#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry + physics feature extractor for Bongard-LOGO shapes.
All methods are safe: they never bubble exceptions.
"""
import time
from shapely.geometry import Polygon, MultiPolygon, LineString
import functools
import logging
import math
import numpy as np
import pymunk

# ───────────── Polygon cache for geometry deduplication ─────────────
_POLY_CACHE: dict[str, Polygon] = {}
# ─────────────────────────────────────────────────────────────────────

def safe_feature(default=0.0):
    """
    Decorator: if the wrapped method throws any Exception,
    log it once and return `default` instead.
    Handles methods that return single values or tuples (e.g., value, confidence).
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logging.warning(
                    f"Feature {fn.__name__} failed ({e!r}), returning default={default}"
                )
                # Check if the function is expected to return a tuple
                if hasattr(fn, '__annotations__') and fn.__annotations__.get('return') and \
                   isinstance(fn.__annotations__['return'], type) and \
                   getattr(fn.__annotations__['return'], '__origin__', None) is tuple:
                    # Construct a default tuple of the correct length if possible
                    # This is a heuristic; ideally, defaults should match function signature
                    num_returns = len(fn.__annotations__['return'].__args__)
                    return tuple([default] * num_returns)
                return default
        return wrapped
    return decorator

def safe_acos(x):
    """
    Clamp input to [-1, 1], then return arccos (in radians) for floats or numpy arrays.
    Prevents math domain errors.
    """
    return np.arccos(np.clip(x, -1.0, 1.0))

# --- Helper for Confidence Scoring (copied from logo_to_shape for consistency) ---
def calculate_confidence(metric_value, max_value, min_value, is_higher_better=True):
    """
    Normalizes a metric value into a confidence score (0-1).
    max_value and min_value define the expected range for the metric.
    """
    if max_value == min_value:
        return 0.5 # Neutral if range is zero

    if is_higher_better:
        return np.clip((metric_value - min_value) / (max_value - min_value), 0.0, 1.0)
    else: # Lower value is better (e.g., error)
        return np.clip(1.0 - (metric_value - min_value) / (max_value - min_value), 0.0, 1.0)


class PhysicsInference:

    @staticmethod
    def _clamped_arccos(dot, norm):
        if norm == 0.0:
            return 0.0
        # returns degrees
        return math.degrees(safe_acos(dot / norm))

    @staticmethod
    def _ensure_polygon(poly_geom):
        if isinstance(poly_geom, MultiPolygon):
            if not poly_geom.is_empty:
                return max(poly_geom.geoms, key=lambda p: p.area)
            return Polygon() # Return empty polygon if MultiPolygon is empty
        return poly_geom

    @staticmethod
    def safe_extract_vertices(obj):
        """
        From list, Polygon, or MultiPolygon → python list of (x,y).
        Handles various input types gracefully.
        """
        if isinstance(obj, np.ndarray) and obj.ndim == 2 and obj.shape[1] == 2:
            return obj.tolist() # Already numpy array of vertices
        if isinstance(obj, list):
            # Ensure it's a list of tuples/lists that look like (x,y)
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in obj):
                return obj
        if hasattr(obj, "exterior") and obj.exterior: # Check if exterior exists and is not empty
            return list(obj.exterior.coords)
        if hasattr(obj, "geoms") and obj.geoms: # Check if geoms exists and is not empty
            largest = max(obj.geoms, key=lambda p: p.area)
            if largest.exterior:
                return list(largest.exterior.coords)
        return []

    @staticmethod
    def polygon_from_vertices(vertices):
        # Convert vertices to a consistent hashable format
        verts_tuple = tuple(tuple(float(coord) for coord in p) for p in vertices)
        key = f"{len(verts_tuple)}-{hash(verts_tuple)}"
        if key in _POLY_CACHE:
            return _POLY_CACHE[key]

        # raw polygon
        t0 = time.time()
        try:
            poly = Polygon(vertices)
        except Exception as e:
            logging.warning(f"Polygon creation from vertices failed: {e}. Attempting repair.")
            # Fallback for invalid polygons: try to buffer or simplify
            if len(vertices) >= 3:
                try:
                    # Attempt to create a LineString and then buffer/polygonize
                    line = LineString(vertices)
                    if not line.is_empty:
                        # Buffer by a small amount and then take convex hull if buffering makes it multipolygon
                        buffered_line = line.buffer(1.0) # Small buffer
                        if isinstance(buffered_line, MultiPolygon):
                            poly = buffered_line.convex_hull # Take convex hull of buffered line
                        else:
                            poly = buffered_line
                    else:
                        poly = Polygon() # Empty polygon if line is empty
                except Exception as e_repair:
                    logging.warning(f"Polygon repair failed: {e_repair}. Returning empty polygon.")
                    poly = Polygon()
            else:
                poly = Polygon() # Default to empty polygon for less than 3 vertices

        # small-polygon repair (fix: use n < 20 for fallback, as in snippet)
        # This part seems to be an old repair mechanism, if polygon() fails, the above try-except should handle.
        # Keeping it for consistency but the buffer logic should be more robust.
        if not poly.is_valid and len(vertices) < 20 and len(vertices) >= 3: # Only attempt if not already valid and enough points
            try:
                # Attempt to create a valid polygon from a simplified version of vertices
                from shapely.validation import make_valid
                poly = make_valid(Polygon(vertices))
            except Exception as e_mv:
                logging.warning(f"make_valid failed for polygon from {len(vertices)} verts: {e_mv}. Trying convex hull.")
                try:
                    poly = Polygon(vertices).convex_hull # Fallback to convex hull
                except Exception as e_ch:
                    logging.warning(f"convex_hull failed: {e_ch}. Returning empty polygon.")
                    poly = Polygon()
            
            if not poly.is_valid: # If still not valid, default to a small square
                logging.warning(f"Polygon still invalid after repair. Defaulting to unit square.")
                poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        poly = PhysicsInference._ensure_polygon(poly)
        _POLY_CACHE[key] = poly
        logging.debug("polygon: build %d verts (valid: %s) in %.3fs", len(vertices), poly.is_valid, time.time()-t0)
        return poly

    @staticmethod
    @safe_feature(default=(0.0, 0.0))
    def centroid(poly_geom) -> tuple[float, float]:
        poly = PhysicsInference._ensure_polygon(poly_geom)
        if poly.is_empty: return (0.0, 0.0) # Handle empty polygon
        c = poly.centroid
        return (float(c.x), float(c.y))

    @staticmethod
    @safe_feature(default=0.0)
    def area(poly_geom) -> float:
        poly = PhysicsInference._ensure_polygon(poly_geom)
        return float(poly.area)

    @staticmethod
    @safe_feature(default=(False, 0.0))
    def is_convex(poly_geom) -> tuple[bool, float]:
        poly = PhysicsInference._ensure_polygon(poly_geom)
        if poly.is_empty: return False, 0.0
        is_conv = poly.equals(poly.convex_hull)
        
        # Confidence: for convex, high. For non-convex, lower confidence.
        # Or based on area ratio: original area / convex hull area. Closer to 1 means more convex.
        if poly.convex_hull.area > 0:
            confidence = poly.area / poly.convex_hull.area
        else:
            confidence = 1.0 if is_conv else 0.0 # If convex hull has zero area, but it's convex
        
        return is_conv, float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    @safe_feature(default=(0.0, 'none', 0.0))
    def symmetry_score(vertices_or_poly) -> tuple[float, str, float]:
        verts = np.array(PhysicsInference.safe_extract_vertices(vertices_or_poly))
        if len(verts) < 2:
            return 0.0, 'none', 0.0
        
        centroid = np.mean(verts, axis=0)
        pts_centered = verts - centroid
        
        from scipy.spatial.distance import cdist

        # Reflectional symmetry (vertical and horizontal axes through centroid)
        mirror_x = pts_centered * np.array([-1, 1])
        mirror_y = pts_centered * np.array([1, -1])

        # Mean distance of mirrored points to the nearest original point
        score_x = np.mean(np.min(cdist(mirror_x, pts_centered), axis=1))
        score_y = np.mean(np.min(cdist(mirror_y, pts_centered), axis=1))

        # Rotational symmetry (180 degrees)
        rot180_pts = -pts_centered # Equivalent to 180 degree rotation
        score_rot180 = np.mean(np.min(cdist(rot180_pts, pts_centered), axis=1))

        # Determine best symmetry type based on minimum score (lower score = better symmetry)
        symmetry_type_str = 'none'
        best_score_val = float(np.inf) # Initialize with infinity

        if score_x < best_score_val:
            best_score_val = score_x
            symmetry_type_str = 'vertical_reflection'
        if score_y < best_score_val: # Use < for strict minimum to prefer y if equal
            best_score_val = score_y
            symmetry_type_str = 'horizontal_reflection'
        if score_rot180 < best_score_val:
            best_score_val = score_rot180
            symmetry_type_str = 'rotational_180'
        
        # Define a threshold for "good enough" symmetry
        symmetry_threshold = 5.0 # pixels mean distance
        if best_score_val > symmetry_threshold:
            symmetry_type_str = 'none' # If no symmetry is strong enough
            best_score_val = np.clip(best_score_val, 0.0, 100.0) # Clip for confidence calculation

        # Confidence: inversely related to the best symmetry score (lower score = higher confidence)
        confidence = calculate_confidence(best_score_val, max_value=50.0, min_value=0.0, is_higher_better=False)
        
        return float(best_score_val), symmetry_type_str, confidence

    @staticmethod
    @safe_feature(default=(0, 0.0))
    def count_arcs(vertices_or_poly, angle_thresh=30) -> tuple[int, float]:
        """
        Estimate the number of arcs in a polyline or polygon.
        An arc is defined as a contiguous segment where the angle between consecutive segments deviates from 180° by more than angle_thresh degrees.
        Args:
            vertices_or_poly: list of (x, y) tuples or Polygon
            angle_thresh: minimum deviation from 180° to count as an arc (degrees)
        Returns:
            (int, float): estimated number of arcs, and confidence
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 0, 0.0
        
        n = len(verts)
        arc_count = 0
        angles = []
        for i in range(n):
            p0 = verts[i - 1]
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            
            v1_vec = np.array(p1) - np.array(p0)
            v2_vec = np.array(p2) - np.array(p1)
            
            norm_v1 = np.linalg.norm(v1_vec)
            norm_v2 = np.linalg.norm(v2_vec)

            if norm_v1 < 1e-6 or norm_v2 < 1e-6: # Degenerate segment, treat as straight for angle calculation
                angles.append(180) 
                continue

            cos_theta = np.dot(v1_vec, v2_vec) / (norm_v1 * norm_v2)
            angle = np.degrees(safe_acos(cos_theta)) # Use safe_acos
            angles.append(angle)
            
            if abs(angle - 180) > angle_thresh: # Angle not straight enough
                arc_count += 1
        
        # Confidence: Higher if angles are clearly different from 180 (for arcs)
        # Or if the distribution of angles confirms the arc count.
        # Simple confidence: proportion of significant angle deviations
        confidence = arc_count / n if n > 0 else 0.0
        return arc_count, confidence

    @staticmethod
    @safe_feature(default=(0, 0.0))
    def count_straight_segments(vertices_or_poly, angle_tol=5) -> tuple[int, float]:
        """
        Counts the number of straight segments in a polyline.
        A segment is considered straight if the angle between consecutive segments is within angle_tol degrees of 180.
        Args:
            vertices_or_poly: list of (x, y) tuples or Polygon
            angle_tol: tolerance in degrees
        Returns:
            (int, float): number of straight segments, and confidence
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 0, 0.0

        count = 0
        n = len(verts)
        straight_segment_angles_deviation_sum = 0
        num_straight_candidates = 0

        for i in range(n):
            p0 = verts[i - 1]
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            
            v1_vec = np.array(p1) - np.array(p0)
            v2_vec = np.array(p2) - np.array(p1)
            
            norm_v1 = np.linalg.norm(v1_vec)
            norm_v2 = np.linalg.norm(v2_vec)

            if norm_v1 < 1e-6 or norm_v2 < 1e-6: # Degenerate segment, implies straight but zero length
                count += 1
                continue

            cos_theta = np.dot(v1_vec, v2_vec) / (norm_v1 * norm_v2)
            angle = np.degrees(safe_acos(cos_theta))
            
            if abs(angle - 180) <= angle_tol:
                count += 1
                num_straight_candidates += 1
                straight_segment_angles_deviation_sum += abs(angle - 180) # Sum of deviations from 180
        
        # Confidence: inversely related to the average deviation from 180 for detected straight segments
        if num_straight_candidates > 0:
            avg_deviation = straight_segment_angles_deviation_sum / num_straight_candidates
            confidence = calculate_confidence(avg_deviation, max_value=angle_tol, min_value=0.0, is_higher_better=False)
        else:
            confidence = 0.0 # No straight segments found
            
        return count, confidence

    @staticmethod
    @safe_feature(default=(False, 0.0))
    def is_balanced(all_shapes, gravity_direction=(0, 1), support_threshold=5.0) -> tuple[bool, float]:
        """
        Infers if a collection of shapes is physically balanced.
        Requires shape centroids and areas.
        Args:
            all_shapes: List of shape dictionaries (must contain 'centroid' and 'area').
            gravity_direction: Tuple (dx, dy) representing direction of gravity. (0,1) for down.
            support_threshold: Max distance from supporting base to be considered supported.
        Returns:
            (bool, float): True if balanced, confidence score.
        """
        if not all_shapes:
            return True, 1.0 # Vacuously true if no shapes

        # Calculate combined center of mass (COM) for all shapes
        total_mass = 0.0
        combined_com_x = 0.0
        combined_com_y = 0.0

        for s in all_shapes:
            # Assuming area is proportional to mass for simplicity
            mass = s.get('area', 0.0)
            centroid = s.get('centroid', (0.0, 0.0))
            
            total_mass += mass
            combined_com_x += centroid[0] * mass
            combined_com_y += centroid[1] * mass
        
        if total_mass == 0:
            return True, 0.5 # If no mass, trivially balanced (neutral confidence)

        combined_com = (combined_com_x / total_mass, combined_com_y / total_mass)

        # Determine the base polygon formed by the bottom-most shapes or supporting shapes
        # For a simple "balanced on a surface" check, project all shapes onto the gravity axis.
        # Find the convex hull of the bottom-most points that could act as a base.
        
        # Identify 'support' points: lowest points of all shapes
        support_points = []
        for s in all_shapes:
            verts = np.array(s.get('vertices', []))
            if verts.size > 0:
                # Find points closest to the "down" direction
                # For gravity_direction (0,1), this is points with max y
                # For gravity_direction (0,-1), this is points with min y
                # More generally, project onto the normalized gravity vector
                
                # Assume gravity is (0,1) for simplicity (downwards in image coords typically)
                # So we look for minimum y values
                if len(verts) > 0:
                    min_y_val = np.min(verts[:, 1])
                    bottom_points = verts[verts[:, 1] <= min_y_val + support_threshold]
                    support_points.extend(bottom_points.tolist())
        
        if not support_points or len(support_points) < 3:
            # Cannot form a base if less than 3 support points
            return False, 0.1 # Unstable, low confidence

        try:
            base_polygon = Polygon(support_points).convex_hull
            if base_polygon.is_empty:
                return False, 0.1
        except Exception as e:
            logging.warning(f"Failed to create base polygon from support points: {e}")
            return False, 0.1

        # Check if the combined COM falls within the base polygon
        from shapely.geometry import Point
        com_point = Point(combined_com)
        
        is_com_within_base = base_polygon.contains(com_point) or base_polygon.touches(com_point)

        # Confidence:
        # If COM is within base: confidence is higher the more central the COM is.
        # If COM is outside base: confidence is lower the further away it is.
        
        if is_com_within_base:
            # Calculate distance of COM to base centroid, normalized by base size
            base_centroid = base_polygon.centroid
            dist_to_base_center = com_point.distance(base_centroid)
            base_bbox = base_polygon.bounds
            base_width = base_bbox[2] - base_bbox[0]
            base_height = base_bbox[3] - base_bbox[1]
            base_diag = np.sqrt(base_width**2 + base_height**2) + 1e-6
            
            norm_dist = dist_to_base_center / base_diag
            confidence = calculate_confidence(norm_dist, max_value=0.5, min_value=0.0, is_higher_better=False) # Closer to center is better
            return True, confidence
        else:
            # If COM is outside, it's unbalanced. Confidence is higher for clearly unbalanced.
            # Distance of COM to the boundary of the base polygon
            dist_to_boundary = com_point.distance(base_polygon.exterior)
            confidence = calculate_confidence(dist_to_boundary, max_value=50.0, min_value=0.0, is_higher_better=True) # Further outside is more confidently unbalanced
            return False, confidence

    @staticmethod
    @safe_feature(default=(False, 0.0))
    def is_occluded_physically(shape_obj_id, all_shapes, visibility_threshold=0.8) -> tuple[bool, float]:
        """
        Infers if a specific shape is physically occluded by other shapes.
        Assumes shapes are 'flat' and occlusion is 2D overlap with a z-order.
        Requires 'id', 'vertices', 'area', and 'draw_order' (higher means drawn later, so on top).
        Args:
            shape_obj_id: ID of the shape to check for occlusion.
            all_shapes: List of all shape dictionaries in the scene.
            visibility_threshold: Minimum proportion of shape's area that must be visible for it not to be considered occluded.
        Returns:
            (bool, float): True if occluded, confidence score.
        """
        target_shape = next((s for s in all_shapes if s['id'] == shape_obj_id), None)
        if not target_shape or not target_shape.get('vertices') or len(target_shape['vertices']) < 3:
            return False, 0.0

        try:
            target_poly = PhysicsInference.polygon_from_vertices(target_shape['vertices'])
            if not target_poly.is_valid or target_poly.area == 0:
                return False, 0.0
        except Exception as e:
            logging.warning(f"Occlusion: invalid polygon for target shape {shape_obj_id}: {e}")
            return False, 0.0

        total_target_area = target_poly.area
        occluded_area = 0.0
        
        # Assuming higher 'draw_order' means on top
        target_draw_order = target_shape.get('draw_order', 0)

        for other_shape in all_shapes:
            if other_shape['id'] == shape_obj_id:
                continue
            
            # Only consider shapes drawn on top of the target shape
            other_draw_order = other_shape.get('draw_order', 0)
            if other_draw_order < target_draw_order: # Other shape is behind, cannot occlude
                continue
            if not other_shape.get('vertices') or len(other_shape['vertices']) < 3:
                continue

            try:
                other_poly = PhysicsInference.polygon_from_vertices(other_shape['vertices'])
                if not other_poly.is_valid or other_poly.area == 0:
                    continue
            except Exception as e:
                logging.warning(f"Occlusion: invalid polygon for other shape {other_shape['id']}: {e}")
                continue

            if target_poly.intersects(other_poly):
                occluded_portion = target_poly.intersection(other_poly)
                occluded_area += occluded_portion.area
        
        # The occluded area cannot exceed the total area of the target shape
        occluded_area = np.clip(occluded_area, 0, total_target_area)

        visible_area = total_target_area - occluded_area
        visibility_ratio = visible_area / (total_target_area + 1e-6)

        is_occ = visibility_ratio < visibility_threshold
        
        # Confidence: higher for clearly occluded or clearly visible
        if is_occ:
            # How much area is occluded, relative to threshold. More occluded = higher confidence of occlusion.
            confidence = calculate_confidence(visibility_ratio, max_value=visibility_threshold, min_value=0.0, is_higher_better=False)
        else:
            # How much area is visible, relative to threshold. More visible = higher confidence of not occluded.
            confidence = calculate_confidence(visibility_ratio, max_value=1.0, min_value=visibility_threshold, is_higher_better=True)
            
        return is_occ, confidence

    @staticmethod
    @safe_feature(default=(False, 0.0))
    def is_touching_ground(shape_obj, ground_y_coord=None, tolerance=5.0) -> tuple[bool, float]:
        """
        Infers if a shape is touching a conceptual 'ground' plane.
        Assumes ground is at a specific Y coordinate.
        Args:
            shape_obj: Dictionary with 'vertices'.
            ground_y_coord: The Y coordinate of the ground plane. If None, uses the lowest Y in all shapes.
            tolerance: Max distance from ground to be considered touching.
        Returns:
            (bool, float): True if touching, confidence.
        """
        verts = np.array(PhysicsInference.safe_extract_vertices(shape_obj))
        if verts.size == 0:
            return False, 0.0
        
        min_y_shape = np.min(verts[:, 1])

        if ground_y_coord is None:
            # Default ground_y_coord to the absolute lowest point found if not provided
            # This requires access to all shapes' vertices, not just `shape_obj`.
            # For simplicity, assuming the "ground" is the lowest Y in this specific shape.
            # A more robust system would pass `all_shapes` to determine scene ground.
            ground_y_coord = min_y_shape # This means any shape will "touch" its own lowest point.
                                         # For a true ground plane, `ground_y_coord` needs to be global.
            logging.warning("No global ground_y_coord provided for is_touching_ground. Using shape's lowest point as reference.")

        dist_to_ground = abs(min_y_shape - ground_y_coord)

        is_touching = dist_to_ground <= tolerance
        
        # Confidence: inversely related to distance from ground
        confidence = calculate_confidence(dist_to_ground, max_value=tolerance*2, min_value=0.0, is_higher_better=False)

        return is_touching, confidence

    @staticmethod
    @safe_feature(default=(False, 0.0))
    def is_floating(shape_obj, ground_y_coord=None, tolerance=5.0) -> tuple[bool, float]:
        """
        Infers if a shape is floating (not touching ground and not supported).
        Returns (bool, confidence).
        """
        # A shape is floating if it's not touching the ground and not supported by another object below it.
        # This requires more complex interaction analysis than just ground contact.
        # For a basic definition: if it's not touching the ground AND its lowest point is significantly above ground.

        is_touching, conf_touching = PhysicsInference.is_touching_ground(shape_obj, ground_y_coord, tolerance)
        
        if is_touching:
            return False, 1.0 - conf_touching # If touching, not floating, confidence inverse of touching confidence
        
        # If not touching, consider it floating with confidence based on distance from ground
        verts = np.array(PhysicsInference.safe_extract_vertices(shape_obj))
        if verts.size == 0: return False, 0.0

        min_y_shape = np.min(verts[:, 1])

        # If no global ground, this function won't be very meaningful for 'floating'
        if ground_y_coord is None:
            return False, 0.0

        dist_from_ground = min_y_shape - ground_y_coord # How far above ground

        is_floating_candidate = dist_from_ground > tolerance
        
        # Confidence: higher if clearly above the ground
        confidence = calculate_confidence(dist_from_ground, max_value=100.0, min_value=tolerance, is_higher_better=True) # Max_value is arbitrary, adjust
        
        return is_floating_candidate, confidence

    @staticmethod
    @safe_feature(default=(False, 0.0))
    def is_stable(shape_obj_id, all_shapes, ground_y_coord=None, tolerance=5.0) -> tuple[bool, float]:
        """
        Infers if a specific shape is physically stable (not likely to topple or fall).
        Requires centroids, polygons, and ground reference.
        This is a simplified check, a full physics simulation would be more accurate.
        Returns (bool, confidence).
        """
        target_shape = next((s for s in all_shapes if s['id'] == shape_obj_id), None)
        if not target_shape or not target_shape.get('vertices') or len(target_shape['vertices']) < 3:
            return False, 0.0
        
        target_poly = PhysicsInference.polygon_from_vertices(target_shape['vertices'])
        if not target_poly.is_valid: return False, 0.0

        # Simple stability: check if COM is over its base of support.
        # If it's on the ground, its base is its own footprint.
        # If it's on another object, its base is the intersection area.
        
        target_centroid = target_shape.get('centroid', PhysicsInference.centroid(target_poly))
        is_on_ground, conf_on_ground = PhysicsInference.is_touching_ground(target_shape, ground_y_coord, tolerance)

        if is_on_ground:
            # Base of support is the shape itself if it's on the ground
            base_of_support_poly = target_poly
            base_poly_conf = 1.0
        else:
            # Look for supporting objects below it (simplified)
            supporting_polys = []
            for other_shape in all_shapes:
                if other_shape['id'] == shape_obj_id: continue
                if not other_shape.get('vertices') or len(other_shape['vertices']) < 3: continue
                
                other_poly = PhysicsInference.polygon_from_vertices(other_shape['vertices'])
                if not other_poly.is_valid: continue

                # Check if other_poly is directly below target_poly and intersects/touches
                # Simplified check: if centroid of other_poly is directly below target_poly
                # and they overlap in X, and other_poly is below target_poly Y-wise
                target_bbox = target_poly.bounds
                other_bbox = other_poly.bounds
                
                # Check for X-overlap and Y-stacking
                x_overlap = max(0, min(target_bbox[2], other_bbox[2]) - max(target_bbox[0], other_bbox[0]))
                if x_overlap > 0 and other_bbox[3] < target_bbox[1] + tolerance: # Other's top is near or below target's bottom
                    if target_poly.intersects(other_poly): # They actually touch/overlap in 2D
                        supporting_polys.append(other_poly)

            if not supporting_polys:
                return False, 0.8 # Not touching ground, no clear support, likely unstable
            
            # Combine all supporting polygons to form a single base of support
            from shapely.ops import unary_union
            base_of_support_poly = unary_union(supporting_polys)
            base_poly_conf = 0.7 # Moderate confidence for combined base
            
        if base_of_support_poly.is_empty:
            return False, 0.0

        # Project target shape's COM onto the horizontal plane and check if it falls within the base
        com_projected = Point(target_centroid[0], base_of_support_poly.centroid.y) # Use COM's X, base's Y
        
        is_com_over_base = base_of_support_poly.contains(Point(target_centroid)) # More accurate: use actual COM point
        
        stability_score = 0.0 # Distance of COM to boundary of base
        if is_com_over_base:
            # Stable: confidence increases closer to center of base
            dist_to_center_of_base = Point(target_centroid).distance(base_of_support_poly.centroid)
            base_dims = max(base_of_support_poly.bounds[2] - base_of_support_poly.bounds[0],
                            base_of_support_poly.bounds[3] - base_of_support_poly.bounds[1])
            stability_score = dist_to_center_of_base / (base_dims + 1e-6)
            confidence = calculate_confidence(stability_score, max_value=0.5, min_value=0.0, is_higher_better=False)
            return True, confidence * base_poly_conf # Combine confidences
        else:
            # Unstable: confidence increases with distance from base boundary
            dist_to_boundary = Point(target_centroid).distance(base_of_support_poly.exterior)
            stability_score = dist_to_boundary
            confidence = calculate_confidence(stability_score, max_value=50.0, min_value=0.0, is_higher_better=True)
            return False, confidence * (1.0 - base_poly_conf) # Inverse confidence for unstable

    @staticmethod
    @safe_feature(default=(0, 0.0))
    def count_contact_points(shape_obj_id, all_shapes, contact_tolerance=2.0) -> tuple[int, float]:
        """
        Counts the number of distinct contact points a shape has with other shapes or ground.
        Returns (count, confidence).
        """
        target_shape = next((s for s in all_shapes if s['id'] == shape_obj_id), None)
        if not target_shape or not target_shape.get('vertices') or len(target_shape['vertices']) < 3:
            return 0, 0.0
        
        target_poly = PhysicsInference.polygon_from_vertices(target_shape['vertices'])
        if not target_poly.is_valid: return 0, 0.0

        contact_points = []

        # Check contact with other shapes
        for other_shape in all_shapes:
            if other_shape['id'] == shape_obj_id: continue
            if not other_shape.get('vertices') or len(other_shape['vertices']) < 3: continue

            other_poly = PhysicsInference.polygon_from_vertices(other_shape['vertices'])
            if not other_poly.is_valid: continue

            if target_poly.touches(other_poly) or target_poly.overlaps(other_poly):
                intersection = target_poly.intersection(other_poly)
                
                # If intersection is a LineString or a MultiPoint, it's a contact point/line
                if isinstance(intersection, LineString):
                    # For a line contact, consider its endpoints or a representative point
                    contact_points.append(intersection.centroid.coords[0])
                elif intersection.geom_type == 'MultiPoint':
                    contact_points.extend([p.coords[0] for p in intersection.geoms])
                elif not intersection.is_empty: # Any non-empty intersection implies contact
                    contact_points.append(intersection.centroid.coords[0]) # Use centroid as a representative

        # Check contact with a conceptual ground (if defined globally)
        # This requires `ground_y_coord` to be passed or globally accessible.
        # For now, let's assume `ground_y_coord` is the minimum Y across all shapes.
        
        all_verts_in_scene = []
        for s in all_shapes:
            verts = PhysicsInference.safe_extract_vertices(s)
            if verts:
                all_verts_in_scene.extend(verts)
        
        if all_verts_in_scene:
            scene_min_y = np.min(np.array(all_verts_in_scene)[:, 1])
            # If the shape is close to the scene's minimum Y, consider it touching ground
            target_min_y = np.min(np.array(target_shape['vertices'])[:, 1])
            if abs(target_min_y - scene_min_y) < contact_tolerance:
                # Add a representative contact point for ground (e.g., bottom-center of shape)
                target_bbox = target_poly.bounds
                contact_points.append(((target_bbox[0] + target_bbox[2]) / 2, scene_min_y))

        # Filter unique contact points (cluster nearby points)
        if not contact_points:
            return 0, 0.0

        from sklearn.cluster import DBSCAN
        # Adjust eps based on contact_tolerance
        clustering = DBSCAN(eps=contact_tolerance, min_samples=1).fit(contact_points)
        num_distinct_contacts = len(set(clustering.labels_))
        
        # Confidence: higher for more distinct contacts
        confidence = calculate_confidence(num_distinct_contacts, max_value=5, min_value=1) # Arbitrary max_value of 5 contacts

        return num_distinct_contacts, confidence
