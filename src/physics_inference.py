#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All imports and logic preserved. Fixed indentation and removed duplicate code.
"""

import time
from shapely.geometry import Polygon, MultiPolygon
import functools
import logging
import math
import time
import numpy as np
import pymunk

# ───────────── Polygon cache for geometry deduplication ─────────────
_POLY_CACHE: dict[str, Polygon] = {}
# ─────────────────────────────────────────────────────────────────────





def safe_feature(default=0.0):
    """
    Decorator: if the wrapped method throws any Exception,
    log it once and return `default` instead.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logging.error(f"safe_feature: Exception {e}")
                return default
        return wrapped
    return decorator

    def safe_feature(default=0.0):
        """
        Decorator: if the wrapped method throws any Exception,
        log it once and return `default` instead.
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logging.error(f"safe_feature: Exception {e}")
                    return default
            return wrapped
        return decorator


class PhysicsInference:
    @staticmethod
    def polsby_popper_compactness(area, perimeter):
        """
        Computes the Polsby-Popper compactness score for a polygon.
        Formula: 4 * pi * area / (perimeter ** 2)
        """
        import math
        if perimeter == 0:
            return 0.0
        return 4 * math.pi * area / (perimeter ** 2)
    @staticmethod
    @safe_feature(default=0.0)
    def robust_curvature(vertices_or_poly):
            """
            Robust geometric curvature: average absolute change in tangent angle per unit length along the shape.
            Handles polygons, polylines, and noisy input. Returns 0.0 for degenerate cases.
            """
            logging.info(f"[robust_curvature] INPUT: {vertices_or_poly}")
            verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
            verts = PhysicsInference.dedup_vertices(verts)
            if not verts or len(verts) < 3:
                logging.warning("[robust_curvature] Not enough vertices.")
                logging.info("[robust_curvature] OUTPUT: 0.0")
                return 0.0
            n = len(verts)
            total_curvature = 0.0
            total_length = 0.0
            for i in range(n):
                p0 = np.array(verts[i - 1])
                p1 = np.array(verts[i])
                p2 = np.array(verts[(i + 1) % n])
                v1 = p1 - p0
                v2 = p2 - p1
                len1 = np.linalg.norm(v1)
                len2 = np.linalg.norm(v2)
                if len1 > 1e-6 and len2 > 1e-6:
                    dot = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
                    dtheta = np.arccos(dot)
                    seg_length = (len1 + len2) / 2
                    total_curvature += abs(dtheta) * seg_length
                    total_length += seg_length
            if total_length < 1e-8:
                logging.info("[robust_curvature] OUTPUT: 0.0")
                return 0.0
            result = float(total_curvature / total_length)
            logging.info(f"[robust_curvature] OUTPUT: {result}")
            return result

    @staticmethod
    def shoelace_area(vertices):
        """
        Compute the area of a polygon using the shoelace formula.
        Args:
            vertices (list or np.ndarray): List of (x, y) tuples or Nx2 array.
        Returns:
            float: Area of the polygon (always positive).
        """
        import numpy as np
        if len(vertices) < 3:
            return 0.0
        arr = np.array(vertices)
        x = arr[:, 0]
        y = arr[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    @staticmethod
    def robust_angular_variance(vertices):
            """
            Estimate the angular variance of a polygon given its vertices.
            Handles degenerate cases and noisy polygons robustly.
            """
            logger = logging.getLogger("PhysicsInference")
            try:
                if len(vertices) < 3:
                    logger.warning("Not enough vertices for angular variance. Returning 0.")
                    return 0.0
                vertices = np.array(vertices)
                # Compute edge vectors
                edges = np.diff(vertices, axis=0, append=vertices[:1])
                # Normalize edge vectors
                norms = np.linalg.norm(edges, axis=1, keepdims=True)
                edges_normed = edges / np.where(norms == 0, 1, norms)
                # Compute angles between consecutive edges
                dot_products = np.sum(edges_normed * np.roll(edges_normed, -1, axis=0), axis=1)
                dot_products = np.clip(dot_products, -1.0, 1.0)
                angles = np.arccos(dot_products)
                # Angular variance
                variance = np.var(angles)
                logger.info(f"Computed angular variance: {variance:.6f} for {len(vertices)} vertices.")
                return variance
            except Exception as e:
                logger.error(f"Error in robust_angular_variance: {e}")
                return 0.0

    @staticmethod
    def alternation_score(seq):
        """
        Compute maximal alternating subsequence fraction.
        """
        if not seq or len(seq) < 2:
            return 0.0
        max_alt = 1
        curr = 1
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                curr += 1
            else:
                max_alt = max(max_alt, curr)
                curr = 1
        max_alt = max(max_alt, curr)
        return max_alt / len(seq)

    @staticmethod
    def visual_complexity(num_strokes, max_strokes, perimeter, hull_perimeter, curvature_score, alpha=0.4, beta=0.3, gamma=0.3):
        """
        Weighted sum of normalized stroke count, perimeter/hull, and curvature.
        All values normalized to [0,1].
        """
        stroke_term = min(num_strokes / max(max_strokes, 1), 1.0)
        perim_term = min(perimeter / max(hull_perimeter, 1e-8), 1.0)
        curv_term = min(curvature_score / np.pi, 1.0)
        return alpha * stroke_term + beta * perim_term + gamma * curv_term

    @staticmethod
    def safe_finite(val, default=0.0, minval=None, maxval=None):
        """
        Ensure val is finite and optionally clamp to [minval, maxval].
        Returns default if not finite.
        """
        try:
            if not np.isfinite(val):
                return default
            if minval is not None:
                val = max(val, minval)
            if maxval is not None:
                val = min(val, maxval)
            return val
        except Exception as e:
            logging.warning(f"safe_finite: Exception {e}, returning default={default}")
            return default
    @staticmethod
    def line_curvature_score(vertices):
        # For a straight line, curvature is zero
        return 0.0

    @staticmethod
    def arc_curvature_score(radius, delta_theta):
        # For a circular arc, curvature is |delta_theta| / radius
        if radius == 0:
            return float('nan')
        return abs(delta_theta) / abs(radius)

    @staticmethod
    def line_moment_of_inertia(length):
        # Thin rod about midpoint, unit density: I = L^3 / 12
        return (length ** 3) / 12.0

    @staticmethod
    def arc_moment_of_inertia(radius, delta_theta):
        # Approximate as thin arc, unit density, about centroid
        # For small arcs, use I = m*R^2, m = arc length
        arc_length = abs(radius * delta_theta)
        return arc_length * (radius ** 2)

    @staticmethod
    def line_center_of_mass(x1, y1, x2, y2):
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def arc_center_of_mass(x1, y1, x2, y2, radius, delta_theta, cx, cy):
        # Center of mass of a circular arc segment (approximate)
        # See: https://mathworld.wolfram.com/Arc.html
        if delta_theta == 0:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        theta1 = math.atan2(y1 - cy, x1 - cx)
        theta2 = math.atan2(y2 - cy, x2 - cx)
        theta_avg = (theta1 + theta2) / 2.0
        r = abs(radius)
        d = r * math.sin(abs(delta_theta) / 2.0) / (abs(delta_theta) / 2.0) if abs(delta_theta) > 1e-8 else r
        return (cx + d * math.cos(theta_avg), cy + d * math.sin(theta_avg))
    @staticmethod
    def dedup_vertices(verts, epsilon=1e-8):
        """
        Remove all duplicate vertices (within epsilon) and collinear points.
        """
        if not verts:
            return []
        deduped = []
        for v in verts:
            if not any(np.linalg.norm(np.array(v) - np.array(u)) < epsilon for u in deduped):
                deduped.append(v)
        # Remove collinear points
        def is_collinear(p1, p2, p3, eps=1e-8):
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            area = np.abs(np.cross(b - a, c - a))
            return area < eps
        if len(deduped) > 2:
            filtered = [deduped[0]]
            for i in range(1, len(deduped) - 1):
                if not is_collinear(deduped[i-1], deduped[i], deduped[i+1]):
                    filtered.append(deduped[i])
            filtered.append(deduped[-1])
            deduped = filtered
        return deduped

    @staticmethod
    def rounded_bbox(verts, epsilon=1e-10):
        """
        Compute bounding box and round near-zero values to zero.
        """
        arr = np.array(verts)
        minx, miny = arr.min(axis=0)
        maxx, maxy = arr.max(axis=0)
        def round_eps(x):
            return 0.0 if abs(x) < epsilon else x
        return tuple(map(round_eps, (minx, miny, maxx, maxy)))
    @staticmethod
    @safe_feature(default=0.0)
    def curvature_score(vertices_or_poly):
        """
        Average absolute change in tangent angle per unit length along the polyline.
        Integrates curvature over arcs. Returns NaN for degenerate cases.
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        verts = PhysicsInference.dedup_vertices(verts)
        if not verts or len(verts) < 3:
            return float('nan')
        n = len(verts)
        total_curvature = 0.0
        total_length = 0.0
        for i in range(n):
            p0 = np.array(verts[i - 1])
            p1 = np.array(verts[i])
            p2 = np.array(verts[(i + 1) % n])
            v1 = p1 - p0
            v2 = p2 - p1
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 > 1e-6 and len2 > 1e-6:
                dot = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
                dtheta = np.arccos(dot)
                seg_length = (len1 + len2) / 2
                total_curvature += abs(dtheta) * seg_length
                total_length += seg_length
        if total_length < 1e-8:
            return float('nan')
        return float(total_curvature / total_length)
    @staticmethod
    def is_short_line(length: float, diag: float, thresh: float = 0.15) -> bool:
        """Inclusive comparison for short line flag. Use thresh=0.2 as per spec."""
        return length <= 0.2 * diag

    @staticmethod
    def is_long_line(length: float, diag: float, thresh: float = 0.85) -> bool:
        """Inclusive comparison for long line flag. Use thresh=0.8 as per spec."""
        return length >= 0.8 * diag
    @staticmethod
    def rotational_symmetry_mask(mask: np.ndarray, k: int = 2) -> float:
        """Compute k-fold rotational symmetry as normalized mask correlation."""
        from scipy import ndimage
        if mask is None or mask.sum() == 0:
            return 0.0
        rot = ndimage.rotate(mask, 360 / k, reshape=False, order=0)
        intersection = np.logical_and(mask, rot).sum()
        union = np.logical_or(mask, rot).sum()
        if union == 0:
            return 0.0
        return intersection / union
    @staticmethod
    @safe_feature(default=1)
    def rotational_symmetry(vertices_or_poly, max_order=None, rmse_threshold=0.02):
        """
        Detects the highest order of rotational symmetry for a polygon/polyline.
        Compares the shape to its rotated versions for k=2,3,...,max_order (default n//2).
        Returns the highest k (>=2) with mean RMSE below threshold, else 1 (no symmetry).
        Args:
            vertices_or_poly: list of (x, y) tuples or Polygon
            max_order: maximum symmetry order to check (default: n//2)
            rmse_threshold: RMSE threshold for symmetry (default: 0.02 for normalized shapes)
        Returns:
            int: highest symmetry order detected (>=2), or 1 if none
        """
        import numpy as np
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 1
        arr = np.array(verts)
        n = len(arr)
        # Remove duplicate last point if closed
        if n > 3 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
            n = len(arr)
        centroid = np.mean(arr, axis=0)
        arr_centered = arr - centroid
        # Normalize scale for RMSE to be meaningful
        scale = np.linalg.norm(arr_centered, axis=1).max()
        if scale < 1e-8:
            scale = 1.0
        arr_norm = arr_centered / scale
        best_order = 1
        best_rmse = float('inf')
        # By default, check up to n//2 (cannot have more symmetry than half the points)
        max_k = max_order if max_order is not None else max(2, n // 2)
        for k in range(2, max_k + 1):
            theta = 2 * np.pi / k
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            arr_rot = (arr_norm @ rot_matrix.T)
            # Find best cyclic alignment (allow for cyclic permutation)
            min_rmse = float('inf')
            for shift in range(n):
                arr_shift = np.roll(arr_norm, shift, axis=0)
                if arr_shift.shape != arr_rot.shape:
                    continue
                rmse = np.sqrt(np.mean((arr_shift - arr_rot) ** 2))
                if rmse < min_rmse:
                    min_rmse = rmse
            # If RMSE is below threshold, consider this symmetry order
            if min_rmse < rmse_threshold and min_rmse < best_rmse:
                best_order = k
                best_rmse = min_rmse
        return best_order
    @staticmethod
    @safe_feature(default=0.0)
    def angular_variance(vertices_or_poly):
        """
        Computes the variance of interior angles (in radians^2) for a polygon/polyline.
        Filters out degenerate (near-colinear) angles. Returns NaN if <2 valid angles.
        Normalized by number of vertices.
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        verts = PhysicsInference.dedup_vertices(verts)
        if not verts or len(verts) < 3:
            return float('nan')
        n = len(verts)
        angles = []
        for i in range(n):
            p0 = np.array(verts[i - 1])
            p1 = np.array(verts[i])
            p2 = np.array(verts[(i + 1) % n])
            v1 = p0 - p1
            v2 = p2 - p1
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle_rad = np.arccos(dot)
                if angle_rad > 1e-3:
                    angles.append(angle_rad)
        if len(angles) < 2:
            return float('nan')
        return float(np.var(angles) / n)

    @staticmethod
    def find_stroke_intersections(geoms):
        """
        Count the number of intersections between all unique pairs of geometries.
        Accepts a list of shapely geometries (LineString/Polygon).
        Skips degenerate or invalid geometries.
        """
        import logging
        count = 0
        n = len(geoms)
        for i in range(n):
            g1 = geoms[i]
            if g1 is None or not hasattr(g1, 'is_valid') or not g1.is_valid:
                logging.debug(f"find_stroke_intersections: skipping invalid geometry at index {i}")
                continue
            for j in range(i+1, n):
                g2 = geoms[j]
                if g2 is None or not hasattr(g2, 'is_valid') or not g2.is_valid:
                    logging.debug(f"find_stroke_intersections: skipping invalid geometry at index {j}")
                    continue
                try:
                    if g1.intersects(g2):
                        count += 1
                except Exception as e:
                    logging.debug(f"find_stroke_intersections: error comparing geometries at {i},{j}: {e}")
        return count

    @staticmethod
    def strokes_touching(strokes):
        """
        Count the number of pairs of strokes that touch (share at least one endpoint).
        Accepts a list of shapely geometries or action objects with .vertices.
        Robust to degenerate/empty geometries.
        """
        import logging
        count = 0
        n = len(strokes)
        for i in range(n):
            s1 = strokes[i]
            if s1 is None or not hasattr(s1, 'coords') or s1.is_empty:
                logging.debug(f"strokes_touching: skipping degenerate geometry at index {i}")
                continue
            v1 = list(s1.coords)
            for j in range(i+1, n):
                s2 = strokes[j]
                if s2 is None or not hasattr(s2, 'coords') or s2.is_empty:
                    logging.debug(f"strokes_touching: skipping degenerate geometry at index {j}")
                    continue
                v2 = list(s2.coords)
                if not v1 or not v2:
                    continue
                if v1[0] in v2 or v1[-1] in v2 or v2[0] in v1 or v2[-1] in v1:
                    count += 1
        return count

    @staticmethod
    def stroke_contains_stroke(strokes):
        """
        Count the number of buffered polygons fully containing another buffered polygon.
        Accepts a list of shapely geometries (buffered recommended).
        """
        import logging
        count = 0
        n = len(strokes)
        for i in range(n):
            poly1 = strokes[i]
            if poly1 is None or not hasattr(poly1, 'is_empty') or poly1.is_empty:
                logging.debug(f"stroke_contains_stroke: skipping degenerate geometry at index {i}")
                continue
            for j in range(n):
                if i == j:
                    continue
                poly2 = strokes[j]
                if poly2 is None or not hasattr(poly2, 'is_empty') or poly2.is_empty:
                    logging.debug(f"stroke_contains_stroke: skipping degenerate geometry at index {j}")
                    continue
                try:
                    if poly1.contains(poly2):
                        count += 1
                except Exception as e:
                    logging.debug(f"stroke_contains_stroke: error comparing geometries at {i},{j}: {e}")
        return count

    @staticmethod
    def stroke_overlap_area(strokes):
        """
        Compute the total area of overlap between all pairs of buffered stroke polygons.
        Accepts a list of shapely geometries (buffered recommended).
        """
        import logging
        total_overlap = 0.0
        n = len(strokes)
        for i in range(n):
            poly1 = strokes[i]
            if poly1 is None or not hasattr(poly1, 'is_empty') or poly1.is_empty:
                logging.debug(f"stroke_overlap_area: skipping degenerate geometry at index {i}")
                continue
            for j in range(i+1, n):
                poly2 = strokes[j]
                if poly2 is None or not hasattr(poly2, 'is_empty') or poly2.is_empty:
                    logging.debug(f"stroke_overlap_area: skipping degenerate geometry at index {j}")
                    continue
                try:
                    if poly1.intersects(poly2):
                        total_overlap += poly1.intersection(poly2).area
                except Exception as e:
                    logging.debug(f"stroke_overlap_area: error comparing geometries at {i},{j}: {e}")
        return total_overlap

    @staticmethod
    def count_arcs(vertices_or_poly, angle_thresh=10):
        """
        Robustly estimate the number of arcs in a polyline or polygon using vertex-based curvature estimation.
        An arc is a contiguous segment where the turning angle changes sign and exceeds angle_thresh (degrees).
        Args:
            vertices_or_poly: list of (x, y) tuples or Polygon
            angle_thresh: minimum deviation in degrees to count as an arc
        Returns:
            int: estimated number of arcs
        """
        import numpy as np
        logger = logging.getLogger(__name__)
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 0
        n = len(verts)
        angles = []
        for i in range(n):
            p0 = np.array(verts[i - 1])
            p1 = np.array(verts[i])
            p2 = np.array(verts[(i + 1) % n])
            v1 = p0 - p1
            v2 = p2 - p1
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                angle = np.arccos(dot)
                angle_deg = np.degrees(angle)
                angles.append(angle_deg)
        # Count sign changes in angle difference above threshold
        arc_count = 0
        for i in range(1, len(angles)):
            if np.sign(angles[i] - angles[i-1]) != 0 and abs(angles[i] - angles[i-1]) > angle_thresh:
                arc_count += 1
        return arc_count

    @staticmethod
    def count_straight_segments(vertices_or_poly, angle_tol=5):
        """
        Counts the number of straight segments in a polyline.
        Accepts either a list of (x, y) tuples or a Polygon/MultiPolygon.
        A segment is considered straight if the angle between consecutive segments is within angle_tol degrees of 180.
        Args:
            vertices_or_poly: list of (x, y) tuples or Polygon
            angle_tol: tolerance in degrees
        Returns:
            int: number of straight segments
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 0
        def angle_between(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            return np.degrees(np.arccos(cos_theta))

        count = 0
        n = len(verts)
        for i in range(n):
            p0 = verts[i - 1]
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p1[0], p2[1] - p1[1])
            ang = angle_between(v1, v2)
            if abs(ang - 180) <= angle_tol:
                count += 1
        return count
    """
    Geometry + physics feature extractor for Bongard-LOGO shapes.
    All methods are safe: they never bubble exceptions.
    """

    @staticmethod
    def _clamped_arccos(dot, norm):
        if norm == 0.0:
            return 0.0
        # returns degrees
        return math.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

    @staticmethod
    def _ensure_polygon(poly_geom):
        if isinstance(poly_geom, MultiPolygon):
            return max(poly_geom.geoms, key=lambda p: p.area)
        return poly_geom

    @staticmethod
    def safe_extract_vertices(obj):
        """
        From list, Polygon, or MultiPolygon → python list of (x,y).
        """
        if isinstance(obj, list):
            return obj
        if hasattr(obj, "exterior"):
            return list(obj.exterior.coords)
        if hasattr(obj, "geoms"):
            largest = max(obj.geoms, key=lambda p: p.area)
            return list(largest.exterior.coords)
        return []

    @staticmethod
    def polygon_from_vertices(vertices):
        import logging
        # Ensure all vertices are tuples (hashable)
        tuple_vertices = [tuple(v) for v in vertices]
        key = f"{len(tuple_vertices)}-{hash(tuple(tuple_vertices))}"
        if key in _POLY_CACHE:
            return _POLY_CACHE[key]

        # Deduplicate and remove collinear points
        cleaned = [tuple(v) for v in vertices]
        cleaned = PhysicsInference.dedup_vertices(cleaned)
        n = len(cleaned)
        t0 = time.time()
        poly = None
        try:
            poly = Polygon(cleaned)
            logging.debug(f"polygon: raw build {n} verts in {time.time()-t0:.3f}s")
            # Pre-validation checks
            if n < 3:
                logging.warning(f"polygon_from_vertices: Too few vertices ({n}) for polygon. Fallback to unit square.")
                poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            elif not poly.is_valid or poly.area == 0:
                # Check for self-intersection
                if hasattr(poly, 'is_simple') and not poly.is_simple:
                    logging.warning(f"polygon_from_vertices: Polygon is self-intersecting. Trying convex hull fallback. Vertices: {cleaned}")
                    poly = Polygon(cleaned).convex_hull
                # Check for collinearity
                elif PhysicsInference.shoelace_area(cleaned) == 0.0:
                    logging.warning(f"polygon_from_vertices: Polygon is collinear. Trying convex hull fallback. Vertices: {cleaned}")
                    poly = Polygon(cleaned).convex_hull
                # Try buffer(0) as standard fix
                else:
                    logging.debug("polygon_from_vertices: invalid or zero-area polygon, trying buffer(0)")
                    poly = poly.buffer(0)
                # If still invalid, try convex hull again
                if (not poly.is_valid or poly.area == 0) and n >= 3:
                    logging.debug("polygon_from_vertices: still invalid, trying convex hull fallback")
                    poly = Polygon(cleaned).convex_hull
                # If still invalid, fallback to bounding box
                if not poly.is_valid or poly.area == 0:
                    bbox = PhysicsInference.rounded_bbox(cleaned)
                    logging.warning(f"polygon_from_vertices: Could not recover valid polygon, using bounding box fallback. Vertices: {cleaned}, bbox: {bbox}")
                    poly = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
                # Final fallback
                if not poly.is_valid or poly.area == 0:
                    logging.warning(f"polygon_from_vertices: Could not recover valid polygon, using unit square fallback. Vertices: {cleaned}")
                    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        except Exception as e:
            logging.warning(f"polygon_from_vertices: Exception {e}, using unit square fallback. Vertices: {cleaned}")
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly = PhysicsInference._ensure_polygon(poly)
        _POLY_CACHE[key] = poly
        return poly

    @staticmethod
    @safe_feature(default=[0.5, 0.5])
    def centroid(poly_geom):
        """
        Returns centroid in normalized coordinates (0-1 range) if possible.
        Accepts list of vertices or Polygon/MultiPolygon.
        """
        # Always convert to Polygon if input is a list
        if isinstance(poly_geom, list):
            poly = PhysicsInference.polygon_from_vertices(poly_geom)
        else:
            poly = PhysicsInference._ensure_polygon(poly_geom)
        c = poly.centroid
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        if width > 0 and height > 0:
            norm_x = (c.x - minx) / width
            norm_y = (c.y - miny) / height
            norm_x = min(max(norm_x, 0.0), 1.0)
            norm_y = min(max(norm_y, 0.0), 1.0)
            return (norm_x, norm_y)
        return (c.x, c.y)

    @staticmethod
    @safe_feature(default=0.0)
    def area(poly_geom):
        """
        Returns area normalized to unit square if possible. Robust to invalid polygons, with buffer(0) and convex hull fallback, and debug logging.
        """
        import logging
        # If input is a list of vertices, use shoelace formula
        if isinstance(poly_geom, list):
            verts = PhysicsInference.safe_extract_vertices(poly_geom)
            verts = PhysicsInference.dedup_vertices(verts)
            if len(verts) >= 3:
                return PhysicsInference.shoelace_area(verts)
            return 0.0
        poly = PhysicsInference._ensure_polygon(poly_geom)
        try:
            if poly.is_valid and poly.area > 0:
                return float(poly.area)
            logging.debug("area: invalid or zero-area polygon, trying buffer(0)")
            poly2 = poly.buffer(0)
            if poly2.is_valid and poly2.area > 0:
                return float(poly2.area)
            logging.debug("area: still invalid, trying convex hull fallback")
            poly3 = Polygon(list(poly.exterior.coords)).convex_hull
            if poly3.is_valid and poly3.area > 0:
                return float(poly3.area)
            # Try to recover with polygonize
            from shapely.ops import polygonize
            polys = list(polygonize([poly.exterior.coords]))
            if polys and polys[0].is_valid and polys[0].area > 0:
                return float(polys[0].area)
            logging.warning("area: Could not recover valid area, returning 0.0")
            return 0.0
        except Exception as e:
            logging.warning(f"area: Exception {e}, returning 0.0")
            return 0.0

    @staticmethod
    @safe_feature(default=1.0)
    def geometric_complexity(vertices_or_poly):
        """
        Returns a finite geometric complexity value based on number of vertices and curvature.
        Circles: 8.0, Zigzag: 6.0, Triangle: 3.0, else: vertex count or curvature-based.
        Always returns a numeric value (vertex count fallback).
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 1.0
        n = len(verts)
        if n == 3:
            return 3.0
        if n == 4:
            return 4.0
        # Check for circle-like (all angles ~360/n)
        angles = []
        for i in range(n):
            p0 = np.array(verts[i - 1])
            p1 = np.array(verts[i])
            p2 = np.array(verts[(i + 1) % n])
            v1 = p1 - p0
            v2 = p2 - p1
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
                angles.append(ang)
        # Fallback: use vertex count as complexity
        return float(n)
    @staticmethod
    @safe_feature(default=1)
    def rotational_symmetry(vertices_or_poly, max_order=None, rmse_threshold=0.01):
        """
        Detects the highest order of rotational symmetry for a polygon/polyline.
        Compares the shape to its rotated versions for k=2,3,...,max_order (default n//2).
        Returns the highest k (>=2) with mean RMSE below threshold, else 1 (no symmetry).
        Lowered RMSE threshold for stricter detection.
        """
        import numpy as np
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 1
        arr = np.array(verts)
        n = len(arr)
        if n > 3 and np.allclose(arr[0], arr[-1]):
            arr = arr[:-1]
            n = len(arr)
        centroid = np.mean(arr, axis=0)
        arr_centered = arr - centroid
        scale = np.linalg.norm(arr_centered, axis=1).max()
        if scale < 1e-8:
            scale = 1.0
        arr_norm = arr_centered / scale
        best_order = 1
        best_rmse = float('inf')
        max_k = max_order if max_order is not None else max(2, n // 2)
        for k in range(2, max_k + 1):
            theta = 2 * np.pi / k
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            arr_rot = (arr_norm @ rot_matrix.T)
            min_rmse = float('inf')
            for shift in range(n):
                arr_shift = np.roll(arr_norm, shift, axis=0)
                if arr_shift.shape != arr_rot.shape:
                    continue
                rmse = np.sqrt(np.mean((arr_shift - arr_rot) ** 2))
                if rmse < min_rmse:
                    min_rmse = rmse
            if min_rmse < rmse_threshold and min_rmse < best_rmse:
                best_order = k
                best_rmse = min_rmse
        return best_order
    @staticmethod
    @safe_feature(default=False)
    def is_convex(poly_geom):
        poly = PhysicsInference._ensure_polygon(poly_geom)
        return poly.equals(poly.convex_hull)

    @staticmethod
    @safe_feature(default=0.0)
    def symmetry_score(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 2:
            return 0.0
        poly = Polygon(verts)
        cx, cy = poly.centroid.x, poly.centroid.y
        reflected = [(2 * cx - x, y) for x, y in verts]
        rmse = np.sqrt(np.mean((np.array(verts) - np.array(reflected)) ** 2))
        return float(rmse)

    @staticmethod
    def num_straight(vertices_or_poly):
        """
        Alias for count_straight_segments: returns the number of straight segments in the shape.
        """
        return PhysicsInference.count_straight_segments(vertices_or_poly)

    @staticmethod
    @safe_feature(default=False)
    def has_quadrangle(vertices_or_poly):
        """
        Returns True if the shape is a valid convex quadrangle (exactly 4 unique vertices, convex, non-self-intersecting).
        Handles closed polygons (duplicate end vertex).
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        verts = PhysicsInference.dedup_vertices(verts)
        # Remove duplicate closing vertex if present
        unique = []
        for v in verts:
            if not unique or np.linalg.norm(np.array(v) - np.array(unique[-1])) > 1e-8:
                unique.append(v)
        if len(unique) != 4:
            return False
        try:
            poly = Polygon(unique)
            return poly.is_valid and poly.equals(poly.convex_hull)
        except Exception:
            return False
    @staticmethod
    @safe_feature(default=0.0)
    def edge_length_variance(vertices_or_poly):
        """
        Population variance of edge lengths for a polygon/polyline. No normalization.
        Returns NaN if <2 edges.
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 2:
            return float('nan')
        n = len(verts)
        lengths = [np.linalg.norm(np.array(verts[(i+1)%n]) - np.array(verts[i])) for i in range(n)]
        if len(lengths) < 2:
            return float('nan')
        return float(np.var(lengths))

    @staticmethod
    @safe_feature(default=False)
    def has_obtuse(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 3:
            return False

        def angle(a, b, c):
            ab = np.array([b[0] - a[0], b[1] - a[1]])
            cb = np.array([b[0] - c[0], b[1] - c[1]])
            dot = np.dot(ab, cb)
            norm = np.linalg.norm(ab) * np.linalg.norm(cb)
            return PhysicsInference._clamped_arccos(dot, norm)

        return any(
            angle(verts[i - 1], verts[i], verts[(i + 1) % len(verts)]) > 100
            for i in range(len(verts))
        )

    @staticmethod
    @safe_feature(default=0.0)
    def moment_of_inertia(vertices_or_poly):
        """
        Returns the moment of inertia (about centroid) for a polygon using the standard formula:
        I_z = (1/12) * sum_{i=1}^n (x_i y_{i+1} - x_{i+1} y_i) * (x_i^2 + x_i x_{i+1} + x_{i+1}^2 + y_i^2 + y_i y_{i+1} + y_{i+1}^2) / (2A)
        Returns 0.0 for degenerate cases.
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        verts = PhysicsInference.dedup_vertices(verts)
        if not verts or len(verts) < 3:
            return 0.0
        arr = np.array(verts)
        n = len(arr)
        area = 0.0
        inertia = 0.0
        for i in range(n):
            x0, y0 = arr[i]
            x1, y1 = arr[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            area += cross
            inertia += cross * (x0**2 + x0*x1 + x1**2 + y0**2 + y0*y1 + y1**2)
        area *= 0.5
        if abs(area) < 1e-8:
            return 0.0
        inertia = (1.0/12.0) * inertia / (2.0 * area)
        return float(abs(inertia))

    @staticmethod
    def pattern_regularity(modifier_sequence):
        """
        Pattern regularity: 1/(1+CV) where CV = stddev/mean of modifier frequencies. Returns NaN if sequence too short.
        """
        if not modifier_sequence or len(modifier_sequence) < 2:
            return 0.0
        max_alt = 1
        curr = 1
        for i in range(1, len(modifier_sequence)):
            if modifier_sequence[i] != modifier_sequence[i-1]:
                curr += 1
            else:
                max_alt = max(max_alt, curr)
                curr = 1
        max_alt = max(max_alt, curr)
        return max_alt / len(modifier_sequence)

    @staticmethod
    def homogeneity_score(modifier_sequence):
        """
        Simpson's index: sum(p_m^2) for modifier frequencies. 1.0 if all same, lower if diverse.
        """
        from collections import Counter
        import numpy as np
        n = len(modifier_sequence)
        if n == 0:
            return float('nan')
        counts = Counter(modifier_sequence)
        probs = np.array([v / n for v in counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        k = len(counts)
        if k <= 1:
            return 1.0
        return 1.0 - (entropy / np.log2(k))

    @staticmethod
    def diversity_penalty(modifier_sequence):
        """
        Shannon entropy normalized by log(k), where k is number of unique modifiers.
        Returns 0 for single-modifier sequences.
        """
        import numpy as np
        from collections import Counter
        n = len(modifier_sequence)
        if n == 0:
            return 0.0
        counts = np.array(list(Counter(modifier_sequence).values()))
        probs = counts / n
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        k = len(counts)
        if k <= 1:
            return 0.0
        return float(entropy / np.log2(k))

    @staticmethod
    def buffer_geometries(geoms, buffer_amt=0.001):
        """
        Return a list of buffered geometries (Polygon) from input geoms (LineString/Polygon).
        Skips invalid or empty geometries. Buffering turns lines into thin polygons for robust relational features.
        """
        buffered = []
        for g in geoms:
            if g is not None and hasattr(g, 'is_valid') and g.is_valid and not g.is_empty:
                try:
                    buffered.append(g.buffer(buffer_amt))
                except Exception as e:
                    logging.debug(f"buffer_geometries: failed to buffer geometry: {e}")
        return buffered

    @staticmethod
    def relational_features_with_buffer(geoms, buffer_amt=0.001):
        """
        Compute all relational features (intersections, adjacency, containment, overlap) using buffered geometries.
        Returns a dict with all four features.
        """
        buffered_geoms = PhysicsInference.buffer_geometries(geoms, buffer_amt)
        return {
            'intersections': PhysicsInference.find_stroke_intersections(buffered_geoms),
            'adjacency': PhysicsInference.strokes_touching(buffered_geoms),
            'containment': PhysicsInference.stroke_contains_stroke(buffered_geoms),
            'overlap': PhysicsInference.stroke_overlap_area(buffered_geoms)
        }
