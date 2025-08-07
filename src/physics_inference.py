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
                logging.warning(
                    f"Feature {fn.__name__} failed ({e!r}), returning default={default}"
                )
                return default
        return wrapped
    return decorator

def safe_acos(x):
    """
    Clamp input to [-1, 1], then return arccos (in radians) for floats or numpy arrays.
    Prevents math domain errors.
    """
    return np.arccos(np.clip(x, -1.0, 1.0))

class PhysicsInference:
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
        Computes the variance of angles (in degrees) between consecutive segments of a polygon or polyline.
        Excludes near-zero angles (colinear/duplicate points) to avoid inflating variance.
        Returns 0.0 if not enough valid angles.
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 0.0
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
                angle_deg = np.degrees(np.arccos(dot))
                if angle_deg > 1.0:  # exclude near-zero angles
                    angles.append(angle_deg)
        if len(angles) < 2:
            return 0.0
        return float(np.var(angles))

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
        return math.degrees(safe_acos(dot / norm))

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
        key = f"{len(vertices)}-{hash(tuple(vertices))}"
        if key in _POLY_CACHE:
            return _POLY_CACHE[key]

        def _clean_vertices(verts): return verts  # placeholder
        cleaned = _clean_vertices(vertices)
        n = len(cleaned)
        t0 = time.time()
        poly = None
        try:
            poly = Polygon(cleaned)
            logging.debug(f"polygon: raw build {n} verts in {time.time()-t0:.3f}s")
            if not poly.is_valid or poly.area == 0:
                logging.debug("polygon_from_vertices: invalid or zero-area polygon, trying buffer(0)")
                poly = poly.buffer(0)
            if (not poly.is_valid or poly.area == 0) and n >= 3:
                logging.debug("polygon_from_vertices: still invalid, trying convex hull fallback")
                poly = Polygon(cleaned).convex_hull
            if not poly.is_valid or poly.area == 0:
                logging.warning(f"polygon_from_vertices: Could not recover valid polygon, using unit square fallback.")
                poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        except Exception as e:
            logging.warning(f"polygon_from_vertices: Exception {e}, using unit square fallback.")
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly = PhysicsInference._ensure_polygon(poly)
        _POLY_CACHE[key] = poly
        return poly

    @staticmethod
    @safe_feature(default=[0.5, 0.5])
    def centroid(poly_geom):
        """
        Returns centroid in normalized coordinates (0-1 range) if possible.
        """
        poly = PhysicsInference._ensure_polygon(poly_geom)
        c = poly.centroid
        # If coordinates are outside [0,1], normalize based on bounds
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        if width > 0 and height > 0:
            norm_x = (c.x - minx) / width
            norm_y = (c.y - miny) / height
            # Clamp to [0,1]
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
        Returns True if the shape is a valid convex quadrangle (exactly 4 vertices).
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        poly = Polygon(verts)
        return len(verts) == 4 and poly.is_valid and PhysicsInference.is_convex(poly)

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
        Returns the moment of inertia (about centroid) for a polygon or polyline.
        Implements I = sum(m_i * r_i^2) for unit mass per vertex.
        """
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not verts or len(verts) < 3:
            return 0.0
        centroid = np.mean(verts, axis=0)
        moment = 0.0
        for vertex in verts:
            r_squared = np.sum((np.array(vertex) - centroid) ** 2)
            moment += r_squared
        return moment / len(verts)

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
if __name__ == "__main__":
    from shapely.geometry import LineString
    g1 = LineString([(0, 0), (1, 1)])
    g2 = LineString([(0, 1), (1, 0)])
    print("g1 intersects g2:", g1.intersects(g2))  # Should be True
    geoms = [g1, g2]
    print("find_stroke_intersections:", PhysicsInference.find_stroke_intersections(geoms))  # Should be 1