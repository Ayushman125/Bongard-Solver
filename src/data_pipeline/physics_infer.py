def _clean_vertices(verts, max_len=2000, tol=1e-6):
    """Remove consecutive duplicates and subsample long chains."""
    cleaned = []
    last = None
    for x, y in verts:
        if last is None or (abs(x - last[0]) > tol or abs(y - last[1]) > tol):
            cleaned.append((x, y))
            last = (x, y)
        if len(cleaned) >= max_len:
            break
    # Ensure at least 4 distinct points
    if len(cleaned) < 4:
        cleaned = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return cleaned
import functools
import logging
import math

import numpy as np
import pymunk
from shapely.geometry import Polygon, MultiPoint, MultiPolygon


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
    def count_straight_segments(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 2:
            return 0
        cnt = 0
        for (x0, y0), (x1, y1) in zip(verts, verts[1:]):
            if abs(x1 - x0) > 0.01 or abs(y1 - y0) > 0.01:
                cnt += 1
        return cnt

    @staticmethod
    def count_arcs(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 2:
            return 0
        return max(0, len(verts) - 1 - PhysicsInference.count_straight_segments(verts))

    @staticmethod
    def polygon_from_vertices(vertices):
        """
        Build, repair, and validate a polygon from raw vertices.
        Always returns a valid Polygon (tiny‐square fallback if pathological).
        """
        MIN_AREA = 5.0
        MAX_ASPECT = 20.0

        try:
            # Clean vertices before passing to Shapely
            cleaned_verts = _clean_vertices(vertices)
            n_verts = len(cleaned_verts)
            poly = Polygon(cleaned_verts)

            # Repair self-intersections only for small polygons
            if not poly.is_valid and n_verts < 500:
                poly = poly.buffer(0)

            # If MultiPolygon, pick largest piece
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)

            # Simplify tiny oscillations
            poly = poly.simplify(0.5, preserve_topology=True)

            # Too small → tiny square fallback
            if poly.is_empty or poly.area < MIN_AREA:
                return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

            # Aspect‐ratio check → fallback if extreme sliver
            minx, miny, maxx, maxy = poly.bounds
            w, h = maxx - minx, maxy - miny
            aspect = max(w / (h + 1e-6), h / (w + 1e-6))
            if aspect > MAX_ASPECT:
                return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

            # Good polygon
            return PhysicsInference._ensure_polygon(poly)

        except Exception as e:
            logging.warning(
                f"polygon_from_vertices crashed ({e!r}); returning tiny‐square fallback"
            )
            return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    @staticmethod
    @safe_feature(default=[0.0, 0.0])
    def centroid(poly_geom):
        poly = PhysicsInference._ensure_polygon(poly_geom)
        c = poly.centroid
        return [c.x, c.y]

    @staticmethod
    @safe_feature(default=0.0)
    def area(poly_geom):
        return PhysicsInference._ensure_polygon(poly_geom).area

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
        return len(PhysicsInference.safe_extract_vertices(vertices_or_poly))

    @staticmethod
    @safe_feature(default=False)
    def has_quadrangle(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        poly = Polygon(verts)
        return len(verts) >= 4 and poly.is_valid and PhysicsInference.is_convex(poly)

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
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 3:
            return 0.0
        try:
            coords = [(float(x), float(y)) for x, y in verts]
            return pymunk.moment_for_poly(1.0, coords)
        except Exception:
            poly = PhysicsInference.polygon_from_vertices(verts)
            if not poly.is_empty:
                minx, miny, maxx, maxy = poly.bounds
                dx, dy = maxx - minx, maxy - miny
                # inertia of unit-mass rectangle: (w^2 + h^2) / 12
                return (dx * dx + dy * dy) / 12.0
            return 0.0