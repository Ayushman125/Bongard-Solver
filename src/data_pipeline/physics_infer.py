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
    Clamp input to [-1, 1], return arccos for floats or numpy arrays.
    Prevents math domain errors.
    """
    return np.arccos(np.clip(x, -1.0, 1.0))


class PhysicsInference:
    """
    Compute geometry and physics-derived attributes for
    polygonal shapes in the Bongard-LOGO dataset.
    """

    @staticmethod
    def _clamped_arccos(dot, norm):
        if norm == 0.0:
            return 0.0
        return math.degrees(safe_acos(dot / norm))

    @staticmethod
    def _ensure_polygon(poly_geom):
        """
        If given a MultiPolygon, return its largest component.
        Otherwise, pass through.
        """
        if isinstance(poly_geom, MultiPolygon):
            return max(poly_geom.geoms, key=lambda p: p.area)
        return poly_geom

    @staticmethod
    def safe_extract_vertices(obj):
        """
        Extract a list of (x,y) coordinates from:
        - a Python list of vertices
        - a shapely Polygon
        - a shapely MultiPolygon (largest exterior)
        """
        if isinstance(obj, list):
            return obj

        if hasattr(obj, 'exterior'):
            # Polygon or GeoSeries element
            return list(obj.exterior.coords)

        if hasattr(obj, 'geoms'):
            # MultiPolygon
            largest = max(obj.geoms, key=lambda p: p.area)
            return list(largest.exterior.coords)

        return []

    @staticmethod
    def count_straight_segments(vertices_or_poly):
        """
        Count edges whose delta-x or delta-y exceeds a small threshold.
        """
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 2:
            return 0

        count = 0
        for (x0, y0), (x1, y1) in zip(vertices[:-1], vertices[1:]):
            if abs(x1 - x0) > 0.01 or abs(y1 - y0) > 0.01:
                count += 1
        return count

    @staticmethod
    def count_arcs(vertices_or_poly):
        """
        Count edges likely part of curves: total edges minus straight segments.
        """
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 2:
            return 0

        total = len(vertices) - 1
        straight = PhysicsInference.count_straight_segments(vertices)
        return max(0, total - straight)

    @staticmethod
    def polygon_from_vertices(vertices):
        """
        Build, repair, and validate a polygon from raw vertices.
        Always returns a valid Polygon (or tiny square fallback).
        """
        MIN_AREA = 5.0
        MAX_ASPECT = 20.0

        try:
            raw = Polygon(vertices)

            # Repair self-intersections
            if not raw.is_valid:
                raw = raw.buffer(0)

            # If MultiPolygon, pick the largest
            if isinstance(raw, MultiPolygon):
                raw = max(raw.geoms, key=lambda p: p.area)

            # Simplify tiny wiggles
            raw = raw.simplify(0.5, preserve_topology=True)

            # Fallback tiny square if too small
            if raw.is_empty or raw.area < MIN_AREA:
                return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

            # Aspect‐ratio check: flag extreme slivers
            minx, miny, maxx, maxy = raw.bounds
            w, h = maxx - minx, maxy - miny
            aspect = max(w / (h + 1e-6), h / (w + 1e-6))
            if aspect > MAX_ASPECT:
                return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

            # Return the largest valid piece
            return PhysicsInference._ensure_polygon(raw)

        except Exception as e:
            logging.warning(
                f"polygon_from_vertices crashed ({e!r}); "
                "returning tiny-square fallback"
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
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 2:
            return 0.0

        poly = Polygon(vertices)
        cx, cy = poly.centroid.x, poly.centroid.y
        reflected = [(2*cx - x, y) for (x, y) in vertices]

        orig = np.array(vertices)
        refl = np.array(reflected)
        rmse = np.sqrt(np.mean((orig - refl) ** 2))
        return float(rmse)

    @staticmethod
    def num_straight(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        return len(vertices)

    @staticmethod
    @safe_feature(default=False)
    def has_quadrangle(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        poly = Polygon(vertices)
        return len(vertices) >= 4 and poly.is_valid and PhysicsInference.is_convex(poly)

    @staticmethod
    @safe_feature(default=False)
    def has_obtuse(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 3:
            return False

        def angle(a, b, c):
            ab = np.array([b[0]-a[0], b[1]-a[1]])
            cb = np.array([b[0]-c[0], b[1]-c[1]])
            dot = np.dot(ab, cb)
            norm = np.linalg.norm(ab) * np.linalg.norm(cb)
            return 0.0 if norm == 0 else PhysicsInference._clamped_arccos(dot, norm)

        return any(
            angle(vertices[i-1], vertices[i], vertices[(i+1) % len(vertices)]) > 100
            for i in range(len(vertices))
        )

    @staticmethod
    @safe_feature(default=0.0)
    def moment_of_inertia(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 3:
            return 0.0

        try:
            coords = [(float(x), float(y)) for (x, y) in verts]
            return pymunk.moment_for_poly(1.0, coords)
        except Exception:
            poly = PhysicsInference.polygon_from_vertices(verts)
            if not poly.is_empty:
                minx, miny, maxx, maxy = poly.bounds
                dx, dy = maxx - minx, maxy - miny
                # Inertia of unit-mass rectangle: (w² + h²) / 12
                return (dx*dx + dy*dy) / 12.0
            return 0.0