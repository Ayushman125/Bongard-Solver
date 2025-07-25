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
        key = f"{len(vertices)}-{hash(tuple(vertices))}"
        if key in _POLY_CACHE:
            return _POLY_CACHE[key]

        # one-time clean + build
        def _clean_vertices(verts): return verts  # placeholder
        cleaned = _clean_vertices(vertices)
        n = len(cleaned)

        # raw polygon
        t0 = time.time()
        poly = Polygon(cleaned)
        logging.debug("polygon: raw build %d verts in %.3fs", n, time.time()-t0)

        # small-polygon repair (fix: use n < 20 for fallback, as in snippet)
        if not poly.is_valid and n < 20:
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        poly = PhysicsInference._ensure_polygon(poly)
        _POLY_CACHE[key] = poly
        return poly

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
                return (dx * dx + dy * dy) / 12.0
            return 0.0
