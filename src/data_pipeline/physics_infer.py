#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All imports and logic preserved. Fixed indentation and removed duplicate code.
"""
from shapely.geometry import Polygon, MultiPolygon
import functools
import logging
import math
import time
import numpy as np
import pymunk

# cache cleaned Polygon results by vertex tuple
_POLY_CACHE: dict[str, Polygon] = {}

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
        # cache key based on all vertices
        key = f"{len(vertices)}-{hash(tuple(vertices))}"
        if key in _POLY_CACHE:
            return _POLY_CACHE[key]

        # clean once (placeholder)
        def _clean_vertices(verts):
            return verts
        cleaned = _clean_vertices(vertices)
        n_verts = len(cleaned)

        # raw polygon creation
        t0_poly = time.time()
        poly = Polygon(cleaned)
        t1_poly = time.time()
        logging.debug(f"polygon_from_vertices: raw Polygon({n_verts}) creation took {t1_poly-t0_poly:.3f}s")

        # only repair small polygons
        repaired = False
        if not poly.is_valid:
            if n_verts < 500:
                t2 = time.time()
                poly = poly.buffer(0)
                repaired = True
                t3 = time.time()
                logging.debug(f"polygon_from_vertices: buffer(0) repair on {n_verts} vertices took {t3-t2:.3f}s (valid={poly.is_valid})")
            else:
                logging.debug(f"polygon_from_vertices: skipped buffer(0) repair for {n_verts} vertices (too large)")

        # If MultiPolygon, pick largest piece
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda p: p.area)

        # Simplify once, only for small polygons
        if n_verts < 500:
            t4 = time.time()
            poly = poly.simplify(0.5, preserve_topology=True)
            t5 = time.time()
            logging.debug(f"polygon_from_vertices: simplify on {n_verts} vertices took {t5-t4:.3f}s")
        else:
            logging.debug(f"polygon_from_vertices: skipped simplify for {n_verts} vertices (too large)")

        # Too small → tiny square fallback
        MIN_AREA = 5.0
        if poly.is_empty or poly.area < MIN_AREA:
            logging.debug(f"polygon_from_vertices: fallback to tiny square (area={getattr(poly, 'area', 'N/A')})")
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        # Aspect‐ratio check → fallback if extreme sliver
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny
        MAX_ASPECT = 20.0
        aspect = max(w / (h + 1e-6), h / (w + 1e-6))
        if aspect > MAX_ASPECT:
            logging.debug(f"polygon_from_vertices: fallback to tiny square (aspect={aspect:.2f})")
            poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        t6 = time.time()
        logging.debug(f"polygon_from_vertices: total time {t6-t1_poly:.3f}s (repaired={repaired})")

        _POLY_CACHE[key] = PhysicsInference._ensure_polygon(poly)
        return _POLY_CACHE[key]

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
