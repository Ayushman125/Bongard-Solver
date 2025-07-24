

import math
import numpy as np
from shapely.geometry import Polygon
import pymunk

def safe_acos(x):
    """Clamp to [–1,1], return arccos for scalar or array."""
    return np.arccos(np.clip(x, -1.0, 1.0))

class PhysicsInference:
    @staticmethod
    def _clamped_arccos(dot, norm):
        if norm == 0:
            return 0.0
        return np.degrees(safe_acos(dot / norm))
    @staticmethod
    def _ensure_polygon(poly_geom):
        from shapely.geometry import MultiPolygon, Polygon
        if isinstance(poly_geom, MultiPolygon):
            # pick the largest area polygon
            return max(poly_geom.geoms, key=lambda p: p.area)
        return poly_geom

    @staticmethod
    def count_straight_segments(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not vertices or len(vertices) < 2:
            return 0
        count = 0
        for i in range(1, len(vertices)):
            x0, y0 = vertices[i-1]
            x1, y1 = vertices[i]
            # If delta x or y is much larger than the other, treat as straight
            if abs(x1-x0) > 0.01 or abs(y1-y0) > 0.01:
                count += 1
        return count

    @staticmethod
    def count_arcs(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if not vertices or len(vertices) < 2:
            return 0
        total = len(vertices) - 1
        straight = PhysicsInference.count_straight_segments(vertices)
        return max(0, total - straight)

    @staticmethod
    def safe_extract_vertices(obj):
        """Safely extract vertices from a list, Polygon, or MultiPolygon."""
        from shapely.geometry import Polygon, MultiPolygon
        if isinstance(obj, list):
            return obj
        elif hasattr(obj, 'exterior'):
            return list(obj.exterior.coords)
        elif hasattr(obj, 'geoms'):
            # MultiPolygon: use largest geometry
            largest = max(obj.geoms, key=lambda p: p.area)
            return list(largest.exterior.coords)
        else:
            return []

    @staticmethod
    def polygon_from_vertices(vertices):
        from shapely.geometry import Polygon, MultiPoint
        try:
            poly = Polygon(vertices)
            if not poly.is_valid or poly.area == 0:
                poly = poly.buffer(0)
            if not poly.is_valid or poly.area == 0:
                hull = MultiPoint(vertices).convex_hull
                if hull.geom_type != "Polygon":
                    hull = Polygon([(0,0),(1,0),(1,1),(0,1)])
                poly = hull
            return PhysicsInference._ensure_polygon(poly)
        except Exception:
            return Polygon([(0,0),(1,0),(1,1),(0,1)])

    @staticmethod
    def centroid(poly_geom):
        poly = PhysicsInference._ensure_polygon(poly_geom)
        c = poly.centroid
        return [c.x, c.y]

    @staticmethod
    def area(poly_geom):
        return PhysicsInference._ensure_polygon(poly_geom).area

    @staticmethod
    def is_convex(poly_geom):
        poly = PhysicsInference._ensure_polygon(poly_geom)
        return poly.equals(poly.convex_hull)

    @staticmethod
    def symmetry_score(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 2:
            return 0.0
        from shapely.geometry import Polygon
        poly = Polygon(vertices)
        cx, cy = poly.centroid.x, poly.centroid.y
        reflected = [(2*cx - x, y) for (x, y) in vertices]
        orig = np.array(vertices)
        refl = np.array(reflected)
        rmse = np.sqrt(np.mean((orig - refl) ** 2))
        return rmse

    @staticmethod
    def num_straight(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        return len(vertices)

    @staticmethod
    def has_quadrangle(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        from shapely.geometry import Polygon
        poly = Polygon(vertices)
        return len(vertices) >= 4 and poly.is_valid and PhysicsInference.is_convex(poly)

    @staticmethod
    def has_obtuse(vertices_or_poly):
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 3:
            return False
        def angle(a, b, c):
            ab = np.array([b[0]-a[0], b[1]-a[1]])
            cb = np.array([b[0]-c[0], b[1]-c[1]])
            dot = np.dot(ab, cb)
            norm = np.linalg.norm(ab) * np.linalg.norm(cb)
            return 0.0 if norm == 0 else np.degrees(safe_acos(dot / norm))
        # If any interior angle >100°, we have an obtuse polygon
        return any(
            angle(vertices[i-1], vertices[i], vertices[(i+1)%len(vertices)]) > 100
            for i in range(len(vertices))
        )

    @staticmethod
    def moment_of_inertia(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 3:
            return 0.0
        try:
            coords = [(float(x), float(y)) for (x, y) in verts]
            return pymunk.moment_for_poly(1.0, coords)
        except ValueError:
            from shapely.geometry import Polygon
            poly = PhysicsInference.polygon_from_vertices(verts)
            if poly and not poly.is_empty:
                minx, miny, maxx, maxy = poly.bounds
                dx, dy = maxx - minx, maxy - miny
                return (dx*dx + dy*dy) / 12.0
            return 0.0