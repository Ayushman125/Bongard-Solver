from shapely.geometry import Polygon
import pymunk
import numpy as np

class PhysicsInference:
    @staticmethod
    def _ensure_polygon(poly_geom):
        from shapely.geometry import MultiPolygon, Polygon
        if isinstance(poly_geom, MultiPolygon):
            # pick the largest area polygon
            return max(poly_geom.geoms, key=lambda p: p.area)
        return poly_geom
    @staticmethod
    def count_straight_segments(vertices):
        # Heuristic: count segments with nearly constant angle
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
    def count_arcs(vertices):
        # Heuristic: treat non-straight segments as arcs
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
        from shapely.ops import unary_union
        try:
            poly = Polygon(vertices)
            if not poly or not poly.is_valid or poly.is_empty or poly.area == 0:
                # Try buffer(0) to fix minor self-intersections
                poly = poly.buffer(0)
                if not poly.is_valid or poly.is_empty or poly.area == 0:
                    # Fallback to convex hull
                    hull = MultiPoint(vertices).convex_hull
                    if hull.geom_type == "Point":
                        x, y = hull.x, hull.y
                        hull = Polygon([(x, y), (x+1, y), (x+1, y+1), (x, y+1)])
                    poly = hull
            poly = PhysicsInference._ensure_polygon(poly)
            return poly
        except Exception as e:
            print(f"Polygon creation failed: {e}")
            return None

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
        # Count number of straight line segments in the LOGO program
        # This should be computed from the original command list, but as a fallback, estimate from vertices
        # For best results, compute from LOGO commands in logo_to_shape.py
        return len(vertices_or_poly) if isinstance(vertices_or_poly, list) else len(PhysicsInference.safe_extract_vertices(vertices_or_poly))

    @staticmethod
    def has_quadrangle(vertices_or_poly):
        # Placeholder: true if there are 4+ vertices and the shape is convex
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        from shapely.geometry import Polygon
        poly = Polygon(vertices)
        return len(vertices) >= 4 and poly.is_valid and PhysicsInference.is_convex(poly)

    @staticmethod
    def has_obtuse(vertices_or_poly):
        # Placeholder: true if any angle > 100 degrees
        vertices = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(vertices) < 3:
            return False
        def angle(a, b, c):
            import math
            ab = np.array([b[0]-a[0], b[1]-a[1]])
            cb = np.array([b[0]-c[0], b[1]-c[1]])
            dot = np.dot(ab, cb)
            norm = np.linalg.norm(ab) * np.linalg.norm(cb)
            if norm == 0:
                return 0
            return np.degrees(np.arccos(dot / norm))
        for i in range(len(vertices)):
            a, b, c = vertices[i-1], vertices[i], vertices[(i+1)%len(vertices)]
            if angle(a, b, c) > 100:
                return True
        return False

    @staticmethod
    def moment_of_inertia(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 3:
            return 0.0
        moment = pymunk.moment_for_poly(1.0, [(float(x), float(y)) for (x, y) in verts])
        return moment
