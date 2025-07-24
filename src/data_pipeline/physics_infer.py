from shapely.geometry import Polygon
import pymunk
import numpy as np

class PhysicsInference:
    @staticmethod
    def safe_extract_vertices(obj):
        """Safely extract vertices from a list, Polygon, or MultiPolygon."""
        from shapely.geometry import Polygon, MultiPolygon
        if isinstance(obj, list):
            return obj
        elif hasattr(obj, 'exterior'):
            return list(obj.exterior.coords)
        elif hasattr(obj, 'geoms'):
            # MultiPolygon: use first geometry
            return list(obj.geoms[0].exterior.coords)
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
            return poly
        except Exception as e:
            print(f"Polygon creation failed: {e}")
            return None

    @staticmethod
    def centroid(poly):
        return (poly.centroid.x, poly.centroid.y)

    @staticmethod
    def area(poly):
        return poly.area

    @staticmethod
    def is_convex(poly):
        # Compare polygon with convex hull
        return poly.convex_hull.equals(poly)

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
    def moment_of_inertia(vertices_or_poly):
        verts = PhysicsInference.safe_extract_vertices(vertices_or_poly)
        if len(verts) < 3:
            return 0.0
        moment = pymunk.moment_for_poly(1.0, [(float(x), float(y)) for (x, y) in verts])
        return moment
