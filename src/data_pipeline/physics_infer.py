from shapely.geometry import Polygon
import pymunk
import numpy as np

class PhysicsInference:
    @staticmethod
    def polygon_from_vertices(vertices):
        try:
            poly = Polygon(vertices)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Attempt to fix invalid polygons
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
    def symmetry_score(vertices):
        poly = Polygon(vertices)
        cx, cy = poly.centroid.x, poly.centroid.y
        reflected = [(2*cx - x, y) for (x, y) in vertices]
        orig = np.array(vertices)
        refl = np.array(reflected)
        rmse = np.sqrt(np.mean((orig - refl) ** 2))
        return rmse

    @staticmethod
    def moment_of_inertia(vertices):
        verts = [(float(x), float(y)) for (x, y) in vertices]
        moment = pymunk.moment_for_poly(1.0, verts)
        return moment
