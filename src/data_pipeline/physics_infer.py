from shapely.geometry import Polygon
import pymunk
import numpy as np

class PhysicsInference:
    @staticmethod
    def polygon_from_vertices(vertices):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        try:
            poly = Polygon(vertices)
            if not poly or not poly.is_valid or poly.is_empty or poly.area == 0:
                # Try to fix with buffer(0)
                poly = poly.buffer(0)
                if not poly.is_valid or poly.is_empty or poly.area == 0:
                    from shapely.geometry import MultiPoint
                    poly = MultiPoint(vertices).convex_hull
                    # If still invalid, buffer a tiny amount
                    if not poly.is_valid:
                        poly = poly.buffer(0.1)
                if poly.is_empty or poly.area == 0:
                    if vertices:
                        x0, y0 = vertices[0]
                        poly = Polygon([(x0, y0), (x0+1, y0), (x0+1, y0+1), (x0, y0+1)])
                    else:
                        return None
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
