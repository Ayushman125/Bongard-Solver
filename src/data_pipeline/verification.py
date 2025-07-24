import logging

class Verification:
    @staticmethod
    def has_quadrangle(poly):
        # True if convex hull has 4 sides
        try:
            hull = poly.convex_hull
            coords = list(hull.exterior.coords)
            return len(coords) - 1 == 4
        except Exception:
            return False

    @staticmethod
    def has_obtuse_angle(poly):
        # True if any interior angle > 90°
        import math
        coords = list(poly.exterior.coords)
        n = len(coords) - 1
        for i in range(n):
            a, b, c = coords[i-1], coords[i], coords[(i+1)%n]
            ab = (b[0]-a[0], b[1]-a[1])
            cb = (b[0]-c[0], b[1]-c[1])
            dot = ab[0]*cb[0] + ab[1]*cb[1]
            norm = math.hypot(*ab) * math.hypot(*cb)
            if norm == 0:
                continue
            angle = math.degrees(math.acos(dot/norm))
            if angle > 90:
                return True
        return False
    @staticmethod
    def validate_polygon(poly):
        if not poly.is_valid:
            logging.warning(f"Invalid polygon detected: {poly.wkt}")
            return False
        return True

    @staticmethod
    def log_for_review(problem_id, filename, reason, review_log='data/flagged_cases.txt'):
        with open(review_log, 'a') as f:
            f.write(f"{problem_id},{filename},{reason}\n")
