import logging

class Verification:
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
