import os

class Attributes:
    @staticmethod
    def extract_problem_type(problem_path):
        # Assumes path includes Freeform, Basic, or Abstract folder
        for pt in ['Freeform', 'Basic', 'Abstract']:
            if pt.lower() in problem_path.lower():
                return pt
        return 'Unknown'

    @staticmethod
    def enrich_problem_dict(problem_dict, problem_path):
        problem_type = Attributes.extract_problem_type(problem_path)
        problem_dict['problem_type'] = problem_type
        return problem_dict
