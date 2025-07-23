import os

class BongardLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def iter_problems(self, problem_type=None):
        # problem_type: "Freeform", "Basic", or "Abstract"
        types = [problem_type] if problem_type else ['Freeform', 'Basic', 'Abstract']
        for pt in types:
            base_dir = os.path.join(self.root_dir, pt)
            if not os.path.isdir(base_dir):
                continue
            for problem_id in sorted(os.listdir(base_dir)):
                problem_path = os.path.join(base_dir, problem_id)
                if not os.path.isdir(problem_path):
                    continue
                pos_dir = os.path.join(problem_path, 'category_1')
                neg_dir = os.path.join(problem_path, 'category_0')
                positives = [
                    {'image_path': os.path.join(pos_dir, f), 'label': 1, 'problem_id': problem_id}
                    for f in os.listdir(pos_dir) if f.endswith('.png')
                ]
                negatives = [
                    {'image_path': os.path.join(neg_dir, f), 'label': 0, 'problem_id': problem_id}
                    for f in os.listdir(neg_dir) if f.endswith('.png')
                ]
                yield {'problem_id': problem_id, 'positives': positives, 'negatives': negatives}
