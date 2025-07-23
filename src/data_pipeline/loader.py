import os
import json



import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.problem_folder_utils import get_problem_folders

class BongardLoader:
    def __init__(self, root_dir, problems_list=None, n_select=50):
        self.root_dir = root_dir
        self.problems_list = problems_list
        self.n_select = n_select

    def get_image_dirs(self):
        # Return ff, bd, hd image directories
        return [
            os.path.join(self.root_dir, 'ff', 'images'),
            os.path.join(self.root_dir, 'bd', 'images'),
            os.path.join(self.root_dir, 'hd', 'images'),
        ]

    def iter_problems(self, limit=None):
        # Enumerate problem folders directly, select random subset, ignore missing metadata
        all_folders = get_problem_folders(self.get_image_dirs(), self.n_select)
        allowed_ids = set()
        if self.problems_list:
            with open(self.problems_list) as f:
                allowed_ids = set(line.strip() for line in f.readlines())
        count = 0
        for folder in all_folders:
            pid = os.path.basename(folder)
            if allowed_ids and pid not in allowed_ids:
                continue
            if limit and count >= limit:
                break
            images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            positives = [{'image_path': img, 'label': 1, 'problem_id': pid} for img in images]
            negatives = []  # Negatives can be generated elsewhere if needed
            yield {'problem_id': pid, 'positives': positives, 'negatives': negatives, 'category': None, 'concept_name': None}
            count += 1
