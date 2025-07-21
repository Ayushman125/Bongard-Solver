import os
import json
import numpy as np

class BongardLogoProblem:
    """
    Loader for real Bongard-LOGO problems from ShapeBongard_V2.
    Usage:
        problem = BongardLogoProblem(split, problem_name)
        program = problem.program  # list of shape dicts
    """
    def __init__(self, split, problem_name, root_dir=None):
        # split: 'ff', 'bd', or 'hd'
        # problem_name: e.g. 'ff_nact9_0232'
        # root_dir: path to ShapeBongard_V2 (default: cwd/ShapeBongard_V2)
        if root_dir is None:
            root_dir = os.path.join(os.getcwd(), 'ShapeBongard_V2')
        self.split = split
        self.problem_name = problem_name
        self.root_dir = root_dir
        self.program = self._load_program()

    def _load_program(self):
        # Find the JSON file for the split
        json_path = os.path.join(self.root_dir, self.split, f"{self.split}_action_programs.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Program JSON not found: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        if self.problem_name not in data:
            raise KeyError(f"Problem {self.problem_name} not found in {json_path}")
        symbolic_program = data[self.problem_name]
        # symbolic_program is a nested list of lists of tokens; flatten and parse
        return self._parse_symbolic_program(symbolic_program)

    def _parse_symbolic_program(self, symbolic_program):
        # symbolic_program: nested list of lists of tokens (strings)
        # Output: list of shape dicts for the renderer
        # Flatten all tokens
        tokens = []
        def flatten(l):
            for el in l:
                if isinstance(el, list):
                    yield from flatten(el)
                else:
                    yield el
        for token in flatten(symbolic_program):
            if isinstance(token, str):
                tokens.append(token)
        # Parse each token
        shape_dicts = [self._parse_action_token(t) for t in tokens]
        return shape_dicts

    def _parse_action_token(self, token):
        # Example token: 'line_circle_1.000-0.500'
        # This is a placeholder parser. Adjust as needed for your real encoding.
        # We'll split by '_' and '-' and assign to shape, size, etc.
        parts = token.split('_')
        shape = parts[0] if len(parts) > 0 else 'unknown'
        subtype = parts[1] if len(parts) > 1 else ''
        params = parts[2] if len(parts) > 2 else ''
        # Try to extract size from params (e.g., '1.000-0.500')
        size = 32
        if params:
            try:
                size = int(float(params.split('-')[0]) * 64)
            except Exception:
                size = 32
        # Compose dict
        return {
            'shape': shape,
            'subtype': subtype,
            'size': size,
            'color': 'black',
            'fill': 'solid',
            'x': 64,
            'y': 64,
            'rotation': 0
        }
