"""
Central concept registry for Bongard-Solver.
Auto-populates from dataset problem IDs and enforces coverage.
Provides factory functions for each problem type (freeform, basic, abstract).
Hard failure if a concept is missing.
"""
import os
import sys
import importlib
import importlib.util

# Dynamically import concept functions from src/concepts.py
spec = importlib.util.spec_from_file_location("concepts", os.path.join(os.path.dirname(__file__), "..", "concepts.py"))
concepts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(concepts_mod)
has_four_straight_lines = concepts_mod.has_four_straight_lines
exists_quadrangle = concepts_mod.exists_quadrangle
has_obtuse_angle = concepts_mod.has_obtuse_angle

# Import all concept functions from concepts.py (extend as needed)
ALL_CONCEPTS = {
    'has_four_straight_lines': has_four_straight_lines,
    'exists_quadrangle': exists_quadrangle,
    'has_obtuse_angle': has_obtuse_angle,
    # ... add more as implemented ...
}

# Problem type to dataset folder mapping
PROBLEM_TYPE_DIRS = {
    'ff': 'data/raw/ShapeBongard_V2/ff/images',
    'bd': 'data/raw/ShapeBongard_V2/bd/images',
    'hd': 'data/raw/ShapeBongard_V2/hd/images',
}

# Map problem type to factory function
FACTORIES = {}

class ConceptRegistry:
    def __init__(self):
        self.problem_to_concept = {}
        self._populate_registry()

    def _populate_registry(self):
        for ptype, folder in PROBLEM_TYPE_DIRS.items():
            if not os.path.exists(folder):
                continue
            for pid in os.listdir(folder):
                if not os.path.isdir(os.path.join(folder, pid)):
                    continue
                # Example: parse concept name from pid (customize as needed)
                concept_name = self._parse_concept_name(pid)
                if concept_name not in ALL_CONCEPTS:
                    raise ValueError(f"Missing concept function for problem {pid} (expected: {concept_name})")
                self.problem_to_concept[pid] = ALL_CONCEPTS[concept_name]

    def _parse_concept_name(self, pid):
        # Example: for 'hd_has_four_straight_lines-has_obtuse_angle_0000', extract 'has_four_straight_lines'
        # Customize parsing logic as needed for your naming conventions
        # Here, just split by '_' and take the first concept
        parts = pid.split('_')
        for part in parts:
            if part in ALL_CONCEPTS:
                return part
        # Fallback: try to match known patterns
        for cname in ALL_CONCEPTS:
            if cname in pid:
                return cname
        raise ValueError(f"Cannot parse concept name from problem ID: {pid}")

    def get(self, pid):
        if pid not in self.problem_to_concept:
            raise KeyError(f"No concept function registered for problem ID: {pid}")
        return self.problem_to_concept[pid]

# Factory functions for each problem type
def get_concept_fn_for_problem(pid):
    registry = ConceptRegistry()
    return registry.get(pid)

# Example: get concept function for a problem ID
# concept_fn = get_concept_fn_for_problem('hd_has_four_straight_lines_0010')
