import unittest
from src.Derive_labels.composition import CompositionEngine

class TestComposition(unittest.TestCase):
    def test_extract_primitives(self):
        primitives = CompositionEngine.extract_primitives(['line_normal', 'arc_square'])
        self.assertIsInstance(primitives, list)
    def test_learn_composition_rules(self):
        rules = CompositionEngine.learn_composition_rules(['primitive1'], set())
        self.assertIsInstance(rules, list)
    def test_generate_combinations(self):
        combos = CompositionEngine.generate_combinations(['primitive1'], ['rule1'])
        self.assertIsInstance(combos, list)

if __name__ == "__main__":
    unittest.main()
