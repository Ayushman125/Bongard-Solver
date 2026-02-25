import unittest
from src.physics_inference_advanced import AdvancedPhysicsInference

class TestAdvancedPhysics(unittest.TestCase):
    def test_infer_compositional_physics(self):
        api = AdvancedPhysicsInference()
        result = api.infer_compositional_physics(['line_normal', 'arc_square'])
        self.assertIsInstance(result, dict)
    def test_discover_physics_primitives(self):
        api = AdvancedPhysicsInference()
        primitives = api.discover_physics_primitives(['line_normal'])
        self.assertIsInstance(primitives, list)
    def test_learn_physics_composition(self):
        api = AdvancedPhysicsInference()
        rules = api.learn_physics_composition(['primitive1'])
        self.assertIsInstance(rules, list)
    def test_detect_emergent_physics(self):
        api = AdvancedPhysicsInference()
        emergent = api.detect_emergent_physics(['primitive1'], ['rule1'])
        self.assertIsInstance(emergent, dict)

if __name__ == "__main__":
    unittest.main()
