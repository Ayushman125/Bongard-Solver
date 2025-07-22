import unittest
import numpy as np
from src.system1_al import System1AbstractionLayer

class TestSystem1AL(unittest.TestCase):
    def setUp(self):
        self.s1 = System1AbstractionLayer()

    def test_extract_features(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        features = self.s1.extract_features(img)
        self.assertIn("object_count", features)
        self.assertIn("objects", features)
        self.assertIn("relationships", features)

    def test_process_batch(self):
        masks = [np.zeros((10, 10), dtype=np.uint8) for _ in range(12)]
        bundle = self.s1.process(masks, problem_id="test")
        self.assertEqual(len(bundle["images"]), 12)
        self.assertEqual(bundle["problem_id"], "test")

    def test_extract_attributes(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        attrs = self.s1.extract_attributes(img)
        self.assertIn("hole_count", attrs)
        self.assertIn("symmetry", attrs)

if __name__ == "__main__":
    unittest.main()
