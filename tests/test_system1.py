import unittest
import numpy as np
import os
import time
import json

from src.system1_al import System1AbstractionLayer
from src.utils.fuzzy_tree import FuzzyTree

class TestSystem1AbstractionLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests."""
        cls.test_data_dir = "test_data"
        cls.fuzzy_model_path = os.path.join(cls.test_data_dir, "test_fuzzy_tree.pkl")
        cls.replay_path = os.path.join(cls.test_data_dir, "test_replay.pkl")

        if not os.path.exists(cls.test_data_dir):
            os.makedirs(cls.test_data_dir)

        # Create a dummy fuzzy tree model for testing
        dummy_tree = FuzzyTree()
        dummy_tree.save(cls.fuzzy_model_path)

        cls.s1_al = System1AbstractionLayer(
            fuzzy_model_path=cls.fuzzy_model_path,
            replay_path=cls.replay_path
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists(cls.fuzzy_model_path):
            os.remove(cls.fuzzy_model_path)
        if os.path.exists(cls.replay_path):
            os.remove(cls.replay_path)
        if os.path.exists(cls.test_data_dir):
            os.rmdir(cls.test_data_dir)

    def setUp(self):
        """Set up for each test."""
        # Clear replay buffer before each test
        self.s1_al.replay_buffer.clear()
        
        # Synthetic shapes
        self.circle = self._create_circle_img()
        self.square = self._create_square_img()
        self.empty = np.zeros((100, 100), dtype=np.uint8)

    def _create_circle_img(self, radius=30):
        img = np.zeros((100, 100), dtype=np.uint8)
        cx, cy = 50, 50
        y, x = np.ogrid[-cy:100-cy, -cx:100-cx]
        mask = x*x + y*y <= radius*radius
        img[mask] = 1
        return img

    def _create_square_img(self, size=60):
        img = np.zeros((100, 100), dtype=np.uint8)
        start = (100 - size) // 2
        end = start + size
        img[start:end, start:end] = 1
        return img

    def test_attribute_correctness_circle(self):
        """Test attributes of a synthetic circle."""
        attrs = self.s1_al.extract_attributes(self.circle)
        self.assertGreater(attrs['area'], 2800, "Circle area should be significant.")
        self.assertAlmostEqual(attrs['circularity'], 1.0, delta=0.1, msg="Circle circularity should be close to 1.")
        self.assertEqual(attrs['hole_count'], 0, "Solid circle should have 0 holes.")
        self.assertEqual(attrs['euler_number'], 1, "Solid circle should have Euler number 1.")

    def test_attribute_correctness_square(self):
        """Test attributes of a synthetic square."""
        attrs = self.s1_al.extract_attributes(self.square)
        self.assertEqual(attrs['area'], 3600, "Square area should be exact.")
        self.assertLess(attrs['circularity'], 0.8, msg="Square circularity should be less than a circle's.")

    def test_relation_matrix(self):
        """Test relational cues between two different shapes."""
        attrs_list = [
            self.s1_al.extract_attributes(self.square), # larger area
            self.s1_al.extract_attributes(self.circle)  # smaller area
        ]
        relations = self.s1_al.compute_relations(attrs_list)
        self.assertIn("0_1", relations)
        self.assertEqual(relations["0_1"]["size_relation"], ">", "Square should be larger than circle.")
        self.assertGreater(relations["0_1"]["area_ratio"], 1.0)

    def test_heuristic_output(self):
        """Test that heuristic generation produces a non-empty list of rules."""
        attrs_list = [self.s1_al.extract_attributes(self.square)]
        heuristics = self.s1_al.generate_heuristics(attrs_list)
        self.assertIsInstance(heuristics, list)
        self.assertGreater(len(heuristics), 0, "Should generate at least one heuristic.")
        self.assertTrue(all("rule" in h and "confidence" in h for h in heuristics))

    def test_runtime_performance(self):
        """Test that processing is fast enough."""
        images = [self.square, self.circle] * 6 # A full 12-image problem
        start_time = time.time()
        bundle = self.s1_al.process(images)
        duration_ms = (time.time() - start_time) * 1000
        
        print(f"\nS1-AL processing time for 12 images: {duration_ms:.2f} ms")
        self.assertLess(duration_ms, 200, "Processing 12 images should be under 200ms.")
        self.assertLess(bundle['duration_ms'], 200)

    def test_json_bundle_structure(self):
        """Verify the structure of the output JSON bundle."""
        bundle = self.s1_al.process([self.square, self.circle], problem_id="test_01")
        
        self.assertIn("problem_id", bundle)
        self.assertIn("timestamp", bundle)
        self.assertIn("duration_ms", bundle)
        self.assertIn("images", bundle)
        self.assertIn("heuristics", bundle)
        
        self.assertEqual(len(bundle["images"]), 2)
        self.assertIn("image_id", bundle["images"][0])
        self.assertIn("attrs", bundle["images"][0])
        self.assertIn("relations", bundle["images"][0])
        
        # Test if it's JSON serializable
        try:
            json.dumps(bundle)
        except TypeError:
            self.fail("S1-AL output bundle is not JSON serializable.")

    def test_self_supervision_loop(self):
        """Test the self-supervision and replay buffer mechanism."""
        s1_output = self.s1_al.process([self.square, self.circle], problem_id="supervision_test")
        
        # Simulate a "surprising" outcome where the heuristic was wrong
        # Let's say heuristic confidence was 0.8, but true labels are [0, 1]
        s1_output['heuristics'] = [{'rule': 'mock_rule', 'confidence': 0.8}]
        true_labels = [0, 1] # Model was very confident (0.8) but wrong about the first image
        
        self.s1_al.self_supervise(s1_output, true_labels)
        
        # The first sample should be added because |0.8 - 0| > threshold (0.4)
        # The second sample should not be added because |0.8 - 1| < threshold
        self.assertEqual(self.s1_al.replay_buffer.size(), 1, "One surprising sample should be added.")
        
        sample = self.s1_al.replay_buffer.sample(1)[0]
        self.assertEqual(sample[1], 0) # Check true label
        self.assertAlmostEqual(sample[2], 0.8, delta=0.01) # Check delta

if __name__ == '__main__':
    unittest.main()
