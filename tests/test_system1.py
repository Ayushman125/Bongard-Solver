
import unittest
import numpy as np
import os
import time
import json
from PIL import Image
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.system1_al import System1AbstractionLayer
from src.utils.fuzzy_tree import FuzzyTree
from professional_pipeline import find_bongard_problems, load_bongard_problem

class TestSystem1AbstractionLayerWithRealData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests, including finding real data."""
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
        
        # Find real Bongard problems
        cls.problem_root = 'ShapeBongard_V2'
        cls.problem_paths = find_bongard_problems(cls.problem_root)

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
        self.s1_al.replay_buffer.clear()
        if not self.problem_paths:
            self.skipTest(f"No real Bongard problems found in '{self.problem_root}'. Skipping real data tests.")

    def test_load_real_problem(self):
        """Test loading a real Bongard problem."""
        problem_path = self.problem_paths[0]
        images, labels = load_bongard_problem(problem_path)
        
        self.assertGreater(len(images), 0, "Should load at least one image.")
        self.assertEqual(len(images), len(labels), "Images and labels count should match.")
        self.assertIsInstance(images[0], np.ndarray, "Loaded image should be a numpy array.")
        self.assertTrue(images[0].dtype == np.uint8, "Image should be of type uint8.")
        self.assertTrue(set(np.unique(images[0])).issubset({0, 1}), "Image should be binary.")

    def test_attribute_extraction_on_real_data(self):
        """Test attribute extraction on a real image."""
        problem_path = self.problem_paths[0]
        images, _ = load_bongard_problem(problem_path)
        attrs = self.s1_al.extract_attributes(images[0])
        self.assertIsInstance(attrs, dict)
        for key in [
            'area', 'stroke_count', 'endpoint_count', 'branch_point_count',
            'circularity', 'solidity', 'perimeter', 'convex_hull_area',
            'euler_number', 'hole_count', 'curvature_histogram', 'symmetry']:
            self.assertIn(key, attrs)
        self.assertGreaterEqual(attrs['area'], 0)
        self.assertGreaterEqual(attrs['stroke_count'], 0)
        self.assertGreaterEqual(attrs['endpoint_count'], 0)
        self.assertGreaterEqual(attrs['branch_point_count'], 0)
        self.assertGreaterEqual(attrs['circularity'], 0)
        self.assertGreaterEqual(attrs['solidity'], 0)

    def test_full_pipeline_on_real_problem(self):
        """Test the full S1-AL process method on a real problem, including new relations and JSON serialization."""
        problem_path = self.problem_paths[0]
        problem_id = os.path.basename(problem_path)
        images, labels = load_bongard_problem(problem_path)
        bundle = self.s1_al.process(images, problem_id=problem_id)
        self.assertEqual(bundle['problem_id'], problem_id)
        self.assertEqual(len(bundle['images']), len(images))
        self.assertGreater(bundle['duration_ms'], 0)
        self.assertIn('heuristics', bundle)
        # Check new relations in at least one image
        if bundle['images'] and len(bundle['images']) > 1:
            rels = list(bundle['images'][0]['relations'].values())
            if rels:
                rel = rels[0]
                for key in ['iou', 'directional_relation', 'horizontal_aligned', 'vertical_aligned']:
                    self.assertIn(key, rel)
        # Test if it's JSON serializable
        try:
            json.dumps(bundle)
        except TypeError:
            self.fail("S1-AL output bundle from real data is not JSON serializable.")

    def test_runtime_performance_on_real_data(self):
        """Test performance on a real problem set."""
        problem_path = self.problem_paths[0]
        images, _ = load_bongard_problem(problem_path)
        
        start_time = time.time()
        self.s1_al.process(images)
        duration_ms = (time.time() - start_time) * 1000
        
        print(f"\nS1-AL processing time for real problem ({len(images)} images): {duration_ms:.2f} ms")
        # This threshold might need adjustment based on machine performance and image complexity
        self.assertLess(duration_ms, 500, "Processing a real problem should be reasonably fast.")

if __name__ == '__main__':
    unittest.main()
