import unittest
from src.Derive_labels.spatial_topological_features import compute_spatial_topological_features

class TestSpatialTopologicalFeatures(unittest.TestCase):
    def test_basic(self):
        image_dict = {
            'vertices': [(0,0), (1,0), (1,1), (0,1)],
            'strokes': [
                {'vertices': [(0,0), (1,0)]},
                {'vertices': [(1,0), (1,1)]},
                {'vertices': [(1,1), (0,1)]},
                {'vertices': [(0,1), (0,0)]}
            ]
        }
        features = compute_spatial_topological_features(image_dict)
        self.assertIn('adjacency_degree', features)
        self.assertIn('euler_characteristic', features)
        self.assertIsInstance(features['adjacency_degree'], dict)
        self.assertIsInstance(features['euler_characteristic'], int)

if __name__ == '__main__':
    unittest.main()
