import unittest
from src.Derive_labels.multiscale_geometric import compute_multiscale_geometric_features

class TestMultiscaleGeometricFeatures(unittest.TestCase):
    def test_basic(self):
        image_dict = {
            'vertices': [(0,0), (1,0), (1,1), (0,1)]
        }
        features = compute_multiscale_geometric_features(image_dict)
        self.assertIn('scale_space_curvature', features)
        self.assertIn('moment_invariants', features)
        self.assertIsInstance(features['scale_space_curvature'], list)
        self.assertIsInstance(features['moment_invariants'], list)

if __name__ == '__main__':
    unittest.main()
