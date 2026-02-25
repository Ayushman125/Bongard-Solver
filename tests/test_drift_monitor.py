import unittest
import numpy as np
from src.drift_monitor import DriftMonitor

class TestDriftMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = DriftMonitor()

    def test_detect_drift(self):
        embeddings = np.random.rand(10, 128)
        result = self.monitor.detect_drift(embeddings)
        self.assertFalse(result)  # Placeholder always returns False

if __name__ == "__main__":
    unittest.main()
