__version__ = "0.1.0"
"""
DriftMonitor

Sliding-window Mahalanobis‐distance detector on 1D embeddings:

  dm = DriftMonitor(window_size=100, threshold=3.0)
  dm.update(embedding)
  if dm.check_drift():  # handle OOD
      …
"""
import numpy as np
from collections import deque

class DriftMonitor:
    def __init__(self, window_size=100, threshold=3.0):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def update(self, embedding: np.ndarray):
        if embedding.ndim != 1:
            raise ValueError("Expected 1D embedding")
        self.window.append(embedding)

    def check_drift(self) -> bool:
        if len(self.window) < 2:
            return False
        data = np.stack(self.window)
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False) + 1e-6*np.eye(data.shape[1])
        inv = np.linalg.inv(cov)
        delta = (self.window[-1] - mean)
        maha = float(delta.T @ inv @ delta)
        return maha > self.threshold
