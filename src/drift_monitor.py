
"""
DriftMonitor: sliding-window detection on S1 embeddings.
Version: 0.1.0
"""

__version__ = "0.1.0"

import numpy as np
from collections import deque

class DriftMonitor:
    def detect_drift(self, embeddings: np.ndarray) -> bool:
        """
        Batch‐process a set of embeddings and return whether drift occurred.
        """
        if embeddings.ndim != 2:
            raise ValueError("Expected 2D array of embeddings.")
        for emb in embeddings:
            self.update(emb)
        return self.check_drift()
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        """
        Args:
            window_size: Number of embeddings to keep in sliding window.
            threshold: Mahalanobis distance threshold to trigger drift.
        """
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def update(self, embedding: np.ndarray):
        """Add a new embedding to the window."""
        if embedding.ndim != 1:
            raise ValueError("Embedding must be 1D array.")
        self.window.append(embedding)

    def check_drift(self) -> bool:
        """
        Compute Mahalanobis distance of the latest embedding from the window mean.
        Returns True if distance exceeds threshold.
        """
        if len(self.window) < 2:
            return False
        data = np.stack(self.window)
        mean = data.mean(axis=0)
        cov = np.cov(data, rowvar=False) + 1e-6 * np.eye(data.shape[1])
        inv_cov = np.linalg.inv(cov)
        diff = self.window[-1] - mean
        dist = float(diff.T @ inv_cov @ diff)
        return dist > self.threshold
