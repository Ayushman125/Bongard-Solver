# src/features/topo_features.py

import numpy as np
from ripser import ripser
from scipy.ndimage import label
import logging

logger = logging.getLogger(__name__)

class TopologicalFeatureExtractor:
    """
    Extracts persistent homology features from binary masks.
    """
    def __init__(self, pixel_thresh: float = 0.5):
        """
        Initializes the TopologicalFeatureExtractor.
        Args:
            pixel_thresh (float): Threshold to binarize the mask if it's not already binary.
        """
        self.pixel_thresh = pixel_thresh
        logger.info(f"TopologicalFeatureExtractor initialized with pixel_thresh={pixel_thresh}.")

    def extract_features(self, mask: np.ndarray) -> np.ndarray:
        """
        Extracts topological features (e.g., number of connected components, holes)
        using persistent homology from a binary mask.
        Args:
            mask (np.ndarray): A binary mask (H, W) where values are 0 or 1 (or 0-255).
        Returns:
            np.ndarray: A NumPy array of topological features.
        """
        # Ensure mask is binary (0 or 1)
        mask = (mask > self.pixel_thresh).astype(np.uint8)
        
        if mask.size == 0:
            logger.warning("Empty mask provided to TopologicalFeatureExtractor. Returning zeros.")
            return np.array([0, 0, 0.0]) # Return default features for empty mask

        # Compute persistent homology (H0 for connected components, H1 for holes)
        # `distance_matrix=False` means ripser expects a point cloud or image
        # `maxdim=1` computes homology up to dimension 1 (H0 and H1)
        try:
            ph = ripser(mask, distance_matrix=False, maxdim=1)
            h0 = ph['dgms'][0] # Diagram for H0 (connected components)
            h1 = ph['dgms'][1] # Diagram for H1 (holes)
        except Exception as e:
            logger.error(f"Error computing persistent homology: {e}. Returning default features.")
            return np.array([0, 0, 0.0]) # Fallback in case of ripser error

        # Simple feature: number of connected components (births in H0)
        n_components = len(h0)
        
        # Simple feature: number of holes (births in H1)
        n_holes = len(h1)
        
        # Optionally, use sum of lifetimes of holes as a feature
        # A "lifetime" is birth - death. For infinite deaths, we ignore or handle specially.
        h1_lifetime = sum([pt[1] - pt[0] for pt in h1 if pt[1] < np.inf])
        
        features = np.array([n_components, n_holes, h1_lifetime])
        logger.debug(f"Extracted topological features: {features}")
        return features

