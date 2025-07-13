# Folder: bongard_solver/
# File: topo_features.py
import numpy as np
import logging
from typing import Union, List, Tuple, Any

# Conditional import for ripser and persim
try:
    from ripser import ripser
    from persim import PersistenceImager
    HAS_TOPOLOGY_LIBS = True
    logger = logging.getLogger(__name__)
    logger.info("ripser and persim found and enabled for topological features.")
except ImportError:
    HAS_TOPOLOGY_LIBS = False
    logger = logging.getLogger(__name__)
    logger.warning("ripser or persim not found. Topological features will be disabled.")

class TopologicalFeatureExtractor:
    """
    Extracts topological features (e.g., persistence images) from binary masks
    using Persistent Homology.
    """
    def __init__(self, thr: float = 0.5, pixel_size: float = 0.1, feature_dim: int = 64):
        """
        Args:
            thr (float): Pixel intensity threshold to binarize the mask (0-1 range).
            pixel_size (float): Pixel size for the PersistenceImager.
            feature_dim (int): Desired output feature dimension after flattening the persistence image.
                               The actual output size depends on the imager's grid and pixel_size.
                               This is a target, and the actual output might be larger/smaller.
        """
        self.thr = thr
        self.feature_dim = feature_dim
        self.imager = None
        if HAS_TOPOLOGY_LIBS:
            self.imager = PersistenceImager(pixel_size=pixel_size)
            # Fit the imager with a dummy point cloud to initialize its grid
            # This is important if you want a fixed output size, otherwise it adapts to data.
            # A fixed range for persistence diagrams (e.g., [0,1] for birth/death) is often used.
            # For now, a simple fit.
            self.imager.fit(np.array([[0, 1], [0.5, 0.5]])) # Dummy data to initialize
            logger.info(f"PersistenceImager initialized with pixel_size={pixel_size}.")
        else:
            logger.warning("Topological feature extraction disabled due to missing libraries.")

    def extract(self, mask: np.ndarray) -> np.ndarray:
        """
        Extracts topological features from a binary mask.
        
        Args:
            mask (np.ndarray): A 2D NumPy array representing the binary mask (e.g., from SAM).
                               Values should be 0 or 1, or can be thresholded.
        Returns:
            np.ndarray: A 1D NumPy array of topological features (flattened persistence image).
                        Returns a zero array if libraries are missing or no points found.
        """
        if not HAS_TOPOLOGY_LIBS or self.imager is None:
            return np.zeros(self.feature_dim) # Return dummy zero features

        # Convert mask to a point cloud (coordinates of foreground pixels)
        # np.where returns (row_indices, col_indices). column_stack makes them (x, y) pairs.
        pts = np.column_stack(np.where(mask > self.thr)).astype(np.float32)

        if len(pts) < 5: # Need at least a few points to compute meaningful homology
            logger.debug(f"Mask has too few points ({len(pts)}) for persistent homology. Returning zero features.")
            return np.zeros(self.feature_dim)

        try:
            # Compute persistence diagrams (homology groups H0, H1, H2, ...)
            # dgms[1] typically corresponds to 1-dimensional homology (loops/holes)
            dgms = ripser(pts)['dgms']
            
            if len(dgms) < 2 or dgms[1].shape[0] == 0:
                logger.debug("No 1-dimensional persistence diagram found. Returning zero features.")
                return np.zeros(self.feature_dim)

            # Transform persistence diagram into a persistence image
            # The imager was fitted with dummy data, so its grid is fixed.
            # If dgms[1] is empty, transform will return an empty array.
            img = self.imager.transform(dgms[1])
            
            # Flatten the image to a 1D feature vector
            features = img.flatten()

            # Resize features to target_dim if necessary (e.g., by padding or truncation)
            if features.shape[0] < self.feature_dim:
                padded_features = np.pad(features, (0, self.feature_dim - features.shape[0]), 'constant')
                return padded_features
            elif features.shape[0] > self.feature_dim:
                return features[:self.feature_dim]
            else:
                return features
        except Exception as e:
            logger.error(f"Error during topological feature extraction: {e}. Returning zero features.")
            return np.zeros(self.feature_dim)

