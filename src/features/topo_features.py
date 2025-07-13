import numpy as np
from ripser import ripser
from scipy.ndimage import label

class TopologicalFeatureExtractor:
    """
    Extracts persistent homology features from binary masks.
    """
    def __init__(self, pixel_thresh=0.5):
        self.pixel_thresh = pixel_thresh

    def extract_features(self, mask):
        # mask: np.ndarray, binary (H, W)
        mask = (mask > self.pixel_thresh).astype(np.uint8)
        # Compute persistent homology (H0, H1)
        ph = ripser(mask, distance_matrix=False, maxdim=1)
        h0 = ph['dgms'][0]
        h1 = ph['dgms'][1]
        # Simple feature: number of connected components and holes
        n_components = len(h0)
        n_holes = len(h1)
        # Optionally, use lifetimes as features
        h1_lifetime = sum([pt[1]-pt[0] for pt in h1 if pt[1] < np.inf])
        return np.array([n_components, n_holes, h1_lifetime])
