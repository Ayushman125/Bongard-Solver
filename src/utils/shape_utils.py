import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def estimate_stroke_width(bin_img):
    """
    Estimate average stroke width of a binary shape mask.
    Args:
        bin_img (np.ndarray): 2D binary mask (0/1 or 0/255).
    Returns:
        float: Estimated average stroke width (diameter). Returns 0.0 if empty.
    """
    bin_img = (bin_img > 0)
    if bin_img.sum() == 0:
        return 0.0
    skel = skeletonize(bin_img)
    if skel.sum() == 0:
        return 0.0
    dist = distance_transform_edt(bin_img)
    widths = dist[skel]
    if widths.size == 0:
        return 0.0
    return float(widths.mean() * 2)  # approximate diameter
