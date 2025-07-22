import numpy as np
from skimage.filters import sobel_h, sobel_v

def structure_tensor_coherence(bin_img):
    """
    Compute average structure tensor coherence for a binary mask.
    Args:
        bin_img (np.ndarray): 2D binary mask (0/1 or 0/255).
    Returns:
        float: Average coherence over edge pixels. Returns 0.0 if mask is empty or no edges.
    """
    bin_img = (bin_img > 0)
    if bin_img.sum() == 0:
        return 0.0
    img = bin_img.astype(float)
    gx, gy = sobel_h(img), sobel_v(img)
    # structure tensor elements
    a, b, c = gx*gx, gx*gy, gy*gy
    # coherence = (λ1 - λ2)/(λ1 + λ2)
    trace = a + c
    det = a*c - b*b
    disc = np.sqrt(np.maximum(0, trace**2 - 4*det))
    lam1 = (trace + disc)/2
    lam2 = (trace - disc)/2 + 1e-8
    coherence = (lam1 - lam2) / (lam1 + lam2)
    # average over edge pixels only
    edges = (gx**2 + gy**2) > 1e-6
    if np.any(edges):
        return float(np.mean(coherence[edges]))
    else:
        return 0.0
