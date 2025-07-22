from gudhi import CubicalComplex
import numpy as np

def compute_betti(bin_img):
    """
    Compute Betti numbers (0 and 1) for a binary mask using persistent homology.
    Args:
        bin_img (np.ndarray): 2D binary mask (0/1 or 0/255).
    Returns:
        tuple: (betti0, betti1). Returns (0, 0) if mask is empty or degenerate.
    """
    bin_img = (bin_img > 0)
    if bin_img.sum() == 0:
        return 0, 0
    # invert mask: 1 → empty, 0 → solid
    data = (1 - bin_img.astype(int)).flatten()
    try:
        cc = CubicalComplex(dimensions=bin_img.shape, top_dimensional_cells=data)
        diag = cc.persistence(homology_coeff_field=2, min_persistence=0.01)
        betti0 = sum(1 for dim,p in diag if dim==0)
        betti1 = sum(1 for dim,p in diag if dim==1)
        return betti0, betti1
    except Exception:
        return 0, 0
