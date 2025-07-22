from gudhi import CubicalComplex
import numpy as np

def compute_betti(bin_img):
    # invert mask: 1 → empty, 0 → solid
    data = (1 - bin_img.astype(int)).flatten()
    cc = CubicalComplex(dimensions=bin_img.shape, top_dimensional_cells=data)
    diag = cc.persistence(homology_coeff_field=2, min_persistence=0.01)
    betti0 = sum(1 for dim,p in diag if dim==0)
    betti1 = sum(1 for dim,p in diag if dim==1)
    return betti0, betti1
