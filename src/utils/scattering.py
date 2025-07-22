import torch
from kymatio import Scattering2D

def compute_scattering(bin_img, J=2, L=8):
    img = bin_img.astype(float)
    img = (img - img.mean()) / (img.std() + 1e-8)
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    scattering = Scattering2D(J=J, shape=bin_img.shape, L=L)
    Sx = scattering(x)                       # (1, C, H', W')
    coeffs = Sx.mean(dim=[2,3]).squeeze(0)   # (C,)
    return coeffs.tolist()
