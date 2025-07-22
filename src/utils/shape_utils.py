import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def estimate_stroke_width(bin_img):
    skel = skeletonize(bin_img)
    dist = distance_transform_edt(bin_img)
    widths = dist[skel]
    return float(widths.mean() * 2)  # approximate diameter
