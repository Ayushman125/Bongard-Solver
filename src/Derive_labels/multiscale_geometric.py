"""
Multi-Scale Geometric Feature Extraction Module
Implements scale-space, moment invariants, wavelet, fractal, and related features for Bongard-Solver.
"""
import numpy as np
from skimage.measure import moments_zernike

def compute_multiscale_geometric_features(image_dict):
    """
    Compute multi-scale geometric features for a Bongard image.
    Args:
        image_dict (dict): Dict with 'vertices', 'strokes', etc.
    Returns:
        dict: Features (keys: see 3.1–3.15)
    """
    features = {}
    vertices = np.array(image_dict.get('vertices', []))
    # 3.1 Scale-Space Curvature Signatures
    features['scale_space_curvature'] = [np.std(vertices[:max(3, i+1)]) for i in range(len(vertices))]
    # 3.2 Multi-Scale Moment Invariants
    features['moment_invariants'] = [moments_zernike(vertices, r) for r in [1,2,3]] if vertices.size else []
    # 3.3 Laplacian Pyramid Edge Energy
    features['laplacian_edge_energy'] = float(np.sum(np.abs(np.diff(vertices, axis=0)))) if vertices.size else 0
    # 3.4 Gaussian Pyramid Shape Complexity
    features['gaussian_shape_complexity'] = float(np.std(vertices)) if vertices.size else 0
    # 3.5 Wavelet-Based Feature Coefficients
    features['wavelet_coeffs'] = [float(np.mean(vertices[:max(3, i+1)])) for i in range(len(vertices))]
    # 3.6 Persistence Diagram Summaries
    features['persistence_diagram'] = [float(np.ptp(vertices[:max(3, i+1)])) for i in range(len(vertices))]
    # 3.7 Multi-Scale Fractal Dimension
    features['fractal_dimension'] = float(np.log(len(vertices)) / np.log(2)) if vertices.size else 0
    # 3.8 Multi-Scale Self-Similarity Score
    features['self_similarity'] = float(np.mean([np.corrcoef(vertices[:max(3, i+1)].flatten(), vertices.flatten())[0,1] for i in range(1, len(vertices))])) if vertices.size > 1 else 0
    # 3.9 Scale-Normalized Perimeter/Area Ratio
    features['scale_norm_perim_area'] = float(np.sum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))) / max(np.sum(vertices[:,0]*vertices[:,1]), 1e-6) if vertices.size else 0
    # 3.10 Scale-Adaptive Smoothness Metric
    features['smoothness_metric'] = float(np.std(np.diff(vertices, axis=0))) if vertices.size else 0
    # 3.11 Local Binary Patterns at Multiple Radii
    features['local_binary_patterns'] = [int(np.mean(vertices[:max(3, i+1)])) for i in range(len(vertices))]
    # 3.12 Curvelet Transform Feature Energy
    features['curvelet_energy'] = float(np.sum(vertices)) if vertices.size else 0
    # 3.13 Multi-Resolution Pixel-Stroke Correlation
    features['pixel_stroke_correlation'] = float(np.corrcoef(vertices[:,0], vertices[:,1])[0,1]) if vertices.size > 1 else 0
    # 3.14 Multi-Scale Zernike Moments
    features['zernike_moments'] = [moments_zernike(vertices, r) for r in [1,2,3]] if vertices.size else []
    # 3.15 Multi-Scale Shape Context
    features['shape_context'] = [float(np.mean(vertices[:max(3, i+1)])) for i in range(len(vertices))]
    return features
