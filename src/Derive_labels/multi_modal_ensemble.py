"""
multi_modal_ensemble.py
Aggregates geometric, topological, compositional, and pixel-level features with quality-weighted ensemble.
"""
from .shape_utils import calculate_geometry
from .spatial_topological_features import compute_spatial_topological_features
from .multiscale_geometric import compute_multiscale_geometric_features
from .compositional_features import _calculate_composition_features
from .quality_monitor import quality_monitor

def aggregate_multi_modal_features(image_dict):
    features = {}
    # Geometric
    features['geometry'] = calculate_geometry(image_dict.get('vertices', []))
    # Topological
    features['topological'] = compute_spatial_topological_features(image_dict)
    # Multiscale
    features['multiscale'] = compute_multiscale_geometric_features(image_dict)
    # Compositional
    features['compositional'] = _calculate_composition_features(image_dict.get('action_commands', []))
    # Quality-weighted ensemble (simple average of non-degenerate scores)
    qualities = []
    for k, v in features.items():
        if isinstance(v, dict) and 'degenerate_case' in v:
            qualities.append(1.0 if not v['degenerate_case'] else 0.0)
    ensemble_quality = sum(qualities) / max(1, len(qualities))
    features['ensemble_quality'] = ensemble_quality
    quality_monitor.log_quality('multi_modal_ensemble', {'ensemble_quality': ensemble_quality})
    return features
