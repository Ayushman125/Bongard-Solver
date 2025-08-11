import numpy as np
from collections import Counter
from scipy.stats import entropy, wasserstein_distance
from typing import List, Dict, Any

def positive_negative_contrast_score(pos_features: List[float], neg_features: List[float]) -> float:
    """Compute contrast score between positive and negative sets."""
    if not pos_features or not neg_features:
        print("[WARN][contextual_features] Empty pos_features or neg_features in positive_negative_contrast_score, returning 0.0")
        return 0.0
    return abs(np.mean(pos_features) - np.mean(neg_features))

def support_set_mutual_information(features: List[float], bins: int = 10) -> float:
    """Compute entropy of feature distribution in support set."""
    if features is None or (isinstance(features, np.ndarray) and features.size == 0) or (hasattr(features, '__len__') and len(features) == 0):
        print("[WARN][contextual_features] Empty features in support_set_mutual_information, returning 0.0")
        return 0.0
    hist = np.histogram(features, bins=bins)[0]
    return entropy(hist)

def label_consistency_ratio(labels: List[Any]) -> float:
    """Compute ratio of most common label to total labels."""
    if not labels:
        print("[WARN][contextual_features] Empty labels in label_consistency_ratio, returning 0.0")
        return 0.0
    return max(Counter(labels).values()) / len(labels)

def concept_drift_score(features: List[float]) -> float:
    """Compare feature distributions between early and late samples."""
    n = len(features)
    if n < 2:
        print("[WARN][contextual_features] Not enough features in concept_drift_score, returning 0.0")
        return 0.0
    return wasserstein_distance(features[:n//2], features[n//2:])

def support_set_shape_cooccurrence(shape_types: List[Any]) -> np.ndarray:
    """Count co-occurrence of shape types in support set."""
    if not shape_types:
        print("[WARN][contextual_features] Empty shape_types in support_set_shape_cooccurrence, returning empty array")
        return np.zeros((0, 0))
    unique_types = list(set(shape_types))
    idx_map = {t: i for i, t in enumerate(unique_types)}
    co_occurrence = np.zeros((len(unique_types), len(unique_types)))
    for i, t1 in enumerate(shape_types):
        for t2 in shape_types:
            co_occurrence[idx_map[t1], idx_map[t2]] += 1
    return co_occurrence

def category_consistency_score(shape_types: List[Any]) -> float:
    """Compute ratio of most common shape type to total."""
    if not shape_types:
        print("[WARN][contextual_features] Empty shape_types in category_consistency_score, returning 0.0")
        return 0.0
    return max(Counter(shape_types).values()) / len(shape_types)

def class_prototype_distance(features: np.ndarray, class_labels: List[Any]) -> Dict[Any, List[float]]:
    """Compute distance from each sample to its class prototype."""
    if features is None or len(features) == 0 or not class_labels:
        print("[WARN][contextual_features] Empty features or class_labels in class_prototype_distance, returning empty dict")
        return {}
    prototypes = {}
    distances = {}
    for label in set(class_labels):
        class_feats = features[np.array(class_labels) == label]
        if len(class_feats) == 0:
            continue
        prototype = np.mean(class_feats, axis=0)
        prototypes[label] = prototype
        distances[label] = [np.linalg.norm(f - prototype) for f in class_feats]
    return distances

def feature_importance_ranking(feature_matrix: np.ndarray) -> List[int]:
    """Rank features by variance."""
    if feature_matrix is None or feature_matrix.size == 0:
        print("[WARN][contextual_features] Empty feature_matrix in feature_importance_ranking, returning empty list")
        return []
    variances = np.var(feature_matrix, axis=0)
    return list(np.argsort(variances)[::-1])

def cross_set_symmetry_difference(pos_symmetry: List[float], neg_symmetry: List[float]) -> float:
    """Compare symmetry scores between positive and negative sets."""
    if not pos_symmetry or not neg_symmetry:
        print("[WARN][contextual_features] Empty pos_symmetry or neg_symmetry in cross_set_symmetry_difference, returning 0.0")
        return 0.0
    return abs(np.mean(pos_symmetry) - np.mean(neg_symmetry))
