def discriminative_concepts(pos_features: list, neg_features: list) -> set:
    """
    Find concepts/rules present in the positive set but absent in the negative set.
    Each feature is a list/set of concept tokens (strings).
    Returns a set of discriminative concepts.
    """
    # Flatten and collect all concepts in positives and negatives
    pos_concepts = set()
    for feat in pos_features:
        if isinstance(feat, (list, set)):
            pos_concepts.update(feat)
        elif isinstance(feat, str):
            pos_concepts.add(feat)
    neg_concepts = set()
    for feat in neg_features:
        if isinstance(feat, (list, set)):
            neg_concepts.update(feat)
        elif isinstance(feat, str):
            neg_concepts.add(feat)
    # Discriminative concepts: present in positives, absent in negatives
    return pos_concepts - neg_concepts
import numpy as np
from collections import Counter
from scipy.stats import entropy, wasserstein_distance
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from .contextual_perception import (
    ContextualPerceptionEncoder,
    QueryContextAttention,
    AdaptiveConceptGenerator
)
from src.Derive_labels.emergence import ConceptMemoryBank

# Initialize modules (singleton or per-call)
_encoder = ContextualPerceptionEncoder()
_attention = QueryContextAttention()
_generator = AdaptiveConceptGenerator()

def contextual_concept_hypotheses(support_pos_feats, support_neg_feats, query_feat):
    """
    support_pos_feats/support_neg_feats: list of feature dicts → convert to tensors
    query_feat: tensor
    """
    # 1. Encode support as context
    device = next(_encoder.parameters()).device
    support_feats = torch.stack(support_pos_feats + support_neg_feats).to(device)  # (12, D)
    query_feat = query_feat.to(device)
    context_encoded = _encoder(support_feats)                          # (12, D)
    # 2. Summarize context (e.g., mean pooling)
    context_summary = context_encoded.mean(dim=0)                      # (D,)
    # 3. Cross-attention with query
    query_context = _attention(query_feat, context_encoded)           # (D,)
    # 4. Generate adaptive concept hypotheses
    hypotheses = _generator(query_context, context_summary)           # (C,)
    # 5. Persist learned context for future problems
    ConceptMemoryBank.integrate(context_summary.detach().cpu().numpy())
    return hypotheses
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
