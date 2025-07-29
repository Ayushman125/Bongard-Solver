from collections import defaultdict

# --- Detector Reliability Matrix ---
DETECTOR_RELIABILITY = {
    'physics':     {'convexity': 1.0, 'holes': 0.7, 'arcs': 0.9, 'stability': 1.0, 'symmetry': 0.8},
    'image':       {'convexity': 0.7, 'holes': 1.0, 'arcs': 0.7, 'stability': 0.5, 'symmetry': 0.7},
    'geometric':   {'convexity': 0.9, 'holes': 0.6, 'arcs': 0.8, 'stability': 0.7, 'symmetry': 1.0},
    'semantic':    {'convexity': 0.7, 'holes': 0.7, 'arcs': 0.7, 'stability': 0.6, 'symmetry': 0.7},
    'mcts':        {'convexity': 0.8, 'holes': 0.7, 'arcs': 0.8, 'stability': 0.9, 'symmetry': 0.8},
}

# --- Label Type Extraction Helper ---
def get_label_type(label):
    label = label.lower()
    if 'convex' in label: return 'convexity'
    if 'concave' in label: return 'convexity'
    if 'hole' in label: return 'holes'
    if 'arc' in label: return 'arcs'
    if 'stable' in label or 'balanced' in label: return 'stability'
    if 'symmetry' in label or 'symmetric' in label: return 'symmetry'
    return 'other'

# --- Consensus Voting Function ---
def consensus_vote(candidate_labels, detector_confidences, detector_reliabilities=DETECTOR_RELIABILITY, context=None, ambiguity_threshold=0.1):
    label_scores = defaultdict(float)
    label_sources = defaultdict(list)
    for detector, labels in candidate_labels.items():
        for label, conf in labels.items():
            label_type = get_label_type(label)
            weight = detector_reliabilities.get(detector, {}).get(label_type, 1.0)
            # Contextual arbitration: boost geometric if grid, etc.
            if context and context.get('is_grid') and detector == 'geometric':
                weight *= 1.2
            label_scores[label] += conf * weight
            label_sources[label].append((detector, conf, weight))
    sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])
    if len(sorted_labels) > 1 and abs(sorted_labels[0][1] - sorted_labels[1][1]) < ambiguity_threshold:
        return 'AMBIGUOUS', sorted_labels, label_sources
    return sorted_labels[0][0], sorted_labels, label_sources
import numpy as np

def calculate_confidence(metric_value, max_value, min_value, is_higher_better=True):
    """
    Normalizes a metric value into a confidence score (0-1).
    max_value and min_value define the expected range for the metric.
    """
    if max_value == min_value:
        return 0.5 # Neutral if range is zero

    if is_higher_better:
        return np.clip((metric_value - min_value) / (max_value - min_value), 0.0, 1.0)
    else: # Lower value is better (e.g., error)
        return np.clip(1.0 - (metric_value - min_value) / (max_value - min_value), 0.0, 1.0)
