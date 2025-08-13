#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module replaces geometric feature engineering with symbolic, compositional, and context-aware concept extraction for Bongard-LOGO.
All logic operates on LOGO action sequences and supports problem-level reasoning.
"""

from collections import Counter

def detect_convex_pattern(action_sequence):
    """Detects if the action sequence represents a convex concept (symbolic)."""
    # Example: If all strokes are 'normal' or 'arc' with no sharp changes, treat as convex
    modifiers = [parse_logo_action(a)['modifier'] for a in action_sequence if parse_logo_action(a)]
    return 'zigzag' not in modifiers and 'triangle' not in modifiers

def detect_symmetry_pattern(action_sequence):
    """Detects symbolic symmetry in the action sequence."""
    types = [parse_logo_action(a)['type'] for a in action_sequence if parse_logo_action(a)]
    return types.count('line') == types.count('arc')

def detect_containment_pattern(action_sequence):
    """Detects symbolic containment (e.g., nested shapes) in the action sequence."""
    # Example: If sequence contains both 'arc' and 'line' with certain modifiers
    modifiers = [parse_logo_action(a)['modifier'] for a in action_sequence if parse_logo_action(a)]
    return 'square' in modifiers and 'circle' in modifiers

def symbolic_concept_features(action_sequence):
    """Extract abstract concepts from action sequence."""
    return {
        'convexity': detect_convex_pattern(action_sequence),
        'symmetry': detect_symmetry_pattern(action_sequence),
        'containment': detect_containment_pattern(action_sequence)
    }

def extract_problem_level_features(positive_examples, negative_examples):
    """Extract features by comparing positive vs negative examples."""
    pos_features = [symbolic_concept_features(seq) for seq in positive_examples]
    neg_features = [symbolic_concept_features(seq) for seq in negative_examples]
    def summarize(features, key):
        return Counter([f[key] for f in features])
    discriminative = {}
    for key in ['convexity', 'symmetry', 'containment']:
        pos_summary = summarize(pos_features, key)
        neg_summary = summarize(neg_features, key)
        discriminative[key] = list(set(pos_summary) - set(neg_summary))
    return discriminative

def parse_logo_action(action_str):
    """Parse a LOGO action command into symbolic components."""
    import re
    m = re.match(r'(line|arc)\(([^,]+), ([^,]+), ([^\)]+)\)', action_str)
    if not m:
        return None
    action_type, modifier, length, angle = m.groups()
    return {
        'type': action_type,
        'modifier': modifier,
        'length': float(length),
        'angle': float(angle)
    }

# Example usage for module testing
if __name__ == "__main__":
    positive_examples = [
        ["line(normal, 0.7, 0.2)", "arc(square, 0.6, 0.8)", "line(normal, 0.4, 0.5)"],
        ["line(normal, 0.8, 0.3)", "arc(square, 0.5, 0.7)", "line(normal, 0.5, 0.6)"]
    ]
    negative_examples = [
        ["line(zigzag, 0.3, 0.6)", "arc(circle, 0.4, 0.9)", "line(triangle, 0.3, 0.7)"],
        ["line(zigzag, 0.2, 0.5)", "arc(circle, 0.5, 0.8)", "line(triangle, 0.4, 0.6)"]
    ]
    features = extract_problem_level_features(positive_examples, negative_examples)
    print("Discriminative symbolic features:", features)
