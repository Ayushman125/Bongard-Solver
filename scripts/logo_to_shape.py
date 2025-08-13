# --- Symbolic and Compositional Feature Extraction ---
import re
from collections import Counter

def parse_logo_action(action_str):
    """Parse a LOGO action command into symbolic components."""
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

def extract_symbolic_concepts(action_sequence):
    """Extract symbolic and compositional features from a sequence of LOGO actions."""
    concepts = []
    for action_str in action_sequence:
        parsed = parse_logo_action(action_str)
        if parsed:
            concepts.append(parsed)
    stroke_types = [c['type'] for c in concepts]
    modifiers = [c['modifier'] for c in concepts]
    length_categories = ['short' if c['length'] < 0.5 else 'long' for c in concepts]
    angle_categories = ['small' if c['angle'] < 0.5 else 'large' for c in concepts]
    return {
        'stroke_types': stroke_types,
        'modifiers': modifiers,
        'length_categories': length_categories,
        'angle_categories': angle_categories,
        'num_strokes': len(concepts)
    }

def analyze_action_structure(action_sequence):
    symbolic = extract_symbolic_concepts(action_sequence)
    pattern_counts = Counter(zip(symbolic['stroke_types'], symbolic['modifiers']))
    return {
        'sequence_patterns': dict(pattern_counts),
        'hierarchical_structure': None,
        'compositional_rules': symbolic
    }

def extract_problem_level_features(positive_examples, negative_examples):
    pos_features = [analyze_action_structure(seq) for seq in positive_examples]
    neg_features = [analyze_action_structure(seq) for seq in negative_examples]
    def summarize(features):
        all_types = [f['compositional_rules']['stroke_types'] for f in features]
        all_mods = [f['compositional_rules']['modifiers'] for f in features]
        return {
            'stroke_types': Counter([t for sub in all_types for t in sub]),
            'modifiers': Counter([m for sub in all_mods for m in sub])
        }
    pos_summary = summarize(pos_features)
    neg_summary = summarize(neg_features)
    discriminative_types = set(pos_summary['stroke_types']) - set(neg_summary['stroke_types'])
    discriminative_mods = set(pos_summary['modifiers']) - set(neg_summary['modifiers'])
    return {
        'discriminative_types': list(discriminative_types),
        'discriminative_modifiers': list(discriminative_mods)
    }

if __name__ == "__main__":
    # Example usage: Replace with actual data loading as needed
    positive_examples = [
        ["line(normal, 0.7, 0.2)", "arc(square, 0.6, 0.8)", "line(triangle, 0.4, 0.5)"],
        ["line(normal, 0.8, 0.3)", "arc(square, 0.5, 0.7)", "line(triangle, 0.5, 0.6)"]
    ]
    negative_examples = [
        ["line(zigzag, 0.3, 0.6)", "arc(circle, 0.4, 0.9)", "line(normal, 0.3, 0.7)"],
        ["line(zigzag, 0.2, 0.5)", "arc(circle, 0.5, 0.8)", "line(normal, 0.4, 0.6)"]
    ]
    features = extract_problem_level_features(positive_examples, negative_examples)
    print("Discriminative symbolic features:", features)