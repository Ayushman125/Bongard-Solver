
import logging
from src.physics_inference import (
    symbolic_concept_features,
    extract_problem_level_features,
    analyze_action_structure
)
logger = logging.getLogger(__name__)

def ensure_str_list(obj):
    """Recursively convert all items in a list (or nested lists/tuples) to strings."""
    if isinstance(obj, list):
        return [ensure_str_list(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_str_list(x) for x in obj)
    elif not isinstance(obj, str):
        if hasattr(obj, 'raw_command') and obj.raw_command is not None:
            return str(obj.raw_command)
        return str(obj)
    return obj

def extract_topological_features(action_sequence):
    """
    Extract symbolic topological features from a LOGO action sequence.
    Returns abstract concepts such as connectivity, compositional structure, and symbolic attributes.
    """
    return symbolic_concept_features(action_sequence)

def extract_multiscale_features(action_sequence):
    """
    Extract symbolic features at multiple levels of abstraction from a LOGO action sequence.
    Returns compositional and hierarchical structure, not geometric scales.
    """
    return analyze_action_structure(action_sequence)

def extract_relational_features(action_sequence):
    """
    Extract symbolic relational features from LOGO action sequences.
    Returns abstract relationships such as containment, adjacency, and compositional links.
    """
    return symbolic_concept_features(action_sequence)

def extract_problem_level_features(positive_examples, negative_examples):
    """
    Extract features by comparing positive vs negative LOGO action sequences.
    Returns discriminative, context-dependent patterns.
    """
    return extract_problem_level_features(positive_examples, negative_examples)

