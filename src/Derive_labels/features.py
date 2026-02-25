import logging
logger = logging.getLogger("features")
def ensure_all_strings(obj):
    """Recursively convert all items in a list (or nested lists/tuples) to strings."""
    if isinstance(obj, list):
        return [ensure_all_strings(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_all_strings(x) for x in obj)
    elif not isinstance(obj, str):
        if hasattr(obj, 'raw_command') and obj.raw_command is not None:
            return str(obj.raw_command)
        return str(obj)
    return obj
def extract_topological_features(action_sequence, context_memory=None):
    logger.info(f"[extract_topological_features] action_sequence input: {repr(action_sequence)}")
    # Flatten input if nested
    if isinstance(action_sequence, list) and action_sequence and isinstance(action_sequence[0], list):
        flat_sequence = [cmd for seq in action_sequence for cmd in seq]
    else:
        flat_sequence = action_sequence
    # Stage 1: Mine frequent stroke-pattern motifs
    motifs = EmergenceDetector.mine_stroke_patterns([flat_sequence])
    logger.info(f"[extract_topological_features] motifs: {repr(motifs)}")
    # Stage 2: Detect emergent concepts from motifs
    emergent = EmergenceDetector.detect_emergent_concepts(motifs, context_memory)
    logger.info(f"[extract_topological_features] emergent: {repr(emergent)}")
    # Stage 3: Abstract to higher-level concepts
    abstracted = AbstractionHierarchy.abstract(emergent)
    logger.info(f"[extract_topological_features] abstracted: {repr(abstracted)}")
    # Stage 4: Integrate into memory for cross-problem generalization
    ConceptMemoryBank.integrate(abstracted)
    # Stage 5: Build scene graph embedding
    scene_graph_emb = BongardSceneGraphProcessor.build_scene_graph([flat_sequence])
    # Stage 6: Compose numerical feature vector
    motif_num = len(motifs)
    emergent_num = len(emergent)
    abstracted_num = len(abstracted)
    feature_vector = np.array([motif_num, emergent_num, abstracted_num] + list(scene_graph_emb))
    return feature_vector
def extract_multiscale_features(action_sequence):
    """
    Extract symbolic features at multiple levels of abstraction from a LOGO action sequence.
    Returns compositional and hierarchical structure, not geometric scales.
    """
    # Use local analyze_action_structure from shape_utils
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

import numpy as np
from src.physics_inference import symbolic_concept_features
from src.Derive_labels.emergence import BongardSceneGraphProcessor, EmergenceDetector, AbstractionHierarchy, ConceptMemoryBank
from src.Derive_labels.shape_utils import analyze_action_structure

def extract_topological_features(action_sequence, context_memory=None):
    logger.info(f"[extract_topological_features] action_sequence input: {repr(action_sequence)}")
    motifs = EmergenceDetector.mine_stroke_patterns(action_sequence)
    logger.info(f"[extract_topological_features] motifs: {repr(motifs)}")
    emergent = EmergenceDetector.detect_emergent_concepts(motifs, context_memory)
    logger.info(f"[extract_topological_features] emergent: {repr(emergent)}")
    abstracted = AbstractionHierarchy.abstract(emergent)
    logger.info(f"[extract_topological_features] abstracted: {repr(abstracted)}")
    ConceptMemoryBank.integrate(abstracted)
    scene_graph_emb = BongardSceneGraphProcessor.build_scene_graph(action_sequence)
    motif_num = len(motifs)
    emergent_num = len(emergent)
    abstracted_num = len(abstracted)
    feature_vector = np.array([motif_num, emergent_num, abstracted_num] + list(scene_graph_emb))
    logger.info(f"[extract_topological_features] feature_vector: {feature_vector}")
    return feature_vector

def extract_multiscale_features(action_sequence):
    """
    Extract symbolic features at multiple levels of abstraction from a LOGO action sequence.
    Returns compositional and hierarchical structure.
    """
    return analyze_action_structure(action_sequence)

def extract_relational_features(action_sequence):
    """
    Extract symbolic relational features from LOGO action sequences.
    Returns abstract relationships such as containment, adjacency, and compositional links.
    """
    # Use analyze_action_structure for relational features
    from src.Derive_labels.shape_utils import analyze_action_structure
    structure = analyze_action_structure(action_sequence)
    return structure.get('compositional_rules', [])

def extract_problem_level_features(positive_examples, negative_examples):
    """
    Extract features by comparing positive vs negative LOGO action sequences.
    Returns discriminative, context-dependent patterns.
    """
    # Use shape_utils for discriminative features
    from src.Derive_labels.shape_utils import extract_problem_level_features as shape_extract_problem_level_features
    return shape_extract_problem_level_features(positive_examples, negative_examples)

