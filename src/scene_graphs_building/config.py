# ConceptNet whitelist of relation types to retain
CONCEPTNET_KEEP_RELS = {
    "PartOf", "HasA", "AtLocation", "MadeOf", "DerivedFrom",
    "RelatedTo", "IsA", "UsedFor", "CapableOf", "Desires",
    "HasProperty", "LocatedNear", "SimilarTo", "SymbolOf",
    "HasSubevent", "CausesDesire", "MotivatedByGoal", "ObstructedBy",
    "Causes", "HasPrerequisite", "HasFirstSubevent", "HasLastSubevent",
    "DefinedAs", "MannerOf", "ReceivesAction", "HasContext",
    "EntailsEvent", "InstanceOf", "ExternalURL", "EtymologicallyRelatedTo",
    "FormOf", "Antonym", "Synonym", "DistinctFrom", "NotUsedFor"
}


# Map raw labels (from NV-Logo metadata or motif miner) to normalized shapes
SHAPE_MAP = {
    "line": "line",
    "circle": "circle",
    "arc": "arc",
    "point": "point",
    "dot": "dot",
    "tri": "triangle",
    "triangle": "triangle",
    "quad": "quadrilateral",
    "rectangle": "rectangle",
    "square": "square",
    "pentagon": "pentagon",
    "hexagon": "hexagon",
    "heptagon": "heptagon",
    "octagon": "octagon",
    "polygon": "polygon",
    "parabola": "parabola",
    "bezier": "bezier curve",
    "curve": "curve",
    "spline": "spline",
    "zigzag": "zigzag",
    "star": "star",
    "cross": "cross",
    "hook": "hook",
    "y shape": "Y-shape",
    "t junction": "T-junction",
    "fork": "fork",
    "spiral": "spiral",
    "cluster": "cluster",
    "ring": "ring",
    "chain": "chain",
    "grid": "grid",
    "segment": "segment",
    "junction": "junction",
    "vertex": "vertex",
    # Add more as needed
}

# Commonsense label mapping for KB edge creation
COMMONSENSE_LABEL_MAP = {
    "line": "line",
    "arc": "curve",
    "point": "point",
    "dot": "point",
    "triangle": "triangle",
    "quad": "quadrilateral",
    "quadrilateral": "quadrilateral",
    "polygon": "polygon",
    "circle": "circle",
    "cluster": "group",
    "ring": "circle",
    "chain": "sequence",
    "grid": "grid",
    "star": "star",
    "rectangle": "rectangle",
    "square": "square",
    "arrow": "arrow",
    "curve": "curve",
    "spiral": "spiral",
    "cross": "cross",
    "hook": "hook",
    "vertex": "point",
    "junction": "intersection",
    "segment": "line",
    "unknown": "shape",
    # Extend as you add more motif/shape computing
}

# Motif categories and semantic types (for supernodes)
MOTIF_CATEGORIES = [
    "sequence", "group", "grid", "enclosed", "symmetry", "repeat", "adjacent", "enclosure", "connects-to", "mirrored", "spiral", "reflection", "loop"
]

# Standard semantic fields for objects and motifs
SEMANTIC_FIELDS = [
    "object_id", "vertices", "object_type", "is_closed", "fallback_geometry", "semantic_label", "shape_label", "motif_label", "motif_type", "kb_concept", "pattern_role", "action_program_type", "function_label", "gnn_score", "vl_embed"
]

# LOGO geometric predicate registry for edge construction
import numpy as np
BASIC_LOGO_PREDICATES = {
    'adjacent_endpoints': lambda a, b: (
        a.get('object_type') == 'line' and
        b.get('object_type') == 'line' and
        any(np.allclose(np.array(pt1), np.array(pt2), atol=1e-3)
            for pt1 in a.get('endpoints', [])
            for pt2 in b.get('endpoints', []))
    ),
    'length_sim': lambda a, b: (
        a.get('length') and b.get('length') and
        abs(a['length'] - b['length']) < 0.05 * max(a['length'], b['length'])
    ),
    'angle_sim':  lambda a, b: (
        a.get('orientation') is not None and
        b.get('orientation') is not None and
        abs(a['orientation'] - b['orientation']) < 5
    ),
}
