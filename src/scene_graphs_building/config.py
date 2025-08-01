# ConceptNet whitelist of relation types to retain
CONCEPTNET_KEEP_RELS = {
    "PartOf", "HasA", "AtLocation", "MadeOf", "DerivedFrom"
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
    "cluster": "group",
    "ring": "circle",
    "chain": "sequence",
    "grid": "grid",
    "star": "star",
    "rectangle": "rectangle",
    "arrow": "arrow",
    "arc": "arc",
    "triangle": "triangle",
    "polygon": "polygon",
    "curve": "curve",
    "point": "point",
    "line": "line",
    "circle": "circle",
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
