# ────────────────── concept_inversions.py ──────────────────
def close_arc_60(cmds):
    return [
        ("line", "0.500-0.500") if cmd.startswith("arc") else (cmd, param)
        for cmd, param in cmds
    ]

def rotate_triangle_30(cmds):
    return [("rotate", "30")] + cmds

def insert_partition(cmds):
    return cmds + [("line", "0.200-0.800")]

CONCEPT_INVERSION_STRATEGIES = {
    "bd_two_equilateral_triangles-open_symm_arc60": [
        close_arc_60,
        rotate_triangle_30,
    ],
    "bd_trapez_acute_hexagon-no_two_parts_sector7": [
        insert_partition,
    ],
    # add other concept keys as needed
}
# ─────────────────────────────────────────────────────────────
# per-concept deterministic inversion operators

# --- Deterministic inversion strategies for symmetry/arrangement concepts ---

# --- Deterministic inversion strategies for symmetry/arrangement concepts ---
def close_arc_60(cmds):
    # Replace arc_zigzag_0.500_0.583-0.500 with a line to break open_arc60
    return [
        ("line", "0.500-0.500") if (isinstance(cmd, str) and cmd.startswith("arc_zigzag_0.500_0.583-0.500")) or (isinstance(cmd, tuple) and str(cmd[0]).startswith("arc_zigzag_0.500_0.583-0.500")) else (cmd, param) if isinstance(cmd, tuple) else (cmd, None)
        for cmd, param in cmds
    ]

def rotate_triangle_30(cmds):
    # Prepend a 30 degree rotation to break triangle symmetry
    return [("rotate", "30")] + cmds

def insert_partition(cmds):
    # Insert a partition line to break sector arrangement
    return cmds + [("line", "0.200-0.800")]

CONCEPT_INVERSION_STRATEGIES = {
    "bd_two_equilateral_triangles-open_symm_arc60": [
        close_arc_60,
        rotate_triangle_30,
    ],
    "bd_trapez_acute_hexagon-no_two_parts_sector7": [
        insert_partition,
    ],
    # Add more concept/problem keys and their inversion functions here
}
