# per-concept deterministic inversion operators
def reflect_one_triangle(cmds):
    # example: replace first triangle with its mirror
    return [("reflect", "vertical")] + cmds

def close_arc(cmds):
    # example: find arc command and convert to line
    return [("line", param) if cmd.startswith("arc") else (cmd, param)
            for cmd, param in cmds]

CONCEPT_INVERSION_STRATEGIES = {
    "bd_two_equilateral_triangles-open_symm_arc60": [
        reflect_one_triangle,
        close_arc,
    ],
    # add other problem-specific lists here
}
