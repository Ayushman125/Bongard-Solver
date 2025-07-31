import random
import logging


def add_stroke(program):
    # Add a random straight stroke to a list of (cmd, param) tuples
    prog = program.copy()
    prog.append(('fd', random.randint(10, 40)))
    prog.append(('rt', random.choice([45, 90, 120])))
    return prog


def reflect_shape(program):
    # Reflect sequence of turtle commands (mirroring effect) on a list of (cmd, param) tuples or LOGO strings
    # import logging removed; use global logging
    import re
    reflected = []
    for i, item in enumerate(program):
        # Accept both tuple and string formats
        if isinstance(item, tuple) and len(item) == 2:
            cmd, param = item
        elif isinstance(item, str):
            # Try to parse LOGO string command
            m = re.match(r'(fd|bk|rt|lt)\s*(-?\d+(?:\.\d+)?)', item.strip())
            if m:
                cmd, param = m.group(1), float(m.group(2))
            else:
                # Try to parse Bongard-LOGO format (e.g. line_normal_0.354-0.500)
                parts = item.split('_')
                if len(parts) >= 2 and parts[0] in {'line', 'arc'}:
                    cmd = parts[0]
                    param = '_'.join(parts[1:])
                else:
                    logging.warning(f"reflect_shape: Skipping malformed item at index {i}: {item!r}")
                    continue
        else:
            logging.warning(f"reflect_shape: Skipping malformed item at index {i}: {item!r}")
            continue
        # Mirror right/left turns, otherwise keep as is
        if cmd == 'rt':
            reflected.append(('lt', param))
        elif cmd == 'lt':
            reflected.append(('rt', param))
        else:
            reflected.append((cmd, param))
    return reflected


def split_edge(program):
    # Insert vertex between two points (rudimentary) for a list of (cmd, param) tuples
    prog = program.copy()
    if len(prog) < 2:
        return prog
    idx = random.randint(0, len(prog)-2)
    cmd = prog[idx]
    next_cmd = prog[idx+1]
    if isinstance(cmd, tuple) and isinstance(next_cmd, tuple):
        midpoint = (cmd[1] + next_cmd[1]) / 2 if isinstance(cmd[1], (int, float)) and isinstance(next_cmd[1], (int, float)) else cmd[1]
        prog.insert(idx+1, (cmd[0], midpoint))
    return prog

RULE_SET = [add_stroke, reflect_shape, split_edge]


def mutate(program, rule_id=None, rng=None):
    rule = RULE_SET[rule_id if rule_id is not None else random.randint(0, len(RULE_SET)-1)]
    return rule(program)
