import random

def add_stroke(program):
    # Add a random straight stroke
    program.commands.append(('fd', random.randint(10, 40)))
    program.commands.append(('rt', random.choice([45, 90, 120])))
    return program

def reflect_shape(program):
    # Reflect sequence of turtle commands (mirroring effect)
    reflected = program.copy()
    for i, (cmd, param) in enumerate(reflected.commands):
        if cmd == 'rt':
            reflected.commands[i] = ('lt', param)
        elif cmd == 'lt':
            reflected.commands[i] = ('rt', param)
    return reflected

def split_edge(program):
    # Insert vertex between two points (rudimentary)
    idx = random.randint(0, len(program.commands)-2)
    cmd = program.commands[idx]
    next_cmd = program.commands[idx+1]
    midpoint = (cmd[1] + next_cmd[1]) / 2
    program.commands.insert(idx+1, (cmd[0], midpoint))
    return program

RULE_SET = [add_stroke, reflect_shape, split_edge]

def mutate(program, rule_id=None, rng=None):
    rule = RULE_SET[rule_id or random.randint(0, len(RULE_SET)-1)]
    return rule(program)
