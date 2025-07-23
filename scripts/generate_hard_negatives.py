import random
from src.data_pipeline.logo_parser import LogoParser

def perturb_logo_commands(commands, angle_delta=15, length_jitter=0.05):
    new_commands = []
    for cmd, val in commands:
        if cmd in 'FB':
            val = val * (1 + random.uniform(-length_jitter, length_jitter))
        elif cmd in 'RL':
            val = val + random.choice([-angle_delta, angle_delta])
        new_commands.append((cmd, val))
    return new_commands

def save_logo(commands, path):
    with open(path, 'w') as f:
        for cmd, val in commands:
            f.write(f"{cmd} {val}\n")

# Usage: For each .logo in positives, perturb, save as new .logo, verify if it becomes a negative.
