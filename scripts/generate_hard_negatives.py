import random
import sys
import os
import argparse
import ijson
import json
import re
from collections import defaultdict
import logging
import multiprocessing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference
from src.data_pipeline.verification import Verification
from src.concepts import CONCEPTS
from src.hard_negative.evo_search import EvoPerturber
from src.data_pipeline.logo_mutator import mutate, RULE_SET
def is_valid_geometry(program):
    # Placeholder: checks for self-intersection, implausible polygons, etc.
    return True

def is_diverse(new, existing):
    # Placeholder: compute vector or label difference
    return True

def is_near_flip(conf, threshold=0.05):
    return abs(conf - 0.5) < threshold


# Structural perturbation
def structural_perturb(cmds):
    cmds = cmds.copy()
    ops = ['delete', 'duplicate', 'swap', 'toggle', 'insert_arc', 'reverse', 'block_swap']
    op = random.choice(ops)
    if op == 'delete' and len(cmds) > 3:
        cmds.pop(random.randrange(len(cmds)))
    elif op == 'duplicate':
        cmds.insert(random.randrange(len(cmds)), random.choice(cmds))
    elif op == 'swap' and len(cmds) > 1:
        i, j = random.sample(range(len(cmds)), 2)
        cmds[i], cmds[j] = cmds[j], cmds[i]
    elif op == 'toggle':
        k = random.randrange(len(cmds))
        cmds[k] = cmds[k].replace('line_', 'arc_') if cmds[k].startswith('line_') else cmds[k].replace('arc_', 'line_')
    elif op == 'insert_arc':
        idx = random.randrange(len(cmds)+1)
        arc_cmd = f"arc_normal_{random.random():.3f}-{random.random():.3f}"
        cmds.insert(idx, arc_cmd)
    elif op == 'reverse' and len(cmds) > 2:
        start = random.randrange(len(cmds)-1)
        end = random.randrange(start+1, len(cmds))
        cmds[start:end] = list(reversed(cmds[start:end]))
    elif op == 'block_swap' and len(cmds) > 4:
        block = random.sample(range(len(cmds)), 3)
        block.sort()
        if len(block) == 3:
            cmds[block[0]:block[2]+1] = cmds[block[0]:block[2]+1][::-1]
    return cmds

def numeric_jitter(cmds, ang_jit, len_jit):
    def jitter(c):
        if c.startswith('line_'):
            m = re.match(r'line_(\w+)_([\d.]+)-([\d.]+)', c)
            if m:
                style, L, A = m.groups()
                L = float(L) * (1 + random.uniform(-len_jit, len_jit))
                A = float(A) * (1 + random.uniform(-ang_jit / 360, ang_jit / 360))
                return f'line_{style}_{L:.3f}-{A:.3f}'
        if c.startswith('arc_'):
            parts = c.split('-')
            return parts[0] + '-' + parts[1]
        return c
    return [jitter(c) for c in cmds]

# Structural perturbation
def structural_perturb(cmds):
    cmds = cmds.copy()
    ops = ['delete', 'duplicate', 'swap', 'toggle', 'insert_arc', 'reverse', 'block_swap']
    op = random.choice(ops)
    if op == 'delete' and len(cmds) > 3:
        cmds.pop(random.randrange(len(cmds)))
    elif op == 'duplicate':
        cmds.insert(random.randrange(len(cmds)), random.choice(cmds))
    elif op == 'swap' and len(cmds) > 1:
        i, j = random.sample(range(len(cmds)), 2)
        cmds[i], cmds[j] = cmds[j], cmds[i]
    elif op == 'toggle':
        k = random.randrange(len(cmds))
        cmds[k] = cmds[k].replace('line_', 'arc_', 1) if cmds[k].startswith('line_') \
                 else cmds[k].replace('arc_', 'line_', 1)
    elif op == 'insert_arc':
        idx = random.randrange(len(cmds)+1)
        arc_cmd = f"arc_normal_{random.random():.3f}-{random.random():.3f}"
        cmds.insert(idx, arc_cmd)
    elif op == 'reverse' and len(cmds) > 2:
        start = random.randrange(len(cmds)-1)
        end = random.randrange(start+1, len(cmds))
        cmds[start:end] = list(reversed(cmds[start:end]))
    elif op == 'block_swap' and len(cmds) > 4:
        block = random.sample(range(len(cmds)), 3)
        block.sort()
        if len(block) == 3:
            cmds[block[0]:block[2]+1] = cmds[block[0]:block[2]+1][::-1]
    return cmds

def flatten_action_program(action_program):
    # Flatten nested lists to a single list of strings
    flat = []
    for item in action_program:
        if isinstance(item, list):
            flat.extend(flatten_action_program(item))
        else:
            flat.append(item)
    return flat

def flips_label(original_features, perturbed_features, concept_test):
    original_label = concept_test(original_features)
    perturbed_label = concept_test(perturbed_features)
    return original_label != perturbed_label

def concept_test(features):
    # Placeholder: implement your concept test logic here
    # For example, return True if area > threshold
    return features['area'] > 0.5


# Move process_sample to top-level
def process_sample(args_tuple):
    pid, sample, concept_fn, args = args_tuple
    result = None
    near_miss_result = None
    try:
        commands = sample.get('action_program')
        if not commands or not isinstance(commands, list):
            return None, None
        flat_commands = commands
        original_features = sample['features']
    except Exception:
        return None, None
    evo = EvoPerturber(scorer=concept_fn, seed=42)
    mutated = evo.search(flat_commands)
    if mutated and is_valid_geometry(mutated):
        # Assume concept_fn.is_flip and concept_fn.predict_concept_confidence exist
        if concept_fn.is_flip(mutated):
            result = f"{pid},{sample.get('image_path','')},evo:{mutated}"
        elif args.near_miss:
            conf = concept_fn.predict_concept_confidence(mutated)
            if is_near_flip(conf):
                near_miss_result = f"{pid},{sample.get('image_path','')},near_miss:{mutated}"
    # Fallback: shape-grammar mutation
    for rule_id in range(len(RULE_SET)):
        cand = mutate(flat_commands, rule_id)
        if is_valid_geometry(cand):
            if concept_fn.is_flip(cand):
                result = f"{pid},{sample.get('image_path','')},mutate:{cand}"
                break
            elif args.near_miss:
                conf = concept_fn.predict_concept_confidence(cand)
                if is_near_flip(conf):
                    near_miss_result = f"{pid},{sample.get('image_path','')},near_miss:{cand}"
    return result, near_miss_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jitter-angle', type=float, default=15, help='Base angle jitter for centroid perturbation')
    parser.add_argument('--jitter-length', type=float, default=0.15, help='Base length jitter for area perturbation')
    parser.add_argument('--max-per-problem', type=int, default=14, help='Max hard negatives per problem')
    parser.add_argument('--trials-per-sample', type=int, default=500, help='Trials per positive sample')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--near-miss', action='store_true', help='Store near-miss samples')
    args = parser.parse_args()

    derived_labels_path = os.path.join('data', 'derived_labels.json')
    if not os.path.exists(derived_labels_path):
        print(f"Missing derived_labels.json at {derived_labels_path}")
        return
    with open(derived_labels_path, 'r') as f:
        all_entries = json.load(f)

    problems = defaultdict(list)
    for entry in all_entries:
        problems[entry['problem_id']].append(entry)

    hard_negatives = []
    near_misses = []
    total_problems = 0
    total_samples = 0
    total_hard_negatives = 0

    parser_obj = BongardLogoParser()
    physics_infer = PhysicsInference()

    sample_args = []
    for pid, entries in problems.items():
        total_problems += 1
        labels = [e.get('label') for e in entries]
        print(f"Problem {total_problems}: {pid} labels: {labels}")
        positives = [e for e in entries if e.get('label') == 'category_1']
        print(f"Processing problem {total_problems}: {pid} ({len(positives)} positives)")
        concept_fn = CONCEPTS.get(pid, concept_test)
        for sample_idx, sample in enumerate(positives):
            sample_args.append((pid, sample, concept_fn, args))

    if args.parallel > 1:
        with multiprocessing.Pool(args.parallel) as pool:
            results = pool.map(process_sample, sample_args)
    else:
        results = [process_sample(arg) for arg in sample_args]

    for result, near_miss_result in results:
        if result:
            hard_negatives.append(result)
        if near_miss_result:
            near_misses.append(near_miss_result)

    with open(args.output, 'w') as f:
        for hn in hard_negatives:
            f.write(hn + '\n')
    if args.near_miss:
        with open(args.output.replace('.txt', '_nearmiss.txt'), 'w') as f:
            for nm in near_misses:
                f.write(nm + '\n')
    print(f"\nSummary:")
    print(f"  Problems processed: {total_problems}")
    print(f"  Hard negatives found: {len(hard_negatives)}")
    if args.near_miss:
        print(f"  Near-miss samples found: {len(near_misses)}")

if __name__ == "__main__":
    main()
