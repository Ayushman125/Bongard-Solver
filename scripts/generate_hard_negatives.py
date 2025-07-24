import random
import sys
import os
import argparse
import ijson
import json
import re
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference
from src.data_pipeline.verification import Verification
from src.concepts import CONCEPTS


# Structural perturbation
def structural_perturb(cmds):
    cmds = cmds.copy()
    op = random.choice(['delete','duplicate','swap','toggle'])
    if op == 'delete' and len(cmds) > 3:
        cmds.pop(random.randrange(len(cmds)))
    elif op == 'duplicate':
        cmds.insert(random.randrange(len(cmds)), random.choice(cmds))
    elif op == 'swap' and len(cmds) > 1:
        i,j = random.sample(range(len(cmds)),2); cmds[i],cmds[j]=cmds[j],cmds[i]
    elif op == 'toggle':
        k = random.randrange(len(cmds))
        cmds[k] = cmds[k].replace('line_','arc_') if cmds[k].startswith('line_') else cmds[k].replace('arc_','line_')
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
    op = random.choice(["delete", "duplicate", "swap", "toggle_style"])
    if op == "delete" and len(cmds) > 3:
        cmds.pop(random.randrange(len(cmds)))
    elif op == "duplicate":
        cmds.insert(random.randrange(len(cmds)), random.choice(cmds))
    elif op == "swap" and len(cmds) > 3:
        i, j = random.sample(range(len(cmds)), 2)
        cmds[i], cmds[j] = cmds[j], cmds[i]
    elif op == "toggle_style":
        k = random.randrange(len(cmds))
        cmds[k] = cmds[k].replace("line_", "arc_", 1) if cmds[k].startswith("line_") \
                 else cmds[k].replace("arc_", "line_", 1)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jitter-angle', type=float, default=5, help='Base angle jitter for centroid perturbation')
    parser.add_argument('--jitter-length', type=float, default=0.05, help='Base length jitter for area perturbation')
    parser.add_argument('--max-per-problem', type=int, default=14, help='Max hard negatives per problem')
    parser.add_argument('--trials-per-sample', type=int, default=80, help='Trials per positive sample')
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
    total_problems = 0
    total_samples = 0
    total_hard_negatives = 0

    parser_obj = BongardLogoParser()
    physics_infer = PhysicsInference()

    for pid, entries in problems.items():
        total_problems += 1
        labels = [e.get('label') for e in entries]
        print(f"Problem {total_problems}: {pid} labels: {labels}")
        positives = [e for e in entries if e.get('label') == 'category_1']
        print(f"Processing problem {total_problems}: {pid} ({len(positives)} positives)")
        count = 0
        concept_fn = CONCEPTS.get(pid, concept_test)
        for sample_idx, sample in enumerate(positives):
            if count >= args.max_per_problem:
                break
            total_samples += 1
            try:
                commands = sample.get('action_program')
                if not commands or not isinstance(commands, list):
                    print(f"    [ERROR] No valid action_program for sample {sample_idx+1}")
                    continue
                flat_commands = commands
                original_features = sample['features']
            except Exception as e:
                print(f"    [ERROR] Failed to get features for sample: {e}")
                continue
            flips = 0
            trials = 0
            ang_jit, len_jit = args.jitter_angle, args.jitter_length
            while flips < 2 and trials < args.trials_per_sample:
                trials += 1
                cmds1 = structural_perturb(flat_commands)
                cmds2 = numeric_jitter(cmds1, ang_jit, len_jit)
                try:
                    try:
                        verts = parser_obj.parse_action_program(cmds2, scale=120)
                    except ValueError as e:
                        if "math domain error" in str(e):
                            print(f"    [SKIP] perturb parse math error: {e}")
                            continue
                        else:
                            raise
                    poly = PhysicsInference.polygon_from_vertices(verts)
                    if poly is None or not poly.is_valid:
                        continue
                    feats = {
                        **original_features,
                        'area': poly.area,
                        'centroid': [poly.centroid.x, poly.centroid.y],
                        'is_convex': PhysicsInference.is_convex(poly),
                        'symmetry_score': PhysicsInference.symmetry_score(poly),
                        'moment_of_inertia': PhysicsInference.moment_of_inertia(poly),
                        'num_straight': PhysicsInference.count_straight_segments(poly),
                        'num_arcs': PhysicsInference.count_arcs(poly),
                        'has_quadrangle': PhysicsInference.has_quadrangle(poly),
                        'has_obtuse_angle': PhysicsInference.has_obtuse(poly),
                    }
                except Exception as e:
                    print(f"    [ERROR] Failed to parse perturbed program: {e}")
                    continue
                if concept_fn(original_features) != concept_fn(feats):
                    entry = f"{pid},{sample.get('image_path','')},perturb:{cmds2}"
                    hard_negatives.append(entry)
                    flips += 1
                    count += 1
                    total_hard_negatives += 1
                    print(f"    [FOUND] Hard negative for sample {sample_idx+1}")
            if flips == 0:
                print(f"    [NO FLIP] No hard negative found for sample {sample_idx+1}")

    with open(args.output, 'w') as f:
        for hn in hard_negatives:
            f.write(hn + '\n')
    print(f"\nSummary:")
    print(f"  Problems processed: {total_problems}")
    print(f"  Samples processed: {total_samples}")
    print(f"  Hard negatives found: {total_hard_negatives}")

if __name__ == "__main__":
    main()
