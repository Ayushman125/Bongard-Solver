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

def perturb_bongard_cmd(cmd_str, angle_jitter, length_jitter):
    if cmd_str.startswith('line_'):
        m = re.match(r'line_(\w+)_([0-9.]+)-([0-9.]+)', cmd_str)
        if m:
            style, L, A = m.groups()
            L = float(L) * (1 + random.uniform(-length_jitter, length_jitter))
            A = float(A) * (1 + random.uniform(-angle_jitter/360, angle_jitter/360))
            return f'line_{style}_{L:.3f}-{A:.3f}'
    if cmd_str.startswith('arc_'):
        m = re.match(r'arc_(\w+)_([0-9.]+)_([0-9.]+)-([0-9.]+)', cmd_str)
        if m:
            style, R, S, T = m.groups()
            R = float(R) * (1 + random.uniform(-length_jitter, length_jitter))
            S = float(S) * (1 + random.uniform(-angle_jitter/360, angle_jitter/360))
            return f'arc_{style}_{R:.3f}_{S:.3f}-{T}'
    return cmd_str

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
    parser.add_argument('--jitter-angle', type=float, default=5, help='Angle jitter for centroid perturbation')
    parser.add_argument('--jitter-length', type=float, default=0.05, help='Length jitter for area perturbation')
    parser.add_argument('--max-per-problem', type=int, default=3, help='Max hard negatives per problem')
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
        for sample_idx, sample in enumerate(positives):
            if count >= args.max_per_problem:
                break
            total_samples += 1
            try:
                # Use the true action program from the sample (flat list of LOGO command strings)
                commands = sample.get('action_program')
                if not commands or not isinstance(commands, list):
                    print(f"    [ERROR] No valid action_program for sample {sample_idx+1}")
                    continue
                # If commands is nested, flatten it
                if any(isinstance(cmd, list) for cmd in commands):
                    flat_commands = []
                    for subgroup in commands:
                        if isinstance(subgroup, list):
                            flat_commands.extend(subgroup)
                        else:
                            flat_commands.append(subgroup)
                else:
                    flat_commands = commands
                # Use precomputed features from sample['features']
                if 'features' not in sample:
                    print(f"    [ERROR] No features field in sample {sample_idx+1}")
                    continue
                original_features = sample['features']
            except Exception as e:
                print(f"    [ERROR] Failed to get features for sample: {e}")
                continue
            found_hard_negative = False
            for _ in range(10):
                # Perturb each command string (always operate on LOGO command strings)
                perturbed_commands = [perturb_bongard_cmd(cmd_str, args.jitter_angle, args.jitter_length) for cmd_str in flat_commands]
                try:
                    # Parse perturbed commands into vertices, then polygon
                    pert_verts = parser_obj.parse_action_program(perturbed_commands)
                    pert_poly = PhysicsInference.polygon_from_vertices(pert_verts)
                    if pert_poly is None or not hasattr(pert_poly, 'is_valid') or not pert_poly.is_valid:
                        continue
                    # Always build perturbed_features as a dict of scalars/lists
                    pert_features = {
                        'area': pert_poly.area,
                        'centroid': [pert_poly.centroid.x, pert_poly.centroid.y],
                        'is_convex': PhysicsInference.is_convex(pert_poly),
                        'symmetry_score': PhysicsInference.symmetry_score(pert_poly),
                        'moment_of_inertia': PhysicsInference.moment_of_inertia(pert_poly)
                    }
                except Exception as e:
                    print(f"    [ERROR] Failed to parse perturbed program: {e}")
                    continue
                if flips_label(original_features, pert_features, concept_test):
                    entry = f"{pid},{sample.get('image_path','')},perturbed_cmds:{perturbed_commands}"
                    hard_negatives.append(entry)
                    count += 1
                    total_hard_negatives += 1
                    found_hard_negative = True
                    print(f"    [FOUND] Hard negative for sample {sample_idx+1}")
                    break
            if not found_hard_negative:
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
