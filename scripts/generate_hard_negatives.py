import random
import sys
import os
import argparse
import ijson
import json
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference

def perturb_logo_commands(commands, angle_jitter=5, length_jitter=0.05):
    new_commands = []
    for cmd, val in commands:
        if cmd in 'FB':
            val = val * (1 + random.uniform(-length_jitter, length_jitter))
        elif cmd in 'RL':
            val = val + random.uniform(-angle_jitter, angle_jitter)
        new_commands.append((cmd, val))
    return new_commands

def commands_to_str(commands):
    return ' '.join([f"{cmd} {val:.3f}" for cmd, val in commands])

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

    # Load derived labels (flat array, NVLabs convention)
    derived_labels_path = os.path.join('data', 'derived_labels.json')
    if not os.path.exists(derived_labels_path):
        print(f"Missing derived_labels.json at {derived_labels_path}")
        return
    with open(derived_labels_path, 'r') as f:
        all_entries = json.load(f)

    # Group by problem_id using defaultdict
    problems = defaultdict(list)
    for entry in all_entries:
        problems[entry['problem_id']].append(entry)

    hard_negatives = []
    total_problems = 0
    total_samples = 0
    total_hard_negatives = 0

    for pid, entries in problems.items():
        total_problems += 1
        labels = [e.get('label') for e in entries]
        print(f"Problem {total_problems}: {pid} labels: {labels}")
        positives = [e for e in entries if e.get('label') == 'category_1']
        negatives = [e for e in entries if e.get('label') == 'category_0']
        print(f"Processing problem {total_problems}: {pid} ({len(positives)} positives)")
        count = 0
        for sample_idx, sample in enumerate(positives):
            if count >= args.max_per_problem:
                break
            total_samples += 1
            try:
                original_features = {
                    'area': sample.get('area', 0),
                    'centroid': sample.get('centroid', [0,0])
                }
            except Exception as e:
                print(f"    [ERROR] Failed to get features for sample: {e}")
                continue
            found_hard_negative = False
            for _ in range(10):
                pert_features = {
                    'area': original_features['area'] * (1 + random.uniform(-args.jitter_length, args.jitter_length)),
                    'centroid': [
                        original_features['centroid'][0] + random.uniform(-args.jitter_angle, args.jitter_angle),
                        original_features['centroid'][1] + random.uniform(-args.jitter_angle, args.jitter_angle)
                    ]
                }
                if flips_label(original_features, pert_features, concept_test):
                    entry = f"{pid},{sample.get('image_path','')},area:{original_features['area']:.2f}->{pert_features['area']:.2f},centroid:{original_features['centroid']}->{pert_features['centroid']}"
                    hard_negatives.append(entry)
                    count += 1
                    total_hard_negatives += 1
                    found_hard_negative = True
                    print(f"    [FOUND] Hard negative for sample {sample_idx+1}")
                    break
            if not found_hard_negative:
                print(f"    [NO FLIP] No hard negative found for sample {sample_idx+1}")

    # Save output
    with open(args.output, 'w') as f:
        for hn in hard_negatives:
            f.write(hn + '\n')
    print(f"\nSummary:")
    print(f"  Problems processed: {total_problems}")
    print(f"  Samples processed: {total_samples}")
    print(f"  Hard negatives found: {total_hard_negatives}")

if __name__ == "__main__":
    main()
