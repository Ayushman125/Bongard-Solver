import random
import sys
import os
import argparse
import ijson
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
    parser.add_argument('--jitter-angle', type=float, default=5)
    parser.add_argument('--jitter-length', type=float, default=0.05)
    parser.add_argument('--max-per-problem', type=int, default=3)
    args = parser.parse_args()

    logo_parser = BongardLogoParser()
    physics = PhysicsInference()
    hard_negatives = []
    total_problems = 0
    total_samples = 0
    total_logo_missing = 0
    total_hard_negatives = 0

    # Always load derived_labels.json from canonical location
    input_json = os.path.join(os.path.dirname(__file__), '..', 'data', 'derived_labels.json')
    input_json = os.path.abspath(input_json)
    if not os.path.exists(input_json):
        print(f"Error: derived_labels.json not found at {input_json}")
        return
    print(f"Loading derived_labels.json from: {input_json}")
    with open(input_json, 'r') as f:
        objects = ijson.items(f, 'item')
        for problem_idx, problem in enumerate(objects):
            pid = problem.get('problem_id', '')
            positives = problem.get('positives', [])
            count = 0
            total_problems += 1
            print(f"Processing problem {problem_idx+1}: {pid} ({len(positives)} positives)")
            for sample_idx, sample in enumerate(positives):
                if count >= args.max_per_problem:
                    break
                total_samples += 1
                # Use only features from derived_labels.json
                original_features = {
                    'area': sample.get('area', 0),
                    'centroid': sample.get('centroid', [0, 0])
                }
                found_hard_negative = False
                for _ in range(10):  # Try up to 10 perturbations
                    # Simulate perturbation: jitter area and centroid
                    perturbed_features = {
                        'area': original_features['area'] * (1 + random.uniform(-args.jitter_length, args.jitter_length)),
                        'centroid': [
                            original_features['centroid'][0] + random.uniform(-args.jitter_angle, args.jitter_angle),
                            original_features['centroid'][1] + random.uniform(-args.jitter_angle, args.jitter_angle)
                        ]
                    }
                    if flips_label(original_features, perturbed_features, concept_test):
                        entry = f"{pid},{sample['category']},{sample['image_index']},{original_features}->{perturbed_features}"
                        hard_negatives.append(entry)
                        count += 1
                        total_hard_negatives += 1
                        found_hard_negative = True
                        print(f"    [FOUND] Hard negative for sample {sample_idx+1}")
                        break
                if not found_hard_negative:
                    print(f"    [NO FLIP] No hard negative found for sample {sample_idx+1}")

    # Write output
    with open(args.output, 'w') as fout:
        for entry in hard_negatives:
            fout.write(entry + '\n')
    print(f"\nSummary:")
    print(f"  Problems processed: {total_problems}")
    print(f"  Samples processed: {total_samples}")
    print(f"  Hard negatives found: {total_hard_negatives}")

if __name__ == "__main__":
    main()
