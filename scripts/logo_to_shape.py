


import argparse
import sys
import os
import ijson
import json
import logging

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline.loader import BongardLoader
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference
from src.data_pipeline.verification import Verification

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--problems-list', default=None)
    parser.add_argument('--n-select', type=int, default=50)
    args = parser.parse_args()

    loader = BongardLoader(args.input_dir, problems_list=args.problems_list, n_select=args.n_select)
    logo_parser = BongardLogoParser()
    data = []
    flagged = []
    total = 0
    valid = 0
    skipped = 0

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    categories = ['bd', 'ff', 'hd']
    base_dir = args.input_dir
    output_path = args.output
    problems_list_path = args.problems_list

    # Read problem IDs from problems-list file
    problem_ids = set()
    if args.problems_list:
        with open(args.problems_list) as f:
            problem_ids = set(line.strip() for line in f if line.strip())

    # For each category, load the JSON and process problems
    required_ids = None
    if problems_list_path:
        with open(problems_list_path, 'r') as f:
            required_ids = set(line.strip() for line in f if line.strip())

    all_results = []
    flagged_cases = []

    for cat in categories:
        json_path = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/{cat}_action_programs.json")
        if not os.path.exists(json_path):
            print(f"WARNING: Missing JSON file: {json_path}")
            continue

        with open(json_path, 'r') as f:
            for problem_id, pos_neg_lists in ijson.kvitems(f, ''):
                if required_ids and problem_id not in required_ids:
                    continue
                # Each problem: [positives, negatives]
                for label, group in zip(['category_1', 'category_0'], pos_neg_lists):
                    for idx, action_program in enumerate(group):
                        img_dir = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/images/{problem_id}", label)
                        img_path = os.path.join(img_dir, f"{idx}.png")
                        # Debug: print mapping and action_program type
                        if idx == 0:
                            print(f"Processing {problem_id} {label} -> {img_path}")
                            print(f"Action program type: {type(action_program)}, value: {action_program}")
                        # Flatten action_program if it's a list of lists (common in JSON)
                        if isinstance(action_program, list) and len(action_program) == 1 and isinstance(action_program[0], list):
                            action_program = action_program[0]
                        try:
                            # If action_program is a list of subgroups, parse each subgroup in sequence, persisting turtle state
                            vertices = []
                            if isinstance(action_program, list) and all(isinstance(subgroup, list) for subgroup in action_program):
                                for subgroup in action_program:
                                    pts = logo_parser.parse_action_program(subgroup, scale=120)
                                    vertices.extend(pts)
                            else:
                                vertices = logo_parser.parse_action_program(action_program, scale=120)
                            print(f"Total vertices for {problem_id} {label}: {len(vertices)}")
                            if len(vertices) < 4:
                                flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': 'too_few_vertices'})
                                continue
                            poly = PhysicsInference.polygon_from_vertices(vertices)
                            if poly is None or not poly.is_valid:
                                flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': 'Invalid polygon'})
                                continue
                            features = {
                                'centroid': PhysicsInference.centroid(poly),
                                'area': PhysicsInference.area(poly),
                                'is_convex': PhysicsInference.is_convex(poly),
                                'symmetry_score': PhysicsInference.symmetry_score(vertices),
                                'moment_of_inertia': PhysicsInference.moment_of_inertia(vertices),
                            }
                            result = {
                                'problem_id': problem_id,
                                'category': cat,
                                'label': label,
                                'image_path': img_path,
                                'features': features,
                                'geometry': vertices,
                                'action_program': action_program if isinstance(action_program, list) else [action_program],
                            }
                            all_results.append(result)
                        except Exception as e:
                            flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': str(e)})
    # Add helper to LogoParser for direct command list parsing

    with open(output_path, 'w') as out:
        json.dump(all_results, out, indent=2)
    print(f"INFO: Saved {len(all_results)} valid samples to {output_path} (skipped {len(flagged_cases)} of {len(all_results) + len(flagged_cases)})")
    flagged_path = os.path.join(os.path.dirname(output_path), 'flagged_cases.txt')
    with open(flagged_path, 'w') as out:
        for case in flagged_cases:
            out.write(json.dumps(case) + '\n')
    print(f"INFO: Flagged {len(flagged_cases)} cases for review in {flagged_path}")

if __name__ == '__main__':
    main()
