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
from src.data_pipeline.physics_infer import PhysicsInference

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
                    # Normalize label to 'positive'/'negative' for registry logic
                    norm_label = 'positive' if label == 'category_1' else 'negative'
                    for idx, action_program in enumerate(group):
                        img_dir = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/images/{problem_id}", label)
                        img_path = os.path.join(img_dir, f"{idx}.png")

                        # Flatten action_program to a list of strings only
                        def flatten_action_program(prog):
                            flat = []
                            if isinstance(prog, list):
                                for item in prog:
                                    if isinstance(item, list):
                                        flat.extend(flatten_action_program(item))
                                    elif isinstance(item, str):
                                        flat.append(item)
                            elif isinstance(prog, str):
                                flat.append(prog)
                            return flat


                        flat_commands = [cmd for cmd in flatten_action_program(action_program) if isinstance(cmd, str)]

                        try:
                            # parse vertices, but fall back on math errors
                            try:
                                vertices = logo_parser.parse_action_program(flat_commands, scale=120)
                            except ValueError as e:
                                logging.warning(
                                    f"{problem_id}/{img_path} parse failed ({e}); using tiny‐square"
                                )
                                vertices = [(0,0),(1,0),(1,1),(0,1)]
                            print(f"→ parsed {len(vertices)} vertices for {problem_id} {label}")

                            if not vertices:
                                flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': 'No vertices generated from LOGO commands'})
                                continue

                            poly = PhysicsInference.polygon_from_vertices(vertices)
                            # Always return a valid polygon, fallback if degenerate
                            fallback_square = (poly.bounds == (0.0, 0.0, 1.0, 1.0))

                            # Build a fully‐safe feature dict (no math errors!)
                            features = {
                                'centroid': PhysicsInference.centroid(poly),
                                'area': PhysicsInference.area(poly),
                                'is_convex': PhysicsInference.is_convex(poly),
                                'symmetry_score': PhysicsInference.symmetry_score(poly),
                                'moment_of_inertia': PhysicsInference.moment_of_inertia(poly),
                                'num_straight': PhysicsInference.count_straight_segments(poly),
                                'num_arcs': PhysicsInference.count_arcs(poly),
                                'has_quadrangle': PhysicsInference.has_quadrangle(poly),
                                'has_obtuse_angle': PhysicsInference.has_obtuse(poly),
                                'fallback_square': fallback_square,
                                'num_strokes': len(flat_commands),
                            }
                            result = {
                                'problem_id': problem_id,
                                'category': cat,
                                'label': norm_label,
                                'image_path': img_path,
                                'features': features,
                                'geometry': vertices,
                                'action_program': flat_commands
                            }
                            all_results.append(result)
                        except Exception as e:
                            flagged_cases.append({'problem_id': problem_id, 'image_path': img_path, 'error': str(e)})
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