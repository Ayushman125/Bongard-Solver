import argparse
from src.data_pipeline.loader import BongardLoader
from src.data_pipeline.logo_parser import LogoParser
from src.data_pipeline.physics_infer import PhysicsInference
from src.data_pipeline.attributes import Attributes
from src.data_pipeline.verification import Verification
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--problems-list', default=None)
    args = parser.parse_args()

    loader = BongardLoader(args.input_dir)
    logo_parser = LogoParser()
    data = []

    allowed_ids = set()
    if args.problems_list:
        with open(args.problems_list) as f:
            allowed_ids = set(line.strip() for line in f.readlines())

    for problem in loader.iter_problems():
        pid = problem['problem_id']
        if allowed_ids and pid not in allowed_ids:
            continue
        for sample in (problem['positives'] + problem['negatives']):
            logo_path = sample['image_path'].replace('.png', '.logo')
            vertices = logo_parser.parse_logo_script(logo_path)
            poly = PhysicsInference.polygon_from_vertices(vertices)
            if not Verification.validate_polygon(poly):
                Verification.log_for_review(pid, logo_path, 'Invalid polygon')
                continue
            features = {
                'centroid': PhysicsInference.centroid(poly),
                'area': PhysicsInference.area(poly),
                'is_convex': PhysicsInference.is_convex(poly),
                'symmetry_score': PhysicsInference.symmetry_score(vertices),
                'moment_of_inertia': PhysicsInference.moment_of_inertia(vertices)
            }
            enriched = Attributes.enrich_problem_dict(sample.copy(), sample['image_path'])
            enriched.update(features)
            data.append(enriched)
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
