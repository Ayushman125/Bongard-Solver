import json
import os

def find_arc_strokes_in_action_programs(action_programs_dir, problems_list_path, output_path):
    arc_examples = []
    with open(problems_list_path, 'r', encoding='utf-8') as f:
        problems = [line.strip() for line in f if line.strip()]
    for problem_id in problems:
        json_path = os.path.join(action_programs_dir, f"{problem_id}_action_programs.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for ex_type, ex_list in zip(['pos', 'neg'], data.get(problem_id, [])):
            for idx, strokes in enumerate(ex_list):
                # Unwrap extra nesting
                if isinstance(strokes, list) and len(strokes) > 0 and isinstance(strokes[0], list):
                    strokes = strokes[0]
                if not isinstance(strokes, list):
                    continue
                for stroke in strokes:
                    if isinstance(stroke, str) and stroke.startswith('arc_'):
                        arc_examples.append({
                            'problem_id': problem_id,
                            'example_type': ex_type,
                            'example_idx': idx,
                            'stroke': stroke
                        })
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in arc_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Found {len(arc_examples)} arc strokes. Saved to {output_path}")

if __name__ == "__main__":
    # Update these paths as needed
    action_programs_dir = "data/raw/ShapeBongard_V2/bd"
    problems_list_path = "data/phase1_50puzzles.txt"
    output_path = "arc_stroke_examples.txt"
    find_arc_strokes_in_action_programs(action_programs_dir, problems_list_path, output_path)
