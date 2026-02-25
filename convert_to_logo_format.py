import os
import json

# All supported stroke/action formats

def load_action_programs(base_dir, categories=('bd', 'ff', 'hd')):
    action_programs = {}
    for cat in categories:
        json_path = os.path.join(base_dir, cat, f"{cat}_action_programs.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    action_programs.update(data)
    return action_programs

def load_problem_ids(problem_list_path):
    with open(problem_list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def save_shape_actions(shape_actions, out_path):
    with open(out_path, 'w') as f:
        for action in shape_actions:
            f.write(f"{action}\n")

def convert_action_programs(input_dir, problem_list_path, output_dir):
    action_programs = load_action_programs(input_dir)
    problem_ids = set(load_problem_ids(problem_list_path))

    for prob_id in problem_ids:
        if prob_id not in action_programs:
            print(f"Warning: {prob_id} not found in action programs.")
            continue
        pos_examples, neg_examples = action_programs[prob_id]
        prob_dir = os.path.join(output_dir, prob_id)
        pos_dir = os.path.join(prob_dir, "pos")
        neg_dir = os.path.join(prob_dir, "neg")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)

        for i, example in enumerate(pos_examples):
            for j, shape in enumerate(example):
                out_path = os.path.join(pos_dir, f"{i}_{j}.txt")
                save_shape_actions(shape, out_path)
        for i, example in enumerate(neg_examples):
            for j, shape in enumerate(example):
                out_path = os.path.join(neg_dir, f"{i}_{j}.txt")
                save_shape_actions(shape, out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Bongard action programs to LOGO format for selected puzzles.")
    parser.add_argument('--input-dir', type=str, required=True, help='Base directory for action programs')
    parser.add_argument('--problems-list', type=str, required=True, help='Path to phase1_50puzzles.txt')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for formatted programs')
    args = parser.parse_args()
    convert_action_programs(args.input_dir, args.problems_list, args.outdir)
