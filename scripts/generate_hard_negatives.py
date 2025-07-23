import random
from src.data_pipeline.logo_parser import LogoParser

def perturb_logo_commands(commands, angle_delta=15, length_jitter=0.05):
    new_commands = []
    for cmd, val in commands:
        if cmd in 'FB':
            val = val * (1 + random.uniform(-length_jitter, length_jitter))
        elif cmd in 'RL':
            val = val + random.choice([-angle_delta, angle_delta])
        new_commands.append((cmd, val))
    return new_commands

def save_logo(commands, path):
    with open(path, 'w') as f:
        for cmd, val in commands:
            f.write(f"{cmd} {val}\n")



import argparse
from BongordSolver.src.data_pipeline.loader import BongardLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--problems-list', default=None)
    parser.add_argument('--n-select', type=int, default=50)
    args = parser.parse_args()

    loader = BongardLoader(args.input_dir, problems_list=args.problems_list, n_select=args.n_select)
    logo_parser = LogoParser()
    # Example: For each problem, perturb logos and save as hard negatives
    for problem in loader.iter_problems():
        pid = problem['problem_id']
        for sample in problem['positives']:
            logo_path = sample['image_path'].replace('.png', '.logo')
            commands = logo_parser.parse_logo_script(logo_path)
            perturbed = perturb_logo_commands(commands)
            out_logo_path = f"{args.output}/{pid}_hardneg.logo"
            save_logo(perturbed, out_logo_path)

if __name__ == "__main__":
    main()
