import os
import random

def get_problem_folders(base_dir, n_select):
    # Enumerate all folders in the category
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if len(folders) < n_select:
        return folders
    return random.sample(folders, n_select)

def main():
    # Paths for each category
    ff_dir = r"c:\Users\HP\AI_Projects\BongordSolver\data\raw\ShapeBongard_V2\ff\images"
    bd_dir = r"c:\Users\HP\AI_Projects\BongordSolver\data\raw\ShapeBongard_V2\bd\images"
    hd_dir = r"c:\Users\HP\AI_Projects\BongordSolver\data\raw\ShapeBongard_V2\hd\images"

    # Select required number per category (from project report)
    ff_selected = get_problem_folders(ff_dir, 17)
    bd_selected = get_problem_folders(bd_dir, 17)
    hd_selected = get_problem_folders(hd_dir, 16)

    all_selected = ff_selected + bd_selected + hd_selected

    # Write to file
    out_path = r"c:\Users\HP\AI_Projects\BongordSolver\data\phase1_50puzzles.txt"
    with open(out_path, 'w') as f:
        for folder in all_selected:
            f.write(folder + '\n')
    print(f"Wrote {len(all_selected)} problem IDs to {out_path}")

if __name__ == "__main__":
    main()
