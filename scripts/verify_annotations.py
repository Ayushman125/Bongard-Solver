
import os
import random
import webbrowser

def enumerate_problem_folders(base_dirs, n_select=50):
    all_folders = []
    for base in base_dirs:
        if os.path.exists(base):
            folders = [os.path.join(base, f) for f in os.listdir(base) if os.path.isdir(os.path.join(base, f))]
            all_folders.extend(folders)
    if len(all_folders) < n_select:
        selected = all_folders
    else:
        selected = random.sample(all_folders, n_select)
    return selected

def store_selected_problems(selected_folders, out_path):
    with open(out_path, 'w') as f:
        for folder in selected_folders:
            f.write(folder + '\n')

def open_random_images_from_folders(folders, n_images=1):
    for folder in folders:
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            chosen = random.sample(images, min(n_images, len(images)))
            for img_path in chosen:
                webbrowser.open(img_path)

if __name__ == "__main__":
    # Example usage
    ff_dir = r"c:\Users\HP\AI_Projects\BongordSolver\data\raw\ShapeBongard_V2\ff\images"
    hd_dir = r"c:\Users\HP\AI_Projects\BongordSolver\data\raw\ShapeBongard_V2\hd\images"
    bd_dir = r"c:\Users\HP\AI_Projects\BongordSolver\data\raw\ShapeBongard_V2\bd\images"
    selected = enumerate_problem_folders([ff_dir, hd_dir, bd_dir], n_select=50)
    store_selected_problems(selected, "selected_problems.txt")
    open_random_images_from_folders(selected, n_images=1)
