import numpy as np
import time
import json
import os

from src.system1_al import System1AbstractionLayer

import numpy as np
import time
import json
import os
import random
from PIL import Image

from src.system1_al import System1AbstractionLayer

def find_bongard_problems(root_dir):
    """Scans the directory for valid Bongard problem folders."""
    problem_paths = []
    if not os.path.isdir(root_dir):
        return []
        
    for level1 in os.listdir(root_dir):
        path1 = os.path.join(root_dir, level1)
        if os.path.isdir(path1):
            images_path = os.path.join(path1, 'images')
            if os.path.isdir(images_path):
                for problem_name in os.listdir(images_path):
                    problem_path = os.path.join(images_path, problem_name)
                    # A valid problem has '0' and '1' subdirectories
                    if os.path.isdir(problem_path) and \
                       os.path.isdir(os.path.join(problem_path, '0')) and \
                       os.path.isdir(os.path.join(problem_path, '1')):
                        problem_paths.append(problem_path)
    return problem_paths

def load_bongard_problem(problem_path):
    """
    Loads a real Bongard problem from the specified path.
    It loads PNG images, converts them to binary numpy arrays,
    and returns the images and their ground truth labels.
    """
    print(f"INFO: Loading Bongard problem from: {problem_path}")
    left_set = []
    right_set = []
    
    right_path = os.path.join(problem_path, '0')
    left_path = os.path.join(problem_path, '1')

    def load_images_from_dir(directory):
        images = []
        if not os.path.isdir(directory):
            return images
        for fname in sorted(os.listdir(directory)):
            if fname.endswith('.png'):
                try:
                    img_path = os.path.join(directory, fname)
                    # Open image, convert to grayscale, then to binary numpy array
                    with Image.open(img_path).convert('L') as img:
                        img_array = np.array(img)
                        # Binarize the image: non-black pixels become 1, black pixels remain 0
                        binary_array = (img_array > 10).astype(np.uint8)
                        images.append(binary_array)
                except Exception as e:
                    print(f"WARNING: Could not load or process image {fname}: {e}")
        return images

    left_set = load_images_from_dir(left_path)
    right_set = load_images_from_dir(right_path)
    
    if not left_set or not right_set:
        raise FileNotFoundError(f"Could not load a complete problem set from {problem_path}")

    # The full problem consists of all images
    all_images = left_set + right_set
    # Ground truth labels: 1 for left set, 0 for right set
    true_labels = [1] * len(left_set) + [0] * len(right_set)
    
    return all_images, true_labels

def simulate_downstream_solver(s1_output):
    """
    Simulates the behavior of the downstream reasoning pipeline (System-2).
    
    This function takes the S1 output and "solves" the puzzle, returning
    the ground truth labels. In a real system, this would be a complex
    process of symbolic grounding, search, and validation.
    """
    print("INFO: Simulating downstream solver...")
    # Here, we just pretend to do work and return the known correct labels.
    # The final rule is implicitly "squares vs. circles".
    
    # We can also simulate the "surprise" calculation.
    # Let's assume the top S1 heuristic was "strong_vertical_symmetry" with confidence 0.85
    # This heuristic is true for both squares and circles, so it doesn't distinguish them well.
    
    top_heuristic_confidence = 0.0
    if s1_output['heuristics']:
        top_heuristic_confidence = s1_output['heuristics'][0]['confidence']
        print(f"INFO: Top S1 heuristic was '{s1_output['heuristics'][0]['rule']}' with confidence {top_heuristic_confidence:.2f}")

    # This is a mock object for what a final rule might look like
    final_rule = {
        "name": "shape_is_square",
        "accuracy": 1.0
    }
    
    print("INFO: Downstream solver confirmed final rule.")
    return final_rule

def main():
    """
    Main professional pipeline for running the Bongard solver.
    This script orchestrates the S1-AL and simulates the rest of the pipeline.
    """
    print("--- Professional Bongard-Solver Pipeline ---")
    
    # --- 1. Initialization ---
    print("INFO: Initializing System-1 Abstraction Layer...")
    s1_al = System1AbstractionLayer(
        fuzzy_model_path="data/fuzzy_tree.pkl",
        replay_path="data/system1_replay.pkl"
    )
    
    # --- 2. Discover and Load a Real Problem ---
    problem_root = 'ShapeBongard_V2'
    problem_paths = find_bongard_problems(problem_root)
    
    if not problem_paths:
        print(f"ERROR: No Bongard problems found in '{problem_root}'. Please check the directory structure.")
        return

    # Select a random problem to process
    problem_path = random.choice(problem_paths)
    problem_id = os.path.basename(problem_path)
    
    try:
        images, true_labels = load_bongard_problem(problem_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # --- 3. S1-AL Processing ---
    print(f"INFO: Processing problem '{problem_id}' with S1-AL...")
    s1_output = s1_al.process(images, problem_id=problem_id)
    
    print("\n--- S1-AL Output Bundle ---")
    print(f"  Problem ID: {s1_output['problem_id']}")
    print(f"  Duration: {s1_output['duration_ms']:.2f} ms")
    print(f"  Top Heuristic: {s1_output['heuristics'][0] if s1_output['heuristics'] else 'None'}")
    print("---------------------------\n")

    # The s1_output would now be passed to downstream modules.
    # context["s1_output"] = s1_output
    
    # --- 4. Downstream Solving (Simulated) ---
    final_rule = simulate_downstream_solver(s1_output)
    
    # --- 5. Self-Supervision Step ---
    print("INFO: Starting self-supervision step...")
    s1_al.self_supervise(s1_output, true_labels)
    print(f"INFO: Replay buffer size is now: {s1_al.replay_buffer.size()}")
    
    # --- 6. Periodic Model Update ---
    if s1_al.replay_buffer.size() > 0:
        print("\nINFO: Triggering periodic model update...")
        s1_al.periodic_update(batch_size=1) # Lower batch size for demo
    
    print("\n--- Pipeline Run Complete ---")

if __name__ == '__main__':
    # Ensure data directory and initial model exist for the demo
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('data/fuzzy_tree.pkl'):
        from src.utils.fuzzy_tree import train_and_save_initial_tree
        train_and_save_initial_tree('data/fuzzy_tree.pkl')
        
    main()
