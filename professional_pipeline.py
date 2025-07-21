import numpy as np
import time
import json
import os

from src.system1_al import System1AbstractionLayer

def load_dummy_problem():
    """
    Generates a dummy Bongard problem for demonstration.
    In a real pipeline, this would load images from a dataset.
    
    Problem: "Left set has squares, right set has circles."
    """
    print("INFO: Loading dummy Bongard problem...")
    left_set = []
    right_set = []

    # Create 6 "left" images (squares of varying sizes)
    for i in range(6):
        size = 50 + i * 5
        img = np.zeros((100, 100), dtype=np.uint8)
        start = (100 - size) // 2
        end = start + size
        img[start:end, start:end] = 1
        left_set.append(img)

    # Create 6 "right" images (circles of varying sizes)
    for i in range(6):
        radius = 25 + i * 3
        img = np.zeros((100, 100), dtype=np.uint8)
        cx, cy = 50, 50
        y, x = np.ogrid[-cy:100-cy, -cx:100-cx]
        mask = x*x + y*y <= radius*radius
        img[mask] = 1
        right_set.append(img)
        
    # The full problem consists of all 12 images
    all_images = left_set + right_set
    # Ground truth labels: 1 for left set, 0 for right set
    true_labels = [1] * 6 + [0] * 6
    
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
    # On worker startup, initialize the S1 Abstraction Layer.
    print("INFO: Initializing System-1 Abstraction Layer...")
    s1_al = System1AbstractionLayer(
        fuzzy_model_path="data/fuzzy_tree.pkl",
        replay_path="data/system1_replay.pkl"
    )
    
    # --- 2. Load Problem ---
    # This would typically be in a loop, processing many problems.
    problem_id = "dummy_problem_01"
    images, true_labels = load_dummy_problem()
    
    # --- 3. S1-AL Processing ---
    # Get the initial "intuitive" analysis from the S1-AL.
    print(f"INFO: Processing problem '{problem_id}' with S1-AL...")
    s1_output = s1_al.process(images, problem_id=problem_id)
    
    print("\n--- S1-AL Output Bundle ---")
    # print(json.dumps(s1_output, indent=2))
    print(f"  Problem ID: {s1_output['problem_id']}")
    print(f"  Duration: {s1_output['duration_ms']:.2f} ms")
    print(f"  Top Heuristic: {s1_output['heuristics'][0] if s1_output['heuristics'] else 'None'}")
    print("---------------------------\n")

    # The s1_output would now be passed to downstream modules.
    # context["s1_output"] = s1_output
    
    # --- 4. Downstream Solving (Simulated) ---
    # The rest of the pipeline works to find the final rule.
    final_rule = simulate_downstream_solver(s1_output)
    
    # --- 5. Self-Supervision Step ---
    # Once the final rule is known, feed the outcome back to the S1-AL.
    print("INFO: Starting self-supervision step...")
    s1_al.self_supervise(s1_output, true_labels)
    print(f"INFO: Replay buffer size is now: {s1_al.replay_buffer.size()}")
    
    # --- 6. Periodic Model Update ---
    # Periodically, the fuzzy model can be retrained on the replay buffer.
    # We'll force an update for demonstration if the buffer has samples.
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
