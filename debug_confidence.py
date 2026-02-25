"""Debug S1 confidence and S2 usage to understand the 8% usage rate."""

import sys
sys.path.insert(0, ".")

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver

loader = RawShapeBongardLoader("data/raw/ShapeBongard_V2")

# Test first 10 episodes
for i in range(10):
    task_id = f"ff_nact4_5_{i:04d}"
    episode = loader.build_episode(task_id, query_index=6)
    
    solver = HybridDualSystemSolver(
        image_size=64,
        use_programs=True,
        system1_confidence_threshold=0.35,  # Default from run_experiment.py
        seed=42+i,
    )
    
    metrics = solver.evaluate_episode(episode)
    
    # Extract query data
    pos_p1 = metrics['query_pos_p1']
    pos_p2 = metrics['query_pos_p2']
    pos_combined = metrics['query_pos_combined']
    used_s2_pos = metrics['used_system2_pos']
    
    # Get confidence from predict_item
    solver.fit_episode(episode)
    pos_result = solver.predict_item(episode.query_pos)
    s1_conf = pos_result.system1_confidence
    
    print(f"Task {i}: {task_id}")
    print(f"  S1 confidence: {s1_conf:.4f}")
    print(f"  Threshold: {solver.system1_confidence_threshold}")
    print(f"  Used S2: {bool(used_s2_pos)}")
    print(f"  S1 pred: {pos_p1:.4f}, S2 pred: {pos_p2:.4f}, Combined: {pos_combined:.4f}")
    print(f"  Weights: w1={solver.system1_weight:.4f}, w2={solver.system2_weight:.4f}")
    print()
