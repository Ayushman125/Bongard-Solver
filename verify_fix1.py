"""Verify Fix 1 impact by comparing with/without confidence bypass."""

import sys
sys.path.insert(0, ".")

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver

loader = RawShapeBongardLoader("data/raw/ShapeBongard_V2")

# Test on first 20 episodes
task_ids = loader.splits["test_ff"][:20]

correct_count = 0
s2_usage_count = 0

for task_id in task_ids:
    episode = loader.build_episode(task_id, query_index=6)
    
    solver = HybridDualSystemSolver(
        image_size=64,
        use_programs=True,
        seed=42,
    )
    
    metrics = solver.evaluate_episode(episode)
    
    correct_count += int(metrics['concept_correct'])
    s2_usage_count += int(metrics['used_system2_pos'])
    
    print(f"{task_id}: correct={int(metrics['concept_correct'])}, "
          f"s2_used={int(metrics['used_system2_pos'])}, "
          f"w1={metrics['system1_weight_final']:.4f}, "
          f"w2={metrics['system2_weight_final']:.4f}, "
          f"p1={metrics['query_pos_p1']:.4f}, "
          f"p2={metrics['query_pos_p2']:.4f}, "
          f"combined={metrics['query_pos_combined']:.4f}")

accuracy = correct_count / len(task_ids)
s2_usage = s2_usage_count / len(task_ids)

print(f"\n{'='*60}")
print(f"SUMMARY (n={len(task_ids)} episodes):")
print(f"  Accuracy: {accuracy:.1%}")
print(f"  S2 Usage: {s2_usage:.1%}")
print(f"{'='*60}")
