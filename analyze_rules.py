"""Check what rules System 2 is finding from program features."""

import sys
sys.path.insert(0, ".")

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver

# Load episodes
loader = RawShapeBongardLoader("data/raw/ShapeBongard_V2")

# Test 5 episodes
print("="*80)
print("RULE DISCOVERY ANALYSIS")
print("="*80)

for i in range(5):
    task_id = f"ff_nact4_5_{i:04d}"
    episode = loader.build_episode(task_id, query_index=6)
    
    # Solver with programs
    solver = HybridDualSystemSolver(
        image_size=64,
        use_programs=True,
        seed=42,
    )
    
    # Fit and check rules
    solver.fit_episode(episode)
    
    # Get System 2 rules
    rules = solver.system2._rules
    
    print(f"\nTask: {task_id}")
    print(f"  Rules found: {len(rules)}")
    if rules:
        for j, rule in enumerate(rules[:5]):  # Show top 5
            print(f"    [{j+1}] {rule.name}, weight={rule.weight:.4f}, kind={rule.kind}")
    else:
        print("    (no compositional rules, using Gaussian fallback)")
    print()
