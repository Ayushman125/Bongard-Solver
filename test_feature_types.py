"""Debug script to check if program features are being used."""

import json
import sys
sys.path.insert(0, ".")

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver

# Load test episode
loader = RawShapeBongardLoader("data/raw/ShapeBongard_V2")
episode = loader.build_episode("ff_nact4_5_0000", query_index=6)

# Test with program features ENABLED
print("="*60)
print("TEST 1: Program features ENABLED")
print("="*60)
solver = HybridDualSystemSolver(
    image_size=64,
    use_programs=True,
    seed=0,
)
result_with_programs = solver.solve_episode(episode)
print(f"POS query feature_type: {result_with_programs.pos_trace.get('feature_type', 'NOT FOUND')}")
print(f"NEG query feature_type: {result_with_programs.neg_trace.get('feature_type', 'NOT FOUND')}")
print(f"POS query rule_count: {result_with_programs.pos_trace.get('rule_count', 'NOT FOUND')}")
print(f"Top rule: {result_with_programs.pos_trace.get('top_rule', 'NOT FOUND')}")
print()

# Test with program features DISABLED
print("="*60)
print("TEST 2: Program features DISABLED")
print("="*60)
solver_no_prog = HybridDualSystemSolver(
    image_size=64,
    use_programs=False,
    seed=0,
)
result_without_programs = solver_no_prog.solve_episode(episode)
print(f"POS query feature_type: {result_without_programs.pos_trace.get('feature_type', 'NOT FOUND')}")
print(f"NEG query feature_type: {result_without_programs.neg_trace.get('feature_type', 'NOT FOUND')}")
print(f"POS query rule_count: {result_without_programs.pos_trace.get('rule_count', 'NOT FOUND')}")
print(f"Top rule: {result_without_programs.pos_trace.get('top_rule', 'NOT FOUND')}")
print()

# Compare
print("="*60)
print("COMPARISON")
print("="*60)
print(f"With programs accuracy: POS={result_with_programs.pos_result.combined_prob:.4f}, NEG={result_with_programs.neg_result.combined_prob:.4f}")
print(f"Without programs accuracy: POS={result_without_programs.pos_result.combined_prob:.4f}, NEG={result_without_programs.neg_result.combined_prob:.4f}")
