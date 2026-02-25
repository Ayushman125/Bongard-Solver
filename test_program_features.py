"""Quick test to verify program feature extraction."""

import json
import sys
sys.path.insert(0, ".")

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.program_utils import extract_program_features

# Load test data
loader = RawShapeBongardLoader("data/raw/ShapeBongard_V2")
episode = loader.build_episode("ff_nact4_5_0000", query_index=6)

# Check first positive support example
ex = episode.support_pos[0]
print(f"Task: {ex.task_id}")
print(f"Label: {ex.label}")
print(f"Program type: {type(ex.program)}")
print(f"Program: {ex.program[:3] if isinstance(ex.program, list) else ex.program}")
print()

# Extract features
features = extract_program_features(ex.program)
print(f"Features shape: {features.shape}")
print(f"Features: {features}")
print()

# Check if all zeros (would mean extraction failed)
import numpy as np
if np.allclose(features, 0):
    print("WARNING: All features are zero!")
else:
    print("✓ Features extracted successfully")
    print(f"  - Non-zero elements: {np.count_nonzero(features)}/{len(features)}")
    print(f"  - Feature range: [{features.min():.4f}, {features.max():.4f}]")
