"""
Module: mix_real_and_synthetic.py
Purpose: Mix synthetic and real data for all shape categories, producing a unified dataset for robust training and evaluation.
"""
import os
import numpy as np
import json
from typing import List, Dict
from .synthetic_shapes import generate_dataset

def load_real_labels(real_json_path: str) -> List[Dict]:
    """Load real labeled data from a JSON file (as produced by logo_to_shape.py)."""
    with open(real_json_path, 'r') as f:
        data = json.load(f)
    return data

def mix_datasets(synthetic: List[Dict], real: List[Dict], ratio: float = 0.5) -> List[Dict]:
    """Mix synthetic and real data at the given ratio (synthetic:real)."""
    n_synth = int(len(synthetic) * ratio)
    n_real = int(len(real) * (1 - ratio))
    mixed = synthetic[:n_synth] + real[:n_real]
    np.random.shuffle(mixed)
    return mixed

def save_mixed_dataset(mixed: List[Dict], out_path: str):
    with open(out_path, 'w') as f:
        json.dump(mixed, f, indent=2)

# Example usage:
if __name__ == "__main__":
    # Generate synthetic data for all shape types
    from synthetic_shapes import generate_dataset
    shape_types = ['ellipse', 'circle', 'rectangle', 'polygon', 'triangle', 'point_cloud']
    synthetic = generate_dataset(shape_types, n_per_type=100, noise=0.03, out_dir=None)
    # Load real data (output from logo_to_shape.py)
    real = load_real_labels('real_labels.json')
    # Mix datasets (50% synthetic, 50% real)
    mixed = mix_datasets(synthetic, real, ratio=0.5)
    save_mixed_dataset(mixed, 'mixed_dataset.json')
