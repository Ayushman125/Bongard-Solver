"""
Smoke test: run 20 puzzles through S1-AL pipeline under time budgets.
Version: 0.1.0
"""

import time, glob
import numpy as np
from src.system1_al import extract_com, extract_inertia_tensor

def load_masks(folder="data/phase0_smoke"):
    # Assuming masks saved as .npy files
    return [np.load(fp) for fp in glob.glob(f"{folder}/*.npy")][:20]

def main():
    masks = load_masks()
    start = time.time()
    for mask in masks:
        t0 = time.time()
        com = extract_com(mask)
        inertia = extract_inertia_tensor(mask)
        dt = (time.time() - t0) * 1000
        assert dt <= 100, f"Exceeded 100 ms: {dt:.2f} ms"
    total = time.time() - start
    assert total <= 300, f"Total budget exceeded: {total:.2f} s"
    print("Phase 0 smoke test passed.")

if __name__ == "__main__":
    main()
