import time
import numpy as np
from src.utils.system1_abstraction import extract_all

def test_smoke_phase0():
    # assume data/phase0_smoke/*.npy masks exist
    import glob
    masks = [np.load(fp) for fp in glob.glob("data/phase0_smoke/*.npy")][:20]
    start = time.time()
    for m in masks:
        evt = extract_all(m)
        assert evt["latency_ms"] <= 100
    assert (time.time() - start) <= 300
