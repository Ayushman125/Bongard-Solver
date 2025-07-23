import numpy as np
from data.hard_negative_miner import HardNegativeMiner

def test_mine_returns_list_of_HardNegative():
    miner = HardNegativeMiner(difficulty_threshold=0.0)
    dummy_img = np.zeros((64,64,3), dtype=np.uint8)
    dummy_info = {'mask': np.ones((64,64)), 'com': [32,32]}
    negs = miner.mine([dummy_img]*5, [dummy_info]*5, n_samples=5)
    assert len(negs) == 5
    for neg in negs:
        assert hasattr(neg, 'image')
        assert 0.0 <= neg.difficulty_score <= 1.0
