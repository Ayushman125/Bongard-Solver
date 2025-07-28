import json
import numpy as np
from pathlib import Path

def compute_dynamic_thresholds(log_path="logs/augmentor_benchmark.jsonl", metric="quality_score", percentile=10):
    """
    Compute dynamic QA thresholds from analytics logs.
    Args:
        log_path: Path to analytics log file (jsonl).
        metric: Metric to compute threshold for (e.g., 'quality_score').
        percentile: Lower percentile to use as threshold (e.g., 10 for 10th percentile).
    Returns:
        threshold: Value at the given percentile.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    values = []
    with log_path.open() as f:
        for line in f:
            try:
                event = json.loads(line)
                if metric in event:
                    values.append(event[metric])
            except Exception:
                continue
    if not values:
        return None
    return float(np.percentile(values, percentile))
