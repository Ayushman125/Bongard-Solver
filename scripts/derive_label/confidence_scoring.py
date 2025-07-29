import numpy as np

def calculate_confidence(metric_value, max_value, min_value, is_higher_better=True):
    """
    Normalizes a metric value into a confidence score (0-1).
    max_value and min_value define the expected range for the metric.
    """
    if max_value == min_value:
        return 0.5 # Neutral if range is zero

    if is_higher_better:
        return np.clip((metric_value - min_value) / (max_value - min_value), 0.0, 1.0)
    else: # Lower value is better (e.g., error)
        return np.clip(1.0 - (metric_value - min_value) / (max_value - min_value), 0.0, 1.0)
