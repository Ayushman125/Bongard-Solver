import logging

def euclidean_embed(x, *args, **kwargs):
    """
    Embed a feature dict (or compatible structure) into Euclidean space as a flat vector.
    Accepts dicts or lists of dicts. Returns 1D numpy array.
    """
    # import logging removed; use global logging
    if isinstance(x, dict):
        return feature_dict_to_vector(x)
    elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], dict):
        # Batch: stack all vectors
        return np.stack([feature_dict_to_vector(d) for d in x])
    elif isinstance(x, str):
        logging.error(f"euclidean_embed: received string input: {x}")
        return np.array([])
    elif isinstance(x, tuple):
        logging.error(f"euclidean_embed: received tuple input: {x}")
        return np.array([])
    else:
        raise ValueError(f"euclidean_embed: input must be dict or list of dicts, got {type(x)}")

def hyperbolic_embed(x, *args, **kwargs):
    """
    Embed a feature dict (or compatible structure) into a hyperbolic-like space (Poincaré ball approx).
    Accepts dicts or lists of dicts. Returns 1D numpy array in (-1,1) via tanh.
    """
    import logging
    if isinstance(x, dict):
        vec = feature_dict_to_vector(x)
        # Map to Poincaré ball (approx): tanh for boundedness
        return np.tanh(vec)
    elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], dict):
        return np.stack([np.tanh(feature_dict_to_vector(d)) for d in x])
    elif isinstance(x, str):
        logging.error(f"hyperbolic_embed: received string input: {x}")
        return np.array([])
    elif isinstance(x, tuple):
        logging.error(f"hyperbolic_embed: received tuple input: {x}")
        return np.array([])
    else:
        raise ValueError(f"hyperbolic_embed: input must be dict or list of dicts, got {type(x)}")
import numpy as np
# Utility: Convert a feature dict to a numeric vector in a fixed order
def feature_dict_to_vector(d, key_order=None):
    """
    Convert a feature dict (with keys like 'centroid', 'area', etc.) to a flat numeric vector.
    If key_order is provided, use that order; otherwise, use sorted keys.
    Handles nested lists/tuples for e.g. 'centroid'.
    Returns a 1D numpy array.
    """
    import numpy as np
    if not isinstance(d, dict):
        raise ValueError(f"feature_dict_to_vector: input is not a dict: {d}")
    if key_order is None:
        key_order = sorted(d.keys())
    vec = []
    for k in key_order:
        v = d[k]
        # Flatten any nested lists/tuples and filter out non-numeric values
        if isinstance(v, (int, float, np.integer, np.floating)):
            vec.append(float(v))
        elif isinstance(v, (list, tuple, np.ndarray)):
            for x in v:
                if isinstance(x, (int, float, np.integer, np.floating)):
                    vec.append(float(x))
    arr = np.array(vec, dtype=float)
    return np.tanh(arr)

def mine_hard(embeddings, topk=5):
    # Handle empty or 1D embeddings gracefully
    arr = np.array(embeddings)
    # If empty or 0-d, return immediately to avoid axis errors
    if arr.size == 0 or arr.ndim == 0:
        return arr, 'none'
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # If after reshape, still empty, return
    if arr.shape[0] == 0:
        return arr, 'none'
    dists = np.linalg.norm(arr - arr.mean(axis=0), axis=1)
    idx = np.argsort(-dists)[:topk]
    return arr[idx], 'mine_hard'

def poincare_mixup(u, v, alpha=0.5):
    # Convert dicts to flat numeric arrays if needed
    import logging
    def flatten_numeric(x):
        flat = []
        if isinstance(x, dict):
            for v in x.values():
                flat.extend(flatten_numeric(v))
        elif isinstance(x, (list, tuple)):
            for v in x:
                flat.extend(flatten_numeric(v))
        elif isinstance(x, np.ndarray):
            # If 0-d array, treat as scalar
            if x.ndim == 0:
                if isinstance(x.item(), (int, float, np.integer, np.floating)):
                    flat.append(float(x.item()))
            else:
                for v in x:
                    flat.extend(flatten_numeric(v))
        elif isinstance(x, (int, float, np.integer, np.floating)):
            flat.append(float(x))
        # else: skip non-numeric, non-iterable
        return flat

    # Defensive: if input is string or tuple, log and return (None, 'invalid_input')
    if isinstance(u, str) or isinstance(v, str):
        logging.error(f"poincare_mixup: received string input! u={u}, v={v}")
        return (None, 'invalid_input')
    if isinstance(u, tuple) and not hasattr(u, 'shape'):
        logging.error(f"poincare_mixup: received tuple input! u={u}")
        return (None, 'invalid_input')
    if isinstance(v, tuple) and not hasattr(v, 'shape'):
        logging.error(f"poincare_mixup: received tuple input! v={v}")
        return (None, 'invalid_input')

    # Defensive: if either input is None, return (None, 'invalid_input')
    if u is None or v is None:
        logging.error(f"poincare_mixup: received None input! u={u}, v={v}")
        return (None, 'invalid_input')
    import logging
    def flatten_any(x):
        flat = []
        if isinstance(x, dict):
            for v in x.values():
                flat.extend(flatten_any(v))
        elif isinstance(x, np.ndarray):
            if x.ndim == 0:
                flat.append(float(x.item()))
            else:
                for v in x:
                    flat.extend(flatten_any(v))
        elif isinstance(x, (list, tuple)):
            for v in x:
                flat.extend(flatten_any(v))
        elif isinstance(x, (int, float, np.integer, np.floating)):
            flat.append(float(x))
        return flat

    u_arr = np.array(flatten_any(u), dtype=float)
    v_arr = np.array(flatten_any(v), dtype=float)
    # Input validation: if either input is empty, log and return (None, reason)
    if u_arr.size == 0 and v_arr.size == 0:
        logging.error(f"poincare_mixup: both inputs empty after conversion! u={u}, v={v}")
        result = (None, 'empty')
        assert isinstance(result, tuple) and len(result) == 2
        return result
    if u_arr.size == 0:
        logging.error(f"poincare_mixup: u is empty after conversion! u={u}, v={v}")
        result = (None, 'u_empty')
        assert isinstance(result, tuple) and len(result) == 2
        return result
    if v_arr.size == 0:
        logging.error(f"poincare_mixup: v is empty after conversion! u={u}, v={v}")
        result = (None, 'v_empty')
        assert isinstance(result, tuple) and len(result) == 2
        return result
    # If shapes do not match, pad the smaller with zeros
    if u_arr.shape != v_arr.shape:
        max_len = max(u_arr.size, v_arr.size)
        u_arr = np.resize(u_arr, max_len)
        v_arr = np.resize(v_arr, max_len)
    result = ((u_arr + v_arr) / 2 * alpha, 'mixed')
    assert isinstance(result, tuple) and len(result) == 2
    logging.info(f"poincare_mixup: returning {result} of type {type(result)}")
    return result
