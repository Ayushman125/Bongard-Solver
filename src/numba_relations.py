# Folder: bongard_solver/src/
# File: numba_relations.py
import numpy as np
from numba import njit, prange
import logging

logger = logging.getLogger(__name__)

# Note: The relation codes here (0-6) should ideally be consistent
# with the RELATION_MAP defined in config.py.
# For example, 0='contains' here should map to RELATION_MAP['contains'] in config.
# Ensure this mapping is handled when integrating the output of this function
# into scene graphs or other symbolic representations.

@njit(cache=True, parallel=True)
def compute_relations_typed(xs: np.ndarray, ys: np.ndarray, ws: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """
    Computes spatial relations between objects based on their bounding box coordinates.
    This function is Numba-optimized and operates on NumPy arrays.
    Args:
        xs (np.ndarray): Array of x-coordinates (top-left) of bounding boxes.
        ys (np.ndarray): Array of y-coordinates (top-left) of bounding boxes.
        ws (np.ndarray): Array of widths of bounding boxes.
        hs (np.ndarray): Array of heights of bounding boxes.
    Returns:
        np.ndarray: An array of relations, where each row is (relation_code, obj1_idx, obj2_idx).
                    Relation codes: 0='contains', 1='inside', 2='overlaps', 3='left_of',
                    4='right_of', 5='above', 6='below'.
    """
    n = xs.shape[0]
    
    if n < 2:
        return np.empty((0, 3), np.int32)
    
    # Pre-allocate for the maximum possible number of *primary* relations (one per pair)
    # If multiple relations per pair are needed, this needs to be a Numba List or a much larger pre-allocation.
    # Based on the current logic, it assigns one 'code' per pair.
    max_rel = n * (n - 1) // 2 
    rels_array = np.empty((max_rel, 3), np.int32)
    count = 0

    for i in prange(n):
        x1, y1, w1, h1 = xs[i], ys[i], ws[i], hs[i]
        x1b, y1b = x1 + w1, y1 + h1  # Bounding box end points

        for j in range(i + 1, n):
            x2, y2, w2, h2 = xs[j], ys[j], ws[j], hs[j]
            x2b, y2b = x2 + w2, y2 + h2  # Bounding box end points
            code = -1  # Default: no specific relation

            # Check for 'contains' or 'inside'
            if x1 <= x2 and y1 <= y2 and x1b >= x2b and y1b >= y2b:
                code = 0  # 0 for 'contains' (i contains j)
            elif x2 <= x1 and y2 <= y1 and x2b >= x1b and y2b >= y1b:
                code = 1  # 1 for 'inside' (j contains i)
            else:
                # Calculate intersection rectangle
                ox = max(0.0, min(x1b, x2b) - max(x1, x2))
                oy = max(0.0, min(y1b, y2b) - max(y1, y2))
                
                if ox * oy > 0:
                    code = 2  # 2 for 'overlaps'
                else:
                    # Positional relations if no overlap
                    if x1b < x2:
                        code = 3  # 3 for 'left_of' (i is left of j)
                    elif x2b < x1:
                        code = 4  # 4 for 'right_of' (i is right of j)
                    elif y1b < y2:
                        code = 5  # 5 for 'above' (i is above j)
                    elif y2b < y1:
                        code = 6  # 6 for 'below' (i is below j)
            
            # Only add if a valid code was assigned
            if code != -1:
                rels_array[count, 0] = code
                rels_array[count, 1] = i
                rels_array[count, 2] = j
                count += 1
    return rels_array[:count]

@njit(cache=True)
def compute_difficulty_typed(num_objs: int, avg_comp: float, num_rels: int, w0: float, w1: float, w2: float) -> float:
    """
    Calculates a difficulty score for an image based on object attributes and relations.
    This function is Numba-optimized and operates on primitive types.
    Args:
        num_objs (int): Number of objects detected.
        avg_comp (float): Average complexity of objects.
        num_rels (int): Number of relations detected.
        w0 (float): Weight for number of objects.
        w1 (float): Weight for average complexity.
        w2 (float): Weight for number of relations.
    Returns:
        float: The calculated difficulty score.
    """
    score = 0.0
    score += w0 * num_objs
    score += w1 * avg_comp
    score += w2 * num_rels
    return score
