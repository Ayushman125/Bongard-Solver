from collections import deque
import cv2
import numpy as np
from skimage.morphology import skeletonize

def prune_spurs(skel_bool, max_spur_length=1):
    """
    Remove any tip that is <= max_spur_length away (in graph steps)
    from a branch point.
    """
    adj_kernel = np.ones((3,3), dtype=int)
    adj_kernel[1,1] = 0

    neigh_count8 = cv2.filter2D(skel_bool.astype(np.uint8), -1, adj_kernel)
    endpoints = set(map(tuple, np.argwhere((skel_bool) & (neigh_count8==1))))
    branches  = set(map(tuple, np.argwhere((skel_bool) & (neigh_count8>=3))))

    to_remove = set()
    for ep in endpoints:
        visited = {ep}
        queue = deque([(ep, 0)])
        while queue:
            (r,c), dist = queue.popleft()
            if (r,c) in branches:
                break
            if dist >= max_spur_length:
                break
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    nb = (r+dr, c+dc)
                    if nb not in visited and 0<=nb[0]<skel_bool.shape[0] and 0<=nb[1]<skel_bool.shape[1]:
                        if skel_bool[nb]:
                            visited.add(nb)
                            queue.append((nb, dist+1))
        else:
            to_remove |= visited
    skel_bool_pruned = skel_bool.copy()
    for pix in to_remove:
        skel_bool_pruned[pix] = False
    return skel_bool_pruned

def _simple_prune_to_target(skel_bool, target_endpoints=3):
    """Simply remove endpoints until we reach target count"""
    pruned = skel_bool.copy()
    nb_kernel = np.ones((3, 3), dtype=np.uint8)
    nb_kernel[1, 1] = 0
    
    for _ in range(10):  # max iterations
        neigh_count = cv2.filter2D(pruned.astype(np.uint8), -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.logical_and(pruned, neigh_count == 1)
        endpoint_count = endpoints.sum()
        
        if endpoint_count <= target_endpoints:
            break
            
        # Remove one endpoint
        endpoint_positions = np.argwhere(endpoints)
        if len(endpoint_positions) > 0:
            y, x = endpoint_positions[0]
            pruned[y, x] = 0
    
    return pruned

def _prune_short_spurs(skel_bool, max_spur_length=2, max_iter=10):
    pruned = skel_bool.copy()
    nb_kernel = np.ones((3, 3), dtype=np.uint8)
    nb_kernel[1, 1] = 0
    for _ in range(max_iter):
        changed = False
        neigh_count = cv2.filter2D(pruned.astype(np.uint8), -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.argwhere(np.logical_and(pruned, neigh_count == 1))
        for y, x in endpoints:
            path = [(y, x)]
            visited = set(path)
            for step in range(max_spur_length):
                # Find neighbors
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < pruned.shape[0] and 0 <= nx < pruned.shape[1]:
                            if pruned[ny, nx] and (ny, nx) not in visited:
                                neighbors.append((ny, nx))
                if not neighbors:
                    # Dead end, prune
                    pruned[y, x] = 0
                    changed = True
                    break
                elif len(neighbors) > 1:
                    # Junction, keep endpoint
                    break
                else:
                    y, x = neighbors[0]
                    visited.add((y, x))
                    if neigh_count[y, x] >= 3:
                        # Reached a junction, keep endpoint
                        break
        if not changed:
            break
    return pruned

def _prune_spurs_until(skel_bool, target_endpoints=3, max_iter=10):
    pruned = skel_bool.copy()
    nb_kernel = np.ones((3, 3), dtype=np.uint8)
    nb_kernel[1, 1] = 0
    for _ in range(max_iter):
        neigh_count = cv2.filter2D(pruned.astype(np.uint8), -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.logical_and(pruned, neigh_count == 1)
        if endpoints.sum() <= target_endpoints:
            break
        pruned[endpoints] = 0
    return pruned

def _prune_spurs(skel_bool, iterations=5):
    # Remove endpoints iteratively to prune short spurs
    pruned = skel_bool.copy()
    nb_kernel = np.ones((3, 3), dtype=np.uint8)
    nb_kernel[1, 1] = 0
    for _ in range(iterations):
        neigh_count = cv2.filter2D(pruned.astype(np.uint8), -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.logical_and(pruned, neigh_count == 1)
        pruned[endpoints] = 0
    return pruned

# Assuming preprocess_mask is defined elsewhere or will be provided.
# For demonstration, let's include a dummy preprocess_mask function.
def preprocess_mask(mask):
    """
    Dummy preprocess_mask function for demonstration.
    Replace with actual implementation if available.
    """
    # Example: Simple binary thresholding
    _, processed_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return processed_mask


def compute_skeleton(mask):
    """
    Computes the skeleton of a binary image and extracts key metrics.

    This function preprocesses the mask, computes its skeleton, and then calculates
    metrics like the number of strokes (pixels), endpoints, and branch points.
    It includes a heuristic to prevent 1-pixel-thick lines from being erased
    by the skeletonization algorithm.

    Args:
        mask (np.ndarray): 2D uint8 binary image (0 or 255).

    Returns:
        tuple: A tuple containing:
            - skel (np.ndarray): The skeletonized image (0 or 255).
            - metrics (dict): A dictionary with skeleton metrics:
                - 'stroke_count': Total number of pixels in the skeleton.
                - 'endpoint_count': Number of points with exactly one neighbor.
                - 'branch_point_count': Number of points with three or more neighbors.
                - 'skeleton_length': Same as stroke_count.
    """
    clean = preprocess_mask(mask)
    binary = clean > 0
    skel_bool = skeletonize(binary)

    # Prune diagonal-only spur pixels
    import numpy as np
    from scipy.ndimage import convolve
    four_k = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=int)
    eight_k = np.ones((3,3), dtype=int)
    eight_k[1,1] = 0
    sk = skel_bool.astype(int)
    nb4  = convolve(sk, four_k,  mode='constant', cval=0)
    nb8  = convolve(sk, eight_k, mode='constant', cval=0)
    # Only prune spurs if there is a branch (junction) in the skeleton
    if np.any((skel_bool) & (nb4 >= 3)):
        endpoints = np.argwhere((skel_bool) & (nb4 == 1))
        branches  = set(map(tuple, np.argwhere((skel_bool) & (nb4 >= 3))))
        to_prune = []
        for ep in endpoints:
            visited = {tuple(ep)}
            queue = deque([(tuple(ep), 0)])
            found_branch = False
            while queue:
                (r, c), dist = queue.popleft()
                if (r, c) in branches:
                    found_branch = True
                    break
                if dist >= 2:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < skel_bool.shape[0] and 0 <= nc < skel_bool.shape[1]:
                            if skel_bool[nr, nc] and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append(((nr, nc), dist + 1))
            if not found_branch:
                to_prune.append(tuple(ep))
        for pix in to_prune:
            skel_bool[pix] = False

    # 4-connectivity for metrics
    stroke_count     = int(skel_bool.sum())
    endpoint_count   = int(np.logical_and(skel_bool, nb4 == 1).sum())
    branch_point_count = int(np.logical_and(skel_bool, nb4 >= 3).sum())
    skeleton_length  = stroke_count

    skel = (skel_bool.astype(np.uint8) * 255)
    metrics = {
      "stroke_count": stroke_count,
      "endpoint_count": endpoint_count,
      "branch_point_count": branch_point_count,
      "skeleton_length": skeleton_length
    }
    return skel, metrics

# The following block was part of the original input but seemed misplaced.
# I've included it here, assuming it was intended to be part of a larger process
# or a separate function. It's not directly called by compute_skeleton.
def process_and_analyze_skeleton(mask):
    clean = preprocess_mask(mask)
    skel_bool = skeletonize(clean > 0)
    skel_bool = prune_spurs(skel_bool, max_spur_length=1)

    nb_kernel_4 = np.array([
        [0,1,0],
        [1,0,1],
        [0,1,0]
    ], dtype=np.uint8)

    neigh4 = cv2.filter2D(skel_bool.astype(np.uint8), -1, nb_kernel_4)
    stroke_count     = int(skel_bool.sum())
    endpoint_count   = int(np.logical_and(skel_bool, neigh4 == 1).sum())
    branch_point_count = int(np.logical_and(skel_bool, neigh4 >= 3).sum())
    skeleton_length  = stroke_count

    skel_img = (skel_bool.astype(np.uint8) * 255)
    return skel_img, {
        "stroke_count": stroke_count,
        "endpoint_count": endpoint_count,
        "branch_point_count": branch_point_count,
        "skeleton_length": skeleton_length
    }

# This function was also part of the original input, seemingly incomplete or misplaced.
# It attempts to prune artifacts but its logic for identifying "prunable" endpoints
# based on branch points seems a bit off. It's included as-is for now.
def _prune_artifacts(skel):
    pruned_skel = skel.copy()
    nb_kernel = np.ones((3, 3), dtype=np.uint8)
    nb_kernel[1, 1] = 0
    
    while True:
        # Find endpoints (pixels with 1 neighbor)
        neigh_count = cv2.filter2D(pruned_skel, -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.logical_and(pruned_skel > 0, neigh_count == 1)
        
        if not np.any(endpoints):
            break # No more endpoints to prune

        # Check if any endpoint is adjacent to a branch point (pixel with > 2 neighbors)
        # If so, it's likely an artifact.
        endpoint_coords = np.argwhere(endpoints)
        can_prune = False
        for y, x in endpoint_coords:
            # Check the 3x3 neighborhood of the endpoint
            y_min, y_max = max(0, y - 1), min(pruned_skel.shape[0], y + 2)
            x_min, x_max = max(0, x - 1), min(pruned_skel.shape[1], x + 2)
            neighborhood = pruned_skel[y_min:y_max, x_min:x_max]
            
            # Count neighbors of neighbors
            # This part is a bit problematic: it applies filter2D on a small neighborhood
            # which might not correctly reflect branch points in the *original* skeleton context.
            # A more robust approach might involve checking the neighbor counts of the neighbors
            # in the *full* pruned_skel, or using a BFS/DFS from the endpoint.
            neighbor_neighbors = cv2.filter2D(neighborhood, -1, nb_kernel, borderType=cv2.BORDER_CONSTANT)
            
            # If any neighbor is a branch point, prune the endpoint
            if np.any(neighbor_neighbors > 2): # This condition is applied to the small neighborhood
                pruned_skel[y, x] = 0
                can_prune = True
        
        if not can_prune:
            break # No more prunable endpoints found in this iteration
            
    return pruned_skel
