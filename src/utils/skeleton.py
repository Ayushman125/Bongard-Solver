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


def preprocess_mask(mask):
    """
    Preprocesses a mask for skeletonization. Ensures binary uint8 (0/1), removes small noise.
    Args:
        mask (np.ndarray): Input mask (any dtype, 0/255 or 0/1).
    Returns:
        np.ndarray: Cleaned binary mask (0/1, uint8).
    """
    mask = (mask > 0).astype(np.uint8)
    # Remove small noise (optional, can add morphology if needed)
    return mask


def compute_skeleton(mask):
    """
    Computes the skeleton of a binary image and extracts key metrics.
    Args:
        mask (np.ndarray): 2D uint8 or bool binary image (0/1 or 0/255).
    Returns:
        tuple: (skel_img, metrics_dict)
    """
    clean = preprocess_mask(mask)
    if clean.sum() == 0:
        skel_img = np.zeros_like(clean, dtype=np.uint8)
        metrics = {'stroke_count': 0, 'endpoint_count': 0, 'branch_point_count': 0, 'skeleton_length': 0}
        return skel_img, metrics
    binary = clean > 0
    skel_bool = skeletonize(binary)
    # Prune spurs (short branches) to make endpoint count robust for thick junctions
    skel_bool = prune_spurs(skel_bool, max_spur_length=10)
    # 4-connectivity for metrics
    from scipy.ndimage import convolve
    four_k = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=int)
    nb4  = convolve(skel_bool.astype(int), four_k,  mode='constant', cval=0)
    # Find endpoints and branch points
    endpoints = np.argwhere(np.logical_and(skel_bool, nb4 == 1))
    branch_points = np.argwhere(np.logical_and(skel_bool, nb4 >= 3))
    # Cluster endpoints that are very close (to handle thick Y-junctions)
    def cluster_points(points, eps=2):
        if len(points) == 0:
            return points
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return points  # fallback: no clustering
        clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
        clustered = []
        for label in np.unique(clustering.labels_):
            cluster = points[clustering.labels_ == label]
            centroid = np.mean(cluster, axis=0)
            clustered.append(centroid)
        return np.array(clustered, dtype=int)
    endpoints_clustered = cluster_points(endpoints, eps=2)
    stroke_count     = int(skel_bool.sum())
    endpoint_count   = len(endpoints_clustered)
    branch_point_count = len(branch_points)
    skeleton_length  = stroke_count
    skel_img = (skel_bool.astype(np.uint8) * 255)
    metrics = {
        'stroke_count': stroke_count,
        'endpoint_count': endpoint_count,
        'branch_point_count': branch_point_count,
        'skeleton_length': skeleton_length
    }
    return skel_img, metrics
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
    clean = preprocess_mask(mask)
    if clean.sum() == 0:
        skel_img = np.zeros_like(clean, dtype=np.uint8)
        metrics = {'stroke_count': 0, 'endpoint_count': 0, 'branch_point_count': 0, 'skeleton_length': 0}
        return skel_img, metrics
    binary = clean > 0
    skel_bool = skeletonize(binary)
    # Prune spurs (short branches) to make endpoint count robust for thick junctions
    skel_bool = prune_spurs(skel_bool, max_spur_length=10)
    # 4-connectivity for metrics
    from scipy.ndimage import convolve
    four_k = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=int)
    nb4  = convolve(skel_bool.astype(int), four_k,  mode='constant', cval=0)
    # Find endpoints and branch points
    endpoints = np.argwhere(np.logical_and(skel_bool, nb4 == 1))
    branch_points = np.argwhere(np.logical_and(skel_bool, nb4 >= 3))
    # Cluster endpoints that are very close (to handle thick Y-junctions)
    def cluster_points(points, eps=2):
        if len(points) == 0:
            return points
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return points  # fallback: no clustering
        clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
        clustered = []
        for label in np.unique(clustering.labels_):
            cluster = points[clustering.labels_ == label]
            centroid = np.mean(cluster, axis=0)
            clustered.append(centroid)
        return np.array(clustered, dtype=int)
    endpoints_clustered = cluster_points(endpoints, eps=2)
    stroke_count     = int(skel_bool.sum())
    endpoint_count   = len(endpoints_clustered)
    branch_point_count = len(branch_points)
    skeleton_length  = stroke_count
    skel_img = (skel_bool.astype(np.uint8) * 255)
    metrics = {
        'stroke_count': stroke_count,
        'endpoint_count': endpoint_count,
        'branch_point_count': branch_point_count,
        'skeleton_length': skeleton_length
    }
    return skel_img, metrics
