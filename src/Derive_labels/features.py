import numpy as np
from scipy.ndimage import gaussian_filter1d
def extract_multiscale_features(shape_vertices, scales=[0.1, 0.3, 0.5, 1.0, 2.0]):
    """Extract features at multiple geometric scales using Gaussian smoothing."""
    logger = logging.getLogger(__name__)
    logger.info(f"[extract_multiscale_features] INPUT vertices: {shape_vertices}")
    multiscale_features = {}
    if not shape_vertices or len(shape_vertices) < 3:
        logger.warning("[extract_multiscale_features] Not enough vertices for multiscale analysis.")
        return multiscale_features
    arr = np.array(shape_vertices)
    for scale in scales:
        # Smooth x and y separately
        smoothed_x = gaussian_filter1d(arr[:,0], sigma=scale, mode='wrap')
        smoothed_y = gaussian_filter1d(arr[:,1], sigma=scale, mode='wrap')
        smoothed_vertices = np.stack([smoothed_x, smoothed_y], axis=1)
        curvature = PhysicsInference.robust_curvature(smoothed_vertices)
        angular_variance = PhysicsInference.robust_angular_variance(smoothed_vertices)
        # Complexity: number of inflection points (sign changes in curvature)
        diffs = np.diff(smoothed_vertices, axis=0)
        angles = np.arctan2(diffs[:,1], diffs[:,0])
        angle_diffs = np.diff(angles)
        inflections = np.sum(np.abs(np.diff(np.sign(angle_diffs))) > 0)
        complexity = inflections
        # Hierarchical/grouping: cluster vertices at this scale
        from sklearn.cluster import AgglomerativeClustering
        n_clusters = min(max(2, int(len(smoothed_vertices) // (3 + scale * 5))), len(smoothed_vertices))
        try:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(smoothed_vertices)
            labels = clustering.labels_
            group_features = {}
            for group in range(n_clusters):
                group_idx = np.where(labels == group)[0]
                group_pts = smoothed_vertices[group_idx]
                if len(group_pts) > 1:
                    group_curvature = PhysicsInference.robust_curvature(group_pts)
                    group_complexity = np.sum(np.abs(np.diff(np.sign(np.diff(np.arctan2(np.diff(group_pts, axis=0)[:,1], np.diff(group_pts, axis=0)[:,0]))))) > 0)
                else:
                    group_curvature = 0.0
                    group_complexity = 0
                group_features[f'group_{group}'] = {
                    'size': int(len(group_pts)),
                    'curvature': float(group_curvature),
                    'complexity': int(group_complexity)
                }
            logger.info(f"[extract_multiscale_features] scale={scale}, group_features={group_features}")
        except Exception as e:
            logger.warning(f"[extract_multiscale_features] Hierarchical clustering failed at scale {scale}: {e}")
            group_features = {}
        multiscale_features[f'scale_{scale}'] = {
            'curvature': float(curvature),
            'angular_variance': float(angular_variance),
            'complexity': int(complexity),
            'groups': group_features
        }
        logger.info(f"[extract_multiscale_features] scale={scale}, features={multiscale_features[f'scale_{scale}']}")
    logger.info(f"[extract_multiscale_features] OUTPUT: {multiscale_features}")
    return multiscale_features
import logging
from src.physics_inference import PhysicsInference

def _actions_to_geometries(shape, arc_points=24):
        """
        Convert all basic_actions in a shape to shapely geometries (LineString), using the true world-space vertices from shape.vertices.
        Each stroke is a segment between consecutive vertices. Fallback to synthetic only if vertices are missing.
        """
        from shapely.geometry import LineString
        import logging
        verts = getattr(shape, 'vertices', None)
        geoms = []
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 2:
            for i in range(len(verts) - 1):
                try:
                    seg = LineString([verts[i], verts[i+1]])
                    if seg.is_valid and not seg.is_empty:
                        geoms.append(seg)
                    else:
                        logging.debug(f"Stroke {i}: invalid or empty LineString from vertices {verts[i]}, {verts[i+1]}")
                except Exception as e:
                    logging.debug(f"Stroke {i}: failed to create LineString: {e}")
        else:
            # Fallback: try to synthesize as before (should rarely happen)
            actions = getattr(shape, 'basic_actions', [])
            for i, action in enumerate(actions):
                v = getattr(action, 'vertices', None)
                if v and isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        seg = LineString(v)
                        if seg.is_valid and not seg.is_empty:
                            geoms.append(seg)
                    except Exception as e:
                        logging.debug(f"Fallback: failed to create LineString for stroke {i}: {e}")
        logging.debug(f"Number of stroke geometries: {len(geoms)}")
        return geoms
    
def _extract_ngram_features(sequence, n=2):
        """Extract n-gram counts from a sequence, with string keys for JSON compatibility."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_extract_ngram_features] INPUTS: sequence={sequence}, n={n}")
        from collections import Counter
        ngrams = zip(*[sequence[i:] for i in range(n)])
        ngram_list = ['|'.join(map(str, ng)) for ng in ngrams]
        result = dict(Counter(ngram_list))
        logger.debug(f"[_extract_ngram_features] OUTPUT: {result}")
        return result

def _detect_alternation(sequence):
        """Compute maximal alternation score using PhysicsInference.alternation_score."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_detect_alternation] INPUTS: sequence={sequence}")
        score = PhysicsInference.alternation_score(sequence)
        logger.debug(f"[_detect_alternation] OUTPUT: {score}")
        return score

def _extract_graph_features(strokes):
    """
    Advanced graph topology detection for Bongard-LOGO datasets.
    Accepts adjacency/intersection matrix (undirected), computes graph statistics and classifies topology.
    Input: adjacency_matrix (2D list or np.ndarray), optionally strokes for reference.
    Output: dict with topology type(s), graph statistics, and logs all inputs/outputs.
    """
    import numpy as np
    import logging
    logger = logging.getLogger(__name__)
    # Accept either strokes or adjacency/intersection matrix
    if isinstance(strokes, dict) and 'adjacency_matrix' in strokes:
        adj = np.array(strokes['adjacency_matrix'])
        n = adj.shape[0]
        logger.info(f"[_extract_graph_features] INPUT: adjacency_matrix shape={adj.shape}")
    elif isinstance(strokes, (list, np.ndarray)) and len(strokes) > 0 and isinstance(strokes[0], (list, np.ndarray)):
        adj = np.array(strokes)
        n = adj.shape[0]
        logger.info(f"[_extract_graph_features] INPUT: adjacency_matrix shape={adj.shape}")
    else:
        logger.warning(f"[_extract_graph_features] Invalid input, expected adjacency matrix or dict. Got: {type(strokes)}")
        return {'type': 'none', 'connectivity': 0}

    # Degree for each node
    degrees = np.sum(adj, axis=1)
    logger.info(f"[_extract_graph_features] Node degrees: {degrees.tolist()}")

    # Connected components (BFS)
    visited = [False]*n
    components = []
    for i in range(n):
        if not visited[i]:
            queue = [i]
            comp = []
            while queue:
                node = queue.pop(0)
                if not visited[node]:
                    visited[node] = True
                    comp.append(node)
                    neighbors = np.where(adj[node] > 0)[0]
                    for nb in neighbors:
                        if not visited[nb]:
                            queue.append(nb)
            components.append(comp)
    num_components = len(components)
    logger.info(f"[_extract_graph_features] Connected components: {components}")

    # Cycle detection (DFS)
    def has_cycle(adj):
        parent = [-1]*n
        visited = [False]*n
        def dfs(v):
            visited[v] = True
            for u in np.where(adj[v] > 0)[0]:
                if not visited[u]:
                    parent[u] = v
                    if dfs(u):
                        return True
                elif parent[v] != u:
                    return True
            return False
        for i in range(n):
            if not visited[i]:
                if dfs(i):
                    return True
        return False
    cycle_present = has_cycle(adj)
    logger.info(f"[_extract_graph_features] Cycle present: {cycle_present}")

    # Topology classification
    topology = []
    if num_components > 1:
        topology.append('disconnected')
    else:
        degs = degrees.tolist()
        if all(d == n-1 for d in degs):
            topology.append('clique')
        elif cycle_present and all(d == 2 for d in degs):
            topology.append('cycle')
        elif degs.count(1) == 2 and degs.count(2) == n-2 and not cycle_present:
            topology.append('chain')
        elif degs.count(1) >= 2 and max(degs) > 2 and degs.count(max(degs)) == 1:
            topology.append('star')
        elif not cycle_present:
            topology.append('tree')
        else:
            topology.append('hybrid')

    stats = {
        'num_strokes': n,
        'degrees': degs,
        'num_components': num_components,
        'components': components,
        'cycle_present': cycle_present,
        'topology_type': topology,
        'is_connected': num_components == 1,
    }
    logger.info(f"[_extract_graph_features] OUTPUT: {stats}")
    return stats

