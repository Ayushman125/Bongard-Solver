import json
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.physics_inference import PhysicsInference
from shapely.geometry import LineString
from sklearn.cluster import AgglomerativeClustering
from src.Derive_labels.relational_features import calculate_relationships
from src.Derive_labels.shape_utils import ensure_flat_str_list
from collections import Counter
logger = logging.getLogger(__name__)

def ensure_str_list(obj):
    """Recursively convert all items in a list (or nested lists/tuples) to strings."""
    if isinstance(obj, list):
        # Debug log for list conversion
        logger.debug(f"[ensure_str_list] Converting list: {obj}")
        result = [ensure_str_list(x) for x in obj]
        logger.debug(f"[ensure_str_list] Result: {result}")
        return result
    elif isinstance(obj, tuple):
        logger.debug(f"[ensure_str_list] Converting tuple: {obj}")
        result = tuple(ensure_str_list(x) for x in obj)
        logger.debug(f"[ensure_str_list] Result: {result}")
        return result
    elif not isinstance(obj, str):
        logger.debug(f"[ensure_str_list] Non-str object: {obj} (type: {type(obj)})")
        # Special handling for action objects
        if hasattr(obj, 'raw_command') and obj.raw_command is not None:
            logger.debug(f"[ensure_str_list] Using raw_command: {obj.raw_command}")
            return str(obj.raw_command)
        # Check if it's a LineAction or ArcAction without raw_command
        elif hasattr(obj, 'line_type') or hasattr(obj, 'arc_type'):
            try:
                if hasattr(obj, 'line_type'):  # LineAction
                    val = f"line_{obj.line_type}_{obj.line_length}-{getattr(obj, 'turn_angle', 0.5)}"
                    logger.debug(f"[ensure_str_list] LineAction string: {val}")
                    return val
                elif hasattr(obj, 'arc_type'):  # ArcAction
                    val = f"arc_{obj.arc_type}_{obj.arc_radius}_{obj.arc_angle}-{getattr(obj, 'turn_angle', 0.5)}"
                    logger.debug(f"[ensure_str_list] ArcAction string: {val}")
                    return val
            except Exception as e:
                logger.error(f"[ensure_str_list] Failed to convert action object: {e}")
        val = str(obj)
        logger.debug(f"[ensure_str_list] Fallback string conversion: {val}")
        return val
    logger.debug(f"[ensure_str_list] Returning string: {obj}")
    return obj

def extract_topological_features(shapes):
    """
    Extract topological features (connectivity, type) from a list of shapes.
    Each shape should have a 'vertices' key.
    """
    # logger.info(f"[extract_topological_features] INPUT: {shapes}")
    from collections import Counter
    import numpy as np
    if not shapes:
        logger.warning("[extract_topological_features] No shapes provided.")
        result = {'type': 'none', 'connectivity': '0', 'shape_distribution': {}}
        logger.info(f"[extract_topological_features] OUTPUT: {result}")
        return result
    from .quality_monitor import quality_monitor
    from .shape_utils import calculate_geometry
    shape_types = []
    connectivity = 0
    for idx, shape in enumerate(shapes):
        verts = shape.get('vertices', [])
        geom = calculate_geometry(verts)
        quality_monitor.log_quality('topological_features', {'degenerate': geom['degenerate_case'], 'area': geom['area']})
        if geom['degenerate_case']:
            # Fallback: try symbolic perturbation or alternative extraction
            shape_types.append('degenerate')
        else:
            if geom['area'] and geom['area'] > 0:
                shape_types.append('closed')
            else:
                shape_types.append('open')
        connectivity += 1
    topology_type = shape_types[0] if len(set(shape_types)) == 1 else 'mixed'
    result = {
        'type': topology_type,
        'connectivity': str(connectivity),
        'shape_distribution': dict(Counter(shape_types))
    }
    # logger.info(f"[extract_topological_features] OUTPUT: {result}")
    return result

def extract_multiscale_features(shape_vertices, scales=[0.1, 0.3, 0.5, 1.0, 2.0]):
    """Extract features at multiple geometric scales using Gaussian smoothing."""

    # logger.info(f"[MULTISCALE][INPUT] vertices: {shape_vertices}")
    multiscale_features = {}
    if not shape_vertices or not isinstance(shape_vertices, list) or len(shape_vertices) < 3:
        logger.warning("[MULTISCALE][WARN] Not enough vertices for multiscale analysis.")
        logger.info(f"[MULTISCALE][OUTPUT] (degenerate): {{'scale_1': 0.0, 'scale_2': 0.0, 'scale_3': 0.0}}")
        return {'scale_1': 0.0, 'scale_2': 0.0, 'scale_3': 0.0}
    shape_vertices = [tuple(map(float, v)) if isinstance(v, (list, tuple)) and len(v) == 2 else v for v in shape_vertices]
    arr = np.array(shape_vertices)
    for scale in scales:
    # logger.info(f"[MULTISCALE][PROCESS] scale={scale}, input_vertices={arr.tolist()}")
        smoothed_x = gaussian_filter1d(arr[:,0], sigma=scale, mode='wrap')
        smoothed_y = gaussian_filter1d(arr[:,1], sigma=scale, mode='wrap')
        smoothed_vertices = np.stack([smoothed_x, smoothed_y], axis=1)
        curvature = PhysicsInference.robust_curvature(smoothed_vertices)
        angular_variance = PhysicsInference.robust_angular_variance(smoothed_vertices)
        diffs = np.diff(smoothed_vertices, axis=0)
        angles = np.arctan2(diffs[:,1], diffs[:,0])
        angle_diffs = np.diff(angles)
        inflections = np.sum(np.abs(np.diff(np.sign(angle_diffs))) > 0)
        complexity = inflections
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
            logger.info(f"[MULTISCALE][GROUPS] scale={scale}, group_features={group_features}")
        except Exception as e:
            logger.warning(f"[MULTISCALE][WARN] Hierarchical clustering failed at scale {scale}: {e}")
            group_features = {}
        multiscale_features[f'scale_{scale}'] = {
            'curvature': float(curvature),
            'angular_variance': float(angular_variance),
            'complexity': int(complexity),
            'groups': group_features
        }
        logger.info(f"[MULTISCALE][FEATURES] scale={scale}, features={multiscale_features[f'scale_{scale}']}")
    logger.info(f"[MULTISCALE][OUTPUT] {multiscale_features}")
    return multiscale_features



def _actions_to_geometries(shape, arc_points=24):
        """
        Convert all basic_actions in a shape to shapely geometries (LineString), using the true world-space vertices from shape.vertices.
        Each stroke is a segment between consecutive vertices. Fallback to synthetic only if vertices are missing.
        """
        verts = getattr(shape, 'vertices', None)
        geoms = []
        # Ensure all vertices are tuples for hashing and geometry
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 2:
            verts = [tuple(map(float, v)) if isinstance(v, (list, tuple)) and len(v) == 2 else v for v in verts]
            for i in range(len(verts) - 1):
                try:
                    seg = LineString([verts[i], verts[i+1]])
                    if seg.is_valid and not seg.is_empty:
                        geoms.append(seg)
                    else:
                        logger.debug(f"Stroke {i}: invalid or empty LineString from vertices {verts[i]}, {verts[i+1]}")
                except Exception as e:
                    logger.debug(f"Stroke {i}: failed to create LineString: {e}")
        else:
            # Fallback: try to synthesize as before (should rarely happen)
            actions = getattr(shape, 'basic_actions', [])
            logger.debug(f"[actions_to_geometries] Raw actions before stringification: {actions}")
            from src.Derive_labels.features import ensure_str_list
            actions_str = ensure_str_list(actions)
            logger.debug(f"[actions_to_geometries] Actions after stringification: {actions_str}")
            for i, action in enumerate(actions):
                v = getattr(action, 'vertices', None)
                if v and isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        seg = LineString(v)
                        if seg.is_valid and not seg.is_empty:
                            geoms.append(seg)
                    except Exception as e:
                        logger.debug(f"Fallback: failed to create LineString for stroke {i}: {e}")
        logger.debug(f"Number of stroke geometries: {len(geoms)}")
        # Ensure output is serializable
        try:
            json.dumps([str(g) for g in geoms])
            logger.debug(f"[actions_to_geometries] Output is JSON serializable.")
        except Exception as e:
            logger.error(f"[actions_to_geometries] Output not JSON serializable: {e}")
        return geoms

def extract_relational_features(strokes, buffer_amt=0.001):
    """
    Compute adjacency, intersection, containment, and overlap using robust buffered geometry (delegates to calculate_relationships).
    Args:
        strokes: list of stroke dicts or objects with 'vertices' attribute
        buffer_amt: float, buffer size for robust geometry
    Returns:
        dict with keys: 'adjacency', 'intersections', 'containment', 'overlap'
    """
    logger.info(f"[extract_relational_features] INPUT: {strokes}")
    # Expecting a list of shapes, each with a 'vertices' key
    try:
        for idx, s in enumerate(strokes):
            logger.debug(f"[extract_relational_features] Shape {idx} vertices: {s.get('vertices', None)}")
        rel = calculate_relationships(strokes, buffer_amt)
        logger.info(f"[extract_relational_features] OUTPUT: {rel}")
        try:
            json.dumps(rel)
            logger.debug(f"[extract_relational_features] Output is JSON serializable.")
        except Exception as e:
            logger.error(f"[extract_relational_features] Output not JSON serializable: {e}")
        if not rel or all(v in (None, [], {}, '') for v in rel.values()):
            logger.warning(f"[extract_relational_features] Relationships output is empty or default. Input: {strokes}")
        return rel
    except Exception as exc:
        logger.error(f"[extract_relational_features] Exception occurred: {exc}")
        return {}

    

def _extract_ngram_features(sequence, n=2):
    """Extract n-gram counts from a sequence, with string keys for JSON compatibility."""
    logger.debug(f"[_extract_ngram_features] INPUTS: sequence={sequence}, n={n}")
    sequence_str = ensure_str_list(sequence)
    logger.debug(f"[_extract_ngram_features] Sequence after stringification: {sequence_str}")
    ngrams = zip(*[sequence_str[i:] for i in range(n)])
    ngram_list = ['|'.join(ensure_flat_str_list(ng)) for ng in ngrams]
    ngram_list = ensure_flat_str_list(ngram_list)
    result = dict(Counter(ngram_list))
    logger.debug(f"[_extract_ngram_features] OUTPUT: {result}")
    try:
        json.dumps(result)
        logger.debug(f"[_extract_ngram_features] Output is JSON serializable.")
    except Exception as e:
        logger.error(f"[_extract_ngram_features] Output not JSON serializable: {e}")
    return result


# --- Regularity Feature Extraction ---
def extract_regularity_features(sequence):
    """
    Extract regularity features from a sequence of actions or shape descriptors.
    Returns a dict with regularity score and supporting details.
    """
    logger.info(f"[extract_regularity_features] INPUT: {sequence}")
    sequence_str = ensure_str_list(sequence)
    # Example: regularity as ratio of most common element to total
    counts = Counter(sequence_str)
    most_common = counts.most_common(1)[0][1] if counts else 0
    regularity_score = most_common / len(sequence_str) if sequence_str else 0.0
    result = {
        'regularity_score': regularity_score,
        'most_common_element': counts.most_common(1)[0][0] if counts else None,
        'distribution': dict(counts)
    }
    logger.info(f"[extract_regularity_features] OUTPUT: {result}")
    try:
        json.dumps(result)
        logger.debug(f"[extract_regularity_features] Output is JSON serializable.")
    except Exception as e:
        logger.error(f"[extract_regularity_features] Output not JSON serializable: {e}")
    return result

# --- Dominant Shape Functions/Modifiers Extraction ---
def extract_dominant_shape_modifiers(shape):
    """
    Extract dominant shape functions/modifiers from a shape object or descriptor list.
    Returns a dict with dominant modifier and supporting details.
    """
    logger.info(f"[extract_dominant_shape_modifiers] INPUT: {shape}")
    # Example: look for 'modifiers' attribute or key
    modifiers = []
    if hasattr(shape, 'modifiers'):
        modifiers = getattr(shape, 'modifiers', [])
    elif isinstance(shape, dict) and 'modifiers' in shape:
        modifiers = shape['modifiers']
    # Fallback: try to infer from basic_actions if present
    elif hasattr(shape, 'basic_actions'):
        actions = getattr(shape, 'basic_actions', [])
        for act in actions:
            if hasattr(act, 'modifier'):
                modifiers.append(act.modifier)
            elif isinstance(act, dict) and 'modifier' in act:
                modifiers.append(act['modifier'])
    # Count and find dominant
    modifiers_str = ensure_str_list(modifiers)
    counts = Counter(modifiers_str)
    dominant_modifier = counts.most_common(1)[0][0] if counts else None
    result = {
        'dominant_modifier': dominant_modifier,
        'modifier_distribution': dict(counts)
    }
    logger.info(f"[extract_dominant_shape_modifiers] OUTPUT: {result}")
    try:
        json.dumps(result)
        logger.debug(f"[extract_dominant_shape_modifiers] Output is JSON serializable.")
    except Exception as e:
        logger.error(f"[extract_dominant_shape_modifiers] Output not JSON serializable: {e}")
    return result

def _detect_alternation(sequence):
        """Compute maximal alternation score using PhysicsInference.alternation_score."""
        from src.Derive_labels.features import ensure_str_list
        logger.debug(f"[_detect_alternation] INPUTS: sequence={sequence}")
        # Ensure sequence is stringified
        sequence_str = ensure_str_list(sequence)
        logger.debug(f"[_detect_alternation] Sequence after stringification: {sequence_str}")
        score = PhysicsInference.alternation_score(sequence_str)
        logger.debug(f"[_detect_alternation] OUTPUT: {score}")
        # Validate output is JSON serializable
        try:
            json.dumps(score)
            logger.debug(f"[_detect_alternation] Output is JSON serializable.")
        except Exception as e:
            logger.error(f"[_detect_alternation] Output not JSON serializable: {e}")
        return score

def _extract_graph_features(strokes):
    """
    Advanced graph topology detection for Bongard-LOGO datasets.
    Accepts adjacency/intersection matrix (undirected), computes graph statistics and classifies topology.
    Input: adjacency_matrix (2D list or np.ndarray), optionally strokes for reference.
    Output: dict with topology type(s), graph statistics, and logs all inputs/outputs.
    """
    logger.info(f"[_extract_graph_features] RAW INPUT: {repr(strokes)}")
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

