# --- Standard Library Imports ---
from typing import Tuple, Dict, Any, List
import torch
import argparse
import asyncio
import concurrent.futures
import gc
import hashlib
import json
import logging
import os
import pickle
import sys
import threading
import time
import torch
import math
from collections import Counter, defaultdict
from itertools import combinations # For global graph features
from typing import Tuple

# --- Third-Party Library Imports ---
import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw # Added ImageDraw for visualization
from scipy import stats
from shapely.geometry import Polygon, LineString, MultiLineString # Added MultiLineString
from shapely.ops import polygonize, unary_union
from tqdm import tqdm
from sklearn.cluster import KMeans # For clustering
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_ind

# Conditional imports for torch/kornia/torchvision
TORCH_KORNIA_AVAILABLE = True
try:
    import torch
    import kornia
    import torchvision
    TORCH_KORNIA_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch, Kornia, or Torchvision not found. Real feature extraction will be disabled.")

# Conditional imports for graphtype
GRAPHTYPE_AVAILABLE = True
try:
    from graphtype import graph_validate, NodeData, EdgeData
    GRAPHTYPE_AVAILABLE = True
except ImportError:
    # Dummy decorator if graphtype is not installed
    def graph_validate(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    NodeData = dict
    EdgeData = dict

# --- Local Application Imports ---
# Ensure project root is in sys.path for imports relative to the script's directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bongard_augmentor.memory_efficient_cache import MemoryEfficientFeatureCache
from src.bongard_augmentor.batch_validator import BatchMultimodalValidator
from src.bongard_augmentor.recall_metrics import RecallAtKEvaluator
from src.bongard_augmentor.drift_visualizer import ConceptDriftVisualizer
from src.bongard_augmentor.knowledge_fusion import MultiSourceKnowledgeFusion
from src.bongard_augmentor.sgcore_validator import SGScoreValidator
from src.bongard_augmentor.hierarchical_predicates import HierarchicalPredicatePredictor
from src.bongard_augmentor.feature_extractors import RealFeatureExtractor
from src.bongard_augmentor.adaptive_predicates import AdaptivePredicateThresholds, create_adaptive_predicate_functions
from src.commonsense_kb_api import ConceptNetAPI
from integration.task_profiler import TaskProfiler

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Constants and Utilities ---
EPS = 1e-6 # Increased precision for epsilon
TOP_K = 3 # For Commonsense KB lookups

# Commonsense KB setup (ConceptNet API)
try:
    kb = ConceptNetAPI()
    test_result = kb.query_relations_for_concept("dog")
    if test_result:
        logging.info(f"ConceptNet API connected successfully. Found {len(test_result)} relations for 'dog'")
    else:
        logging.warning("ConceptNet API test query returned no results")
except Exception as e:
    logging.error(f"ConceptNet API connection failed: {e}")
    logging.info("Continuing without ConceptNet knowledge base")
    kb = None

# --- Predicate Registry and Edge Enrichment Functions ---
def parallel(a, b, tol=7):
    """Checks if two orientations are parallel within a tolerance."""
    # Orientations are in degrees, 0-360.
    # Parallel means difference is 0 or 180 (modulo 360)
    diff = abs(a - b) % 180
    return diff < tol or (180 - diff) < tol

def physics_contact(poly_a, poly_b):
    """Robust check for physical contact between two polygons."""
    return poly_a.touches(poly_b)

def extract_line_segments(action_program):
    """
    Extracts line segments from a simplified action program.
    Assumes action_program is a list of commands like:
    [{'type': 'start', 'x': x1, 'y': y1}, {'type': 'line', 'x': x2, 'y': y2}, ...]
    """
    segments = []
    current_point = None
    if not isinstance(action_program, list):
        return [] # Return empty if not a list

    for command in action_program:
        if command['type'] == 'start':
            current_point = (command['x'], command['y'])
        elif command['type'] == 'line' and current_point:
            next_point = (command['x'], command['y'])
            segments.append((current_point, next_point))
            current_point = next_point
        # Add other command types if necessary (e.g., 'arc', 'curve')
    return segments

def load_predicates(adaptive_thresholds: AdaptivePredicateThresholds, canvas_dims: Tuple[int, int] = (128, 128)):
    """
    Loads predicate definitions from schemas/edge_types.json or uses built-in defaults,
    and integrates adaptive thresholds.
    """
    predicate_file = os.path.join(os.path.dirname(__file__), '..', 'schemas', 'edge_types.json')
    predicate_defs = []
    if os.path.exists(predicate_file):
        try:
            with open(predicate_file, 'r', encoding='utf-8') as f:
                predicate_defs = json.load(f)
            logging.info(f"Loaded {len(predicate_defs)} predicate definitions from {predicate_file}.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding predicates from {predicate_file}: {e}. Using built-in defaults.")
    else:
        logging.warning("schemas/edge_types.json not found. Using built-in predicates.")

    registry = {}

    # Override with parameters from adaptive thresholds
    # Robustly get thresholds from adaptive_thresholds object
    if hasattr(adaptive_thresholds, 'get_current_thresholds'):
        learned_params = adaptive_thresholds.get_current_thresholds()
    elif hasattr(adaptive_thresholds, 'thresholds'): # Fallback if direct attribute access is needed
        learned_params = adaptive_thresholds.thresholds
    else:
        learned_params = {} # Default to empty if no known method/attribute
    logging.info(f"Loaded adaptive predicate parameters: {learned_params}")

    # Define all predicates, prioritizing those from file if they exist
    # Each lambda now receives 'a', 'b' (node data) and 'params' (learned thresholds)
    all_candidate_predicates = {
        'left_of':     lambda a, b, params: a.get('cx', 0) + EPS < b.get('cx', 0),
        'right_of':    lambda a, b, params: a.get('cx', 0) > b.get('cx', 0) + EPS,
        'above':       lambda a, b, params: a.get('cy', 0) + EPS < b.get('cy', 0),
        'below':       lambda a, b, params: a.get('cy', 0) > b.get('cy', 0) + EPS,
        'contains':    lambda a, b, params: Polygon(a['vertices']).contains(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
        'inside':      lambda a, b, params: Polygon(b['vertices']).contains(Polygon(a['vertices'])) if 'vertices' in a and 'vertices' in b else False,
        'supports':    lambda a, b, params: (physics_contact(Polygon(a['vertices']), Polygon(b['vertices'])) and a.get('cy', 0) > b.get('cy', 0)) if 'vertices' in a and 'vertices' in b else False,
        'supported_by':lambda a, b, params: (physics_contact(Polygon(b['vertices']), Polygon(a['vertices'])) and b.get('cy', 0) > a.get('cy', 0)) if 'vertices' in a and 'vertices' in b else False,
        'touches':     lambda a, b, params: Polygon(a['vertices']).touches(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
        'overlaps':    lambda a, b, params: Polygon(a['vertices']).overlaps(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
        'parallel_to': lambda a, b, params: parallel(a.get('orientation', 0), b.get('orientation', 0), params.get('orient_tol', 7)), # Parameterized
        'perpendicular_to': lambda a, b, params: abs(abs(a.get('orientation', 0) - b.get('orientation', 0)) - 90) < params.get('orient_tol', 7), # Parameterized
        'aligned_left':lambda a, b, params: abs(a.get('bbox', [0,0,0,0])[0] - b.get('bbox', [0,0,0,0])[0]) < EPS,
        'aligned_right':lambda a, b, params: abs(a.get('bbox', [0,0,0,0])[2] - b.get('bbox', [0,0,0,0])[2]) < EPS,
        'aligned_top': lambda a, b, params: abs(a.get('bbox', [0,0,0,0])[1] - b.get('bbox', [0,0,0,0])[1]) < EPS,
        'aligned_bottom':lambda a, b, params: abs(a.get('bbox', [0,0,0,0])[3] - b.get('bbox', [0,0,0,0])[3]) < EPS,
        'proximal_to': lambda a, b, params: np.linalg.norm(np.array([a.get('cx',0), a.get('cy',0)]) - np.array([b.get('cx',0), b.get('cy',0)])) < params.get('near_threshold', 50), # Parameterized
        'contains_text': lambda a, b, params: False, # KB-based, handled in add_commonsense_edges
        'is_arrow_for': lambda a, b, params: False, # KB-based, handled in add_commonsense_edges
        'has_sides':   lambda a, b, params: False, # KB-based, handled in add_commonsense_edges
        'same_shape':  lambda a, b, params: a.get('shape_label') == b.get('shape_label'),
        'symmetry_axis':lambda a, b, params: abs(a.get('symmetry_axis',0) - b.get('orientation',0)) < params.get('orient_tol', 7), # Parameterized
        'same_color':  lambda a, b, params: np.all(np.abs(np.array(a.get('color',[0,0,0])) - np.array(b.get('color',[0,0,0]))) < params.get('color_tol', 10)), # Parameterized
        # New predicates from prompt
        'larger_than': lambda a, b, params: a.get('area', 0) > params.get('larger_than_alpha', 1.1) * b.get('area', 0), # Parameterized
        'near':        lambda a, b, params: math.hypot(a.get('cx',0)-b.get('cx',0), a.get('cy',0)-b.get('cy',0)) < params.get('near_threshold', 50), # Parameterized
        'same_aspect': lambda a, b, params: abs(a.get('aspect_ratio', 1.0) - b.get('aspect_ratio', 1.0)) < params.get('aspect_tol', 0.2), # Parameterized
        'clustered':   lambda a, b, params: a.get('cluster_label') == b.get('cluster_label'), # New, uses cluster_label
        'high_compact':lambda a, b, params: a.get('compactness', 0.0) > params.get('compactness_thresh', 0.5), # New, uses compactness, single node predicate
        'symmetry_pair': lambda a, b, params: abs(a.get('cx',0) + b.get('cx',0) - canvas_dims[0]) < params.get('symmetry_tol', 10) and \
                                              abs(a.get('cy',0) + b.get('cy',0) - canvas_dims[1]) < params.get('symmetry_tol', 10), # Parameterized with canvas_dims
        'part_of':     lambda a, b, params: Polygon(b['vertices']).contains(Polygon(a['vertices'])) and a.get('parent_id') == b.get('id') # For decomposed parts
    }

    # Wrap predicates to pass learned_params
    for name, func in all_candidate_predicates.items():
        # All predicates now receive params, even if they don't use them.
        # This simplifies the calling signature in add_predicate_edges.
        registry[name] = lambda a, b, func=func: func(a, b, learned_params)

    # Add predicates from file if they are not already in all_candidate_predicates
    for entry in predicate_defs:
        name = entry['predicate']
        if name not in registry:
            # If custom logic is provided in schemas/edge_types.json, add it here.
            # For now, assuming file predicates are just names that map to built-in logic.
            if name in all_candidate_predicates:
                registry[name] = lambda a, b, func=all_candidate_predicates[name]: func(a, b, learned_params)
            else:
                logging.warning(f"Predicate '{name}' from edge_types.json has no defined logic in build_scene_graphs.py. Skipping.")

    return registry

# PREDICATES will be initialized in main() after adaptive_thresholds are loaded
PREDICATES = {}

def induce_predicate_automl(objects, automl_type='tpot', max_time_mins=2):
    """
    Uses TPOT or AutoSklearn to automate feature selection and rule induction for predicate induction.
    Returns the best feature and threshold found by the AutoML pipeline.
    """
    try:
        features = []
        labels = []
        for obj in objects:
            features.append([
                obj.get('area', 0),
                obj.get('aspect_ratio', 1),
                obj.get('compactness', 0),
                obj.get('orientation', 0)
            ])
            labels.append(obj.get('category', 0))
        import numpy as np
        features = np.array(features)
        labels = np.array(labels)
        feature_names = ['area', 'aspect_ratio', 'compactness', 'orientation']
        if automl_type == 'tpot':
            try:
                from tpot import TPOTClassifier
                tpot = TPOTClassifier(generations=2, population_size=20, max_time_mins=max_time_mins, random_state=42)
                tpot.fit(features, labels)
                # Check for both fitted_pipeline_ and fitted_pipeline attributes
                pipeline = None
                if hasattr(tpot, 'fitted_pipeline_'):
                    pipeline = tpot.fitted_pipeline_
                elif hasattr(tpot, 'fitted_pipeline'):
                    pipeline = tpot.fitted_pipeline
                if pipeline is not None and hasattr(pipeline, 'feature_importances_'):
                    importances = pipeline.feature_importances_
                    best_idx = int(np.argmax(importances))
                    best_feature = feature_names[best_idx]
                    return f"{best_feature}_automl_tpot", best_feature
                else:
                    logging.warning("TPOT did not produce a valid pipeline with feature_importances_. Returning fallback.")
                    return "same_shape", None
            except Exception as e:
                logging.warning(f"TPOT failed: {e}")
                return "same_shape", None
        elif automl_type == 'autosklearn':
            try:
                import autosklearn.classification
                automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=max_time_mins*60, per_run_time_limit=60)
                automl.fit(features, labels)
                # Extract feature importances if available
                if hasattr(automl, 'feature_importances_'):
                    importances = automl.feature_importances_
                    best_idx = int(np.argmax(importances))
                    best_feature = feature_names[best_idx]
                    return f"{best_feature}_automl_autosklearn", best_feature
                else:
                    return "same_shape", None
            except Exception as e:
                logging.warning(f"AutoSklearn failed: {e}")
                return "same_shape", None
        else:
            logging.warning(f"Unknown automl_type: {automl_type}")
            return "same_shape", None
    except Exception as e:
        logging.warning(f"AutoML predicate induction failed: {e}")
        return "same_shape", None


def induce_predicate_statistical(objects):
    pos = [o for o in objects if o.get('category') == 1]
    neg = [o for o in objects if o.get('category') == 0]
    features = ['area', 'aspect_ratio', 'compactness', 'orientation']
    best_feature = None
    best_p = 1.0
    for feat in features:
        pos_vals = [o.get(feat, 0) for o in pos]
        neg_vals = [o.get(feat, 0) for o in neg]
        if len(pos_vals) > 1 and len(neg_vals) > 1:
            stat, p = ttest_ind(pos_vals, neg_vals, equal_var=False)
            if p < best_p:
                best_p = p
                best_feature = feat
    if best_feature and best_p < 0.05:
        return f"{best_feature}_statistically_significant", best_feature
    return "same_shape", None

def induce_predicate_decision_tree(objects):
    # Prepare features and labels
    features = []
    labels = []
    for obj in objects:
        # Example features: area, aspect_ratio, compactness, orientation
        features.append([
            obj.get('area', 0),
            obj.get('aspect_ratio', 1),
            obj.get('compactness', 0),
            obj.get('orientation', 0)
        ])
        labels.append(obj.get('category', 0))
    features = np.array(features)
    labels = np.array(labels)

    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(features, labels)

    # Extract rules (predicates)
    # You can use tree_ structure to extract splits and thresholds
    # For demonstration, return the most important feature and threshold
    feature_names = ['area', 'aspect_ratio', 'compactness', 'orientation']
    if hasattr(clf, 'tree_'):
        tree = clf.tree_
        if tree.feature[0] != -2:  # -2 means leaf
            split_feature = feature_names[tree.feature[0]]
            threshold = tree.threshold[0]
            return f"{split_feature}_gt_{threshold:.2f}", (split_feature, threshold)
    return "same_shape", None


def add_predicate_edges(G):
    """Iterates through all node pairs and adds edges based on the predicate registry."""
    node_list = list(G.nodes(data=True))
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if i == j:
                continue
            # Ensure a and b are valid dicts and not None
            if not isinstance(data_u, dict) or not isinstance(data_v, dict):
                continue
            for pred, fn in PREDICATES.items():
                try:
                    # Pass node data (data_u, data_v) to predicate function
                    # If predicate function expects only a, b, pass them
                    # If it expects params, it should be handled in the registry
                    if fn(data_u, data_v):
                        G.add_edge(u, v, predicate=pred, source='spatial')
                except Exception as e:
                    logging.debug(f"Predicate function '{pred}' failed for ({u}, {v}): {e}")
                    continue

    # Add global graph features
    G.graph['node_count'] = G.number_of_nodes()
    G.graph['edge_count'] = G.number_of_edges()
    if G.number_of_nodes() > 0:
        G.graph['avg_degree'] = np.mean([d for _,d in G.degree()])
    else:
        G.graph['avg_degree'] = 0.0
    # Use robust clustering coefficient calculation for MultiDiGraph
    try:
        if G.number_of_nodes() > 1:
            G.graph['clustering_coeff'] = compute_clustering_coefficient_multidigraph(G)
        else:
            G.graph['clustering_coeff'] = 0.0
    except Exception as e:
        logging.warning(f"Could not compute clustering coefficient: {e}. Setting to 0.0")
        G.graph['clustering_coeff'] = 0.0
def compute_clustering_coefficient_multidigraph(G):
    """Convert MultiDiGraph to simple Graph (undirected, no parallel edges) and compute clustering coefficient."""
    simple_G = nx.Graph(G)
    return nx.average_clustering(simple_G)

    # New global graph features
    dists = []
    if len(G.nodes) > 1:
        for u_id, v_id in combinations(G.nodes(), 2):
            u_data = G.nodes[u_id]
            v_data = G.nodes[v_id]
            if 'cx' in u_data and 'cy' in u_data and 'cx' in v_data and 'cy' in v_data:
                dists.append(math.hypot(u_data['cx'] - v_data['cx'], u_data['cy'] - v_data['cy']))
    
    avg_area_ratio = 0.0
    if G.number_of_nodes() > 0:
        all_areas = [n.get('area', 0) for _, n in G.nodes(data=True)]
        mean_area = np.mean(all_areas) if all_areas else EPS
        avg_area_ratio = np.mean([n.get('area', 0) / (mean_area + EPS) for _, n in G.nodes(data=True)])

    edge_diversity = 0.0
    if G.number_of_edges() > 0:
        edge_diversity = len({d['predicate'] for _,_,d in G.edges(data=True)}) / (G.number_of_nodes()**2) # Using N^2 as maximum possible edges

    motif_coverage = 0.0
    if G.number_of_nodes() > 0:
        motif_coverage = len([n for n in G.nodes if '_part' in n]) / len(G.nodes)

    G.graph.update({
        'avg_area_ratio': float(avg_area_ratio),
        'std_centroid_dist': float(np.std(dists)) if dists else 0.0,
        'edge_diversity': float(edge_diversity),
        'motif_coverage': float(motif_coverage)
    })


def add_commonsense_edges(G):
    """Adds semantic edges by querying the commonsense knowledge base."""
    if not kb:
        return
    nodes_with_data = list(G.nodes(data=True))
    for u, data_u in nodes_with_data:
        label = data_u.get('shape_label')
        if not label:
            continue
        try:
            # Query the KB for related concepts
            # Ensure the kb.related method exists and returns (relation, concept) tuples
            related_concepts = kb.related(label) if hasattr(kb, 'related') else []
            for rel, other_concept in related_concepts[:TOP_K]:
                for v, data_v in nodes_with_data:
                    if u != v and data_v.get('shape_label') == other_concept:
                        G.add_edge(u, v, predicate=rel, source='kb')
                        break
        except Exception as e:
            logging.warning(f"Commonsense KB query failed for label '{label}': {e}")


# --- Physics Attribute Computation ---
def compute_basic_features(poly: Polygon) -> Dict[str, Any]:
    """Computes basic geometric features for a shapely Polygon."""
    if not poly or not poly.is_valid:
        return {'area': 0.0, 'perimeter': 0.0, 'cx': 0.0, 'cy': 0.0, 'bbox': [0,0,0,0], 'aspect_ratio': 1.0, 'orientation': 0.0}

    area = poly.area
    perimeter = poly.length
    centroid = poly.centroid
    minx, miny, maxx, maxy = poly.bounds
    width, height = maxx - minx, maxy - miny
    aspect_ratio = width / height if height > 0 else 1.0

    # Orientation via PCA of vertices
    orientation = 0.0
    try:
        if len(poly.exterior.coords) >= 2:
            coords = np.array(poly.exterior.coords)
            if coords.shape[0] > 1 and coords.shape[1] == 2:
                # Ensure at least 2 points for PCA
                if coords.shape[0] > 1:
                    cov_matrix = np.cov(coords.T)
                    if cov_matrix.shape == (2, 2): # Ensure it's a 2x2 matrix
                        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                        # The first principal component (eigenvector corresponding to largest eigenvalue)
                        # gives the direction of the major axis.
                        major_axis_idx = np.argmax(eigenvalues)
                        principal_axis = eigenvectors[:, major_axis_idx]
                        orientation = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))) % 360
    except Exception as e:
        logging.debug(f"Could not compute orientation for polygon: {e}")
        orientation = 0.0 # Default to 0 on error

    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'cx': float(centroid.x),
        'cy': float(centroid.y),
        'bbox': [float(minx), float(miny), float(maxx), float(maxy)],
        'aspect_ratio': float(aspect_ratio),
        'orientation': float(orientation)
    }

def compute_physics_attributes(node_data):
    """
    Computes robust physics attributes and asserts their domain validity.
    Now also computes perimeter and compactness.
    """
    if not isinstance(node_data, dict):
        logging.error(f"compute_physics_attributes: node_data is not a dict: type={type(node_data)}, value={repr(node_data)}")
        # Initialize with default values to prevent further errors
        node_data.update({'area': 0.0, 'inertia': 0.0, 'convexity': 0.0, 'cx': 0.0, 'cy': 0.0, 'bbox': [0,0,0,0], 'aspect_ratio': 1.0, 'perimeter': 0.0, 'orientation': 0.0, 'compactness': 0.0, 'num_segments': 0, 'num_junctions': 0})
        return

    vertices = node_data.get('vertices')
    if not vertices or len(vertices) < 3:
        logging.warning(f"compute_physics_attributes: Invalid or missing vertices for node {node_data.get('id', 'unknown')}. Initializing with default physics attributes.")
        node_data.update({'area': 0.0, 'inertia': 0.0, 'convexity': 0.0, 'cx': 0.0, 'cy': 0.0, 'bbox': [0,0,0,0], 'aspect_ratio': 1.0, 'perimeter': 0.0, 'orientation': 0.0, 'compactness': 0.0, 'num_segments': 0, 'num_junctions': 0})
        return

    poly = Polygon(vertices)
    
    # Compute basic features first
    basic_features = compute_basic_features(poly)
    node_data.update(basic_features)

    area = node_data['area'] # Use the computed area

    assert area >= 0.0, f"Negative area {area} for node {node_data.get('id')}"

    try:
        # Shapely's moment_of_inertia is about the centroid
        inertia = poly.moment_of_inertia
    except Exception:
        inertia = 0.0 # Default to 0 on error

    convexity = poly.convex_hull.area / area if area > 0 else 0.0

    # New attributes: compactness, num_segments, num_junctions
    perimeter = node_data['perimeter'] # Use computed perimeter
    compactness = perimeter**2 / (4 * math.pi * area) if area > 0 else 0.0

    action_program = node_data.get('action_program', [])
    num_segments = len(extract_line_segments(action_program))
    
    # Simple way to count junctions (points that appear more than once in exterior coords)
    num_junctions = 0
    if poly.exterior:
        coords_counter = Counter(tuple(c) for c in poly.exterior.coords)
        num_junctions = sum(1 for count in coords_counter.values() if count > 1)

    node_data.update({
        'inertia': float(inertia),
        'convexity': float(convexity),
        'compactness': float(compactness),
        'num_segments': int(num_segments),
        'num_junctions': int(num_junctions),
    })

# --- Graph Building Functions with/without Validation ---
# @graph_validate is commented out as GRAPHTYPE_AVAILABLE is False in the logs
# and it causes issues if not properly set up.
# @graph_validate(
#     NodeData(id=str, area=float, inertia=float, convexity=float, cx=float, cy=float, bbox=list, aspect_ratio=float, perimeter=float, orientation=float, compactness=float, num_segments=int, num_junctions=int), # Updated NodeData schema
#     EdgeData(predicate=str, source=str),
# )
def build_graph_unvalidated(record):
    """Builds a single scene graph without runtime schema validation."""
    G = nx.MultiDiGraph()
    objects = record.get('objects', [])
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            logging.error(f"build_graph_unvalidated: Skipping non-dict object at index {idx}: type={type(obj)}, value={repr(obj)}")
            continue
        try:
            compute_physics_attributes(obj)
        except Exception as e:
            import traceback
            logging.error(f"build_graph_unvalidated: Exception in compute_physics_attributes for object at index {idx}: {e}\n{traceback.format_exc()}")
            continue
        try:
            G.add_node(obj['id'], **obj)
        except Exception as e:
            import traceback
            logging.error(f"build_graph_unvalidated: Exception adding node at index {idx}: {e}\n{traceback.format_exc()}")
            continue
    add_predicate_edges(G)
    add_commonsense_edges(G)
    # logging.warning("Could not compute clustering coefficient: not implemented for multigraph type. Setting to 0.0")
    return G

# --- Diversity KPI Logger ---
def compute_pc_error(graphs):
    """Computes Physical Consistency error (e.g., overlapping solids)."""
    error_count = 0
    total_graphs = len(graphs)
    if total_graphs == 0: return 0.0

    for G_data in graphs: # G_data is now a dict containing 'scene_graph' and other info
        # Extract the NetworkX graph from the scene_graph dict
        G = G_data.get('scene_graph', {}).get('graph')
        if not G:
            logging.warning(f"No NetworkX graph found for problem {G_data.get('problem_id')}. Skipping PC error calculation.")
            continue

        nodes = list(G.nodes(data=True))
        polygons = []
        for _, data in nodes:
            if data.get('vertices') and len(data['vertices']) >= 3:
                try:
                    polygons.append(Polygon(data['vertices']))
                except Exception as e:
                    logging.warning(f"Invalid polygon vertices for node {data.get('id')}: {e}")
                    continue

        has_overlap = False
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                try:
                    # Check for actual intersection, not just touching
                    if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                        has_overlap = True
                        break
                except Exception as e:
                    logging.warning(f"Error checking polygon overlap: {e}")
            if has_overlap:
                break
        if has_overlap:
            error_count += 1

    return error_count / total_graphs

def log_diversity_metrics(graphs, out_path='logs/graph_diversity.jsonl'):
    """Computes and logs coverage, entropy, and physical consistency."""
    if not graphs: return

    # Collect unique triples (subject, predicate, object) from problem-level graphs
    unique_triples = set()
    predicate_counts = Counter()
    total_possible_triples = 0

    for G_data in graphs:
        G = G_data.get('scene_graph', {}).get('graph')
        if not G:
            continue

        # Only count for graphs with more than one node to avoid division by zero
        if len(G.nodes) > 1:
            total_possible_triples += len(G.nodes) * (len(G.nodes) - 1) # N*(N-1) for directed pairs

        for u, v, data in G.edges(data=True):
            unique_triples.add((u, data.get('predicate', 'unknown'), v))
            predicate_counts[data.get('predicate', 'unknown')] += 1

    C = len(unique_triples) / total_possible_triples if total_possible_triples > 0 else 0
    H = stats.entropy(list(predicate_counts.values()), base=2) if predicate_counts else 0
    pc_error = compute_pc_error(graphs) # This function already updated to handle new graph structure

    metrics = {'coverage': C, 'entropy': H, 'pc_error': pc_error}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a') as f:
        f.write(json.dumps(metrics) + "\n")

    logging.info(f"Diversity Metrics: Coverage={C:.3f}, Entropy={H:.3f} bits, PC-Error={pc_error:.3f}")

# --- Caching and Profiling Utilities ---
def compute_hash(input_paths, params_dict):
    """Computes a hash based on input files and parameters for cache validation."""
    h = hashlib.sha256()
    for path in input_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                h.update(f.read())
        else:
            logging.warning(f"Input file for hash computation not found: {path}. Hash will be incomplete.")
    for k in sorted(params_dict.keys()):
        h.update(str(k).encode())
        h.update(str(params_dict[k]).encode())
    return h.hexdigest()

def cache_valid(cache_path, hash_path, current_hash):
    """Checks if the cache is valid based on stored hash."""
    if not os.path.exists(cache_path) or not os.path.exists(hash_path):
        return False
    with open(hash_path, 'r') as f:
        cached_hash = f.read().strip()
    return cached_hash == current_hash

def save_hash(hash_path, hash_val):
    """Saves the current hash to a file."""
    with open(hash_path, 'w') as f:
        f.write(hash_val)

def profile_optimal_batch_size(records, build_func, args):
    """Profiles different batch sizes to find the one with optimal graph build throughput."""
    candidate_sizes = [8, 16, 32, 64, 128]
    best_size = candidate_sizes[0]
    best_throughput = 0
    logging.info("[INFO] Profiling batch sizes for optimal graph build throughput...")
    for size in candidate_sizes:
        start = time.time()
        try:
            # For profiling, we need to simulate the problem-level merging
            # Take a slice of records, group them by problem_id, then build graphs
            # This is a simplified profiling, as it doesn't run the full async enhanced builder
            sample_records = records[:size * 12] # Assuming 12 images per problem for a rough estimate
            if not sample_records:
                logging.warning(f"Not enough records to profile batch size {size}. Skipping.")
                continue

            # Group sample records by problem_id
            grouped_sample = defaultdict(list)
            for rec in sample_records:
                pid = rec.get('problem_id')
                if pid:
                    grouped_sample[pid].append(rec)

            temp_graphs = []
            for pid, problem_recs in grouped_sample.items():
                merged_rec = {'objects': []}
                temp_objects_for_clustering_profiling = [] # Separate list for profiling
                for idx, rec in enumerate(problem_recs):
                    # Normalize action_program in rec
                    raw_ap = rec.get('action_program', [])
                    normalized_ap = []
                    for cmd in raw_ap:
                        parsed = parse_action_command(cmd)
                        if parsed:
                            normalized_ap.append(parsed)
                    rec['action_program'] = normalized_ap

                    # If rec contains 'objects' key and it's a list, flatten and process each
                    if 'objects' in rec and isinstance(rec['objects'], list):
                        for obj_idx, obj in enumerate(rec['objects']):
                            # Normalize action_program in obj
                            raw_obj_ap = obj.get('action_program', [])
                            normalized_obj_ap = []
                            for cmd in raw_obj_ap:
                                parsed = parse_action_command(cmd)
                                if parsed:
                                    normalized_obj_ap.append(parsed)
                            obj['action_program'] = normalized_obj_ap
                            obj_data = {
                                'id': obj.get('id', f"{problem_id}_{idx}_{obj_idx}"),
                                'vertices': obj.get('vertices', []),
                                **obj,
                                'label': obj.get('label', rec.get('label', '')),
                                'shape_label': obj.get('shape_label', obj.get('label', rec.get('label', ''))),
                                'category': obj.get('category', rec.get('category', '')),
                                'original_image_path': remap_path(rec.get('image_path', '')),
                                'original_record_idx': idx,
                                'action_program': obj['action_program']
                            }
                            # Compute additional node attributes
                            compute_physics_attributes(obj_data) # This will add cx, cy, bbox, aspect_ratio, perimeter, orientation, compactness, num_segments, num_junctions

                            # Optionally decompose complex polygons during profiling
                            if args.decompose_polygons and obj_data.get('action_program'):
                                segments = extract_line_segments(obj_data['action_program'])
                                if segments:
                                    try:
                                        from shapely.geometry import MultiLineString
                                        multi_line = MultiLineString(segments)
                                        unary_union_result = unary_union(multi_line)
                                        sub_polys = list(polygonize(unary_union_result))

                                        if len(sub_polys) > 1:
                                            for i, sub in enumerate(sub_polys):
                                                new_obj_id = f"{obj_data['id']}_part{i}"
                                                new_obj_data = {
                                                    'id': new_obj_id,
                                                    'vertices': list(sub.exterior.coords) if sub.exterior else [],
                                                    'shape_label': obj_data.get('shape_label', 'part'),
                                                    'category': obj_data.get('category', 'part'),
                                                    'parent_id': obj_data['id']
                                                }
                                                compute_physics_attributes(new_obj_data)
                                                temp_objects_for_clustering_profiling.append(new_obj_data)
                                        else:
                                            temp_objects_for_clustering_profiling.append(obj_data)
                                    except Exception as e:
                                        logging.warning(f"Failed to decompose object {obj_data['id']} during profiling: {e}. Adding original object.")
                                        temp_objects_for_clustering_profiling.append(obj_data)
                                else:
                                    temp_objects_for_clustering_profiling.append(obj_data)
                            else:
                                temp_objects_for_clustering_profiling.append(obj_data)

                # Assign cluster labels for profiling purposes
                if temp_objects_for_clustering_profiling and len(temp_objects_for_clustering_profiling) > 1:
                    coords = np.array([[o['cx'], o['cy']] for o in temp_objects_for_clustering_profiling if 'cx' in o and 'cy' in o])
                    if len(coords) > 1:
                        k = min(3, len(coords))
                        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                        labels = kmeans.fit_predict(coords)
                        valid_coords_idx = 0
                        for obj in temp_objects_for_clustering_profiling:
                            if 'cx' in obj and 'cy' in obj:
                                obj['cluster_label'] = int(labels[valid_coords_idx])
                                valid_coords_idx += 1
                            else:
                                obj['cluster_label'] = -1
                    else:
                        for obj in temp_objects_for_clustering_profiling:
                            obj['cluster_label'] = 0
                elif temp_objects_for_clustering_profiling:
                    temp_objects_for_clustering_profiling[0]['cluster_label'] = 0
                
                merged_rec['objects'] = temp_objects_for_clustering_profiling # Assign to merged_rec

                if merged_rec['objects']:
                    G = build_func(merged_rec)
                    temp_graphs.append(G)

            elapsed = time.time() - start
            throughput = len(temp_graphs) / elapsed if elapsed > 0 else 0 # Throughput in problems/sec
            logging.info(f"Batch size {size} (approx {len(temp_graphs)} problems): {throughput:.2f} problems/sec")
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size
        except Exception as e:
            logging.error(f"Batch size {size} failed during profiling: {e}")
    logging.info(f"[INFO] Selected optimal batch size (problems): {best_size}")
    return best_size


# --- Enhanced Scene Graph Builder Class ---

# --- Singleton for RealFeatureExtractor ---
_REAL_FEATURE_EXTRACTOR_INSTANCE = None
def get_real_feature_extractor():
    global _REAL_FEATURE_EXTRACTOR_INSTANCE
    if _REAL_FEATURE_EXTRACTOR_INSTANCE is not None:
        return _REAL_FEATURE_EXTRACTOR_INSTANCE
    if not TORCH_KORNIA_AVAILABLE:
        logging.warning("Torch/Kornia/Torchvision not available. RealFeatureExtractor will not be initialized.")
        return None
    try:
        _REAL_FEATURE_EXTRACTOR_INSTANCE = RealFeatureExtractor(
            clip_model_name="openai/clip-vit-base-patch32",
            sam_encoder_path="sam_checkpoints/sam_vit_h_4b8939.pth",
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_features=True
        )
        logging.info("RealFeatureExtractor singleton initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize RealFeatureExtractor: {e}. Feature extraction will be skipped.")
        _REAL_FEATURE_EXTRACTOR_INSTANCE = None
    return _REAL_FEATURE_EXTRACTOR_INSTANCE


class EnhancedSceneGraphBuilder:
    def __init__(self):
        self.knowledge_fusion = MultiSourceKnowledgeFusion()
        self.sgcore_validator = SGScoreValidator()
        self.hierarchical_predictor = HierarchicalPredicatePredictor()
        self.feature_extractor = get_real_feature_extractor()

    # --- Caching for mask creation ---
    _mask_cache = {}

    def _create_mask_from_vertices(self, vertices, image_shape):
        key = (tuple(map(tuple, vertices)), image_shape)
        if key in self._mask_cache:
            return self._mask_cache[key]
        mask = np.zeros(image_shape, dtype=np.uint8)
        vertices_array = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [vertices_array], 255)
        self._mask_cache[key] = mask
        return mask

    def _create_mask_from_bbox(self, bbox, image_shape):
        key = (tuple(bbox), image_shape)
        if key in self._mask_cache:
            return self._mask_cache[key]
        mask = np.zeros(image_shape, dtype=np.uint8)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(image_shape[0], y2), min(image_shape[1], x2)
        mask[y1:y2, x1:x2] = 255
        self._mask_cache[key] = mask
        return mask

    async def build_enhanced_scene_graph(self, image_path: str, base_scene_graph: dict) -> dict:
        """
        Builds an enhanced scene graph by adding real features, knowledge fusion,
        and hierarchical predicate refinement.
        The base_scene_graph now contains a NetworkX graph for the entire problem.
        """
        # --- Fast image loading and caching ---
        image_np = None
        if not os.path.exists(image_path):
            logging.error(f"Representative image file not found: {image_path}. Skipping enhanced graph build.")
            return {'scene_graph': base_scene_graph, 'quality_metrics': {}}

        try:
            image_np = cv2.imread(image_path)
            if image_np is None:
                raise ValueError("cv2.imread returned None")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"Failed to load or process representative image {image_path} with OpenCV: {e}. Trying PIL fallback.")
            try:
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
            except Exception as e2:
                logging.error(f"Failed to load image with PIL fallback: {e2}. Skipping enhanced graph build.")
                return {'scene_graph': base_scene_graph, 'quality_metrics': {}}
        # Feature extraction for each node in the graph
        G = base_scene_graph.get('graph')
        object_features = {}
        if G:
            for node_id, node_data in list(G.nodes(data=True)):
                mask = None
                if 'vertices' in node_data and len(node_data['vertices']) >= 3:
                    mask = self._create_mask_from_vertices(node_data['vertices'], image_np.shape[:2])
                elif 'bbox' in node_data and len(node_data['bbox']) == 4:
                    mask = self._create_mask_from_bbox(node_data['bbox'], image_np.shape[:2])
                else:
                    logging.warning(f"Node {node_id} missing valid vertices or bbox. Skipping feature extraction for this node.")
                    continue

                if self.feature_extractor and mask is not None:
                    try:
                        features = self.feature_extractor.extract_object_features(image_np, mask, node_id)
                        object_features[node_id] = features
                        G.nodes[node_id]['real_features'] = features # Store raw features
                        logging.debug(f"Extracted features for node {node_id}: shape {features.shape}")
                    except Exception as e:
                        import traceback
                        logging.error(f"Feature extraction failed for node {node_id}: {e}. No features for this node.")
                        logging.error(traceback.format_exc())
                else:
                    logging.debug(f"Feature extractor not available or mask is None for node {node_id}. Skipping feature extraction.")
        
        # Knowledge fusion and Hierarchical predicate refinement now operate on the problem-level graph
        # This part needs to iterate through existing edges and potentially add new ones based on features
        # The original code's `enhanced_relationships` and `refined_relationships` are for a list of relationships.
        # We need to apply this to the edges of the NetworkX graph G.

        # Step 1: Knowledge Fusion (semantic enrichment of existing edges)
        # Collect existing relationships to process
        current_relationships = []
        for u, v, data in G.edges(data=True):
            current_relationships.append({
                'subject_id': u,
                'object_id': v,
                'predicate': data.get('predicate', 'unknown'),
                'source': data.get('source', 'spatial')
            })

        enhanced_relationships = []
        for rel in current_relationships:
            subject_id = rel.get('subject_id')
            object_id = rel.get('object_id')
            
            # Use shape_label for knowledge fusion if available, otherwise node ID
            subject_label = G.nodes[subject_id].get('shape_label', subject_id)
            object_label = G.nodes[object_id].get('shape_label', object_id)

            if subject_id in object_features and object_id in object_features:
                try:
                    # Assuming knowledge_fusion can take labels and return enriched relations
                    enriched_rels = await self.knowledge_fusion.get_enriched_relationships(
                        subject_label, object_label, [rel.get('predicate', '')]
                    )
                    # Add enriched relations back to the graph or a temporary list
                    for erel in enriched_rels:
                        # Ensure enriched relations have subject_id and object_id for graph
                        erel['subject_id'] = subject_id
                        erel['object_id'] = object_id
                        enhanced_relationships.append(erel)
                except Exception as e:
                    logging.warning(f"Knowledge fusion failed for relationship {rel}: {e}. Appending original relationship.")
                    enhanced_relationships.append(rel)
            else:
                logging.debug(f"Skipping knowledge fusion for relationship {rel} due to missing features.")
                enhanced_relationships.append(rel) # Append original if features are missing

        # Step 2: Hierarchical predicate refinement
        refined_relationships = []
        for rel in enhanced_relationships:
            subject_id = rel.get('subject_id')
            object_id = rel.get('object_id')
            
            subject_features = object_features.get(subject_id)
            object_features_ = object_features.get(object_id)
            knowledge_embeddings = None # Placeholder, as per original code

            if subject_features is not None and object_features_ is not None:
                try:
                    # Assuming predict_with_bayesian_inference returns a list of refined relations
                    refined = self.hierarchical_predictor.predict_with_bayesian_inference(
                        subject_features, object_features_, knowledge_embeddings,
                        current_predicates=[rel.get('predicate')] # Pass current predicate for context
                    )
                    # Add refined relations back to the list
                    for r_rel in refined:
                        r_rel['subject_id'] = subject_id
                        r_rel['object_id'] = object_id
                        refined_relationships.append(r_rel)
                except Exception as e:
                    logging.warning(f"Hierarchical predicate prediction failed for rel {rel}: {e}. Appending original.")
                    refined_relationships.append(rel)
            else:
                logging.debug(f"Skipping hierarchical prediction for rel {rel} due to missing features.")
                refined_relationships.append(rel)

        # Final validation using SGScoreValidator
        # Prepare objects list for SGScoreValidator (from NetworkX nodes)
        objects_for_validator = [G.nodes[node_id] for node_id in G.nodes()]

        # Only call validator if image_np is loaded
        if image_np is not None:
            validation_results = await self.sgcore_validator.validate_scene_graph(
                image_np, {'objects': objects_for_validator, 'relationships': refined_relationships}
            )
        else:
            logging.warning(f"No image loaded for validation at {image_path}. Skipping validation.")
            validation_results = {'overall_score': 0.0, 'relationship_accuracy_score': 0.0}

        final_scene_graph = base_scene_graph.copy()
        # Update the 'relationships' list in the output dict with the refined ones
        final_scene_graph['relationships'] = refined_relationships
        # The NetworkX graph G itself is passed by reference and modified with 'real_features'

        return {
            'scene_graph': final_scene_graph, # Contains the NetworkX graph and refined relationships
            'quality_metrics': {
                'knowledge_confidence': float(np.mean([r.get('final_confidence', 1.0) for r in enhanced_relationships]) if enhanced_relationships else 1.0),
                'validation_score': validation_results.get('overall_score', 0.0),
                'relationship_accuracy': validation_results.get('relationship_accuracy_score', 0.0)
            }
        }

# Utility to parse action_program commands into dicts
from typing import Any, Optional, Dict

def parse_action_command(cmd: Any) -> Optional[Dict[str,Any]]:
    if isinstance(cmd, dict):
        return cmd
    if not isinstance(cmd, str):
        return None
    parts = cmd.split('_', 2)
    if len(parts) < 3:
        return None
    shape, mode, rest = parts  # e.g. ["line","normal","0.200-0.667"]
    if shape == "line" and '-' in rest:
        a, b = rest.split('-',1)
        try:
            return {'type':'line', 'mode':mode, 'x':float(a), 'y':float(b)}
        except Exception:
            return None
    if shape == "start" and '-' in rest:
        a, b = rest.split('-',1)
        try:
            return {'type':'start', 'x':float(a), 'y':float(b)}
        except Exception:
            return None
    # Extend for 'arc', 'curve', etc. as needed
    return None

# --- Main Functions for Data Loading and Processing ---
def load_data(input_path):
    """Simple pickle/json loader for demonstration; replace as needed."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported input file type: {input_path}")

def mask_quality_stats(image, mask):
    """Calculates mask quality statistics like Edge IoU, precision, and recall."""
    if image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    mask_bin = (mask > 0).astype(np.uint8)
    edges_img = cv2.Canny(image_gray.astype(np.uint8), 50, 150)
    edges_mask = cv2.Canny(mask.astype(np.uint8), 50, 150)
    edges_img_flat = (edges_img > 0).flatten()
    edges_mask_flat = (edges_mask > 0).flatten()
    intersection = np.logical_and(edges_img_flat, edges_mask_flat).sum()
    union = np.logical_or(edges_img_flat, edges_mask_flat).sum()
    edge_iou = intersection / (union + 1e-6) if union > 0 else 1.0
    precision = intersection / (edges_mask_flat.sum() + 1e-6) if edges_mask_flat.sum() > 0 else 1.0
    recall = intersection / (edges_img_flat.sum() + 1e-6) if edges_img_flat.sum() > 0 else 1.0
    clinically_acceptable = (edge_iou > 0.5) or (precision > 0.7 and recall > 0.7)
    stats = {
        'edge_iou': edge_iou,
        'edge_precision': precision,
        'edge_recall': recall,
        'clinically_acceptable': clinically_acceptable
    }
    return stats

def save_feedback_images(image, mask, base_name, feedback_dir, scene_graph=None):
    """Saves input image, mask, scene graph visualization, and side-by-side comparisons for feedback."""
    logging.info(f"[save_feedback_images] Called with base_name={base_name}, image type={type(image)}, mask type={type(mask)}, feedback_dir={feedback_dir}, scene_graph type={type(scene_graph)}")
    import matplotlib.pyplot as plt
    os.makedirs(feedback_dir, exist_ok=True)
    img_save_path = os.path.join(feedback_dir, f"{base_name}_input.png")
    mask_save_path = os.path.join(feedback_dir, f"{base_name}_mask.png")
    side_by_side_path = os.path.join(feedback_dir, f"{base_name}_side_by_side.png")
    graph_img_path = os.path.join(feedback_dir, f"{base_name}_graph.png")
    img_graph_side_by_side_path = os.path.join(feedback_dir, f"{base_name}_img_graph.png")
    actmap_path = os.path.join(feedback_dir, f"{base_name}_actmap.png") # For visualization of edges on image

    # Save real image
    if image.ndim == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image
    cv2.imwrite(img_save_path, img_bgr)

    # Save mask if provided
    if mask is not None:
        cv2.imwrite(mask_save_path, mask)
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if img_bgr.shape[:2] != mask_color.shape[:2]:
            mask_color = cv2.resize(mask_color, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        side_by_side = cv2.hconcat([img_bgr, mask_color])
        cv2.imwrite(side_by_side_path, side_by_side)

        # Save scene graph visualization if provided
    if scene_graph is not None and 'graph' in scene_graph:
        G = scene_graph['graph']
        # Robustly check if G is a valid NetworkX graph
        is_nx_graph = hasattr(G, 'nodes') and hasattr(G, 'edges')
        if not is_nx_graph:
            logging.error(f"Scene graph for {base_name} is not a valid NetworkX graph object: {type(G)}. Attempting to visualize anyway.")
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            plt.figure(figsize=(5, 5))
            if is_nx_graph and len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else None
                # Node labels: use shape_label if present, else id
                node_labels = {n: (G.nodes[n].get('shape_label', n)) for n in G.nodes()}
                nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
                nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.7)
                nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')
                # Edge labels: use 'predicate' if present
                edge_labels = {(u, v): d.get('predicate', '') for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=9)
                plt.title(f"Scene Graph: {base_name}")
                plt.axis('off')
            else:
                plt.text(0.5, 0.5, 'No graph', ha='center', va='center', fontsize=12)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(graph_img_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.error(f"Failed to save scene graph visualization for {base_name}: {e}")

        # Always attempt to save side-by-side image and graph, even if graph is empty
        try:
            graph_img = cv2.imread(graph_img_path)
            if graph_img is not None:
                # Resize graph image to match input image height
                if graph_img.shape[0] != img_bgr.shape[0]:
                    scale = img_bgr.shape[0] / graph_img.shape[0]
                    new_w = int(graph_img.shape[1] * scale)
                    graph_img = cv2.resize(graph_img, (new_w, img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
                img_graph_side_by_side = cv2.hconcat([img_bgr, graph_img])
                cv2.imwrite(img_graph_side_by_side_path, img_graph_side_by_side)
            else:
                logging.warning(f"Graph image not found for {base_name}, skipping side-by-side save.")
        except Exception as e:
            logging.warning(f"Failed to create side-by-side image and graph for {base_name}: {e}")

        # Visualization of edges on image (Activation Map style)
        try:
            from PIL import Image, ImageDraw
            pil_image = Image.open(os.path.join(feedback_dir, f"{base_name}_input.png")).convert('RGBA')
            draw = ImageDraw.Draw(pil_image)
            if is_nx_graph and len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else {n: (0.5, 0.5) for n in G.nodes()}
                # Draw edges as lines
                width, height = pil_image.size
                for u, v in G.edges():
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    draw.line([
                        int(x1 * width), int(y1 * height),
                        int(x2 * width), int(y2 * height)
                    ], fill=(255, 0, 0, 128), width=2)
                # Draw nodes as circles
                for n in G.nodes():
                    x, y = pos[n]
                    r = 8
                    draw.ellipse([
                        int(x * width) - r, int(y * height) - r,
                        int(x * width) + r, int(y * height) + r
                    ], fill=(0, 255, 0, 128))
            pil_image.save(actmap_path)
        except Exception as e:
            logging.warning(f"Failed to create activation map image for {base_name}: {e}")


    # ...existing code...
def remap_path(path):
    """Remaps image paths to match expected dataset structure."""
    return path.replace('category_1', '1').replace('category_0', '0')

def induce_predicate_for_problem(objects: List[Dict[str, Any]], adaptive_thresholds: AdaptivePredicateThresholds, method: str = 'auto') -> Tuple[str, Any]:
    """
    Induces the best predicate for a given problem using the selected method.
    method: 'auto', 'automl', 'statistical', 'decision_tree'.
    Fallbacks are used if a method fails.
    """
    if not objects or len(objects) < 2:
        return "same_shape", None

    # GNN-based predicate induction logic
    def induce_predicate_gnn(objects):
        try:
            from scripts.train_relation_gnn import train_and_extract_predicates_from_gnn
            # This function should train the GNN and return the most discriminative predicate(s)
            pred, params = train_and_extract_predicates_from_gnn(objects)
            if pred:
                return pred, params
        except Exception as e:
            logging.warning(f"GNN predicate induction failed: {e}")
        return "same_shape", None

    # Try the requested method, with fallbacks
    if method == 'gnn':
        pred, params = induce_predicate_gnn(objects)
        if pred != "same_shape":
            return pred, params
        # fallback
        pred, params = induce_predicate_automl(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_statistical(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_decision_tree(objects)
        return pred, params
    elif method == 'automl':
        pred, params = induce_predicate_automl(objects)
        if pred != "same_shape":
            return pred, params
        # fallback
        pred, params = induce_predicate_gnn(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_statistical(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_decision_tree(objects)
        return pred, params
    elif method == 'statistical':
        pred, params = induce_predicate_statistical(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_gnn(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_decision_tree(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_automl(objects)
        return pred, params
    elif method == 'decision_tree':
        pred, params = induce_predicate_decision_tree(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_gnn(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_statistical(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_automl(objects)
        return pred, params
    else:  # 'auto' or unknown
        # Try all, in order: gnn, automl, statistical, decision_tree
        pred, params = induce_predicate_gnn(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_automl(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_statistical(objects)
        if pred != "same_shape":
            return pred, params
        pred, params = induce_predicate_decision_tree(objects)
        return pred, params
    """
    A simplified placeholder for automated predicate induction via contrastive analysis.
    This function should identify a predicate that best separates positive from negative examples.
    For demonstration, it will try to find a simple rule or fallback.
    """
    # This is a very basic example. A real inducer would be much more complex.
    # It would iterate through predicate templates and parameter values,
    # and evaluate their ability to discriminate positive vs. negative sets.

    # Assume 'category' '1' is positive, '0' is negative for this induction step
    pos_objects = [o for o in objects if o.get('category') == '1']
    neg_objects = [o for o in objects if o.get('category') == '0']

    # Get current adaptive parameters
    # Use get_learned_thresholds() for compatibility
    if hasattr(adaptive_thresholds, 'get_learned_thresholds'):
        params = adaptive_thresholds.get_learned_thresholds()
    elif hasattr(adaptive_thresholds, 'thresholds'):
        params = adaptive_thresholds.thresholds
    else:
        params = {}
    canvas_width = params.get('canvas_width', 128)
    canvas_height = params.get('canvas_height', 128)

    # Example: Try to induce 'larger_than'
    if pos_objects and neg_objects:
        # Simple heuristic: find an alpha where positive objects are generally larger than negative ones
        # This is highly simplified and for illustration only.
        
        # Check 'larger_than'
        larger_than_alpha_candidates = np.linspace(0.8, 1.5, 10)
        for alpha in larger_than_alpha_candidates:
            # Check if positives are generally larger than other positives (vacuously true or not meaningful)
            # and if positives are larger than negatives
            pos_vs_neg_holds = all(any(o1.get('area',0) > alpha * o2.get('area',0) for o2 in neg_objects) for o1 in pos_objects)
            neg_vs_pos_fails = all(not any(o1.get('area',0) > alpha * o2.get('area',0) for o2 in pos_objects) for o1 in neg_objects)

            if pos_vs_neg_holds and neg_vs_pos_fails:
                logging.info(f"Induced 'larger_than' with alpha={alpha:.2f}")
                adaptive_thresholds.update_threshold('larger_than_alpha', alpha)
                return 'larger_than', alpha

        # Check 'near'
        near_threshold_candidates = np.linspace(20, 80, 10)
        for threshold in near_threshold_candidates:
            # Check if positives are near each other, and negatives are not near positives
            pos_near_pos = all(any(math.hypot(o1.get('cx',0)-o2.get('cx',0), o1.get('cy',0)-o2.get('cy',0)) < threshold for o2 in pos_objects if o1 != o2) for o1 in pos_objects)
            neg_not_near_pos = all(not any(math.hypot(o1.get('cx',0)-o2.get('cx',0), o1.get('cy',0)-o2.get('cy',0)) < threshold for o2 in pos_objects) for o1 in neg_objects)
            
            if pos_near_pos and neg_not_near_pos:
                logging.info(f"Induced 'near' with threshold={threshold:.1f}")
                adaptive_thresholds.update_threshold('near_threshold', threshold)
                return 'near', threshold

    logging.info("Could not induce a specific predicate. Falling back to 'same_shape'.")
    return 'same_shape', None # Fallback


async def _process_single_problem(problem_id: str, problem_records: List[Dict[str, Any]], feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate, decompose_polygons: bool, adaptive_thresholds: AdaptivePredicateThresholds, args=None):
    logging.info(f"Starting _process_single_problem for {problem_id}. problem_records type: {type(problem_records)}")
    # Do NOT split by category; merge all records for this problem
    merged_record = {'objects': []}
    representative_image_path = None
    for idx, rec in enumerate(problem_records):
        if representative_image_path is None:
            representative_image_path = remap_path(rec.get('image_path', rec.get('mask_path', '')))
        # Normalize action_program in rec
        raw_ap = rec.get('action_program', [])
        normalized_ap = []
        for cmd in raw_ap:
            parsed = parse_action_command(cmd)
            if parsed:
                normalized_ap.append(parsed)
        rec['action_program'] = normalized_ap
        # Always treat each record as a single object (one node per image)
        obj_data = {
            'id': f"{problem_id}_{idx}",
            'vertices': rec.get('vertices', rec.get('geometry', [])),
            **rec.get('features', {}),
            'label': rec.get('label', ''),
            'shape_label': rec.get('label', ''),
            'category': rec.get('category', ''),
            'original_image_path': remap_path(rec.get('image_path', '')),
            'original_record_idx': idx,
            'action_program': normalized_ap
        }
        compute_physics_attributes(obj_data)
        merged_record['objects'].append(obj_data)
    if not merged_record['objects']:
        logging.error(f"Problem {problem_id}: No valid objects to build graph. Skipping.")
        return None

    # All downstream logic (clustering, decomposition, predicate induction, graph build, visualization, etc.) remains unchanged and operates on the full 12-node graph
    # ...existing code continues...

    # Assign objects for clustering
    temp_objects_for_clustering = merged_record['objects']

    # Step 1b: Assign cluster labels by K-means on centroids after all objects (and parts) are collected
    if temp_objects_for_clustering and len(temp_objects_for_clustering) > 1:
        # Filter objects that have valid 'cx' and 'cy' for clustering
        clusterable_objects = [o for o in temp_objects_for_clustering if 'cx' in o and 'cy' in o]
        # Deduplicate points by unique (cx, cy)
        unique_coords = {}
        for idx, obj in enumerate(clusterable_objects):
            key = (round(obj['cx'], 6), round(obj['cy'], 6)) # rounding for floating point stability
            if key not in unique_coords:
                unique_coords[key] = idx
        unique_points = list(unique_coords.keys())
        n_unique = len(unique_points)
        default_k = 3
        k = min(default_k, n_unique) if n_unique > 1 else 1
        # Only run KMeans if enough unique points
        if n_unique >= k and k > 1:
            # Optionally add jitter if points are nearly identical
            coords = np.array(unique_points)
            if np.ptp(coords, axis=0).max() < 1e-3:  # If all points are nearly identical
                coords = coords + np.random.normal(0, 1e-2, coords.shape)
            # Now run KMeans on (possibly jittered) coords
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(coords)
            # First, assign labels to unique points
            unique_labels = {pt: int(label) for pt, label in zip(unique_points, labels)}
            # Then, assign to all objects with matching centroid
            for obj in temp_objects_for_clustering:
                if 'cx' in obj and 'cy' in obj:
                    key = (round(obj['cx'], 6), round(obj['cy'], 6))
                    obj['cluster_label'] = unique_labels.get(key, -1)
                else:
                    obj['cluster_label'] = -1  # Assign a default for objects without valid centroids
        else:
            # Not enough unique points for clustering, assign all to one cluster
            for obj in temp_objects_for_clustering:
                obj['cluster_label'] = 0
    elif temp_objects_for_clustering:
        temp_objects_for_clustering[0]['cluster_label'] = 0


    # --- Recursive flattening of nested 'objects' lists ---
    def flatten_objects(obj_list, problem_id):
        flat = []
        for obj in obj_list:
            if isinstance(obj, dict):
                # If this object has its own 'objects' key and it's a list, flatten recursively
                if 'objects' in obj and isinstance(obj['objects'], list):
                    flat.extend(flatten_objects(obj['objects'], problem_id))
                    # Optionally remove the nested 'objects' key to avoid confusion downstream
                    obj = {k: v for k, v in obj.items() if k != 'objects'}
                flat.append(obj)
            elif isinstance(obj, list):
                flat.extend(flatten_objects(obj, problem_id))
            else:
                logging.error(f"Problem {problem_id}: Skipping non-dict/non-list object during flattening: type={type(obj)}, value={repr(obj)}")
        return flat

    # Flatten temp_objects_for_clustering before validation
    flattened_objects = flatten_objects(temp_objects_for_clustering, problem_id)

    # Final cleaning: remove any 'objects' key from all dicts, ensure required keys
    cleaned_objects = []
    for idx, obj in enumerate(flattened_objects):
        if isinstance(obj, dict):
            # Remove any nested 'objects' key
            if 'objects' in obj:
                obj = {k: v for k, v in obj.items() if k != 'objects'}
            # Check for required keys
            if 'id' in obj and 'vertices' in obj:
                cleaned_objects.append(obj)
            else:
                logging.warning(f"Problem {problem_id}: Skipping object at index {idx} missing required keys: {list(obj.keys())}")
        else:
            logging.error(f"Problem {problem_id}: Skipping non-dict object at index {idx} in flattened_objects: type={type(obj)}, value={repr(obj)}")

    merged_record['objects'] = cleaned_objects

    # Log only summary info to avoid terminal flooding
    # logging.info(f"Problem {problem_id}: Number of objects before graph build: {len(cleaned_objects)}")
    if not merged_record['objects']:
        logging.error(f"Problem {problem_id}: No valid objects to build graph. Skipping.")
        return None
    # debug_summary = [
    #     {'id': obj.get('id'), 'shape_label': obj.get('shape_label'), 'cluster_label': obj.get('cluster_label')} for obj in merged_record['objects']
    # ]
    # logging.info(f"Merged objects for problem {problem_id} (id, shape_label, cluster_label): {json.dumps(debug_summary)}")

    # --- Robust exception handling for graph building ---
    try:
        # Automated Predicate Induction via Contrastive Analysis (Step 3)
        # This will try to find a distinguishing predicate for the problem
        chosen_predicate, _ = induce_predicate_for_problem(merged_record['objects'], adaptive_thresholds)
        logging.info(f"Problem {problem_id}: Induced predicate: {chosen_predicate}")

        # Update PREDICATES with adaptive thresholds before building graph for this problem
        global PREDICATES
        # Pass canvas dimensions to load_predicates for symmetry_pair predicate
        # Use get_current_thresholds() or the correct method/property to access thresholds
        if hasattr(adaptive_thresholds, 'get_current_thresholds'):
            thresholds = adaptive_thresholds.get_current_thresholds()
        elif hasattr(adaptive_thresholds, 'thresholds'):
            thresholds = adaptive_thresholds.thresholds
        else:
            thresholds = {}
        canvas_width = thresholds.get('canvas_width', 128)
        canvas_height = thresholds.get('canvas_height', 128)
        PREDICATES = load_predicates(adaptive_thresholds, canvas_dims=(canvas_width, canvas_height))

        # 2. Build the unvalidated base scene graph for the entire problem
        # This graph will have N nodes (e.g., 12), one for each image-object.
        base_graph_nx = build_graph_unvalidated(merged_record)
    except Exception as e:
        import traceback
        logging.error(f"Problem {problem_id}: Exception during graph building phase: {e}\n{traceback.format_exc()}")
        return None
    
    # 3. Prepare base_scene_graph for enhanced builder
    base_scene_graph_for_enhanced = {
        'objects': merged_record['objects'], # These are the node data for the graph
        'relationships': [], # Relationships will be filled by add_predicate_edges and enhanced_builder
        'graph': base_graph_nx # Pass the networkx graph directly
    }

    # 4. Enhance the scene graph (knowledge fusion, hierarchical predicates)
    scene_graph_data = await enhanced_builder.build_enhanced_scene_graph(representative_image_path, base_scene_graph_for_enhanced)
    final_scene_graph = scene_graph_data['scene_graph']
    quality_metrics = scene_graph_data['quality_metrics']

    # 6. Iterative Rule Refinement & Human-in-the-Loop (Ambiguity check & trigger)
    rels = final_scene_graph.get('relationships', [])
    preds = [r['predicate'] for r in rels]
    support_rate = preds.count(chosen_predicate) / len(preds) if preds else 0

    if support_rate < 0.95 and chosen_predicate != 'same_shape': # Only flag if not the fallback and support is low
        queue_file = os.path.join(feedback_dir, 'review_queue.jsonl')
        os.makedirs(os.path.dirname(queue_file), exist_ok=True) # Ensure directory exists
        with open(queue_file, 'a') as q:
            q.write(json.dumps({
                'problem_id': problem_id,
                'induced_predicate': chosen_predicate,
                'support_rate': support_rate,
                'timestamp': time.time()
            }) + "\n")
        logging.info(f"Problem {problem_id}: Ambiguous case detected for predicate '{chosen_predicate}' (support: {support_rate:.2f}). Added to review queue.")


    # 5. Save feedback images (always, if enabled)
    if feedback_dir and representative_image_path:
        try:
            img = cv2.imread(representative_image_path)
            logging.info(f"[Feedback Save] Problem {problem_id}: representative_image_path={representative_image_path}, img type={type(img)}, img shape={getattr(img, 'shape', None)}")
            # Log scene graph summary
            if final_scene_graph is not None and 'graph' in final_scene_graph:
                G = final_scene_graph['graph']
                logging.info(f"[Feedback Save] Problem {problem_id}: scene_graph type={type(G)}, num_nodes={getattr(G, 'number_of_nodes', lambda: 'NA')()}, num_edges={getattr(G, 'number_of_edges', lambda: 'NA')()}")
            else:
                logging.info(f"[Feedback Save] Problem {problem_id}: scene_graph missing or invalid.")
            if img is not None:
                # Use problem_id as base_name for feedback images
                save_feedback_images(img, None, problem_id, feedback_dir, scene_graph=final_scene_graph)
            else:
                logging.warning(f"Could not read representative image for feedback: {representative_image_path}")
        except Exception as e:
            logging.warning(f"Failed to save feedback images for problem {problem_id} ({representative_image_path}): {e}")

    # 6. Batch validation (if enabled)
    if batch_validator:
        try:
            img_pil = Image.open(representative_image_path)
            validation_result = await batch_validator.validate_scene_graph_batch(
                [(img_pil, final_scene_graph)]
            )
            final_scene_graph['validation'] = validation_result[0] if validation_result else None
        except Exception as e:
            logging.error(f"Batch validation failed for problem {problem_id} ({representative_image_path}): {e}")
            final_scene_graph['validation'] = {'error': str(e)}

    # 7. Recall evaluation (if ground_truth is available)
    metrics = None
    # Check if problem_records[0] exists and has 'ground_truth' key
    if problem_records and 'ground_truth' in problem_records[0] and recall_evaluator:
        try:
            gt = problem_records[0]['ground_truth']
            metrics = recall_evaluator.evaluate_scene_graph(
                final_scene_graph['relationships'],
                gt['relationships'],
                final_scene_graph['objects'],
                gt['objects']
            )
            final_scene_graph['metrics'] = metrics
        except Exception as e:
            logging.error(f"Recall evaluation failed for problem {problem_id}: {e}")
            final_scene_graph['metrics'] = {'error': str(e)}

    # 8. Drift visualization (if enabled)
    if drift_visualizer:
        perf_val = None
        # Check if metrics object has 'overall_statistics' and 'recall'
        if metrics and hasattr(metrics, 'overall_statistics') and 'recall' in metrics.overall_statistics:
            perf_val = metrics.overall_statistics['recall']
        else:
            # Fallback to knowledge confidence if recall metrics are not available
            perf_val = quality_metrics.get('knowledge_confidence', 0.5)

        for rel in final_scene_graph.get('relationships', []):
            predicate = rel['predicate']
            confidence = rel.get('final_confidence', quality_metrics.get('knowledge_confidence', 0.5))
            drift_visualizer.add_threshold_data(predicate, confidence, perf_val)

    return {
        'problem_id': problem_id,
        'scene_graph_data': scene_graph_data, # Contains 'scene_graph' (with NetworkX graph) and 'quality_metrics'
        'processing_metadata': {
            'representative_image_path': representative_image_path,
            'num_nodes_in_graph': base_graph_nx.number_of_nodes()
        }
    }

def process_problem_batch_sync_wrapper(list_of_problem_records: List[List[Dict[str, Any]]], feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate, decompose_polygons: bool, adaptive_thresholds: AdaptivePredicateThresholds, args=None):
    # Synchronous wrapper to run async processing for a batch of problems.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for problem_records in list_of_problem_records:
        if problem_records: # Ensure problem_records is not empty
            problem_id = problem_records[0].get('problem_id', 'unknown_problem')
            tasks.append(_process_single_problem(problem_id, problem_records, feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate, decompose_polygons, adaptive_thresholds, args=args))
        else:
            # logging.warning("Empty problem record list passed to process_problem_batch_sync_wrapper.")
            pass
    try:
        results = loop.run_until_complete(asyncio.gather(*tasks))
    except Exception as e:
        # logging.error(f"Error processing problem batch 1: {e}")
        pass
    finally:
        loop.close()
    return results if 'results' in locals() else []

    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return results

# --- Main Execution Logic ---
async def main(): # Main is now async because it calls async functions
    parser = argparse.ArgumentParser(description="Build scene graphs from augmented images and derived labels")
    parser.add_argument('--aug', type=str, required=True, help='Path to augmented.pkl containing object records')
    parser.add_argument('--labels', type=str, default='data/derived_labels.json', help='Path to derived_labels.json')
    parser.add_argument('--out', type=str, required=True, help='Output pickle file for scene graphs')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel workers for batch processing')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for graph building (0=auto-profile). Now refers to problems per batch.')
    parser.add_argument('--feedback-dir', type=str, default='feedback_samples', help='Directory to save QA/feedback images')
    parser.add_argument('--feedback-rate', type=int, default=20, help='Save every Nth sample for feedback (default: 20, set 1 to save all)')
    parser.add_argument('--decompose-polygons', action='store_true', help='Enable decomposition of complex polygons into sub-polygons.')
    args = parser.parse_args()

    # Initialize components
    feature_cache = MemoryEfficientFeatureCache(
        cache_dir="cache/features",
        max_memory_cache_mb=1024,
        max_gpu_cache_mb=256
    )
    batch_validator = BatchMultimodalValidator(
        max_batch_size=args.batch_size if args.batch_size > 0 else 16, # Use a default if auto-profiling
        max_memory_mb=2048,
        enable_gpu_batching=True
    )
    # Define PREDICATES globally or load from config (already done above)
    recall_evaluator = RecallAtKEvaluator(
        predicate_classes=[], # Will be populated after PREDICATES are loaded
        k_values=[10, 20, 50],
        iou_threshold=0.5
    )
    drift_visualizer = ConceptDriftVisualizer(
        window_size=50,
        visualization_dir="visualizations/drift"
    ) if args.feedback_rate > 0 else None # Only enable if feedback_rate allows visualization
    enhanced_builder = EnhancedSceneGraphBuilder()

    # Adaptive predicate threshold learning
    adaptive_thresholds = AdaptivePredicateThresholds(
        history_size=1000,
        confidence_level=0.95,
        adaptation_rate=0.1,
        min_samples=50
    )
    threshold_cache_path = args.out.replace('.pkl', '_adaptive_thresholds.json')
    adaptive_thresholds.load_learned_thresholds(threshold_cache_path)
    logging.info(f"Loaded adaptive thresholds from {threshold_cache_path}")

    # Initialize PREDICATES after adaptive_thresholds are loaded
    global PREDICATES
    # Pass canvas dimensions from adaptive_thresholds for symmetry_pair predicate
    # Use get_learned_thresholds() or the correct method/property to access thresholds
    if hasattr(adaptive_thresholds, 'get_learned_thresholds'):
        thresholds = adaptive_thresholds.get_learned_thresholds()
    elif hasattr(adaptive_thresholds, 'thresholds'):
        thresholds = adaptive_thresholds.thresholds
    else:
        thresholds = {}
    canvas_width = thresholds.get('canvas_width', 128)
    canvas_height = thresholds.get('canvas_height', 128)
    PREDICATES = load_predicates(adaptive_thresholds, canvas_dims=(canvas_width, canvas_height))
    recall_evaluator.predicate_classes = list(PREDICATES.keys()) # Update recall_evaluator with all predicates

    # Compute hash for cache validation
    # Get adaptive thresholds for hashing
    if hasattr(adaptive_thresholds, 'get_learned_thresholds'):
        adaptive_params = adaptive_thresholds.get_learned_thresholds()
    elif hasattr(adaptive_thresholds, 'thresholds'):
        adaptive_params = adaptive_thresholds.thresholds
    else:
        adaptive_params = {}
    params_dict = {
        'aug': args.aug,
        'labels': args.labels,
        'parallel': args.parallel,
        'batch_size': args.batch_size,
        'decompose_polygons': args.decompose_polygons,
        'predicates_hash': hashlib.sha256(json.dumps(list(PREDICATES.keys()), sort_keys=True).encode()).hexdigest(),
        'adaptive_params_hash': hashlib.sha256(json.dumps(adaptive_params, sort_keys=True).encode()).hexdigest(),
        # Add a hash for code versioning if needed
    }
    current_hash = compute_hash([args.aug, args.labels], params_dict)
    hash_path = args.out + '.hash'

    if cache_valid(args.out, hash_path, current_hash):
        logging.info(f"[INFO] Cache valid for {args.out}. Skipping graph build.")
        # Load cached graphs if valid and return
        try:
            with open(args.out, 'rb') as f:
                graphs = pickle.load(f)
            logging.info(f"Loaded {len(graphs)} graphs from cache.")
            return # Exit if cache is valid and loaded
        except Exception as e:
            logging.warning(f"Failed to load cached graphs: {e}. Rebuilding...")

    # Load data
    logging.info(f"Loading data from {args.aug}")
    try:
        augmented_data = load_data(args.aug)
        logging.info(f"Loaded {len(augmented_data)} records from {args.aug}")
        if len(augmented_data) > 0:
            logging.info(f"Sample augmented record keys: {list(augmented_data[0].keys()) if isinstance(augmented_data[0], dict) else type(augmented_data[0])}")
    except FileNotFoundError as e:
        logging.error(f"Error loading augmented data: {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading augmented data: {e}. Exiting.")
        sys.exit(1)

    logging.info(f"Loading labels from {args.labels}")
    try:
        derived_labels = load_data(args.labels)
        logging.info(f"Loaded labels.")
    except (FileNotFoundError, ValueError) as e:
        logging.warning(f"Labels file not found or invalid at {args.labels}: {e}. Continuing without them.")
        derived_labels = {}
    except Exception as e:
        logging.error(f"Unexpected error loading labels: {e}. Continuing without them.")
        derived_labels = {}


    # --- Merge labels and group by problem_id ---
    # Build a lookup from image_path -> features/geometry
    label_lookup = {}
    if isinstance(derived_labels, list):
        for lbl in derived_labels:
            normalized_image_path = remap_path(lbl['image_path'])
            label_lookup[normalized_image_path] = {
                'features': lbl['features'],
                'vertices': lbl['geometry'],
                'label': lbl['label'],
                'category': lbl['category'],
                'action_program': lbl.get('action_program', [])
            }

    # Enrich augmented_data with label info
    enriched_augmented_data = []
    for rec in augmented_data:
        info = label_lookup.get(remap_path(rec['image_path']))
        if not info:
            logging.warning(f"No derived_labels entry for {rec['image_path']}. Skipping record.")
            continue
        new_rec = rec.copy()
        new_rec.update(info)
        enriched_augmented_data.append(new_rec)

    # Group records by problem_id
    grouped_problems = defaultdict(list)
    for rec in enriched_augmented_data:
        pid = rec.get('problem_id')
        if pid:
            grouped_problems[pid].append(rec)
        else:
            logging.warning(f"Record missing 'problem_id': {rec.get('image_path')}. Skipping.")

    logging.info(f"Grouped {len(enriched_augmented_data)} records into {len(grouped_problems)} problems.")
    # --- END Merge labels and group by problem_id ---

    # Per-problem async processing (restored logic)
    graphs = []
    for problem_id, problem_records in tqdm(grouped_problems.items(), desc="Processing problems", mininterval=0.5):
        result = await _process_single_problem(
            problem_id,
            problem_records,
            feature_cache,
            batch_validator,
            recall_evaluator,
            drift_visualizer,
            enhanced_builder,
            args.feedback_dir,
            args.feedback_rate,
            args.decompose_polygons,
            adaptive_thresholds,
            args=args
        )
        if result is not None:
            graphs.append(result)

    if len(graphs) == 0:
        logging.error(f"No scene graphs were produced. Check input data and filtering logic.")
        if len(enriched_augmented_data) > 0:
            logging.debug(f"Sample input record: {enriched_augmented_data[0]}")
            if isinstance(enriched_augmented_data[0], dict):
                logging.debug(f"Sample input record keys: {list(enriched_augmented_data[0].keys())}")
    else:
        logging.info(f"Successfully built {len(graphs)} scene graphs.")

    # Save results
    try:
        with open(args.out, 'wb') as f:
            pickle.dump(graphs, f)
        save_hash(hash_path, current_hash)
        logging.info(f"Saved {len(graphs)} scene graphs to {args.out}")
    except Exception as e:
        logging.error(f"Failed to save scene graphs to {args.out}: {e}")

    # Log diversity metrics (now operates on problem-level graphs)
    log_diversity_metrics(graphs)

    # Save adaptive thresholds
    try:
        adaptive_thresholds.save_learned_thresholds(threshold_cache_path)
        logging.info(f"Saved adaptive thresholds to {threshold_cache_path}")
    except Exception as e:
        logging.error(f"Failed to save adaptive thresholds to {threshold_cache_path}: {e}")

    # Generate drift report if enabled
    if drift_visualizer:
        all_predicates = [rel.get('predicate', 'unknown')
                         for G_data in graphs
                         for rel in G_data.get('scene_graph_data', {}).get('scene_graph', {}).get('relationships', [])]
        common_predicates = [p for p, _ in Counter(all_predicates).most_common(5)]
        if common_predicates:
            logging.info(f"Generating drift report for predicates: {common_predicates}")
            drift_report_path = drift_visualizer.generate_drift_report(common_predicates)
            viz_path = drift_visualizer.visualize_threshold_evolution(common_predicates)
            logging.info(f"Drift report saved to: {drift_report_path}")
            logging.info(f"Threshold evolution visualization saved to: {viz_path}")
        else:
            logging.warning("No common predicates found to generate drift report.")

    # Log overall performance stats
    cache_stats = feature_cache.get_cache_stats()
    logging.info("Processing complete!")
    logging.info(f"Cache efficiency: Memory: {cache_stats.get('memory_usage_mb', 0):.1f}MB used, GPU: {cache_stats.get('gpu_usage_mb', 0):.1f}MB used")

    # --- Relational GNN Training Integration ---
    try:
        if TORCH_KORNIA_AVAILABLE:
            from train_relation_gnn import train_relational_gnn
            logging.info("Attempting to call train_relational_gnn...")
            train_relational_gnn(graphs, device='cuda' if torch.cuda.is_available() else 'cpu', epochs=10, batch_size=8)
        else:
            logging.warning("PyTorch or Torchvision not available. Skipping Relational GNN training.")
    except ImportError:
        logging.warning("train_relation_gnn module not found. Skipping Relational GNN training.")
    except Exception as e:
        logging.error(f"Relational GNN training failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

