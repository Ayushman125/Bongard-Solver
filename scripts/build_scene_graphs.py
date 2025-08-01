import os
import logging
os.environ["HF_HOME"] = os.path.abspath("model_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.abspath("model_cache")
os.environ["TORCH_HOME"] = os.path.abspath("model_cache")
# --- Standard Library Imports ---
from typing import Tuple, Dict, Any, List
import torch
import argparse
import asyncio
import concurrent.futures
import gc
import hashlib
import json
import pickle
import sys
import threading
import time
import torch
import logging
import math
from collections import Counter, defaultdict
from itertools import combinations # For global graph features
from typing import Tuple

# Global predicate registry (must be defined before any global usage)
PREDICATES = {}


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
TORCH_KORNIA_AVAILABLE = False
try:
    import torch
    import torchvision
    import kornia
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

from src.scene_graphs_building.memory_efficient_cache import MemoryEfficientFeatureCache
from src.scene_graphs_building.batch_validator import BatchMultimodalValidator
from src.scene_graphs_building.recall_metrics import RecallAtKEvaluator
from src.scene_graphs_building.drift_visualizer import ConceptDriftVisualizer
from src.scene_graphs_building.knowledge_fusion import MultiSourceKnowledgeFusion
from src.scene_graphs_building.sgcore_validator import SGScoreValidator
from src.scene_graphs_building.hierarchical_predicates import HierarchicalPredicatePredictor
from src.scene_graphs_building.feature_extractors import RealFeatureExtractor
from src.scene_graphs_building.feature_extraction import get_real_feature_extractor, compute_physics_attributes, extract_line_segments
from src.scene_graphs_building.data_loading import remap_path, load_data
from src.scene_graphs_building.predicate_induction import induce_predicate_for_problem
from src.scene_graphs_building.graph_building import build_graph_unvalidated, add_commonsense_edges
from src.scene_graphs_building.visualization import save_feedback_images
from src.scene_graphs_building.recall_metrics import log_diversity_metrics
from src.scene_graphs_building.adaptive_predicates import AdaptivePredicateThresholds, create_adaptive_predicate_functions
# --- Research-grade modules ---
from src.scene_graphs_building.predicate_miner import PredicateMiner
from src.scene_graphs_building.motif_miner import MotifMiner
from src.scene_graphs_building.vl_features import CLIPEmbedder
from src.reasoner.gnn_reasoner import BongardGNN
from src.data_pipeline.adaptive_thresholds import AdaptiveThresholds
from src.commonsense_kb_api import ConceptNetAPI
from integration.task_profiler import TaskProfiler

# --- ConceptNetAPI Singleton Wrapper ---
class ConceptNetClient:
    _instance = None
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = ConceptNetAPI()
        return cls._instance

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Constants and Utilities ---
EPS = 1e-6 # Increased precision for epsilon
TOP_K = 3 # For Commonsense KB lookups

# --- ConceptNetAPI Singleton ---
_CONCEPTNET_API_INSTANCE = None
def get_conceptnet_api():
    global _CONCEPTNET_API_INSTANCE
    if _CONCEPTNET_API_INSTANCE is None:
        try:
            from src.commonsense_kb_api import ConceptNetAPI
            logging.info("Attempting to initialize ConceptNetAPI client...")
            _CONCEPTNET_API_INSTANCE = ConceptNetAPI()
            logging.info("ConceptNetAPI client initialized. Running test query...")
            test_result = _CONCEPTNET_API_INSTANCE.query_relations_for_concept("dog")
            if test_result:
                logging.info(f"ConceptNet API connected successfully. Found {len(test_result)} relations for 'dog'")
            else:
                logging.warning("ConceptNet API test query returned no results. test_result=%s", test_result)
        except Exception as e:
            import traceback
            logging.error(f"ConceptNet API connection failed: {e}\n{traceback.format_exc()}")
            logging.info("Continuing without ConceptNet knowledge base")
            _CONCEPTNET_API_INSTANCE = None
    return _CONCEPTNET_API_INSTANCE

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
    # SOTA: add new feature predicates for programmatic, KB, global stats, stroke, curvature
    all_candidate_predicates = {
        # SOTA: new feature predicates
        'curvature_sim': lambda a, b, params: abs(a.get('curvature', 0.0) - b.get('curvature', 0.0)) < params.get('curvature_tol', 0.2),
        'stroke_count_sim': lambda a, b, params: abs(a.get('stroke_count', 0) - b.get('stroke_count', 0)) < params.get('stroke_count_tol', 1),
        'programmatic_sim': lambda a, b, params: a.get('programmatic_label') == b.get('programmatic_label'),
        'kb_sim': lambda a, b, params: a.get('kb_concept') == b.get('kb_concept'),
        'global_stat_sim': lambda a, b, params: abs(a.get('global_stat', 0.0) - b.get('global_stat', 0.0)) < params.get('global_stat_tol', 0.2),
        # aspect_sim: True if aspect ratios are similar within a threshold
        'aspect_sim': lambda a, b, params: abs(a.get('aspect_ratio', 1.0) - b.get('aspect_ratio', 1.0)) < params.get('aspect_tol', 0.2),
        # para: True if orientations are similar within a threshold (parallel)
        'para': lambda a, b, params: abs(a.get('orientation', 0) - b.get('orientation', 0)) < params.get('orient_tol', 7),
        'left_of':     lambda a, b, params: a.get('cx', 0) + EPS < b.get('cx', 0),
        'right_of':    lambda a, b, params: a.get('cx', 0) > b.get('cx', 0) + EPS,
        'above':       lambda a, b, params: a.get('cy', 0) + EPS < b.get('cy', 0),
        'below':       lambda a, b, params: a.get('cy', 0) > b.get('cy', 0) + EPS,
        'contains':    lambda a, b, params: a.get('object_type') == 'polygon' and b.get('object_type') == 'polygon' and Polygon(a['vertices']).contains(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
        'inside':      lambda a, b, params: a.get('object_type') == 'polygon' and b.get('object_type') == 'polygon' and Polygon(b['vertices']).contains(Polygon(a['vertices'])) if 'vertices' in a and 'vertices' in b else False,
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
                temp_nodes_for_clustering = []  # List of all nodes for this problem
                for idx, rec in enumerate(problem_recs):
                    # Normalize action_program in rec
                    raw_ap = rec.get('action_program', [])
                    normalized_ap = []
                    for cmd in raw_ap:
                        parsed = parse_action_command(cmd)
                        if parsed:
                            normalized_ap.append(parsed)
                    rec['action_program'] = normalized_ap

                    # Use geometry field for node data
                    geometry = rec.get('geometry', [])
                    for node_idx, node in enumerate(geometry):
                        # Normalize action_program in node
                        raw_node_ap = node.get('action_program', [])
                        normalized_node_ap = []
                        for cmd in raw_node_ap:
                            parsed = parse_action_command(cmd)
                            if parsed:
                                normalized_node_ap.append(parsed)
                        node['action_program'] = normalized_node_ap
                        node_data = {
                            'id': node.get('id', f"{pid}_{idx}_{node_idx}"),
                            'vertices': node.get('vertices', []),
                            **node,
                            'label': node.get('label', rec.get('label', '')),
                            'shape_label': node.get('shape_label', node.get('label', rec.get('label', ''))),
                            'category': node.get('category', rec.get('category', '')),
                            'original_image_path': remap_path(rec.get('image_path', '')),
                            'original_record_idx': idx,
                            'action_program': node['action_program']
                        }
                        # Compute additional node attributes
                        compute_physics_attributes(node_data)

                        # Optionally decompose complex polygons during profiling
                        if args.decompose_polygons and node_data.get('action_program'):
                            segments = extract_line_segments(node_data['action_program'])
                            if segments:
                                try:
                                    from shapely.geometry import MultiLineString
                                    multi_line = MultiLineString(segments)
                                    unary_union_result = unary_union(multi_line)
                                    sub_polys = list(polygonize(unary_union_result))

                                    if len(sub_polys) > 1:
                                        for i, sub in enumerate(sub_polys):
                                            new_node_id = f"{node_data['id']}_part{i}"
                                            new_node_data = {
                                                'id': new_node_id,
                                                'vertices': list(sub.exterior.coords) if sub.exterior else [],
                                                'shape_label': node_data.get('shape_label', 'part'),
                                                'category': node_data.get('category', 'part'),
                                                'parent_id': node_data['id']
                                            }
                                            compute_physics_attributes(new_node_data)
                                            temp_nodes_for_clustering.append(new_node_data)
                                    else:
                                        temp_nodes_for_clustering.append(node_data)
                                except Exception as e:
                                    logging.warning(f"Failed to decompose node {node_data['id']} during profiling: {e}. Adding original node.")
                                    temp_nodes_for_clustering.append(node_data)
                            else:
                                temp_nodes_for_clustering.append(node_data)
                        else:
                            temp_nodes_for_clustering.append(node_data)

                # Assign cluster labels for profiling purposes
                if temp_nodes_for_clustering and len(temp_nodes_for_clustering) > 1:
                    coords = np.array([[o['cx'], o['cy']] for o in temp_nodes_for_clustering if 'cx' in o and 'cy' in o])
                    if len(coords) > 1:
                        k = min(3, len(coords))
                        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
                        labels = kmeans.fit_predict(coords)
                        valid_coords_idx = 0
                        for node in temp_nodes_for_clustering:
                            if 'cx' in node and 'cy' in node:
                                node['cluster_label'] = int(labels[valid_coords_idx])
                                valid_coords_idx += 1
                            else:
                                node['cluster_label'] = -1
                    else:
                        for node in temp_nodes_for_clustering:
                            node['cluster_label'] = 0
                elif temp_nodes_for_clustering:
                    temp_nodes_for_clustering[0]['cluster_label'] = 0

                merged_rec = {'geometry': temp_nodes_for_clustering}  # Use geometry field

                if merged_rec['geometry']:
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



class EnhancedSceneGraphBuilder:
    def __init__(self):
        self.knowledge_fusion = MultiSourceKnowledgeFusion()
        self.sgcore_validator = SGScoreValidator()
        self.hierarchical_predictor = HierarchicalPredicatePredictor()
        self.feature_extractor = get_real_feature_extractor()
        # SOTA: ensure all new features are processed in the pipeline
        self.extra_features = ['curvature', 'stroke_count', 'programmatic_label', 'kb_concept', 'global_stat']

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
                from src.scene_graphs_building.data_loading import robust_image_open
                image = robust_image_open(image_path).convert("RGB")
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

import numpy as np
from shapely.geometry import Polygon, LineString

def are_points_collinear(verts, tol=1e-6):
    arr = np.asarray(verts)
    if len(arr) < 3:
        return True
    v0 = arr[0]
    v1 = arr[1]
    direction = v1 - v0
    norm = np.linalg.norm(direction)
    if norm < tol:
        return True
    direction = direction / norm
    for v in arr[2:]:
        rel = v - v0
        proj = np.dot(rel, direction)
        perp = rel - proj * direction
        if np.linalg.norm(perp) > tol:
            return False
    return True

def assign_object_type(verts):
    arr = np.asarray(verts)
    if len(arr) == 1 or np.allclose(np.std(arr, axis=0), 0, atol=1e-8):
        return "point"
    elif len(arr) == 2 or are_points_collinear(arr):
        return "line"
    elif len(arr) >= 3 and not are_points_collinear(arr):
        return "polygon"
    else:
        # SOTA: treat degenerate/ambiguous as 'curve' if possible
        return "curve" if len(arr) > 1 else "unknown"

async def _process_single_problem(
    problem_id: str,
    problem_records: List[Dict[str, Any]],
    feature_cache,
    batch_validator,
    recall_evaluator,
    drift_visualizer,
    enhanced_builder,
    feedback_dir,
    feedback_rate,
    decompose_polygons: bool,
    adaptive_thresholds: AdaptivePredicateThresholds,
    args=None
):
    global PREDICATES
    import traceback

    logging.info(f"Starting _process_single_problem for {problem_id}. problem_records type: {type(problem_records)}")


    # 1. Merge all image records into one list of objects
    merged_record = {'objects': []}
    representative_image_path = None

    from src.scene_graphs_building.clip_embedder import CLIPEmbedder
    from src.scene_graphs_building.predicate_miner import PredicateMiner
    from src.scene_graphs_building.motif_miner import MotifMiner
    from src.reasoner.gnn_reasoner import GNNReasoner
    from integration.rule_inducer import RuleInducer
    import numpy as np
    # 1. Build objects with enriched features
    clip_embedder = CLIPEmbedder()
    objects = []
    for idx, rec in enumerate(problem_records):
        if representative_image_path is None:
            representative_image_path = remap_path(rec.get('image_path', ''))
        verts = rec.get('vertices') or rec.get('geometry') or []
        obj_type = assign_object_type(verts)
        arr = np.array(verts)
        obj = {
            'object_id': f"{problem_id}_{idx}",
            'vertices': verts,
            'object_type': obj_type,
            'is_closed': (len(verts) > 2 and np.allclose(verts[0], verts[-1], atol=1e-6)),
            'fallback_geometry': False, # will update below
            'bounding_box': [float(np.min(arr[:,0])) if verts else 0.0,
                             float(np.min(arr[:,1])) if verts else 0.0,
                             float(np.max(arr[:,0])) if verts else 0.0,
                             float(np.max(arr[:,1])) if verts else 0.0],
            'centroid': [0.0, 0.0], # will update below
            'area': 0.0,
            'perimeter': 0.0,
            'orientation': 0.0,
            'aspect_ratio': 1.0,
            'curvature': 0.0,
            'skeleton_length': 0.0,
            'shape_label': rec.get('label', ''),
            'action_program': rec.get('action_program', []),
            'stroke_type': rec.get('stroke_type', ''),
            'symmetry_axis': 0.0,
            'component_index': idx,
            'is_valid': True,
            'geometry_reason': '',
            'object_color': rec.get('object_color', ''),
            'category': rec.get('category'),
            'label': rec.get('label'),
            'image_path': rec.get('image_path'),
            'feature_valid': {},
        }
        # SOTA: Feature validity and geometry for all types
        if obj_type == "polygon":
            try:
                poly = Polygon(arr)
                obj['centroid'] = list(poly.centroid.coords[0])
                obj['area'] = float(poly.area)
                obj['perimeter'] = float(poly.length)
                obj['orientation'] = float(np.degrees(np.arctan2(arr[-1][1]-arr[0][1], arr[-1][0]-arr[0][0])))
                obj['aspect_ratio'] = (obj['bounding_box'][2] - obj['bounding_box'][0]) / max(obj['bounding_box'][3] - obj['bounding_box'][1], 1e-6)
                obj['curvature'] = 0.0 # can be updated with more logic
                obj['skeleton_length'] = 0.0 # can be updated with more logic
                obj['symmetry_axis'] = obj['orientation']
                obj['is_valid'] = poly.is_valid and obj['area'] > 1e-6
                obj['geometry_reason'] = "polygon" if obj['is_valid'] else "degenerate"
                obj['fallback_geometry'] = not obj['is_valid']
                for feat in ['curvature', 'skeleton_length', 'symmetry_axis', 'centroid', 'orientation', 'area', 'perimeter', 'aspect_ratio']:
                    obj['feature_valid'][feat] = True if obj['is_valid'] else False
            except Exception as e:
                obj['is_valid'] = False
                obj['geometry_reason'] = f"polygon_error: {e}"
                obj['fallback_geometry'] = True
                for feat in ['curvature', 'skeleton_length', 'symmetry_axis', 'centroid', 'orientation', 'area', 'perimeter', 'aspect_ratio']:
                    obj['feature_valid'][feat] = False
        elif obj_type in ["line", "curve"]:
            # SOTA: Always set geometry for lines/curves
            if arr.shape[0] > 1:
                centroid = np.mean(arr, axis=0)
                obj['centroid'] = centroid.tolist()
                obj['perimeter'] = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
                obj['orientation'] = float(np.degrees(np.arctan2(arr[-1][1]-arr[0][1], arr[-1][0]-arr[0][0])))
                obj['skeleton_length'] = obj['perimeter']
                obj['curvature'] = None # can be updated if needed
                obj['area'] = 0.0
                obj['aspect_ratio'] = (obj['bounding_box'][2] - obj['bounding_box'][0]) / max(obj['bounding_box'][3] - obj['bounding_box'][1], 1e-6)
                obj['symmetry_axis'] = None
                obj['is_valid'] = True
                obj['geometry_reason'] = obj_type
                obj['fallback_geometry'] = False
                for feat in ['centroid', 'orientation', 'perimeter', 'skeleton_length', 'aspect_ratio']:
                    obj['feature_valid'][feat] = True
                for feat in ['area', 'curvature', 'symmetry_axis']:
                    obj['feature_valid'][feat] = False
            else:
                obj['is_valid'] = False
                obj['geometry_reason'] = f"{obj_type}_degenerate"
                obj['fallback_geometry'] = True
                for feat in ['centroid', 'orientation', 'perimeter', 'skeleton_length', 'aspect_ratio', 'area', 'curvature', 'symmetry_axis']:
                    obj['feature_valid'][feat] = False
        elif obj_type == "point":
            obj['centroid'] = arr[0].tolist() if len(arr) == 1 else [0.0, 0.0]
            obj['area'] = 0.0
            obj['perimeter'] = 0.0
            obj['orientation'] = 0.0
            obj['aspect_ratio'] = 1.0
            obj['curvature'] = None
            obj['skeleton_length'] = None
            obj['symmetry_axis'] = None
            obj['is_valid'] = True
            obj['geometry_reason'] = "point"
            obj['fallback_geometry'] = True
            for feat in ['centroid', 'area', 'perimeter', 'orientation', 'aspect_ratio', 'curvature', 'skeleton_length', 'symmetry_axis']:
                obj['feature_valid'][feat] = feat == 'centroid'
        else:
            obj['is_valid'] = False
            obj['geometry_reason'] = "unknown"
            obj['fallback_geometry'] = True
            for feat in ['centroid', 'area', 'perimeter', 'orientation', 'aspect_ratio', 'curvature', 'skeleton_length', 'symmetry_axis']:
                obj['feature_valid'][feat] = False
        # Logging for interpretability
        logging.info(f"Object {obj['object_id']}: type={obj['object_type']}, valid={obj['is_valid']}, fallback={obj['fallback_geometry']}, reason={obj['geometry_reason']}, feature_valid={obj['feature_valid']}")
        # SOTA: Vision-language and motif fields
        obj['clip_sim'] = None
        obj['vl_sim'] = None
        obj['motif_score'] = None
        if getattr(args, 'use_vl', False) and obj.get('image_path'):
            try:
                from src.scene_graphs_building.data_loading import robust_image_open
                img = robust_image_open(obj['image_path'])
                # SOTA: Use ROI/mask-based embedding for lines/curves
                if obj_type in ["line", "curve"] and arr.shape[0] > 1:
                    x1, y1, x2, y2 = [int(round(x)) for x in obj['bounding_box']]
                    img_crop = img.crop((x1, y1, x2, y2))
                    obj['vl_embed'] = clip_embedder.embed_image(img_crop)
                else:
                    obj['vl_embed'] = clip_embedder.embed_image(img)
            except Exception as e:
                obj['vl_embed'] = np.zeros(512)
        # Log non-polygon objects for feedback
        if obj_type != "polygon":
            logging.warning(f"Inserted non-polygon object: id={obj['object_id']}, type={obj['object_type']}, vertices={verts}")
        objects.append(obj)
    merged_record['objects'] = objects
    # 2. Motif mining and super-nodes (SOTA: graph-based, type-aware)
    motif_nodes = []
    motif_edges = []
    if getattr(args, 'use_motifs', False):
        motif_dict, motif_nodes = MotifMiner().cluster_motifs(objects, method='graph+type')
        for motif_node in motif_nodes:
            supernode_id = motif_node['id']
            member_nodes = [obj for obj in objects if obj.get('object_id', obj.get('id')) in motif_node['member_nodes']]
            # SOTA: Aggregate motif features (stroke count, mean direction, etc.)
            motif_node['stroke_count'] = len(member_nodes)
            motif_node['mean_orientation'] = float(np.mean([o['orientation'] for o in member_nodes if o['feature_valid'].get('orientation')])) if member_nodes else 0.0
            motif_node['motif_type'] = motif_dict.get(supernode_id, 'unknown')
            # Add part_of_motif edges
            for nid in motif_node['member_nodes']:
                motif_edges.append((nid, supernode_id, {'predicate': 'part_of', 'source': 'motif'}))
    merged_record['geometry'] = objects + motif_nodes
    # Compute mean VL embedding and assign clip_sim/vl_sim for all nodes
    all_embeds = [obj.get('vl_embed') for obj in objects + motif_nodes if obj.get('vl_embed') is not None]
    if all_embeds:
        mean_embed = np.mean(np.array(all_embeds), axis=0)
        def cosine_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        for obj in objects + motif_nodes:
            if obj.get('vl_embed') is not None:
                obj['clip_sim'] = cosine_sim(obj['vl_embed'], mean_embed)
                obj['vl_sim'] = cosine_sim(obj['vl_embed'], mean_embed)
    # SOTA: log node/edge type and feature coverage
    n_poly = sum(1 for o in objects if o['object_type'] == 'polygon')
    n_line = sum(1 for o in objects if o['object_type'] == 'line')
    n_curve = sum(1 for o in objects if o['object_type'] == 'curve')
    n_point = sum(1 for o in objects if o['object_type'] == 'point')
    logging.info(f"Node type counts: polygon={n_poly}, line={n_line}, curve={n_curve}, point={n_point}")
    n_valid_geom = sum(1 for o in objects if o['is_valid'])
    logging.info(f"Valid geometry nodes: {n_valid_geom}/{len(objects)}")
    # 3. Adaptive predicate mining (SOTA: branch by type, ensure all pairs get at least one predicate)
    pred_miner = PredicateMiner()
    learned_thresholds = pred_miner.fit(objects)
    dynamic_predicates = pred_miner.build_edge_fn(type_aware=True)  # SOTA: type-aware branching
    for k, v in dynamic_predicates.items():
        logging.info(f"Predicate '{k}' type: {type(v)}")
    # SOTA: ensure every node pair gets at least one predicate (esp. for lines/curves)
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j: continue
            found = any(fn(a, b) for fn in dynamic_predicates.values())
            if not found:
                logging.warning(f"No predicate found for node pair {a['object_id']} ({a['object_type']}), {b['object_id']} ({b['object_type']})")
    # 4. VLM edges (SOTA: include lines/curves, use ROI/mask)
    vl_edges = []
    if getattr(args, 'use_vl', False):
        vl_edges = clip_embedder.contrastive_edges(objects, use_roi=getattr(args, 'use_roi', False))
    # 5. Build graph
    base_graph_nx = build_graph_unvalidated(merged_record, dynamic_predicates, TOP_K, extra_edges=vl_edges+motif_edges, kb=kb)
    logging.info(f"[DEBUG] After build_graph_unvalidated: type={type(base_graph_nx)}, nodes={getattr(base_graph_nx, 'number_of_nodes', lambda: 'NA')()}, edges={getattr(base_graph_nx, 'number_of_edges', lambda: 'NA')()}")
    if base_graph_nx is None:
        logging.error(f"[ERROR] build_graph_unvalidated returned None for problem {problem_id}. Skipping.")
        return {'scene_graph': None, 'rules': None}
    # SOTA: log edge predicate type mix
    pred_types = [edata.get('predicate') for _,_,edata in base_graph_nx.edges(data=True)]
    from collections import Counter
    pred_counter = Counter(pred_types)
    logging.info(f"Predicate type counts: {dict(pred_counter)}")
    # --- Global Graph Statistics ---
    node_centroids = [d.get('centroid', [0,0]) for n, d in base_graph_nx.nodes(data=True)]
    def is_valid_centroid(c):
        return (
            isinstance(c, (list, tuple, np.ndarray)) and
            len(c) == 2 and
            all(isinstance(v, (int, float, np.floating, np.integer)) and not np.isnan(v) for v in c)
        )
    if node_centroids:
        node_centroids_clean = [
            c if is_valid_centroid(c) else [np.nan, np.nan]
            for c in node_centroids
        ]
        arr = np.array(node_centroids_clean)
        center = np.nanmean(arr, axis=0)
        sym_pairs = 0
        total_pairs = 0
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                # SOTA: symmetry for all types
                if np.all(np.isfinite(arr[i])) and np.all(np.isfinite(arr[j])):
                    if np.allclose(arr[i]+arr[j], 2*center, atol=10):
                        sym_pairs += 1
                    total_pairs += 1
        base_graph_nx.graph['symmetry_score'] = sym_pairs/total_pairs if total_pairs else 0.0
    else:
        base_graph_nx.graph['symmetry_score'] = 0.0
    base_graph_nx.graph['motif_diversity'] = len(set([d.get('motif_type','') for n, d in base_graph_nx.nodes(data=True) if d.get('is_motif')]))
    from scipy.stats import entropy
    pred_counts = [edata.get('predicate') for _,_,edata in base_graph_nx.edges(data=True)]
    _, counts = np.unique(pred_counts, return_counts=True)
    base_graph_nx.graph['pred_entropy'] = float(entropy(counts)) if len(counts)>0 else 0.0
    # --- GNN/Transformer Reasoner ---
    if getattr(args, 'use_gnn', False):
        from src.reasoner.gnn_reasoner import GNNReasoner
        gnn = GNNReasoner(in_dim=10)
        try:
            base_graph_nx = gnn.predict(base_graph_nx)
            logging.info(f"[DEBUG] After GNN: type={type(base_graph_nx)}, nodes={getattr(base_graph_nx, 'number_of_nodes', lambda: 'NA')()}, edges={getattr(base_graph_nx, 'number_of_edges', lambda: 'NA')()}")
        except Exception as e:
            logging.error(f"[ERROR] GNNReasoner failed for problem {problem_id}: {e}")
            return {'scene_graph': None, 'rules': None}
    # --- Symbolic Rule Inducer ---
    from integration.rule_inducer import RuleInducer
    try:
        rules = RuleInducer().induce(base_graph_nx)
        logging.info(f"[DEBUG] After RuleInducer: rules type={type(rules)}, rules={rules}")
    except Exception as e:
        logging.error(f"[ERROR] RuleInducer failed for problem {problem_id}: {e}")
        return {'scene_graph': base_graph_nx, 'rules': None}
    # 9. Save feedback images with all features visualized
    if feedback_dir and representative_image_path:
        try:
            from src.scene_graphs_building.data_loading import robust_image_open
            img = robust_image_open(representative_image_path)
            from src.scene_graphs_building.visualization import save_feedback_images
            save_feedback_images(img, None, problem_id, feedback_dir, scene_graph={'graph': base_graph_nx, 'rules': rules})
        except Exception as e:
            logging.warning(f"Failed to save feedback images for problem {problem_id} ({representative_image_path}): {e}")
    return {'scene_graph': base_graph_nx, 'rules': rules}


    # Debug: log merged object count and details
    logging.info(f"Problem {problem_id}: merged {len(merged_record['objects'])} objects")
    for i, obj in enumerate(merged_record['objects']):
        logging.info(f"Problem {problem_id}: object {i}: id={obj.get('id')}, shape_label={obj.get('shape_label')}, category={obj.get('category')}, vertices={obj.get('vertices')[:5]}... (total {len(obj.get('vertices', []))} vertices)")

    if not merged_record['objects']:
        logging.error(f"No objects for problem {problem_id}; skipping.")
        return None

    # 2. Predicate induction and load predicates
    try:
        # Use dynamic predicates
        chosen_predicate = 'dynamic'  # Placeholder
        logging.info(f"Problem {problem_id}: Learned dynamic thresholds: {learned_thresholds}")

        # Save learned thresholds to adaptive cache
        if adaptive_thresholds:
            adaptive_thresholds.update(problem_id, learned_thresholds)

        # Ensure geometry is set for graph building
        merged_record['geometry'] = merged_record['objects']

        # 2b. Add motif/super-node edges (not shown, but could be added to edge list)

        # 2c. Add vision-language (CLIP) edges if enabled
        vl_edges = []
        if getattr(args, 'use_vl', False):
            # Always use CUDA for CLIP if available, fallback to CPU
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                logging.info('Using CUDA for CLIP vision-language embedding.')
            else:
                logging.warning('CUDA not available, using CPU for CLIP vision-language embedding.')
            clip_embedder = CLIPEmbedder(device=device)
            vl_edges = clip_embedder.contrastive_edges(merged_record['objects'], use_roi=getattr(args, 'use_roi', False))
            logging.info(f"Problem {problem_id}: Added {len(vl_edges)} CLIP edges.")

        # 3. Build base scene graph with dynamic predicates
        # (You must update build_graph_unvalidated to accept dynamic_predicates)
        base_graph_nx = build_graph_unvalidated(merged_record, dynamic_predicates, TOP_K, extra_edges=vl_edges, kb=kb)
        # Log produced scene graph summary
        if base_graph_nx is not None:
            logging.info(f"Problem {problem_id}: base_graph_nx nodes={base_graph_nx.number_of_nodes()}, edges={base_graph_nx.number_of_edges()}")
            # Log a sample of node and edge data
            node_list = list(base_graph_nx.nodes(data=True))
            edge_list = list(base_graph_nx.edges(data=True))
            for i, (nid, ndata) in enumerate(node_list[:3]):
                logging.info(f"Problem {problem_id}: node {i}: id={nid}, data_keys={list(ndata.keys())}, shape_label={ndata.get('shape_label')}, category={ndata.get('category')}, vertices={ndata.get('vertices')[:5]}... (total {len(ndata.get('vertices', []))} vertices)")
            for i, (u, v, edata) in enumerate(edge_list[:3]):
                logging.info(f"Problem {problem_id}: edge {i}: {u}->{v}, predicate={edata.get('predicate')}, source={edata.get('source', '')}, data_keys={list(edata.keys())}")
        else:
            logging.warning(f"Problem {problem_id}: base_graph_nx is None after build_graph_unvalidated.")

    except Exception as e:
        logging.error(f"Problem {problem_id}: Exception during graph building phase: {e}\n{traceback.format_exc()}")
        return None

    # ... rest of the function remains unchanged, operating on `base_graph_nx` ...




    
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
            from src.scene_graphs_building.data_loading import robust_image_open
            img = robust_image_open(representative_image_path)
            logging.info(f"[Feedback Save] Problem {problem_id}: representative_image_path={representative_image_path}, img type={type(img)}, img shape={getattr(img, 'size', None)}")
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
            from src.scene_graphs_building.data_loading import robust_image_open
            img_pil = robust_image_open(representative_image_path)
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
            problem_id = problem_records[0].get('problem_id', 'unknown')
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
    parser.add_argument('--feedback-rate', type=int, default=1, help='Save every Nth sample for feedback (default: 1 = save all)')
    parser.add_argument('--decompose-polygons', action='store_true', help='Enable decomposition of complex polygons into sub-polygons.')
    parser.add_argument('--tpot-max-time', type=int, default=5, help='Max time in minutes for TPOT AutoML (default: 5)')
    parser.add_argument('--tpot-generations', type=int, default=5, help='Number of generations for TPOT AutoML (default: 5)')
    parser.add_argument('--use-vl', action='store_true', help='Enable vision-language (CLIP) edges in scene graph')
    parser.add_argument('--use-gnn', action='store_true', help='Enable GNN/Transformer reasoning in scene graph')
    parser.add_argument('--use-motifs', action='store_true', help='Enable motif mining and super-nodes in scene graph')
    parser.add_argument('--use-roi', action='store_true', help='Use ROI (bounding box/mask) for CLIP vision-language edges (SOTA)')
    args = parser.parse_args()
    global kb
    kb = ConceptNetClient.get()
    if kb is None:
        logging.warning("ConceptNet KB client could not be initialized. Commonsense edges will be skipped.")

    # Initialize components
    feature_cache = MemoryEfficientFeatureCache(
        cache_dir="cache/features",
        max_memory_cache_mb=1024,
        max_gpu_cache_mb=256
    )
    # Only initialize BatchMultimodalValidator if batch_size > 2 (i.e., if Dask or large batch logic is used)
    batch_validator = None
    if args.batch_size > 2:
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

    # Track all feedback images saved for summary
    saved_feedback_images = []
    enhanced_builder = EnhancedSceneGraphBuilder()

    # Adaptive predicate threshold learning (now uses SQLite)
    adaptive_thresholds = AdaptiveThresholds()
    threshold_cache_path = args.out.replace('.pkl', '_adaptive_thresholds.json')

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
        new_rec.pop('objects', None)  # Remove legacy primitives
        new_rec['geometry'] = info['vertices']  # Set geometry field for downstream use
        enriched_augmented_data.append(new_rec)

    # Group records by problem_id
    grouped_problems = defaultdict(list)
    for rec in enriched_augmented_data:
        pid = rec.get('problem_id')
        if pid:
            grouped_problems[pid].append(rec)
        else:
            logging.warning(f"Record missing 'problem_id': {rec.get('image_path')}. Skipping.")

    logging.info(f"Grouped records into {len(grouped_problems)} problems.")
    
    # --- END Merge labels and group by problem_id ---
    use_dask = args.batch_size > 2
    graphs = []  # Always define graphs before branching
    gnn_model = None
    if not use_dask:
        logging.info("Batch size is small; using asyncio for processing instead of Dask.")
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
                1,  # Always save feedback images for every problem
                args.decompose_polygons,
                adaptive_thresholds,
                args=args
            )
            if result is not None:
                graphs.append(result)
                # Track feedback image for summary
                feedback_img_path = os.path.join(args.feedback_dir, f"{problem_id}_img_graph.png")
                if os.path.exists(feedback_img_path):
                    saved_feedback_images.append(feedback_img_path)
        # Cutting-edge GNN training block (after all graphs are built)
        if getattr(args, 'use_gnn', False) and len(graphs) > 0:
            try:
                from src.reasoner.gnn_reasoner import GNNReasoner
                # Prepare torch_geometric Data objects and labels
                graph_data_list = []
                graph_labels = []
                for g_result in graphs:
                    scene_graph = g_result.get('scene_graph')
                    if scene_graph is not None and hasattr(scene_graph, 'nodes'):
                        gnn = GNNReasoner(in_dim=10)
                        data, node_ids = gnn.nx_to_pyg(scene_graph)
                        graph_data_list.append(data)
                        # Use a placeholder label (e.g., 1.0) or extract from your data
                        graph_labels.append(1.0)
                        # Log GNN node scores if available
                        if hasattr(scene_graph, 'nodes'):
                            for n in scene_graph.nodes:
                                score = scene_graph.nodes[n].get('gnn_score')
                                logging.info(f"[GNN] Node {n} gnn_score: {score}")
                        logging.info(f"[GNN] Graph status: {scene_graph.graph.get('gnn_status')}")
                if len(graph_data_list) > 0:
                    gnn_model = GNNReasoner.train(graph_data_list, graph_labels, epochs=50, batch_size=8, val_split=0.2, patience=5, lr=1e-3, weight_decay=1e-4)
                    # Save model weights for reproducibility
                    torch.save(gnn_model.state_dict(), "gnn_model_best.pt")
                    logging.info("[GNN] Model trained and saved as gnn_model_best.pt")
            except Exception as e:
                logging.error(f"[GNN] Training failed: {e}")
    else:
        logging.info("Using Dask for parallel processing.")
        # Only import and initialize Dask if needed for large batches
        try:
            from dask.distributed import Client, LocalCluster
            # Guard Dask cluster initialization
            cluster = LocalCluster(n_workers=args.parallel, threads_per_worker=1)
            client = Client(cluster)
            # ...Dask-based batch processing if implemented...
            # (Your Dask logic here)
            # For now, leave graphs empty or implement Dask logic as needed
            client.close()
            cluster.close()
        except ImportError:
            logging.error("Dask is not installed but required for large batch processing.")
        except Exception as e:
            logging.error(f"Dask-based processing failed: {e}")
        # If Dask fails, graphs remains as the empty list

    if len(graphs) == 0:
        logging.error(f"No scene graphs were produced. Check input data and filtering logic.")
        if len(enriched_augmented_data) > 0:
            logging.debug(f"Sample input record: {enriched_augmented_data[0]}")
            if isinstance(enriched_augmented_data[0], dict):
                logging.debug(f"Sample input record keys: {list(enriched_augmented_data[0].keys())}")
    else:
        logging.info(f"Successfully built {len(graphs)} scene graphs.")
        # Optionally re-rank with GNN (now handled above)

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

    # Print summary of saved feedback images
    if saved_feedback_images:
        print("\n[Scene Graph Visualization Summary]")
        print(f"Saved {len(saved_feedback_images)} scene graph visualizations to '{args.feedback_dir}':")
        for img_path in saved_feedback_images:
            print(f"  {img_path}")
    else:
        print("No scene graph visualizations were saved.")

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



if __name__ == "__main__":
    asyncio.run(main())

