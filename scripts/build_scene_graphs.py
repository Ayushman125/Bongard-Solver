# --- Local Application Imports ---
# Ensure project root is in sys.path for imports relative to the script's directory

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import traceback
import json
import asyncio
import argparse
import hashlib
import pickle
import math
import sys
from collections import defaultdict, Counter
from typing import Tuple, Dict, Any, List
import logging
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
from src.scene_graphs_building.process_single_problem import _process_single_problem



from src.scene_graphs_building.memory_efficient_cache import MemoryEfficientFeatureCache
from src.scene_graphs_building.batch_validator import BatchMultimodalValidator
from src.scene_graphs_building.recall_metrics import RecallAtKEvaluator
from src.scene_graphs_building.drift_visualizer import ConceptDriftVisualizer
from src.scene_graphs_building.knowledge_fusion import MultiSourceKnowledgeFusion
from src.scene_graphs_building.sgcore_validator import SGScoreValidator
from src.scene_graphs_building.hierarchical_predicates import HierarchicalPredicatePredictor
from src.scene_graphs_building.feature_extractors import RealFeatureExtractor
from src.scene_graphs_building.feature_extraction import compute_physics_attributes, extract_line_segments
from src.scene_graphs_building.data_loading import remap_path, load_action_programs, get_problem_data, load_json_data
from src.scene_graphs_building.predicate_induction import induce_predicate_for_problem
from src.scene_graphs_building.graph_building import build_graph_unvalidated, add_commonsense_edges
from src.scene_graphs_building.visualization import save_feedback_images
from src.scene_graphs_building.recall_metrics import log_diversity_metrics
from src.scene_graphs_building.adaptive_predicates import AdaptivePredicateThresholds, create_adaptive_predicate_functions
from src.scene_graphs_building.advanced_predicates import ADVANCED_PREDICATE_REGISTRY
# --- Enhanced visualization imports ---
from scripts.scene_graph_visualization import save_scene_graph_csv
# --- Research-grade modules ---
from src.scene_graphs_building.predicate_miner import PredicateMiner
from src.scene_graphs_building.motif_miner import MotifMiner
from src.scene_graphs_building.vl_features import CLIPEmbedder
from src.reasoner.gnn_reasoner import BongardGNN
from src.data_pipeline.adaptive_thresholds import AdaptiveThresholds
from src.commonsense_kb_api import ConceptNetAPI
from integration.task_profiler import TaskProfiler

def get_real_feature_extractor():
    """Create and return a RealFeatureExtractor instance"""
    return RealFeatureExtractor()

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

# --- Bongard LOGO dataset main entry ---
async def run_bongard_logo_scene_graph_pipeline(args):
    """Enhanced LOGO mode pipeline with proper action program handling - ACTION PROGRAMS ONLY MODE"""
    
    # Load action programs from the correct directory structure
    action_base_dir = args.action_base_dir or "data/raw/ShapeBongard_V2"
    action_programs = load_action_programs(action_base_dir)
    
    if not action_programs:
        logging.error(f"No action programs loaded from {action_base_dir}")
        return
    
    logging.info(f"[LOGO] ACTION-ONLY MODE: Loaded {len(action_programs)} action programs")
    
    # Optional: Load derived labels as fallback, but not required
    derived_labels = None
    try:
        if args.labels and os.path.exists(args.labels):
            derived_labels = load_json_data(args.labels)
            logging.info(f"[LOGO] Loaded {len(derived_labels)} derived labels as optional fallback")
        else:
            logging.info(f"[LOGO] No derived labels provided or file not found. Using ACTION PROGRAMS ONLY.")
    except Exception as e:
        logging.warning(f"[LOGO] Could not load derived labels: {e}. Using ACTION PROGRAMS ONLY.")
        derived_labels = None

    # Handle puzzle list - check if it's from split file or manual list
    puzzle_ids = []
    if args.puzzles_list:
        if args.puzzles_list.endswith('.json'):
            # It's a split file
            from src.scene_graphs_building.data_loading import load_split_file, get_problem_ids_from_split
            split_data = load_split_file(args.puzzles_list)
            if split_data:
                # Use validation split by default, limit to 50 for testing
                puzzle_ids = get_problem_ids_from_split(split_data, 'val', max_problems=50)
        else:
            # It's a text file with problem IDs or paths
            with open(args.puzzles_list, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            # If lines contain paths, extract problem IDs
            puzzle_ids = []
            for line in lines:
                if '/' in line or '\\' in line:
                    # Extract problem ID from path
                    problem_id = os.path.basename(line)
                    puzzle_ids.append(problem_id)
                else:
                    # Assume it's already a problem ID
                    puzzle_ids.append(line)
    else:
        # Use all available action programs
        puzzle_ids = list(action_programs.keys())[:50]  # Limit for testing
    
    logging.info(f"[LOGO] Processing {len(puzzle_ids)} puzzles from action programs")
    
    # Process each problem
    processed_count = 0
    total_examples = 0
    command_type_stats = {'line': 0, 'arc': 0, 'unknown': 0}
    shape_type_stats = {}
    
    for problem_id in tqdm(puzzle_ids, desc="Processing LOGO Problems (Action-Only)"):
        if problem_id not in action_programs:
            logging.warning(f"Problem {problem_id} not found in action_programs.")
            continue
            
        # ACTION PROGRAMS ONLY: No derived_labels needed
        pdata = get_problem_data(problem_id, action_programs)
        if pdata is None:
            logging.warning(f"Failed to get problem data for {problem_id}")
            continue

        try:
            problem_records = pdata['records']
            total_examples += len(problem_records)
            
            # Collect statistics about action commands
            for record in problem_records:
                if 'features' in record and 'command_type_counts' in record['features']:
                    for cmd_type, count in record['features']['command_type_counts'].items():
                        command_type_stats[cmd_type] = command_type_stats.get(cmd_type, 0) + count
                if 'features' in record and 'shape_type_counts' in record['features']:
                    for shape_type, count in record['features']['shape_type_counts'].items():
                        shape_type_stats[shape_type] = shape_type_stats.get(shape_type, 0) + count
            
            logging.info(f"[LOGO] Processing {problem_id}: {len(problem_records)} records ({pdata['total_positive']} pos, {pdata['total_negative']} neg)")
            
            # The new function returns a dictionary containing the scene graphs and all objects
            processed_data = await _process_single_problem(problem_id, problem_records, None, args=args)
            
            if not processed_data or not processed_data.get('scene_graphs'):
                logging.warning(f"[LOGO] No scene graphs were generated for problem {problem_id}. Skipping.")
                continue
            
            # The core logic is now inside _process_single_problem. 
            # This script's role is primarily to orchestrate the calls and save outputs.
            final_graphs = processed_data['scene_graphs']
            logging.info(f"[LOGO] Successfully processed problem {problem_id}, generated {len(final_graphs)} scene graphs.")
            processed_count += 1

        except Exception as e:
            logging.error(f"Error processing problem {problem_id}: {e}")
            logging.error(traceback.format_exc())
            continue
    
    # Log comprehensive statistics
    logging.info(f"[LOGO] ===== ACTION-ONLY PIPELINE COMPLETED =====")
    logging.info(f"[LOGO] Successfully processed {processed_count} out of {len(puzzle_ids)} problems")
    logging.info(f"[LOGO] Total examples processed: {total_examples}")
    logging.info(f"[LOGO] Command type statistics: {command_type_stats}")
    logging.info(f"[LOGO] Shape type statistics: {shape_type_stats}")
    logging.info(f"[LOGO] All action command categories parsed: {list(command_type_stats.keys())}")
    logging.info(f"[LOGO] All shape categories parsed: {list(shape_type_stats.keys())}")
    logging.info(f"[LOGO] ==============================================")
    
    return processed_count, total_examples, command_type_stats, shape_type_stats

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
    STATE-OF-THE-ART: Loads predicate definitions including abstract BONGARD-LOGO style predicates,
    and integrates adaptive thresholds and advanced reasoning predicates.
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
    if hasattr(adaptive_thresholds, 'get_current_thresholds'):
        learned_params = adaptive_thresholds.get_current_thresholds()
    elif hasattr(adaptive_thresholds, 'thresholds'):
        learned_params = adaptive_thresholds.thresholds
    else:
        learned_params = {}
    logging.info(f"Loaded adaptive predicate parameters: {learned_params}")

    args = None
    if 'args' in globals():
        args = globals()['args']
    elif hasattr(sys.modules['__main__'], 'args'):
        args = getattr(sys.modules['__main__'], 'args')

    # === STATE-OF-THE-ART: INTEGRATE ADVANCED PREDICATES ===
    # Load advanced predicates from registry
    for predicate_name, predicate_func in ADVANCED_PREDICATE_REGISTRY.items():
        registry[predicate_name] = predicate_func
    
    logging.info(f"Loaded {len(ADVANCED_PREDICATE_REGISTRY)} STATE-OF-THE-ART advanced predicates: {list(ADVANCED_PREDICATE_REGISTRY.keys())}")

    # LOGO mode: restrict to LOGO semantics only
    def next_action(a, b, params):
        return a.get('parent_shape_id') is not None and a.get('parent_shape_id') == b.get('parent_shape_id') and a.get('action_index', -2) + 1 == b.get('action_index', -1)

    def turn_left(a, b, params):
        return next_action(a, b, params) and b.get('turn_direction') == 'left'

    def turn_right(a, b, params):
        return next_action(a, b, params) and b.get('turn_direction') == 'right'

    def repeat_action(a, b, params):
        return a.get('parent_shape_id') == b.get('parent_shape_id') and a.get('command') is not None and a.get('command') == b.get('command') and a.get('object_id') != b.get('object_id')

    def adjacent_endpoints(a, b, params):
        ep_a = a.get('endpoints', [None, None])[1]
        sp_b = b.get('endpoints', [None, None])[0]
        if ep_a is None or sp_b is None:
            return False
        try:
            diff = np.linalg.norm(np.array(ep_a) - np.array(sp_b))
        except Exception:
            diff = 0.0
        return diff < params.get('angle_tol', 10.0)

    def same_program_type(a, b, params):
        return a.get('parent_shape_id') == b.get('parent_shape_id') and a.get('programmatic_label') is not None and a.get('programmatic_label') == b.get('programmatic_label')

    def stroke_type_sim(a, b, params):
        return a.get('parent_shape_id') == b.get('parent_shape_id') and a.get('stroke_type') is not None and a.get('stroke_type') == b.get('stroke_type')

    def junction(a, b, params):
        endpoints_a = [a.get('start_point'), a.get('end_point')]
        endpoints_b = [b.get('start_point'), b.get('end_point')]
        shared = [pt for pt in endpoints_a if pt is not None and pt in endpoints_b]
        if not shared:
            return False
        return a.get('object_id') != b.get('object_id')

    def intersects(a, b, params):
        verts_a = a.get('vertices', [])
        verts_b = b.get('vertices', [])
        if len(verts_a) < 2 or len(verts_b) < 2:
            return False
        try:
            from shapely.geometry import LineString
            line_a = LineString(verts_a)
            line_b = LineString(verts_b)
            return line_a.intersects(line_b) and not line_a.touches(line_b)
        except Exception:
            return False

    def part_of_motif(a, b, params):
        return False

    def motif_similarity(a, b, params):
        return False

    def symmetric_with(a, b, params):
        return False

    # STATE-OF-THE-ART: Import abstract predicates from config
    from src.scene_graphs_building.config import ABSTRACT_PREDICATES, BASIC_LOGO_PREDICATES
    
    # Add abstract predicates to the registry
    for pred_name, pred_func in ABSTRACT_PREDICATES.items():
        registry[pred_name] = pred_func
    
    # Add basic LOGO predicates
    for pred_name, pred_func in BASIC_LOGO_PREDICATES.items():
        registry[pred_name] = pred_func

    if args is not None and hasattr(args, 'mode') and args.mode == 'logo':
        all_candidate_predicates = {
            'next_action': next_action,
            'turn_left': turn_left,
            'turn_right': turn_right,
            'repeat_action': repeat_action,
            # 'forms_loop': forms_loop,  # Undefined, remove or implement
            # 'length_sim': length_sim,  # Undefined, remove or implement
            # 'angle_sim': angle_sim,    # Undefined, remove or implement
            # 'angle_between': angle_between,  # Undefined, remove or implement
            'adjacent_endpoints': adjacent_endpoints,
            'same_program_type': same_program_type,
            'stroke_type_sim': stroke_type_sim,
            'junction': junction,
            'intersects': intersects,
            'part_of_motif': part_of_motif,
            'motif_similarity': motif_similarity,
            'symmetric_with': symmetric_with,
        }
        for name, func in all_candidate_predicates.items():
            if callable(func):
                registry[name] = lambda a, b, params, func=func: func(a, b, params)
            else:
                logging.warning(f"[LOGO] Predicate '{name}' is not callable and will be skipped.")
        for entry in predicate_defs:
            name = entry['predicate']
            func = all_candidate_predicates.get(name)
            if callable(func):
                if name not in registry:
                    registry[name] = lambda a, b, params, func=func: func(a, b, params)
            else:
                logging.warning(f"[LOGO] Predicate '{name}' from edge_types.json has no defined logic in build_scene_graphs.py or is not callable. Skipping.")
    else:
        # SOTA mode predicate logic is not implemented in this patch. Uncomment and complete if needed.
        # all_candidate_predicates = {
        #     'curvature_sim': lambda a, b, params: abs(a.get('curvature', 0.0) - b.get('curvature', 0.0)) < params.get('curvature_tol', 0.2),
        #     ...
        # }
        # for name, func in all_candidate_predicates.items():
        #     if callable(func):
        #         registry[name] = lambda a, b, params, func=func: func(a, b, params)
        #     else:
        #         logging.warning(f"Predicate '{name}' is not callable and will be skipped.")
        # for entry in predicate_defs:
        #     name = entry['predicate']
        #     func = all_candidate_predicates.get(name)
        #     if callable(func):
        #         if name not in registry:
        #             registry[name] = lambda a, b, params=learned_params, func=func: func(a, b, params)
        #     else:
        #         logging.warning(f"Predicate '{name}' from edge_types.json has no defined logic in build_scene_graphs.py or is not callable. Skipping.")
        pass
    return registry



# --- Utility Functions ---
def _robust_edge_unpack(edges):
    """Yield (u, v, k, data) for all edge tuples, handling 2, 3, 4-item cases and non-dict data."""
    for edge in edges:
        if len(edge) == 4:
            u, v, k, data = edge
        elif len(edge) == 3:
            u, v, data = edge
            k = None
        elif len(edge) == 2:
            u, v = edge
            k = None
            data = {}
        else:
            logging.warning(f"Skipping unexpectedly short/long edge tuple: {edge}")
            continue
        if not isinstance(data, dict):
            logging.warning(f"Edge data not dict: {repr(data)} for edge {edge}; using empty dict.")
            data = {}
        yield u, v, k, data

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
        hierarchical predicate refinement, and STATE-OF-THE-ART contrastive reasoning.
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
        node_objects = []  # For contrastive analysis
        
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

                # Store node object data for contrastive analysis
                node_objects.append({
                    'id': node_id,
                    'vertices': node_data.get('vertices', []),
                    'bbox': node_data.get('bbox', []),
                    'stroke_count': node_data.get('stroke_count', len(node_data.get('vertices', []))),
                    'compactness': node_data.get('compactness', 0.0),
                    'aspect_ratio': node_data.get('aspect_ratio', 1.0),
                    'orientation': node_data.get('orientation', 0.0),
                    'is_closed': node_data.get('is_closed', False),
                    'curvature_score': node_data.get('curvature_score', 0.0),
                    'area': node_data.get('area', 0.0),
                    'length': node_data.get('length', 0.0),
                    'shape_label': node_data.get('shape_label', ''),
                    'category': node_data.get('category', ''),
                    'action_program': node_data.get('action_program', []),
                    'object_type': node_data.get('object_type', 'unknown')
                })

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
        
        # === STATE-OF-THE-ART: CONTRASTIVE PREDICATE INDUCTION ===
        contrastive_results = {}
        analogical_patterns = []
        program_rules = []
        
        if hasattr(self.feature_extractor, 'compute_contrastive_features') and len(node_objects) >= 2:
            try:
                # Group objects by their labels for contrastive analysis
                label_groups = {}
                for obj in node_objects:
                    label = obj.get('shape_label', obj.get('category', 'unknown'))
                    if label not in label_groups:
                        label_groups[label] = []
                    label_groups[label].append(obj)
                
                # Perform contrastive analysis between different label groups
                label_keys = list(label_groups.keys())
                if len(label_keys) >= 2:
                    positive_objects = label_groups[label_keys[0]]
                    negative_objects = label_groups[label_keys[1]]
                    
                    contrastive_results = self.feature_extractor.compute_contrastive_features(
                        positive_objects, negative_objects
                    )
                    
                    if contrastive_results:
                        logging.info(f"Contrastive analysis found {len(contrastive_results.get('discriminative_rules', []))} discriminative rules")
                        
                        # Add contrastive predicates to graph
                        for rule in contrastive_results.get('discriminative_rules', []):
                            if rule['confidence'] > 0.7:  # High confidence rules
                                # Add as graph metadata
                                if 'contrastive_rules' not in G.graph:
                                    G.graph['contrastive_rules'] = []
                                G.graph['contrastive_rules'].append(rule)
                
            except Exception as e:
                logging.warning(f"Contrastive analysis failed: {e}")
        
        # === STATE-OF-THE-ART: ANALOGICAL REASONING ===
        if hasattr(self.feature_extractor, 'find_analogical_patterns') and len(node_objects) >= 4:
            try:
                # Split objects into two sets for analogical comparison
                mid_point = len(node_objects) // 2
                set_a = node_objects[:mid_point]
                set_b = node_objects[mid_point:]
                
                analogical_patterns = self.feature_extractor.find_analogical_patterns(set_a, set_b)
                
                if analogical_patterns:
                    logging.info(f"Analogical reasoning found {len(analogical_patterns)} patterns")
                    G.graph['analogical_patterns'] = analogical_patterns
                    
            except Exception as e:
                logging.warning(f"Analogical reasoning failed: {e}")
        
        # === STATE-OF-THE-ART: PROGRAM SYNTHESIS FOR RULE EXTRACTION ===
        if hasattr(self.feature_extractor, 'extract_program_rules'):
            try:
                action_programs = [obj.get('action_program', []) for obj in node_objects]
                object_labels = [obj.get('shape_label', obj.get('category', 'unknown')) for obj in node_objects]
                
                program_rules = self.feature_extractor.extract_program_rules(action_programs, object_labels)
                
                if program_rules:
                    logging.info(f"Program synthesis extracted {len(program_rules)} rules")
                    G.graph['program_rules'] = program_rules
                    
            except Exception as e:
                logging.warning(f"Program rule extraction failed: {e}")
        
        # Knowledge fusion and Hierarchical predicate refinement now operate on the problem-level graph
        # This part needs to iterate through existing edges and potentially add new ones based on features
        # The original code's `enhanced_relationships` and `refined_relationships` are for a list of relationships.
        # We need to apply this to the edges of the NetworkX graph G.

        # Step 1: Knowledge Fusion (semantic enrichment of existing edges)
        # Collect existing relationships to process
        current_relationships = []
        if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
            for u, v, k, data in _robust_edge_unpack(G.edges(keys=True, data=True)):
                current_relationships.append({
                    'subject_id': u,
                    'object_id': v,
                    'predicate': data.get('predicate', 'unknown'),
                    'source': data.get('source', 'spatial')
                })
        else:
            for u, v, k, data in _robust_edge_unpack(G.edges(data=True)):
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
                'relationship_accuracy': validation_results.get('relationship_accuracy_score', 0.0),
                # STATE-OF-THE-ART quality metrics
                'contrastive_rules_count': len(contrastive_results.get('discriminative_rules', [])),
                'analogical_patterns_count': len(analogical_patterns),
                'program_rules_count': len(program_rules),
                'has_advanced_reasoning': len(contrastive_results.get('discriminative_rules', [])) > 0 or len(analogical_patterns) > 0 or len(program_rules) > 0
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
    if shape == "arc" and '-' in rest:
        radius, angle = rest.split('-',1)
        try:
            return {'type':'arc', 'mode':mode, 'radius':float(radius), 'angle':float(angle)}
        except Exception:
            return None
    if shape == "turn" and '-' in rest:
        angle = rest
        try:
            return {'type':'turn', 'mode':mode, 'angle':float(angle)}
        except Exception:
            return None
    # Extend for other commands as needed
    return None


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

# REMOVED: assign_object_type function that used vertices without action programs
# All object type assignment now happens in process_single_problem.py using action commands only

def process_problem_batch_sync_wrapper(list_of_problem_records: List[List[Dict[str, Any]]], feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate, decompose_polygons: bool, adaptive_thresholds: AdaptivePredicateThresholds, args=None):
    # Synchronous wrapper to run async processing for a batch of problems.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for problem_records in list_of_problem_records:
        if problem_records: # Ensure problem_records is not empty
            problem_id = problem_records[0].get('problem_id', 'unknown')
            tasks.append(_process_single_problem(problem_id, problem_records, feature_cache, args=args))
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
    parser.add_argument('--use-vl', action='store_true', default=True, help='Enable enhanced vision-language (CLIP) features with missing data calculations')
    parser.add_argument('--use-gnn', action='store_true', help='Enable GNN/Transformer reasoning in scene graph')
    parser.add_argument('--use-motifs', action='store_true', default=True, help='Enable enhanced motif mining with comprehensive feature calculations')
    parser.add_argument('--use-roi', action='store_true', default=True, help='Use ROI (bounding box/mask) for CLIP vision-language edges (SOTA)')
    parser.add_argument('--enhanced-viz', action='store_true', default=True, help='Generate enhanced multi-view visualizations with missing data analysis')
    parser.add_argument('--mode', type=str, default='sota', choices=['sota', 'logo'], help='Pipeline mode: "sota" (default, all features) or "logo" (LOGO-only, NVLabs baseline)')
    parser.add_argument('--puzzles-list', type=str, required=False, help='Path to puzzle list for LOGO mode')
    parser.add_argument('--action-base-dir', type=str, required=False, help='Base directory for action program JSONs for LOGO mode')
    parser.add_argument('--use-semantic', action='store_true', default=True, help='Enable semantic action parsing for Bongard-relevant features')
    args = parser.parse_args()
    # [LOGO] Log all input arguments and label values at the start
    if hasattr(args, 'mode') and args.mode == 'logo':
    
        logging.info(f"[LOGO] All input arguments: {vars(args)}")
        # Print all label values loaded
        try:
            with open(args.labels, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                logging.info(f"[LOGO] All label values from {args.labels}: {json.dumps(label_data, indent=2, ensure_ascii=False)[:2000]}{'...TRUNCATED...' if len(json.dumps(label_data))>2000 else ''}")
        except Exception as e:
            logging.warning(f"[LOGO] Could not load or print all label values: {e}")
    global kb
    kb = ConceptNetClient.get()
    if kb is None:
        logging.warning("ConceptNet KB client could not be initialized. Commonsense edges will be skipped.")
    # LOGO mode: allow advanced techniques if explicitly enabled
    if args.mode == 'logo':
        # Only disable if not explicitly enabled
        if not args.use_vl and not args.use_gnn and not args.use_motifs and not args.use_roi:
            args.use_vl = False
            args.use_gnn = False
            args.use_motifs = False
            kb = None
            logging.info("LOGO mode: VL, GNN, motifs, and KB edges are disabled. Only programmatic and minimal geometry predicates will be used.")
        else:
            logging.info(f"LOGO mode with advanced techniques enabled: VL={args.use_vl}, GNN={args.use_gnn}, motifs={args.use_motifs}, ROI={args.use_roi}")
            # Keep kb enabled if using advanced techniques

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
    if args.mode == 'logo':
        recall_evaluator = None
        drift_visualizer = None
        logging.info("LOGO mode: Using simple accuracy metric for validation. SGScore/RecallAtK/DriftVisualizer are disabled.")
    else:
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
    if recall_evaluator:
        recall_evaluator.predicate_classes = list(PREDICATES.keys()) # Update recall_evaluator with all predicates

    # Compute hash for cache validation
    # Get adaptive thresholds for hashing
    if hasattr(adaptive_thresholds, 'get_learned_thresholds'):
        adaptive_params = adaptive_thresholds.get_learned_thresholds()
    elif hasattr(adaptive_thresholds, 'thresholds'):
        adaptive_params = adaptive_thresholds.thresholds
    else:
        adaptive_params = {}
    # Robustly ensure PREDICATES is a dict before using .keys()
    if PREDICATES is None or not isinstance(PREDICATES, dict):
        PREDICATES = {}
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
        augmented_data = load_json_data(args.aug)
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
        derived_labels = load_json_data(args.labels)
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
    
    # Check if we should use LOGO mode pipeline
    if args.mode == 'logo':
        logging.info("Using specialized LOGO mode pipeline")
        await run_bongard_logo_scene_graph_pipeline(args)
        return  # Exit after LOGO pipeline completes
    
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
                import torch
                from src.reasoner.gnn_reasoner import GNNReasoner
                # Prepare torch_geometric Data objects and labels
                graph_data_list = []
                graph_labels = []
                for g_result in graphs:
                    scene_graph = g_result.get('scene_graph')
                    if scene_graph is not None and hasattr(scene_graph, 'nodes'):
                        # Calculate correct feature dimension: type_onehot + features + validity + action + vl
                        # 5 + 10 + 10 + 4 + 10 = 39 features total
                        expected_dim = 39
                        gnn = GNNReasoner(in_dim=expected_dim)
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

    # Generate comprehensive analysis report if enhanced features were used
    if getattr(args, 'enhanced_viz', True) and (args.use_vl or args.use_motifs):
        try:
            logging.info("Generating comprehensive multi-puzzle analysis...")
            
            from scripts.scene_graph_visualization import (
                create_multi_puzzle_visualization, 
                create_missing_data_analysis
            )
            
            # Create multi-puzzle overview
            if os.path.exists(args.feedback_dir):
                multi_viz_path = create_multi_puzzle_visualization(
                    csv_dir=args.feedback_dir,
                    out_dir=args.feedback_dir,
                    puzzle_pattern="*",
                    max_puzzles=12
                )
                
                # Create missing data analysis  
                analysis_paths = create_missing_data_analysis(
                    csv_dir=args.feedback_dir,
                    out_dir=args.feedback_dir,
                    puzzle_pattern="*"
                )
                
                if multi_viz_path:
                    print(f"\n[Enhanced Analysis] Multi-puzzle visualization: {multi_viz_path}")
                if analysis_paths:
                    print(f"[Enhanced Analysis] Missing data analysis: {analysis_paths[0]}")
                    print(f"[Enhanced Analysis] Completeness report: {analysis_paths[1]}")
                    
        except Exception as e:
            logging.warning(f"Failed to generate comprehensive analysis: {e}")

    # Log overall performance stats
    cache_stats = feature_cache.get_cache_stats()
    logging.info("Processing complete!")
    logging.info(f"Cache efficiency: Memory: {cache_stats.get('memory_usage_mb', 0):.1f}MB used, GPU: {cache_stats.get('gpu_usage_mb', 0):.1f}MB used")



if __name__ == "__main__":
    asyncio.run(main())

