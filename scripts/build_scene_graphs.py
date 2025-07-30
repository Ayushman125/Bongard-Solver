# --- Typing Imports for Function Annotations ---
from typing import List, Dict, Any
# --- Standard Library Imports ---
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
from collections import Counter, defaultdict

# --- Third-Party Library Imports ---
import cv2
import networkx as nx
import numpy as np
from PIL import Image
from scipy import stats
from shapely.geometry import Polygon
from tqdm import tqdm

# Conditional imports for torch/kornia/torchvision
TORCH_KORNIA_AVAILABLE = False
try:
    import torch
    import kornia
    import torchvision
    TORCH_KORNIA_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch, Kornia, or Torchvision not found. Real feature extraction will be disabled.")

# Conditional imports for graphtype
GRAPHTYPE_AVAILABLE = False
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
EPS = 1e-3
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
    return abs(a - b) < tol or abs(abs(a - b) - 180) < tol

def physics_contact(poly_a, poly_b):
    """Robust check for physical contact between two polygons."""
    return poly_a.touches(poly_b)

def load_predicates():
    """Loads predicate definitions from schemas/edge_types.json or uses built-in defaults."""
    predicate_file = os.path.join(os.path.dirname(__file__), '..', 'schemas', 'edge_types.json')
    if not os.path.exists(predicate_file):
        logging.warning("schemas/edge_types.json not found. Using built-in predicates.")
        return None
    try:
        with open(predicate_file, 'r', encoding='utf-8') as f:
            predicate_defs = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding predicates from {predicate_file}: {e}. Using built-in defaults.")
        return None

    registry = {}
    for entry in predicate_defs:
        name = entry['predicate']
        # Map each predicate to its logic (use built-in if available, else skip)
        if name == 'left_of':
            registry[name] = lambda a, b: a.get('cx', 0) + EPS < b.get('cx', 0)
        elif name == 'right_of':
            registry[name] = lambda a, b: a.get('cx', 0) > b.get('cx', 0) + EPS
        elif name == 'above':
            registry[name] = lambda a, b: a.get('cy', 0) + EPS < b.get('cy', 0)
        elif name == 'below':
            registry[name] = lambda a, b: a.get('cy', 0) > b.get('cy', 0) + EPS
        elif name == 'contains':
            registry[name] = lambda a, b: Polygon(a['vertices']).contains(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False
        elif name == 'inside':
            registry[name] = lambda a, b: Polygon(b['vertices']).contains(Polygon(a['vertices'])) if 'vertices' in a and 'vertices' in b else False
        elif name == 'supports':
            registry[name] = lambda a, b: (physics_contact(Polygon(a['vertices']), Polygon(b['vertices'])) and a.get('cy', 0) > b.get('cy', 0)) if 'vertices' in a and 'vertices' in b else False
        elif name == 'supported_by':
            registry[name] = lambda a, b: (physics_contact(Polygon(b['vertices']), Polygon(a['vertices'])) and b.get('cy', 0) > a.get('cy', 0)) if 'vertices' in a and 'vertices' in b else False
        elif name == 'touches':
            registry[name] = lambda a, b: Polygon(a['vertices']).touches(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False
        elif name == 'overlaps':
            registry[name] = lambda a, b: Polygon(a['vertices']).overlaps(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False
        elif name == 'parallel_to':
            registry[name] = lambda a, b: parallel(a.get('orientation', 0), b.get('orientation', 0))
        elif name == 'perpendicular_to':
            registry[name] = lambda a, b: abs(abs(a.get('orientation', 0) - b.get('orientation', 0)) - 90) < 7
        elif name == 'aligned_left':
            registry[name] = lambda a, b: abs(a.get('bbox', [0,0,0,0])[0] - b.get('bbox', [0,0,0,0])[0]) < EPS
        elif name == 'aligned_right':
            registry[name] = lambda a, b: abs(a.get('bbox', [0,0,0,0])[2] - b.get('bbox', [0,0,0,0])[2]) < EPS
        elif name == 'aligned_top':
            registry[name] = lambda a, b: abs(a.get('bbox', [0,0,0,0])[1] - b.get('bbox', [0,0,0,0])[1]) < EPS
        elif name == 'aligned_bottom':
            registry[name] = lambda a, b: abs(a.get('bbox', [0,0,0,0])[3] - b.get('bbox', [0,0,0,0])[3]) < EPS
        elif name == 'proximal_to':
            registry[name] = lambda a, b: np.linalg.norm(np.array([a.get('cx',0), a.get('cy',0)]) - np.array([b.get('cx',0), b.get('cy',0)])) < 50
        elif name == 'contains_text':
            registry[name] = lambda a, b: False # KB-based, handled in add_commonsense_edges
        elif name == 'is_arrow_for':
            registry[name] = lambda a, b: False # KB-based, handled in add_commonsense_edges
        elif name == 'has_sides':
            registry[name] = lambda a, b: False # KB-based, handled in add_commonsense_edges
        elif name == 'same_shape':
            registry[name] = lambda a, b: a.get('shape_label') == b.get('shape_label')
        elif name == 'symmetry_axis':
            registry[name] = lambda a, b: abs(a.get('symmetry_axis',0) - b.get('orientation',0)) < 7
        elif name == 'same_color':
            registry[name] = lambda a, b: np.all(np.abs(np.array(a.get('color',[0,0,0])) - np.array(b.get('color',[0,0,0]))) < 10)
        else:
            registry[name] = lambda a, b: False
    return registry

PREDICATES = load_predicates() or {
    'left_of':     lambda a, b: a.get('cx', 0) + EPS < b.get('cx', 0),
    'right_of':    lambda a, b: a.get('cx', 0) > b.get('cx', 0) + EPS,
    'above':       lambda a, b: a.get('cy', 0) + EPS < b.get('cy', 0),
    'below':       lambda a, b: a.get('cy', 0) > b.get('cy', 0) + EPS,
    'contains':    lambda a, b: Polygon(a['vertices']).contains(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
    'inside':      lambda a, b: Polygon(b['vertices']).contains(Polygon(a['vertices'])) if 'vertices' in a and 'vertices' in b else False,
    'supports':    lambda a, b: (physics_contact(Polygon(a['vertices']), Polygon(b['vertices'])) and a.get('cy', 0) > b.get('cy', 0)) if 'vertices' in a and 'vertices' in b else False,
    'touches':     lambda a, b: Polygon(a['vertices']).touches(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
    'overlaps':    lambda a, b: Polygon(a['vertices']).overlaps(Polygon(b['vertices'])) if 'vertices' in a and 'vertices' in b else False,
    'parallel_to': lambda a, b: parallel(a.get('orientation', 0), b.get('orientation', 0)),
}

def add_predicate_edges(G):
    """Iterates through all node pairs and adds edges based on the predicate registry."""
    node_list = list(G.nodes(data=True))
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if i == j:
                continue
            for pred, fn in PREDICATES.items():
                try:
                    # Use binarized mask and object attributes only
                    if fn(data_u, data_v):
                        G.add_edge(u, v, predicate=pred, source='spatial')
                except Exception as e:
                    logging.debug(f"Predicate function '{pred}' failed for ({u}, {v}): {e}")
                    continue

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
def compute_physics_attributes(node_data):
    """Computes robust physics attributes and asserts their domain validity."""
    vertices = node_data.get('vertices')
    if not vertices or len(vertices) < 3:
        node_data.update({'area': 0.0, 'inertia': 0.0, 'convexity': 0.0, 'cx': 0.0, 'cy': 0.0})
        return

    poly = Polygon(vertices)
    area = poly.area

    assert area >= 0.0, f"Negative area {area} for node {node_data.get('id')}"

    try:
        # Shapely's moment_of_inertia is about the centroid
        inertia = poly.moment_of_inertia
    except Exception:
        inertia = 0.0 # Default to 0 on error

    convexity = poly.convex_hull.area / area if area > 0 else 0.0
    centroid = poly.centroid

    node_data.update({
        'area': float(area),
        'inertia': float(inertia),
        'convexity': float(convexity),
        'cx': centroid.x,
        'cy': centroid.y,
    })

# --- Graph Building Functions with/without Validation ---
@graph_validate(
    NodeData(id=str, area=float, inertia=float, convexity=float),
    EdgeData(predicate=str, source=str),
)
def build_graph_validated(record):
    """Builds a single scene graph with runtime schema validation."""
    G = nx.MultiDiGraph()
    for obj in record.get('objects', []):
        compute_physics_attributes(obj)
        G.add_node(obj['id'], **obj)
    add_predicate_edges(G)
    add_commonsense_edges(G)
    return G

def build_graph_unvalidated(record):
    """Builds a single scene graph without runtime schema validation."""
    G = nx.MultiDiGraph()
    for obj in record.get('objects', []):
        compute_physics_attributes(obj)
        G.add_node(obj['id'], **obj)
    add_predicate_edges(G)
    add_commonsense_edges(G)
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
                    if polygons[i].overlaps(polygons[j]):
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
            total_possible_triples += len(G.nodes)**2 - len(G.nodes)

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

def profile_optimal_batch_size(records, build_func):
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
                for idx, rec in enumerate(problem_recs):
                    merged_rec['objects'].append({
                        'id': f"{pid}_{idx}",
                        'vertices': rec.get('vertices', []),
                        **rec.get('features', {}),
                        'label': rec.get('label', ''),
                        'shape_label': rec.get('label', ''),
                        'category': rec.get('category', '')
                    })
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

    def _create_mask_from_vertices(self, vertices, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        vertices_array = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [vertices_array], 255)
        return mask

    def _create_mask_from_bbox(self, bbox, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # Ensure coordinates are within image bounds
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(image_shape[0], y2), min(image_shape[1], x2)
        mask[y1:y2, x1:x2] = 255
        return mask

    async def build_enhanced_scene_graph(self, image_path: str, base_scene_graph: dict) -> dict:
        """
        Builds an enhanced scene graph by adding real features, knowledge fusion,
        and hierarchical predicate refinement.
        The base_scene_graph now contains a NetworkX graph for the entire problem.
        """
        if not os.path.exists(image_path):
            logging.error(f"Representative image file not found: {image_path}. Skipping enhanced graph build.")
            return {'scene_graph': base_scene_graph, 'quality_metrics': {}}

        try:
            image = Image.open(image_path).convert("RGB") # Ensure RGB for consistent processing
            image_np = np.array(image)
        except Exception as e:
            logging.error(f"Failed to load or process representative image {image_path}: {e}. Skipping enhanced graph build.")
            return {'scene_graph': base_scene_graph, 'quality_metrics': {}}

        # The base_scene_graph now contains the NetworkX graph in 'graph' key
        G = base_scene_graph.get('graph')
        if not G:
            logging.error("No NetworkX graph provided in base_scene_graph. Cannot enhance.")
            return {'scene_graph': base_scene_graph, 'quality_metrics': {}}

        object_features = {} # To store extracted features for nodes in G

        # Iterate over nodes in the NetworkX graph to extract features
        for node_id, node_data in list(G.nodes(data=True)): # Use list() to allow modification during iteration
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
                    # Update the node data in the graph with the extracted features
                    G.nodes[node_id]['real_features'] = features # Store raw features
                    logging.debug(f"Extracted features for node {node_id}: shape {features.shape}")
                except Exception as e:
                    # Add detailed tensor shape/type logging if possible
                    import traceback
                    logging.error(f"Feature extraction failed for node {node_id}: {e}. No features for this node.")
                    # Log the traceback for the original error
                    logging.error(traceback.format_exc())
            else:
                logging.debug(f"Feature extractor not available or mask is None for node {node_id}. Skipping feature extraction.")

        # Knowledge fusion and Hierarchical predicate refinement now operate on the problem-level graph
        # This part needs to iterate through existing edges and potentially add new ones based on features
        # The original code's `enhanced_relationships` and `refined_relationships` are for a list of relationships.
        # We need to apply this to the edges of the NetworkX graph G.

        # Step 1: Knowledge Fusion (semantic enrichment of existing edges)
        # This part is tricky. The original `get_enriched_relationships` expects subject/object *names*
        # and a list of predicates. We have node IDs and attributes.
        # Assuming `knowledge_fusion` can work with node data directly or needs adaptation.
        # For now, I'll adapt it to iterate over existing edges in G.
        
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
                    # This might need adjustment based on the actual MultiSourceKnowledgeFusion API
                    enriched_rels = await self.knowledge_fusion.get_enriched_relationships(
                        subject_label, object_label, [rel.get('predicate', '')]
                    )
                    # Add enriched relations back to the graph or a temporary list
                    for erel in enriched_rels:
                        # Ensure enriched relations have subject_id and object_id for graph
                        erel['subject_id'] = subject_id
                        erel['object_id'] = object_id
                        enhanced_relationships.append(erel)
                        # Optionally, update existing edge or add new one to G
                        # G.add_edge(subject_id, object_id, predicate=erel['predicate'], source='kb_fusion', confidence=erel.get('confidence'))
                except Exception as e:
                    logging.warning(f"Knowledge fusion failed for relationship {rel}: {e}. Appending original relationship.")
                    enhanced_relationships.append(rel)
            else:
                logging.debug(f"Skipping knowledge fusion for relationship {rel} due to missing features.")
                enhanced_relationships.append(rel) # Append original if features are missing

        # Clear existing edges and add enhanced ones, or add new edges
        # For simplicity, let's just add new edges if they don't exist, or update existing ones.
        # To avoid duplicates, we might want to rebuild edges or manage them carefully.
        # For now, let's just update the `relationships` list that will be part of the final output.
        # The NetworkX graph `G` itself is modified in place (nodes have 'real_features').
        # The relationships in `final_scene_graph['relationships']` will be the refined ones.

        # Step 2: Hierarchical predicate refinement
        refined_relationships = []
        # This part of the original code was designed for a list of relationships.
        # It needs to be re-thought for a NetworkX graph where edges are already present.
        # The `predict_with_bayesian_inference` takes features.
        # We need to iterate over pairs of nodes in G and apply hierarchical prediction.
        
        # This is a conceptual adaptation. The exact API of HierarchicalPredicatePredictor
        # and how it interacts with NetworkX edges might need more specific implementation.
        # For now, I'll simulate applying it to the `enhanced_relationships` list.
        
        # Create a mapping from (u,v) to existing edge data for easy lookup
        edge_data_map = {}
        for u, v, data in G.edges(data=True):
            edge_data_map[(u,v)] = data

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
                        # Optionally, update edge in G (e.g., add new predicate or update confidence)
                        # G.add_edge(subject_id, object_id, predicate=r_rel['predicate'], source='hierarchical', confidence=r_rel.get('confidence'))
                except Exception as e:
                    logging.warning(f"Hierarchical predicate prediction failed for rel {rel}: {e}. Appending original.")
                    refined_relationships.append(rel)
            else:
                logging.debug(f"Skipping hierarchical prediction for rel {rel} due to missing features.")
                refined_relationships.append(rel)

        # Final validation using SGScoreValidator
        # This expects a PIL Image and a dict with 'objects' and 'relationships'.
        # The 'objects' here should be the list of object dictionaries, not the NetworkX nodes.
        # The relationships should be the `refined_relationships`.
        
        # Prepare objects list for SGScoreValidator (from NetworkX nodes)
        objects_for_validator = [G.nodes[node_id] for node_id in G.nodes()]
        
        validation_results = await self.sgcore_validator.validate_scene_graph(
            image, {'objects': objects_for_validator, 'relationships': refined_relationships}
        )

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
    import matplotlib.pyplot as plt
    import networkx as nx
    os.makedirs(feedback_dir, exist_ok=True)
    img_save_path = os.path.join(feedback_dir, f"{base_name}_input.png")
    mask_save_path = os.path.join(feedback_dir, f"{base_name}_mask.png")
    side_by_side_path = os.path.join(feedback_dir, f"{base_name}_side_by_side.png")
    graph_img_path = os.path.join(feedback_dir, f"{base_name}_graph.png")
    img_graph_side_by_side_path = os.path.join(feedback_dir, f"{base_name}_img_graph.png")

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
        # Draw the graph using networkx and matplotlib
        plt.figure(figsize=(5, 5))
        pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else nx.random_layout(G)
        node_labels = {n: str(n) for n in G.nodes()}
        edge_labels = {(u, v): d.get('predicate', '') for u, v, d in G.edges(data=True)}
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(graph_img_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # Combine real image and graph viz side by side
        try:
            graph_img = cv2.imread(graph_img_path)
            if graph_img is not None:
                # Resize graph image to match height of real image
                h_img = img_bgr.shape[0]
                h_graph, w_graph = graph_img.shape[:2]
                scale = h_img / h_graph
                new_w = int(w_graph * scale)
                graph_img_resized = cv2.resize(graph_img, (new_w, h_img), interpolation=cv2.INTER_AREA)
                img_graph_side = cv2.hconcat([img_bgr, graph_img_resized])
                cv2.imwrite(img_graph_side_by_side_path, img_graph_side)
        except Exception as e:
            logging.warning(f"Failed to create side-by-side image and graph: {e}")

def remap_path(path):
    """Remaps image paths to match expected dataset structure."""
    return path.replace('category_1', '1').replace('category_0', '0')

async def _process_single_problem(problem_id: str, problem_records: List[Dict[str, Any]], feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate):
    """
    Processes a single Bongard problem (12 images) to build one scene graph.
    Each image within the problem becomes a node in the single problem-level graph.
    """
    if not problem_records:
        logging.warning(f"No records found for problem {problem_id}. Skipping graph build.")
        return None

    # 1. Merge records into a single object for graph building
    merged_record = {'objects': []}
    representative_image_path = None # Path to one of the images for visualization/feature extraction

    for idx, rec in enumerate(problem_records):
        # Use the first image path as representative for the problem
        if representative_image_path is None:
            representative_image_path = remap_path(rec.get('image_path', rec.get('mask_path', '')))

        # Ensure 'id' is unique within the problem's graph and corresponds to the original image
        obj_id = f"{problem_id}_{idx}" # Unique ID for each image-object within the problem graph
        # Log missing label for debugging
        if 'label' not in rec:
            logging.warning(f"Missing 'label' in record for problem {problem_id}, image idx {idx}, image_path: {rec.get('image_path', '')}")
        merged_record['objects'].append({
            'id': obj_id,
            'vertices': rec.get('vertices', []),
            **rec.get('features', {}), # Unpack centroid, area etc. from derived_labels
            'label': rec.get('label', ''), # Always include 'label' key for downstream compatibility
            'shape_label': rec.get('label', ''), # Use 'label' from derived_labels as shape_label
            'category': rec.get('category', ''),
            'original_image_path': remap_path(rec.get('image_path', '')), # Keep original image path for tracing
            'original_record_idx': idx # For debugging/tracing if needed
        })

    # Log merged objects for debugging (show only id and shape_label for each object)
    debug_summary = [
        {'id': obj.get('id'), 'shape_label': obj.get('shape_label')} for obj in merged_record['objects']
    ]
    logging.info(f"Merged objects for problem {problem_id} (id, shape_label): {json.dumps(debug_summary)}")

    # 2. Build the unvalidated base scene graph for the entire problem
    # This graph will have N nodes (e.g., 12), one for each image-object.
    base_graph_nx = build_graph_unvalidated(merged_record)

    # 3. Prepare base_scene_graph for enhanced builder
    # The enhanced builder expects 'objects' (list of object dicts) and 'relationships' (list of dicts)
    # and the NetworkX graph itself.
    base_scene_graph_for_enhanced = {
        'objects': merged_record['objects'], # These are the node data for the graph
        'relationships': [], # Relationships will be filled by add_predicate_edges and enhanced_builder
        'graph': base_graph_nx # Pass the networkx graph directly
    }

    # 4. Enhance the scene graph (knowledge fusion, hierarchical predicates)
    # This still needs a single 'image_path' for image-based feature extraction/validation.
    scene_graph_data = await enhanced_builder.build_enhanced_scene_graph(representative_image_path, base_scene_graph_for_enhanced)
    final_scene_graph = scene_graph_data['scene_graph']
    quality_metrics = scene_graph_data['quality_metrics']

    # 5. Save feedback images (if enabled)
    if feedback_dir and representative_image_path:
        # Save only for a subset of problems based on feedback_rate
        problem_numeric_id = int(''.join(filter(str.isdigit, problem_id))) if any(char.isdigit() for char in problem_id) else 0
        if feedback_rate == 1 or (problem_numeric_id % feedback_rate == 0):
            try:
                img = cv2.imread(representative_image_path)
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
            validation_result = await batch_validator.validate_scene_graphs_batch(
                [(img_pil, final_scene_graph)]
            )
            final_scene_graph['validation'] = validation_result[0] if validation_result else None
        except Exception as e:
            logging.error(f"Batch validation failed for problem {problem_id} ({representative_image_path}): {e}")
            final_scene_graph['validation'] = {'error': str(e)}

    # 7. Recall evaluation (if ground_truth is available)
    # This part is highly dependent on how problem-level ground truth is structured.
    # The original `recall_evaluator` expects ground truth per image.
    # For problem-level graphs, you would need problem-level ground truth.
    # As the prompt does not provide this, I'm commenting out the active evaluation,
    # but keeping the evaluator initialized for potential future use.
    metrics = None
    # if 'ground_truth' in problem_records[0] and recall_evaluator:
    #     try:
    #         # This is a placeholder. You would need problem-level ground truth here.
    #         # For example, if problem_records[0]['ground_truth'] represents the *problem's* GT
    #         metrics = recall_evaluator.evaluate_scene_graph(
    #             final_scene_graph['relationships'],
    #             problem_records[0]['ground_truth']['relationships'], # This is likely incorrect for problem-level GT
    #             final_scene_graph['objects'],
    #             problem_records[0]['ground_truth']['objects'] # This is likely incorrect for problem-level GT
    #         )
    #         final_scene_graph['metrics'] = metrics
    #     except Exception as e:
    #         logging.error(f"Recall evaluation failed for problem {problem_id}: {e}")
    #         final_scene_graph['metrics'] = {'error': str(e)}

    # 8. Drift visualization (if enabled)
    # This also relies on `metrics` from recall evaluation, which is currently paused.
    # If you re-enable recall evaluation with problem-level GT, this can be re-enabled.
    if drift_visualizer and metrics: # metrics will be None if recall_evaluator is skipped
        for rel in final_scene_graph.get('relationships', []):
            predicate = rel['predicate']
            confidence = rel.get('final_confidence', quality_metrics.get('knowledge_confidence', 0.5))
            performance = metrics.get('overall_recall', 0.5) # Placeholder: needs actual problem-level performance
            drift_visualizer.add_threshold_data(predicate, confidence, performance)

    return {
        'problem_id': problem_id,
        'scene_graph_data': scene_graph_data, # Contains 'scene_graph' (with NetworkX graph) and 'quality_metrics'
        'processing_metadata': {
            'representative_image_path': representative_image_path,
            'num_nodes_in_graph': base_graph_nx.number_of_nodes()
        }
    }

def process_problem_batch_sync_wrapper(list_of_problem_records: List[List[Dict[str, Any]]], feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate):
    """Synchronous wrapper to run async processing for a batch of problems."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for problem_records in list_of_problem_records:
        if problem_records: # Ensure problem_records is not empty
            problem_id = problem_records[0].get('problem_id', 'unknown_problem')
            tasks.append(_process_single_problem(problem_id, problem_records, feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, feedback_dir, feedback_rate))
        else:
            logging.warning("Empty problem record list passed to process_problem_batch_sync_wrapper.")

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
        predicate_classes=list(PREDICATES.keys()), # Use the globally defined PREDICATES
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

    # Compute hash for cache validation
    params_dict = {
        'aug': args.aug,
        'labels': args.labels,
        'parallel': args.parallel,
        'batch_size': args.batch_size,
        'predicates_hash': hashlib.sha256(json.dumps(list(PREDICATES.keys()), sort_keys=True).encode()).hexdigest(),
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

    # --- START PATCH: Merge labels and group by problem_id ---
    # Build a lookup from image_path -> features/geometry
    label_lookup = {}
    if isinstance(derived_labels, list):
        for lbl in derived_labels:
            # normalize paths if needed so they match augmented_data['image_path']
            normalized_image_path = remap_path(lbl['image_path'])
            label_lookup[normalized_image_path] = {
                'features': lbl['features'],
                'vertices': lbl['geometry'],
                'label': lbl['label'],
                'category': lbl['category']
            }

    # Enrich augmented_data with label info
    enriched_augmented_data = []
    for rec in augmented_data:
        info = label_lookup.get(remap_path(rec['image_path']))
        if not info:
            logging.warning(f"No derived_labels entry for {rec['image_path']}. Skipping record.")
            continue
        new_rec = rec.copy() # Create a copy to update
        new_rec.update(info)  # now new_rec has new_rec['features'], new_rec['vertices'], new_rec['label'], new_rec['category']
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
    # --- END PATCH ---

    # Determine which graph building function to use (with or without validation)
    # This is now used only for profiling, as the main loop calls _process_single_problem
    build_func_for_profiling = build_graph_validated if GRAPHTYPE_AVAILABLE else build_graph_unvalidated
    if GRAPHTYPE_AVAILABLE:
        logging.info("Graphtype detected. Runtime schema validation is ENABLED.")
    else:
        logging.info("Graphtype not found. Runtime schema validation is DISABLED.")

    # Auto-profile batch size if requested (batch size now refers to problems per batch)
    batch_size = args.batch_size
    if batch_size == 0 and len(grouped_problems) > 0:
        # Pass a sample of individual records for profiling, as profile_optimal_batch_size simulates grouping
        batch_size = profile_optimal_batch_size(enriched_augmented_data, build_func_for_profiling)
    elif batch_size == 0:
        logging.warning("Cannot auto-profile batch size: no problems available. Using default batch size 16.")
        batch_size = 16
    logging.info(f"Using batch size (problems per batch): {batch_size}")


    graphs = [] # This will store the problem-level scene graph data (dict from _process_single_problem)
    profiler = TaskProfiler()

    # Split problem_ids into batches for parallel processing
    problem_ids = list(grouped_problems.keys())
    problem_batches = [problem_ids[i:i + batch_size] for i in range(0, len(problem_ids), batch_size)]
    logging.info(f"Starting problem batch processing: {len(problem_batches)} batches, problem_batch_size={batch_size}, parallel={args.parallel}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit tasks for each problem batch
        futures = {
            executor.submit(
                process_problem_batch_sync_wrapper,
                [grouped_problems[pid] for pid in batch_pids], # Pass list of problem_records for each problem in batch
                feature_cache, batch_validator, recall_evaluator, drift_visualizer, enhanced_builder, args.feedback_dir, args.feedback_rate
            ): batch_idx for batch_idx, batch_pids in enumerate(problem_batches)
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(problem_batches), desc="Building problem graphs", mininterval=0.5):
            batch_idx = futures[future]
            start_time_batch = time.time()
            try:
                batch_results = future.result()
                # Extend graphs with valid results (filter out None from skipped problems)
                graphs.extend([res for res in batch_results if res is not None])
                latency = (time.time() - start_time_batch) * 1000
                profiler.log_latency('scene_graph_build_problem', latency, {
                    'num_problems_in_batch': len(batch_results),
                    'latency_ms': latency,
                    'throughput_problems_per_sec': len(batch_results) / (latency / 1000) if latency > 0 else float('inf')
                })
                logging.info(f"Processed problem batch {batch_idx+1}/{len(problem_batches)}. Problems processed: {len(batch_results)}")

                # Periodically clear cache and log memory stats
                if (batch_idx + 1) % 4 == 0: # Check every 4 batches
                    if TORCH_KORNIA_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    memory_stats = feature_cache.get_cache_stats()
                    logging.info(f"Memory usage: RAM: {memory_stats.get('memory_usage_mb', 0):.1f}MB, GPU: {memory_stats.get('gpu_usage_mb', 0):.1f}MB")

            except Exception as e:
                logging.error(f"Error processing problem batch {batch_idx+1}: {e}")
                problematic_problem_ids = problem_batches[batch_idx]
                if problematic_problem_ids:
                    logging.debug(f"Sample from problematic problem batch (first problem ID): {problematic_problem_ids[0]}")

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
        # Collect all predicates from all relationships in all scene graphs
        all_predicates = [rel.get('predicate', 'unknown')
                         for G_data in graphs # Iterate over problem-level graph data
                         for rel in G_data.get('scene_graph_data', {}).get('scene_graph', {}).get('relationships', [])]
        common_predicates = [p for p, _ in Counter(all_predicates).most_common(5)]
        if common_predicates:
            logging.info(f"Generating drift report for predicates: {common_predicates}")
            drift_visualizer.generate_drift_report(common_predicates)
            drift_visualizer.visualize_threshold_evolution(common_predicates)
        else:
            logging.warning("No common predicates found to generate drift report.")

    # Log overall performance stats
    cache_stats = feature_cache.get_cache_stats()
    logging.info("Processing complete!")
    logging.info(f"Cache efficiency: Memory: {cache_stats.get('memory_usage_mb', 0):.1f}MB used, GPU: {cache_stats.get('gpu_usage_mb', 0):.1f}MB used")

if __name__ == "__main__":
    asyncio.run(main())
