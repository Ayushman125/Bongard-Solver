from collections import Counter
from scipy import stats
from shapely.geometry import Polygon

import os
import sys
import pickle
import json
import hashlib
import time
import argparse
import concurrent.futures
from tqdm import tqdm
import numpy as np
from PIL import Image
import networkx as nx
import logging
import cv2
try:
    from src.bongard_augmentor.kb_sqlite import CommonsenseKB
except ImportError:
    CommonsenseKB = None
import threading
import time
from PIL import Image
# Modern image loading and GPU augmentation
try:
    import torch
    import kornia
    import torchvision
    TORCH_KORNIA_AVAILABLE = True
except ImportError:
    TORCH_KORNIA_AVAILABLE = False

# --- Advanced Knowledge and Validation Imports ---
from src.bongard_augmentor.knowledge_fusion import MultiSourceKnowledgeFusion
from src.bongard_augmentor.sgcore_validator import SGScoreValidator
from src.bongard_augmentor.hierarchical_predicates import HierarchicalPredicatePredictor
from src.bongard_augmentor.feature_extractors import RealFeatureExtractor
from src.bongard_augmentor.adaptive_predicates import AdaptivePredicateThresholds, create_adaptive_predicate_functions

class EnhancedSceneGraphBuilder:
    def __init__(self):
        self.knowledge_fusion = MultiSourceKnowledgeFusion()
        self.sgcore_validator = SGScoreValidator()
        self.hierarchical_predictor = HierarchicalPredicatePredictor()
        self.feature_extractor = RealFeatureExtractor(
            clip_model_name="openai/clip-vit-base-patch32",
            sam_encoder_path="sam_checkpoints/sam_vit_h_4b8939.pth",
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_features=True
        )
        logging.info("EnhancedSceneGraphBuilder initialized with real feature extraction")

    def _create_mask_from_vertices(self, vertices, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        vertices_array = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [vertices_array], 255)
        return mask

    def _create_mask_from_bbox(self, bbox, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        mask[y1:y2, x1:x2] = 255
        return mask

    async def build_enhanced_scene_graph(self, image_path: str, base_scene_graph: dict) -> dict:
        image = Image.open(image_path)
        image_np = np.array(image)
        objects = base_scene_graph.get('objects', [])
        object_features = {}
        for obj in objects:
            obj_id = obj.get('id', f"obj_{len(object_features)}")
            if 'vertices' in obj and len(obj['vertices']) >= 3:
                mask = self._create_mask_from_vertices(obj['vertices'], image_np.shape[:2])
            else:
                bbox = obj.get('bbox', [0, 0, image_np.shape[1], image_np.shape[0]])
                mask = self._create_mask_from_bbox(bbox, image_np.shape[:2])
            try:
                features = self.feature_extractor.extract_object_features(image_np, mask, obj_id)
                object_features[obj_id] = features
                logging.info(f"Extracted features for object {obj_id}: shape {features.shape}")
            except Exception as e:
                logging.error(f"Feature extraction failed for object {obj_id}: {e}")
                object_features[obj_id] = torch.zeros(384)
        enhanced_relationships = []
        for rel in base_scene_graph.get('relationships', []):
            subject_id = rel.get('subject_id')
            object_id = rel.get('object_id')
            if subject_id in object_features and object_id in object_features:
                try:
                    enriched_rels = await self.knowledge_fusion.get_enriched_relationships(
                        rel.get('subject', ''), rel.get('object', ''), [rel.get('predicate', '')]
                    )
                    enhanced_relationships.extend(enriched_rels)
                except Exception as e:
                    logging.warning(f"Knowledge fusion failed for relationship: {e}")
                    enhanced_relationships.append(rel)
            else:
                enhanced_relationships.append(rel)
        # Hierarchical predicate refinement (example: process pairs)
        refined_relationships = []
        for i in range(0, len(enhanced_relationships), 2):
            if i + 1 < len(enhanced_relationships):
                rel1 = enhanced_relationships[i]
                rel2 = enhanced_relationships[i+1]
                # Example: use features for refinement (dummy call)
                subject_features = object_features.get(rel1.get('subject_id'), torch.zeros(384))
                object_features_ = object_features.get(rel1.get('object_id'), torch.zeros(384))
                knowledge_embeddings = None
                refined = self.hierarchical_predictor.predict_with_bayesian_inference(
                    subject_features, object_features_, knowledge_embeddings
                )
                refined_relationships.extend(refined)
            else:
                refined_relationships.append(enhanced_relationships[i])
        validation_results = await self.sgcore_validator.validate_scene_graph(
            image, {'objects': objects, 'relationships': refined_relationships}
        )
        final_scene_graph = base_scene_graph.copy()
        final_scene_graph['relationships'] = refined_relationships
        return {
            'scene_graph': final_scene_graph,
            'quality_metrics': {
                'knowledge_confidence': float(np.mean([r.get('final_confidence', 1.0) for r in enhanced_relationships]) if enhanced_relationships else 1.0),
                'validation_score': validation_results['overall_score'],
                'relationship_accuracy': validation_results['relationship_accuracy_score']
            }
        }

# Ensure project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.commonsense_kb_sqlite import SQLiteCommonsenseKB
from integration.task_profiler import TaskProfiler

try:
    from graphtype import graph_validate, NodeData, EdgeData
    GRAPHTYPE_AVAILABLE = True
except ImportError:
    GRAPHTYPE_AVAILABLE = False
    # Dummy decorator if graphtype is not installed
    def graph_validate(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    NodeData = dict
    EdgeData = dict

# --- Hardening Plan Implementation ---

# 1. Predicate Registry & Edge Enrichment
EPS = 1e-3

def parallel(a, b, tol=7):
    """Checks if two orientations are parallel within a tolerance."""
    return abs(a - b) < tol or abs(abs(a - b) - 180) < tol

def physics_contact(poly_a, poly_b):
    """Robust check for physical contact between two polygons."""
    return poly_a.touches(poly_b)


# Loader for all predicates from schemas/edge_types.json
def load_predicates():
    predicate_file = os.path.join(os.path.dirname(__file__), '..', 'schemas', 'edge_types.json')
    if not os.path.exists(predicate_file):
        print("Warning: schemas/edge_types.json not found. Using built-in predicates.")
        return None
    with open(predicate_file, 'r', encoding='utf-8') as f:
        predicate_defs = json.load(f)
    # Map predicate names to logic
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
            registry[name] = lambda a, b: Polygon(a['vertices']).contains(Polygon(b['vertices']))
        elif name == 'inside':
            registry[name] = lambda a, b: Polygon(b['vertices']).contains(Polygon(a['vertices']))
        elif name == 'supports':
            registry[name] = lambda a, b: physics_contact(Polygon(a['vertices']), Polygon(b['vertices'])) and a.get('cy', 0) > b.get('cy', 0)
        elif name == 'supported_by':
            registry[name] = lambda a, b: physics_contact(Polygon(b['vertices']), Polygon(a['vertices'])) and b.get('cy', 0) > a.get('cy', 0)
        elif name == 'touches':
            registry[name] = lambda a, b: Polygon(a['vertices']).touches(Polygon(b['vertices']))
        elif name == 'overlaps':
            registry[name] = lambda a, b: Polygon(a['vertices']).overlaps(Polygon(b['vertices']))
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

# Use loader if file exists, else fallback to built-in
PREDICATES = load_predicates() or {
    'left_of':     lambda a, b: a.get('cx', 0) + EPS < b.get('cx', 0),
    'right_of':    lambda a, b: a.get('cx', 0) > b.get('cx', 0) + EPS,
    'above':       lambda a, b: a.get('cy', 0) + EPS < b.get('cy', 0),
    'below':       lambda a, b: a.get('cy', 0) > b.get('cy', 0) + EPS,
    'contains':    lambda a, b: Polygon(a['vertices']).contains(Polygon(b['vertices'])),
    'inside':      lambda a, b: Polygon(b['vertices']).contains(Polygon(a['vertices'])),
    'supports':    lambda a, b: physics_contact(Polygon(a['vertices']), Polygon(b['vertices'])) and a.get('cy', 0) > b.get('cy', 0),
    'touches':     lambda a, b: Polygon(a['vertices']).touches(Polygon(b['vertices'])),
    'overlaps':    lambda a, b: Polygon(a['vertices']).overlaps(Polygon(b['vertices'])),
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
                except Exception:
                    continue

# 2. Commonsense KB Look-Ups
kb_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'conceptnet_lite.json')
if os.path.exists(kb_path):
    kb = CommonsenseKB(path=kb_path)
else:
    kb = None
    print(f"Warning: CommonsenseKB not found at {kb_path}. Semantic edges will be skipped.")

TOP_K = 3

def add_commonsense_edges(G):
    """Adds semantic edges by querying the commonsense knowledge base."""
    if not kb:
        return
    nodes_with_data = list(G.nodes(data=True))
    for u, data_u in nodes_with_data:
        label = data_u.get('shape_label')
        if not label:
            continue
        for rel, other_concept in kb.related(label)[:TOP_K]:
            for v, data_v in nodes_with_data:
                if u != v and data_v.get('shape_label') == other_concept:
                    G.add_edge(u, v, predicate=rel, source='kb')
                    break

# 3. Robust Physics Attributes & Assertions
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
        inertia = 0.0
    
    convexity = poly.convex_hull.area / area if area > 0 else 0.0
    centroid = poly.centroid
    
    node_data.update({
        'area': float(area),
        'inertia': float(inertia),
        'convexity': float(convexity),
        'cx': centroid.x,
        'cy': centroid.y,
    })

# 4. Schema Validation via graphtype
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

# 5. Diversity KPI Logger
def compute_pc_error(graphs):
    """Computes Physical Consistency error (e.g., overlapping solids)."""
    error_count = 0
    total_graphs = len(graphs)
    if total_graphs == 0: return 0.0

    for G in graphs:
        nodes = list(G.nodes(data=True))
        polygons = [Polygon(data['vertices']) for _, data in nodes if data.get('vertices') and len(data['vertices']) >= 3]
        has_overlap = any(polygons[i].overlaps(polygons[j]) for i in range(len(polygons)) for j in range(i + 1, len(polygons)))
        if has_overlap:
            error_count += 1
            
    return error_count / total_graphs

def log_diversity_metrics(graphs, out_path='logs/graph_diversity.jsonl'):
    """Computes and logs coverage, entropy, and physical consistency."""
    if not graphs: return

    possible_triples = sum(len(G.nodes)**2 - len(G.nodes) for G in graphs)
    unique_triples = {(u, data.get('predicate', 'unknown'), v) for G in graphs for u, v, data in G.edges(data=True)}
    predicate_counts = Counter(d.get('predicate', 'unknown') for G in graphs for u,v,d in G.edges(data=True))

    C = len(unique_triples) / possible_triples if possible_triples > 0 else 0
    H = stats.entropy(list(predicate_counts.values()), base=2) if predicate_counts else 0
    pc_error = compute_pc_error(graphs)
    
    metrics = {'coverage': C, 'entropy': H, 'pc_error': pc_error}
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'a') as f:
        f.write(json.dumps(metrics) + "\n")
    
    print(f"Diversity Metrics: Coverage={C:.3f}, Entropy={H:.3f} bits, PC-Error={pc_error:.3f}")

def compute_hash(input_paths, params_dict):
    h = hashlib.sha256()
    for path in input_paths:
        with open(path, 'rb') as f:
            h.update(f.read())
    for k in sorted(params_dict.keys()):
        h.update(str(k).encode())
        h.update(str(params_dict[k]).encode())
    return h.hexdigest()

def cache_valid(cache_path, hash_path, current_hash):
    if not os.path.exists(cache_path) or not os.path.exists(hash_path):
        return False
    with open(hash_path, 'r') as f:
        cached_hash = f.read().strip()
    return cached_hash == current_hash

def save_hash(hash_path, hash_val):
    with open(hash_path, 'w') as f:
        f.write(hash_val)

def profile_optimal_batch_size(records, build_func):
    candidate_sizes = [8, 16, 32, 64, 128]
    best_size = candidate_sizes[0]
    best_throughput = 0
    print("[INFO] Profiling batch sizes for optimal graph build throughput...")
    for size in candidate_sizes:
        start = time.time()
        try:
            batch = records[:size]
            graphs = []
            for record in batch:
                G = build_func(record)
                graphs.append(G)
            elapsed = time.time() - start
            throughput = size / elapsed if elapsed > 0 else 0
            print(f"Batch size {size}: {throughput:.2f} graphs/sec")
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size
        except Exception as e:
            print(f"Batch size {size} failed: {e}")
    print(f"[INFO] Selected optimal batch size: {best_size}")
    return best_size


def main():
    parser = argparse.ArgumentParser(description="Build scene graphs from augmented images and derived labels")
    parser.add_argument('--aug', type=str, required=True, help='Path to augmented.pkl containing object records')
    parser.add_argument('--labels', type=str, default='data/derived_labels.json', help='Path to derived_labels.json')
    parser.add_argument('--out', type=str, required=True, help='Output pickle file for scene graphs')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for graph building (0=auto)')
    parser.add_argument('--feedback-dir', type=str, default='feedback_samples', help='Directory to save QA/feedback images')
    parser.add_argument('--feedback-rate', type=int, default=20, help='Save every Nth sample for feedback (default: 20, set 1 to save all)')
    args = parser.parse_args()

    # Adaptive predicate threshold learning
    adaptive_thresholds = AdaptivePredicateThresholds(
        history_size=1000,
        confidence_level=0.95,
        adaptation_rate=0.1,
        min_samples=50
    )
    threshold_cache_path = args.out.replace('.pkl', '_thresholds.json')
    adaptive_thresholds.load_learned_thresholds(threshold_cache_path)

    # ...existing code for loading data...
    import cv2
    import numpy as np
    import shutil
    def mask_quality_stats(image, mask):
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

    def save_feedback_images(image, mask, base_name, feedback_dir):
        os.makedirs(feedback_dir, exist_ok=True)
        img_save_path = os.path.join(feedback_dir, f"{base_name}_input.png")
        mask_save_path = os.path.join(feedback_dir, f"{base_name}_mask.png")
        side_by_side_path = os.path.join(feedback_dir, f"{base_name}_side_by_side.png")
        if image.ndim == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image
        cv2.imwrite(img_save_path, img_bgr)
        cv2.imwrite(mask_save_path, mask)
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if img_bgr.shape[:2] != mask_color.shape[:2]:
            mask_color = cv2.resize(mask_color, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        side_by_side = cv2.hconcat([img_bgr, mask_color])
        cv2.imwrite(side_by_side_path, side_by_side)

    params_dict = {
        'aug': args.aug,
        'labels': args.labels,
        'parallel': args.parallel,
        'batch_size': args.batch_size
    }
    current_hash = compute_hash([args.aug, args.labels], params_dict)
    hash_path = args.out + '.hash'
    if cache_valid(args.out, hash_path, current_hash):
        print(f"[INFO] Cache valid for {args.out}. Skipping graph build.")
        return

    def async_load_pickle(path):
        print(f"[LOG] Loading pickle file: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"[LOG] Loaded pickle file: {path}, records: {len(data) if hasattr(data, '__len__') else 'unknown'}")
        return data
    def async_load_json(path):
        print(f"[LOG] Loading JSON file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[LOG] Loaded JSON file: {path}, records: {len(data) if hasattr(data, '__len__') else 'unknown'}")
        return data

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_aug = executor.submit(async_load_pickle, args.aug)
        future_labels = executor.submit(async_load_json, args.labels)
        augmented_data = future_aug.result()
        print(f"[INFO] Loaded {len(augmented_data)} records from {args.aug}")
        if len(augmented_data) > 0:
            print(f"[LOG] Sample augmented record keys: {list(augmented_data[0].keys()) if isinstance(augmented_data[0], dict) else type(augmented_data[0])}")
        try:
            derived_labels = future_labels.result()
        except Exception:
            print(f"Warning: Labels file not found at {args.labels}. Continuing without them.")
            derived_labels = {}

    build_func = build_graph_validated if GRAPHTYPE_AVAILABLE else build_graph_unvalidated
    if GRAPHTYPE_AVAILABLE:
        print("graphtype detected. Runtime schema validation is ENABLED.")
    else:
        print("graphtype not found. Runtime schema validation is DISABLED.")

    if args.batch_size == 0:
        batch_size = profile_optimal_batch_size(augmented_data, build_func)
    else:
        batch_size = args.batch_size

    graphs = []
    profiler = TaskProfiler()
    import asyncio
    enhanced_builder = EnhancedSceneGraphBuilder()

    def process_batch(batch):
        batch_graphs = []
        skipped = 0
        sample_skipped = None
        for idx, record in enumerate(batch):
            if not isinstance(record, dict) or 'objects' not in record:
                skipped += 1
                if sample_skipped is None:
                    sample_skipped = record
                print(f"[WARN] Skipping record at batch index {idx}: not a dict or missing 'objects'.")
                continue
            img_id = record.get('id') or record.get('problem_id')
            img_path = record.get('spath') or record.get('image_path')
            if img_path:
                img_path = img_path.replace('category_1', '1').replace('category_0', '0')
            print(f"[LOG] Processing record idx={idx}, id={img_id}, img_path={img_path}")
            base_scene_graph = {
                'objects': record.get('objects', []),
                'relationships': []
            }
            try:
                enhanced_result = asyncio.run(
                    enhanced_builder.build_enhanced_scene_graph(img_path, base_scene_graph)
                )
                batch_graphs.append(enhanced_result)
            except Exception as e:
                print(f"[ERROR] Enhanced scene graph build failed for idx={idx}, id={img_id}: {e}")
        if skipped > 0:
            print(f"[WARN] Skipped {skipped} records in batch due to missing or invalid format.")
            if sample_skipped is not None:
                print(f"[DEBUG] Sample skipped record: {sample_skipped}")
        return batch_graphs

    batches = [augmented_data[i:i+batch_size] for i in range(0, len(augmented_data), batch_size)]
    print(f"[LOG] Starting batch processing: {len(batches)} batches, batch_size={batch_size}, parallel={args.parallel}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        for batch_idx, batch in enumerate(tqdm(batches, desc="Building graphs", mininterval=0.5)):
            start = time.time()
            print(f"[LOG] Submitting batch {batch_idx+1}/{len(batches)} (records {batch_idx*batch_size} to {(batch_idx+1)*batch_size-1})")
            future = executor.submit(process_batch, batch)
            batch_graphs = future.result()
            graphs.extend(batch_graphs)
            latency = (time.time() - start) * 1000
            profiler.log_latency('scene_graph_build', latency, {
                'batch_size': len(batch),
                'latency_ms': latency,
                'throughput_graphs_per_sec': len(batch) / (latency / 1000)
            })

    if len(graphs) == 0:
        print(f"[ERROR] No scene graphs were produced. Check input data and filtering logic.")
        if len(augmented_data) > 0:
            print(f"[DEBUG] Sample input record: {augmented_data[0]}")
            if isinstance(augmented_data[0], dict):
                print(f"[DEBUG] Sample input record keys: {list(augmented_data[0].keys())}")
    else:
        print(f"[LOG] Successfully built {len(graphs)} scene graphs.")
    with open(args.out, 'wb') as f:
        pickle.dump(graphs, f)
    save_hash(hash_path, current_hash)
    print(f"Saved {len(graphs)} scene graphs to {args.out}")
    log_diversity_metrics(graphs)
    adaptive_thresholds.save_learned_thresholds(threshold_cache_path)
    logging.info(f"Saved adaptive thresholds to {threshold_cache_path}")

if __name__ == "__main__":
    main()
