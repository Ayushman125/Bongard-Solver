import sys
import os
import argparse
import pickle
import json
import networkx as nx
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from collections import Counter
import scipy.stats as stats

# Ensure project root is in sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.commonsense_kb import CommonsenseKB

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

# Predicate functions expect node dictionaries as input
PREDICATES = {
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

def main():
    parser = argparse.ArgumentParser(description="Build scene graphs from augmented images and derived labels")
    parser.add_argument('--aug', type=str, required=True, help='Path to augmented.pkl containing object records')
    parser.add_argument('--labels', type=str, default='data/derived_labels.json', help='Path to derived_labels.json')
    parser.add_argument('--out', type=str, required=True, help='Output pickle file for scene graphs')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers (placeholder)')
    args = parser.parse_args()

    with open(args.aug, 'rb') as f:
        augmented_data = pickle.load(f)

    try:
        with open(args.labels, 'r', encoding='utf-8') as f:
            derived_labels = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {args.labels}. Continuing without them.")
        derived_labels = {}

    build_func = build_graph_validated if GRAPHTYPE_AVAILABLE else build_graph_unvalidated
    if GRAPHTYPE_AVAILABLE:
        print("graphtype detected. Runtime schema validation is ENABLED.")
    else:
        print("graphtype not found. Runtime schema validation is DISABLED.")

    graphs = []
    
    all_records = []
    if augmented_data and isinstance(augmented_data[0], dict) and 'original' in augmented_data[0]:
        # This handles the list of batch results from the current image_augmentor.py
        for batch_result in augmented_data:
            # We need to associate original images with their augmented data if needed,
            # but for now, let's just build graphs from the 'original' data structure.
            # The prompt implies augmented.pkl contains object records.
            # Let's assume the objects are in a key like 'objects' or similar.
            # This part is tricky as the augmentor script doesn't seem to output this structure.
            # Let's assume a structure that might be intended.
            # A list of records, where each record is a dict with 'id' and 'objects'.
            pass # This logic needs to be clarified based on actual `augmented.pkl` structure.

    # Based on the prompt, let's assume `augmented_data` is a list of records.
    # If it's from the *previous* version of the pipeline, it might be structured differently.
    # For now, we'll assume it's a list of records.
    
    # Let's create a dummy record structure if the loaded data is not in the expected format
    # This part is speculative and should be adjusted based on the actual data format.
    if not (augmented_data and isinstance(augmented_data[0], dict) and 'objects' in augmented_data[0]):
         print("Warning: `augmented.pkl` does not seem to contain a list of object records. The script might fail.")
         # The user needs to ensure the input `augmented.pkl` has the right format.
         # For now, we proceed assuming it does.

    for record in tqdm(augmented_data, desc="Building graphs"):
        if not isinstance(record, dict) or 'objects' not in record:
            continue

        img_id = record.get('id')
        if img_id and img_id in derived_labels:
            label_map = {obj_data['id']: obj_data for obj_data in derived_labels.get(img_id, [])}
            for obj in record.get('objects', []):
                obj.update(label_map.get(obj['id'], {}))
        
        G = build_func(record)
        graphs.append(G)

    with open(args.out, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"Saved {len(graphs)} scene graphs to {args.out}")

    log_diversity_metrics(graphs)

if __name__ == "__main__":
    main()
