import networkx as nx
import logging
import numpy as np
from src.scene_graphs_building.feature_extraction import compute_physics_attributes
from src.scene_graphs_building.config import CONCEPTNET_KEEP_RELS, SHAPE_MAP, COMMONSENSE_LABEL_MAP


def add_predicate_edges(G, predicates):
    """Iterates through all node pairs and adds edges based on the predicate registry, with robust feature validity guards and line-specific proxies."""
    def safe_get(node, key):
        val = node.get(key)
        return val if val is not None else None

    node_list = list(G.nodes(data=True))
    edge_count = 0
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if i == j:
                continue
            if not isinstance(data_u, dict) or not isinstance(data_v, dict):
                continue
            # Robust geometry guards
            if not (data_u.get('geometry_valid', False) and data_v.get('geometry_valid', False)):
                # For line-based predicates, allow if both are lines
                if not (data_u.get('object_type') == 'line' and data_v.get('object_type') == 'line'):
                    continue
            # Line-specific proxies for 'near' and 'para'
            if data_u.get('object_type') == 'line' and data_v.get('object_type') == 'line':
                mid1, mid2 = safe_get(data_u, 'centroid'), safe_get(data_v, 'centroid')
                orient1, orient2 = safe_get(data_u, 'orientation'), safe_get(data_v, 'orientation')
                near_thresh = 32.0  # or from config
                para_thresh = 12.0  # or from config
                if mid1 and mid2 and all(isinstance(x, (int, float)) for x in mid1+mid2):
                    dist = np.linalg.norm(np.array(mid1) - np.array(mid2))
                    if dist < near_thresh:
                        G.add_edge(u, v, predicate='near', source='spatial')
                        edge_count += 1
                if orient1 is not None and orient2 is not None:
                    if abs(orient1 - orient2) < para_thresh:
                        G.add_edge(u, v, predicate='para', source='spatial')
                        edge_count += 1
                continue  # skip other predicates for lines
            # For polygons and valid geometry
            for pred, fn in predicates.items():
                # Area-based predicates
                if pred in ['larger_than', 'aspect_sim']:
                    if not (data_u.get('feature_valid', {}).get('area_valid', False) and data_v.get('feature_valid', {}).get('area_valid', False)):
                        continue
                    area1 = safe_get(data_u, 'area')
                    area2 = safe_get(data_v, 'area')
                    if area1 is None or area2 is None:
                        continue
                # Near/para require centroid/orientation
                if pred in ['near', 'para']:
                    if not (data_u.get('feature_valid', {}).get('centroid_valid', False) and data_v.get('feature_valid', {}).get('centroid_valid', False)):
                        continue
                    c1 = safe_get(data_u, 'centroid')
                    c2 = safe_get(data_v, 'centroid')
                    if c1 is None or c2 is None or not all(isinstance(x, (int, float)) for x in c1+c2):
                        continue
                    if pred == 'near':
                        dist = np.linalg.norm(np.array(c1) - np.array(c2))
                        if dist >= 32.0:
                            continue
                    if pred == 'para':
                        o1 = safe_get(data_u, 'orientation')
                        o2 = safe_get(data_v, 'orientation')
                        if o1 is None or o2 is None:
                            continue
                        if abs(o1 - o2) >= 12.0:
                            continue
                try:
                    result = fn(data_u, data_v)
                    logging.debug(f"[add_predicate_edges] Predicate '{pred}' between {u} and {v}: {result}")
                    if result:
                        G.add_edge(u, v, predicate=pred, source='spatial')
                        edge_count += 1
                        logging.debug(f"[add_predicate_edges] Added edge: {u}->{v} predicate={pred}")
                except Exception as e:
                    logging.error(f"[add_predicate_edges] Exception for predicate '{pred}' between {u} and {v}: {e}")
                    continue
    logging.info(f"[add_predicate_edges] Finished: edges added={edge_count}")

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

def normalize_shape_label(lbl):
    if not lbl:
        return None
    lbl = lbl.lower().replace("_", " ").replace("-", " ").strip()
    # Remove generic or non-shape labels
    if lbl in ("positive", "negative", "pos", "neg"):
        return None
    return SHAPE_MAP.get(lbl, lbl)

def add_commonsense_edges(G, top_k, kb=None):
    """Adds semantic edges by querying the commonsense knowledge base with normalized shape labels and pruned relations."""
    logging.info(f"[add_commonsense_edges] Called with kb type: {type(kb)}, value: {kb}")
    if kb is None:
        logging.info("[add_commonsense_edges] KB not available, skipping.")
        return
    nodes_with_data = list(G.nodes(data=True))
    # Normalize all shape_label values (motifs and shapes)
    for _, d in nodes_with_data:
        if d.get('shape_label') is not None:
            from src.scene_graphs_building.config import SHAPE_MAP
            lbl = str(d['shape_label']).lower().replace('_',' ').replace('-',' ').strip()
            d['shape_label'] = SHAPE_MAP.get(lbl, lbl)
    all_shape_labels = [d.get('kb_concept') for _, d in nodes_with_data]
    logging.info(f"[add_commonsense_edges] All node kb_concepts: {all_shape_labels}")
    for _, d in nodes_with_data:
        # Use shape semantics, not just positive/negative
        raw_label = d.get('shape_label')
        if raw_label in (None, '', 'positive', 'negative'):
            # Try to use object_type or other semantic field
            raw_label = d.get('object_type', None)
        # Map to commonsense concept
        kb_concept = COMMONSENSE_LABEL_MAP.get(raw_label, raw_label)
        d['kb_concept'] = kb_concept
    edge_count = 0
    for u, data_u in nodes_with_data:
        raw = data_u.get('shape_label', '')
        concept = normalize_shape_label(raw)
        if not concept:
            continue
        try:
            related = kb.related(concept) if hasattr(kb, 'related') else []
        except Exception as e:
            logging.warning(f"Commonsense KB query failed for label '{concept}': {e}")
            continue
        # Prune to allowed relations and top_k
        added = 0
        for rel, other in related:
            if rel not in CONCEPTNET_KEEP_RELS:
                continue
            if added >= top_k:
                break
            # Find matching node v
            for v, data_v in nodes_with_data:
                v_concept = normalize_shape_label(data_v.get('shape_label', ''))
                if u != v and v_concept == other:
                    G.add_edge(u, v, predicate=rel, source='kb')
                    edge_count += 1
                    added += 1
                    logging.debug(f"[add_commonsense_edges] Added KB edge: {u}->{v} predicate={rel}")
                    break
    logging.info(f"[add_commonsense_edges] Finished: KB edges added={edge_count}")

def build_graph_unvalidated(record, predicates, top_k, extra_edges=None, kb=None):
    """Builds a single scene graph without runtime schema validation. Optionally adds extra edges (e.g., CLIP/vision-language)."""
    G = nx.MultiDiGraph()
    geometry = record.get('geometry', [])
    logging.info(f"[build_graph_unvalidated] Called with {len(geometry)} objects, {len(predicates)} predicates, top_k={top_k}")
    for idx, node in enumerate(geometry):
        if not isinstance(node, dict):
            logging.error(f"build_graph_unvalidated: Skipping non-dict node at index {idx}: type={type(node)}, value={repr(node)}")
            continue
        # Motif handling: assign geometry and string label if missing
        if node.get('is_motif'):
            if node.get('vertices') is None or len(node.get('vertices')) < 3:
                # Try to aggregate geometry from members
                from src.scene_graphs_building.motif_miner import MotifMiner
                member_nodes = [n for n in geometry if n.get('object_id', n.get('id')) in node.get('members', [])]
                node['vertices'] = MotifMiner().aggregate_motif_vertices(member_nodes)
            if isinstance(node.get('shape_label'), int):
                from src.scene_graphs_building.motif_miner import MotifMiner
                node['shape_label'] = MotifMiner.MOTIF_LABELS.get(node['shape_label'], f"motif_{node['shape_label']}")
        try:
            # Bypass strict repair for motifs
            if node.get('is_motif'):
                pass
            else:
                compute_physics_attributes(node)
        except Exception as e:
            import traceback
            logging.error(f"build_graph_unvalidated: Exception in compute_physics_attributes for node at index {idx}: {e}\n{traceback.format_exc()}")
            continue
        try:
            node_id = node.get('object_id', node.get('id'))
            G.add_node(node_id, **node)
            logging.debug(f"[build_graph_unvalidated] Added node: id={node_id}, shape_label={node.get('shape_label')}, category={node.get('category')}, vertices_len={len(node.get('vertices', []) if 'vertices' in node else [])}")
        except Exception as e:
            import traceback
            logging.error(f"build_graph_unvalidated: Exception adding node at index {idx}: {e}\n{traceback.format_exc()}")
            continue
    # Normalize features for motif and regular nodes separately
    try:
        from src.scene_graphs_building.motif_miner import MotifMiner
        MotifMiner.normalize_features([d for _, d in G.nodes(data=True)])
    except Exception as e:
        logging.warning(f"Motif feature normalization failed: {e}")
    add_predicate_edges(G, predicates)
    add_commonsense_edges(G, top_k, kb=kb)
    # Add extra edges if provided (e.g., CLIP/vision-language edges)
    if extra_edges is not None:
        for edge in extra_edges:
            if len(edge) == 3:
                u, v, data = edge
                G.add_edge(u, v, **data)
            elif len(edge) == 2:
                u, v = edge
                G.add_edge(u, v)
    logging.info(f"[build_graph_unvalidated] Finished: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G

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


