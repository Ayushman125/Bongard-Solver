
import networkx as nx
import logging
import numpy as np
from itertools import combinations
from src.scene_graphs_building.feature_extraction import compute_physics_attributes
from src.scene_graphs_building.config import CONCEPTNET_KEEP_RELS, SHAPE_MAP, COMMONSENSE_LABEL_MAP


def add_predicate_edges(G, predicates):
    """Iterates through all node pairs and adds edges based on the predicate registry, with robust feature validity guards and line-specific proxies."""
    def safe_get(node, key):
        val = node.get(key)
        return val if val is not None else None


    node_list = list(G.nodes(data=True))
    edge_count = 0
    # --- LOGO mode: use predicate registry for all edge types ---
    # Only add edges for pairs where predicate function returns True
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if u == v:
                continue  # No self-loops unless predicate explicitly allows
            for pred, fn in predicates.items():
                try:
                    if fn(data_u, data_v):
                        # Always add predicate and source
                        source = 'geometry' if pred in ('length_sim', 'angle_sim') else 'program'
                        G.add_edge(u, v, predicate=pred, source=source)
                except Exception as e:
                    logging.error(f"[add_predicate_edges] Exception for predicate '{pred}' between {u} and {v}: {e}")
    logging.info(f"[add_predicate_edges] Finished: edges added={G.number_of_edges()}")

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

    # --- Map kb_concept for all nodes using COMMONSENSE_LABEL_MAP ---
    for _, data in G.nodes(data=True):
        shape_label = data.get('shape_label')
        if shape_label:
            data['kb_concept'] = COMMONSENSE_LABEL_MAP.get(shape_label, shape_label)

    # --- Add global predicates for rule induction (restore nodes/num_lines logic) ---
    nodes = [d for _, d in G.nodes(data=True)]
    num_lines = len([n for n in nodes if n.get('curvature_type') == 'line'])
    G.graph['count_lines'] = num_lines
    num_junctions = sum(1 for n in nodes if n.get('junction_degree', 0) >= 3)
    G.graph['count_lines'] = num_lines
    num_junctions = sum(1 for n in nodes if n.get('junction_degree', 0) >= 3)
    G.graph['junction_count'] = num_junctions

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
            logging.debug(f"Node {u}: no valid concept found (raw='{raw}')")
            continue
        logging.debug(f"Node {u}: processing concept '{concept}' (from raw='{raw}')")
        try:
            related = kb.related(concept) if hasattr(kb, 'related') else []
            logging.info(f"ConceptNet query for concept '{concept}': found {len(related)} relations")
            if related:
                logging.debug(f"Relations for '{concept}': {related[:5]}")  # Log first 5 for debugging
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
    # --- SOTA: Per-stroke feature extraction and caching ---
    from src.scene_graphs_building.feature_extraction import get_real_feature_extractor
    feature_extractor = get_real_feature_extractor()
    problem_id = record.get('problem_id', None)
    node_features = {}
    for idx, node in enumerate(geometry):
        if not isinstance(node, dict):
            logging.error(f"build_graph_unvalidated: Skipping non-dict node at index {idx}: type={type(node)}, value={repr(node)}")
            continue
        # Motif handling: assign geometry and string label if missing
        if node.get('is_motif'):
            if node.get('vertices') is None or len(node.get('vertices')) < 3:
                from src.scene_graphs_building.motif_miner import MotifMiner
                member_ids = node.get('member_nodes', node.get('members', []))
                member_nodes = [n for n in geometry if n.get('object_id', n.get('id')) in member_ids]
                node['vertices'] = MotifMiner().aggregate_motif_vertices(member_nodes)
            if isinstance(node.get('shape_label'), int):
                from src.scene_graphs_building.motif_miner import MotifMiner
                node['shape_label'] = MotifMiner.MOTIF_LABELS.get(node['shape_label'], f"motif_{node['shape_label']}")
        try:
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
            # --- SOTA: Extract and cache per-stroke features ---
            if feature_extractor is not None and 'image' in record and 'mask' in node:
                try:
                    feat = feature_extractor.extract_object_features(
                        record['image'], node['mask'], object_id=node_id, problem_id=problem_id, node_attrs=node
                    )
                    node['vl_embed'] = feat.detach().cpu().numpy().astype(np.float32)
                    node_features[node_id] = node['vl_embed']
                except Exception as e:
                    logging.warning(f"VL feature extraction failed for node {node_id}: {e}")
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
    # --- SOTA: Add CLIP/vision-language similarity edges for high-similarity strokes ---
    try:
        from src.scene_graphs_building.vl_features import CLIPEmbedder
        clip_embedder = CLIPEmbedder()
        # Prepare objects for CLIPEmbedder: must have image_path, bounding_box, mask, id
        objects = []
        for node_id in node_features:
            node = G.nodes[node_id]
            obj = {
                'id': node_id,
                'image_path': record.get('image', None),
                'bounding_box': node.get('bounding_box', None),
                'mask': node.get('mask', None)
            }
            objects.append(obj)
        if objects and objects[0]['image_path'] is not None:
            clip_edges = clip_embedder.contrastive_edges(objects, threshold=0.2)
            for edge in clip_edges:
                if len(edge) == 3:
                    u, v, data = edge
                elif len(edge) == 2:
                    u, v = edge
                    data = {}
                else:
                    logging.warning(f"[graph_building] Skipping unexpectedly short/long edge tuple: {edge}")
                    continue
                if not isinstance(data, dict):
                    logging.warning(f"[graph_building] Edge data not dict: {repr(data)} for edge {edge}; using empty dict.")
                    data = {}
                G.add_edge(u, v, **data)
    except Exception as e:
        logging.warning(f"CLIPEmbedder contrastive_edges failed: {e}")
    # Retain legacy VL similarity edges for backward compatibility
    if node_features:
        import numpy as np
        node_ids = list(node_features.keys())
        feats = np.stack([node_features[nid] for nid in node_ids])
        normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(normed, normed.T)
        np.fill_diagonal(sim, -np.inf)
        for i, src_id in enumerate(node_ids):
            top_idx = np.argsort(sim[i])[-2:][::-1]  # Top 2 most similar
            for j in top_idx:
                tgt_id = node_ids[j]
                if sim[i, j] > 0.7:  # High similarity threshold
                    G.add_edge(src_id, tgt_id, predicate='vl_sim', weight=float(sim[i, j]), source='vl')
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


