
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
    # --- 1. Programmatic semantics: next_action, turns_toward ---
    # Group nodes by OneStrokeShape if available (assume 'action_index' and 'stroke_type' present)
    nodes_by_action = {}
    for u, data_u in node_list:
        idx = data_u.get('action_index', None)
        if 'action_index' in data_u and idx is not None:
            nodes_by_action[(data_u.get('parent_shape_id', None), idx)] = (u, data_u)
    # Add next_action and turns_toward edges
    for (parent_id, idx), (u, data_u) in nodes_by_action.items():
        if idx is None:
            continue
        next_key = (parent_id, idx+1)
        if next_key in nodes_by_action:
            v, data_v = nodes_by_action[next_key]
            G.add_edge(u, v, predicate='next_action', source='program')
            if data_v.get('turn_direction') is not None:
                G.add_edge(u, v, predicate=f'turns_{data_v["turn_direction"]}', angle=data_v.get('turn_angle'), source='program')

    # --- 2. Topological & connectivity relations ---
    # Endpoint adjacency and intersection
    from collections import defaultdict
    endpoints = defaultdict(list)
    for u, data_u in node_list:
        verts = data_u.get('vertices', [])
        if verts and len(verts) >= 2:
            endpoints[tuple(np.round(verts[0], 2))].append(u)
            endpoints[tuple(np.round(verts[-1], 2))].append(u)
    # Add adjacent_endpoints edges
    for coord, stroke_ids in endpoints.items():
        for i, j in combinations(stroke_ids, 2):
            G.add_edge(i, j, predicate='adjacent_endpoints', source='topology')
    # Compute junction_degree
    for coord, stroke_ids in endpoints.items():
        if len(stroke_ids) >= 3:
            for sid in stroke_ids:
                if 'junction_degree' not in G.nodes[sid]:
                    G.nodes[sid]['junction_degree'] = 0
                G.nodes[sid]['junction_degree'] = max(G.nodes[sid]['junction_degree'], len(stroke_ids))
    # Intersection count (segment-segment)
    def count_segment_intersections(verts1, verts2):
        # Discretize both polylines into segments
        def segments(verts):
            return list(zip(verts[:-1], verts[1:]))
        segs1 = segments(verts1)
        segs2 = segments(verts2)
        def seg_intersect(a, b, c, d):
            # Return True if segments ab and cd intersect
            from shapely.geometry import LineString
            return LineString([a, b]).intersects(LineString([c, d]))
        count = 0
        for a, b in segs1:
            for c, d in segs2:
                if seg_intersect(a, b, c, d):
                    count += 1
        return count
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if i >= j:
                continue
            verts1 = data_u.get('vertices', [])
            verts2 = data_v.get('vertices', [])
            if verts1 and verts2 and len(verts1) >= 2 and len(verts2) >= 2:
                count = count_segment_intersections(verts1, verts2)
                if count > 0:
                    G.add_edge(u, v, predicate='intersects', count=count, source='topology')

    # --- 3. Stroke-level geometric predicates ---
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if i == j:
                continue
            # Only for lines/polylines
            if data_u.get('object_type') == 'line' and data_v.get('object_type') == 'line':
                # longer_than
                if data_u.get('length') is not None and data_v.get('length') is not None:
                    if data_u['length'] > data_v['length']:
                        G.add_edge(u, v, predicate='longer_than', source='geometry')
                # parallel
                if data_u.get('orientation') is not None and data_v.get('orientation') is not None:
                    if abs(data_u['orientation'] - data_v['orientation']) < 12.0:
                        G.add_edge(u, v, predicate='parallel', source='geometry')
                # collinear (endpoints nearly aligned)
                if data_u.get('centroid') is not None and data_v.get('centroid') is not None:
                    c1, c2 = np.array(data_u['centroid']), np.array(data_v['centroid'])
                    if np.linalg.norm(c1 - c2) < 6.0:
                        G.add_edge(u, v, predicate='collinear', source='geometry')
                # angle_between
                if data_u.get('orientation') is not None and data_v.get('orientation') is not None:
                    angle = abs(data_u['orientation'] - data_v['orientation'])
                    G.add_edge(u, v, predicate='angle_between', angle=angle, source='geometry')

    # --- 4. Existing logic for polygons and valid geometry ---
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

    # --- Add global predicates for rule induction ---
    nodes = [d for _, d in G.nodes(data=True)]
    num_lines = len([n for n in nodes if n.get('curvature_type') == 'line'])
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
                # Try to aggregate geometry from member_nodes (new) or members (legacy)
                from src.scene_graphs_building.motif_miner import MotifMiner
                member_ids = node.get('member_nodes', node.get('members', []))
                member_nodes = [n for n in geometry if n.get('object_id', n.get('id')) in member_ids]
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


