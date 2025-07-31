import networkx as nx
import logging
import numpy as np
from src.scene_graphs_building.feature_extraction import compute_physics_attributes


def add_predicate_edges(G, predicates):
    """Iterates through all node pairs and adds edges based on the predicate registry."""
    node_list = list(G.nodes(data=True))
    edge_count = 0
    for i, (u, data_u) in enumerate(node_list):
        for j, (v, data_v) in enumerate(node_list):
            if i == j:
                continue
            # Ensure a and b are valid dicts and not None
            if not isinstance(data_u, dict) or not isinstance(data_v, dict):
                continue
            for pred, fn in predicates.items():
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

def add_commonsense_edges(G, top_k, kb=None):
    """Adds semantic edges by querying the commonsense knowledge base."""
    logging.info(f"[add_commonsense_edges] Called with kb type: {type(kb)}, value: {kb}")
    if kb is None:
        logging.info("[add_commonsense_edges] KB not available, skipping.")
        return
    nodes_with_data = list(G.nodes(data=True))
    all_shape_labels = [d.get('shape_label') for _, d in nodes_with_data]
    logging.info(f"[add_commonsense_edges] All node shape_labels: {all_shape_labels}")
    edge_count = 0
    for u, data_u in nodes_with_data:
        # Try label, then category, then shape_label
        label = data_u.get('label') or data_u.get('category') or data_u.get('shape_label')
        label_source = None
        if data_u.get('label'):
            label_source = 'label'
        elif data_u.get('category'):
            label_source = 'category'
        elif data_u.get('shape_label'):
            label_source = 'shape_label'
        if not label:
            continue
        try:
            related_concepts = kb.related(label) if hasattr(kb, 'related') else []
            logging.info(f"[add_commonsense_edges] Node {u} ({label_source}={label}): kb.related -> {related_concepts}")
            for rel, other_concept in related_concepts[:top_k]:
                found_match = False
                for v, data_v in nodes_with_data:
                    v_label = data_v.get('label') or data_v.get('category') or data_v.get('shape_label')
                    if u != v and v_label == other_concept:
                        G.add_edge(u, v, predicate=rel, source='kb')
                        edge_count += 1
                        found_match = True
                        logging.debug(f"[add_commonsense_edges] Added KB edge: {u}->{v} predicate={rel}")
                        break
                if not found_match:
                    logging.info(f"[add_commonsense_edges] Related concept '{other_concept}' (relation '{rel}') for node {u} ({label_source} '{label}') did not match any node's label/category/shape_label.")
        except Exception as e:
            logging.warning(f"Commonsense KB query failed for label '{label}': {e}")
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
        try:
            compute_physics_attributes(node)
        except Exception as e:
            import traceback
            logging.error(f"build_graph_unvalidated: Exception in compute_physics_attributes for node at index {idx}: {e}\n{traceback.format_exc()}")
            continue
        try:
            G.add_node(node['id'], **node)
            logging.debug(f"[build_graph_unvalidated] Added node: id={node.get('id')}, shape_label={node.get('shape_label')}, category={node.get('category')}, vertices_len={len(node.get('vertices', []) if 'vertices' in node else [])}")
        except Exception as e:
            import traceback
            logging.error(f"build_graph_unvalidated: Exception adding node at index {idx}: {e}\n{traceback.format_exc()}")
            continue
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


