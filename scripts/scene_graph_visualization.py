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
            logging.warning(f"[LOGO Visualization] Skipping unexpectedly short/long edge tuple: {edge}")
            continue
        if not isinstance(data, dict):
            logging.warning(f"[LOGO Visualization] Edge data not dict: {repr(data)} for edge {edge}; using empty dict.")
            data = {}
        yield u, v, k, data
import os
import logging
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import pandas as pd

__all__ = [
    'save_scene_graph_visualization',
    'save_scene_graph_csv',
]

def save_scene_graph_visualization(G, image_path, out_dir, problem_id, abstract_view=True):
    """Save a visualization of the scene graph G overlaid on the real image."""
    os.makedirs(out_dir, exist_ok=True)
    # Node positions: use centroid, else first vertex, else (0,0)
    pos = {}
    for n, data in G.nodes(data=True):
        if 'centroid' in data and data['centroid'] is not None:
            pos[n] = data['centroid']
        elif 'vertices' in data and data['vertices']:
            pos[n] = data['vertices'][0]
        else:
            pos[n] = (0, 0)

    # For visualization: if MultiDiGraph, convert to DiGraph for plotting edge labels
    # This simplifies the visualization by showing one edge between nodes, labeled with all predicates.
    plotG = nx.DiGraph()
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        plotG.add_nodes_from(G.nodes(data=True))
        # Aggregate edges
        edge_pred_map = defaultdict(list)
        for u, v, data in G.edges(data=True):
            pred = data.get('predicate', 'unknown')
            edge_pred_map[(u, v)].append(pred)
        
        for (u, v), preds in edge_pred_map.items():
            plotG.add_edge(u, v, label=', '.join(sorted(list(set(preds)))))
    else:
        plotG.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            plotG.add_edge(u, v, label=data.get('predicate', 'unknown'))

    # --- Node color/border/label logic ---
    node_labels = {}
    node_colors = []
    node_border_colors = []
    for n, data in plotG.nodes(data=True):
        label = data.get('object_type', 'obj')
        if data.get('source') == 'geometric_grouping':
            node_colors.append('skyblue' if not data.get('is_closed') else 'lightgreen')
            node_border_colors.append('blue')
            label += f"\n(strokes: {data.get('stroke_count', 1)})"
        else: # Primitive
            node_colors.append('lightcoral')
            node_border_colors.append('red')
        node_labels[n] = label

    # --- Edge color/label logic ---
    edge_labels = nx.get_edge_attributes(plotG, 'label')
    edge_colors = []
    color_map = {
        'part_of': 'orange',
        'adjacent_endpoints': 'blue',
        'is_above': 'purple',
        'is_parallel': 'green',
        'contains': 'brown',
        'intersects': 'red',
    }
    for u, v, data in plotG.edges(data=True):
        pred = data.get('label', '').split(',')[0] # Color by first predicate
        edge_colors.append(color_map.get(pred, 'gray'))

    # Filter out labels for self-loops or edges between co-located nodes to prevent drawing errors
    drawable_edge_labels = {
        (u, v): label for (u, v), label in edge_labels.items()
        if u != v and pos.get(u) != pos.get(v)
    }
    if len(drawable_edge_labels) < len(edge_labels):
        logging.warning(f"Skipped {len(edge_labels) - len(drawable_edge_labels)} edge labels for self-loops or co-located nodes in {problem_id}.")

    # 1. Save graph overlayed on real image (if image exists)
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                plt.figure(figsize=(12, 12))
                plt.imshow(img)
                nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=500, edgecolors=node_border_colors, linewidths=1.5)
                nx.draw_networkx_edges(plotG, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.8)
                nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=7)
                nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='black', font_size=6, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1))
                plt.title(f"Full Scene Graph - {problem_id}")
                plt.axis('on')
                out_path_full = os.path.join(out_dir, f"{problem_id}_scene_graph.png")
                plt.savefig(out_path_full, bbox_inches='tight', dpi=150)
                plt.close()
        except Exception as e:
            logging.error(f"Failed to create full visualization for {problem_id}: {e}")

    # 2. Save pure matplotlib graph visualization (no image)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=800, edgecolors=node_border_colors, linewidths=1.5)
    nx.draw_networkx_edges(plotG, pos, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.8)
    nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='black', font_size=7)
    plt.title(f"Graph Only - {problem_id}")
    plt.axis('off')
    out_path_graph = os.path.join(out_dir, f"{problem_id}_graph_only.png")
    plt.savefig(out_path_graph, bbox_inches='tight')
    plt.close()

    # 3. Save abstract graph view (if enabled and applicable)
    if abstract_view:
        abstract_nodes = [n for n, data in G.nodes(data=True) if data.get('source') == 'geometric_grouping']
        if abstract_nodes:
            abstract_G = G.subgraph(abstract_nodes).copy()
            
            # Since we are creating a new figure, we need to redefine plot elements
            abstract_pos = {n: pos[n] for n in abstract_nodes}
            abstract_node_colors = ['lightgreen' if data.get('is_closed') else 'skyblue' for n, data in abstract_G.nodes(data=True)]
            abstract_node_labels = {n: f"{data.get('object_type')}\n(strokes: {data.get('stroke_count', 1)})" for n, data in abstract_G.nodes(data=True)}
            
            # Aggregate edges for abstract view
            abstract_edge_map = defaultdict(list)
            for u, v, data in abstract_G.edges(data=True):
                abstract_edge_map[(u,v)].append(data.get('predicate', 'unknown'))
            
            abstract_plot_G = nx.DiGraph()
            abstract_plot_G.add_nodes_from(abstract_G.nodes(data=True))
            for (u,v), preds in abstract_edge_map.items():
                abstract_plot_G.add_edge(u, v, label=', '.join(sorted(list(set(preds)))))

            abstract_edge_labels = nx.get_edge_attributes(abstract_plot_G, 'label')
            abstract_edge_colors = [color_map.get(l.split(',')[0], 'gray') for l in abstract_edge_labels.values()]

            plt.figure(figsize=(8, 8))
            nx.draw_networkx_nodes(abstract_plot_G, abstract_pos, node_color=abstract_node_colors, node_size=1200, edgecolors='black', linewidths=1.5)
            nx.draw_networkx_edges(abstract_plot_G, abstract_pos, edge_color=abstract_edge_colors, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.9)
            nx.draw_networkx_labels(abstract_plot_G, abstract_pos, labels=abstract_node_labels, font_size=8)
            nx.draw_networkx_edge_labels(abstract_plot_G, abstract_pos, edge_labels=abstract_edge_labels, font_color='red', font_size=7)
            
            plt.title(f"Abstract View - {problem_id}")
            plt.axis('off')
            out_path_abstract = os.path.join(out_dir, f"{problem_id}_graph_abstract.png")
            plt.savefig(out_path_abstract, bbox_inches='tight')
            plt.close()

def save_scene_graph_csv(G, out_dir, problem_id):
    """Save node and edge data of the scene graph as CSV files."""
    logging.info(f"[LOGO Visualization] save_scene_graph_csv called for problem_id={problem_id}")
    import json
    nodes = []
    for n, data in G.nodes(data=True):
        d = data.copy()
        d['id'] = n
        d['all_data'] = json.dumps(data, ensure_ascii=False)
        nodes.append(d)
    edges = []
    logging.info(f"[LOGO Visualization] (CSV) Graph type: {type(G)}. is_multigraph={isinstance(G, (nx.MultiDiGraph, nx.MultiGraph))}")
    try:
        if hasattr(G, 'is_multigraph') and G.is_multigraph():
            for u, v, k, data in _robust_edge_unpack(G.edges(keys=True, data=True)):
                d = data.copy()
                d['source'] = u
                d['target'] = v
                d['key'] = k
                d['all_data'] = json.dumps(data, ensure_ascii=False)
                edges.append(d)
        else:
            for u, v, k, data in _robust_edge_unpack(G.edges(data=True)):
                d = data.copy()
                d['source'] = u
                d['target'] = v
                d['all_data'] = json.dumps(data, ensure_ascii=False)
                edges.append(d)
    except Exception as e:
        logging.error(f"[LOGO Visualization] Error unpacking edges for CSV: {e}")
        logging.error(f"[LOGO Visualization] Edges: {list(G.edges(data=True))}")
        raise
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(nodes).to_csv(os.path.join(out_dir, f"{problem_id}_nodes.csv"), index=False)
    if edges:
        pd.DataFrame(edges).to_csv(os.path.join(out_dir, f"{problem_id}_edges.csv"), index=False)
    else:
        logging.warning(f"[LOGO Visualization] No edges found for problem {problem_id}, skipping edge CSV.")
