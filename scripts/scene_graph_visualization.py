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
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import logging
import pandas as pd

__all__ = [
    'save_scene_graph_visualization',
    'save_scene_graph_csv',
]

def save_scene_graph_visualization(G, image_path, out_dir, problem_id):
    """Save a visualization of the scene graph G overlaid on the real image."""
    os.makedirs(out_dir, exist_ok=True)
    # Node positions: use centroid, else first vertex, else (0,0)
    import numpy as np
    pos = {}
    for n, data in G.nodes(data=True):
        if 'centroid' in data:
            # Ensure centroid is a numpy array for compatibility
            pos[n] = tuple(np.array(data['centroid'], dtype=np.float64))
        elif 'vertices' in data and data['vertices']:
            # Ensure first vertex is a numpy array for compatibility
            pos[n] = tuple(np.array(data['vertices'][0], dtype=np.float64))
        else:
            pos[n] = (0.0, 0.0)

    # Ensure all nodes are shown, even if disconnected (add to plotG if missing)
    import networkx as nx
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        plotG = nx.DiGraph()
        plotG.add_nodes_from(G.nodes(data=True))
        seen = set()
        for u, v, k, data in _robust_edge_unpack(G.edges(keys=True, data=True)):
            if (u, v) not in seen:
                plotG.add_edge(u, v, **data)
                seen.add((u, v))
    else:
        plotG = nx.DiGraph()
        plotG.add_nodes_from(G.nodes(data=True))
        for u, v, k, data in _robust_edge_unpack(G.edges(data=True)):
            plotG.add_edge(u, v, **data)

    # If any node is missing from plotG, add it (ensures isolated nodes are shown)
    for n, data in G.nodes(data=True):
        if n not in plotG:
            plotG.add_node(n, **data)

    # 1. Save graph overlayed on real image (if image exists)
    # For visualization: if MultiDiGraph, convert to DiGraph for plotting edge labels
    import json
    from collections import defaultdict
    import matplotlib.patches as mpatches
    
    # --- Node color/border/label logic ---
    node_labels = {}
    node_colors = []
    node_border_colors = []
    for n, data in plotG.nodes(data=True):
        # Color and border logic
        if data.get('is_motif'):
            node_colors.append('gold')
            node_border_colors.append('goldenrod')
        elif not data.get('geometry_valid', True):
            node_colors.append('red')
            node_border_colors.append('black')
        elif data.get('object_type') == 'line':
            node_colors.append('orange')
            node_border_colors.append('black')
        elif data.get('object_type') in ('arc', 'point'):
            node_colors.append('purple')
            node_border_colors.append('black')
        elif data.get('gnn_score') is not None:
            node_colors.append('lightgreen')
            node_border_colors.append('green')
        else:
            node_colors.append('skyblue')
            node_border_colors.append('blue')
        # Label logic: show program index, command, and shape_label for LOGO clarity
        label = f"{n}"
        if data.get('action_index') is not None:
            label += f"\n[action {data['action_index']}]"
        if data.get('command'):
            label += f"\n{data['command']}"
        if data.get('shape_label'):
            label += f"\n{data['shape_label']}"
        if not data.get('geometry_valid', True):
            label += '\n[degenerate]'
        elif data.get('object_type') == 'line':
            label += '\n[line]'
        elif data.get('object_type') == 'arc':
            label += '\n[arc]'
        elif data.get('object_type') == 'point':
            label += '\n[point]'
        fv = data.get('feature_valid', {})
        if fv:
            invalid_feats = [k for k, v in fv.items() if not v]
            if invalid_feats:
                label += f"\n[invalid: {', '.join(invalid_feats)}]"
        node_labels[n] = label

    # --- Edge color/label logic ---
    edge_labels = {}
    edge_colors = []
    edge_list = []
    edge_pred_map = defaultdict(list)
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        import networkx as nx
        plotG = nx.DiGraph()
        plotG.add_nodes_from(G.nodes(data=True))
        seen = set()
        for u, v, k, data in _robust_edge_unpack(G.edges(keys=True, data=True)):
            if (u, v) not in seen:
                plotG.add_edge(u, v, **data)
                seen.add((u, v))
            pred = data.get('predicate', '')
            edge_pred_map[(u, v)].append(pred)
    else:
        for u, v, k, data in _robust_edge_unpack(G.edges(data=True)):
            pred = data.get('predicate', '')
            edge_pred_map[(u, v)].append(pred)
    # Assign edge colors and labels
    color_map = {
        'vl_sim': 'red',
        'part_of': 'orange',
        'near': 'green',
        'para': 'green',
        'aspect_sim': 'green',
        'programmatic_sim': 'blue',
        'kb_sim': 'purple',
        'global_stat_sim': 'brown',
    }
    for (u, v), preds in edge_pred_map.items():
        edge_list.append((u, v))
        # Pick color by first predicate, fallback to gray
        color = color_map.get(preds[0], 'gray') if preds else 'gray'
        edge_colors.append(color)
        edge_labels[(u, v)] = ', '.join(sorted(set(preds)))

    # Filter out labels for self-loops or edges between co-located nodes to prevent drawing errors
    drawable_edge_labels = {
        (u, v): label for (u, v), label in edge_labels.items()
        if u != v and pos.get(u) != pos.get(v)
    }
    if len(drawable_edge_labels) < len(edge_labels):
        logging.warning(f"Skipped {len(edge_labels) - len(drawable_edge_labels)} edge labels for self-loops or co-located nodes in {problem_id}.")

    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 8))
            plt.imshow(img_rgb)
            nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=700, edgecolors=node_border_colors, linewidths=2)
            nx.draw_networkx_edges(plotG, pos, edgelist=edge_list, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=2, alpha=0.8)
            nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=9, font_family='sans-serif', font_color='black')
            # Legend
            legend_handles = [
                mpatches.Patch(color='gold', label='Motif'),
                mpatches.Patch(color='red', label='Degenerate'),
                mpatches.Patch(color='orange', label='Line'),
                mpatches.Patch(color='purple', label='Arc/Point'),
                mpatches.Patch(color='lightgreen', label='Valid Polygon'),
                mpatches.Patch(color='skyblue', label='Other'),
                mpatches.Patch(color='blue', label='Programmatic Edge'),
                mpatches.Patch(color='purple', label='KB Edge'),
                mpatches.Patch(color='brown', label='Global Stat Edge'),
                mpatches.Patch(color='gray', label='Other Edge'),
            ]
            plt.legend(handles=legend_handles, loc='upper right', fontsize=8)
            # Draw edge labels using filtered labels; catch errors to avoid crashes
            try:
                nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='blue', font_size=7)
            except ValueError as e:
                logging.warning(f"[LOGO Visualization] Edge label drawing failed for {problem_id}: {e}")
            plt.axis('off')
            out_path_overlay = os.path.join(out_dir, f"{problem_id}_scene_graph.png")
            plt.savefig(out_path_overlay, bbox_inches='tight')
            plt.close()

    # 2. Save pure matplotlib graph visualization (no image)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(plotG, pos, node_color=node_colors, node_size=700, edgecolors=node_border_colors, linewidths=2)
    nx.draw_networkx_edges(plotG, pos, edgelist=edge_list, edge_color=edge_colors, arrows=True, arrowstyle='-|>', width=2, alpha=0.8)
    nx.draw_networkx_labels(plotG, pos, labels=node_labels, font_size=9, font_family='sans-serif', font_color='black')
    legend_handles = [
        mpatches.Patch(color='gold', label='Motif'),
        mpatches.Patch(color='red', label='Degenerate'),
        mpatches.Patch(color='orange', label='Line'),
        mpatches.Patch(color='purple', label='Arc/Point'),
        mpatches.Patch(color='lightgreen', label='Valid Polygon'),
        mpatches.Patch(color='skyblue', label='Other'),
        mpatches.Patch(color='blue', label='Programmatic Edge'),
        mpatches.Patch(color='purple', label='KB Edge'),
        mpatches.Patch(color='brown', label='Global Stat Edge'),
        mpatches.Patch(color='gray', label='Other Edge'),
    ]
    plt.legend(handles=legend_handles, loc='upper right', fontsize=8)
    # Draw edge labels using filtered labels; catch errors to avoid crashes
    try:
        nx.draw_networkx_edge_labels(plotG, pos, edge_labels=drawable_edge_labels, font_color='blue', font_size=7)
    except ValueError as e:
        logging.warning(f"[LOGO Visualization] Edge label drawing failed for {problem_id}: {e}")
    plt.axis('off')
    out_path_graph = os.path.join(out_dir, f"{problem_id}_graph_only.png")
    plt.savefig(out_path_graph, bbox_inches='tight')
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
    pd.DataFrame(edges).to_csv(os.path.join(out_dir, f"{problem_id}_edges.csv"), index=False)
