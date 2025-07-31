import os
import cv2
import logging
import matplotlib
matplotlib.use('Agg')

def save_feedback_images(image, mask, base_name, feedback_dir, scene_graph=None):
    """Saves input image, mask, scene graph visualization, and side-by-side comparisons for feedback."""
    logging.info(f"[save_feedback_images] Called with base_name={base_name}, image type={type(image)}, mask type={type(mask)}, feedback_dir={feedback_dir}, scene_graph type={type(scene_graph)}")
    import matplotlib.pyplot as plt
    os.makedirs(feedback_dir, exist_ok=True)
    img_save_path = os.path.join(feedback_dir, f"{base_name}_input.png")
    mask_save_path = os.path.join(feedback_dir, f"{base_name}_mask.png")
    side_by_side_path = os.path.join(feedback_dir, f"{base_name}_side_by_side.png")
    graph_img_path = os.path.join(feedback_dir, f"{base_name}_graph.png")
    img_graph_side_by_side_path = os.path.join(feedback_dir, f"{base_name}_img_graph.png")
    actmap_path = os.path.join(feedback_dir, f"{base_name}_actmap.png") # For visualization of edges on image

    # Convert PIL image to numpy if needed (robust)
    import numpy as np
    if not hasattr(image, 'ndim') or not isinstance(image, np.ndarray):
        image = np.array(image)
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
    import networkx as nx
    if scene_graph is not None and 'graph' in scene_graph:
        G = scene_graph['graph']
    else:
        G = None

    if not isinstance(G, nx.Graph):
        logging.error(f"Scene graph for {base_name} is not a valid NetworkX graph object: {type(G)}. Skipping visualization for this problem.")
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else None
        # Node labels: show shape_label and image name (from original_image_path or image_path)
        def get_img_name(node):
            path = node.get('original_image_path') or node.get('image_path')
            if path:
                return os.path.basename(path)
            return ''
        from src.scene_graphs_building.config import SHAPE_MAP
        def normalize_shape_label(lbl):
            if not lbl:
                return None
            lbl = lbl.lower().replace("_", " ").replace("-", " ").strip()
            if lbl in ("positive", "negative", "pos", "neg"):
                return None
            return SHAPE_MAP.get(lbl, lbl)
        node_labels = {}
        node_colors = []
        for n in G.nodes():
            node = G.nodes[n]
            raw = node.get('shape_label', n)
            norm = normalize_shape_label(raw)
            label = norm if norm else str(raw)
            img_name = get_img_name(node)
            gnn_score = node.get('gnn_score')
            if node.get('is_motif'):
                label = f"[motif]\n{label}"
                node_colors.append('gold')
            elif gnn_score is not None:
                label = f"{label}\nGNN:{gnn_score:.2f}"
                node_colors.append('lightgreen')
            else:
                node_colors.append('skyblue')
            if img_name:
                node_labels[n] = f"{label}\n{img_name}"
            else:
                node_labels[n] = label
        # Draw edges: color VLM edges red, motif edges orange, others gray
        edges_to_draw = list(G.edges(data=True))
        vl_edges = [(u,v) for u,v,d in edges_to_draw if d.get('predicate')=='vl_sim']
        motif_edges = [(u,v) for u,v,d in edges_to_draw if d.get('predicate')=='part_of']
        other_edges = [(u,v) for u,v,d in edges_to_draw if d.get('predicate') not in ('vl_sim','part_of')]
        import networkx as nx
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
        nx.draw_networkx_edges(G, pos, edgelist=other_edges, arrows=True, arrowstyle='-|>', width=1.5, alpha=0.7, edge_color='gray')
        nx.draw_networkx_edges(G, pos, edgelist=vl_edges, arrows=True, arrowstyle='-|>', width=2.5, alpha=0.8, edge_color='red')
        nx.draw_networkx_edges(G, pos, edgelist=motif_edges, arrows=True, arrowstyle='-|>', width=2.5, alpha=0.8, edge_color='orange')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')
        from collections import defaultdict
        combined = defaultdict(list)
        for u, v, d in edges_to_draw:
            pred = d.get('predicate', '')
            if pred:
                combined[(u, v)].append(pred)
        edge_labels = { (u, v): ", ".join(sorted(set(preds))) for (u, v), preds in combined.items() }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=9)
        # Annotate rules if present
        rules = scene_graph.get('rules') if scene_graph else None
        if rules is not None:
            plt.gcf().text(0.01, 0.01, f"Rules: {getattr(rules, 'tree_', None)}", fontsize=8, color='purple', ha='left', va='bottom')
        plt.title(f"Scene Graph: {base_name}")
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No graph', ha='center', va='center', fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(graph_img_path, bbox_inches='tight')
    plt.close()

    # Always attempt to save side-by-side image and graph, even if graph is empty
    try:
        graph_img = cv2.imread(graph_img_path)
        if graph_img is not None:
            # Resize graph image to match input image height
            if graph_img.shape[0] != img_bgr.shape[0]:
                scale = img_bgr.shape[0] / graph_img.shape[0]
                new_w = int(graph_img.shape[1] * scale)
                graph_img = cv2.resize(graph_img, (new_w, img_bgr.shape[0]), interpolation=cv2.INTER_AREA)
            img_graph_side_by_side = cv2.hconcat([img_bgr, graph_img])
            cv2.imwrite(img_graph_side_by_side_path, img_graph_side_by_side)
        else:
            logging.warning(f"Graph image not found for {base_name}, skipping side-by-side save.")
    except Exception as e:
        logging.warning(f"Failed to create side-by-side image and graph for {base_name}: {e}")

    # Visualization of edges on image (Activation Map style)
    try:
        from PIL import Image, ImageDraw
        pil_image = Image.open(os.path.join(feedback_dir, f"{base_name}_input.png")).convert('RGBA')
        draw = ImageDraw.Draw(pil_image)
        # Robust check for NetworkX graph type
        import networkx as nx
        is_nx_graph = isinstance(G, nx.Graph)
        if is_nx_graph and len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else {n: (0.5, 0.5) for n in G.nodes()}
            # Draw edges as lines
            width, height = pil_image.size
            for u, v in G.edges():
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                draw.line([
                    int(x1 * width), int(y1 * height),
                    int(x2 * width), int(y2 * height)
                ], fill=(255, 0, 0, 128), width=2)
            # Draw nodes as circles
            for n in G.nodes():
                x, y = pos[n]
                r = 8
                draw.ellipse([
                    int(x * width) - r, int(y * height) - r,
                    int(x * width) + r, int(y * height) + r
                ], fill=(0, 255, 0, 128))
        pil_image.save(actmap_path)
    except Exception as e:
        logging.warning(f"Failed to create activation map image for {base_name}: {e}")