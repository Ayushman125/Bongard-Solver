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
        # Robustly check if G is a valid NetworkX graph
        is_nx_graph = hasattr(G, 'nodes') and hasattr(G, 'edges')
        if not is_nx_graph:
            logging.error(f"Scene graph for {base_name} is not a valid NetworkX graph object: {type(G)}. Attempting to visualize anyway.")
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            plt.figure(figsize=(5, 5))
            if is_nx_graph and len(G.nodes) > 0:
                pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else None
                # Node labels: show shape_label and image name (from original_image_path or image_path)
                def get_img_name(node):
                    path = node.get('original_image_path') or node.get('image_path')
                    if path:
                        return os.path.basename(path)
                    return ''
                node_labels = {}
                for n in G.nodes():
                    node = G.nodes[n]
                    label = str(node.get('shape_label', n))
                    img_name = get_img_name(node)
                    if img_name:
                        node_labels[n] = f"{label}\n{img_name}"
                    else:
                        node_labels[n] = label
                # --- Predicate Filtering ---
                allowed_predicates = set(['left_of', 'right_of', 'above', 'below', 'contains'])
                edges_to_draw = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('predicate') in allowed_predicates]
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
                # Draw only filtered edges
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, d in edges_to_draw], arrows=True, arrowstyle='-|>', width=1.5, alpha=0.7)
                nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')
                # Edge labels: use 'predicate' if present, but only for filtered edges
                edge_labels = {(u, v): d.get('predicate', '') for u, v, d in edges_to_draw}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=9)
                plt.title(f"Scene Graph: {base_name}")
                plt.axis('off')
            else:
                plt.text(0.5, 0.5, 'No graph', ha='center', va='center', fontsize=12)
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(graph_img_path, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logging.error(f"Failed to save scene graph visualization for {base_name}: {e}")

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