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
            logging.warning(f"[scene_graphs_building.visualization] Skipping unexpectedly short/long edge tuple: {edge}")
            continue
        if not isinstance(data, dict):
            logging.warning(f"[scene_graphs_building.visualization] Edge data not dict: {repr(data)} for edge {edge}; using empty dict.")
            data = {}
        yield u, v, k, data
import logging
import os
import matplotlib.pyplot as plt

# --- GNN Training/Validation Logging Utilities ---
def log_gnn_training(epoch, train_loss, val_loss, val_acc=None, patience=None, best_val=None):
    msg = f"[GNN][Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    if val_acc is not None:
        msg += f" | Val Acc: {val_acc:.4f}"
    if best_val is not None:
        msg += f" | Best Val: {best_val:.4f}"
    if patience is not None:
        msg += f" | Patience: {patience}"
    logging.info(msg)

def plot_gnn_training_curves(train_losses, val_losses, val_accs=None, out_dir="visualizations/gnn"):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    if val_accs is not None:
        plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.title("GNN Training Curves")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "gnn_training_curves.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"[GNN] Saved training curves to {out_path}")
def save_fallback_centroid_visualizations(samples, output_dir):
    """
    Save visualizations of fallback centroid cases to output_dir.
    Each sample should have 'vertices' and either 'cx'/'cy' or 'centroid'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        vertices = np.array(sample['vertices'])
        # Use computed centroid if available, else fallback to original
        if 'cx' in sample and 'cy' in sample:
            centroid = [sample['cx'], sample['cy']]
        else:
            centroid = sample.get('centroid', None)
        node_id = sample.get('id', f'sample_{i}')
        save_path = os.path.join(output_dir, f"physics_fallback_{node_id}.png")
        plt.figure(figsize=(6, 6))
        plt.plot(vertices[:,0], vertices[:,1], 'b-', lw=2, label='Shape')
        # Only plot centroid if valid (not None, not [None, None])
        if centroid is not None and all(c is not None for c in centroid):
            plt.scatter([centroid[0]], [centroid[1]], c='r', s=80, label='Centroid')
        else:
            plt.text(0.5, 0.5, 'Centroid: N/A', ha='center', va='center', fontsize=10, color='red', transform=plt.gca().transAxes)
        plt.title(f"Node {node_id} Fallback Centroid")
        plt.axis('equal')
        plt.legend()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved physics fallback visualization for node {node_id} to {save_path}")
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
    G = None
    # Accept both MultiDiGraph and DiGraph, and handle direct graph input
    if scene_graph is not None:
        if isinstance(scene_graph, dict) and 'graph' in scene_graph:
            G = scene_graph['graph']
        elif isinstance(scene_graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            G = scene_graph
    if G is None or not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        logging.error(f"Scene graph for {base_name} is not a valid NetworkX graph object: {type(G)}. Skipping visualization for this problem.")
        return

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    # --- LOGO mode detection ---
    is_logo_mode = False
    # Try to detect LOGO mode from scene_graph or base_name
    if scene_graph and scene_graph.get('mode', None) == 'logo':
        is_logo_mode = True
    elif 'logo' in str(base_name).lower():
        is_logo_mode = True

    # Only show program-graph and minimal geometric edges in LOGO mode
    # Robust edge unpacking for MultiDiGraph/DiGraph
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        all_edges = list(_robust_edge_unpack(G.edges(keys=True, data=True)))
        if is_logo_mode:
            allowed_predicates = {'next_action', 'turn_left', 'turn_right', 'length_sim', 'angle_sim', 'adjacent_endpoints', 'intersects'}
            edges_to_draw = [(u, v, d) for u, v, k, d in all_edges if d.get('predicate') in allowed_predicates]
        else:
            edges_to_draw = [(u, v, d) for u, v, k, d in all_edges]
    else:
        all_edges = list(_robust_edge_unpack(G.edges(data=True)))
        if is_logo_mode:
            allowed_predicates = {'next_action', 'turn_left', 'turn_right', 'length_sim', 'angle_sim', 'adjacent_endpoints', 'intersects'}
            edges_to_draw = [(u, v, d) for u, v, k, d in all_edges if d.get('predicate') in allowed_predicates]
        else:
            edges_to_draw = [(u, v, d) for u, v, k, d in all_edges]

    # Draw nodes and edges
    pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else None
    node_labels = {n: G.nodes[n].get('shape_label', n) for n in G.nodes()}
    node_colors = ['skyblue' for _ in G.nodes()]
    node_border_colors = ['blue' for _ in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors=node_border_colors, linewidths=2)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edges_to_draw], arrows=True, arrowstyle='-|>', width=2.0, alpha=0.8, edge_color='blue')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_family='sans-serif', font_color='black')
    # Add LOGO mode banner if active
    if is_logo_mode:
        plt.gcf().text(0.5, 0.95, 'LOGO MODE: Program Graph Only', fontsize=12, color='navy', ha='center', va='top', bbox=dict(facecolor='lightyellow', alpha=0.7, edgecolor='navy'))
        # No advanced logic, no edge labels, no advanced features
        rules = scene_graph.get('rules') if scene_graph and isinstance(scene_graph, dict) else None
        if rules is not None:
            plt.gcf().text(0.01, 0.01, f"Rules: {getattr(rules, 'tree_', None)}", fontsize=8, color='purple', ha='left', va='bottom')
        plt.title(f"Scene Graph: {base_name}")
        plt.axis('off')
    else:
        plt.title(f"Scene Graph: {base_name}")
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
        import networkx as nx
        is_nx_graph = isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
        if is_nx_graph and len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 1 else {n: (0.5, 0.5) for n in G.nodes()}
            width, height = pil_image.size
            # Only draw minimal edges in LOGO mode
            if is_logo_mode:
                if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                    all_edges = list(_robust_edge_unpack(G.edges(keys=True, data=True)))
                    allowed_predicates = {'next_action', 'turn_left', 'turn_right', 'length_sim', 'angle_sim', 'adjacent_endpoints', 'intersects'}
                    edges_to_draw = [(u, v) for u, v, k, d in all_edges if d.get('predicate') in allowed_predicates]
                else:
                    all_edges = list(_robust_edge_unpack(G.edges(data=True)))
                    allowed_predicates = {'next_action', 'turn_left', 'turn_right', 'length_sim', 'angle_sim', 'adjacent_endpoints', 'intersects'}
                    edges_to_draw = [(u, v) for u, v, k, d in all_edges if d.get('predicate') in allowed_predicates]
            else:
                if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                    edges_to_draw = []
                    for edge in G.edges(keys=True, data=True):
                        if len(edge) == 4:
                            u, v, k, d = edge
                            edges_to_draw.append((u, v))
                        elif len(edge) == 3:
                            u, v, d = edge
                            edges_to_draw.append((u, v))
                        else:
                            logging.warning(f"[scene_graphs_building.visualization] Edge tuple malformed (length={len(edge)}): {edge}")
                            continue
                else:
                    edges_to_draw = []
                    for edge in G.edges(data=True):
                        if len(edge) == 3:
                            u, v, d = edge
                            edges_to_draw.append((u, v))
                        else:
                            logging.warning(f"[scene_graphs_building.visualization] Edge tuple malformed (length={len(edge)}): {edge}")
                            continue
            for u, v in edges_to_draw:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                draw.line([
                    int(x1 * width), int(y1 * height),
                    int(x2 * width), int(y2 * height)
                ], fill=(255, 0, 0, 128), width=2)
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