import numpy as np
import networkx as nx
import logging
import os
import glob # For finding label files
import cv2 # For image processing if needed for visualization or context

# Local imports
from .metrics import iou_xyxy # Reusing IoU function from metrics.py
from .logger import log_mlflow_param, log_mlflow_metric # For MLflow logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _yolo_to_xyxy(bbox_yolo: list, img_w: int, img_h: int) -> list:
    """
    Converts a YOLO format bounding box [cx, cy, w, h] to absolute pixel [x0, y0, x1, y1].
    Args:
        bbox_yolo (list): Bounding box in YOLO format [cx, cy, w, h] (normalized).
        img_w (int): Image width.
        img_h (int): Image height.
    Returns:
        list: Bounding box in [x0, y0, x1, y1] format (absolute pixels).
    """
    cx, cy, w, h = bbox_yolo
    x0 = (cx - w/2) * img_w
    y0 = (cy - h/2) * img_h
    x1 = (cx + w/2) * img_w
    y1 = (cy + h/2) * img_h
    return [x0, y0, x1, y1]

def _xyxy_to_yolo(bbox_xyxy: list, img_w: int, img_h: int) -> list:
    """
    Converts an absolute pixel bounding box [x0, y0, x1, y1] to normalized YOLO [cx, cy, w, h].
    Args:
        bbox_xyxy (list): Bounding box in [x0, y0, x1, y1] format (absolute pixels).
        img_w (int): Image width.
        img_h (int): Image height.
    Returns:
        list: Bounding box in YOLO format [cx, cy, w, h] (normalized).
    """
    x0, y0, x1, y1 = bbox_xyxy
    cx = ((x0 + x1) / 2) / img_w
    cy = ((y0 + y1) / 2) / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    # Clip to [0,1] to handle potential floating point errors
    cx, cy, w, h = [max(0.0, min(1.0, v)) for v in [cx, cy, w, h]]
    return [cx, cy, w, h]


def load_graph(graph_path: str) -> nx.Graph:
    """
    Loads a graph from a specified path (e.g., GML, GraphML, or a custom JSON format).
    Args:
        graph_path (str): Path to the graph file.
    Returns:
        networkx.Graph: Loaded NetworkX graph.
    Raises:
        FileNotFoundError: If the graph file does not exist.
        Exception: For other loading errors.
    """
    if not os.path.exists(graph_path):
        logging.error(f"Graph file not found: {graph_path}")
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    try:
        # Infer format from extension or try common formats
        if graph_path.endswith('.gml'):
            graph = nx.read_gml(graph_path)
        elif graph_path.endswith('.graphml'):
            graph = nx.read_graphml(graph_path)
        elif graph_path.endswith('.json'):
            # Assuming a custom JSON format, e.g., from symbolic_fusion
            with open(graph_path, 'r') as f:
                data = json.load(f)
            # Reconstruct graph from nodes and edges in JSON
            graph = nx.Graph()
            for node_data in data.get('nodes', []):
                graph.add_node(node_data['id'], **{k: v for k, v in node_data.items() if k != 'id'})
            for edge_data in data.get('edges', []):
                graph.add_edge(edge_data['source'], edge_data['target'], **{k: v for k, v in edge_data.items() if k not in ['source', 'target']})
        else:
            logging.warning(f"Unsupported graph file format for {graph_path}. Attempting to read as GraphML.")
            graph = nx.read_graphml(graph_path) # Default fallback
        logging.info(f"Graph loaded from {graph_path} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph
    except Exception as e:
        logging.error(f"Error loading graph from {graph_path}: {e}")
        raise

def load_yolo_predictions(label_file_path: str, image_width: int, image_height: int) -> list:
    """
    Loads YOLO format predictions from a .txt file and converts them to a list of dicts.
    Args:
        label_file_path (str): Path to the YOLO label (.txt) file.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    Returns:
        list: List of dictionaries, each representing a prediction:
              {'class_id': int, 'bbox_yolo': [cx,cy,w,h], 'bbox_xyxy': [x0,y0,x1,y1], 'score': float}.
              Note: 'score' is a placeholder if not present in YOLO format, typically 1.0 for GT.
    """
    predictions = []
    if not os.path.exists(label_file_path):
        logging.warning(f"YOLO label file not found: {label_file_path}")
        return predictions

    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        bbox_yolo = [cx, cy, w, h]
                        bbox_xyxy = _yolo_to_xyxy(bbox_yolo, image_width, image_height)
                        predictions.append({
                            'class_id': class_id,
                            'bbox_yolo': bbox_yolo,
                            'bbox_xyxy': bbox_xyxy,
                            'score': 1.0 # Assuming 1.0 for ground truth labels loaded as predictions
                        })
                    except ValueError as ve:
                        logging.warning(f"Invalid numeric value in {label_file_path} line: '{line.strip()}'. Error: {ve}")
                else:
                    logging.warning(f"Malformed line in {label_file_path}: '{line.strip()}'. Skipping.")
        logging.debug(f"Loaded {len(predictions)} YOLO predictions from {label_file_path}.")
    except Exception as e:
        logging.error(f"Error loading YOLO predictions from {label_file_path}: {e}")
    return predictions

def non_max_suppression(predictions: list, iou_threshold: float = 0.5, score_threshold: float = 0.01) -> list:
    """
    Applies Non-Maximum Suppression (NMS) to a list of predictions.
    Args:
        predictions (list): List of dictionaries, each with 'bbox_xyxy', 'score', 'class_id'.
        iou_threshold (float): IoU threshold for suppressing overlapping boxes.
        score_threshold (float): Minimum score to consider a prediction.
    Returns:
        list: Filtered list of predictions after NMS.
    """
    if not predictions:
        return []

    # Filter by score threshold first
    predictions = [p for p in predictions if p['score'] >= score_threshold]
    if not predictions:
        return []

    # Sort predictions by score in descending order
    predictions.sort(key=lambda x: x['score'], reverse=True)

    keep_indices = []
    suppressed = [False] * len(predictions)

    for i in range(len(predictions)):
        if suppressed[i]:
            continue

        keep_indices.append(i)
        current_bbox = predictions[i]['bbox_xyxy']
        current_class_id = predictions[i]['class_id']

        for j in range(i + 1, len(predictions)):
            if suppressed[j]:
                continue

            # Only suppress if same class and high IoU
            if predictions[j]['class_id'] == current_class_id:
                iou = iou_xyxy(current_bbox, predictions[j]['bbox_xyxy'])
                if iou >= iou_threshold:
                    suppressed[j] = True
    
    filtered_predictions = [predictions[i] for i in keep_indices]
    logging.debug(f"NMS reduced {len(predictions)} to {len(filtered_predictions)} predictions.")
    return filtered_predictions

def check_relation(graph: nx.Graph, yolo_predictions: list, relation_type: str = 'spatial_proximity', threshold: float = 50.0) -> list:
    """
    Checks if a specified relation exists between objects in the graph and matches YOLO predictions.
    This function is conceptual and depends heavily on the graph's structure and attributes.
    Args:
        graph (nx.Graph): The symbolic graph of objects. Nodes should have 'bbox_xyxy' or 'centroid' attributes.
        yolo_predictions (list): List of YOLO predictions (from load_yolo_predictions).
        relation_type (str): The type of relation to check (e.g., 'spatial_proximity', 'contains', 'aligned').
        threshold (float): A threshold relevant to the relation (e.g., distance for proximity).
    Returns:
        list: A list of dictionaries describing detected relations, potentially linking to YOLO predictions.
              Example: [{'relation': 'spatial_proximity', 'source_obj_id': 0, 'target_obj_id': 1, 'distance': 25.0, 'matched_yolo_preds': [idx1, idx2]}]
    """
    detected_relations = []
    
    # Map YOLO predictions to graph nodes based on spatial overlap (e.g., IoU)
    # This is a critical step: how do predictions map to abstract graph nodes?
    # For simplicity, assume graph nodes are indexed 0 to N-1 and correspond to initial detections.
    # A more robust system would use a matching algorithm.

    # For this example, we'll assume graph nodes already have 'bbox_xyxy' or 'centroid' attributes
    # from the symbolic_fusion step.

    logging.info(f"Checking for relation '{relation_type}' in graph...")

    if relation_type == 'spatial_proximity':
        for u, v, data in graph.edges(data=True):
            if 'distance' in data and data['distance'] < threshold:
                relation_info = {
                    'relation': 'spatial_proximity',
                    'source_obj_id': u,
                    'target_obj_id': v,
                    'distance': data['distance']
                }
                
                # Try to find corresponding YOLO predictions for these graph objects
                # This is a simplified matching. In reality, you'd use the original detections
                # that formed the graph nodes.
                matched_yolo_preds = []
                # Assuming graph nodes have a 'bbox_xyxy' attribute
                if u in graph.nodes and 'bbox_xyxy' in graph.nodes[u] and \
                   v in graph.nodes and 'bbox_xyxy' in graph.nodes[v]:
                    
                    # Find YOLO predictions that overlap significantly with graph node's bbox
                    for idx, yolo_pred in enumerate(yolo_predictions):
                        if iou_xyxy(graph.nodes[u]['bbox_xyxy'], yolo_pred['bbox_xyxy']) > 0.8:
                            matched_yolo_preds.append(idx)
                        if iou_xyxy(graph.nodes[v]['bbox_xyxy'], yolo_pred['bbox_xyxy']) > 0.8:
                            matched_yolo_preds.append(idx)
                
                if matched_yolo_preds:
                    relation_info['matched_yolo_preds_indices'] = list(set(matched_yolo_preds))
                
                detected_relations.append(relation_info)
    
    # Add other relation types here (e.g., 'contains', 'aligned', 'same_shape')
    # These would typically be attributes on the graph edges or inferred from node attributes.
    elif relation_type == 'contains':
        for u, v, data in graph.edges(data=True):
            if 'contains' in data and data['contains'] is True: # Assuming 'contains' is a boolean flag on edge
                detected_relations.append({
                    'relation': 'contains',
                    'container_obj_id': u,
                    'contained_obj_id': v
                })
    elif relation_type == 'aligned':
        for u, v, data in graph.edges(data=True):
            if 'vertically-aligned' in data and data['vertically-aligned'] is True:
                detected_relations.append({
                    'relation': 'vertically-aligned',
                    'obj1_id': u,
                    'obj2_id': v
                })
            if 'horizontally-aligned' in data and data['horizontally-aligned'] is True:
                detected_relations.append({
                    'relation': 'horizontally-aligned',
                    'obj1_id': u,
                    'obj2_id': v
                })
    else:
        logging.warning(f"Unsupported relation type: {relation_type}")

    logging.info(f"Found {len(detected_relations)} instances of relation type '{relation_type}'.")
    return detected_relations

def label_quality(yolo_predictions: list, ground_truth_labels: list, iou_threshold: float = 0.5) -> dict:
    """
    Assesses the quality of YOLO predictions against ground truth labels.
    Calculates False Positives (FP), False Negatives (FN), and potentially
    identifies low-confidence predictions.
    Args:
        yolo_predictions (list): List of dictionaries for predictions
                                 (e.g., from non_max_suppression, with 'bbox_xyxy', 'score', 'class_id').
        ground_truth_labels (list): List of dictionaries for ground truths
                                    (e.g., from load_yolo_predictions, with 'bbox_xyxy', 'class_id').
        iou_threshold (float): IoU threshold for considering a prediction a match to a ground truth.
    Returns:
        dict: A report on label quality including FP, FN counts.
    """
    quality_report = {
        'false_positives': 0,
        'false_negatives': 0,
        'true_positives': 0,
        'low_confidence_preds': 0,
        'total_predictions': len(yolo_predictions),
        'total_ground_truths': len(ground_truth_labels)
    }

    if not yolo_predictions and not ground_truth_labels:
        logging.info("No predictions or ground truths to evaluate label quality.")
        return quality_report

    # Mark ground truths as matched or not
    matched_gts = [False] * len(ground_truth_labels)

    for pred in yolo_predictions:
        if pred['score'] < 0.3: # Example threshold for 'low confidence'
            quality_report['low_confidence_preds'] += 1
            # A low-confidence prediction that doesn't match a GT could be an FP, but we separate it.
            # If it matches a GT, it's still a low-conf TP.
            
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth_labels):
            if gt['class_id'] == pred['class_id'] and not matched_gts[gt_idx]:
                current_iou = iou_xyxy(pred['bbox_xyxy'], gt['bbox_xyxy'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            quality_report['true_positives'] += 1
            matched_gts[best_gt_idx] = True
        else:
            quality_report['false_positives'] += 1
    
    # False negatives are ground truths that were not matched by any prediction
    quality_report['false_negatives'] = sum(1 for matched in matched_gts if not matched)

    logging.info(f"Label quality report: {quality_report}")
    return quality_report

def log_graph_metadata(graph: nx.Graph, mlflow_enabled: bool = False):
    """
    Logs metadata about the graph structure.
    Args:
        graph (nx.Graph): The NetworkX graph.
        mlflow_enabled (bool): If True, also logs to MLflow.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    
    # Calculate average degree
    degrees = [degree for node, degree in graph.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0

    # Check for connected components
    num_connected_components = nx.number_connected_components(graph)
    
    graph_metadata = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_node_degree": float(avg_degree),
        "num_connected_components": num_connected_components
    }

    logging.info(f"Graph metadata: {graph_metadata}")

    if mlflow_enabled:
        log_mlflow_param("graph_num_nodes", num_nodes)
        log_mlflow_param("graph_num_edges", num_edges)
        log_mlflow_param("graph_avg_node_degree", f"{avg_degree:.2f}")
        log_mlflow_param("graph_num_connected_components", num_connected_components)
        logging.info("Graph metadata logged to MLflow.")

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dummy image and label files for testing
    dummy_img_dir = "dummy_images"
    dummy_lbl_dir = "dummy_labels"
    os.makedirs(dummy_img_dir, exist_ok=True)
    os.makedirs(dummy_lbl_dir, exist_ok=True)

    img_path = os.path.join(dummy_img_dir, "test_image.jpg")
    lbl_path = os.path.join(dummy_lbl_dir, "test_image.txt")
    
    # Create a dummy image (just a placeholder, content doesn't matter for this test)
    cv2.imwrite(img_path, np.zeros((640, 480, 3), dtype=np.uint8))

    # Create a dummy label file
    with open(lbl_path, 'w') as f:
        f.write("0 0.2 0.2 0.1 0.1\n") # Class 0, small box
        f.write("1 0.5 0.5 0.2 0.2\n") # Class 1, medium box
        f.write("0 0.21 0.21 0.11 0.11\n") # Class 0, duplicate/overlapping with first
        f.write("1 0.7 0.7 0.15 0.15\n") # Class 1, another box

    img_w, img_h = 640, 480

    # Test load_yolo_predictions
    yolo_preds = load_yolo_predictions(lbl_path, img_w, img_h)
    print("\n--- Loaded YOLO Predictions ---")
    for p in yolo_preds:
        print(p)

    # Test non_max_suppression
    # Create some overlapping predictions for NMS test
    nms_test_preds = [
        {'bbox_xyxy': [10, 10, 50, 50], 'score': 0.9, 'class_id': 0},
        {'bbox_xyxy': [12, 12, 52, 52], 'score': 0.85, 'class_id': 0}, # Overlapping with first
        {'bbox_xyxy': [100, 100, 150, 150], 'score': 0.7, 'class_id': 1},
        {'bbox_xyxy': [102, 102, 152, 152], 'score': 0.65, 'class_id': 1}, # Overlapping with third
        {'bbox_xyxy': [200, 200, 210, 210], 'score': 0.2, 'class_id': 0} # Low score
    ]
    filtered_preds = non_max_suppression(nms_test_preds, iou_threshold=0.5, score_threshold=0.3)
    print("\n--- NMS Filtered Predictions (score > 0.3, IoU > 0.5) ---")
    for p in filtered_preds:
        print(p)

    # Test check_relation (requires a graph, using a dummy one)
    G = nx.Graph()
    G.add_node(0, bbox_xyxy=[10, 10, 100, 100], centroid=[55, 55])
    G.add_node(1, bbox_xyxy=[120, 120, 200, 200], centroid=[160, 160])
    G.add_node(2, bbox_xyxy=[15, 15, 60, 60], centroid=[37, 37]) # Close to node 0
    G.add_edge(0, 1, distance=np.linalg.norm(np.array([55,55]) - np.array([160,160])))
    G.add_edge(0, 2, distance=np.linalg.norm(np.array([55,55]) - np.array([37,37])))
    G.add_edge(0, 3, relation_type='contains', contains=True) # Dummy relation

    relations = check_relation(G, yolo_preds, relation_type='spatial_proximity', threshold=100)
    print("\n--- Detected Spatial Proximity Relations ---")
    for r in relations:
        print(r)
    
    relations_contains = check_relation(G, yolo_preds, relation_type='contains')
    print("\n--- Detected Contains Relations ---")
    for r in relations_contains:
        print(r)

    # Test label_quality
    # Use NMS filtered predictions as model output, and original YOLO preds as GT
    gt_for_quality = []
    for p in yolo_preds:
        gt_for_quality.append({'bbox_xyxy': p['bbox_xyxy'], 'class_id': p['class_id']})

    quality_report = label_quality(filtered_preds, gt_for_quality, iou_threshold=0.5)
    print("\n--- Label Quality Report ---")
    print(quality_report)

    # Test log_graph_metadata
    print("\n--- Logging Graph Metadata ---")
    log_graph_metadata(G, mlflow_enabled=False) # Set to True to test MLflow logging

    # Clean up dummy files and directories
    os.remove(img_path)
    os.remove(lbl_path)
    os.rmdir(dummy_img_dir)
    os.rmdir(dummy_lbl_dir)
    print("\nCleaned up dummy files and directories.")
