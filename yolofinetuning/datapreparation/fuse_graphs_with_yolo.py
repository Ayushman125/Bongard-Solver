import networkx as nx
import json
import numpy as np
import os
from collections import defaultdict
import logging # Added for logging

try:
    import mlflow
except ImportError:
    mlflow = None
    logging.warning("MLflow not installed. MLflow logging will be skipped.")

def load_graph(path: str):
    """
    Load a relational graph from a JSON file.
    The JSON is expected to be in NetworkX node-link format.
    Args:
        path (str): Path to the JSON graph file.
    Returns:
        networkx.Graph: The loaded graph.
    """
    if not os.path.exists(path):
        logging.error(f"Graph file not found: {path}")
        raise FileNotFoundError(f"Graph file not found: {path}")
    try:
        with open(path, 'r') as f:
            graph_data = json.load(f)
        graph = nx.node_link_graph(graph_data)
        logging.info(f"Loaded graph from {path} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph
    except Exception as e:
        logging.error(f"Error loading graph from {path}: {e}")
        raise

def load_yolo(path: str):
    """
    Load YOLO predictions from a text file.
    Expected format per line: class_id cx cy w h [score]
    Args:
        path (str): Path to the YOLO prediction text file.
    Returns:
        list: A list of dictionaries, each representing a prediction.
              Example: [{'class': 0, 'x': 0.5, 'y': 0.5, 'w': 0.1, 'h': 0.1, 'score': 0.9}]
    """
    if not os.path.exists(path):
        logging.error(f"YOLO prediction file not found: {path}")
        raise FileNotFoundError(f"YOLO prediction file not found: {path}")
    preds = []
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5: # class_id cx cy w h
                    c, x, y, w, h = map(float, parts)
                    preds.append({'class': int(c), 'x': x, 'y': y, 'w': w, 'h': h, 'score': 1.0}) # Assume score 1.0 if not provided
                elif len(parts) == 6: # class_id cx cy w h score
                    c, x, y, w, h, score = map(float, parts)
                    preds.append({'class': int(c), 'x': x, 'y': y, 'w': w, 'h': h, 'score': score})
                else:
                    logging.warning(f"Malformed YOLO prediction line in {path}: '{line.strip()}'. Skipping.")
        logging.info(f"Loaded {len(preds)} YOLO predictions from {path}.")
        return preds
    except Exception as e:
        logging.error(f"Error loading YOLO predictions from {path}: {e}")
        raise

def iou_yolo_dict(boxA: dict, boxB: dict):
    """
    Compute Intersection-over-Union (IoU) for two bounding boxes given as dictionaries.
    Boxes are expected in YOLO format: {'x': cx, 'y': cy, 'w': w, 'h': h}.
    Args:
        boxA (dict): Bounding box A.
        boxB (dict): Bounding box B.
    Returns:
        float: IoU value.
    """
    # Convert YOLO center format to corners (x1, y1, x2, y2)
    def to_corners(box_dict):
        cx, cy, w, h = box_dict['x'], box_dict['y'], box_dict['w'], box_dict['h']
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    a1, a2, a3, a4 = to_corners(boxA)
    b1, b2, b3, b4 = to_corners(boxB)

    # Determine the coordinates of the intersection rectangle
    xi1, yi1 = max(a1, b1), max(a2, b2)
    xi2, yi2 = min(a3, b3), min(a4, b4)

    # Compute the area of intersection rectangle
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    # Compute the area of both boxes
    areaA = (a3 - a1) * (a4 - a2)
    areaB = (b3 - b1) * (b4 - b2)

    # Compute the union area
    union_area = areaA + areaB - inter_area

    # Handle case where union_area is zero to avoid division by zero
    return inter_area / union_area if union_area > 0 else 0.0

def nms(boxes: list, scores: list, iou_thresh: float = 0.5):
    """
    Performs Non-Maximum Suppression (NMS) on a list of bounding boxes.
    Boxes are expected as dictionaries in YOLO format: {'x': cx, 'y': cy, 'w': w, 'h': h}.
    Args:
        boxes (list): List of bounding box dictionaries.
        scores (list): List of confidence scores corresponding to the boxes.
        iou_thresh (float): IoU threshold for suppression.
    Returns:
        list: Indices of the boxes to keep after NMS.
    """
    if not boxes:
        return []

    # Convert boxes to a format compatible with numpy operations if needed, or iterate
    # For simplicity, assuming boxes are already dicts, we'll work with indices
    
    # Sort by scores in descending order
    idxs = np.argsort(scores)[::-1]
    
    keep = []
    while len(idxs) > 0:
        i = idxs[0] # Index of the box with the highest score
        keep.append(i)
        
        # Indices to remove in the current iteration
        remove_indices = [0] # Always remove the current highest scoring box
        
        # Compare the current box with the rest
        for j in range(1, len(idxs)):
            current_idx = idxs[j]
            # Use the iou_yolo_dict function defined above
            if iou_yolo_dict(boxes[i], boxes[current_idx]) > iou_thresh:
                remove_indices.append(j)
        
        # Remove the suppressed boxes from the list of indices to consider
        idxs = np.delete(idxs, remove_indices)
        
    return keep

def check_relation(graph: nx.Graph, predictions: list):
    """
    Check relational graph constraints against YOLO predictions.
    Args:
        graph (networkx.Graph): The relational graph where nodes correspond to object indices
                                and edges have 'relation' attributes.
        predictions (list): List of YOLO prediction dictionaries, where index corresponds to graph node.
                            Example: [{'class': 0, 'x': 0.5, 'y': 0.5, 'w': 0.1, 'h': 0.1, 'score': 0.9}, ...]
    Returns:
        tuple: (list of violations, dict of relation statistics)
    """
    violations = []
    stats = defaultdict(int)

    # Ensure predictions list is long enough for graph nodes
    if not predictions:
        logging.warning("No predictions provided for relation checking.")
        return [], {}

    for u, v, d in graph.edges(data=True):
        rel = d.get('relation')
        if rel is None:
            logging.warning(f"Edge ({u}, {v}) has no 'relation' attribute. Skipping.")
            continue

        # Ensure u and v are valid indices in predictions
        if u >= len(predictions) or v >= len(predictions):
            logging.warning(f"Graph node index out of bounds for predictions: ({u}, {v}). Skipping edge.")
            continue
        
        # Get the bounding box dictionaries for the current nodes
        box_u = predictions[u]
        box_v = predictions[v]

        stats[rel] += 1 # Count occurrences of each relation type

        if rel == 'left-of':
            # Check if object u's center x is indeed to the left of object v's center x
            if box_u['x'] >= box_v['x']: # Violation if u is not strictly left of v
                violations.append((u, v, 'left-of'))
        elif rel == 'above':
            # Check if object u's center y is indeed above object v's center y (smaller y is higher)
            if box_u['y'] >= box_v['y']: # Violation if u is not strictly above v
                violations.append((u, v, 'above'))
        elif rel == 'touches':
            # Simple proximity check using IoU. If IoU is very low, they don't "touch".
            # A low IoU threshold (e.g., < 0.05) indicates they are not touching significantly.
            if iou_yolo_dict(box_u, box_v) < 0.05: # Use the correct iou function
                violations.append((u, v, 'touches'))
        elif rel == 'contains':
            # Check if box u fully contains box v
            ux1, uy1, ux2, uy2 = iou_yolo_dict_to_corners(box_u)
            vx1, vy1, vx2, vy2 = iou_yolo_dict_to_corners(box_v)

            # A contains B if B's corners are all within A's corners
            if not (ux1 <= vx1 and uy1 <= vy1 and ux2 >= vx2 and uy2 >= vy2):
                violations.append((u, v, 'contains'))
        else:
            logging.warning(f"Unknown relation type '{rel}' for edge ({u}, {v}). Skipping check.")

    return violations, dict(stats)

def iou_yolo_dict_to_corners(box_dict):
    """Helper to convert YOLO dict to corners [x1, y1, x2, y2]"""
    cx, cy, w, h = box_dict['x'], box_dict['y'], box_dict['w'], box_dict['h']
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return x1, y1, x2, y2


def label_quality(predictions: list, confidences: list = None, iou_thresh: float = 0.5):
    """
    Runs NMS (Non-Maximum Suppression) and potentially other checks to assess label quality.
    Args:
        predictions (list): List of YOLO prediction dictionaries.
        confidences (list, optional): List of confidence scores for predictions. If None, assumes all 1.0.
        iou_thresh (float): IoU threshold for NMS.
    Returns:
        tuple: (list of filtered predictions, list of flagged outlier predictions)
    """
    if not predictions:
        return [], []

    if confidences is None:
        # If confidences are not provided, try to extract from prediction dicts or default to 1.0
        scores = [p.get('score', 1.0) for p in predictions]
    else:
        scores = confidences
    
    if len(predictions) != len(scores):
        logging.error("Mismatch between number of predictions and scores. Cannot perform NMS.")
        return predictions, [] # Return original if mismatch

    # Perform NMS to get indices of boxes to keep
    keep_idxs = nms(predictions, scores, iou_thresh)
    
    filtered_preds = [predictions[i] for i in keep_idxs]
    
    outliers = []
    # Identify outliers (suppressed boxes)
    for i, p in enumerate(predictions):
        if i not in keep_idxs:
            outliers.append(p)
            
    logging.info(f"Label quality check: {len(filtered_preds)} detections kept, {len(outliers)} outliers (suppressed) found.")
    return filtered_preds, outliers


def log_graph_metadata(image_id: str, predictions: list, violations: list, stats: dict, out_path: str):
    """
    Log metadata related to graph fusion analysis for a single image.
    Args:
        image_id (str): Unique identifier for the image.
        predictions (list): List of YOLO prediction dictionaries after quality checks.
        violations (list): List of detected relational violations.
        stats (dict): Statistics about relation types.
        out_path (str): Path to the JSONL file where metadata will be appended.
    """
    avg_area = float(np.mean([p['w']*p['h'] for p in predictions]) if predictions else 0.0)

    meta = {
        'id': image_id,
        'obj_count': len(predictions),
        'rel_violations_count': len(violations),
        'rel_stats': stats,
        'avg_bbox_area_filtered': avg_area, # Renamed for clarity
        'violations_details': violations # Detailed list of violations
    }
    
    try:
        with open(out_path, 'a') as f:
            f.write(json.dumps(meta) + '\n')
        logging.info(f"Graph metadata logged to {out_path} for image ID: {image_id}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to log graph metadata to file {out_path}: {e}")

    # MLflow logging
    if mlflow:
        try:
            # Log as a dictionary artifact.
            artifact_name = f"graph_metadata_{image_id}.json"
            mlflow.log_dict(meta, artifact_file=artifact_name)
            logging.info(f"Graph metadata logged to MLflow as artifact: {artifact_name}")
        except Exception as e:
            logging.error(f"[ERROR] MLflow graph metadata logging failed: {e}")

if __name__ == '__main__':
    # Configure basic logging for CLI usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    import argparse
    parser = argparse.ArgumentParser(description="Fuse Graphs with YOLO Predictions Utility")
    parser.add_argument('--graph_path', required=True, help='Path to relational graph JSON file')
    parser.add_argument('--yolo_pred_path', required=True, help='Path to YOLO prediction TXT file')
    parser.add_argument('--output_log_path', default='fuse_graph_metadata_log.jsonl', help='Output path for metadata log')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IoU threshold for NMS and relation checks')
    args = parser.parse_args()

    try:
        graph = load_graph(args.graph_path)
        preds = load_yolo(args.yolo_pred_path)

        # Extract confidences from predictions (assuming load_yolo returns dicts with 'score')
        confs = [p.get('score', 1.0) for p in preds]

        # 1. Label Quality Check (NMS)
        filtered_preds, outliers = label_quality(preds, confidences=confs, iou_thresh=args.iou_thresh)
        logging.info(f"Original predictions: {len(preds)}, Filtered predictions (after NMS): {len(filtered_preds)}")
        logging.info(f"Outlier boxes (suppressed by NMS): {len(outliers)}")

        # 2. Check Relations against the graph
        # Note: The graph nodes must correspond to indices in 'filtered_preds' if you want to check
        # relations on the NMS-filtered set. This is a common point of confusion.
        # For this example, we assume graph nodes refer to the original indices,
        # or that the graph itself is built on the filtered set.
        # If graph nodes are 0-indexed and correspond to the filtered_preds, then it works.
        # If graph nodes are arbitrary IDs, you'd need a mapping.
        # For a simple example, let's assume the graph refers to indices of 'filtered_preds'.
        # If your graph is based on original indices, you might need to adjust 'filtered_preds'
        # to include dummy entries for removed boxes, or re-index the graph.
        
        # For demo, let's assume graph nodes are 0-indexed and refer to the filtered_preds.
        # This means the graph should ideally be built *after* initial filtering or NMS.
        # If the graph is from GT, then 'preds' should be compared to GT.
        # This part of the logic is highly dependent on how your graph is constructed.
        # For simplicity, we'll use filtered_preds directly.
        violations, stats = check_relation(graph, filtered_preds)
        
        logging.info(f"Relational violations found: {len(violations)}")
        logging.info(f"Relation stats: {stats}")

        # 3. Log Fusion Metadata
        image_id = os.path.splitext(os.path.basename(args.yolo_pred_path))[0]
        log_graph_metadata(image_id, filtered_preds, violations, stats, args.output_log_path)
        
        print("\n--- Summary ---")
        print(f"Image ID: {image_id}")
        print(f"Original Detections: {len(preds)}")
        print(f"Detections after NMS: {len(filtered_preds)}")
        print(f"NMS Outliers: {len(outliers)}")
        print(f"Relational Violations: {violations}")
        print(f"Relation Statistics: {stats}")
        print(f"Metadata logged to: {args.output_log_path}")

    except FileNotFoundError as e:
        logging.critical(f"Required file not found: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
