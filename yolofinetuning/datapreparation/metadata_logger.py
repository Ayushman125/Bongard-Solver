import json
import numpy as np
import logging # Added for logging errors

try:
    import mlflow
except ImportError:
    mlflow = None
    logging.warning("MLflow not installed. MLflow metadata logging will be skipped.")

def compute_metadata(image_id: str, labels: list):
    """
    Computes various metadata for a single image and its labels.
    Args:
        image_id (str): Unique identifier for the image.
        labels (list): List of labels for the image. Each label is expected to be
                       in YOLO format: [class_id, cx, cy, w, h].
                       Example: [[0, 0.5, 0.5, 0.1, 0.1], [1, 0.2, 0.2, 0.05, 0.05]]
    Returns:
        dict: A dictionary containing computed metadata.
    """
    areas = []
    class_counts = {}
    for label in labels:
        if len(label) == 5: # Ensure it's a valid YOLO label
            class_id, _, _, w, h = label
            areas.append(w * h)
            class_counts[int(class_id)] = class_counts.get(int(class_id), 0) + 1
        else:
            logging.warning(f"Malformed label entry for image_id {image_id}: {label}. Skipping.")

    avg_bbox_area = float(np.mean(areas)) if areas else 0.0
    min_bbox_area = float(np.min(areas)) if areas else 0.0
    max_bbox_area = float(np.max(areas)) if areas else 0.0

    return {
        'id': image_id,
        'obj_count': len(labels),
        'avg_bbox_area': avg_bbox_area,
        'min_bbox_area': min_bbox_area,
        'max_bbox_area': max_bbox_area,
        'class_distribution': class_counts # Added class distribution
    }


def log_dataset_metadata(meta: dict, path: str = 'dataset_metadata_log.jsonl'):
    """
    Logs computed metadata for a dataset or image to a JSONL file.
    Args:
        meta (dict): Dictionary containing metadata for a dataset or image.
        path (str): Path to the JSONL file where metadata will be appended.
    """
    try:
        with open(path, 'a') as f:
            f.write(json.dumps(meta) + "\n")
        logging.info(f"Metadata logged to {path} for image/dataset ID: {meta.get('id', 'N/A')}")
    except Exception as e:
        logging.error(f"[ERROR] Failed to log metadata to file {path}: {e}")

    # MLflow logging
    if mlflow:
        try:
            # Log as a dictionary artifact. If 'id' is present, use it for artifact naming.
            artifact_name = f"metadata_{meta.get('id', 'overall')}.json"
            mlflow.log_dict(meta, artifact_file=artifact_name)
            logging.info(f"Metadata logged to MLflow as artifact: {artifact_name}")
        except Exception as e:
            logging.error(f"[ERROR] MLflow metadata logging failed: {e}")
