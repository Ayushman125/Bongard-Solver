import json
import os
import logging
import numpy as np
from collections import Counter, defaultdict

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metadata(image_id: str, labels: list, image_width: int = None, image_height: int = None) -> dict:
    """
    Computes various metadata for a single image and its labels.
    Args:
        image_id (str): Unique identifier for the image.
        labels (list): List of YOLO format labels for the image:
                       [[class_id, cx, cy, w, h], ...].
        image_width (int, optional): Width of the image. Required for absolute bbox calculations.
        image_height (int, optional): Height of the image. Required for absolute bbox calculations.
    Returns:
        dict: A dictionary containing computed metadata.
    """
    metadata = {
        "image_id": image_id,
        "num_objects": len(labels),
        "class_distribution": {},
        "bbox_areas_normalized": [],
        "bbox_aspect_ratios": [],
        "avg_bbox_area_normalized": 0.0,
        "avg_bbox_aspect_ratio": 0.0,
        "image_width": image_width,
        "image_height": image_height,
        "avg_bbox_area_abs": 0.0, # Absolute pixel area
        "object_density": 0.0 # Total normalized bbox area / image area
    }

    if not labels:
        logging.debug(f"No labels found for image_id: {image_id}. Returning basic metadata.")
        return metadata

    class_ids = [int(l[0]) for l in labels]
    class_counts = Counter(class_ids)
    
    total_area_normalized = 0.0
    bbox_areas_abs = []

    for label in labels:
        try:
            # YOLO format: class_id, cx, cy, w, h (all normalized [0,1])
            _, cx, cy, w, h = map(float, label)
            
            # Normalized area
            normalized_area = w * h
            metadata["bbox_areas_normalized"].append(normalized_area)
            total_area_normalized += normalized_area

            # Aspect ratio (width / height)
            if h > 0:
                aspect_ratio = w / h
                metadata["bbox_aspect_ratios"].append(aspect_ratio)
            else:
                metadata["bbox_aspect_ratios"].append(0.0) # Handle zero height

            # Absolute area (if image dimensions are available)
            if image_width is not None and image_height is not None:
                abs_w = w * image_width
                abs_h = h * image_height
                bbox_areas_abs.append(abs_w * abs_h)

        except ValueError as ve:
            logging.error(f"Invalid label format for {image_id}: {label}. Error: {ve}")
        except Exception as e:
            logging.error(f"Error processing label for {image_id}: {label}. Error: {e}")
            
    # Compute averages
    if metadata["bbox_areas_normalized"]:
        metadata["avg_bbox_area_normalized"] = float(np.mean(metadata["bbox_areas_normalized"]))
    if metadata["bbox_aspect_ratios"]:
        metadata["avg_bbox_aspect_ratio"] = float(np.mean(metadata["bbox_aspect_ratios"]))
    if bbox_areas_abs:
        metadata["avg_bbox_area_abs"] = float(np.mean(bbox_areas_abs))
    
    # Object density
    if image_width is not None and image_height is not None and image_width * image_height > 0:
        metadata["object_density"] = total_area_normalized / (1.0) # Total normalized area is already relative to image size
        # If you want pixel density: sum(bbox_areas_abs) / (image_width * image_height)
    
    # Convert class counts to percentages
    total_objects = len(labels)
    metadata["class_distribution"] = {
        str(cls_id): float(count / total_objects) for cls_id, count in class_counts.items()
    }
    
    logging.debug(f"Computed metadata for {image_id}: {metadata}")
    return metadata

def log_dataset_metadata(all_metadata: list, output_path: str, mlflow_enabled: bool = False):
    """
    Logs a list of metadata dictionaries to a JSONL file and optionally to MLflow.
    Args:
        all_metadata (list): A list of metadata dictionaries, one per image.
        output_path (str): The base path for the output JSONL file.
                           The file will be named 'output_path.jsonl'.
        mlflow_enabled (bool): If True, also logs metadata as an MLflow artifact.
    """
    jsonl_output_path = f"{output_path}.jsonl"
    
    try:
        with open(jsonl_output_path, 'w') as f:
            for meta in all_metadata:
                f.write(json.dumps(meta) + '\n')
        logging.info(f"Dataset metadata logged to JSONL file: {jsonl_output_path}")
    except Exception as e:
        logging.error(f"Error writing dataset metadata to {jsonl_output_path}: {e}")

    if mlflow_enabled:
        try:
            import mlflow
            # Log the JSONL file as an artifact
            mlflow.log_artifact(jsonl_output_path, "dataset_metadata")
            logging.info(f"Dataset metadata logged to MLflow as artifact: {jsonl_output_path}")

            # Optionally, log summary statistics as MLflow params
            if all_metadata:
                total_images = len(all_metadata)
                total_objects = sum(m.get('num_objects', 0) for m in all_metadata)
                avg_objects_per_image = total_objects / total_images if total_images > 0 else 0

                all_normalized_areas = [area for m in all_metadata for area in m.get('bbox_areas_normalized', [])]
                avg_normalized_bbox_area_overall = float(np.mean(all_normalized_areas)) if all_normalized_areas else 0.0

                mlflow.log_param("total_images", total_images)
                mlflow.log_param("total_objects", total_objects)
                mlflow.log_param("avg_objects_per_image", f"{avg_objects_per_image:.2f}")
                mlflow.log_param("avg_normalized_bbox_area_overall", f"{avg_normalized_bbox_area_overall:.4f}")
                
                # Aggregate class distribution across the whole dataset
                overall_class_counts = Counter()
                for meta in all_metadata:
                    for cls_id_str, percentage in meta.get('class_distribution', {}).items():
                        # Convert percentage back to count for aggregation, then re-normalize
                        # This is an approximation if original counts aren't available
                        overall_class_counts[int(cls_id_str)] += round(percentage * meta.get('num_objects', 0))
                
                total_overall_objects = sum(overall_class_counts.values())
                overall_class_distribution_percent = {
                    str(cls_id): float(count / total_overall_objects) for cls_id, count in overall_class_counts.items()
                } if total_overall_objects > 0 else {}

                mlflow.log_dict(overall_class_distribution_percent, "overall_class_distribution.json")
                logging.info("Summary metadata logged to MLflow parameters.")

        except ImportError:
            logging.warning("MLflow not installed. Skipping MLflow logging for dataset metadata.")
        except Exception as e:
            logging.error(f"Error logging dataset metadata to MLflow: {e}")

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example 1: Image with objects
    labels1 = [
        [0, 0.5, 0.5, 0.2, 0.3], # class 0, center x,y, width, height
        [1, 0.1, 0.1, 0.05, 0.05], # class 1
        [0, 0.8, 0.2, 0.1, 0.15]  # class 0
    ]
    meta1 = compute_metadata("img_A", labels1, image_width=1000, image_height=800)
    print(f"Metadata for img_A:\n{json.dumps(meta1, indent=2)}\n")

    # Example 2: Image with no objects
    labels2 = []
    meta2 = compute_metadata("img_B", labels2, image_width=640, image_height=480)
    print(f"Metadata for img_B:\n{json.dumps(meta2, indent=2)}\n")

    # Example 3: Image with malformed label
    labels3 = [
        [0, 0.5, 0.5, 0.2, 0.3],
        [1, 0.1, 0.1, 0.05], # Malformed (missing one value)
        [0, 0.8, 0.2, 0.1, 0.15]
    ]
    meta3 = compute_metadata("img_C", labels3, image_width=720, image_height=540)
    print(f"Metadata for img_C:\n{json.dumps(meta3, indent=2)}\n")

    all_dataset_metadata = [meta1, meta2, meta3]
    output_base_path = "test_dataset_metadata"
    
    # Clean up previous test file if exists
    if os.path.exists(f"{output_base_path}.jsonl"):
        os.remove(f"{output_base_path}.jsonl")

    # Test logging to JSONL
    log_dataset_metadata(all_dataset_metadata, output_base_path, mlflow_enabled=False)
    print(f"\nCheck '{output_base_path}.jsonl' for logged metadata.")

    # To test MLflow logging, you would need to have MLflow installed and a tracking server running.
    # For demonstration, we'll just show the call.
    # try:
    #     import mlflow
    #     mlflow.set_tracking_uri("file:///tmp/mlruns") # Example local tracking URI
    #     mlflow.set_experiment("metadata_logging_test")
    #     with mlflow.start_run(run_name="test_metadata_run"):
    #         log_dataset_metadata(all_dataset_metadata, output_base_path, mlflow_enabled=True)
    #     print("\nMLflow logging test completed. Check your MLflow UI.")
    # except ImportError:
    #     print("\nMLflow not installed, skipping MLflow logging test.")
    # except Exception as e:
    #     print(f"\nError during MLflow test: {e}")

    # Clean up created test file
    if os.path.exists(f"{output_base_path}.jsonl"):
        os.remove(f"{output_base_path}.jsonl")
