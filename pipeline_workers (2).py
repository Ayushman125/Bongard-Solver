# Folder: bongard_solver/
# File: pipeline_workers.py

import os
import logging
import random
import numpy as np
import cv2 # For image loading and processing
from PIL import Image # For image handling
from typing import Dict, Any, List, Tuple

# Import from config
from config import CONFIG, DEVICE

# Import SAM utilities (assuming they handle YOLO and SAM model loading)
# Note: YOLO and SAM models are large. For process-based parallelism,
# it's often best to initialize them *within* the worker function
# to ensure each process has its own isolated instance.
# sam_utils should have a function like `load_yolo_and_sam_models_for_worker`
# that can be called repeatedly or has internal caching.
# For this example, we'll assume `detect_and_segment_image` handles model loading
# or that models are passed (though passing large models between processes is inefficient).
# A better approach is to load them inside the worker.
from sam_utils import load_yolo_and_sam_models, detect_and_segment_image, HAS_YOLO, HAS_SAM

# Import utilities (e.g., _calculate_iou if needed here)
from utils import _calculate_iou

logger = logging.getLogger(__name__)

# Global variables for models (initialized once per worker process)
_yolo_model = None
_sam_predictor = None

def _initialize_models_for_worker(config: Dict[str, Any]):
    """
    Initializes YOLO and SAM models for a single worker process.
    This function should be called once per worker.
    """
    global _yolo_model, _sam_predictor

    if _yolo_model is None:
        yolo_config = config['model']['object_detector_config']
        if HAS_YOLO:
            from ultralytics import YOLO as RealYOLO
            _yolo_model = RealYOLO(yolo_config['model_name'])
            logger.info(f"Worker: Loaded YOLO model: {yolo_config['model_name']}")
        else:
            logger.warning("Worker: YOLO library not available. YOLO part of detection will be dummy.")

    if _sam_predictor is None:
        sam_config = config['model']['object_detector_config']
        if HAS_SAM and sam_config['sam_model_type'] and sam_config['sam_checkpoint_path']:
            if os.path.exists(sam_config['sam_checkpoint_path']):
                try:
                    from segment_anything import SamPredictor, sam_model_registry
                    sam_model = sam_model_registry[sam_config['sam_model_type']](checkpoint=sam_config['sam_checkpoint_path'])
                    # SAM model should be moved to CPU if workers are CPU-bound, or to specific GPU if worker is GPU-bound
                    # For general multiprocessing.Pool, assume CPU.
                    _sam_predictor = SamPredictor(sam_model.to('cpu')) 
                    logger.info(f"Worker: Loaded SAM model: {sam_config['sam_model_type']}")
                except Exception as e:
                    logger.error(f"Worker: Failed to load SAM model: {e}. SAM will be disabled for this worker.")
            else:
                logger.warning(f"Worker: SAM checkpoint not found at {sam_config['sam_checkpoint_path']}. SAM will be disabled for this worker.")
        else:
            logger.warning("Worker: SAM libraries or config not fully available. SAM will be dummy.")

def process_image_worker(image_path: str, config: Dict[str, Any], worker_id: int, total_workers: int) -> Dict[str, Any]:
    """
    Worker function to process a single image, perform object detection,
    and extract symbolic scene graph information.

    This function is designed to be run in a multiprocessing Pool.
    It initializes models locally to ensure isolated state.

    Args:
        image_path (str): Path to the input image file.
        config (Dict[str, Any]): The global configuration dictionary.
        worker_id (int): The ID of the current worker process.
        total_workers (int): Total number of workers in the pool.

    Returns:
        Dict[str, Any]: A dictionary containing the processed image's
                        metadata, bounding boxes, masks, and inferred scene graph.
    """
    # Set random seed for reproducibility within this worker
    random.seed(config['training']['seed'] + worker_id)
    np.random.seed(config['training']['seed'] + worker_id)

    logger.info(f"Worker {worker_id}/{total_workers}: Processing image: {image_path}")

    # Initialize models for this worker if not already initialized
    _initialize_models_for_worker(config)

    # Load image using OpenCV (NumPy array format)
    try:
        img_np = cv2.imread(image_path)
        if img_np is None:
            raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # Convert to RGB
    except Exception as e:
        logger.error(f"Worker {worker_id}: Error loading image {image_path}: {e}")
        return {
            'image_path': image_path,
            'status': 'failed_loading',
            'error': str(e),
            'bboxes': [],
            'masks': [],
            'inferred_scene_graph': {'objects': [], 'relations': []}
        }

    # Perform object detection and segmentation using YOLO and SAM
    # This function is assumed to be from sam_utils and uses the global _yolo_model and _sam_predictor
    # or takes them as arguments. For multiprocessing, passing them is better.
    # Let's assume detect_and_segment_image can take optional model instances.
    try:
        bboxes, masks, sg_objects_list = detect_and_segment_image(
            image_np=img_np,
            yolo_model=_yolo_model,
            sam_predictor=_sam_predictor,
            yolo_conf_threshold=config['model']['object_detector_config']['yolo_conf_threshold'],
            yolo_iou_threshold=config['model']['object_detector_config']['yolo_iou_threshold'],
            max_objects=config['model']['object_detector_config']['num_objects']
        )
        logger.info(f"Worker {worker_id}: Detected {len(bboxes)} objects in {image_path}.")

        # Further symbolic fact extraction (attributes, relations)
        # This is where more complex, CPU-bound symbolic reasoning would occur.
        # For demonstration, we'll just populate a basic scene graph.
        inferred_scene_graph = {'objects': [], 'relations': []}
        for i, bbox in enumerate(bboxes):
            obj_id = i
            # Dummy attributes for illustration
            dummy_attributes = {
                'shape': random.choice(['circle', 'square', 'triangle']),
                'color': random.choice(['red', 'blue', 'green']),
                'size': random.choice(['small', 'medium', 'large'])
            }
            inferred_scene_graph['objects'].append({
                'id': obj_id,
                'bbox': bbox,
                'attributes': dummy_attributes,
                'mask': masks[i].tolist() # Convert mask to list for serialization
            })
        
        # Dummy relations based on object count
        if len(bboxes) >= 2:
            # Example: Add a dummy 'left_of' relation between first two objects
            inferred_scene_graph['relations'].append({
                'subject_id': 0,
                'object_id': 1,
                'type': 'left_of',
                'score': 0.95 # Dummy score
            })

    except Exception as e:
        logger.error(f"Worker {worker_id}: Error during detection/segmentation for {image_path}: {e}")
        return {
            'image_path': image_path,
            'status': 'failed_processing',
            'error': str(e),
            'bboxes': [],
            'masks': [],
            'inferred_scene_graph': {'objects': [], 'relations': []}
        }

    return {
        'image_path': image_path,
        'status': 'success',
        'bboxes': bboxes,
        'masks': [m.tolist() for m in masks], # Ensure masks are serializable
        'inferred_scene_graph': inferred_scene_graph
    }

