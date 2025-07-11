# Folder: bongard_solver/
# File: pipeline_workers.py
import os
import logging
import random
import numpy as np
import cv2  # For image loading and processing
from PIL import Image  # For image handling
from typing import Dict, Any, List, Tuple
import threading
from queue import Queue, Empty # Import Queue and Empty
import time # Import time for sleep

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
            try:
                from ultralytics import YOLO as RealYOLO
                _yolo_model = RealYOLO(yolo_config['model_name'])
                logger.info(f"Worker: Loaded YOLO model: {yolo_config['model_name']}")
            except ImportError:
                logger.warning("Worker: ultralytics (YOLO) library not available. YOLO part of detection will be dummy.")
                HAS_YOLO = False # Update global HAS_YOLO
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
                    HAS_SAM = False # Update global HAS_SAM
            else:
                logger.warning(f"Worker: SAM checkpoint not found at {sam_config['sam_checkpoint_path']}. SAM will be disabled for this worker.")
                HAS_SAM = False # Update global HAS_SAM
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
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # Convert to RGB
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
                'mask': masks[i].tolist()  # Convert mask to list for serialization
            })
        
        # Dummy relations based on object count
        if len(bboxes) >= 2:
            # Example: Add a dummy 'left_of' relation between first two objects
            inferred_scene_graph['relations'].append({
                'subject_id': 0,
                'object_id': 1,
                'type': 'left_of',
                'score': 0.95  # Dummy score
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
        'masks': [m.tolist() for m in masks],  # Ensure masks are serializable
        'inferred_scene_graph': inferred_scene_graph
    }

def start_prefetch_worker(dali_pipeline: Any, cfg: Dict[str, Any]) -> Queue:
    """
    Starts a DALI prefetch worker in a separate thread.
    This worker continuously runs the DALI pipeline and puts outputs into a queue.
    It includes enhanced error handling for queue underflow/overflow.

    Args:
        dali_pipeline: An initialized DALI pipeline object.
        cfg (Dict[str, Any]): The configuration dictionary, specifically for 'dali' settings.

    Returns:
        Queue: A queue from which processed batches can be retrieved.
    """
    queue_size = cfg['dali'].get('queue_size', 3)
    put_timeout = cfg['dali'].get('put_timeout', 1.0) # Timeout for putting data into queue
    worker_sleep = cfg['dali'].get('worker_sleep', 0.01) # Sleep interval for worker loop

    q = Queue(maxsize=queue_size)
    
    def worker():
        logger.info("DALI prefetch worker started.")
        while True:
            try:
                # Attempt to run the DALI pipeline
                data = dali_pipeline.run()
                try:
                    # Attempt to put data into the queue with a timeout
                    q.put(data, timeout=put_timeout)
                except Exception as e: # Catch any exception during put (e.g., Full, Timeout)
                    logging.warning(f"DALI queue overflow (put timeout: {put_timeout}s). Error: {e}")
            except Empty: # This is for dali_pipeline.run() returning Empty, indicating underflow
                logging.warning("DALI pipeline underflow (no data available from pipeline).")
            except Exception as e:
                logging.error(f"DALI worker error: {e}")
                # In a real scenario, you might want to handle specific DALI errors
                # or signal the main thread to stop. For now, break the loop.
                break
            time.sleep(worker_sleep) # Sleep to prevent busy-waiting
        logger.info("DALI prefetch worker stopped.")

    # Start the worker thread as a daemon so it exits when the main program exits
    threading.Thread(target=worker, daemon=True).start()
    logger.info(f"DALI prefetch worker thread launched with queue size {queue_size}.")
    return q

