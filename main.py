# Folder: bongard_solver/
# File: main.py
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import logging
import random
import numpy as np
import yaml # For saving config
import json # For handling scene graph bytes (if needed for dummy data)
import cv2 # For creating dummy images in scene graph demo
from PIL import Image # For saving dummy images
from typing import List, Dict, Any, Tuple, Optional

# --- Hydra Imports ---
import hydra
from omegaconf import DictConfig, OmegaConf

# Import configuration (Note: with Hydra, CONFIG will largely be replaced by cfg)
from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD, \
                   HAS_WANDB, HAS_TORCH_QUANTIZATION, HAS_ULTRALYTICS, HAS_SAM, HAS_TOPOLOGY_LIBS

# Import data loading components (for DALI processor if needed in main)
from data import build_dali_image_processor, get_dataloader # Used for pruning/quantization validation and main training
# Import model and training components
from training import run_training_pipeline, run_training_once # Main training orchestrator and HPO helper
from yolo_trainer import fine_tune_yolo # For YOLO fine-tuning
from prune_quantize import run_pruning_and_quantization # For pruning and quantization
from hpo import run_hpo # For Hyperparameter Optimization

# Import scene_graph_builder for demonstration
try:
    from scene_graph_builder import SceneGraphBuilder # Import the class
    HAS_SCENE_GRAPH_BUILDER = True
    logger.info("scene_graph_builder.py found.")
except ImportError:
    HAS_SCENE_GRAPH_BUILDER = False
    logger.warning("scene_graph_builder.py not found. Scene graph building demonstration will be skipped.")
    # Dummy class for demonstration if not found
    class SceneGraphBuilder:
        def __init__(self, config): pass
        def build_scene_graph(self, image_np, detected_bboxes, detected_masks, attribute_logits, relation_logits):
            logger.warning("Dummy SceneGraphBuilder: Returning empty scene graph.")
            return {'objects': [], 'relations': []}

# Import sam_utils for object detection and segmentation demo
try:
    from sam_utils import load_yolo_and_sam_models, detect_and_segment_image, get_masked_crop, HAS_YOLO, HAS_SAM
    logger.info("sam_utils.py found.")
except ImportError:
    logger.warning("sam_utils.py not found. Object detection and segmentation demonstration will be skipped.")
    # Set HAS_YOLO and HAS_SAM to False if sam_utils is not found
    HAS_YOLO = False
    HAS_SAM = False
    # Dummy functions to prevent errors
    def load_yolo_and_sam_models(cfg): return None, None
    def detect_and_segment_image(image_np, yolo_model, sam_predictor, cfg): return [], [], []
    def get_masked_crop(image_np, mask, bbox): return np.zeros((1,1,3))

# Import utils for set_seed
try:
    from utils import set_seed
except ImportError:
    logger.warning("Could not import set_seed from utils.py. Using local dummy function.")
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main function definition with Hydra
@hydra.main(config_path="conf", config_name="config", version_base=None) # Added version_base
def main(cfg: DictConfig):
    """
    Main entry point for the Bongard Solver project.
    Orchestrates the entire pipeline: YOLO fine-tuning, HPO, main training,
    pruning, and quantization.
    """
    logger.info("--- Starting Bongard Solver Main Pipeline ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}") # Use OmegaConf.to_yaml for pretty printing

    # Set logging level from config
    logger.setLevel(cfg.debug.log_level)

    # Create checkpoints directory if it doesn't exist
    os.makedirs(cfg.debug.save_model_checkpoints, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {cfg.debug.save_model_checkpoints}")

    # Initialize Weights & Biases if enabled
    # Note: HAS_WANDB is a global flag from config.py, which might be less ideal with Hydra.
    # For full Hydra integration, you'd typically manage this directly from cfg.
    if HAS_WANDB and cfg.training.get('use_wandb', False):
        import wandb # Import here to avoid issues if not installed
        wandb.init(project="bongard_solver", config=OmegaConf.to_container(cfg, resolve=True))
        logger.info("Weights & Biases initialized.")
    else:
        logger.info("Weights & Biases is disabled or not found.")

    # Set random seed
    set_seed(cfg.training.seed)

    # 2. YOLO Object Detector Fine-tuning (if enabled)
    if cfg.object_detector.fine_tune_yolo:
        logger.info("Initiating YOLO fine-tuning.")
        # Pass the Hydra config object directly
        best_yolo_weights_path = fine_tune_yolo(cfg)
        if best_yolo_weights_path:
            cfg.object_detector.yolo_pretrained = best_yolo_weights_path
            logger.info(f"YOLO fine-tuning completed. Using best weights: {best_yolo_weights_path}")
        else:
            logger.error("YOLO fine-tuning failed. Proceeding with original YOLO weights or skipping YOLO.")
            # Optionally, disable YOLO if fine-tuning is critical and failed
            # cfg.object_detector.use_yolo = False

    # 3. Hyperparameter Optimization (HPO) (if enabled)
    if cfg.hpo.enabled:
        logger.info("Initiating Hyperparameter Optimization with Optuna.")
        # Optuna runs its own trials, each calling `run_training_once`
        run_hpo(
            cfg=cfg, # Pass the entire config
            n_trials=cfg.hpo.trials,
            timeout=cfg.hpo.timeout,
            n_jobs=cfg.hpo.n_jobs,
            study_path=cfg.hpo.study_path
        )
        logger.info("Hyperparameter Optimization completed.")
        # After HPO, you might want to load the best HPs into CONFIG
        # For simplicity, this example assumes you'd manually update config.yaml
        # with the best HPs found by HPO.

    # 4. Main Bongard Solver Training
    logger.info("Initiating main Bongard Solver training pipeline.")
    run_training_pipeline(cfg) # Pass the Hydra config object
    logger.info("Main Bongard Solver training completed.")

    # 5. Pruning and Quantization (if enabled)
    # These can be run after the main training to optimize the trained model.
    if cfg.pruning.enabled or cfg.quantization.qat or cfg.quantization.ptq:
        logger.info("Initiating Pruning and Quantization pipeline.")
        run_pruning_and_quantization(cfg) # Pass the Hydra config object
        logger.info("Pruning and Quantization pipeline completed.")
    else:
        logger.info("Pruning and Quantization are disabled in config.")

    logger.info("--- Bongard Solver Main Pipeline finished. ---")

    # --- Demonstrating Scene Graph Building with new components ---
    # This section requires YOLO, SAM, and SceneGraphBuilder to be available.
    if HAS_SCENE_GRAPH_BUILDER and HAS_YOLO and HAS_SAM:
        logger.info("\n--- Demonstrating Scene Graph Building with YOLO, SAM, and Persistent Homology ---")
        
        # Create a dummy image for demonstration
        # Use cfg.paths.data_root for consistent path management
        dummy_image_dir = os.path.join(cfg.paths.data_root, "demo")
        os.makedirs(dummy_image_dir, exist_ok=True)
        dummy_image_path = os.path.join(dummy_image_dir, "demo_image.png")
        
        # Create a more complex dummy image for better demonstration
        demo_image = np.zeros((cfg.data.image_size, cfg.data.image_size, 3), dtype=np.uint8)
        cv2.circle(demo_image, (70, 70), 30, (255, 255, 255), -1)   # White circle
        cv2.circle(demo_image, (70, 70), 15, (0, 0, 0), -1)   # Black circle inside (hole)
        cv2.rectangle(demo_image, (180, 50), (230, 100), (0, 255, 0), -1)   # Green square
        cv2.ellipse(demo_image, (128, 180), (60, 30), 0, 0, 360, (255, 0, 0), -1)   # Red ellipse
        
        Image.fromarray(demo_image).save(dummy_image_path)
        logger.info(f"Created dummy image at {dummy_image_path}")
        yolo_model, sam_predictor = None, None
        try:
            yolo_model, sam_predictor = load_yolo_and_sam_models(cfg) # Pass cfg
            logger.info("YOLO and SAM models loaded for demonstration.")
            # Detect and segment objects
            detected_bboxes, detected_masks, _ = detect_and_segment_image(
                demo_image, yolo_model, sam_predictor, cfg
            )
            logger.info(f"Detected {len(detected_bboxes)} objects.")
            if detected_bboxes:
                # Dummy attribute and relation logits for demonstration
                # In a real scenario, these would come from your PerceptionModule
                dummy_attribute_logits = {
                    'shape': torch.randn(len(detected_bboxes), cfg.model.attribute_classifier_config.shape),
                    'color': torch.randn(len(detected_bboxes), cfg.model.attribute_classifier_config.color),
                    'fill': torch.randn(len(detected_bboxes), cfg.model.attribute_classifier_config.fill),
                    'size': torch.randn(len(detected_bboxes), cfg.model.attribute_classifier_config.size),
                    'orientation': torch.randn(len(detected_bboxes), cfg.model.attribute_classifier_config.orientation),
                    'texture': torch.randn(len(detected_bboxes), cfg.model.attribute_classifier_config.texture),
                }
                # For relations, it's (N_edges, N_relations). N_edges = N_objects * (N_objects - 1)
                num_objects = len(detected_bboxes)
                num_edges = num_objects * (num_objects - 1)
                dummy_relation_logits = torch.randn(num_edges, cfg.model.relation_gnn_config.num_relations)
                
                # Build the scene graph
                scene_graph_builder = SceneGraphBuilder(cfg) # Pass cfg
                inferred_scene_graph = scene_graph_builder.build_scene_graph(
                    demo_image, detected_bboxes, detected_masks,
                    dummy_attribute_logits, dummy_relation_logits
                )
                
                logger.info("\n--- Inferred Scene Graph for Demo Image ---")
                logger.info(json.dumps(inferred_scene_graph, indent=2))
            else:
                logger.warning("No objects detected for scene graph building demonstration.")
        except Exception as e:
            logger.error(f"Error during scene graph demonstration: {e}", exc_info=True)
        finally:
            # Clean up dummy file
            if os.path.exists(dummy_image_path):
                os.remove(dummy_image_path)
                logger.info(f"Cleaned up dummy image: {dummy_image_path}")
    else:
        logger.warning("Skipping scene graph demonstration as required components (YOLO, SAM, SceneGraphBuilder) are not available.")
    
    logger.info("--- Bongard Solver Main Pipeline finished. ---")
    
    if HAS_WANDB and cfg.training.get('use_wandb', False):
        wandb.finish()

if __name__ == "__main__":
    # This block ensures that multiprocessing works correctly when running
    # from a script (especially on Windows).
    mp.set_start_method('spawn', force=True)
    main()

