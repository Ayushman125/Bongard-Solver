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

# --- Emergent System Imports ---
import sys
# Add the parent directory of 'emergent' to the Python path
# Assuming main.py is in src/, and emergent is in src/emergent/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))) # Add src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'emergent'))) # Add src/emergent/

from src.emergent.workspace_ext import Workspace
from src.emergent.codelets import Scout, GroupScout, RuleTester, RuleBuilder # Import new codelets
from src.utils.compute_temperature import compute_temperature # Now from utils/
from src.utils.general import set_seed # Assuming set_seed is in utils/general

# --- Import the actual DSL, ILP, Causal, and RL classes ---
from src.dsl import DSL # Import the real DSL class
from src.ilp import RuleInducer # Import the real ILP class
from src.causal import CausalFilter # Import the real CausalFilter class
from src.rl_module import RLAgent # Import the real RLAgent class

# --- Import Core Models Components ---
from core_models.ensemble import EnsembleSolver, load_trained_model, train_ensemble, train_distilled_student_orchestrator_combined # For ensembling
from core_models.training import train_perception_with_buffer, _run_single_training_session_ensemble # For main model training
from core_models.training_moco import main as run_moco_pretraining # For MoCo pretraining
from core_models.training_args import load_config as load_training_config # For structured config loading

# from core_models.hpo import run_hpo # Assuming HPO function exists here or in a separate hpo.py
# from core_models.pruning_quantization import run_pruning_and_quantization # Assuming these functions exist

# Instantiate the real DSL class (it's a singleton)
dsl = DSL()
# Instantiate the real ILP and Causal modules
ilp = RuleInducer()
causal = CausalFilter()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import scene_graph_builder for demonstration
try:
    from src.scene_graph_builder import SceneGraphBuilder # Import the class
    HAS_SCENE_GRAPH_BUILDER = True
    logger.info("src/scene_graph_builder.py found.")
except ImportError:
    HAS_SCENE_GRAPH_BUILDER = False
    logger.warning("src/scene_graph_builder.py not found. Scene graph building will be mocked.")
    # Dummy class for demonstration if not found
    class SceneGraphBuilder:
        def __init__(self, images: List[Any], config: Optional[Dict[str, Any]] = None):
            logger.warning("Dummy SceneGraphBuilder initialized for main.py. Problem solving will be mocked.")
            self.images = images
            self.objects = [f"obj_{i}" for i in range(len(images))] if images else ["obj_0", "obj_1"] # Mock object IDs
            self._solution_found = False
            self._solution = None
            self.config = config if config is not None else {} # Store config even for dummy
        def extract_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
            """Mocks feature extraction for main.py's dummy SceneGraphBuilder."""
            if feat_type == 'shape': return "circle", 0.8
            if feat_type == 'color': return "red", 0.7
            if feat_type == 'size': return "small", 0.6
            if feat_type == 'position_h': return "center_h", 0.5
            if feat_type == 'position_v': return "center_v", 0.5
            if feat_type == 'fill': return "solid", 0.7
            if feat_type == 'orientation': return "upright", 0.6
            if feat_type == 'texture': return "none_texture", 0.6
            return "unknown", 0.1
        def problem_solved(self) -> bool:
            return self._solution_found
        
        def mark_solution(self, solution: Any):
            self._solution = solution
            self._solution_found = True
        def get_solution(self) -> Optional[Any]:
            return self._solution
        
        def build_scene_graph(self, image_np: np.ndarray, **kwargs) -> Dict[str, Any]:
            """Dummy build_scene_graph for mock data generation."""
            # Create a simple mock scene graph for a single image
            mock_objects = []
            if image_np is not None: # Ensure image_np is not None
                obj_id = f"obj_{random.randint(0, 99)}" # Random ID
                # Mock some attributes based on random choices
                shape = random.choice(['circle', 'square', 'triangle'])
                color = random.choice(['red', 'blue', 'green'])
                size = random.choice(['small', 'medium', 'large'])
                mock_objects.append({
                    'id': obj_id,
                    'attributes': {'shape': shape, 'color': color, 'size': size},
                    'bbox_xyxy': [0,0,10,10], # Dummy bbox
                    'centroid': [5,5] # Dummy centroid
                })
            return {'objects': mock_objects, 'relations': []}

        def get_object_image(self, obj_id: str) -> Optional[Any]:
            """Dummy: Returns a dummy image for an object."""
            return Image.new('RGB', (50, 50), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))


# Main function definition with Hydra
@hydra.main(config_path="conf", config_name="config", version_base=None) # Added version_base
def main(cfg: DictConfig):
    """
    Main entry point for the Bongard Solver project.
    Orchestrates the entire pipeline: MoCo pretraining, YOLO fine-tuning, HPO, main training,
    pruning, quantization, and the emergent symbolic reasoning loop.
    """
    logger.info("--- Starting Bongard Solver Main Pipeline ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}") # Use OmegaConf.to_yaml for pretty printing
    
    # Set logging level from config
    logger.setLevel(getattr(logging, cfg.debug.log_level.upper())) # Use getattr for dynamic level
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs(cfg.debug.save_model_checkpoints, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {cfg.debug.save_model_checkpoints}")
    
    # Initialize Weights & Biases if enabled
    HAS_WANDB = False # Default to False, will be updated by imports.py if wandb is found
    try:
        import wandb
        HAS_WANDB = True
    except ImportError:
        logger.warning("Weights & Biases library not found. Skipping W&B initialization.")
    if HAS_WANDB and cfg.debug.get('use_wandb', False): # Use cfg.debug.use_wandb
        wandb.init(project=cfg.debug.wandb_project, entity=cfg.debug.wandb_entity, config=OmegaConf.to_container(cfg, resolve=True))
        logger.info("Weights & Biases initialized.")
    else:
        logger.info("Weights & Biases is disabled or not found.")
    
    # Set random seed
    set_seed(cfg.debug.seed) # Use cfg.debug.seed
    
    # --- 1. MoCo Self-Supervised Pretraining ---
    if cfg.model.simclr_config.get('enabled', False):
        logger.info("Initiating MoCo-V2 self-supervised pretraining.")
        # The run_moco_pretraining function is now main in training_moco.py
        # It expects a config object. We need to pass the resolved DictConfig.
        # For simplicity, if run_moco_pretraining is designed to be called directly,
        # it might need a similar argparse setup or accept DictConfig directly.
        # For now, let's assume it can take the DictConfig.
        run_moco_pretraining(cfg) # Calling the main function from training_moco.py
        logger.info("MoCo-V2 pretraining completed.")
    else:
        logger.info("MoCo-V2 pretraining is disabled in config.")

    # --- 2. YOLO Object Detector Fine-tuning (if enabled) ---
    if cfg.model.get('fine_tune_yolo', False):
        logger.info("Initiating YOLO object detector fine-tuning.")
        try:
            from ultralytics import YOLO as RealYOLO
            yolo_model = RealYOLO(cfg.model.object_detector_model_path)
            logger.info(f"Fine-tuning YOLO model: {cfg.model.object_detector_model_path}")
            yolo_model.train(
                data=cfg.model.yolo_data_yaml,
                epochs=cfg.model.fine_tune_epochs,
                imgsz=cfg.model.img_size,
                batch=cfg.training.batch_size, # Use training batch size
                lr0=cfg.training.learning_rate, # Use training learning rate
                project=os.path.join(cfg.debug.logs_dir, "yolo_runs"), # Use debug logs_dir
                name="yolo_finetune"
            )
            # Save the fine-tuned YOLO model path
            cfg.model.object_detector_model_path = os.path.join(cfg.debug.logs_dir, "yolo_runs", "yolo_finetune", "weights", "best.pt")
            logger.info(f"YOLO fine-tuning completed. Best model saved to: {cfg.model.object_detector_model_path}")
        except ImportError:
            logger.error("Ultralytics YOLO not found. Skipping YOLO fine-tuning.")
        except Exception as e:
            logger.error(f"Error during YOLO fine-tuning: {e}", exc_info=True)
    else:
        logger.info("YOLO object detector fine-tuning is disabled in config.")

    # --- 3. Hyperparameter Optimization (HPO) (if enabled) ---
    if cfg.hpo.get('enabled', False):
        logger.info("Initiating Hyperparameter Optimization.")
        try:
            # Assuming run_hpo is available, e.g., from core_models.hpo
            # from core_models.hpo import run_hpo
            # run_hpo(cfg=cfg, n_trials=cfg.hpo.trials, timeout=cfg.hpo.timeout, n_jobs=cfg.hpo.n_jobs, study_path=cfg.hpo.study_path)
            logger.warning("HPO function (run_hpo) is a placeholder. Skipping actual HPO run.")
        except ImportError:
            logger.error("HPO module (core_models.hpo) not found. Skipping HPO.")
        except Exception as e:
            logger.error(f"Error during HPO: {e}", exc_info=True)
        logger.info("Hyperparameter Optimization completed.")
    else:
        logger.info("Hyperparameter Optimization is disabled in config.")

    # --- 4. Main Bongard Solver Training (Phase 1: Offline Training of Perception Model) ---
    logger.info("Initiating main Bongard Solver training pipeline (Phase 1: Offline Perception Training).")
    
    # Load the structured configuration for training
    training_cfg = load_training_config(config_path=None) # Load default if no path specified, or pass cfg.yaml
    # Update with values from Hydra config if needed
    training_cfg.device = str(cfg.device)
    training_cfg.model.backbone = cfg.model.backbone
    training_cfg.model.pretrained = cfg.model.pretrained
    training_cfg.model.bongard_head_config = OmegaConf.to_container(cfg.model.bongard_head_config, resolve=True)
    training_cfg.model.relation_gnn_config = OmegaConf.to_container(cfg.model.relation_gnn_config, resolve=True)
    training_cfg.model.attribute_classifier_config = OmegaConf.to_container(cfg.model.attribute_classifier_config, resolve=True)
    training_cfg.data.image_size = cfg.data.image_size
    training_cfg.data.dataloader_workers = cfg.data.dataloader_workers
    training_cfg.data.use_synthetic_data = cfg.data.use_synthetic_data
    training_cfg.data.synthetic_data_config = OmegaConf.to_container(cfg.data.synthetic_data_config, resolve=True)
    training_cfg.training.epochs = cfg.training.epochs
    training_cfg.training.batch_size = cfg.training.batch_size
    training_cfg.training.learning_rate = cfg.training.learning_rate
    training_cfg.training.optimizer = cfg.training.optimizer
    training_cfg.training.scheduler = cfg.training.scheduler
    training_cfg.training.early_stop_patience = cfg.training.early_stop_patience
    training_cfg.training.use_amp = cfg.training.use_amp
    training_cfg.training.log_every_n_steps = cfg.training.log_every_n_steps
    training_cfg.debug.save_model_checkpoints = cfg.debug.save_model_checkpoints
    training_cfg.debug.logs_dir = cfg.debug.logs_dir
    training_cfg.debug.use_wandb = cfg.debug.use_wandb
    training_cfg.debug.wandb_project = cfg.debug.wandb_project
    training_cfg.debug.wandb_entity = cfg.debug.wandb_entity
    training_cfg.training.use_knowledge_distillation = cfg.training.use_knowledge_distillation
    training_cfg.training.distillation_config = OmegaConf.to_container(cfg.training.distillation_config, resolve=True)
    training_cfg.training.num_ensemble_members = cfg.training.num_ensemble_members
    training_cfg.training.ensemble_training_epochs = cfg.training.ensemble_training_epochs
    training_cfg.training.teacher_ensemble_type = cfg.training.teacher_ensemble_type
    training_cfg.training.stacked_ensemble_config = OmegaConf.to_container(cfg.training.stacked_ensemble_config, resolve=True)
    training_cfg.training.curriculum_learning = cfg.training.curriculum_learning
    training_cfg.training.curriculum_config = OmegaConf.to_container(cfg.training.curriculum_config, resolve=True)
    training_cfg.training.use_mean_teacher = cfg.training.use_mean_teacher
    training_cfg.training.mean_teacher_config = OmegaConf.to_container(cfg.training.mean_teacher_config, resolve=True)
    training_cfg.training.use_mixup_cutmix = cfg.training.use_mixup_cutmix
    training_cfg.training.mixup_cutmix_config = OmegaConf.to_container(cfg.training.mixup_cutmix_config, resolve=True)
    training_cfg.model.use_scene_gnn = cfg.model.use_scene_gnn
    training_cfg.model.use_glu = cfg.model.use_glu
    training_cfg.model.use_lora = cfg.model.use_lora
    training_cfg.model.lora_config = OmegaConf.to_container(cfg.model.lora_config, resolve=True)
    training_cfg.model.use_dropblock = cfg.model.use_dropblock
    training_cfg.model.dropblock_config = OmegaConf.to_container(cfg.model.dropblock_config, resolve=True)
    training_cfg.model.use_stochastic_depth = cfg.model.use_stochastic_depth
    training_cfg.model.drop_path_max = cfg.model.drop_path_max
    training_cfg.model.drop_path_layers = cfg.model.drop_path_layers
    training_cfg.training.attribute_loss_weight = cfg.training.attribute_loss_weight
    training_cfg.training.relation_loss_weight = cfg.training.relation_loss_weight
    training_cfg.training.consistency_loss_weight = cfg.training.consistency_loss_weight
    training_cfg.training.feature_consistency_weight = cfg.training.feature_consistency_weight
    training_cfg.training.symbolic_consistency_weight = cfg.training.symbolic_consistency_weight
    training_cfg.training.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
    training_cfg.training.max_grad_norm = cfg.training.max_grad_norm


    if training_cfg.training.num_ensemble_members > 1 and training_cfg.training.use_knowledge_distillation:
        logger.info(f"Training ensemble of {training_cfg.training.num_ensemble_members} teachers and distilling to a student.")
        # train_distilled_student_orchestrator_combined handles both teacher training/loading and student distillation
        trained_student_model, teacher_ensemble_metrics, student_metrics = train_distilled_student_orchestrator_combined(
            num_ensemble_members=training_cfg.training.num_ensemble_members,
            train_loader=None, # Will be handled internally by data module
            val_loader=None, # Will be handled internally by data module
            student_model_config=training_cfg, # Student uses the main training config
            epochs_student=training_cfg.training.epochs,
            teacher_ensemble_type=training_cfg.training.teacher_ensemble_type,
            train_members=True, # Always train teachers in this orchestration
            cfg=training_cfg,
            current_rank=0, # Assuming single GPU for now
            world_size=1,
            is_ddp_initialized=False
        )
        logger.info(f"Distillation completed. Student model saved.")
        logger.info(f"Teacher Ensemble Metrics: Accuracy={teacher_ensemble_metrics.get('accuracy', 0.0):.4f}")
        logger.info(f"Student Model Metrics: Accuracy={student_metrics.get('accuracy', 0.0):.4f}")
    elif training_cfg.training.num_ensemble_members > 1:
        logger.info(f"Training an ensemble of {training_cfg.training.num_ensemble_members} models (no distillation).")
        trained_ensemble_models, all_members_best_metrics = train_ensemble(
            num_members=training_cfg.training.num_ensemble_members,
            train_loader=None,
            val_loader=None,
            cfg=training_cfg,
            current_rank=0,
            world_size=1,
            is_ddp_initialized=False
        )
        logger.info(f"Ensemble training completed. Models saved.")
    else:
        logger.info("Training a single perception model.")
        # This function will now handle data loading internally using the config
        best_model_path, val_predictions_logits, val_true_labels, best_metrics = train_perception_with_buffer(training_cfg)
        logger.info("Main Bongard Solver training completed.")
        logger.info(f"Best Perception Model saved to: {best_model_path}")
        logger.info(f"Validation Accuracy: {best_metrics.get('val_accuracy', 0.0):.4f}")


    # --- Optional: Periodic Online Fine-tuning of Perception Model ---
    if cfg.training.get('online_finetuning', False):
        logger.info("Initiating periodic online fine-tuning of the perception model.")
        # This part would typically run in a loop during the emergent phase,
        # collecting new data (e.g., from human feedback or successful problem solutions)
        # and using it to fine-tune the perception model.
        
        # Placeholder for online fine-tuning logic:
        # 1. Collect new data (e.g., from replay buffer or human-in-the-loop)
        #    This data would likely be scene graphs + images.
        # 2. Add to replay buffer (if not already there)
        # 3. Periodically sample from replay buffer and fine-tune perception model
        
        # Example: Load the best trained model (either single or student)
        # For simplicity, assume the last saved model is the one to fine-tune
        perception_model_to_finetune_path = os.path.join(training_cfg.debug.save_model_checkpoints, "best_perception_model.pt")
        if training_cfg.training.use_knowledge_distillation:
            perception_model_to_finetune_path = os.path.join(training_cfg.debug.save_model_checkpoints, "distilled_student_perception_model.pt")
        elif training_cfg.training.num_ensemble_members > 1:
            # If ensemble, fine-tune the first member or a specific one
            perception_model_to_finetune_path = os.path.join(training_cfg.debug.save_model_checkpoints, "member_0_best_model.pt")

        if os.path.exists(perception_model_to_finetune_path):
            logger.info(f"Loading perception model for online fine-tuning from {perception_model_to_finetune_path}")
            # Load the model using load_trained_model, which returns PerceptionModule
            perception_model, _ = load_trained_model(
                model_path=perception_model_to_finetune_path,
                cfg=training_cfg,
                current_rank=0,
                is_ddp_initialized=False,
                val_loader=None # No validation needed during this specific load
            )
            
            # Simulate collecting new data and adding to buffer
            # In a real scenario, this would involve actual data generation/collection
            logger.info("Simulating collection of new data for online fine-tuning...")
            # This would involve generating a small dataset or sampling from a live source
            # For this placeholder, we'll just log the intent.
            
            # Call a hypothetical fine-tuning function
            # This function would take the loaded model and a small new dataset/buffer
            # For now, it's a dummy call.
            logger.info("Performing a dummy online fine-tuning step...")
            # hypothetical_online_finetune(perception_model, new_data_buffer, training_cfg)
            logger.info("Dummy online fine-tuning completed.")
        else:
            logger.warning(f"Perception model not found at {perception_model_to_finetune_path}. Skipping online fine-tuning.")
    else:
        logger.info("Periodic online fine-tuning is disabled in config.")

    # --- 5. Pruning and Quantization (if enabled) ---
    if cfg.pruning.get('enabled', False) or cfg.quantization.get('qat', False) or cfg.quantization.get('ptq', False):
        logger.info("Initiating Pruning and Quantization pipeline.")
        try:
            # Assuming run_pruning_and_quantization is available, e.g., from core_models.pruning_quantization
            # from core_models.pruning_quantization import run_pruning_and_quantization
            # run_pruning_and_quantization(cfg)
            logger.warning("Pruning and Quantization function (run_pruning_and_quantization) is a placeholder. Skipping actual run.")
        except ImportError:
            logger.error("Pruning/Quantization module (core_models.pruning_quantization) not found. Skipping.")
        except Exception as e:
            logger.error(f"Error during Pruning/Quantization: {e}", exc_info=True)
        logger.info("Pruning and Quantization pipeline completed.")
    else:
        logger.info("Pruning and Quantization are disabled in config.")

    # --- 6. Emergent Perception and Symbolic Reasoning Loop ---
    # Load dummy images for demonstration. In a real scenario, these would be actual image data.
    # For now, we generate a list of strings representing image paths/identifiers.
    dummy_image_paths = [f"path/to/image_{i}.png" for i in range(10)] # Example: 10 dummy image paths
    
    logger.info("Initiating emergent perception and symbolic reasoning loop.")
    # Pass the Hydra config directly to solve
    solution = solve(dummy_image_paths, cfg) 
    logger.info(f"Emergent system solution: {solution}")
    
    logger.info("--- Bongard Solver Main Pipeline finished. ---")
    
    if HAS_WANDB and cfg.debug.get('use_wandb', False): # Use cfg.debug.use_wandb
        wandb.finish() # Ensure wandb run is finished

def load_images(data_path: str) -> List[Any]:
    """
    Dummy function to simulate loading images.
    In a real scenario, this would load actual image data.
    """
    logger.info(f"Loading dummy images from (simulated) {data_path}...")
    # For demonstration, return a list of placeholder strings or dummy numpy arrays
    return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(10)] # Example: 10 dummy images as numpy arrays

def solve(images: List[Any], cfg: DictConfig) -> Optional[Any]:
    """
    The main emergent perception and symbolic reasoning loop.
    Replaces the linear pipeline with a dynamic, codelet-driven process.
    Args:
        images (List[Any]): List of image data (e.g., file paths or numpy arrays) for the problem.
        cfg (DictConfig): The Hydra configuration object.
    Returns:
        Optional[Any]: The inferred solution to the Bongard problem, or None if not found.
    """
    logger.info("Starting emergent solve loop.")
    # 0. Initialize emergent workspace
    # Pass the actual Hydra config to Workspace
    ws = Workspace(images, config=OmegaConf.to_container(cfg, resolve=True)) 
    temperature = 1.0 # Initial temperature
    
    # 1. Seed bottom-up Scouts and GroupScouts
    for obj_id_list in ws.object_ids_per_image: # Iterate through objects for each image
        for obj_id in obj_id_list:
            for feat in ['shape', 'color', 'size', 'position_h', 'position_v', 'fill', 'orientation', 'texture']:
                ws.post_codelet(Scout(obj_id, feat, urgency=0.1))
    logger.info(f"Seeded initial Scout codelets for {len(ws.objects)} objects across all images.")
    
    # Seed GroupScouts
    ws.post_codelet(GroupScout(urgency=0.2)) # Initial GroupScout
    logger.info("Seeded initial GroupScout codelet.")

    # --- Generate "real" (dynamically mocked) support set data ---
    # Select a subset of images for the support set (e.g., first 2-4 images)
    num_support_images = min(len(images), 4) # Use up to 4 images for support set
    support_image_data = random.sample(images, num_support_images) if num_support_images > 0 else []
    support_set_features: List[torch.Tensor] = []
    support_set_labels: List[int] = []
    support_set_scene_graphs: List[Dict[str, Any]] = []
    
    # Process each support image to create scene graphs and features
    for i, img_data_np in enumerate(support_image_data):
        # Build scene graph for this dummy image using the (real or dummy) SceneGraphBuilder
        scene_graph_for_image = ws.sg.build_scene_graph(img_data_np)
        support_set_scene_graphs.append({'image_data': img_data_np.tolist(), 'scene_graph': scene_graph_for_image}) # Convert numpy to list for JSON compatibility
        
        # Assign dummy label (e.g., alternating positive/negative)
        support_set_labels.append(1 if i % 2 == 0 else 0)

        # Convert scene graph to a feature vector for the RL agent
        feature_vector_elements = []
        num_objects_in_sg = len(scene_graph_for_image.get('objects', []))
        feature_vector_elements.append(num_objects_in_sg) # Simple count
        
        # Count specific shapes and colors for a richer feature vector
        shape_counts = {'circle': 0, 'square': 0, 'triangle': 0}
        color_counts = {'red': 0, 'blue': 0, 'green': 0}
        for obj in scene_graph_for_image.get('objects', []):
            if 'attributes' in obj:
                shape = obj['attributes'].get('shape')
                color = obj['attributes'].get('color')
                if shape in shape_counts:
                    shape_counts[shape] += 1
                if color in color_counts:
                    color_counts[color] += 1
        
        feature_vector_elements.extend(shape_counts.values())
        feature_vector_elements.extend(color_counts.values())
        
        # Ensure the feature vector has a consistent size. Pad with zeros if necessary.
        # This `D_FEATURE` should ideally be derived from the actual feature space.
        # For this dummy, let's assume a fixed size.
        D_FEATURE = 7 # 1 (num_objects) + 3 (shapes) + 3 (colors)
        current_feature_tensor = torch.tensor(feature_vector_elements, dtype=torch.float32)
        if current_feature_tensor.shape[0] < D_FEATURE:
            padding = torch.zeros(D_FEATURE - current_feature_tensor.shape[0], dtype=torch.float32)
            current_feature_tensor = torch.cat((current_feature_tensor, padding))
        elif current_feature_tensor.shape[0] > D_FEATURE:
            current_feature_tensor = current_feature_tensor[:D_FEATURE] # Truncate if too long
        support_set_features.append(current_feature_tensor)
    
    # Instantiate RLAgent here, after support_context_dim is determined
    rl_agent = RLAgent(support_context_dim=D_FEATURE)
    logger.info(f"RLAgent instantiated with support_context_dim={D_FEATURE}.")
    # --- End of support set generation ---

    # 2. Emergent perception + symbolic loop
    max_solve_iterations = 100 # Prevent infinite loops for demonstration
    iteration = 0
    while not ws.sg.problem_solved() and ws.coderack and iteration < max_solve_iterations:
        iteration += 1
        # Compute temperature based on the current iteration and max_solve_iterations
        temperature = compute_temperature(iteration, max_solve_iterations,
                                          initial_temperature=cfg.slipnet_config.get('initial_temperature', 1.0),
                                          final_temperature=cfg.slipnet_config.get('final_temperature', 0.1),
                                          annealing_type=cfg.slipnet_config.get('annealing_type', 'linear'))

        logger.info(f"\n--- Solve Loop Iteration {iteration} (Temperature: {temperature:.4f}) ---")
        
        # A. Run codelets
        ws.run_codelets(temperature, max_steps=20) 
        
        # B. Decay & spread activation in the Concept Network
        # The ConceptNet is now part of Workspace
        ws.concept_net.step(decay_factor=cfg.slipnet_config.get('decay_factor', 0.01),
                            max_activation=cfg.slipnet_config.get('max_activation', 1.0))
        
        # C. Update temperature (already done at the beginning of the loop for next iteration)
        
        # D. Convert built features into DSL facts
        dsl.clear_facts() 
        for obj_id, feat_type, value, confidence in ws.built: # Now 'built' includes confidence
            fact = f"{feat_type.upper()}({obj_id},{value})" 
            dsl.add_fact(fact, source='emergent', confidence=confidence) # Pass confidence to DSL
        logger.info(f"Converted {len(ws.built)} built features into DSL facts. Total DSL facts: {len(dsl.facts)}")
        
        # E. Run ILP → causal → RL/MCTS
        if dsl.facts: 
            rules_from_ilp = ilp.generate(dsl.get_facts()) 
            logger.info(f"ILP generated {len(rules_from_ilp)} rules.")
            
            rules_after_causal = causal.filter(rules_from_ilp)
            logger.info(f"Causal filtered to {len(rules_after_causal)} rules.")
            
            best_solution_candidate = rl_agent.search(
                rules=rules_after_causal, 
                support_set_features=support_set_features,
                support_set_labels=support_set_labels,
                support_set_scene_graphs=support_set_scene_graphs,
                num_episodes=5 
            )
            logger.info(f"RL/MCTS found solution candidate with score: {best_solution_candidate.score:.4f}")
            if best_solution_candidate.score > 0.9: # Threshold for considering a solution
                ws.sg.mark_solution(best_solution_candidate.description)
                logger.info(f"Problem solved with score {best_solution_candidate.score:.4f}: {best_solution_candidate.description}")
                break
        else:
            logger.warning("No DSL facts generated in this iteration. Skipping ILP/Causal/RL.")
    
    if not ws.sg.problem_solved():
        logger.warning("Emergent solve loop finished without finding a strong solution or coderack is empty.")
    return ws.sg.get_solution()

if __name__ == "__main__":
    # Ensure multiprocessing start method is 'spawn' for CUDA compatibility
    # This should be done once at the beginning of the program.
    mp.set_start_method('spawn', force=True)
    main()
