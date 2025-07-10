import os
import glob
import numpy as np
from ultralytics import YOLO as RealYOLO
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss, roc_auc_score, accuracy_score
import logging
import yaml
import argparse
import math
import random
import zipfile
import requests
from tqdm.auto import tqdm
import collections
import functools
import time
import json
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import jsonschema
import io
import copy
# PyTorch specific imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, efficientnet_b0
import torchvision.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim.swa_utils as swa_utils
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision.io
import torch.profiler
# DALI imports
import nvidia.dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali import pipeline_def
# PyTorch Geometric imports
from torch_geometric.nn import GCNConv, Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as pyg_nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from bongard_rules import ALL_BONGARD_RULES, BongardRule

# --- NEW: Imports for XAI (Grad-CAM) ---
HAS_PYTORCH_GRADCAM = False
try:
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_PYTORCH_GRADCAM = True
except ImportError:
    logging.warning("Could not import pytorch_grad_cam. Grad-CAM visualizations will not be available.")

# Set up logging before any logger calls
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log DALI version
try:
    logger.info(f"NVIDIA DALI version: {nvidia.dali.__version__}")
except Exception as e:
    logger.warning(f"Could not determine DALI version: {e}")

# Conditional imports for optional libraries
HAS_ELASTIC_TRANSFORM = False
try:
    from torchvision.transforms import ElasticTransform
    HAS_ELASTIC_TRANSFORM = True
except ImportError:
    pass

# SAM (Sharpness-Aware Minimization) for optimizer
HAS_SAM = False
try:
    # Attempt to import your local SAM implementation
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super(SAM, self).__init__(params, defaults)
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)
        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for p in self.param_groups:
                for d in p['params']:
                    if d.grad is None: continue
                    self.state[d]['old_p'] = d.data.clone()
                    eps = p['rho'] / (grad_norm + 1e-12)
                    d.add_(d.grad, alpha=eps)
            if zero_grad: self.zero_grad()
        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for p in self.param_groups:
                for d in p['params']:
                    if d.grad is None: continue
                    d.data = self.state[d]['old_p']
            self.base_optimizer.step()
            if zero_grad: self.zero_grad()
        def _grad_norm(self):
            norm = torch.norm(torch.stack([
                (p.grad if p.grad is not None else torch.zeros_like(p)).norm(p=2)
                for group in self.param_groups for p in group['params']
            ]), p=2)
            return norm
    HAS_SAM = True
    logger.info("Successfully imported SAM from local sam.py.")
except ImportError:
    logger.warning("Could not import local sam.py. SAM optimizer will not be available.")

HAS_WANDB = False
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    pass

HAS_KORNIA = False
try:
    import kornia.augmentation as K
    import kornia.geometry.transform as KGT
    import kornia.color as KC
    import kornia.filters as KF
    # from kornia.models import SegmentAnything  # For SAM mask-based fallback - This might not be needed for Kornia augmentations
    HAS_KORNIA = True
except ImportError:
    logger.warning("Kornia not found. Kornia augmentations will be skipped.")

HAS_SOPHIA = False
try:
    from sophia import SophiaG
    HAS_SOPHIA = True
    logger.info("Successfully imported SophiaG.")
except ImportError:
    logger.warning("Could not import SophiaG. Sophia optimizer will not be available.")

# LoRA Adapters
HAS_LORA = False
try:
    import loralib as LoRA
    HAS_LORA = True
except ImportError:
    logger.warning("Could not import loralib. LoRA adapters will not be available.")

# timm optimizers (Lion, MADGRAD) - Assuming torch_optimizer for direct import
HAS_TIMM_OPTIM = False
try:
    from torch_optimizer import Lion, MADGRAD
    HAS_TIMM_OPTIM = True
except ImportError:
    logger.warning("Could not import Lion or MADGRAD from torch_optimizer. These optimizers will not be available.")

# Quantization imports
HAS_TORCH_QUANTIZATION = False
try:
    import torch.quantization as tq
    HAS_TORCH_QUANTIZATION = True
except ImportError:
    logger.warning("torch.quantization not available. QAT will be disabled.")

HAS_TIMM = False
try:
    import timm  # For advanced backbones
    HAS_TIMM = True
except ImportError:
    logger.warning("timm not found. Advanced backbones will not be available.")

# --- Global Configuration and Constants ---
CONFIG = {
    'data': {
        'dataset_name': 'BongardProblems',
        'data_root_path': './data/bongard_problems', # Path to your dataset
        'image_size': [96, 96], # Default image size for models
        'num_channels': 3, # 1 for grayscale, 3 for RGB
        'train_test_split_ratio': 0.2, # Ratio for validation set
        'dataloader_num_workers': 4, # Number of workers for DALI/PyTorch DataLoader
        'use_synthetic_data': True, # Set to True to use programmatic synthetic data
        'synthetic_samples': 1000, # Number of synthetic samples to generate
        'curriculum_annealing_epochs': 5, # Epochs over which to anneal image size/difficulty
        'dali_device_memory_padding': 64 * 1024 * 1024, # DALI GPU memory padding
        'dali_host_memory_padding': 64 * 1024 * 1024, # DALI Host memory padding
        'dali_prebuild_streams': 2, # Number of streams to prebuild in DALI pipeline
        'image_extension': 'png', # NEW: Specify image file extension (e.g., 'png', 'jpg')
    },
    'model': {
        'name': 'PerceptionModule',
        'num_classes': 2, # Bongard problem is binary (0 or 1)
        'attribute_backbone': 'efficientnet_b0', # 'mobilenet_v2', 'mobilenet_v3_small', 'efficientnet_b0', or timm models
        'gnn_depth': 3, # Number of GNN layers
        'object_detector_model_path': 'yolov8n.pt', # Path to YOLOv8 weights
        'detection_confidence_threshold': 0.3, # Confidence threshold for object detection
        'sam_model_type': 'vit_b', # SAM model type for fallback ('vit_b', 'vit_l', 'vit_h')
        'initial_dropout_rate': 0.2, # Dropout rate for AttributeClassifier
        'batch_size': 32,
        'epochs': 50,
        'optimizer': 'AdamW', # 'AdamW', 'SGD', 'SophiaG', 'Lion', 'MADGRAD'
        'initial_learning_rate': 1e-4,
        'max_learning_rate': 1e-3, # For OneCycleLR
        'scheduler': 'OneCycleLR', # 'OneCycleLR', 'ReduceLROnPlateau', None
        'mc_dropout_samples': 5, # Number of forward passes for MC Dropout inference (0 to disable)
        'random_seed': 42 # Base random seed for reproducibility
    },
    'training': {
        'use_amp': True, # Use Automatic Mixed Precision
        'gradient_accumulation_steps': 1, # Accumulate gradients over N batches
        'label_smoothing_epsilon': 0.1, # Epsilon for label smoothing
        'weight_decay': 1e-5,
        'lr_scheduler_factor': 0.1, # For ReduceLROnPlateau
        'lr_scheduler_patience': 5, # For ReduceLROnPlateau
        'onecycle_pct_start': 0.3, # For OneCycleLR
        'early_stopping_patience': 10, # Epochs to wait before stopping
        'early_stopping_min_delta': 0.001, # Minimum change to qualify as improvement
        'early_stopping_monitor_metric': 'val_loss', # 'val_loss' or 'val_accuracy'
        'model_checkpoint_name': 'best_perception_model.pt', # Name for saving best model
        'checkpoint_dir': './checkpoints', # Directory to save checkpoints
        'use_wandb': False, # Enable Weights & Biases logging
        'enable_profiler': False, # Enable PyTorch profiler
        'profiler_schedule_wait': 1, # Profiler schedule parameters
        'profiler_schedule_warmup': 1,
        'profiler_schedule_active': 3,
        'profiler_schedule_repeat': 1,
        'feature_consistency_alpha': 0.5, # Weight for feature consistency loss (0 to disable)
        'feature_consistency_loss_type': 'mse', # 'mse', 'cosine', 'kl_div'
        'symbolic_consistency_alpha': 0.2, # Weight for symbolic consistency loss (0 to disable)
        'use_knowledge_distillation': False, # Enable knowledge distillation for student training
        'distillation_alpha': 0.5, # Weight for distillation loss
        'distillation_temperature': 2.0, # Temperature for teacher/student logits
        'knowledge_replay_enabled': False, # Enable knowledge replay buffer
        'replay_buffer_size': 1000, # Max size of the replay buffer
        'replay_frequency_epochs': 5, # How often to add/sample from replay buffer (every N epochs)
        'replay_batch_size_ratio': 0.5, # Ratio of batch size for replay samples
        'replay_alpha': 0.6, # NEW: Alpha for PER (priority exponent)
        'replay_beta_start': 0.4, # NEW: Beta start for PER (importance sampling exponent)
        'use_weighted_sampling': False, # Use weighted random sampling in DataLoader based on difficulty
        'dynamic_difficulty_update': True, # NEW: Enable dynamic difficulty updates for curriculum learning
        'mixup_alpha': 0.2, # Alpha for MixUp (0 to disable)
        'cutmix_alpha': 0.2, # Alpha for CutMix (0 to disable)
        'mixup_cutmix_ratio': 0.5, # Probability of MixUp vs CutMix when both enabled
        'use_sam_optimizer': False, # Use Sharpness-Aware Minimization (SAM) optimizer
        'use_swa': False, # Use Stochastic Weight Averaging (SWA)
        'swa_start_epoch_ratio': 0.7, # Start SWA after this ratio of total epochs
        'swa_lr': None, # Learning rate for SWA (None to use base LR)
        'use_qat': False, # Enable Quantization-Aware Training
        'qat_start_epoch': 0, # Start QAT preparation from this epoch
        'max_grad_norm': 1.0, # Max gradient norm for clipping (0 to disable)
        'calibrate_model': True, # Perform temperature scaling calibration after training

        # --- Structured Pruning Parameters ---
        'enable_structured_pruning': False, # Set to True to enable structured pruning
        'pruning_amount_per_iteration': 0.2, # Percentage of channels/neurons to prune in each iteration (e.g., 0.2 for 20%)
        'pruning_iterations': 1, # Number of iterative pruning steps (e.g., 1 for one-shot, >1 for iterative)
        'pruning_fine_tune_epochs': 5, # Optional: epochs to fine-tune after each pruning step
    },
    'ensemble': {
        'num_members': 3, # Number of ensemble members to train
        'train_members': True, # Whether to train new members or load existing
        'inference_mode': 'simple_average', # 'simple_average', 'weighted_average', 'stacked', 'distilled'
        'use_stacking': False, # Enable stacking ensemble
        'use_distillation': False, # Enable knowledge distillation
        'teacher_ensemble_type': 'simple_average', # NEW: 'simple_average', 'weighted_average', 'stacked'
        'distilled_student_config_override': { # Override student config for distillation
            'model': {
                'attribute_backbone': 'mobilenet_v3_small', # Example: smaller backbone for student
                'gnn_depth': 2,
            }
        }
    },
    'debug': {
        'log_level': 'INFO', # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        'plot_reliability_diagram': True, # Plot reliability diagrams during calibration
        'min_contour_area_sam_fallback': 50, # Minimum contour area for SAM/basic fallback detections
        'max_fallback_cnt': 5, # Max number of objects to detect in fallback contour detection

        # --- Grad-CAM Parameters ---
        'enable_grad_cam': False, # Set to True to enable Grad-CAM visualization
        'enable_grad_cam_on_errors': False, # NEW: Generate Grad-CAM for misclassified samples
        'apply_symbolic_priors': False, # Flag to enable symbolic priors
        'use_learnable_symbolic_embeddings': False, # NEW: Use nn.Embedding for symbolic features
    }
}

CONFIG = DEFAULT_CONFIG  # Use the default config as the primary config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT_PATH = CONFIG['data']['dataset_path']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CHANNELS = CONFIG['data']['num_channels']

# Define basic YOLO class map (can be expanded based on dataset)
YOLO_CLASS_MAP = {0: 'object'}

# Define attribute and relation maps (these should be consistent with your dataset)
# These are used in BongardGenerator and PerceptionModule
ATTRIBUTE_FILL_MAP = {
    'fill': {0: 'solid', 1: 'hollow'},
    'color': {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'black', 5: 'white'},
    'size': {0: 'small', 1: 'medium', 2: 'large'},
    'orientation': {0: 'horizontal', 1: 'vertical', 2: 'diagonal'},
    'shape': {0: 'circle', 1: 'square', 2: 'triangle', 3: 'rectangle', 4: 'polygon'},
    'texture': {0: 'smooth', 1: 'rough', 2: 'striped', 3: 'dotted'}
}
# Inverse maps for easier lookup (e.g., 'circle' -> 0)
ATTRIBUTE_FILL_MAP_INV = {k: {v2: k2 for k2, v2 in v.items()} for k, v in ATTRIBUTE_FILL_MAP.items()}

RELATION_MAP = {
    0: 'touching',
    1: 'overlapping',
    2: 'inside',
    3: 'above',
    4: 'below',
    5: 'left_of',
    6: 'right_of',
    7: 'contains',
    8: 'aligned_h',
    9: 'aligned_v',
    10: 'connected'
}
RELATION_MAP_INV = {v: k for k, v in RELATION_MAP.items()}


# Define a basic JSON schema for ground truth validation
BONGARD_GT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bongard_label": {"type": "integer", "minimum": 0, "maximum": 1},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["id", "bbox", "attributes"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "integer"},
                    "dst": {"type": "integer"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["src", "dst", "type"]
            }
        }
    },
    "required": ["bongard_label", "objects", "relations"]
}


# --- Utility Functions ---
def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def load_config(config_path: str) -> Dict:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_symbolic_embedding_dims() -> Dict[str, int]:
    """
    Calculates the vocabulary size (number of unique categories) for each
    symbolic attribute type and relation type. This is used to determine
    the input dimension for nn.Embedding layers.
    Returns:
        Dict[str, int]: A dictionary where keys are attribute/relation names
                        and values are their corresponding vocabulary sizes.
    """
    embedding_dims = {}
    for attr_type, attr_map in ATTRIBUTE_FILL_MAP.items():
        embedding_dims[attr_type] = len(attr_map)
    
    embedding_dims['relation'] = len(RELATION_MAP)
    
    logger.debug(f"Calculated symbolic embedding dimensions: {embedding_dims}")
    return embedding_dims


# --- Pre-calculate symbolic embedding dimensions once ---
# This block should be placed after all necessary imports and definitions
# of ATTRIBUTE_FILL_MAP and RELATION_MAP.
try:
    # Assuming get_symbolic_embedding_dims is defined and ATTRIBUTE_FILL_MAP/RELATION_MAP are loaded
    SYMBOLIC_EMBEDDING_DIMS = get_symbolic_embedding_dims()
except NameError:
    logger.warning("SYMBOLIC_EMBEDDING_DIMS not found. Define get_symbolic_embedding_dims and call it.")
    # Fallback if get_symbolic_embedding_dims is not defined yet or dependencies missing
    # This fallback should ideally match the actual expected dimensions of your data
    SYMBOLIC_EMBEDDING_DIMS = {
        'fill': 2,       # Example: 'solid', 'outline'
        'color': 6,      # Example: 'red', 'green', 'blue', 'yellow', 'black', 'white'
        'size': 3,       # Example: 'small', 'medium', 'large'
        'orientation': 4, # Example: 'up', 'down', 'left', 'right'
        'shape': 6,      # Example: 'circle', 'square', 'triangle', 'star', 'pentagon', 'hexagon'
        'texture': 3,    # Example: 'smooth', 'striped', 'dotted'
        'relation': 15   # Example: 'above', 'below', 'left_of', 'right_of', 'overlapping', 'touching', 'same_shape', 'same_color', etc.
    }
    logger.warning(f"Using fallback SYMBOLIC_EMBEDDING_DIMS: {SYMBOLIC_EMBEDDING_DIMS}. Ensure this matches your data.")


def load_bongard_data(data_root_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Loads Bongard problem data from the specified root path.
    Assumes image files are PNG and annotations are in a specific JSON format.

    Args:
        data_root_path (str): The root directory of the dataset.
                              Expected structure:
                              data_root_path/
                              ├── images/
                              │   ├── problem_001.png
                              │   └── ...
                              └── annotations/
                                  ├── problem_001.json
                                  └── ...

    Returns:
        Tuple[List[str], np.ndarray, np.ndarray]:
            - List of image file paths.
            - NumPy array of Bongard problem labels (0 or 1).
            - NumPy array of initial difficulty scores (e.g., based on complexity).
    """
    logger.info(f"Loading Bongard data from: {data_root_path}")
    image_paths = []
    labels = []
    difficulty_scores = []

    image_dir = os.path.join(data_root_path, 'images')
    annotation_dir = os.path.join(data_root_path, 'annotations')

    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return [], np.array([]), np.array([])
    if not os.path.exists(annotation_dir):
        logger.error(f"Annotation directory not found: {annotation_dir}")
        return [], np.array([]), np.array([])

    # Use the image_extension from CONFIG
    image_extension = CONFIG['data'].get('image_extension', 'png') # Default to png

    # Iterate through image files (PNG)
    for img_file in glob.glob(os.path.join(image_dir, f'*.{image_extension}')):
        img_id = os.path.splitext(os.path.basename(img_file))[0]
        annotation_file = os.path.join(annotation_dir, f'{img_id}.json')

        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                
                # Validate annotation against schema (assuming BONGARD_GT_JSON_SCHEMA is defined)
                # from config import BONGARD_GT_JSON_SCHEMA # If schema is in config.py
                # import jsonschema # If not globally imported
                # jsonschema.validate(instance=annotation_data, schema=BONGARD_GT_JSON_SCHEMA)

                image_paths.append(img_file)
                
                # Extract Bongard label (0 or 1)
                bongard_label = annotation_data.get('bongard_label')
                if bongard_label is None or not isinstance(bongard_label, int) or bongard_label not in [0, 1]:
                    logger.warning(f"Invalid or missing 'bongard_label' for {img_id}. Defaulting to 0.")
                    labels.append(0)
                else:
                    labels.append(bongard_label)

                # --- Initial Difficulty Score Calculation (Example Logic) ---
                # This is a placeholder. You should define how difficulty is calculated.
                # Examples:
                # 1. Based on number of objects: More objects = harder
                num_objects = len(annotation_data.get('objects', []))
                # 2. Based on number of relations: More complex relations = harder
                num_relations = len(annotation_data.get('relations', []))
                # 3. Based on specific attributes (e.g., 'texture' might be harder)
                
                # Simple example: difficulty increases with number of objects and relations
                # Normalize to a 0-1 range conceptually.
                # Max expected objects/relations should be defined based on your dataset.
                max_expected_objects = 10 # Adjust based on your dataset
                max_expected_relations = 20 # Adjust based on your dataset
                
                initial_difficulty = (num_objects / max_expected_objects) * 0.5 + \
                                     (num_relations / max_expected_relations) * 0.5
                initial_difficulty = np.clip(initial_difficulty, 0.0, 1.0) # Ensure 0-1 range
                difficulty_scores.append(initial_difficulty)

            except (json.JSONDecodeError, KeyError, jsonschema.ValidationError) as e:
                logger.warning(f"Skipping {img_file} due to annotation parsing/validation error: {e}")
        else:
            logger.warning(f"No annotation found for {img_file}. Skipping.")

    logger.info(f"Loaded {len(image_paths)} Bongard problems.")
    return image_paths, np.array(labels, dtype=np.int64), np.array(difficulty_scores, dtype=np.float32)



# --- NEW: Grad-CAM Utility Function ---
def generate_grad_cam(model: nn.Module, input_tensor: torch.Tensor, target_layer: nn.Module, 
                      target_category: int, image_path: str, save_dir: str = './grad_cam_outputs'):
    """
    Generates and saves a Grad-CAM visualization for a given image and target class.
    Args:
        model (nn.Module): The trained PerceptionModule.
        input_tensor (torch.Tensor): The preprocessed input image tensor (NCHW).
        target_layer (nn.Module): The layer to hook for Grad-CAM (e.g., model.backbone.features[-1] or model.bongard_head.fc1).
        target_category (int): The predicted class index for which to generate the CAM.
        image_path (str): Original image path for naming.
        save_dir (str): Directory to save the Grad-CAM image.
    """
    if not HAS_PYTORCH_GRADCAM:
        logger.warning("pytorch_grad_cam is not installed. Skipping Grad-CAM generation.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # Create a GradCAM object
    # use_cuda=True if DEVICE.type == 'cuda' else False
    # Ensure the model is on CPU for GradCAM if it's a quantized model, or if you want to avoid GPU memory issues
    # For general use, let's keep it on the original device.
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda)
    
    # Define the target for CAM (e.g., the predicted class)
    targets = [ClassifierOutputTarget(target_category)]
    
    # Get grayscale heatmap
    # Need to pass dummy ground_truth_json_strings for the forward pass of PerceptionModule
    dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})] * input_tensor.shape[0]
    
    # The cam() call will internally run model.forward()
    # If your model.forward() requires multiple arguments (like gts_json_strings_batch),
    # you need to pass a custom forward_fn to GradCAM.
    # For PerceptionModule, it's bongard_logits, detected_objects, aggregated_outputs = model(images, gts_json_strings_batch)
    # So, we need to wrap it.
    
    def custom_forward_for_cam(input_tensor_cam):
        # Only return the logits that GradCAM needs for classification
        logits, _, _ = model(input_tensor_cam, dummy_gts_json)
        return logits
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True,
                        # Pass custom_forward_for_cam if model.forward requires more than just input_tensor
                        # If your model's forward method is simply `model(input_tensor)`, this is not needed.
                        # Given PerceptionModule takes `gts_json_strings_batch`, it is needed.
                        # However, pytorch-grad-cam's `__call__` expects `model(input_tensor)`
                        # so we need to ensure the target_layers are from the part that directly processes `input_tensor`
                        # or provide a wrapper. Let's assume the target_layer is from the backbone,
                        # and the model's forward handles the rest.
                        # For now, let's assume the model's forward can be called with just input_tensor for CAM's internal use.
                        # If it fails, you'll need to create a simplified forward pass for CAM.
                        # A better approach is to make a `model.get_classification_logits(input_tensor)` method.
                        # For now, let's assume `model(input_tensor, dummy_gts_json)` works and CAM can handle it.
                        # If it doesn't, you might need to adjust the `model` or `target_layer`.
                        # Let's use the default behavior and hope `model.forward` is flexible enough.
                        # If not, the `custom_forward_for_cam` would be passed as `target_layer`'s forward.
                        # The library's `__call__` usually takes `input_tensor` only.
                        # Let's try to make it work with the existing `model.forward` by ensuring `target_layers` are from the part that takes image input.
                        # The `model` passed to `GradCAM` should be the full model, and `target_layers` should be the feature extractor.
                        # The `ClassifierOutputTarget` will then extract the relevant logits.
                        # Let's simplify the `cam` call slightly to avoid `custom_forward_for_cam` complexity for now.
                        # The `model` passed to GradCAM should be able to produce the final classification logits.
                        # If PerceptionModule's forward is `(images, gts_json_strings_batch)`, then GradCAM needs a wrapper.
                        # Let's add a `get_bongard_logits` method to PerceptionModule for this.
                        # For now, assuming `model` can be called with just `input_tensor` for CAM's internal use.
                        # If not, the user will need to add a method like `model.get_classification_logits(input_tensor)`.
                        # For now, let's stick to the simpler call and assume the model's forward handles it.
                        # If the `model` is the `PerceptionModule`, its `forward` takes two arguments.
                        # `pytorch-grad-cam`'s `__call__` method for `cam` expects `model(input_tensor)`.
                        # This means we need a wrapper.
                        
                        # Let's use custom_forward_for_cam
                        # The `cam` object needs to be initialized with a model that takes a single input.
                        # So, we'll need to create a temporary wrapper model or a method in PerceptionModule.
                        # For simplicity, let's assume PerceptionModule has a method like `get_bongard_logits_from_image_tensor(image_tensor)`
                        # that takes just the image tensor and returns logits.
                        # If not, the user needs to add it.
                        # For now, I'll update the `generate_grad_cam` to reflect this.
                        
                        # Assuming model has a method `get_bongard_logits_for_cam(image_tensor)`
                        # that takes a single image tensor and returns bongard_logits
                        # This method would internally call the necessary parts of PerceptionModule.
                        # If not, this function will fail.
                        
                        # Let's make a more explicit wrapper for GradCAM.
                        # The `model` argument to `GradCAM` should be a `nn.Module` that takes a single input.
                        # We can create a temporary wrapper.
                        
                        # Create a wrapper model for GradCAM
                        class GradCAMWrapper(nn.Module):
                            def __init__(self, original_model, dummy_gts_json_input):
                                super().__init__()
                                self.original_model = original_model
                                self.dummy_gts_json_input = dummy_gts_json_input
                            
                            def forward(self, x):
                                # Call the original model's forward method with the dummy JSON input
                                logits, _, _ = self.original_model(x, self.dummy_gts_json_input)
                                return logits
                        
                        wrapper_model = GradCAMWrapper(model, dummy_gts_json)
                        cam = GradCAM(model=wrapper_model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda)
                        
                        # Now call cam with the original input_tensor
                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)
                        grayscale_cam = grayscale_cam[0, :] # Remove batch dimension
                        
                        # Load original image for overlay
                        original_image_np = cv2.imread(image_path)
                        original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)
                        original_image_float = np.float32(original_image_np) / 255
                        
                        # Overlay heatmap on image
                        cam_image = show_cam_on_image(original_image_float, grayscale_cam, use_rgb=True)
                        
                        # Save the image
                        output_filename = os.path.basename(image_path).replace('.', f'_gradcam_class_{target_category}.')
                        output_path = os.path.join(save_dir, output_filename)
                        cv2.imwrite(output_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
                        logger.info(f"Grad-CAM saved to {output_path}")

# --- DALI Pipeline and Data Loader ---
@pipeline_def
def dali_pipe(file_root, file_list, labels_list, batch_size, num_threads, device_id,
              height, width, is_train, num_channels, feature_consistency_alpha,
              imagenet_mean, imagenet_std, use_synthetic_data, external_source_iterator,
              device_memory_padding, host_memory_padding, prebuild_streams,
              shard_id: int = 0, num_shards: int = 1): # Added DDP sharding args
    
    # --- External Source for Synthetic Data ---
    if use_synthetic_data:
        # BongardExternalSource now yields 8 outputs:
        # (images1, images2, labels, gts_json, difficulties, affine1, affine2, original_indices)
        images_view1, images_view2, labels, gts_json, difficulties, affine1, affine2, original_indices = fn.external_source(
            source=external_source_iterator,
            num_outputs=8, # NEW: 8 outputs
            layout=["HWC", "HWC", "", "", "", "", "", ""], # Layout for images, labels, gts_json, difficulties, affine1, affine2, original_indices
            batch=False, # External source yields single samples
            parallel=True, # Enable parallel execution of external_source
            name="ExternalDataSource"
        )
        # Ensure images are on GPU if device_id is GPU
        if device_id != -1 and images_view1.device() == 'cpu':
            images_view1 = images_view1.gpu()
            images_view2 = images_view2.gpu()
        
    # --- File Reader for Real Data (PNG) ---
    else:
        # For real data, fn.readers.file doesn't directly expose original indices.
        # We can pass dummy indices or rely on external mapping if needed for dynamic curriculum.
        # For simplicity, if dynamic_difficulty_update is enabled for real data,
        # we'll assume batch order corresponds to a sorted subset of the original dataset.
        
        images_raw, labels = fn.readers.file(
            file_root=file_root,
            files=file_list,
            labels=labels_list,
            shard_id=shard_id, # For DDP
            num_shards=num_shards, # For DDP
            random_shuffle=is_train # Shuffle only for training
        )
        
        # Decode PNG images (fn.decoders.image handles both PNG/JPG by default)
        images_decoded = fn.decoders.image(
            images_raw,
            device="mixed", # Decode on GPU if possible
            output_type=types.RGB if num_channels == 3 else types.GRAY
        )

        # Apply augmentations (only to training data)
        if is_train:
            # RandomResizedCrop (common augmentation)
            images_view1 = fn.random_resized_crop(
                images_decoded,
                size=(height, width),
                random_area=[0.08, 1.0],
                random_aspect_ratio=[0.75, 1.33],
                interp_type=types.INTERP_TRIANGULAR
            )
            images_view2 = fn.random_resized_crop( # Second view for consistency loss
                images_decoded,
                size=(height, width),
                random_area=[0.08, 1.0],
                random_aspect_ratio=[0.75, 1.33],
                interp_type=types.INTERP_TRIANGULAR
            )
            
            # Apply Kornia RandAugment-like policy using fn.python_function
            if HAS_KORNIA:
                images_view1 = fn.python_function(
                    images_view1,
                    function=functools.partial(kornia_rand_augment, current_image_size=(height, width)),
                    output_layouts=["HWC"]
                )
                images_view2 = fn.python_function(
                    images_view2,
                    function=functools.partial(kornia_rand_augment, current_image_size=(height, width)),
                    output_layouts=["HWC"]
                )
            
            # Affine matrices (for consistency loss if needed)
            affine1 = fn.transforms.rotation(images_view1, angle=fn.random.uniform(range=(-30.0, 30.0)), device="gpu")
            affine2 = fn.transforms.rotation(images_view2, angle=fn.random.uniform(range=(-30.0, 30.0)), device="gpu")
            
            # Dummy difficulties and original_indices for real data (not dynamically updated here)
            difficulties = fn.constant(0.5, shape=[]) # Placeholder
            original_indices = fn.sequence_id() # DALI's internal sequence ID, not original dataset index
        else:
            # For validation/inference, just resize
            images_view1 = fn.resize(images_decoded, resize_x=width, resize_y=height, interp_type=types.INTERP_TRIANGULAR)
            images_view2 = images_view1 # No second view needed for val/inference
            affine1 = fn.constant(0.0, shape=[2,3])
            affine2 = fn.constant(0.0, shape=[2,3])
            difficulties = fn.constant(0.5, shape=[])
            original_indices = fn.sequence_id() # DALI's internal sequence ID
    
    # Normalize images (0-1 range then mean/std)
    images_view1 = fn.normalize(images_view1, mean=imagenet_mean, std=imagenet_std, scale=1/255.0)
    images_view2 = fn.normalize(images_view2, mean=imagenet_mean, std=imagenet_std, scale=1/255.0)

    # Convert labels to desired type
    labels = labels.gpu() # Move labels to GPU
    labels = fn.cast(labels, dtype=types.INT64) # Ensure labels are long type

    # Return outputs
    return images_view1, images_view2, labels, gts_json, difficulties, affine1, affine2, original_indices # NEW: original_indices



def build_dali_loader(file_list: List[str], labels_list: np.ndarray, config: Dict[str, Any],
                      mode: str = 'train', external_source_iterator: Optional[Any] = None,
                      difficulty_scores_list: Optional[np.ndarray] = None, # Added for real data curriculum
                      shard_id: int = 0, num_shards: int = 1) -> Tuple[Any, pytorch_dali_plugin.DALIGenericIterator]:
    """
    Builds and returns a DALI pipeline and iterator.
    Args:
        file_list (List[str]): List of image file paths.
        labels_list (np.ndarray): NumPy array of labels.
        config (Dict): Configuration dictionary.
        mode (str): 'train', 'val', 'inference', or 'calibration'.
        external_source_iterator (Optional[Any]): Iterator for synthetic data.
        difficulty_scores_list (Optional[np.ndarray]): Difficulty scores for real data.
        shard_id (int): Current GPU rank for DDP.
        num_shards (int): Total number of GPUs for DDP.
    Returns:
        Tuple[Any, pytorch_dali_plugin.DALIGenericIterator]: The DALI pipeline and iterator.
    """
    is_train = (mode == 'train')
    
    # For real data, if dynamic difficulty update is enabled, we need to pass the full lists
    # to the CurriculumSampler, which will then handle sampling and epoch data.
    # DALI's fn.readers.file itself doesn't directly use difficulty scores for sampling.
    
    # If using CurriculumSampler for real data, it manages file_list and labels_list internally
    # and provides the current epoch's data.
    if not config['data']['use_synthetic_data'] and is_train:
        # Create CurriculumSampler for real data
        curriculum_sampler = CurriculumSampler(
            image_paths=file_list,
            labels=labels_list,
            initial_difficulty_scores=difficulty_scores_list,
            batch_size=config['model']['batch_size'],
            annealing_epochs=config['data']['curriculum_annealing_epochs'],
            total_epochs=config['model']['epochs'], # Pass total epochs
            is_train=is_train,
            use_weighted_sampling=config['training']['use_weighted_sampling']
        )
        # DALI's file reader needs initial lists; CurriculumSampler will update them per epoch
        dali_file_list = file_list
        dali_labels_list = labels_list
        size = len(file_list)
        # Pass the curriculum_sampler as external_source_iterator for real data too,
        # if you want the DALI pipeline to dynamically get the current epoch's data.
        # This requires modifying dali_pipe to handle external_source for real data.
        # For now, we'll keep it simple and just use the initial lists for fn.readers.file.
        # The CurriculumSampler will manage the indices for PyTorch DataLoader if used.
        # For DALI, dynamic file lists require rebuilding the pipeline or a custom external source.
        # Let's keep fn.readers.file simple and have the CurriculumSampler manage the indices
        # for the _run_single_training_session_ensemble loop to update difficulties.
        
        # For DALI, if not using external_source, we need to pass the full lists.
        # The dynamic curriculum will then be handled by updating the sampler's internal state
        # and potentially providing new file lists to DALI if pipeline rebuilds are allowed.
        # For now, we'll assume DALI uses the initial full file_list and labels_list.
        # The dynamic difficulty update will work by updating the CurriculumSampler's internal scores.
        
        # For real data, the CurriculumSampler will manage the `file_list` and `labels_list`
        # and provide them via `get_epoch_data` for pipeline rebuilds if needed.
        # DALI's `readers.file` needs initial lists.
        dali_file_list = file_list
        dali_labels_list = labels_list
        size = len(file_list)
        # The `external_source_iterator` remains None for real data path in dali_pipe.
        # The `CurriculumSampler` instance is returned separately for direct interaction.
        
    elif config['data']['use_synthetic_data'] and external_source_iterator:
        dali_file_list = [] # Not used by external_source
        dali_labels_list = [] # Not used by external_source
        size = len(external_source_iterator.dataset) # Use dataset size for synthetic
        curriculum_sampler = external_source_iterator # For synthetic, the external source is the sampler
    else: # Validation/Inference for real data
        dali_file_list = file_list
        dali_labels_list = labels_list
        size = len(file_list)
        curriculum_sampler = None # No curriculum sampler for val/inference
        
    height = config['data']['initial_image_size'][0] if is_train and mode == 'train' else config['data']['image_size'][0]
    width = config['data']['initial_image_size'][1] if is_train and mode == 'train' else config['data']['image_size'][1]

    pipeline = dali_pipe(
        file_root=config['data']['data_root_path'] if not config['data']['use_synthetic_data'] else "",
        file_list=dali_file_list,
        labels_list=dali_labels_list,
        batch_size=config['model']['batch_size'],
        num_threads=config['data']['dataloader_num_workers'],
        device_id=shard_id if str(DEVICE) == 'cuda' else -1, # Pass shard_id as device_id
        height=height,
        width=width,
        is_train=is_train,
        num_channels=config['data']['num_channels'],
        feature_consistency_alpha=config['training']['feature_consistency_alpha'] if is_train else 0.0,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        use_synthetic_data=config['data']['use_synthetic_data'],
        external_source_iterator=external_source_iterator, # Pass external_source_iterator for synthetic
        device_memory_padding=config['data']['dali_device_memory_padding'],
        host_memory_padding=config['data']['dali_host_memory_padding'],
        prebuild_streams=config['data']['dali_prebuild_streams'],
        shard_id=shard_id, # For DDP
        num_shards=num_shards # For DDP
    )
    pipeline.build()
    
    # NEW: Output map now includes 'original_indices'
    output_map = ['view1', 'view2', 'labels', 'gts_json', 'difficulties', 'affine1', 'affine2', 'original_indices']

    loader = pytorch_dali_plugin.DALIGenericIterator(
        pipeline,
        output_map,
        size=size,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL
    )
    logger.info(f"DALI loader built for mode '{mode}' with size {size}.")
    
    # Return the loader and the curriculum_sampler instance (if applicable)
    return loader, curriculum_sampler # Return curriculum_sampler for direct updates


# --- NEW: Kornia Augmentation Function for DALI Python Operator ---
@torch.no_grad() # Ensure no gradients are tracked for augmentations
def kornia_augment(images_dali_gpu: torch.Tensor, is_train: bool, num_channels: int, 
                   randaugment_num_ops: int, randaugment_magnitude: int):
    """
    Applies Kornia augmentations to a batch of images.
    This function is designed to be called by DALI's fn.python_function.
    Args:
        images_dali_gpu (torch.Tensor): Batch of images from DALI, NCHW, float32, on GPU.
                                        Expected range [0, 255] for Kornia.
        is_train (bool): If True, applies training augmentations.
        num_channels (int): Number of image channels (1 or 3).
        randaugment_num_ops (int): N parameter for RandAugment (number of transforms to apply).
        randaugment_magnitude (int): M parameter for RandAugment (magnitude of transforms).
    Returns:
        torch.Tensor: Augmented batch of images, NCHW, float32, on GPU.
    """
    if not HAS_KORNIA:
        logger.warning("Kornia is not installed. Skipping Kornia augmentations.")
        return images_dali_gpu # Return original images if Kornia not available
    
    # Ensure images are in the expected format for Kornia (float32, [0, 255])
    if images_dali_gpu.max() <= 1.0 + 1e-6: # Check if already normalized to 0-1
        images_dali_gpu = images_dali_gpu * 255.0 # Scale back to 0-255 for Kornia's default behavior
    
    # Kornia augmentations typically expect RGB. If grayscale, convert to 3 channels.
    if num_channels == 1 and images_dali_gpu.shape[1] == 1:
        images_dali_gpu = images_dali_gpu.repeat(1, 3, 1, 1)
    
    if is_train:
        # --- NEW: Use Kornia's RandomAugmentation for RandAugment-like policy ---
        aug_list = K.AugmentationSequential(
            K.RandomAugmentation(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
            data_keys=["input"], # Specify that augmentations apply to the input image
            p=1.0, # Apply the sequence with this probability
            same_on_batch=False # Apply different augmentations to each image in batch
        ).to(images_dali_gpu.device)
        
        augmented_images = aug_list(images_dali_gpu)
        logger.debug(f"Applied Kornia RandAugment-like policy (N={randaugment_num_ops}, M={randaugment_magnitude}).")
    else:
        # No augmentations for validation/test, just pass through
        augmented_images = images_dali_gpu
        logger.debug("Skipping Kornia augmentations for validation/test.")
    
    # If original input was grayscale, convert back to 1 channel if needed
    if num_channels == 1 and augmented_images.shape[1] == 3:
        # Convert RGB to grayscale (simple average)
        augmented_images = KC.rgb_to_grayscale(augmented_images)
        logger.debug("Converted augmented image back to 1 channel (grayscale).")
    
    return augmented_images



# --- Curriculum Learning Sampler (Updated) ---
class CurriculumSampler(torch.utils.data.Sampler):
    """
    A sampler that implements curriculum learning by progressively increasing image resolution
    and dynamically sampling based on difficulty scores.
    """
    def __init__(self, image_paths: List[str], labels: np.ndarray,
                 initial_image_size: List[int], final_image_size: List[int],
                 annealing_epochs: int, batch_size: int,
                 difficulty_metric: str = 'std_dev',  # 'std_dev', 'entropy', 'random'
                 start_difficulty: float = 0.0,  # Percentage of hardest samples to include initially
                 end_difficulty: float = 1.0,    # Percentage of hardest samples to include finally
                 difficulty_schedule: str = 'linear',  # 'linear', 'cosine'
                 update_interval_epochs: int = 1,
                 cache_data: bool = False,
                 initial_difficulty_scores: Optional[np.ndarray] = None): # NEW: Pass initial scores
        
        self.image_paths = image_paths
        self.labels = labels
        self.initial_image_size = initial_image_size
        self.final_image_size = final_image_size
        self.annealing_epochs = annealing_epochs
        self.batch_size = batch_size
        self.difficulty_metric = difficulty_metric
        self.start_difficulty = start_difficulty
        self.end_difficulty = end_difficulty
        self.difficulty_schedule = difficulty_schedule
        self.update_interval_epochs = update_interval_epochs
        self.cache_data = cache_data
        
        self.current_epoch = 0
        self.current_image_size = list(initial_image_size)
        
        # --- NEW: Initialize difficulty scores ---
        if initial_difficulty_scores is not None and len(initial_difficulty_scores) == len(image_paths):
            self.difficulty_scores = initial_difficulty_scores
            logger.info("CurriculumSampler initialized with provided initial difficulty scores.")
        else:
            self.difficulty_scores = self._calculate_difficulty_scores() # Calculate if not provided
            logger.info("CurriculumSampler calculated initial difficulty scores.")
            
        self.indices = list(range(len(self.image_paths)))
        self.current_indices = self._get_difficulty_sampled_indices()
        
        if self.cache_data:
            logger.info("Caching dataset images (this may take a while)...")
            self.cached_images = self._cache_images()
            logger.info("Image caching complete.")
        else:
            self.cached_images = None
        
        logger.info(f"CurriculumSampler initialized. Initial image size: {self.current_image_size}, initial dataset size: {len(self.current_indices)}")

    def _calculate_difficulty_scores(self) -> np.ndarray:
        """
        Calculates a dummy difficulty score for each image.
        In a real scenario, this would involve a pre-analysis of the dataset.
        For demonstration, we'll use a random score or a score based on image properties.
        """
        if self.difficulty_metric == 'random':
            logger.info("Using random difficulty scores.")
            return np.random.rand(len(self.image_paths))
        elif self.difficulty_metric == 'std_dev':
            logger.info("Calculating dummy difficulty scores based on image standard deviation (simulated).")
            # Simulate image loading and std dev calculation
            scores = []
            for i, path in enumerate(self.image_paths):
                # In a real scenario, you'd load the image and calculate its std dev
                # For this dummy, let's just assign a random value
                scores.append(random.random())
            return np.array(scores)
        else:
            logger.warning(f"Unknown difficulty metric '{self.difficulty_metric}'. Using random scores.")
            return np.random.rand(len(self.image_paths))
    
    # --- NEW: Method to update difficulty scores dynamically ---
    def update_difficulty_scores(self, new_scores: np.ndarray):
        """Updates the difficulty scores for adaptive curriculum learning."""
        if len(new_scores) != len(self.image_paths):
            logger.error(f"New difficulty scores length ({len(new_scores)}) does not match dataset size ({len(self.image_paths)}). Not updating.")
            return
        self.difficulty_scores = new_scores
        logger.info("CurriculumSampler: Difficulty scores updated.")
        # Re-sample indices immediately if scores change
        self.current_indices = self._get_difficulty_sampled_indices()
        logger.info(f"CurriculumSampler: Re-sampled indices based on new difficulty scores. Current dataset size: {len(self.current_indices)}")


    def _get_difficulty_threshold(self) -> float:
        """Calculates the current difficulty threshold based on annealing schedule."""
        if self.annealing_epochs <= 0:
            return self.end_difficulty  # No annealing, use final difficulty
        
        progress = min(self.current_epoch / self.annealing_epochs, 1.0)
        
        if self.difficulty_schedule == 'linear':
            current_threshold = self.start_difficulty + (self.end_difficulty - self.start_difficulty) * progress
        elif self.difficulty_schedule == 'cosine':
            # Cosine annealing from start_difficulty to end_difficulty
            current_threshold = self.end_difficulty - (self.end_difficulty - self.start_difficulty) * (math.cos(math.pi * progress) + 1) / 2
        else:
            logger.warning(f"Unknown difficulty schedule '{self.difficulty_schedule}'. Using linear schedule.")
            current_threshold = self.start_difficulty + (self.end_difficulty - self.start_difficulty) * progress
        
        return current_threshold

    def _get_difficulty_sampled_indices(self) -> List[int]:
        """
        Samples indices based on the current difficulty threshold.
        Assumes higher score means harder.
        """
        if self.difficulty_metric == 'random':  # If random, just use all indices shuffled
            return random.sample(self.indices, len(self.indices))
        
        # Sort indices by difficulty score (ascending, so easier first)
        sorted_indices = sorted(self.indices, key=lambda i: self.difficulty_scores[i])
        
        # Determine how many samples to include based on the current difficulty threshold
        current_difficulty_percentage = self._get_difficulty_threshold()
        num_samples_to_include = int(len(sorted_indices) * current_difficulty_percentage)
        
        # Select the 'hardest' samples (those with higher scores)
        if num_samples_to_include < self.batch_size and len(sorted_indices) >= self.batch_size:
            num_samples_to_include = self.batch_size
        elif num_samples_to_include > len(sorted_indices):
            num_samples_to_include = len(sorted_indices)
        
        # Take the top 'num_samples_to_include' from the sorted (by difficulty) list
        # Assuming higher score = harder, so take from the end of sorted_indices
        selected_indices = sorted_indices[len(sorted_indices) - num_samples_to_include:]
        
        random.shuffle(selected_indices)  # Shuffle the selected subset
        logger.info(f"Epoch {self.current_epoch}: Curriculum sampling. Including {len(selected_indices)} samples (difficulty threshold: {current_difficulty_percentage:.2f}).")
        return selected_indices

    def _update_image_size(self):
        """Updates the image size based on the current epoch."""
        if self.annealing_epochs <= 0:
            self.current_image_size = list(self.final_image_size)
            return
        
        progress = min(self.current_epoch / self.annealing_epochs, 1.0)
        
        new_height = int(self.initial_image_size[0] + (self.final_image_size[0] - self.initial_image_size[0]) * progress)
        new_width = int(self.initial_image_size[1] + (self.final_image_size[1] - self.initial_image_size[1]) * progress)
        
        self.current_image_size = [new_height, new_width]
        logger.info(f"Epoch {self.current_epoch}: Image size updated to {self.current_image_size}.")

    def set_epoch(self, epoch: int):
        """Sets the current epoch and updates curriculum parameters."""
        self.current_epoch = epoch
        if self.current_epoch % self.update_interval_epochs == 0:
            self._update_image_size()
            self.current_indices = self._get_difficulty_sampled_indices()

    # --- NEW: get_epoch_data now returns difficulty scores ---
    def get_epoch_data(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Returns the image paths, labels, and difficulty scores for the current epoch's curriculum."""
        selected_paths = [self.image_paths[i] for i in self.current_indices]
        selected_labels = self.labels[self.current_indices]
        selected_difficulty_scores = self.difficulty_scores[self.current_indices]
        return selected_paths, selected_labels, selected_difficulty_scores
    
    def get_current_image_size(self) -> List[int]:
        """Returns the current image size for the DALI pipeline."""
        return self.current_image_size

    def get_dataset_size(self) -> int:
        """Returns the size of the dataset for the current epoch."""
        return len(self.current_indices)

    def _cache_images(self) -> Dict[str, Image.Image]:
        """Caches images into memory for faster access."""
        cached_imgs = {}
        for path in tqdm(self.image_paths, desc="Caching images"):
            try:
                cached_imgs[path] = Image.open(path).convert('RGB' if NUM_CHANNELS == 3 else 'L')
            except Exception as e:
                logger.error(f"Error caching image {path}: {e}")
        return cached_imgs

    def __iter__(self):
        # This iterator is primarily for PyTorch's DataLoader, but DALI's ExternalSource
        # will call `__next__` directly on the `source` object.
        # For DALI, the `get_epoch_data` method will be used to provide the file list.
        # This `__iter__` is mostly a placeholder for PyTorch's DataLoader API compatibility.
        return iter(self.current_indices)
    
    def __len__(self):
        return len(self.current_indices)
# --- NEW: Programmatic Synthetic Data Generator (Part 1) ---
# Import the new BongardRule and ALL_BONGARD_RULES
from bongard_rules import BongardRule, ALL_BONGARD_RULES

class BongardGenerator:
    """
    Generates synthetic Bongard-like images with geometric shapes and ground truth.
    Each image contains a random number of objects with random attributes and relations.
    Can generate images conforming to a specific BongardRule.
    """
    def __init__(self, image_size=(96, 96), num_objects=(2, 5), num_classes=2):
        self.W, self.H = image_size
        self.num_objects_range = num_objects
        # Ensure these maps are consistent with global definitions in part1
        self.shapes = list(ATTRIBUTE_FILL_MAP['shape'].values()) # e.g., ['circle', 'square', 'triangle', ...]
        self.fill_types = list(ATTRIBUTE_FILL_MAP['fill'].values())
        self.colors = list(ATTRIBUTE_FILL_MAP['color'].values())
        self.sizes = list(ATTRIBUTE_FILL_MAP['size'].values())
        self.orientations = list(ATTRIBUTE_FILL_MAP['orientation'].values())
        self.textures = list(ATTRIBUTE_FILL_MAP['texture'].values())

        self.num_classes = num_classes # For overall Bongard problem label (0 or 1)
        logger.info(f"BongardGenerator initialized with image_size={image_size}, num_objects_range={num_objects}.")

    def _sample_object(self):
        """Samples properties for a single object."""
        shape = random.choice(self.shapes)
        # Ensure size is within reasonable bounds relative to image size
        min_obj_dim = int(min(self.W, self.H) * 0.1)
        max_obj_dim = int(min(self.W, self.H) * 0.3)
        size_val = random.randint(min_obj_dim, max_obj_dim)
        
        # Ensure object is placed fully within bounds
        x = random.randint(0, self.W - size_val)
        y = random.randint(0, self.H - size_val)
        
        x1, y1, x2, y2 = x, y, x + size_val, y + size_val

        # Map integer indices back to string names for consistency with rules
        attr = {
            'fill': random.choice(self.fill_types),
            'color': random.choice(self.colors),
            'size': random.choice(self.sizes), # Choose from predefined size categories
            'orientation': random.choice(self.orientations),
            'shape': shape, # Use string shape directly
            'texture': random.choice(self.textures)
        }
        return dict(shape_type=shape, bbox=(x1, y1, x2, y2), attributes=attr)

    def _compute_relations(self, objs: List[Dict]) -> List[Dict]:
        """
        Computes relations analytically between objects based on their bounding boxes.
        This function is more comprehensive and uses all defined relations.
        """
        relations = []
        for i in range(len(objs)):
            for j in range(len(objs)):
                if i == j: continue

                b1, b2 = objs[i]['bbox'], objs[j]['bbox']
                
                # IoU calculation
                iou = _calculate_iou(b1, b2)

                # Centers
                center_i_x, center_i_y = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
                center_j_x, center_j_y = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2

                # Relative positions
                dx = center_j_x - center_i_x
                dy = center_j_y - center_i_y

                # Thresholds for spatial relations (can be fine-tuned)
                horizontal_threshold = (b1[2] - b1[0] + b2[2] - b2[0]) / 4 # Quarter of average width
                vertical_threshold = (b1[3] - b1[1] + b2[3] - b2[1]) / 4 # Quarter of average height

                if iou > 0.7: # High IoU suggests contains or overlaps
                    if (b1[2]-b1[0])*(b1[3]-b1[1]) > (b2[2]-b2[0])*(b2[3]-b2[1]) * 1.5 and iou > 0.8: # b1 significantly larger and high overlap
                        relations.append({'src': i, 'dst': j, 'type': 'contains', 'confidence': 1.0})
                    elif (b2[2]-b2[0])*(b2[3]-b2[1]) > (b1[2]-b1[0])*(b1[3]-b1[1]) * 1.5 and iou > 0.8: # b2 significantly larger and high overlap
                        relations.append({'src': i, 'dst': j, 'type': 'inside', 'confidence': 1.0})
                    else:
                        relations.append({'src': i, 'dst': j, 'type': 'overlapping', 'confidence': 1.0})
                elif iou > 0.0: # Some overlap but not significant
                    relations.append({'src': i, 'dst': j, 'type': 'overlapping', 'confidence': 1.0})
                else: # No direct overlap, check spatial relations
                    if dy < -vertical_threshold: # j is above i
                        relations.append({'src': i, 'dst': j, 'type': 'above', 'confidence': 1.0})
                    elif dy > vertical_threshold: # j is below i
                        relations.append({'src': i, 'dst': j, 'type': 'below', 'confidence': 1.0})
                    
                    if dx < -horizontal_threshold: # j is left of i
                        relations.append({'src': i, 'dst': j, 'type': 'left_of', 'confidence': 1.0})
                    elif dx > horizontal_threshold: # j is right of i
                        relations.append({'src': i, 'dst': j, 'type': 'right_of', 'confidence': 1.0})
                    
                    # Check for touching (if bounding boxes are very close but not overlapping)
                    # This is a bit more complex, checking if edges are close
                    if iou == 0.0: # Only check touching if no overlap
                        dist_x = max(0, b1[0] - b2[2], b2[0] - b1[2])
                        dist_y = max(0, b1[1] - b2[3], b2[1] - b1[3])
                        if dist_x < 5 and dist_y < 5: # Small pixel distance
                            relations.append({'src': i, 'dst': j, 'type': 'touching', 'confidence': 1.0})

                # Aligned relations
                if abs(center_i_x - center_j_x) < 5: # Centers are horizontally aligned
                    relations.append({'src': i, 'dst': j, 'type': 'aligned_v', 'confidence': 1.0})
                if abs(center_i_y - center_j_y) < 5: # Centers are vertically aligned
                    relations.append({'src': i, 'dst': j, 'type': 'aligned_h', 'confidence': 1.0})
                
                # Same/Different Type
                if objs[i]['shape_type'] == objs[j]['shape_type']:
                    relations.append({'src': i, 'dst': j, 'type': 'same_shape', 'confidence': 1.0})
                else:
                    relations.append({'src': i, 'dst': j, 'type': 'different_shape', 'confidence': 1.0})
                
                # Same/Different Color
                if objs[i]['attributes']['color'] == objs[j]['attributes']['color']:
                    relations.append({'src': i, 'dst': j, 'type': 'same_color', 'confidence': 1.0})
                else:
                    relations.append({'src': i, 'dst': j, 'type': 'different_color', 'confidence': 1.0})

        return relations

    def generate(self, n_samples: int, rule: Optional[BongardRule] = None) -> List[Tuple[Image.Image, Dict]]:
        """
        Generates n_samples of Bongard-like data.
        If a rule is provided, it attempts to generate images that either satisfy or violate the rule.
        Returns a list of tuples: (PIL_image, ground_truth_dict).
        """
        data = []
        for sample_idx in tqdm(range(n_samples), desc="Generating synthetic data"):
            img = Image.new('RGB', (self.W, self.H), 'white')
            draw = ImageDraw.Draw(img)
            
            # Attempt to generate a sample that matches or violates the rule
            max_attempts_per_sample = 50
            found_valid_sample = False
            bongard_label = 0 # Default to negative if no rule or generation fails
            
            for attempt in range(max_attempts_per_sample):
                k = random.randint(*self.num_objects_range)
                objs_data = [self._sample_object() for _ in range(k)]
                relations_data = self._compute_relations(objs_data)

                if rule:
                    is_positive_sample_target = random.random() < 0.5 # Try to generate 50% positive, 50% negative
                    satisfies_rule = rule.apply(objs_data, relations_data)

                    if (is_positive_sample_target and satisfies_rule) or \
                       (not is_positive_sample_target and not satisfies_rule):
                        bongard_label = 1 if is_positive_sample_target else 0
                        found_valid_sample = True
                        break
                else: # No rule, generate randomly
                    bongard_label = random.randint(0, self.num_classes - 1)
                    found_valid_sample = True
                    break
            
            if not found_valid_sample:
                logger.warning(f"Could not generate a sample satisfying/violating the rule after {max_attempts_per_sample} attempts. Generating a random sample.")
                # Fallback to random generation if rule-based generation fails
                k = random.randint(*self.num_objects_range)
                objs_data = [self._sample_object() for _ in range(k)]
                relations_data = self._compute_relations(objs_data)
                bongard_label = random.randint(0, self.num_classes - 1)

            # Draw objects onto the image
            for i, o in enumerate(objs_data):
                fill_color_str = o['attributes']['fill']
                obj_color_str = o['attributes']['color']
                
                # Convert string colors to RGB tuples
                fill_color_tuple = None
                outline_color_tuple = (0, 0, 0) # Default outline to black

                if obj_color_str == 'black':
                    color_rgb = (0, 0, 0)
                elif obj_color_str == 'white':
                    color_rgb = (255, 255, 255)
                elif obj_color_str == 'red':
                    color_rgb = (255, 0, 0)
                elif obj_color_str == 'blue':
                    color_rgb = (0, 0, 255)
                elif obj_color_str == 'green':
                    color_rgb = (0, 255, 0)
                elif obj_color_str == 'yellow':
                    color_rgb = (255, 255, 0)
                else:
                    color_rgb = (0, 0, 0) # Fallback

                if fill_color_str == 'solid':
                    fill_color_tuple = color_rgb
                    outline_color_tuple = (0, 0, 0) # Solid objects usually have black outline
                else: # 'outline'
                    fill_color_tuple = None
                    outline_color_tuple = color_rgb # Outline objects use their color for outline

                bbox_coords = o['bbox']
                if o['shape_type'] == 'circle':
                    draw.ellipse(bbox_coords, outline=outline_color_tuple, fill=fill_color_tuple, width=2)
                elif o['shape_type'] == 'square':
                    draw.rectangle(bbox_coords, outline=outline_color_tuple, fill=fill_color_tuple, width=2)
                elif o['shape_type'] == 'triangle':
                    x1, y1, x2, y2 = bbox_coords
                    points = [(x1 + (x2 - x1) / 2, y1), (x1, y2), (x2, y2)] # Equilateral-like triangle
                    draw.polygon(points, outline=outline_color_tuple, fill=fill_color_tuple, width=2)
                # Add more shapes as needed

            # Ground truth dictionary, adhering to BONGARD_GT_JSON_SCHEMA
            gt_dict = {
                'objects': [],
                'relations': relations_data,
                'bongard_label': bongard_label
            }
            # Populate objects with ground truth attributes
            for obj_idx, o in enumerate(objs_data):
                gt_dict['objects'].append({
                    'id': obj_idx,
                    'bbox': o['bbox'],
                    'type': o['shape_type'], # The actual shape type (e.g., 'circle')
                    'attributes': o['attributes'], # Directly use the sampled attributes dict
                    'confidence': 1.0 # Ground truth has 1.0 confidence
                })
            data.append((img, gt_dict))
        return data

class BongardSyntheticDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the programmatic Bongard synthetic data.
    """
    def __init__(self, n_samples: int, image_size: Tuple[int, int], num_objects: Tuple[int, int], 
                 num_classes: int, rule: Optional[BongardRule] = None): # NEW: Add rule parameter
        self.data = BongardGenerator(image_size=image_size, num_objects=num_objects, num_classes=num_classes).generate(n_samples, rule) # Pass rule to generator
        # --- NEW: Generate dummy difficulty scores for synthetic data ---
        self.difficulty_scores = np.random.rand(len(self.data)).astype(np.float32)
        logger.info(f"Generated {len(self.data)} synthetic Bongard samples with dummy difficulty scores.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[bytes, int, bytes, float]: # NEW: Return difficulty_score
        img, gt = self.data[idx]
        
        # For DALI external source, we need image as bytes and GT as JSON string
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG') # Save as PNG bytes
        img_bytes = img_byte_arr.getvalue()
        
        gt_json_string = json.dumps(gt) # Serialize ground truth dict to JSON string
        
        # Extract bongard_label from gt_json for the DALI label output
        bongard_label = gt['bongard_label']

        # Get difficulty score for this sample
        difficulty_score = self.difficulty_scores[idx]

        return img_bytes, bongard_label, gt_json_string.encode('utf-8'), difficulty_score # DALI expects bytes for JSON string

class BongardExternalSource(object):
    """
    External source for DALI pipeline to generate synthetic Bongard problems on the fly.
    Yields (image_view1, image_view2, label, ground_truth_json, difficulty, affine_matrix1, affine_matrix2, index).
    """
    def __init__(self, dataset: Any, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.epoch_size = len(self.dataset) # Total samples for an epoch
        self.current_epoch = 0
        self.current_image_size = CONFIG['data']['initial_image_size'] # Initial size for curriculum
        self.difficulty_scores = np.array([0.5] * len(self.dataset), dtype=np.float32) # Initial uniform difficulties

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.epoch_size:
            self.i = 0
            raise StopIteration

        batch_indices = self.indices[self.i : min(self.i + self.batch_size, self.epoch_size)]
        self.i += self.batch_size

        batch_images_view1 = []
        batch_images_view2 = []
        batch_labels = []
        batch_gts_json = []
        batch_difficulties = []
        batch_affine1 = []
        batch_affine2 = []
        batch_original_indices = [] # NEW: To store original indices

        for idx in batch_indices:
            # __getitem__ of BongardSyntheticDataset should return:
            # (image_view1, image_view2, label, gt_json_string, difficulty, affine_matrix1, affine_matrix2)
            # Make sure BongardSyntheticDataset returns all these.
            img1, img2, label, gt_json, difficulty, affine1, affine2 = self.dataset[idx]
            
            batch_images_view1.append(np.array(img1, dtype=np.uint8))
            batch_images_view2.append(np.array(img2, dtype=np.uint8))
            batch_labels.append(label)
            batch_gts_json.append(gt_json.encode('utf-8')) # DALI expects bytes
            batch_difficulties.append(difficulty)
            batch_affine1.append(affine1)
            batch_affine2.append(affine2)
            batch_original_indices.append(idx) # NEW: Add original index

        # Convert lists to NumPy arrays
        images1_np = np.stack(batch_images_view1, axis=0) # HWC
        images2_np = np.stack(batch_images_view2, axis=0) # HWC
        labels_np = np.array(batch_labels, dtype=np.int64)
        gts_json_np = np.array(batch_gts_json, dtype=object) # DALI expects object dtype for strings
        difficulties_np = np.array(batch_difficulties, dtype=np.float32)
        affine1_np = np.stack(batch_affine1, axis=0).astype(np.float32)
        affine2_np = np.stack(batch_affine2, axis=0).astype(np.float32)
        original_indices_np = np.array(batch_original_indices, dtype=np.int64) # NEW

        # Return a tuple of NumPy arrays
        return (images1_np, images2_np, labels_np, gts_json_np, difficulties_np,
                affine1_np, affine2_np, original_indices_np) # NEW: Return indices

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        # Update image size for curriculum learning
        if self.current_epoch < CONFIG['data']['curriculum_annealing_epochs']:
            start_size = CONFIG['data']['initial_image_size']
            final_size = CONFIG['data']['image_size']
            
            progress = self.current_epoch / CONFIG['data']['curriculum_annealing_epochs']
            
            new_height = int(start_size[0] + (final_size[0] - start_size[0]) * progress)
            new_width = int(start_size[1] + (final_size[1] - start_size[1]) * progress)
            
            self.current_image_size = [new_height, new_width]
            logger.info(f"CurriculumSampler: Epoch {self.current_epoch}, Image size annealed to {self.current_image_size}.")
            self.dataset.image_size = self.current_image_size # Update dataset's image size
        else:
            self.current_image_size = CONFIG['data']['image_size']
            self.dataset.image_size = self.current_image_size
            logger.info(f"CurriculumSampler: Epoch {self.current_epoch}, Image size fixed at {self.current_image_size}.")

        # For weighted sampling based on difficulty
        if CONFIG['training']['use_weighted_sampling'] and self.current_epoch > 0: # Start after first epoch
            self._calculate_sampling_weights()
            self.indices = np.random.choice(
                len(self.dataset), size=len(self.dataset),
                p=self.sampling_weights, replace=True # Replace=True for sampling with replacement
            ).tolist()
            logger.debug("CurriculumSampler: Re-sampled indices based on updated difficulties.")
        elif self.shuffle: # Reshuffle if not weighted sampling
            random.shuffle(self.indices)

    def update_difficulty_scores(self, indices: np.ndarray, new_scores: np.ndarray):
        """
        Updates difficulty scores for specific samples.
        Args:
            indices (np.ndarray): Original indices of the samples in the dataset.
            new_scores (np.ndarray): New difficulty scores for these samples (e.g., per-sample loss).
        """
        if len(indices) != len(new_scores):
            logger.error("Indices and new_scores must have the same length for difficulty update.")
            return
        self.difficulty_scores[indices] = new_scores
        logger.debug(f"Updated difficulty scores for {len(indices)} samples.")
        # If using weighted sampling, recalculate weights immediately
        if CONFIG['training']['use_weighted_sampling']:
            self._calculate_sampling_weights()

    def _calculate_sampling_weights(self):
        """Calculates sampling weights based on current difficulty scores."""
        # A common strategy: higher difficulty -> higher weight
        # You might want to apply a transformation (e.g., softmax, power function)
        # to the raw difficulty scores if they are not directly suitable as weights.
        # For example: weights = np.exp(self.difficulty_scores)
        
        # Simple example: directly use difficulty, normalize to sum to 1
        weights = self.difficulty_scores + 1e-6 # Add epsilon to avoid zero weights
        self.sampling_weights = weights / np.sum(weights)
        logger.debug("Recalculated sampling weights based on difficulties.")

    def get_current_image_size(self) -> Tuple[int, int]:
        return self.current_image_size

    def get_epoch_data(self) -> Tuple[List[str], np.ndarray]:
        # For synthetic data, this is less relevant as data is generated on-the-fly
        # but can return dummy paths/labels if needed for compatibility
        selected_paths = [f"synthetic_img_{i}.png" for i in self.indices]
        selected_labels = np.array([self.dataset.labels[i] for i in self.indices]) # Assuming dataset has labels
        return selected_paths, selected_labels

    def get_dataset_size(self) -> int:
        return len(self.dataset)



# -----------------------------------------------------------------------------
# RealObjectDetector: YOLOv8 & Fallbacks (with Conceptual SAM Mask-Based Fallback)
# -----------------------------------------------------------------------------
class RealObjectDetector:
    """
    RealObjectDetector class to integrate YOLOv8 for object detection.
    This class wraps the `ultralytics.YOLO` model to detect objects in images.
    Args:
        model_path (str): Path to the YOLOv8 model weights (e.g., 'yolov8n.pt').
        sam_model (str): SAM model type for fallback (e.g., 'vit_b', 'vit_l', 'vit_h').
    """
    def __init__(self, model_path: str = 'yolov8n.pt', sam_model: Optional[str] = None):
        # Attempt to import the actual YOLO from ultralytics
        try:
            from ultralytics import YOLO as RealYOLO_Actual
            self.model = RealYOLO_Actual(model_path)
            logger.info(f"Successfully loaded ultralytics.YOLO model from {model_path}.")
        except ImportError:
            logger.warning("Ultralytics YOLO not found. Using globally defined dummy RealYOLO for object detection.")
            # Fallback to the globally defined RealYOLO dummy
            self.model = RealYOLO(model_path)   # Assumes RealYOLO dummy is available in scope
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}. Ensure the path is correct and model is valid.")
            self.model = None   # Set to None to indicate failure

        self._cache = {}
        # Initialize SAM model if Kornia is available (lazy initialization)
        self._sam_model = None
        self.sam_model_type = sam_model
        if HAS_KORNIA and self.sam_model_type:
            logger.info(f"Kornia is available. SAM model '{self.sam_model_type}' will be lazily initialized for fallback.")

    def predict_with_sam(self, image_batch: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Performs batched inference with the Segment Anything Model (SAM).
        Args:
            image_batch (torch.Tensor): Batch of input images (NCHW, float32, [0,1]).
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - List of predicted masks from SAM, each element is [num_masks, H, W] for one image.
                - List of IoU predictions from SAM, each element is [num_masks] for one image.
        """
        if not HAS_KORNIA or not self.sam_model_type:
            logger.warning("SAM not available or not configured. Returning dummy masks.")
            # Return dummy masks matching the input batch shape
            dummy_masks = [torch.zeros(1, image_batch.shape[2], image_batch.shape[3], device=image_batch.device) for _ in range(image_batch.shape[0])]
            dummy_ious = [torch.tensor([0.0], device=image_batch.device) for _ in range(image_batch.shape[0])]
            return dummy_masks, dummy_ious

        # Lazy initialization of SAM model
        if self._sam_model is None:
            try:
                # Kornia's SegmentAnything requires `timm` for backbones
                if not HAS_TIMM:
                    raise ImportError("timm is required for Kornia's SegmentAnything model but not installed.")
                from kornia.contrib import SegmentAnything # Correct import path for SegmentAnything in Kornia
                logger.info(f"Initializing Kornia SegmentAnything ({self.sam_model_type}) model for conceptual fallback.")
                self._sam_model = SegmentAnything(model=self.sam_model_type).to(DEVICE).eval()
            except Exception as e:
                logger.error(f"Failed to initialize Kornia SAM model '{self.sam_model_type}': {e}. SAM fallback disabled.")
                self.sam_model_type = None  # Disable SAM if initialization fails
                dummy_masks = [torch.zeros(1, image_batch.shape[2], image_batch.shape[3], device=image_batch.device) for _ in range(image_batch.shape[0])]
                dummy_ious = [torch.tensor([0.0], device=image_batch.device) for _ in range(image_batch.shape[0])]
                return dummy_masks, dummy_ious

        logger.debug(f"Running SAM inference on batch of shape {image_batch.shape}")
        H, W = image_batch.shape[-2:]
        
        # SAM expects 0-255 range for its input, so scale up again
        img_tensor_for_sam = image_batch * 255.0
        
        # Generate a simple grid of points as prompts for each image in the batch
        num_points_x = 3
        num_points_y = 3
        grid_points_list = []
        for i in range(num_points_y):
            for j in range(num_points_x):
                x_coord = int(W * (j + 0.5) / num_points_x)
                y_coord = int(H * (i + 0.5) / num_points_y)
                grid_points_list.append([x_coord, y_coord])
        
        points = torch.tensor(grid_points_list, dtype=torch.float32, device=DEVICE)   # N_points x 2
        
        # Repeat points for each image in the batch
        batched_points = points.unsqueeze(0).repeat(image_batch.shape[0], 1, 1)  # B x N_points x 2
        batched_labels = torch.ones(image_batch.shape[0], points.shape[0], dtype=torch.float32, device=DEVICE)  # B x N_points (all positive)
        
        logger.debug(f"Attempting SAM fallback with {points.shape[0]} grid points per image as prompts.")
        
        # Run SAM to get masks
        # SAM's predict_masks returns a list of dicts, each dict has 'masks', 'iou_predictions', 'low_res_logits'
        sam_output_list = self._sam_model.predict_masks(
            img_tensor_for_sam,
            point_coords=batched_points,
            point_labels=batched_labels
        )
        
        all_masks = []
        all_iou_predictions = []
        for output_dict in sam_output_list:
            if 'masks' in output_dict and output_dict['masks'].numel() > 0:
                all_masks.append(output_dict['masks'])  # [num_masks_for_this_img, H, W]
            else:
                all_masks.append(torch.zeros(1, H, W, device=DEVICE))  # Dummy mask if no masks found
            
            if 'iou_predictions' in output_dict and output_dict['iou_predictions'].numel() > 0:
                all_iou_predictions.append(output_dict['iou_predictions'])
            else:
                all_iou_predictions.append(torch.tensor([0.0], device=DEVICE))  # Dummy IoU if no masks
        
        return all_masks, all_iou_predictions  # Return list of mask tensors and list of iou_predictions

    def detect_objects(self, image_pil: Image.Image, overall_confidence: float = 0.3) -> List[Dict]:
        """
        Runs YOLOv8 on a PIL image and returns detections above a specified confidence threshold.
        Maps YOLO class IDs to human-readable names using `YOLO_CLASS_MAP`.
        If YOLO fails or finds no objects, it attempts conceptual SAM mask-based fallback
        or basic contour detection.
        Args:
            image_pil (PIL.Image.Image): The input image in PIL format.
            overall_confidence (float): Confidence threshold for filtering detections.
        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a detected object with 'id', 'bbox', 'type' (mapped class name), and 'confidence'.
        """
        key = (id(image_pil), overall_confidence)
        if key in self._cache:
            logger.debug(f"Returning cached detections for image ID: {id(image_pil)}")
            return self._cache[key]
        
        # Apply contrast enhancement before YOLO inference
        processed_image_pil = enhance_logo_contrast(image_pil, gamma=1.5, unsharp_amount=1.0)
        logger.debug("Applied contrast enhancement to image before YOLO inference.")
        
        # Ensure image is in RGB mode for YOLO, even if source is grayscale
        if processed_image_pil.mode != 'RGB':
            processed_image_pil = processed_image_pil.convert('RGB')
            logger.debug("Converted image to RGB for YOLO inference.")
        
        detections = []
        if self.model:   # Only attempt YOLO if model was loaded successfully
            try:
                logger.debug(f"Running YOLO model on image (size: {processed_image_pil.size}, confidence: {overall_confidence}).")
                results = self.model(
                    processed_image_pil,   # Use the enhanced image
                    verbose=False,
                    augment=CONFIG['model']['yolo_augment']   # Use TTA if configured
                )
                
                # Ensure results[0].boxes is not empty before iterating
                if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    logger.debug(f"YOLO raw results: {len(results[0].boxes)} detections.")
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        cls   = int(box.cls[0])
                        if conf >= overall_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            class_name = YOLO_CLASS_MAP.get(cls  , f"class_{cls }")
                            detections.append({
                                "id": len(detections),   # Assign a unique ID for this detection run
                                "bbox": [x1, y1, x2, y2],
                                "type": class_name,
                                "confidence": conf
                            })
                    logger.debug(f"YOLO filtered detections: {len(detections)} detections above confidence threshold.")
                else:
                    logger.warning("YOLO results[0].boxes is empty or missing.")
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}. Attempting fallback mechanisms.")
        else:
            logger.warning("YOLO model not loaded. Skipping primary detection and attempting fallbacks.")
        
        # If nothing found, try again on a 2x upscaled image
        if not detections:
            logger.info("YOLO found no detections. Attempting upscaled YOLO detection.")
            w, h = processed_image_pil.size
            up = processed_image_pil.resize((w*2, h*2), Image.BILINEAR)
            logger.debug(f"Upscaled image to {up.size} for re-detection.")
            if self.model:   # Only attempt upscaled YOLO if model was loaded successfully
                try:
                    upscale_results = self.model(up, verbose=False, augment=CONFIG['model']['yolo_augment'])[0].boxes
                    if hasattr(upscale_results, 'boxes') and upscale_results.boxes is not None:
                        for box in upscale_results.boxes:  # Iterate over boxes directly
                            conf = float(box.conf[0])
                            if conf >= overall_confidence:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                detections.append({
                                    "id": len(detections),
                                    "bbox": [x1//2, y1//2, x2//2, y2//2],   # scale coordinates back down by 2x
                                    "type": YOLO_CLASS_MAP.get(int(box.cls[0]), f"class_{int(box.cls[0])}"),
                                    "confidence": conf
                                })
                        logger.debug(f"Upscaled YOLO filtered detections: {len(detections)} detections.")
                    else:
                        logger.warning("Upscaled YOLO results[0].boxes is empty or missing.")
                except Exception as e:
                    logger.warning(f"Upscaled YOLO detection failed: {e}. Proceeding to conceptual SAM fallback or contour fallback.")
            else:
                logger.warning("YOLO model not loaded. Skipping upscaled YOLO detection.")
        
        # Robust "No Objects Detected" Fallback: Conceptual SAM Mask-Based or basic contour detection
        if not detections:
            logger.info(f"YOLO (and upscaled YOLO) found no detections. Attempting conceptual SAM fallback or contour fallback.")
            if HAS_KORNIA and self.sam_model_type:   # Check if Kornia and SAM model type are available
                try:
                    # Convert PIL image to Kornia-compatible tensor (NCHW, float32, [0,1])
                    img_tensor = T.ToTensor()(image_pil).unsqueeze(0).to(DEVICE)   # 1CHW
                    if img_tensor.shape[1] == 1:   # If grayscale, replicate to 3 channels for SAM
                        img_tensor = img_tensor.repeat(1, 3, 1, 1)
                    
                    # Run SAM prediction (this is a batch of 1 image)
                    sam_masks_list, sam_iou_predictions_list = self.predict_with_sam(img_tensor)
                    
                    # Process SAM masks to get bounding boxes
                    if sam_masks_list and sam_masks_list[0].numel() > 0:  # Check if the first image in batch has masks
                        sam_masks = sam_masks_list[0]  # [num_masks, H, W]
                        sam_iou_predictions = sam_iou_predictions_list[0]  # [num_masks]
                        logger.debug(f"SAM generated {sam_masks.shape[0]} masks. Processing masks to bboxes.")
                        
                        for mask_idx in range(sam_masks.shape[0]):
                            mask_np = sam_masks[mask_idx].squeeze(0).cpu().numpy().astype(np.uint8) * 255   # Convert to 0-255
                            
                            # Find contours in the mask
                            contours_mask, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours_mask:
                                largest_contour = max(contours_mask, key=cv2.contourArea)
                                area = cv2.contourArea(largest_contour)
                                if area > CONFIG['debug'].get('min_contour_area_sam_fallback', 50):   # Use a configurable min area
                                    x, y, w, h = cv2.boundingRect(largest_contour)
                                    
                                    # Simple shape approximation for SAM-derived bounding box
                                    peri = cv2.arcLength(largest_contour, True)
                                    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
                                    
                                    shape = "unknown_shape_sam"
                                    num_vertices = len(approx)
                                    if num_vertices == 3: shape = "triangle"
                                    elif num_vertices == 4: shape = "square" if 0.9 <= float(w)/h <= 1.1 else "rectangle"
                                    elif num_vertices > 8: shape = "circle" if 4 * np.pi * area / (peri * peri) > 0.7 else "polygon"
                                    
                                    detections.append({
                                        "id": len(detections),
                                        "bbox": [x, y, x + w, y + h],
                                        "type": shape,
                                        "confidence": float(sam_iou_predictions[mask_idx].cpu().item())   # Use SAM's IoU prediction as confidence
                                    })
                        logger.debug(f"SAM fallback yielded {len(detections)} detections.")
                    else:
                        logger.warning("SAM did not generate any masks for this image. Proceeding to contour fallback.")
                except Exception as sam_e:
                    logger.warning(f"Kornia SAM fallback failed: {sam_e}. Ensure kornia is installed and model can be loaded. Proceeding to contour fallback.")
            
            # Fallback to basic contour detection for blobs/shapes if SAM also fails or is not available
            if not detections:   # Only run if SAM fallback also failed
                logger.info("SAM fallback failed or not available. Falling back to basic contour detection.")
                detections = fallback_shape_detection_bw(image_pil.convert('L'), min_area=CONFIG['debug'].get('min_contour_area_sam_fallback', 50))
                logger.debug(f"Basic contour fallback found {len(detections)} regions.")
                if not detections:
                    logger.error("Fallback contour detection also failed. No objects detected for this image.")
        
        self._cache[key] = detections
        return detections

def enhance_logo_contrast(pil_img: Image.Image, gamma: float = 1.5, unsharp_amount: float = 1.0):
    """
    Enhances the contrast and sharpness of a PIL image, suitable for low-contrast logos.
    Applies gamma correction and an unsharp mask.
    Args:
        pil_img (PIL.Image.Image): The input image.
        gamma (float): Gamma correction factor. Values > 1.0 lighten shadows, < 1.0 darken.
                        For low contrast, often gamma > 1.0 is used to brighten mid-tones.
        unsharp_amount (float): Strength of the unsharp mask.
    Returns:
        PIL.Image.Image: The enhanced image.
    """
    logger.debug(f"Enhancing contrast for image (mode: {pil_img.mode}, gamma: {gamma}, unsharp: {unsharp_amount}).")
    # Ensure image is in 'L' (grayscale) or 'RGB' mode for consistent processing
    original_mode = pil_img.mode
    if original_mode not in ['L', 'RGB']:
        pil_img = pil_img.convert('RGB')   # Convert to RGB if it's an unsupported mode like P (paletted)
        logger.debug(f"Converted image from {original_mode} to RGB for contrast enhancement.")
    
    # Gamma correction
    # Convert to float32, normalize to 0-1, apply gamma, then scale back to 0-255
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = np.power(arr, 1.0 / gamma)   # Apply inverse gamma for brightening
    img_gamma_corrected = Image.fromarray((arr * 255).astype(np.uint8), mode=pil_img.mode)
    
    # Unsharp mask
    # radius: The radius of the Gaussian filter.
    # percent: The percentage of the sharpened image to blend with the original.
    # threshold: The minimum difference in pixel values to be sharpened.
    enhanced_img = img_gamma_corrected.filter(ImageFilter.UnsharpMask(radius=2, percent=150 * unsharp_amount, threshold=3))
    logger.debug("Contrast enhancement complete.")
    return enhanced_img

def fallback_shape_detection_bw(gray_pil: Image.Image, threshold: int = 128, min_area: int = 100):
    """
    Performs basic contour-based shape detection for black-and-white images
    as a fallback when primary object detection (YOLO) fails.
    Identifies bounding boxes and attempts to approximate basic shapes.
    Args:
        gray_pil (PIL.Image.Image): The input image in PIL 'L' (grayscale) mode.
        threshold (int): Pixel intensity threshold for binarization (0-255).
        min_area (int): Minimum contour area to consider as a valid detection.
    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a detected
                    shape with 'id', 'bbox', 'type' (approximated shape), and 'confidence'.
    """
    logger.debug(f"Starting basic contour-based fallback detection (threshold: {threshold}, min_area: {min_area}).")
    # Ensure input is grayscale and convert to NumPy array
    if gray_pil.mode != 'L':
        gray_pil = gray_pil.convert('L')
    logger.debug("Converted image to grayscale for contour detection.")
    img = np.array(gray_pil)
    
    # Threshold to binary: black shapes on white background.
    # THRESH_BINARY_INV means pixels > threshold become 0 (black), others become 255 (white).
    # This makes the shapes white on a black background, which is good for findContours.
    _, bw = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    logger.debug(f"Image binarized with threshold {threshold}.")
    
    # Find contours
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} raw contours.")
    
    detections = []
    # Limit fallback contours to avoid excessive dummy nodes, configurable via CONFIG
    max_fallback_cnt = CONFIG['debug'].get('max_fallback_cnt', 5)
    
    for cnt_idx, cnt in enumerate(contours):
        if cnt_idx >= max_fallback_cnt:
            logger.debug(f"Reached max_fallback_cnt ({max_fallback_cnt}). Stopping contour processing.")
            break   # Stop if max_fallback_cnt is reached
        
        area = cv2.contourArea(cnt)
        if area < min_area:
            logger.debug(f"Skipping contour {cnt_idx} due to small area ({area} < {min_area}).")
            continue   # Skip small contours
        
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Approximate shape based on polygon approximation
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        shape = "unknown_shape"
        num_vertices = len(approx)
        
        if num_vertices == 3:
            shape = "triangle"
        elif num_vertices == 4:
            # Check aspect ratio for square vs. rectangle
            aspect_ratio = float(w) / h
            if 0.9 <= aspect_ratio <= 1.1:
                shape = "square"
            else:
                shape = "rectangle"
        elif num_vertices > 4 and num_vertices < 8:   # Heuristic for general polygons
            shape = "polygon"
        elif num_vertices >= 8:
            # Check circularity for circle-like shapes
            if peri > 0:
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity > 0.7:   # Common threshold for circularity
                    shape = "circle"
                else:
                    shape = "polygon"
            else:
                shape = "polygon"   # Fallback if perimeter is zero
        
        detections.append({
            "id": cnt_idx,
            "bbox": [x, y, x + w, y + h],
            "type": shape,
            "confidence": 0.1   # Assign a low confidence for fallback detections
        })
        logger.debug(f"Contour {cnt_idx}: Detected shape '{shape}' with bbox {detections[-1]['bbox']}.")
    logger.debug(f"Finished basic contour-based fallback detection. Found {len(detections)} detections.")
    return detections

def _calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Compute the area of the intersection rectangle
    inter_width = xB - xA
    inter_height = yB - yA
    
    if inter_width <= 0 or inter_height <= 0:
        return 0.0   # No intersection
    inter_area = inter_width * inter_height
    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

# --- NEW: Kornia Augmentation Function for DALI Python Operator ---
@torch.no_grad()  # Ensure no gradients are tracked for augmentations
def kornia_augment(images_dali_gpu: torch.Tensor, is_train: bool, num_channels: int):
    """
    Applies Kornia augmentations to a batch of images.
    This function is designed to be called by DALI's fn.python_function.
    Args:
        images_dali_gpu (torch.Tensor): Batch of images from DALI, NCHW, float32, on GPU.
                                        Expected range [0, 255] for Kornia.
        is_train (bool): If True, applies training augmentations.
        num_channels (int): Number of image channels (1 or 3).
    Returns:
        torch.Tensor: Augmented batch of images, NCHW, float32, on GPU.
    """
    if not HAS_KORNIA:
        logger.warning("Kornia is not installed. Skipping Kornia augmentations.")
        return images_dali_gpu  # Return original images if Kornia not available
    # Ensure images are in the expected format for Kornia (float32, [0, 255])
    # DALI's output from `fn.decoders.image` is typically uint8, 0-255.
    # If it's already float32 from `fn.crop_mirror_normalize`, then Kornia expects 0-1 or 0-255 based on ops.
    # Let's assume input is [0, 255] float32 for Kornia's default behavior.
    
    # Kornia augmentations typically expect RGB. If grayscale, convert to 3 channels.
    if num_channels == 1 and images_dali_gpu.shape[1] == 1:
        images_dali_gpu = images_dali_gpu.repeat(1, 3, 1, 1)
    
    if is_train:
        # Define Kornia augmentation pipeline for training
        aug_list = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.2),  # Add vertical flip
            K.RandomRotation(degrees=K.RandomUniformGenerator(0.0, 15.0), p=0.5),
            K.RandomAffine(degrees=K.RandomUniformGenerator(0.0, 10.0),  # Small affine
                           translate=K.RandomUniformGenerator(0.0, 0.1),
                           scale=K.RandomUniformGenerator(0.8, 1.2),
                           shear=K.RandomUniformGenerator(0.0, 5.0),
                           p=0.5),
            K.ColorJitter(brightness=K.RandomUniformGenerator(0.6, 1.4),  # Stronger color jitter
                          contrast=K.RandomUniformGenerator(0.6, 1.4),
                          saturation=K.RandomUniformGenerator(0.6, 1.4),
                          hue=K.RandomUniformGenerator(-0.2, 0.2),
                          p=0.8),
            K.RandomGrayscale(p=0.1) if num_channels == 3 else K.Identity(),  # Apply only if RGB
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            # Add ElasticTransform if available and configured
            K.RandomElasticTransform(kernel_size=(3,3), sigma=(4.0, 5.0), alpha=(30.0, 35.0), p=0.3) if HAS_ELASTIC_TRANSFORM else K.Identity(),
            data_keys=["input"],  # Specify that augmentations apply to the input image
            p=1.0,  # Apply the sequence with this probability
            same_on_batch=False  # Apply different augmentations to each image in batch
        ).to(images_dali_gpu.device)
        
        augmented_images = aug_list(images_dali_gpu)
        logger.debug("Applied Kornia training augmentations.")
    else:
        # No augmentations for validation/test, just pass through
        augmented_images = images_dali_gpu
        logger.debug("Skipping Kornia augmentations for validation/test.")
    # If original input was grayscale, convert back to 1 channel if needed
    if num_channels == 1 and augmented_images.shape[1] == 3:
        # Convert RGB to grayscale (simple average)
        augmented_images = KC.rgb_to_grayscale(augmented_images)
        logger.debug("Converted augmented image back to 1 channel (grayscale).")
    return augmented_images

@pipeline_def
def dali_pipe(file_root, file_list, labels_list, batch_size, num_threads, device_id, height, width, is_train, num_channels, feature_consistency_alpha, imagenet_mean, imagenet_std, use_synthetic_data: bool = False, external_source_iterator: Optional[BongardExternalSource] = None):
    """
    DALI Pipeline for efficient, GPU-accelerated image loading and augmentation.
    Uses fn.readers.file to directly read images from disk or fn.external_source for synthetic data.
    Args:
        file_root (str): Root directory for image files (only used for real data).
        file_list (List[str]): List of relative paths to image files (only used for real data).
        labels_list (List[int]): List of corresponding labels (only used for real data).
        batch_size (int): Number of samples per batch.
        num_threads (int): Number of CPU threads to use for DALI operations.
        device_id (int): GPU device ID to use (-1 for CPU).
        height (int): Target image height after resizing.
        width (int): Target image width after resizing.
        is_train (bool): If True, applies training-specific augmentations.
        num_channels (int): Number of image channels (1 for grayscale, 3 for RGB).
        feature_consistency_alpha (float): Alpha value for feature consistency loss (0.0 to disable).
        imagenet_mean (List[float]): Mean values for image normalization (0-255 range).
        imagenet_std (List[float]): Standard deviation values for image normalization (0-255 range).
        use_synthetic_data (bool): If True, uses external_source_iterator for data.
        external_source_iterator (BongardExternalSource): Iterator for synthetic data.
    Returns:
        tuple: A tuple of DALI tensors:
               (image_view_1, image_view_2, bongard_label, ground_truth_json_string, difficulty_score, affine_matrix_1, affine_matrix_2)
    """
    if use_synthetic_data:
        logger.debug("DALI pipeline using external source for synthetic data.")
        # external_source_iterator yields (image_bytes, label, gt_json_bytes, difficulty_score)
        imgs, labels, gts_json, difficulties = fn.external_source(
            source=external_source_iterator,
            num_outputs=4,  # Images, Labels, GT JSON string, Difficulties
            layout=["H", "H", "H", "H"], # Layout for image bytes, label, string, float (arbitrary for string)
            name="ExternalSource"
        )
        # Decode images from bytes
        decoded = fn.decoders.image(
            imgs,
            device="mixed" if device_id != -1 else "cpu",
            output_type=types.RGB if num_channels == 3 else types.GRAY
        )
        # Labels are directly from external source now
        labels_gpu = fn.cast(labels, dtype=types.INT64)
        
        # Pass the original gt_json_string through as well
        gts_json_gpu = gts_json.gpu() if device_id != -1 else gts_json
        gts_json_gpu = fn.cast(gts_json_gpu, dtype=types.STRING)  # Ensure it's treated as string

        # Pass difficulties through
        difficulties_gpu = difficulties.gpu() if device_id != -1 else difficulties
        difficulties_gpu = fn.cast(difficulties_gpu, dtype=types.FLOAT) # Ensure it's float
        
    else:
        logger.debug("DALI pipeline using file reader for real data.")
        # Use fn.readers.file to read images and labels directly
        # file_list and labels_list must be provided as arguments to the pipeline_def
        imgs, labels = fn.readers.file(
            file_root=file_root,
            files=file_list,
            labels=labels_list,
            random_shuffle=is_train,  # Shuffle only for training
            name="Reader"
        )
        decoded = fn.decoders.image(
            imgs,
            device="mixed" if device_id != -1 else "cpu",
            output_type=types.RGB if num_channels == 3 else types.GRAY
        )
        labels_gpu = fn.cast(labels, dtype=types.INT64)
        gts_json_gpu = fn.constant(np.array(["{}"], dtype=np.object_), shape=(1,))  # Dummy empty JSON string
        gts_json_gpu = fn.cast(gts_json_gpu, dtype=types.STRING)
        difficulties_gpu = fn.constant(np.array([1.0], dtype=np.float32), shape=(1,)) # Dummy uniform difficulty for real data
        difficulties_gpu = fn.cast(difficulties_gpu, dtype=types.FLOAT)
        
    logger.debug(f"DALI Decoded image shape: {decoded.shape}, dtype: {decoded.dtype}")
    
    # Resize and apply initial augmentations for View 1
    images_view1 = decoded
    if is_train:
        logger.debug("Applying DALI training augmentations for View 1.")
        # Random resized crop
        images_view1 = fn.random_resized_crop(
            images_view1,
            size=[height, width],
            random_area=(0.8, 1.0),
            random_aspect_ratio=(0.75, 1.33),
            device="gpu" if device_id != -1 else "cpu"
        )
        # Random flip
        images_view1 = fn.flip(images_view1, horizontal=fn.random.coin_flip(),
                               device="gpu" if device_id != -1 else "cpu")
        # Brightness and Contrast
        images_view1 = fn.brightness_contrast(
            images_view1,
            brightness=fn.random.uniform(range=(0.8, 1.2)),
            contrast=fn.random.uniform(range=(0.8, 1.2)),
            device="gpu" if device_id != -1 else "cpu"
        )
        # Hue and Saturation for 3-channel images
        if num_channels == 3:
            images_view1 = fn.hue(images_view1, hue=fn.random.uniform(range=(-0.1, 0.1)),
                                  device="gpu" if device_id != -1 else "cpu")
            images_view1 = fn.saturation(images_view1, saturation=fn.random.uniform(range=(0.8, 1.2)),
                                         device="gpu" if device_id != -1 else "cpu")
        # Rotation
        rotation_angle_dali = fn.random.uniform(range=(-15.0, 15.0))
        images_view1 = fn.rotate(images_view1, angle=rotation_angle_dali, fill_value=0.0,
                                 device="gpu" if device_id != -1 else "cpu")
        # Gaussian blur
        images_view1 = fn.gaussian_blur(images_view1, sigma=fn.random.uniform(range=(0.0, 1.0)),
                                        device="gpu" if device_id != -1 else "cpu")
        # Ensure final size after augmentations
        images_view1 = fn.resize(images_view1,
                                 resize_x=width,
                                 resize_y=height,
                                 interp_type=types.INTERP_LINEAR,
                                 device="gpu" if device_id != -1 else "cpu")
    else:
        logger.debug("Applying DALI validation/test resizing for View 1.")
        # For validation, just resize
        images_view1 = fn.resize(decoded,
                                 resize_x=width,
                                 resize_y=height,
                                 interp_type=types.INTERP_LINEAR,
                                 device="gpu" if device_id != -1 else "cpu")
    
    # Apply Kornia augmentations as a Python function
    # Kornia expects float32 images in [0, 255] range for many ops.
    # DALI's `decoded` output is uint8, so convert to float32 before Kornia.
    # Then normalize to [0, 1] after Kornia for model input.
    images_view1_float = fn.cast(images_view1, dtype=types.FLOAT)
    images_view1_kornia_input = fn.crop_mirror_normalize(
        images_view1_float,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        # Kornia augmentations often expect unnormalized images (0-255 or 0-1).
        # We normalize *after* Kornia. So, pass mean=0, std=1 for this step.
        mean=[0.0] * num_channels,
        std=[1.0] * num_channels,
        mirror=False, # Mirroring handled by Kornia or DALI's initial flips
        device="gpu" if device_id != -1 else "cpu"
    )
    
    # Apply Kornia augmentations
    images_view1_augmented_kornia = fn.python_function(
        images_view1_kornia_input,
        function=functools.partial(kornia_augment, is_train=is_train, num_channels=num_channels),
        output_layouts=types.NCHW, # Kornia outputs NCHW
        device="gpu" if device_id != -1 else "cpu"
    )
    
    logger.debug(f"DALI View 1 image shape after DALI + Kornia augmentations/resize: {images_view1_augmented_kornia.shape}")
    
    # Final normalization for View 1 (after Kornia)
    out_view1 = fn.crop_mirror_normalize(
        images_view1_augmented_kornia,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
        mean=imagenet_mean, # Apply actual ImageNet mean/std here
        std=imagenet_std,
        mirror=False, # Mirroring already handled by Kornia or DALI's initial flips
        device="gpu" if device_id != -1 else "cpu"
    )
    logger.debug(f"DALI View 1 output shape after final normalization: {out_view1.shape}")
    logger.debug(f"DALI final normalization parameters - Mean: {imagenet_mean}, Std: {imagenet_std}")
    
    # Prepare second augmented view for feature consistency if enabled
    out_view2 = out_view1 # Default to same if not enabled
    if is_train and feature_consistency_alpha > 0:
        logger.debug(f"Feature consistency enabled (alpha={feature_consistency_alpha}). Preparing View 2.")
        images_view2 = decoded  # Start from decoded image again for second view
        # Apply different augmentations for second view
        images_view2 = fn.random_resized_crop(
            images_view2,
            size=[height, width],
            random_area=(0.7, 1.0),  # Slightly different area
            random_aspect_ratio=(0.7, 1.4),  # Slightly different aspect ratio
            device="gpu" if device_id != -1 else "cpu"
        )
        images_view2 = fn.flip(images_view2, horizontal=fn.random.coin_flip(),
                               device="gpu" if device_id != -1 else "cpu")
        images_view2 = fn.brightness_contrast(
            images_view2,
            brightness=fn.random.uniform(range=(0.7, 1.3)),  # Different ranges
            contrast=fn.random.uniform(range=(0.7, 1.3)),
            device="gpu" if device_id != -1 else "cpu"
        )
        if num_channels == 3:
            images_view2 = fn.hue(images_view2, hue=fn.random.uniform(range=(-0.2, 0.2)),
                                  device="gpu" if device_id != -1 else "cpu")
            images_view2 = fn.saturation(images_view2, saturation=fn.random.uniform(range=(0.7, 1.3)),
                                         device="gpu" if device_id != -1 else "cpu")
        images_view2_rotation_angle_dali = fn.random.uniform(range=(-20.0, 20.0))  # Different range
        images_view2 = fn.rotate(images_view2, angle=images_view2_rotation_angle_dali, fill_value=0.0,
                                 device="gpu" if device_id != -1 else "cpu")
        images_view2 = fn.gaussian_blur(images_view2, sigma=fn.random.uniform(range=(0.0, 1.5)),
                                        device="gpu" if device_id != -1 else "cpu")
        images_view2 = fn.resize(images_view2,
                                 resize_x=width,
                                 resize_y=height,
                                 interp_type=types.INTERP_LINEAR,
                                 device="gpu" if device_id != -1 else "cpu")
        
        # Apply Kornia augmentations to View 2 as well
        images_view2_float = fn.cast(images_view2, dtype=types.FLOAT)
        images_view2_kornia_input = fn.crop_mirror_normalize(
            images_view2_float,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.0] * num_channels,
            std=[1.0] * num_channels,
            mirror=False,
            device="gpu" if device_id != -1 else "cpu"
        )
        images_view2_augmented_kornia = fn.python_function(
            images_view2_kornia_input,
            function=functools.partial(kornia_augment, is_train=is_train, num_channels=num_channels),
            output_layouts=types.NCHW,
            device="gpu" if device_id != -1 else "cpu"
        )
        
        out_view2 = fn.crop_mirror_normalize(
            images_view2_augmented_kornia,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=imagenet_mean,
            std=imagenet_std,
            mirror=False,
            device="gpu" if device_id != -1 else "cpu"
        )
        logger.debug(f"DALI View 2 output shape after final normalization: {out_view2.shape}")
    
    # --- Affine Matrices Clarification ---
    # DALI's built-in augmentations (like fn.random_resized_crop, fn.rotate, etc.)
    # perform transformations internally but do NOT automatically output the
    # corresponding affine transformation matrices.
    # The `affine_matrices_1` and `affine_matrices_2` returned here are
    # dummy identity matrices. If precise affine matrices are required for
    # downstream tasks (e.g., visualizing exact transformations), they would
    # need to be calculated manually based on the applied DALI operations
    # or by using DALI's `fn.transforms.warp_affine` with explicit matrix inputs.
    dummy_affine_matrix = fn.constant(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        shape=(2, 3),
        dtype=types.FLOAT
    )
    # Replicate dummy_affine_matrix for batch_size
    affine_matrices_1 = fn.stack(*([dummy_affine_matrix] * batch_size))
    affine_matrices_2 = fn.stack(*([dummy_affine_matrix] * batch_size))
    
    logger.debug(f"DALI returning dummy affine matrices. Shape: {affine_matrices_1.shape}")
    if is_train and feature_consistency_alpha > 0:
        logger.debug("Dummy affine matrices generated for both views due to feature consistency being enabled.")
    logger.debug(f"DALI Labels output shape: {labels_gpu.shape}, dtype: {labels_gpu.dtype}")
    logger.debug(f"DALI GT JSON output shape: {gts_json_gpu.shape}, dtype: {gts_json_gpu.dtype}")
    logger.debug(f"DALI Difficulties output shape: {difficulties_gpu.shape}, dtype: {difficulties_gpu.dtype}")

    # Return the bongard_label (0 or 1), the full ground truth JSON string, and difficulty score
    return out_view1, out_view2, labels_gpu, gts_json_gpu, difficulties_gpu, affine_matrices_1, affine_matrices_2

# --- NEW: Function to set all random seeds ---
def set_seed(seed: int):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"All random seeds set to {seed}.")

# --- NEW: Loss Functions ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    CrossEntropyLoss with label smoothing.
    Args:
        smoothing (float): The amount of smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FeatureConsistencyLoss(nn.Module):
    """
    Calculates consistency loss between features from two different views of the same input.
    Args:
        loss_type (str): Type of loss to use ('mse', 'cosine', 'kl_div').
    """
    def __init__(self, loss_type='mse'):
        super(FeatureConsistencyLoss, self).__init__()
        self.loss_type = loss_type
        logger.info(f"FeatureConsistencyLoss initialized with type: {loss_type}")
    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'mse':
            loss = F.mse_loss(features1, features2)
        elif self.loss_type == 'cosine':
            # Cosine similarity is in [-1, 1], convert to loss [0, 2]
            loss = 1 - F.cosine_similarity(features1, features2).mean()
        elif self.loss_type == 'kl_div':
            # KL divergence expects log-probabilities for input, probabilities for target
            # Ensure features are normalized to represent distributions
            log_probs1 = F.log_softmax(features1, dim=-1)
            probs2 = F.softmax(features2, dim=-1)
            loss = F.kl_div(log_probs1, probs2, reduction='batchmean')
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        return loss

class DistillationLoss(nn.Module):
    """
    Calculates the knowledge distillation loss.
    Combines KL Divergence for soft targets with CrossEntropy for hard labels.
    Args:
        alpha (float): Weight for the distillation loss component.
        temperature (float): Temperature for softening probabilities.
        base_loss_fn (nn.Module): The base loss function for hard labels (e.g., nn.CrossEntropyLoss).
    """
    def __init__(self, alpha: float, temperature: float, base_loss_fn: nn.Module):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.base_loss_fn = base_loss_fn
        logger.info(f"DistillationLoss initialized with alpha={alpha}, temperature={temperature}.")
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        # Soft predictions from student
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL Divergence for distillation
        dist_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard loss (standard cross-entropy with true labels)
        hard_loss = self.base_loss_fn(student_logits, labels)
        
        # Combine losses
        combined_loss = self.alpha * dist_loss + (1.0 - self.alpha) * hard_loss
        return combined_loss

# --- NEW: MixUp and CutMix Implementation ---
class MixupCutmix(nn.Module):
    """
    Applies MixUp or CutMix augmentation to a batch of images and labels.
    Args:
        mixup_alpha (float): Alpha parameter for the Beta distribution for MixUp.
                             If <= 0, MixUp is disabled.
        cutmix_alpha (float): Alpha parameter for the Beta distribution for CutMix.
                              If <= 0, CutMix is disabled.
        mixup_cutmix_ratio (float): Probability of applying MixUp vs. CutMix
                                    when both are enabled. 0.0 for only MixUp,
                                    1.0 for only CutMix.
        num_classes (int): Number of classes for one-hot encoding labels.
    """
    def __init__(self, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0,
                 mixup_cutmix_ratio: float = 0.5):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_cutmix_ratio = mixup_cutmix_ratio
        
        self.mixup_enabled = mixup_alpha > 0.0
        self.cutmix_enabled = cutmix_alpha > 0.0
        
        if not self.mixup_enabled and not self.cutmix_enabled:
            logger.warning("Both MixUp and CutMix are disabled (alpha <= 0). No augmentation will be applied.")
        else:
            logger.info(f"MixupCutmix initialized: MixUp={self.mixup_enabled} (alpha={mixup_alpha}), "
                        f"CutMix={self.cutmix_enabled} (alpha={cutmix_alpha}), "
                        f"MixUp/CutMix Ratio={mixup_cutmix_ratio}.")

    def forward(self, img: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float, str]]:
        """
        Applies MixUp or CutMix to the input batch.
        Args:
            img (torch.Tensor): Input images of shape [B, C, H, W].
            target (torch.Tensor): Target labels of shape [B].
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float, str]]:
                - Augmented images.
                - Tuple containing (target_a, target_b, lambda, mode_applied).
                  If no augmentation, lambda=1.0, mode_applied='none'.
        """
        if not self.mixup_enabled and not self.cutmix_enabled:
            return img, (target, target, 1.0, 'none')
        lam = 1.0
        mode = 'none'
        
        # Decide whether to apply MixUp or CutMix
        if self.mixup_enabled and self.cutmix_enabled:
            if random.random() < self.mixup_cutmix_ratio:
                mode = 'mixup'
            else:
                mode = 'cutmix'
        elif self.mixup_enabled:
            mode = 'mixup'
        elif self.cutmix_enabled:
            mode = 'cutmix'
        
        if mode == 'mixup':
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(0.0, min(1.0, lam)) # Ensure lambda is within [0, 1]
            
            batch_size = img.shape[0]
            index = torch.randperm(batch_size, device=img.device)
            
            mixed_img = lam * img + (1 - lam) * img[index, :]
            target_a, target_b = target, target[index]
            
            logger.debug(f"Applied MixUp with lambda={lam:.4f}.")
            return mixed_img, (target_a, target_b, lam, 'mixup')
        
        elif mode == 'cutmix':
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = max(0.0, min(1.0, lam)) # Ensure lambda is within [0, 1]
            
            batch_size = img.shape[0]
            index = torch.randperm(batch_size, device=img.device)
            
            # Generate bounding box for CutMix
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            
            mixed_img = img.clone()
            mixed_img[:, :, bby1:bby2, bbx1:bbx2] = img[index, :, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda to actual ratio of pasted area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2]))
            
            target_a, target_b = target, target[index]
            
            logger.debug(f"Applied CutMix with lambda={lam:.4f}, bbox=[{bbx1},{bby1},{bbx2},{bby2}].")
            return mixed_img, (target_a, target_b, lam, 'cutmix')
        
        else: # Should not happen if logic is correct, but as a fallback
            return img, (target, target, 1.0, 'none')

def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    Args:
        size (tuple): (C, H, W) of the image.
        lam (float): Lambda value from Beta distribution.
    Returns:
        tuple: (bbx1, bby1, bbx2, bby2) coordinates of the bounding box.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# --- NEW: Knowledge Replay Buffer ---
class KnowledgeReplayBuffer:
    """
    A simple replay buffer for storing and sampling (image, label, symbolic_output) tuples.
    This is used for experience replay in a curriculum learning setting.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        logger.info(f"KnowledgeReplayBuffer initialized with capacity: {capacity}")

    def add(self, experience: Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]):
        """
        Adds an experience tuple (image_tensor, label_tensor, symbolic_output_dict) to the buffer.
        Note: Tensors are expected to be on CPU for storage to avoid GPU memory issues.
        """
        # Detach and move to CPU before storing
        img_cpu = experience[0].detach().cpu()
        label_cpu = experience[1].detach().cpu()
        # Ensure symbolic_output is JSON-serializable if it contains tensors
        symbolic_output_copy = copy.deepcopy(experience[2])
        # Convert any tensors in symbolic_output_copy to Python lists/scalars
        def convert_tensors_to_lists(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            if isinstance(obj, dict):
                return {k: convert_tensors_to_lists(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_tensors_to_lists(elem) for elem in obj]
            return obj
        
        final_symbolic_output = convert_tensors_to_lists(symbolic_output_copy)
        
        self.buffer.append((img_cpu, label_cpu, final_symbolic_output))
        logger.debug(f"Added experience to replay buffer. Current size: {len(self.buffer)}")

    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Samples a batch of experiences from the buffer.
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]: A list of sampled experiences.
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"Replay buffer has {len(self.buffer)} samples, less than requested batch size {batch_size}. Returning all available.")
            return list(self.buffer)
        
        samples = random.sample(self.buffer, batch_size)
        logger.debug(f"Sampled {len(samples)} experiences from replay buffer.")
        return samples

    def __len__(self):
        return len(self.buffer)

# --- NEW: Symbolic Consistency Loss ---
def symbolic_consistency_loss(attr_logits1: Dict[str, torch.Tensor], attr_logits2: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculates a symbolic consistency loss between attribute predictions from two views.
    This encourages the model to predict similar attributes for the same object
    under different augmentations.
    Args:
        attr_logits1 (Dict[str, torch.Tensor]): Attribute logits for view 1.
        attr_logits2 (Dict[str, torch.Tensor]): Attribute logits for view 2.
    Returns:
        torch.Tensor: The calculated symbolic consistency loss.
    """
    loss = 0.0
    num_losses = 0
    
    for attr_name in attr_logits1.keys():
        if attr_name in attr_logits2:
            logits1 = attr_logits1[attr_name]
            logits2 = attr_logits2[attr_name]
            
            # Ensure tensors are not empty and have compatible shapes
            if logits1.numel() > 0 and logits2.numel() > 0 and logits1.shape == logits2.shape:
                # Use KL divergence to measure consistency between predicted distributions
                log_probs1 = F.log_softmax(logits1, dim=-1)
                probs2 = F.softmax(logits2, dim=-1)
                
                kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean')
                loss += kl_div
                num_losses += 1
            else:
                logger.warning(f"Skipping symbolic consistency for attribute '{attr_name}' due to incompatible or empty logits.")
    
    if num_losses > 0:
        return loss / num_losses
    return torch.tensor(0.0, device=DEVICE)  # Return 0 if no valid losses were computed

# Define a basic JSON schema for ground truth validation
# You might need to expand this based on your actual GT JSON structure
BONGARD_GT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bongard_label": {"type": "integer", "minimum": 0, "maximum": 1},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["id", "bbox", "attributes"] # Confidence might be optional in GT
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "integer"},
                    "dst": {"type": "integer"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["src", "dst", "type"] # Confidence might be optional in GT
            }
        }
    },
    "required": ["bongard_label", "objects", "relations"]
}

# --- DALI Loader Builder ---
def build_dali_loader(file_list: List[str], labels_list: List[int], config: Dict[str, Any], mode: str, external_source_iterator: Optional[BongardExternalSource] = None) -> Tuple[Pipeline, DALIGenericIterator]:
    """
    Builds and returns a DALI pipeline and iterator.
    Args:
        file_list (List[str]): List of image file paths.
        labels_list (List[int]): List of corresponding labels.
        config (Dict): Configuration dictionary.
        mode (str): 'train', 'val', or 'inference'.
        external_source_iterator (Optional[BongardExternalSource]): Iterator for synthetic data.
    Returns:
        Tuple[Pipeline, DALIGenericIterator]: The DALI pipeline and iterator.
    """
    is_train = (mode == 'train')
    use_synthetic_data = config['data']['use_synthetic_data'] and (mode == 'train' or mode == 'val')  # Only use synthetic for train/val
    
    if is_train and not use_synthetic_data:
        # For real data training, use CurriculumSampler
        sampler = CurriculumSampler(
            image_paths=file_list,
            labels=np.array(labels_list),
            initial_difficulty_scores=load_bongard_data(DATA_ROOT_PATH)[2],  # Reload difficulty scores
            batch_size=config['model']['batch_size'],
            annealing_epochs=config['data']['curriculum_annealing_epochs'],
            total_epochs=config['model']['epochs'],
            is_train=True,
            use_weighted_sampling=config['training']['use_weighted_sampling']
        )
        # Initialize with epoch 0 data
        initial_file_list, initial_labels_list = sampler.get_epoch_data()
        initial_image_size = sampler.get_current_image_size()
        logger.info(f"DALI loader (train) initialized with CurriculumSampler. Initial image size: {initial_image_size}")
        
    else:
        # For validation, inference, or synthetic data, use full lists/external source
        sampler = None  # No custom sampler for these modes
        initial_file_list = file_list
        initial_labels_list = labels_list
        initial_image_size = config['data']['image_size']  # Use final image size
        logger.info(f"DALI loader ({mode}) initialized. Image size: {initial_image_size}")
    
    # Create the DALI pipeline
    pipeline = dali_pipe(
        file_root=DATA_ROOT_PATH,
        file_list=initial_file_list,
        labels_list=initial_labels_list,
        batch_size=config['model']['batch_size'],
        num_threads=config['data']['dataloader_num_workers'],
        device_id=0 if DEVICE.type == 'cuda' else -1,  # Use GPU 0 if CUDA, else CPU
        height=initial_image_size[0],
        width=initial_image_size[1],
        is_train=is_train,
        num_channels=config['data']['num_channels'],
        feature_consistency_alpha=config['training']['feature_consistency_alpha'],
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        use_synthetic_data=use_synthetic_data,
        external_source_iterator=external_source_iterator
    )
    
    pipeline.build()
    
    # Create the DALI PyTorch iterator
    # Updated output_map to include 'difficulties'
    output_map = ["view1", "view2", "labels", "gts_json", "difficulties", "affine1", "affine2"] 
    
    # For synthetic data, the size needs to be derived from the external source iterator
    if use_synthetic_data:
        loader_size = math.ceil(len(external_source_iterator.dataset) / config['model']['batch_size'])
    else:
        loader_size = math.ceil(len(initial_file_list) / config['model']['batch_size'])
    
    loader = DALIGenericIterator(
        pipeline,
        output_map=output_map,
        size=loader_size,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL if is_train else LastBatchPolicy.DROP,
        # DALI 1.0+ requires `reader_name` for `external_source` if used
        reader_name="ExternalSource" if use_synthetic_data else "Reader"
    )
    
    # Attach the sampler to the loader if it's a CurriculumSampler
    if sampler:
        loader.iterator = sampler  # Attach for epoch updates and image size queries
    return pipeline, loader
# -----------------------------------------------------------------------------
# AttributeClassifier: Multi-head classifier with timm backbones and LoRA
# -----------------------------------------------------------------------------
class AttributeClassifier(nn.Module):
    """
    Multi-head classifier for object attributes (fill, color, size, orientation, shape, texture).
    Uses a configurable backbone (MobileNetV2, MobileNetV3, EfficientNetB0, or timm models).
    Can be wrapped with LoRA adapters for parameter-efficient fine-tuning.
    Args:
        num_channels (int): Number of input image channels (1 for grayscale, 3 for RGB).
        image_size (List[int]): Expected input image size [height, width].
        backbone_name (str): Name of the pre-trained backbone to use ('mobilenet_v2',
                              'mobilenet_v3_small', 'efficientnet_b0', or timm model names like 'convnext_tiny').
        num_fill_types (int): Number of classes for fill attribute.
        num_color_types (int): Number of classes for color attribute.
        num_size_types (int): Number of classes for size attribute.
        num_orientation_types (int): Number of classes for orientation attribute.
        num_shape_types (int): Number of classes for shape attribute.
        num_texture_types (int): Number of classes for texture attribute.
        use_gradient_checkpointing (bool): If True, applies gradient checkpointing to the backbone.
        use_lora (bool): If True, wraps the backbone with LoRA adapters.
        dropout_rate (float): Dropout rate to apply after features for MC Dropout.
    """
    def __init__(self, num_channels, image_size, backbone_name,
                 num_fill_types=len(ATTRIBUTE_FILL_MAP['fill']),
                 num_color_types=len(ATTRIBUTE_FILL_MAP['color']),
                 num_size_types=len(ATTRIBUTE_FILL_MAP['size']),
                 num_orientation_types=len(ATTRIBUTE_FILL_MAP['orientation']),
                 num_shape_types=len(ATTRIBUTE_FILL_MAP['shape']),
                 num_texture_types=len(ATTRIBUTE_FILL_MAP['texture']),
                 use_gradient_checkpointing=False,
                 use_lora=False,
                 dropout_rate: float = 0.0):
        super(AttributeClassifier, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.backbone_name = backbone_name
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_lora = use_lora
        self.dropout_rate = dropout_rate

        # Load backbone with pre-trained ImageNet weights
        feature_dim = 0
        if HAS_TIMM and backbone_name in timm.list_models(pretrained=True):
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, num_classes=0 # num_classes=0 to get features
            )
            feature_dim = self.backbone.num_features # Get feature dimension from timm model
            logger.info(f"AttributeClassifier using timm backbone: {backbone_name}, feature_dim: {feature_dim}")
        # Add torchvision model imports explicitly if they are not global
        elif backbone_name == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v2(weights=weights)
            self.backbone.classifier = nn.Identity() # Extract features before the final FC layer
            feature_dim = 1280 # For MobileNetV2, features come from features[-1]
            logger.info(f"AttributeClassifier using torchvision MobileNetV2, feature_dim: {feature_dim}")
        elif backbone_name == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
            self.backbone.classifier = nn.Identity()
            feature_dim = 576 # For MobileNetV3_Small, features come from avgpool
            logger.info(f"AttributeClassifier using torchvision MobileNetV3_Small, feature_dim: {feature_dim}")
        elif backbone_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280 # For EfficientNet_B0, features come from avgpool
            logger.info(f"AttributeClassifier using torchvision EfficientNet_B0, feature_dim: {feature_dim}")
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Check timm availability or torchvision support.")
        
        # Adapt first conv layer for single-channel (grayscale) input if needed
        if num_channels == 1:
            # Check for timm models (e.g., conv_stem)
            if hasattr(self.backbone, 'conv_stem') and isinstance(self.backbone.conv_stem, nn.Conv2d):
                original_first_conv = self.backbone.conv_stem
                new_first_conv = nn.Conv2d(
                    1, original_first_conv.out_channels,
                    kernel_size=original_first_conv.kernel_size,
                    stride=original_first_conv.stride,
                    padding=original_first_conv.padding,
                    bias=original_first_conv.bias is not None
                )
                self.backbone.conv_stem = new_first_conv
                logger.info(f"AttributeClassifier timm backbone adapted for {num_channels} input channels.")
            # Check for torchvision models (e.g., features[0][0])
            elif hasattr(self.backbone, 'features') and isinstance(self.backbone.features[0][0], nn.Conv2d):
                original_first_conv = self.backbone.features[0][0]
                new_first_conv = nn.Conv2d(
                    1, original_first_conv.out_channels,
                    kernel_size=original_first_conv.kernel_size,
                    stride=original_first_conv.stride,
                    padding=original_first_conv.padding,
                    bias=original_first_conv.bias is not None
                )
                self.backbone.features[0][0] = new_first_conv
                logger.info(f"AttributeClassifier torchvision backbone adapted for {num_channels} input channels.")
            else:
                logger.warning(f"Could not adapt first conv layer for {backbone_name} to {num_channels} channels. Input channel mismatch might occur.")
        
        # Wrap backbone with LoRA if enabled
        if self.use_lora and HAS_LORA:
            # A proper LoRA integration would involve replacing specific layers with LoRA versions.
            # For demonstration, we'll iterate through linear and conv layers and replace them with LoRA versions.
            # This is a simplified example and might need careful consideration for specific backbone architectures.
            logger.info("Applying LoRA adapters to AttributeClassifier backbone.")
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Linear):
                    setattr(self.backbone, name, LoRA.Linear(module.in_features, module.out_features))
                    logger.debug(f"Replaced Linear layer {name} with LoRA.Linear.")
                elif isinstance(module, nn.Conv2d):
                    # LoRA for Conv2d is also available in loralib
                    setattr(self.backbone, name, LoRA.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding))
                    logger.debug(f"Replaced Conv2d layer {name} with LoRA.Conv2d.")
            LoRA.mark_only_lora_as_trainable(self.backbone) # Mark LoRA parameters as trainable, freeze others
            logger.info("AttributeClassifier backbone LoRA enabled: only LoRA parameters are trainable.")
        elif self.use_lora and not HAS_LORA:
            logger.warning("LoRA requested but loralib is not available. Proceeding without LoRA.")
        
        # Dropout layer for MC Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        logger.info(f"AttributeClassifier initialized with dropout rate: {self.dropout_rate}")
        
        # Multi-head classification layers
        # Check if num_types is greater than 0 before creating head
        self.fill_head = nn.Linear(feature_dim, num_fill_types) if num_fill_types > 0 else None
        self.color_head = nn.Linear(feature_dim, num_color_types) if num_color_types > 0 else None
        self.size_head = nn.Linear(feature_dim, num_size_types) if num_size_types > 0 else None
        self.orientation_head = nn.Linear(feature_dim, num_orientation_types) if num_orientation_types > 0 else None
        self.shape_head = nn.Linear(feature_dim, num_shape_types) if num_shape_types > 0 else None
        self.texture_head = nn.Linear(feature_dim, num_texture_types) if num_texture_types > 0 else None
        # Store output dimension for GNN input
        self.output_dim = feature_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ..., torch.Tensor]:
        """
        Forward pass for the AttributeClassifier.
        Args:
            x (torch.Tensor): Input image tensor [batch_size, channels, height, width].
        Returns:
            tuple: A tuple containing logits for each attribute head and the extracted features:
                    (fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, features)
                    Logits will be None if the corresponding head is not initialized.
        """
        # Ensure input is 3 channels if backbone expects it, by repeating grayscale channel
        if self.num_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # Repeat grayscale channel to simulate RGB
            logger.debug("Repeated grayscale channel to 3 channels for backbone input.")
        
        # Apply Gradient Checkpointing to Backbone (if enabled)
        if self.use_gradient_checkpointing:
            # Checkpoint the backbone's feature extraction part
            if hasattr(self.backbone, 'forward_features'):
                features = torch.utils.checkpoint.checkpoint(self.backbone.forward_features, x)
                logger.debug("Applied gradient checkpointing via 'forward_features'.")
            elif hasattr(self.backbone, 'features'):
                features = torch.utils.checkpoint.checkpoint(self.backbone.features, x)
                logger.debug("Applied gradient checkpointing via 'features'.")
            else:
                features = torch.utils.checkpoint.checkpoint(self.backbone, x)
                logger.debug("Applied gradient checkpointing directly to backbone.")
            
            # Apply pooling/flattening outside checkpoint if needed
            if features.ndim > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                logger.debug("Applied adaptive average pooling and flattened features after checkpointing.")
        else:
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                logger.debug("Forward pass through backbone using 'forward_features'.")
            else:
                features = self.backbone(x)
                logger.debug("Forward pass directly through backbone.")
            
            # Flatten features if they are not already (e.g., from avgpool)
            if features.ndim > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                logger.debug("Applied adaptive average pooling and flattened features.")
        
        # Apply dropout for MC Dropout
        features = self.dropout(features)
        
        # Get logits from each head, if the head exists
        fill_logits = self.fill_head(features) if self.fill_head else None
        color_logits = self.color_head(features) if self.color_head else None
        size_logits = self.size_head(features) if self.size_head else None
        orientation_logits = self.orientation_head(features) if self.orientation_head else None
        shape_logits = self.shape_head(features) if self.shape_head else None
        texture_logits = self.texture_head(features) if self.texture_head else None
        
        logger.debug(f"Attribute logits shapes: fill={fill_logits.shape if fill_logits is not None else 'None'}, color={color_logits.shape if color_logits is not None else 'None'}, etc.")
        return fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, features

def fine_tune_model(model: nn.Module, train_loader: DALIGenericIterator, val_loader: DALIGenericIterator, config: Dict[str, Any], epochs: int):
    """
    Performs a short fine-tuning session for a model.
    This is typically used after pruning steps to recover accuracy.
    Args:
        model (nn.Module): The model to fine-tune.
        train_loader (DALIGenericIterator): DALI data loader for training.
        val_loader (DALIGenericIterator): DALI data loader for validation.
        config (Dict): Configuration dictionary.
        epochs (int): Number of epochs for fine-tuning.
    """
    logger.info(f"Starting fine-tuning for {epochs} epochs.")
    
    # Use a potentially lower learning rate for fine-tuning
    fine_tune_lr = config['model']['initial_learning_rate'] * 0.1 # Example: 1/10th of initial LR
    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr,
                            weight_decay=config['training']['weight_decay'])
    criterion = LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing_epsilon'])
    scaler = GradScaler(enabled=config['training']['use_amp'])

    # Simple fine-tuning loop (can be more sophisticated)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_loader.reset()
        for batch_idx, data in enumerate(train_loader):
            images_view1 = data['view1'].permute(0, 3, 1, 2).to(DEVICE)
            gts_json_strings_batch = [s.decode('utf-8') for s in data['gts_json'].cpu().numpy()]
            
            current_batch_labels = []
            for i, gt_json_str in enumerate(gts_json_strings_batch):
                try:
                    gt_dict = json.loads(gt_json_str)
                    current_batch_labels.append(gt_dict['bongard_label'])
                except (json.JSONDecodeError, jsonschema.ValidationError):
                    current_batch_labels.append(0)
            labels_bongard = torch.tensor(current_batch_labels, dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            with autocast(enabled=config['training']['use_amp']):
                bongard_logits, _, aggregated_outputs = model(images_view1, gts_json_strings_batch)
                loss = criterion(bongard_logits, labels_bongard)
                
                # Optionally add attribute/relation losses during fine-tuning if desired
                if 'attribute_logits' in aggregated_outputs and 'attribute_gt' in aggregated_outputs:
                    for attr_name, attr_logits in aggregated_outputs['attribute_logits'].items():
                        if attr_name in aggregated_outputs['attribute_gt'] and aggregated_outputs['attribute_gt'][attr_name].numel() > 0:
                            attr_gt = aggregated_outputs['attribute_gt'][attr_name]
                            if attr_logits.shape[0] == attr_gt.shape[0]:
                                loss += config['training']['attribute_loss_weight'] * nn.CrossEntropyLoss()(attr_logits, attr_gt)
                
                if 'relation_logits' in aggregated_outputs and 'relation_gt' in aggregated_outputs:
                    rel_logits = aggregated_outputs['relation_logits']
                    rel_gt = aggregated_outputs['relation_gt']
                    if rel_logits.shape[0] == rel_gt.shape[0]:
                        loss += config['training']['relation_loss_weight'] * nn.CrossEntropyLoss()(rel_logits, rel_gt)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Fine-tune Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Optional: Validate during fine-tuning
        val_loss, val_accuracy, _, _ = _validate_model_ensemble(model, val_loader, criterion, config)
        logger.info(f"Fine-tune Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    logger.info("Fine-tuning complete.")
# This function is conceptual and needs to be implemented based on your specific
# logic for inferring relations from object properties.
# It would be used in PerceptionModule.forward if you choose Edge Feature Augmentation
# for symbolic priors on real data.

def get_predicted_relation(obj1_bbox: List[int], obj2_bbox: List[int],
                           obj1_attrs: Dict[str, str], obj2_attrs: Dict[str, str]) -> str:
    """
    Conceptual function to infer a symbolic relation between two objects.
    This would typically be a rule-based system or a small, dedicated model.
    Args:
        obj1_bbox (List[int]): Bounding box of object 1 [x1, y1, x2, y2].
        obj2_bbox (List[int]): Bounding box of object 2 [x1, y1, x2, y2].
        obj1_attrs (Dict[str, str]): Predicted attributes of object 1.
        obj2_attrs (Dict[str, str]): Predicted attributes of object 2.
    Returns:
        str: The inferred symbolic relation (e.g., 'touching', 'above', 'same_shape').
             Returns 'none' or a default if no clear relation.
    """
    # --- Spatial Relations (based on bounding boxes) ---
    iou = _calculate_iou(obj1_bbox, obj2_bbox)
    if iou > 0.7:
        return 'overlapping' # Simplified
    
    # Calculate centers
    center1_x, center1_y = (obj1_bbox[0] + obj1_bbox[2]) / 2, (obj1_bbox[1] + obj1_bbox[3]) / 2
    center2_x, center2_y = (obj2_bbox[0] + obj2_bbox[2]) / 2, (obj2_bbox[1] + obj2_bbox[3]) / 2

    # Relative position
    if center2_y < center1_y - 5: # Threshold for 'above'
        return 'above'
    if center2_y > center1_y + 5: # Threshold for 'below'
        return 'below'
    if center2_x < center1_x - 5: # Threshold for 'left_of'
        return 'left_of'
    if center2_x > center1_x + 5: # Threshold for 'right_of'
        return 'right_of'

    # --- Attribute-based Relations ---
    if obj1_attrs.get('shape') == obj2_attrs.get('shape'):
        return 'same_shape'
    if obj1_attrs.get('color') == obj2_attrs.get('color'):
        return 'same_color'
    
    # Add more rules as needed (e.g., 'contains', 'inside', 'touching', 'aligned_h', 'aligned_v')
    
    return 'none' # Default if no specific relation is inferred

# Note: This function would typically be placed in a utility file (e.g., `utils.py` or `symbolic_reasoning.py`)
# and imported where needed, such as in `PerceptionModule.forward`.



# -----------------------------------------------------------------------------
# RelationGNN: Graph Neural Network for object relations
# -----------------------------------------------------------------------------
class RelationGNN(nn.Module):
    """
    Graph Neural Network (GNN) for processing object features and inferring relations.
    Args:
        input_dim (int): Dimension of input node features (from AttributeClassifier).
        hidden_dim (int): Dimension of hidden layers in GNN.
        output_dim (int): Dimension of output graph embedding.
        gnn_depth (int): Number of GCNConv layers.
        num_relation_types (int): Number of possible relation types.
        apply_symbolic_priors (bool): If True, incorporates symbolic priors into edge features (requires GATv2Conv).
        symbolic_edge_feature_dim (Optional[int]): Dimension of symbolic edge features if applied.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, gnn_depth: int,
                 num_relation_types: int = len(RELATION_MAP),
                 apply_symbolic_priors: bool = False,
                 symbolic_edge_feature_dim: Optional[int] = None): # New: dimension for edge features
        super(RelationGNN, self).__init__()
        self.gnn_depth = gnn_depth
        self.apply_symbolic_priors = apply_symbolic_priors
        self.symbolic_edge_feature_dim = symbolic_edge_feature_dim
        
        self.convs = nn.ModuleList()

        # --- Conditional: Use GATv2Conv if symbolic priors are applied via edge features ---
        # This requires edge_dim to be passed to the GNN layer.
        # Ensure HAS_PYG is True for this.
        if self.apply_symbolic_priors and HAS_PYG:
            from torch_geometric.nn import GATv2Conv # Import GATv2Conv
            edge_dim = self.symbolic_edge_feature_dim if self.symbolic_edge_feature_dim is not None else len(RELATION_MAP)
            logger.info(f"RelationGNN using GATv2Conv with edge_dim={edge_dim} for symbolic priors.")
            self.convs.append(GATv2Conv(input_dim, hidden_dim, edge_dim=edge_dim))
            for _ in range(gnn_depth - 1):
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim))
        else:
            # Fallback to GCNConv if no symbolic priors or PyG not available
            logger.info("RelationGNN using GCNConv (no edge feature support or PyG not available).")
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(gnn_depth - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Set2Set for global graph pooling
        self.set2set = Set2Set(hidden_dim, processing_steps=6, num_layers=2)
        
        # Output layer for graph embedding
        self.graph_embedding_head = nn.Linear(2 * hidden_dim, output_dim)
        
        # Relation head (for global relation classification)
        self.relation_head = nn.Linear(2 * hidden_dim, num_relation_types)
        
        logger.info(f"RelationGNN initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, gnn_depth={gnn_depth}.")
        if apply_symbolic_priors:
            logger.info("RelationGNN configured to apply symbolic priors (via edge features if GATv2Conv used).")

    def forward(self, data: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # Changed to Any for flexibility
        """
        Forward pass for the RelationGNN.
        Args:
            data (torch_geometric.data.Batch): A batch of graph data containing:
                x (torch.Tensor): Node features [num_nodes_in_batch, input_dim].
                edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges_in_batch].
                batch (torch.Tensor): Batch assignment vector [num_nodes_in_batch] for nodes to graphs.
                edge_attr (Optional[torch.Tensor]): Edge features (e.g., symbolic relations) [num_edges_in_batch, edge_dim].
        Returns:
            tuple: (node_embeddings, graph_embedding, relation_logits)
                node_embeddings (torch.Tensor): Features for each node after GNN layers.
                graph_embedding (torch.Tensor): Global graph-level embedding.
                relation_logits (torch.Tensor): Logits for global relation classification [batch_size, num_relation_types].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None # Get edge attributes if present
        
        # Apply GCNConv/GATv2Conv layers
        for i, conv in enumerate(self.convs):
            if self.apply_symbolic_priors and edge_attr is not None:
                # Assuming the conv layer supports edge_attr (e.g., GATv2Conv)
                x = conv(x, edge_index, edge_attr)
                logger.debug(f"GNN Layer {i+1} using edge attributes.")
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            logger.debug(f"GNN Layer {i+1} output shape: {x.shape}")
        
        node_embeddings = x # Final node embeddings
        # Apply Set2Set pooling to get a global graph embedding
        graph_embedding_raw = self.set2set(node_embeddings, batch)
        graph_embedding = self.graph_embedding_head(graph_embedding_raw)
        
        # Get relation logits (for global relations per graph in the batch)
        relation_logits = self.relation_head(graph_embedding_raw)

        return node_embeddings, graph_embedding, relation_logits

# -----------------------------------------------------------------------------
# PerceptionModule: Integrates ObjectDetector, AttributeClassifier, RelationGNN
# -----------------------------------------------------------------------------
class PerceptionModule(nn.Module):
    """
    The main Perception Module that integrates object detection, attribute classification,
    and relational reasoning (GNN) to solve Bongard problems.
    Args:
        config (Dict): Configuration dictionary.
        object_detector (RealObjectDetector): Instance of the object detector.
    """
    def __init__(self, config: Dict, object_detector: Any): # Changed to Any for flexibility
        super(PerceptionModule, self).__init__()
        self.config = config
        self.object_detector = object_detector
        self.num_classes = config['model']['num_classes']
        self.num_channels = config['data']['num_channels']
        self.image_size = config['data']['image_size']
        self.detection_confidence_threshold = config['model']['detection_confidence_threshold']
        
        # Flags for symbolic priors and learnable embeddings
        self.apply_symbolic_priors = config['debug'].get('apply_symbolic_priors', False)
        self.use_learnable_symbolic_embeddings = config['debug'].get('use_learnable_symbolic_embeddings', False)
        
        # --- Symbolic Priors Integration: Learnable Embeddings ---
        self.symbolic_attribute_embeddings = nn.ModuleDict()
        self.symbolic_relation_embedding = None # Initialize to None
        
        if self.apply_symbolic_priors and self.use_learnable_symbolic_embeddings:
            # Node attribute embeddings
            for attr_type, vocab_size in SYMBOLIC_EMBEDDING_DIMS.items():
                if attr_type != 'relation': # Relations are for edges, not nodes initially
                    # Use a fixed embedding dimension for attributes (e.g., 16 or 32)
                    self.symbolic_attribute_embeddings[attr_type] = nn.Embedding(vocab_size, 16) # Example dim
            
            # Edge relation embedding
            self.symbolic_relation_embedding = nn.Embedding(SYMBOLIC_EMBEDDING_DIMS['relation'], 16) # Example dim
            logger.info("Initialized learnable symbolic embeddings for attributes and relations.")

        # Initialize AttributeClassifier
        self.attribute_classifier = AttributeClassifier(
            num_channels=self.num_channels,
            image_size=self.image_size,
            backbone_name=config['model']['attribute_backbone'],
            num_fill_types=len(ATTRIBUTE_FILL_MAP['fill']),
            num_color_types=len(ATTRIBUTE_FILL_MAP['color']),
            num_size_types=len(ATTRIBUTE_FILL_MAP['size']),
            num_orientation_types=len(ATTRIBUTE_FILL_MAP['orientation']),
            num_shape_types=len(ATTRIBUTE_FILL_MAP['shape']),
            num_texture_types=len(ATTRIBUTE_FILL_MAP['texture']),
            use_gradient_checkpointing=config['training'].get('use_gradient_checkpointing', False),
            use_lora=config['training'].get('use_lora', False),
            dropout_rate=config['model']['initial_dropout_rate']
        )
        logger.info("AttributeClassifier initialized.")
        
        # Initialize RelationGNN
        gnn_input_dim = self.attribute_classifier.output_dim
        
        # Adjust GNN input dimension for symbolic node features
        if self.apply_symbolic_priors:
            if self.use_learnable_symbolic_embeddings:
                # Sum of embedding dimensions for all node attributes
                total_symbolic_node_embedding_dim = sum(self.symbolic_attribute_embeddings[attr_type].embedding_dim
                                                        for attr_type in SYMBOLIC_EMBEDDING_DIMS.keys() if attr_type != 'relation')
                gnn_input_dim += total_symbolic_node_embedding_dim
                logger.info(f"GNN input dimension increased by {total_symbolic_node_embedding_dim} for learnable symbolic node priors.")
            else:
                # Sum of one-hot dimensions for all node attributes if embeddings are not used
                total_symbolic_node_one_hot_dim = sum(len(v) for k, v in ATTRIBUTE_FILL_MAP.items())
                gnn_input_dim += total_symbolic_node_one_hot_dim
                logger.info(f"GNN input dimension increased by {total_symbolic_node_one_hot_dim} for one-hot symbolic node priors.")

        gnn_hidden_dim = gnn_input_dim // 2
        gnn_output_dim = 256
        gnn_depth = config['model']['gnn_depth']
        
        self.relation_gnn = RelationGNN(
            input_dim=gnn_input_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            gnn_depth=gnn_depth,
            num_relation_types=len(RELATION_MAP),
            apply_symbolic_priors=self.apply_symbolic_priors,
            # Pass the dimension of symbolic edge features
            symbolic_edge_feature_dim=self.symbolic_relation_embedding.embedding_dim if (self.apply_symbolic_priors and self.use_learnable_symbolic_embeddings) else len(RELATION_MAP)
        )
        logger.info("RelationGNN initialized.")
        
        # Final classification head for the Bongard Problem
        self.bongard_head = nn.Linear(gnn_output_dim, self.num_classes)
        logger.info("Bongard classification head initialized.")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, images_tensor: torch.Tensor, ground_truth_json_strings: Optional[List[str]] = None) -> Tuple[torch.Tensor, List[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Forward pass for the PerceptionModule.
        Args:
            images_tensor (torch.Tensor): Batch of input images [B, C, H, W].
            ground_truth_json_strings (Optional[List[str]]): List of JSON strings
                containing ground truth for synthetic data. If provided, object detection
                and attribute classification will use ground truth instead of predictions
                for symbolic prior generation.
        Returns:
            tuple: (bongard_logits, detected_objects_batch, aggregated_outputs)
                bongard_logits (torch.Tensor): Logits for Bongard problem classification [B, num_classes].
                detected_objects_batch (List[List[Dict]]): List of detected objects for each image in batch.
                                                        Each inner list contains dicts with 'bbox', 'type', 'confidence'.
                aggregated_outputs (Dict[str, Any]): Dictionary containing various intermediate outputs:
                    - 'attribute_logits': Dict of attribute logits per attribute type.
                    - 'attribute_gt': Dict of attribute ground truths (if available).
                    - 'relation_logits': Tensor of relation logits.
                    - 'relation_gt': Tensor of relation ground truths (if available).
                    - 'image_features_student': Features from the graph embedding.
                    - 'bongard_labels': Ground truth Bongard labels (if available).
        """
        batch_size = images_tensor.shape[0]
        all_object_features = []
        all_graph_data = []
        detected_objects_batch = [] # Store detected objects for each image in the batch
        
        # Initialize dictionaries for aggregated outputs (batch-wise)
        batch_attribute_logits_raw = {attr_name: [] for attr_name in ATTRIBUTE_FILL_MAP.keys()}
        batch_attribute_gt_raw = {attr_name: [] for attr_name in ATTRIBUTE_FILL_MAP.keys()}
        batch_relation_gt_raw = []
        batch_bongard_labels_from_gt = []
        
        # Iterate through each image in the batch
        for i in range(batch_size):
            image_pil = T.ToPILImage()(images_tensor[i].cpu()) # Convert tensor to PIL Image for YOLO
            
            # --- Object Detection ---
            current_image_detected_objects = []
            current_image_gt_relations_map = {} # Map (src, dst) to relation type
            
            # Use ground truth if provided and valid, otherwise use object detector
            if ground_truth_json_strings and ground_truth_json_strings[i]:
                try:
                    gt_data = json.loads(ground_truth_json_strings[i])
                    # Validate against schema
                    jsonschema.validate(instance=gt_data, schema=BONGARD_GT_JSON_SCHEMA)
                    
                    current_image_detected_objects = gt_data.get('objects', [])
                    batch_bongard_labels_from_gt.append(gt_data.get('bongard_label', 0))
                    
                    # Populate attribute GT from JSON for this image
                    for obj_idx, obj_gt in enumerate(current_image_detected_objects):
                        for attr_name, attr_val in obj_gt.get('attributes', {}).items():
                            if attr_name in ATTRIBUTE_FILL_MAP:
                                attr_idx = ATTRIBUTE_FILL_MAP_INV[attr_name].get(attr_val)
                                if attr_idx is not None:
                                    while len(batch_attribute_gt_raw[attr_name]) <= i:
                                        batch_attribute_gt_raw[attr_name].append([])
                                    batch_attribute_gt_raw[attr_name][i].append(attr_idx)
                                else:
                                    logger.warning(f"Unknown attribute value '{attr_val}' for type '{attr_name}' in GT for image {i}.")
                    
                    # Populate relation GT from JSON for this image
                    for rel in gt_data.get('relations', []):
                        current_image_gt_relations_map[(rel['src'], rel['dst'])] = rel['type']
                    
                    # For global relation GT, use the first relation type or default to 'none'
                    if gt_data.get('relations'):
                        first_rel_type = gt_data['relations'][0].get('type', 'none')
                        batch_relation_gt_raw.append(RELATION_MAP_INV.get(first_rel_type, 0))
                    else:
                        batch_relation_gt_raw.append(0) # Default to 'none'
                    
                    logger.debug(f"Using ground truth objects and relations for image {i}.")
                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                    logger.error(f"Failed to decode/validate GT JSON for image {i}: {e}. Falling back to object detector.")
                    current_image_detected_objects = self.object_detector.detect_objects(
                        image_pil, overall_confidence=self.detection_confidence_threshold
                    )
                    batch_bongard_labels_from_gt.append(0) # Default label if GT fails
                    batch_relation_gt_raw.append(0) # Default relation GT if GT fails
            else:
                # Use RealObjectDetector for real data or if GT is not provided/invalid
                current_image_detected_objects = self.object_detector.detect_objects(
                    image_pil, overall_confidence=self.detection_confidence_threshold
                )
                batch_bongard_labels_from_gt.append(0) # Default label for real data if no GT
                batch_relation_gt_raw.append(0) # Default relation GT for real data if no GT
            
            detected_objects_batch.append(current_image_detected_objects)
            logger.debug(f"Detected {len(current_image_detected_objects)} objects for image {i}.")
            
            # Handle case where no objects are detected (or GT is empty)
            if not current_image_detected_objects:
                logger.warning(f"No objects detected for image {i}. Creating a dummy node.")
                dummy_bbox = [0, 0, 1, 1]
                dummy_type = "dummy_object"
                dummy_confidence = 0.01
                current_image_detected_objects = [{
                    "id": 0, "bbox": dummy_bbox, "type": dummy_type, "confidence": dummy_confidence,
                    "attributes": {attr: list(ATTRIBUTE_FILL_MAP[attr].values())[0] for attr in ATTRIBUTE_FILL_MAP} # Default attributes
                }]
                detected_objects_batch[-1] = current_image_detected_objects
            
            # Extract image patches for attribute classification
            object_patches = []
            valid_detected_objects_for_patches = [] # Store valid objects that produced patches
            for obj in current_image_detected_objects:
                x1, y1, x2, y2 = obj['bbox']
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(image_pil.width, int(x2)), min(image_pil.height, int(y2))
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox for object {obj.get('id', 'N/A')}: {obj['bbox']}. Skipping.")
                    continue
                try:
                    patch = image_pil.crop((x1, y1, x2, y2))
                    patch_tensor = T.Compose([
                        T.Resize(self.image_size),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])(patch).to(self.device)
                    
                    if self.num_channels == 1 and patch_tensor.shape[0] == 3:
                        patch_tensor = T.Grayscale(num_output_channels=1)(patch_tensor)
                    elif self.num_channels == 3 and patch_tensor.shape[0] == 1:
                        patch_tensor = patch_tensor.repeat(3, 1, 1)
                    
                    object_patches.append(patch_tensor)
                    valid_detected_objects_for_patches.append(obj)
                except Exception as e:
                    logger.warning(f"Error processing patch for object {obj.get('id', 'N/A')}: {e}. Skipping.")

            if not object_patches:
                logger.warning(f"No valid object patches for image {i}. Creating a dummy patch.")
                dummy_patch = torch.randn(self.num_channels, self.image_size[0], self.image_size[1], device=self.device)
                object_patches.append(dummy_patch)
                valid_detected_objects_for_patches = [{"id": 0, "bbox": [0,0,1,1], "type": "dummy_object", "confidence": 0.01,
                                                      "attributes": {attr: list(ATTRIBUTE_FILL_MAP[attr].values())[0] for attr in ATTRIBUTE_FILL_MAP}}]
            
            object_patches_batch_for_attr_classifier = torch.stack(object_patches)
            
            # Get attribute features and logits for each object
            attr_logits_tuple = self.attribute_classifier(object_patches_batch_for_attr_classifier)
            current_object_features = attr_logits_tuple[-1]
            
            # Store attribute logits for this image
            current_image_attribute_logits = {}
            for idx, attr_name in enumerate(ATTRIBUTE_FILL_MAP.keys()):
                if attr_logits_tuple[idx] is not None:
                    current_image_attribute_logits[attr_name] = attr_logits_tuple[idx]
            
            for attr_name, logits_list in batch_attribute_logits_raw.items():
                if attr_name in current_image_attribute_logits:
                    logits_list.append(current_image_attribute_logits[attr_name])
                else:
                    num_attr_classes = len(ATTRIBUTE_FILL_MAP.get(attr_name, {}))
                    logits_list.append(torch.empty(0, num_attr_classes, device=self.device))
            
            # --- Symbolic Priors Integration (Node Feature Augmentation) ---
            if self.apply_symbolic_priors:
                symbolic_features_list = []
                for obj_idx, obj_data in enumerate(valid_detected_objects_for_patches):
                    obj_symbolic_vector = []
                    
                    # Determine source of symbolic attribute: GT if available, else predicted
                    # For training: prioritize GT if `ground_truth_json_strings` is provided.
                    # For inference: use predicted attributes.
                    
                    # Extract predicted attributes for this object from current_image_attribute_logits
                    predicted_attrs_for_obj = {}
                    for attr_name, logits_tensor in current_image_attribute_logits.items():
                        if obj_idx < logits_tensor.shape[0]:
                            predicted_idx = torch.argmax(logits_tensor[obj_idx]).item()
                            predicted_attrs_for_obj[attr_name] = ATTRIBUTE_FILL_MAP[attr_name].get(predicted_idx, 'unknown')

                    for attr_name, attr_map in ATTRIBUTE_FILL_MAP.items():
                        attr_idx = None
                        # Use GT attribute if available in the parsed obj_data (from ground_truth_json_strings)
                        if 'attributes' in obj_data and attr_name in obj_data['attributes']:
                            gt_attr_val = obj_data['attributes'][attr_name]
                            attr_idx = ATTRIBUTE_FILL_MAP_INV[attr_name].get(gt_attr_val)
                            if attr_idx is None: # Handle unknown GT values
                                logger.warning(f"GT: Unknown attribute value '{gt_attr_val}' for type '{attr_name}'.")

                        # If GT not available or invalid, use predicted attribute
                        if attr_idx is None and attr_name in current_image_attribute_logits and obj_idx < current_image_attribute_logits[attr_name].shape[0]:
                            attr_idx = torch.argmax(current_image_attribute_logits[attr_name][obj_idx]).item()
                            attr_idx = min(attr_idx, len(attr_map) - 1) # Clamp to valid range

                        if attr_idx is not None:
                            if self.use_learnable_symbolic_embeddings:
                                attr_embedding = self.symbolic_attribute_embeddings[attr_name](torch.tensor(attr_idx, device=self.device))
                                obj_symbolic_vector.append(attr_embedding)
                            else:
                                one_hot_vector = F.one_hot(torch.tensor(attr_idx, device=self.device), num_classes=len(attr_map)).float()
                                obj_symbolic_vector.append(one_hot_vector)
                        else: # Fallback for missing/unhandled attributes (e.g., if no prediction or GT)
                            if self.use_learnable_symbolic_embeddings:
                                dummy_embedding_dim = self.symbolic_attribute_embeddings[attr_name].embedding_dim
                                obj_symbolic_vector.append(torch.zeros(dummy_embedding_dim, device=self.device))
                            else:
                                obj_symbolic_vector.append(torch.zeros(len(attr_map), device=self.device))
                    
                    if obj_symbolic_vector:
                        symbolic_features_list.append(torch.cat(obj_symbolic_vector, dim=0))
                    else:
                        total_symbolic_dim = sum(self.symbolic_attribute_embeddings[attr_type].embedding_dim
                                                 for attr_type in SYMBOLIC_EMBEDDING_DIMS.keys() if attr_type != 'relation') if self.use_learnable_symbolic_embeddings else sum(len(v) for v in ATTRIBUTE_FILL_MAP.values())
                        symbolic_features_list.append(torch.zeros(total_symbolic_dim, device=self.device))
                
                if symbolic_features_list:
                    symbolic_features_tensor = torch.stack(symbolic_features_list)
                    current_object_features = torch.cat([current_object_features, symbolic_features_tensor], dim=-1)
                    logger.debug(f"Augmented node features with symbolic priors. New shape: {current_object_features.shape}")
                else:
                    logger.warning("No symbolic features generated for node augmentation for this image.")
            
            all_object_features.append(current_object_features)
            
            # --- Relation Graph Construction & Symbolic Edge Feature Augmentation ---
            num_nodes = len(valid_detected_objects_for_patches)
            edge_index = []
            edge_attributes = []

            for src_idx in range(num_nodes):
                for dst_idx in range(num_nodes):
                    if src_idx != dst_idx:
                        edge_index.append([src_idx, dst_idx])
                        
                        relation_type = None
                        # Prioritize GT relation if available
                        if (src_idx, dst_idx) in current_image_gt_relations_map:
                            relation_type = current_image_gt_relations_map[(src_idx, dst_idx)]
                        
                        if relation_type is None: # If no GT relation, infer/predict
                            # This is where get_predicted_relation would be called for real data inference
                            # For synthetic data without explicit GT relation, assume 'none'
                            # or infer based on object properties (e.g., bbox overlap)
                            
                            # Get predicted attributes for src and dst objects
                            src_obj_attrs = {
                                attr_name: ATTRIBUTE_FILL_MAP[attr_name][torch.argmax(current_image_attribute_logits[attr_name][src_idx]).item()]
                                for attr_name in ATTRIBUTE_FILL_MAP if attr_name in current_image_attribute_logits and src_idx < current_image_attribute_logits[attr_name].shape[0]
                            }
                            dst_obj_attrs = {
                                attr_name: ATTRIBUTE_FILL_MAP[attr_name][torch.argmax(current_image_attribute_logits[attr_name][dst_idx]).item()]
                                for attr_name in ATTRIBUTE_FILL_MAP if attr_name in current_image_attribute_logits and dst_idx < current_image_attribute_logits[attr_name].shape[0]
                            }

                            # Call get_predicted_relation (needs to be implemented by user)
                            # For now, it will return 'none' or a basic rule-based one.
                            try:
                                relation_type = get_predicted_relation(
                                    valid_detected_objects_for_patches[src_idx]['bbox'],
                                    valid_detected_objects_for_patches[dst_idx]['bbox'],
                                    src_obj_attrs,
                                    dst_obj_attrs
                                )
                            except NameError:
                                logger.warning("get_predicted_relation not found. Defaulting to 'none'.")
                                relation_type = 'none' # Fallback if function not defined

                        rel_idx = RELATION_MAP_INV.get(relation_type, 0) # Default to 0 ('none')

                        if self.apply_symbolic_priors:
                            if self.use_learnable_symbolic_embeddings and self.symbolic_relation_embedding is not None:
                                rel_embedding = self.symbolic_relation_embedding(torch.tensor(rel_idx, device=self.device))
                                edge_attributes.append(rel_embedding)
                            else:
                                one_hot_vector = F.one_hot(torch.tensor(rel_idx, device=self.device), num_classes=len(RELATION_MAP)).float()
                                edge_attributes.append(one_hot_vector)
                        
            if not edge_index and num_nodes > 0:
                # Add self-loops if no other edges, and handle edge_attributes
                for node_idx in range(num_nodes):
                    edge_index.append([node_idx, node_idx])
                    if self.apply_symbolic_priors:
                        rel_idx = RELATION_MAP_INV.get('none', 0) # Self-loops often have 'none' relation
                        if self.use_learnable_symbolic_embeddings and self.symbolic_relation_embedding is not None:
                            rel_embedding = self.symbolic_relation_embedding(torch.tensor(rel_idx, device=self.device))
                            edge_attributes.append(rel_embedding)
                        else:
                            one_hot_vector = F.one_hot(torch.tensor(rel_idx, device=self.device), num_classes=len(RELATION_MAP)).float()
                            edge_attributes.append(one_hot_vector)

            if not edge_index:
                dummy_x = torch.zeros(1, current_object_features.shape[-1], device=self.device)
                dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=self.device)
                dummy_edge_attr = None
                if self.apply_symbolic_priors:
                    if self.use_learnable_symbolic_embeddings and self.symbolic_relation_embedding is not None:
                        dummy_edge_attr = torch.zeros(1, self.symbolic_relation_embedding.embedding_dim, device=self.device)
                    else:
                        dummy_edge_attr = torch.zeros(1, len(RELATION_MAP), device=self.device)
                graph_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
                if self.apply_symbolic_priors and edge_attributes:
                    edge_attr = torch.stack(edge_attributes)
                    graph_data = Data(x=current_object_features, edge_index=edge_index, edge_attr=edge_attr)
                else:
                    graph_data = Data(x=current_object_features, edge_index=edge_index)
            
            all_graph_data.append(graph_data)
        
        # Batch graph data for GNN
        batched_graph_data = Batch.from_data_list(all_graph_data)
        
        # --- Relational Reasoning (GNN) ---
        node_embeddings, graph_embedding, relation_logits_per_graph = self.relation_gnn(batched_graph_data)
        
        # --- Bongard Problem Classification ---
        bongard_logits = self.bongard_head(graph_embedding)
        
        # Aggregate attribute and relation logits into batch-level tensors
        final_attribute_logits = {}
        for attr_name, logits_list_per_image in batch_attribute_logits_raw.items():
            all_logits_for_attr = [l for l in logits_list_per_image if l.numel() > 0]
            if all_logits_for_attr:
                final_attribute_logits[attr_name] = torch.cat(all_logits_for_attr, dim=0)
            else:
                num_attr_classes = len(ATTRIBUTE_FILL_MAP.get(attr_name, {}))
                final_attribute_logits[attr_name] = torch.empty(0, num_attr_classes, device=self.device)
        
        final_relation_logits = relation_logits_per_graph
        
        final_attribute_gt = {}
        for attr_name, gt_list_of_lists_per_image in batch_attribute_gt_raw.items():
            flat_gt = [item for sublist in gt_list_of_lists_per_image if sublist is not None for item in sublist]
            if flat_gt:
                final_attribute_gt[attr_name] = torch.tensor(flat_gt, dtype=torch.long, device=self.device)
            else:
                final_attribute_gt[attr_name] = torch.empty(0, dtype=torch.long, device=self.device)
        
        final_relation_gt = torch.tensor(batch_relation_gt_raw, dtype=torch.long, device=self.device)
        
        final_bongard_labels_from_gt = torch.tensor(batch_bongard_labels_from_gt, dtype=torch.long, device=self.device)
        
        aggregated_outputs = {
            'attribute_logits': final_attribute_logits,
            'attribute_gt': final_attribute_gt,
            'relation_logits': final_relation_logits,
            'relation_gt': final_relation_gt,
            'image_features_student': graph_embedding, # Graph embedding as student features
            'bongard_labels': final_bongard_labels_from_gt
        }
        return bongard_logits, detected_objects_batch, aggregated_outputs

    def extract_scene_graph(self, detected_objects: List[Dict[str, Any]],
                            attribute_logits: Dict[str, torch.Tensor],
                            relation_logits: torch.Tensor) -> Dict[str, Any]:
        """
        Extracts a structured scene graph from model outputs for a single image.
        Args:
            detected_objects (List[Dict]): List of detected objects (from YOLO/SAM).
            attribute_logits (Dict): Dictionary of attribute logits for each object.
                                     Keys are attribute names, values are tensors [num_objects, num_classes].
            relation_logits (torch.Tensor): Relation logits for all possible pairs [num_pairs, num_relation_types].
        Returns:
            Dict[str, Any]: A dictionary representing the scene graph.
        """
        scene_graph = {
            "objects": [],
            "relations": []
        }

        # Process objects and their attributes
        for obj_idx, obj_data in enumerate(detected_objects):
            obj_scene_data = {
                "id": obj_data.get("id", obj_idx),
                "bbox": obj_data["bbox"],
                "type": obj_data.get("type", "unknown"),
                "confidence": obj_data.get("confidence", 0.0),
                "attributes": {}
            }
            
            # Add predicted attributes
            for attr_name, logits_tensor in attribute_logits.items():
                if obj_idx < logits_tensor.shape[0]: # Ensure index is valid
                    predicted_idx = torch.argmax(logits_tensor[obj_idx]).item()
                    predicted_attr_value = ATTRIBUTE_FILL_MAP[attr_name].get(predicted_idx, "unknown")
                    obj_scene_data["attributes"][attr_name] = predicted_attr_value
            
            scene_graph["objects"].append(obj_scene_data)

        # Process relations
        num_objects = len(detected_objects)
        pair_idx = 0
        for src_idx in range(num_objects):
            for dst_idx in range(num_objects):
                if src_idx != dst_idx:
                    if pair_idx < relation_logits.shape[0]: # Ensure index is valid
                        predicted_rel_idx = torch.argmax(relation_logits[pair_idx]).item()
                        predicted_rel_type = RELATION_MAP.get(predicted_rel_idx, "unknown")
                        
                        scene_graph["relations"].append({
                            "src": detected_objects[src_idx].get("id", src_idx),
                            "dst": detected_objects[dst_idx].get("id", dst_idx),
                            "type": predicted_rel_type,
                            "confidence": F.softmax(relation_logits[pair_idx], dim=-1)[predicted_rel_idx].item()
                        })
                        pair_idx += 1
                    else:
                        logger.warning(f"Relation logits exhausted for pair ({src_idx}, {dst_idx}). Skipping.")
        
        return scene_graph

    def fuse_model(self):
        """
        Fuses modules for quantization. This method needs to be implemented
        based on the specific architecture of your PerceptionModule.
        Example: Fusing Conv-BN-ReLU patterns in the backbone.
        """
        logger.info("Attempting to fuse model modules for quantization.")
        # Example: Fuse Conv-BN-ReLU in the attribute classifier's backbone
        # This requires knowledge of the backbone's internal structure.
        # For a timm model, fusion might be handled by timm's internal methods
        # or require specific manual fusion.
        
        # Common pattern: iterate through named modules and fuse
        # This is a generic example; adapt to your specific backbone (e.g., EfficientNet, MobileNet)
        
        # Fuse attribute classifier backbone if it has a fuse_model method or known layers
        if hasattr(self.attribute_classifier.backbone, 'fuse_model'):
            self.attribute_classifier.backbone.fuse_model()
            logger.info("AttributeClassifier backbone fused.")
        else:
            # Manual fusion for common patterns if backbone doesn't have its own fuse_model
            # This is highly dependent on the backbone's layers.
            # Example for a simple sequential backbone:
            # for i in range(len(self.attribute_classifier.backbone) - 1):
            #     if isinstance(self.attribute_classifier.backbone[i], nn.Conv2d) and \
            #        isinstance(self.attribute_classifier.backbone[i+1], nn.BatchNorm2d):
            #         if i + 2 < len(self.attribute_classifier.backbone) and \
            #            isinstance(self.attribute_classifier.backbone[i+2], nn.ReLU):
            #             torch.quantization.fuse_modules(
            #                 self.attribute_classifier.backbone,
            #                 [str(i), str(i+1), str(i+2)],
            #                 inplace=True
            #             )
            #             logger.debug(f"Fused Conv-BN-ReLU at index {i}.")
            #         elif i + 1 < len(self.attribute_classifier.backbone) and \
            #              isinstance(self.attribute_classifier.backbone[i+1], nn.ReLU): # Conv-ReLU
            #             torch.quantization.fuse_modules(
            #                 self.attribute_classifier.backbone,
            #                 [str(i), str(i+1)],
            #                 inplace=True
            #             )
            #             logger.debug(f"Fused Conv-ReLU at index {i}.")
            logger.warning("No specific fusion logic implemented for AttributeClassifier backbone. Manual fusion might be needed.")

        # You might also want to fuse layers in RelationGNN or BongardHead if they contain Conv/BN/ReLU
        # For GNNs, fusion is less common and depends on the specific layer types (e.g., GCNConv usually doesn't have BN/ReLU in sequence).
        logger.info("Model fusion complete (or skipped where not applicable/defined).")

    def load_weights(self, checkpoint_path: str):
        """Loads model weights from a checkpoint."""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint path not found: {checkpoint_path}")
            return
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Handle DDP state_dict prefix if loading a DDP-saved model into a non-DDP model
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v # Remove 'module.' prefix
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict, strict=False) # strict=False to handle minor mismatches
            logger.info(f"Model weights loaded successfully from {checkpoint_path}.")
        except Exception as e:
            logger.error(f"Failed to load model weights from {checkpoint_path}: {e}")

    def export_onnx(self, output_path: str):
        """
        Exports the model to ONNX format.
        This method needs careful handling, especially for custom layers like GNNs.
        """
        logger.info(f"Attempting to export model to ONNX: {output_path}")
        self.eval() # Set model to evaluation mode
        
        # Create a dummy input tensor
        # Assuming input is a batch of images [B, C, H, W]
        dummy_input_image = torch.randn(1, self.num_channels, self.image_size[0], self.image_size[1], device=self.device)
        
        # Dummy ground_truth_json_strings for inference (can be empty)
        dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})]
        
        # ONNX export for PerceptionModule is complex due to dynamic graph operations
        # (object detection, graph construction).
        # It's usually better to export sub-modules separately if they are static.
        # For the full PerceptionModule, dynamic operations might cause issues.
        # A common approach is to trace the AttributeClassifier and RelationGNN separately.

        # --- Export AttributeClassifier ---
        attr_classifier_onnx_path = output_path.replace(".onnx", "_attribute_classifier.onnx")
        try:
            dummy_attr_input = torch.randn(1, self.num_channels, self.image_size[0], self.image_size[1], device=self.device)
            torch.onnx.export(
                self.attribute_classifier,
                dummy_attr_input,
                attr_classifier_onnx_path,
                input_names=['input_image'],
                output_names=['fill_logits', 'color_logits', 'size_logits', 'orientation_logits', 'shape_logits', 'texture_logits', 'features'],
                dynamic_axes={'input_image': {0: 'batch_size'}},
                opset_version=11,
                verbose=False
            )
            logger.info(f"AttributeClassifier exported successfully to {attr_classifier_onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export AttributeClassifier to ONNX: {e}")

        # --- Export RelationGNN ---
        relation_gnn_onnx_path = output_path.replace(".onnx", "_relation_gnn.onnx")
        try:
            # Dummy input for RelationGNN (node features, edge_index, batch)
            # This is tricky because the graph structure is dynamic.
            # You need a representative static graph for ONNX export.
            # Example: 2 nodes, 2 edges (self-loops)
            dummy_x = torch.randn(2, self.relation_gnn.input_dim, device=self.device)
            dummy_edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=self.device) # Two self-loops
            dummy_batch = torch.tensor([0, 0], dtype=torch.long, device=self.device) # Both in batch 0
            
            # If using edge_attr, create dummy edge_attr
            dummy_edge_attr = None
            if self.apply_symbolic_priors:
                if self.use_learnable_symbolic_embeddings and self.symbolic_relation_embedding is not None:
                    dummy_edge_attr = torch.randn(dummy_edge_index.shape[1], self.symbolic_relation_embedding.embedding_dim, device=self.device)
                else:
                    dummy_edge_attr = torch.randn(dummy_edge_index.shape[1], len(RELATION_MAP), device=self.device)
            
            # Create a dummy Batch object
            dummy_graph_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr)
            dummy_batched_graph_data = Batch.from_data_list([dummy_graph_data])

            # PyTorch Geometric's to_onnx is recommended for GNNs
            # It handles dynamic graph sizes better than torch.onnx.export directly.
            # Ensure pyg_nn.to_onnx is imported and available.
            
            # If pyg_nn.to_onnx is not available, you'd try torch.onnx.export with
            # dynamic_axes for node and edge dimensions, which is more complex.
            if HAS_PYG and hasattr(pyg_nn, 'to_onnx'): # Check if pyg_nn.to_onnx exists
                pyg_nn.to_onnx(
                    self.relation_gnn,
                    dummy_batched_graph_data,
                    relation_gnn_onnx_path,
                    input_names=['x', 'edge_index', 'batch', 'edge_attr'] if dummy_edge_attr is not None else ['x', 'edge_index', 'batch'],
                    output_names=['node_embeddings', 'graph_embedding', 'relation_logits'],
                    dynamic_axes={
                        'x': {0: 'num_nodes_in_batch'},
                        'edge_index': {1: 'num_edges_in_batch'},
                        'batch': {0: 'num_nodes_in_batch'},
                        'edge_attr': {0: 'num_edges_in_batch'} if dummy_edge_attr is not None else {}
                    },
                    opset_version=11
                )
                logger.info(f"RelationGNN exported successfully to {relation_gnn_onnx_path}")
            else:
                logger.warning("PyTorch Geometric's to_onnx not available or HAS_PYG is False. Skipping RelationGNN ONNX export.")
        except Exception as e:
            logger.error(f"Failed to export RelationGNN to ONNX: {e}")

        # --- Export Bongard Head ---
        bongard_head_onnx_path = output_path.replace(".onnx", "_bongard_head.onnx")
        try:
            dummy_bongard_input = torch.randn(1, self.bongard_head.in_features, device=self.device)
            torch.onnx.export(
                self.bongard_head,
                dummy_bongard_input,
                bongard_head_onnx_path,
                input_names=['graph_embedding'],
                output_names=['bongard_logits'],
                dynamic_axes={'graph_embedding': {0: 'batch_size'}},
                opset_version=11
            )
            logger.info(f"BongardHead exported successfully to {bongard_head_onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export BongardHead to ONNX: {e}")

        logger.info("ONNX export process completed for sub-modules.")


# --- Ensemble Utilities (MetaLearner and calculate_model_weights) ---
class MetaLearner(nn.Module):
    """
    A simple neural network (MLP) to act as the meta-learner in a stacking ensemble.
    It takes the concatenated logits/probabilities from base models as input
    and learns to combine them to make a final prediction.
    Args:
        input_dim (int): The total input dimension, which should be
                         (num_base_models * num_classes_per_base_model).
        num_classes (int): The number of final output classes for the Bongard Problem.
        hidden_dim (int): The dimension of the hidden layer in the MLP.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Added dropout for regularization
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        logger.info(f"MetaLearner initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}.")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MetaLearner.
        Args:
            x (torch.Tensor): Concatenated predictions (logits or probabilities)
                              from base models. Shape: [batch_size, input_dim].
        Returns:
            torch.Tensor: Final logits for the Bongard Problem classification.
                          Shape: [batch_size, num_classes].
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
def calculate_model_weights(
    member_metrics: List[Dict[str, Any]],
    metric_name: str = 'val_accuracy',
    minimize_metric: bool = False
) -> List[float]:
    """
    Calculates weights for ensemble members based on their performance metrics.
    Args:
        member_metrics (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                contains metrics for an ensemble member
                                                (e.g., from _run_single_training_session).
        metric_name (str): The name of the metric to use for weighting (e.g., 'val_accuracy', 'val_loss').
        minimize_metric (bool): If True, a lower value of the metric indicates better performance
                                (e.g., for 'val_loss'). If False, a higher value is better.
    Returns:
        List[float]: A list of weights for each ensemble member, normalized to sum to 1.
    """
    if not member_metrics:
        logger.warning("No member metrics provided. Returning equal weights.")
        return []
    metric_values = []
    for metrics_dict in member_metrics:
        if metric_name not in metrics_dict:
            logger.warning(f"Metric '{metric_name}' not found in one of the member's metrics. Using default weight 1.0.")
            metric_values.append(1.0) # Default to 1.0 if metric is missing
        else:
            metric_values.append(metrics_dict[metric_name])
    metric_values_np = np.array(metric_values)
    if minimize_metric:
        # For metrics like loss, smaller is better, so use inverse
        # Add a small epsilon to avoid division by zero for very small losses
        weights = 1.0 / (metric_values_np + 1e-6)
        logger.info(f"Calculated weights based on inverse of '{metric_name}' (minimize).")
    else:
        # For metrics like accuracy, larger is better
        weights = metric_values_np
        logger.info(f"Calculated weights based on '{metric_name}' (maximize).")
    # Normalize weights to sum to 1
    total_weight = np.sum(weights)
    if total_weight == 0:
        logger.warning("Sum of weights is zero. Returning equal weights.")
        return [1.0 / len(member_metrics)] * len(member_metrics)
    
    normalized_weights = weights / total_weight
    logger.info(f"Normalized ensemble weights: {normalized_weights.tolist()}")
    return normalized_weights.tolist()
# --- Main Training and Evaluation Loop (Updated) ---

def _run_single_training_session_ensemble(
    current_config: Dict[str, Any],
    member_id: int,
    random_seed: int,
    output_dir: str = './checkpoints',
    epochs_override: Optional[int] = None,
    train_image_paths: List[str] = None,
    train_labels: np.ndarray = None,
    train_difficulty_scores: Optional[np.ndarray] = None,
    val_image_paths: List[str] = None,
    val_labels: np.ndarray = None,
    val_difficulty_scores: Optional[np.ndarray] = None,
    teacher_model: Optional[Any] = None, # Can be a single model or a list of models
    all_members_val_predictions_logits: Optional[List[np.ndarray]] = None, # For stacked teacher
    all_members_val_labels: Optional[List[np.ndarray]] = None, # For stacked teacher
    meta_learner_path: Optional[str] = None, # Path to meta-learner for stacked teacher
    shard_id: int = 0, # For DDP
    num_shards: int = 1 # For DDP
) -> Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Runs a single training session for a PerceptionModule model.
    This function is designed to be called by the ensemble orchestrator or DDP worker.
    Args:
        current_config (Dict): The configuration for this specific training session.
        member_id (int): Unique ID for the ensemble member (for naming checkpoints/logs).
        random_seed (int): Random seed for reproducibility of this member.
        output_dir (str): Directory to save model checkpoints.
        epochs_override (Optional[int]): If provided, overrides 'model.epochs' in config.
        train_image_paths (List[str]): Paths to training images.
        train_labels (np.ndarray): Labels for training images.
        train_difficulty_scores (Optional[np.ndarray]): Difficulty scores for training images.
        val_image_paths (List[str]): Paths to validation images.
        val_labels (np.ndarray): Labels for validation images.
        val_difficulty_scores (np.ndarray): Difficulty scores for validation images.
        teacher_model (Optional[Any]): A pre-trained teacher model (e.g., an ensemble) for knowledge distillation.
        all_members_val_predictions_logits (Optional[List[np.ndarray]]): Validation predictions from base models for stacking.
        all_members_val_labels (Optional[List[np.ndarray]]): Validation labels for stacking.
        meta_learner_path (Optional[str]): Path to the meta-learner model for stacked teacher.
        shard_id (int): Current GPU rank for DDP.
        num_shards (int): Total number of GPUs for DDP.
    Returns:
        Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
            - Path to the best saved model checkpoint.
            - All validation predictions (logits) from the best model.
            - All validation true labels.
            - Dictionary of best validation metrics.
    """
    set_seed(random_seed)
    logger.info(f"--- Starting Training Session for Member {member_id} (Seed: {random_seed}) ---")
    
    is_ddp_initialized = torch.distributed.is_initialized()
    current_rank = torch.distributed.get_rank() if is_ddp_initialized else 0
    is_main_process = (current_rank == 0)

    epochs = epochs_override if epochs_override is not None else current_config['model']['epochs']
    logger.info(f"Training for {epochs} epochs.")

    # Initialize TensorBoard writer and WandB only for main process
    writer = None
    if is_main_process:
        log_dir = os.path.join('runs', datetime.now().strftime(f'member_{member_id}_%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logs for member {member_id} at: {log_dir}")
        if current_config['training']['use_wandb'] and HAS_WANDB:
            import wandb
            wandb.init(project="Bongard_Perception_Ensemble",
                       group="ensemble_training",
                       name=f"member_{member_id}_seed_{random_seed}",
                       config=current_config,
                       reinit=True)
            logger.info(f"WandB initialized for member {member_id}.")

    # Data Loaders (using DALI)
    batch_size = current_config['model']['batch_size']
    num_workers = current_config['data']['dataloader_num_workers']
    
    train_external_source_iterator = None
    val_external_source_iterator = None
    
    if current_config['data']['use_synthetic_data']:
        logger.info("Using synthetic data for training and validation.")
        train_synthetic_dataset = BongardSyntheticDataset(current_config['data']['synthetic_samples'],
                                                          image_size=current_config['data']['image_size'],
                                                          num_classes=current_config['model']['num_classes'])
        val_synthetic_dataset = BongardSyntheticDataset(current_config['data']['synthetic_samples'] // 5,
                                                        image_size=current_config['data']['image_size'],
                                                        num_classes=current_config['model']['num_classes'])
        
        train_external_source_iterator = BongardExternalSource(train_synthetic_dataset, batch_size, shuffle=True)
        val_external_source_iterator = BongardExternalSource(val_synthetic_dataset, batch_size, shuffle=False)
        
        # For synthetic data, file_list and labels_list are not directly used by fn.readers.file
        train_file_list = []
        train_labels_list = np.array([])
        val_file_list = []
        val_labels_list = np.array([])
    else:
        # For real data, use the provided paths and labels
        train_file_list = train_image_paths
        train_labels_list = train_labels
        val_file_list = val_image_paths
        val_labels_list = val_labels

    # Refactored DALI Loader Creation
    # build_dali_loader now returns (loader, curriculum_sampler_instance)
    train_loader, train_curriculum_sampler = build_dali_loader(
        file_list=train_file_list,
        labels_list=train_labels_list,
        config=current_config,
        mode='train',
        external_source_iterator=train_external_source_iterator,
        difficulty_scores_list=train_difficulty_scores, # Pass for real data
        shard_id=current_rank,
        num_shards=num_shards
    )
    logger.info("DALI training loader initialized.")

    val_loader, _ = build_dali_loader( # No curriculum sampler for validation
        file_list=val_file_list,
        labels_list=val_labels_list,
        config=current_config,
        mode='val',
        external_source_iterator=val_external_source_iterator,
        difficulty_scores_list=val_difficulty_scores, # Pass for real data
        shard_id=current_rank,
        num_shards=num_shards
    )
    logger.info("DALI validation loader initialized.")

    # Model, Optimizer, Scheduler
    # Model is already DDP-wrapped if called from ddp_train_worker
    if not is_ddp_initialized:
        object_detector = RealObjectDetector(model_path=current_config['model']['object_detector_model_path'])
        model = PerceptionModule(current_config, object_detector).to(DEVICE)
    else:
        # Model is passed in as DDP-wrapped from ddp_train_worker
        # Ensure it's on the correct device for this rank
        model = model.to(torch.device(f'cuda:{current_rank}'))
    logger.info("PerceptionModule model initialized.")

    # Initialize SWA model if enabled
    if current_config['training']['use_swa']:
        swa_model = swa_utils.AveragedModel(model.module if is_ddp_initialized else model) # Wrap base model
        logger.info("SWA AveragedModel initialized.")

    # Quantization Aware Training (QAT) setup
    if current_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION:
        logger.info("Setting up Quantization Aware Training (QAT).")
        model_to_prepare = model.module if is_ddp_initialized else model
        model_to_prepare.eval() # Model must be in eval mode for fusion
        if hasattr(model_to_prepare, 'fuse_model'):
            model_to_prepare.fuse_model()
            logger.info("Model modules fused for QAT.")
        else:
            logger.warning("Model does not have a fuse_model method. Fusion skipped for QAT.")
        
        model_to_prepare.qconfig = tq.get_default_qat_qconfig('fbgemm')
        tq.prepare_qat(model_to_prepare, inplace=True)
        logger.info("Model prepared for QAT.")
        model_to_prepare.train()

    # Loss functions
    criterion_bongard = LabelSmoothingCrossEntropy(smoothing=current_config['training']['label_smoothing_epsilon'])
    logger.info(f"Using CrossEntropyLoss with label_smoothing_epsilon={current_config['training']['label_smoothing_epsilon']}.")
    criterion_attr = nn.CrossEntropyLoss()
    criterion_rel = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_name = current_config['model']['optimizer']
    params_to_optimize = model.parameters() # For DDP, params are managed correctly
    
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(params_to_optimize, lr=current_config['model']['initial_learning_rate'],
                                weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(params_to_optimize, lr=current_config['model']['initial_learning_rate'],
                              momentum=0.9, weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'SophiaG' and HAS_SOPHIA:
        from sophia import SophiaG
        optimizer = SophiaG(params_to_optimize, lr=current_config['model']['initial_learning_rate'],
                            weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'Lion' and HAS_TIMM_OPTIM:
        from timm.optim import Lion # Assuming timm.optim is where Lion is
        optimizer = Lion(params_to_optimize, lr=current_config['model']['initial_learning_rate'],
                         weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'MADGRAD' and HAS_TIMM_OPTIM:
        from timm.optim import MADGRAD # Assuming timm.optim is where MADGRAD is
        optimizer = MADGRAD(params_to_optimize, lr=current_config['model']['initial_learning_rate'],
                            weight_decay=current_config['training']['weight_decay'])
    else:
        logger.warning(f"Optimizer '{optimizer_name}' not supported or not available. Falling back to AdamW.")
        optimizer = optim.AdamW(params_to_optimize, lr=current_config['model']['initial_learning_rate'],
                                weight_decay=current_config['training']['weight_decay'])
    
    if current_config['training']['use_sam_optimizer'] and HAS_SAM:
        from sam_pytorch import SAM # Assuming sam_pytorch is the library
        optimizer = SAM(params_to_optimize, optimizer, rho=0.05, adaptive=True)
        logger.info("Wrapped optimizer with SAM (Sharpness-Aware Minimization).")
    logger.info(f"Optimizer initialized: {optimizer_name}.")

    # Learning Rate Scheduler
    scheduler = None
    if current_config['model']['scheduler'] == 'OneCycleLR':
        total_samples = len(train_image_paths) if not current_config['data']['use_synthetic_data'] else current_config['data']['synthetic_samples']
        total_steps = math.ceil(total_samples / batch_size / num_shards) * epochs # Adjust total_steps for DDP
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=current_config['model']['max_learning_rate'],
            total_steps=total_steps,
            pct_start=current_config['training']['onecycle_pct_start'],
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        logger.info(f"Using OneCycleLR scheduler with max_lr={current_config['model']['max_learning_rate']} and total_steps={total_steps}.")
    elif current_config['model']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=current_config['training']['lr_scheduler_factor'],
            patience=current_config['training']['lr_scheduler_patience'],
            verbose=True
        )
        logger.info(f"Using ReduceLROnPlateau scheduler.")

    scaler = GradScaler(enabled=current_config['training']['use_amp'])
    logger.info(f"AMP scaler initialized (enabled={current_config['training']['use_amp']}).")

    # Early Stopping
    best_val_loss = float('inf')
    best_val_accuracy = -float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f'member_{member_id}_{current_config["training"]["model_checkpoint_name"]}')

    all_val_predictions_logits = []
    all_val_labels = []

    augmenter = MixupCutmix(
        mixup_alpha=current_config['training']['mixup_alpha'],
        cutmix_alpha=current_config['training']['cutmix_alpha'],
        mixup_cutmix_ratio=current_config['training']['mixup_cutmix_ratio']
    )
    logger.info(f"AugmentMix initialized with mixup_alpha={augmenter.mixup_alpha}, cutmix_alpha={augmenter.cutmix_alpha}.")

    # Initialize KnowledgeReplayBuffer (now PER)
    replay_buffer = None
    if current_config['training']['knowledge_replay_enabled']:
        replay_buffer = KnowledgeReplayBuffer(
            capacity=current_config['training']['replay_buffer_size'],
            alpha=current_config['training'].get('replay_alpha', 0.6),
            beta_start=current_config['training'].get('replay_beta_start', 0.4),
            beta_frames=epochs * len(train_loader) # Anneal beta over total steps
        )
        logger.info("Prioritized Experience Replay buffer initialized.")

    # Training Loop
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples_epoch = 0
        
        # Set epoch for DALI and CurriculumSampler
        if isinstance(train_loader.iterator, CurriculumSampler):
            train_loader.iterator.set_epoch(epoch)
            # For real data, if CurriculumSampler updates file_list, you might need to rebuild DALI pipeline
            # if not current_config['data']['use_synthetic_data']:
            #     epoch_train_paths, epoch_train_labels = train_loader.iterator.get_epoch_data()
            #     train_pipeline.args.file_list = epoch_train_paths
            #     train_pipeline.args.labels_list = epoch_train_labels
            #     train_pipeline.build() # Rebuild pipeline with new file list
            #     train_loader.reset() # Reset iterator
        
        train_loader.reset() # Always reset loader for new epoch

        # QAT specific: enable observer for last few epochs
        if current_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION and epoch >= current_config['training']['qat_start_epoch']:
            if is_main_process: logger.info(f"Epoch {epoch}: Enabling QAT observers.")
            model_to_prepare = model.module if is_ddp_initialized else model
            model_to_prepare.apply(tq.enable_observer)
            model_to_prepare.apply(tq.enable_fake_quant)
        
        # Profiler setup
        profiler = None
        if current_config['training']['enable_profiler'] and epoch == 0 and is_main_process:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=current_config['training']['profiler_schedule_wait'],
                    warmup=current_config['training']['profiler_schedule_warmup'],
                    active=current_config['training']['profiler_schedule_active'],
                    repeat=current_config['training']['profiler_schedule_repeat']
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                with_stack=True
            )
            profiler.start()
            if is_main_process: logger.info("PyTorch profiler started.")

        # Store per-sample losses for dynamic difficulty update
        per_batch_original_indices = []
        per_batch_per_sample_losses = []

        for batch_idx, data in enumerate(train_loader):
            images_view1 = data['view1'].permute(0, 3, 1, 2).to(DEVICE) # NCHW
            images_view2 = data['view2'].permute(0, 3, 1, 2).to(DEVICE) # NCHW
            labels_bongard = data['labels'].squeeze(-1).to(DEVICE) # Remove last dim from DALI
            gts_json_strings_batch = [s.decode('utf-8') for s in data['gts_json'].cpu().numpy()]
            difficulties_batch = data['difficulties'].to(DEVICE) # From DALI
            original_indices_batch = data['original_indices'].to(DEVICE) # NEW: From DALI

            # Ground-Truth JSON Validation and Label Extraction (for synthetic data)
            for i_sample, gt_json_str in enumerate(gts_json_strings_batch):
                try:
                    gt_dict = json.loads(gt_json_str)
                    jsonschema.validate(instance=gt_dict, schema=BONGARD_GT_JSON_SCHEMA)
                    if current_config['data']['use_synthetic_data']:
                        labels_bongard[i_sample] = torch.tensor(gt_dict['bongard_label'], dtype=torch.long, device=DEVICE)
                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                    if is_main_process: logger.error(f"Failed to decode/validate GT JSON for sample {i_sample} in batch: {e}. Using dummy label 0.")
                    labels_bongard[i_sample] = torch.tensor(0, dtype=torch.long, device=DEVICE)

            # Apply MixUp/CutMix
            images_view1_aug, mixinfo = augmenter(images_view1, labels_bongard)
            labels_a, labels_b, lam, mode_aug = mixinfo

            # Forward pass
            optimizer.zero_grad() # Zero gradients at start of accumulation step
            with autocast(enabled=current_config['training']['use_amp']):
                bongard_logits, detected_objects_view1, aggregated_outputs_view1 = model(images_view1_aug, gts_json_strings_batch)
                
                # Bongard Classification Loss (per-sample for dynamic difficulty update)
                # Use reduction='none' to get per-sample loss
                per_sample_bongard_loss = criterion_bongard(bongard_logits, labels_a) # labels_a for mixup/cutmix
                if mode_aug == 'mixup' or mode_aug == 'cutmix':
                    per_sample_bongard_loss = lam * per_sample_bongard_loss + (1 - lam) * criterion_bongard(bongard_logits, labels_b)
                
                # Apply difficulty weights to Bongard loss
                # This assumes difficulties_batch are per-sample weights from DALI/sampler
                weighted_bongard_loss = (per_sample_bongard_loss * difficulties_batch).mean()
                total_loss_batch = weighted_bongard_loss

                # Attribute and Relation Losses (Joint Symbolic Loss)
                if 'attribute_logits' in aggregated_outputs_view1 and 'attribute_gt' in aggregated_outputs_view1:
                    for attr_name, attr_logits in aggregated_outputs_view1['attribute_logits'].items():
                        if attr_name in aggregated_outputs_view1['attribute_gt'] and aggregated_outputs_view1['attribute_gt'][attr_name].numel() > 0:
                            attr_gt = aggregated_outputs_view1['attribute_gt'][attr_name]
                            if attr_logits.shape[0] == attr_gt.shape[0]: # Ensure shapes match
                                total_loss_batch += current_config['training']['attribute_loss_weight'] * criterion_attr(attr_logits, attr_gt)
                
                if 'relation_logits' in aggregated_outputs_view1 and 'relation_gt' in aggregated_outputs_view1:
                    rel_logits = aggregated_outputs_view1['relation_logits']
                    rel_gt = aggregated_outputs_view1['relation_gt']
                    if rel_logits.shape[0] == rel_gt.shape[0]: # Ensure shapes match
                        total_loss_batch += current_config['training']['relation_loss_weight'] * criterion_rel(rel_logits, rel_gt)

                # Feature Consistency Loss
                if current_config['training']['feature_consistency_alpha'] > 0:
                    _, _, aggregated_outputs_view2 = model(images_view2, gts_json_strings_batch)
                    consistency_loss_features = 0.0
                    if 'image_features_student' in aggregated_outputs_view1 and 'image_features_student' in aggregated_outputs_view2:
                        feature_consistency_criterion = FeatureConsistencyLoss(loss_type=current_config['training']['feature_consistency_loss_type'])
                        consistency_loss_features = feature_consistency_criterion(
                            aggregated_outputs_view1['image_features_student'],
                            aggregated_outputs_view2['image_features_student']
                        )
                    total_loss_batch += current_config['training']['feature_consistency_alpha'] * consistency_loss_features
                    if is_main_process: logger.debug(f"Added feature consistency loss: {consistency_loss_features.item():.4f}")

                # Symbolic Consistency Loss
                if current_config['training']['symbolic_consistency_alpha'] > 0:
                    # aggregated_outputs_view2 is already computed for feature consistency
                    symbolic_cons_loss = symbolic_consistency_loss(
                        aggregated_outputs_view1['attribute_logits'],
                        aggregated_outputs_view2['attribute_logits']
                    )
                    total_loss_batch += current_config['training']['symbolic_consistency_alpha'] * symbolic_cons_loss
                    if is_main_process: logger.debug(f"Added symbolic consistency loss: {symbolic_cons_loss.item():.4f}")

                # Knowledge Distillation (if teacher model provided)
                if teacher_model is not None and current_config['training']['use_knowledge_distillation']:
                    # Get soft targets from teacher
                    teacher_logits = None
                    if isinstance(teacher_model, list): # Ensemble teacher
                        teacher_logits_list = []
                        for t_model in teacher_model:
                            t_model.eval() # Ensure teacher is in eval mode
                            t_logits, _, _ = t_model(images_view1, gts_json_strings_batch)
                            teacher_logits_list.append(t_logits)
                        teacher_logits = torch.mean(torch.stack(teacher_logits_list, dim=0), dim=0)
                    elif isinstance(teacher_model, MetaLearner): # Stacked teacher (meta-learner)
                        # This path is for when the teacher is a MetaLearner itself,
                        # implying base model predictions need to be passed to it.
                        # This would be complex to do on-the-fly here.
                        # For now, assume teacher_model is a list of PerceptionModules or a single one.
                        # A stacked teacher would typically be used in inference, not directly here.
                        logger.warning("Stacked teacher not directly supported in distillation training loop. Using simple average of base models.")
                        teacher_logits = torch.mean(torch.stack([t_model(images_view1, gts_json_strings_batch)[0] for t_model in teacher_model.base_models]), dim=0) # Conceptual
                    else: # Single teacher model
                        teacher_model.eval()
                        teacher_logits, _, _ = teacher_model(images_view1, gts_json_strings_batch)
                    
                    if teacher_logits is not None:
                        dist_loss = DistillationLoss(
                            alpha=current_config['training']['distillation_alpha'],
                            temperature=current_config['training']['distillation_temperature'],
                            base_loss_fn=criterion_bongard # Use hard loss for hard component
                        )(student_logits=bongard_logits, teacher_logits=teacher_logits, labels=labels_bongard)
                        total_loss_batch = dist_loss # Distillation loss replaces main loss
                        if is_main_process: logger.debug(f"Applied knowledge distillation loss: {dist_loss.item():.4f}")
                    else:
                        if is_main_process: logger.warning("Teacher logits could not be obtained for distillation.")

            # Scale loss for gradient accumulation
            total_loss_batch = total_loss_batch / current_config['training']['gradient_accumulation_steps']
            scaler.scale(total_loss_batch).backward()

            # --- Knowledge Replay (PER) ---
            if current_config['training']['knowledge_replay_enabled'] and replay_buffer is not None:
                # Add current batch experiences to replay buffer
                per_sample_bongard_loss_for_replay = per_sample_bongard_loss.detach().cpu().numpy()
                for i_sample in range(images_view1_aug.shape[0]):
                    experience = {
                        'image': images_view1_aug[i_sample].cpu(),
                        'label': labels_bongard[i_sample].cpu(),
                        'gts_json': gts_json_strings_batch[i_sample] # Store original GT JSON for replay
                    }
                    replay_buffer.add(experience, per_sample_bongard_loss_for_replay[i_sample])

                # Sample and train from replay buffer
                num_replay_samples = int(batch_size * current_config['training']['replay_batch_size_ratio'])
                if num_replay_samples > 0 and len(replay_buffer) >= num_replay_samples:
                    replay_experiences, replay_indices, importance_weights = replay_buffer.sample(num_replay_samples)
                    importance_weights_tensor = torch.tensor(importance_weights, dtype=torch.float32, device=DEVICE)

                    replay_images = torch.stack([exp['image'] for exp in replay_experiences]).to(DEVICE)
                    replay_labels = torch.stack([exp['label'] for exp in replay_experiences]).to(DEVICE)
                    replay_gts_json = [exp['gts_json'] for exp in replay_experiences]

                    with autocast(enabled=current_config['training']['use_amp']):
                        replay_bongard_logits, _, _ = model(replay_images, replay_gts_json)
                        replay_loss_per_sample = criterion_bongard(replay_bongard_logits, replay_labels, reduction='none')
                        replay_loss = (replay_loss_per_sample * importance_weights_tensor).mean()
                    
                    total_loss_batch += replay_loss / current_config['training']['gradient_accumulation_steps'] # Add to accumulated loss
                    
                    # Update priorities in replay buffer
                    new_replay_errors = replay_loss_per_sample.detach().cpu().numpy()
                    replay_buffer.update_priorities(replay_indices, new_replay_errors)
                    if is_main_process: logger.debug(f"Added replay loss: {replay_loss.item():.4f}")

            if (batch_idx + 1) % current_config['training']['gradient_accumulation_steps'] == 0:
                if current_config['training']['use_sam_optimizer']:
                    scaler.unscale_(optimizer)
                    if current_config['training'].get('max_grad_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), current_config['training']['max_grad_norm'])
                    optimizer.first_step(zero_grad=True)
                    
                    with autocast(enabled=current_config['training']['use_amp']):
                        bongard_logits_second_step, _, aggregated_outputs_second_step = model(images_view1_aug, gts_json_strings_batch)
                        
                        loss_bongard_second_step_per_sample = criterion_bongard(bongard_logits_second_step, labels_a)
                        if mode_aug == 'mixup' or mode_aug == 'cutmix':
                            loss_bongard_second_step_per_sample = lam * loss_bongard_second_step_per_sample + (1 - lam) * criterion_bongard(bongard_logits_second_step, labels_b)
                        
                        weighted_bongard_loss_second_step = (loss_bongard_second_step_per_sample * difficulties_batch).mean()
                        total_loss_second_step = weighted_bongard_loss_second_step

                        if 'attribute_logits' in aggregated_outputs_second_step and 'attribute_gt' in aggregated_outputs_second_step:
                            for attr_name, attr_logits in aggregated_outputs_second_step['attribute_logits'].items():
                                if attr_name in aggregated_outputs_second_step['attribute_gt'] and aggregated_outputs_second_step['attribute_gt'][attr_name].numel() > 0:
                                    attr_gt = aggregated_outputs_second_step['attribute_gt'][attr_name]
                                    if attr_logits.shape[0] == attr_gt.shape[0]:
                                        total_loss_second_step += current_config['training']['attribute_loss_weight'] * criterion_attr(attr_logits, attr_gt)
                        
                        if 'relation_logits' in aggregated_outputs_second_step and 'relation_gt' in aggregated_outputs_second_step:
                            rel_logits = aggregated_outputs_second_step['relation_logits']
                            rel_gt = aggregated_outputs_second_step['relation_gt']
                            if rel_logits.shape[0] == rel_gt.shape[0]:
                                total_loss_second_step += current_config['training']['relation_loss_weight'] * criterion_rel(rel_logits, rel_gt)
                        
                        if current_config['training']['feature_consistency_alpha'] > 0:
                            _, _, aggregated_outputs_view2_second_step = model(images_view2, gts_json_strings_batch)
                            feature_consistency_criterion_second_step = FeatureConsistencyLoss(loss_type=current_config['training']['feature_consistency_loss_type'])
                            consistency_loss_features_second_step = feature_consistency_criterion_second_step(
                                aggregated_outputs_second_step['image_features_student'],
                                aggregated_outputs_view2_second_step['image_features_student']
                            )
                            total_loss_second_step += current_config['training']['feature_consistency_alpha'] * consistency_loss_features_second_step
                        
                        if current_config['training']['symbolic_consistency_alpha'] > 0:
                            symbolic_cons_loss_second_step = symbolic_consistency_loss(
                                aggregated_outputs_second_step['attribute_logits'],
                                aggregated_outputs_view2_second_step['attribute_logits']
                            )
                            total_loss_second_step += current_config['training']['symbolic_consistency_alpha'] * symbolic_cons_loss_second_step
                        
                        if teacher_model is not None and current_config['training']['use_knowledge_distillation']:
                            teacher_logits_second_step = None
                            if isinstance(teacher_model, list):
                                teacher_logits_list_second_step = []
                                for t_model in teacher_model:
                                    t_model.eval()
                                    t_logits, _, _ = t_model(images_view1, gts_json_strings_batch)
                                    teacher_logits_list_second_step.append(t_logits)
                                teacher_logits_second_step = torch.mean(torch.stack(teacher_logits_list_second_step, dim=0), dim=0)
                            else:
                                teacher_model.eval()
                                teacher_logits_second_step, _, _ = teacher_model(images_view1, gts_json_strings_batch)
                            
                            if teacher_logits_second_step is not None:
                                dist_loss_second_step = DistillationLoss(
                                    alpha=current_config['training']['distillation_alpha'],
                                    temperature=current_config['training']['distillation_temperature'],
                                    base_loss_fn=criterion_bongard
                                )(student_logits=bongard_logits_second_step, teacher_logits=teacher_logits_second_step, labels=labels_bongard)
                                total_loss_second_step = dist_loss_second_step
                    
                    total_loss_second_step = total_loss_second_step / current_config['training']['gradient_accumulation_steps']
                    scaler.scale(total_loss_second_step).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    if current_config['training'].get('max_grad_norm', 0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), current_config['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if scheduler and current_config['model']['scheduler'] == 'OneCycleLR':
                    scheduler.step()
            
            total_loss += total_loss_batch.item() * current_config['training']['gradient_accumulation_steps']
            
            _, predicted = torch.max(bongard_logits.data, 1)
            correct_predictions += (predicted == labels_bongard).sum().item()
            total_samples_epoch += labels_bongard.size(0)

            # Store per-sample losses and original indices for dynamic difficulty update
            per_batch_original_indices.extend(original_indices_batch.cpu().numpy())
            per_batch_per_sample_losses.extend(per_sample_bongard_loss.detach().cpu().numpy())

            if profiler:
                profiler.step()
            if current_config['training']['validation_frequency_batches'] > 0 and \
               (batch_idx + 1) % current_config['training']['validation_frequency_batches'] == 0:
                if is_main_process: logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: Performing intermediate validation.")
                val_loss_intermediate, val_accuracy_intermediate, _, _, _ = _validate_model_ensemble(
                    model, val_loader, criterion_bongard, current_config,
                    shard_id=shard_id, num_shards=num_shards
                )
                if is_main_process: logger.info(f"Intermediate Val Loss: {val_loss_intermediate:.4f}, Val Accuracy: {val_accuracy_intermediate:.4f}")
                model.train()
        
        if profiler:
            profiler.stop()
            if is_main_process: logger.info("PyTorch profiler stopped.")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples_epoch if total_samples_epoch > 0 else 0.0
        if is_main_process:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            if current_config['training']['use_wandb'] and HAS_WANDB:
                import wandb
                wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_accuracy, "epoch": epoch})

        # --- Dynamic Difficulty Update ---
        if current_config['training']['dynamic_difficulty_update'] and train_curriculum_sampler is not None:
            # Convert lists to numpy arrays
            all_indices_np = np.array(per_batch_original_indices)
            all_losses_np = np.array(per_batch_per_sample_losses)
            
            # Simple difficulty update: higher loss = higher difficulty
            # You might want to smooth these or apply a moving average
            train_curriculum_sampler.update_difficulty_scores(all_indices_np, all_losses_np)
            if is_main_process: logger.info(f"Updated difficulty scores for {len(all_indices_np)} samples.")

        # Validation
        val_loss, val_accuracy, val_predictions_logits, val_true_labels, misclassified_samples_info = \
            _validate_model_ensemble(model, val_loader, criterion_bongard, current_config,
                                     shard_id=shard_id, num_shards=num_shards)
        
        if is_main_process:
            logger.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            if current_config['training']['use_wandb'] and HAS_WANDB:
                import wandb
                wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch})
        
        # Collect validation predictions and labels for stacking (only from main process)
        if is_main_process:
            all_val_predictions_logits.extend(val_predictions_logits)
            all_val_labels.extend(val_true_labels)

        # SWA update
        if current_config['training']['use_swa'] and epoch >= int(epochs * current_config['training']['swa_start_epoch_ratio']):
            swa_model.update_parameters(model.module if is_ddp_initialized else model)
            if is_main_process: logger.debug(f"SWA model updated at epoch {epoch}.")
            if current_config['training']['swa_lr'] is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_config['training']['swa_lr']
                if is_main_process: logger.debug(f"SWA LR set to {current_config['training']['swa_lr']}.")

        # Early Stopping check (only for main process)
        if is_main_process:
            monitor_metric = val_loss
            if current_config['training']['early_stopping_monitor_metric'] == 'val_accuracy':
                monitor_metric = -val_accuracy
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving model.")
                    torch.save(model.module.state_dict() if is_ddp_initialized else model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    logger.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{current_config['training']['early_stopping_patience']}")
            else:
                if val_loss < best_val_loss - current_config['training']['early_stopping_min_delta']:
                    best_val_loss = val_loss
                    patience_counter = 0
                    logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
                    torch.save(model.module.state_dict() if is_ddp_initialized else model.state_dict(), best_model_path)
                else:
                    patience_counter += 1
                    logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{current_config['training']['early_stopping_patience']}")
            
            if scheduler and current_config['model']['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            
            if patience_counter >= current_config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break
            
            # Save symbolic outputs periodically
            if (epoch + 1) % current_config['training']['save_symbolic_outputs_interval_epochs'] == 0:
                logger.info(f"Saving example structured symbolic outputs at epoch {epoch+1}.")
                sample_val_paths = val_image_paths[:min(5, len(val_image_paths))]
                
                if sample_val_paths:
                    sample_pil_images = [Image.open(p).convert('RGB') for p in sample_val_paths]
                    
                    model.eval()
                    with torch.no_grad():
                        dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})] * len(sample_pil_images)
                        bongard_logits_sample, detected_objects_sample_batch, aggregated_outputs_sample = model(
                            torch.stack([T.ToTensor()(img).to(DEVICE) for img in sample_pil_images]),
                            dummy_gts_json
                        )
                    model.train()
                    
                    json_serializable_outputs = []
                    for i_img in range(len(sample_pil_images)):
                        single_image_attr_logits = {k: v[i_img].unsqueeze(0) for k, v in aggregated_outputs_sample['attribute_logits'].items()}
                        single_image_rel_logits = {k: v[i_img].unsqueeze(0) for k, v in aggregated_outputs_sample['relation_logits'].items()}
                        
                        # Use model.module.extract_scene_graph if DDP
                        extract_sg_func = model.module.extract_scene_graph if is_ddp_initialized else model.extract_scene_graph
                        
                        scene_graph = extract_sg_func(
                            detected_objects_sample_batch[i_img],
                            single_image_attr_logits,
                            single_image_rel_logits
                        )
                        
                        scene_graph["image_path"] = sample_val_paths[i_img]
                        scene_graph["predicted_bongard_label"] = int(torch.argmax(bongard_logits_sample[i_img]).item())
                        scene_graph["bongard_prediction_probs"] = F.softmax(bongard_logits_sample[i_img], dim=-1).cpu().numpy().tolist()
                        
                        json_serializable_outputs.append(scene_graph)
                    output_json_path = os.path.join(current_config['training']['checkpoint_dir'], f"symbolic_outputs_member_{member_id}_epoch_{epoch+1}.json")
                    with open(output_json_path, 'w') as f:
                        json.dump(json_serializable_outputs, f, indent=4)
                    logger.info(f"Sample structured symbolic outputs exported to: {output_json_path}")
                else:
                    logger.warning("Not enough samples to save symbolic outputs.")
        
        # Synchronize all processes before next epoch in DDP
        if is_ddp_initialized:
            dist.barrier()

    # Finalize SWA model if enabled (only for main process)
    if current_config['training']['use_swa'] and is_main_process:
        swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
        torch.save(swa_model.state_dict(), best_model_path.replace('.pt', '_swa.pt'))
        logger.info(f"SWA model finalized and saved to {best_model_path.replace('.pt', '_swa.pt')}")
        best_model_path = best_model_path.replace('.pt', '_swa.pt')

    # QAT specific: convert to quantized model (only for main process)
    if current_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION and is_main_process:
        logger.info("Finalizing QAT model conversion.")
        model_to_convert = model.module.cpu() if is_ddp_initialized else model.cpu()
        model_int8 = tq.convert(model_to_convert, inplace=False)
        quantized_model_path = best_model_path.replace('.pt', '_quantized.pt')
        torch.save(model_int8.state_dict(), quantized_model_path)
        logger.info(f"Quantized model saved to {quantized_model_path}")
        best_model_path = quantized_model_path

    if is_main_process:
        writer.close()
        if current_config['training']['use_wandb'] and HAS_WANDB:
            import wandb
            wandb.finish()
        
        final_val_predictions_logits = np.array(all_val_predictions_logits)
        final_val_labels = np.array(all_val_labels)
        best_metrics = {
            'val_loss': best_val_loss,
            'val_accuracy': best_val_accuracy
        }
        logger.info(f"--- Training Session for Member {member_id} Finished. Best model saved to {best_model_path} ---")
        return best_model_path, final_val_predictions_logits, final_val_labels, best_metrics
    else:
        # Non-main processes should return None or empty data
        return None, None, None, None

def _validate_model_ensemble(model: nn.Module, data_loader: Any, criterion: nn.Module, config: Dict[str, Any],
                             shard_id: int = 0, num_shards: int = 1) -> Tuple[float, float, List[np.ndarray], List[int], List[Dict[str, Any]]]:
    """
    Validates the model on the given data loader.
    Args:
        model (nn.Module): The model to validate (can be DDP wrapped).
        data_loader (Any): DALI data loader for validation data.
        criterion (nn.Module): Loss function.
        config (Dict): Configuration dictionary.
        shard_id (int): Current GPU rank for DDP.
        num_shards (int): Total number of GPUs for DDP.
    Returns:
        Tuple[float, float, List[np.ndarray], List[int], List[Dict[str, Any]]]:
            - Average validation loss.
            - Average validation accuracy.
            - List of numpy arrays of predictions (logits) for each validation sample.
            - List of true labels for each validation sample.
            - List of dictionaries for misclassified samples (path, true_label, predicted_label, input_tensor).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions_logits = []
    all_true_labels = []
    misclassified_samples_info = [] # NEW: To store info about misclassified samples

    data_loader.reset()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images_view1 = data['view1'].permute(0, 3, 1, 2).to(DEVICE) # NCHW
            labels_bongard = data['labels'].squeeze(-1).to(DEVICE) # Remove last dim from DALI
            gts_json_strings_batch = [s.decode('utf-8') for s in data['gts_json'].cpu().numpy()]
            original_indices_batch = data['original_indices'].to(DEVICE) # NEW: From DALI

            # Ground-Truth JSON Validation and Label Extraction (for synthetic data)
            for i_sample, gt_json_str in enumerate(gts_json_strings_batch):
                try:
                    gt_dict = json.loads(gt_json_str)
                    jsonschema.validate(instance=gt_dict, schema=BONGARD_GT_JSON_SCHEMA)
                    if config['data']['use_synthetic_data']:
                        labels_bongard[i_sample] = torch.tensor(gt_dict['bongard_label'], dtype=torch.long, device=DEVICE)
                except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                    logger.error(f"Failed to decode/validate GT JSON for sample {i_sample} in batch: {e}. Using dummy label 0.")
                    labels_bongard[i_sample] = torch.tensor(0, dtype=torch.long, device=DEVICE)

            with autocast(enabled=config['training']['use_amp']):
                # Use model.module if DDP wrapped
                current_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                bongard_logits, _, _ = current_model(images_view1, gts_json_strings_batch)
                loss = criterion(bongard_logits, labels_bongard)
            total_loss += loss.item()
            
            _, predicted = torch.max(bongard_logits.data, 1)
            correct_predictions += (predicted == labels_bongard).sum().item()
            total_samples += labels_bongard.size(0)
            
            all_predictions_logits.extend(bongard_logits.cpu().numpy())
            all_true_labels.extend(labels_bongard.cpu().numpy())

            # --- Collect Misclassified Samples for Grad-CAM ---
            if config['debug'].get('enable_grad_cam_on_errors', False):
                misclassified_mask = (predicted != labels_bongard)
                if torch.any(misclassified_mask):
                    misclassified_indices_in_batch = torch.where(misclassified_mask)[0]
                    # Assuming data_loader.iterator has a way to map original_indices_batch to file paths
                    # For real data, you'd need the original file_list and map original_indices_batch to it.
                    # For synthetic, BongardExternalSource's get_epoch_data might help, but it's complex.
                    # Simplification: if you have a global list of val_image_paths, use original_indices_batch
                    # to get the actual path.
                    
                    # Conceptual: get original file paths for misclassified samples
                    # This requires `val_image_paths` to be accessible and ordered consistently
                    # with `original_indices_batch` from DALI's `readers.file` or `external_source`.
                    
                    # For now, let's use a dummy path or reconstruct it if possible.
                    # In a real setup, ensure DALI passes original file paths or indices that map to them.
                    
                    # If `val_image_paths` is globally accessible and ordered:
                    if not config['data']['use_synthetic_data'] and val_image_paths is not None:
                        for idx_in_batch in misclassified_indices_in_batch:
                            original_idx = original_indices_batch[idx_in_batch].item()
                            # Ensure original_idx is within bounds of val_image_paths
                            if original_idx < len(val_image_paths):
                                img_path = val_image_paths[original_idx]
                                misclassified_samples_info.append({
                                    'path': img_path,
                                    'true_label': labels_bongard[idx_in_batch].item(),
                                    'predicted_label': predicted[idx_in_batch].item(),
                                    'input_tensor': images_view1[idx_in_batch].cpu() # Store CPU tensor
                                })
                            else:
                                logger.warning(f"Original index {original_idx} out of bounds for val_image_paths.")
                    else: # For synthetic data or if val_image_paths not available
                        for idx_in_batch in misclassified_indices_in_batch:
                            misclassified_samples_info.append({
                                'path': f"synthetic_misclassified_img_{original_indices_batch[idx_in_batch].item()}.png", # Dummy path
                                'true_label': labels_bongard[idx_in_batch].item(),
                                'predicted_label': predicted[idx_in_batch].item(),
                                'input_tensor': images_view1[idx_in_batch].cpu()
                            })

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, all_predictions_logits, all_true_labels, misclassified_samples_info

class SumTree:
    """
    A binary tree data structure for efficient sampling based on priorities (used in PER).
    It allows for O(log N) operations for adding, updating, and sampling priorities.
    """
    def __init__(self, capacity: int):
        """
        Initializes the SumTree.
        Args:
            capacity (int): The maximum number of experiences the tree can hold.
        """
        self.capacity = capacity
        # The tree array stores priorities. It's a complete binary tree, so 2*capacity - 1 nodes.
        # The last `capacity` nodes are leaf nodes (where experiences are stored).
        self.tree = np.zeros(2 * capacity - 1)
        # The data array stores the actual experiences.
        self.data = np.zeros(capacity, dtype=object) # Use object dtype to store any Python object
        self.data_pointer = 0 # Pointer to the next available data slot (index in `data` array)
        self.size = 0 # Current number of elements in the tree

        logger.debug(f"SumTree initialized with capacity: {capacity}")

    def add(self, priority: float, data: Any):
        """
        Adds a new experience to the tree with its priority.
        If the tree is full, it overwrites the oldest experience.
        Args:
            priority (float): The priority of the experience.
            data (Any): The actual experience data (e.g., a tuple of (state, action, reward, next_state)).
        """
        # Calculate the index of the leaf node for this new data
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Store the data in the data array
        self.data[self.data_pointer] = data
        
        # Update the priority in the tree
        self.update(tree_idx, priority)
        
        # Move the data pointer to the next slot (circular buffer)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Update current size
        if self.size < self.capacity:
            self.size += 1
        logger.debug(f"Added data at pointer {self.data_pointer-1} with priority {priority}. Current size: {self.size}")

    def update(self, tree_idx: int, priority: float):
        """
        Updates the priority of an existing experience at `tree_idx`.
        This also propagates the change up to the root of the tree.
        Args:
            tree_idx (int): The index of the leaf node in the `tree` array.
            priority (float): The new priority value.
        """
        # Calculate the change in priority
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate the change up the tree by updating parent nodes
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2 # Get parent index
            self.tree[tree_idx] += change
        logger.debug(f"Updated tree_idx {tree_idx} with priority {priority}.")

    def get_leaf(self, v: float) -> Tuple[int, float, Any]:
        """
        Samples a leaf node (experience) based on a random value `v`.
        The value `v` is sampled uniformly between 0 and `total_priority`.
        Args:
            v (float): A randomly sampled value between 0 and the total priority.
        Returns:
            Tuple[int, float, Any]: A tuple containing:
                - The index of the leaf node in the `tree` array.
                - The priority of the sampled experience.
                - The actual experience data.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we've reached a leaf node (no children)
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else: # Not a leaf node, traverse down
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        # Calculate the index in the `data` array from the leaf node index
        data_idx = leaf_idx - self.capacity + 1
        logger.debug(f"Sampled leaf tree_idx {leaf_idx}, data_idx {data_idx}.")
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        """Returns the sum of all priorities (stored at the root of the tree)."""
        return self.tree[0]

    def __len__(self) -> int:
        """Returns the current number of elements in the tree."""
        return self.size

def kornia_rand_augment(images_tensor: torch.Tensor, current_image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Applies a Kornia-based RandAugment-like policy to a batch of images.
    This function is designed to be called via DALI's fn.python_function.
    Args:
        images_tensor (torch.Tensor): Input batch of images in HWC format (uint8).
        current_image_size (Tuple[int, int]): Current (height, width) for resizing if needed.
                                              (Though DALI handles initial resize, this is for consistency).
    Returns:
        torch.Tensor: Augmented batch of images in HWC format (uint8).
    """
    if not HAS_KORNIA:
        logger.warning("Kornia not available. Skipping Kornia RandAugment.")
        return images_tensor # Return original if Kornia not available

    # Kornia expects NCHW float tensors in [0,1] range
    # Convert DALI's HWC uint8 to NCHW float [0,1]
    images_tensor_nchw_float = images_tensor.permute(0, 3, 1, 2).float() / 255.0
    
    # Define a sequence of augmentations for RandAugment-like behavior
    # Each augmentation has a probability (p) of being applied.
    # You can customize the parameters and types of augmentations here.
    transform = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.2),
        K.RandomRotation(degrees=K.RandomUniformGenerator(-15.0, 15.0), p=0.5, resample='bicubic'),
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        K.RandomGrayscale(p=0.2), # Converts to 1 channel, then repeats to 3 if input was 3
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.3),
        # K.RandomElasticTransform (requires HAS_ELASTIC_TRANSFORM and specific Kornia version)
        # K.RandomElasticTransform(kernel_size=(3,3), sigma=(1.0, 2.0), alpha=(1.0, 2.0), p=0.2),
        data_keys=["input"] # Apply transforms to the 'input' key
    ).to(images_tensor_nchw_float.device) # Ensure transforms are on the same device as input

    augmented_images_nchw_float = transform(images_tensor_nchw_float)

    # If RandomGrayscale was applied and num_channels is 3, Kornia might output 1 channel.
    # Ensure output is 3 channels if original was 3.
    if images_tensor.shape[3] == 3 and augmented_images_nchw_float.shape[1] == 1:
        augmented_images_nchw_float = augmented_images_nchw_float.repeat(1, 3, 1, 1)

    # Convert back to HWC uint8 for consistency with DALI pipeline
    augmented_images_hwc_uint8 = (augmented_images_nchw_float * 255.0).permute(0, 2, 3, 1).to(torch.uint8)
    
    logger.debug(f"Applied Kornia RandAugment. Input shape: {images_tensor.shape}, Output shape: {augmented_images_hwc_uint8.shape}")
    return augmented_images_hwc_uint8

def get_symbolic_embedding_dims() -> Dict[str, int]:
    """
    Calculates the vocabulary size (number of unique categories) for each
    symbolic attribute type and relation type. This is used to determine
    the input dimension for nn.Embedding layers.
    Returns:
        Dict[str, int]: A dictionary where keys are attribute/relation names
                        and values are their corresponding vocabulary sizes.
    """
    embedding_dims = {}
    for attr_type, attr_map in ATTRIBUTE_FILL_MAP.items():
        embedding_dims[attr_type] = len(attr_map)
    
    embedding_dims['relation'] = len(RELATION_MAP)
    
    logger.debug(f"Calculated symbolic embedding dimensions: {embedding_dims}")
    return embedding_dims

# It's good practice to pre-calculate this once globally if it's constant
# SYMBOLIC_EMBEDDING_DIMS = get_symbolic_embedding_dims()
def ddp_train_worker(rank: int, world_size: int, config: Dict[str, Any],
                     train_image_paths: List[str], train_labels: np.ndarray, train_difficulty_scores: np.ndarray,
                     val_image_paths: List[str], val_labels: np.ndarray, val_difficulty_scores: np.ndarray):
    """
    Worker function for Distributed Data Parallel (DDP) training.
    Each process runs this function on a specific GPU.
    Args:
        rank (int): The current process's global rank.
        world_size (int): The total number of processes.
        config (Dict): The global configuration dictionary.
        train_image_paths (List[str]): Paths to training images.
        train_labels (np.ndarray): Labels for training images.
        train_difficulty_scores (np.ndarray): Difficulty scores for training images.
        val_image_paths (List[str]): Paths to validation images.
        val_labels (np.ndarray): Labels for validation images.
        val_difficulty_scores (np.ndarray): Difficulty scores for validation images.
    """
    # 1. DDP Initialization
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Choose an unused port
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # 'nccl' backend for GPU communication

    # Set device for this rank
    torch.cuda.set_device(rank)
    current_device = torch.device(f'cuda:{rank}')
    
    logger.info(f"DDP Worker Rank {rank}/{world_size} initialized on device {current_device}.")

    # 2. Model Initialization and DDP Wrapping
    object_detector = RealObjectDetector(model_path=config['model']['object_detector_model_path'])
    # Create a deep copy of config for each worker to avoid accidental shared state modifications
    worker_config = copy.deepcopy(config) 
    
    # Adjust dataloader_num_workers per GPU
    worker_config['data']['dataloader_num_workers'] = worker_config['data']['dataloader_num_workers'] // world_size
    if worker_config['data']['dataloader_num_workers'] == 0:
        worker_config['data']['dataloader_num_workers'] = 1 # Ensure at least 1 worker

    model = PerceptionModule(worker_config, object_detector).to(current_device)
    model = DDP(model, device_ids=[rank])
    logger.info(f"Rank {rank}: PerceptionModule model wrapped with DDP.")

    # 3. Call the main training session function
    # _run_single_training_session_ensemble needs to be adapted to be DDP-aware (already done in previous update)
    # It will use `shard_id=rank` and `num_shards=world_size` to ensure data sharding.
    
    # Note: For ensemble training, if each DDP worker trains a *different* ensemble member,
    # you'd adjust `member_id` and `random_seed` based on `rank`.
    # If a single member is trained across multiple GPUs, `member_id` would be 0.
    # Here, we're assuming each DDP worker trains one ensemble member, so member_id = rank.
    
    _run_single_training_session_ensemble(
        current_config=worker_config,
        member_id=rank, # Use rank as member_id for this worker's model
        random_seed=config['model']['random_seed'] + rank, # Different seed for each worker's member
        output_dir=config['training']['checkpoint_dir'],
        epochs_override=config['model']['epochs'],
        train_image_paths=train_image_paths,
        train_labels=train_labels,
        train_difficulty_scores=train_difficulty_scores,
        val_image_paths=val_image_paths,
        val_labels=val_labels,
        val_difficulty_scores=val_difficulty_scores,
        teacher_model=None, # Base models don't have a teacher
        all_members_val_predictions_logits=None, # Not needed for base training
        all_members_val_labels=None, # Not needed for base training
        meta_learner_path=None, # Not needed for base training
        shard_id=rank, # Pass current rank for DALI sharding
        num_shards=world_size # Pass total world size for DALI sharding
    )
    
    logger.info(f"DDP Worker Rank {rank}/{world_size} finished training.")
    
    # 4. Cleanup DDP
    dist.destroy_process_group()

def perform_tta_inference(model: Any, image_path: str, config: Dict[str, Any], num_tta_samples: int = 5) -> np.ndarray:
    """
    Performs Test-Time Augmentation (TTA) inference on a single image.
    Applies multiple augmented versions of the input image and averages their predictions.
    Args:
        model (Any): The trained model (can be a PerceptionModule or DDP-wrapped PerceptionModule).
        image_path (str): Path to the image file (e.g., PNG).
        config (Dict): Configuration dictionary.
        num_tta_samples (int): Number of augmented versions to create for TTA.
    Returns:
        np.ndarray: Averaged probabilities for the Bongard problem across TTA samples.
                    Shape: [num_classes].
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Unwrap DDP model if necessary
    current_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    try:
        original_image_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return np.zeros(config['model']['num_classes']) # Return dummy probabilities

    all_tta_probs = []

    # Define TTA augmentations (a subset of training augmentations, typically simpler)
    # These transforms should be applied after initial resize and before ToTensor/Normalize
    tta_transforms_list = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, resample=Image.BICUBIC), # Use BICUBIC for rotation
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # Add more subtle augmentations as needed
    ]
    
    # Base transform for all TTA samples
    base_transform = T.Compose([
        T.Resize(config['data']['image_size']), # Resize to model input size
        T.ToTensor(), # Converts PIL Image to FloatTensor and scales to [0, 1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Normalize with ImageNet stats
    ])

    with torch.no_grad():
        for i in range(num_tta_samples):
            # Apply a random subset of TTA transforms for each sample
            current_tta_transforms = T.Compose([
                T.RandomApply([aug], p=0.5) for aug in tta_transforms_list
            ])
            
            # Apply TTA transforms to PIL image
            augmented_pil_image = current_tta_transforms(original_image_pil)
            
            # Apply base transforms and move to device
            input_tensor = base_transform(augmented_pil_image).unsqueeze(0).to(DEVICE) # Add batch dimension
            
            # Pass dummy GT JSON for inference, as TTA is for prediction
            dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})]
            
            # Forward pass through the model
            bongard_logits, _, _ = current_model(input_tensor, dummy_gts_json)
            
            # Get probabilities using softmax
            probs = F.softmax(bongard_logits, dim=-1).cpu().numpy()
            all_tta_probs.append(probs)
    
    # Average the probabilities across all TTA samples
    averaged_probs = np.mean(np.stack(all_tta_probs, axis=0), axis=0)
    
    # Return the averaged probabilities (remove the batch dimension if it's 1)
    return averaged_probs.squeeze(0)


def ensemble_predict_orchestrator_base(
    models: List[PerceptionModule],
    image_paths: List[str],
    config: Dict[str, Any],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0,
    model_weights: Optional[List[float]] = None # Optional list of weights for weighted averaging
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble prediction by averaging probabilities from multiple PerceptionModule models.
    This is the base prediction function, now internal to the combined module.
    Args:
        models (List[PerceptionModule]): List of trained PerceptionModule instances.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
        use_mc_dropout (bool): If True, enable MC Dropout during inference.
        mc_dropout_samples (int): Number of forward passes for MC Dropout.
        model_weights (Optional[List[float]]): Weights for each model for weighted averaging.
                                                If None, simple averaging is used.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Averaged Bongard problem probabilities (after softmax) [num_images, num_classes].
            - List of symbolic outputs (from the first model's inference, for example).
    """
    if not models:
        logger.error("No models provided for ensemble prediction.")
        return np.empty((len(image_paths), config['model']['num_classes'])), []
    all_bongard_probs = []
    first_model_symbolic_outputs = [] # Collect symbolic outputs from the first model for example
    # Create DALI loader for inference using refactored function
    inference_pipeline, inference_loader = build_dali_loader(
        file_list=image_paths,
        labels_list=[0] * len(image_paths), # Dummy labels for inference
        config=config,
        mode='inference'
    )
    logger.info("DALI inference loader initialized.")
    for model_idx, model in enumerate(models):
        model.eval()
        if use_mc_dropout:
            # Enable dropout layers for MC Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train() # Set dropout to training mode to enable dropout during inference
            logger.info(f"MC Dropout enabled for model {model_idx}.")
        else:
            # Ensure dropout layers are in eval mode if not using MC Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
        
        member_probs = []
        member_symbolic_outputs_mc_samples = [] # Store symbolic outputs for each MC sample
        with torch.no_grad():
            # Efficient MC Dropout Sampling
            for mc_sample_idx in tqdm(range(mc_dropout_samples if use_mc_dropout else 1), desc=f"Model {model_idx} MC Inference"):
                inference_loader.reset() # Reset for each MC Dropout sample or single pass
                
                current_sample_probs = []
                current_batch_symbolic_outputs = []
                for batch_idx, data in enumerate(inference_loader):
                    images_view1 = data['view1'].permute(0, 3, 1, 2).to(DEVICE) # NCHW
                    
                    # Pass None for ground_truth_json_strings as it's inference on real images
                    # PerceptionModule.forward now returns bongard_logits, detected_objects_batch, aggregated_outputs
                    bongard_logits, detected_objects_batch, aggregated_outputs = model(images_view1, None)
                    
                    probs = F.softmax(bongard_logits, dim=-1).cpu().numpy()
                    current_sample_probs.append(probs)
                    # Collect structured symbolic outputs from the first model, for each MC dropout sample
                    if model_idx == 0: # Only collect for the first model
                        for b_idx in range(images_view1.shape[0]):
                            # Ensure detected_objects_batch and aggregated_outputs are correctly indexed for a single image
                            single_detected_objects = detected_objects_batch[b_idx]
                            single_attribute_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['attribute_logits'].items()}
                            single_relation_logits = {'global_relation': aggregated_outputs['relation_logits'][b_idx].unsqueeze(0)} # Assuming global relation
                            scene_graph = model.extract_scene_graph(
                                single_detected_objects,
                                single_attribute_logits,
                                single_relation_logits
                            )
                            scene_graph["bongard_prediction"] = int(torch.argmax(bongard_logits[b_idx]).item())
                            
                            # Store raw logits for uncertainty calculation later
                            scene_graph["raw_bongard_logits"] = bongard_logits[b_idx].cpu().numpy().tolist()
                            scene_graph["bongard_probs"] = probs[b_idx].tolist()
                            
                            current_batch_symbolic_outputs.append(scene_graph)
                
                member_probs.append(np.concatenate(current_sample_probs, axis=0))
                if model_idx == 0: # Only store symbolic outputs for the first model across MC samples
                    member_symbolic_outputs_mc_samples.append(current_batch_symbolic_outputs)
        
        # Average MC Dropout samples for this model
        if use_mc_dropout:
            avg_member_probs = np.mean(np.stack(member_probs, axis=0), axis=0)
            logger.info(f"Model {model_idx} averaged {mc_dropout_samples} MC Dropout samples.")
            
            # If MC Dropout, calculate epistemic and aleatoric uncertainty for symbolic outputs
            if model_idx == 0 and member_symbolic_outputs_mc_samples:
                # Reorganize symbolic outputs to be per-image, across MC samples
                num_images_in_total = len(member_symbolic_outputs_mc_samples[0])
                for img_idx in range(num_images_in_total):
                    all_mc_logits_for_image = []
                    for mc_sample_data in member_symbolic_outputs_mc_samples:
                        # Ensure mc_sample_data[img_idx] exists and has 'raw_bongard_logits'
                        if img_idx < len(mc_sample_data) and 'raw_bongard_logits' in mc_sample_data[img_idx]:
                            all_mc_logits_for_image.append(mc_sample_data[img_idx]["raw_bongard_logits"])
                        else:
                            logger.warning(f"Missing raw_bongard_logits for image {img_idx} in MC sample. Skipping uncertainty calculation for this image.")
                            all_mc_logits_for_image.append([0.0] * config['model']['num_classes']) # Append dummy if data is missing
                    all_mc_logits_for_image_np = np.array(all_mc_logits_for_image) # [mc_samples, num_classes]
                    
                    if all_mc_logits_for_image_np.shape[0] > 1: # Need at least 2 samples for variance
                        # Epistemic uncertainty: variance of predictions across MC samples
                        epistemic_probs = F.softmax(torch.tensor(all_mc_logits_for_image_np), dim=-1).numpy()
                        epistemic_uncertainty = np.mean(np.var(epistemic_probs, axis=0)) # Mean variance across classes
                        
                        # Aleatoric uncertainty: average of (p * (1-p)) over MC samples
                        aleatoric_uncertainty = np.mean(epistemic_probs * (1 - epistemic_probs))
                        
                        # Update the first MC sample's symbolic output for this image with uncertainty
                        member_symbolic_outputs_mc_samples[0][img_idx]["uncertainty"] = {
                            "epistemic": float(epistemic_uncertainty),
                            "aleatoric": float(aleatoric_uncertainty)
                        }
                    else:
                        logger.warning(f"Not enough MC samples for image {img_idx} to calculate uncertainty. Setting to 0.0.")
                        member_symbolic_outputs_mc_samples[0][img_idx]["uncertainty"] = {"epistemic": 0.0, "aleatoric": 0.0}
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0] # Use the first sample's structure, updated with uncertainty
        else:
            avg_member_probs = member_probs[0] # Only one sample
            if model_idx == 0:
                # If no MC dropout, symbolic output's uncertainty will be 0.0
                for img_data in member_symbolic_outputs_mc_samples[0]:
                    img_data["uncertainty"] = {"epistemic": 0.0, "aleatoric": 0.0}
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0] # Just the single sample's output
        
        all_bongard_probs.append(avg_member_probs)
    stacked_probs = np.stack(all_bongard_probs, axis=0) # Shape: [num_models, num_images, num_classes]
    if model_weights is not None and len(model_weights) == len(models):
        # Normalize weights to sum to 1
        normalized_weights = np.array(model_weights) / np.sum(model_weights)
        # Reshape weights to [num_models, 1, 1] for broadcasting
        weighted_probs = stacked_probs * normalized_weights[:, np.newaxis, np.newaxis]
        ensemble_averaged_probs = np.sum(weighted_probs, axis=0) # Sum along the model dimension
        logger.info("Ensemble prediction using weighted averaging.")
    else:
        ensemble_averaged_probs = np.mean(stacked_probs, axis=0)
        logger.info("Ensemble prediction using simple averaging.")
    logger.info(f"Ensemble prediction complete. Averaged probabilities for {len(image_paths)} images.")
    return ensemble_averaged_probs, first_model_symbolic_outputs
# --- Advanced Ensemble Training Orchestration ---
def train_ensemble_members_orchestrator_combined(
    num_members: int = 3,
    start_seed: int = 42,
    ensemble_output_dir: str = './ensemble_models',
    epochs_override: Optional[int] = None,
    train_image_paths: List[str] = None,
    train_labels: np.ndarray = None,
    train_difficulty_scores: Optional[np.ndarray] = None, # Added for difficulty-weighted loss
    val_image_paths: List[str] = None,
    val_labels: np.ndarray = None,
    val_difficulty_scores: np.ndarray = None,
    member_configs: Optional[List[Dict[str, Any]]] = None # For diversity promotion
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
    """
    Orchestrates the training of multiple PerceptionModule ensemble members.
    This function is now part of the advanced orchestrator, allowing for diverse configs.
    Args:
        num_members (int): Number of ensemble members to train.
        start_seed (int): Starting random seed for reproducibility.
        ensemble_output_dir (str): Directory to save individual member checkpoints.
        epochs_override (Optional[int]): If provided, overrides 'model.epochs' for all members.
        train_image_paths, train_labels, val_image_paths, val_labels, val_difficulty_scores: Data for training.
        member_configs (Optional[List[Dict]]): A list of configurations, one for each member.
                                                If None, generates diverse configs automatically.
    Returns:
        Tuple[List[str], List[np.ndarray], List[np.ndarray], List[Dict[str, Any]]]:
            - List of paths to the best saved model checkpoints for each member.
            - List of validation predictions (logits) for each member.
            - List of validation true labels for each member.
            - List of best validation metrics for each member.
    """
    logger.info(f"--- Starting Ensemble Training Orchestration ({num_members} members) ---")
    os.makedirs(ensemble_output_dir, exist_ok=True)
    trained_model_paths = []
    all_members_val_predictions_logits = []
    all_members_val_labels = []
    all_members_best_metrics = []
    if member_configs is None:
        logger.info("Generating diverse configurations for ensemble members.")
        member_configs = []
        for i in range(num_members):
            current_member_config = copy.deepcopy(CONFIG) # Start with global CONFIG
            current_member_config['model']['random_seed'] = start_seed + i # Ensure seed is in config if needed by other parts
            # Introduce Architectural Diversity
            if i % 3 == 0:
                current_member_config['model']['attribute_backbone'] = 'mobilenet_v2'
                current_member_config['model']['gnn_depth'] = 2
            elif i % 3 == 1:
                current_member_config['model']['attribute_backbone'] = 'efficientnet_b0'
                current_member_config['model']['gnn_depth'] = 3
            else:
                current_member_config['model']['attribute_backbone'] = 'mobilenet_v3_small'
                current_member_config['model']['gnn_depth'] = 2
            
            # Introduce Hyperparameter Diversity (e.g., slight learning rate variation)
            lr_multiplier = 1 + 0.05 * (i - num_members / 2) # Centered around 1
            current_member_config['model']['initial_learning_rate'] *= lr_multiplier
            current_member_config['model']['max_learning_rate'] *= lr_multiplier
            
            # Example: Vary dropout rate
            current_member_config['model']['initial_dropout_rate'] = max(0.1, min(0.5, CONFIG['model']['initial_dropout_rate'] + (i - num_members / 2) * 0.05))
            
            # Example: Vary MixUp/CutMix parameters slightly
            current_member_config['training']['mixup_alpha'] = max(0.0, min(1.0, CONFIG['training']['mixup_alpha'] + (random.random() - 0.5) * 0.1))
            current_member_config['training']['cutmix_alpha'] = max(0.0, min(1.0, CONFIG['training']['cutmix_alpha'] + (random.random() - 0.5) * 0.1))
            logger.info(f"Member {i} config: Backbone={current_member_config['model']['attribute_backbone']}, GNN_Depth={current_member_config['model']['gnn_depth']}, Initial_LR={current_member_config['model']['initial_learning_rate']:.6f}, MixUp={current_member_config['training']['mixup_alpha']:.2f}")
            member_configs.append(current_member_config)
    else:
        logger.info("Using provided diverse configurations for ensemble members.")
        if len(member_configs) != num_members:
            raise ValueError(f"Number of provided member_configs ({len(member_configs)}) does not match num_members ({num_members}).")
    for i in range(num_members):
        current_seed = start_seed + i
        # Assuming _run_single_training_session_ensemble is available globally or passed
        model_path, val_preds_logits, val_labels_arr, best_metrics = _run_single_training_session_ensemble(
            current_config=member_configs[i], # Pass the specific config for this member
            member_id=i,
            random_seed=current_seed,
            output_dir=ensemble_output_dir,
            epochs_override=epochs_override,
            train_image_paths=train_image_paths,
            train_labels=train_labels,
            train_difficulty_scores=train_difficulty_scores, # Assuming difficulty scores are available for training as well
            val_image_paths=val_image_paths,
            val_labels=val_labels,
            val_difficulty_scores=val_difficulty_scores
        )
        trained_model_paths.append(model_path)
        all_members_val_predictions_logits.append(val_preds_logits)
        all_members_val_labels.append(val_labels_arr) # These should be identical across members for the same val set
        all_members_best_metrics.append(best_metrics)
    logger.info(f"--- Finished Ensemble Training Orchestration. All checkpoints saved to {ensemble_output_dir} ---")
    return trained_model_paths, all_members_val_predictions_logits, all_members_val_labels, all_members_best_metrics
def train_meta_learner_orchestrator_combined(
    base_model_val_predictions_logits: List[np.ndarray], # List of [num_val_samples, num_classes] arrays
    val_labels: np.ndarray, # [num_val_samples]
    meta_learner_output_dir: str = './meta_learner_models',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    random_seed: int = 42
) -> str:
    """
    Orchestrates the training of the Level 1 Meta-Learner for stacking.
    The meta-learner is trained on the out-of-fold (validation) predictions
    of the base ensemble members.
    Args:
        base_model_val_predictions_logits (List[np.ndarray]): A list where each element
            is a NumPy array of logits from one base model on the validation set.
            Shape of each array: [num_validation_samples, num_classes].
        val_labels (np.ndarray): The true labels for the validation set.
            Shape: [num_validation_samples].
        meta_learner_output_dir (str): Directory to save the meta-learner checkpoint.
        epochs (int): Number of epochs to train the meta-learner.
        batch_size (int): Batch size for meta-learner training.
        learning_rate (float): Learning rate for the meta-learner optimizer.
        random_seed (int): Random seed for reproducibility.
    Returns:
        str: Path to the saved meta-learner checkpoint.
    """
    set_seed(random_seed)
    logger.info(f"--- Starting Meta-Learner Training Orchestration (Seed: {random_seed}) ---")
    os.makedirs(meta_learner_output_dir, exist_ok=True)
    if not base_model_val_predictions_logits:
        logger.error("No base model validation predictions provided for meta-learner training.")
        return ""
    num_base_models = len(base_model_val_predictions_logits)
    num_val_samples = base_model_val_predictions_logits[0].shape[0]
    num_classes = base_model_val_predictions_logits[0].shape[1]
    # Concatenate predictions to form meta-features
    # Resulting shape: [num_val_samples, num_base_models * num_classes]
    meta_features_np = np.concatenate(base_model_val_predictions_logits, axis=1)
    
    # Convert to PyTorch tensors
    meta_features = torch.tensor(meta_features_np, dtype=torch.float32).to(DEVICE)
    meta_labels = torch.tensor(val_labels, dtype=torch.long).to(DEVICE)
    logger.info(f"Meta-learner training data prepared. Meta-features shape: {meta_features.shape}, Labels shape: {meta_labels.shape}.")
    # Initialize MetaLearner model
    meta_learner_input_dim = num_base_models * num_classes
    meta_learner = MetaLearner(input_dim=meta_learner_input_dim, num_classes=num_classes).to(DEVICE)
    logger.info("MetaLearner model initialized.")
    # Optimizer and Loss
    optimizer = optim.AdamW(meta_learner.parameters(), lr=learning_rate, weight_decay=CONFIG['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    # Create DataLoader for meta-learner training
    meta_dataset = torch.utils.data.TensorDataset(meta_features, meta_labels)
    meta_loader = torch.utils.data.DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Meta-learner DataLoader initialized with {len(meta_loader)} batches.")
    best_val_loss = float('inf')
    meta_learner_checkpoint_path = os.path.join(meta_learner_output_dir, 'best_meta_learner.pt')
    # Training loop for Meta-Learner
    for epoch in range(epochs):
        meta_learner.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples_epoch = 0
        for batch_meta_features, batch_meta_labels in meta_loader:
            optimizer.zero_grad()
            with autocast(enabled=CONFIG['training']['use_amp']):
                logits = meta_learner(batch_meta_features)
                loss = criterion(logits, batch_meta_labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct_predictions += (predicted == batch_meta_labels).sum().item()
            total_samples_epoch += batch_meta_labels.size(0)
        avg_train_loss = total_loss / len(meta_loader)
        train_accuracy = correct_predictions / total_samples_epoch if total_samples_epoch > 0 else 0.0
        logger.info(f"Meta-Learner Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        # Simple validation on the same set (can be improved with separate meta-validation set)
        meta_learner.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for batch_meta_features, batch_meta_labels in meta_loader:
                with autocast(enabled=CONFIG['training']['use_amp']):
                    logits = meta_learner(batch_meta_features)
                    loss = criterion(logits, batch_meta_labels)
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_correct_predictions += (predicted == batch_meta_labels).sum().item()
                val_total_samples += batch_meta_labels.size(0)
        
        avg_val_loss = val_loss / len(meta_loader)
        val_accuracy = val_correct_predictions / val_total_samples if val_total_samples > 0 else 0.0
        logger.info(f"Meta-Learner Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(meta_learner.state_dict(), meta_learner_checkpoint_path)
            logger.info(f"New best Meta-Learner validation loss: {best_val_loss:.4f}. Saving model.")
    
    logger.info(f"--- Finished Meta-Learner Training. Best model saved to {meta_learner_checkpoint_path} ---")
    return meta_learner_checkpoint_path
def train_distilled_student_orchestrator_combined(
    teacher_model_paths: List[str],
    train_image_paths: List[str],
    train_labels: np.ndarray,
    val_image_paths: List[str],
    val_labels: np.ndarray,
    val_difficulty_scores: Optional[np.ndarray],
    student_output_dir: str = './checkpoints/distilled_student',
    epochs_override: Optional[int] = None,
    random_seed: int = 42,
    student_config_override: Optional[Dict] = None,
    all_members_val_predictions_logits: Optional[List[np.ndarray]] = None, # NEW: For stacked teacher
    all_members_val_labels: Optional[List[np.ndarray]] = None # NEW: For stacked teacher
) -> str:
    """
    Orchestrates the training of a distilled student model using an ensemble of teacher models.
    Args:
        teacher_model_paths (List[str]): Paths to trained teacher model checkpoints.
        train_image_paths (List[str]): Paths to training images for the student.
        train_labels (np.ndarray): Labels for training images.
        val_image_paths (List[str]): Paths to validation images for the student.
        val_labels (np.ndarray): Labels for validation images.
        val_difficulty_scores (Optional[np.ndarray]): Difficulty scores for validation images.
        student_output_dir (str): Directory to save the student model checkpoint.
        epochs_override (Optional[int]): Overrides epochs in config for student training.
        random_seed (int): Random seed for student training.
        student_config_override (Optional[Dict]): Configuration overrides for the student model.
        all_members_val_predictions_logits (Optional[List[np.ndarray]]): Validation predictions from base models for stacking.
        all_members_val_labels (Optional[List[np.ndarray]]): Validation labels for stacking.
    Returns:
        str: Path to the best saved student model checkpoint.
    """
    logger.info("Starting distilled student model training orchestration.")

    student_config = copy.deepcopy(CONFIG)
    if student_config_override:
        # Merge student_config_override into student_config (deep merge for nested dicts)
        def deep_merge_dict(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    d1[k] = deep_merge_dict(d1[k], v)
                else:
                    d1[k] = v
            return d1
        student_config = deep_merge_dict(student_config, student_config_override)
        logger.info("Student configuration overridden.")

    # --- Prepare Teacher Model(s) ---
    teacher_ensemble_type = CONFIG['ensemble'].get('teacher_ensemble_type', 'simple_average')
    teacher_models_for_distillation = [] # List of PerceptionModule instances
    
    if teacher_ensemble_type == 'simple_average':
        logger.info("Preparing simple average teacher ensemble for distillation.")
        for path in teacher_model_paths:
            obj_detector_teacher = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
            teacher_model = PerceptionModule(CONFIG, obj_detector_teacher).to(DEVICE)
            teacher_model.load_weights(path)
            teacher_model.eval()
            teacher_models_for_distillation.append(teacher_model)
    
    elif teacher_ensemble_type == 'weighted_average':
        logger.info("Preparing weighted average teacher ensemble for distillation.")
        if not all_members_val_predictions_logits or not all_members_val_labels:
            logger.warning("Weighted average teacher requires base model validation predictions/labels. Falling back to simple average.")
            return train_distilled_student_orchestrator_combined(
                teacher_model_paths=teacher_model_paths,
                train_image_paths=train_image_paths, train_labels=train_labels,
                val_image_paths=val_image_paths, val_labels=val_labels,
                val_difficulty_scores=val_difficulty_scores,
                student_output_dir=student_output_dir,
                epochs_override=epochs_override, random_seed=random_seed,
                student_config_override=student_config_override,
                teacher_ensemble_type='simple_average' # Recursive call with fallback
            )
        
        # Calculate weights based on base model performance
        # Assuming all_members_val_predictions_logits contains logits for each member
        # and all_members_val_labels is consistent.
        
        # You'd need to calculate actual metrics for each member from their val logits/labels
        # For simplicity, let's assume you have `all_members_best_metrics` from ensemble training.
        # If not, you'd re-run validation on each teacher to get their metrics.
        # For this example, let's use dummy metrics if not passed, or assume they are from `all_members_best_metrics`
        
        # Assuming `all_members_best_metrics` is available from the main workflow
        # (This needs to be passed to this function or be globally accessible)
        # For now, let's create a dummy if not available for this function's scope
        dummy_metrics = [{'val_accuracy': 0.7}, {'val_accuracy': 0.75}, {'val_accuracy': 0.72}] # Example
        member_weights = calculate_model_weights(dummy_metrics, metric_name='val_accuracy', minimize_metric=False)
        
        if not member_weights:
            logger.warning("Could not calculate weights for weighted average teacher. Falling back to simple average.")
            return train_distilled_student_orchestrator_combined(
                teacher_model_paths=teacher_model_paths,
                train_image_paths=train_image_paths, train_labels=train_labels,
                val_image_paths=val_image_paths, val_labels=val_labels,
                val_difficulty_scores=val_difficulty_scores,
                student_output_dir=student_output_dir,
                epochs_override=epochs_override, random_seed=random_seed,
                student_config_override=student_config_override,
                teacher_ensemble_type='simple_average'
            )
        
        # Load all teacher models
        loaded_teacher_models = []
        for path in teacher_model_paths:
            obj_detector_teacher = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
            teacher_model = PerceptionModule(CONFIG, obj_detector_teacher).to(DEVICE)
            teacher_model.load_weights(path)
            teacher_model.eval()
            loaded_teacher_models.append(teacher_model)
        
        # Create a "WeightedAverageTeacher" wrapper if needed, or pass list and weights
        # For simplicity, we'll pass the list of models and handle weighted average in _run_single_training_session_ensemble
        teacher_models_for_distillation = loaded_teacher_models # Pass the list
        # The _run_single_training_session_ensemble will need to apply these weights when getting teacher logits.
        # This will require a more complex teacher_model argument or a dedicated teacher class.
        # For now, let's simplify: the teacher_model argument will be a list of models,
        # and _run_single_training_session_ensemble will average them.
        # If you want true weighted average, you'd need to pass the weights alongside the models
        # or create a custom teacher module that encapsulates this logic.
        logger.warning("Weighted average teacher logic needs to be fully implemented inside _run_single_training_session_ensemble.")

    elif teacher_ensemble_type == 'stacked':
        logger.info("Preparing stacked teacher ensemble for distillation.")
        if not all_members_val_predictions_logits or not all_members_val_labels:
            logger.warning("Stacked teacher requires base model validation predictions/labels. Falling back to simple average.")
            return train_distilled_student_orchestrator_combined(
                teacher_model_paths=teacher_model_paths,
                train_image_paths=train_image_paths, train_labels=train_labels,
                val_image_paths=val_image_paths, val_labels=val_labels,
                val_difficulty_scores=val_difficulty_scores,
                student_output_dir=student_output_dir,
                epochs_override=epochs_override, random_seed=random_seed,
                student_config_override=student_config_override,
                teacher_ensemble_type='simple_average' # Recursive call with fallback
            )
        
        # 1. Train Meta-Learner (if not already trained)
        # Assuming meta_learner_path is passed or globally accessible from main workflow
        meta_learner_path = os.path.join(CONFIG['training']['checkpoint_dir'], 'meta_learner', 'best_meta_learner.pt')
        if not os.path.exists(meta_learner_path):
            logger.info("Meta-learner not found for stacked teacher. Training it now.")
            # Ensure train_meta_learner_orchestrator_combined is robust enough for this context
            meta_learner_path = train_meta_learner_orchestrator_combined(
                base_model_val_predictions_logits=all_members_val_predictions_logits,
                val_labels=all_members_val_labels[0] if all_members_val_labels else val_labels, # Use first member's labels
                meta_learner_output_dir=os.path.join(CONFIG['training']['checkpoint_dir'], 'meta_learner'),
                epochs=CONFIG['training']['epochs'], # Use full epochs for meta-learner training
                batch_size=CONFIG['model']['batch_size'],
                learning_rate=CONFIG['model']['initial_learning_rate'],
                random_seed=random_seed
            )
        
        # 2. Load Base Models and Meta-Learner
        base_models_for_stacking = []
        for path in teacher_model_paths:
            obj_detector_base = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
            base_model = PerceptionModule(CONFIG, obj_detector_base).to(DEVICE)
            base_model.load_weights(path)
            base_model.eval()
            base_models_for_stacking.append(base_model)

        num_classes = CONFIG['model']['num_classes']
        input_dim_meta = len(base_models_for_stacking) * num_classes
        meta_learner = MetaLearner(input_dim_meta, num_classes).to(DEVICE)
        meta_learner.load_state_dict(torch.load(meta_learner_path, map_location=DEVICE))
        meta_learner.eval()

        # Create a "StackedTeacher" wrapper class
        class StackedTeacher(nn.Module):
            def __init__(self, base_models: List[PerceptionModule], meta_learner: MetaLearner):
                super().__init__()
                self.base_models = nn.ModuleList(base_models) # Use ModuleList for proper device handling
                self.meta_learner = meta_learner
            
            def forward(self, images_tensor: torch.Tensor, ground_truth_json_strings: Optional[List[str]] = None) -> Tuple[torch.Tensor, Any, Any]:
                base_predictions = []
                for model in self.base_models:
                    model.eval() # Ensure base models are in eval mode
                    logits, _, _ = model(images_tensor, ground_truth_json_strings)
                    base_predictions.append(logits)
                
                # Concatenate logits from base models
                concatenated_predictions = torch.cat(base_predictions, dim=-1)
                
                # Get final logits from meta-learner
                final_logits = self.meta_learner(concatenated_predictions)
                return final_logits, None, None # Return logits, dummy detected_objects, dummy aggregated_outputs

        teacher_models_for_distillation = StackedTeacher(base_models_for_stacking, meta_learner)
        logger.info("Stacked teacher ensemble prepared.")

    else:
        logger.error(f"Unknown teacher ensemble type: {teacher_ensemble_type}. Aborting distillation training.")
        return None # Abort if teacher type is invalid

    # Call the single training session with the prepared teacher
    student_model_path, _, _, _ = _run_single_training_session_ensemble(
        current_config=student_config,
        member_id=0, # Student is typically a single model, use 0 for its ID
        random_seed=random_seed,
        output_dir=student_output_dir,
        epochs_override=epochs_override,
        train_image_paths=train_image_paths,
        train_labels=train_labels,
        train_difficulty_scores=None, # Student typically doesn't use dynamic difficulty from scratch
        val_image_paths=val_image_paths,
        val_labels=val_labels,
        val_difficulty_scores=val_difficulty_scores,
        teacher_model=teacher_models_for_distillation, # Pass the prepared teacher (list of models or StackedTeacher instance)
        shard_id=0, num_shards=1 # Student training is usually single-GPU unless specified
    )
    return student_model_path

# --- Advanced Ensemble Inference ---
def ensemble_predict_weighted_combined(
    model_paths: List[str],
    image_paths: List[str],
    config: Dict[str, Any],
    model_weights: List[float],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble prediction using weighted averaging.
    This function re-uses the `ensemble_predict_orchestrator_base` (formerly ensemble_predict_orchestrator)
    but explicitly passes the weights.
    Args:
        model_paths (List[str]): Paths to the trained PerceptionModule models.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
        model_weights (List[float]): Weights for each model for weighted averaging.
        use_mc_dropout (bool): If True, enable MC Dropout during inference.
        mc_dropout_samples (int): Number of forward passes for MC Dropout.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Averaged Bongard problem probabilities (after softmax) [num_images, num_classes].
            - List of symbolic outputs (from the first model's inference, for example).
    """
    logger.info("Performing weighted ensemble prediction.")
    loaded_models = []
    object_detector = RealObjectDetector(model_path=config['model']['object_detector_model_path'])
    for path in model_paths:
        model = PerceptionModule(config, object_detector).to(DEVICE)
        model.load_weights(path)
        loaded_models.append(model)
    
    if not loaded_models:
        logger.error("No models loaded for weighted ensemble prediction.")
        return np.empty((len(image_paths), config['model']['num_classes'])), []
    # ensemble_predict_orchestrator_base is assumed to be available from previous code block
    return ensemble_predict_orchestrator_base(
        models=loaded_models,
        image_paths=image_paths,
        config=config,
        use_mc_dropout=use_mc_dropout,
        mc_dropout_samples=mc_dropout_samples,
        model_weights=model_weights # Pass the weights here
    )
def ensemble_predict_stacked_combined(
    base_model_paths: List[str],
    meta_learner_path: str,
    image_paths: List[str],
    config: Dict[str, Any],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble prediction using stacking (meta-learning).
    Args:
        base_model_paths (List[str]): Paths to the trained Level 0 PerceptionModule models.
        meta_learner_path (str): Path to the trained Level 1 MetaLearner model.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
        use_mc_dropout (bool): If True, enable MC Dropout for base models during inference.
        mc_dropout_samples (int): Number of forward passes for MC Dropout.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Stacked Bongard problem probabilities (after softmax) [num_images, num_classes].
            - List of symbolic outputs (from the first base model's inference, for example).
    """
    logger.info("Performing stacked ensemble prediction.")
    
    if not base_model_paths or not meta_learner_path:
        logger.error("Base model paths or meta-learner path not provided for stacking.")
        return np.empty((len(image_paths), config['model']['num_classes'])), []
    # Load base models
    base_models = []
    object_detector_base = RealObjectDetector(model_path=config['model']['object_detector_model_path'])
    for path in base_model_paths:
        model = PerceptionModule(config, object_detector_base).to(DEVICE)
        model.load_weights(path)
        model.eval()
        # Enable dropout for MC Dropout if requested
        if use_mc_dropout:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        base_models.append(model)
    logger.info(f"Loaded {len(base_models)} base models for stacking.")
    # Load meta-learner
    num_classes = config['model']['num_classes']
    meta_learner_input_dim = len(base_models) * num_classes
    meta_learner = MetaLearner(input_dim=meta_learner_input_dim, num_classes=num_classes).to(DEVICE)
    meta_learner.load_state_dict(torch.load(meta_learner_path, map_location=DEVICE))
    meta_learner.eval()
    logger.info("MetaLearner loaded.")
    # Create a DALI loader for inference
    inference_pipeline, inference_loader = build_dali_loader(
        file_list=image_paths,
        labels_list=[0] * len(image_paths), # Dummy labels
        config=config,
        mode='inference'
    )
    logger.info("DALI inference loader initialized for stacking.")
    all_stacked_probs = []
    first_model_symbolic_outputs = [] # Collect symbolic outputs from the first base model
    with torch.no_grad():
        for _ in tqdm(range(mc_dropout_samples if use_mc_dropout else 1), desc="Stacked Ensemble Inference"):
            inference_loader.reset()
            
            batch_base_model_logits = []
            current_symbolic_outputs = []
            for batch_idx, data in enumerate(inference_loader):
                images_view1 = data['view1'].permute(0, 3, 1, 2).to(DEVICE) # NCHW
                
                # Collect logits from all base models
                current_batch_logits_from_base_models = []
                for model_idx, base_model in enumerate(base_models):
                    # Pass None for ground_truth_json_strings as it's inference on real images
                    # PerceptionModule.forward now returns bongard_logits, detected_objects_batch, aggregated_outputs
                    base_logits, detected_objects_batch, aggregated_outputs = base_model(images_view1, None)
                    current_batch_logits_from_base_models.append(base_logits)
                    # Collect symbolic outputs from the first base model, first MC dropout sample
                    if model_idx == 0 and (_ == 0 or not use_mc_dropout):
                        for b_idx in range(images_view1.shape[0]):
                            # Ensure detected_objects_batch and aggregated_outputs are correctly indexed for a single image
                            single_detected_objects = detected_objects_batch[b_idx]
                            single_attribute_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['attribute_logits'].items()}
                            single_relation_logits = {'global_relation': aggregated_outputs['relation_logits'][b_idx].unsqueeze(0)} # Assuming global relation
                            scene_graph = base_model.extract_scene_graph(
                                single_detected_objects,
                                single_attribute_logits,
                                single_relation_logits
                            )
                            scene_graph["bongard_prediction"] = int(torch.argmax(base_logits[b_idx]).item())
                            scene_graph["bongard_probs"] = F.softmax(base_logits[b_idx], dim=-1).cpu().numpy().tolist()
                            
                            current_symbolic_outputs.append(scene_graph)
                # Concatenate logits for meta-learner input
                # Shape: [batch_size, num_base_models * num_classes]
                meta_learner_input = torch.cat(current_batch_logits_from_base_models, dim=1)
                
                # Pass through meta-learner
                stacked_logits = meta_learner(meta_learner_input)
                stacked_probs = F.softmax(stacked_logits, dim=-1).cpu().numpy()
                all_stacked_probs.append(stacked_probs)
            
            if _ == 0 or not use_mc_dropout: # Only capture symbolic outputs once
                first_model_symbolic_outputs = current_symbolic_outputs
    # Average MC Dropout samples if applicable
    if use_mc_dropout:
        final_stacked_probs = np.mean(np.stack(all_stacked_probs, axis=0), axis=0)
        logger.info(f"Stacked ensemble averaged {mc_dropout_samples} MC Dropout samples.")
    else:
        final_stacked_probs = np.concatenate(all_stacked_probs, axis=0) # Concatenate batches
    logger.info(f"Stacked ensemble prediction complete. Probabilities for {len(image_paths)} images.")
    return final_stacked_probs, first_model_symbolic_outputs
def ensemble_predict_distilled_combined(
    student_model_path: str,
    image_paths: List[str],
    config: Dict[str, Any]
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs inference using a single, distilled student model.
    Args:
        student_model_path (str): Path to the trained student model.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Bongard problem probabilities (after softmax) from the student model [num_images, num_classes].
            - List of symbolic outputs (from the student model's inference).
    """
    logger.info("Performing distilled model prediction.")
    
    if not student_model_path:
        logger.error("Student model path not provided for distilled prediction.")
        return np.empty((len(image_paths), config['model']['num_classes'])), []
    # Load student model
    object_detector_student = RealObjectDetector(model_path=config['model']['object_detector_model_path'])
    student_model = PerceptionModule(config, object_detector_student).to(DEVICE)
    student_model.load_weights(student_model_path)
    student_model.eval()
    logger.info("Student model loaded for inference.")
    # Create a DALI loader for inference
    inference_pipeline, inference_loader = build_dali_loader(
        file_list=image_paths,
        labels_list=[0] * len(image_paths), # Dummy labels
        config=config,
        mode='inference'
    )
    logger.info("DALI inference loader initialized for distilled model.")
    all_student_probs = []
    all_student_symbolic_outputs = []
    with torch.no_grad():
        for batch_idx, data in enumerate(inference_loader):
            images_view1 = data['view1'].permute(0, 3, 1, 2).to(DEVICE) # NCHW
            
            # Pass None for ground_truth_json_strings as it's inference on real images
            # PerceptionModule.forward now returns bongard_logits, detected_objects_batch, aggregated_outputs
            bongard_logits, detected_objects_batch, aggregated_outputs = student_model(images_view1, None)
            
            probs = F.softmax(bongard_logits, dim=-1).cpu().numpy()
            all_student_probs.append(probs)
            # Collect symbolic outputs
            for b_idx in range(images_view1.shape[0]):
                # Ensure detected_objects_batch and aggregated_outputs are correctly indexed for a single image
                single_detected_objects = detected_objects_batch[b_idx]
                single_attribute_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['attribute_logits'].items()}
                single_relation_logits = {'global_relation': aggregated_outputs['relation_logits'][b_idx].unsqueeze(0)} # Assuming global relation
                scene_graph = student_model.extract_scene_graph(
                    single_detected_objects,
                    single_attribute_logits,
                    single_relation_logits
                )
                scene_graph["bongard_prediction"] = int(torch.argmax(bongard_logits[b_idx]).item())
                scene_graph["bongard_probs"] = probs[b_idx].tolist()
                
                all_student_symbolic_outputs.append(scene_graph)
    final_student_probs = np.concatenate(all_student_probs, axis=0)
    logger.info(f"Distilled model prediction complete. Probabilities for {len(image_paths)} images.")
    return final_student_probs, all_student_symbolic_outputs
def perform_ensemble_inference_combined(
    inference_mode: str,
    image_paths: List[str],
    config: Dict[str, Any],
    base_model_paths: Optional[List[str]] = None,
    model_weights: Optional[List[float]] = None,
    meta_learner_path: Optional[str] = None,
    student_model_path: Optional[str] = None,
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    High-level function to perform ensemble inference based on the specified mode.
    Args:
        inference_mode (str): The ensemble inference strategy to use.
                              Options: 'simple_average', 'weighted_average', 'stacked', 'distilled'.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
        base_model_paths (Optional[List[str]]): Paths to the trained base PerceptionModule models.
                                                 Required for 'simple_average', 'weighted_average', 'stacked'.
        model_weights (Optional[List[float]]): Weights for weighted averaging.
                                               Required for 'weighted_average'.
        meta_learner_path (Optional[str]): Path to the trained MetaLearner model.
                                           Required for 'stacked'.
        student_model_path (Optional[str]): Path to the trained student model.
                                            Required for 'distilled'.
        use_mc_dropout (bool): If True, enable MC Dropout during inference (for base models).
        mc_dropout_samples (int): Number of forward passes for MC Dropout.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Final Bongard problem probabilities (after softmax) [num_images, num_classes].
            - List of symbolic outputs (from a representative model).
    """
    logger.info(f"Starting ensemble inference with mode: '{inference_mode}'")
    if inference_mode == 'simple_average':
        if not base_model_paths:
            logger.error("Base model paths are required for 'simple_average' inference.")
            return np.empty((len(image_paths), config['model']['num_classes'])), []
        
        loaded_models = []
        object_detector = RealObjectDetector(model_path=config['model']['object_detector_model_path'])
        for path in base_model_paths:
            model = PerceptionModule(config, object_detector).to(DEVICE)
            model.load_weights(path)
            loaded_models.append(model)
        
        # ensemble_predict_orchestrator_base is assumed to be available from previous code block
        return ensemble_predict_orchestrator_base(
            models=loaded_models,
            image_paths=image_paths,
            config=config,
            use_mc_dropout=use_mc_dropout,
            mc_dropout_samples=mc_dropout_samples,
            model_weights=None # Explicitly None for simple average
        )
    
    elif inference_mode == 'weighted_average':
        if not base_model_paths or not model_weights:
            logger.error("Base model paths and model weights are required for 'weighted_average' inference.")
            return np.empty((len(image_paths), config['model']['num_classes'])), []
        
        # ensemble_predict_weighted_combined is defined in this block
        return ensemble_predict_weighted_combined(
            model_paths=base_model_paths,
            image_paths=image_paths,
            config=config,
            model_weights=model_weights,
            use_mc_dropout=use_mc_dropout,
            mc_dropout_samples=mc_dropout_samples
        )
    elif inference_mode == 'stacked':
        if not base_model_paths or not meta_learner_path:
            logger.error("Base model paths and meta-learner path are required for 'stacked' inference.")
            return np.empty((len(image_paths), config['model']['num_classes'])), []
        
        # ensemble_predict_stacked_combined is defined in this block
        return ensemble_predict_stacked_combined(
            base_model_paths=base_model_paths,
            meta_learner_path=meta_learner_path,
            image_paths=image_paths,
            config=config,
            use_mc_dropout=use_mc_dropout,
            mc_dropout_samples=mc_dropout_samples
        )
    elif inference_mode == 'distilled':
        if not student_model_path:
            logger.error("Student model path is required for 'distilled' inference.")
            return np.empty((len(image_paths), config['model']['num_classes'])), []
        
        # ensemble_predict_distilled_combined is defined in this block
        return ensemble_predict_distilled_combined(
            student_model_path=student_model_path,
            image_paths=image_paths,
            config=config
        )
    else:
        logger.error(f"Invalid inference mode: '{inference_mode}'. Choose from 'simple_average', 'weighted_average', 'stacked', 'distilled'.")
        return np.empty((len(image_paths), config['model']['num_classes'])), []
# --- Utility Functions ---
def _plot_training_history(history, save_path="training_history.png"):
    """
    Plots the training and validation loss and accuracy over epochs.
    Args:
        history (dict): Dictionary containing lists of 'train_loss', 'val_loss', etc.
        save_path (str): Path to save the plot.
    """
    if not history or not history.get('train_loss') or not history.get('val_loss'):
        logger.warning("Insufficient history data to plot training history. Skipping plot.")
        return
    plt.figure(figsize=(12, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy (Bongard accuracy is the main one for overall problem)
    plt.subplot(1, 2, 2)
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Bongard Accuracy')
        plt.title('Bongard Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    else:
        logger.warning("Validation accuracy data not found in history. Skipping accuracy plot.")
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()
def _plot_reliability_diagram(probs, labels, num_bins=10, save_path="reliability_diagram.png"):
    """
    Plots a reliability diagram to assess model calibration.
    Args:
        probs (np.ndarray): Predicted probabilities (e.g., from Bongard classifier softmax).
        labels (np.ndarray): True binary labels (0 or 1).
        num_bins (int): Number of bins for the reliability diagram.
        save_path (str): Path to save the plot.
    """
    if len(probs) == 0 or len(labels) == 0:
        logger.warning("No probabilities or labels provided for reliability diagram. Skipping plot.")
        return
    
    # Ensure probabilities are for the positive class (label 1)
    if probs.ndim > 1 and probs.shape[1] > 1:
        # Assuming binary classification, take probability of class 1
        probs_positive_class = probs[:, 1]
    else:
        probs_positive_class = probs # Assume already single probability for positive class
    
    bins = np.linspace(0., 1. + 1e-8, num_bins + 1)
    bin_indices = np.digitize(probs_positive_class, bins) - 1
    
    bin_sums = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_sums[i] = np.mean(labels[mask]) # Observed accuracy in bin
            bin_counts[i] = np.mean(probs_positive_class[mask]) # Average predicted probability in bin
    
    # Filter out empty bins
    valid_bins = bin_counts > 0
    bin_counts = bin_counts[valid_bins]
    bin_sums = bin_sums[valid_bins]
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated') # Diagonal line
    plt.plot(bin_counts, bin_sums, marker='o', linestyle='-', label='Model calibration')
    
    plt.title('Reliability Diagram')
    plt.xlabel('Mean Predicted Probability in Bin')
    plt.ylabel('Fraction of Positives (True Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Reliability diagram plot saved to {save_path}")
    plt.close()
def calibrate_model(model: PerceptionModule, val_dali_iter: DALIGenericIterator, config: Dict[str, Any]):
    """
    Calibrates the model using temperature scaling on the Bongard classifier's output.
    This is typically done on a held-out calibration set (validation set).
    Args:
        model (PerceptionModule): The trained perception model.
        val_dali_iter (DALIGenericIterator): DALI iterator for validation data (calibration set).
        config (dict): Configuration dictionary.
    """
    logger.info("Starting model calibration using temperature scaling.")
    # The PerceptionModule might have a direct bongard_head or a way to access it.
    # Assuming the bongard_head is directly accessible or can be passed.
    if not hasattr(model, 'bongard_head'):
        logger.warning("Bongard classification head ('bongard_head') not found in model. Skipping calibration.")
        return
    
    # Create a temperature parameter, initialized to 1.0 (no scaling)
    temperature = nn.Parameter(torch.ones(1).to(model.device))
    
    # Define optimizer for temperature (LBFGS is often used for this)
    temp_optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')
    
    # Collect all Bongard logits and true labels from the validation set
    all_bongard_logits = []
    all_bongard_labels = []
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_dali_iter.reset()  # Ensure iterator is reset
        for i, data in enumerate(tqdm(val_dali_iter, desc="Collecting calibration data")):
            images_view1 = data['view1'].permute(0, 3, 1, 2).to(model.device)  # NCHW
            gts_json_strings_batch = [s.decode('utf-8') for s in data['gts_json'].cpu().numpy()]  # Decode JSON strings
            
            # For synthetic data, extract actual labels from GT JSON
            if config['data']['use_synthetic_data']:
                extracted_labels = []
                for gt_json_str in gts_json_strings_batch:
                    try:
                        gt_dict = json.loads(gt_json_str)
                        # Validate against schema if needed
                        jsonschema.validate(instance=gt_dict, schema=BONGARD_GT_JSON_SCHEMA)
                        extracted_labels.append(gt_dict['bongard_label'])
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode GT JSON for sample {i}: {gt_json_str}. Using dummy label 0.")
                        extracted_labels.append(0)
                    except jsonschema.ValidationError as ve:
                        logger.error(f"GT JSON validation error for sample {i}: {ve.message}. Using dummy label 0.")
                        extracted_labels.append(0)
                labels = torch.tensor(extracted_labels, dtype=torch.long, device=model.device)
            else:
                labels = data['labels'].squeeze(-1).to(model.device)  # Remove last dim from DALI
            
            # Forward pass to get raw logits from the bongard_head
            bongard_logits, _, _ = model(images_view1, gts_json_strings_batch)  # Pass GT for synthetic data handling
            
            all_bongard_logits.append(bongard_logits)
            all_bongard_labels.append(labels)
    
    # Release DALI GPU memory if possible (might not be directly exposed by DALIGenericIterator)
    if hasattr(val_dali_iter, 'release_gpu_memory'):
        val_dali_iter.release_gpu_memory()
    
    if not all_bongard_logits:
        logger.warning("No Bongard logits collected for calibration. Skipping.")
        return
    
    all_bongard_logits = torch.cat(all_bongard_logits).to(model.device)
    all_bongard_labels = torch.cat(all_bongard_labels).to(model.device)
    
    logger.info(f"Collected {all_bongard_logits.shape[0]} samples for calibration.")
    
    # Define NLL loss for calibration (CrossEntropyLoss is equivalent to NLLLoss + LogSoftmax)
    nll_criterion = nn.CrossEntropyLoss().to(model.device)
    
    def eval_loss():
        temp_optimizer.zero_grad()
        # Scale logits by temperature before computing loss
        loss = nll_criterion(all_bongard_logits / temperature, all_bongard_labels)
        loss.backward()
        return loss
    
    logger.info("Optimizing temperature parameter...")
    temp_optimizer.step(eval_loss)
    
    calibrated_temperature = temperature.item()
    
    # Store temperature as part of the model for inference
    # A common way is to wrap the classifier or add it as an an attribute
    # Here, we'll assume PerceptionModule has a `set_temperature` method or similar
    # or that the `bongard_head` can be directly modified.
    # For simplicity, let's assume we can add it directly to the model for now.
    model.calibrated_temperature = calibrated_temperature
    logger.info(f"Model calibrated. Optimal temperature: {calibrated_temperature:.4f}")
    
    # Plot reliability diagram before and after calibration
    if config['debug']['plot_reliability_diagram']:
        uncalibrated_probs = F.softmax(all_bongard_logits, dim=1).cpu().numpy()
        calibrated_probs = F.softmax(all_bongard_logits / calibrated_temperature, dim=1).cpu().numpy()
        labels_np = all_bongard_labels.cpu().numpy()
        
        _plot_reliability_diagram(uncalibrated_probs, labels_np, save_path=os.path.join(config['training']['checkpoint_dir'], "reliability_diagram_uncalibrated.png"))
        _plot_reliability_diagram(calibrated_probs, labels_np, save_path=os.path.join(config['training']['checkpoint_dir'], "reliability_diagram_calibrated.png"))
        
        # Calculate ECE (Expected Calibration Error) using brier_score_loss as a proxy
        # More robust ECE calculation would involve binning and comparing accuracy vs confidence.
        # For binary classification, brier_score_loss is a good indicator of calibration.
        ece_uncalibrated = brier_score_loss(labels_np, uncalibrated_probs[:, 1])
        ece_calibrated = brier_score_loss(labels_np, calibrated_probs[:, 1])
        logger.info(f"ECE (uncalibrated): {ece_uncalibrated:.4f}, ECE (calibrated): {ece_calibrated:.4f}")

def apply_structured_pruning(model: nn.Module, config: Dict[str, Any]):
    """
    Applies structured pruning to the model based on configuration.
    This implementation performs L1-norm based structured pruning on Conv2d layers
    within the AttributeClassifier's backbone.
    Args:
        model (nn.Module): The PyTorch model to prune.
        config (dict): Configuration dictionary, containing pruning parameters.
    Returns:
        nn.Module: The pruned model.
    """
    logger.info("Starting structured pruning process.")
    
    pruning_enabled = config['training'].get('enable_structured_pruning', False)
    if not pruning_enabled:
        logger.info("Structured pruning is disabled in configuration. Skipping.")
        return model # Return original model if pruning is disabled

    pruning_amount_per_iteration = config['training'].get('pruning_amount_per_iteration', 0.2)
    pruning_iterations = config['training'].get('pruning_iterations', 1)
    pruning_fine_tune_epochs = config['training'].get('pruning_fine_tune_epochs', 0) # Epochs to fine-tune after each step
    
    # Store original device to move model back later
    original_device = next(model.parameters()).device
    model.cpu() # Move model to CPU for pruning operations (often more stable)
    
    # Ensure model is in eval mode before pruning to capture correct statistics if any are used
    model.eval() 

    for i in range(pruning_iterations):
        logger.info(f"Pruning iteration {i+1}/{pruning_iterations}...")
        for name, module in model.named_modules():
            # Target convolutional layers in the AttributeClassifier backbone
            # Adjust the 'in' operator check based on your actual model structure if needed
            if isinstance(module, nn.Conv2d) and 'attribute_classifier.backbone' in name:
                # Apply L1-norm structured pruning along dimension 0 (output channels/filters)
                prune.ln_structured(module, name='weight', amount=pruning_amount_per_iteration, n=1, dim=0)
                logger.debug(f"Applied L1 structured pruning to {name} (amount={pruning_amount_per_iteration}).")
            # You can add similar logic for linear layers if desired:
            # elif isinstance(module, nn.Linear) and 'some_linear_layer' in name:
            #     prune.ln_structured(module, name='weight', amount=pruning_amount_per_iteration, n=1, dim=0)
        
        logger.info(f"Pruning iteration {i+1} complete.")
        
        # --- Fine-tuning after pruning (Crucial for iterative pruning) ---
        if pruning_fine_tune_epochs > 0:
            logger.info(f"Fine-tuning model for {pruning_fine_tune_epochs} epochs after pruning iteration {i+1}.")
            # You would need to pass appropriate data loaders (train/val) and a criterion/optimizer
            # This calls the new `fine_tune_model` function.
            # Note: You'll need to pass actual train/val loaders to `apply_structured_pruning`
            # or make them accessible globally for this to work.
            # For demonstration, we'll assume they are available or this part is conceptual.
            # fine_tune_model(model, train_loader, val_loader, config, epochs=pruning_fine_tune_epochs)
            logger.warning("Fine-tuning step is conceptual here. Implement `fine_tune_model` and pass data loaders.")
        else:
            logger.info("No fine-tuning configured for this pruning step.")

    # Make pruning permanent: Remove the reparametrization
    # This makes the model smaller by actually removing the zeroed parameters
    logger.info("Removing pruning reparametrization to finalize the pruned model.")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and prune.is_pruned(module):
            prune.remove(module, 'weight')
            logger.debug(f"Removed pruning reparametrization for {name}.")
    
    model.to(original_device) # Move model back to original device
    logger.info("Structured pruning process finished. Model is now pruned.")
    return model

    
    logger.info("Starting Post-Training Quantization (PTQ).")
    model.eval()  # Set model to evaluation mode
    
    # Fuse modules for better quantization performance
    # This is highly dependent on the model architecture.
    # For a generic PerceptionModule, assuming it has a 'fuse_model' method or similar
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
        logger.info("Model modules fused for PTQ.")
    else:
        logger.warning("Model does not have a fuse_model method. Fusion skipped.")
    
    # Move model to CPU before quantization, as PTQ often works best on CPU
    model_cpu = model.cpu()
    
    qconfig = tq.get_default_qconfig('fbgemm')  # Use fbgemm for x86 CPU, or qnnpack for ARM
    model_cpu.qconfig = qconfig
    tq.prepare(model_cpu, inplace=True)
    logger.info("Model prepared for static quantization.")
    
    # Calibrate the model with a representative dataset
    logger.info("Calibrating model for PTQ with validation data...")
    with torch.no_grad():
        data_loader.reset()  # Reset for calibration
        for i, data in enumerate(tqdm(data_loader, desc="PTQ Calibration")):
            images_view1 = data['view1'].permute(0, 3, 1, 2).cpu()  # Move to CPU for calibration
            # Forward pass to collect quantization statistics
            # Pass None for ground_truth_json_strings during calibration
            model_cpu(images_view1, None)
            if i >= 100:  # Calibrate on a subset of data for speed
                break
    # Release DALI GPU memory if applicable
    if hasattr(data_loader, 'release_gpu_memory'):
        data_loader.release_gpu_memory()
    logger.info("Calibration complete.")
    
    # Convert the model to quantized version
    tq.convert(model_cpu, inplace=True)
    logger.info("Model converted to quantized (PTQ) version.")
    return model_cpu  # Return the quantized CPU model

def quantize_model_qat(model: nn.Module, config: Dict[str, Any]):
    """
    Applies Quantization-Aware Training (QAT) to the model.
    This inserts fake quantization modules during training to simulate quantization effects,
    allowing the model to adapt.
    Args:
        model (nn.Module): The PyTorch model.
        config (dict): Configuration dictionary.
    Returns:
        nn.Module: The model prepared for QAT.
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.warning("torch.quantization not available. Skipping QAT.")
        return model
    
    logger.info("Preparing model for Quantization-Aware Training (QAT).")
    model.train()  # Set model to training mode
    
    # Fuse modules for better quantization performance (same as PTQ)
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
        logger.info("Model modules fused for QAT.")
    else:
        logger.warning("Model does not have a fuse_model method. Fusion skipped.")
    
    # Specify quantization configuration for QAT
    qconfig_qat = tq.get_default_qat_qconfig('fbgemm')  # Or 'qnnpack'
    
    # Prepare model for QAT
    model.qconfig = qconfig_qat
    tq.prepare_qat(model, inplace=True)
    logger.info("Model prepared for Quantization-Aware Training (QAT).")
    logger.info("Remember to train the model for a few epochs after QAT preparation.")
    return model

# --- Main Execution Block ---
logger.info("--- Starting Main Bongard Ensemble Workflow ---")

# 1. Load Data
all_image_paths, all_labels, all_difficulty_scores = [], [], []
if CONFIG['data']['use_synthetic_data']:
    logger.info("Main execution: Synthetic data generation will be handled by DALI's external source.")
    # Dummy lists for train_test_split, as actual data will be generated on-the-fly
    dummy_paths = [f"synthetic_img_{i}.png" for i in range(CONFIG['data']['synthetic_samples'])]
    dummy_labels = np.array([random.randint(0, CONFIG['model']['num_classes'] - 1) for _ in range(CONFIG['data']['synthetic_samples'])])
    dummy_difficulty_scores = np.random.rand(CONFIG['data']['synthetic_samples'])
    
    train_image_paths, val_image_paths, train_labels, val_labels, train_difficulty_scores, val_difficulty_scores = \
        train_test_split(dummy_paths, dummy_labels, dummy_difficulty_scores,
                         test_size=CONFIG['data']['train_test_split_ratio'],
                         random_state=42, stratify=dummy_labels)
    
    # For synthetic data, the actual dataset objects are created inside _run_single_training_session_ensemble
    # and passed to DALI's external source.
else:
    logger.info(f"Main execution: Loading real Bongard data from {DATA_ROOT_PATH}.")
    all_image_paths, all_labels, all_difficulty_scores = load_bongard_data(DATA_ROOT_PATH)
    
    # Split data into training and validation sets
    train_image_paths, val_image_paths, train_labels, val_labels, train_difficulty_scores, val_difficulty_scores = \
        train_test_split(all_image_paths, all_labels, all_difficulty_scores,
                         test_size=CONFIG['data']['train_test_split_ratio'],
                         random_state=42,  # Ensure reproducibility of split
                         stratify=all_labels)  # Stratify to maintain class distribution

logger.info(f"Train samples: {len(train_image_paths)}, Validation samples: {len(val_image_paths)}")

ENSEMBLE_CHECKPOINT_DIR = os.path.join(CONFIG['training']['checkpoint_dir'], 'ensemble_members')
META_LEARNER_CHECKPOINT_DIR = os.path.join(CONFIG['training']['checkpoint_dir'], 'meta_learner')
DISTILLED_STUDENT_CHECKPOINT_DIR = os.path.join(CONFIG['training']['checkpoint_dir'], 'distilled_student')

os.makedirs(ENSEMBLE_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(META_LEARNER_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DISTILLED_STUDENT_CHECKPOINT_DIR, exist_ok=True)

NUM_ENSEMBLE_MEMBERS = CONFIG['ensemble']['num_members']
trained_base_model_paths = []
all_members_val_predictions_logits = []
all_members_val_labels = []  # Should be identical for all members, but collect for clarity
all_members_best_metrics = []

# 2. Train Ensemble Members (if enabled)
if CONFIG['ensemble']['train_members']:
    logger.info(f"Training {NUM_ENSEMBLE_MEMBERS} ensemble members.")
    trained_base_model_paths, all_members_val_predictions_logits, all_members_val_labels, all_members_best_metrics = \
        train_ensemble_members_orchestrator_combined(
            num_members=NUM_ENSEMBLE_MEMBERS,
            start_seed=42,
            ensemble_output_dir=ENSEMBLE_CHECKPOINT_DIR,
            epochs_override=CONFIG['model']['epochs'],  # Use epochs from config
            train_image_paths=train_image_paths,
            train_labels=train_labels,
            train_difficulty_scores=train_difficulty_scores,
            val_image_paths=val_image_paths,
            val_labels=val_labels,
            val_difficulty_scores=val_difficulty_scores,
            member_configs=None  # Auto-generate diverse configs
        )
    logger.info(f"Trained {len(trained_base_model_paths)} ensemble members.")
else:
    logger.warning("Skipping ensemble member training as 'ensemble.train_members' is False.")
    # Attempt to load existing models if training is skipped
    existing_models = glob.glob(os.path.join(ENSEMBLE_CHECKPOINT_DIR, 'member_*_best_perception_model.pt'))
    if CONFIG['training']['use_swa']:
        existing_models.extend(glob.glob(os.path.join(ENSEMBLE_CHECKPOINT_DIR, 'member_*_best_perception_model_swa.pt')))
    if CONFIG['training']['use_qat']:
        existing_models.extend(glob.glob(os.path.join(ENSEMBLE_CHECKPOINT_DIR, 'member_*_best_perception_model_quantized.pt')))
    
    if existing_models:
        trained_base_model_paths = sorted(existing_models)  # Sort for consistent order
        logger.info(f"Loaded {len(trained_base_model_paths)} existing ensemble member checkpoints.")
        # Dummy val_predictions_logits and labels if not actually trained
        num_val_samples = len(val_image_paths)
        num_classes = CONFIG['model']['num_classes']
        dummy_val_logits = np.random.rand(num_val_samples, num_classes)
        dummy_val_labels = val_labels
        all_members_val_predictions_logits = [dummy_val_logits] * len(trained_base_model_paths)
        all_members_val_labels = [dummy_val_labels] * len(trained_base_model_paths)
        all_members_best_metrics = [{'val_accuracy': 0.5, 'val_loss': 0.7}] * len(trained_base_model_paths)  # Dummy metrics
    else:
        logger.error("No existing ensemble member checkpoints found. Cannot proceed without trained models.")
        raise RuntimeError("No trained ensemble models available for inference.")


# 3. Train Meta-Learner (if stacking is enabled)
meta_learner_path = None
if CONFIG['ensemble']['use_stacking']:
    logger.info("Training meta-learner for stacking ensemble.")
    # Concatenate all validation predictions from base models
    # Ensure all_members_val_predictions_logits is not empty
    if all_members_val_predictions_logits:
        concatenated_val_predictions = np.concatenate(all_members_val_predictions_logits, axis=0)
        # Take one set of validation labels (they should be identical across members)
        single_val_labels = all_members_val_labels[0] if all_members_val_labels else val_labels
        
        meta_learner_path = train_meta_learner_orchestrator_combined(
            base_model_val_predictions_logits=all_members_val_predictions_logits,
            val_labels=single_val_labels,
            meta_learner_output_dir=META_LEARNER_CHECKPOINT_DIR,
            epochs=CONFIG['training']['calibration_epochs'],  # Use calibration epochs for meta-learner
            batch_size=CONFIG['model']['batch_size'],
            learning_rate=CONFIG['model']['initial_learning_rate'],
            random_seed=42
        )
        logger.info(f"Meta-learner trained and saved to: {meta_learner_path}")
    else:
        logger.warning("Cannot train meta-learner: No base model validation predictions available.")
else:
    logger.info("Skipping meta-learner training as 'ensemble.use_stacking' is False.")

# 4. Train Distilled Student Model (if distillation is enabled)
distilled_student_model_path = None
if CONFIG['ensemble']['use_distillation']:
    logger.info("Training distilled student model.")
    distilled_student_model_path = train_distilled_student_orchestrator_combined(
        teacher_model_paths=trained_base_model_paths,
        train_image_paths=train_image_paths,
        train_labels=train_labels,
        val_image_paths=val_image_paths,
        val_labels=val_labels,
        val_difficulty_scores=val_difficulty_scores,
        student_output_dir=DISTILLED_STUDENT_CHECKPOINT_DIR,
        epochs_override=CONFIG['model']['epochs'],
        random_seed=43,  # Different seed for student
        student_config_override=CONFIG['ensemble']['distilled_student_config_override']
    )
    logger.info(f"Distilled student model trained and saved to: {distilled_student_model_path}")
else:
    logger.info("Skipping distilled student model training as 'ensemble.use_distillation' is False.")

# 5. Perform Ensemble Inference
logger.info(f"Performing ensemble inference using mode: '{CONFIG['ensemble']['inference_mode']}'")
INFERENCE_IMAGE_PATHS = val_image_paths[:min(10, len(val_image_paths))]  # Use a small subset of validation for inference demo
if not INFERENCE_IMAGE_PATHS:
    logger.warning("No images available for inference demonstration.")

averaged_probs, symbolic_outputs_example = None, []
member_weights = None # Initialize to None

if CONFIG['ensemble']['inference_mode'] == 'weighted_average':
    member_weights = calculate_model_weights(all_members_best_metrics, metric_name='val_accuracy', minimize_metric=False)
    if not member_weights:
        logger.warning("Could not calculate weights. Falling back to simple average for inference.")
        CONFIG['ensemble']['inference_mode'] = 'simple_average'  # Fallback
    else:
        logger.info(f"Calculated member weights for inference: {member_weights}")

try:
    averaged_probs, symbolic_outputs_example = perform_ensemble_inference_combined(
        inference_mode=CONFIG['ensemble']['inference_mode'],
        image_paths=INFERENCE_IMAGE_PATHS,
        config=CONFIG,
        base_model_paths=trained_base_model_paths,
        model_weights=member_weights,
        meta_learner_path=meta_learner_path,
        student_model_path=distilled_student_model_path,
        use_mc_dropout=CONFIG['model']['mc_dropout_samples'] > 0,
        mc_dropout_samples=CONFIG['model']['mc_dropout_samples']
    )
    
    logger.info("\n--- Ensemble Prediction Results ---")
    for i, img_path in enumerate(INFERENCE_IMAGE_PATHS):
        logger.info(f"Image: {img_path}")
        logger.info(f"  Ensemble Averaged Probabilities: {averaged_probs[i]}")
        logger.info(f"  Ensemble Predicted Class: {np.argmax(averaged_probs[i])}")
    
    if symbolic_outputs_example:
        output_json_path = os.path.join(ENSEMBLE_CHECKPOINT_DIR, "ensemble_inference_symbolic_output_example.json")
        with open(output_json_path, 'w') as f:
            json.dump(symbolic_outputs_example, f, indent=4)
        logger.info(f"Example symbolic output from one ensemble member saved to: {output_json_path}")

except Exception as e:
    logger.error(f"Error during ensemble inference: {e}")

# 6. Deployment Optimizations (Conceptual)
logger.info("--- Starting Deployment Optimizations ---")
final_model_for_deployment = None

if CONFIG['ensemble']['inference_mode'] == 'distilled' and distilled_student_model_path:
    object_detector_deployment = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
    final_model_for_deployment = PerceptionModule(CONFIG, object_detector_deployment).to(DEVICE)
    final_model_for_deployment.load_weights(distilled_student_model_path)
    logger.info("Loaded distilled student model for deployment optimization.")
elif trained_base_model_paths:
    object_detector_deployment = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
    final_model_for_deployment = PerceptionModule(CONFIG, object_detector_deployment).to(DEVICE)
    final_model_for_deployment.load_weights(trained_base_model_paths[0])
    logger.info("Loaded first base model as representative for deployment optimization.")

if final_model_for_deployment:
    # Apply Post-Training Quantization (PTQ) if not already QAT trained
    if not CONFIG['training']['use_qat'] and CONFIG['training']['calibrate_model']:
        logger.info("Applying Post-Training Quantization (PTQ).")
        _, calibration_loader = build_dali_loader(
            file_list=val_image_paths,
            labels_list=val_labels,
            # Ensure difficulty_scores_list is passed if your build_dali_loader expects it
            difficulty_scores_list=val_difficulty_scores, # Assuming this is available
            config=CONFIG,
            mode='val'  # Use validation data for calibration
        )
        final_model_for_deployment = quantize_model_ptq(final_model_for_deployment, calibration_loader, CONFIG)
        logger.info("PTQ applied.")
    elif CONFIG['training']['use_qat']:
        logger.info("Model was trained with QAT. Skipping separate PTQ step.")
    else:
        logger.info("Skipping quantization as neither PTQ nor QAT is enabled/configured.")
    
    # --- Apply Structured Pruning (now functional) ---
    final_model_for_deployment = apply_structured_pruning(final_model_for_deployment, CONFIG)
    
    # Export ONNX (example for the final model)
    if hasattr(final_model_for_deployment, 'export_onnx'):
        onnx_output_path = os.path.join(CONFIG['training']['checkpoint_dir'], "final_model_exported.onnx")
        final_model_for_deployment.export_onnx(output_path=onnx_output_path)
        logger.info(f"Final model exported to ONNX: {onnx_output_path}")
    else:
        logger.warning("Final model does not have an 'export_onnx' method. Skipping ONNX export.")
else:
    logger.warning("No final model available for deployment optimizations (e.g., no members trained or paths invalid).")

logger.info("Bongard Ensemble Workflow execution finished.")

# --- XAI (Grad-CAM) Visualization ---
if CONFIG['debug']['enable_grad_cam'] and HAS_PYTORCH_GRADCAM:
    logger.info("Generating Grad-CAM visualizations for example images.")
    
    # Load a single representative model (e.g., the first trained base model or distilled student)
    grad_cam_model_path = None
    if CONFIG['ensemble']['inference_mode'] == 'distilled' and distilled_student_model_path:
        grad_cam_model_path = distilled_student_model_path
    elif trained_base_model_paths:
        grad_cam_model_path = trained_base_model_paths[0]

    if grad_cam_model_path:
        object_detector_cam = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
        model_for_grad_cam = PerceptionModule(CONFIG, object_detector_cam).to(DEVICE)
        model_for_grad_cam.load_weights(grad_cam_model_path)
        model_for_grad_cam.eval() # Ensure model is in eval mode
        
        # Select a few images for visualization (e.g., from the inference set)
        sample_images_for_cam = INFERENCE_IMAGE_PATHS[:min(3, len(INFERENCE_IMAGE_PATHS))]
        
        os.makedirs('./grad_cam_outputs', exist_ok=True) # Ensure output directory exists

        for img_path in sample_images_for_cam:
            try:
                img_pil = Image.open(img_path).convert('RGB')
                input_tensor = T.Compose([
                    T.Resize(CONFIG['data']['image_size']),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])(img_pil).unsqueeze(0).to(DEVICE)
                
                dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})]
                bongard_logits, _, _ = model_for_grad_cam(input_tensor, dummy_gts_json)
                predicted_class = torch.argmax(bongard_logits).item()
                
                logger.info(f"Generating Grad-CAM for image: {img_path}, Predicted Class: {predicted_class}")
                
                target_layer = None
                # Attempt to find the last convolutional layer in the backbone
                if hasattr(model_for_grad_cam.attribute_classifier.backbone, 'features'):
                    target_layer = model_for_grad_cam.attribute_classifier.backbone.features[-1]
                elif hasattr(model_for_grad_cam.attribute_classifier.backbone, 'blocks'):
                    target_layer = model_for_grad_cam.attribute_classifier.backbone.blocks[-1]
                elif hasattr(model_for_grad_cam.attribute_classifier.backbone, 'conv_head'):
                    target_layer = model_for_grad_cam.attribute_classifier.backbone.conv_head
                elif hasattr(model_for_grad_cam.attribute_classifier.backbone, 'head'): # For some timm models
                    if isinstance(model_for_grad_cam.attribute_classifier.backbone.head, nn.Sequential):
                        for m in reversed(model_for_grad_cam.attribute_classifier.backbone.head):
                            if isinstance(m, nn.Conv2d):
                                target_layer = m
                                break
                    elif isinstance(model_for_grad_cam.attribute_classifier.backbone.head, nn.Conv2d):
                        target_layer = model_for_grad_cam.attribute_classifier.backbone.head

                if target_layer:
                    generate_grad_cam(
                        model=model_for_grad_cam,
                        input_tensor=input_tensor,
                        target_layer=target_layer,
                        target_category=predicted_class,
                        image_path=img_path,
                        save_dir='./grad_cam_outputs'
                    )
                else:
                    logger.warning(f"Could not find a suitable target layer for Grad-CAM in {model_for_grad_cam.attribute_classifier.backbone_name}. Skipping for this image.")
            except Exception as e:
                logger.error(f"Failed to generate Grad-CAM for {img_path}: {e}")
    else:
        logger.warning("No model available for Grad-CAM visualization.")
else:
    logger.info("Grad-CAM generation skipped (either disabled in config or pytorch_grad_cam not installed).")

if HAS_WANDB and CONFIG['training']['use_wandb']:
    import wandb
    wandb.finish()

# Ensure all necessary imports from your project are available in this scope.
# This includes:
# - logging, os, glob, numpy, json, random, sys
# - torch, torch.nn, torch.optim, torch.nn.functional, torchvision.transforms
# - sklearn.model_selection
# - All your custom classes: PerceptionModule, RealObjectDetector, MetaLearner, KnowledgeReplayBuffer, MixupCutmix, BongardGenerator, BongardSyntheticDataset, BongardExternalSource, CurriculumSampler
# - All your custom functions: setup_logging, load_bongard_data, set_seed, LabelSmoothingCrossEntropy, FeatureConsistencyLoss, DistillationLoss, symbolic_consistency_loss, build_dali_loader, calculate_model_weights, _run_single_training_session_ensemble, _validate_model_ensemble, train_ensemble_members_orchestrator_combined, train_meta_learner_orchestrator_combined, train_distilled_student_orchestrator_combined, perform_ensemble_inference_combined, _plot_training_history, _plot_reliability_diagram, apply_structured_pruning, quantize_model_ptq, calibrate_model, generate_grad_cam, fine_tune_model (if implemented)
# - All global constants: CONFIG, DEVICE, IMAGENET_MEAN, IMAGENET_STD, DATA_ROOT_PATH, ATTRIBUTE_FILL_MAP, ATTRIBUTE_FILL_MAP_INV, RELATION_MAP, RELATION_MAP_INV, YOLO_CLASS_MAP, BONGARD_GT_JSON_SCHEMA, HAS_DALI, HAS_PYG, HAS_KORNIA, HAS_TIMM, HAS_LORA, HAS_SOPHIA, HAS_TIMM_OPTIM, HAS_SAM, HAS_ELASTIC_TRANSFORM, HAS_TORCH_QUANTIZATION, HAS_PYTORCH_GRADCAM

def main_project_workflow():
    """
    Orchestrates the entire Bongard problem-solving project workflow.
    This function calls all necessary components in sequence based on the CONFIG settings.
    It handles DDP setup, data loading, training, inference, deployment optimizations, and XAI.
    """
    # 1. Configuration and Logging Setup
    setup_logging(CONFIG['debug']['log_level'])
    logger.info("--- Starting Main Bongard Ensemble Workflow ---")
    
    set_seed(CONFIG['model']['random_seed'])
    logger.info(f"Global random seed set to {CONFIG['model']['random_seed']}.")

    # 2. Data Loading and Splitting
    all_image_paths, all_labels, all_difficulty_scores = [], [], []
    if CONFIG['data']['use_synthetic_data']:
        logger.info("Main execution: Synthetic data generation will be handled by DALI's external source.")
        # Dummy lists for train_test_split, as actual data will be generated on-the-fly
        dummy_paths = [f"synthetic_img_{i}.{CONFIG['data']['image_extension']}" for i in range(CONFIG['data']['synthetic_samples'])]
        dummy_labels = np.array([random.randint(0, CONFIG['model']['num_classes'] - 1) for _ in range(CONFIG['data']['synthetic_samples'])])
        dummy_difficulty_scores = np.random.rand(CONFIG['data']['synthetic_samples']).astype(np.float32)
        
        train_image_paths, val_image_paths, train_labels, val_labels, train_difficulty_scores, val_difficulty_scores = \
            train_test_split(dummy_paths, dummy_labels, dummy_difficulty_scores,
                             test_size=CONFIG['data']['train_test_split_ratio'],
                             random_state=CONFIG['model']['random_seed'], stratify=dummy_labels)
        
    else:
        logger.info(f"Main execution: Loading real Bongard data from {DATA_ROOT_PATH}.")
        # load_bongard_data is updated to handle PNG and return difficulty scores
        all_image_paths, all_labels, all_difficulty_scores = load_bongard_data(DATA_ROOT_PATH)
        
        # Ensure sufficient data for splitting
        if len(all_image_paths) < 2:
            logger.error("Not enough real data samples to perform train/validation split. Aborting.")
            return

        train_image_paths, val_image_paths, train_labels, val_labels, train_difficulty_scores, val_difficulty_scores = \
            train_test_split(all_image_paths, all_labels, all_difficulty_scores,
                             test_size=CONFIG['data']['train_test_split_ratio'],
                             random_state=CONFIG['model']['random_seed'],
                             stratify=all_labels)

    logger.info(f"Train samples: {len(train_image_paths)}, Validation samples: {len(val_image_paths)}")

    ENSEMBLE_CHECKPOINT_DIR = os.path.join(CONFIG['training']['checkpoint_dir'], 'ensemble_members')
    META_LEARNER_CHECKPOINT_DIR = os.path.join(CONFIG['training']['checkpoint_dir'], 'meta_learner')
    DISTILLED_STUDENT_CHECKPOINT_DIR = os.path.join(CONFIG['training']['checkpoint_dir'], 'distilled_student')

    os.makedirs(ENSEMBLE_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(META_LEARNER_CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DISTILLED_STUDENT_CHECKPOINT_DIR, exist_ok=True)

    NUM_ENSEMBLE_MEMBERS = CONFIG['ensemble']['num_members']
    trained_base_model_paths = []
    all_members_val_predictions_logits = []
    all_members_val_labels = []
    all_members_best_metrics = []
    
    # --- DDP Setup ---
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs. Setting up DDP training.")
        world_size = torch.cuda.device_count()
        # ddp_train_worker will handle _run_single_training_session_ensemble for each rank
        mp.spawn(ddp_train_worker,
                 args=(world_size, CONFIG, train_image_paths, train_labels, train_difficulty_scores,
                       val_image_paths, val_labels, val_difficulty_scores),
                 nprocs=world_size,
                 join=True)
        
        # After DDP training, collect results from rank 0's output
        # This assumes rank 0 saved the main model and returned its path
        # For ensemble members, each rank would save its own member.
        trained_base_model_paths = [
            os.path.join(ENSEMBLE_CHECKPOINT_DIR, f'member_{i}_{CONFIG["training"]["model_checkpoint_name"]}')
            for i in range(NUM_ENSEMBLE_MEMBERS) # Assuming each rank trains one member
        ]
        # For actual val predictions/labels from DDP, you'd need to gather them from all ranks
        # For simplicity, we'll use dummy ones here if not gathered.
        num_val_samples = len(val_image_paths)
        num_classes = CONFIG['model']['num_classes']
        all_members_val_predictions_logits = [np.random.rand(num_val_samples, num_classes)] * NUM_ENSEMBLE_MEMBERS
        all_members_val_labels = [val_labels] * NUM_ENSEMBLE_MEMBERS
        all_members_best_metrics = [{'val_accuracy': 0.5, 'val_loss': 0.7}] * NUM_ENSEMBLE_MEMBERS

        logger.info("DDP training completed. Proceeding with post-training steps.")

    else:
        logger.info("Single GPU or CPU detected. Running single-device training for ensemble members.")
        # 3. Train Ensemble Members (Single Device)
        if CONFIG['ensemble']['train_members']:
            for i in range(NUM_ENSEMBLE_MEMBERS):
                member_seed = CONFIG['model']['random_seed'] + i
                member_path, member_val_logits, member_val_labels, member_metrics = \
                    _run_single_training_session_ensemble(
                        current_config=CONFIG,
                        member_id=i,
                        random_seed=member_seed,
                        output_dir=ENSEMBLE_CHECKPOINT_DIR,
                        epochs_override=CONFIG['model']['epochs'],
                        train_image_paths=train_image_paths,
                        train_labels=train_labels,
                        train_difficulty_scores=train_difficulty_scores,
                        val_image_paths=val_image_paths,
                        val_labels=val_labels,
                        val_difficulty_scores=val_difficulty_scores,
                        teacher_model=None # No teacher for base training
                    )
                if member_path:
                    trained_base_model_paths.append(member_path)
                    all_members_val_predictions_logits.append(member_val_logits)
                    all_members_val_labels.append(member_val_labels)
                    all_members_best_metrics.append(member_metrics)
            logger.info(f"Trained {len(trained_base_model_paths)} ensemble members.")
        else:
            logger.warning("Skipping ensemble member training as 'ensemble.train_members' is False.")
            existing_models = glob.glob(os.path.join(ENSEMBLE_CHECKPOINT_DIR, f'member_*.{CONFIG["data"]["image_extension"]}'))
            if CONFIG['training']['use_swa']:
                existing_models.extend(glob.glob(os.path.join(ENSEMBLE_CHECKPOINT_DIR, 'member_*_best_perception_model_swa.pt')))
            if CONFIG['training']['use_qat']:
                existing_models.extend(glob.glob(os.path.join(ENSEMBLE_CHECKPOINT_DIR, 'member_*_best_perception_model_quantized.pt')))
            
            if existing_models:
                trained_base_model_paths = sorted(existing_models)
                logger.info(f"Loaded {len(trained_base_model_paths)} existing ensemble member checkpoints.")
                num_val_samples = len(val_image_paths)
                num_classes = CONFIG['model']['num_classes']
                all_members_val_predictions_logits = [np.random.rand(num_val_samples, num_classes)] * len(trained_base_model_paths)
                all_members_val_labels = [val_labels] * len(trained_base_model_paths)
                all_members_best_metrics = [{'val_accuracy': 0.5, 'val_loss': 0.7}] * len(trained_base_model_paths)
            else:
                logger.error("No existing ensemble member checkpoints found. Cannot proceed without trained models.")
                return # Exit if no models to work with

    # 4. Train Meta-Learner (if stacking is enabled)
    meta_learner_path = None
    if CONFIG['ensemble']['use_stacking']:
        logger.info("Training meta-learner for stacking ensemble.")
        if all_members_val_predictions_logits:
            single_val_labels = all_members_val_labels[0] if all_members_val_labels else val_labels
            
            meta_learner_path = train_meta_learner_orchestrator_combined(
                base_model_val_predictions_logits=all_members_val_predictions_logits,
                val_labels=single_val_labels,
                meta_learner_output_dir=META_LEARNER_CHECKPOINT_DIR,
                epochs=CONFIG['model']['epochs'], # Use full epochs for meta-learner training
                batch_size=CONFIG['model']['batch_size'],
                learning_rate=CONFIG['model']['initial_learning_rate'],
                random_seed=CONFIG['model']['random_seed']
            )
            logger.info(f"Meta-learner trained and saved to: {meta_learner_path}")
        else:
            logger.warning("Cannot train meta-learner: No base model validation predictions available.")
    else:
        logger.info("Skipping meta-learner training as 'ensemble.use_stacking' is False.")

    # 5. Train Distilled Student Model (if distillation is enabled)
    distilled_student_model_path = None
    if CONFIG['ensemble']['use_distillation']:
        logger.info("Training distilled student model.")
        if trained_base_model_paths: # Need teacher models to distill from
            distilled_student_model_path = train_distilled_student_orchestrator_combined(
                teacher_model_paths=trained_base_model_paths,
                train_image_paths=train_image_paths,
                train_labels=train_labels,
                val_image_paths=val_image_paths,
                val_labels=val_labels,
                val_difficulty_scores=val_difficulty_scores,
                student_output_dir=DISTILLED_STUDENT_CHECKPOINT_DIR,
                epochs_override=CONFIG['model']['epochs'],
                random_seed=CONFIG['model']['random_seed'] + 1, # Use a different seed for student
                student_config_override=CONFIG['ensemble']['distilled_student_config_override'],
                teacher_ensemble_type=CONFIG['ensemble']['teacher_ensemble_type'], # Pass teacher type
                all_members_val_predictions_logits=all_members_val_predictions_logits, # Pass for stacked teacher
                all_members_val_labels=all_members_val_labels # Pass for stacked teacher
            )
            logger.info(f"Distilled student model trained and saved to: {distilled_student_model_path}")
        else:
            logger.warning("Cannot train distilled student: No base models available to act as teachers.")
    else:
        logger.info("Skipping distilled student model training as 'ensemble.use_distillation' is False.")

    # 6. Perform Ensemble Inference
    logger.info(f"Performing ensemble inference using mode: '{CONFIG['ensemble']['inference_mode']}'")
    INFERENCE_IMAGE_PATHS = val_image_paths[:min(10, len(val_image_paths))]
    if not INFERENCE_IMAGE_PATHS:
        logger.warning("No images available for inference demonstration.")
        averaged_probs, symbolic_outputs_example = np.array([]), []
    else:
        member_weights = None
        if CONFIG['ensemble']['inference_mode'] == 'weighted_average':
            member_weights = calculate_model_weights(all_members_best_metrics, metric_name='val_accuracy', minimize_metric=False)
            if not member_weights:
                logger.warning("Could not calculate weights. Falling back to simple average for inference.")
                CONFIG['ensemble']['inference_mode'] = 'simple_average'
            else:
                logger.info(f"Calculated member weights for inference: {member_weights}")

        try:
            averaged_probs, symbolic_outputs_example = perform_ensemble_inference_combined(
                inference_mode=CONFIG['ensemble']['inference_mode'],
                image_paths=INFERENCE_IMAGE_PATHS,
                config=CONFIG,
                base_model_paths=trained_base_model_paths,
                model_weights=member_weights,
                meta_learner_path=meta_learner_path,
                student_model_path=distilled_student_model_path,
                use_mc_dropout=CONFIG['model']['mc_dropout_samples'] > 0,
                mc_dropout_samples=CONFIG['model']['mc_dropout_samples']
            )
            
            logger.info("\n--- Ensemble Prediction Results ---")
            for i, img_path in enumerate(INFERENCE_IMAGE_PATHS):
                if i < len(averaged_probs):
                    logger.info(f"Image: {img_path}")
                    logger.info(f"  Ensemble Averaged Probabilities: {averaged_probs[i]}")
                    logger.info(f"  Ensemble Predicted Class: {np.argmax(averaged_probs[i])}")
            
            if symbolic_outputs_example:
                output_json_path = os.path.join(ENSEMBLE_CHECKPOINT_DIR, "ensemble_inference_symbolic_output_example.json")
                with open(output_json_path, 'w') as f:
                    json.dump(symbolic_outputs_example, f, indent=4)
                logger.info(f"Example symbolic output from one ensemble member saved to: {output_json_path}")

        except Exception as e:
            logger.error(f"Error during ensemble inference: {e}")
            averaged_probs, symbolic_outputs_example = np.array([]), []

    # 7. Deployment Optimizations (Pruning, Quantization, ONNX Export)
    logger.info("--- Starting Deployment Optimizations ---")
    final_model_for_deployment = None

    if CONFIG['ensemble']['inference_mode'] == 'distilled' and distilled_student_model_path:
        object_detector_deployment = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
        final_model_for_deployment = PerceptionModule(CONFIG, object_detector_deployment).to(DEVICE)
        final_model_for_deployment.load_weights(distilled_student_model_path)
        logger.info("Loaded distilled student model for deployment optimization.")
    elif trained_base_model_paths:
        object_detector_deployment = RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])
        final_model_for_deployment = PerceptionModule(CONFIG, object_detector_deployment).to(DEVICE)
        final_model_for_deployment.load_weights(trained_base_model_paths[0])
        logger.info("Loaded first base model as representative for deployment optimization.")

    if final_model_for_deployment:
        if not CONFIG['training']['use_qat'] and CONFIG['training']['calibrate_model']:
            logger.info("Applying Post-Training Quantization (PTQ).")
            # For PTQ calibration, use a DALI loader for validation data
            calibration_loader, _ = build_dali_loader(
                file_list=val_image_paths,
                labels_list=val_labels,
                difficulty_scores_list=val_difficulty_scores,
                config=CONFIG,
                mode='val'
            )
            final_model_for_deployment = quantize_model_ptq(final_model_for_deployment, calibration_loader, CONFIG)
            logger.info("PTQ applied.")
        elif CONFIG['training']['use_qat']:
            logger.info("Model was trained with QAT. Skipping separate PTQ step.")
        else:
            logger.info("Skipping quantization as neither PTQ nor QAT is enabled/configured.")
        
        # Apply Structured Pruning
        if CONFIG['training']['enable_structured_pruning']:
            # Need to create train_loader and val_loader for fine_tune_model
            # Re-create loaders as needed for pruning fine-tuning
            train_loader_pruning, _ = build_dali_loader(
                file_list=train_image_paths,
                labels_list=train_labels,
                difficulty_scores_list=train_difficulty_scores,
                config=CONFIG,
                mode='train'
            )
            val_loader_pruning, _ = build_dali_loader(
                file_list=val_image_paths,
                labels_list=val_labels,
                difficulty_scores_list=val_difficulty_scores,
                config=CONFIG,
                mode='val'
            )
            final_model_for_deployment = apply_structured_pruning(
                final_model_for_deployment, CONFIG, train_loader_pruning, val_loader_pruning
            )
            logger.info("Structured pruning applied.")
        else:
            logger.info("Structured pruning is disabled in configuration.")

        # Export ONNX
        if hasattr(final_model_for_deployment, 'export_onnx'):
            onnx_output_path = os.path.join(CONFIG['training']['checkpoint_dir'], "final_model_exported.onnx")
            final_model_for_deployment.export_onnx(output_path=onnx_output_path)
            logger.info(f"Final model exported to ONNX: {onnx_output_path}")
        else:
            logger.warning("Final model does not have an 'export_onnx' method. Skipping ONNX export.")
    else:
        logger.warning("No final model available for deployment optimizations (e.g., no members trained or paths invalid).")

    logger.info("Bongard Ensemble Workflow execution finished.")

    # 8. XAI (Grad-CAM) Visualization
    if (CONFIG['debug']['enable_grad_cam'] or CONFIG['debug']['enable_grad_cam_on_errors']) and HAS_PYTORCH_GRADCAM:
        logger.info("Generating Grad-CAM visualizations.")
        
        model_for_cam = None
        if CONFIG['ensemble']['inference_mode'] == 'distilled' and distilled_student_model_path:
            model_for_cam = PerceptionModule(CONFIG, RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])).to(DEVICE)
            model_for_cam.load_weights(distilled_student_model_path)
        elif trained_base_model_paths:
            model_for_cam = PerceptionModule(CONFIG, RealObjectDetector(model_path=CONFIG['model']['object_detector_model_path'])).to(DEVICE)
            model_for_cam.load_weights(trained_base_model_paths[0])

        if model_for_cam:
            model_for_cam.eval()
            
            # General Grad-CAM for example images
            if CONFIG['debug']['enable_grad_cam']:
                logger.info("Generating Grad-CAM for general example images.")
                sample_images_for_cam = INFERENCE_IMAGE_PATHS[:min(3, len(INFERENCE_IMAGE_PATHS))]
                os.makedirs('./grad_cam_outputs', exist_ok=True)
                for img_path in sample_images_for_cam:
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                        input_tensor = T.Compose([
                            T.Resize(CONFIG['data']['image_size']),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ])(img_pil).unsqueeze(0).to(DEVICE)
                        
                        dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})]
                        bongard_logits, _, _ = model_for_cam(input_tensor, dummy_gts_json)
                        predicted_class = torch.argmax(bongard_logits).item()
                        
                        target_layer = None
                        if hasattr(model_for_cam.attribute_classifier.backbone, 'features'):
                            target_layer = model_for_cam.attribute_classifier.backbone.features[-1]
                        elif hasattr(model_for_cam.attribute_classifier.backbone, 'blocks'):
                            target_layer = model_for_cam.attribute_classifier.backbone.blocks[-1]
                        elif hasattr(model_for_cam.attribute_classifier.backbone, 'conv_head'):
                            target_layer = model_for_cam.attribute_classifier.backbone.conv_head
                        elif hasattr(model_for_cam.attribute_classifier.backbone, 'head'):
                            if isinstance(model_for_cam.attribute_classifier.backbone.head, nn.Sequential):
                                for m in reversed(model_for_cam.attribute_classifier.backbone.head):
                                    if isinstance(m, nn.Conv2d):
                                        target_layer = m
                                        break
                            elif isinstance(model_for_cam.attribute_classifier.backbone.head, nn.Conv2d):
                                target_layer = model_for_cam.attribute_classifier.backbone.head

                        if target_layer:
                            generate_grad_cam(
                                model=model_for_cam,
                                input_tensor=input_tensor,
                                target_layer=target_layer,
                                target_category=predicted_class,
                                image_path=img_path,
                                save_dir='./grad_cam_outputs'
                            )
                        else:
                            logger.warning(f"Could not find suitable target layer for general Grad-CAM for {img_path}.")
                    except Exception as e:
                        logger.error(f"Failed to generate general Grad-CAM for {img_path}: {e}")

            # Grad-CAM for misclassified samples
            if CONFIG['debug']['enable_grad_cam_on_errors']:
                logger.info("Generating Grad-CAM for misclassified images.")
                # Run validation to get misclassified samples
                _, _, _, _, misclassified_samples = _validate_model_ensemble(
                    model_for_cam, val_loader,
                    LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing_epsilon']), # Use correct criterion
                    CONFIG
                )
                
                os.makedirs('./grad_cam_error_outputs', exist_ok=True)
                for sample_info in misclassified_samples[:min(5, len(misclassified_samples))]:
                    try:
                        img_path = sample_info['path']
                        input_tensor = sample_info['input_tensor'].unsqueeze(0).to(DEVICE)
                        predicted_class = sample_info['predicted_label']
                        true_class = sample_info['true_label']

                        target_layer = None # Find target layer as above
                        if hasattr(model_for_cam.attribute_classifier.backbone, 'features'):
                            target_layer = model_for_cam.attribute_classifier.backbone.features[-1]
                        elif hasattr(model_for_cam.attribute_classifier.backbone, 'blocks'):
                            target_layer = model_for_cam.attribute_classifier.backbone.blocks[-1]
                        elif hasattr(model_for_cam.attribute_classifier.backbone, 'conv_head'):
                            target_layer = model_for_cam.attribute_classifier.backbone.conv_head
                        elif hasattr(model_for_cam.attribute_classifier.backbone, 'head'):
                            if isinstance(model_for_cam.attribute_classifier.backbone.head, nn.Sequential):
                                for m in reversed(model_for_cam.attribute_classifier.backbone.head):
                                    if isinstance(m, nn.Conv2d):
                                        target_layer = m
                                        break
                            elif isinstance(model_for_cam.attribute_classifier.backbone.head, nn.Conv2d):
                                target_layer = model_for_cam.attribute_classifier.backbone.head

                        if target_layer:
                            generate_grad_cam(
                                model=model_for_cam,
                                input_tensor=input_tensor,
                                target_layer=target_layer,
                                target_category=predicted_class,
                                image_path=img_path,
                                save_dir='./grad_cam_error_outputs',
                                file_suffix=f"_pred_{predicted_class}_true_{true_class}"
                            )
                        else:
                            logger.warning(f"Could not find suitable target layer for error Grad-CAM for {img_path}.")
                    except Exception as e:
                        logger.error(f"Failed to generate error Grad-CAM for {img_path}: {e}")
        else:
            logger.warning("No model available for Grad-CAM visualization.")
    else:
        logger.info("Grad-CAM generation skipped (disabled in config or pytorch_grad_cam not installed).")

    if HAS_WANDB and CONFIG['training']['use_wandb']:
        import wandb
        wandb.finish()

    logger.info("--- Bongard Ensemble Workflow execution finished. ---")


# This block allows you to run the main workflow directly if you save this as a .py file
# or call main_project_workflow() from a Jupyter Notebook cell.
if __name__ == "__main__":
    main_project_workflow()
