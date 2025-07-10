import os
import glob
import numpy as np
from ultralytics import YOLO as RealYOLO # Import RealYOLO from ultralytics
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss, roc_auc_score
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
import time # Added for batch timing
import json # Added for JSON export of symbolic outputs

# PyTorch specific imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
# UPDATED: Import specific MobileNetV3 and EfficientNet models and their weights
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, efficientnet_b0
import torchvision.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
# UPDATED: Use torch.cuda.amp for autocast and GradScaler (corrected from torch.amp)
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
    from sam import SAM  # Assuming sam.py is in the same directory or on PYTHONPATH
    HAS_SAM = True
    logger.info("Successfully imported SAM from local sam.py.")
except ImportError:
    logger.warning("Could not import local sam.py. SAM optimizer will not be available.")
    # Define a dummy SAM class to prevent NameError if SAM is not available
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            logger.warning("Using dummy SAM optimizer. Please install sam-pytorch or ensure sam.py is accessible for full SAM functionality.")
            self.base_optimizer = base_optimizer
            super().__init__(params, base_optimizer.defaults)
        
        # SAM's first_step and second_step API
        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for p in self.param_groups:
                for n,d in p['params']:
                    if d.grad is None: continue
                    self.state[d]['old_p'] = d.data.clone() # Store original parameter
                    eps = self.rho / (grad_norm + 1e-12)
                    d.add_(d.grad, alpha=eps) # Update for first step

            if zero_grad: self.zero_grad()
        
        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for p in self.param_groups:
                for n,d in p['params']:
                    if d.grad is None: continue
                    d.data = self.state[d]['old_p'] # Restore original parameter
            self.base_optimizer.step() # Apply base optimizer step

            if zero_grad: self.zero_grad()

        def _grad_norm(self):
            # Calculate the gradient norm for SAM
            norm = torch.norm(torch.stack([
                (p.grad if p.grad is not None else torch.zeros_like(p)).norm(p=2)
                for p in functools.reduce(lambda a,b: a + b, [group['params'] for group in self.param_groups])
            ]), p=2)
            return norm


HAS_WANDB = False
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    pass

HAS_KORNIA = False
try:
    import kornia.augmentation as K
    HAS_KORNIA = True
except ImportError:
    pass

# PyTorch Geometric for GNNs
HAS_PYG = False
try:
    from torch_geometric.nn import GCNConv, Set2Set # Assuming torch_geometric library
    HAS_PYG = True
    logger.info("PyTorch Geometric (torch_geometric) imported successfully.")
except ImportError:
    logger.warning("PyTorch Geometric (PyG) not installed. GNN functionality will be limited to dummy implementations.")
    # Define dummy classes to prevent NameError if PyG is not available
    class GCNConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)
            logger.warning("Using dummy GCNConv. Please install torch_geometric for full GNN functionality.")
        def forward(self, x, edge_index):
            return self.linear(x) # Simple linear transformation as a dummy
    
    class Set2Set(nn.Module):
        def __init__(self, in_channels, processing_steps, num_layers):
            super().__init__()
            self.linear = nn.Linear(in_channels, in_channels * 2) # Dummy output for pooling (2 * in_channels as per Set2Set output)
            logger.warning("Using dummy Set2Set. Please install torch_geometric for full GNN functionality.")
        def forward(self, x, batch):
            if x.numel() == 0:
                # If there are no nodes, return a tensor of appropriate shape.
                num_expected_batches = batch.max().item() + 1 if batch.numel() > 0 else 0
                return torch.zeros(num_expected_batches, self.linear.out_features, device=x.device)

            unique_batches = torch.unique(batch)
            pooled_features = []
            for b in unique_batches:
                mask = (batch == b)
                if x[mask].numel() > 0: # Check if there are nodes for this batch
                    pooled_features.append(torch.mean(x[mask], dim=0))
                else:
                    # If no nodes for this batch, append a zero vector of the correct feature dimension
                    pooled_features.append(torch.zeros(x.size(1), device=x.device))
            
            if not pooled_features and batch.numel() == 0:
                return torch.empty(0, self.linear.out_features, device=x.device)
            elif not pooled_features:
                return torch.zeros(batch.max().item() + 1, self.linear.out_features, device=x.device)

            return self.linear(torch.stack(pooled_features))

# Dummy for timm and sophia-pytorch if not installed
HAS_TIMM = False
try:
    # from timm.optim import Lion # Assuming timm is installed for Lion
    # HAS_TIMM = True
    pass # Keeping this commented out as timm is not explicitly requested for installation
except ImportError:
    pass

# SophiaG optimizer
HAS_SOPHIA = False
try:
    from sophia import SophiaG  # Assuming sophia.py is in the same directory or on PYTHONPATH
    HAS_SOPHIA = True
    logger.info("Successfully imported SophiaG from local sophia.py.")
except ImportError:
    logger.warning("Could not import SophiaG from sophia.py. Sophia optimizer will not be available.")
    # Dummy SophiaG class if not installed
    class SophiaG(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
            logger.warning("Using dummy SophiaG optimizer. Please ensure sophia.py is accessible.")
            defaults = dict(lr=lr, weight_decay=weight_decay)
            super().__init__(params, defaults)

        def step(self, closure=None):
            # Dummy step: simply calls the closure if provided
            if closure is not None:
                closure()
            logger.debug("Dummy SophiaG step called.") # Indicate dummy behavior


# --- Global Configuration and Constants ---
# DEFAULT_CONFIG is the base, which will be loaded first.
DEFAULT_CONFIG = {
    'data': {
        'dataset_path': './ShapeBongard_V2', # Unified source of truth
        'image_size': [224, 224], # Default image size
        'num_channels': 1, # 1 for grayscale, 3 for RGB
        'train_test_split_ratio': 0.2,
        'dataloader_num_workers': 1, # Default worker count. Consider increasing for performance (e.g., 4 or 8)
        'dataloader_pin_memory': True,
        'dataloader_persistent_workers': False,
        'dataloader_prefetch_factor': 2,
        'class_imbalance_threshold': 0.2,
        'progressive_resizing_epochs': 5, # Epochs for small image size
        'initial_image_size': [112, 112], # Smaller initial size for DALI decoding
        'curriculum_learning_enabled': True,
        'curriculum_difficulty_metric': 'std_dev',
        'curriculum_annealing_epochs': 5,
        'cache_data': False,
        'curriculum_start_difficulty': 0.0,
        'curriculum_end_difficulty': 1.0,
        'curriculum_difficulty_schedule': 'linear',
        'curriculum_update_interval_epochs': 1
    },
    'model': {
        'num_classes': 2,
        'initial_learning_rate': 0.001,
        'max_learning_rate': 0.01, # ADDED: Required for OneCycleLR scheduler
        'epochs': 15,
        'batch_size': 32,
        'mc_dropout_samples': 25,
        'unfreeze_schedule': {
            'backbone_unfreeze_epoch': -1 # Default: don't unfreeze backbone
        },
        'unfreeze_lr_multiplier': 0.1,
        'initial_dropout_rate': 0.5,
        'final_dropout_rate': 0.1,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'teacher_model_path': None,
        'object_detector_model_path': 'yolov8n.pt',
        'attribute_classifier_model_path': None,
        'relation_gnn_model_path': None,
        'attribute_backbone': 'mobilenet_v2', # NEW: Option for MobileNetV2, MobileNetV3, EfficientNet
        'gnn_depth': 2, # NEW: Depth of the GNN (number of GCNConv layers)
        'detection_confidence_threshold': 0.3 # NEW: Confidence threshold for object detection
    },
    'training': {
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 0.0001,
        'checkpoint_dir': './checkpoints',
        'model_checkpoint_name': 'best_perception_model.pt',
        'lr_scheduler_factor': 0.1,
        'lr_scheduler_patience': 3,
        'use_amp': True,
        'max_grad_norm': 1.0,
        'gradient_accumulation_steps': 1, # Default to 1
        'calibrate_model': True,
        'calibration_epochs': 50,
        'mixup_alpha': 0.0, # Default to 0 (disabled)
        'cutmix_alpha': 0.0, # Default to 0 (disabled)
        'mixup_cutmix_ratio': 0.5,
        'label_smoothing_epsilon': 0.0, # Default to 0 (disabled)
        'weight_decay': 1e-4,
        'use_sam_optimizer': True,
        'swa_start_epoch_ratio': 0.75,
        'swa_lr': 0.05,
        'self_distillation_alpha': 0.5,
        'feature_consistency_alpha': 0.1,
        'knowledge_replay_enabled': True,
        'replay_buffer_size': 100,
        'replay_frequency_epochs': 2,
        'replay_batch_size_ratio': 0.1,
        'use_wandb': True,
        'early_stopping_monitor_metric': 'val_loss', # Can be 'val_loss', 'val_accuracy', etc.
        'use_torch_compile': True,
        'enable_profiler': False,
        'profiler_schedule_wait': 1,
        'profiler_schedule_warmup': 1,
        'profiler_schedule_active': 3,
        'profiler_schedule_repeat': 1,
        'validation_frequency_batches': 0,
        'distillation_temperature': 2.0,
        'distillation_alpha': 0.5,
        'use_knowledge_distillation': False,
        'profile_performance': False,
        'use_qat': False,
        'qat_start_epoch': 5,
        'use_gradient_checkpointing': False,
        'save_symbolic_outputs_interval_epochs': 5, # NEW: Interval to save symbolic outputs
        'onecycle_pct_start': 0.3 # ADDED: Percentage of steps to increase learning rate in OneCycleLR
    },
    'debug': {
        'visualize_data': True,
        'visualize_training_history': True,
        'visualize_perception_output': True,
        'plot_reliability_diagram': True,
        'log_level': 'INFO',
        'visualize_gnn_weights': False # NEW: Toggle for GNN edge weight visualization (conceptual)
    }
}

# load_config function
def load_config(config_path=None):
    """Loads configuration from a YAML file, merging with default config."""
    config = DEFAULT_CONFIG.copy()
    if config_path:
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            logger.info(f"Configuration loaded from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using default configuration.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}. Using default configuration.")
    return config


# Argument parsing and initial config loading
parser = argparse.ArgumentParser(description="Bongard Problem Phase 1: Advanced Perception Module")
parser.add_argument('--config', type=str, default=None, help='Path to a YAML configuration file.')
args, unknown = parser.parse_known_args()
CONFIG = load_config(args.config)

# --- CRITICAL: Force Memory-Optimized CONFIG Values for 4GB GPU ---
# These values will override anything set in a config file or elsewhere,
# ensuring the DALI pipeline is initialized with memory-safe parameters.
# Moved after load_config to ensure they override any YAML settings.
# After MEMORY-SAFE OVERRIDES
CONFIG['model']['batch_size'] = 4            # Reduced batch size
CONFIG['data']['image_size'] = [96, 96]      # Reduced image size
# NEW: Smaller initial_image_size for progressive resizing
CONFIG['data']['initial_image_size'] = [64, 64] # Smaller initial size for DALI decoding
CONFIG['training']['feature_consistency_alpha'] = 0.0 # Disable second view
CONFIG['training']['use_amp'] = True       # Ensure AMP is enabled
# OPTIMIZATION: Increased worker count for DALI parallelism
CONFIG['data']['dataloader_num_workers'] = 6 # Increased worker count
# Added missing 'use_swa' key to CONFIG['training']
CONFIG['training']['use_swa'] = True # Default to True as per report
CONFIG['training']['swa_start_epoch_ratio'] = 0.75 # SWA start epoch ratio
CONFIG['training']['swa_lr'] = 0.05 # Default SWA learning rate
# OPTIMIZATION: Gradient Accumulation
CONFIG['training']['gradient_accumulation_steps'] = 2 # Set to 2 as per report
# OPTIMIZATION: MixUp & CutMix (enable with non-zero alpha)
CONFIG['training']['mixup_alpha'] = 0.2 # Enable MixUp
CONFIG['training']['cutmix_alpha'] = 1.0 # Enable CutMix
# OPTIMIZATION: Label Smoothing
CONFIG['training']['label_smoothing_epsilon'] = 0.1 # Enable Label Smoothing

# UPDATED: Lower detection threshold and enable TTA for YOLO
CONFIG['model']['detection_confidence_threshold'] = 0.1   # ← lower threshold
CONFIG['model']['yolo_augment'] = True                  # ← new flag for TTA

logger.info(f"Effective CONFIG for DALI Pipeline: Batch Size={CONFIG['model']['batch_size']}, Image Size={CONFIG['data']['image_size']}, Feature Consistency Alpha={CONFIG['training']['feature_consistency_alpha']}")
logger.info(f"DALI Dataloader Workers: {CONFIG['data']['dataloader_num_workers']}")
logger.info(f"Gradient Accumulation Steps: {CONFIG['training']['gradient_accumulation_steps']}")
logger.info(f"MixUp Alpha: {CONFIG['training']['mixup_alpha']}, CutMix Alpha: {CONFIG['training']['cutmix_alpha']}")
logger.info(f"Label Smoothing Epsilon: {CONFIG['training']['label_smoothing_epsilon']}")

# --- END CRITICAL CONFIG OVERRIDE ---

# Set global logging level based on config
logging.getLogger().setLevel(getattr(logging, CONFIG['debug']['log_level'].upper()))

# Unified DATA_ROOT_PATH and other globals from CONFIG
DATA_ROOT_PATH = CONFIG['data']['dataset_path'] # Unified source of truth
FINAL_IMAGE_SIZE = tuple(CONFIG['data']['image_size'])
NUM_CHANNELS = CONFIG['data']['num_channels']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ImageNet Mean and Std for normalization (DALI uses these)
# DALI expects mean/std in pixel value range (0-255) when used with fn.crop_mirror_normalize
IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255] # For 0-255 range
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255] # For 0-255 range

# --- Data Download & Unzip Execution ---
# These are global constants used for data download, not part of CONFIG
GOOGLE_DRIVE_FILE_ID = '1-1j7EBriRpxI-xIVqE6UEXt-SzoWvwLx'
OUTPUT_ZIP_FILENAME = 'bongard_dataset.zip'
DATA_ROOT_FOLDER_NAME = 'ShapeBongard_V2'
FULL_DATA_ROOT_PATH = os.path.join('.', DATA_ROOT_FOLDER_NAME)
DIFFICULTY_SCORES_FILE = os.path.join(FULL_DATA_ROOT_PATH, 'difficulty_scores.npy')
FULL_DATA_CACHE_FILE = os.path.join(FULL_DATA_ROOT_PATH, 'bongard_data_full_cache.npz')

print(f"Checking for existing dataset folder at: {DATA_ROOT_PATH}")
if not os.path.exists(DATA_ROOT_PATH):
    print(f"Dataset folder '{DATA_ROOT_FOLDER_NAME}' not found. Downloading and unzipping...")
    try:
        # This block is for gdown, but since it's not available, it will fall back to requests.
        # Keeping it as is from original code.
        raise ImportError("gdown not available in this environment. Falling back to requests.")
    except ImportError:
        print("gdown failed or not available. Attempting download with requests...")
        try:
            download_url = f'https://docs.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}'
            session = requests.Session()
            response = session.get(download_url, stream=True)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                params = {'id': GOOGLE_DRIVE_FILE_ID, 'confirm': token}
                response = session.get(download_url, params=params, stream=True)
            response.raise_for_status()
            with open(OUTPUT_ZIP_FILENAME, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {OUTPUT_ZIP_FILENAME} successfully with requests.")
        except Exception as e_requests:
            print(f"Requests download also failed ({e_requests}).")
            print("Please download the dataset manually from: "
                  f"https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view?usp=sharing "
                  f"and place '{OUTPUT_ZIP_FILENAME}' in the same directory as your notebook.")
            print("Then, manually unzip it to create the folder structure starting with "
                  f"'{DATA_ROOT_FOLDER_NAME}'.")
            raise RuntimeError("Failed to download dataset automatically.")
    print(f"Unzipping {OUTPUT_ZIP_FILENAME}...")
    try:
        with zipfile.ZipFile(OUTPUT_ZIP_FILENAME, 'r') as zip_ref:
            zip_ref.extractall('./')
        print(f"Dataset unzipped. Please confirm '{DATA_ROOT_FOLDER_NAME}' folder exists in your current directory.")
    except Exception as e_unzip:
        print(f"Error unzipping {OUTPUT_ZIP_FILENAME}: {e_unzip}")
        print("Please ensure the zip file is not corrupted and try unzipping manually.")
        raise RuntimeError("Failed to unzip dataset.")
else:
    print(f"Dataset folder '{DATA_ROOT_FOLDER_NAME}' already exists at {DATA_ROOT_PATH}. Skipping download and unzip.")

# --- 1.1 Data Loading (Memory Efficient & Corrected Labeling) ---
def load_bongard_data(data_root_path):
    """
    Loads Bongard problem image *paths* and labels from the specified directory structure,
    correctly assigning labels based on the '0' or '1' subfolder.
    Images themselves are NOT loaded into memory here.
    Also calculates a heuristic difficulty score (image std dev) for curriculum learning.
    This function now checks for and saves pre-calculated difficulty scores.
    This version also caches the full image paths, labels, and difficulty scores
    to speed up subsequent runs.
    """
    # Try to load from the full data cache first
    if os.path.exists(FULL_DATA_CACHE_FILE):
        try:
            logger.info(f"Attempting to load full data cache from {FULL_DATA_CACHE_FILE}")
            cached_data = np.load(FULL_DATA_CACHE_FILE, allow_pickle=True)
            cached_image_paths = cached_data['image_paths'].tolist() # Convert back to list of strings
            cached_labels = cached_data['labels']
            cached_difficulty_scores = cached_data['difficulty_scores']
            logger.info(f"Successfully loaded full data cache with {len(cached_image_paths)} entries.")
            return cached_image_paths, cached_labels, cached_difficulty_scores
        except Exception as e:
            logger.warning(f"Error loading full data cache from {FULL_DATA_CACHE_FILE}: {e}. Recalculating all data.")

    logger.info("Full data cache not found or invalid. Performing full data scan and difficulty calculation...")
    all_image_paths = []
    all_labels = []

    # Define base directories for collecting paths
    set_a_base_dirs = [os.path.join(data_root_path, 'bd', 'images')]
    set_b_base_dirs = [
        os.path.join(data_root_path, 'ff', 'images'),
        os.path.join(data_root_path, 'hd', 'images')
    ]

    # Collect all image paths and labels
    temp_image_paths = []
    temp_labels = []

    def collect_paths_and_labels(base_dirs):
        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                logger.warning(f"Base image directory not found: {base_dir}. Skipping.")
                continue
            for problem_folder_path in glob.glob(os.path.join(base_dir, '*')):
                if os.path.isdir(problem_folder_path):
                    for sub_label_folder_name in ['0', '1']:
                        sub_label_path = os.path.join(problem_folder_path, sub_label_folder_name)
                        if not os.path.exists(sub_label_path):
                            continue
                        current_label = int(sub_label_folder_name)
                        for img_path in glob.glob(os.path.join(sub_label_path, '*.png')):
                            temp_image_paths.append(img_path)
                            temp_labels.append(current_label)

    collect_paths_and_labels(set_a_base_dirs)
    collect_paths_and_labels(set_b_base_dirs)

    all_image_paths = temp_image_paths
    all_labels = temp_labels

    total_images = len(all_image_paths)
    if total_images == 0:
        logger.error("No image paths collected. Please check your DATA_ROOT_PATH and directory structure.")
        raise FileNotFoundError("No images found in the specified data root path.")

    labels_array = np.array(all_labels)
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    label_distribution = dict(zip(unique_labels, counts))
    logger.info(f"Collected {total_images} image paths in total.")
    logger.info(f"Label distribution: {label_distribution}")

    if len(unique_labels) < CONFIG['model']['num_classes']:
        logger.error(f"Expected {CONFIG['model']['num_classes']} classes but found {len(unique_labels)}. Check data integrity.")
        raise ValueError("Insufficient number of classes found in dataset.")

    expected_count_per_class = total_images / CONFIG['model']['num_classes']
    for label, count in label_distribution.items():
        if abs(count - expected_count_per_class) / expected_count_per_class > CONFIG['data']['class_imbalance_threshold']:
            logger.warning(f"Significant class imbalance detected for label {label}: {count} samples (expected approx. {expected_count_per_class:.0f}).")
            logger.warning("Consider using WeightedRandomSampler or custom loss weighting for balanced batches.")

    # Now, calculate or load difficulty scores (old logic for difficulty_scores.npy)
    all_difficulty_scores = []
    difficulty_scores_loaded = False # Flag to track if scores were successfully loaded

    if os.path.exists(DIFFICULTY_SCORES_FILE):
        try:
            loaded_difficulty_scores = np.load(DIFFICULTY_SCORES_FILE)
            if len(loaded_difficulty_scores) == total_images: # Check against total_images
                logger.info(f"Loading pre-calculated difficulty scores from {DIFFICULTY_SCORES_FILE}")
                all_difficulty_scores = loaded_difficulty_scores
                difficulty_scores_loaded = True # Set flag to True
            else:
                logger.warning(f"Mismatch in number of images ({total_images}) and loaded difficulty scores ({len(loaded_difficulty_scores)}). Recalculating scores.")
        except Exception as e:
            logger.warning(f"Error loading difficulty scores from {DIFFICULTY_SCORES_FILE}: {e}. Recalculating scores.")

    if not difficulty_scores_loaded: # Use the boolean flag here
        logger.info("Calculating difficulty scores (this may take a while)...")
        for img_path in tqdm(all_image_paths, desc="Calculating image difficulty scores"):
            try:
                img_pil = Image.open(img_path).convert('L')
                img_np = np.array(img_pil)
                difficulty = img_np.std()
                all_difficulty_scores.append(difficulty)
            except Exception as e:
                logger.warning(f"Could not calculate difficulty for {img_path}: {e}. Assigning default difficulty.")
                all_difficulty_scores.append(100.0) # Assign a high difficulty for errors
        
        difficulty_scores_np = np.array(all_difficulty_scores)
        if difficulty_scores_np.max() > 0:
            difficulty_scores_normalized = difficulty_scores_np / difficulty_scores_np.max()
        else:
            difficulty_scores_normalized = np.zeros_like(difficulty_scores_np) # All zeros if max is 0
        np.save(DIFFICULTY_SCORES_FILE, difficulty_scores_normalized)
        logger.info(f"Saved calculated difficulty scores to {DIFFICULTY_SCORES_FILE}")
        all_difficulty_scores = difficulty_scores_normalized

    # Save the full data metadata to the new cache file
    np.savez_compressed(FULL_DATA_CACHE_FILE,
                        image_paths=np.array(all_image_paths, dtype=object), # Use dtype=object for string paths
                        labels=np.array(all_labels),
                        difficulty_scores=np.array(all_difficulty_scores))
    logger.info(f"Saved full data metadata cache to {FULL_DATA_CACHE_FILE}")

    return all_image_paths, labels_array, np.array(all_difficulty_scores)
# -----------------------------------------------------------------------------
# YOLOv8 Class Mapping (Example for common objects, extend as needed)
# This maps the integer class IDs returned by YOLO to more descriptive names.
# You would need to align this with the specific YOLO model's training classes.
# -----------------------------------------------------------------------------
YOLO_CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
    # Add specific Bongard-Logo shapes if YOLO was fine-tuned for them
    80: "circle", 81: "square", 82: "triangle", 83: "star", 84: "hexagon",
    85: "pentagon", 86: "polygon", 87: "line", 88: "arc", 89: "cross",
    90: "diamond", 91: "oval", 92: "rectangle", 93: "trapezoid", 94: "arrow",
    95: "spiral", 96: "heart", 97: "cloud", 98: "lightning", 99: "abstract",
    100: "text_character"
}


# -----------------------------------------------------------------------------
# Replace dummy YOLO with a real Ultralytics YOLOv8 detector
# -----------------------------------------------------------------------------
class RealObjectDetector:
    """
    RealObjectDetector class to integrate YOLOv8 for object detection.
    This class wraps the `ultralytics.YOLO` model to detect objects in images.
    Args:
        model_path (str): Path to the YOLOv8 model weights (e.g., 'yolov8n.pt').
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = RealYOLO(model_path)
        self._cache = {}  # ← Add this line

    def detect_objects(self, image_pil: Image.Image, overall_confidence: float = 0.3):
        """
        Runs YOLOv8 on a PIL image and returns detections above a specified confidence threshold.
        Maps YOLO class IDs to human-readable names using `YOLO_CLASS_MAP`.
        Args:
            image_pil (PIL.Image.Image): The input image in PIL format.
            overall_confidence (float): Confidence threshold for filtering detections.
        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a detected object with 'id', 'bbox', 'type' (mapped class name), and 'confidence'.
        """
        # ← Insert at top of method
        key = (id(image_pil), overall_confidence)
        if key in self._cache:
            return self._cache[key]

        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        results = self.model(
            image_pil,
            verbose=False,
            augment=CONFIG['model']['yolo_augment']  # ← add augment, using CONFIG flag
        )
        detections = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf >= overall_confidence:
                # Map YOLO class ID to a human-readable name
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_name = YOLO_CLASS_MAP.get(cls, f"class_{cls}")
                detections.append({
                    "id": len(detections),  # Unique ID for this detection within the image
                    "bbox": [x1, y1, x2, y2],
                    "type": class_name,  # Use mapped class name
                    "confidence": conf
                })

        # If nothing found, try again on a 2× upscaled image
        if not detections:
            w, h = image_pil.size
            up = image_pil.resize((w*2, h*2), Image.BILINEAR)
            # Use the same augment setting from config for the upscaled image
            upscale_results = self.model(up, verbose=False, augment=CONFIG['model']['yolo_augment'])[0].boxes
            for box in upscale_results:
                conf = float(box.conf[0])
                if conf >= overall_confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # scale coordinates back down by 2×
                    detections.append({
                        "id": len(detections),
                        "bbox": [x1//2, y1//2, x2//2, y2//2],
                        "type": YOLO_CLASS_MAP.get(int(box.cls[0]), f"class_{int(box.cls[0])}"),
                        "confidence": conf
                    })
        
        self._cache[key] = detections
        return detections


# -----------------------------------------------------------------------------
# Bongard-Logo Dataset: Attribute & Relation Maps
# -----------------------------------------------------------------------------
# 1) Fill type: logos can be solid, outline, gradient, or patterned
ATTRIBUTE_FILL_MAP = {
    0: "outline",
    1: "solid",
    2: "gradient",
    3: "patterned", # e.g., stripes, dots, checkerboard
    4: "empty_transparent", # for transparent or invisible fills
}

# 2) Color palette: common logo colors (expanded)
ATTRIBUTE_COLOR_MAP = {
    0: "black",
    1: "white",
    2: "red",
    3: "blue",
    4: "green",
    5: "yellow",
    6: "orange",
    7: "purple",
    8: "pink",
    9: "brown",
    10: "gray",
    11: "gold",
    12: "silver",
    13: "cyan",
    14: "magenta",
    15: "lime",
    16: "teal",
    17: "indigo",
    18: "violet",
    19: "maroon",
    20: "navy",
    21: "olive",
    22: "coral",
    23: "beige",
    24: "multicolor", # for objects with multiple distinct colors
}

# 3) Size categories (object emerges at different scales in logo)
ATTRIBUTE_SIZE_MAP = {
    0: "tiny",
    1: "small",
    2: "medium",
    3: "large",
    4: "huge",
    5: "full_frame", # object occupies most of the image
}

# 4) Orientation: some logos contain rotated elements
ATTRIBUTE_ORIENTATION_MAP = {
    0: "upright",        # 0 degrees
    1: "rotated_45",
    2: "rotated_90",
    3: "rotated_135",
    4: "rotated_180",
    5: "rotated_225",
    6: "rotated_270",
    7: "rotated_315",
    8: "inverted",       # flipped vertically
    9: "horizontal",     # aligned horizontally
    10: "vertical",      # aligned vertically
    11: "diagonal_up",   # top-right to bottom-left
    12: "diagonal_down", # top-left to bottom-right
}

# 5) Shape: primitive and common shapes used in logos
ATTRIBUTE_SHAPE_MAP = {
    0: "circle",
    1: "square",
    2: "triangle",
    3: "star",
    4: "hexagon",
    5: "pentagon",
    6: "polygon",      # general polygon (more than 6 sides, or irregular)
    7: "line",
    8: "arc",
    9: "cross",
    10: "diamond",
    11: "oval",
    12: "rectangle",
    13: "trapezoid",
    14: "arrow",
    15: "spiral",
    16: "heart",
    17: "cloud",
    18: "lightning",
    19: "abstract",     # for highly stylized or non-geometric shapes
    20: "text_character", # if a detected object is a letter/number
}

# 6) Texture/pattern: flat vs textured vs striped
ATTRIBUTE_TEXTURE_MAP = {
    0: "flat",
    1: "striped_horizontal",
    2: "striped_vertical",
    3: "striped_diagonal",
    4: "dotted",
    5: "checkered",
    6: "gradient_linear",
    7: "gradient_radial",
    8: "noise",
    9: "grid",
    10: "wavy",
    11: "rough",
    12: "smooth",
    13: "pixelated",
    14: "blurred",
}

# -----------------------------------------------------------------------------
# Spatial Relation Map: how two logo elements relate (expanded)
# -----------------------------------------------------------------------------
RELATION_MAP = {
    0: "unrelated",
    1: "above",
    2: "below",
    3: "left_of",
    4: "right_of",
    5: "inside",         # object A is entirely within object B
    6: "contains",       # object A entirely contains object B (inverse of inside)
    7: "overlaps",       # objects share common area but neither contains other
    8: "touches",        # objects share a boundary but no common area
    9: "adjacent",       # near but not overlapping or touching
    10: "aligned_horizontally", # centers or edges are horizontally aligned
    11: "aligned_vertically",   # centers or edges are vertically aligned
    12: "symmetrical_to", # mirror symmetry with respect to an axis
    13: "connected_to",   # physically joined or linked
    14: "intersects_with", # lines/curves cross
    15: "parallel_to",    # lines/edges are parallel
    16: "perpendicular_to", # lines/edges are perpendicular
    17: "part_of",        # object A is a component of object B
    18: "surrounds",      # object A forms a boundary around object B
    19: "same_type_as",   # objects are of the same detected type (e.g., both circles)
    20: "different_type_from", # objects are of different detected types
}

class AttributeClassifier(nn.Module):
    """
    Multi-head classifier for object attributes (fill, color, size, orientation, shape, texture).
    Uses a configurable backbone (MobileNetV2, MobileNetV3, EfficientNetB0).

    Args:
        num_channels (int): Number of input image channels (1 for grayscale, 3 for RGB).
        image_size (List[int]): Expected input image size [height, width].
        backbone_name (str): Name of the pre-trained backbone to use ('mobilenet_v2',
                             'mobilenet_v3_small', or 'efficientnet_b0').
        num_fill_types (int): Number of classes for fill attribute.
        num_color_types (int): Number of classes for color attribute.
        num_size_types (int): Number of classes for size attribute.
        num_orientation_types (int): Number of classes for orientation attribute.
        num_shape_types (int): Number of classes for shape attribute.
        num_texture_types (int): Number of classes for texture attribute.
    """
    def __init__(self, num_channels, image_size, backbone_name,
                 num_fill_types=len(ATTRIBUTE_FILL_MAP),
                 num_color_types=len(ATTRIBUTE_COLOR_MAP),
                 num_size_types=len(ATTRIBUTE_SIZE_MAP),
                 num_orientation_types=len(ATTRIBUTE_ORIENTATION_MAP),
                 num_shape_types=len(ATTRIBUTE_SHAPE_MAP),
                 num_texture_types=len(ATTRIBUTE_TEXTURE_MAP)):
        super(AttributeClassifier, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.backbone_name = backbone_name

        # Load backbone with pre-trained ImageNet weights
        if backbone_name == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
            # Replace classifier with Identity to extract features before the final FC layer
            self.backbone.classifier = nn.Identity()
            # Determine feature dimension dynamically
            # For MobileNetV2, features come from features[-1] (Conv2d with 1280 output channels)
            feature_dim = 1280
        elif backbone_name == 'mobilenet_v3_small':
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.backbone = mobilenet_v3_small(weights=weights)
            # Replace classifier with Identity
            self.backbone.classifier = nn.Identity()
            # For MobileNetV3_Small, features come from avgpool (576 output channels)
            feature_dim = 576
        elif backbone_name == 'efficientnet_b0':
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
            # Replace classifier with Identity
            self.backbone.classifier = nn.Identity()
            # For EfficientNet_B0, features come from avgpool (1280 output channels)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Adapt first conv layer for single-channel (grayscale) input if needed
        if num_channels == 1:
            # Get the original first convolutional layer
            original_first_conv = self.backbone.features[0][0]
            # Create a new conv layer with 1 input channel
            new_first_conv = nn.Conv2d(
                1,
                original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias is not None
            )
            # Replace the first layer
            self.backbone.features[0][0] = new_first_conv
            logger.info(f"AttributeClassifier backbone adapted for {num_channels} input channels.")

        # Multi-head classification layers
        self.fill_head = nn.Linear(feature_dim, num_fill_types)
        self.color_head = nn.Linear(feature_dim, num_color_types)
        self.size_head = nn.Linear(feature_dim, num_size_types)
        self.orientation_head = nn.Linear(feature_dim, num_orientation_types)
        self.shape_head = nn.Linear(feature_dim, num_shape_types)
        self.texture_head = nn.Linear(feature_dim, num_texture_types)

        logger.info(f"AttributeClassifier initialized with backbone: {backbone_name}, feature_dim: {feature_dim}")

    def forward(self, x):
        """
        Forward pass for the AttributeClassifier.

        Args:
            x (torch.Tensor): Input image tensor [batch_size, channels, height, width].

        Returns:
            tuple: A tuple containing logits for each attribute head and the extracted features:
                   (fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, features)
        """
        # Ensure input is 3 channels if backbone expects it, by repeating grayscale channel
        if self.num_channels == 1 and x.shape[1] == 1 and self.backbone_name != 'mobilenet_v2': # MobileNetV2's first conv is replaced
            x = x.repeat(1, 3, 1, 1) # Repeat grayscale channel to simulate RGB for ImageNet backbones

        features = self.backbone(x)
        # Flatten features if they are not already (e.g., from avgpool)
        if features.ndim > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        fill_logits = self.fill_head(features)
        color_logits = self.color_head(features)
        size_logits = self.size_head(features)
        orientation_logits = self.orientation_head(features)
        shape_logits = self.shape_head(features)
        texture_logits = self.texture_head(features)

        return fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, features


class CurriculumSampler(torch.utils.data.Sampler):
    """
    A curriculum learning sampler that gradually increases sample difficulty over epochs.
    It provides batches of image paths and labels based on precomputed difficulty scores.

    Args:
        image_paths (List[str]): List of image file paths.
        labels (np.ndarray): Corresponding labels.
        initial_difficulty_scores (np.ndarray): Precomputed difficulty scores for each image.
        batch_size (int): The number of samples per batch.
        annealing_epochs (int): The number of early epochs during which curriculum learning
                                (gradual inclusion of harder samples) is applied.
        total_epochs (int): Total number of training epochs.
        is_train (bool): If True, this is a training sampler and applies curriculum learning.
                         If False, it's a validation sampler and uses uniform sampling.
    """
    def __init__(self, image_paths, labels, initial_difficulty_scores, batch_size, annealing_epochs, total_epochs, is_train):
        self.image_paths = image_paths
        self.labels = labels
        self.initial_difficulty_scores = np.array(initial_difficulty_scores)
        self.batch_size = batch_size
        self.annealing_epochs = annealing_epochs
        self.total_epochs = total_epochs
        self.is_train = is_train
        self.current_epoch = 0
        self.indices = list(range(len(self.image_paths)))
        # Sort indices by difficulty for curriculum learning (easiest first)
        self.difficulty_order = np.argsort(self.initial_difficulty_scores)
        logger.info(f"CurriculumSampler initialized. is_train={is_train}, annealing_epochs={annealing_epochs}")

        # Assuming CONFIG is globally available for image_size
        # These will be used by get_current_image_size
        self.initial_image_size = CONFIG['data']['initial_image_size']
        self.final_image_size = CONFIG['data']['image_size']
        self.progressive_resizing_epochs = CONFIG['data']['progressive_resizing_epochs']


    def __iter__(self):
        """
        Yields batches of (image_paths, labels) according to the current curriculum stage.
        """
        if self.is_train and self.current_epoch < self.annealing_epochs:
            # Curriculum learning phase: gradually include harder samples
            progress = self.current_epoch / self.annealing_epochs
            # Start with 10% easiest samples and linearly increase to 100%
            num_samples_to_include = int(len(self.indices) * (0.1 + 0.9 * progress))
            
            # Select the easiest samples based on difficulty scores
            current_indices = self.difficulty_order[:num_samples_to_include]
            
            # Shuffle the selected subset to ensure randomness within the current difficulty band
            np.random.shuffle(current_indices)
            logger.debug(f"Epoch {self.current_epoch}: Curriculum sampling, including {num_samples_to_include} easiest samples.")
        else:
            # Uniform sampling phase (after annealing or for validation)
            current_indices = np.array(self.indices)
            np.random.shuffle(current_indices) # Always shuffle for randomness
            logger.debug(f"Epoch {self.current_epoch}: Uniform sampling.")
        
        # Yield batches of (paths, labels)
        for i in range(0, len(current_indices), self.batch_size):
            batch_indices = current_indices[i:i + self.batch_size]
            batch_paths = [self.image_paths[j] for j in batch_indices]
            batch_labels = [self.labels[j] for j in batch_indices]
            yield batch_paths, batch_labels

    def __len__(self):
        """
        Returns the number of batches in the current epoch, considering curriculum learning.
        """
        if self.is_train and self.current_epoch < self.annealing_epochs:
            progress = self.current_epoch / self.annealing_epochs
            num_samples_to_include = int(len(self.indices) * (0.1 + 0.9 * progress))
            return math.ceil(num_samples_to_include / self.batch_size)
        return math.ceil(len(self.indices) / self.batch_size)

    def set_epoch(self, epoch):
        """
        Sets the current epoch for the sampler, used to control the curriculum progression.

        Args:
            epoch (int): The current training epoch.
        """
        self.current_epoch = epoch
        logger.debug(f"Sampler epoch set to {self.current_epoch}")

    def get_current_image_size(self):
        """
        Calculates the current image size for progressive resizing based on the epoch.

        Returns:
            List[int]: The [height, width] for the current epoch's image size.
        """
        if self.current_epoch < self.progressive_resizing_epochs:
            # Linear interpolation for image size
            ratio = self.current_epoch / max(1, self.progressive_resizing_epochs - 1)
            h = int(self.initial_image_size[0] + ratio * (self.final_image_size[0] - self.initial_image_size[0]))
            w = int(self.initial_image_size[1] + ratio * (self.final_image_size[1] - self.initial_image_size[1]))
            return [h, w]
        return self.final_image_size




# --- DALI Pipeline Helper Functions ---
# --- DALI Pipeline Helper Functions ---
def get_dummy_decoded_image(image_size, num_channels):
    """
    Generates a dummy image (NumPy array) for use when actual image loading fails
    or for filling partial batches in DALI.

    Args:
        image_size (List[int]): Desired [height, width] of the dummy image.
        num_channels (int): Number of channels for the dummy image (1 for grayscale, 3 for RGB).

    Returns:
        np.ndarray: A contiguous NumPy array representing the dummy image.
    """
    dummy_height, dummy_width = image_size[0], image_size[1]
    if num_channels == 1:
        dummy_img_base = np.full((dummy_height, dummy_width, 1), 127, dtype=np.uint8)
    else:
        dummy_img_base = np.full((dummy_height, dummy_width, num_channels), 127, dtype=np.uint8)
    expected_shape = (dummy_height, dummy_width, num_channels) if num_channels > 1 else (dummy_height, dummy_width, 1)
    if num_channels == 1 and dummy_img_base.shape[2] == 1:
        expected_shape = (dummy_height, dummy_width, 1)
    assert dummy_img_base.shape == expected_shape, \
        f"Generated dummy image has shape {dummy_img_base.shape}, expected {expected_shape}"
    return np.ascontiguousarray(dummy_img_base)

def infinite_batch_generator(sampler, batch_size, image_size, num_channels, logger):
    """
    An infinite generator that yields batches of pre-decoded images, labels, and affine matrices
    for use with DALI's `fn.external_source`. It handles `StopIteration` from the sampler
    by resetting it for the next epoch and filling incomplete batches with dummy data.

    Args:
        sampler (CurriculumSampler): The sampler instance to draw image paths and labels from.
        batch_size (int): The desired batch size.
        image_size (List[int]): The [height, width] to resize images to after decoding.
        num_channels (int): Number of channels for the images (1 for grayscale, 3 for RGB).
        logger (logging.Logger): Logger instance for logging messages.

    Yields:
        tuple: A tuple containing:
            - images (np.ndarray): A batch of images as NumPy arrays (HWC, UINT8).
            - labels (np.ndarray): A batch of labels as NumPy array (B, 1, INT64).
            - affine_matrices_1 (np.ndarray): Identity affine matrices for view 1 (B, 2, 3, FLOAT32).
            - affine_matrices_2 (np.ndarray): Identity affine matrices for view 2 (B, 2, 3, FLOAT32).
    """
    pre_decoded_dummy_image = get_dummy_decoded_image(image_size, num_channels)
    identity_matrix = np.array([ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0] ], dtype=np.float32)
    it = iter(sampler)
    while True:
        paths_batch, labels_raw = [], []
        try:
            paths_batch, labels_raw = next(it)
        except StopIteration:
            sampler.set_epoch(sampler.current_epoch + 1)
            it = iter(sampler)
            try:
                paths_batch, labels_raw = next(it)
            except StopIteration:
                logger.warning("Sampler yielded StopIteration even after reset. Filling batch with dummies.")
                paths_batch = [None] * batch_size
                labels_raw = [0] * batch_size

        if len(paths_batch) < batch_size:
            deficit = batch_size - len(paths_batch)
            paths_batch.extend(paths_batch[:deficit])
            labels_raw.extend(labels_raw[:deficit])

        # ✅ CORRECTION STARTS HERE
        # Ensure batch is exactly batch_size
        paths_batch = paths_batch[:batch_size]
        labels_raw = labels_raw[:batch_size]
        # ✅ CORRECTION ENDS HERE

        image_np_list = []
        for i, path in enumerate(paths_batch):
            current_image_np = None
            try:
                if path is None:
                    current_image_np = pre_decoded_dummy_image
                else:
                    with open(path, 'rb') as f:
                        img_bytes = f.read()
                    if not img_bytes: raise ValueError(f"Image file {path} is empty.")
                    
                    image_np_from_bytes = np.frombuffer(img_bytes, dtype=np.uint8)
                    if not isinstance(image_np_from_bytes, np.ndarray) or image_np_from_bytes.dtype != np.uint8 or image_np_from_bytes.size == 0:
                        raise TypeError(f"Image bytes from {path} invalid numpy array conversion.")
                    
                    cv2_decode_flag = cv2.IMREAD_GRAYSCALE if num_channels == 1 else cv2.IMREAD_COLOR
                    decoded_img = cv2.imdecode(image_np_from_bytes, cv2_decode_flag)
                    
                    if decoded_img is None: raise ValueError(f"Image file {path} could not be decoded by OpenCV with flag {cv2_decode_flag}.")
                    if num_channels == 1:
                        if decoded_img.ndim == 3 and decoded_img.shape[2] > 1: decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
                        if decoded_img.ndim == 2: decoded_img = np.expand_dims(decoded_img, axis=2)
                    elif num_channels == 3:
                        if decoded_img.ndim == 2: decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_GRAY2BGR)
                        elif decoded_img.ndim == 3 and decoded_img.shape[2] == 4: decoded_img = decoded_img[:, :, :3]
                    
                    resized_img = cv2.resize(decoded_img, (image_size[1], image_size[0]))
                    if num_channels == 1:
                        if resized_img.ndim == 3 and resized_img.shape[2] == 3: resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                        if resized_img.ndim == 2: resized_img = np.expand_dims(resized_img, axis=2)
                    elif num_channels == 3:
                        if resized_img.ndim == 2: resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
                        elif resized_img.ndim == 3 and resized_img.shape[2] == 4: resized_img = resized_img[:, :, :3]
                    
                    final_expected_shape = (image_size[0], image_size[1], num_channels) if num_channels > 1 else (image_size[0], image_size[1], 1)
                    if resized_img.shape != final_expected_shape:
                        raise RuntimeError(f"Resized image {path} has shape {resized_img.shape}, expected {final_expected_shape} after resize and channel enforcement.")
                    
                    current_image_np = resized_img
            except Exception as e:
                logger.error(f"Error reading or processing image {path}: {e}. Replacing with dummy image to maintain batch consistency.")
                current_image_np = pre_decoded_dummy_image
            
            current_image_np = np.ascontiguousarray(current_image_np)
            assert current_image_np.flags['C_CONTIGUOUS'], f"Batch element {i} is not contiguous after processing."
            # print(f"Image {i} shape: {current_image_np.shape}, dtype: {current_image_np.dtype}, contiguous: {current_image_np.flags['C_CONTIGUOUS']}") # Commented out for cleaner logs
            image_np_list.append(current_image_np)

        labels_np = np.array(labels_raw, dtype=np.int64).reshape(batch_size, 1)
        labels_np = np.ascontiguousarray(labels_np)
        identity_matrix = np.array([ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0] ], dtype=np.float32) # Re-define for local scope
        affine_matrices_1 = np.stack([identity_matrix.copy() for _ in range(batch_size)]).astype(np.float32)
        affine_matrices_1 = np.ascontiguousarray(affine_matrices_1)
        affine_matrices_2 = np.stack([identity_matrix.copy() for _ in range(batch_size)]).astype(np.float32)
        affine_matrices_2 = np.ascontiguousarray(affine_matrices_2)

        if image_np_list:
            assert all(img.shape == image_np_list[0].shape for img in image_np_list), "Not all images in image_np_list have the same shape"
        
        expected_image_shape = (batch_size, image_size[0], image_size[1], num_channels) if num_channels > 1 else (batch_size, image_size[0], image_size[1], 1)
        image_batch_tensor = np.stack(image_np_list, axis=0)
        assert image_batch_tensor.shape == expected_image_shape, f"Stacked image batch shape {image_batch_tensor.shape} does not match expected {expected_image_shape}"
        assert image_batch_tensor.flags['C_CONTIGUOUS'], "Stacked image batch tensor is not contiguous"
        # print(f"Stacked batch shape: {image_batch_tensor.shape}, dtype: {image_batch_tensor.dtype}") # Commented out for cleaner logs

        assert labels_np.shape == (batch_size, 1), f"Labels shape {labels_np.shape} does not match expected {(batch_size, 1)}"
        assert labels_np.flags['C_CONTIGUOUS'], "Labels array is not contiguous"
        assert affine_matrices_1.shape == (batch_size, 2, 3), f"Affine 1 shape {affine_matrices_1.shape} does not match expected {(batch_size, 2, 3)}"
        assert affine_matrices_1.flags['C_CONTIGUOUS'], "Affine 1 array is not contiguous"
        assert affine_matrices_2.shape == (batch_size, 2, 3), f"Affine 2 shape {affine_matrices_2.shape} does not match expected {(batch_size, 2, 3)}"
        assert affine_matrices_2.flags['C_CONTIGUOUS'], "Affine 2 array is not contiguous"

        def log_gen_print(name, arr):
            # print(f"[GEN] {name:14s} shape={arr.shape}, " # Commented out for cleaner logs
            #              f"dtype={arr.dtype}, "
            #              f"contig={arr.flags['C_CONTIGUOUS']}")
            pass # Keep function but disable print for cleaner logs

        log_gen_print("images", image_batch_tensor)
        log_gen_print("labels", labels_np)
        log_gen_print("affine1", affine_matrices_1)
        log_gen_print("affine2", affine_matrices_2)

        try:
            yield image_batch_tensor, labels_np, affine_matrices_1, affine_matrices_2
        except Exception as e:
            logger.error(f"Error yielding batch from infinite_batch_generator: {e}. Attempting to recover with dummy batch.")
            dummy_image_np_stacked = np.ascontiguousarray(np.stack([pre_decoded_dummy_image] * batch_size, axis=0))
            dummy_labels = np.zeros((batch_size, 1), dtype=np.int64)
            dummy_labels = np.ascontiguousarray(dummy_labels)
            dummy_affine_1 = np.ascontiguousarray(np.stack([identity_matrix.copy() for _ in range(batch_size)]).astype(np.float32))
            dummy_affine_2 = np.ascontiguousarray(np.stack([identity_matrix.copy() for _ in range(batch_size)]).astype(np.float32))
            log_gen_print("images (dummy)", dummy_image_np_stacked)
            log_gen_print("labels (dummy)", dummy_labels)
            log_gen_print("affine1 (dummy)", dummy_affine_1)
            log_gen_print("affine2 (dummy)", dummy_affine_2)
            yield dummy_image_np_stacked, dummy_labels, dummy_affine_1, dummy_affine_2

class DaliPipeline(Pipeline):
    """
    DALI Pipeline for efficient, GPU-accelerated image loading and augmentation.
    Handles decoding, resizing, various augmentations, and normalization. Supports single-channel (grayscale) and three-channel (RGB) images.
    Can generate a second augmented view for feature consistency loss.
    Uses fn.external_source to directly consume batches of raw NumPy images and labels from a Python generator.

    Args:
        batch_size (int): Number of samples per batch.
        num_threads (int): Number of CPU threads to use for DALI operations.
        device_id (int): GPU device ID to use (-1 for CPU).
        seed (int): Random seed for reproducibility.
        image_size (List[int]): Target image size [height, width] after resizing.
        is_train (bool): If True, applies training-specific augmentations.
        sampler (CurriculumSampler): The sampler providing image paths and labels.
        num_channels (int): Number of image channels (1 for grayscale, 3 for RGB).
        feature_consistency_alpha (float): Alpha value for feature consistency loss (0.0 to disable).
        logger (logging.Logger): Logger instance for logging messages.
        imagenet_mean (List[float]): Mean values for image normalization (0-255 range).
        imagenet_std (List[float]): Standard deviation values for image normalization (0-255 range).
        initial_image_size (List[int], optional): Initial image size for DALI decoding,
                                                  used for progressive resizing. Defaults to image_size.
    """
    def __init__(self, batch_size, num_threads, device_id, seed, image_size, is_train, sampler, num_channels, feature_consistency_alpha, logger, imagenet_mean, imagenet_std, initial_image_size=None):
        super(DaliPipeline, self).__init__(
            batch_size, num_threads, device_id, seed, prefetch_queue_depth=2, exec_async=True, exec_pipelined=True
        )
        self.is_train = is_train
        self.image_size = tuple(image_size)
        self.initial_image_size = initial_image_size if initial_image_size else self.image_size
        self.num_channels = num_channels
        self.feature_consistency_alpha = feature_consistency_alpha
        self.logger = logger
        self.batch_gen = infinite_batch_generator(sampler, batch_size, self.initial_image_size, self.num_channels, self.logger) # Use initial_image_size for decoding
        self.mean = np.array(imagenet_mean[:num_channels] if num_channels == 1 else imagenet_mean, dtype=np.float32)
        self.std = np.array(imagenet_std[:num_channels] if num_channels == 1 else imagenet_std, dtype=np.float32)
        self.rng = np.random.default_rng(seed=seed)
        self._python_batch_size = batch_size

    def define_graph(self):
        """
        Defines the DALI graph, specifying the sequence of operations for data loading and augmentation.
        Returns 5 dense Tensors for each batch:
        1) image_view_1 (out)
           • dtype: FLOAT32
           • layout: NCHW
           • shape: (B, C, H, W)
           • normalized via (x-mean)/std
        2) image_view_2
           • same as image_view_1, second augmented view
           • dtype: FLOAT32, NCHW, (B, C, H, W)
        3) label
           • dtype: INT64
           • shape: (B,)  — one scalar per sample
        4) affine_matrix_1
           • dtype: FLOAT32
           • shape: (B, 2, 3)
           • the exact same matrix used in warp_affine for view-1
        5) affine_matrix_2
           • dtype: FLOAT32
           • shape: (B, 2, 3)
           • the matrix for view-2 warp_affine
        """
        # pull in raw NumPy arrays from Python generator
        images, labels, aff1, aff2 = fn.external_source(
            source=self.batch_gen, num_outputs=4,
            layout=["HWC", "", "", ""],  # layout only matters for images
            dtype=[types.UINT8, types.INT64, types.FLOAT, types.FLOAT],
            batch=True)

        # 2) image+label → GPU
        images_gpu = fn.copy(images, device="gpu")
        labels_gpu = fn.cast(fn.copy(labels, device="gpu"), dtype=types.INT64)

        # 3) AFFINES → GPU **and** force into a real Tensor
        #    copy+cast gives you a TensorList on GPU; reshape makes it dense
        aff1_gpu = fn.cast(fn.copy(aff1, device="gpu"), dtype=types.FLOAT)
        # CORRECT: reshape each sample from [2,3] → [2,3]
        aff1_gpu = fn.reshape(aff1_gpu, shape=(2, 3))

        aff2_gpu = fn.cast(fn.copy(aff2, device="gpu"), dtype=types.FLOAT)
        aff2_gpu = fn.reshape(aff2_gpu, shape=(2, 3))

        # 4) your view - 1 augmentations on images_gpu …
        images_view1_temp = images_gpu # Start with GPU-copied images
        if self.is_train:
            out = fn.random_resized_crop(images_view1_temp,
                                         size=self.image_size,    # HxW here just sets crop area
                                         random_area=(0.8,1.0),
                                         random_aspect_ratio=(0.75,1.33),
                                         device="gpu")
            out = fn.flip(out, horizontal=fn.random.coin_flip(), device="gpu")
            out = fn.warp_affine(out, aff1_gpu,
                                 fill_value=0.0, inverse_map=False,
                                 device="gpu")
            # Apply brightness and contrast adjustments
            out = fn.brightness_contrast(out, brightness=fn.random.uniform(range=(0.8, 1.2)), contrast=fn.random.uniform(range=(0.8, 1.2)), device="gpu")
            # Apply hue and saturation for 3-channel images
            if self.num_channels == 3:
                out = fn.hue(out, hue=fn.random.uniform(range=(-0.1, 0.1)), device="gpu")
                out = fn.saturation(out, saturation=fn.random.uniform(range=(0.8, 1.2)), device="gpu")
            # Apply rotation
            rotation_angle_dali = fn.random.uniform(range=(-15.0, 15.0))
            out = fn.rotate(out, angle=rotation_angle_dali, fill_value=0.0, device="gpu")
            # Apply Gaussian blur
            out = fn.gaussian_blur(out, sigma=fn.random.uniform(range=(0.0, 1.0)), device="gpu")
            # ENFORCE uniform output size after all spatial augmentations
            out = fn.resize(out,
                            resize_x=self.image_size[1],
                            resize_y=self.image_size[0],
                            interp_type=types.INTERP_LINEAR,
                            device="gpu")
        else:
            out = fn.resize(images_view1_temp,
                            resize_x=self.image_size[1],
                            resize_y=self.image_size[0],
                            interp_type=types.INTERP_LINEAR,
                            device="gpu")

        # 5) normalize & NCHW - No 'crop' argument needed here as fn.resize already handled sizing
        output_layout = types.NCHW
        out = fn.crop_mirror_normalize(
            out,
            dtype=types.FLOAT,
            output_layout=output_layout,
            mean=self.mean,
            std=self.std,
            device="gpu"
        )

        # 6) second view, exact same pattern with aff2_gpu
        images_2_temp = fn.copy(images_gpu, device="gpu") # Start with GPU-copied images
        if self.is_train and self.feature_consistency_alpha > 0:
            images_2 = fn.random_resized_crop(images_2_temp,
                                              size=self.image_size, # HxW here just sets crop area
                                              random_area=(0.7,1.0),
                                              random_aspect_ratio=(0.7,1.4),
                                              device="gpu")
            images_2 = fn.flip(images_2, horizontal=fn.random.coin_flip(), device="gpu")
            images_2 = fn.warp_affine(images_2, aff2_gpu,
                                      fill_value=0.0, inverse_map=False,
                                      device="gpu")
            # Apply brightness and contrast adjustments
            images_2 = fn.brightness_contrast(images_2, brightness=fn.random.uniform(range=(0.7, 1.3)), contrast=fn.random.uniform(range=(0.7, 1.3)), device="gpu")
            # Apply hue and saturation for 3-channel images
            if self.num_channels == 3:
                images_2 = fn.hue(images_2, hue=fn.random.uniform(range=(-0.2, 0.2)), device="gpu")
                images_2 = fn.saturation(images_2, saturation=fn.random.uniform(range=(0.7, 1.3)), device="gpu")
            # Apply rotation
            images_2_rotation_angle_dali = fn.random.uniform(range=(-20.0, 20.0))
            images_2 = fn.rotate(images_2, angle=images_2_rotation_angle_dali, fill_value=0.0, device="gpu")
            # Apply Gaussian blur
            images_2 = fn.gaussian_blur(images_2, sigma=fn.random.uniform(range=(0.0, 1.5)), device="gpu")
            # ENFORCE uniform output size after all spatial augmentations
            images_2 = fn.resize(images_2,
                                 resize_x=self.image_size[1],
                                 resize_y=self.image_size[0],
                                 interp_type=types.INTERP_LINEAR,
                                 device="gpu")
        else:
            images_2 = fn.resize(images_2_temp,
                                 resize_x=self.image_size[1],
                                 resize_y=self.image_size[0],
                                 interp_type=types.INTERP_LINEAR,
                                 device="gpu")

        images_2 = fn.crop_mirror_normalize(
            images_2,
            dtype=types.FLOAT,
            output_layout=output_layout,
            mean=self.mean,
            std=self.std,
            device="gpu"
        )

        # 7) Return five *real* Tensors:
        return out, images_2, labels_gpu, aff1_gpu, aff2_gpu

import numpy as np
from scipy.spatial.distance import cdist # Import for graph construction

class RelationGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, num_relation_types, gnn_depth=2):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_relation_types = num_relation_types
        self.gnn_depth = gnn_depth # Initial depth from config
        
        # Dynamic GNN Depth: Store max and min depth from config
        self.max_depth = CONFIG['model'].get('gnn_max_depth', gnn_depth)
        self.min_depth = CONFIG['model'].get('gnn_min_depth', 1)
        self.depth_schedule = CONFIG['model'].get('gnn_depth_schedule', 'fixed') # 'fixed', 'linear', 'adaptive'

        if not HAS_PYG:
            logger.warning("PyTorch Geometric (PyG) not installed. Using dummy GCNConv. RelationGNN will not function correctly.")
        hidden_dim = node_feature_dim * 2 # A common choice for hidden dimension
        
        self.convs = nn.ModuleList([
            GCNConv(node_feature_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(self.max_depth) # Construct up to max_depth
        ])
        
        # Learnable normalization layer for edge attributes
        self.edge_norm = nn.LayerNorm(edge_feature_dim)
        
        # Ensure Set2Set is initialized if PyG is present
        if HAS_PYG:
            self.set2set = Set2Set(hidden_dim, processing_steps=6, num_layers=2)
        else:
            self.set2set = Set2Set(hidden_dim, 6, 2) # Dummy Set2Set
            
        logger.info(f"RelationGNN initialized with node_feature_dim={node_feature_dim}, gnn_depth={gnn_depth} (max_depth={self.max_depth})")

    def forward(self, x, edge_index, batch):
        """
        Forward pass for the RelationGNN.
        Args:
            x (torch.Tensor): Node features (object embeddings) [num_nodes, node_feature_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            batch (torch.Tensor): Batch vector, which assigns each node to a specific example [num_nodes].
        Returns:
            tuple: A tuple containing:
                - graph_embedding (torch.Tensor): Graph-level representation for each problem.
                - final_node_features (torch.Tensor): Node features after the last GCN layer.
        """
        # Determine actual depth based on schedule
        current_depth = self.gnn_depth # Default to initial gnn_depth
        if self.depth_schedule == 'linear':
            # Assuming 'current_epoch' is set by the PerceptionModule
            curr_epoch = getattr(self, 'current_epoch', 0)
            max_e = CONFIG['model']['epochs'] # Assuming CONFIG is accessible
            if max_e > 0:
                depth = int(self.max_depth * (curr_epoch / max_e))
                current_depth = max(self.min_depth, depth)
            else:
                current_depth = self.min_depth # If 0 epochs, use min depth
        elif self.depth_schedule == 'adaptive':
            # Example: adaptive depth based on graph density or size
            # For simplicity, let's keep it fixed for now or implement a placeholder
            # A more complex adaptive strategy would involve analyzing graph properties here
            current_depth = self.max_depth # For now, adaptive defaults to max
        
        # Ensure current_depth does not exceed the actual number of conv layers
        current_depth = min(current_depth, len(self.convs))

        for i, conv in enumerate(self.convs[:current_depth]): # Iterate up to current_depth
            x = conv(x, edge_index)
            if i < current_depth - 1: # Apply ReLU after all but the last GCN layer
                x = F.relu(x)
        
        # Global pooling (e.g., Set2Set) to get a graph-level representation for each problem
        graph_embedding = self.set2set(x, batch)
        
        # Return graph_embedding directly, as it's the input for the final classifier head
        return graph_embedding, x # Return graph_embedding and final node features
    
    def classify_edges(self, node_features, edge_index, edge_attributes):
        """
        Classifies edges based on concatenated features of connected nodes and edge attributes.
        Args:
            node_features (torch.Tensor): Output features from the final GNN layer for all nodes.
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            edge_attributes (torch.Tensor): Features describing the edge itself (e.g., relative bbox info).
        Returns:
            torch.Tensor: Logits for each relation type for each edge.
        """
        if edge_index.numel() == 0:
            # Handle empty edge_index: return an empty tensor of appropriate shape
            return torch.empty((0, self.num_relation_types), device=node_features.device)
        row, col = edge_index
        
        # Concatenate features of connected nodes
        node_pair_features = torch.cat([node_features[row], node_features[col]], dim=-1)
        
        # Apply learnable normalization to edge attributes
        if edge_attributes.numel() > 0:
            normalized_edge_attributes = self.edge_norm(edge_attributes)
            # Concatenate node pair features with normalized edge attributes
            combined_features = torch.cat([node_pair_features, normalized_edge_attributes], dim=-1)
        else:
            # If no edge attributes, just use node_pair_features
            combined_features = node_pair_features
        # Simple linear layer to classify edge type
        # The input dimension is (2 * node_feature_dim) + edge_feature_dim
        input_dim_edge_mlp = (2 * node_features.size(-1)) + self.edge_feature_dim
        if not hasattr(self, 'edge_mlp'):
            self.edge_mlp = nn.Linear(input_dim_edge_mlp, self.num_relation_types).to(node_features.device)
        
        edge_logits = self.edge_mlp(combined_features)
        
        # NEW: Log or save edge probabilities for visualization
        if CONFIG['debug']['visualize_gnn_weights']:
            edge_probs = F.softmax(edge_logits, dim=1)
            logger.debug(f"GNN Edge Probabilities - Shape: {edge_probs.shape}, Mean: {edge_probs.mean().item():.4f}")
        
        return edge_logits

class RealObjectDetector:
    """
    RealObjectDetector class to integrate YOLOv8 for object detection.
    This class wraps the `ultralytics.YOLO` model to detect objects in images.
    Args:
        model_path (str): Path to the YOLOv8 model weights (e.g., 'yolov8n.pt').
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        # Dummy RealYOLO if not available
        try:
            from ultralytics import YOLO as RealYOLO_Actual
            self.model = RealYOLO_Actual(model_path)
        except ImportError:
            logger.warning("Ultralytics YOLO not found. Using dummy RealYOLO for object detection.")
            self.model = RealYOLO(model_path) # Fallback to dummy
        self._cache = {}  # Add this line
    def detect_objects(self, image_pil: Image.Image, overall_confidence: float = 0.3):
        """
        Runs YOLOv8 on a PIL image and returns detections above a specified confidence threshold.
        Maps YOLO class IDs to human-readable names using `YOLO_CLASS_MAP`.
        Args:
            image_pil (PIL.Image.Image): The input image in PIL format.
            overall_confidence (float): Confidence threshold for filtering detections.
        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a detected object with 'id', 'bbox', 'type' (mapped class name), and 'confidence'.
        """
        key = (id(image_pil), overall_confidence)
        if key in self._cache:
            return self._cache[key]
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        detections = []
        try:
            results = self.model(
                image_pil,
                verbose=False,
                augment=CONFIG['model']['yolo_augment']  # ← add augment, using CONFIG flag
            )
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if conf >= overall_confidence:
                    # Map YOLO class ID to a human-readable name
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_name = YOLO_CLASS_MAP.get(cls, f"class_{cls}")
                    detections.append({
                        "id": len(detections),  # Unique ID for this detection within the image
                        "bbox": [x1, y1, x2, y2],
                        "type": class_name,  # Use mapped class name
                        "confidence": conf
                    })
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}. Attempting fallback mechanisms.")

        # If nothing found, try again on a 2× upscaled image
        if not detections:
            w, h = image_pil.size
            up = image_pil.resize((w*2, h*2), Image.BILINEAR)
            try:
                # Use the same augment setting from config for the upscaled image
                upscale_results = self.model(up, verbose=False, augment=CONFIG['model']['yolo_augment'])[0].boxes
                for box in upscale_results:
                    conf = float(box.conf[0])
                    if conf >= overall_confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        # scale coordinates back down by 2×
                        detections.append({
                            "id": len(detections),
                            "bbox": [x1//2, y1//2, x2//2, y2//2],
                            "type": YOLO_CLASS_MAP.get(int(box.cls[0]), f"class_{int(box.cls[0])}"),
                            "confidence": conf
                        })
            except Exception as e:
                logger.warning(f"Upscaled YOLO detection failed: {e}. Proceeding to contour fallback.")

        # Robust "No Objects Detected" Fallback: basic contour detection for blobs
        if not detections:
            logger.warning(f"YOLO failed, attempting fallback blob detection for image.")
            try:
                img_gray = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Limit fallback contours to avoid excessive dummy nodes
                max_fallback_cnt = CONFIG['debug'].get('max_fallback_cnt', 5)
                for cnt_idx, cnt in enumerate(contours[:max_fallback_cnt]):
                    x,y,w,h = cv2.boundingRect(cnt)
                    detections.append({
                        "id": len(detections),
                        "bbox": [x,y,x+w,y+h],
                        "type": "fallback_blob",
                        "confidence": 0.1 # Assign a low confidence for fallback detections
                    })
                logger.warning(f"Fallback blob detection found {len(detections)} regions.")
            except Exception as e:
                logger.error(f"Fallback contour detection failed: {e}. No objects detected for this image.")
        
        self._cache[key] = detections
        return detections

# --- 1. Label Smoothing Loss ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Implements Label Smoothing Cross Entropy Loss.

    Args:
        epsilon (float): The smoothing factor, typically between 0.0 (no smoothing) and 1.0.
    """
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, target):
        """
        Calculates the label smoothed cross-entropy loss.

        Args:
            logits (torch.Tensor): Raw, unnormalized scores from the model [batch_size, num_classes].
            target (torch.Tensor): Ground truth labels [batch_size].

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # logits: [B, C], target: [B]
        num_classes = logits.size(1)
        log_preds = self.log_softmax(logits)
        with torch.no_grad():
            # one-hot with smoothing
            one_hot = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
            one_hot = one_hot * (1 - self.epsilon) + self.epsilon / num_classes
        loss = - (one_hot * log_preds).sum(dim=1).mean()
        return loss

# --- 2. Knowledge Distillation Loss ---
class DistillationLoss(nn.Module):
    """
    Implements Knowledge Distillation Loss, combining a KL Divergence term
    between student and teacher softened probabilities and a Cross Entropy term
    for the student's predictions against hard labels.

    Args:
        temperature (float): Temperature for softening logits in KL divergence.
        alpha (float): Weight for the KL divergence term (1 - alpha for CE term).
    """
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        """
        Calculates the distillation loss.

        Args:
            student_logits (torch.Tensor): Logits from the student model.
            teacher_logits (torch.Tensor): Logits from the teacher model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The scalar distillation loss value.
        """
        T = self.temperature
        # softened probabilities
        student_logprob = F.log_softmax(student_logits / T, dim=1)
        teacher_prob    = F.softmax(teacher_logits / T, dim=1)
        kd_loss = self.kl_div(student_logprob, teacher_prob) * (T * T)
        ce_loss = self.ce(student_logits, labels)
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

# --- 3. MixUp / CutMix Utilities ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Applies MixUp augmentation to a batch of data.

    Args:
        x (torch.Tensor): Input images tensor [batch_size, channels, height, width].
        y (torch.Tensor): Original labels tensor [batch_size].
        alpha (float): Beta distribution parameter for lambda.
        device (str): Device to perform operations on.

    Returns:
        tuple: A tuple containing:
            - mixed_x (torch.Tensor): Augmented images.
            - y_a (torch.Tensor): Original labels.
            - y_b (torch.Tensor): Shuffled labels.
            - lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """
    Applies CutMix augmentation to a batch of data.

    Args:
        x (torch.Tensor): Input images tensor [batch_size, channels, height, width].
        y (torch.Tensor): Original labels tensor [batch_size].
        alpha (float): Beta distribution parameter for lambda.
        device (str): Device to perform operations on.

    Returns:
        tuple: A tuple containing:
            - x (torch.Tensor): Augmented images.
            - y_a (torch.Tensor): Original labels.
            - y_b (torch.Tensor): Shuffled labels.
            - lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    b, c, h, w = x.size()
    index = torch.randperm(b, device=device)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates loss for MixUp/CutMix augmented data.

    Args:
        criterion (torch.nn.Module): The base loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): First set of labels.
        y_b (torch.Tensor): Second set of labels.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: The mixed loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- 4. Early Stopping ---
class EarlyStopping:
    """
    Monitors a validation metric and stops training when it stops improving
    for a specified number of epochs. Saves the best model checkpoint.

    Args:
        patience (int): How many epochs to wait after last improvement before stopping.
        delta (float): Minimum change in the monitored metric to qualify as an improvement.
        path (str): Path to save the best model checkpoint.
        monitor_metric (str): Name of the metric to monitor (e.g., 'val_loss', 'val_accuracy').
        verbose (bool): If True, prints messages when early stopping is triggered or counter increases.
    """
    def __init__(self, patience=5, delta=0.0, path='checkpoint.pt', monitor_metric='val_loss', verbose=False):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.monitor = monitor_metric
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metrics, model, optimizer, epoch):
        """
        Checks the current metrics against the best score and updates the early stopping state.

        Args:
            metrics (dict): Dictionary of current validation metrics.
            model (torch.nn.Module): The model to save if it's the best.
            optimizer (torch.optim.Optimizer): The optimizer to save its state.
            epoch (int): Current epoch number.
        """
        score = metrics.get(self.monitor, None)
        if score is None:
            logger.warning(f"Monitored metric '{self.monitor}' not found in metrics. Early stopping skipped.")
            return
        
        # Determine if current score is an improvement based on monitor_metric
        # For loss, lower is better; for accuracy, higher is better.
        if self.monitor.startswith('val_loss'):
            should_improve = score < (self.best_score - self.delta)
        else: # Assuming accuracy or similar where higher is better
            should_improve = score > (self.best_score + self.delta)

        if self.best_score is None or should_improve:
            self.best_score = score
            self._save_checkpoint(model, optimizer, epoch, score)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model, optimizer, epoch, score):
        """
        Saves the model and optimizer state to a checkpoint file.

        Args:
            model (torch.nn.Module): The model to save.
            optimizer (torch.optim.Optimizer): The optimizer to save.
            epoch (int): Current epoch number.
            score (float): The monitored metric score.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            self.monitor: score
        }, self.path)


class PerceptionModule(nn.Module):
    """
    The core Perception Module (System 1) for Bongard problems.
    It integrates an object detector (YOLOv8), a multi-head attribute classifier,
    and a Graph Neural Network (GNN) for relational reasoning.
    Handles training, evaluation, logging, and various optimization techniques
    like AMP, SWA, MixUp/CutMix, Label Smoothing, and QAT.
    Args:
        config (dict): A dictionary containing all configuration parameters
                       for data, model, and training.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config['model']['num_classes']
        self.use_amp = config['training']['use_amp']
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32
        self.use_gradient_checkpointing = config['training']['use_gradient_checkpointing']
        self.use_swa = config['training']['use_swa']
        self.feature_consistency_alpha = config['training']['feature_consistency_alpha']
        self.log_interval = config['training'].get('log_interval', 10) # Log every N batches
        self.eval_interval = config['training'].get('eval_interval', 1) # Evaluate every N epochs
        self.save_interval = config['training'].get('save_interval', 5) # Save every N epochs
        self.save_path = config['training']['checkpoint_dir']
        self.epochs = config['model']['epochs']
        self.optimizer_name = config['model']['optimizer']
        self.scheduler_name = config['model']['scheduler']
        self.learning_rate = config['model']['initial_learning_rate'] # Use initial_learning_rate
        self.weight_decay = config['training']['weight_decay']
        self.gradient_clip_norm = config['training']['max_grad_norm'] # Use max_grad_norm from training
        self.swa_start_epoch = int(self.epochs * config['training']['swa_start_epoch_ratio']) # Use ratio from training
        
        # Define edge_feature_dim here as an instance attribute
        self.edge_feature_dim = 6 # dx, dy, dw, dh, angle, iou
        # NEW: Knowledge Distillation setup
        self.use_knowledge_distillation = config['training']['use_knowledge_distillation']
        self.teacher_model = None
        self.distillation_criterion = None
        if self.use_knowledge_distillation:
            teacher_model_path = config['model'].get('teacher_model_path')
            if teacher_model_path and os.path.exists(teacher_model_path):
                logger.info(f"Loading teacher model from {teacher_model_path}")
                self.teacher_model = nn.Module() # Placeholder, replace with actual model loading
                # Example: self.teacher_model = models.mobilenet_v2(weights=None)
                # ... load state dict ...
            else:
                logger.info("No teacher model checkpoint found or specified. Initializing new teacher model with ImageNet weights.")
                self.teacher_model = nn.Module() # Placeholder, replace with actual model initialization
                # Example: self.teacher_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            self.teacher_model.to(DEVICE).eval()
            for param in self.teacher_model.parameters(): param.requires_grad = False
            logger.info("Teacher model initialized for Knowledge Distillation.")
            # Dummy DistillationLoss if not imported
            class DistillationLoss(nn.Module):
                def __init__(self, temperature, alpha):
                    super().__init__()
                    self.temperature = temperature
                    self.alpha = alpha
                    self.kl_div = nn.KLDivLoss(reduction='batchmean')
                    self.ce = nn.CrossEntropyLoss()
                def forward(self, student_logits, teacher_logits, labels):
                    T = self.temperature
                    student_logprob = F.log_softmax(student_logits / T, dim=1)
                    teacher_prob    = F.softmax(teacher_logits / T, dim=1)
                    kd_loss = self.kl_div(student_logprob, teacher_prob) * (T * T)
                    ce_loss = self.ce(student_logits, labels)
                    return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

            self.distillation_criterion = DistillationLoss(temperature=config['training']['distillation_temperature'], alpha=config['training']['distillation_alpha'])
        # --- NEW: Integrated Perceptual Modules ---
        self.object_detector = RealObjectDetector(model_path=config['model']['object_detector_model_path'])
        self.attribute_classifier = AttributeClassifier(
            num_channels=NUM_CHANNELS, image_size=config['data']['image_size'], backbone_name=config['model']['attribute_backbone'],
            num_fill_types=len(ATTRIBUTE_FILL_MAP), num_color_types=len(ATTRIBUTE_COLOR_MAP), num_size_types=len(ATTRIBUTE_SIZE_MAP),
            num_orientation_types=len(ATTRIBUTE_ORIENTATION_MAP), num_shape_types=len(ATTRIBUTE_SHAPE_MAP), num_texture_types=len(ATTRIBUTE_TEXTURE_MAP)
        )
        if config['model']['attribute_classifier_model_path']:
            try:
                self.attribute_classifier.load_state_dict(torch.load(config['model']['attribute_classifier_model_path'], map_location=DEVICE))
                logger.info(f"AttributeClassifier loaded from {config['model']['attribute_classifier_model_path']}")
                for param in self.attribute_classifier.parameters(): param.requires_grad = False
                self.attribute_classifier.eval()
                logger.info("AttributeClassifier parameters frozen (loaded pre-trained).")
            except Exception as e:
                logger.warning(f"Could not load pre-trained AttributeClassifier from {config['model']['attribute_classifier_model_path']}: {e}. Starting with randomly initialized weights and making them trainable.")
                for param in self.attribute_classifier.parameters(): param.requires_grad = True
                self.attribute_classifier.train()
        else:
            logger.info("No pre-trained AttributeClassifier path specified. Starting with randomly initialized and trainable weights.")
            for param in self.attribute_classifier.parameters(): param.requires_grad = True
            self.attribute_classifier.train()
        self.attribute_classifier.to(DEVICE)
        dummy_attr_classifier = AttributeClassifier(num_channels=NUM_CHANNELS, image_size=config['data']['image_size'], backbone_name=config['model']['attribute_backbone'])
        dummy_input = torch.randn(1, NUM_CHANNELS, config['data']['image_size'][0], config['data']['image_size'][1])
        _, _, _, _, _, _, dummy_features = dummy_attr_classifier(dummy_input)
        node_feature_dim = dummy_features.shape[-1]
        hidden_dim_gnn = node_feature_dim * 2
        num_relations = len(RELATION_MAP)
        
        # Pass dynamic GNN depth parameters to RelationGNN
        self.relation_gnn = RelationGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=self.edge_feature_dim,
            num_relation_types=num_relations,
            gnn_depth=config['model']['gnn_depth'] # Initial depth
        )
        if config['model']['relation_gnn_model_path']:
            try:
                self.relation_gnn.load_state_dict(torch.load(config['model']['relation_gnn_model_path'], map_location=DEVICE))
                logger.info(f"RelationGNN loaded from {config['model']['relation_gnn_model_path']}")
                for param in self.relation_gnn.parameters(): param.requires_grad = False
                self.relation_gnn.eval()
                logger.info("RelationGNN parameters frozen (loaded pre-trained).")
            except Exception as e:
                logger.warning(f"Could not load pre-trained RelationGNN from {config['model']['relation_gnn_model_path']}: {e}. Starting with randomly initialized weights and making them trainable.")
                for param in self.relation_gnn.parameters(): param.requires_grad = True
                self.relation_gnn.train()
        else:
            logger.info("No pre-trained RelationGNN path specified. Starting with randomly initialized and trainable weights.")
            for param in self.relation_gnn.parameters(): param.requires_grad = True
            self.relation_gnn.train()
        self.relation_gnn.to(DEVICE)
        
        final_classifier_input_dim = 2 * hidden_dim_gnn
        self.final_classifier_head = nn.Sequential(
            nn.Linear(final_classifier_input_dim, 256), nn.ReLU(),
            nn.Dropout(config['model']['initial_dropout_rate']),
            nn.Linear(256, self.num_classes)
        )
        logger.info("Final classifier head for Bongard problem initialized.")
        self = self.to(memory_format=torch.channels_last)
        logger.info("Model converted to channels_last memory format.")
        self.use_qat = config['training']['use_qat']
        self.qat_start_epoch = config['training']['qat_start_epoch']
        if self.use_qat:
            self.final_classifier_head.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare_qat(self.final_classifier_head, inplace=True)
            logger.info("Final classifier head prepared for Quantization-Aware Training (QAT).")
        if self.use_knowledge_distillation: self.criterion = self.distillation_criterion
        else:
            # Dummy LabelSmoothingCrossEntropy if not imported
            class LabelSmoothingCrossEntropy(nn.Module):
                def __init__(self, epsilon: float = 0.1):
                    super().__init__()
                    self.epsilon = epsilon
                    self.log_softmax = nn.LogSoftmax(dim=1)
                def forward(self, logits, target):
                    num_classes = logits.size(1)
                    log_preds = self.log_softmax(logits)
                    with torch.no_grad():
                        one_hot = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
                        one_hot = one_hot * (1 - self.epsilon) + self.epsilon / num_classes
                    loss = - (one_hot * log_preds).sum(dim=1).mean()
                    return loss

            if config['training'].get('label_smoothing_epsilon', 0.0) > 0: self.criterion = LabelSmoothingCrossEntropy(epsilon=config['training']['label_smoothing_epsilon'])
            else: self.criterion = nn.CrossEntropyLoss()
        trainable_params = []
        trainable_params.extend(self.final_classifier_head.parameters())
        if any(p.requires_grad for p in self.attribute_classifier.parameters()): trainable_params.extend(self.attribute_classifier.parameters())
        if any(p.requires_grad for p in self.relation_gnn.parameters()): trainable_params.extend(self.relation_gnn.parameters())
        
        # Optimizer and Scheduler setup (simplified for brevity, assuming external imports for SAM/Sophia)
        if self.optimizer_name == 'AdamW': self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SGD': self.optimizer = torch.optim.SGD(trainable_params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SAM': # Assuming SAM class is available globally
            # from sam import SAM # Local import for SAM
            class SAM: # Dummy SAM for demonstration
                def __init__(self, params, base_optimizer, rho, adaptive):
                    self.optimizer = base_optimizer
                def step(self, closure):
                    closure()
            base_optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.optimizer = SAM(trainable_params, base_optimizer, rho=0.05, adaptive=True)
        elif self.optimizer_name == 'Sophia': # Assuming SophiaG class is available globally
            # from sophia import SophiaG # Local import for SophiaG
            class SophiaG: # Dummy SophiaG for demonstration
                def __init__(self, params, lr, weight_decay):
                    self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
                def step(self, closure):
                    closure()
            self.optimizer = SophiaG(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            logger.warning(f"Optimizer {self.optimizer_name} not supported or required library not found. Falling back to AdamW.")
            self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2.**16, enabled=self.use_amp)
        logger.info(f"Automatic Mixed Precision (AMP) enabled using {self.amp_dtype}.")
        if self.use_swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self)
            self.swa_scheduler = None
            logger.info("SWA AveragedModel initialized.")
        
        # Kornia normalization (if HAS_KORNIA is True and Kornia is imported)
        self.normalize_kornia = None
        # if HAS_KORNIA:
        #     import kornia.augmentation as K # Local import for Kornia
        #     kornia_mean = torch.tensor([m / 255.0 for m in IMAGENET_MEAN[:NUM_CHANNELS]], device=DEVICE).view(1, -1, 1, 1)
        #     kornia_std = torch.tensor([s / 255.0 for s in IMAGENET_STD[:NUM_CHANNELS]], device=DEVICE).view(1, -1, 1, 1)
        #     self.normalize_kornia = K.Normalize(mean=kornia_mean, std=kornia_std).to(DEVICE)
        
        os.makedirs(self.save_path, exist_ok=True)
        self.gpu_memory_history = []
        logger.info("GPU memory profiling enabled.")
        class TemperatureScaler(nn.Module): # Dummy TemperatureScaler
            def __init__(self):
                super().__init__()
                self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initialize temperature
            def forward(self, logits):
                return logits / self.temperature
            def calibrate(self, evaluate_fn, val_loader):
                # Dummy calibration: just sets temperature to a fixed value
                self.temperature.data = torch.ones(1) * 1.0 # Example fixed value
                logger.warning("Using dummy calibration. Implement actual calibration logic.")


        self.temperature_scaler = TemperatureScaler().to(DEVICE) if config['training']['calibrate_model'] else None

        # Batch timing
        self.batch_times = collections.deque(maxlen=50) # sliding window for throughput

    def _process_batch_for_symbolic_outputs(self, images_tensor_batch):
        batch_size = images_tensor_batch.size(0)
        pil_images_batch = []
        # Denormalize images for PIL conversion
        for i in range(batch_size):
            img_chw_normalized = images_tensor_batch[i].cpu().numpy()
            IMAGENET_MEAN_PLOT = np.array(IMAGENET_MEAN[:NUM_CHANNELS]) / 255.0
            IMAGENET_STD_PLOT = np.array(IMAGENET_STD[:NUM_CHANNELS]) / 255.0
            img_denormalized = (img_chw_normalized.transpose(1, 2, 0) * IMAGENET_STD_PLOT) + IMAGENET_MEAN_PLOT
            img_denormalized = np.clip(img_denormalized, 0, 1)
            img_np = (img_denormalized * 255).astype(np.uint8)
            if NUM_CHANNELS == 1:
                pil_images_batch.append(Image.fromarray(img_np.squeeze(2), 'L'))
            else:
                pil_images_batch.append(Image.fromarray(img_np, 'RGB'))

        all_node_features_batch = []
        all_edge_indices_batch = []
        all_edge_attrs_batch = []
        batch_indices_for_pooling = [] # This is the 'batch' tensor for PyG
        num_nodes_so_far = 0 # This tracks the cumulative number of nodes across images in the batch
        all_symbolic_outputs = []

        for i, pil_image in enumerate(pil_images_batch):
            current_image_symbolic_output = {"image_idx": i, "objects": [], "attributes": [], "relations": []}
            thresh = self.config['model'].get('detection_confidence_threshold', 0.3)
            detected_objects = self.object_detector.detect_objects(pil_image, overall_confidence=thresh)
            logger.debug(f"Detected {len(detected_objects)} objects in image {i}")

            node_features_list_current_image = []
            
            # --- Process detected objects (including fallback) ---
            if not detected_objects:
                logger.warning(f"No objects detected in image {i} after all attempts. Appending a single dummy node feature.")
                dummy_node_feature_dim = self.attribute_classifier.fill_head.in_features
                node_features_list_current_image.append(torch.zeros(dummy_node_feature_dim, device=DEVICE))
                
                all_edge_indices_batch.append(torch.empty((2, 0), dtype=torch.long, device=DEVICE))
                all_edge_attrs_batch.append(torch.empty((0, self.edge_feature_dim), dtype=torch.float32, device=DEVICE))
                
                all_symbolic_outputs.append(current_image_symbolic_output)
                num_nodes_so_far += 1 # Increment for the dummy node
                
                # After processing all objects for the current image, if any nodes were added,
                # append their batch indices. This handles the case where `detected_objects` was empty.
                node_features_current_image = torch.stack(node_features_list_current_image)
                all_node_features_batch.append(node_features_current_image)
                batch_indices_for_pooling.extend([i] * node_features_current_image.size(0))
                
                continue # Move to the next image in the batch

            for obj_idx, obj in enumerate(detected_objects):
                x1, y1, x2, y2 = obj['bbox']
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(pil_image.width, x2); y2 = min(pil_image.height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox for object {obj['id']}: {obj['bbox']}. Skipping attribute classification for this object.")
                    dummy_node_feature_dim = self.attribute_classifier.fill_head.in_features
                    node_features_list_current_image.append(torch.zeros(dummy_node_feature_dim, device=DEVICE))
                    
                    current_image_symbolic_output["objects"].append({"id": obj["id"], "type": obj["type"], "bbox": obj["bbox"], "confidence": obj["confidence"]})
                    current_image_symbolic_output["attributes"].extend([{"obj_id": obj["id"], "property": p, "value": "unknown", "confidence": 0.0} for p in ["fill", "color", "size", "orientation", "shape", "texture"]])
                    continue

                cropped_patch_pil = pil_image.crop((x1, y1, x2, y2))
                patch_tensor = T.Compose([
                    T.Resize(self.config['data']['image_size']),
                    T.ToTensor(),
                    T.Normalize(mean=[m / 255.0 for m in IMAGENET_MEAN[:NUM_CHANNELS]], std=[s / 255.0 for s in IMAGENET_STD[:NUM_CHANNELS]])
                ])(cropped_patch_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, features = self.attribute_classifier(patch_tensor)
                
                node_features_list_current_image.append(features.squeeze(0))
                
                # Attribute prediction and logging (unchanged)
                fill_prob, color_prob, size_prob, orientation_prob, shape_prob, texture_prob = [F.softmax(l, dim=1) for l in [fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits]]
                fill_pred, color_pred, size_pred, orientation_pred, shape_pred, texture_pred = [p.argmax(dim=1).item() for p in [fill_prob, color_prob, size_prob, orientation_prob, shape_prob, texture_prob]]
                fill_conf, color_conf, size_conf, orientation_conf, shape_conf, texture_conf = [p.max(dim=1)[0].item() for p in [fill_prob, color_prob, size_prob, orientation_prob, shape_prob, texture_prob]]

                fill_val = ATTRIBUTE_FILL_MAP.get(fill_pred, "unknown")
                color_val = ATTRIBUTE_COLOR_MAP.get(color_pred, "unknown")
                size_val = ATTRIBUTE_SIZE_MAP.get(size_pred, "unknown")
                orientation_val = ATTRIBUTE_ORIENTATION_MAP.get(orientation_pred, "unknown")
                shape_val = ATTRIBUTE_SHAPE_MAP.get(shape_pred, "unknown")
                texture_val = ATTRIBUTE_TEXTURE_MAP.get(texture_pred, "unknown")
                
                current_image_symbolic_output["objects"].append({"id": obj["id"], "type": obj["type"], "bbox": obj["bbox"], "confidence": obj["confidence"]})
                current_image_symbolic_output["attributes"].extend([
                    {"obj_id": obj["id"], "property": "fill", "value": fill_val, "confidence": fill_conf},
                    {"obj_id": obj["id"], "property": "color", "value": color_val, "confidence": color_conf},
                    {"obj_id": obj["id"], "property": "size", "value": size_val, "confidence": size_conf},
                    {"obj_id": obj["id"], "property": "orientation", "value": orientation_val, "confidence": orientation_conf},
                    {"obj_id": obj["id"], "property": "shape", "value": shape_val, "confidence": shape_conf},
                    {"obj_id": obj["id"], "property": "texture", "value": texture_val, "confidence": texture_conf},
                ])
            
            # If after processing, no valid nodes were added (e.g., all bboxes were invalid)
            if not node_features_list_current_image:
                logger.warning(f"No valid objects after bbox filtering for image {i}. Appending a single dummy node feature.")
                dummy_node_feature_dim = self.attribute_classifier.fill_head.in_features
                node_features_list_current_image.append(torch.zeros(dummy_node_feature_dim, device=DEVICE))
                
                all_edge_indices_batch.append(torch.empty((2, 0), dtype=torch.long, device=DEVICE))
                all_edge_attrs_batch.append(torch.empty((0, self.edge_feature_dim), dtype=torch.float32, device=DEVICE))
                
                all_symbolic_outputs.append(current_image_symbolic_output)
                num_nodes_so_far += 1 # Increment for the dummy node
                
                # After processing all objects for the current image, if any nodes were added,
                # append their batch indices. This handles the case where `node_features_list_current_image` became empty due to invalid bboxes.
                node_features_current_image = torch.stack(node_features_list_current_image)
                all_node_features_batch.append(node_features_current_image)
                batch_indices_for_pooling.extend([i] * node_features_current_image.size(0))

                continue # Move to the next image in the batch

            node_features_current_image = torch.stack(node_features_list_current_image)
            all_node_features_batch.append(node_features_current_image)
            
            # --- FIXED: Append batch indices once per image, after all nodes for the image are collected ---
            batch_indices_for_pooling.extend([i] * node_features_current_image.size(0)) 
            
            num_nodes_current_image = node_features_current_image.size(0) # Use the actual number of valid nodes
            
            edge_index_current_image = []
            edge_attr_list_current_image = []
            current_image_relations = []

            if num_nodes_current_image > 1:
                # Filter detected_objects to only include valid ones that made it into node_features_list_current_image
                # This assumes order is preserved; if not, you might need mapping from original obj_id to new node index
                valid_detected_objects = []
                for obj_idx, obj in enumerate(detected_objects):
                    x1, y1, x2, y2 = obj['bbox']
                    if not (x2 <= x1 or y2 <= y1):
                        valid_detected_objects.append(obj)

                bboxes_current_image = np.array([obj['bbox'] for obj in valid_detected_objects])
                centers = np.array([((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2) for bbox in bboxes_current_image])
                dist_matrix = cdist(centers, centers)
                img_w, img_h = pil_image.width, pil_image.height
                image_diagonal = np.sqrt(img_w**2 + img_h**2)
                proximity_threshold = 0.25 * image_diagonal
                
                for src_idx in range(num_nodes_current_image):
                    for dest_idx in range(num_nodes_current_image):
                        if src_idx != dest_idx and dist_matrix[src_idx, dest_idx] < proximity_threshold:
                            # Local indices for the edge within the current image's graph
                            edge_index_current_image.append([src_idx, dest_idx])
                            
                            bbox_src = valid_detected_objects[src_idx]["bbox"]
                            bbox_dest = valid_detected_objects[dest_idx]["bbox"]
                            
                            center_src_x = (bbox_src[0] + bbox_src[2]) / 2.0; center_src_y = (bbox_src[1] + bbox_src[3]) / 2.0
                            center_dest_x = (bbox_dest[0] + bbox_dest[2]) / 2.0; center_dest_y = (bbox_dest[1] + bbox_dest[3]) / 2.0
                            width_src = bbox_src[2] - bbox_src[0]; height_src = bbox_src[3] - bbox_src[1]
                            width_dest = bbox_dest[2] - bbox_dest[0]; height_dest = bbox_dest[3] - bbox_dest[1]
                            
                            rel_dx = (center_dest_x - center_src_x) / img_w
                            rel_dy = (center_dest_y - center_src_y) / img_h
                            rel_dw = (width_dest - width_src) / img_w
                            rel_dh = (height_dest - height_src) / img_h
                            angle = math.atan2(rel_dy, rel_dx)

                            # Compute IoU between bbox_src and bbox_dest
                            xA = max(bbox_src[0], bbox_dest[0])
                            yA = max(bbox_src[1], bbox_dest[1])
                            xB = min(bbox_src[2], bbox_dest[2])
                            yB = min(bbox_src[3], bbox_dest[3])
                            interArea = max(0, xB - xA) * max(0, yB - yA)
                            boxAArea = (bbox_src[2] - bbox_src[0]) * (bbox_src[3] - bbox_src[1])
                            boxBArea = (bbox_dest[2] - bbox_dest[0]) * (bbox_dest[3] - bbox_dest[1])
                            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)  # Add epsilon to avoid division by zero
                            
                            edge_attr_list_current_image.append(torch.tensor([rel_dx, rel_dy, rel_dw, rel_dh, angle, iou], dtype=torch.float32, device=DEVICE))
                
                if edge_index_current_image:
                    edge_index_current_image_tensor = torch.tensor(edge_index_current_image, dtype=torch.long, device=DEVICE).t().contiguous()
                    edge_attr_current_image_tensor = torch.stack(edge_attr_list_current_image)
                    
                    with torch.no_grad():
                        edge_logits = self.relation_gnn.classify_edges(node_features_current_image, edge_index_current_image_tensor, edge_attr_current_image_tensor)
                        edge_preds = torch.argmax(edge_logits, dim=1)
                        edge_probs = F.softmax(edge_logits, dim=1)
                    
                    # Edge Pruning by Confidence
                    prune_thresh = self.config['debug'].get('edge_prune_threshold', 0.2)
                    kept_edges = []
                    kept_attrs = []
                    kept_relations_info = [] # To store relations that pass pruning
                    for k, (src, dst) in enumerate(edge_index_current_image):
                        conf = edge_probs[k].max().item()
                        if conf >= prune_thresh:
                            kept_edges.append([src, dst])
                            kept_attrs.append(edge_attr_list_current_image[k])
                            
                            src_node_obj_id = valid_detected_objects[src]["id"]
                            dest_node_obj_id = valid_detected_objects[dst]["id"]
                            relation_val = RELATION_MAP.get(edge_preds[k].item(), "unknown_relation")
                            kept_relations_info.append({"source_obj_id": src_node_obj_id, "target_obj_id": dest_node_obj_id, "relation": relation_val, "confidence": conf})

                    if kept_edges:
                        edge_index_current_image_tensor = torch.tensor(kept_edges, dtype=torch.long, device=DEVICE).t().contiguous()
                        edge_attr_current_image_tensor = torch.stack(kept_attrs)
                        current_image_relations.extend(kept_relations_info) # Add only kept relations
                    else:
                        edge_index_current_image_tensor = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
                        edge_attr_current_image_tensor = torch.empty((0, self.edge_feature_dim), dtype=torch.float32, device=DEVICE)
                        logger.debug(f"All edges pruned for image {i} below threshold {prune_thresh}.")

                    all_edge_indices_batch.append(edge_index_current_image_tensor)
                    all_edge_attrs_batch.append(edge_attr_current_image_tensor)
                    
                else:
                    all_edge_indices_batch.append(torch.empty((2, 0), dtype=torch.long, device=DEVICE))
                    all_edge_attrs_batch.append(torch.empty((0, self.edge_feature_dim), dtype=torch.float32, device=DEVICE))
            else:
                all_edge_indices_batch.append(torch.empty((2, 0), dtype=torch.long, device=DEVICE))
                all_edge_attrs_batch.append(torch.empty((0, self.edge_feature_dim), dtype=torch.float32, device=DEVICE))
            
            current_image_symbolic_output["relations"] = current_image_relations
            all_symbolic_outputs.append(current_image_symbolic_output)
            num_nodes_so_far += num_nodes_current_image # Update cumulative node count

        # --- Concatenate all features and edges for the batch ---
        if not all_node_features_batch:
            logger.warning("No valid node features generated for the entire batch. Returning dummy logits.")
            return torch.zeros(batch_size, self.num_classes, device=DEVICE), []

        x_batch = torch.cat(all_node_features_batch, dim=0)
        
        # Correctly offset edge indices for the batch
        if all_edge_indices_batch and any(e.numel() > 0 for e in all_edge_indices_batch):
            offsets = torch.cumsum(torch.tensor([nf.size(0) for nf in all_node_features_batch], dtype=torch.long, device=DEVICE), dim=0)
            offsets = torch.cat([torch.tensor([0], dtype=torch.long, device=DEVICE), offsets[:-1]])
            
            processed_edge_indices = []
            for img_idx, (e_idx_img, node_offset) in enumerate(zip(all_edge_indices_batch, offsets)):
                if e_idx_img.numel() > 0:
                    processed_edge_indices.append(e_idx_img + node_offset.item()) 
            
            if processed_edge_indices:
                edge_index_batch = torch.cat(processed_edge_indices, dim=1)
            else:
                edge_index_batch = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
        else:
            edge_index_batch = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

        if all_edge_attrs_batch and any(e.numel() > 0 for e in all_edge_attrs_batch):
            edge_attr_batch = torch.cat(all_edge_attrs_batch, dim=0)
        else:
            edge_attr_batch = torch.empty((0, self.edge_feature_dim), dtype=torch.float32, device=DEVICE)

        # Ensure batch_tensor_for_pooling is correctly sized and contiguous
        batch_tensor_for_pooling = torch.tensor(batch_indices_for_pooling, dtype=torch.long, device=DEVICE)
        
        # --- OPTIONAL SANITY CHECK ---
        logger.debug(f"x_batch size: {x_batch.size(0)}, batch_tensor size: {batch_tensor_for_pooling.size(0)}")
        assert x_batch.size(0) == batch_tensor_for_pooling.size(0), \
            f"Mismatch: x_batch has {x_batch.size(0)} nodes, but batch_tensor has {batch_tensor_for_pooling.size(0)} entries"

        if batch_tensor_for_pooling.numel() > 0: # Only check if tensor is not empty
            if batch_size > 0: # Avoid division by zero if batch_size is 0 for some reason
                assert batch_tensor_for_pooling.max().item() < batch_size, \
                    f"Invalid batch index: max={batch_tensor_for_pooling.max().item()}, batch_size={batch_size}. This means a node is assigned to a graph index out of bounds for the batch."
        
        # Check if x_batch is empty, which can happen if all images had no valid objects
        if x_batch.numel() == 0:
            logger.warning("x_batch is empty after processing. Returning dummy logits.")
            return torch.zeros(batch_size, self.num_classes, device=DEVICE), all_symbolic_outputs

        graph_embeddings_per_image, final_node_features = self.relation_gnn(x_batch, edge_index_batch, batch_tensor_for_pooling)
        logits = self.final_classifier_head(graph_embeddings_per_image)
        return logits, all_symbolic_outputs

    def forward(self, images_tensor_batch):
        logits, symbolic_outputs = self._process_batch_for_symbolic_outputs(images_tensor_batch)
        return logits, symbolic_outputs

    def train_model(self, train_loader, val_loader, train_sampler, writer):
        """
        Trains the PerceptionModule using the provided data loaders and sampler.
        Args:
            train_loader (None): Placeholder, as DALI pipeline is managed internally.
            val_loader (DALIGenericIterator): Validation data iterator.
            train_sampler (CurriculumSampler): Sampler for curriculum learning.
            writer (SummaryWriter or wandb): Logger for metrics.
        """
        self.train()
        self.writer = writer
        self.num_train_batches = len(train_sampler) # Number of batches per epoch
        
        early_stopper = EarlyStopping(
            patience=self.config['training']['early_stopping_patience'],
            delta=self.config['training']['early_stopping_min_delta'],
            path=os.path.join(self.save_path, self.config['training']['model_checkpoint_name']),
            monitor_metric=self.config['training']['early_stopping_monitor_metric'],
            verbose=True
        )
        self.early_stopper = early_stopper # Assign to self for potential access later
        for epoch in range(self.epochs):
            train_sampler.set_epoch(epoch)
            # Set current_epoch in RelationGNN for dynamic depth
            self.relation_gnn.current_epoch = epoch 

            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            epoch_loss, correct, total = 0.0, 0, 0
            current_image_size = train_sampler.get_current_image_size()
            logger.info(f"  Epoch {epoch+1}: Progressive Resizing to {current_image_size[0]}x{current_image_size[1]}")
            train_pipeline = DaliPipeline(
                batch_size=self.config['model']['batch_size'],
                num_threads=self.config['data']['dataloader_num_workers'],
                device_id=0 if str(DEVICE) == "cuda" else -1,
                seed=epoch,
                image_size=current_image_size,
                is_train=True,
                sampler=train_sampler,
                num_channels=NUM_CHANNELS,
                feature_consistency_alpha=self.feature_consistency_alpha,
                logger=logger,
                imagenet_mean=IMAGENET_MEAN,
                imagenet_std=IMAGENET_STD,
                initial_image_size=self.config['data']['initial_image_size']
            )
            train_pipeline.build()
            
            total_train_samples = len(train_sampler.image_paths)
            num_train_batches_dali = math.ceil(total_train_samples / self.config['model']['batch_size'])
            dali_train_iterator_size = int(num_train_batches_dali * self.config['model']['batch_size'])
            train_iterator = DALIGenericIterator(
                pipelines=[train_pipeline],
                output_map=["data", "images_2", "label", "affine_1", "affine_2"],
                size=dali_train_iterator_size,
                last_batch_policy=LastBatchPolicy.FILL,
                auto_reset=True
            )
            logger.info(f"DALI Train Iterator initialized for epoch {epoch+1} with {len(train_sampler)} batches.")
            if self.scheduler_name == 'OneCycleLR' and self.scheduler is None:
                total_steps = self.epochs * self.num_train_batches
                max_lr = self.config['model']['max_learning_rate']
                # Dummy OneCycleLR if not imported
                class OneCycleLR:
                    def __init__(self, optimizer, max_lr, total_steps, pct_start, anneal_strategy, cycle_momentum, base_momentum, max_momentum, div_factor, final_div_factor):
                        self.optimizer = optimizer
                        self.max_lr = max_lr
                        self.total_steps = total_steps
                        self.current_step = 0
                        logger.warning("Using dummy OneCycleLR scheduler. Please import torch.optim.lr_scheduler.OneCycleLR for full functionality.")
                    def step(self):
                        # Simple linear decay for dummy
                        self.current_step += 1
                        progress = self.current_step / self.total_steps
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.max_lr * (1 - progress)
                self.scheduler = OneCycleLR(
                    self.optimizer, max_lr=max_lr, total_steps=total_steps,
                    pct_start=self.config['training']['onecycle_pct_start'],
                    anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85,
                    max_momentum=0.95, div_factor=25.0, final_div_factor=1e4
                )
                logger.info(f"OneCycleLR scheduler initialized with total_steps={total_steps}.")
            profiler = None
            if self.config['training']['enable_profiler'] and epoch == 0 and self.config['training']['profile_performance']:
                # Dummy profiler if not available
                try:
                    import torch.profiler
                    profiler = torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/profiler_log'),
                        with_stack=True
                    )
                    profiler.__enter__()
                except ImportError:
                    logger.warning("torch.profiler not available. Profiling disabled.")
                    profiler = None
                logger.info("PyTorch Profiler initialized for first 5 batches.")

            for batch_idx, dali_output in enumerate(tqdm(train_iterator, desc=f"Training Epoch {epoch+1}")):
                # Batch Timing: Start
                batch_start_time = time.time()

                if profiler:
                    with profiler:
                        batch_loss, batch_correct, batch_total = self._train_step(dali_output, batch_idx, epoch, self.writer, profiler)
                else:
                    batch_loss, batch_correct, batch_total = self._train_step(dali_output, batch_idx, epoch, self.writer)
                
                epoch_loss += batch_loss
                correct += batch_correct
                total += batch_total

                # Batch Timing: End and Log Throughput
                batch_time = time.time() - batch_start_time
                self.batch_times.append(batch_time)
                if batch_idx % self.log_interval == 0 and self.batch_times:
                    avg_time = sum(self.batch_times) / len(self.batch_times)
                    throughput = self.config['model']['batch_size'] / avg_time
                    logger.info(f" Throughput: {throughput:.1f} samples/sec (avg over {len(self.batch_times)} batches)")
                    if self.writer:
                        if hasattr(self.writer, 'log') and callable(self.writer.log): # Likely WandB
                            self.writer.log({"Train/Throughput_samples_per_sec": throughput, "global_step": epoch * self.num_train_batches + batch_idx})
                        else: # Likely TensorBoard
                            self.writer.add_scalar("Train/Throughput_samples_per_sec", throughput, epoch * self.num_train_batches + batch_idx)

            if profiler:
                profiler.__exit__(None, None, None)
                logger.info("PyTorch Profiler finished.")
            avg_loss = epoch_loss / total
            accuracy = correct / total
            logger.info(f"Epoch {epoch+1} Summary — Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
            if (epoch + 1) % self.eval_interval == 0:
                val_metrics = self.evaluate_model(val_loader)
                val_loss, val_acc = val_metrics[0], val_metrics[1]
                logger.info(f"  Epoch {epoch+1} Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                if self.writer:
                    # Dummy wandb if not available
                    class wandb:
                        @staticmethod
                        def Image(fig):
                            return fig
                        @staticmethod
                        def log(data, step):
                            pass
                    if hasattr(self.writer, 'log') and callable(self.writer.log): # Likely WandB
                        self.writer.log({
                            "Epoch/Train_Loss": avg_loss, "Epoch/Train_Accuracy": accuracy,
                            "Epoch/Val_Loss": val_loss, "Epoch/Val_Accuracy": val_acc,
                            "Epoch/Learning_Rate": self.optimizer.param_groups[0]['lr'], "epoch": epoch + 1
                        })
                    else:
                        self.writer.add_scalars('Epoch_Metrics/Loss', {'Train': avg_loss, 'Validation': val_loss}, epoch + 1)
                        self.writer.add_scalars('Epoch_Metrics/Accuracy', {'Train': accuracy, 'Validation': val_acc}, epoch + 1)
                        self.writer.add_scalar('Epoch_Metrics/Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch + 1)
                        self.writer.add_scalar('Epoch_Metrics/Epoch', epoch + 1, epoch + 1)
                early_stopper({'val_loss': val_loss, 'val_accuracy': val_acc}, self, self.optimizer, epoch + 1)
                if early_stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break
            
            if self.use_swa and (epoch + 1) >= self.swa_start_epoch:
                self.swa_model.update_parameters(self)
                # Dummy SWALR if not imported
                try:
                    from torch.optim.swa_utils import SWALR
                except ImportError:
                    class SWALR:
                        def __init__(self, optimizer, swa_lr):
                            self.optimizer = optimizer
                            self.swa_lr = swa_lr
                            logger.warning("Using dummy SWALR. Please import torch.optim.swa_utils.SWALR for full functionality.")
                        def step(self):
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = self.swa_lr
                if self.swa_scheduler is None:
                    self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.config['training']['swa_lr']) # Corrected swa_lr
                self.swa_scheduler.step()
                logger.info(f"SWA model updated at epoch {epoch+1}.")
            if self.use_qat and (epoch + 1) >= self.qat_start_epoch:
                logger.info(f"Applying QAT observe_and_quantize at epoch {epoch+1}.")
                # This is where you'd typically call torch.quantization.fuse_modules and then prepare_qat
                # For now, just a placeholder.
                pass
            
            # Symbolic Output Export to JSON
            if (epoch + 1) % self.config['training'].get('save_symbolic_outputs_interval_epochs', 5) == 0:
                logger.info(f"Saving sample symbolic outputs for epoch {epoch+1}...")
                # Get a sample of symbolic outputs from the validation set
                all_symbolic_outputs_sample = self._get_sample_symbolic_outputs_from_val_loader(
                    val_loader, num_samples=self.config['training'].get('num_symbolic_dump', 5)
                )
                json_path = os.path.join(self.save_path, f"symbolic_epoch_{epoch+1}.json")
                with open(json_path, 'w') as f:
                    json.dump(all_symbolic_outputs_sample, f, indent=2)
                logger.info(f"Symbolic outputs saved to {json_path}")


        logger.info("Training complete.")
        if self.use_swa:
            if self.use_qat: logger.info("Disabling QAT for SWA model before `swap_parameters`.")
            # Dummy swap_parameters_with_avg if not imported
            try:
                from torch.optim.swa_utils import swap_parameters_with_avg
            except ImportError:
                def swap_parameters_with_avg(model, swa_model):
                    logger.warning("Using dummy swap_parameters_with_avg. Please import torch.optim.swa_utils.swap_parameters_with_avg for full functionality.")
                    model.load_state_dict(swa_model.module.state_dict()) # Simple copy for dummy
            swap_parameters_with_avg(self, self.swa_model)
            logger.info("SWA parameters swapped with averaged model.")
        if self.use_qat:
            logger.info("Finalizing QAT conversion of the model.")
            torch.quantization.convert(self.final_classifier_head, inplace=True)
            logger.info("Model converted to quantized version.")
        
        if self.config['training']['calibrate_model'] and self.temperature_scaler:
            logger.info("Calibrating model using Temperature Scaling.")
            # Create a temporary DALI iterator for calibration data
            calib_sampler = val_loader.sampler # Use the existing validation sampler
            calib_pipeline = DaliPipeline(
                batch_size=self.config['model']['batch_size'],
                num_threads=self.config['data']['dataloader_num_workers'],
                device_id=0 if str(DEVICE) == "cuda" else -1,
                seed=42,
                image_size=list(CONFIG['data']['image_size']), # Use final image size
                is_train=False, # No augmentations
                sampler=calib_sampler,
                num_channels=NUM_CHANNELS,
                feature_consistency_alpha=0.0,
                logger=logger,
                imagenet_mean=IMAGENET_MEAN,
                imagenet_std=IMAGENET_STD,
                initial_image_size=CONFIG['data']['initial_image_size']
            )
            calib_pipeline.build()
            total_calib_samples = len(calib_sampler.image_paths)
            num_calib_batches_dali = math.ceil(total_calib_samples / self.config['model']['batch_size'])
            dali_calib_iterator_size = int(num_calib_batches_dali * self.config['model']['batch_size'])
            calib_iterator = DALIGenericIterator(
                pipelines=[calib_pipeline],
                output_map=["data", "images_2", "label", "affine_1", "affine_2"],
                size=dali_calib_iterator_size,
                last_batch_policy=LastBatchPolicy.PARTIAL,
                auto_reset=True
            )
            self.temperature_scaler.calibrate(self.evaluate_model, calib_iterator) # Pass the DALI iterator
            logger.info(f"Optimal Temperature: {self.temperature_scaler.temperature.item()}")
            if self.writer:
                if hasattr(self.writer, 'log') and callable(self.writer.log): # Likely WandB
                    self.writer.log({"Model_Calibration/Optimal_Temperature": self.temperature_scaler.temperature.item()})
                else: self.writer.add_scalar("Model_Calibration/Optimal_Temperature", self.temperature_scaler.temperature.item(), 0)
            
            # Clean up calibration DALI resources
            del calib_iterator
            del calib_pipeline
            torch.cuda.empty_cache()

        if 'train_pipeline' in locals() and train_pipeline:
            del train_pipeline
            del train_iterator
            torch.cuda.empty_cache()


    def _train_step(self, dali_output, batch_idx, epoch, writer, profiler=None):
        start_time = time.time()
        if str(DEVICE) == 'cuda':
            self.gpu_memory_history.append(torch.cuda.memory_allocated(DEVICE) / (1024**2))
            if batch_idx % self.log_interval == 0: logger.info(f"  GPU Memory Allocated: {self.gpu_memory_history[-1]:.2f} MB")

        data = dali_output[0]
        images = data["data"]
        labels = data["label"].squeeze(1)
        images = images.to(device=DEVICE, memory_format=torch.channels_last)
        self.optimizer.zero_grad()
        
        y_a, y_b, lam = labels, labels, 1.0
        mix_type = "None"
        if self.config['training']['mixup_alpha'] > 0 and self.config['training']['cutmix_alpha'] > 0:
            if random.random() < self.config['training']['mixup_cutmix_ratio']:
                images, y_a, y_b, lam = mixup_data(images, labels, self.config['training']['mixup_alpha'], DEVICE)
                mix_type = "MixUp"
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels, self.config['training']['cutmix_alpha'], DEVICE)
                mix_type = "CutMix"
        elif self.config['training']['mixup_alpha'] > 0:
            images, y_a, y_b, lam = mixup_data(images, labels, self.config['training']['mixup_alpha'], DEVICE)
            mix_type = "MixUp"
        elif self.config['training']['cutmix_alpha'] > 0:
            images, y_a, y_b, lam = cutmix_data(images, labels, self.config['training']['cutmix_alpha'], DEVICE)
            mix_type = "CutMix"

        # Fix: Removed device_type='cuda' from autocast
        with autocast(dtype=self.amp_dtype):
            student_outputs, symbolic_outputs = self(images)
            if self.use_knowledge_distillation and self.teacher_model:
                with torch.no_grad(): teacher_outputs, _ = self.teacher_model(images)
                loss = self.distillation_criterion(student_outputs, teacher_outputs, labels)
            else:
                if mix_type in ["MixUp", "CutMix"]: loss = mixup_criterion(self.criterion, student_outputs, y_a, y_b, lam)
                else: loss = self.criterion(student_outputs, labels)
        
            if self.feature_consistency_alpha > 0 and "images_2" in data:
                images_2 = data["images_2"].to(device=DEVICE, memory_format=torch.channels_last)
                outputs_2, _ = self(images_2)
                feature_loss = F.mse_loss(student_outputs, outputs_2)
                loss = loss + self.feature_consistency_alpha * feature_loss
                if batch_idx % self.log_interval == 0 and writer:
                    if HAS_WANDB and CONFIG['training']['use_wandb']: writer.log({"Train/Feature_Consistency_Loss": feature_loss.item(), "batch_idx": batch_idx, "epoch": epoch})
                    else: writer.add_scalar("Train/Feature_Consistency_Loss", feature_loss.item(), epoch * self.num_train_batches + batch_idx)

        loss = loss / self.config['training']['gradient_accumulation_steps']
        
        if self.optimizer_name == 'SAM' and HAS_SAM:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler and self.scheduler_name == 'OneCycleLR': self.scheduler.step()
        else:
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler and self.scheduler_name == 'OneCycleLR': self.scheduler.step()
        
        _, predicted = torch.max(student_outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        
        batch_time = time.time() - start_time
        if batch_idx % self.log_interval == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            global_step = epoch * self.num_train_batches + batch_idx
            logger.info(f"  Batch {batch_idx}/{self.num_train_batches} - Loss: {loss.item() * self.config['training']['gradient_accumulation_steps']:.4f}, Accuracy: {correct/total:.4f}, LR: {current_lr:.6f}, Batch Time: {batch_time:.4f}s")
            if writer:
                if HAS_WANDB and CONFIG['training']['use_wandb']:
                    writer.log({"Train/Batch_Loss": loss.item() * self.config['training']['gradient_accumulation_steps'], "Train/Batch_Accuracy": correct/total, "Train/Learning_Rate": current_lr, "Train/Batch_Time_s": batch_time, "global_step": global_step})
                else:
                    writer.add_scalar("Train/Batch_Loss", loss.item() * self.config['training']['gradient_accumulation_steps'], global_step)
                    writer.add_scalar("Train/Batch_Accuracy", correct/total, global_step)
                    writer.add_scalar("Train/Learning_Rate", current_lr, global_step)
                    writer.add_scalar("Train/Batch_Time_s", batch_time, global_step)
        
        if profiler: profiler.step()
        return loss.item() * total * self.config['training']['gradient_accumulation_steps'], correct, total

    def evaluate_model(self, data_loader):
        self.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        data_loader.reset()
        
        with torch.no_grad():
            for i, dali_output in enumerate(tqdm(data_loader, desc="Evaluating")):
                data = dali_output[0]
                images = data["data"]
                labels = data["label"].squeeze(1)
                images = images.to(device=DEVICE, memory_format=torch.channels_last)
                labels = labels.to(DEVICE)

                # Fix: Removed device_type='cuda'
                with autocast(dtype=self.amp_dtype):
                    outputs, _ = self(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        num_val_samples = len(data_loader.sampler.image_paths)
        avg_loss = total_loss / num_val_samples
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = np.mean(all_predictions == all_labels)
        if len(np.unique(all_labels)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary', zero_division=0)
            auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            logger.warning("Only one class present in validation labels. Precision, Recall, F1, AUC will be 0.")
            precision, recall, f1, auc_score = 0.0, 0.0, 0.0, 0.0
        
        brier_score = brier_score_loss(all_labels, all_probabilities[:, 1])
        ece_calculator = ExpectedCalibrationError(num_bins=10)
        ece = ece_calculator.compute_ece(torch.tensor(all_probabilities), torch.tensor(all_labels))
        
        if CONFIG['debug']['plot_reliability_diagram']:
            ece_calculator.plot_reliability_diagram(torch.tensor(all_probabilities), torch.tensor(all_labels), title=f"Reliability Diagram (Epoch {self.current_epoch if hasattr(self, 'current_epoch') else 0})", writer=self.writer, epoch=self.current_epoch if hasattr(self, 'current_epoch') else 0)
        
        return avg_loss, accuracy, precision, recall, f1, auc_score, brier_score, ece

    def save_model(self, path, epoch, accuracy, symbolic_outputs_sample=None):
        if self.use_swa: model_to_save = self.swa_model.module.state_dict()
        else: model_to_save = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': model_to_save, 'optimizer_state_dict': self.optimizer.state_dict(), 'val_accuracy': accuracy, 'config': self.config, 'symbolic_outputs_sample': symbolic_outputs_sample}, path)

    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=DEVICE)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            accuracy = checkpoint['val_accuracy']
            if 'config' in checkpoint:
                loaded_config = checkpoint['config']
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in self.config: self.config[key].update(value)
                    else: self.config[key] = value
                logger.info(f"Configuration merged from checkpoint: {path}")
            logger.info(f"Model loaded from {path} (Epoch: {epoch}, Accuracy: {accuracy:.4f})")
            return epoch, accuracy
        except FileNotFoundError: logger.error(f"Model checkpoint not found at {path}. Starting training from scratch."); return 0, 0.0
        except Exception as e: logger.error(f"Error loading model from {path}: {e}. Starting training from scratch."); return 0, 0.0

    def _get_sample_symbolic_outputs_from_val_loader(self, val_loader, num_samples=5):
        self.eval()
        sample_symbolic_outputs = []
        temp_val_pipeline = DaliPipeline(batch_size=self.config['model']['batch_size'], num_threads=CONFIG['data']['dataloader_num_workers'], device_id=0 if str(DEVICE) == "cuda" else -1, seed=42, image_size=list(FINAL_IMAGE_SIZE), is_train=False, sampler=val_loader.sampler, num_channels=NUM_CHANNELS, feature_consistency_alpha=0.0, logger=logger, imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, initial_image_size=self.config['data']['initial_image_size'])
        temp_val_pipeline.build()
        total_val_samples = len(val_loader.sampler.image_paths)
        num_val_batches = math.ceil(total_val_samples / self.config['model']['batch_size'])
        dali_val_iterator_size = int(num_val_batches * self.config['model']['batch_size'])
        temp_val_iterator = DALIGenericIterator(pipelines=[temp_val_pipeline], output_map=["data", "images_2", "label", "affine_1", "affine_2"], size=dali_val_iterator_size, last_batch_policy=LastBatchPolicy.FILL, auto_reset=True)
        with torch.no_grad():
            for i, dali_output in enumerate(temp_val_iterator):
                images = dali_output[0]["data"].to(device=DEVICE, memory_format=torch.channels_last)
                if len(sample_symbolic_outputs) >= num_samples: break
                _, current_batch_symbolic_outputs = self._process_batch_for_symbolic_outputs(images)
                sample_symbolic_outputs.extend(current_batch_symbolic_outputs)
        temp_val_pipeline.release_outputs() # This line should be removed according to DALI best practices
        del temp_val_iterator; del temp_val_pipeline; torch.cuda.empty_cache()
        return sample_symbolic_outputs[:num_samples]



# --- Visualization Functions ---
def visualize_raw_images(image_paths, labels, title="Raw Images", num_images=5):
    """
    Visualizes a few raw images with their labels.

    Args:
        image_paths (List[str]): List of paths to the images.
        labels (np.ndarray): Array of corresponding labels.
        title (str): Title for the plot.
        num_images (int): Number of images to display.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    axes = axes.flatten()
    for i in range(min(num_images, len(image_paths))):
        img_path = image_paths[i]
        label = labels[i]
        try:
            img = Image.open(img_path).convert('L') # Load as grayscale
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        except Exception as e:
            logger.error(f"Error loading image {img_path} for visualization: {e}")
            axes[i].set_title("Load Error")
            axes[i].axis('off')
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_training_history(history):
    """
    Plots training and validation loss and accuracy over epochs.

    Args:
        history (collections.defaultdict): A dictionary containing 'train_loss',
                                           'val_loss', 'train_acc', and 'val_acc' lists.
    """
    epochs = range(len(history['train_loss']))
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_augmented_images(sampler, image_size, num_channels, num_images=5):
    """
    Visualizes a few augmented images from a DALI pipeline using a temporary pipeline.

    Args:
        sampler (CurriculumSampler): The sampler instance to draw image paths from.
        image_size (List[int]): The [height, width] dimensions for the images.
        num_channels (int): Number of channels (1 for grayscale, 3 for RGB).
        num_images (int): Number of augmented images to display.
    """
    logger.info(f"Visualizing {num_images} augmented images from DALI pipeline...")
    
    # Create a temporary DALI pipeline for visualization (train mode to see augmentations)
    temp_pipeline = DaliPipeline(
        batch_size=num_images, # Get a small batch for visualization
        num_threads=1, # Use 1 thread for simplicity
        device_id=0 if str(DEVICE) == "cuda" else -1,
        seed=42, # Fixed seed for reproducibility
        image_size=list(image_size), # Use the passed image_size
        is_train=True, # To see augmentations
        sampler=sampler, # Use the actual sampler directly
        num_channels=num_channels,
        feature_consistency_alpha=0.0, # No second view needed for visualization
        logger=logger,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        initial_image_size=CONFIG['data']['initial_image_size']
    )
    temp_pipeline.build()
    
    # Create a temporary iterator
    temp_iterator = DALIGenericIterator(
        pipelines=[temp_pipeline],
        output_map=["data", "images_2", "label", "affine_1", "affine_2"],
        size=num_images, # Fetch exactly num_images
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True
    )

    try:
        data = next(temp_iterator)[0]
        images_tensor = data["data"].cpu().numpy()
        labels = data["label"].cpu().numpy().flatten()
        
        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
        axes = axes.flatten()
        
        for i in range(min(num_images, images_tensor.shape[0])):
            img_chw_normalized = images_tensor[i]
            
            # Denormalize for plotting: (normalized_value * std_plot) + mean_plot
            IMAGENET_MEAN_PLOT = np.array(IMAGENET_MEAN[:NUM_CHANNELS]) / 255.0
            IMAGENET_STD_PLOT = np.array(IMAGENET_STD[:NUM_CHANNELS]) / 255.0
            
            # Transpose from NCHW to HWC for plotting
            img_denormalized = (img_chw_normalized.transpose(1, 2, 0) * IMAGENET_STD_PLOT) + IMAGENET_MEAN_PLOT
            img_denormalized = np.clip(img_denormalized, 0, 1) # Clip to 0-1 range
            
            if NUM_CHANNELS == 1:
                axes[i].imshow(img_denormalized.squeeze(2), cmap='gray')
            else:
                axes[i].imshow(img_denormalized)
            axes[i].set_title(f"Label: {labels[i]}")
            axes[i].axis('off')
        
        fig.suptitle("Augmented Images from DALI Pipeline")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing augmented images: {e}")
    finally:
        # Corrected DALI resource release for iterator-based pipelines
        del temp_iterator
        del temp_pipeline
        torch.cuda.empty_cache() # Clear GPU memory


def visualize_conceptual_detection_output(image_path, symbolic_output, title="Conceptual Detection Output"):
    """
    Visualizes detected objects, their attributes, and relations on an image.
    Relations are drawn as arrows, color-coded by relation type, with attribute legends.

    Args:
        image_path (str): Path to the original image file.
        symbolic_output (dict): A dictionary containing detected objects, attributes, and relations.
        title (str): Title for the plot.
    """
    try:
        # Load the original image
        img_pil = Image.open(image_path).convert('RGB')
        img_np = np.array(img_pil)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(title)

        # Object bounding boxes and labels
        for obj in symbolic_output.get("objects", []):
            x1, y1, x2, y2 = obj['bbox']
            obj_id = obj['id']
            obj_type = obj['type']
            obj_conf = obj['confidence']

            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor='cyan', linewidth=2)
            ax.add_patch(rect)
            
            # Display object ID and type
            ax.text(x1, y1 - 10, f"ID:{obj_id} Type:{obj_type} ({obj_conf:.2f})",
                    color='cyan', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        # Attributes as text labels near objects
        attributes_by_obj = collections.defaultdict(list)
        for attr in symbolic_output.get("attributes", []):
            attributes_by_obj[attr["obj_id"]].append(f"{attr['property']}:{attr['value']} ({attr['confidence']:.2f})")
        
        for obj in symbolic_output.get("objects", []):
            obj_id = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            if obj_id in attributes_by_obj:
                attr_text = "\n".join(attributes_by_obj[obj_id])
                ax.text(x2 + 5, y1, attr_text,
                        color='lime', fontsize=7, ha='left', va='top',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        # Relations as arrows with color coding
        relation_colors = {
            "above": "red", "below": "blue", "left_of": "green", "right_of": "purple",
            "inside": "orange", "contains": "darkorange", "overlaps": "brown", "touches": "pink",
            "adjacent": "gray", "aligned_horizontally": "gold", "aligned_vertically": "silver",
            "symmetrical_to": "magenta", "connected_to": "cyan", "intersects_with": "lime",
            "parallel_to": "teal", "perpendicular_to": "navy", "part_of": "olive",
            "surrounds": "maroon", "same_type_as": "darkgreen", "different_type_from": "darkred",
            "unrelated": "lightgray" # Default for unrelated or unknown
        }
        
        # Create a legend for relation types
        legend_handles = []
        for rel_type, color in relation_colors.items():
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=rel_type))
        
        # Collect object centers for drawing arrows
        obj_centers = {}
        for obj in symbolic_output.get("objects", []):
            obj_id = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            obj_centers[obj_id] = ((x1 + x2) / 2, (y1 + y2) / 2)

        for rel_info in symbolic_output.get("relations", []):
            src_id = rel_info["source_obj_id"]
            dest_id = rel_info["target_obj_id"]
            relation_type = rel_info["relation"]
            confidence = rel_info["confidence"]

            if src_id in obj_centers and dest_id in obj_centers:
                src_center = obj_centers[src_id]
                dest_center = obj_centers[dest_id]
                
                # Draw arrow from source to destination
                dx = dest_center[0] - src_center[0]
                dy = dest_center[1] - src_center[1]
                
                # Adjust arrow starting/ending points to be slightly away from center
                arrow_length = np.sqrt(dx**2 + dy**2)
                if arrow_length > 0:
                    unit_dx = dx / arrow_length
                    unit_dy = dy / arrow_length
                    
                    # Offset from center to avoid drawing over object itself
                    offset = 15 # pixels
                    start_x = src_center[0] + unit_dx * offset
                    start_y = src_center[1] + unit_dy * offset
                    end_x = dest_center[0] - unit_dx * offset
                    end_y = dest_center[1] - unit_dy * offset
                    
                    # Reduce length for arrow head
                    plot_dx = end_x - start_x
                    plot_dy = end_y - start_y

                    ax.arrow(start_x, start_y, plot_dx, plot_dy,
                             head_width=8, head_length=10, fc=relation_colors.get(relation_type, "black"),
                             ec=relation_colors.get(relation_type, "black"), linewidth=1.5, length_includes_head=True)
                    
                    # Label relation type and confidence near the arrow midpoint
                    mid_x = (src_center[0] + dest_center[0]) / 2
                    mid_y = (src_center[1] + dest_center[1]) / 2
                    ax.text(mid_x, mid_y, f"{relation_type} ({confidence:.2f})",
                            color='white', fontsize=7, ha='center', va='center',
                            bbox=dict(facecolor=relation_colors.get(relation_type, "black"), alpha=0.7, edgecolor='none', pad=1))
        
        # Add legend for relation types
        ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, title="Relations")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing conceptual detection output for {image_path}: {e}")
        # Fallback to just showing the image if visualization fails
        try:
            img_pil = Image.open(image_path).convert('RGB')
            plt.imshow(img_pil)
            plt.title(f"Error in Visualization: {e}")
            plt.axis('off')
            plt.show()
        except Exception as img_load_e:
            logger.error(f"Could not even load image for fallback: {img_load_e}")


# Main execution block
def main_phase1():
    logger.info("--- Starting Phase 1: Advanced Perception Module (System 1) ---")
    
    # Initialize TensorBoard/WandB writer
    writer = None
    if HAS_WANDB and CONFIG['training']['use_wandb']:
        wandb.init(project="bongard_perception_phase1", config=CONFIG)
        writer = wandb
        logger.info("WandB initialized.")
    else:
        writer = SummaryWriter(log_dir='./runs/bongard_perception')
        logger.info("TensorBoard SummaryWriter initialized.")

    # Load data paths and labels
    image_paths, labels, difficulty_scores = load_bongard_data(DATA_ROOT_PATH)
    logger.info(f"Loaded {len(image_paths)} image paths and labels.")

    # Visualize raw images (optional)
    if CONFIG['debug']['visualize_data']:
        logger.info("Visualizing raw images...")
        visualize_raw_images(image_paths, labels, num_images=min(5, len(image_paths)))

    # Split data into training and validation sets
    train_paths, val_paths, train_labels, val_labels, train_difficulty, val_difficulty = train_test_split(
        image_paths, labels, difficulty_scores,
        test_size=CONFIG['data']['train_test_split_ratio'],
        random_state=42,
        stratify=labels # Ensure balanced classes in splits
    )
    logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    # Initialize Curriculum Samplers
    train_sampler = CurriculumSampler(
        image_paths=train_paths,
        labels=train_labels,
        initial_difficulty_scores=train_difficulty,
        batch_size=CONFIG['model']['batch_size'],
        annealing_epochs=CONFIG['data']['curriculum_annealing_epochs'],
        total_epochs=CONFIG['model']['epochs'],
        is_train=True
    )
    val_sampler = CurriculumSampler(
        image_paths=val_paths,
        labels=val_labels,
        initial_difficulty_scores=val_difficulty,
        batch_size=CONFIG['model']['batch_size'],
        annealing_epochs=0, # No annealing for validation
        total_epochs=1,
        is_train=False
    )
    logger.info("Curriculum Samplers initialized for training and validation.")

    # Initialize DALI Iterators for validation (train iterator is managed by PerceptionModule)
    val_pipeline = DaliPipeline(
        batch_size=CONFIG['model']['batch_size'],
        num_threads=CONFIG['data']['dataloader_num_workers'],
        device_id=0 if str(DEVICE) == "cuda" else -1,
        seed=42,
        image_size=list(FINAL_IMAGE_SIZE), # Use final image size for validation
        is_train=False, # No augmentations for validation
        sampler=val_sampler, # Pass the validation sampler
        num_channels=NUM_CHANNELS,
        feature_consistency_alpha=0.0, # Not relevant for validation
        logger=logger,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        initial_image_size=CONFIG['data']['initial_image_size']
    )
    logger.info("Building val_pipeline...")
    val_pipeline.build()
    
    total_val_samples = len(val_sampler.image_paths)
    num_val_batches_dali = math.ceil(total_val_samples / CONFIG['model']['batch_size'])
    dali_val_iterator_size = int(num_val_batches_dali * CONFIG['model']['batch_size'])

    val_dali_iterator = DALIGenericIterator(
        pipelines=[val_pipeline],
        output_map=["data", "images_2", "label", "affine_1", "affine_2"],
        size=dali_val_iterator_size,
        last_batch_policy=LastBatchPolicy.FILL,
        auto_reset=True
    )
    logger.info("DALI Validation Iterator initialized.")

    # Visualize augmented images (optional)
    if CONFIG['debug']['visualize_data']:
        logger.info("Visualizing augmented images from DALI pipeline (using train sampler)...")
        # Pass the train_sampler directly to visualize_augmented_images
        visualize_augmented_images(
            sampler=train_sampler,
            image_size=CONFIG['data']['image_size'],
            num_channels=NUM_CHANNELS,
            num_images=min(5, CONFIG['model']['batch_size'])
        )

    # Instantiate the PerceptionModule
    perception_module = PerceptionModule(CONFIG)
    
    # Apply torch.compile to the instance after creation
    if hasattr(torch, 'compile') and CONFIG['training']['use_torch_compile']:
        perception_module = torch.compile(perception_module)
        logger.info("Compiled PerceptionModule instance via torch.compile()")

    perception_module.to(DEVICE)
    # Assign the writer to the perception_module for logging within its methods
    perception_module.writer = writer
    
    # Train the model
    # Pass the initial train_sampler, PerceptionModule will manage pipelines for progressive resizing
    perception_module.train_model(None, val_dali_iterator, train_sampler, writer) # train_loader is now managed internally by PerceptionModule
    
    logger.info("DALI pipeline resources released.") # This message is now less relevant as pipelines are managed internally
    logger.info("\n--- Phase 1: Advanced Perception Module (System 1) Complete ---")

    # Visualize conceptual detection output for a sample image (optional)
    if CONFIG['debug']['visualize_perception_output'] and len(val_paths) > 0:
        logger.info("Visualizing conceptual detection output for a sample validation image...")
        sample_image_path = val_paths[0]
        # Get symbolic output for the sample image
        # Note: extract_features_and_attributes returns a list of symbolic outputs (one per image in batch)
        # We need to wrap the single image path in a list for the method.
        sample_symbolic_outputs_list = perception_module.extract_features_and_attributes([sample_image_path])
        if sample_symbolic_outputs_list:
            visualize_conceptual_detection_output(sample_image_path, sample_symbolic_outputs_list[0],
                                                  title=f"Conceptual Output for: {os.path.basename(sample_image_path)}")
        else:
            logger.warning("Could not generate symbolic output for sample image for visualization.")

    # Corrected logic for closing writer and finishing wandb
    if writer:
        if not (HAS_WANDB and CONFIG['training']['use_wandb']):
            writer.close()
    if HAS_WANDB and CONFIG['training']['use_wandb']:
        wandb.finish()

# Main execution block
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method...{e}")
    main_phase1()
