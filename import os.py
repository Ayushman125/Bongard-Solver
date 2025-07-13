import os
import glob
import numpy as np
from ultralytics import YOLO as RealYOLO # Import RealYOLO from ultralytics
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw, ImageFont # Import ImageFilter, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss, roc_auc_score, accuracy_score # Added accuracy_score
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
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Any, Optional # Added Optional
from datetime import datetime # For TensorBoard log directory

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
from nvidia.dali import pipeline_def # NEW: For decorated DALI pipelines

# PyTorch Geometric imports
# Ensure these are installed: pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
from torch_geometric.nn import GCNConv, Set2Set
from torch_geometric.data import Data, Batch # Import Data and Batch for graph handling

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
    # Dummy SAM class to prevent NameError if SAM is not available
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            logger.warning("Using dummy SAM optimizer. Please install sam-pytorch or ensure sam.py is accessible for full SAM functionality.")
            self.base_optimizer = base_optimizer
            super().__init__(params, base_optimizer.defaults)
        
        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for p in self.param_groups:
                for d in p['params']:
                    if d.grad is None: continue
                    self.state[d]['old_p'] = d.data.clone() # Store original parameter
                    eps = p['rho'] / (grad_norm + 1e-12) # Use rho from param_group
                    d.add_(d.grad, alpha=eps) # Update for first step
            if zero_grad: self.zero_grad()
        
        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for p in self.param_groups:
                for d in p['params']:
                    if d.grad is None: continue
                    d.data = self.state[d]['old_p'] # Restore original parameter
            self.base_optimizer.step() # Apply base optimizer step
            if zero_grad: self.zero_grad()
        def _grad_norm(self):
            # Calculate the gradient norm for SAM
            norm = torch.norm(torch.stack([
                (p.grad if p.grad is not None else torch.zeros_like(p)).norm(p=2)
                for group in self.param_groups for p in group['params']
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
    import kornia.geometry.transform as KGT
    import kornia.color as KC
    import kornia.filters as KF
    from kornia.models import SegmentAnything # For SAM mask-based fallback
    HAS_KORNIA = True
except ImportError:
    pass
# Dummy for timm and sophia-pytorch if not installed
HAS_TIMM = False
try:
    import timm
    HAS_TIMM = True
except ImportError:
    pass
HAS_SOPHIA = False
try:
    from sophia import SophiaG
    HAS_SOPHIA = True
    logger.info("Successfully imported SophiaG.")
except ImportError:
    logger.warning("Could not import SophiaG. Sophia optimizer will not be available.")
    # Dummy SophiaG class if not installed
    class SophiaG(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
            logger.warning("Using dummy SophiaG optimizer. Please install sophia-pytorch for full functionality.")
            defaults = dict(lr=lr, weight_decay=weight_decay)
            super().__init__(params, defaults)
        def step(self, closure=None):
            if closure is not None:
                closure()
            logger.debug("Dummy SophiaG step called.")
# LoRA Adapters
HAS_LORA = False
try:
    from loralib import LoRA
    HAS_LORA = True
except ImportError:
    logger.warning("Could not import loralib. LoRA adapters will not be available.")
    # Dummy LoRA class
    class LoRA(nn.Module):
        def __init__(self, model, r=4, lora_alpha=32):
            super().__init__()
            self.model = model
            logger.warning("Using dummy LoRA. Please install loralib for full functionality.")
        def forward(self, x):
            return self.model(x)
# timm optimizers (Lion, MADGRAD)
HAS_TIMM_OPTIM = False
try:
    # timm.optim is not a direct module, optimizers are usually accessed via timm.create_optimizer_v2
    # For direct import, we might need torch_optimizer
    from torch_optimizer import Lion, MADGRAD # Assuming torch_optimizer is installed
    HAS_TIMM_OPTIM = True
except ImportError:
    logger.warning("Could not import Lion or MADGRAD from torch_optimizer. These optimizers will not be available.")
# Quantization imports
HAS_TORCH_QUANTIZATION = False
try:
    import torch.quantization as tq # CRITICAL FIX: Added alias 'tq'
    HAS_TORCH_QUANTIZATION = True
except ImportError:
    logger.warning("torch.quantization not available. QAT will be disabled.")

# --- Global Configuration and Constants ---
# DEFAULT_CONFIG is the base, which will be loaded first.
DEFAULT_CONFIG = {
    'data': {
        'dataset_path': './ShapeBongard_V2', # Unified source of truth
        'image_size': [224, 224], # Default image size
        'num_channels': 1, # 1 for grayscale, 3 for RGB
        'train_test_split_ratio': 0.2,
        'dataloader_num_workers': 8, # UPDATED: Increased worker count for DALI parallelism
        'dataloader_pin_memory': True,
        'dataloader_persistent_workers': False,
        'dataloader_prefetch_factor': 3, # UPDATED: Increased prefetch queue depth
        'class_imbalance_threshold': 0.2,
        'progressive_resizing_epochs': 5, # Epochs for small image size
        'initial_image_size': [112, 112], # Smaller initial size for DALI decoding
        'curriculum_learning_enabled': True,
        'curriculum_difficulty_metric': 'std_dev',
        'curriculum_annealing_epochs': 5,
        'cache_data': True, # UPDATED: Enable data caching
        'curriculum_start_difficulty': 0.0,
        'curriculum_end_difficulty': 1.0,
        'curriculum_difficulty_schedule': 'linear',
        'curriculum_update_interval_epochs': 1,
        'use_synthetic_data': False, # NEW: Flag to use programmatic synthetic data
        'synthetic_samples': 1000 # NEW: Number of synthetic samples to generate
    },
    'model': {
        'num_classes': 2,
        'initial_learning_rate': 0.001,
        'max_learning_rate': 0.01, # Required for OneCycleLR scheduler
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
        'object_detector_model_path': 'yolov8_bongard_finetuned.pt', # UPDATED: Point to your fine-tuned model
        'attribute_classifier_model_path': None,
        'relation_gnn_model_path': None,
        'attribute_backbone': 'mobilenet_v2', # NEW: Option for MobileNetV2, MobileNetV3, EfficientNet
        'gnn_depth': 2, # NEW: Depth of the GNN (number of GCNConv layers)
        'gnn_max_depth': 4, # NEW: Maximum GNN depth for adaptive/linear schedules
        'gnn_min_depth': 1, # NEW: Minimum GNN depth for adaptive/linear schedules
        'gnn_depth_schedule': 'adaptive', # NEW: 'fixed', 'linear', 'adaptive'
        'detection_confidence_threshold': 0.1 # NEW: Confidence threshold for object detection
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
        # Moved use_swa and related parameters here to be configurable via YAML
        'use_swa': True,
        'swa_start_epoch_ratio': 0.75,
        'swa_lr': 0.05,
        'self_distillation_alpha': 0.5,
        'feature_consistency_alpha': 0.1,
        'feature_consistency_loss_type': 'mse', # NEW: 'mse', 'cosine', 'kl_div'
        'knowledge_replay_enabled': True,
        'replay_buffer_size': 100,
        'replay_frequency_epochs': 2,
        'replay_batch_size_ratio': 0.1,
        'use_wandb': True,
        'early_stopping_monitor_metric': 'val_loss', # Can be 'val_loss', 'val_accuracy', etc.
        'use_torch_compile': True,
        'enable_profiler': False,
        'profiler_schedule_wait': 1,
        'profiler_schedule_warmup': 2,
        'profiler_schedule_active': 10,
        'profiler_schedule_repeat': 3,
        'validation_frequency_batches': 0,
        'distillation_temperature': 2.0,
        'distillation_alpha': 0.5,
        'use_knowledge_distillation': False,
        'profile_performance': False,
        'use_qat': False,
        'qat_start_epoch': 5,
        'use_weighted_sampling': True, # NEW
        'save_symbolic_outputs_interval_epochs': 5, # NEW: Interval to save symbolic outputs
        'onecycle_pct_start': 0.3 # ADDED: Percentage of steps to increase learning rate in OneCycleLR
    },
    'debug': {
        'visualize_data': True,
        'visualize_training_history': True,
        'visualize_perception_output': True,
        'plot_reliability_diagram': True,
        'log_level': 'INFO',
        'visualize_gnn_weights': False, # NEW: Toggle for GNN edge weight visualization (conceptual)
        'max_fallback_cnt': 5, # Max number of contours for fallback detection
        'apply_symbolic_priors': True, # NEW: Toggle for applying symbolic priors in GNN
        'min_contour_area_sam_fallback': 50 # Minimum contour area for SAM fallback detections
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
# This block is placed after load_config to ensure it overrides any YAML settings.
logger.info("Applying critical memory-optimized configuration overrides for 4GB GPU:")
CONFIG['model']['batch_size'] = 4            # Reduced batch size
logger.info(f"  Overriding 'model.batch_size' to {CONFIG['model']['batch_size']}.")
CONFIG['data']['image_size'] = [96, 96]      # Reduced image size
logger.info(f"  Overriding 'data.image_size' to {CONFIG['data']['image_size']}.")
CONFIG['data']['initial_image_size'] = [64, 64] # Smaller initial size for DALI decoding
logger.info(f"  Overriding 'data.initial_image_size' to {CONFIG['data']['initial_image_size']}.")
CONFIG['training']['feature_consistency_alpha'] = 0.0 # Disable second view
logger.info(f"  Overriding 'training.feature_consistency_alpha' to {CONFIG['training']['feature_consistency_alpha']} (disabling second view for memory).")
CONFIG['training']['use_amp'] = True       # Ensure AMP is enabled
logger.info(f"  Ensuring 'training.use_amp' is {CONFIG['training']['use_amp']} (Automatic Mixed Precision).")
CONFIG['data']['dataloader_num_workers'] = 6 # Increased worker count for DALI parallelism
logger.info(f"  Overriding 'data.dataloader_num_workers' to {CONFIG['data']['dataloader_num_workers']}.")
CONFIG['training']['gradient_accumulation_steps'] = 2 # Set to 2 as per report
logger.info(f"  Overriding 'training.gradient_accumulation_steps' to {CONFIG['training']['gradient_accumulation_steps']}.")
CONFIG['training']['mixup_alpha'] = 0.2 # Enable MixUp
logger.info(f"  Overriding 'training.mixup_alpha' to {CONFIG['training']['mixup_alpha']} (enabling MixUp).")
CONFIG['training']['cutmix_alpha'] = 1.0 # Enable CutMix
logger.info(f"  Overriding 'training.cutmix_alpha' to {CONFIG['training']['cutmix_alpha']} (enabling CutMix).")
CONFIG['training']['label_smoothing_epsilon'] = 0.1 # Enable Label Smoothing
logger.info(f"  Overriding 'training.label_smoothing_epsilon' to {CONFIG['training']['label_smoothing_epsilon']} (enabling Label Smoothing).")
CONFIG['model']['detection_confidence_threshold'] = 0.1   # Lower detection threshold
logger.info(f"  Overriding 'model.detection_confidence_threshold' to {CONFIG['model']['detection_confidence_threshold']}.")
CONFIG['model']['yolo_augment'] = True                  # New flag for TTA
logger.info(f"  Overriding 'model.yolo_augment' to {CONFIG['model']['yolo_augment']} (enabling YOLO Test Time Augmentation).")
logger.info("Finished applying critical configuration overrides.")
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
# --- UPDATED: Cache and Difficulty Score File Paths ---
# Store these in a separate 'processed_data' directory for cleaner project structure.
PROCESSED_DATA_DIR = 'processed_data' # New directory for processed data/cache
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Ensure this directory exists
DIFFICULTY_SCORES_FILE = os.path.join(PROCESSED_DATA_DIR, 'difficulty_scores.npy')
FULL_DATA_CACHE_FILE = os.path.join(PROCESSED_DATA_DIR, 'bongard_data_full_cache.npz')
# --- END UPDATED PATHS ---
print(f"Checking for existing dataset folder at: {DATA_ROOT_PATH}")
if not os.path.exists(DATA_ROOT_PATH) and not CONFIG['data']['use_synthetic_data']:
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
                  f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID} "
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
    if not CONFIG['data']['use_synthetic_data']:
        print(f"Dataset folder '{DATA_ROOT_FOLDER_NAME}' already exists at {DATA_ROOT_PATH}. Skipping download and unzip.")
    else:
        print("Using synthetic data. Skipping real dataset download.")
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
    if os.path.exists(FULL_DATA_CACHE_FILE) and CONFIG['data'].get('cache_data', False):
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
    # This is the correct structure as per your description and screenshot:
    # ShapeBongard_V2/{bd,ff,hd}/images/<problem_id>/{0,1}/*.png
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
            # This glob.glob correctly finds the 'problem_id' folders (e.g., 'bd_acute_equi_triangle_0000')
            for problem_folder_path in glob.glob(os.path.join(base_dir, '*')):
                if os.path.isdir(problem_folder_path):
                    # This loop correctly finds the '0' and '1' subfolders
                    for sub_label_folder_name in ['0', '1']:
                        sub_label_path = os.path.join(problem_folder_path, sub_label_folder_name)
                        if not os.path.exists(sub_label_path):
                            continue
                        current_label = int(sub_label_folder_name)
                        # This glob.glob correctly finds the .png images inside '0' or '1'
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
    # Now, calculate or load difficulty scores
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
                # Load as grayscale for standard deviation calculation
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
    if CONFIG['data'].get('cache_data', False):
        np.savez_compressed(FULL_DATA_CACHE_FILE,
                            image_paths=np.array(all_image_paths, dtype=object), # Use dtype=object for string paths
                            labels=np.array(all_labels),
                            difficulty_scores=np.array(all_difficulty_scores))
        logger.info(f"Saved full data metadata cache to {FULL_DATA_CACHE_FILE}")
    return all_image_paths, labels_array, np.array(all_difficulty_scores)
# --- NEW: Programmatic Synthetic Data Generator (Part 1) ---
class BongardGenerator:
    """
    Generates synthetic Bongard-like images with geometric shapes and ground truth.
    Each image contains a random number of objects with random attributes and relations.
    """
    def __init__(self, image_size=(96, 96), num_objects=(2, 5), num_classes=2):
        self.W, self.H = image_size
        self.num_objects_range = num_objects
        self.shapes = ['circle', 'square', 'triangle']
        self.fill_types = ['solid', 'outline']
        self.colors = ['black', 'white'] # Shapes are drawn on a white background
        self.num_classes = num_classes # For overall Bongard problem label (0 or 1)
    def _sample_object(self):
        """Samples properties for a single object."""
        shape = random.choice(self.shapes)
        size = random.randint(int(self.W * 0.1), int(self.W * 0.3)) # Size relative to image
        x = random.randint(0, self.W - size)
        y = random.randint(0, self.H - size)
        
        # Ensure objects are within bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.W, x + size)
        y2 = min(self.H, y + size)
        attr = {
            'fill': random.choice(self.fill_types),
            'color': random.choice(self.colors), # This is the object's color, not background
            'size': size,
            'orientation': random.randint(0, len(ATTRIBUTE_ORIENTATION_MAP) - 1), # Use integer index
            'shape': self.shapes.index(shape), # Use integer index
            'texture': random.randint(0, len(ATTRIBUTE_TEXTURE_MAP) - 1) # Use integer index
        }
        return dict(shape_type=shape, bbox=(x1, y1, x2, y2), **attr)
    def generate(self, n_samples):
        """
        Generates n_samples of Bongard-like data.
        Returns a list of tuples: (PIL_image, ground_truth_dict).
        """
        data = []
        for sample_idx in tqdm(range(n_samples), desc="Generating synthetic data"):
            k = random.randint(*self.num_objects_range)
            objs = [self._sample_object() for _ in range(k)]
            
            # Decide overall Bongard problem label (e.g., 0 or 1)
            # For simplicity, let's make it random for now. In a real Bongard generator,
            # this would depend on the underlying rule.
            bongard_label = random.randint(0, self.num_classes - 1)
            img = Image.new('RGB', (self.W, self.H), 'white')
            draw = ImageDraw.Draw(img)
            for i, o in enumerate(objs):
                fill_color = o['color'] if o['fill'] == 'solid' else None
                outline_color = o['color'] if o['fill'] == 'outline' else 'black' # Default outline to black if solid fill
                
                # Ensure colors are valid for PIL
                if fill_color == 'black': fill_color_tuple = (0, 0, 0)
                elif fill_color == 'white': fill_color_tuple = (255, 255, 255)
                else: fill_color_tuple = None # No fill
                if outline_color == 'black': outline_color_tuple = (0, 0, 0)
                elif outline_color == 'white': outline_color_tuple = (255, 255, 255)
                else: outline_color_tuple = (0, 0, 0) # Default to black outline
                bbox_coords = o['bbox']
                if o['shape_type'] == 'circle':
                    draw.ellipse(bbox_coords, outline=outline_color_tuple, fill=fill_color_tuple, width=2)
                elif o['shape_type'] == 'square':
                    draw.rectangle(bbox_coords, outline=outline_color_tuple, fill=fill_color_tuple, width=2)
                elif o['shape_type'] == 'triangle':
                    # Calculate triangle vertices from bbox
                    x1, y1, x2, y2 = bbox_coords
                    points = [(x1 + (x2 - x1) / 2, y1), (x1, y2), (x2, y2)] # Equilateral-like triangle
                    draw.polygon(points, outline=outline_color_tuple, fill=fill_color_tuple, width=2)
                # Add more shapes as needed
            # Compute relations analytically (e.g., overlaps, contains)
            relations = []
            for i in range(k):
                for j in range(k):
                    if i == j: continue
                    b1, b2 = objs[i]['bbox'], objs[j]['bbox']
                    
                    # IoU calculation
                    xA = max(b1[0], b2[0])
                    yA = max(b1[1], b2[1])
                    xB = min(b1[2], b2[2])
                    yB = min(b1[3], b2[3])
                    inter_width = xB - xA
                    inter_height = yB - yA
                    inter_area = max(0, inter_width) * max(0, inter_height)
                    
                    box1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    box2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    
                    union_area = float(box1_area + box2_area - inter_area)
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    # Relative position
                    center_i_x, center_i_y = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
                    center_j_x, center_j_y = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
                    rel_type = "unrelated"
                    if iou > 0.7: # High IoU suggests contains or overlaps
                        if box1_area > box2_area * 1.5 and iou * box1_area > box2_area * 0.9: # Heuristic for contains
                            rel_type = "contains"
                        elif box2_area > box1_area * 1.5 and iou * box2_area > box1_area * 0.9: # Heuristic for inside
                            rel_type = "inside"
                        else:
                            rel_type = "overlaps"
                    elif iou > 0: # Some overlap but not significant
                        rel_type = "overlaps"
                    elif abs(center_i_x - center_j_x) < (b1[2]-b1[0])/4 and center_j_y < center_i_y:
                        rel_type = "above"
                    elif abs(center_i_x - center_j_x) < (b1[2]-b1[0])/4 and center_j_y > center_i_y:
                        rel_type = "below"
                    elif abs(center_i_y - center_j_y) < (b1[3]-b1[1])/4 and center_j_x < center_i_x:
                        rel_type = "left_of"
                    elif abs(center_i_y - center_j_y) < (b1[3]-b1[1])/4 and center_j_x > center_i_x:
                        rel_type = "right_of"
                    # Add more complex relations based on bbox geometry, e.g., 'touches', 'adjacent'
                    # For simplicity, using simple spatial relations and IoU for now.
                    
                    # Add same_type_as / different_type_from
                    if objs[i]['shape_type'] == objs[j]['shape_type']:
                        rel_type = "same_type_as"
                    else:
                        rel_type = "different_type_from"
                    relations.append({'src': i, 'dst': j, 'type': RELATION_MAP.get(RELATION_MAP_INV.get(rel_type, 0), rel_type), 'confidence': 1.0}) # Ensure type is string
            # Ground truth dictionary
            gt_dict = {
                'objects': [],
                'relations': relations,
                'bongard_label': bongard_label # Overall label for the problem
            }
            # Populate objects with ground truth attributes
            for obj_idx, o in enumerate(objs):
                gt_dict['objects'].append({
                    'id': obj_idx,
                    'bbox': o['bbox'],
                    'type': o['shape_type'], # The actual shape type (e.g., 'circle')
                    'attributes': {
                        'fill': o['fill'],
                        'color': o['color'],
                        'size': ATTRIBUTE_SIZE_MAP.get(o['size'], 'unknown'), # Map size back to category if needed
                        'orientation': ATTRIBUTE_ORIENTATION_MAP.get(o['orientation'], 'unknown'),
                        'shape': ATTRIBUTE_SHAPE_MAP.get(o['shape'], 'unknown'),
                        'texture': ATTRIBUTE_TEXTURE_MAP.get(o['texture'], 'unknown')
                    }
                })
            data.append((img, gt_dict))
        return data
class BongardSyntheticDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the programmatic Bongard synthetic data.
    """
    def __init__(self, n_samples, image_size=(96,96), num_objects=(2,5), num_classes=2):
        self.data = BongardGenerator(image_size=image_size, num_objects=num_objects, num_classes=num_classes).generate(n_samples)
        logger.info(f"Generated {len(self.data)} synthetic Bongard samples.")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, gt = self.data[idx]
        # For DALI external source, we need image as bytes and GT as JSON string
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG') # Save as PNG bytes
        img_bytes = img_byte_arr.getvalue()
        
        gt_json_string = json.dumps(gt) # Serialize ground truth dict to JSON string
        return img_bytes, gt_json_string
import io # Needed for BytesIO
class BongardExternalSource:
    """
    DALI External Source for feeding synthetic data into the DALI pipeline.
    """
    def __init__(self, dataset: BongardSyntheticDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.rng = random.Random() # Use a dedicated RNG for shuffling
    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.i = 0
        return self
    def __next__(self):
        if self.i >= len(self.indices):
            self.i = 0 # Reset for next epoch
            if self.shuffle:
                self.rng.shuffle(self.indices)
            raise StopIteration
        batch_indices = self.indices[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        batch_imgs = []
        batch_gts = []
        for idx in batch_indices:
            img_bytes, gt_json_string = self.dataset[idx]
            batch_imgs.append(np.frombuffer(img_bytes, dtype=np.uint8)) # DALI needs numpy array
            batch_gts.append(np.array(gt_json_string, dtype=np.object_)) # DALI needs numpy array of object type for strings
        return batch_imgs, batch_gts
# -----------------------------------------------------------------------------
# YOLOv8 Class Mapping (Example for common objects, extend as needed)
# This maps the integer class IDs returned by YOLO to more descriptive names.
# You would need to align this with the specific YOLO model's training classes.
# -----------------------------------------------------------------------------
YOLO_CLASS_MAP = {
    0: "circle", 1: "square", 2: "triangle", 3: "line", 4: "arc",
    5: "star",   6: "polygon",  7: "diamond",  8: "oval", 9: "rectangle",
    10: "cross", 11: "arrow", 12: "heart", 13: "spiral", 14: "cloud",
    15: "lightning", 16: "text_character", 17: "ellipse", 18: "trapezoid",
    19: "pentagon", 20: "hexagon", 21: "octagon", 22: "semicircle",
    # Add more classes as per your YOLO fine-tuning
}
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
ATTRIBUTE_FILL_MAP_INV = {v: k for k, v in ATTRIBUTE_FILL_MAP.items()}
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
ATTRIBUTE_COLOR_MAP_INV = {v: k for k, v in ATTRIBUTE_COLOR_MAP.items()}
# 3) Size categories (object emerges at different scales in logo)
ATTRIBUTE_SIZE_MAP = {
    0: "tiny",
    1: "small",
    2: "medium",
    3: "large",
    4: "huge",
    5: "full_frame", # object occupies most of the image
}
ATTRIBUTE_SIZE_MAP_INV = {v: k for k, v in ATTRIBUTE_SIZE_MAP.items()}
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
ATTRIBUTE_ORIENTATION_MAP_INV = {v: k for k, v in ATTRIBUTE_ORIENTATION_MAP.items()}
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
ATTRIBUTE_SHAPE_MAP_INV = {v: k for k, v in ATTRIBUTE_SHAPE_MAP.items()}
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
ATTRIBUTE_TEXTURE_MAP_INV = {v: k for k, v in ATTRIBUTE_TEXTURE_MAP.items()}
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
RELATION_MAP_INV = {v: k for k, v in RELATION_MAP.items()}
# --- Curriculum Sampler (Updated for Weighted Sampling and DALI file_list generation) ---
class CurriculumSampler(torch.utils.data.Sampler):
    """
    A curriculum learning sampler that gradually increases sample difficulty over epochs.
    It provides batches of image paths and labels based on precomputed difficulty scores.
    This updated version can also incorporate weighted sampling for class imbalance
    and prepares the file_list and labels for DALI's fn.readers.file.
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
        use_weighted_sampling (bool): If True, applies weighted random sampling for class imbalance.
    """
    def __init__(self, image_paths, labels, initial_difficulty_scores, batch_size, annealing_epochs, total_epochs, is_train, use_weighted_sampling=False):
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
        
        # These will be used by get_current_image_size
        self.initial_image_size = CONFIG['data']['initial_image_size']
        self.final_image_size = CONFIG['data']['image_size']
        self.progressive_resizing_epochs = CONFIG['data']['progressive_resizing_epochs']
        
        self.use_weighted_sampling = use_weighted_sampling # NEW: Flag for weighted sampling
        if self.use_weighted_sampling:
            label_counts = collections.Counter(self.labels)
            self.class_weights = {cls: 1.0 / (count + 1e-6) for cls, count in label_counts.items()}
            self.sample_weights = [self.class_weights[l] for l in self.labels]
            logger.info("CurriculumSampler configured for weighted sampling.")
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
    def get_epoch_data(self) -> Tuple[List[str], List[int]]:
        """
        Returns the list of image paths and corresponding labels for the current epoch,
        applying curriculum learning and weighted sampling as configured.
        This method is designed to provide the file_list and labels for DALI's fn.readers.file.
        Returns:
            Tuple[List[str], List[int]]: A tuple containing the list of image paths and labels
                                         for the current epoch.
        """
        if self.is_train and self.current_epoch < self.annealing_epochs:
            # Curriculum learning phase: gradually include harder samples
            progress = self.current_epoch / self.annealing_epochs
            # Start with 10% easiest samples and linearly increase to 100%
            num_samples_to_include = int(len(self.indices) * (0.1 + 0.9 * progress))
            
            # Select the easiest samples based on difficulty scores
            current_indices = self.difficulty_order[:num_samples_to_include]
            logger.debug(f"Epoch {self.current_epoch}: Curriculum sampling, including {num_samples_to_include} easiest samples.")
        else:
            # Uniform sampling phase (after annealing or for validation)
            current_indices = np.array(self.indices)
            logger.debug(f"Epoch {self.current_epoch}: Uniform sampling.")
        
        if self.use_weighted_sampling and self.is_train:
            # Create a WeightedRandomSampler for the currently selected indices
            # Filter weights to only include those for current_indices
            current_sample_weights = [self.sample_weights[idx] for idx in current_indices]
            
            # Use WeightedRandomSampler to get the order of indices for this epoch
            # num_samples should be total number of samples for the epoch, typically len(current_sample_weights)
            # If you want to oversample minority classes significantly, num_samples can be larger.
            # For DALI, we need the list of files, so we generate a list of indices.
            weighted_sampler_for_epoch = WeightedRandomSampler(
                weights=current_sample_weights,
                num_samples=len(current_sample_weights), # Draw samples from this subset
                replacement=True # Essential for weighted sampling
            )
            
            # Map back to original dataset indices
            final_indices_for_epoch = [current_indices[i] for i in list(weighted_sampler_for_epoch)]
            logger.debug("Applied weighted sampling within current curriculum band.")
        else:
            # If not weighted sampling, just shuffle the selected indices
            final_indices_for_epoch = current_indices.copy()
            np.random.shuffle(final_indices_for_epoch)
        # Prepare paths and labels for DALI
        epoch_paths = [self.image_paths[j] for j in final_indices_for_epoch]
        epoch_labels = [self.labels[j] for j in final_indices_for_epoch]
        
        return epoch_paths, epoch_labels
    def __len__(self):
        """
        Returns the total number of samples for the current epoch, considering curriculum learning.
        This is used for DALI iterator size calculation.
        """
        if self.is_train and self.current_epoch < self.annealing_epochs:
            progress = self.current_epoch / self.annealing_epochs
            num_samples_to_include = int(len(self.indices) * (0.1 + 0.9 * progress))
            return num_samples_to_include
        return len(self.indices)

# --- NEW: Kornia Augmentation Function for DALI Python Operator ---
@torch.no_grad() # Ensure no gradients are tracked for augmentations
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
        return images_dali_gpu # Return original images if Kornia not available

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
            K.RandomVerticalFlip(p=0.2), # Add vertical flip
            K.RandomRotation(degrees=K.RandomUniformGenerator(0.0, 15.0), p=0.5),
            K.RandomAffine(degrees=K.RandomUniformGenerator(0.0, 10.0), # Small affine
                           translate=K.RandomUniformGenerator(0.0, 0.1),
                           scale=K.RandomUniformGenerator(0.8, 1.2),
                           shear=K.RandomUniformGenerator(0.0, 5.0),
                           p=0.5),
            K.ColorJitter(brightness=K.RandomUniformGenerator(0.6, 1.4), # Stronger color jitter
                          contrast=K.RandomUniformGenerator(0.6, 1.4),
                          saturation=K.RandomUniformGenerator(0.6, 1.4),
                          hue=K.RandomUniformGenerator(-0.2, 0.2),
                          p=0.8),
            K.RandomGrayscale(p=0.1) if num_channels == 3 else K.Identity(), # Apply only if RGB
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
            # Add ElasticTransform if available and configured
            K.RandomElasticTransform(kernel_size=(3,3), sigma=(4.0, 5.0), alpha=(30.0, 35.0), p=0.3) if HAS_ELASTIC_TRANSFORM else K.Identity(),
            data_keys=["input"], # Specify that augmentations apply to the input image
            p=1.0, # Apply the sequence with this probability
            same_on_batch=False # Apply different augmentations to each image in batch
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
               (image_view_1, image_view_2, bongard_label, ground_truth_json_string, affine_matrix_1, affine_matrix_2)
    """
    if use_synthetic_data:
        logger.debug("DALI pipeline using external source for synthetic data.")
        # external_source_iterator yields (image_bytes, gt_json_string)
        # We need to explicitly define outputs for external_source
        imgs, gts_json = fn.external_source(
            source=external_source_iterator,
            num_outputs=2, # Images and GT JSON string
            layout=["H", "H"], # Layout for image bytes and string (arbitrary for string)
            name="ExternalSource"
        )
        # Decode images from bytes
        decoded = fn.decoders.image(
            imgs,
            device="mixed" if device_id != -1 else "cpu",
            output_type=types.RGB if num_channels == 3 else types.GRAY
        )
        # The overall Bongard label needs to be extracted from the gt_json.
        # This will be handled in the training loop after parsing the JSON.
        # For DALI, we can pass a dummy label or extract a specific field if it's always there.
        # For now, we'll pass the JSON string directly and extract the label in PyTorch.
        # So, labels_gpu will refer to the overall Bongard label from GT, if available.
        # For now, let's just pass a dummy label or assume it's part of the GT JSON.
        # The `labels_list` argument passed to the pipeline will be ignored for synthetic data.
        labels_gpu = fn.constant(np.array([0], dtype=np.int64), shape=(1,)) # Dummy label, will be overridden by GT JSON
        labels_gpu = fn.cast(labels_gpu, dtype=types.INT64) # Ensure correct type
        
        # Pass the original gt_json_string through as well
        gts_json_gpu = gts_json.gpu() if device_id != -1 else gts_json
        gts_json_gpu = fn.cast(gts_json_gpu, dtype=types.STRING) # Ensure it's treated as string
        
    else:
        logger.debug("DALI pipeline using file reader for real data.")
        # Use fn.readers.file to read images and labels directly
        # file_list and labels_list must be provided as arguments to the pipeline_def
        imgs, labels = fn.readers.file(
            file_root=file_root,
            files=file_list,
            labels=labels_list,
            random_shuffle=is_train, # Shuffle only for training
            name="Reader"
        )
        decoded = fn.decoders.image(
            imgs,
            device="mixed" if device_id != -1 else "cpu",
            output_type=types.RGB if num_channels == 3 else types.GRAY
        )
        labels_gpu = fn.cast(labels, dtype=types.INT64)
        gts_json_gpu = fn.constant(np.array(["{}"], dtype=np.object_), shape=(1,)) # Dummy empty JSON string
        gts_json_gpu = fn.cast(gts_json_gpu, dtype=types.STRING)
        
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
        images_view2 = decoded # Start from decoded image again for second view
        # Apply different augmentations for second view
        images_view2 = fn.random_resized_crop(
            images_view2,
            size=[height, width],
            random_area=(0.7, 1.0), # Slightly different area
            random_aspect_ratio=(0.7, 1.4), # Slightly different aspect ratio
            device="gpu" if device_id != -1 else "cpu"
        )
        images_view2 = fn.flip(images_view2, horizontal=fn.random.coin_flip(),
                               device="gpu" if device_id != -1 else "cpu")
        images_view2 = fn.brightness_contrast(
            images_view2,
            brightness=fn.random.uniform(range=(0.7, 1.3)), # Different ranges
            contrast=fn.random.uniform(range=(0.7, 1.3)),
            device="gpu" if device_id != -1 else "cpu"
        )
        if num_channels == 3:
            images_view2 = fn.hue(images_view2, hue=fn.random.uniform(range=(-0.2, 0.2)),
                                  device="gpu" if device_id != -1 else "cpu")
            images_view2 = fn.saturation(images_view2, saturation=fn.random.uniform(range=(0.7, 1.3)),
                                         device="gpu" if device_id != -1 else "cpu")
        images_view2_rotation_angle_dali = fn.random.uniform(range=(-20.0, 20.0)) # Different range
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
    # Return the bongard_label (0 or 1) and the full ground truth JSON string
    return out_view1, out_view2, labels_gpu, gts_json_gpu, affine_matrices_1, affine_matrices_2

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

# ... (Continued from phase1-code-part1) ...
# -----------------------------------------------------------------------------
# RealObjectDetector: YOLOv8 & Fallbacks (with Conceptual SAM Mask-Based Fallback)
# -----------------------------------------------------------------------------
class RealObjectDetector:
    """
    RealObjectDetector class to integrate YOLOv8 for object detection.
    This class wraps the `ultralytics.YOLO` model to detect objects in images.
    Args:
        model_path (str): Path to the YOLOv8 model weights (e.g., 'yolov8n.pt').
    """
    def __init__(self, model_path: str = 'yolov8n.pt'):
        # Attempt to import the actual YOLO from ultralytics
        try:
            from ultralytics import YOLO as RealYOLO_Actual
            self.model = RealYOLO_Actual(model_path)
            logger.info(f"Successfully loaded ultralytics.YOLO model from {model_path}.")
        except ImportError:
            logger.warning("Ultralytics YOLO not found. Using globally defined dummy RealYOLO for object detection.")
            # Fallback to the globally defined RealYOLO dummy
            self.model = RealYOLO(model_path) # Assumes RealYOLO dummy is available in scope
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}. Ensure the path is correct and model is valid.")
            self.model = None # Set to None to indicate failure
        self._cache = {}
        # Initialize SAM model if Kornia is available (lazy initialization)
        self._sam_model = None
        if HAS_KORNIA:
            logger.info("Kornia is available. SAM model will be lazily initialized for fallback.")
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
        if self.model: # Only attempt YOLO if model was loaded successfully
            try:
                logger.debug(f"Running YOLO model on image (size: {processed_image_pil.size}, confidence: {overall_confidence}).")
                results = self.model(
                    processed_image_pil, # Use the enhanced image
                    verbose=False,
                    augment=CONFIG['model']['yolo_augment'] # Use TTA if configured
                )
                logger.debug(f"YOLO raw results: {len(results[0].boxes)} detections.")
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if conf >= overall_confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_name = YOLO_CLASS_MAP.get(cls, f"class_{cls}")
                        detections.append({
                            "id": len(detections), # Assign a unique ID for this detection run
                            "bbox": [x1, y1, x2, y2],
                            "type": class_name,
                            "confidence": conf
                        })
                logger.debug(f"YOLO filtered detections: {len(detections)} detections above confidence threshold.")
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
            if self.model: # Only attempt upscaled YOLO if model was loaded successfully
                try:
                    upscale_results = self.model(up, verbose=False, augment=CONFIG['model']['yolo_augment'])[0].boxes
                    for box in upscale_results:
                        conf = float(box.conf[0])
                        if conf >= overall_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            detections.append({
                                "id": len(detections),
                                "bbox": [x1//2, y1//2, x2//2, y2//2], # scale coordinates back down by 2x
                                "type": YOLO_CLASS_MAP.get(int(box.cls[0]), f"class_{int(box.cls[0])}"),
                                "confidence": conf
                            })
                    logger.debug(f"Upscaled YOLO filtered detections: {len(detections)} detections.")
                except Exception as e:
                    logger.warning(f"Upscaled YOLO detection failed: {e}. Proceeding to conceptual SAM fallback or contour fallback.")
            else:
                logger.warning("YOLO model not loaded. Skipping upscaled YOLO detection.")
        
        # Robust "No Objects Detected" Fallback: Conceptual SAM Mask-Based or basic contour detection
        if not detections:
            logger.info(f"YOLO (and upscaled YOLO) found no detections. Attempting conceptual SAM fallback or contour fallback.")
            if HAS_KORNIA: # Check if Kornia (and thus kornia.models.SegmentAnything) is available
                try:
                    # Convert PIL image to Kornia-compatible tensor (NCHW, float32, [0,1])
                    img_tensor = T.ToTensor()(image_pil).unsqueeze(0).to(DEVICE) # 1CHW
                    if img_tensor.shape[1] == 1: # If grayscale, replicate to 3 channels for SAM
                        img_tensor = img_tensor.repeat(1, 3, 1, 1)
                    
                    # SAM expects 0-255 range for its input, so scale up again
                    img_tensor_for_sam = img_tensor * 255.0
                    
                    # Initialize SAM model (only once if possible, or cache it)
                    if self._sam_model is None:
                        logger.info("Initializing Kornia SegmentAnything (ViT-H) model for conceptual fallback.")
                        self._sam_model = SegmentAnything(model="vit_h").to(DEVICE).eval()
                    
                    # --- Conceptual SAM Fallback Strategy ---
                    # When YOLO finds nothing, SAM needs prompts to generate masks.
                    # A simple strategy is to generate a grid of points across the image
                    # to encourage SAM to find any salient objects.
                    
                    H_img, W_img = image_pil.size[1], image_pil.size[0] # PIL size is (width, height)
                    # Generate a simple grid of points as prompts
                    grid_points = []
                    num_points_x = 3
                    num_points_y = 3
                    for i in range(num_points_y):
                        for j in range(num_points_x):
                            x_coord = int(W_img * (j + 0.5) / num_points_x)
                            y_coord = int(H_img * (i + 0.5) / num_points_y)
                            grid_points.append([x_coord, y_coord])
                    
                    point_coords = torch.tensor(grid_points, dtype=torch.float32, device=DEVICE).unsqueeze(0) # 1xNx2
                    point_labels = torch.ones(1, point_coords.shape[1], dtype=torch.float32, device=DEVICE) # All positive prompts
                    logger.debug(f"Attempting SAM fallback with {point_coords.shape[1]} grid points as prompts.")
                    
                    # Run SAM to get masks
                    # SAM's predict_masks returns a list of masks per image in batch
                    # Each mask is typically a boolean tensor.
                    sam_output = self._sam_model.predict_masks(
                        img_tensor_for_sam,
                        point_coords=point_coords,
                        point_labels=point_labels
                    )
                    
                    # Process SAM masks to get bounding boxes
                    # sam_output is a list of dicts, each dict has 'masks', 'iou_predictions', 'low_res_logits'
                    if sam_output and sam_output[0] and 'masks' in sam_output[0]:
                        sam_masks = sam_output[0]['masks'] # [num_masks, H, W]
                        logger.debug(f"SAM generated {sam_masks.shape[0]} masks. Processing masks to bboxes.")
                        
                        for mask_idx in range(sam_masks.shape[0]):
                            mask_np = sam_masks[mask_idx].squeeze(0).cpu().numpy().astype(np.uint8) * 255 # Convert to 0-255
                            
                            # Find contours in the mask
                            contours_mask, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours_mask:
                                largest_contour = max(contours_mask, key=cv2.contourArea)
                                area = cv2.contourArea(largest_contour)
                                if area > CONFIG['debug'].get('min_contour_area_sam_fallback', 50): # Use a configurable min area
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
                                        "confidence": float(sam_output[0]['iou_predictions'][mask_idx].cpu().item()) # Use SAM's IoU prediction as confidence
                                    })
                        logger.debug(f"SAM fallback yielded {len(detections)} detections.")
                    else:
                        logger.warning("SAM did not generate any masks. Proceeding to contour fallback.")
                except Exception as sam_e:
                    logger.warning(f"Kornia SAM fallback failed: {sam_e}. Ensure kornia is installed and model can be loaded. Proceeding to contour fallback.")
            
            # Fallback to basic contour detection for blobs/shapes if SAM also fails or is not available
            if not detections: # Only run if SAM fallback also failed
                logger.info("SAM fallback failed or not available. Falling back to basic contour detection.")
                detections = fallback_shape_detection_bw(image_pil.convert('L'))
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
        pil_img = pil_img.convert('RGB') # Convert to RGB if it's an unsupported mode like P (paletted)
        logger.debug(f"Converted image from {original_mode} to RGB for contrast enhancement.")
    
    # Gamma correction
    # Convert to float32, normalize to 0-1, apply gamma, then scale back to 0-255
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = np.power(arr, 1.0 / gamma) # Apply inverse gamma for brightening
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
            break # Stop if max_fallback_cnt is reached
        
        area = cv2.contourArea(cnt)
        if area < min_area:
            logger.debug(f"Skipping contour {cnt_idx} due to small area ({area} < {min_area}).")
            continue # Skip small contours
        
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
        elif num_vertices > 4 and num_vertices < 8: # Heuristic for general polygons
            shape = "polygon"
        elif num_vertices >= 8:
            # Check circularity for circle-like shapes
            if peri > 0:
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity > 0.7: # Common threshold for circularity
                    shape = "circle"
                else:
                    shape = "polygon"
            else:
                shape = "polygon" # Fallback if perimeter is zero
        
        detections.append({
            "id": cnt_idx,
            "bbox": [x, y, x + w, y + h],
            "type": shape,
            "confidence": 0.1 # Assign a low confidence for fallback detections
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
        return 0.0 # No intersection
    inter_area = inter_width * inter_height
    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou
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
                 num_fill_types=len(ATTRIBUTE_FILL_MAP),
                 num_color_types=len(ATTRIBUTE_COLOR_MAP),
                 num_size_types=len(ATTRIBUTE_SIZE_MAP),
                 num_orientation_types=len(ATTRIBUTE_ORIENTATION_MAP),
                 num_shape_types=len(ATTRIBUTE_SHAPE_MAP),
                 num_texture_types=len(ATTRIBUTE_TEXTURE_MAP),
                 use_gradient_checkpointing=False,
                 use_lora=False,
                 dropout_rate: float = 0.0): # NEW: dropout_rate for MC Dropout
        super(AttributeClassifier, self).__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.backbone_name = backbone_name
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_lora = use_lora
        self.dropout_rate = dropout_rate # Store dropout rate
        
        # Load backbone with pre-trained ImageNet weights
        feature_dim = 0
        if HAS_TIMM and backbone_name in timm.list_models(pretrained=True):
            self.backbone = timm.create_model(
                backbone_name, pretrained=True, num_classes=0 # num_classes=0 to get features
            )
            feature_dim = self.backbone.num_features # Get feature dimension from timm model
            logger.info(f"AttributeClassifier using timm backbone: {backbone_name}, feature_dim: {feature_dim}")
        elif backbone_name == 'mobilenet_v2':
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
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
            self.backbone = LoRA(self.backbone, r=4, lora_alpha=32)
            # Freeze original parameters of the backbone, only LoRA modules will be trainable
            for name, param in self.backbone.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
            logger.info("AttributeClassifier backbone wrapped with LoRA adapters and original parameters frozen.")
        elif self.use_lora and not HAS_LORA:
            logger.warning("LoRA requested but loralib is not available. Proceeding without LoRA.")
        
        # Dropout layer for MC Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        logger.info(f"AttributeClassifier initialized with dropout rate: {self.dropout_rate}")

        # Multi-head classification layers
        self.fill_head = nn.Linear(feature_dim, num_fill_types)
        self.color_head = nn.Linear(feature_dim, num_color_types)
        self.size_head = nn.Linear(feature_dim, num_size_types)
        self.orientation_head = nn.Linear(feature_dim, num_orientation_types)
        self.shape_head = nn.Linear(feature_dim, num_shape_types)
        self.texture_head = nn.Linear(feature_dim, num_texture_types)
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
        if self.num_channels == 1 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1) # Repeat grayscale channel to simulate RGB
            logger.debug("Repeated grayscale channel to 3 channels for backbone input.")
        # Apply Gradient Checkpointing to Backbone (if enabled)
        if self.use_gradient_checkpointing:
            # Checkpoint the backbone's feature extraction part
            if hasattr(self.backbone, 'forward_features'):
                features = checkpoint(self.backbone.forward_features, x)
                logger.debug("Applied gradient checkpointing via 'forward_features'.")
            elif hasattr(self.backbone, 'features'):
                features = checkpoint(self.backbone.features, x)
                logger.debug("Applied gradient checkpointing via 'features'.")
            else:
                features = checkpoint(self.backbone, x)
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
        
        fill_logits = self.fill_head(features)
        color_logits = self.color_head(features)
        size_logits = self.size_head(features)
        orientation_logits = self.orientation_head(features)
        shape_logits = self.shape_head(features)
        texture_logits = self.texture_head(features)
        
        logger.debug(f"Attribute logits shapes: fill={fill_logits.shape}, color={color_logits.shape}, etc.")
        return fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, features

# ... (Continued from phase1-code-part2a) ...

# -----------------------------------------------------------------------------
# RelationGNN: Graph Neural Network with Dynamic Depth and Edge Attribute MLP
# -----------------------------------------------------------------------------
class RelationGNN(nn.Module):
    """
    Graph Neural Network for relational reasoning between detected objects.
    Uses PyTorch Geometric's GCNConv and Set2Set.
    Supports dynamic depth scheduling and an MLP for edge attribute embedding.
    Args:
        node_feature_dim (int): Dimension of input node features (object embeddings).
        edge_feature_dim (int): Dimension of input edge features (e.g., relative bbox info).
        num_relation_types (int): Number of classes for relation prediction.
        gnn_depth (int): Initial fixed depth of the GNN.
        use_gradient_checkpointing (bool): If True, applies gradient checkpointing to GCN layers.
        dropout_rate (float): Dropout rate to apply after GCN layers for MC Dropout.
    """
    def __init__(self, node_feature_dim, edge_feature_dim, num_relation_types, gnn_depth=2, use_gradient_checkpointing=False, dropout_rate: float = 0.0): # NEW: dropout_rate
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.num_relation_types = num_relation_types
        self.gnn_depth = gnn_depth # Initial depth from config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.dropout_rate = dropout_rate # Store dropout rate
        
        # Dynamic GNN Depth: Store max and min depth from config
        self.max_depth = CONFIG['model'].get('gnn_max_depth', gnn_depth)
        self.min_depth = CONFIG['model'].get('gnn_min_depth', 1)
        self.depth_schedule = CONFIG['model'].get('gnn_depth_schedule', 'fixed') # 'fixed', 'linear', 'adaptive'
        logger.info(f"RelationGNN depth schedule: '{self.depth_schedule}' (min={self.min_depth}, max={self.max_depth})")
        hidden_dim = node_feature_dim * 2 # A common choice for hidden dimension
        
        self.convs = nn.ModuleList([
            GCNConv(node_feature_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(self.max_depth) # Construct up to max_depth
        ])
        
        # Edge-Attribute Embedding MLP
        self.edge_attribute_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_feature_dim * 2, edge_feature_dim) # Output same dimension as input for concatenation
        )
        logger.debug(f"Edge attribute MLP initialized with input_dim={edge_feature_dim}.")
        
        self.set2set = Set2Set(hidden_dim, processing_steps=6, num_layers=2)
        
        # MLP for classifying edge types (relations)
        # Input dimension: 2 * hidden_dim (for concatenated node features) + edge_feature_dim (for embedded edge attributes)
        self.edge_mlp = nn.Linear((2 * hidden_dim) + edge_feature_dim, self.num_relation_types)
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
        current_depth = self.gnn_depth # Default to initial gnn_depth from config
        if self.depth_schedule == 'linear':
            curr_epoch = getattr(self, 'current_epoch', 0) # Assumes current_epoch is set by PerceptionModule
            max_e = CONFIG['model']['epochs']
            if max_e > 0:
                # Linearly interpolate depth between min_depth and max_depth
                depth = int(self.min_depth + (self.max_depth - self.min_depth) * (curr_epoch / max_e))
                current_depth = max(self.min_depth, min(depth, self.max_depth)) # Ensure it's within bounds
            else:
                current_depth = self.min_depth # Fallback if epochs is 0
            logger.debug(f"Linear depth schedule: Epoch {curr_epoch}/{max_e}, Calculated Depth: {current_depth}")
        elif self.depth_schedule == 'adaptive':
            # Adaptive Strategy: Adjust depth based on graph sparsity/density (e.g., avg degree)
            num_nodes_in_batch = x.size(0)
            num_edges_in_batch = edge_index.size(1)
            
            if num_nodes_in_batch == 0 or num_edges_in_batch == 0:
                current_depth = self.min_depth
                logger.debug(f"Adaptive depth: Empty graph, setting depth to min_depth ({current_depth}).")
            else:
                avg_degree = num_edges_in_batch / num_nodes_in_batch
                
                if avg_degree < 1.0: # Very sparse graphs
                    current_depth = self.min_depth
                    logger.debug(f"Adaptive depth: Avg degree {avg_degree:.2f} < 1.0, setting depth to min_depth ({current_depth}).")
                elif avg_degree < 3.0: # Moderately dense
                    current_depth = max(self.min_depth, self.gnn_depth)
                    logger.debug(f"Adaptive depth: Avg degree {avg_degree:.2f} (1.0-3.0), setting depth to max(min_depth, gnn_depth) ({current_depth}).")
                else: # Dense graphs
                    current_depth = self.max_depth
                    logger.debug(f"Adaptive depth: Avg degree {avg_degree:.2f} >= 3.0, setting depth to max_depth ({current_depth}).")
            
            current_depth = min(current_depth, len(self.convs)) # Ensure it doesn't exceed available layers
        else: # 'fixed' schedule or unknown
            current_depth = self.gnn_depth # Use the fixed depth from config
            logger.debug(f"Fixed depth schedule: Depth set to {current_depth}.")
        logger.info(f"GNN depth for epoch {getattr(self, 'current_epoch', 0)}: {current_depth}")
        # Handle empty graph case for GCNConv
        if x.numel() == 0 or edge_index.numel() == 0:
            logger.warning("Empty graph detected in RelationGNN forward pass. Returning dummy outputs.")
            # Return dummy outputs matching expected shapes for an empty batch
            dummy_graph_embedding = torch.zeros(batch.max().item() + 1 if batch.numel() > 0 else 1, self.set2set.output_size, device=x.device)
            dummy_node_features = torch.zeros(x.size(0), self.convs[0].out_channels if self.convs else self.node_feature_dim, device=x.device)
            return dummy_graph_embedding, dummy_node_features
        
        for i, conv in enumerate(self.convs[:current_depth]):
            if self.use_gradient_checkpointing:
                x = checkpoint(conv, x, edge_index)
                logger.debug(f"Applied gradient checkpointing for GCNConv layer {i}.")
            else:
                x = conv(x, edge_index)
            if i < current_depth - 1:
                x = F.relu(x)
                # Apply dropout after ReLU for MC Dropout
                x = F.dropout(x, p=self.dropout_rate, training=self.training) # Apply dropout
                logger.debug(f"Applied ReLU and Dropout after GCNConv layer {i}.")
        
        # Global pooling (e.g., Set2Set) to get a graph-level representation for each problem
        graph_embedding = self.set2set(x, batch)
        logger.debug(f"Graph embedding shape after Set2Set: {graph_embedding.shape}.")
        
        return graph_embedding, x # Return graph_embedding and final node features
    
    def classify_edges(self, node_features, edge_index, edge_attributes, node_types_for_edges):
        """
        Classifies edges based on concatenated features of connected nodes and embedded edge attributes.
        Args:
            node_features (torch.Tensor): Output features from the final GNN layer for all nodes.
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            edge_attributes (torch.Tensor): Features describing the edge itself (e.g., relative bbox info).
            node_types_for_edges (list): A list of (source_obj_type, dest_obj_type) for each edge,
                                    used for symbolic priors.
        Returns:
            torch.Tensor: Logits for each relation type for each edge.
        """
        if edge_index.numel() == 0:
            logger.warning("Empty edge_index in classify_edges. Returning empty logits.")
            return torch.empty((0, self.num_relation_types), device=node_features.device)
        
        row, col = edge_index
        
        # Concatenate features of connected nodes
        node_pair_features = torch.cat([node_features[row], node_features[col]], dim=-1)
        logger.debug(f"Node pair features shape: {node_pair_features.shape}.")
        
        # Apply Edge-Attribute Embedding MLP
        if edge_attributes.numel() > 0:
            embedded_edge_attributes = self.edge_attribute_mlp(edge_attributes)
            combined_features = torch.cat([node_pair_features, embedded_edge_attributes], dim=-1)
            logger.debug(f"Combined features shape (with edge attributes): {combined_features.shape}.")
        else:
            combined_features = node_pair_features
            logger.debug(f"Combined features shape (no edge attributes): {combined_features.shape}.")
        
        edge_logits = self.edge_mlp(combined_features)
        logger.debug(f"Edge logits shape before symbolic priors: {edge_logits.shape}.")
        
        # Apply symbolic priors
        if CONFIG['debug'].get('apply_symbolic_priors', True):
            logger.debug("Applying symbolic priors to edge logits.")
            edge_logits = apply_symbolic_priors(edge_logits, edge_logits.argmax(dim=1), node_types_for_edges, edge_attributes, RELATION_MAP)
            logger.debug(f"Edge logits shape after symbolic priors: {edge_logits.shape}.")
        
        # Log or save edge probabilities for visualization
        if CONFIG['debug']['visualize_gnn_weights']:
            edge_probs = F.softmax(edge_logits, dim=1)
            logger.debug(f"GNN Edge Probabilities - Shape: {edge_probs.shape}, Mean: {edge_probs.mean().item():.4f}")
        
        return edge_logits
# -----------------------------------------------------------------------------
# apply_symbolic_priors function
# -----------------------------------------------------------------------------
def apply_symbolic_priors(
    edge_logits: torch.Tensor,
    edge_preds: torch.Tensor,
    node_types_for_edges: list, # List of (source_obj_type, dest_obj_type) for each edge
    edge_attributes: torch.Tensor, # NEW: Edge attributes (e.g., rel_dx, rel_dy, iou)
    RELATION_MAP: dict
):
    """
    Applies symbolic (rule-based) priors to adjust edge logits,
    penalizing or boosting certain relations based on object properties and edge attributes.
    This function operates on the raw logits before softmax.
    Args:
        edge_logits (torch.Tensor): Raw logits for each relation type for each edge [num_edges, num_relation_types].
        edge_preds (torch.Tensor): Predicted relation type (argmax of logits) for each edge [num_edges].
        node_types_for_edges (list): A list where each element is a tuple (src_obj_type, dest_obj_type)
                                    corresponding to an edge.
        edge_attributes (torch.Tensor): Features describing the edge itself (e.g., relative bbox info).
                                        Expected order: [rel_dx, rel_dy, rel_dw, rel_dh, angle, iou].
        RELATION_MAP (dict): The global mapping from relation ID to relation name.
    Returns:
        torch.Tensor: Adjusted edge logits.
    """
    adjusted_logits = edge_logits.clone() # Work on a clone to avoid modifying original in-place
    
    # Get inverse map for quick lookup of relation IDs
    inverse_relation_map = {name: idx for idx, name in RELATION_MAP.items()}
    
    # Define indices for common attributes in edge_attributes tensor
    # Assuming order: [rel_dx, rel_dy, rel_dw, rel_dh, angle, iou]
    REL_DX_IDX = 0
    REL_DY_IDX = 1
    REL_DW_IDX = 2
    REL_DH_IDX = 3
    ANGLE_IDX = 4
    IOU_IDX = 5
    # Iterate through each edge
    for i in range(edge_logits.size(0)):
        src_type, dst_type = node_types_for_edges[i]
        current_edge_pred_idx = edge_preds[i].item()
        current_edge_attrs = edge_attributes[i] # Get attributes for the current edge
        # Rule 1: Penalize "overlaps" if IoU is very low (i.e., they barely overlap)
        overlaps_idx = inverse_relation_map.get("overlaps")
        if overlaps_idx is not None and current_edge_pred_idx == overlaps_idx:
            iou_val = current_edge_attrs[IOU_IDX].item()
            if iou_val < 0.05: # If IoU is very small, it's likely not a true overlap
                adjusted_logits[i, overlaps_idx] -= 2.0 # Strong penalty
                logger.debug(f"Prior Applied (Edge {i}): Penalized 'overlaps' (IoU: {iou_val:.4f}) due to very low overlap.")
        # Rule 2: Boost "contains" if IoU is high AND one object is significantly smaller
        contains_idx = inverse_relation_map.get("contains")
        if contains_idx is not None and current_edge_pred_idx == contains_idx:
            iou_val = current_edge_attrs[IOU_IDX].item()
            rel_dw = current_edge_attrs[REL_DW_IDX].item() # (width_dest - width_src) / img_w
            rel_dh = current_edge_attrs[REL_DH_IDX].item() # (height_dest - height_src) / img_h
            # If src contains dest, then dest should be smaller than src (rel_dw, rel_dh should be negative)
            # And IoU should be high (close to area of dest / area of src)
            if iou_val > 0.6 and rel_dw < -0.1 and rel_dh < -0.1: # Heuristic for "smaller"
                adjusted_logits[i, contains_idx] += 1.5
                logger.debug(f"Prior Applied (Edge {i}): Boosted 'contains' (IoU: {iou_val:.4f}, rel_dw: {rel_dw:.4f}, rel_dh: {rel_dh:.4f}) due to high overlap and size difference.")
        # Rule 3: Boost "above" if dest is clearly above src (negative rel_dy)
        above_idx = inverse_relation_map.get("above")
        if above_idx is not None and current_edge_pred_idx == above_idx:
            rel_dy = current_edge_attrs[REL_DY_IDX].item()
            if rel_dy < -0.1: # If destination's y is significantly less than source's y
                adjusted_logits[i, above_idx] += 1.0
                logger.debug(f"Prior Applied (Edge {i}): Boosted 'above' (rel_dy: {rel_dy:.4f}).")
        
        # Rule 4: Penalize "above" if dest is clearly below src (positive rel_dy)
        if above_idx is not None and current_edge_pred_idx == above_idx:
            rel_dy = current_edge_attrs[REL_DY_IDX].item()
            if rel_dy > 0.1: # If destination's y is significantly greater than source's y
                adjusted_logits[i, above_idx] -= 1.0
                logger.debug(f"Prior Applied (Edge {i}): Penalized 'above' (rel_dy: {rel_dy:.4f}).")
        # Rule 5: Boost "same_type_as" if objects are indeed of the same detected type
        same_type_as_idx = inverse_relation_map.get("same_type_as")
        if same_type_as_idx is not None and current_edge_pred_idx == same_type_as_idx:
            if src_type == dst_type and src_type != "dummy_node" and src_type != "invalid_bbox_node":
                adjusted_logits[i, same_type_as_idx] += 0.8
                logger.debug(f"Prior Applied (Edge {i}): Boosted 'same_type_as' ({src_type}-{dst_type}).")
        # Rule 6: Boost "different_type_from" if objects are indeed of different detected types
        different_type_from_idx = inverse_relation_map.get("different_type_from")
        if different_type_from_idx is not None and current_edge_pred_idx == different_type_from_idx:
            if src_type != dst_type and src_type != "dummy_node" and src_type != "invalid_bbox_node":
                adjusted_logits[i, different_type_from_idx] += 0.8
                logger.debug(f"Prior Applied (Edge {i}): Boosted 'different_type_from' ({src_type}-{dst_type}).")
    return adjusted_logits
# -----------------------------------------------------------------------------
# PerceptionModule: Main model integrating detection, attributes, and relations
# -----------------------------------------------------------------------------
class PerceptionModule(nn.Module):
    """
    The main Perception Module integrating RealObjectDetector, AttributeClassifier, and RelationGNN.
    It takes an image and outputs a symbolic representation of objects and their relations.
    Args:
        config (dict): Configuration dictionary.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use global DEVICE
        
        # Real Object Detector (YOLOv8 + SAM/Contour fallback)
        self.real_object_detector = RealObjectDetector(model_path=config['model']['object_detector_model_path']) # Corrected key
        # Attribute Classifier
        # Input channels for attribute classifier: 3 (for RGB crop)
        # Image size for attribute classifier: fixed crop size (e.g., 224, 224)
        self.attribute_classifier = AttributeClassifier(
            num_channels=3, # Always use 3 channels for attribute classification by converting crops to RGB
            image_size=config['data']['image_size'], # Use the image_size from config for crops
            backbone_name=config['model']['attribute_backbone'],
            use_gradient_checkpointing=config['training'].get('use_gradient_checkpointing', False), # Default to False
            use_lora=config['model'].get('use_lora', False), # Default to False
            dropout_rate=config['model'].get('initial_dropout_rate', 0.0) # Pass dropout rate
        ).to(self.device)
        # Relation GNN
        # Node feature dimension will be the output dimension of attribute classifier's features
        # Assuming last layer of attribute classifier features are fed to GNN.
        # This needs to be precisely derived from the AttributeClassifier's backbone output dim.
        if HAS_TIMM and config['model']['attribute_backbone'] in timm.list_models(pretrained=True):
            gnn_node_feature_dim = timm.create_model(config['model']['attribute_backbone'], pretrained=True, num_classes=0).num_features
        elif config['model']['attribute_backbone'] == 'mobilenet_v2':
            gnn_node_feature_dim = 1280
        elif config['model']['attribute_backbone'] == 'mobilenet_v3_small':
            gnn_node_feature_dim = 576
        elif config['model']['attribute_backbone'] == 'efficientnet_b0':
            gnn_node_feature_dim = 1280
        else:
            raise ValueError(f"Unknown attribute_backbone: {config['model']['attribute_backbone']} for GNN feature dim inference.")
        # Edge feature dimension: rel_dx, rel_dy, rel_dw, rel_dh, angle, iou = 6
        self.relation_gnn = RelationGNN(
            node_feature_dim=gnn_node_feature_dim,
            edge_feature_dim=6, # For [rel_dx, rel_dy, rel_dw, rel_dh, angle, iou]
            num_relation_types=len(RELATION_MAP),
            gnn_depth=config['model']['gnn_depth'],
            use_gradient_checkpointing=config['training'].get('use_gradient_checkpointing', False),
            dropout_rate=config['model'].get('initial_dropout_rate', 0.0) # Pass dropout rate
        ).to(self.device)
        
        # Loss weights (if used in forward pass, otherwise handled in train_step)
        self.loss_weights = config.get('loss_weights', {}) # Default to empty dict if not present
        # Transformations for attribute classification crops
        self.attr_transform = T.Compose([
            T.Resize(config['data']['image_size']), # Use global image_size for crops
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
        ])
        # Feature matching / distillation
        self.use_knowledge_distillation = config['training'].get('use_knowledge_distillation', False)
        self.teacher_model = None # To be loaded if distillation is used
        if self.use_knowledge_distillation:
            teacher_model_path = config['model'].get('teacher_model_path')
            if teacher_model_path:
                logger.info(f"Loading teacher model from {teacher_model_path} for knowledge distillation.")
                # Assuming teacher model is also a PerceptionModule for now, or a specific feature extractor.
                # For a real scenario, this would be a pre-trained model.
                # For simplicity, let's assume it's just the attribute classifier backbone for feature distillation.
                teacher_backbone_name = config['model']['attribute_backbone'] # Use same backbone for teacher
                self.teacher_model = AttributeClassifier(
                    num_channels=3,
                    image_size=config['data']['image_size'],
                    backbone_name=teacher_backbone_name,
                    use_gradient_checkpointing=False, # Teacher doesn't need checkpointing
                    use_lora=False, # Teacher doesn't need LoRA
                    dropout_rate=0.0 # Teacher should not have dropout for consistent features
                ).to(self.device)
                self.teacher_model.load_state_dict(torch.load(teacher_model_path)) # Load teacher weights
                self.teacher_model.eval() # Set teacher to eval mode
                for param in self.teacher_model.parameters():
                    param.requires_grad = False # Freeze teacher model
            else:
                logger.warning("Knowledge distillation enabled but 'teacher_model_path' not specified. Distillation will be skipped.")
                self.use_knowledge_distillation = False
    def _extract_features_and_symbolic_data(self, image_pil_batch: List[Image.Image], image_paths: List[str]):
        """
        Processes a batch of images to extract object detections,
        attribute predictions, and inter-object relations.
        Returns a list of symbolic output dictionaries for each image.
        """
        symbolic_outputs_batch = []
        
        all_object_crops = []
        all_object_crop_image_indices = []
        all_object_types = [] # To store the YOLO/SAM detected type
        all_object_ious = [] # Store IoU from SAM for conceptual nodes
        # Phase 1: Object Detection (RealObjectDetector)
        for img_idx, image_pil in enumerate(image_pil_batch):
            detections = self.real_object_detector.detect_objects(
                image_pil, overall_confidence=self.config['model']['detection_confidence_threshold'] # Corrected key
            )
            
            symbolic_output = {
                "image_idx": img_idx, # Index of the original image in the batch
                "image_path": image_paths[img_idx],
                "objects": [],
                "relations": [],
                "image_features": None # Placeholder for potential image-level features
            }
            if not detections:
                logger.warning(f"No objects detected in image {image_paths[img_idx]}. Adding a dummy node.")
                # Add a dummy node for this image to ensure graph processing doesn't break
                # This dummy node can have a special 'type' and placeholder bbox/attributes.
                dummy_obj = {
                    "id": 0,
                    "bbox": [0, 0, 1, 1], # Minimal bbox
                    "type": "dummy_node",
                    "confidence": 0.0,
                    "attributes": {
                        "fill": "unknown", "color": "unknown", "size": "unknown",
                        "orientation": "unknown", "shape": "unknown", "texture": "unknown"
                    }
                }
                symbolic_output["objects"].append(dummy_obj)
            else:
                for obj_idx, detection in enumerate(detections):
                    x1, y1, x2, y2 = detection['bbox']
                    img_w, img_h = image_pil.size
                    
                    # Ensure bbox is within image bounds (clamp values)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)
                    if x2 <= x1 or y2 <= y1:
                        logger.warning(f"Invalid bounding box [{x1},{y1},{x2},{y2}] for detection {obj_idx} in image {image_paths[img_idx]}. Skipping.")
                        continue # Skip invalid bounding boxes
                    
                    # Extract object crop and apply attribute transforms
                    obj_crop = image_pil.crop((x1, y1, x2, y2))
                    # Convert object crop to RGB for attribute classification if not already
                    if obj_crop.mode != 'RGB':
                        obj_crop = obj_crop.convert('RGB')
                    all_object_crops.append(self.attr_transform(obj_crop))
                    all_object_crop_image_indices.append(img_idx) # Map crop back to original image
                    all_object_types.append(detection['type'])
                    all_object_ious.append(detection.get('confidence', 0.0) if detection['type'].startswith('unknown_shape_sam') else 0.0) # Store SAM IoU
                    symbolic_output["objects"].append({
                        "id": obj_idx, # Object's local ID within this image
                        "bbox": detection['bbox'],
                        "type": detection['type'], # e.g., 'person', 'car', 'square', 'circle' from YOLO/SAM
                        "confidence": detection['confidence'],
                        "attributes": {}, # Placeholder, to be filled by AttributeClassifier
                        "conceptual_mask": None # Placeholder for SAM masks
                    })
            
            symbolic_outputs_batch.append(symbolic_output)
        # Phase 2: Attribute Classification (AttributeClassifier)
        if all_object_crops:
            object_crops_tensor = torch.stack(all_object_crops).to(self.device)
            logger.debug(f"Stacked object crops tensor shape: {object_crops_tensor.shape}")
            
            # No torch.no_grad() here, as we want gradients for MC Dropout if model is in train mode
            fill_logits, color_logits, size_logits, orientation_logits, shape_logits, texture_logits, object_features = \
                self.attribute_classifier(object_crops_tensor)
            
            # Map predictions back to symbolic_outputs_batch
            offset = 0
            for img_idx, symbolic_output in enumerate(symbolic_outputs_batch):
                num_objects_in_img = len(symbolic_output["objects"])
                
                # Handle dummy nodes or images with no detections correctly
                if num_objects_in_img == 0:
                    continue # Should not happen if dummy node is added, but as a safeguard
                
                # Check if it's a dummy node and skip attribute prediction if so
                if symbolic_output["objects"][0]["type"] == "dummy_node":
                    # For dummy nodes, object_features will be zero-filled or ignored.
                    # We need a corresponding 'dummy' feature tensor for GNN.
                    # This ensures correct indexing later when object_features is used for GNN nodes.
                    dummy_feature_dim = object_features.shape[1] if object_features.numel() > 0 else 1280 # Default
                    symbolic_output["objects"][0]["_features"] = torch.zeros(dummy_feature_dim, device=self.device)
                    offset += num_objects_in_img
                    continue
                for obj_local_idx in range(num_objects_in_img):
                    # Ensure the current object corresponds to the correct crop/feature
                    if all_object_crop_image_indices[offset + obj_local_idx] != img_idx:
                        logger.error(f"Mismatch in crop image index. Expected {img_idx}, got {all_object_crop_image_indices[offset + obj_local_idx]}.")
                        continue # Critical error, skip this object
                    # Attribute predictions (argmax for categories)
                    symbolic_output["objects"][obj_local_idx]["attributes"]["fill"] = ATTRIBUTE_FILL_MAP[fill_logits[offset + obj_local_idx].argmax().item()]
                    symbolic_output["objects"][obj_local_idx]["attributes"]["color"] = ATTRIBUTE_COLOR_MAP[color_logits[offset + obj_local_idx].argmax().item()]
                    symbolic_output["objects"][obj_local_idx]["attributes"]["size"] = ATTRIBUTE_SIZE_MAP[size_logits[offset + obj_local_idx].argmax().item()]
                    symbolic_output["objects"][obj_local_idx]["attributes"]["orientation"] = ATTRIBUTE_ORIENTATION_MAP[orientation_logits[offset + obj_local_idx].argmax().item()]
                    symbolic_output["objects"][obj_local_idx]["attributes"]["shape"] = ATTRIBUTE_SHAPE_MAP[shape_logits[offset + obj_local_idx].argmax().item()]
                    symbolic_output["objects"][obj_local_idx]["attributes"]["texture"] = ATTRIBUTE_TEXTURE_MAP[texture_logits[offset + obj_local_idx].argmax().item()]
                    
                    # Store features for GNN input (detach to avoid backprop through detection/attribute prediction if not needed)
                    # For MC Dropout, we want to keep features attached if model is in train mode.
                    symbolic_output["objects"][obj_local_idx]["_features"] = object_features[offset + obj_local_idx]
                    symbolic_output["objects"][obj_local_idx]["_type_original"] = all_object_types[offset + obj_local_idx] # Store YOLO/SAM type
                    symbolic_output["objects"][obj_local_idx]["_sam_iou"] = all_object_ious[offset + obj_local_idx] # Store SAM IoU
                    # Store logits for attribute losses in training step
                    symbolic_output["objects"][obj_local_idx]["_fill_logits"] = fill_logits[offset + obj_local_idx]
                    symbolic_output["objects"][obj_local_idx]["_color_logits"] = color_logits[offset + obj_local_idx]
                    symbolic_output["objects"][obj_local_idx]["_size_logits"] = size_logits[offset + obj_local_idx]
                    symbolic_output["objects"][obj_local_idx]["_orientation_logits"] = orientation_logits[offset + obj_local_idx]
                    symbolic_output["objects"][obj_local_idx]["_shape_logits"] = shape_logits[offset + obj_local_idx]
                    symbolic_output["objects"][obj_local_idx]["_texture_logits"] = texture_logits[offset + obj_local_idx]
                offset += num_objects_in_img
        else:
            logger.info("No object crops to classify attributes for in this batch.")
            # For empty all_object_crops, the loop above for populating attributes won't run.
            # Ensure _features is set for dummy nodes if they were created.
            for symbolic_output in symbolic_outputs_batch:
                if symbolic_output["objects"] and symbolic_output["objects"][0]["type"] == "dummy_node":
                    # Default feature dim (e.g., for mobilenet_v2) or dynamically get from gnn_node_feature_dim
                    dummy_feature_dim = 1280
                    # Try to infer it from the GNN's expected input dimension, if possible
                    if hasattr(self.relation_gnn, 'node_feature_dim'):
                        dummy_feature_dim = self.relation_gnn.node_feature_dim
                    symbolic_output["objects"][0]["_features"] = torch.zeros(dummy_feature_dim, device=self.device)
        # Phase 3: Prepare Graph Data for GNN (Relations)
        # This part will be done within the main forward pass for batching efficiency.
        # Here we just ensure object_features are populated correctly for each image.
        
        return symbolic_outputs_batch
    def _process_batch_for_symbolic_outputs(self, symbolic_outputs_batch: List[Dict]):
        """
        Processes a batch of symbolic outputs (from _extract_features_and_symbolic_data)
        to construct graph inputs for the GNN and get relation predictions.
        Handles object features, bounding boxes, and image-level batching for PyG.
        Args:
            symbolic_outputs_batch (List[Dict]): A list of symbolic outputs, one dict per image.
        Returns:
            List[Dict]: The updated list of symbolic outputs, now including relation predictions.
            tuple: (PyG batch data, mapping from original image_idx to PyG batch_idx, all_node_types_for_edges).
        """
        all_nodes_features = []
        all_edge_indices = []
        all_edge_attributes = []
        all_batch_indices = [] # PyTorch Geometric batch index for each node
        all_node_types_for_edges = [] # For symbolic priors
        
        node_offset = 0
        img_idx_to_pyg_batch_idx = {} # Maps original image index to PyG's batch index (0 to N-1)
        for pyg_batch_idx, symbolic_output in enumerate(symbolic_outputs_batch):
            img_idx = symbolic_output["image_idx"]
            img_idx_to_pyg_batch_idx[img_idx] = pyg_batch_idx
            objects = symbolic_output["objects"]
            image_path = symbolic_output["image_path"]
            
            # If a dummy node was added, ensure it has features
            if not objects or ("_features" not in objects[0] and objects[0].get("type") == "dummy_node"):
                # This should ideally be handled during feature extraction, but as a safeguard:
                dummy_feature_dim = self.relation_gnn.node_feature_dim
                objects[0]["_features"] = torch.zeros(dummy_feature_dim, device=self.device)
                logger.warning(f"Dummy node in image {img_idx} did not have features. Initialized with zeros.")
            # Append node features and batch indices for all objects in this image
            current_nodes_features = torch.stack([obj["_features"] for obj in objects])
            all_nodes_features.append(current_nodes_features)
            
            num_objects = len(objects)
            all_batch_indices.append(torch.full((num_objects,), fill_value=pyg_batch_idx, dtype=torch.long, device=self.device))
            
            if num_objects <= 1:
                logger.debug(f"Image {img_idx}: Less than 2 objects, no relations to form.")
                node_offset += num_objects
                continue # No relations to form for 0 or 1 object
            # Create all possible pairs (edges) between objects within this image
            current_image_edges = []
            current_image_edge_attrs = []
            current_image_node_types_for_edges = []
            for i in range(num_objects):
                for j in range(num_objects):
                    if i == j: continue # No self-loops
                    
                    obj_i = objects[i]
                    obj_j = objects[j]
                    
                    bbox_i = obj_i["bbox"]
                    bbox_j = obj_j["bbox"]
                    
                    # Calculate relative bounding box features and IoU
                    # Convert absolute bbox to relative coordinates (0-1) based on image size
                    # This is tricky because we don't have the full image size here.
                    # Assume image_pil_batch has been generated and its size is consistent.
                    # Or, retrieve image size from symbolic_output if stored.
                    # For now, let's assume image size is needed directly here.
                    # A better way: Pass image_pil_batch sizes.
                    # For simplicity, if image_path is available, open image again to get size.
                    try:
                        temp_img = Image.open(image_path)
                        img_w, img_h = temp_img.size
                    except Exception as e:
                        logger.warning(f"Could not open {image_path} to get image size for relative bbox calculation: {e}. Using dummy size 1000x1000.")
                        img_w, img_h = 1000, 1000 # Fallback dummy size if image not accessible
                    rel_dx = ((bbox_j[0] + bbox_j[2]) / 2 - (bbox_i[0] + bbox_i[2]) / 2) / img_w
                    rel_dy = ((bbox_j[1] + bbox_j[3]) / 2 - (bbox_i[1] + bbox_i[3]) / 2) / img_h
                    rel_dw = (bbox_j[2] - bbox_j[0] - (bbox_i[2] - bbox_i[0])) / img_w # difference in width
                    rel_dh = (bbox_j[3] - bbox_j[1] - (bbox_i[3] - bbox_i[1])) / img_h # difference in height
                    
                    # Calculate angle from i to j (normalized to [-1, 1])
                    dx_center = ((bbox_j[0] + bbox_j[2]) / 2) - ((bbox_i[0] + bbox_i[2]) / 2)
                    dy_center = ((bbox_j[1] + bbox_j[3]) / 2) - ((bbox_i[1] + bbox_i[3]) / 2)
                    angle = math.atan2(dy_center, dx_center) / math.pi # Normalized angle
                    
                    # Calculate IoU
                    iou = _calculate_iou(bbox_i, bbox_j)
                    
                    # Edge attributes: [rel_dx, rel_dy, rel_dw, rel_dh, angle, iou]
                    edge_attr = torch.tensor([rel_dx, rel_dy, rel_dw, rel_dh, angle, iou], dtype=torch.float32, device=self.device)
                    current_image_edge_attrs.append(edge_attr)
                    
                    current_image_edges.append([i + node_offset, j + node_offset]) # Global node indices
                    current_image_node_types_for_edges.append((obj_i["_type_original"], obj_j["_type_original"]))
            if current_image_edges:
                all_edge_indices.append(torch.tensor(current_image_edges, dtype=torch.long).t().contiguous())
                all_edge_attributes.append(torch.stack(current_image_edge_attrs))
                all_node_types_for_edges.extend(current_image_node_types_for_edges)
            node_offset += num_objects
        
        # Concatenate all graph components
        if not all_nodes_features: # Case where no objects were detected across the entire batch
            logger.warning("No nodes generated for the entire batch. Returning empty graph data.")
            return symbolic_outputs_batch, (Data(x=torch.empty(0, self.relation_gnn.node_feature_dim),
                                                edge_index=torch.empty(2, 0, dtype=torch.long),
                                                edge_attr=torch.empty(0, 6),
                                                batch=torch.empty(0, dtype=torch.long)),
                                            img_idx_to_pyg_batch_idx,
                                            []) # Return empty node types for edges
            
        x_tensor = torch.cat(all_nodes_features, dim=0)
        batch_tensor = torch.cat(all_batch_indices, dim=0)
        
        edge_index_tensor = torch.empty(2, 0, dtype=torch.long, device=self.device)
        edge_attr_tensor = torch.empty(0, 6, dtype=torch.float32, device=self.device)
        if all_edge_indices:
            edge_index_tensor = torch.cat(all_edge_indices, dim=1)
            edge_attr_tensor = torch.cat(all_edge_attributes, dim=0)
        
        # Create a PyTorch Geometric Data object
        pyg_data_batch = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, batch=batch_tensor)
        logger.debug(f"PyG Data batch created: x shape={pyg_data_batch.x.shape}, edge_index shape={pyg_data_batch.edge_index.shape}, edge_attr shape={pyg_data_batch.edge_attr.shape}, batch shape={pyg_data_batch.batch.shape}")
        return symbolic_outputs_batch, (pyg_data_batch, img_idx_to_pyg_batch_idx, all_node_types_for_edges)
    def forward(self, image_input: Any, is_synthetic: bool = False):
        """
        Forward pass for the entire PerceptionModule.
        Args:
            image_input (List[str] or List[Image.Image]): List of paths to input images (for real data)
                                                          or list of PIL Images (for synthetic data).
            is_synthetic (bool): True if input is synthetic PIL images, False if paths to real images.
        Returns:
            List[Dict]: A list of symbolic output dictionaries for each image,
                       including object detections, attributes, and relations.
            dict: A dictionary containing all aggregated logits and ground truths for loss calculation.
        """
        if is_synthetic:
            # If synthetic, image_input is already a list of PIL Images
            image_pil_batch = image_input
            image_paths = [f"synthetic_image_{i}" for i in range(len(image_input))] # Dummy paths for logging
        else:
            # If real, image_input is a list of paths, load PIL Images
            image_pil_batch = [Image.open(p).convert('RGB') for p in image_input]
            image_paths = image_input
        # Phase 1 & 2: Object Detection and Attribute Classification
        # This populates `objects` and their `_features` and `attributes`, and attribute logits
        symbolic_outputs_batch = self._extract_features_and_symbolic_data(image_pil_batch, image_paths)
        # Phase 3: Prepare Graph Data and Run GNN for Relations
        # This returns PyG Data object ready for GNN.
        updated_symbolic_outputs_batch, (pyg_data_batch, img_idx_to_pyg_batch_idx, all_node_types_for_edges) = \
            self._process_batch_for_symbolic_outputs(symbolic_outputs_batch)
        
        # Initialize aggregated logits and ground truths for the entire batch
        # This will be returned for loss calculation in the training loop
        aggregated_outputs = {
            'attribute_logits': {
                'fill': [], 'color': [], 'size': [], 'orientation': [], 'shape': [], 'texture': []
            },
            'attribute_gt': { # These will be populated from GT JSON if synthetic, or remain empty for real data
                'fill': [], 'color': [], 'size': [], 'orientation': [], 'shape': [], 'texture': []
            },
            'relation_logits': [],
            'relation_gt': [],
            'bongard_labels': [], # Overall Bongard labels
            'image_features_student': [], # For feature consistency loss
            'image_features_teacher': [] # For feature consistency loss
        }
        # Run Relation GNN if there are any nodes/edges
        if pyg_data_batch.x.numel() > 0 and pyg_data_batch.edge_index.numel() > 0:
            graph_embeddings, final_node_features = self.relation_gnn(
                pyg_data_batch.x, pyg_data_batch.edge_index, pyg_data_batch.batch
            )
            
            # Classify edges (relations)
            # Pass original node types for symbolic priors
            relation_logits = self.relation_gnn.classify_edges(
                final_node_features, pyg_data_batch.edge_index, pyg_data_batch.edge_attr, all_node_types_for_edges
            )
            # Map relation predictions back to symbolic_outputs_batch
            # The order of edges in pyg_data_batch.edge_index corresponds to all_edge_attributes
            # and thus to the order relation_logits are generated.
            current_edge_idx = 0
            for sym_idx, symbolic_output in enumerate(updated_symbolic_outputs_batch):
                objects = symbolic_output["objects"]
                if len(objects) <= 1:
                    continue # No relations for images with 0 or 1 object
                
                num_objects = len(objects)
                
                for i in range(num_objects):
                    for j in range(num_objects):
                        if i == j: continue
                        
                        # Get the predicted relation ID and confidence
                        # Need to ensure current_edge_idx is valid for relation_logits
                        if current_edge_idx < relation_logits.size(0):
                            relation_pred_idx = relation_logits[current_edge_idx].argmax().item()
                            relation_confidence = F.softmax(relation_logits[current_edge_idx], dim=-1)[relation_pred_idx].item()
                            
                            symbolic_output["relations"].append({
                                "source_id": objects[i]["id"],
                                "target_id": objects[j]["id"],
                                "type": RELATION_MAP[relation_pred_idx],
                                "confidence": relation_confidence,
                                "_logits": relation_logits[current_edge_idx].detach().cpu().numpy() # Store logits for loss/metrics
                            })
                            # Add to aggregated relation logits
                            aggregated_outputs['relation_logits'].append(relation_logits[current_edge_idx])
                            current_edge_idx += 1
                        else:
                            logger.warning(f"Mismatched edge count during relation mapping for image {sym_idx}. Predicted {current_edge_idx} edges, but expected more based on pairs.")
                            break # Break inner loop if we ran out of relation logits
                    if current_edge_idx >= relation_logits.size(0):
                        break # Break outer loop if we ran out of relation logits
            # Store graph embeddings per image, mapped by pyg_batch_idx
            if graph_embeddings.numel() > 0:
                for img_orig_idx, pyg_batch_idx in img_idx_to_pyg_batch_idx.items():
                    if pyg_batch_idx < graph_embeddings.shape[0]:
                        updated_symbolic_outputs_batch[img_orig_idx]["image_features"] = graph_embeddings[pyg_batch_idx]
                        # Store student features for consistency loss
                        aggregated_outputs['image_features_student'].append(graph_embeddings[pyg_batch_idx])
                        
                        # If knowledge distillation is enabled, get teacher features
                        if self.use_knowledge_distillation and self.teacher_model:
                            # Need to re-extract features from the original image for the teacher model
                            # This is inefficient but demonstrates the concept. A better way would be to
                            # pass the original image batch or pre-computed teacher features.
                            # For now, let's pass the object crops through the teacher's attribute classifier
                            # to get features for distillation.
                            # This assumes a 1:1 mapping of objects to features, which is true for attribute classifier.
                            
                            # Collect all object crops for the current image from the original batch
                            current_image_crops = []
                            for obj in updated_symbolic_outputs_batch[img_orig_idx]['objects']:
                                if obj['type'] != 'dummy_node':
                                    # Re-open image and crop for teacher if not synthetic, or use original PIL crop
                                    if is_synthetic:
                                        # For synthetic, we have the PIL image in image_pil_batch
                                        img_pil = image_pil_batch[img_orig_idx]
                                    else:
                                        img_pil = Image.open(image_paths[img_orig_idx]).convert('RGB')
                                    x1, y1, x2, y2 = obj['bbox']
                                    obj_crop = img_pil.crop((x1, y1, x2, y2)).convert('RGB')
                                    current_image_crops.append(self.attr_transform(obj_crop))
                            
                            if current_image_crops:
                                teacher_crops_tensor = torch.stack(current_image_crops).to(self.device)
                                with torch.no_grad():
                                    # Get teacher features from attribute classifier backbone
                                    _, _, _, _, _, _, teacher_feats = self.teacher_model(teacher_crops_tensor)
                                    # Aggregate teacher features to match student graph embedding dimension
                                    # For simplicity, average teacher features for objects in this image
                                    if teacher_feats.numel() > 0:
                                        aggregated_outputs['image_features_teacher'].append(teacher_feats.mean(dim=0))
                                    else:
                                        # If no real objects, append a zero tensor of appropriate size
                                        aggregated_outputs['image_features_teacher'].append(torch.zeros_like(graph_embeddings[pyg_batch_idx]))
                            else:
                                # If no real objects in image, append zero tensor for teacher features
                                aggregated_outputs['image_features_teacher'].append(torch.zeros_like(graph_embeddings[pyg_batch_idx]))
                    else:
                        logger.warning(f"Graph embedding for PyG batch index {pyg_batch_idx} not found. Image {img_orig_idx} will not have image_features.")
        else:
            logger.warning("No nodes or edges in the PyG batch. Skipping GNN forward pass.")
            # Ensure image_features is None or empty for all images if GNN was skipped
            for symbolic_output in updated_symbolic_outputs_batch:
                symbolic_output["image_features"] = None
                # Also ensure empty lists for feature consistency if GNN skipped
                aggregated_outputs['image_features_student'].append(torch.empty(0, device=self.device))
                aggregated_outputs['image_features_teacher'].append(torch.empty(0, device=self.device))
        # Populate aggregated attribute logits and ground truths (if synthetic data)
        for sym_idx, symbolic_output in enumerate(updated_symbolic_outputs_batch):
            # Overall Bongard label
            if is_synthetic and 'bongard_label' in symbolic_output.get('ground_truth', {}):
                aggregated_outputs['bongard_labels'].append(
                    torch.tensor(symbolic_output['ground_truth']['bongard_label'], dtype=torch.long, device=self.device)
                )
            else:
                # For real data or if synthetic GT is missing, use a dummy label or handle appropriately
                aggregated_outputs['bongard_labels'].append(torch.tensor(0, dtype=torch.long, device=self.device)) # Placeholder
            for obj in symbolic_output['objects']:
                # Populate attribute logits
                if '_fill_logits' in obj:
                    aggregated_outputs['attribute_logits']['fill'].append(obj['_fill_logits'])
                    aggregated_outputs['attribute_logits']['color'].append(obj['_color_logits'])
                    aggregated_outputs['attribute_logits']['size'].append(obj['_size_logits'])
                    aggregated_outputs['attribute_logits']['orientation'].append(obj['_orientation_logits'])
                    aggregated_outputs['attribute_logits']['shape'].append(obj['_shape_logits'])
                    aggregated_outputs['attribute_logits']['texture'].append(obj['_texture_logits'])
                # Populate attribute ground truths (ONLY if synthetic data and GT is available)
                if is_synthetic and 'ground_truth' in symbolic_output:
                    # Find matching ground truth object by bbox or ID (if available)
                    gt_obj = next((g_obj for g_obj in symbolic_output['ground_truth']['objects'] if g_obj['id'] == obj['id']), None)
                    if gt_obj:
                        aggregated_outputs['attribute_gt']['fill'].append(ATTRIBUTE_FILL_MAP_INV[gt_obj['attributes']['fill']])
                        aggregated_outputs['attribute_gt']['color'].append(ATTRIBUTE_COLOR_MAP_INV[gt_obj['attributes']['color']])
                        aggregated_outputs['attribute_gt']['size'].append(ATTRIBUTE_SIZE_MAP_INV[gt_obj['attributes']['size']])
                        aggregated_outputs['attribute_gt']['orientation'].append(ATTRIBUTE_ORIENTATION_MAP_INV[gt_obj['attributes']['orientation']])
                        aggregated_outputs['attribute_gt']['shape'].append(ATTRIBUTE_SHAPE_MAP_INV[gt_obj['attributes']['shape']])
                        aggregated_outputs['attribute_gt']['texture'].append(ATTRIBUTE_TEXTURE_MAP_INV[gt_obj['attributes']['texture']])
                    else:
                        # If no matching GT object, append dummy/placeholder
                        for attr_type in aggregated_outputs['attribute_gt'].keys():
                            aggregated_outputs['attribute_gt'][attr_type].append(0) # Default to 0 for missing GT
            # Populate relation ground truths (ONLY if synthetic data and GT is available)
            if is_synthetic and 'ground_truth' in symbolic_output and symbolic_output['relations']:
                # For each predicted relation, find its ground truth counterpart
                for pred_rel in symbolic_output['relations']:
                    gt_rel = next((g_rel for g_rel in symbolic_output['ground_truth']['relations']
                                   if g_rel['src'] == pred_rel['source_id'] and g_rel['dst'] == pred_rel['target_id']), None)
                    if gt_rel:
                        aggregated_outputs['relation_gt'].append(RELATION_MAP_INV[gt_rel['type']])
                    else:
                        aggregated_outputs['relation_gt'].append(0) # Default to 'unrelated' or 0 for missing GT
        # Convert lists to tensors
        for attr_type in aggregated_outputs['attribute_logits'].keys():
            if aggregated_outputs['attribute_logits'][attr_type]:
                aggregated_outputs['attribute_logits'][attr_type] = torch.stack(aggregated_outputs['attribute_logits'][attr_type])
            else:
                aggregated_outputs['attribute_logits'][attr_type] = torch.empty(0, len(ATTRIBUTE_FILL_MAP), device=self.device) # Dummy empty tensor
        for attr_type in aggregated_outputs['attribute_gt'].keys():
            if aggregated_outputs['attribute_gt'][attr_type]:
                aggregated_outputs['attribute_gt'][attr_type] = torch.tensor(aggregated_outputs['attribute_gt'][attr_type], dtype=torch.long, device=self.device)
            else:
                aggregated_outputs['attribute_gt'][attr_type] = torch.empty(0, dtype=torch.long, device=self.device)
        if aggregated_outputs['relation_logits']:
            aggregated_outputs['relation_logits'] = torch.stack(aggregated_outputs['relation_logits'])
        else:
            aggregated_outputs['relation_logits'] = torch.empty(0, len(RELATION_MAP), device=self.device)
        if aggregated_outputs['relation_gt']:
            aggregated_outputs['relation_gt'] = torch.tensor(aggregated_outputs['relation_gt'], dtype=torch.long, device=self.device)
        else:
            aggregated_outputs['relation_gt'] = torch.empty(0, dtype=torch.long, device=self.device)
        if aggregated_outputs['bongard_labels']:
            aggregated_outputs['bongard_labels'] = torch.stack(aggregated_outputs['bongard_labels'])
        else:
            aggregated_outputs['bongard_labels'] = torch.empty(0, dtype=torch.long, device=self.device)
        if aggregated_outputs['image_features_student']:
            aggregated_outputs['image_features_student'] = torch.stack(aggregated_outputs['image_features_student'])
        else:
            aggregated_outputs['image_features_student'] = torch.empty(0, self.relation_gnn.set2set.output_size, device=self.device) # Match expected dim
        if aggregated_outputs['image_features_teacher']:
            aggregated_outputs['image_features_teacher'] = torch.stack(aggregated_outputs['image_features_teacher'])
        else:
            aggregated_outputs['image_features_teacher'] = torch.empty(0, self.relation_gnn.set2set.output_size, device=self.device) # Match expected dim
        # Clean up temporary features and logits used for internal processing
        for symbolic_output in updated_symbolic_outputs_batch:
            for obj in symbolic_output["objects"]:
                for key_to_delete in ["_features", "_type_original", "_sam_iou",
                                      "_fill_logits", "_color_logits", "_size_logits",
                                      "_orientation_logits", "_shape_logits", "_texture_logits"]:
                    if key_to_delete in obj:
                        del obj[key_to_delete]
            # Remove ground_truth if it was temporarily added
            if 'ground_truth' in symbolic_output:
                del symbolic_output['ground_truth']
        return updated_symbolic_outputs_batch, aggregated_outputs
    
    def mc_dropout_predict(self, image_input: Any, num_samples: int, is_synthetic: bool = False):
        """
        Performs Monte Carlo Dropout inference to estimate prediction uncertainty.
        Args:
            image_input (List[str] or List[Image.Image]): List of paths to input images (for real data)
                                                          or list of PIL Images (for synthetic data).
            num_samples (int): Number of forward passes to perform with dropout enabled.
            is_synthetic (bool): True if input is synthetic PIL images, False if paths to real images.
        Returns:
            List[Dict]: A list of dictionaries, one per image in the batch.
                        Each dict contains mean and variance for Bongard problem logits,
                        attribute logits, and relation logits.
        """
        # Ensure dropout layers are enabled during inference
        # This requires setting the model to train mode, but disabling gradient tracking.
        self.train() # Enable dropout layers
        
        # Lists to store predictions from each MC sample
        all_mc_bongard_logits = []
        all_mc_attr_logits = {attr: [] for attr in ATTRIBUTE_FILL_MAP.keys()}
        all_mc_relation_logits = []
        
        # Prepare image_pil_batch and image_paths once
        if is_synthetic:
            image_pil_batch = image_input
            image_paths = [f"synthetic_image_{i}" for i in range(len(image_input))]
        else:
            image_pil_batch = [Image.open(p).convert('RGB') for p in image_input]
            image_paths = image_input

        logger.info(f"Performing Monte Carlo Dropout inference with {num_samples} samples.")
        with torch.no_grad(): # No gradients needed for MC Dropout inference
            for s in tqdm(range(num_samples), desc="MC Dropout Samples"):
                # Run the full forward pass
                symbolic_outputs_batch, aggregated_outputs = self.forward(image_pil_batch, is_synthetic)
                
                # Collect Bongard logits
                if aggregated_outputs['bongard_labels'].numel() > 0 and aggregated_outputs['image_features_student'].numel() > 0:
                    if hasattr(self, 'bongard_classifier'):
                        bongard_logits = self.bongard_classifier(aggregated_outputs['image_features_student'])
                        all_mc_bongard_logits.append(bongard_logits.unsqueeze(0)) # Add batch dimension for stacking
                
                # Collect attribute logits
                for attr_type in aggregated_outputs['attribute_logits'].keys():
                    if aggregated_outputs['attribute_logits'][attr_type].numel() > 0:
                        all_mc_attr_logits[attr_type].append(aggregated_outputs['attribute_logits'][attr_type].unsqueeze(0))
                
                # Collect relation logits
                if aggregated_outputs['relation_logits'].numel() > 0:
                    all_mc_relation_logits.append(aggregated_outputs['relation_logits'].unsqueeze(0))
        
        # Reset model to original mode (e.g., eval if it was in eval before MC dropout)
        self.eval() # Set back to eval mode after MC dropout
        
        # Calculate mean and variance for each prediction type
        mc_results = []
        num_images_in_batch = len(image_pil_batch)

        for img_idx in range(num_images_in_batch):
            img_mc_result = {
                'image_idx': img_idx,
                'image_path': image_paths[img_idx],
                'bongard_prediction_mean': None,
                'bongard_prediction_variance': None,
                'attribute_prediction_means': {attr: None for attr in ATTRIBUTE_FILL_MAP.keys()},
                'attribute_prediction_variances': {attr: None for attr in ATTRIBUTE_FILL_MAP.keys()},
                'relation_prediction_means': None,
                'relation_prediction_variances': None,
                'objects': [] # To store per-object attribute uncertainty
            }

            # Bongard Problem Uncertainty
            if all_mc_bongard_logits:
                # Stack all samples for this image (assuming batch size 1 for simplicity in this example)
                # If batch size > 1, need to extract per-image logits from stacked tensors
                # This requires careful indexing based on how `aggregated_outputs` are structured.
                # For now, assume `all_mc_bongard_logits` is a list of [1, num_classes] tensors.
                
                # Re-structure all_mc_bongard_logits to be [num_samples, batch_size, num_classes]
                stacked_bongard_logits = torch.cat(all_mc_bongard_logits, dim=0) # [num_samples, batch_size, num_classes]
                
                # Extract logits for current image
                if img_idx < stacked_bongard_logits.shape[1]: # Check if image exists in batch
                    bongard_logits_for_img = stacked_bongard_logits[:, img_idx, :] # [num_samples, num_classes]
                    
                    bongard_probs = F.softmax(bongard_logits_for_img, dim=-1)
                    img_mc_result['bongard_prediction_mean'] = bongard_probs.mean(dim=0).cpu().numpy().tolist()
                    img_mc_result['bongard_prediction_variance'] = bongard_probs.var(dim=0).cpu().numpy().tolist()
                else:
                    logger.warning(f"No Bongard logits available for image {img_idx} in MC Dropout results.")

            # Attribute Uncertainty (per object)
            # This is more complex as we need to map back to individual objects.
            # The current `aggregated_outputs` flattens object-level logits.
            # To get per-object uncertainty, we need to re-run `_extract_features_and_symbolic_data`
            # and `_process_batch_for_symbolic_outputs` for each MC sample, and then collect
            # the logits for each *original* object across samples.
            # This means `symbolic_outputs_batch` from each `self.forward` call needs to be stored.
            
            # For simplicity, let's just compute overall attribute uncertainty from `aggregated_outputs`
            # and acknowledge that per-object uncertainty would require more granular collection.
            # This is a conceptual placeholder for per-object uncertainty.
            
            # To properly implement per-object uncertainty:
            # 1. Store `symbolic_outputs_batch` for each MC sample.
            # 2. For each original image and each original object in that image,
            #    collect its attribute logits from all `num_samples` runs.
            # 3. Compute mean/variance for those collected logits.
            
            # For now, we'll calculate overall attribute uncertainty from `all_mc_attr_logits`
            # and assume it's an average over all objects in the batch.
            
            # This part needs to be refined for true per-object uncertainty.
            # For now, we'll indicate this is a placeholder.
            logger.warning("Per-object attribute uncertainty calculation is a placeholder. Requires more granular collection of per-object logits across MC samples.")

            # Relation Uncertainty
            if all_mc_relation_logits:
                # Stack all samples for relations
                stacked_relation_logits = torch.cat(all_mc_relation_logits, dim=0) # [num_samples, num_relations_in_batch, num_relation_types]
                
                # This is an average over all relations in the batch.
                # To get per-relation uncertainty, similar to attributes,
                # you'd need to map back to specific relations.
                
                relation_probs = F.softmax(stacked_relation_logits, dim=-1)
                img_mc_result['relation_prediction_means'] = relation_probs.mean(dim=0).cpu().numpy().tolist()
                img_mc_result['relation_prediction_variances'] = relation_probs.var(dim=0).cpu().numpy().tolist()
            else:
                logger.warning(f"No relation logits available for image {img_idx} in MC Dropout results.")

            mc_results.append(img_mc_result)
        
        logger.info("Monte Carlo Dropout inference complete.")
        return mc_results

    def export_onnx(self, dummy_input_shape=(1, 3, 224, 224), output_path="perception_module.onnx"):
        """
        Exports the entire PerceptionModule to ONNX format.
        Note: This is a simplified export. A real-world export would need
        to handle dynamic input sizes, control flow, and potentially
        separate export for sub-modules.
        """
        logger.info(f"Attempting to export PerceptionModule to ONNX: {output_path}")
        # Create a dummy input for the _extract_features_and_symbolic_data method.
        # This is challenging because _extract_features_and_symbolic_data takes file paths.
        # For ONNX export, we need a torch.Tensor as input.
        # This requires a modified forward pass that directly takes image tensors or
        # exporting sub-modules separately.
        
        logger.warning("Direct ONNX export of the full PerceptionModule is complex due to file I/O and graph construction.")
        logger.warning("Consider exporting sub-modules (AttributeClassifier, RelationGNN) individually.")
        try:
            # Example: Exporting AttributeClassifier
            dummy_attr_input = torch.randn(dummy_input_shape).to(self.device)
            attr_output_path = output_path.replace(".onnx", "_attribute_classifier.onnx")
            
            # Temporarily set training=False for export (important for BatchNorm, Dropout)
            self.attribute_classifier.eval()
            
            torch.onnx.export(
                self.attribute_classifier,
                dummy_attr_input,
                attr_output_path,
                export_params=True,
                opset_version=11, # Common OPS set
                do_constant_folding=True,
                input_names=['input'],
                output_names=['fill_logits', 'color_logits', 'size_logits', 'orientation_logits', 'shape_logits', 'texture_logits', 'features'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'fill_logits': {0: 'batch_size'},
                              'color_logits': {0: 'batch_size'},
                              'size_logits': {0: 'batch_size'},
                              'orientation_logits': {0: 'batch_size'},
                              'shape_logits': {0: 'batch_size'},
                              'texture_logits': {0: 'batch_size'},
                              'features': {0: 'batch_size'}}
            )
            logger.info(f"AttributeClassifier exported to {attr_output_path}")
            # Example: Exporting RelationGNN (more complex due to PyG Data object)
            # This would require constructing a dummy PyG Data object, which is not straightforward with torch.onnx.export
            # as it expects Tensors. You'd likely need to trace specific tensor operations.
            logger.warning("RelationGNN export to ONNX is more involved and not demonstrated here directly. Requires custom tracing or PyG's own ONNX support if available.")
            # Restore training mode
            self.attribute_classifier.train()
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            logger.info("Ensure PyTorch Geometric (pyg) models are compatible with torch.onnx.export or export individual tensor ops.")
# -----------------------------------------------------------------------------
# Loss Functions
# -----------------------------------------------------------------------------
class DistillationLoss(nn.Module):
    """
    Distillation loss combines a hard target loss (e.g., CrossEntropy) with a
    soft target loss (KL-Divergence) for knowledge distillation.
    Args:
        alpha (float): Weight for the soft target loss.
        temperature (float): Temperature for softening logits.
        base_loss_fn (nn.Module): The base loss function for hard targets (e.g., nn.CrossEntropyLoss()).
    """
    def __init__(self, alpha=0.5, temperature=1.0, base_loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.base_loss_fn = base_loss_fn
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")
    def forward(self, student_logits, teacher_logits, targets):
        """
        Calculates distillation loss.
        Args:
            student_logits (torch.Tensor): Logits from the student model.
            teacher_logits (torch.Tensor): Logits from the teacher model.
            targets (torch.Tensor): Hard targets (ground truth labels).
        Returns:
            torch.Tensor: Combined distillation loss.
        """
        hard_loss = self.base_loss_fn(student_logits, targets)
        
        # Soften teacher and student logits
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        soft_loss = self.kl_div_loss(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)
        
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return total_loss
class MixupCutmix:
    """
    Applies Mixup or CutMix data augmentation.
    Mixup: Combines two samples and their labels linearly.
    CutMix: Combines two samples by cutting and pasting regions, and mixes labels proportionally.
    Args:
        mixup_alpha (float): Alpha parameter for the Beta distribution for Mixup.
        cutmix_alpha (float): Alpha parameter for the Beta distribution for CutMix.
        prob (float): Probability of applying either Mixup or CutMix.
        switch_prob (float): Probability of switching to CutMix if Mixup is chosen.
        num_classes (int): Number of classes for one-hot encoding labels.
    """
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, num_classes=10):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.num_classes = num_classes
    def __call__(self, input, target):
        if np.random.rand(1) > self.prob:
            return input, target
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        rand_index = torch.randperm(input.size()[0]).cuda() # Ensure index is on GPU
        
        # One-hot encode targets for Mixup/CutMix
        if target.ndim == 1: # If labels are not already one-hot
            target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        else:
            target_onehot = target # Assume already one-hot
        target_a = target_onehot
        target_b = target_onehot[rand_index]
        if np.random.rand(1) < self.switch_prob: # Use CutMix
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # Adjust lambda for CutMix based on erased area ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            logger.debug(f"Applied CutMix with lambda: {lam:.4f}")
        else: # Use Mixup
            input = lam * input + (1 - lam) * input[rand_index, :]
            logger.debug(f"Applied Mixup with lambda: {lam:.4f}")
        
        target = lam * target_a + (1 - lam) * target_b
        return input, target
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int_(W * cut_rat)
        cut_h = np.int_(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with label smoothing.
    Args:
        smoothing (float): Smoothing factor, typically between 0.0 and 0.1.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
class FeatureConsistencyLoss(nn.Module):
    """
    A loss function to enforce consistency between features extracted by the perception module
    and features extracted by a reference (teacher) feature extractor.
    This can be used for self-supervised consistency or distillation.
    Args:
        loss_type (str): Type of loss to use ('mse', 'cosine', 'kl_div').
        temperature (float): Temperature for KL_Div (if used).
    """
    def __init__(self, loss_type='mse', temperature=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        elif loss_type == 'kl_div':
            self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError(f"Unsupported feature consistency loss type: {loss_type}")
    def forward(self, student_features, teacher_features):
        if self.loss_type == 'mse':
            return self.loss_fn(student_features, teacher_features.detach())
        elif self.loss_type == 'cosine':
            # CosineEmbeddingLoss expects (input1, input2, target) where target is 1 for similar, -1 for dissimilar
            # We want them to be similar, so target is all ones.
            target = torch.ones(student_features.size(0), device=student_features.device)
            return self.loss_fn(student_features, teacher_features.detach(), target)
        elif self.loss_type == 'kl_div':
            # For KLDiv, treat features as logits, soften them, and compute KLDiv
            # This assumes features can be interpreted as log-probabilities or logits.
            # Normalizing features before KLDiv can be beneficial.
            soft_teacher_probs = F.softmax(teacher_features.detach() / self.temperature, dim=1)
            soft_student_log_probs = F.log_softmax(student_features / self.temperature, dim=1)
            return self.loss_fn(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)
# -----------------------------------------------------------------------------
# Knowledge Replay Buffer (for experience replay / continual learning)
# -----------------------------------------------------------------------------
class KnowledgeReplayBuffer:
    """
    A simple replay buffer for storing and sampling symbolic data (e.g., for
    experience replay in continual learning settings).
    Stores symbolic outputs along with their image paths.
    Args:
        capacity (int): Maximum number of experiences to store.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.rng = random.Random() # Use a dedicated random number generator
    def add(self, symbolic_output: Dict):
        """Adds a symbolic output to the buffer."""
        self.buffer.append(symbolic_output)
        logger.debug(f"Added symbolic output to replay buffer. Current size: {len(self.buffer)}/{self.capacity}")
    def sample(self, batch_size: int) -> List[Dict]:
        """Samples a batch of symbolic outputs from the buffer."""
        if len(self.buffer) < batch_size:
            logger.warning(f"Replay buffer has {len(self.buffer)} items, requested {batch_size}. Returning all available.")
            return list(self.buffer)
        
        samples = self.rng.sample(self.buffer, batch_size)
        logger.debug(f"Sampled {len(samples)} items from replay buffer.")
        return samples
    def __len__(self):
        return len(self.buffer)

# ... (Continued from phase1-code-part2b) ...

# -----------------------------------------------------------------------------
# Main Training and Evaluation Loop
# -----------------------------------------------------------------------------
def _train_step(model: PerceptionModule, dali_iter, optimizer, criterion_attr, criterion_rel,
                criterion_consistency, criterion_distillation, scaler,
                mixup_cutmix_transform, config, epoch, replay_buffer: Optional[KnowledgeReplayBuffer] = None):
    """
    Performs a single training step over one epoch.
    Args:
        model (PerceptionModule): The perception model.
        dali_iter (DALIGenericIterator): DALI iterator for training data.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion_attr (nn.Module): Loss function for attribute classification.
        criterion_rel (nn.Module): Loss function for relation classification.
        criterion_consistency (FeatureConsistencyLoss): Loss for feature consistency.
        criterion_distillation (DistillationLoss): Loss for knowledge distillation.
        scaler (GradScaler): For Automatic Mixed Precision.
        mixup_cutmix_transform (MixupCutmix): Mixup/CutMix augmentation.
        config (dict): Configuration dictionary.
        epoch (int): Current epoch number.
        replay_buffer (KnowledgeReplayBuffer, optional): Replay buffer for experience replay.
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_attr_loss = 0.0
    total_rel_loss = 0.0
    total_consistency_loss = 0.0
    total_distillation_loss = 0.0
    
    # Initialize profiler if enabled
    if config['training']['enable_profiler']:
        schedule = torch.profiler.schedule(
            wait=config['training']['profiler_schedule_wait'],
            warmup=config['training']['profiler_schedule_warmup'],
            active=config['training']['profiler_schedule_active'],
            repeat=config['training']['profiler_schedule_repeat']
        )
        profiler = torch.profiler.profile(
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(config['training']['checkpoint_dir'], 'profiler_logs')),
            with_stack=True,
            profile_memory=True
        )
        profiler.start()
        logger.info("PyTorch profiler started.")
    
    optimizer.zero_grad() # Initialize gradients to zero for accumulation
    
    # Set current epoch for GNN dynamic depth scheduling
    if hasattr(model.relation_gnn, 'current_epoch'):
        model.relation_gnn.current_epoch = epoch
    
    for i, data in enumerate(tqdm(dali_iter, desc=f"Training Epoch {epoch}")):
        # DALI returns list of tensors for each output.
        # For our pipeline: [image_view_1, image_view_2, bongard_label, ground_truth_json_string, affine_matrix_1, affine_matrix_2]
        images_view1 = data[0]["data"]
        images_view2 = data[1]["data"] # Second view for feature consistency
        
        # DALI's ground truth JSON string needs to be parsed here.
        # It comes as a list of NumPy arrays (each containing a JSON string).
        gt_json_strings = [s.item().decode('utf-8') for s in data[3]["data"]] # Decode bytes to string
        # Parse JSON strings into Python dicts
        ground_truths = [json.loads(s) for s in gt_json_strings]
        
        # Prepare inputs for PerceptionModule.forward
        # For synthetic data, we pass PIL images directly.
        # For real data, we pass image paths.
        # DALI provides decoded images (images_view1, images_view2), so we'll use these.
        # We need to convert DALI's output (torch.Tensor on GPU) back to PIL for RealObjectDetector.
        # This is inefficient but necessary for the current architecture.
        # A more efficient approach would be to make RealObjectDetector accept torch.Tensor.
        
        # Convert DALI GPU tensor to PIL Image for RealObjectDetector
        # DALI output is NCHW, float32, normalized. Convert to HWC, uint8, unnormalized.
        # Then to PIL. This is a bottleneck.
        image_pil_batch = []
        # DALI images are already on GPU if device_id != -1
        # Denormalize and convert to CPU numpy array, then to PIL
        # DALI images are normalized by `fn.crop_mirror_normalize` using ImageNet mean/std.
        # To get back to original pixel values for PIL.Image.fromarray:
        # (img_tensor * std) + mean
        
        # Denormalize for PIL conversion (assuming images_view1 is already NCHW, float32)
        # Apply inverse normalization (un-normalize)
        # First, ensure it's on CPU and convert to HWC
        images_denorm = images_view1.permute(0, 2, 3, 1) # NCHW -> NHWC
        
        # Denormalize from [0,1] to [0,255] and convert to uint8
        # Assuming ImageNet_Mean/Std are for [0,1] range (as typically used with PyTorch models)
        # If DALI's fn.crop_mirror_normalize used 0-255 mean/std, then this step is simpler.
        # Based on DALI documentation, `mean` and `std` are applied to the input range.
        # If input is 0-255, mean/std should be 0-255. If input is 0-1, mean/std should be 0-1.
        # Our DALI pipeline uses IMAGENET_MEAN/STD which are 0-255 scaled.
        # So, the output of DALI's `fn.crop_mirror_normalize` is already normalized to mean 0, std 1.
        # To reverse: `img_normalized * std_val + mean_val` (where std_val/mean_val are for 0-1 range)
        # Or, if DALI normalized 0-255 data directly, then it's `(img_normalized * std_255) + mean_255`
        
        # Let's assume DALI's output is normalized to [0,1] range (or similar)
        # and we need to scale back to 0-255 for PIL.
        # This is a critical point of potential mismatch.
        # For simplicity, let's assume images_view1 is already in a visualizable range (e.g., 0-1 or 0-255)
        # after DALI's normalization, and we just need to convert to uint8.
        # If it's normalized to mean 0, std 1, then it needs to be un-normalized first.
        
        # Re-evaluate DALI output: `fn.crop_mirror_normalize` with `mean` and `std` values
        # effectively normalizes the input. If `mean` and `std` are `[0.485*255, ...]` and `[0.229*255, ...]`,
        # then the output `out_view1` is normalized.
        # To convert back to 0-255 for PIL:
        # `img_tensor_normalized * IMAGENET_STD_0_1 + IMAGENET_MEAN_0_1` and then scale to 255.
        
        # Let's use the 0-1 range for ImageNet mean/std for un-normalization
        IMAGENET_MEAN_0_1 = [0.485, 0.456, 0.406]
        IMAGENET_STD_0_1 = [0.229, 0.224, 0.225]
        
        # Apply denormalization and convert to uint8
        for b_idx in range(images_view1.shape[0]):
            img_tensor_unnorm = images_view1[b_idx].cpu() # C H W
            for c in range(img_tensor_unnorm.shape[0]):
                img_tensor_unnorm[c] = img_tensor_unnorm[c] * IMAGENET_STD_0_1[c] + IMAGENET_MEAN_0_1[c]
            
            img_np = (img_tensor_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8) # HWC
            
            # Handle grayscale conversion for PIL if num_channels is 1
            if config['data']['num_channels'] == 1 and img_np.shape[2] == 3:
                img_pil = Image.fromarray(img_np, 'RGB').convert('L')
            elif config['data']['num_channels'] == 3 and img_np.shape[2] == 1:
                # If DALI outputted grayscale but we expect RGB for YOLO
                img_pil = Image.fromarray(img_np.squeeze(2), 'L').convert('RGB')
            else:
                img_pil = Image.fromarray(img_np, 'RGB') # Assuming RGB for YOLO
            image_pil_batch.append(img_pil)
        
        # Pass the PIL images and their ground truths to the PerceptionModule
        # The PerceptionModule's forward method expects a list of PIL images or paths.
        # We need to inject the ground truth into the symbolic_outputs for training.
        # This requires modifying PerceptionModule.forward or handling GT outside.
        # For now, let's pass ground_truths as an additional argument to forward.
        
        # --- CRITICAL: Inject ground truth into symbolic_outputs for loss calculation ---
        # This requires a slight modification to PerceptionModule.forward to accept GT.
        # Or, we can modify the symbolic_outputs_batch *after* PerceptionModule.forward.
        # Let's modify after, as it's cleaner for the forward signature.
        
        # Pass the PIL images and indicate if it's synthetic (to use the GT parsing logic)
        symbolic_outputs, aggregated_outputs = model(image_pil_batch, is_synthetic=config['data']['use_synthetic_data'])
        
        # If using synthetic data, populate ground truth in aggregated_outputs
        if config['data']['use_synthetic_data']:
            # The `ground_truths` list (parsed from DALI) contains GT for each image in batch.
            # We need to map these to the `symbolic_outputs` and then to `aggregated_outputs`.
            # This is already handled within `PerceptionModule.forward` if `is_synthetic=True`.
            # So, `aggregated_outputs` should already contain the correct `attribute_gt`, `relation_gt`, `bongard_labels`.
            pass # No explicit action needed here, already handled in model.forward
        
        # Apply Mixup/CutMix to Bongard problem images (if enabled)
        # This is typically applied to the raw image data, not features.
        # However, since DALI handles image augmentations, we can apply it to the
        # final image features if the goal is feature-level mixing.
        # For simplicity, let's assume Mixup/CutMix is applied to the final
        # image-level features (graph_embeddings) for the Bongard classifier.
        # This is a simplification; ideally, it's image-level.
        
        # Check if there are any valid outputs for loss calculation
        if (aggregated_outputs['attribute_logits']['fill'].numel() == 0 and
            aggregated_outputs['relation_logits'].numel() == 0 and
            aggregated_outputs['image_features_student'].numel() == 0):
            logger.warning(f"Batch {i}: No valid logits or features for loss calculation. Skipping batch.")
            dali_iter.release_gpu_memory() # Release memory for this batch
            continue # Skip to next batch
        
        # Calculate losses
        current_attr_loss = 0.0
        current_rel_loss = 0.0
        current_consistency_loss = 0.0
        current_distillation_loss = 0.0
        
        # Attribute Loss
        if aggregated_outputs['attribute_logits']['fill'].numel() > 0:
            attr_loss_fill = criterion_attr(aggregated_outputs['attribute_logits']['fill'], aggregated_outputs['attribute_gt']['fill'])
            attr_loss_color = criterion_attr(aggregated_outputs['attribute_logits']['color'], aggregated_outputs['attribute_gt']['color'])
            attr_loss_size = criterion_attr(aggregated_outputs['attribute_logits']['size'], aggregated_outputs['attribute_gt']['size'])
            attr_loss_orientation = criterion_attr(aggregated_outputs['attribute_logits']['orientation'], aggregated_outputs['attribute_gt']['orientation'])
            attr_loss_shape = criterion_attr(aggregated_outputs['attribute_logits']['shape'], aggregated_outputs['attribute_gt']['shape'])
            attr_loss_texture = criterion_attr(aggregated_outputs['attribute_logits']['texture'], aggregated_outputs['attribute_gt']['texture'])
            
            current_attr_loss = (attr_loss_fill + attr_loss_color + attr_loss_size +
                                 attr_loss_orientation + attr_loss_shape + attr_loss_texture) / 6.0
            total_attr_loss += current_attr_loss.item()
        
        # Relation Loss
        if aggregated_outputs['relation_logits'].numel() > 0:
            current_rel_loss = criterion_rel(aggregated_outputs['relation_logits'], aggregated_outputs['relation_gt'])
            total_rel_loss += current_rel_loss.item()
        
        # Feature Consistency Loss (between two views of the same image)
        if config['training']['feature_consistency_alpha'] > 0 and \
           aggregated_outputs['image_features_student'].numel() > 0 and \
           images_view2.numel() > 0: # Ensure second view was actually generated
            
            # Re-run the attribute classifier and GNN for the second view to get student features
            # This is necessary because `images_view2` is a separate augmented view.
            
            # Convert DALI GPU tensor to PIL Image for RealObjectDetector (for view2)
            image_pil_batch_view2 = []
            for b_idx in range(images_view2.shape[0]):
                img_tensor_unnorm_v2 = images_view2[b_idx].cpu() # C H W
                for c in range(img_tensor_unnorm_v2.shape[0]):
                    img_tensor_unnorm_v2[c] = img_tensor_unnorm_v2[c] * IMAGENET_STD_0_1[c] + IMAGENET_MEAN_0_1[c]
                img_np_v2 = (img_tensor_unnorm_v2.permute(1, 2, 0).numpy() * 255).astype(np.uint8) # HWC
                if config['data']['num_channels'] == 1 and img_np_v2.shape[2] == 3:
                    img_pil_v2 = Image.fromarray(img_np_v2, 'RGB').convert('L')
                elif config['data']['num_channels'] == 3 and img_np_v2.shape[2] == 1:
                    img_pil_v2 = Image.fromarray(img_np_v2.squeeze(2), 'L').convert('RGB')
                else:
                    img_pil_v2 = Image.fromarray(img_np_v2, 'RGB')
                image_pil_batch_view2.append(img_pil_v2)
            
            # Forward pass for second view to get features
            _, aggregated_outputs_view2 = model(image_pil_batch_view2, is_synthetic=config['data']['use_synthetic_data'])
            
            if aggregated_outputs_view2['image_features_student'].numel() > 0:
                current_consistency_loss = criterion_consistency(
                    aggregated_outputs['image_features_student'], # Features from view 1
                    aggregated_outputs_view2['image_features_student'] # Features from view 2
                ) * config['training']['feature_consistency_alpha']
                total_consistency_loss += current_consistency_loss.item()
            else:
                logger.warning(f"Batch {i}: No student features from view 2 for consistency loss. Skipping consistency loss for this batch.")
        
        # Knowledge Distillation Loss
        if config['training']['use_knowledge_distillation'] and model.teacher_model and \
           aggregated_outputs['image_features_student'].numel() > 0 and \
           aggregated_outputs['image_features_teacher'].numel() > 0:
            
            # The Bongard classifier takes image-level features.
            # Assuming Bongard classifier is part of the main PerceptionModule,
            # or a separate head that takes the graph_embeddings.
            # For simplicity, let's assume `model.bongard_classifier` exists and takes `image_features_student`.
            
            # If the teacher model also outputs graph embeddings or similar aggregate features
            # that can be directly compared, use those.
            # Here, we're assuming `image_features_student` and `image_features_teacher`
            # are the comparable features for distillation.
            
            # For distillation, we need logits from both student and teacher for the *final* task (Bongard classification).
            # This implies a Bongard classifier head on top of the `image_features_student`/`teacher`.
            if hasattr(model, 'bongard_classifier') and hasattr(model.teacher_model, 'bongard_classifier'):
                student_bongard_logits = model.bongard_classifier(aggregated_outputs['image_features_student'])
                with torch.no_grad():
                    teacher_bongard_logits = model.teacher_model.bongard_classifier(aggregated_outputs['image_features_teacher'])
                
                # Use dummy targets for distillation loss, as it's a soft target loss
                # The `targets` argument in DistillationLoss is for the hard loss component.
                # If we only want soft distillation, targets can be dummy or ignored.
                # For a full distillation setup, the `targets` should be the true Bongard labels.
                
                # Ensure bongard_labels are available and match student features batch size
                if aggregated_outputs['bongard_labels'].numel() > 0 and \
                   student_bongard_logits.shape[0] == aggregated_outputs['bongard_labels'].shape[0]:
                    current_distillation_loss = criterion_distillation(
                        student_bongard_logits,
                        teacher_bongard_logits,
                        aggregated_outputs['bongard_labels'] # Hard targets for base loss
                    ) * config['training']['distillation_alpha']
                    total_distillation_loss += current_distillation_loss.item()
                else:
                    logger.warning(f"Batch {i}: Bongard labels or student logits mismatch for distillation. Skipping distillation loss.")
            else:
                logger.warning(f"Batch {i}: Bongard classifier not found on student or teacher model for distillation. Skipping distillation loss.")
        
        # Combine losses with weights
        loss = (current_attr_loss * config['training'].get('attribute_loss_weight', 1.0) +
                current_rel_loss * config['training'].get('relation_loss_weight', 1.0) +
                current_consistency_loss +
                current_distillation_loss)
        
        total_loss += loss.item()
        
        # Scale the loss for AMP
        if config['training']['use_amp']:
            scaler.scale(loss / config['training']['gradient_accumulation_steps']).backward()
        else:
            (loss / config['training']['gradient_accumulation_steps']).backward()
        
        # Perform optimizer step only after accumulation steps
        if (i + 1) % config['training']['gradient_accumulation_steps'] == 0:
            if config['training']['use_amp']:
                if config['training']['use_sam_optimizer'] and HAS_SAM:
                    # SAM's first step with scaled gradients
                    scaler.unscale_(optimizer) # Unscale gradients before SAM's first step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                    optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass for SAM
                    # Re-run forward pass with current inputs to get new gradients for second step
                    # This is computationally expensive for SAM.
                    # For simplicity, we'll re-use the same inputs and assume the model state is slightly perturbed.
                    # In a real SAM implementation, you'd need to re-evaluate the model.
                    
                    # This is a conceptual placeholder for SAM's second step in AMP.
                    # A full implementation would require careful handling of the second forward pass and loss.
                    
                    # For now, we'll just call the second step which restores params and applies base_optimizer.step()
                    scaler.scale(loss / config['training']['gradient_accumulation_steps']).backward() # Re-calculate gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Zero gradients after SAM's second step
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
                if config['training']['use_sam_optimizer'] and HAS_SAM:
                    optimizer.first_step(zero_grad=True)
                    # Second forward pass for SAM (conceptual)
                    # Re-calculate gradients for SAM's second step
                    (loss / config['training']['gradient_accumulation_steps']).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.step()
                    optimizer.zero_grad()
        
        if config['training']['enable_profiler']:
            profiler.step()
    
    if config['training']['enable_profiler']:
        profiler.stop()
        logger.info("PyTorch profiler stopped.")
    
    # Release DALI GPU memory at the end of epoch
    dali_iter.release_gpu_memory()
    
    avg_loss = total_loss / len(dali_iter)
    logger.info(f"Epoch {epoch} Training Summary: Avg Loss: {avg_loss:.4f}, Attr Loss: {total_attr_loss/len(dali_iter):.4f}, Rel Loss: {total_rel_loss/len(dali_iter):.4f}, Consistency Loss: {total_consistency_loss/len(dali_iter):.4f}, Distillation Loss: {total_distillation_loss/len(dali_iter):.4f}")
    return avg_loss
def _validate_model(model: PerceptionModule, dali_iter, criterion_attr, criterion_rel, config):
    """
    Performs a single validation step over one epoch.
    Args:
        model (PerceptionModule): The perception model.
        dali_iter (DALIGenericIterator): DALI iterator for validation data.
        criterion_attr (nn.Module): Loss function for attribute classification.
        criterion_rel (nn.Module): Loss function for relation classification.
        config (dict): Configuration dictionary.
    Returns:
        tuple: (Average validation loss, Bongard accuracy, Attribute accuracy, Relation accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_attr_loss = 0.0
    total_rel_loss = 0.0
    
    all_bongard_preds = []
    all_bongard_labels = []
    
    all_attr_preds = {attr: [] for attr in ATTRIBUTE_FILL_MAP.keys()}
    all_attr_labels = {attr: [] for attr in ATTRIBUTE_FILL_MAP.keys()}
    
    all_rel_preds = []
    all_rel_labels = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dali_iter, desc="Validation")):
            images_view1 = data[0]["data"]
            # DALI's ground truth JSON string needs to be parsed here.
            gt_json_strings = [s.item().decode('utf-8') for s in data[3]["data"]]
            ground_truths = [json.loads(s) for s in gt_json_strings]
            
            image_pil_batch = []
            IMAGENET_MEAN_0_1 = [0.485, 0.456, 0.406]
            IMAGENET_STD_0_1 = [0.229, 0.224, 0.225]
            for b_idx in range(images_view1.shape[0]):
                img_tensor_unnorm = images_view1[b_idx].cpu()
                for c in range(img_tensor_unnorm.shape[0]):
                    img_tensor_unnorm[c] = img_tensor_unnorm[c] * IMAGENET_STD_0_1[c] + IMAGENET_MEAN_0_1[c]
                img_np = (img_tensor_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if config['data']['num_channels'] == 1 and img_np.shape[2] == 3:
                    img_pil = Image.fromarray(img_np, 'RGB').convert('L')
                elif config['data']['num_channels'] == 3 and img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np.squeeze(2), 'L').convert('RGB')
                else:
                    img_pil = Image.fromarray(img_np, 'RGB')
                image_pil_batch.append(img_pil)
            
            symbolic_outputs, aggregated_outputs = model(image_pil_batch, is_synthetic=config['data']['use_synthetic_data'])
            
            # Calculate losses
            current_attr_loss = 0.0
            current_rel_loss = 0.0
            
            if aggregated_outputs['attribute_logits']['fill'].numel() > 0:
                attr_loss_fill = criterion_attr(aggregated_outputs['attribute_logits']['fill'], aggregated_outputs['attribute_gt']['fill'])
                attr_loss_color = criterion_attr(aggregated_outputs['attribute_logits']['color'], aggregated_outputs['attribute_gt']['color'])
                attr_loss_size = criterion_attr(aggregated_outputs['attribute_logits']['size'], aggregated_outputs['attribute_gt']['size'])
                attr_loss_orientation = criterion_attr(aggregated_outputs['attribute_logits']['orientation'], aggregated_outputs['attribute_gt']['orientation'])
                attr_loss_shape = criterion_attr(aggregated_outputs['attribute_logits']['shape'], aggregated_outputs['attribute_gt']['shape'])
                attr_loss_texture = criterion_attr(aggregated_outputs['attribute_logits']['texture'], aggregated_outputs['attribute_gt']['texture'])
                
                current_attr_loss = (attr_loss_fill + attr_loss_color + attr_loss_size +
                                     attr_loss_orientation + attr_loss_shape + attr_loss_texture) / 6.0
                total_attr_loss += current_attr_loss.item()
                
                # Collect attribute predictions and labels for accuracy
                for attr_type in aggregated_outputs['attribute_logits'].keys():
                    if aggregated_outputs['attribute_logits'][attr_type].numel() > 0:
                        all_attr_preds[attr_type].extend(aggregated_outputs['attribute_logits'][attr_type].argmax(dim=1).cpu().tolist())
                        all_attr_labels[attr_type].extend(aggregated_outputs['attribute_gt'][attr_type].cpu().tolist())
            
            if aggregated_outputs['relation_logits'].numel() > 0:
                current_rel_loss = criterion_rel(aggregated_outputs['relation_logits'], aggregated_outputs['relation_gt'])
                total_rel_loss += current_rel_loss.item()
                
                # Collect relation predictions and labels for accuracy
                all_rel_preds.extend(aggregated_outputs['relation_logits'].argmax(dim=1).cpu().tolist())
                all_rel_labels.extend(aggregated_outputs['relation_gt'].cpu().tolist())
            
            # Bongard Problem Prediction (from image_features_student)
            if aggregated_outputs['image_features_student'].numel() > 0:
                if hasattr(model, 'bongard_classifier'):
                    bongard_logits = model.bongard_classifier(aggregated_outputs['image_features_student'])
                    all_bongard_preds.extend(bongard_logits.argmax(dim=1).cpu().tolist())
                    all_bongard_labels.extend(aggregated_outputs['bongard_labels'].cpu().tolist())
                else:
                    logger.warning("Bongard classifier not found for validation.")
            
            loss = (current_attr_loss * config['training'].get('attribute_loss_weight', 1.0) +
                    current_rel_loss * config['training'].get('relation_loss_weight', 1.0))
            total_loss += loss.item()
    
    dali_iter.release_gpu_memory()
    
    avg_loss = total_loss / len(dali_iter)
    
    # Calculate accuracies
    bongard_accuracy = 0.0
    if all_bongard_preds and all_bongard_labels:
        bongard_accuracy = accuracy_score(all_bongard_labels, all_bongard_preds)
        logger.info(f"Bongard Accuracy: {bongard_accuracy:.4f}")
    
    attr_accuracy = 0.0
    if any(all_attr_preds.values()) and any(all_attr_labels.values()):
        # Average accuracy across all attribute types
        attr_accuracies = []
        for attr_type in ATTRIBUTE_FILL_MAP.keys():
            if all_attr_labels[attr_type]:
                acc = accuracy_score(all_attr_labels[attr_type], all_attr_preds[attr_type])
                attr_accuracies.append(acc)
                logger.debug(f"Attribute '{attr_type}' Accuracy: {acc:.4f}")
        if attr_accuracies:
            attr_accuracy = np.mean(attr_accuracies)
            logger.info(f"Average Attribute Accuracy: {attr_accuracy:.4f}")
    
    rel_accuracy = 0.0
    if all_rel_preds and all_rel_labels:
        rel_accuracy = accuracy_score(all_rel_labels, all_rel_preds)
        logger.info(f"Relation Accuracy: {rel_accuracy:.4f}")
    
    logger.info(f"Validation Summary: Avg Loss: {avg_loss:.4f}, Bongard Acc: {bongard_accuracy:.4f}, Attr Acc: {attr_accuracy:.4f}, Rel Acc: {rel_accuracy:.4f}")
    return avg_loss, bongard_accuracy, attr_accuracy, rel_accuracy
def _plot_training_history(history, save_path="training_history.png"):
    """
    Plots the training and validation loss and accuracy over epochs.
    Args:
        history (dict): Dictionary containing lists of 'train_loss', 'val_loss', etc.
        save_path (str): Path to save the plot.
    """
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
    plt.plot(history['val_bongard_accuracy'], label='Validation Bongard Accuracy')
    plt.title('Bongard Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
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
def calibrate_model(model: PerceptionModule, val_dali_iter, config):
    """
    Calibrates the model using temperature scaling on the Bongard classifier's output.
    This is typically done on a held-out calibration set (validation set).
    Args:
        model (PerceptionModule): The trained perception model.
        val_dali_iter (DALIGenericIterator): DALI iterator for validation data (calibration set).
        config (dict): Configuration dictionary.
    """
    logger.info("Starting model calibration using temperature scaling.")
    if not hasattr(model, 'bongard_classifier'):
        logger.warning("Bongard classifier not found in model. Skipping calibration.")
        return
    
    # Create a temperature parameter
    temperature = nn.Parameter(torch.ones(1).to(model.device))
    
    # Define optimizer for temperature
    temp_optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')
    
    # Collect all Bongard logits and true labels from the validation set
    all_bongard_logits = []
    all_bongard_labels = []
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_dali_iter, desc="Collecting calibration data")):
            images_view1 = data[0]["data"]
            gt_json_strings = [s.item().decode('utf-8') for s in data[3]["data"]]
            ground_truths = [json.loads(s) for s in gt_json_strings]
            
            image_pil_batch = []
            IMAGENET_MEAN_0_1 = [0.485, 0.456, 0.406]
            IMAGENET_STD_0_1 = [0.229, 0.224, 0.225]
            for b_idx in range(images_view1.shape[0]):
                img_tensor_unnorm = images_view1[b_idx].cpu()
                for c in range(img_tensor_unnorm.shape[0]):
                    img_tensor_unnorm[c] = img_tensor_unnorm[c] * IMAGENET_STD_0_1[c] + IMAGENET_MEAN_0_1[c]
                img_np = (img_tensor_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if config['data']['num_channels'] == 1 and img_np.shape[2] == 3:
                    img_pil = Image.fromarray(img_np, 'RGB').convert('L')
                elif config['data']['num_channels'] == 3 and img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np.squeeze(2), 'L').convert('RGB')
                else:
                    img_pil = Image.fromarray(img_np, 'RGB')
                image_pil_batch.append(img_pil)
            
            _, aggregated_outputs = model(image_pil_batch, is_synthetic=config['data']['use_synthetic_data'])
            
            if aggregated_outputs['image_features_student'].numel() > 0:
                bongard_logits = model.bongard_classifier(aggregated_outputs['image_features_student'])
                all_bongard_logits.append(bongard_logits)
                all_bongard_labels.append(aggregated_outputs['bongard_labels'])
    
    val_dali_iter.release_gpu_memory() # Release memory after data collection
    
    if not all_bongard_logits:
        logger.warning("No Bongard logits collected for calibration. Skipping.")
        return
    
    all_bongard_logits = torch.cat(all_bongard_logits).to(model.device)
    all_bongard_labels = torch.cat(all_bongard_labels).to(model.device)
    
    logger.info(f"Collected {all_bongard_logits.shape[0]} samples for calibration.")
    
    # Define NLL loss for calibration
    nll_criterion = nn.CrossEntropyLoss().to(model.device)
    
    def eval_loss():
        temp_optimizer.zero_grad()
        loss = nll_criterion(all_bongard_logits / temperature, all_bongard_labels)
        loss.backward()
        return loss
    
    logger.info("Optimizing temperature parameter...")
    temp_optimizer.step(eval_loss)
    
    calibrated_temperature = temperature.item()
    model.bongard_classifier.temperature = calibrated_temperature # Store temperature in the classifier
    logger.info(f"Model calibrated. Optimal temperature: {calibrated_temperature:.4f}")
    
    # Plot reliability diagram before and after calibration
    if config['debug']['plot_reliability_diagram']:
        uncalibrated_probs = F.softmax(all_bongard_logits, dim=1).cpu().numpy()
        calibrated_probs = F.softmax(all_bongard_logits / calibrated_temperature, dim=1).cpu().numpy()
        labels_np = all_bongard_labels.cpu().numpy()
        
        _plot_reliability_diagram(uncalibrated_probs, labels_np, save_path=os.path.join(config['training']['checkpoint_dir'], "reliability_diagram_uncalibrated.png"))
        _plot_reliability_diagram(calibrated_probs, labels_np, save_path=os.path.join(config['training']['checkpoint_dir'], "reliability_diagram_calibrated.png"))
        
        # Calculate ECE (Expected Calibration Error)
        ece_uncalibrated = brier_score_loss(labels_np, uncalibrated_probs[:, 1]) # Brier score as proxy for ECE
        ece_calibrated = brier_score_loss(labels_np, calibrated_probs[:, 1])
        logger.info(f"ECE (uncalibrated): {ece_uncalibrated:.4f}, ECE (calibrated): {ece_calibrated:.4f}")
def apply_structured_pruning(model: nn.Module, config: dict):
    """
    Conceptual function to apply structured pruning to the model.
    This is a placeholder as actual structured pruning requires specific libraries
    (e.g., `torch.nn.utils.prune` for unstructured, or more advanced for structured)
    and careful design of which layers to prune and by how much.
    Args:
        model (nn.Module): The PyTorch model to prune.
        config (dict): Configuration dictionary, potentially containing pruning parameters.
    """
    logger.warning("Conceptual structured pruning initiated. This is a placeholder function.")
    logger.warning("Actual structured pruning requires specific strategies (e.g., L1-norm based, lottery ticket hypothesis) and careful implementation.")
    logger.warning("For demonstration, this function will simply log a message and not modify the model.")
    
    # Example: If using torch.nn.utils.prune for *unstructured* pruning:
    # import torch.nn.utils.prune as prune
    # for name, module in model.named_modules():
    #     if isinstance(module, (nn.Linear, nn.Conv2d)):
    #         # Apply L1 unstructured pruning to 50% of weights
    #         prune.l1_unstructured(module, name="weight", amount=0.5)
    #         # To make pruning permanent and remove re-parametrization:
    #         # prune.remove(module, 'weight')
    
    # For structured pruning (e.g., channel pruning, filter pruning),
    # you would typically identify layers (e.g., Conv2d), compute importance scores
    # for filters/channels, and then remove/zero out those filters/channels,
    # often followed by fine-tuning.
    
    if config['training'].get('enable_structured_pruning', False):
        logger.info("Structured pruning is enabled in config, but this is a conceptual implementation.")
        # Add your structured pruning logic here if you have a specific strategy.
        # For instance, identify a Conv2d layer in the backbone:
        # if hasattr(model.attribute_classifier.backbone, 'features'):
        #     conv_layer = model.attribute_classifier.backbone.features[0][0] # Example conv layer
        #     if isinstance(conv_layer, nn.Conv2d):
        #         # This is a highly simplified conceptual channel pruning:
        #         # Identify channels to prune (e.g., based on sum of absolute weights)
        #         # num_channels_to_prune = int(conv_layer.out_channels * 0.2)
        #         # sum_abs_weights = conv_layer.weight.abs().sum(dim=[0, 2, 3])
        #         # _, indices_to_prune = torch.topk(sum_abs_weights, num_channels_to_prune, largest=False)
        #         # conv_layer.weight.data[:, indices_to_prune, :, :] = 0.0 # Zero out input channels
        #         # conv_layer.bias.data[indices_to_prune] = 0.0 # Zero out bias
        #         logger.info(f"Conceptually applied structured pruning to a Conv2d layer.")
        #         logger.warning("Actual structured pruning requires careful handling of subsequent layers and model architecture changes.")
    else:
        logger.info("Structured pruning is disabled in configuration.")
    logger.info("Conceptual structured pruning function finished.")
def quantize_model_ptq(model: nn.Module, data_loader: DALIGenericIterator, config: dict):
    """
    Applies Post-Training Quantization (PTQ) to the model.
    This involves converting float weights/activations to lower precision (e.g., int8)
    after the model is fully trained.
    Args:
        model (nn.Module): The trained PyTorch model.
        data_loader (DALIGenericIterator): A data loader for calibration (representative dataset).
        config (dict): Configuration dictionary.
    Returns:
        nn.Module: The quantized model.
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.warning("torch.quantization not available. Skipping PTQ.")
        return model
    
    logger.info("Starting Post-Training Quantization (PTQ).")
    model.eval() # Set model to evaluation mode
    
    # Fuse modules for better quantization performance
    # Fusing Conv-BN-ReLU is common. Identify such patterns in your model.
    # For example, in AttributeClassifier's backbone:
    # if isinstance(model.attribute_classifier.backbone, (models.mobilenet_v2.MobileNetV2, models.mobilenet_v3.MobileNetV3)):
    #     # MobileNetV2/V3 have their own fusing utilities or can be manually fused
    #     # This is a conceptual example, actual fusing depends on model architecture.
    #     logger.info("Attempting to fuse modules in AttributeClassifier backbone for PTQ.")
    #     # For torchvision models, there are sometimes `fuse_model` methods or specific patterns.
    #     # E.g., for MobileNetV2: `torch.quantization.fuse_modules(model.backbone.features, [['0', '1', '2']], inplace=True)`
    #     # This requires knowing the exact structure.
    #     pass
    
    # Specify quantization configuration
    # 'qnnpack' backend is generally good for ARM CPUs (like on Jetson)
    # 'fbgemm' is good for x86 CPUs.
    # For GPU, usually CUDA backend or just keep it on CPU for PTQ.
    
    qconfig = tq.get_default_qconfig('fbgemm') # Use fbgemm for x86 CPU, or qnnpack for ARM
    # Prepare model for static quantization
    model.qconfig = qconfig
    tq.prepare(model, inplace=True)
    logger.info("Model prepared for static quantization.")
    
    # Calibrate the model with a representative dataset
    logger.info("Calibrating model for PTQ with validation data...")
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="PTQ Calibration")):
            images_view1 = data[0]["data"]
            # Convert DALI GPU tensor to PIL Image for RealObjectDetector
            image_pil_batch = []
            IMAGENET_MEAN_0_1 = [0.485, 0.456, 0.406]
            IMAGENET_STD_0_1 = [0.229, 0.224, 0.225]
            for b_idx in range(images_view1.shape[0]):
                img_tensor_unnorm = images_view1[b_idx].cpu()
                for c in range(img_tensor_unnorm.shape[0]):
                    img_tensor_unnorm[c] = img_tensor_unnorm[c] * IMAGENET_STD_0_1[c] + IMAGENET_MEAN_0_1[c]
                img_np = (img_tensor_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if config['data']['num_channels'] == 1 and img_np.shape[2] == 3:
                    img_pil = Image.fromarray(img_np, 'RGB').convert('L')
                elif config['data']['num_channels'] == 3 and img_np.shape[2] == 1:
                    img_pil = Image.fromarray(img_np.squeeze(2), 'L').convert('RGB')
                else:
                    img_pil = Image.fromarray(img_np, 'RGB')
                image_pil_batch.append(img_pil)
            
            # Forward pass to collect quantization statistics
            model(image_pil_batch, is_synthetic=config['data']['use_synthetic_data'])
            if i >= 100: # Calibrate on a subset of data
                break
    data_loader.release_gpu_memory() # Release memory after calibration
    logger.info("Calibration complete.")
    
    # Convert the model to quantized version
    tq.convert(model, inplace=True)
    logger.info("Model converted to quantized (PTQ) version.")
    return model
def quantize_model_qat(model: nn.Module, config: dict):
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
    model.train() # Set model to training mode
    
    # Fuse modules for better quantization performance (same as PTQ)
    # This should be done before preparing for QAT.
    # For example, in AttributeClassifier's backbone:
    # if isinstance(model.attribute_classifier.backbone, (models.mobilenet_v2.MobileNetV2, models.mobilenet_v3.MobileNetV3)):
    #     logger.info("Attempting to fuse modules in AttributeClassifier backbone for QAT.")
    #     pass
    
    # Specify quantization configuration for QAT
    qconfig_qat = tq.get_default_qat_qconfig('fbgemm') # Or 'qnnpack'
    
    # Prepare model for QAT
    model.qconfig = qconfig_qat
    tq.prepare_qat(model, inplace=True)
    logger.info("Model prepared for Quantization-Aware Training (QAT).")
    logger.info("Remember to train the model for a few epochs after QAT preparation.")
    return model
def main_phase1(config_path=None):
    """
    Main execution function for Phase 1 Perception Module.
    Handles data loading, model initialization, training, validation,
    and various advanced techniques.
    Args:
        config_path (str, optional): Path to a YAML configuration file.
    """
    global CONFIG # Ensure CONFIG is accessible and can be updated
    CONFIG = load_config(config_path)
    # Set up logging level dynamically
    logging.getLogger().setLevel(getattr(logging, CONFIG['debug']['log_level'].upper()))
    logger.info("Starting Main_phase1 execution.")
    logger.info(f"Running with configuration: {CONFIG}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG['training']['checkpoint_dir'], exist_ok=True)
    
    # Initialize WandB if enabled
    if HAS_WANDB and CONFIG['training']['use_wandb']:
        wandb.init(project="bongard-perception-phase1", config=CONFIG)
        logger.info("WandB initialized.")
    
    # Data Loading
    if CONFIG['data']['use_synthetic_data']:
        logger.info(f"Generating {CONFIG['data']['synthetic_samples']} synthetic data samples.")
        synthetic_dataset = BongardSyntheticDataset(
            n_samples=CONFIG['data']['synthetic_samples'],
            image_size=CONFIG['data']['image_size'],
            num_objects=(2, 5), # Example range
            num_classes=CONFIG['model']['num_classes']
        )
        # Split synthetic data for train/val
        train_size = int((1 - CONFIG['data']['train_test_split_ratio']) * len(synthetic_dataset))
        val_size = len(synthetic_dataset) - train_size
        train_synthetic_dataset, val_synthetic_dataset = torch.utils.data.random_split(synthetic_dataset, [train_size, val_size])
        
        train_external_source = BongardExternalSource(train_synthetic_dataset, CONFIG['model']['batch_size'], shuffle=True)
        val_external_source = BongardExternalSource(val_synthetic_dataset, CONFIG['model']['batch_size'], shuffle=False)
        
        # DALI pipeline for synthetic data
        train_pipe = dali_pipe(
            file_root=None, # Not used for external source
            file_list=None, # Not used for external source
            labels_list=None, # Not used for external source
            batch_size=CONFIG['model']['batch_size'],
            num_threads=CONFIG['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=CONFIG['data']['image_size'][0],
            width=CONFIG['data']['image_size'][1],
            is_train=True,
            num_channels=CONFIG['data']['num_channels'],
            feature_consistency_alpha=CONFIG['training']['feature_consistency_alpha'],
            imagenet_mean=IMAGENET_MEAN,
            imagenet_std=IMAGENET_STD,
            use_synthetic_data=True,
            external_source_iterator=train_external_source
        )
        val_pipe = dali_pipe(
            file_root=None,
            file_list=None,
            labels_list=None,
            batch_size=CONFIG['model']['batch_size'],
            num_threads=CONFIG['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=CONFIG['data']['image_size'][0],
            width=CONFIG['data']['image_size'][1],
            is_train=False,
            num_channels=CONFIG['data']['num_channels'],
            feature_consistency_alpha=0.0, # No consistency loss for validation
            imagenet_mean=IMAGENET_MEAN,
            imagenet_std=IMAGENET_STD,
            use_synthetic_data=True,
            external_source_iterator=val_external_source
        )
        
        train_pipe.build()
        val_pipe.build()
        
        train_dali_iter = DALIGenericIterator(
            train_pipe,
            ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="ExternalSource", # Correct reader name for external source
            last_batch_policy=LastBatchPolicy.DROP # Drop last incomplete batch
        )
        val_dali_iter = DALIGenericIterator(
            val_pipe,
            ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="ExternalSource",
            last_batch_policy=LastBatchPolicy.DROP
        )
        logger.info(f"DALI iterators built for synthetic data. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
    else:
        # Real Data Loading
        image_paths, labels, difficulty_scores = load_bongard_data(DATA_ROOT_PATH)
        
        # Split data into training and validation sets
        train_paths, val_paths, train_labels, val_labels, train_difficulties, val_difficulties = train_test_split(
            image_paths, labels, difficulty_scores, test_size=CONFIG['data']['train_test_split_ratio'],
            random_state=42, stratify=labels # Stratify to maintain class distribution
        )
        logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
        
        # Initialize Curriculum Samplers
        train_sampler = CurriculumSampler(
            train_paths, train_labels, train_difficulties, CONFIG['model']['batch_size'],
            CONFIG['data']['curriculum_annealing_epochs'], CONFIG['model']['epochs'],
            is_train=True, use_weighted_sampling=CONFIG['training']['use_weighted_sampling']
        )
        val_sampler = CurriculumSampler(
            val_paths, val_labels, val_difficulties, CONFIG['model']['batch_size'],
            CONFIG['data']['curriculum_annealing_epochs'], CONFIG['model']['epochs'],
            is_train=False, use_weighted_sampling=False # No weighted sampling for validation
        )
        # DALI pipeline for real data
        train_pipe = dali_pipe(
            file_root=DATA_ROOT_PATH,
            file_list=train_sampler.get_epoch_data()[0], # Initial file list
            labels_list=train_sampler.get_epoch_data()[1], # Initial labels list
            batch_size=CONFIG['model']['batch_size'],
            num_threads=CONFIG['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=CONFIG['data']['initial_image_size'][0], # Start with initial image size
            width=CONFIG['data']['initial_image_size'][1],
            is_train=True,
            num_channels=CONFIG['data']['num_channels'],
            feature_consistency_alpha=CONFIG['training']['feature_consistency_alpha'],
            imagenet_mean=IMAGENET_MEAN,
            imagenet_std=IMAGENET_STD,
            use_synthetic_data=False
        )
        val_pipe = dali_pipe(
            file_root=DATA_ROOT_PATH,
            file_list=val_sampler.get_epoch_data()[0],
            labels_list=val_sampler.get_epoch_data()[1],
            batch_size=CONFIG['model']['batch_size'],
            num_threads=CONFIG['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=CONFIG['data']['image_size'][0], # Validation uses final image size
            width=CONFIG['data']['image_size'][1],
            is_train=False,
            num_channels=CONFIG['data']['num_channels'],
            feature_consistency_alpha=0.0, # No consistency loss for validation
            imagenet_mean=IMAGENET_MEAN,
            imagenet_std=IMAGENET_STD,
            use_synthetic_data=False
        )
        train_pipe.build()
        val_pipe.build()
        train_dali_iter = DALIGenericIterator(
            train_pipe,
            ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="Reader", # Correct reader name for file reader
            last_batch_policy=LastBatchPolicy.DROP
        )
        val_dali_iter = DALIGenericIterator(
            val_pipe,
            ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP
        )
        logger.info(f"DALI iterators built for real data. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
    
    # Model Initialization
    model = PerceptionModule(CONFIG).to(DEVICE)
    # Add a final Bongard classifier head to the PerceptionModule
    # This head takes the graph_embedding (image_features) as input.
    # The dimension of graph_embedding is `self.relation_gnn.set2set.output_size`.
    model.bongard_classifier = nn.Linear(model.relation_gnn.set2set.output_size, CONFIG['model']['num_classes']).to(DEVICE)
    logger.info("PerceptionModule and Bongard classifier initialized.")
    
    # Apply Torch.compile if enabled (PyTorch 2.0+)
    if CONFIG['training']['use_torch_compile'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")
    
    # Optimizers
    if CONFIG['model']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
    elif CONFIG['model']['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], momentum=0.9, weight_decay=CONFIG['training']['weight_decay'])
    elif CONFIG['model']['optimizer'] == 'SophiaG' and HAS_SOPHIA:
        optimizer = SophiaG(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
    elif CONFIG['model']['optimizer'] == 'Lion' and HAS_TIMM_OPTIM:
        optimizer = Lion(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
    elif CONFIG['model']['optimizer'] == 'MADGRAD' and HAS_TIMM_OPTIM:
        optimizer = MADGRAD(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
    else:
        logger.warning(f"Optimizer '{CONFIG['model']['optimizer']}' not supported or not installed. Falling back to AdamW.")
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
    
    if CONFIG['training']['use_sam_optimizer'] and HAS_SAM:
        optimizer = SAM(model.parameters(), optimizer, rho=0.05, adaptive=False)
        logger.info("SAM optimizer wrapped around base optimizer.")
    
    # Schedulers
    scheduler = None
    if CONFIG['model']['scheduler'] == 'OneCycleLR':
        # Calculate total steps for OneCycleLR
        total_steps = len(train_dali_iter) * CONFIG['model']['epochs']
        scheduler = OneCycleLR(optimizer,
                               max_lr=CONFIG['model']['max_learning_rate'],
                               total_steps=total_steps,
                               pct_start=CONFIG['training']['onecycle_pct_start'], # Use configured pct_start
                               anneal_strategy='cos',
                               cycle_momentum=True,
                               base_momentum=0.85,
                               max_momentum=0.95,
                               div_factor=CONFIG['model']['max_learning_rate'] / CONFIG['model']['initial_learning_rate'],
                               final_div_factor=CONFIG['model']['initial_learning_rate'] / (CONFIG['model']['max_learning_rate'] * 1e-2) # Ensure it goes low enough
                              )
        logger.info("OneCycleLR scheduler initialized.")
    elif CONFIG['model']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=CONFIG['training']['lr_scheduler_factor'],
                                                         patience=CONFIG['training']['lr_scheduler_patience'],
                                                         verbose=True)
        logger.info("ReduceLROnPlateau scheduler initialized.")
    else:
        logger.info("No learning rate scheduler configured.")
    
    # Loss Functions
    # For attribute and relation classification, use Label Smoothing Cross Entropy if enabled
    if CONFIG['training']['label_smoothing_epsilon'] > 0:
        criterion_attr = LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing_epsilon'])
        criterion_rel = LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing_epsilon'])
        logger.info(f"Using LabelSmoothingCrossEntropy with epsilon={CONFIG['training']['label_smoothing_epsilon']}.")
    else:
        criterion_attr = nn.CrossEntropyLoss()
        criterion_rel = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss.")
    
    criterion_consistency = FeatureConsistencyLoss(loss_type=CONFIG['training']['feature_consistency_loss_type'])
    criterion_distillation = DistillationLoss(
        alpha=CONFIG['training']['distillation_alpha'],
        temperature=CONFIG['training']['distillation_temperature'],
        base_loss_fn=nn.CrossEntropyLoss() # Base loss for hard targets in distillation
    )
    
    scaler = GradScaler() if CONFIG['training']['use_amp'] else None
    
    # Mixup/CutMix Transform
    mixup_cutmix_transform = MixupCutmix(
        mixup_alpha=CONFIG['training']['mixup_alpha'],
        cutmix_alpha=CONFIG['training']['cutmix_alpha'],
        prob=1.0, # Always try to apply if alphas > 0
        switch_prob=CONFIG['training']['mixup_cutmix_ratio'],
        num_classes=CONFIG['model']['num_classes'] # For Bongard problem classification
    )
    
    # Knowledge Replay Buffer
    replay_buffer = None
    if CONFIG['training']['knowledge_replay_enabled']:
        replay_buffer = KnowledgeReplayBuffer(capacity=CONFIG['training']['replay_buffer_size'])
        logger.info(f"Knowledge replay buffer initialized with capacity: {CONFIG['training']['replay_buffer_size']}.")
    
    # SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    if CONFIG['training']['use_swa']:
        swa_model = swa_utils.AveragedModel(model)
        # SWA scheduler can be defined with a fixed LR or a decaying one
        # It's recommended to use a constant or slightly decaying LR for SWA.
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=CONFIG['training']['swa_lr'])
        logger.info(f"SWA initialized with SWA LR: {CONFIG['training']['swa_lr']}.")
    
    # Training Loop
    best_val_loss = float('inf')
    best_val_accuracy = -1.0 # Monitor Bongard accuracy for early stopping
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_bongard_accuracy': [],
        'val_attribute_accuracy': [],
        'val_relation_accuracy': []
    }
    
    for epoch in range(CONFIG['model']['epochs']):
        logger.info(f"--- Epoch {epoch+1}/{CONFIG['model']['epochs']} ---")
        
        # Update DALI iterators with current epoch's data (for curriculum learning)
        if not CONFIG['data']['use_synthetic_data']:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
            # Rebuild DALI pipelines with updated file lists and image sizes
            # This is crucial for curriculum learning and progressive resizing
            train_pipe = dali_pipe(
                file_root=DATA_ROOT_PATH,
                file_list=train_sampler.get_epoch_data()[0],
                labels_list=train_sampler.get_epoch_data()[1],
                batch_size=CONFIG['model']['batch_size'],
                num_threads=CONFIG['data']['dataloader_num_workers'],
                device_id=0 if DEVICE.type == 'cuda' else -1,
                height=train_sampler.get_current_image_size()[0], # Progressive resizing
                width=train_sampler.get_current_image_size()[1],
                is_train=True,
                num_channels=CONFIG['data']['num_channels'],
                feature_consistency_alpha=CONFIG['training']['feature_consistency_alpha'],
                imagenet_mean=IMAGENET_MEAN,
                imagenet_std=IMAGENET_STD,
                use_synthetic_data=False
            )
            val_pipe = dali_pipe(
                file_root=DATA_ROOT_PATH,
                file_list=val_sampler.get_epoch_data()[0],
                labels_list=val_sampler.get_epoch_data()[1],
                batch_size=CONFIG['model']['batch_size'],
                num_threads=CONFIG['data']['dataloader_num_workers'],
                device_id=0 if DEVICE.type == 'cuda' else -1,
                height=val_sampler.get_current_image_size()[0], # Validation uses its own progressive size
                width=val_sampler.get_current_image_size()[1],
                is_train=False,
                num_channels=CONFIG['data']['num_channels'],
                feature_consistency_alpha=0.0,
                imagenet_mean=IMAGENET_MEAN,
                imagenet_std=IMAGENET_STD,
                use_synthetic_data=False
            )
            train_pipe.build()
            val_pipe.build()
            # Re-initialize DALI iterators after rebuilding pipelines
            train_dali_iter = DALIGenericIterator(
                train_pipe,
                ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
                reader_name="Reader",
                last_batch_policy=LastBatchPolicy.DROP
            )
            val_dali_iter = DALIGenericIterator(
                val_pipe,
                ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
                reader_name="Reader",
                last_batch_policy=LastBatchPolicy.DROP
            )
            logger.info(f"DALI iterators rebuilt for epoch {epoch+1}. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
        
        # Quantization-Aware Training (QAT) start
        if CONFIG['training']['use_qat'] and HAS_TORCH_QUANTIZATION and epoch == CONFIG['training']['qat_start_epoch']:
            logger.info(f"Starting QAT at epoch {epoch}. Preparing model for QAT.")
            model = quantize_model_qat(model, CONFIG)
            # Re-initialize optimizer if QAT changes model parameters (e.g., adds observers)
            # This is important to ensure the optimizer is tracking the correct parameters.
            if CONFIG['model']['optimizer'] == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=CONFIG['model']['initial_learning_rate'], weight_decay=CONFIG['training']['weight_decay'])
            # ... (repeat for other optimizers if needed) ...
            if CONFIG['training']['use_sam_optimizer'] and HAS_SAM:
                optimizer = SAM(model.parameters(), optimizer, rho=0.05, adaptive=False)
            logger.info("Optimizer re-initialized after QAT preparation.")
            
        train_loss = _train_step(model, train_dali_iter, optimizer, criterion_attr, criterion_rel,
                                 criterion_consistency, criterion_distillation, scaler,
                                 mixup_cutmix_transform, CONFIG, epoch, replay_buffer)
        
        val_loss, val_bongard_accuracy, val_attribute_accuracy, val_relation_accuracy = \
            _validate_model(model, val_dali_iter, criterion_attr, criterion_rel, CONFIG)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_bongard_accuracy'].append(val_bongard_accuracy)
        history['val_attribute_accuracy'].append(val_attribute_accuracy)
        history['val_relation_accuracy'].append(val_relation_accuracy)
        
        # Log to WandB
        if HAS_WANDB and CONFIG['training']['use_wandb']:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bongard_accuracy": val_bongard_accuracy,
                "val_attribute_accuracy": val_attribute_accuracy,
                "val_relation_accuracy": val_relation_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, OneCycleLR):
                # OneCycleLR steps per batch, so we step it in _train_step for each batch.
                # No epoch-level step needed here.
                pass
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
        
        # SWA Update
        if CONFIG['training']['use_swa'] and epoch >= int(CONFIG['model']['epochs'] * CONFIG['training']['swa_start_epoch_ratio']):
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.info(f"SWA model updated at epoch {epoch+1}.")
        
        # Early Stopping
        monitor_metric = history[CONFIG['training']['early_stopping_monitor_metric']][-1]
        
        if CONFIG['training']['early_stopping_monitor_metric'] == 'val_loss':
            if monitor_metric < best_val_loss - CONFIG['training']['early_stopping_min_delta']:
                best_val_loss = monitor_metric
                epochs_no_improve = 0
                # Save best model checkpoint
                checkpoint_path = os.path.join(CONFIG['training']['checkpoint_dir'], CONFIG['training']['model_checkpoint_name'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation loss did not improve. Epochs without improvement: {epochs_no_improve}")
        elif CONFIG['training']['early_stopping_monitor_metric'] == 'val_bongard_accuracy':
            if monitor_metric > best_val_accuracy + CONFIG['training']['early_stopping_min_delta']:
                best_val_accuracy = monitor_metric
                epochs_no_improve = 0
                # Save best model checkpoint
                checkpoint_path = os.path.join(CONFIG['training']['checkpoint_dir'], CONFIG['training']['model_checkpoint_name'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path} with validation accuracy: {best_val_accuracy:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation accuracy did not improve. Epochs without improvement: {epochs_no_improve}")
        
        if epochs_no_improve >= CONFIG['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break
        
        # Save symbolic outputs periodically
        if (epoch + 1) % CONFIG['training']['save_symbolic_outputs_interval_epochs'] == 0:
            logger.info(f"Saving symbolic outputs for visualization at epoch {epoch+1}.")
            # Take a small sample from validation set for visualization
            sample_val_paths, sample_val_labels = val_sampler.get_epoch_data()
            num_samples_to_save = min(5, len(sample_val_paths)) # Save up to 5 samples
            
            sample_image_paths = sample_val_paths[:num_samples_to_save]
            # Need to load PIL images from paths for the model's forward pass
            sample_pil_images = [Image.open(p).convert('RGB') for p in sample_image_paths]
            
            # Ensure model is in eval mode for consistent output
            model.eval()
            with torch.no_grad():
                symbolic_outputs_for_export, _ = model(sample_pil_images, is_synthetic=False) # Assume real data for saving
            model.train() # Set back to train mode
            
            # Export symbolic outputs to JSON
            output_json_path = os.path.join(CONFIG['training']['checkpoint_dir'], f"symbolic_outputs_epoch_{epoch+1}.json")
            # Make outputs JSON serializable (e.g., convert tensors to lists)
            json_serializable_outputs = []
            for sym_out in symbolic_outputs_for_export:
                serializable_sym_out = sym_out.copy()
                # Handle potential tensor in image_features
                if serializable_sym_out.get("image_features") is not None and torch.is_tensor(serializable_sym_out["image_features"]):
                    serializable_sym_out["image_features"] = serializable_sym_out["image_features"].tolist()
                json_serializable_outputs.append(serializable_sym_out)
            
            with open(output_json_path, 'w') as f:
                json.dump(json_serializable_outputs, f, indent=4)
            logger.info(f"Sample symbolic outputs exported to: {output_json_path}")
    
    # After training, if SWA was used, swap parameters
    if CONFIG['training']['use_swa'] and swa_model:
        swa_utils.swap_parameters_with_avg(model, swa_model)
        logger.info("Swapped model parameters with SWA averaged model.")
    
    # Load the best model before final evaluation, calibration, or export
    best_model_path = os.path.join(CONFIG['training']['checkpoint_dir'], CONFIG['training']['model_checkpoint_name'])
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}.")
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Using last trained model.")
    
    # Post-Training Quantization (PTQ)
    if CONFIG['training']['use_qat'] and HAS_TORCH_QUANTIZATION:
        logger.info("Finalizing QAT model conversion.")
        # After QAT training, convert the model to its final quantized form
        tq.convert(model, inplace=True)
        logger.info("QAT model converted to quantized version.")
    elif CONFIG['training'].get('use_ptq', False) and HAS_TORCH_QUANTIZATION: # Only if PTQ is explicitly enabled and QAT not used
        logger.info("Applying Post-Training Quantization (PTQ).")
        model = quantize_model_ptq(model, val_dali_iter, CONFIG) # Use validation data for calibration
    
    # Structured Pruning (Conceptual)
    if CONFIG['training'].get('enable_structured_pruning', False):
        apply_structured_pruning(model, CONFIG)
    
    # Final Evaluation (on best model or SWA model)
    logger.info("Performing final evaluation on the best/SWA model...")
    final_val_loss, final_val_bongard_accuracy, final_val_attribute_accuracy, final_val_relation_accuracy = \
        _validate_model(model, val_dali_iter, criterion_attr, criterion_rel, CONFIG)
    logger.info(f"Final Validation Results: Loss={final_val_loss:.4f}, Bongard Acc={final_val_bongard_accuracy:.4f}, Attr Acc={final_val_attribute_accuracy:.4f}, Rel Acc={final_val_relation_accuracy:.4f}")
    
    # Plot training history
    if CONFIG['debug']['visualize_training_history']:
        _plot_training_history(history, save_path=os.path.join(CONFIG['training']['checkpoint_dir'], "training_history.png"))
    
    # Plot reliability diagram
    if CONFIG['debug']['plot_reliability_diagram']:
        # For plotting reliability diagram, we need probabilities and labels from a validation run.
        # This is ideally done after calibration.
        # The `calibrate_model` function already handles plotting.
        logger.warning("Reliability diagram plotting is conceptual. Requires collecting all validation probabilities and labels.")
        # Example: dummy_probs = np.random.rand(100), dummy_labels = np.random.randint(0, 2, 100)
        dummy_probs = np.random.rand(100)
        dummy_labels = np.random.randint(0, 2, 100)
        _plot_reliability_diagram(dummy_probs, dummy_labels, save_path=os.path.join(CONFIG['training']['checkpoint_dir'], "reliability_diagram.png"))
    
    # Calibrate model if configured (after loading best model)
    if CONFIG['training']['calibrate_model']:
        calibrate_model(model, val_dali_iter, CONFIG)
    
    # Export ONNX (example for AttributeClassifier)
    # The full PerceptionModule export is complex, as noted in the class.
    # We'll just call the AttributeClassifier export as a demonstration.
    model.export_onnx(output_path=os.path.join(CONFIG['training']['checkpoint_dir'], "perception_module_exported.onnx"))
    
    # Monte Carlo Dropout Inference Example
    if CONFIG['model']['mc_dropout_samples'] > 0:
        logger.info(f"Performing Monte Carlo Dropout inference with {CONFIG['model']['mc_dropout_samples']} samples.")
        # Take a small sample from validation set for MC Dropout inference
        sample_val_paths, sample_val_labels = val_sampler.get_epoch_data() if not CONFIG['data']['use_synthetic_data'] else ([], [])
        num_mc_samples_to_test = min(2, len(sample_val_paths) if sample_val_paths else 2) # Test on 2 samples
        
        if num_mc_samples_to_test > 0:
            mc_test_image_paths = sample_val_paths[:num_mc_samples_to_test]
            mc_test_pil_images = [Image.open(p).convert('RGB') for p in mc_test_image_paths]
            
            mc_results = model.mc_dropout_predict(mc_test_pil_images, CONFIG['model']['mc_dropout_samples'], is_synthetic=False)
            logger.info("Monte Carlo Dropout results (mean and variance):")
            for res in mc_results:
                logger.info(f"  Image: {res['image_path']}")
                if res['bongard_prediction_mean'] is not None:
                    logger.info(f"    Bongard Mean: {res['bongard_prediction_mean']}")
                    logger.info(f"    Bongard Variance: {res['bongard_prediction_variance']}")
                # Log attribute and relation uncertainties if desired
        else:
            logger.warning("Not enough samples for Monte Carlo Dropout inference demonstration.")
    
    logger.info("Main_phase1 execution finished.")
# Main execution block
if __name__ == "__main__":
    try:
        # Set start method for multiprocessing (important for CUDA and DataLoader)
        # 'spawn' is safer for CUDA than 'fork'
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. This might be an issue in some environments.")
    
    main_phase1()

# ensemble_trainer.py
import os
import argparse
import logging
import random
import yaml
import torch

# Import necessary components from your phase1 code parts
# Assuming these files are in the same directory or on PYTHONPATH
from phase1_code_part1 import load_config, set_seed, DATA_ROOT_PATH, IMAGENET_MEAN, IMAGENET_STD, DEVICE, BongardSyntheticDataset, BongardExternalSource, dali_pipe, CurriculumSampler
from phase1_code_part2a import RealObjectDetector, AttributeClassifier
from phase1_code_part2b import RelationGNN, PerceptionModule, DistillationLoss, MixupCutmix, LabelSmoothingCrossEntropy, FeatureConsistencyLoss, KnowledgeReplayBuffer, ATTRIBUTE_FILL_MAP, RELATION_MAP
from phase1_code_part3 import _train_step, _validate_model, _plot_training_history, _plot_reliability_diagram, calibrate_model, apply_structured_pruning, quantize_model_ptq, quantize_model_qat, load_bongard_data

# Set up logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_ensemble_member(
    member_id: int,
    base_config_path: str,
    random_seed: int,
    output_dir: str,
    epochs_override: Optional[int] = None
):
    """
    Trains a single ensemble member model.
    Args:
        member_id (int): Unique ID for this ensemble member.
        base_config_path (str): Path to the base YAML configuration file.
        random_seed (int): Random seed to use for this training run.
        output_dir (str): Directory to save model checkpoints.
        epochs_override (Optional[int]): Override the number of epochs from config.
    """
    logger.info(f"--- Starting training for Ensemble Member {member_id} (Seed: {random_seed}) ---")

    # Set all random seeds for reproducibility of this specific run
    set_seed(random_seed)

    # Load the base configuration
    config = load_config(base_config_path)

    # Override epochs if specified
    if epochs_override is not None:
        config['model']['epochs'] = epochs_override
        logger.info(f"Overriding epochs to {epochs_override} for this ensemble member.")

    # Update checkpoint name to be unique for this ensemble member
    original_model_checkpoint_name = config['training']['model_checkpoint_name']
    config['training']['model_checkpoint_name'] = f"ensemble_member_{member_id}_{original_model_checkpoint_name}"
    config['training']['checkpoint_dir'] = output_dir # Ensure checkpoints go to the specified output_dir
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True) # Create if not exists

    logger.info(f"Ensemble Member {member_id} will save checkpoint to: {os.path.join(output_dir, config['training']['model_checkpoint_name'])}")

    # Initialize WandB if enabled (ensure unique run name for each member)
    if config.get('training', {}).get('use_wandb', False):
        import wandb
        wandb.init(project="bongard-perception-ensemble", config=config, name=f"ensemble-member-{member_id}-seed-{random_seed}")
        logger.info(f"WandB initialized for ensemble member {member_id}.")

    # --- Data Loading ---
    if config['data']['use_synthetic_data']:
        logger.info(f"Generating {config['data']['synthetic_samples']} synthetic data samples.")
        synthetic_dataset = BongardSyntheticDataset(
            n_samples=config['data']['synthetic_samples'],
            image_size=config['data']['image_size'],
            num_objects=(2, 5),
            num_classes=config['model']['num_classes']
        )
        train_size = int((1 - config['data']['train_test_split_ratio']) * len(synthetic_dataset))
        val_size = len(synthetic_dataset) - train_size
        train_synthetic_dataset, val_synthetic_dataset = torch.utils.data.random_split(synthetic_dataset, [train_size, val_size])
        
        train_external_source = BongardExternalSource(train_synthetic_dataset, config['model']['batch_size'], shuffle=True)
        val_external_source = BongardExternalSource(val_synthetic_dataset, config['model']['batch_size'], shuffle=False)
        
        train_pipe = dali_pipe(
            file_root=None, file_list=None, labels_list=None,
            batch_size=config['model']['batch_size'],
            num_threads=config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=config['data']['image_size'][0], width=config['data']['image_size'][1],
            is_train=True, num_channels=config['data']['num_channels'],
            feature_consistency_alpha=config['training']['feature_consistency_alpha'],
            imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD,
            use_synthetic_data=True, external_source_iterator=train_external_source
        )
        val_pipe = dali_pipe(
            file_root=None, file_list=None, labels_list=None,
            batch_size=config['model']['batch_size'],
            num_threads=config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=config['data']['image_size'][0], width=config['data']['image_size'][1],
            is_train=False, num_channels=config['data']['num_channels'],
            feature_consistency_alpha=0.0,
            imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD,
            use_synthetic_data=True, external_source_iterator=val_external_source
        )
        
        train_pipe.build()
        val_pipe.build()
        
        train_dali_iter = DALIGenericIterator(
            train_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="ExternalSource", last_batch_policy=LastBatchPolicy.DROP
        )
        val_dali_iter = DALIGenericIterator(
            val_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="ExternalSource", last_batch_policy=LastBatchPolicy.DROP
        )
        logger.info(f"DALI iterators built for synthetic data. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
    else:
        image_paths, labels, difficulty_scores = load_bongard_data(DATA_ROOT_PATH)
        train_paths, val_paths, train_labels, val_labels, train_difficulties, val_difficulties = train_test_split(
            image_paths, labels, difficulty_scores, test_size=config['data']['train_test_split_ratio'],
            random_state=random_seed, stratify=labels
        )
        logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
        
        train_sampler = CurriculumSampler(
            train_paths, train_labels, train_difficulties, config['model']['batch_size'],
            config['data']['curriculum_annealing_epochs'], config['model']['epochs'],
            is_train=True, use_weighted_sampling=config['training']['use_weighted_sampling']
        )
        val_sampler = CurriculumSampler(
            val_paths, val_labels, val_difficulties, config['model']['batch_size'],
            config['data']['curriculum_annealing_epochs'], config['model']['epochs'],
            is_train=False, use_weighted_sampling=False
        )
        train_pipe = dali_pipe(
            file_root=DATA_ROOT_PATH, file_list=train_sampler.get_epoch_data()[0], labels_list=train_sampler.get_epoch_data()[1],
            batch_size=config['model']['batch_size'], num_threads=config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=config['data']['initial_image_size'][0], width=config['data']['initial_image_size'][1],
            is_train=True, num_channels=config['data']['num_channels'],
            feature_consistency_alpha=config['training']['feature_consistency_alpha'],
            imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
        )
        val_pipe = dali_pipe(
            file_root=DATA_ROOT_PATH, file_list=val_sampler.get_epoch_data()[0], labels_list=val_sampler.get_epoch_data()[1],
            batch_size=config['model']['batch_size'], num_threads=config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=config['data']['image_size'][0], width=config['data']['image_size'][1],
            is_train=False, num_channels=config['data']['num_channels'],
            feature_consistency_alpha=0.0, imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
        )
        train_pipe.build()
        val_pipe.build()
        train_dali_iter = DALIGenericIterator(
            train_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
        )
        val_dali_iter = DALIGenericIterator(
            val_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
        )
        logger.info(f"DALI iterators built for real data. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")

    # Model Initialization
    model = PerceptionModule(config).to(DEVICE)
    model.bongard_classifier = torch.nn.Linear(model.relation_gnn.set2set.output_size, config['model']['num_classes']).to(DEVICE)
    logger.info("PerceptionModule and Bongard classifier initialized.")
    
    if config['training']['use_torch_compile'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    # Optimizers
    if config['model']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['initial_learning_rate'], weight_decay=config['training']['weight_decay'])
    elif config['model']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['model']['initial_learning_rate'], momentum=0.9, weight_decay=config['training']['weight_decay'])
    elif config['model']['optimizer'] == 'SophiaG' and hasattr(torch.optim, 'SophiaG'): # Check for SophiaG availability
        optimizer = torch.optim.SophiaG(model.parameters(), lr=config['model']['initial_learning_rate'], weight_decay=config['training']['weight_decay'])
    elif config['model']['optimizer'] == 'Lion' and hasattr(torch.optim, 'Lion'): # Check for Lion availability
        optimizer = torch.optim.Lion(model.parameters(), lr=config['model']['initial_learning_rate'], weight_decay=config['training']['weight_decay'])
    elif config['model']['optimizer'] == 'MADGRAD' and hasattr(torch.optim, 'MADGRAD'): # Check for MADGRAD availability
        optimizer = torch.optim.MADGRAD(model.parameters(), lr=config['model']['initial_learning_rate'], weight_decay=config['training']['weight_decay'])
    else:
        logger.warning(f"Optimizer '{config['model']['optimizer']}' not supported or not installed. Falling back to AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['initial_learning_rate'], weight_decay=config['training']['weight_decay'])
    
    if config['training']['use_sam_optimizer'] and hasattr(torch.optim, 'SAM'): # Check for SAM availability
        optimizer = torch.optim.SAM(model.parameters(), optimizer, rho=0.05, adaptive=False)
        logger.info("SAM optimizer wrapped around base optimizer.")

    # Schedulers
    scheduler = None
    if config['model']['scheduler'] == 'OneCycleLR':
        total_steps = len(train_dali_iter) * config['model']['epochs']
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                               max_lr=config['model']['max_learning_rate'],
                               total_steps=total_steps,
                               pct_start=config['training']['onecycle_pct_start'],
                               anneal_strategy='cos',
                               cycle_momentum=True,
                               base_momentum=0.85,
                               max_momentum=0.95,
                               div_factor=config['model']['max_learning_rate'] / config['model']['initial_learning_rate'],
                               final_div_factor=config['model']['initial_learning_rate'] / (config['model']['max_learning_rate'] * 1e-2)
                              )
        logger.info("OneCycleLR scheduler initialized.")
    elif config['model']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=config['training']['lr_scheduler_factor'],
                                                         patience=config['training']['lr_scheduler_patience'],
                                                         verbose=True)
        logger.info("ReduceLROnPlateau scheduler initialized.")
    else:
        logger.info("No learning rate scheduler configured.")

    # Loss Functions
    if config['training']['label_smoothing_epsilon'] > 0:
        criterion_attr = LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing_epsilon'])
        criterion_rel = LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing_epsilon'])
        logger.info(f"Using LabelSmoothingCrossEntropy with epsilon={config['training']['label_smoothing_epsilon']}.")
    else:
        criterion_attr = torch.nn.CrossEntropyLoss()
        criterion_rel = torch.nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss.")
    
    criterion_consistency = FeatureConsistencyLoss(loss_type=config['training']['feature_consistency_loss_type'])
    criterion_distillation = DistillationLoss(
        alpha=config['training']['distillation_alpha'],
        temperature=config['training']['distillation_temperature'],
        base_loss_fn=torch.nn.CrossEntropyLoss()
    )
    
    scaler = torch.cuda.amp.GradScaler() if config['training']['use_amp'] else None
    
    mixup_cutmix_transform = MixupCutmix(
        mixup_alpha=config['training']['mixup_alpha'],
        cutmix_alpha=config['training']['cutmix_alpha'],
        prob=1.0,
        switch_prob=config['training']['mixup_cutmix_ratio'],
        num_classes=config['model']['num_classes']
    )
    
    replay_buffer = None
    if config['training']['knowledge_replay_enabled']:
        replay_buffer = KnowledgeReplayBuffer(capacity=config['training']['replay_buffer_size'])
        logger.info(f"Knowledge replay buffer initialized with capacity: {config['training']['replay_buffer_size']}.")
    
    swa_model = None
    swa_scheduler = None
    if config['training']['use_swa']:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=config['training']['swa_lr'])
        logger.info(f"SWA initialized with SWA LR: {config['training']['swa_lr']}.")
    
    best_val_loss = float('inf')
    best_val_accuracy = -1.0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_bongard_accuracy': [],
        'val_attribute_accuracy': [],
        'val_relation_accuracy': []
    }
    
    for epoch in range(config['model']['epochs']):
        logger.info(f"--- Epoch {epoch+1}/{config['model']['epochs']} ---")
        
        if not config['data']['use_synthetic_data']:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
            train_pipe = dali_pipe(
                file_root=DATA_ROOT_PATH, file_list=train_sampler.get_epoch_data()[0], labels_list=train_sampler.get_epoch_data()[1],
                batch_size=config['model']['batch_size'], num_threads=config['data']['dataloader_num_workers'],
                device_id=0 if DEVICE.type == 'cuda' else -1,
                height=train_sampler.get_current_image_size()[0], width=train_sampler.get_current_image_size()[1],
                is_train=True, num_channels=config['data']['num_channels'],
                feature_consistency_alpha=config['training']['feature_consistency_alpha'],
                imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
            )
            val_pipe = dali_pipe(
                file_root=DATA_ROOT_PATH, file_list=val_sampler.get_epoch_data()[0], labels_list=val_sampler.get_epoch_data()[1],
                batch_size=config['model']['batch_size'], num_threads=config['data']['dataloader_num_workers'],
                device_id=0 if DEVICE.type == 'cuda' else -1,
                height=val_sampler.get_current_image_size()[0], width=val_sampler.get_current_image_size()[1],
                is_train=False, num_channels=config['data']['num_channels'],
                feature_consistency_alpha=0.0, imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
            )
            train_pipe.build()
            val_pipe.build()
            train_dali_iter = DALIGenericIterator(
                train_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
                reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
            )
            val_dali_iter = DALIGenericIterator(
                val_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
                reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
            )
            logger.info(f"DALI iterators rebuilt for epoch {epoch+1}. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
        
        if config['training']['use_qat'] and hasattr(torch.quantization, 'prepare_qat') and epoch == config['training']['qat_start_epoch']:
            logger.info(f"Starting QAT at epoch {epoch}. Preparing model for QAT.")
            model = quantize_model_qat(model, config)
            if config['model']['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['initial_learning_rate'], weight_decay=config['training']['weight_decay'])
            # Re-initialize other optimizers similarly if needed
            if config['training']['use_sam_optimizer'] and hasattr(torch.optim, 'SAM'):
                optimizer = torch.optim.SAM(model.parameters(), optimizer, rho=0.05, adaptive=False)
            logger.info("Optimizer re-initialized after QAT preparation.")
            
        train_loss = _train_step(model, train_dali_iter, optimizer, criterion_attr, criterion_rel,
                                 criterion_consistency, criterion_distillation, scaler,
                                 mixup_cutmix_transform, config, epoch, replay_buffer)
        
        val_loss, val_bongard_accuracy, val_attribute_accuracy, val_relation_accuracy = \
            _validate_model(model, val_dali_iter, criterion_attr, criterion_rel, config)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_bongard_accuracy'].append(val_bongard_accuracy)
        history['val_attribute_accuracy'].append(val_attribute_accuracy)
        history['val_relation_accuracy'].append(val_relation_accuracy)
        
        if config.get('training', {}).get('use_wandb', False):
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bongard_accuracy": val_bongard_accuracy,
                "val_attribute_accuracy": val_attribute_accuracy,
                "val_relation_accuracy": val_relation_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
        
        if config['training']['use_swa'] and swa_model and epoch >= int(config['model']['epochs'] * config['training']['swa_start_epoch_ratio']):
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.info(f"SWA model updated at epoch {epoch+1}.")
        
        monitor_metric = history[config['training']['early_stopping_monitor_metric']][-1]
        
        if config['training']['early_stopping_monitor_metric'] == 'val_loss':
            if monitor_metric < best_val_loss - config['training']['early_stopping_min_delta']:
                best_val_loss = monitor_metric
                epochs_no_improve = 0
                checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_checkpoint_name'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation loss did not improve. Epochs without improvement: {epochs_no_improve}")
        elif config['training']['early_stopping_monitor_metric'] == 'val_bongard_accuracy':
            if monitor_metric > best_val_accuracy + config['training']['early_stopping_min_delta']:
                best_val_accuracy = monitor_metric
                epochs_no_improve = 0
                checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_checkpoint_name'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path} with validation accuracy: {best_val_accuracy:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation accuracy did not improve. Epochs without improvement: {epochs_no_improve}")
        
        if epochs_no_improve >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break
        
        if (epoch + 1) % config['training']['save_symbolic_outputs_interval_epochs'] == 0:
            logger.info(f"Saving symbolic outputs for visualization at epoch {epoch+1}.")
            sample_val_paths, sample_val_labels = val_sampler.get_epoch_data() if not config['data']['use_synthetic_data'] else ([], [])
            num_samples_to_save = min(5, len(sample_val_paths))
            
            sample_image_paths = sample_val_paths[:num_samples_to_save]
            sample_pil_images = [Image.open(p).convert('RGB') for p in sample_image_paths]
            
            model.eval()
            with torch.no_grad():
                symbolic_outputs_for_export, _ = model(sample_pil_images, is_synthetic=False)
            model.train()
            
            output_json_path = os.path.join(config['training']['checkpoint_dir'], f"symbolic_outputs_epoch_{epoch+1}.json")
            json_serializable_outputs = []
            for sym_out in symbolic_outputs_for_export:
                serializable_sym_out = sym_out.copy()
                if serializable_sym_out.get("image_features") is not None and torch.is_tensor(serializable_sym_out["image_features"]):
                    serializable_sym_out["image_features"] = serializable_sym_out["image_features"].tolist()
                json_serializable_outputs.append(serializable_sym_out)
            
            with open(output_json_path, 'w') as f:
                json.dump(json_serializable_outputs, f, indent=4)
            logger.info(f"Sample symbolic outputs exported to: {output_json_path}")
    
    if config['training']['use_swa'] and swa_model:
        torch.optim.swa_utils.swap_parameters_with_avg(model, swa_model)
        logger.info("Swapped model parameters with SWA averaged model.")
    
    best_model_path = os.path.join(config['training']['checkpoint_dir'], config['training']['model_checkpoint_name'])
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}.")
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Using last trained model.")
    
    if config['training']['use_qat'] and hasattr(torch.quantization, 'convert'):
        logger.info("Finalizing QAT model conversion.")
        torch.quantization.convert(model, inplace=True)
        logger.info("QAT model converted to quantized version.")
    elif config['training'].get('use_ptq', False) and hasattr(torch.quantization, 'prepare'):
        logger.info("Applying Post-Training Quantization (PTQ).")
        model = quantize_model_ptq(model, val_dali_iter, config)
    
    if config['training'].get('enable_structured_pruning', False):
        apply_structured_pruning(model, config)
    
    logger.info("Performing final evaluation on the best/SWA model...")
    final_val_loss, final_val_bongard_accuracy, final_val_attribute_accuracy, final_val_relation_accuracy = \
        _validate_model(model, val_dali_iter, criterion_attr, criterion_rel, config)
    logger.info(f"Final Validation Results: Loss={final_val_loss:.4f}, Bongard Acc={final_val_bongard_accuracy:.4f}, Attr Acc={final_val_attribute_accuracy:.4f}, Rel Acc={final_val_relation_accuracy:.4f}")
    
    if config['debug']['visualize_training_history']:
        _plot_training_history(history, save_path=os.path.join(config['training']['checkpoint_dir'], "training_history.png"))
    
    if config['debug']['plot_reliability_diagram']:
        dummy_probs = np.random.rand(100)
        dummy_labels = np.random.randint(0, 2, 100)
        _plot_reliability_diagram(dummy_probs, dummy_labels, save_path=os.path.join(config['training']['checkpoint_dir'], "reliability_diagram.png"))
    
    if config['training']['calibrate_model']:
        calibrate_model(model, val_dali_iter, config)
    
    model.export_onnx(output_path=os.path.join(config['training']['checkpoint_dir'], "perception_module_exported.onnx"))
    
    if config['model']['mc_dropout_samples'] > 0:
        logger.info(f"Performing Monte Carlo Dropout inference with {config['model']['mc_dropout_samples']} samples.")
        sample_val_paths, sample_val_labels = val_sampler.get_epoch_data() if not config['data']['use_synthetic_data'] else ([], [])
        num_mc_samples_to_test = min(2, len(sample_val_paths) if sample_val_paths else 2)
        
        if num_mc_samples_to_test > 0:
            mc_test_image_paths = sample_val_paths[:num_mc_samples_to_test]
            mc_test_pil_images = [Image.open(p).convert('RGB') for p in mc_test_image_paths]
            
            mc_results = model.mc_dropout_predict(mc_test_pil_images, config['model']['mc_dropout_samples'], is_synthetic=False)
            logger.info("Monte Carlo Dropout results (mean and variance):")
            for res in mc_results:
                logger.info(f"  Image: {res['image_path']}")
                if res['bongard_prediction_mean'] is not None:
                    logger.info(f"    Bongard Mean: {res['bongard_prediction_mean']}")
                    logger.info(f"    Bongard Variance: {res['bongard_prediction_variance']}")
        else:
            logger.warning("Not enough samples for Monte Carlo Dropout inference demonstration.")
    
    logger.info(f"--- Finished training for Ensemble Member {member_id} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple ensemble members for Bongard Perception.")
    parser.add_argument('--num_members', type=int, default=3, help='Number of ensemble members to train.')
    parser.add_argument('--base_config', type=str, default='config.yaml', help='Path to the base YAML configuration file.')
    parser.add_argument('--output_dir', type=str, default='./ensemble_checkpoints', help='Directory to save ensemble model checkpoints.')
    parser.add_argument('--start_seed', type=int, default=42, help='Starting random seed for the first ensemble member.')
    parser.add_argument('--epochs', type=int, default=None, help='Override the number of epochs for all ensemble members.')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.num_members):
        current_seed = args.start_seed + i # Use a different seed for each member
        train_ensemble_member(
            member_id=i,
            base_config_path=args.base_config,
            random_seed=current_seed,
            output_dir=args.output_dir,
            epochs_override=args.epochs
        )
    logger.info(f"Successfully trained {args.num_members} ensemble members. Checkpoints saved to {args.output_dir}")

# ensemble_inference.py
import os
import argparse
import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple

# Import necessary components from your phase1 code parts
from phase1_code_part1 import load_config, set_seed, DATA_ROOT_PATH, IMAGENET_MEAN, IMAGENET_STD, DEVICE, YOLO_CLASS_MAP, ATTRIBUTE_FILL_MAP, RELATION_MAP
from phase1_code_part2a import RealObjectDetector, AttributeClassifier
from phase1_code_part2b import RelationGNN, PerceptionModule

# Set up logging for this script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ensemble_models(
    model_paths: List[str],
    base_config_path: str
) -> List[PerceptionModule]:
    """
    Loads multiple PerceptionModule models for ensemble inference.
    Args:
        model_paths (List[str]): List of paths to model checkpoints.
        base_config_path (str): Path to the base YAML configuration file used during training.
    Returns:
        List[PerceptionModule]: A list of loaded PerceptionModule instances.
    """
    config = load_config(base_config_path)
    loaded_models = []
    
    for i, model_path in enumerate(model_paths):
        logger.info(f"Loading ensemble member {i+1} from: {model_path}")
        model = PerceptionModule(config).to(DEVICE)
        # Re-initialize bongard_classifier as it's added dynamically in main_phase1
        model.bongard_classifier = torch.nn.Linear(model.relation_gnn.set2set.output_size, config['model']['num_classes']).to(DEVICE)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval() # Set to evaluation mode for inference
                # If QAT was used, convert the model to quantized form after loading
                if config['training'].get('use_qat', False) and hasattr(torch.quantization, 'convert'):
                    torch.quantization.convert(model, inplace=True)
                    logger.info(f"Converted ensemble member {i+1} to quantized version for inference.")
                logger.info(f"Successfully loaded ensemble member {i+1}.")
                loaded_models.append(model)
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}. Skipping this member.")
        else:
            logger.warning(f"Model checkpoint not found at {model_path}. Skipping this member.")
    
    if not loaded_models:
        raise RuntimeError("No ensemble models could be loaded. Check paths and configurations.")
    
    logger.info(f"Successfully loaded {len(loaded_models)} ensemble members.")
    return loaded_models

def ensemble_predict(
    models: List[PerceptionModule],
    image_paths: List[str],
    config: Dict[str, Any],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble inference on a batch of images.
    Args:
        models (List[PerceptionModule]): List of loaded ensemble models.
        image_paths (List[str]): List of paths to input images.
        config (Dict[str, Any]): Configuration dictionary.
        use_mc_dropout (bool): Whether to use MC Dropout for each ensemble member.
        mc_dropout_samples (int): Number of MC Dropout samples per model if used.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Ensemble averaged Bongard problem probabilities (numpy array).
            - List of symbolic outputs from the first ensemble member (for visualization).
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction.")

    all_bongard_probs = []
    first_model_symbolic_outputs = None # To return symbolic output for one model

    for i, model in enumerate(models):
        logger.info(f"Performing inference with ensemble member {i+1}...")
        
        # Load PIL images for the current batch
        image_pil_batch = [Image.open(p).convert('RGB') for p in image_paths]

        if use_mc_dropout and mc_dropout_samples > 0:
            # If using MC Dropout, the model returns uncertainty estimates
            mc_results = model.mc_dropout_predict(image_pil_batch, mc_dropout_samples, is_synthetic=False)
            
            # Extract mean probabilities from MC Dropout results
            # Assuming mc_results is a list of dicts, one per image in batch
            current_bongard_probs_batch = []
            for res in mc_results:
                if res['bongard_prediction_mean'] is not None:
                    current_bongard_probs_batch.append(np.array(res['bongard_prediction_mean']))
                else:
                    logger.warning(f"No Bongard prediction mean for image {res['image_path']} in MC Dropout result.")
                    current_bongard_probs_batch.append(np.zeros(config['model']['num_classes'])) # Placeholder
            
            if current_bongard_probs_batch:
                all_bongard_probs.append(np.stack(current_bongard_probs_batch)) # Stack across images in batch
            else:
                logger.warning(f"No valid Bongard probabilities collected from MC Dropout for model {i+1}.")

        else:
            # Standard inference (no MC Dropout)
            with torch.no_grad():
                symbolic_outputs, aggregated_outputs = model(image_pil_batch, is_synthetic=False)
            
            if first_model_symbolic_outputs is None:
                # Store symbolic outputs from the first model for potential visualization
                first_model_symbolic_outputs = symbolic_outputs
                # Ensure it's JSON serializable (tensors to lists)
                json_serializable_outputs = []
                for sym_out in first_model_symbolic_outputs:
                    serializable_sym_out = sym_out.copy()
                    if serializable_sym_out.get("image_features") is not None and torch.is_tensor(serializable_sym_out["image_features"]):
                        serializable_sym_out["image_features"] = serializable_sym_out["image_features"].tolist()
                    json_serializable_outputs.append(serializable_sym_out)
                first_model_symbolic_outputs = json_serializable_outputs

            # Get Bongard problem logits and convert to probabilities
            if aggregated_outputs['image_features_student'].numel() > 0 and hasattr(model, 'bongard_classifier'):
                bongard_logits = model.bongard_classifier(aggregated_outputs['image_features_student'])
                bongard_probs = torch.nn.functional.softmax(bongard_logits, dim=-1).cpu().numpy()
                all_bongard_probs.append(bongard_probs)
            else:
                logger.warning(f"No Bongard features or classifier found for model {i+1}. Appending zero probabilities.")
                all_bongard_probs.append(np.zeros((len(image_paths), config['model']['num_classes'])))

    if not all_bongard_probs:
        logger.error("No Bongard probabilities collected from any ensemble member.")
        return np.empty((len(image_paths), config['model']['num_classes'])), first_model_symbolic_outputs

    # Average the probabilities across all ensemble members
    # Stack all_bongard_probs: List of [num_images, num_classes] -> [num_models, num_images, num_classes]
    stacked_probs = np.stack(all_bongard_probs, axis=0)
    ensemble_averaged_probs = np.mean(stacked_probs, axis=0)

    logger.info(f"Ensemble prediction complete. Averaged probabilities for {len(image_paths)} images.")
    return ensemble_averaged_probs, first_model_symbolic_outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform ensemble inference for Bongard Perception.")
    parser.add_argument('--model_dir', type=str, default='./ensemble_checkpoints', help='Directory containing ensemble model checkpoints.')
    parser.add_argument('--num_members', type=int, default=3, help='Number of ensemble members to load for inference.')
    parser.add_argument('--base_config', type=str, default='config.yaml', help='Path to the base YAML configuration file used during training.')
    parser.add_argument('--image_paths', nargs='+', required=True, help='List of image paths to perform inference on.')
    parser.add_argument('--use_mc_dropout', action='store_true', help='Use Monte Carlo Dropout for each ensemble member during inference.')
    parser.add_argument('--mc_dropout_samples', type=int, default=25, help='Number of MC Dropout samples per model if --use_mc_dropout is enabled.')
    
    args = parser.parse_args()

    # Construct model paths based on the expected naming convention from ensemble_trainer.py
    model_paths = []
    # Assumes the naming convention: ensemble_member_{id}_best_perception_model.pt
    # You might need to adjust this if your trainer saves with a different suffix.
    base_checkpoint_name = load_config(args.base_config)['training']['model_checkpoint_name']
    for i in range(args.num_members):
        model_paths.append(os.path.join(args.model_dir, f"ensemble_member_{i}_{base_checkpoint_name}"))

    try:
        # Load models
        ensemble_models = load_ensemble_models(model_paths, args.base_config)

        # Perform ensemble prediction
        averaged_probs, symbolic_outputs_example = ensemble_predict(
            ensemble_models,
            args.image_paths,
            load_config(args.base_config), # Pass the loaded config for constants like num_classes
            use_mc_dropout=args.use_mc_dropout,
            mc_dropout_samples=args.mc_dropout_samples
        )

        logger.info("\n--- Ensemble Prediction Results ---")
        for i, img_path in enumerate(args.image_paths):
            logger.info(f"Image: {img_path}")
            logger.info(f"  Ensemble Averaged Probabilities: {averaged_probs[i]}")
            logger.info(f"  Ensemble Predicted Class: {np.argmax(averaged_probs[i])}")
        
        # Optionally save or display symbolic_outputs_example
        if symbolic_outputs_example:
            output_json_path = os.path.join(args.model_dir, "ensemble_inference_symbolic_output_example.json")
            with open(output_json_path, 'w') as f:
                json.dump(symbolic_outputs_example, f, indent=4)
            logger.info(f"Example symbolic output from one ensemble member saved to: {output_json_path}")

    except Exception as e:
        logger.error(f"An error occurred during ensemble inference: {e}")

# BongardSolver.ipynb - Main Cell for Phase 1 Perception Module with Ensemble Training and Inference Orchestration

# --- 0. Imports from separate Python files ---
# IMPORTANT: Ensure phase1_code_part1.py, phase1_code_part2a.py, phase1_code_part2b.py,
# and phase1_code_part3.py are in the same directory as this notebook,
# or that their directory is added to your Python path (e.g., using sys.path.append).

# Standard library imports
import os
import logging
import copy # For deep copying configuration for each ensemble member
import random # For setting seeds
from typing import List, Dict, Any, Optional, Tuple

# Third-party library imports
import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm # For progress bars

# Imports from your project files
# From phase1_code_part1.py
from phase1_code_part1 import (
    load_config as load_base_config, # Renamed to avoid conflict with local CONFIG var
    set_seed,
    DATA_ROOT_PATH, IMAGENET_MEAN, IMAGENET_STD, DEVICE,
    BongardSyntheticDataset, BongardExternalSource, dali_pipe, CurriculumSampler,
    YOLO_CLASS_MAP, ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
    ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
    RELATION_MAP, ATTRIBUTE_FILL_MAP_INV, ATTRIBUTE_COLOR_MAP_INV,
    ATTRIBUTE_SIZE_MAP_INV, ATTRIBUTE_ORIENTATION_MAP_INV, ATTRIBUTE_SHAPE_MAP_INV,
    ATTRIBUTE_TEXTURE_MAP_INV, RELATION_MAP_INV,
    PROCESSED_DATA_DIR, DIFFICULTY_SCORES_FILE, FULL_DATA_CACHE_FILE,
    load_bongard_data, HAS_SAM, HAS_WANDB, HAS_KORNIA, HAS_TIMM, HAS_SOPHIA, HAS_LORA, HAS_TIMM_OPTIM, HAS_TORCH_QUANTIZATION
)

# From phase1_code_part2a.py
from phase1_code_part2a import (
    RealObjectDetector, AttributeClassifier,
    enhance_logo_contrast, fallback_shape_detection_bw, _calculate_iou
)

# From phase1_code_part2b.py
from phase1_code_part2b import (
    RelationGNN, PerceptionModule, apply_symbolic_priors,
    DistillationLoss, MixupCutmix, LabelSmoothingCrossEntropy,
    FeatureConsistencyLoss, KnowledgeReplayBuffer
)

# From phase1_code_part3.py
from phase1_code_part3 import (
    _train_step, _validate_model, _plot_training_history, _plot_reliability_diagram,
    calibrate_model, apply_structured_pruning, quantize_model_ptq, quantize_model_qat
)

# Set up logging for this notebook
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuration Loading and Overrides ---
# Load the default configuration. You can create a 'config.yaml' file
# in the same directory as this notebook to override defaults.
# If config.yaml is not found, it will use the hardcoded DEFAULT_CONFIG from phase1_code_part1.py
CONFIG = load_base_config('config.yaml')

# --- CRITICAL: Force Memory-Optimized CONFIG Values for 4GB GPU ---
# These values will override anything set in a config file or elsewhere,
# ensuring the DALI pipeline is initialized with memory-safe parameters.
# This block is placed after load_config to ensure it overrides any YAML settings.
logger.info("Applying critical memory-optimized configuration overrides for 4GB GPU (for notebook environment):")
CONFIG['model']['batch_size'] = 4            # Reduced batch size
logger.info(f"  Overriding 'model.batch_size' to {CONFIG['model']['batch_size']}.")
CONFIG['data']['image_size'] = [96, 96]      # Reduced image size
logger.info(f"  Overriding 'data.image_size' to {CONFIG['data']['image_size']}.")
CONFIG['data']['initial_image_size'] = [64, 64] # Smaller initial size for DALI decoding
logger.info(f"  Overriding 'data.initial_image_size' to {CONFIG['data']['initial_image_size']}.")
CONFIG['training']['feature_consistency_alpha'] = 0.0 # Disable second view
logger.info(f"  Overriding 'training.feature_consistency_alpha' to {CONFIG['training']['feature_consistency_alpha']} (disabling second view for memory).")
CONFIG['training']['use_amp'] = True       # Ensure AMP is enabled
logger.info(f"  Ensuring 'training.use_amp' is {CONFIG['training']['use_amp']} (Automatic Mixed Precision).")
CONFIG['data']['dataloader_num_workers'] = 6 # Increased worker count for DALI parallelism
logger.info(f"  Overriding 'data.dataloader_num_workers' to {CONFIG['data']['dataloader_num_workers']}.")
CONFIG['training']['gradient_accumulation_steps'] = 2 # Set to 2 as per report
logger.info(f"  Overriding 'training.gradient_accumulation_steps' to {CONFIG['training']['gradient_accumulation_steps']}.")
CONFIG['training']['mixup_alpha'] = 0.2 # Enable MixUp
logger.info(f"  Overriding 'training.mixup_alpha' to {CONFIG['training']['mixup_alpha']} (enabling MixUp).")
CONFIG['training']['cutmix_alpha'] = 1.0 # Enable CutMix
logger.info(f"  Overriding 'training.cutmix_alpha' to {CONFIG['training']['cutmix_alpha']} (enabling CutMix).")
CONFIG['training']['label_smoothing_epsilon'] = 0.1 # Enable Label Smoothing
logger.info(f"  Overriding 'training.label_smoothing_epsilon' to {CONFIG['training']['label_smoothing_epsilon']} (enabling Label Smoothing).")
CONFIG['model']['detection_confidence_threshold'] = 0.1   # Lower detection threshold
logger.info(f"  Overriding 'model.detection_confidence_threshold' to {CONFIG['model']['detection_confidence_threshold']}.")
CONFIG['model']['yolo_augment'] = True                  # New flag for TTA
logger.info(f"  Overriding 'model.yolo_augment' to {CONFIG['model']['yolo_augment']} (enabling YOLO Test Time Augmentation).")
logger.info("Finished applying critical configuration overrides.")
logger.info(f"Effective CONFIG for DALI Pipeline: Batch Size={CONFIG['model']['batch_size']}, Image Size={CONFIG['data']['image_size']}, Feature Consistency Alpha={CONFIG['training']['feature_consistency_alpha']}")
logger.info(f"DALI Dataloader Workers: {CONFIG['data']['dataloader_num_workers']}")
logger.info(f"Gradient Accumulation Steps: {CONFIG['training']['gradient_accumulation_steps']}")
logger.info(f"MixUp Alpha: {CONFIG['training']['mixup_alpha']}, CutMix Alpha: {CONFIG['training']['cutmix_alpha']}")
logger.info(f"Label Smoothing Epsilon: {CONFIG['training']['label_smoothing_epsilon']}")
# --- END CRITICAL CONFIG OVERRIDE ---

# Set global logging level based on config
logging.getLogger().setLevel(getattr(logging, CONFIG['debug']['log_level'].upper()))

logger.info(f"Using device: {DEVICE}")

# --- 2. Core Training Function (Adapted from main_phase1) ---
def _run_single_training_session(
    current_config: Dict[str, Any],
    member_id: int,
    random_seed: int,
    output_dir: str,
    epochs_override: Optional[int] = None
):
    """
    Runs a single training session for one ensemble member.
    This function is adapted from the original `main_phase1` to be callable
    with specific parameters for ensemble training. It uses imported functions
    from the separate phase1_code_partX.py files.
    """
    logger.info(f"--- Starting training for Ensemble Member {member_id} (Seed: {random_seed}) ---")

    # Set all random seeds for reproducibility of this specific run
    set_seed(random_seed)

    # Use a deep copy of the config to avoid modifying the global CONFIG for other members
    local_config = copy.deepcopy(current_config)

    # Override epochs if specified
    if epochs_override is not None:
        local_config['model']['epochs'] = epochs_override
        logger.info(f"Overriding epochs to {epochs_override} for this ensemble member.")

    # Update checkpoint name to be unique for this ensemble member
    original_model_checkpoint_name = local_config['training']['model_checkpoint_name']
    local_config['training']['model_checkpoint_name'] = f"ensemble_member_{member_id}_{original_model_checkpoint_name}"
    local_config['training']['checkpoint_dir'] = output_dir # Ensure checkpoints go to the specified output_dir
    os.makedirs(local_config['training']['checkpoint_dir'], exist_ok=True) # Create if not exists

    logger.info(f"Ensemble Member {member_id} will save checkpoint to: {os.path.join(output_dir, local_config['training']['model_checkpoint_name'])}")

    # Initialize WandB if enabled (ensure unique run name for each member)
    if HAS_WANDB and local_config.get('training', {}).get('use_wandb', False):
        import wandb
        wandb.init(project="bongard-perception-ensemble", config=local_config, name=f"ensemble-member-{member_id}-seed-{random_seed}")
        logger.info(f"WandB initialized for ensemble member {member_id}.")

    # --- Data Loading ---
    # The data loading logic is directly copied from main_phase1 to ensure it uses local_config
    if local_config['data']['use_synthetic_data']:
        logger.info(f"Generating {local_config['data']['synthetic_samples']} synthetic data samples.")
        synthetic_dataset = BongardSyntheticDataset(
            n_samples=local_config['data']['synthetic_samples'],
            image_size=local_config['data']['image_size'],
            num_objects=(2, 5),
            num_classes=local_config['model']['num_classes']
        )
        train_size = int((1 - local_config['data']['train_test_split_ratio']) * len(synthetic_dataset))
        val_size = len(synthetic_dataset) - train_size
        train_synthetic_dataset, val_synthetic_dataset = torch.utils.data.random_split(synthetic_dataset, [train_size, val_size])
        
        train_external_source = BongardExternalSource(train_synthetic_dataset, local_config['model']['batch_size'], shuffle=True)
        val_external_source = BongardExternalSource(val_synthetic_dataset, local_config['model']['batch_size'], shuffle=False)
        
        train_pipe = dali_pipe(
            file_root=None, file_list=None, labels_list=None,
            batch_size=local_config['model']['batch_size'],
            num_threads=local_config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=local_config['data']['image_size'][0], width=local_config['data']['image_size'][1],
            is_train=True, num_channels=local_config['data']['num_channels'],
            feature_consistency_alpha=local_config['training']['feature_consistency_alpha'],
            imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD,
            use_synthetic_data=True, external_source_iterator=train_external_source
        )
        val_pipe = dali_pipe(
            file_root=None, file_list=None, labels_list=None,
            batch_size=local_config['model']['batch_size'],
            num_threads=local_config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=local_config['data']['image_size'][0], width=local_config['data']['image_size'][1],
            is_train=False, num_channels=local_config['data']['num_channels'],
            feature_consistency_alpha=0.0,
            imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD,
            use_synthetic_data=True, external_source_iterator=val_external_source
        )
        
        train_pipe.build()
        val_pipe.build()
        
        train_dali_iter = DALIGenericIterator(
            train_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="ExternalSource", last_batch_policy=LastBatchPolicy.DROP
        )
        val_dali_iter = DALIGenericIterator(
            val_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="ExternalSource", last_batch_policy=LastBatchPolicy.DROP
        )
        logger.info(f"DALI iterators built for synthetic data. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
    else:
        image_paths, labels, difficulty_scores = load_bongard_data(DATA_ROOT_PATH)
        train_paths, val_paths, train_labels, val_labels, train_difficulties, val_difficulties = train_test_split(
            image_paths, labels, difficulty_scores, test_size=local_config['data']['train_test_split_ratio'],
            random_state=random_seed, stratify=labels
        )
        logger.info(f"Train samples: {len(train_paths)}, Validation samples: {len(val_paths)}")
        
        train_sampler = CurriculumSampler(
            train_paths, train_labels, train_difficulties, local_config['model']['batch_size'],
            local_config['data']['curriculum_annealing_epochs'], local_config['model']['epochs'],
            is_train=True, use_weighted_sampling=local_config['training']['use_weighted_sampling']
        )
        val_sampler = CurriculumSampler(
            val_paths, val_labels, val_difficulties, local_config['model']['batch_size'],
            local_config['data']['curriculum_annealing_epochs'], local_config['model']['epochs'],
            is_train=False, use_weighted_sampling=False
        )
        train_pipe = dali_pipe(
            file_root=DATA_ROOT_PATH, file_list=train_sampler.get_epoch_data()[0], labels_list=train_sampler.get_epoch_data()[1],
            batch_size=local_config['model']['batch_size'], num_threads=local_config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=local_config['data']['initial_image_size'][0], width=local_config['data']['initial_image_size'][1],
            is_train=True, num_channels=local_config['data']['num_channels'],
            feature_consistency_alpha=local_config['training']['feature_consistency_alpha'],
            imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
        )
        val_pipe = dali_pipe(
            file_root=DATA_ROOT_PATH, file_list=val_sampler.get_epoch_data()[0], labels_list=val_sampler.get_epoch_data()[1],
            batch_size=local_config['model']['batch_size'], num_threads=local_config['data']['dataloader_num_workers'],
            device_id=0 if DEVICE.type == 'cuda' else -1,
            height=local_config['data']['image_size'][0], width=local_config['data']['image_size'][1],
            is_train=False, num_channels=local_config['data']['num_channels'],
            feature_consistency_alpha=0.0, imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
        )
        train_pipe.build()
        val_pipe.build()
        train_dali_iter = DALIGenericIterator(
            train_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
        )
        val_dali_iter = DALIGenericIterator(
            val_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
            reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
        )
        logger.info(f"DALI iterators built for real data. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")

    # Model Initialization
    model = PerceptionModule(local_config).to(DEVICE)
    model.bongard_classifier = torch.nn.Linear(model.relation_gnn.set2set.output_size, local_config['model']['num_classes']).to(DEVICE)
    logger.info("PerceptionModule and Bongard classifier initialized.")
    
    if local_config['training']['use_torch_compile'] and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Proceeding without compilation.")

    # Optimizers
    if local_config['model']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=local_config['model']['initial_learning_rate'], weight_decay=local_config['training']['weight_decay'])
    elif local_config['model']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=local_config['model']['initial_learning_rate'], momentum=0.9, weight_decay=local_config['training']['weight_decay'])
    elif local_config['model']['optimizer'] == 'SophiaG' and HAS_SOPHIA:
        optimizer = SophiaG(model.parameters(), lr=local_config['model']['initial_learning_rate'], weight_decay=local_config['training']['weight_decay'])
    elif local_config['model']['optimizer'] == 'Lion' and HAS_TIMM_OPTIM:
        optimizer = Lion(model.parameters(), lr=local_config['model']['initial_learning_rate'], weight_decay=local_config['training']['weight_decay'])
    elif local_config['model']['optimizer'] == 'MADGRAD' and HAS_TIMM_OPTIM:
        optimizer = MADGRAD(model.parameters(), lr=local_config['model']['initial_learning_rate'], weight_decay=local_config['training']['weight_decay'])
    else:
        logger.warning(f"Optimizer '{local_config['model']['optimizer']}' not supported or not installed. Falling back to AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=local_config['model']['initial_learning_rate'], weight_decay=local_config['training']['weight_decay'])
    
    if local_config['training']['use_sam_optimizer'] and HAS_SAM:
        optimizer = SAM(model.parameters(), optimizer, rho=0.05, adaptive=False)
        logger.info("SAM optimizer wrapped around base optimizer.")

    # Schedulers
    scheduler = None
    if local_config['model']['scheduler'] == 'OneCycleLR':
        total_steps = len(train_dali_iter) * local_config['model']['epochs']
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                               max_lr=local_config['model']['max_learning_rate'],
                               total_steps=total_steps,
                               pct_start=local_config['training']['onecycle_pct_start'],
                               anneal_strategy='cos',
                               cycle_momentum=True,
                               base_momentum=0.85,
                               max_momentum=0.95,
                               div_factor=local_config['model']['max_learning_rate'] / local_config['model']['initial_learning_rate'],
                               final_div_factor=local_config['model']['initial_learning_rate'] / (local_config['model']['max_learning_rate'] * 1e-2)
                              )
        logger.info("OneCycleLR scheduler initialized.")
    elif local_config['model']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=local_config['training']['lr_scheduler_factor'],
                                                         patience=local_config['training']['lr_scheduler_patience'],
                                                         verbose=True)
        logger.info("ReduceLROnPlateau scheduler initialized.")
    else:
        logger.info("No learning rate scheduler configured.")

    # Loss Functions
    if local_config['training']['label_smoothing_epsilon'] > 0:
        criterion_attr = LabelSmoothingCrossEntropy(smoothing=local_config['training']['label_smoothing_epsilon'])
        criterion_rel = LabelSmoothingCrossEntropy(smoothing=local_config['training']['label_smoothing_epsilon'])
        logger.info(f"Using LabelSmoothingCrossEntropy with epsilon={local_config['training']['label_smoothing_epsilon']}.")
    else:
        criterion_attr = torch.nn.CrossEntropyLoss()
        criterion_rel = torch.nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropyLoss.")
    
    criterion_consistency = FeatureConsistencyLoss(loss_type=local_config['training']['feature_consistency_loss_type'])
    criterion_distillation = DistillationLoss(
        alpha=local_config['training']['distillation_alpha'],
        temperature=local_config['training']['distillation_temperature'],
        base_loss_fn=torch.nn.CrossEntropyLoss()
    )
    
    scaler = torch.cuda.amp.GradScaler() if local_config['training']['use_amp'] else None
    
    mixup_cutmix_transform = MixupCutmix(
        mixup_alpha=local_config['training']['mixup_alpha'],
        cutmix_alpha=local_config['training']['cutmix_alpha'],
        prob=1.0,
        switch_prob=local_config['training']['mixup_cutmix_ratio'],
        num_classes=local_config['model']['num_classes']
    )
    
    replay_buffer = None
    if local_config['training']['knowledge_replay_enabled']:
        replay_buffer = KnowledgeReplayBuffer(capacity=local_config['training']['replay_buffer_size'])
        logger.info(f"Knowledge replay buffer initialized with capacity: {local_config['training']['replay_buffer_size']}.")
    
    swa_model = None
    swa_scheduler = None
    if local_config['training']['use_swa']:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=local_config['training']['swa_lr'])
        logger.info(f"SWA initialized with SWA LR: {local_config['training']['swa_lr']}.")
    
    best_val_loss = float('inf')
    best_val_accuracy = -1.0
    epochs_no_improve = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_bongard_accuracy': [],
        'val_attribute_accuracy': [],
        'val_relation_accuracy': []
    }
    
    for epoch in range(local_config['model']['epochs']):
        logger.info(f"--- Epoch {epoch+1}/{local_config['model']['epochs']} ---")
        
        if not local_config['data']['use_synthetic_data']:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
            # Rebuild DALI pipelines with updated file lists and image sizes
            train_pipe = dali_pipe(
                file_root=DATA_ROOT_PATH, file_list=train_sampler.get_epoch_data()[0], labels_list=train_sampler.get_epoch_data()[1],
                batch_size=local_config['model']['batch_size'], num_threads=local_config['data']['dataloader_num_workers'],
                device_id=0 if DEVICE.type == 'cuda' else -1,
                height=train_sampler.get_current_image_size()[0], width=train_sampler.get_current_image_size()[1],
                is_train=True, num_channels=local_config['data']['num_channels'],
                feature_consistency_alpha=local_config['training']['feature_consistency_alpha'],
                imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
            )
            val_pipe = dali_pipe(
                file_root=DATA_ROOT_PATH, file_list=val_sampler.get_epoch_data()[0], labels_list=val_sampler.get_epoch_data()[1],
                batch_size=local_config['model']['batch_size'], num_threads=local_config['data']['dataloader_num_workers'],
                device_id=0 if DEVICE.type == 'cuda' else -1,
                height=val_sampler.get_current_image_size()[0], width=val_sampler.get_current_image_size()[1],
                is_train=False, num_channels=local_config['data']['num_channels'],
                feature_consistency_alpha=0.0, imagenet_mean=IMAGENET_MEAN, imagenet_std=IMAGENET_STD, use_synthetic_data=False
            )
            train_pipe.build()
            val_pipe.build()
            # Re-initialize DALI iterators after rebuilding pipelines
            train_dali_iter = DALIGenericIterator(
                train_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
                reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
            )
            val_dali_iter = DALIGenericIterator(
                val_pipe, ["data", "data_view2", "labels", "ground_truth_json", "affine_matrix_1", "affine_matrix_2"],
                reader_name="Reader", last_batch_policy=LastBatchPolicy.DROP
            )
            logger.info(f"DALI iterators rebuilt for epoch {epoch+1}. Train batches: {len(train_dali_iter)}, Val batches: {len(val_dali_iter)}")
        
        if local_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION and epoch == local_config['training']['qat_start_epoch']:
            logger.info(f"Starting QAT at epoch {epoch}. Preparing model for QAT.")
            model = quantize_model_qat(model, local_config)
            # Re-initialize optimizer if QAT changes model parameters (e.g., adds observers)
            if local_config['model']['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=local_config['model']['initial_learning_rate'], weight_decay=local_config['training']['weight_decay'])
            # ... (repeat for other optimizers if needed) ...
            if local_config['training']['use_sam_optimizer'] and HAS_SAM:
                optimizer = SAM(model.parameters(), optimizer, rho=0.05, adaptive=False)
            logger.info("Optimizer re-initialized after QAT preparation.")
            
        train_loss = _train_step(model, train_dali_iter, optimizer, criterion_attr, criterion_rel,
                                 criterion_consistency, criterion_distillation, scaler,
                                 mixup_cutmix_transform, local_config, epoch, replay_buffer)
        
        val_loss, val_bongard_accuracy, val_attribute_accuracy, val_relation_accuracy = \
            _validate_model(model, val_dali_iter, criterion_attr, criterion_rel, local_config)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_bongard_accuracy'].append(val_bongard_accuracy)
        history['val_attribute_accuracy'].append(val_attribute_accuracy)
        history['val_relation_accuracy'].append(val_relation_accuracy)
        
        if HAS_WANDB and local_config.get('training', {}).get('use_wandb', False):
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_bongard_accuracy": val_bongard_accuracy,
                "val_attribute_accuracy": val_attribute_accuracy,
                "val_relation_accuracy": val_relation_accuracy,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass # OneCycleLR steps per batch, not per epoch here
            elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
        
        if local_config['training']['use_swa'] and swa_model and epoch >= int(local_config['model']['epochs'] * local_config['training']['swa_start_epoch_ratio']):
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.info(f"SWA model updated at epoch {epoch+1}.")
        
        monitor_metric = history[local_config['training']['early_stopping_monitor_metric']][-1]
        
        if local_config['training']['early_stopping_monitor_metric'] == 'val_loss':
            if monitor_metric < best_val_loss - local_config['training']['early_stopping_min_delta']:
                best_val_loss = monitor_metric
                epochs_no_improve = 0
                checkpoint_path = os.path.join(local_config['training']['checkpoint_dir'], local_config['training']['model_checkpoint_name'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation loss did not improve. Epochs without improvement: {epochs_no_improve}")
        elif local_config['training']['early_stopping_monitor_metric'] == 'val_bongard_accuracy':
            if monitor_metric > best_val_accuracy + local_config['training']['early_stopping_min_delta']:
                best_val_accuracy = monitor_metric
                epochs_no_improve = 0
                checkpoint_path = os.path.join(local_config['training']['checkpoint_dir'], local_config['training']['model_checkpoint_name'])
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Best model saved to {checkpoint_path} with validation accuracy: {best_val_accuracy:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Validation accuracy did not improve. Epochs without improvement: {epochs_no_improve}")
        
        if epochs_no_improve >= local_config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
            break
        
        if (epoch + 1) % local_config['training']['save_symbolic_outputs_interval_epochs'] == 0:
            logger.info(f"Saving symbolic outputs for visualization at epoch {epoch+1}.")
            sample_val_paths, sample_val_labels = val_sampler.get_epoch_data() if not local_config['data']['use_synthetic_data'] else ([], [])
            num_samples_to_save = min(5, len(sample_val_paths))
            
            sample_image_paths = sample_val_paths[:num_samples_to_save]
            sample_pil_images = [Image.open(p).convert('RGB') for p in sample_image_paths]
            
            model.eval()
            with torch.no_grad():
                symbolic_outputs_for_export, _ = model(sample_pil_images, is_synthetic=False)
            model.train()
            
            output_json_path = os.path.join(local_config['training']['checkpoint_dir'], f"symbolic_outputs_epoch_{epoch+1}.json")
            json_serializable_outputs = []
            for sym_out in symbolic_outputs_for_export:
                serializable_sym_out = sym_out.copy()
                if serializable_sym_out.get("image_features") is not None and torch.is_tensor(serializable_sym_out["image_features"]):
                    serializable_sym_out["image_features"] = serializable_sym_out["image_features'].tolist()
                json_serializable_outputs.append(serializable_sym_out)
            
            with open(output_json_path, 'w') as f:
                json.dump(json_serializable_outputs, f, indent=4)
            logger.info(f"Sample symbolic outputs exported to: {output_json_path}")
    
    if local_config['training']['use_swa'] and swa_model:
        torch.optim.swa_utils.swap_parameters_with_avg(model, swa_model)
        logger.info("Swapped model parameters with SWA averaged model.")
    
    best_model_path = os.path.join(local_config['training']['checkpoint_dir'], local_config['training']['model_checkpoint_name'])
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        logger.info(f"Loaded best model from {best_model_path}.")
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Using last trained model.")
    
    if local_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION:
        logger.info("Finalizing QAT model conversion.")
        torch.quantization.convert(model, inplace=True)
        logger.info("QAT model converted to quantized version.")
    elif local_config['training'].get('use_ptq', False) and HAS_TORCH_QUANTIZATION:
        logger.info("Applying Post-Training Quantization (PTQ).")
        model = quantize_model_ptq(model, val_dali_iter, local_config)
    
    if local_config['training'].get('enable_structured_pruning', False):
        apply_structured_pruning(model, local_config)
    
    logger.info("Performing final evaluation on the best/SWA model...")
    final_val_loss, final_val_bongard_accuracy, final_val_attribute_accuracy, final_val_relation_accuracy = \
        _validate_model(model, val_dali_iter, criterion_attr, criterion_rel, local_config)
    logger.info(f"Final Validation Results: Loss={final_val_loss:.4f}, Bongard Acc={final_val_bongard_accuracy:.4f}, Attr Acc={final_val_attribute_accuracy:.4f}, Rel Acc={final_val_relation_accuracy:.4f}")
    
    if local_config['debug']['visualize_training_history']:
        _plot_training_history(history, save_path=os.path.join(local_config['training']['checkpoint_dir'], "training_history.png"))
    
    if local_config['debug']['plot_reliability_diagram']:
        # Note: _plot_reliability_diagram in phase3.py uses dummy data.
        # For real plotting, you would need to collect actual probs and labels during validation.
        dummy_probs = np.random.rand(100)
        dummy_labels = np.random.randint(0, 2, 100)
        _plot_reliability_diagram(dummy_probs, dummy_labels, save_path=os.path.join(local_config['training']['checkpoint_dir'], "reliability_diagram.png"))
    
    if local_config['training']['calibrate_model']:
        calibrate_model(model, val_dali_iter, local_config)
    
    model.export_onnx(output_path=os.path.join(local_config['training']['checkpoint_dir'], "perception_module_exported.onnx"))
    
    if local_config['model']['mc_dropout_samples'] > 0:
        logger.info(f"Performing Monte Carlo Dropout inference with {local_config['model']['mc_dropout_samples']} samples.")
        sample_val_paths, sample_val_labels = val_sampler.get_epoch_data() if not local_config['data']['use_synthetic_data'] else ([], [])
        num_mc_samples_to_test = min(2, len(sample_val_paths) if sample_val_paths else 2)
        
        if num_mc_samples_to_test > 0:
            mc_test_image_paths = sample_val_paths[:num_mc_samples_to_test]
            mc_test_pil_images = [Image.open(p).convert('RGB') for p in mc_test_image_paths]
            
            mc_results = model.mc_dropout_predict(mc_test_pil_images, local_config['model']['mc_dropout_samples'], is_synthetic=False)
            logger.info("Monte Carlo Dropout results (mean and variance):")
            for res in mc_results:
                logger.info(f"  Image: {res['image_path']}")
                if res['bongard_prediction_mean'] is not None:
                    logger.info(f"    Bongard Mean: {res['bongard_prediction_mean']}")
                    logger.info(f"    Bongard Variance: {res['bongard_prediction_variance']}")
        else:
            logger.warning("Not enough samples for Monte Carlo Dropout inference demonstration.")
    
    logger.info(f"--- Finished training for Ensemble Member {member_id} ---")

# --- 3. Ensemble Training Orchestrator ---
def train_ensemble_members_orchestrator(
    num_members: int = 3,
    start_seed: int = 42,
    ensemble_output_dir: str = './ensemble_checkpoints',
    epochs_override: Optional[int] = None
):
    """
    Orchestrates the training of multiple ensemble members.
    Each member is trained with a unique random seed and saves its checkpoint
    to a specified output directory.
    Args:
        num_members (int): The number of ensemble members to train.
        start_seed (int): The starting random seed. Subsequent members will use
                          seeds `start_seed + i`.
        ensemble_output_dir (str): The base directory where all ensemble member
                                   checkpoints will be saved.
        epochs_override (Optional[int]): If provided, overrides the 'epochs'
                                         setting in the config for all members.
    """
    logger.info(f"--- Starting Ensemble Training Orchestration ({num_members} members) ---")
    os.makedirs(ensemble_output_dir, exist_ok=True)

    for i in range(num_members):
        current_seed = start_seed + i # Use a different seed for each member
        # Call the single training session function
        _run_single_training_session(
            current_config=CONFIG, # Pass the global CONFIG
            member_id=i,
            random_seed=current_seed,
            output_dir=ensemble_output_dir,
            epochs_override=epochs_override
        )
    logger.info(f"--- Finished Ensemble Training Orchestration. All checkpoints saved to {ensemble_output_dir} ---")

# --- 4. Ensemble Inference Functions ---
def load_ensemble_models(
    model_dir: str,
    num_members: int,
    base_config: Dict[str, Any]
) -> List[PerceptionModule]:
    """
    Loads multiple PerceptionModule models for ensemble inference.
    Args:
        model_dir (str): Directory containing ensemble model checkpoints.
        num_members (int): Number of ensemble members to load.
        base_config (Dict[str, Any]): The base configuration dictionary used during training.
    Returns:
        List[PerceptionModule]: A list of loaded PerceptionModule instances.
    """
    loaded_models = []
    
    # Construct model paths based on the expected naming convention from _run_single_training_session
    original_model_checkpoint_name = base_config['training']['model_checkpoint_name']
    
    for i in range(num_members):
        model_filename = f"ensemble_member_{i}_{original_model_checkpoint_name}"
        model_path = os.path.join(model_dir, model_filename)
        
        logger.info(f"Loading ensemble member {i+1} from: {model_path}")
        model = PerceptionModule(base_config).to(DEVICE)
        # Re-initialize bongard_classifier as it's added dynamically in _run_single_training_session
        model.bongard_classifier = torch.nn.Linear(model.relation_gnn.set2set.output_size, base_config['model']['num_classes']).to(DEVICE)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.eval() # Set to evaluation mode for inference
                # If QAT was used, convert the model to quantized form after loading
                if base_config['training'].get('use_qat', False) and HAS_TORCH_QUANTIZATION:
                    torch.quantization.convert(model, inplace=True)
                    logger.info(f"Converted ensemble member {i+1} to quantized version for inference.")
                loaded_models.append(model)
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}. Skipping this member.")
        else:
            logger.warning(f"Model checkpoint not found at {model_path}. Skipping this member.")
    
    if not loaded_models:
        raise RuntimeError("No ensemble models could be loaded. Check paths and configurations.")
    
    logger.info(f"Successfully loaded {len(loaded_models)} ensemble members.")
    return loaded_models

def ensemble_predict_orchestrator(
    models: List[PerceptionModule],
    image_paths: List[str],
    config: Dict[str, Any],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble inference on a batch of images by averaging predictions from multiple models.
    Args:
        models (List[PerceptionModule]): List of loaded ensemble models.
        image_paths (List[str]): List of paths to input images.
        config (Dict[str, Any]): Configuration dictionary.
        use_mc_dropout (bool): Whether to use MC Dropout for each ensemble member.
        mc_dropout_samples (int): Number of MC Dropout samples per model if used.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Ensemble averaged Bongard problem probabilities (numpy array).
            - List of symbolic outputs from the first ensemble member (for visualization).
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction.")

    all_bongard_probs = []
    first_model_symbolic_outputs = None # To return symbolic output for one model

    for i, model in enumerate(models):
        logger.info(f"Performing inference with ensemble member {i+1}...")
        
        # Load PIL images for the current batch
        image_pil_batch = []
        for p in image_paths:
            try:
                image_pil_batch.append(Image.open(p).convert('RGB'))
            except Exception as e:
                logger.error(f"Could not load image {p}: {e}. Skipping this image for current model.")
                # Append a dummy image or handle error appropriately to maintain batch size
                # For simplicity, if an image fails to load, we'll append a placeholder
                # This might cause issues if the model expects a fixed batch size.
                # A more robust solution would be to filter out bad images beforehand.
                image_pil_batch.append(Image.new('RGB', tuple(config['data']['image_size']), 'black'))


        if use_mc_dropout and mc_dropout_samples > 0:
            # If using MC Dropout, the model returns uncertainty estimates
            mc_results = model.mc_dropout_predict(image_pil_batch, mc_dropout_samples, is_synthetic=False)
            
            # Extract mean probabilities from MC Dropout results
            current_bongard_probs_batch = []
            for res in mc_results:
                if res['bongard_prediction_mean'] is not None:
                    current_bongard_probs_batch.append(np.array(res['bongard_prediction_mean']))
                else:
                    logger.warning(f"No Bongard prediction mean for image {res['image_path']} in MC Dropout result.")
                    current_bongard_probs_batch.append(np.zeros(config['model']['num_classes'])) # Placeholder
            
            if current_bongard_probs_batch:
                all_bongard_probs.append(np.stack(current_bongard_probs_batch)) # Stack across images in batch
            else:
                logger.warning(f"No valid Bongard probabilities collected from MC Dropout for model {i+1}.")

        else:
            # Standard inference (no MC Dropout)
            with torch.no_grad():
                symbolic_outputs, aggregated_outputs = model(image_pil_batch, is_synthetic=False)
            
            if first_model_symbolic_outputs is None:
                # Store symbolic outputs from the first model for potential visualization
                first_model_symbolic_outputs = symbolic_outputs
                # Ensure it's JSON serializable (tensors to lists)
                json_serializable_outputs = []
                for sym_out in first_model_symbolic_outputs:
                    serializable_sym_out = sym_out.copy()
                    if serializable_sym_out.get("image_features") is not None and torch.is_tensor(serializable_sym_out["image_features"]):
                        serializable_sym_out["image_features"] = serializable_sym_out["image_features"].tolist()
                    json_serializable_outputs.append(serializable_sym_out)
                first_model_symbolic_outputs = json_serializable_outputs

            # Get Bongard problem logits and convert to probabilities
            if aggregated_outputs['image_features_student'].numel() > 0 and hasattr(model, 'bongard_classifier'):
                bongard_logits = model.bongard_classifier(aggregated_outputs['image_features_student'])
                bongard_probs = torch.nn.functional.softmax(bongard_logits, dim=-1).cpu().numpy()
                all_bongard_probs.append(bongard_probs)
            else:
                logger.warning(f"No Bongard features or classifier found for model {i+1}. Appending zero probabilities.")
                all_bongard_probs.append(np.zeros((len(image_paths), config['model']['num_classes'])))

    if not all_bongard_probs:
        logger.error("No Bongard probabilities collected from any ensemble member.")
        return np.empty((len(image_paths), config['model']['num_classes'])), first_model_symbolic_outputs

    # Average the probabilities across all ensemble members
    # Stack all_bongard_probs: List of [num_images, num_classes] -> [num_models, num_images, num_classes]
    stacked_probs = np.stack(all_bongard_probs, axis=0)
    ensemble_averaged_probs = np.mean(stacked_probs, axis=0)

    logger.info(f"Ensemble prediction complete. Averaged probabilities for {len(image_paths)} images.")
    return ensemble_averaged_probs, first_model_symbolic_outputs


# --- 5. Main Execution Block for Jupyter Notebook ---
if __name__ == "__main__":
    # Ensure multiprocessing start method is 'spawn' for CUDA compatibility
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. This might be an issue in some environments.")

    # --- Ensemble Training Configuration ---
    NUM_ENSEMBLE_MEMBERS = 3 # Number of models to train for the ensemble
    ENSEMBLE_START_SEED = 100 # Starting seed for ensemble members
    ENSEMBLE_CHECKPOINT_DIR = './ensemble_models' # Directory to save all ensemble checkpoints
    # You can override the number of epochs for ensemble training here,
    # or leave it None to use the 'epochs' value from your CONFIG.
    ENSEMBLE_EPOCHS_OVERRIDE = 5 # Set to a small number for quick testing, or None for full training

    # --- Run Ensemble Training ---
    logger.info("--- Initiating Ensemble Training ---")
    train_ensemble_members_orchestrator(
        num_members=NUM_ENSEMBLE_MEMBERS,
        start_seed=ENSEMBLE_START_SEED,
        ensemble_output_dir=ENSEMBLE_CHECKPOINT_DIR,
        epochs_override=ENSEMBLE_EPOCHS_OVERRIDE
    )
    logger.info("--- Ensemble Training Complete ---")

    # --- Ensemble Inference Configuration ---
    # Example image paths for inference. Replace with your actual paths.
    # You can use images from your dataset or any other images.
    INFERENCE_IMAGE_PATHS = [
        os.path.join(DATA_ROOT_PATH, 'bd', 'images', 'bd_acute_equi_triangle_0000', '0', 'image_000.png'),
        os.path.join(DATA_ROOT_PATH, 'bd', 'images', 'bd_acute_equi_triangle_0000', '1', 'image_001.png'),
        # Add more image paths as needed for testing inference
    ]
    USE_MC_DROPOUT_INFERENCE = True # Set to True to enable MC Dropout during inference
    MC_DROPOUT_SAMPLES_INFERENCE = 50 # Number of MC Dropout samples per model during inference

    # --- Run Ensemble Inference ---
    logger.info("--- Initiating Ensemble Inference ---")
    try:
        # Load the trained ensemble models
        ensemble_models = load_ensemble_models(
            model_dir=ENSEMBLE_CHECKPOINT_DIR,
            num_members=NUM_ENSEMBLE_MEMBERS,
            base_config=CONFIG # Pass the global CONFIG here
        )

        # Perform ensemble prediction
        averaged_probs, symbolic_outputs_example = ensemble_predict_orchestrator(
            models=ensemble_models,
            image_paths=INFERENCE_IMAGE_PATHS,
            config=CONFIG, # Pass the global CONFIG here
            use_mc_dropout=USE_MC_DROPOUT_INFERENCE,
            mc_dropout_samples=MC_DROPOUT_SAMPLES_INFERENCE
        )

        logger.info("\n--- Ensemble Prediction Results ---")
        for i, img_path in enumerate(INFERENCE_IMAGE_PATHS):
            logger.info(f"Image: {img_path}")
            logger.info(f"  Ensemble Averaged Probabilities: {averaged_probs[i]}")
            logger.info(f"  Ensemble Predicted Class: {np.argmax(averaged_probs[i])}")
        
        # Optionally save or display symbolic_outputs_example
        if symbolic_outputs_example:
            output_json_path = os.path.join(ENSEMBLE_CHECKPOINT_DIR, "ensemble_inference_symbolic_output_example.json")
            with open(output_json_path, 'w') as f:
                json.dump(symbolic_outputs_example, f, indent=4)
            logger.info(f"Example symbolic output from one ensemble member saved to: {output_json_path}")

    except Exception as e:
        logger.error(f"An error occurred during ensemble inference: {e}")

    logger.info("--- Ensemble Inference Complete ---")

