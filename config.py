# Folder: bongard_solver/
# File: config.py
import torch
import os
import logging
import yaml
from typing import Dict, Any

# Set up basic logging for config loading
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- ImageNet Normalization Constants (Standard) ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Global Configuration Dictionary ---
# This dictionary will hold all hyperparameters and settings.
# It can be loaded from a YAML file.
CONFIG: Dict[str, Any] = {
    # --- Data Configuration ---
    'data': {
        'image_size': 224,    # Target size for image processing
        'use_synthetic_data': True,  # True to generate synthetic, False to load real
        'data_root_path': './data',  # Base path for datasets (real or for synthetic temp files)
        'dataloader_workers': 4,  # Number of worker processes for DataLoader
        'synthetic_data_config': {
            'num_train_problems': 10000,
            'num_val_problems': 1000,
            'min_objects_per_image': 1,
            'max_objects_per_image': 5,
            'object_size_range': (20, 80),  # Min/Max pixel size of objects
            'padding': 10,  # Padding around objects/scene
            'font_path': None,  # Path to a .ttf font file for text objects (e.g., './fonts/arial.ttf')
            'scene_layout_types': ['random', 'grid', 'polar'],  # Types of layouts for objects
            'min_support_images_per_problem': 2,  # Minimum positive/negative support examples
            'max_support_images_per_problem': 5,  # Maximum total support examples
        },
        'real_data_config': {
            'dataset_name': 'bongard_dataset_v1',  # Name of the real dataset folder
            'dataset_path': './data/real_bongard',  # Full path to the real dataset root
            'train_split': 0.8,  # Train-validation split ratio for real data
        },
        'use_dali': False,  # Flag to enable NVIDIA DALI (set dynamically by main.py based on HAS_DALI)
        'dali_augmentations': {  # DALI-specific augmentation parameters
            'jpeg_p': 0.1,  # Probability of applying JPEG compression distortion
            'jpeg_q': (50, 100),  # Quality range for JPEG compression
            'gaussian_blur_p': 0.1,  # Probability of applying Gaussian blur
            'gaussian_blur_sigma_range': (0.1, 2.0),  # Sigma range for Gaussian blur
            'color_twist_p': 0.1,  # Probability of applying color twist
            'color_twist_brightness_range': (0.8, 1.2),
            'color_twist_contrast_range': (0.8, 1.2),
            'color_twist_saturation_range': (0.8, 1.2),
            'color_twist_hue_range': (-0.1, 0.1)
        }
    },
    # --- Training Configuration ---
    'training': {
        'seed': 42,
        'epochs': 50,
        'batch_size': 16,  # Per-GPU batch size
        'learning_rate': 1e-4,
        'optimizer': 'AdamW',  # Options: 'AdamW', 'ranger', 'lion', 'sam'
        'weight_decay': 1e-5,
        'use_amp': True,  # Use Automatic Mixed Precision (FP16)
        'max_grad_norm': 1.0,  # Gradient clipping
        'scheduler': 'ReduceLROnPlateau',  # Options: 'CosineAnnealingLR', 'OneCycleLR', 'ReduceLROnPlateau'
        'scheduler_config': {
            'CosineAnnealingLR': {'eta_min': 1e-6},
            'OneCycleLR': {'max_lr': 1e-3, 'pct_start': 0.3, 'div_factor': 25, 'final_div_factor': 1e4, 'steps_per_epoch': 1000},  # steps_per_epoch needs to be calculated
            'ReduceLROnPlateau': {'mode': 'min', 'factor': 0.1, 'patience': 5},
        },
        'early_stop_patience': 10,  # Number of epochs with no improvement after which training will be stopped
        # Augmentation Strategy
        'augmentation_config': {
            'use_autoaugment': False,
            'autoaugment_policy': 'imagenet',  # 'imagenet' or 'cifar10'
            'use_random_erasing': False,
            'random_erasing_p': 0.5,
            'use_randaugment': False,
            'randaugment_num_ops': 2,
            'randaugment_magnitude': 9,
            'use_augmix': False,  # AugMix requires custom implementation
        },
        'use_mixup_cutmix': True,  # Apply Mixup or CutMix
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,  # Probability of applying Mixup
        'cutmix_prob': 0.5,  # Probability of applying CutMix
        'label_smoothing': 0.1,  # Label smoothing for classification losses
        # Curriculum Learning / Prioritized Experience Replay (PER)
        'curriculum_learning': True,
        'curriculum_config': {
            'enabled': True,
            'start_image_size': 64,  # Starting image size for curriculum
            'end_image_size': 224,  # Ending image size for curriculum
            'size_increase_epochs': 20,  # Epochs over which image size increases
            'difficulty_sampling': True,  # Use PER for difficulty sampling
            'difficulty_update_frequency_batches': 100,  # How often to update PER priorities
        },
        'per_config': {  # Prioritized Experience Replay (PER) parameters
            'capacity': 10000,  # Capacity of the replay buffer
            'alpha': 0.6,  # Prioritization exponent (0=uniform, 1=full priority)
            'beta_start': 0.4,  # Importance sampling weight annealing start
            'beta_end': 1.0,  # Importance sampling weight annealing end
        },
        # Replay (general replay buffer for PER)
        'replay': {
            'capacity': 10000,
            'alpha_start': 0.6,
            'alpha_end': 1.0,
            'beta_start': 0.4,
            'beta_end': 1.0,
            'anneal_epochs': 100
        },
        # Consistency Losses (for self-supervised regularization)
        'consistency_loss_weight': 0.1,  # Overall weight for consistency losses
        'feature_consistency_weight': 0.5,  # Weight for feature consistency (e.g., MSE between views)
        'symbolic_consistency_weight': 0.5,  # Weight for symbolic consistency (between inferred SGs)
        # Mean Teacher
        'use_mean_teacher': True,
        'mean_teacher_config': {
            'alpha': 0.99,  # EMA decay rate
        },
        # Knowledge Distillation
        'use_knowledge_distillation': False,  # Enable/disable knowledge distillation
        'distillation_config': {
            'temperature': 4.0,  # Temperature for softmax in distillation
            'alpha': 0.5,  # Weight for soft targets vs. hard targets (0=hard, 1=soft)
            'ensemble_dropout_prob': 0.2,  # Dropout for teacher ensemble during inference
            'use_mask_distillation': True,  # Only distill for samples where teachers agree
            'mask_agreement_threshold': 0.8,  # Min agreement ratio for distillation mask
        },
        'proto_loss_weight': 1.0,  # Weight for Prototypical Network loss
        'log_interval_batches': 50,  # Log training metrics every N batches
        'validation_frequency_batches': 200,  # Run validation every N batches (if val_loader is not None)
    },
    # --- Model Configuration ---
    'model': {
        'backbone': 'resnet18',  # Options: 'resnet18', 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224', 'swin_base_patch4_window7_224'
        'pretrained': True,  # Use ImageNet pretrained weights for backbone
        'feature_dim': None,  # This will be dynamically inferred in AttributeModel.__init__
        'attribute_classifier_config': {
            'shape': 4,  # circle, square, triangle, star
            'color': 6,  # red, blue, green, yellow, black, white
            'fill': 4,  # solid, hollow, striped, dotted
            'size': 3,  # small, medium, large
            'orientation': 2,  # upright, inverted (for triangles)
            'texture': 3  # none, striped, dotted
        },
        'relation_gnn_config': {
            'hidden_dim': 256,
            'num_layers': 3,
            'num_relations': 10,  # e.g., left_of, right_of, above, below, is_close_to, intersects, aligned_horizontally, aligned_vertically, inside, outside, none
            'dropout_prob': 0.1,
            'use_edge_features': True,  # Use spatial/geometric edge features
        },
        'bongard_head_config': {
            'hidden_dim': 512,
            'num_classes': 2,  # 0 for negative, 1 for positive Bongard problem
            'dropout_prob': 0.2,
        },
        'attn': {
            'type': 'multihead', # default: multihead | performer | nystrom
            'heads': 8,
            'landmarks': 64
        },
        'use_dropblock': False,  # Use DropBlock regularization
        'dropblock_config': {
            'block_size': 7,
            'drop_prob': 0.1,
        },
        'use_stochastic_depth': False,  # Use Stochastic Depth regularization
        'stochastic_depth_p': 0.1,  # Probability of dropping a layer (for ViT/Swin)
        'use_persistent_homology': True,  # Use topological features from persistent homology
        'ph_pixel_thresh': 0.5,  # Pixel intensity threshold for binarizing masks for PH
        'ph_feature_dim': 64,  # Desired output dimension for topological features
        'simclr_config': {
            'enabled': True,  # Enable SimCLR pretraining
            'pretrain_epochs': 10,
            'temperature': 0.07,
            'projection_dim': 128,  # Dimension of the projection head
            'mlp_hidden_size': 256,  # Hidden size for the MLP projection head
        },
    },
    # --- Object Detector (YOLO) Configuration ---
    'object_detector': {
        'use_yolo': True,  # Enable YOLO object detection
        'yolo_pretrained': 'yolov8n.pt',  # Pretrained YOLO weights (e.g., 'yolov8n.pt', 'yolov8s.pt') or path to custom .pt
        'yolo_data_yaml': './data/yolo_dataset/data.yaml',  # Path to YOLO dataset config YAML
        'fine_tune_yolo': False,  # Whether to fine-tune YOLO before main training
        'fine_tune_epochs': 5,  # Epochs for YOLO fine-tuning
        'img_size': 640,  # Image size for YOLO training/inference
        'batch_size': 8,  # Batch size for YOLO fine-tuning
        'lr': 0.001,  # Learning rate for YOLO fine-tuning
        'yolo_conf_threshold': 0.25,  # Confidence threshold for YOLO detections
        'yolo_iou_threshold': 0.7,  # IoU threshold for NMS
    },
    # --- Segmentation (SAM) Configuration ---
    'segmentation': {
        'use_sam': True,  # Enable SAM for instance segmentation
        'sam_model_type': 'vit_b',  # 'vit_b', 'vit_l', 'vit_h'
        'sam_checkpoint_path': './checkpoints/sam_vit_b_01ec64.pth',  # Path to SAM checkpoint
        'sam_points_per_side': 32,  # Points per side for grid sampling in SAM
        'sam_pred_iou_thresh': 0.88,  # IoU threshold for SAM mask selection
    },
    # --- Ensemble Configuration ---
    'ensemble': {
        'num_members': 5,  # Number of ensemble members (if training multiple models)
        'train_members': False,  # If True, train individual members; if False, load pretrained
        'teacher_model_paths': [],  # Paths to pretrained teacher models for distillation
                                     # e.g., ['./checkpoints/teacher1.pth', './checkpoints/teacher2.pth']
    },
    # --- Pruning Configuration ---
    'pruning': {  # Top-level key for pruning
        'enabled': False,  # Enable model pruning
        'checkpoint': './checkpoints/best_bongard_model.pth',  # Path to a trained model for pruning
        'method': 'ln_structured',  # 'l1_unstructured', 'random_unstructured', 'ln_structured'
        'amount': 0.2,  # Amount of sparsity (e.g., 0.2 for 20% pruned)
        'iterations': 2,  # Iterative pruning and fine-tuning steps
        'fine_tune_epochs_per_iter': 3,  # Epochs to fine-tune after each pruning step
        'use_sensitivity': True,  # Use sensitivity analysis to select layers for pruning
        'pruning_target_layers_ratio': 0.5,  # Ratio of least sensitive layers to prune
    },
    # --- Quantization Configuration ---
    'quantization': {  # Top-level key for quantization
        'qat': False,  # Quantization Aware Training
        'ptq': False,  # Post-Training Quantization
        'calibration_batches': 10,  # Number of batches for PTQ calibration
    },
    # --- Hyperparameter Optimization (HPO) Configuration ---
    'hpo': {
        'enabled': False,  # Enable HPO
        'trials': 50,  # Number of Optuna trials
        'timeout': 3600, # seconds
        'n_jobs': 4,
        'study_path': 'artifacts/hpo_study.pkl'
    },
    # --- Debugging & Logging ---
    'debug': {
        'log_level': 'INFO',    # DEBUG, INFO, WARNING, ERROR, CRITICAL
        'save_model_checkpoints': './checkpoints',  # Directory to save model checkpoints
        'save_grad_cam_images': False,  # Save Grad-CAM visualizations during validation
        'grad_cam_target_layer': 'perception_module.attribute_model.feature_extractor.layer4',  # Example layer for Grad-CAM
    },
    # --- New: Centralized File Paths ---
    'paths': {
        'data_root': './data', # Default value, update as needed
        'scene_graph_dir': './scene_graphs', # Default value, update as needed
        'dsl_dir': './dsl_rules', # Default value, update as needed
        'logs_dir': './logs',
        'cache_dir': './.cache',
        'mlflow_artifact_dir': './mlruns'
        },
    'rule_engine': {
            'n_workers': 8,      # Number of parallel workers for rule scoring
            'max_rules': 1000    # Maximum number of candidate rules to score
        },

}

# --- Attribute and Relation Mappings (for symbolic representations) ---
# These are used by BongardGenerator and SymbolicEngine
ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2, 'star': 3}
ATTRIBUTE_COLOR_MAP = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'black': 4, 'white': 5}
ATTRIBUTE_FILL_MAP = {'solid': 0, 'hollow': 1, 'striped': 2, 'dotted': 3}
ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
ATTRIBUTE_ORIENTATION_MAP = {'upright': 0, 'inverted': 1}
ATTRIBUTE_TEXTURE_MAP = {'none': 0, 'striped': 1, 'dotted': 2}

# Calculate TOTAL_ATTRIBUTE_CLASSES dynamically
TOTAL_ATTRIBUTE_CLASSES = sum([
    len(ATTRIBUTE_SHAPE_MAP),
    len(ATTRIBUTE_COLOR_MAP),
    len(ATTRIBUTE_FILL_MAP),
    len(ATTRIBUTE_SIZE_MAP),
    len(ATTRIBUTE_ORIENTATION_MAP),
    len(ATTRIBUTE_TEXTURE_MAP)
])

RELATION_MAP = {
    'none': 0,  # No relation
    'left_of': 1, 'right_of': 2, 'above': 3, 'below': 4,
    'is_close_to': 5, 'intersects': 6,
    'aligned_horizontally': 7, 'aligned_vertically': 8,
    'inside': 9, 'outside': 10,
}

# --- Conditional Feature Flags (Dynamically set based on imports) ---
# These flags are set to False by default and will be updated by
# try-except blocks in main.py or other modules if the libraries are detected.
HAS_WANDB = False
HAS_TIMM_OPTIM = False
HAS_TORCH_QUANTIZATION = False
HAS_SAM_OPTIMIZER = False
HAS_RANGER = False
HAS_LION = False
HAS_AUTOAUGMENT = False
HAS_DALI = False
HAS_ULTRALYTICS = False
HAS_SAM = False
HAS_YOLO = False  # Redundant with HAS_ULTRALYTICS, but kept for clarity with sam_utils
HAS_TOPOLOGY_LIBS = False

# --- Function to load configuration from YAML ---
def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Loads configuration from a YAML file, merging it with default CONFIG.
    """
    global CONFIG
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Deep merge YAML config into default CONFIG
        def deep_merge(dict1, dict2):
            for k, v in dict2.items():
                if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
                    deep_merge(dict1[k], v)
                else:
                    dict1[k] = v
        deep_merge(CONFIG, yaml_config)
        logger.info(f"Configuration loaded from {config_path}.")
    else:
        logger.warning(f"Config file not found at {config_path}. Using default configuration.")
    return CONFIG

# Call load_config once to initialize CONFIG and feature flags when config.py is imported
# This ensures CONFIG is populated with defaults and then potentially overridden by config.yaml
CONFIG = load_config()
