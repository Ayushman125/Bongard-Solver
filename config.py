# Folder: bongard_solver/
# File: config.py
import torch
import os
import logging
import yaml
import argparse
from typing import Dict, Any

# Set up basic logging for config loading
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- ImageNet Normalization Constants (Standard) ---
# DALI expects mean/std in pixel value range (0-255) when used with fn.crop_mirror_normalize
IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]   # For 0-255 range
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]   # For 0-255 range

 # --- Global Configuration Dictionary ---
 # This dictionary will hold all hyperparameters and settings.
 # It can be loaded from a YAML file.
from genetic_config import GENETIC_CONFIG
CONFIG: Dict[str, Any] = {
    # --- Data Configuration ---
    'data': {
        'image_size': [224, 224],   # Target size for image processing [height, width]
        'num_channels': 3,          # 1 for grayscale, 3 for RGB
        'use_synthetic_data': True,   # True to generate synthetic, False to load real
        'data_root_path': './data',   # Base path for datasets (real or for synthetic temp files)
        'dataloader_workers': 4,    # Number of worker processes for DataLoader
        'synthetic_data_config': {
            'num_train_problems': 10000,
            'num_val_problems': 1000,
            'min_objects_per_image': 1,
            'max_objects_per_image': 5,
            'object_size_range': (20, 80),   # Min/Max pixel size of objects
            'padding': 10,   # Padding around objects/scene
            'font_path': None,   # Path to a .ttf font file for text objects (e.g., './fonts/arial.ttf')
            'scene_layout_types': ['random', 'grid', 'polar'],   # Types of layouts for objects
            'min_support_images_per_problem': 2,   # Minimum positive/negative support examples
            'max_support_images_per_problem': 5,   # Maximum total support examples
            'synthetic_samples': 1000,  # Number of synthetic samples to generate if use_synthetic_data is True
            # New: Domain Randomization parameters for LogoGenerator
            'use_background_textures': True, # Enable random background textures
            'background_texture_path': './data/textures', # Path to directory with background images
            'background_texture_alpha_range': (0.1, 0.5), # Opacity range for textures
            'use_jitter': True, # Enable jitter for drawing operations
            'jitter_strength_range': (0.5, 2.0), # Strength of jitter
            'use_occluders': True, # Introduce random occluding shapes
            'occluder_count_range': (1, 3), # Number of occluders per image
            'occluder_size_range': (10, 50), # Size of occluders
            'occluder_color_range': ((0, 255), (0, 255), (0, 255)), # RGB range for occluders
            'use_blur': True, # Apply optional blur to the final image
            'blur_radius_range': (0.5, 2.0), # Blur radius range
            'confidence_decay_rate': 0.05, # Rate at which confidence decays with added complexity/augmentation
        },
        'real_data_path': './data/real_bongard', # Path to the real Bongard dataset
        'real_data_config': {
            'dataset_name': 'bongard_dataset_v1',   # Name of the real dataset folder
            'dataset_path': './data/real_bongard',   # Full path to the real dataset root
            'train_split': 0.8,   # Train-validation split ratio for real data
        },
        'use_dali': False,   # Flag to enable NVIDIA DALI (set dynamically by main.py based on HAS_DALI)
        'dali_augmentations': {   # DALI-specific augmentation parameters
            'jpeg_p': 0.1,   # Probability of applying JPEG compression distortion
            'jpeg_q': (50, 100),   # Quality range for JPEG compression
            'gaussian_blur_p': 0.1,   # Probability of applying Gaussian blur
            'gaussian_blur_sigma_range': (0.1, 2.0),   # Sigma range for Gaussian blur
            'color_twist_p': 0.1,   # Probability of applying color twist
            'color_twist_brightness_range': (0.8, 1.2),
            'color_twist_contrast_range': (0.8, 1.2),
            'color_twist_saturation_range': (0.8, 1.2),
            'color_twist_hue_range': (-0.1, 0.1)
        },
        'initial_image_size': [112, 112],  # Smaller initial size for DALI decoding
        'progressive_resizing_epochs': 5,  # Epochs for small image size
        'curriculum_learning_enabled': True,
        'curriculum_difficulty_metric': 'std_dev',
        'curriculum_annealing_epochs': 5,
        'cache_data': True,  # Enable data caching
        'curriculum_start_difficulty': 0.0,
        'curriculum_end_difficulty': 1.0,
        'curriculum_difficulty_schedule': 'linear',
        'curriculum_update_interval_epochs': 1,
        'class_imbalance_threshold': 0.2,  # Threshold for warning about class imbalance
    },
    # --- Training Configuration ---
    'training': {
        'seed': 42,
        'epochs': 50,
        'batch_size': 16,   # Per-GPU batch size
        'learning_rate': 1e-4,
        'optimizer': 'AdamW',   # Options: 'AdamW', 'ranger', 'lion', 'sam', 'SophiaG'
        'weight_decay': 1e-5,
        'use_amp': True,   # Use Automatic Mixed Precision (FP16)
        'max_grad_norm': 1.0,   # Gradient clipping
        'scheduler': 'ReduceLROnPlateau',   # Options: 'CosineAnnealingLR', 'OneCycleLR', 'ReduceLROnPlateau'
        'scheduler_config': {
            'CosineAnnealingLR': {'eta_min': 1e-6},
            'OneCycleLR': {'max_lr': 1e-3, 'pct_start': 0.3, 'div_factor': 25, 'final_div_factor': 1e4, 'steps_per_epoch': 1000},   # steps_per_epoch needs to be calculated
            'ReduceLROnPlateau': {'mode': 'min', 'factor': 0.1, 'patience': 5},
        },
        'early_stop_patience': 10,   # Number of epochs with no improvement after which training will be stopped
        'early_stopping_min_delta': 0.0001,
        'early_stopping_monitor_metric': 'val_loss',
        # Augmentation Strategy
        'augmentation_config': {
            'use_autoaugment': False,
            'autoaugment_policy': 'imagenet',   # 'imagenet' or 'cifar10'
            'use_random_erasing': False,
            'random_erasing_p': 0.5,
            'use_randaugment': False,
            'randaugment_num_ops': 2,
            'randaugment_magnitude': 9,
            'use_augmix': False,   # AugMix requires custom implementation
            'use_color_jitter': False,  # For PyTorch transforms
            'color_jitter_brightness': 0.8,
            'color_jitter_contrast': 0.8,
            'color_jitter_saturation': 0.8,
            'color_jitter_hue': 0.2,
            'use_gaussian_blur': False,  # For PyTorch transforms
            'gaussian_blur_sigma': 1.0,
            'use_random_flip': False,  # For PyTorch transforms
            'use_random_resized_crop': False,  # For PyTorch transforms
            'random_resized_crop_area_min': 0.08,
            'random_resized_crop_ratio_min': 0.75,
            'random_resized_crop_ratio_max': 1.33,
            # New: Albumentations specific augmentations
            'use_albumentations': True, # Enable Albumentations for realistic augmentations
            'albumentations_transforms': [
                {'name': 'HorizontalFlip', 'p': 0.5},
                {'name': 'ShiftScaleRotate', 'p': 0.5, 'shift_limit': 0.0625, 'scale_limit': 0.1, 'rotate_limit': 15},
                {'name': 'RandomBrightnessContrast', 'p': 0.5},
                {'name': 'GaussNoise', 'p': 0.2},
                {'name': 'CoarseDropout', 'p': 0.2, 'max_holes': 8, 'max_height': 8, 'max_width': 8},
                {'name': 'Blur', 'p': 0.1, 'blur_limit': 3},
                {'name': 'CLAHE', 'p': 0.1},
                {'name': 'ColorJitter', 'p': 0.5, 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.2},
            ],
        },
        'use_mixup_cutmix': True,   # Apply Mixup or CutMix
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,   # Probability of applying Mixup
        'cutmix_prob': 0.5,   # Probability of applying CutMix
        'label_smoothing_epsilon': 0.1,   # Label smoothing for classification losses
        # Curriculum Learning / Prioritized Experience Replay (PER)
        'curriculum_learning': True,
        'curriculum_config': {
            'enabled': True,
            'start_image_size': 64,   # Starting image size for curriculum
            'end_image_size': 224,   # Ending image size for curriculum
            'size_increase_epochs': 20,   # Epochs over which image size increases
            'difficulty_sampling': True,   # Use PER for difficulty sampling
            'difficulty_update_frequency_batches': 100,   # How often to update PER priorities
            'alpha': 0.6,  # PER alpha
            'beta_start': 0.4,  # PER beta start
            'beta_end': 1.0,  # PER beta end
            'beta_anneal_epochs': 100  # Epochs for beta annealing
        },
        'replay_buffer_size': 10000,  # Capacity of the replay buffer (for PER)
        'replay_frequency_epochs': 2,  # How often to sample from replay buffer
        'replay_batch_size_ratio': 0.1,  # Ratio of batch size to draw from replay buffer
        # Consistency Losses (for self-supervised regularization)
        'consistency_loss_weight': 0.1,   # Overall weight for consistency losses
        'feature_consistency_weight': 0.5,   # Weight for feature consistency (e.g., MSE between views)
        'feature_consistency_loss_type': 'mse',  # 'mse', 'cosine', 'kl_div'
        'symbolic_consistency_weight': 0.5,   # Weight for symbolic consistency (between inferred SGs)
        # Mean Teacher
        'use_mean_teacher': True,
        'mean_teacher_config': {
            'alpha': 0.99,   # EMA decay rate
        },
        # Knowledge Distillation
        'use_knowledge_distillation': False,   # Enable/disable knowledge distillation
        'distillation_config': {
            'temperature': 4.0,   # Temperature for softening logits.
            'alpha': 0.5,   # Weight for soft targets vs. hard targets (0=hard, 1=soft)
            'ensemble_dropout_prob': 0.2,   # Dropout for teacher ensemble during inference
            'use_mask_distillation': True,   # Only distill for samples where teachers agree
            'mask_agreement_threshold': 0.8,   # Min agreement ratio for distillation mask
            'loss_weight': 1.0, # Weight for distillation loss
        },
        'proto_loss_weight': 1.0,   # Weight for Prototypical Network loss
        'log_interval_batches': 50,   # Log training metrics every N batches
        'validation_frequency_batches': 200,   # Run validation every N batches (if val_loader is not None)
        'gradient_accumulation_steps': 1,  # Default to 1
        'calibrate_model': True,  # Enable post-training calibration (e.g., Temperature Scaling)
        'calibration_epochs': 50,  # Epochs for calibration
        'use_sam_optimizer': True,  # Use Sharpness-Aware Minimization
        'use_swa': True,  # Use Stochastic Weight Averaging
        'swa_start_epoch_ratio': 0.75,  # Start SWA after 75% of epochs
        'swa_lr': 0.05,  # Learning rate for SWA
        'use_wandb': True,  # Enable Weights & Biases logging
        'use_torch_compile': False,  # Use torch.compile for performance
        'enable_profiler': False,  # Enable PyTorch profiler
        'profiler_schedule_wait': 1,
        'profiler_schedule_warmup': 2,
        'profiler_schedule_active': 10,
        'profiler_schedule_repeat': 3,
        'use_qat': False,  # Quantization Aware Training
        'qat_start_epoch': 5,  # Start QAT after N epochs
        'use_weighted_sampling': True,  # Use weighted sampling for class imbalance
        'save_symbolic_outputs_interval_epochs': 5,  # Interval to save symbolic outputs
        'onecycle_pct_start': 0.3,  # Percentage of steps to increase learning rate in OneCycleLR
        # New: Domain Adaptation parameters
        'use_domain_adaptation': True, # Enable adversarial domain adaptation
        'grl_alpha': 1.0, # Alpha parameter for Gradient Reversal Layer
        'lambda_style': 0.1, # Weight for the style discriminator loss
        'lr_disc': 1e-4, # Learning rate for the StyleDiscriminator
    },
    # --- Model Configuration ---
    'model': {
        'backbone': 'resnet18',   # Options: 'resnet18', 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224', 'swin_base_patch4_window7_224'
        'pretrained': True,   # Use ImageNet pretrained weights for backbone
        'feature_dim': None,   # This will be dynamically inferred in AttributeModel.__init__
        'attribute_classifier_config': {
            'shape': 4,   # circle, square, triangle, star
            'color': 6,   # red, blue, green, yellow, black, white
            'fill': 4,   # solid, hollow, striped, dotted
            'size': 3,   # small, medium, large
            'orientation': 2,   # upright, inverted (for triangles)
            'texture': 3    # none, striped, dotted
        },
        'relation_gnn_config': {
            'hidden_dim': 256,
            'num_layers': 3,
            'num_relations': 10,   # e.g., left_of, right_of, above, below, is_close_to, intersects, aligned_horizontally, aligned_vertically, inside, outside, none
            'dropout_prob': 0.1,
            'use_edge_features': True,   # Use spatial/geometric edge features
        },
        'bongard_head_config': {
            'hidden_dim': 512,
            'num_classes': 2,   # 0 for negative, 1 for positive Bongard problem
            'dropout_prob': 0.2,
        },
        'attn': {
            'type': 'multihead',   # default: multihead | performer | nystrom
            'heads': 8,
            'landmarks': 64
        },
        'use_dropblock': False,   # Use DropBlock regularization
        'dropblock_config': {
            'block_size': 7,
            'drop_prob': 0.1,
        },
        'use_stochastic_depth': False,   # Use Stochastic Depth regularization
        'stochastic_depth_p': 0.1,   # Probability of dropping a layer (for ViT/Swin)
        'use_persistent_homology': True,   # Use topological features from persistent homology
        'ph_pixel_thresh': 0.5,   # Pixel intensity threshold for binarizing masks for PH
        'ph_feature_dim': 64,   # Desired output dimension for topological features
        'simclr_config': {
            'enabled': True,   # Enable SimCLR pretraining
            'pretrain_epochs': 10,
            'temperature': 0.07,
            'projection_dim': 128,   # Dimension of the projection head
            'mlp_hidden_size': 256,   # Hidden size for the MLP projection head
            'pretrain_on_real': True, # New: Flag to enable MoCo pretraining on real data
        },
        'object_detector_model_path': 'yolov8n.pt',  # Pretrained YOLO weights or path to custom .pt
        'yolo_data_yaml': './data/yolo_dataset/data.yaml',   # Path to YOLO dataset config YAML
        'fine_tune_yolo': False,   # Whether to fine-tune YOLO before main training
        'fine_tune_epochs': 5,   # Epochs for YOLO fine-tuning
        'img_size': 640,   # Image size for YOLO training/inference
        'batch_size': 8,   # Batch size for YOLO fine-tuning (for YOLO's internal training)
        'lr': 0.001,   # Learning rate for YOLO fine-tuning
        'yolo_conf_threshold': 0.25,   # Confidence threshold for YOLO detections
        'yolo_iou_threshold': 0.7,   # IoU threshold for NMS
        'yolo_augment': True,  # Enable YOLO Test Time Augmentation
        'detection_confidence_threshold': 0.1,  # Confidence threshold for object detection
        'attribute_backbone': 'mobilenet_v2',  # Option for AttributeClassifier backbone
        'gnn_depth': 2,  # Initial fixed depth of the GNN
        'gnn_max_depth': 4,  # Maximum GNN depth for adaptive/linear schedules
        'gnn_min_depth': 1,  # Minimum GNN depth for adaptive/linear schedules
        'gnn_depth_schedule': 'fixed',  # 'fixed', 'linear', 'adaptive'
        'use_lora': False,  # Use LoRA adapters for attribute classifier backbone
        'initial_dropout_rate': 0.5,  # Initial dropout rate for MC Dropout
        'final_dropout_rate': 0.1,  # Final dropout rate for MC Dropout
        'mc_dropout_samples': 25,  # Number of MC Dropout samples
    },
    # --- Segmentation (SAM) Configuration ---
    'segmentation': {
        'use_sam': True,   # Enable SAM for instance segmentation
        'sam_model_type': 'vit_b',   # 'vit_b', 'vit_l', 'vit_h'
        'sam_checkpoint_path': './checkpoints/sam_vit_b_01ec64.pth',   # Path to SAM checkpoint
        'sam_points_per_side': 32,   # Points per side for grid sampling in SAM
        'sam_pred_iou_thresh': 0.88,   # IoU threshold for SAM mask selection
    },
    # --- Ensemble Configuration ---
    'ensemble': {
        'num_members': 5,   # Number of ensemble members (if training multiple models)
        'train_members': False,   # If True, train individual members; if False, load pretrained
        'teacher_model_paths': [],   # Paths to pretrained teacher models for distillation
                                     # e.g., ['./checkpoints/teacher1.pth', './checkpoints/teacher2.pth']
    },
    # --- Pruning Configuration ---
    'pruning': {   # Top-level key for pruning
        'enabled': False,   # Enable model pruning
        'checkpoint': './checkpoints/best_bongard_model.pth',   # Path to a trained model for pruning
        'method': 'ln_structured',   # 'l1_unstructured', 'random_unstructured', 'ln_structured'
        'amount': 0.2,   # Amount of sparsity (e.g., 0.2 for 20% pruned)
        'iterations': 2,   # Iterative pruning and fine-tuning steps
        'fine_tune_epochs_per_iter': 3,   # Epochs to fine-tune after each pruning step
        'use_sensitivity': True,   # Use sensitivity analysis to select layers for pruning
        'pruning_target_layers_ratio': 0.5,   # Ratio of least sensitive layers to prune
    },
    # --- Genetic Algorithm Configuration ---
    'genetic': GENETIC_CONFIG,
    # --- Quantization Configuration ---
    'quantization': {   # Top-level key for quantization
        'qat': False,   # Quantization Aware Training
        'ptq': False,   # Post-Training Quantization
        'calibration_batches': 10,   # Number of batches for PTQ calibration
    },
    # --- Hyperparameter Optimization (HPO) Configuration ---
    'hpo': {
        'enabled': False,   # Enable HPO
        'trials': 50,   # Number of Optuna trials
        'timeout': 3600,   # seconds
        'n_jobs': 4,
        'study_path': 'artifacts/hpo_study.pkl'
    },
    # --- Debugging & Logging ---
    'debug': {
        'log_level': 'INFO',   # DEBUG, INFO, WARNING, ERROR, CRITICAL
        'save_model_checkpoints': './checkpoints',   # Directory to save model checkpoints
        'save_grad_cam_images': False,   # Save Grad-CAM visualizations during validation
        'grad_cam_target_layer': 'perception_module.attribute_model.feature_extractor.layer4',   # Example layer for Grad-CAM
        'visualize_data': True,  # Visualize data samples
        'visualize_training_history': True,  # Plot training history
        'visualize_perception_output': True,  # Visualize perception module outputs
        'plot_reliability_diagram': True,  # Plot reliability diagrams for calibration
        'visualize_gnn_weights': False,  # Toggle for GNN edge weight visualization (conceptual)
        'max_fallback_cnt': 5,  # Max number of contours for fallback detection
        'apply_symbolic_priors': True,  # Toggle for applying symbolic priors in GNN
        'min_contour_area_sam_fallback': 50,  # Minimum contour area for SAM fallback detections
        # New: Validation thresholds
        'validation_accuracy_threshold': 0.8, # Threshold for logging validation failures
        'validation_f1_threshold': 0.7, # Threshold for rule F1 score in validation
    },
    # --- New: Centralized File Paths ---
    'paths': {
        'data_root': './data',   # Default value, update as needed
        'scene_graph_dir': './scene_graphs',   # Default value, update as needed
        'dsl_dir': './dsl_rules',   # Default value, update as needed
        'logs_dir': './logs',
        'cache_dir': './.cache',
        'mlflow_artifact_dir': './mlruns'
    },

    'rule_engine': {
        'n_workers': 8,   # Number of parallel workers for rule scoring
        'max_rules': 1000   # Maximum number of candidate rules to score
    },

    # --- Generator Configuration ---
    'generator': {
        'cp_quota': 0.5,      # Fraction of scenes generated with CP-SAT
        'ga_quota': 0.3,      # Fraction of scenes generated with genetic algorithms
        'pt_quota': 0.2,      # Fraction of scenes generated with prototype templates
        'use_gan': False,     # Enable GAN-based stylization
        'use_gnn': False,     # Enable GNN-based scene filtering
        'gnn_ckpt': 'checkpoints/scene_gnn.pth',  # Path to GNN model checkpoint
        'gnn_thresh': 0.5,    # Quality threshold for GNN filtering (0-1)
        'gnn_radius': 0.3,    # Connectivity radius for scene graphs (fraction of canvas)
        'gnn_hidden': 64,     # Hidden dimensions for GNN
        'gnn_layers': 2,      # Number of GNN layers
        'gnn_dropout': 0.1,   # Dropout rate for GNN
        'gnn_attention': False, # Use Graph Attention Networks instead of GCN
        'bg_texture': 'none', # Background texture: 'none', 'noise', 'checker'
        'canvas_size': 256,   # Canvas size for generated scenes
        'enable_rotation': True,  # Enable random rotation of objects
        'jitter_px': 2,       # Maximum jitter in pixels
        'stroke_min': 2,      # Minimum stroke width
        'shapes_dir': 'src/bongard_generator/shapes',  # Directory for prototype shapes
        'coverage_target': 100  # Target coverage per rule
    },

    # --- Phase 1 Validation/Training Script Configs ---
    'phase1': {
        'best_model_path': 'checkpoints/bongard_perception_best.pth',
        'last_model_path': 'checkpoints/bongard_perception_last.pth',
        'checkpoint_dir': 'checkpoints',
        'synth_holdout_count': 200,
        'img_size': 128,
        'textures_dir': 'data/textures',
        'real_holdout_root': 'data/real_bongard/VAL',
        'validate_bins': 10
    },
}

# --- Expose CONFIG_OBJECT for compatibility with dataclass config imports ---
CONFIG_OBJECT = CONFIG

# --- Attribute and Relation Mappings (for symbolic representations) ---
# These are used by BongardGenerator and SymbolicEngine
ATTRIBUTE_SHAPE_MAP = {
    'circle': 0, 'square': 1, 'triangle': 2, 'star': 3, 'hexagon': 4, 'pentagon': 5,
    'polygon': 6, 'line': 7, 'arc': 8, 'cross': 9, 'diamond': 10, 'oval': 11,
    'rectangle': 12, 'trapezoid': 13, 'arrow': 14, 'spiral': 15, 'heart': 16,
    'cloud': 17, 'lightning': 18, 'abstract': 19, 'text_character': 20
}
ATTRIBUTE_SHAPE_MAP_INV = {v: k for k, v in ATTRIBUTE_SHAPE_MAP.items()}

ATTRIBUTE_COLOR_MAP = {
    'black': 0, 'white': 1, 'red': 2, 'blue': 3, 'green': 4, 'yellow': 5,
    'orange': 6, 'purple': 7, 'pink': 8, 'brown': 9, 'gray': 10, 'gold': 11,
    'silver': 12, 'cyan': 13, 'magenta': 14, 'lime': 15, 'teal': 16,
    'indigo': 17, 'violet': 18, 'maroon': 19, 'navy': 20, 'olive': 21,
    'coral': 22, 'beige': 23, 'multicolor': 24
}
ATTRIBUTE_COLOR_MAP_INV = {v: k for k, v in ATTRIBUTE_COLOR_MAP.items()}

ATTRIBUTE_FILL_MAP = {
    'outline': 0, 'solid': 1, 'gradient': 2, 'patterned': 3, 'empty_transparent': 4
}
ATTRIBUTE_FILL_MAP_INV = {v: k for k, v in ATTRIBUTE_FILL_MAP.items()}

ATTRIBUTE_SIZE_MAP = {
    'tiny': 0, 'small': 1, 'medium': 2, 'large': 3, 'huge': 4, 'full_frame': 5
}
ATTRIBUTE_SIZE_MAP_INV = {v: k for k, v in ATTRIBUTE_SIZE_MAP.items()}

ATTRIBUTE_ORIENTATION_MAP = {
    'upright': 0, 'rotated_45': 1, 'rotated_90': 2, 'rotated_135': 3,
    'rotated_180': 4, 'rotated_225': 5, 'rotated_270': 6, 'rotated_315': 7,
    'inverted': 8, 'horizontal': 9, 'vertical': 10, 'diagonal_up': 11,
    'diagonal_down': 12
}
ATTRIBUTE_ORIENTATION_MAP_INV = {v: k for v, k in ATTRIBUTE_ORIENTATION_MAP.items()} # Fix: should be v:k

ATTRIBUTE_TEXTURE_MAP = {
    'flat': 0, 'striped_horizontal': 1, 'striped_vertical': 2, 'striped_diagonal': 3,
    'dotted': 4, 'checkered': 5, 'gradient_linear': 6, 'gradient_radial': 7,
    'noise': 8, 'grid': 9, 'wavy': 10, 'rough': 11, 'smooth': 12,
    'pixelated': 13, 'blurred': 14
}
ATTRIBUTE_TEXTURE_MAP_INV = {v: k for k, v in ATTRIBUTE_TEXTURE_MAP.items()}

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
    'unrelated': 0, 'above': 1, 'below': 2, 'left_of': 3, 'right_of': 4,
    'inside': 5, 'contains': 6, 'overlaps': 7, 'touches': 8, 'adjacent': 9,
    'aligned_horizontally': 10, 'aligned_vertically': 11, 'symmetrical_to': 12,
    'connected_to': 13, 'intersects_with': 14, 'parallel_to': 15,
    'perpendicular_to': 16, 'part_of': 17, 'surrounds': 18,
    'same_type_as': 19, 'different_type_from': 20
}
RELATION_MAP_INV = {v: k for k, v in RELATION_MAP.items()}

# YOLOv8 Class Mapping (Example for common objects, extend as needed)
# This maps the integer class IDs returned by YOLO to more descriptive names.
# You would need to align this with the specific YOLO model's training classes.
YOLO_CLASS_MAP = {
    0: "circle", 1: "square", 2: "triangle", 3: "line", 4: "arc",
    5: "star",   6: "polygon",  7: "diamond",  8: "oval", 9: "rectangle",
    10: "cross", 11: "arrow", 12: "heart", 13: "spiral", 14: "cloud",
    15: "lightning", 16: "text_character", 17: "ellipse", 18: "trapezoid",
    19: "pentagon", 20: "hexagon", 21: "octagon", 22: "semicircle",
    # Add more classes as per your YOLO fine-tuning
}

# --- Conditional Feature Flags (Dynamically set based on imports) ---
# These flags are set to False by default and will be updated by
# try-except blocks in main.py or other modules if the libraries are detected.
HAS_WANDB = False
HAS_TIMM_OPTIM = False
HAS_TORCH_QUANTIZATION = False
HAS_SAM = False  # For SAM optimizer
HAS_RANGER = False
HAS_LION = False
HAS_AUTOAUGMENT = False  # For torchvision.transforms.AutoAugment
HAS_DALI = False
HAS_ULTRALYTICS = False  # For YOLO
HAS_SAM_SEG = False  # For Segment Anything Model (segmentation)
HAS_TOPOLOGY_LIBS = False  # For persistent homology
HAS_KORNIA = False  # For Kornia augmentations
HAS_SOPHIA = False  # For SophiaG optimizer
HAS_ELASTIC_TRANSFORM = False  # For torchvision.transforms.ElasticTransform
HAS_ALBUMENTATIONS = False # New: For Albumentations library

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

# Argument parsing and initial config loading

parser = argparse.ArgumentParser(description="Bongard Problem Phase 1: Advanced Perception Module")
parser.add_argument('--config', type=str, default=None, help='Path to a YAML configuration file.')
args, unknown = parser.parse_known_args()
# Fix: Use default config path if args.config is None
config_path = args.config if args.config is not None else 'config.yaml'
CONFIG = load_config(config_path)

# --- CRITICAL: Force Memory-Optimized CONFIG Values for 4GB GPU ---
# These values will override anything set in a config file or elsewhere,
# ensuring the DALI pipeline is initialized with memory-safe parameters.
# This block is placed after load_config to ensure it overrides any YAML settings.
logger.info("Applying critical memory-optimized configuration overrides for 4GB GPU:")
CONFIG['model']['batch_size'] = 4          # Reduced batch size
logger.info(f"  Overriding 'model.batch_size' to {CONFIG['model']['batch_size']}.")
CONFIG['data']['image_size'] = [96, 96]       # Reduced image size
logger.info(f"  Overriding 'data.image_size' to {CONFIG['data']['image_size']}.")
CONFIG['data']['initial_image_size'] = [64, 64]  # Smaller initial size for DALI decoding
logger.info(f"  Overriding 'data.initial_image_size' to {CONFIG['data']['initial_image_size']}.")
CONFIG['training']['feature_consistency_weight'] = 0.0  # Disable second view (was alpha)
logger.info(f"  Overriding 'training.feature_consistency_weight' to {CONFIG['training']['feature_consistency_weight']} (disabling second view for memory).")
CONFIG['training']['use_amp'] = True        # Ensure AMP is enabled
logger.info(f"  Ensuring 'training.use_amp' is {CONFIG['training']['use_amp']} (Automatic Mixed Precision).")
CONFIG['data']['dataloader_workers'] = 6  # Increased worker count for DALI parallelism
logger.info(f"  Overriding 'data.dataloader_workers' to {CONFIG['data']['dataloader_workers']}.")
CONFIG['training']['gradient_accumulation_steps'] = 2  # Set to 2 as per report
logger.info(f"  Overriding 'training.gradient_accumulation_steps' to {CONFIG['training']['gradient_accumulation_steps']}.")
CONFIG['training']['mixup_alpha'] = 0.2  # Enable MixUp
logger.info(f"  Overriding 'training.mixup_alpha' to {CONFIG['training']['mixup_alpha']} (enabling MixUp).")
CONFIG['training']['cutmix_alpha'] = 1.0  # Enable CutMix
logger.info(f"  Overriding 'training.cutmix_alpha' to {CONFIG['training']['cutmix_alpha']} (enabling CutMix).")
CONFIG['training']['label_smoothing_epsilon'] = 0.1  # Enable Label Smoothing
logger.info(f"  Overriding 'training.label_smoothing_epsilon' to {CONFIG['training']['label_smoothing_epsilon']} (enabling Label Smoothing).")
CONFIG['model']['detection_confidence_threshold'] = 0.1    # Lower detection threshold
logger.info(f"  Overriding 'model.detection_confidence_threshold' to {CONFIG['model']['detection_confidence_threshold']}.")
CONFIG['model']['yolo_augment'] = True              # New flag for TTA
logger.info(f"  Overriding 'model.yolo_augment' to {CONFIG['model']['yolo_augment']} (enabling YOLO Test Time Augmentation).")
logger.info("Finished applying critical configuration overrides.")
logger.info(f"Effective CONFIG for DALI Pipeline: Batch Size={CONFIG['model']['batch_size']}, Image Size={CONFIG['data']['image_size']}, Feature Consistency Weight={CONFIG['training']['feature_consistency_weight']}")
logger.info(f"DALI Dataloader Workers: {CONFIG['data']['dataloader_workers']}")
logger.info(f"Gradient Accumulation Steps: {CONFIG['training']['gradient_accumulation_steps']}")
logger.info(f"MixUp Alpha: {CONFIG['training']['mixup_alpha']}, CutMix Alpha: {CONFIG['training']['cutmix_alpha']}")
logger.info(f"Label Smoothing Epsilon: {CONFIG['training']['label_smoothing_epsilon']}")
# --- END CRITICAL CONFIG OVERRIDE ---

# Set global logging level based on config
logging.getLogger().setLevel(getattr(logging, CONFIG['debug']['log_level'].upper()))

# Unified DATA_ROOT_PATH and other globals from CONFIG
DATA_ROOT_PATH = CONFIG['data']['data_root_path']  # Unified source of truth
FINAL_IMAGE_SIZE = tuple(CONFIG['data']['image_size'])
NUM_CHANNELS = CONFIG['data']['num_channels']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- UPDATED: Cache and Difficulty Score File Paths ---
# Store these in a separate 'processed_data' directory for cleaner project structure.
PROCESSED_DATA_DIR = 'processed_data'  # New directory for processed data/cache
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Ensure this directory exists
DIFFICULTY_SCORES_FILE = os.path.join(PROCESSED_DATA_DIR, 'difficulty_scores.npy')
FULL_DATA_CACHE_FILE = os.path.join(PROCESSED_DATA_DIR, 'bongard_data_full_cache.npz')
# --- END UPDATED PATHS ---

# Dummy values for dataset download (if not using synthetic data and dataset not found)
# These would typically be defined in a separate constants file or passed as arguments.
# For now, keeping them here as placeholders if the download logic is still present.
GOOGLE_DRIVE_FILE_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID_HERE'  # Replace with actual ID
OUTPUT_ZIP_FILENAME = 'bongard_dataset.zip'
DATA_ROOT_FOLDER_NAME = 'ShapeBongard_V2'  # Expected folder name after unzipping

# Check for dataset existence and attempt download/unzip if not using synthetic data
if not os.path.exists(os.path.join(DATA_ROOT_PATH, DATA_ROOT_FOLDER_NAME)) and not CONFIG['data']['use_synthetic_data']:
    print(f"Dataset folder '{DATA_ROOT_FOLDER_NAME}' not found at {DATA_ROOT_PATH}. Attempting download and unzipping...")
    try:
        # This block is for gdown, but since it's not available, it will fall back to requests.
        # Keeping it as is from original code.
        raise ImportError("gdown not available in this environment. Falling back to requests.")
    except ImportError:
        print("gdown failed or not available. Attempting download with requests...")
        try:
            import requests
            import zipfile
            from tqdm.auto import tqdm
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
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 Kibibytes
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(OUTPUT_ZIP_FILENAME, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
            print(f"Downloaded {OUTPUT_ZIP_FILENAME} successfully with requests.")
        except Exception as e_requests:
            print(f"Requests download also failed ({e_requests}).")
            print("Please download the dataset manually from: "
                  f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID} "
                  f"and place '{OUTPUT_ZIP_FILENAME}' in the same directory as your project root.")
            print("Then, manually unzip it to create the folder structure starting with "
                  f"'{DATA_ROOT_FOLDER_NAME}' inside '{DATA_ROOT_PATH}'.")
            raise RuntimeError("Failed to download dataset automatically.")
    print(f"Unzipping {OUTPUT_ZIP_FILENAME} to {DATA_ROOT_PATH}...")
    try:
        with zipfile.ZipFile(OUTPUT_ZIP_FILENAME, 'r') as zip_ref:
            zip_ref.extractall(DATA_ROOT_PATH)  # Extract to the data_root_path
        print(f"Dataset unzipped. Please confirm '{DATA_ROOT_FOLDER_NAME}' folder exists in '{DATA_ROOT_PATH}'.")
        # Clean up the zip file after extraction
        os.remove(OUTPUT_ZIP_FILENAME)
        print(f"Removed {OUTPUT_ZIP_FILENAME}.")
    except Exception as e_unzip:
        print(f"Error unzipping {OUTPUT_ZIP_FILENAME}: {e_unzip}")
        print("Please ensure the zip file is not corrupted and try unzipping manually.")
        raise RuntimeError("Failed to unzip dataset.")
else:
    if not CONFIG['data']['use_synthetic_data']:
        print(f"Dataset folder '{DATA_ROOT_FOLDER_NAME}' already exists at {DATA_ROOT_PATH}. Skipping download and unzip.")
    else:
        print("Using synthetic data. Skipping real dataset download.")

