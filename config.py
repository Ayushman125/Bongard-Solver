# Folder: bongard_solver/

import torch
import os
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# --- Global Constants ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Conditional Imports (for feature toggles) ---
HAS_TIMM_OPTIM = False
try:
    import timm.optim # Check if timm optimizers are available
    HAS_TIMM_OPTIM = True
    logger.info("timm.optim found and enabled.")
except ImportError:
    logger.warning("timm.optim not found. Some optimizers might not be available.")

HAS_TORCH_QUANTIZATION = False
try:
    import torch.quantization as tq # Check if torch.quantization is available
    HAS_TORCH_QUANTIZATION = True
    logger.info("torch.quantization found and enabled.")
except ImportError:
    logger.warning("torch.quantization not found. QAT/PTQ will be disabled.")

HAS_WANDB = False
try:
    import wandb # Check if wandb is available
    HAS_WANDB = True
    logger.info("Weights & Biases found and enabled.")
except ImportError:
    logger.warning("Weights & Biases not found. Experiment tracking will be limited.")

# --- Attribute Mappings (for symbolic representation) ---
# These map string labels to integer IDs for classification heads
ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2, 'star': 3}
ATTRIBUTE_COLOR_MAP = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'black': 4, 'white': 5}
ATTRIBUTE_FILL_MAP = {'solid': 0, 'hollow': 1, 'striped': 2, 'dotted': 3}
ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
ATTRIBUTE_ORIENTATION_MAP = {'upright': 0, 'inverted': 1} # For triangles
ATTRIBUTE_TEXTURE_MAP = {'none': 0, 'striped': 1, 'dotted': 2} # For more complex textures

# Relation mapping (e.g., spatial relations)
RELATION_MAP = {'none': 0, 'left_of': 1, 'right_of': 2, 'above': 3, 'below': 4, 'is_close_to': 5, 'aligned_horizontally': 6, 'aligned_vertically': 7, 'intersects': 8, 'contains': 9} # Added intersects, contains


# --- Global Configuration Dictionary ---
CONFIG = {
    # --- Data Configuration ---
    'data': {
        'image_size': 224, # Default image size for models (final DALI resize target)
        'dataloader_workers': 4, # Number of PyTorch DataLoader workers (CPU)
        'num_processing_workers': 4, # Number of multiprocessing.Pool workers for CPU-bound tasks
        'use_synthetic_data': True, # True for synthetic, False for real
        'data_root_path': './data', # Base path for real datasets
        'synthetic_data_config': {
            'num_samples_per_rule': 1000, # Number of synthetic samples per Bongard rule
            'min_objects_per_image': 1,
            'max_objects_per_image': 3,
            'object_size_range': (20, 80), # Min and max size of objects in pixels
            'padding': 10, # Padding around objects
            'font_path': None, # Path to a .ttf font if text is used
            'scene_layout_types': ['random', 'grid', 'polar'], # Types of structured layouts
            'min_support_images_per_problem': 2, # Minimum positive/negative support images
            'max_support_images_per_problem': 5, # Maximum total support images per problem
        },
        'real_data_config': {
            'dataset_name': 'bongard_dataset_v1', # Name of the real dataset folder
            'train_split_ratio': 0.8,
        },
        # DALI Augmentation specific parameters
        'dali_augmentations': {
            'elastic_p': 0.0, # Probability of applying elastic transform (set to 0 to disable)
            'elastic': { # Kept for reference but not used if elastic_p is 0
                'alpha': 5.0, # Scaling factor for displacements
                'sigma': 10.0 # Gaussian smoothing kernel size
            },
            'jpeg_p': 0.5, # Probability of applying JPEG compression distortion
            'jpeg_q': [50, 90], # Quality range for JPEG compression [min_q, max_q]
            'gaussian_blur_p': 0.3, # Probability of applying Gaussian blur
            'gaussian_blur_sigma_range': [0.1, 2.0], # Sigma range for Gaussian blur
            'color_twist_p': 0.3, # Probability of applying color twist
            'color_twist_brightness_range': [0.8, 1.2], # Brightness factor range
            'color_twist_contrast_range': [0.8, 1.2], # Contrast factor range
            'color_twist_saturation_range': [0.8, 1.2], # Saturation factor range
            'color_twist_hue_range': [-0.1, 0.1], # Hue shift range
        }
    },

    # --- Model Configuration ---
    'model': {
        'object_detector_config': {
            'model_name': 'yolov8n.pt', # YOLO model variant (e.g., yolov8n.pt, yolov8s.pt)
            'pretrained': True,
            'num_objects': 5, # Maximum number of objects to detect per image
            'feature_dim': 512, # Dummy feature dimension, will be overridden by backbone
            'yolo_conf_threshold': 0.25, # Confidence threshold for YOLO detections
            'yolo_iou_threshold': 0.7, # IoU threshold for NMS in YOLO
            'sam_model_type': 'vit_h', # SAM model type: 'vit_h', 'vit_l', 'vit_b'
            'sam_checkpoint_path': 'checkpoints/sam_vit_h_4b8939.pth', # Path to SAM checkpoint
        },
        'attribute_classifier_config': {
            'backbone_name': 'efficientnet_b0', # 'mobilenet_v3_small' or 'efficientnet_b0'
            'pretrained': True,
            'freeze_backbone': False, # Freeze backbone weights during training
            # Number of classes for each attribute (derived from mappings above)
            'shape': len(ATTRIBUTE_SHAPE_MAP),
            'color': len(ATTRIBUTE_COLOR_MAP),
            'fill': len(ATTRIBUTE_FILL_MAP),
            'size': len(ATTRIBUTE_SIZE_MAP),
            'orientation': len(ATTRIBUTE_ORIENTATION_MAP),
            'texture': len(ATTRIBUTE_TEXTURE_MAP),
        },
        'relation_gnn_config': {
            'hidden_dim': 256,
            'num_layers': 3,
            'num_relation_classes': len(RELATION_MAP),
        },
        'bongard_head_config': {
            'input_dim': 0, # This will be calculated dynamically in PerceptionModule
            'hidden_dim': 512,
            'dropout_prob': 0.3,
            'num_classes': 2, # Binary classification: positive or negative Bongard problem
        },
        'MAX_GNN_OBJECTS': 5, # Max objects for GNN, used for edge mapping

        # SimCLR Configuration
        'simclr_config': {
            'enabled': True,
            'projection_dim': 128, # Dimension of the projection head output
            'temperature': 0.5, # Temperature for NT-Xent loss
            'pretrain_epochs': 10, # Number of epochs for SimCLR pretraining
            'pretrain_batch_size': 64,
            'pretrain_lr': 1e-3,
        },
        # DeepSets/Set Transformer for Support Set Context Modeling
        'support_set_encoder_config': {
            'enabled': True,
            'encoder_type': 'deep_sets', # 'deep_sets', 'set_transformer', or 'linear_attention_set_transformer'
            'input_dim': 0, # Will be derived from object features
            'hidden_dim': 256,
            'output_dim': 128, # Dimension of the context vector
            'num_heads': 4, # For Set Transformer
            'num_blocks': 2, # For Set Transformer
        },
        'use_cross_attention_for_bongard_head': True, # Use cross-attention in BongardHead
    },

    # --- Training Configuration ---
    'training': {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'optimizer': 'AdamW', # AdamW, SGD, SAM, SophiaG
        'scheduler': 'OneCycleLR', # OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
        'scheduler_config': {
            'OneCycleLR': {'max_lr': 1e-3, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
            'ReduceLRO2nPlateau': {'mode': 'min', 'factor': 0.5, 'patience': 5},
            'CosineAnnealingLR': {'T_max': 50, 'eta_min': 1e-6},
        },
        'weight_decay': 1e-2,
        'max_grad_norm': 1.0, # Gradient clipping
        'use_amp': True, # Automatic Mixed Precision
        'label_smoothing': 0.1,
        'log_interval_batches': 10,
        'validation_frequency_batches': 50, # Validate every N batches
        'seed': 42,

        # Curriculum Learning
        'curriculum_learning': True,
        'curriculum_config': {
            'enabled': True,
            'start_image_size': 64, # Start with smaller images
            'end_image_size': 224, # Gradually increase to full size
            'size_increase_epochs': 20, # Number of epochs over which to increase image size
            'difficulty_sampling': True, # Enable Prioritized Experience Replay (PER)
            'difficulty_update_frequency_batches': 5, # How often to update priorities in PER
        },
        'per_config': { # Prioritized Experience Replay configuration
            'capacity': 10000, # Max size of the replay buffer
            'alpha': 0.6, # Controls how much prioritization is used (0 = uniform, 1 = full)
            'beta_start': 0.4, # Initial importance sampling (IS) weight compensation
            'beta_frames': 100000, # Number of frames over which to anneal beta to 1.0
        },

        # Data Augmentations
        'use_mixup_cutmix': True,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
        'mixup_prob': 0.5,
        'cutmix_prob': 0.5,

        # Model Optimization
        'use_swa': False, # Stochastic Weight Averaging
        'swa_start_epoch': 30,
        'swa_lr': 5e-4,

        'use_sam': False, # Sharpness-Aware Minimization (SAM)
        'sam_rho': 0.05, # SAM's neighborhood size

        'use_knowledge_distillation': False, # Knowledge Distillation (for student training)
        'distillation_config': {
            'temperature': 2.0, # Softmax temperature for teacher and student logits
            'alpha': 0.5, # Weight for the soft target loss
            'ensemble_dropout_prob': 0.3, # Probability of dropping a teacher model in ensemble distillation
            'use_mask_distillation': False, # Whether to use mask-based distillation
            'mask_agreement_threshold': 0.9, # Threshold for ensemble agreement to create a mask
        },
        'teacher_logits': None, # Placeholder for teacher logits during student training

        'use_quantization_aware_training': False, # QAT
        'use_post_training_quantization': False, # PTQ
        'use_structured_pruning': False, # Pruning
        'pruning_config': {
            'enabled': False, # Master switch for pruning
            'method': 'l1_unstructured', # 'l1_unstructured', 'random_unstructured', 'ln_structured'
            'amount': 0.5, # Percentage of connections to prune
            'iterations': 3, # Number of pruning + fine-tuning cycles
            'fine_tune_epochs_per_iter': 5, # Epochs to fine-tune after each pruning step
            'fine_tune_lr': 1e-5, # Learning rate for fine-tuning
            'use_sensitivity_analysis': False, # Enable sensitivity analysis to select layers
            'pruning_target_layers_ratio': 0.5, # Ratio of layers (ranked by sensitivity) to prune
            'prune_interval_epochs': 10, # Apply pruning every N epochs
        },
        'export_quantized_model': False, # Export model after PTQ/QAT

        # Loss Weights
        'consistency_loss_weight': 1.0,
        'feature_consistency_weight': 0.5, # Weight for feature consistency part of consistency loss
        'symbolic_consistency_weight': 0.5, # Weight for symbolic consistency part of consistency loss
        'relation_loss_weight': 1.0, # Weight for relation classification loss

        # RL Reformulation
        'use_rl_reformulation': False, # Enable RL-based rule search
        'rl_config': {
            'policy_lr': 1e-4,
            'num_episodes': 1000,
            'max_steps_per_episode': 10, # Max rule atoms/ops to propose
            'gamma': 0.99, # Discount factor
            'entropy_coeff': 0.01, # For exploration
        },
    },

    # --- Ensemble Configuration ---
    'ensemble': {
        'num_members': 3, # Number of ensemble members (teachers)
        'train_members': True, # True to train new members, False to load existing
        'teacher_ensemble_type': 'weighted_average', # 'weighted_average' or 'stacked'
        'stacked_ensemble_config': {
            'meta_learner_epochs': 100,
            'meta_learner_lr': 1e-3,
        },
        'student_model_config': { # Configuration for the student model (can be smaller)
            'object_detector_config': {
                'model_name': 'yolov8n.pt', # Smaller YOLO for student
                'pretrained': True,
                'num_objects': 5,
                'feature_dim': 512,
                'yolo_conf_threshold': 0.25,
                'yolo_iou_threshold': 0.7,
                'sam_model_type': 'vit_b', # Smaller SAM for student
                'sam_checkpoint_path': 'checkpoints/sam_vit_b_01ec64.pth', # Path to SAM checkpoint
            },
            'attribute_classifier_config': {
                'backbone_name': 'mobilenet_v3_small', # Smaller backbone for student
                'pretrained': True,
                'freeze_backbone': False,
                'shape': len(ATTRIBUTE_SHAPE_MAP), 'color': len(ATTRIBUTE_COLOR_MAP),
                'fill': len(ATTRIBUTE_FILL_MAP), 'size': len(ATTRIBUTE_SIZE_MAP),
                'orientation': len(ATTRIBUTE_ORIENTATION_MAP), 'texture': len(ATTRIBUTE_TEXTURE_MAP),
            },
            'relation_gnn_config': {
                'hidden_dim': 128, # Smaller GNN for student
                'num_layers': 2,
                'num_relation_classes': len(RELATION_MAP),
            },
            'bongard_head_config': {
                'input_dim': 0, # Will be calculated dynamically
                'hidden_dim': 256, # Smaller head for student
                'dropout_prob': 0.2,
                'num_classes': 2,
            },
        },
        'student_epochs': 50,
        'student_lr': 5e-5,
    },

    # --- Meta-Learner Configuration (Moved from training loop) ---
    'meta': {
        'epochs': 50, # Example epochs for meta-learner
        'lr': 1e-4,   # Example learning rate for meta-learner
    },

    # --- Slipnet Configuration ---
    'slipnet_config': {
        'initial_activation_decay_rate': 0.05, # Faster decay for initial activations
        'general_decay_factor': 0.01, # General decay applied per step
        'max_activation': 1.0, # Max activation value for any node
        'activation_threshold': 0.1, # Minimum activation for a node to be considered 'active'
        'link_strength_decay_rate': 0.001, # Rate at which link weights decay (if dynamic)
        'link_reinforce_factor': 0.1, # Factor by which link weights are reinforced (if dynamic)
    },

    # --- Debugging & Logging ---
    'debug': {
        'log_level': 'INFO', # DEBUG, INFO, WARNING, ERROR, CRITICAL
        'save_model_checkpoints': './checkpoints',
        'save_grad_cam_outputs': './grad_cam_outputs',
        'enable_grad_cam': True, # Enable Grad-CAM visualization for misclassified samples
        'rule_eval_log_interval_batches': 100, # Log rule evaluation every N batches
        'workspace_snapshot_interval_steps': 10, # How often to save workspace snapshots
        'slipnet_snapshot_interval_steps': 10, # How often to save Slipnet snapshots
    }
}

# Create checkpoints directory if it doesn't exist
os.makedirs(CONFIG['debug']['save_model_checkpoints'], exist_ok=True)
os.makedirs(CONFIG['debug']['save_grad_cam_outputs'], exist_ok=True)

