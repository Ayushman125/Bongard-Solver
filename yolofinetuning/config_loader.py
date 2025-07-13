# yolofinetuning/config_loader.py
import os
import logging
import torch

# Define YOLO_CLASSES_LIST here as it's used in CONFIG.
# This assumes YOLO_CLASSES_LIST is a static list of class names.
# If it's dynamically generated, its source might need further refactoring
# to avoid new circular dependencies.
YOLO_CLASSES_LIST = ['circle', 'square', 'triangle', 'line', 'dot', 'polygon',
                     'pentagon', 'hexagon', 'star', 'cross', 'plus', 'ellipse', 'unknown_shape'] # Extended for multi-class

# --- GLOBAL CONFIG ────────────────────────────────────────────────────────────
CONFIG = {
    'bongard_root':       './ShapeBongard_V2',  # Relative to project root
    'background_root': './backgrounds',  # Relative to project root
    'output_root':        './datasets/bongard_objects',  # Relative to project root
    'splits':             {'train':0.7, 'val':0.15, 'test':0.15, 'challenge_val': 0.1},  # Added challenge_val split
    'seed':               42,
    'image_size':         [224, 224],  # Target H, W for generated images and YOLO training
    # contour detection (used by _detect_labels_and_difficulty_wrapper in pipeline_workers)
    'cnt_block': 11, 'cnt_C':2, 'min_area':100, 'max_cnt':10,
    'morphological_ops': 'open',  # 'open', 'close', or None for morphological cleanup
    # initial augmentation hyperparams (these will be tuned by Optuna, used by DALI)
    'elastic_p':          0.3,
    'elastic':            {'alpha':1,'sigma':50},
    'phot_blur_p':        0.3,
    'phot_blur':          (3,7),
    'jpeg_p':             0.3,
    'jpeg_q':             (30,70),
    'stroke_ker':         3,
    'occlude_p':          0.3,
    'occlude_max':        0.3,
    'add_clutter_p':      0.2,
    'num_clutter_patches': 3,
    'clutter_max_factor':0.1,
    'augmix_p':           0.5,
    'rand_nops':          2,
    'rand_mag':           9,
    'mixup_alpha':        0.4,  # Alpha for MixUp/CutMix
    'fract_depth':        5,
    'fill_contour_p':     0.5,
    'gan_generate_n':     20000,    # Increased GAN images significantly
    'pil_generate_n_per_class': 5000,  # Increased PIL images per class
    # NEW DALI geometric augmentations
    'dali_rotation_p': 0.5,
    'dali_rotation_degrees': 90,
    'dali_translation_p': 0.5,
    'dali_translation_range': (0.2, 0.2),    # fraction of width/height
    'dali_shear_p': 0.3,
    'dali_shear_range': (10.0, 10.0),        # degrees
    # NEW DALI color augmentations
    'dali_hsv_p': 0.5,
    'dali_hsv_h_range': (-10, 10),
    'dali_hsv_s_range': (0.8, 1.2),
    'dali_hsv_v_range': (0.8, 1.2),
    # NEW DALI noise augmentations
    'dali_gaussian_noise_p': 0.3,
    'dali_gaussian_noise_std': 10.0,
    'dali_salt_pepper_p': 0.2,
    # Curriculum Difficulty Scoring (weights passed to DALI's python_function via DIFFICULTY_WEIGHTS)
    'difficulty_weights': {
        'num_objects': 0.4,
        'avg_complexity': 0.3,
        'num_relations': 0.3,
    },
    'difficulty_csv_path': './datasets/bongard_objects/difficulty_summary.csv',  # Relative to project root
    # Optuna Tuning Parameters
    'tuning_n_trials': 100,  # Increased trials for Optuna
    'tuning_subset_size': 500,  # Increased images to sample for tuner evaluation
    'tuning_min_labels_per_sample': 10,  # Minimum labels required for a valid tuning sample
    'tuning_min_diffs_per_sample': 5,  # Minimum difficulty scores required for a valid tuning sample
    'tuning_db_path': 'sqlite:///datasets/bongard_objects/tuning_results.db',  # Relative to project root, in output dir
    # NEW Optuna parameters for YOLO-centric study
    'yolo_label_smoothing': 0.0,  # Default, will be tuned
    'yolo_dropout': 0.0,          # Default, will be tuned
    'yolo_weight_decay': 1e-5,    # Default, will be tuned
    'hard_mining_topk_frac': 0.3,  # Fraction of hard examples to keep after sorting by difficulty
    # YOLO Model Fine-tuning Parameters
    'model_save_dir': './runs/train/yolov8_bongard',  # Relative to project root
    'model_name': 'yolov8s.pt',  # UPGRADED to YOLOv8s
    'num_classes': len(YOLO_CLASSES_LIST),  # Dynamically set based on YOLO_CLASSES_LIST
    'class_names': YOLO_CLASSES_LIST,  # Dynamically set based on YOLO_CLASSES_LIST
    'yolo_img_size': [640, 640],  # Consistent image size for YOLO training (simplified from progressive resize)
                                 # This will be the target size for DALI/FFCV/PyTorch loaders
    'yolo_epochs': 60,  # Total epochs for two-stage training
    'yolo_batch_size': 2,  # Smaller batch size for accumulate (Empirically selected for 4GB VRAM)
    'yolo_accumulate': 8,  # Gradient accumulation steps (effective batch = 16) (Empirically selected for 4GB VRAM)
    'yolo_learning_rate': 1e-3,  # Initial learning rate for YOLO
    'yolo_final_lr_factor': 0.05,  # Final learning rate factor for cosine scheduler
    'yolo_optimizer': 'AdamW',  # AdamW is a good choice for fine-tuning, SAM will be handled in yolo_fine_tuning.py
    'yolo_device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Device for YOLO training
    'yolo_ema': True,  # Enable Exponential Moving Average
    'yolo_patience': 10,  # Early stopping patience
    'yolo_save_period': 1,  # Save every epoch
    # DALI Specific Parameters for Data Generation
    'dali_batch_size': 64,  # Batch size for DALI pipeline during data generation
    'dali_num_threads': os.cpu_count() - 1,  # Number of CPU threads for DALI - Tweak after profiling
    'dali_py_num_workers': os.cpu_count(),  # Number of Python workers for fn.python_function - Tweak after profiling
    'dali_prefetch_queue': 8,  # DALI prefetch queue depth - Increased for better GPU utilization
    'force_cpu_dali': False,  # Force DALI ops to CPU even if GPU is available
    # NEW DALI Memory Preallocation
    'dali_gpu_memory_bytes': 1024 * 1024 * 1024,  # 1GB for DALI GPU pool (Increased)
    'dali_pinned_memory_bytes': 512 * 1024 * 1024,  # 512MB for DALI pinned host memory (Increased)
    'dali_buffer_growth_factor': 2.0,  # Factor by which DALI buffers grow
    'dali_host_buffer_shrink_threshold': 0.1,  # Threshold for host buffer shrinking
    # Debugging / Visualization (add if needed)
    'visualize_predictions_samples': 10,
    'visualize_yolo_save_dir': './runs/predict/yolov8_bongard_val_preds',  # Relative to project root
    # Cache for raw image paths
    'raw_image_paths_cache': './datasets/bongard_objects/raw_image_paths_cache.json',  # Relative to project root
    'cache_freshness_days': 7,  # How many days until cache is considered stale
    # Flag file to indicate if data generation is complete
    'data_generated_flag_file': './datasets/bongard_objects/.data_generation_complete',  # Relative to project root
    # NEW: Flag file for raw annotation pre-computation
    'raw_annotations_precomputed_flag_file': './datasets/bongard_objects/.raw_annotations_precomputed',  # Relative to project root
    # Directory to store pre-computed raw annotations
    'raw_annotations_dir': './datasets/bongard_objects/raw_annotations',  # Relative to project root
    # Temporary directory for all generated images before splitting
    'temp_generated_data_dir': './datasets/bongard_objects/temp_generated_data',  # Relative to project root
    # File list for DALI reader (will now include annotation paths)
    'dali_file_list_path': './datasets/bongard_objects/dali_file_list.txt',  # Relative to project root
    # NEW: Persistent directory for GAN images
    'gan_persistent_dir': './datasets/bongard_objects/gan_raw_images',  # Relative to project root
    # NEW: Persistent directory for PIL generated images
    'pil_persistent_dir': './datasets/bongard_objects/pil_raw_images',  # Relative to project root
    # NEW: Persistent file to track DALI processed stems
    'dali_processed_stems_tracker': './datasets/bongard_objects/.dali_processed_stems_tracker.txt',  # Relative to project root
    # Hard Example Mining
    'hard_mining_conf_thresh': 0.05,  # Confidence threshold for identifying hard examples (Refined)
    'hard_examples_dir': './datasets/bongard_objects/hard_examples',  # Relative to project root
    'hard_mining_epochs': 5,  # Epochs for retraining on hard examples
    # --- NEW SECTIONS FROM PATCH ---
    # Attention Modules
    'attention': {
        'type': 'se',          # options: 'none', 'cbam', 'se'
        'se_reduction': 16     # Reduction ratio for SEBlock
    },
    # Multi-Backbone Ensemble
    'ensemble': {
        'enabled': True,
        'detectors': [
            {'type': 'yolov8', 'weights': 'runs/train/best.pt'},
            {'type': 'fcos',   'weights': 'runs/fcos/best.pt', 'config': 'configs/fcos.yaml'}
        ]
    },
    # Detector-Segmentor Joint Head
    'mask_head': {
        'enabled': True,
        'type': 'yolact',
        'num_classes': 1,  # Number of classes for segmentation (e.g., 1 for foreground/background)
        'prototype_channels': 32,  # Channels for YOLACT prototypes
        'mask_size': 28  # Output mask size (e.g., 28x28)
    },
    # Learned NMS
    'learned_nms': {
        'enabled': True,
        'hidden_dim': 64,
        'num_layers': 2,
        'score_threshold': 0.05,
        'iou_threshold': 0.5
    },
    # Inference-Time Settings
    'inference': {
        'tta': True,
        'soft_nms': True,
        'nms_threshold': 0.6
    },
    # Pretraining (SSL)
    'pretrain': {
        'enabled': False,  # Set to True to enable SSL pretraining
        'ssl_epochs': 50,
        'encoder': 'resnet18',  # or 'pvtv2_b0' if you have it implemented
        'contrastive': {
            'temperature': 0.07,
            'batch_size': 256
        }
    },
    # SimGAN for Domain Randomization
    'simgan': {
        'enabled': True,  # Set to True to enable SimGAN
    },
    # CycleGAN for Domain Randomization
    'cyclegan': {
        'enabled': False,  # Set to True to enable CycleGAN
        'path': '/path/to/cyclegan.pth'  # Path to CycleGAN generator weights
    },
    # Augmentation specific settings (for procedural.py and my_data_utils.py)
    'augmentation': {
        'simgan': False,  # Controlled by 'simgan.enabled' now
        'cyclegan': False,  # Controlled by 'cyclegan.enabled' now
        'occlusion': {
            'max_shapes': 5,
            'occlusion_prob': 0.5
        }
    },
    # Model Architecture Enhancements
    'model': {
        'backbone': 'yolov8s.pt',  # Default YOLOv8s, can be 'pvtv2_b0' if implemented
        'attention': 'none',  # options: 'none', 'cbam', 'se' # This will be overridden by 'attention' key directly
        'neck_nas': False  # Enable NAS-searchable neck (requires custom YOLOv8 implementation)
    },
    # Loss Function Enhancements
    'loss': {
        'dynamic_focal': True,
        'initial_gamma': 2.0,
        'min_gamma': 0.5,
        'max_gamma': 5.0,
        'dynamic_ciou': True
    },
    # Optimizer Enhancements
    'optimizer': {
        'lr': 1e-3,  # Base learning rate, will be overridden by Optuna if enabled
        'wd': 1e-4,  # Weight decay, will be overridden by Optuna if enabled
        'sam': {
            'enabled': False,  # Set to True to enable SAM optimizer
            'rho': 0.05
        }
    },
    # Semi-Supervised Learning
    'semi_supervised': {
        'enabled': False,  # Set to True to enable semi-supervised learning
        'teacher_weights': '/path/to/teacher.pt',  # Path to pre-trained teacher model
        'confidence_threshold': 0.7,
        'co_teaching_rate': 0.2,  # Not directly used in current pseudo-labeling logic
        'start_epoch': 10  # Epoch to start pseudo-labeling
    },
    # Curriculum Learning
    'curriculum': {
        'enabled': False,  # Set to True to enable curriculum learning
        'score_map': 'data/curriculum_scores.json'  # Path to JSON mapping image stems to difficulty scores
    },
    # Knowledge Distillation
    'distillation': {
        'enabled': False,  # Set to True to enable distillation
        'teacher': 'yolov8l.pt',  # Teacher model weights (e.g., larger YOLOv8)
        'student': 'yolov8s.pt',  # Student model weights (e.g., current YOLOv8s)
        'alpha': 0.9,  # Weight for distillation loss
        'temperature': 4.0
    },
    # Quantization-Aware Training (QAT)
    'qat': {
        'enabled': False,  # Set to True to enable QAT
        'bitwidth': 8
    },
    # Pruning
    'pruning': {
        'enabled': False,  # Set to True to enable pruning
        'target_sparsity': 0.3
    },
    # Hard Example Mining (OHEM and Active Learning)
    'hard_mining': {
        'ohem': {
            'enabled': False,  # Set to True to enable OHEM
            'top_k': 128
        },
        'active': {
            'enabled': False,  # Set to True to enable active learning
            'pool_path': './datasets/bongard_objects/unlabeled_pool',  # Path to unlabeled image pool
            'entropy_threshold': 1.5  # Threshold for selecting uncertain samples
        }
    },
    # Training Configuration (newly added)
    'train': {
        'epochs': 50,
        'batch_size': 8,
        'resume_from': 'checkpoints/last.pt',     # path to checkpoint to resume (or None)
        'checkpoint_dir': 'checkpoints',          # where to write .pt files
        'save_every': 5                           # save a checkpoint every N epochs
    },
    # Data Pipeline Configuration (newly added)
    'data_pipeline': {
        'type': 'pytorch',  # Default to 'pytorch', options: 'dali', 'ffcv', 'pytorch'
        'prefetch': True,  # Whether to use prefetch_loader for PyTorch/FFCV loaders
        'dali': { # DALI specific settings
            'batch_size': 64,
            'num_threads': os.cpu_count() - 1,
        },
        'ffcv': { # FFCV specific settings
            'batch_size': 16,
            'num_workers': os.cpu_count(),
        }
    },
    # YOLO Training Settings (These are top-level for convenience in some training scripts)
    "weights":           "yolov8s.pt",
    "data_yaml":         "/workspace/data/data.yaml", # Ensure this path is correct for your environment
    "epochs":            50,
    "img_size":          640, # This should match yolo_img_size[0] if it's a single value, or be the target size
    "batch_size":        16, # This should be the effective batch size after accumulation
    "project_dir":       "/workspace/runs/train",
    "exp_name":          "exp1",
    "conf_thresh":       0.25,
    "iou_thresh":        0.45,
    "class_names":       YOLO_CLASSES_LIST, # Use the global list
    # Data Root for loaders (consistent with prepare_yolo_dataset.py output)
    "data_root": "./data", # This is crucial for pipeline_workers.py

    # --- NEW: Multi-Modal Label Directory Mapping ---
    'label_directories': {
        "boxes":       "labels",
        "masks":       "masks",
        "polygons":    "polygons",
        "programs":    "programs",
        "relations":   "relations",
        "topo":        "topo",
        "descriptors": "stats", # Folder name 'stats' for descriptors
        "captions":    "captions"
    }
}

# --- LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger()  # Get the logger instance
