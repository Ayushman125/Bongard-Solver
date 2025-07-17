# Folder: bongard_solver/core_models/
# File: training_args.py

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml
import os
import torch
import logging
import torchvision.transforms as T # For mo_transform

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    backbone: str = "mobilenet_v3_small"
    pretrained: bool = True
    feat_dim: int = 576   # Example feature dimension from backbone
    proj_dim: int = 128   # Projection dimension for contrastive learning
    
    # SimCLR/MoCo specific configuration
    simclr_config: Dict[str, Any] = field(default_factory=lambda: {
        'projection_dim': 128,
        'head_layers': 2,
        'mlp_hidden_size': 2048,
        'use_moco': True,
        'moco_k': 65536,
        'moco_m': 0.999,
        'temperature': 0.07,
        'pretrain_epochs': 100,
        'head_dropout_prob': 0.2,
        'moco_ckpt_name': 'moco_combined_encoder.pth' # Name for combined data checkpoint
    })
    # Bongard Head configuration
    bongard_head_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_classes': 2, # Binary classification for Bongard problems (positive/negative)
        'hidden_dim': 256,
        'attn_dim': 256, # Attention dimension for FiLM
        'dropout_prob': 0.3
    })
    # Relation GNN configuration
    relation_gnn_config: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_dim': 256,
        'num_layers': 2,
        'num_relations': 11, # Example: left_of, above, inside, etc. + 'none'
        'dropout_prob': 0.3,
        'use_edge_features': False,
        'global_pool': 'mean' # 'mean' or 'attention'
    })
    # Attribute Classifier configuration
    attribute_classifier_config: Dict[str, Any] = field(default_factory=lambda: {
        'shape': 5, 'color': 7, 'size': 3, 'fill': 2, 'orientation': 4, 'texture': 2, # Example attribute class counts
        'mlp_dim': 256,
        'head_dropout_prob': 0.3
    })
    # Regularization
    use_lora: bool = False
    lora_config: Dict[str, Any] = field(default_factory=lambda: {
        'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.1, 'target_modules': ["q_proj", "v_proj"]
    })
    use_dropblock: bool = False
    dropblock_config: Dict[str, Any] = field(default_factory=lambda: {
        'block_size': 7, 'drop_prob': 0.1
    })
    use_stochastic_depth: bool = False
    drop_path_max: float = 0.1
    drop_path_layers: int = 5 # Number of layers over which to apply stochastic depth schedule
    use_scene_gnn: bool = False # Whether to use SceneGNN instead of RelationGNN
    use_glu: bool = False # Whether to use Gated Linear Units in BongardHead

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    image_size: List[int] = field(default_factory=lambda: [224, 224]) # Changed to list for [H,W]
    dataloader_workers: int = 4
    use_synthetic_data: bool = True
    synthetic_data_config: Dict[str, Any] = field(default_factory=lambda: {
        'num_train_problems': 10000,
        'num_val_problems': 2000,
        'num_test_problems': 2000,
        'max_objects_per_image': 5,
        'max_support_images_per_problem': 5,
        'min_dist_objects': 0.1, # Minimum normalized distance between objects
        'object_size_range': [0.1, 0.3], # Normalized size range
        'attr_distribution': {'shape': [0.5, 0.5], 'color': [0.5, 0.5]}, # Example distribution
        'relation_density': 0.5, # Probability of a relation existing between two objects
        'seed': 42,
        'background_texture_path': './data/textures', # Path to background textures
        'num_positive_examples': 6, # For LogoGenerator
        'num_negative_examples': 6, # For LogoGenerator
        'occluder_prob': 0.3, # For LogoGenerator
        'blur_prob': 0.2, # For LogoGenerator
        'min_occluder_size': 5, # For LogoGenerator
        'max_occluder_size': 20, # For LogoGenerator
        'jitter_width_range': [1, 3], # For LogoGenerator
        'jitter_dash_options': [None, (4,2), (2,2,2)] # For LogoGenerator
    })
    real_data_path: str = "./data/real_bongard_problems"
    dataset_cache_dir: str = "./data/cache"
    
    # Bongard-LOGO specific
    bongardlogo_root: str = 'data/Bongard-LOGO/data' # Root for Bongard-LOGO dataset
    
    # DALI specific
    use_dali: bool = False
    dali_gpu_augmentation: bool = True

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "AdamW" # AdamW, SGD, Ranger, Lion, SAM
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        'SGD': {'momentum': 0.9},
        'AdamW': {'eps': 1e-8},
        'SAM': {'base_optimizer': 'AdamW', 'rho': 0.05, 'adaptive': False}
    })
    scheduler: str = "CosineAnnealingLR" # OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
    scheduler_config: Dict[str, Any] = field(default_factory=lambda: {
        'OneCycleLR': {'max_lr': 1e-3, 'pct_start': 0.3, 'anneal_strategy': 'cos', 'cycle_momentum': True, 'base_momentum': 0.85, 'max_momentum': 0.95, 'div_factor': 25.0, 'final_div_factor': 10000.0, 'steps_per_epoch': 1000},
        'ReduceLROnPlateau': {'mode': 'min', 'factor': 0.1, 'patience': 5},
        'CosineAnnealingLR': {'eta_min': 1e-6}
    })
    early_stop_patience: int = 10
    early_stop_delta: float = 0.001
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0 # Gradient clipping
    use_amp: bool = False # Automatic Mixed Precision
    log_every_n_steps: int = 50
    
    # Knowledge Distillation
    use_knowledge_distillation: bool = False
    distillation_config: Dict[str, Any] = field(default_factory=lambda: {
        'temperature': 2.0,
        'alpha': 0.5, # Weight for soft targets
        'loss_weight': 1.0, # Overall weight for distillation loss
        'use_mask_distillation': False, # Whether to apply a mask to distillation loss
        'mask_threshold': 0.8 # Confidence threshold for masking
    })
    # Ensemble Training
    num_ensemble_members: int = 5
    ensemble_training_epochs: int = 20
    teacher_ensemble_type: str = 'weighted_average' # 'soft_vote', 'hard_vote', 'weighted_average', 'stacked'
    stacked_ensemble_config: Dict[str, Any] = field(default_factory=lambda: {
        'meta_learner_epochs': 50,
        'meta_learner_lr': 0.001,
        'meta_learner_hidden_dim': 128
    })
    # Consistency Losses
    consistency_loss_weight: float = 0.1 # Overall weight for consistency losses
    feature_consistency_weight: float = 0.5
    symbolic_consistency_weight: float = 0.5
    # Loss Weights for multi-task learning
    attribute_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    
    # Curriculum Learning / Prioritized Experience Replay
    curriculum_learning: bool = False
    curriculum_config: Dict[str, Any] = field(default_factory=lambda: {
        'difficulty_sampling': False, # Use PER for difficulty sampling
        'difficulty_metric': 'rule_complexity', # 'rule_complexity', 'object_count', 'relation_count'
        'difficulty_update_frequency_batches': 100,
        'per_alpha': 0.6, # PER alpha parameter
        'per_beta_start': 0.4, # PER beta parameter (annealed)
        'per_beta_frames': 100000 # Total frames for beta annealing
    })
    # Mean Teacher
    use_mean_teacher: bool = False
    mean_teacher_config: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 0.99 # EMA decay rate
    })
    # Data Augmentation
    use_mixup_cutmix: bool = False
    mixup_cutmix_config: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 1.0, # Beta distribution parameter
        'prob': 0.5, # Probability of applying mixup/cutmix
        'cutmix_beta': 1.0, # Beta distribution parameter for cutmix
        'cutmix_prob': 0.5 # Probability of applying cutmix
    })
    # Domain Adaptation
    use_domain_adaptation: bool = False
    grl_alpha: float = 1.0 # Gradient Reversal Layer alpha
    lambda_style: float = 0.1 # Weight for style loss
    lr_disc: float = 1e-4 # Learning rate for discriminator
    
    # Quantization
    quantization: Dict[str, Any] = field(default_factory=lambda: {
        'qat': False, # Quantization Aware Training
        'ptq': False, # Post-Training Quantization
        'backend': 'fbgemm' # 'fbgemm', 'qnnpack'
    })
    
    # Online Fine-tuning
    online_finetuning: bool = False
    fine_tune_threshold: int = 100 # Minimum samples in buffer to trigger fine-tuning
    
    # Bongard-LOGO specific training args
    logo_epochs: int = 5 # Epochs for supervised fine-tuning on Bongard-LOGO
    bongardlogo_ckpt: str = 'checkpoints/bongardlogo_cnn.pth' # Checkpoint path for Bongard-LOGO model

@dataclass
class DebugConfig:
    """Configuration for debugging and logging."""
    seed: int = 42
    log_level: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
    save_model_checkpoints: str = "./checkpoints"
    logs_dir: str = "./logs"
    use_wandb: bool = False
    wandb_project: str = "bongard-solver"
    wandb_entity: Optional[str] = None
    save_diversity_metrics: bool = True
    diversity_metrics_path: str = "./diversity_metrics.json"

@dataclass
class Config:
    """Overall configuration for the Bongard solver."""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    # Phase 1 hyper-configs and checkpointing
    # Perception CNN
    perception_device: str = "cuda"
    img_size: int = 128
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-3

    # Replay-buffer fine-tuning
    fine_tune_threshold: int = 500
    fine_tune_epochs: int = 1

    # Synthetic generator
    textures_dir: str = "data/textures"

    # Validation sets
    synth_holdout_count: int = 200
    real_holdout_root: str = "data/real_bongard/VAL"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_path: str = f"checkpoints/bongard_perception_last.pth"
    best_model_path: str = f"checkpoints/bongard_perception_best.pth"
    last_model_path: str = f"checkpoints/bongard_perception_last.pth"

    # Script mode
    validate_bins: int = 10

    # DALLIPipeline feature flag
    use_dallipipeline: bool = False

    # MoCo-specific transform for combined dataset (added here for central management)
    # This transform needs to handle both RGB (synthetic) and grayscale (Bongard-LOGO) inputs.
    # For grayscale, convert to RGB before applying color jitter/normalize with 3 channels.
    # This is a property, so it's initialized when Config is created.
    @property
    def mo_transform(self):
        # We need to access image_size from self.data.image_size
        img_h, img_w = self.data.image_size[0], self.data.image_size[1]
        
        # Define ImageNet mean and std for 3 channels
        IMAGENET_MEAN_RGB = [0.485, 0.456, 0.406]
        IMAGENET_STD_RGB = [0.229, 0.224, 0.225]

        return T.Compose([
            # Convert 'L' (grayscale) to 'RGB' if needed, before other transforms
            # This lambda ensures the image is always RGB before subsequent transforms
            lambda img: img.convert('RGB') if img.mode == 'L' else img,
            T.RandomResizedCrop(img_h, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN_RGB, std=IMAGENET_STD_RGB)
        ])


def load_config(config_path: str) -> Config:
    """
    Loads configuration from a YAML file and creates a Config object.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        Config: A Config object populated with values from the YAML file.
    """
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Using default configuration.")
        return Config()
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Create instances of nested dataclasses, handling potential missing keys
    model_config = ModelConfig(**yaml_config.get('model', {}))
    data_config = DataConfig(**yaml_config.get('data', {}))
    training_config = TrainingConfig(**yaml_config.get('training', {}))
    debug_config = DebugConfig(**yaml_config.get('debug', {}))
    
    # Populate the main Config object
    config = Config(
        device=yaml_config.get('device', "cuda" if torch.cuda.is_available() else "cpu"),
        model=model_config,
        data=data_config,
        training=training_config,
        debug=debug_config
    )
    logger.info(f"Configuration loaded from {config_path}.")
    return config

def save_config(config: Config, save_path: str):
    """
    Saves the current Config object to a YAML file.
    Args:
        config (Config): The Config object to save.
        save_path (str): The path to save the YAML file.
    """
    try:
        # Convert dataclass to dictionary
        config_dict = dataclasses.asdict(config)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, indent=4)
        logger.info(f"Configuration saved to {save_path}.")
    except Exception as e:
        logger.error(f"Error saving configuration to {save_path}: {e}")

if __name__ == "__main__":
    # Example usage:
    # Create a default config
    default_config = Config()
    
    # Define a path to save a sample config
    sample_config_path = "sample_config.yaml"
    save_config(default_config, sample_config_path)
    # Load the config back
    loaded_config = load_config(sample_config_path)
    
    # Access parameters
    logger.info(f"Loaded device: {loaded_config.device}")
    logger.info(f"Loaded backbone: {loaded_config.model.backbone}")
    logger.info(f"Loaded batch size: {loaded_config.training.batch_size}")
    logger.info(f"Loaded SimCLR pretrain epochs: {loaded_config.model.simclr_config['pretrain_epochs']}")
    logger.info(f"Loaded Bongard-LOGO root: {loaded_config.data.bongardlogo_root}")
    logger.info(f"Loaded Bongard-LOGO checkpoint: {loaded_config.training.bongardlogo_ckpt}")
    logger.info(f"Loaded MoCo synthetic count: {loaded_config.data.synthetic_data_config['num_train_problems']}")
    logger.info(f"Loaded MoCo transform (first element): {loaded_config.mo_transform.transforms[0]}")

    # You can also modify and save
    loaded_config.training.epochs = 100
    loaded_config.model.backbone = "resnet50"
    loaded_config.data.image_size = [256, 256]
    save_config(loaded_config, "modified_config.yaml")
    logger.info("Modified config saved to modified_config.yaml")

# --- Expose a module-level config instance for direct import ---
config = Config()
