from pathlib import Path
# --- Helper functions ---
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
# Folder: bongard_solver/core_models/
# File: training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import logging
import os
import collections
import random
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import torchvision.transforms as T
import threading
import copy

# PyTorch Profiler imports
from torch.profiler import profile, record_function, ProfilerActivity

# Import configuration (from parent directory)
try:
    from ..config import (
        CONFIG, DEVICE, HAS_WANDB, HAS_TORCH_QUANTIZATION,
        ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
        ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
        RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD
    )
except ImportError:
    # Dummy imports for standalone execution if config is not set up yet
    CONFIG = {}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_WANDB = False
    HAS_TORCH_QUANTIZATION = False
    ATTRIBUTE_FILL_MAP = {'filled': 0, 'outlined': 1}
    ATTRIBUTE_COLOR_MAP = {'black': 0, 'white': 1} # Example
    ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2} # Example
    ATTRIBUTE_ORIENTATION_MAP = {'upright': 0} # Example
    ATTRIBUTE_SHAPE_MAP = {'triangle': 0, 'quadrilateral': 1, 'circle': 2, 'pentagon': 3, 'star': 4, 'text_character': 5}
    ATTRIBUTE_TEXTURE_MAP = {'flat': 0} # Example
    RELATION_MAP = {'none': 0, 'left_of': 1, 'above': 2, 'right_of': 3, 'below': 4, 'intersects': 5, 'contains': 6, 'is_contained_by': 7, 'aligned_vertically': 8, 'aligned_horizontally': 9} # Example
    IMAGENET_MEAN = [0.485, 0.456, 0.406] # For RGB images
    IMAGENET_STD = [0.229, 0.224, 0.225] # For RGB images
    logging.warning("Could not import full config. Using dummy values for some config items.")

# Import PyTorch Lightning modules (if used, though this script is more manual training loop)
import pytorch_lightning as pl # Keep for potential future use or if other parts rely on it

# Import models (from current directory)
from .models import LitBongard, LitSimCLR, PerceptionModule, BongardPerceptionModel # Added BongardPerceptionModel

# Import data module and loader (from parent directory's src folder)
try:
    from ..data import BongardDataModule, get_loader
    from ..data.generator import LogoGenerator # Import LogoGenerator
except ImportError:
    logging.warning("Could not import BongardDataModule, get_loader, or LogoGenerator. Dummy classes/functions will be used.")
    class BongardDataModule: # Dummy
        def __init__(self, config): pass
        def train_dataloader(self): return []
        def val_dataloader(self): return []
    def get_loader(dataset, batch_size, shuffle, num_workers): return [] # Dummy
    class LogoGenerator: # Dummy
        def __init__(self, cfg, bg_textures_dir="data/textures"): # Updated constructor to match generator.py
            self.size = cfg['data']['image_size'][0]
            self.canvas_width, self.canvas_height = self.size, self.size
            self.cfg = cfg['data']['synthetic_data_config'] # Access synthetic_data_config
        def make_problem(self, problem_id): # Dummy make_problem matching new signature
            img_size = CONFIG['data']['image_size'][0] if 'data' in CONFIG and 'image_size' in CONFIG['data'] else 128
            img = np.random.randint(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
            # Dummy support images (padded to max_support_images_per_problem)
            max_support = CONFIG['data']['synthetic_data_config'].get('max_support_images_per_problem', 0)
            support_imgs_np = [np.zeros((img_size, img_size, 3), dtype=np.uint8)] * max_support
            support_labels = [-1] * max_support
            support_sgs = [b'{}'] * max_support
            support_confs = [0.0] * max_support
            return (img, img, 0, b'{}', b'{}', 0.0, np.eye(3).tolist(), np.eye(3).tolist(), problem_id,
                    support_imgs_np, support_labels, support_sgs, max_support, problem_id, 1.0,
                    [], [], [], [], [], [], 1.0, 1.0, support_confs, "DUMMY_RULE") # Added confidences and gt_rule

# Import losses (from current directory)
from .losses import LabelSmoothingCrossEntropy, DistillationLoss, FeatureConsistencyLoss, SymbolicConsistencyLoss, GradNorm, CrossEntropyWithConfidence # Added CrossEntropyWithConfidence

# Import optimizers and schedulers (from current directory)
from .optimizers import get_optimizer, get_scheduler, SAM # Assuming SAM is exposed by optimizers.py

# Import for torchvision.transforms.v2 for MixUp/CutMix
HAS_TORCHVISION_V2 = False
try:
    import torchvision.transforms.v2 as T_v2
    HAS_TORCHVISION_V2 = True
    logging.getLogger(__name__).info("torchvision.transforms.v2 found and enabled.")
except ImportError:
    logging.getLogger(__name__).warning("torchvision.transforms.v2 not found. MixUp/CutMix will be disabled.")

# Import GradualWarmupScheduler and CosineAnnealingWarmRestarts (for 7.1)
HAS_GRADUAL_WARMUP = False
try:
    from warmup_scheduler import GradualWarmupScheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    HAS_GRADUAL_WARMUP = True
except ImportError:
    pass # Handled by optimizers.py

# Import StyleDiscriminator and GradReverse
try:
    from .style_disc import StyleDiscriminator
    from .grad_rev import grad_reverse
    HAS_DOMAIN_ADAPTATION = True
    logging.getLogger(__name__).info("StyleDiscriminator and GradReverse found for domain adaptation.")
except ImportError:
    HAS_DOMAIN_ADAPTATION = False
    logging.getLogger(__name__).warning("StyleDiscriminator or GradReverse not found. Domain adaptation will be disabled.")
    class StyleDiscriminator(nn.Module): # Dummy
        def __init__(self, feat_dim): super().__init__(); self.net = nn.Identity()
        def forward(self, x, alpha=1.0): return torch.zeros(x.shape[0], 1)
    def grad_reverse(x, alpha): return x

logger = logging.getLogger(__name__)

# --- Helper functions ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# Import async_update_priorities from replay_buffer.py (now in core_models)
from .replay_buffer import KnowledgeReplayBuffer, ReplayBuffer # Import both classes

# Dummy function for asynchronous priority update for replay buffer
def async_update_priorities(replay_buffer: KnowledgeReplayBuffer, tree_indices: List[int], losses: List[float], cfg: Dict[str, Any]):
    """
    Dummy function for asynchronous priority update.
    This function simulates updating replay buffer priorities.
    In a real scenario, this might involve a separate thread or process.
    """
    # The replay_buffer.update_priorities method now takes cfg directly for annealing.
    replay_buffer.update_priorities(tree_indices, losses, cfg) 
    logger.debug(f"Async update priorities called for {len(tree_indices)} samples.")

# Dummy QAT/PTQ functions (actual implementations are in prune_quantize.py, which is in the project root)
def quantize_model_qat(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Dummy function for Quantization Aware Training (QAT)."""
    logger.info("Performing dummy Quantization Aware Training (QAT).")
    return model

def quantize_model_ptq(model: nn.Module, val_loader: DataLoader, cfg: Dict[str, Any]) -> nn.Module:
    """Dummy function for Post-Training Quantization (PTQ)."""
    logger.info("Performing dummy Post-Training Quantization (PTQ).")
    return model

# Early Stopping class
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0, path: str = 'checkpoint.pt', trace_func=logger.info):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        # Save only the state_dict of the core model (PerceptionModule)
        if isinstance(model, DDP):
            torch.save(model.module.perception_module.state_dict(), self.path)
        elif isinstance(model, LitBongard):
            torch.save(model.perception_module.state_dict(), self.path)
        elif isinstance(model, BongardPerceptionModel): # Added for direct PerceptionModel saving
            torch.save(model.state_dict(), self.path)
        else: # Fallback for other model types
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Mixup/CutMix Augmenter
class MixupCutmixAugmenter:
    def __init__(self, training_cfg: Dict[str, Any], num_classes: int):
        self.use_mixup = training_cfg.get('use_mixup', False)
        self.use_cutmix = training_cfg.get('use_cutmix', False)
        self.mixup_prob = training_cfg.get('mixup_prob', 0.0)
        self.cutmix_prob = training_cfg.get('cutmix_prob', 0.0)
        self.mixup_alpha = training_cfg.get('mixup_alpha', 0.2)
        self.cutmix_alpha = training_cfg.get('cutmix_alpha', 1.0)
        self.num_classes = num_classes
        if not HAS_TORCHVISION_V2:
            logger.warning("torchvision.transforms.v2 not available. MixUp/CutMix will not function.")
            self.aug = None
        elif self.use_mixup and self.mixup_prob > 0:
            self.aug = T_v2.MixUp(
                num_classes=self.num_classes,
                prob=self.mixup_prob,
                alpha=self.mixup_alpha
            )
            logger.info(f"MixUp enabled with prob={self.mixup_prob}, alpha={self.mixup_alpha}")
        elif self.use_cutmix and self.cutmix_prob > 0:
            self.aug = T_v2.CutMix(
                num_classes=self.num_classes,
                prob=self.cutmix_prob,
                alpha=self.cutmix_alpha
            )
            logger.info(f"CutMix enabled with prob={self.cutmix_prob}, alpha={self.cutmix_alpha}")
        else:
            self.aug = None
            logger.info("MixUp/CutMix disabled by config.")

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.aug:
            # T_v2.MixUp/CutMix expects labels as (N,) LongTensor, returns (N, C) float tensor
            return self.aug(images, labels)
        # If no augmentation, convert labels to one-hot for consistency with augmented labels
        return images, F.one_hot(labels, num_classes=self.num_classes).float()

# Ensemble Teacher Logits (from previous context, updated to take config)
def _get_ensemble_teacher_logits(
    teacher_models: Union[nn.ModuleList, nn.Module], # Can be a list of models or a single meta-learner
    raw_images_np: List[np.ndarray],
    raw_gt_json_strings: List[bytes],
    raw_support_images_np: List[np.ndarray],
    distillation_config: Dict[str, Any],
    dali_image_processor: Any = None,  # DALI processor from datamodule
    config: Dict[str, Any], # Pass the full config here
    detected_bboxes_batch: Optional[List[List[List[float]]]] = None,
    detected_masks_batch: Optional[List[List[np.ndarray]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Helper function to get ensemble teacher logits for distillation.
    Args:
        teacher_models (Union[nn.ModuleList, nn.Module]): A list of trained ensemble member models
                                                         (PerceptionModule) or a single meta-learner.
        raw_images_np (List[np.ndarray]): List of raw query images (numpy arrays).
        raw_gt_json_strings (List[bytes]): List of ground truth JSON strings for query images.
        raw_support_images_np (List[np.ndarray]): List of raw support images (numpy arrays).
        distillation_config (Dict[str, Any]): Distillation configuration.
        dali_image_processor (Any, optional): DALI processor. Defaults to None.
        config (Dict[str, Any]): The full configuration dictionary.
        detected_bboxes_batch (Optional[List[List[List[float]]]]): Pre-detected bboxes for query images.
        detected_masks_batch (Optional[List[List[np.ndarray]]]): Pre-detected masks for query images.
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - ensemble_logits (torch.Tensor): Averaged teacher logits (B, num_classes).
            - final_distillation_mask (Optional[torch.Tensor]): Mask for distillation (B,).
    """
    if not teacher_models:
        logger.warning("No teacher models provided for distillation. Returning empty logits.")
        return torch.empty(0), None
    
    all_teacher_logits = []
    all_distillation_masks = []
    
    # Process images once for all teachers
    if dali_image_processor is None or not config['training'].get('use_dali', False):
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(tuple(config['data']['image_size'])), # Use tuple for Resize
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        processed_images = torch.stack([transform(img_np) for img_np in raw_images_np]).to(DEVICE)
        
        # Handle support images for teacher's forward pass
        max_support_imgs = config['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = processed_images.shape[0]
        
        # Ensure raw_support_images_np is not empty before stacking
        if raw_support_images_np and len(raw_support_images_np) > 0:
            # Check if raw_support_images_np contains actual image data or padding arrays
            # If it's a list of lists (from collate_fn), flatten it first
            if isinstance(raw_support_images_np[0], list):
                flat_support_images_np = [item for sublist in raw_support_images_np for item in sublist]
            else:
                flat_support_images_np = raw_support_images_np
            # Filter out empty/padding images if they are all zeros
            actual_support_images_np = [img for img in flat_support_images_np if np.sum(img) > 0]
            if actual_support_images_np:
                processed_support_images_flat = torch.stack([transform(img_np) for img_np in actual_support_images_np]).to(DEVICE)
                # Reshape to (batch_size, max_support_imgs, C, H, W)
                processed_support_images_reshaped = processed_support_images_flat.view(
                    batch_size_actual, max_support_imgs, 
                    processed_images.shape[1], processed_images.shape[2], processed_images.shape[3]
                )
            else:
                # If no actual support images, create a dummy tensor of correct shape
                processed_support_images_reshaped = torch.zeros(
                    batch_size_actual, max_support_imgs, 
                    processed_images.shape[1], processed_images.shape[2], processed_images.shape[3],
                    device=DEVICE
                )
        else:
            processed_support_images_reshaped = torch.zeros(
                batch_size_actual, max_support_imgs, 
                processed_images.shape[1], processed_images.shape[2], processed_images.shape[3],
                device=DEVICE
            )
    else:
        processed_images, _, processed_support_images_flat = dali_image_processor.run(
            raw_images_np,
            [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_images_np), # Dummy view2
            raw_support_images_np
        )
        max_support_imgs = config['data']['synthetic_data_config']['max_support_images_per_problem']
        batch_size_actual = processed_images.shape[0]
        processed_support_images_reshaped = processed_support_images_flat.view(
            batch_size_actual, max_support_imgs, 
            processed_images.shape[1], processed_images.shape[2], processed_images.shape[3]
        )
    # Dummy support labels, as they are not used by teacher's PerceptionModule directly for classification
    dummy_support_labels_flat = torch.zeros_like(processed_support_images_reshaped[:,:,0,0,0], dtype=torch.long) # Use a slice to get correct shape

    # If teacher_models is a single meta-learner (e.g., from stacked ensemble)
    if not isinstance(teacher_models, nn.ModuleList):
        teacher_model = teacher_models
        teacher_model.eval()
        with torch.no_grad():
            if isinstance(teacher_model, PerceptionModule) or isinstance(teacher_model, LitBongard): # If it's a single PerceptionModule or LitBongard
                # If it's LitBongard, get its internal perception_module
                core_model = teacher_model.perception_module if isinstance(teacher_model, LitBongard) else teacher_model
                teacher_outputs = core_model(
                    processed_images,
                    ground_truth_json_strings=raw_gt_json_strings, # Pass GT for scene graph building
                    detected_bboxes_batch=detected_bboxes_batch,
                    detected_masks_batch=detected_masks_batch,
                    support_images=processed_support_images_reshaped,
                    support_labels_flat=dummy_support_labels_flat
                )
                teacher_logits = teacher_outputs['bongard_logits']
            else:
                logger.error(f"Unsupported single teacher_model type: {type(teacher_model)}. Cannot get logits.")
                return torch.empty(0), None
            
            all_teacher_logits.append(teacher_logits)
            
            current_distillation_mask = torch.ones(batch_size_actual, dtype=torch.float, device=DEVICE)
            if distillation_config.get('use_mask_distillation', False):
                teacher_probs = F.softmax(teacher_logits / distillation_config['temperature'], dim=-1)
                max_teacher_probs, _ = torch.max(teacher_probs, dim=-1)
                current_distillation_mask = (max_teacher_probs > distillation_config.get('mask_threshold', 0.8)).float()
            all_distillation_masks.append(current_distillation_mask)
    else: # If teacher_models is a list of models (for ensemble averaging)
        for teacher_model_member in teacher_models:
            # Ensure it's a PerceptionModule or LitBongard
            if isinstance(teacher_model_member, LitBongard):
                core_model = teacher_model_member.perception_module
            elif isinstance(teacher_model_member, PerceptionModule):
                core_model = teacher_model_member
            else:
                logger.warning(f"Teacher model member type {type(teacher_model_member)} not recognized. Skipping.")
                continue
            core_model.eval()
            with torch.no_grad():
                teacher_outputs = core_model(
                    processed_images,
                    ground_truth_json_strings=raw_gt_json_strings, # Pass GT for scene graph building
                    detected_bboxes_batch=detected_bboxes_batch,
                    detected_masks_batch=detected_masks_batch,
                    support_images=processed_support_images_reshaped,
                    support_labels_flat=dummy_support_labels_flat
                )
                teacher_logits = teacher_outputs['bongard_logits']
                
                current_distillation_mask = torch.ones(batch_size_actual, dtype=torch.float, device=DEVICE)
                if distillation_config.get('use_mask_distillation', False):
                    teacher_probs = F.softmax(teacher_logits / distillation_config['temperature'], dim=-1)
                    max_teacher_probs, _ = torch.max(teacher_probs, dim=-1)
                    current_distillation_mask = (max_teacher_probs > distillation_config.get('mask_threshold', 0.8)).float()
                
                all_teacher_logits.append(teacher_logits)
                all_distillation_masks.append(current_distillation_mask)
    
    if all_teacher_logits:
        ensemble_logits = torch.stack(all_teacher_logits, dim=0).mean(dim=0) # Average across teachers
        final_distillation_mask = torch.stack(all_distillation_masks, dim=0).prod(dim=0) # Product of masks
        return ensemble_logits, final_distillation_mask
    return torch.empty(0), None

# Helper for IoU calculation (moved from LitBongard for general use)
def _calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates Intersection over Union (IoU) of two bounding boxes.
    Boxes are in (x1, y1, x2, y2) format.
    """
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the area of union
    union_area = float(box1_area + box2_area - inter_area)

    # Compute the IoU
    iou = inter_area / (union_area + 1e-6) # Add epsilon for numerical stability
    return iou

# Helper for edge index map (moved from LitBongard for general use)
def make_edge_index_map(num_objects: int) -> Dict[Tuple[int, int], int]:
    """
    Creates a mapping from (subject_id, object_id) to a linear edge index
    for a fully connected graph of `num_objects`.
    Excludes self-loops.
    """
    edge_map = {}
    idx_counter = 0
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                edge_map[(i, j)] = idx_counter
                idx_counter += 1
    return edge_map

# --- Main Training Loop for Ensemble Members and Student ---
def train_perception_with_buffer(
    cfg: Dict[str, Any],
    current_rank: int = 0, # Default to 0 for single GPU
    is_ddp_initialized: bool = False # Default to False for single GPU
) -> Tuple[str, List[float], List[int], Dict[str, Any]]:
    """
    Runs a single training session for the main perception model,
    integrating data loading, curriculum learning, and domain adaptation.
    This function effectively replaces `_run_single_training_session_ensemble`
    when training a single model, handling its own data loaders.
    Args:
        cfg (Dict[str, Any]): The configuration dictionary.
        current_rank (int): Current GPU rank (0 for single GPU).
        is_ddp_initialized (bool): True if DDP is already initialized.
    Returns:
        Tuple[str, List[float], List[int], Dict[str, Any]]:
            - Path to the best saved model checkpoint.
            - List of validation logits (flattened).
            - List of true validation labels (flattened).
            - Dictionary of best validation metrics.
    """
    set_seed(cfg['training']['seed']) # Use a different seed for each member

    # Initialize data module and loaders
    dm = BongardDataModule(cfg)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Initialize model
    model = LitBongard(cfg).to(DEVICE)
    if is_ddp_initialized:
        model = DDP(model, device_ids=[current_rank], output_device=current_rank, find_unused_parameters=False)

    # Initialize optimizer and scheduler using centralized functions
    optimizer = get_optimizer(model.parameters(), cfg['training'])

    # total_steps is needed for OneCycleLR
    num_epochs = cfg['training']['epochs']
    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, cfg['training'], total_steps)

    scaler = GradScaler() if cfg['training']['use_amp'] else None

    # Checkpointing setup
    from core_models.training_args import Config
    config = Config()
    ensure_dir(config.checkpoint_dir)
    last_ckpt_path = config.checkpoint_path
    best_ckpt_path = config.best_model_path
    best_acc = 0.0

    # ...existing early stopping setup...
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stop_patience'],
        delta=cfg['training']['early_stop_delta'],
        verbose=True,
        path=best_ckpt_path
    )
    for epoch in range(num_epochs):
        # ...existing training logic...
        avg_loss, avg_acc = 0.0, 0.0
        steps = len(train_loader)
        for _ in range(steps):
            # ...backward, step, accumulate avg_loss, avg_acc...
            pass
        print(f"[Epoch {epoch}] Train Loss={avg_loss/steps:.4f} Acc={avg_acc/steps:.4f}")

        # 2) validate on synthetic hold-out
        # val_acc, _, _ = validate_on_synthetic_holdout(model)  # Implement as needed
        val_acc = 0.0  # Placeholder
        print(f"[Epoch {epoch}] Val (Synth) Acc={val_acc:.4f}")

        # 3) checkpointing
        torch.save(model.state_dict(), last_ckpt_path)  # last
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)  # best
    logger.info(f"Early Stopping initialized with patience={early_stopping.patience}, delta={early_stopping.delta}.")
    
    num_bongard_classes = cfg['model']['bongard_head_config']['num_classes']
    mixup_cutmix_augmenter = MixupCutmixAugmenter(cfg['training'], num_bongard_classes)
    
    # Mean Teacher (EMA) model
    ema_model = None
    if cfg['training'].get('use_mean_teacher', False):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info("Mean Teacher (EMA) model initialized.")
    
    # GradNorm
    grad_norm_instance = None
    if cfg['training'].get('use_grad_norm', False):
        initial_task_weights = {
            'bongard_loss': 1.0,
            'attribute_loss': cfg['training'].get('attribute_loss_weight', 1.0),
            'relation_loss': cfg['training'].get('relation_loss_weight', 1.0),
            'consistency_loss': cfg['training'].get('consistency_loss_weight', 1.0),
        }
        initial_task_weights = {k: v for k, v in initial_task_weights.items() if v > 0}
        if initial_task_weights:
            grad_norm_instance = GradNorm(initial_task_weights, cfg['training'].get('grad_norm_alpha', 1.5))
            grad_norm_instance.to(DEVICE)
            logger.info("GradNorm initialized.")
        else:
            logger.warning("No active loss components for GradNorm. GradNorm will not be used.")
    
    # SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    if cfg['training'].get('use_swa', False):
        swa_config = cfg['training'].get('swa_config', {})
        swa_start_epoch = swa_config.get('swa_start_epoch', num_epochs // 2)
        if num_epochs > swa_start_epoch:
            swa_model = swa_utils.AveragedModel(model)
            swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=swa_config.get('swa_lr', 0.05))
            logger.info(f"SWA enabled starting at epoch {swa_start_epoch}.")
        else:
            logger.warning(f"SWA start epoch {swa_start_epoch} is not less than total epochs {num_epochs}. SWA disabled.")
    best_val_accuracy = 0.0
    
    # Domain Adaptation: Style Discriminator
    style_discriminator = None
    optimizer_d = None
    bce_loss = None
    if HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
        feat_dim = cfg['model'].get('feat_dim', 576) # Default to MobileNetV3 small output
        style_discriminator = StyleDiscriminator(feat_dim=feat_dim).to(DEVICE)
        optimizer_d = torch.optim.Adam(style_discriminator.parameters(), lr=cfg['training'].get('lr_disc', 1e-4))
        bce_loss = nn.BCELoss()
        logger.info(f"Style Discriminator initialized with feat_dim={feat_dim}, lr_disc={cfg['training'].get('lr_disc', 1e-4)}.")

    # Compile model if enabled (requires PyTorch 2.0+)
    if cfg['training'].get('use_torch_compile', False):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}. Continuing without compilation.")

    # Profiler setup
    profiler_enabled = cfg['debug'].get('enable_profiler', False)
    if profiler_enabled:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=cfg['debug'].get('profiler_schedule_wait', 1),
                warmup=cfg['debug'].get('profiler_schedule_warmup', 1),
                active=cfg['debug'].get('profiler_schedule_active', 3),
                repeat=cfg['debug'].get('profiler_schedule_repeat', 1)
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(cfg['debug']['logs_dir'], "profiler/main_model")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
    else:
        prof = None

    all_val_logits = []
    all_val_labels = []

    for epoch in range(num_epochs):
        if profiler_enabled:
            prof.step()
        
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        if style_discriminator:
            style_discriminator.train()

        total_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_loop):
            (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
             query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
             raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
             tree_indices, is_weights,
             query_bboxes_view1, query_masks_view1,
             query_bboxes_view2, query_masks_view2,
             support_bboxes_flat, support_masks_flat,
             query_conf1, query_conf2, support_confs_flat
            ) = batch
            
            query_labels = query_labels.to(DEVICE).long()
            support_labels_flat = support_labels_flat.to(DEVICE).long()
            query_conf1 = query_conf1.to(DEVICE).float()
            
            dali_processor = getattr(train_loader.dataset, 'dali_image_processor', None)
            
            if dali_processor is None or not cfg['data'].get('use_dali', False):
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize(tuple(cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(DEVICE)
                
                max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual = processed_query_images_view1.shape[0]
                
                if raw_support_images_flat_np and len(raw_support_images_flat_np) > 0:
                    if isinstance(raw_support_images_flat_np[0], list):
                        flat_support_images_np = [item for sublist in raw_support_images_flat_np for item in sublist]
                    else:
                        flat_support_images_np = raw_support_images_flat_np
                    actual_support_images_np = [img for img in flat_support_images_np if np.sum(img) > 0]
                    if actual_support_images_np:
                        processed_support_images_flat = torch.stack([transform(img_np) for img_np in actual_support_images_np]).to(DEVICE)
                        processed_support_images_reshaped = processed_support_images_flat.view(
                            batch_size_actual, max_support_imgs, 
                            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
                        )
                    else:
                        processed_support_images_reshaped = torch.zeros(
                            batch_size_actual, max_support_imgs, 
                            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3],
                            device=DEVICE
                        )
                else:
                    processed_support_images_reshaped = torch.zeros(
                        batch_size_actual, max_support_imgs, 
                        processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3],
                        device=DEVICE
                    )
            else:
                processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_processor.run(
                    raw_query_images_view1_np,
                    raw_query_images_view2_np,
                    raw_support_images_flat_np
                )
                max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual = processed_query_images_view1.shape[0]
                processed_support_images_reshaped = processed_support_images_flat.view(
                    batch_size_actual, max_support_imgs, 
                    processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
                )
            support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
            
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels)
            
            optimizer.zero_grad()
            if style_discriminator:
                optimizer_d.zero_grad()

            with record_function("forward_backward_pass"):
                if isinstance(optimizer, SAM) and cfg['training'].get('use_sam_optimizer', False):
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs_sam_first = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                                detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                        bongard_logits_sam_first = outputs_sam_first['bongard_logits']
                        
                        per_sample_bongard_losses_sam_first = model.bongard_criterion(bongard_logits_sam_first, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard_sam_first = (per_sample_bongard_losses_sam_first * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses_sam_first.mean()
                        
                    if cfg['training']['use_amp']:
                        scaler.scale(loss_bongard_sam_first).backward()
                        optimizer.first_step(zero_grad=True)
                    else:
                        loss_bongard_sam_first.backward()
                        optimizer.first_step(zero_grad=True)
                    
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        attribute_logits = outputs['attribute_logits']
                        relation_logits = outputs['relation_logits']
                        attribute_features = outputs['attribute_features']
                        scene_graphs = outputs['scene_graphs']
                        
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard = (per_sample_bongard_losses * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses.mean()
                        
                        total_batch_loss = loss_bongard
                        
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        num_attribute_losses = 0
                        current_flat_idx = 0
                        for i_img in range(len(scene_graphs)):
                            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
                            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                                gt_obj = None
                                inferred_bbox = inferred_obj.get('bbox_xyxy')
                                if inferred_bbox:
                                    for gt_o in sg_gt['objects']:
                                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                            gt_obj = gt_o
                                            break
                                if gt_obj:
                                    for attr_name in cfg['model']['attribute_classifier_config'].keys():
                                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                                            continue
                                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                                            if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                                    predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                                    loss_attribute += model.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                                    num_attribute_losses += 1
                                current_flat_idx += 1
                        if num_attribute_losses > 0:
                            loss_attribute /= num_attribute_losses
                            total_batch_loss += loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0)
                        
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        if not model.perception_module.use_scene_gnn and relation_logits.numel() > 0:
                            all_gt_edge_labels_flat = []
                            for b in range(batch_size_actual):
                                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                                num_gt_objects = len(sg_gt['objects'])
                                if num_gt_objects > 1:
                                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                                    for rel in sg_gt['relations']:
                                        subj_id = int(rel['subject_id'].split('_')[1])
                                        obj_id = int(rel['object_id'].split('_')[1])
                                        rel_type = rel['type']
                                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                                    all_gt_edge_labels_flat.append(temp_gt_labels)
                            if all_gt_edge_labels_flat:
                                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                                loss_relation = model.relation_criterion(relation_logits, labels_flat)
                                total_batch_loss += loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                        
                        loss_consistency = torch.tensor(0.0, device=DEVICE)
                        if cfg['training']['consistency_loss_weight'] > 0:
                            outputs_view2 = model(images_view2_aug, ground_truth_json_strings=query_gts_json_view2,
                                                detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                            if cfg['training']['feature_consistency_weight'] > 0 and attribute_features.numel() > 0 and outputs_view2['attribute_features'].numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(attribute_features, outputs_view2['attribute_features'])
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=scene_graphs,
                                    scene_graphs2=outputs_view2['scene_graphs'],
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency
                        
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        # Distillation loss is only for student model, not for main perception training
                        
                        loss_domain_adaptation = torch.tensor(0.0, device=DEVICE)
                        if style_discriminator and HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
                            features_for_discriminator = attribute_features
                            if features_for_discriminator.numel() > 0:
                                synthetic_labels_disc = torch.zeros(features_for_discriminator.shape[0], 1).to(DEVICE)
                                
                                disc_preds_synthetic = style_discriminator(features_for_discriminator.detach())
                                disc_loss = bce_loss(disc_preds_synthetic, synthetic_labels_disc)
                                
                                if cfg['training']['use_amp']:
                                    scaler.scale(disc_loss).backward()
                                    scaler.unscale_(optimizer_d)
                                    scaler.step(optimizer_d)
                                else:
                                    disc_loss.backward()
                                    optimizer_d.step()
                                
                                real_labels_disc = torch.ones(features_for_discriminator.shape[0], 1).to(DEVICE)
                                disc_preds_adversarial = style_discriminator(features_for_discriminator, alpha=cfg['training'].get('grl_alpha', 1.0))
                                loss_domain_adaptation = bce_loss(disc_preds_adversarial, real_labels_disc)
                                total_batch_loss += loss_domain_adaptation * cfg['training'].get('lambda_style', 1.0)
                        
                        current_loss = total_batch_loss.item()
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        total_batch_loss.backward()
                    
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else: # Standard optimizer path (not SAM)
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        attribute_logits = outputs['attribute_logits']
                        relation_logits = outputs['relation_logits']
                        attribute_features = outputs['attribute_features']
                        scene_graphs = outputs['scene_graphs']
                        
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard = (per_sample_bongard_losses * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses.mean()
                        
                        total_batch_loss = loss_bongard
                        
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        num_attribute_losses = 0
                        current_flat_idx = 0
                        for i_img in range(len(scene_graphs)):
                            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
                            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                                gt_obj = None
                                inferred_bbox = inferred_obj.get('bbox_xyxy')
                                if inferred_bbox:
                                    for gt_o in sg_gt['objects']:
                                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                            gt_obj = gt_o
                                            break
                                if gt_obj:
                                    for attr_name in cfg['model']['attribute_classifier_config'].keys():
                                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                                            continue
                                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                                            if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                                    predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                                    loss_attribute += model.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                                    num_attribute_losses += 1
                                current_flat_idx += 1
                        if num_attribute_losses > 0:
                            loss_attribute /= num_attribute_losses
                            total_batch_loss += loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0)
                        
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        if not model.perception_module.use_scene_gnn and relation_logits.numel() > 0:
                            all_gt_edge_labels_flat = []
                            for b in range(batch_size_actual):
                                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                                num_gt_objects = len(sg_gt['objects'])
                                if num_gt_objects > 1:
                                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                                    for rel in sg_gt['relations']:
                                        subj_id = int(rel['subject_id'].split('_')[1])
                                        obj_id = int(rel['object_id'].split('_')[1])
                                        rel_type = rel['type']
                                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                                    all_gt_edge_labels_flat.append(temp_gt_labels)
                            if all_gt_edge_labels_flat:
                                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                                loss_relation = model.relation_criterion(relation_logits, labels_flat)
                                total_batch_loss += loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                        
                        loss_consistency = torch.tensor(0.0, device=DEVICE)
                        if cfg['training']['consistency_loss_weight'] > 0:
                            outputs_view2 = model(images_view2_aug, ground_truth_json_strings=query_gts_json_view2,
                                                detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                            if cfg['training']['feature_consistency_weight'] > 0 and attribute_features.numel() > 0 and outputs_view2['attribute_features'].numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(attribute_features, outputs_view2['attribute_features'])
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=scene_graphs,
                                    scene_graphs2=outputs_view2['scene_graphs'],
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency
                        
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        # Distillation loss is only for student model, not for main perception training
                        
                        loss_domain_adaptation = torch.tensor(0.0, device=DEVICE)
                        if style_discriminator and HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
                            features_for_discriminator = attribute_features
                            if features_for_discriminator.numel() > 0:
                                synthetic_labels_disc = torch.zeros(features_for_discriminator.shape[0], 1).to(DEVICE)
                                
                                disc_preds_synthetic = style_discriminator(features_for_discriminator.detach())
                                disc_loss = bce_loss(disc_preds_synthetic, synthetic_labels_disc)
                                
                                if cfg['training']['use_amp']:
                                    scaler.scale(disc_loss).backward()
                                    scaler.unscale_(optimizer_d)
                                    scaler.step(optimizer_d)
                                else:
                                    disc_loss.backward()
                                    optimizer_d.step()
                                
                                real_labels_disc = torch.ones(features_for_discriminator.shape[0], 1).to(DEVICE)
                                disc_preds_adversarial = style_discriminator(features_for_discriminator, alpha=cfg['training'].get('grl_alpha', 1.0))
                                loss_domain_adaptation = bce_loss(disc_preds_adversarial, real_labels_disc)
                                total_batch_loss += loss_domain_adaptation * cfg['training'].get('lambda_style', 1.0)
                        
                        current_loss = total_batch_loss.item()
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        total_batch_loss.backward()
                    
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
            
            total_train_loss += current_loss
            
            if scheduler is not None and (isinstance(scheduler, OneCycleLR) or (HAS_GRADUAL_WARMUP and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
                scheduler.step()
            
            if ema_model and cfg['training'].get('use_mean_teacher', False):
                ema_decay = cfg['training']['mean_teacher_config'].get('alpha', 0.99)
                for student_param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            
            if grad_norm_instance:
                losses_for_gradnorm = {
                    'bongard_loss': loss_bongard.item(),
                    'attribute_loss': loss_attribute.item(),
                    'relation_loss': loss_relation.item(),
                    'consistency_loss': loss_consistency.item(),
                    'domain_adaptation_loss': loss_domain_adaptation.item() if style_discriminator else 0.0
                }
                losses_for_gradnorm = {k: v for k, v in losses_for_gradnorm.items() if v != 0.0}
                if losses_for_gradnorm:
                    grad_norm_instance.update_weights(losses_for_gradnorm, list(model.parameters()))
            
            train_loop.set_postfix(loss=current_loss)
            
            predictions = torch.argmax(bongard_logits, dim=1)
            train_correct_predictions += (predictions == query_labels).sum().item()
            train_total_samples += query_labels.size(0)
            
            if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and train_loader.dataset.sampler is not None:
                per_sample_losses = model.bongard_criterion(bongard_logits, query_labels, reduction='none', confidences=query_conf1)
                # The sampler in BongardDataModule handles the replay buffer directly
                # We need to pass the per_sample_losses and original_indices to the sampler
                # Assuming the sampler has an update_priorities method
                if hasattr(train_loader.dataset.sampler, 'update_priorities'):
                    train_loader.dataset.sampler.update_priorities(original_indices.cpu().tolist(), per_sample_losses.cpu().tolist())
                else:
                    logger.warning("Sampler does not have update_priorities method. Skipping PER priority update.")
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        if swa_model and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.debug(f"SWA model updated at epoch {epoch+1}.")
        
        val_model = swa_model.module if (swa_model and epoch >= swa_start_epoch) else model
        val_model.eval()
        if style_discriminator:
            style_discriminator.eval()
        total_val_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0
        
        current_epoch_val_logits = []
        current_epoch_val_labels = []

        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(val_loop):
                (raw_query_images_view1_np, _, query_labels_val,
                 query_gts_json_view1_val, _, _, _, _, _,
                 raw_support_images_flat_np_val, support_labels_flat_val, _, _, _, _,
                 query_bboxes_view1_val, query_masks_view1_val,
                 _, _,
                 support_bboxes_flat_val, support_masks_flat_val,
                 query_conf1_val, query_conf2_val, support_confs_flat_val
                ) = batch_val
                
                query_labels_val = query_labels_val.to(DEVICE).long()
                support_labels_flat_val = support_labels_flat_val.to(DEVICE).long()
                query_conf1_val = query_conf1_val.to(DEVICE).float()

                if dali_processor is None or not cfg['data'].get('use_dali', False):
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize(tuple(cfg['data']['image_size'])),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    processed_query_images_view1_val = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                    
                    max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    batch_size_actual_val = processed_query_images_view1_val.shape[0]
                    if raw_support_images_flat_np_val and len(raw_support_images_flat_np_val) > 0:
                        if isinstance(raw_support_images_flat_np_val[0], list):
                            flat_support_images_np_val = [item for sublist in raw_support_images_flat_np_val for item in sublist]
                        else:
                            flat_support_images_np_val = raw_support_images_flat_np_val
                        actual_support_images_np_val = [img for img in flat_support_images_np_val if np.sum(img) > 0]
                        if actual_support_images_np_val:
                            processed_support_images_flat_val = torch.stack([transform(img_np) for img_np in actual_support_images_np_val]).to(DEVICE)
                            processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                                batch_size_actual_val, max_support_imgs_val, 
                                processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                            )
                        else:
                            processed_support_images_reshaped_val = torch.zeros(
                                batch_size_actual_val, max_support_imgs_val, 
                                processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3],
                                device=DEVICE
                            )
                    else:
                        processed_support_images_reshaped_val = torch.zeros(
                            batch_size_actual_val, max_support_imgs_val, 
                            processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3],
                            device=DEVICE
                        )
                else:
                    processed_query_images_view1_val, _, processed_support_images_flat_val = dali_processor.run(
                        raw_query_images_view1_np,
                        [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np),
                        raw_support_images_flat_np_val
                    )
                    max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    batch_size_actual_val = processed_query_images_view1_val.shape[0]
                    processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                        batch_size_actual_val, max_support_imgs_val, 
                        processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                    )
                support_labels_reshaped_val = support_labels_flat_val.view(batch_size_actual_val, max_support_imgs_val)

                outputs_val = val_model(processed_query_images_view1_val, ground_truth_json_strings=query_gts_json_view1_val,
                                                detected_bboxes_batch=query_bboxes_view1_val, detected_masks_batch=query_masks_view1_val,
                                                support_images=processed_support_images_reshaped_val,
                                                support_labels_flat=support_labels_reshaped_val)
                bongard_logits_val = outputs_val['bongard_logits']
                
                loss_val = model.bongard_criterion(bongard_logits_val, query_labels_val, confidences=query_conf1_val).mean()
                total_val_loss += loss_val.item()
                predictions_val = torch.argmax(bongard_logits_val, dim=1)
                val_correct_predictions += (predictions_val == query_labels_val).sum().item()
                val_total_samples += query_labels_val.size(0)
                val_loop.set_postfix(loss=loss_val.item())
                current_epoch_val_logits.extend(bongard_logits_val.cpu().numpy().tolist())
                current_epoch_val_labels.extend(query_labels_val.cpu().numpy().tolist())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_samples
        logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        if scheduler is not None and not (isinstance(scheduler, OneCycleLR) or (HAS_GRADUAL_WARMUP and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
    
    if profiler_enabled:
        prof.stop()
        logger.info("PyTorch Profiler trace saved.")
    
    if swa_model:
        logger.info("Updating BatchNorm for SWA model.")
        pass # Placeholder for update_bn if it's not defined
        swa_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_swa_model.pt")
        torch.save(swa_model.state_dict(), swa_model_path)
        logger.info(f"SWA model saved to: {swa_model_path}")
    
    if os.path.exists(best_model_path):
        final_model_for_quant = LitBongard(cfg).to(DEVICE)
        loaded_state_dict = torch.load(best_model_path, map_location=DEVICE)
        if list(loaded_state_dict.keys())[0].startswith('module.'):
            loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        final_model_for_quant.perception_module.load_state_dict(loaded_state_dict)
        logger.info(f"Loaded best model from {best_model_path} for final evaluation/quantization.")
    else:
        logger.warning(f"No best model checkpoint found at {best_model_path} to load for final evaluation/quantization. Using current model state.")
        final_model_for_quant = model

    if cfg['training']['quantization'].get('qat', False):
        final_model_for_quant = quantize_model_qat(final_model_for_quant, cfg)
        optimized_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_qat_optimized_bongard_model.pth")
        torch.save(final_model_for_quant.state_dict(), optimized_model_path)
        logger.info(f"QAT optimized model saved to: {optimized_model_path}")

    if cfg['training']['quantization'].get('ptq', False):
        final_model_for_quant = quantize_model_ptq(final_model_for_quant, val_loader, cfg)
        optimized_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_ptq_optimized_bongard_model.pth")
        torch.save(final_model_for_quant.state_dict(), optimized_model_path)
        logger.info(f"PTQ optimized model saved to: {optimized_model_path}")
    
    logger.info("--- Main training session finished. ---")
    
    return best_model_path, current_epoch_val_logits, current_epoch_val_labels, {
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'logits': np.array(current_epoch_val_logits),
        'labels': np.array(current_epoch_val_labels)
    }

def train_ensemble(
    num_members: int,
    train_loader: Any,
    val_loader: Any,
    cfg: Dict[str, Any],
    current_rank: int,
    world_size: int,
    is_ddp_initialized: bool
) -> Tuple[List[LitBongard], List[Dict[str, Any]]]:
    """
    Trains an ensemble of `num_members` PerceptionModule models.
    Args:
        num_members (int): Number of ensemble members to train.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        cfg (Dict[str, Any]): Configuration dictionary.
        current_rank (int): Current GPU rank.
        world_size (int): Total number of GPUs.
        is_ddp_initialized (bool): True if DDP is already initialized.
    Returns:
        Tuple[List[LitBongard], List[Dict[str, Any]]]:
            - List of trained ensemble member models.
            - List of best validation metrics for each member.
    """
    logger.info(f"Starting training for an ensemble of {num_members} models.")
    trained_members = []
    all_members_best_metrics = []
    for i in range(num_members):
        logger.info(f"--- Training Ensemble Member {i+1}/{num_members} ---")
        model = LitBongard(cfg).to(DEVICE)
        if is_ddp_initialized:
            model = DDP(model, device_ids=[current_rank], output_device=current_rank, find_unused_parameters=False)
        
        member_best_path, member_val_logits, member_val_labels, member_metrics = \
            _run_single_training_session_ensemble_member( # Renamed to avoid confusion with main training function
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                current_rank=current_rank,
                is_ddp_initialized=is_ddp_initialized,
                model_idx=i, # Pass member index
                cfg=cfg
            )
        
        loaded_model = LitBongard(cfg).to(DEVICE)
        loaded_state_dict = torch.load(member_best_path, map_location=DEVICE)
        if list(loaded_state_dict.keys())[0].startswith('module.'):
            loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        loaded_model.perception_module.load_state_dict(loaded_state_dict)
        loaded_model.eval()
        trained_members.append(loaded_model)
        all_members_best_metrics.append(member_metrics)
        logger.info(f"Ensemble Member {i+1} training completed. Best accuracy: {member_metrics.get('val_accuracy', 0.0):.4f}")
    
    logger.info("All ensemble members trained.")
    return trained_members, all_members_best_metrics

def _run_single_training_session_ensemble_member( # Renamed for clarity
    model: LitBongard,
    train_loader: Any,
    val_loader: Any,
    current_rank: int,
    is_ddp_initialized: bool,
    model_idx: int, # For naming checkpoints (e.g., member_0, member_1, student)
    cfg: Dict[str, Any],
    replay_buffer: Optional[KnowledgeReplayBuffer] = None, # Pass replay buffer directly
    teacher_model: Optional[Union[nn.ModuleList, nn.Module]] = None # For knowledge distillation
) -> Tuple[str, List[float], List[int], Dict[str, Any]]:
    """
    Runs a single training session for an ensemble member or student model.
    This function replaces the previous `run_training_once` for more granular control
    and integration with ensemble/distillation and domain adaptation.
    Args:
        model (LitBongard): The model to train (already moved to DEVICE).
        train_loader: DALI or PyTorch DataLoader for training.
        val_loader: DALI or PyTorch DataLoader for validation.
        current_rank (int): Current GPU rank (0 for single GPU).
        is_ddp_initialized (bool): True if DDP is already initialized.
        model_idx (int): Index of the model (e.g., 0, 1, ..., -1 for student).
        cfg (Dict[str, Any]): The configuration dictionary.
        replay_buffer (Optional[KnowledgeReplayBuffer]): The PER replay buffer instance.
        teacher_model (Optional[Union[nn.ModuleList, nn.Module]]): Teacher model(s) for knowledge distillation.
    Returns:
        Tuple[str, List[float], List[int], Dict[str, Any]]:
            - Path to the best saved model checkpoint.
            - List of validation logits (flattened).
            - List of true validation labels (flattened).
            - Dictionary of best validation metrics.
    """
    set_seed(cfg['training']['seed'] + model_idx) # Use a different seed for each member
    
    # Initialize optimizer and scheduler using centralized functions
    optimizer = get_optimizer(model.parameters(), cfg['training'])
    
    # total_steps is needed for OneCycleLR
    num_epochs = cfg['training']['epochs'] if model_idx != -1 else cfg['training']['distillation_config'].get('student_epochs', cfg['training']['epochs']) # Use student epochs if student
    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, cfg['training'], total_steps)
    
    scaler = GradScaler() if cfg['training']['use_amp'] else None
    
    # Define checkpoint path based on model_idx
    model_name_prefix = f"member_{model_idx}" if model_idx != -1 else "student"
    checkpoint_dir = cfg['debug']['save_model_checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_best_model.pt")
    
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stop_patience'],
        delta=cfg['training']['early_stop_delta'],
        verbose=True,
        path=best_model_path # Save to the specific model's path
    )
    logger.info(f"Early Stopping initialized for {model_name_prefix} with patience={early_stopping.patience}, delta={early_stopping.delta}.")
    
    num_bongard_classes = cfg['model']['bongard_head_config']['num_classes']
    mixup_cutmix_augmenter = MixupCutmixAugmenter(cfg['training'], num_bongard_classes)
    
    # Mean Teacher (EMA) model
    ema_model = None
    if cfg['training'].get('use_mean_teacher', False) and model_idx != -1: # Only for teacher members
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info(f"Mean Teacher (EMA) model initialized for {model_name_prefix}.")
    
    # GradNorm
    grad_norm_instance = None
    if cfg['training'].get('use_grad_norm', False):
        initial_task_weights = {
            'bongard_loss': 1.0,
            'attribute_loss': cfg['training'].get('attribute_loss_weight', 1.0),
            'relation_loss': cfg['training'].get('relation_loss_weight', 1.0),
            'consistency_loss': cfg['training'].get('consistency_loss_weight', 1.0),
            'distillation_loss': cfg['training']['distillation_config'].get('loss_weight', 1.0) if model_idx == -1 else 0.0 # Only for student
        }
        initial_task_weights = {k: v for k, v in initial_task_weights.items() if v > 0}
        if initial_task_weights:
            grad_norm_instance = GradNorm(initial_task_weights, cfg['training'].get('grad_norm_alpha', 1.5))
            grad_norm_instance.to(DEVICE)
            logger.info(f"GradNorm initialized for {model_name_prefix}.")
        else:
            logger.warning(f"No active loss components for GradNorm for {model_name_prefix}. GradNorm will not be used.")
    
    # SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    if cfg['training'].get('use_swa', False):
        # Ensure 'swa_config' exists in cfg['training']
        swa_config = cfg['training'].get('swa_config', {})
        swa_start_epoch = swa_config.get('swa_start_epoch', num_epochs // 2)
        if num_epochs > swa_start_epoch:
            swa_model = swa_utils.AveragedModel(model)
            swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=swa_config.get('swa_lr', 0.05))
            logger.info(f"SWA enabled for {model_name_prefix} starting at epoch {swa_start_epoch}.")
        else:
            logger.warning(f"SWA start epoch {swa_start_epoch} is not less than total epochs {num_epochs}. SWA disabled for {model_name_prefix}.")
    best_val_accuracy = 0.0
    
    # Domain Adaptation: Style Discriminator
    style_discriminator = None
    optimizer_d = None
    bce_loss = None
    if HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
        # Assuming feat_dim is available in model config (e.g., from backbone output)
        feat_dim = cfg['model'].get('feat_dim', 576) # Default to MobileNetV3 small output
        style_discriminator = StyleDiscriminator(feat_dim=feat_dim).to(DEVICE)
        optimizer_d = torch.optim.Adam(style_discriminator.parameters(), lr=cfg['training'].get('lr_disc', 1e-4))
        bce_loss = nn.BCELoss()
        logger.info(f"Style Discriminator initialized with feat_dim={feat_dim}, lr_disc={cfg['training'].get('lr_disc', 1e-4)}.")

    # Compile model if enabled (requires PyTorch 2.0+)
    if cfg['training'].get('use_torch_compile', False):
        try:
            model = torch.compile(model)
            logger.info(f"Model for {model_name_prefix} compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"Failed to compile model for {model_name_prefix}: {e}. Continuing without compilation.")

    # Profiler setup
    profiler_enabled = cfg['debug'].get('enable_profiler', False) # Use 'enable_profiler' from debug
    if profiler_enabled:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=cfg['debug'].get('profiler_schedule_wait', 1),
                warmup=cfg['debug'].get('profiler_schedule_warmup', 1),
                active=cfg['debug'].get('profiler_schedule_active', 3),
                repeat=cfg['debug'].get('profiler_schedule_repeat', 1)
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(cfg['debug']['logs_dir'], f"profiler/{model_name_prefix}")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
    else:
        prof = None

    all_val_logits = []
    all_val_labels = []

    for epoch in range(num_epochs):
        if profiler_enabled:
            prof.step() # Step profiler at the start of each epoch
        
        # Set train loader epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        if style_discriminator:
            style_discriminator.train() # Set discriminator to train mode

        total_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train {model_name_prefix}]")
        for batch_idx, batch in enumerate(train_loop):
            # Unpack batch data (now includes confidences)
            (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
             query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
             raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
             tree_indices, is_weights,
             query_bboxes_view1, query_masks_view1,
             query_bboxes_view2, query_masks_view2,
             support_bboxes_flat, support_masks_flat,
             query_conf1, query_conf2, support_confs_flat # Added confidences
            ) = batch
            
            query_labels = query_labels.to(DEVICE).long()
            support_labels_flat = support_labels_flat.to(DEVICE).long()
            query_conf1 = query_conf1.to(DEVICE).float() # Move confidences to device
            
            # DALI Image Processor or torchvision transforms
            dali_processor = getattr(train_loader.dataset, 'dali_image_processor', None) # Access from dataset if DALI is used
            
            if dali_processor is None or not cfg['data'].get('use_dali', False): # Use cfg['data']['use_dali']
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize(tuple(cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(DEVICE)
                
                # Handle support images for main model's forward pass
                max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual = processed_query_images_view1.shape[0]
                
                if raw_support_images_flat_np and len(raw_support_images_flat_np) > 0:
                    if isinstance(raw_support_images_flat_np[0], list): # If it's a list of lists
                        flat_support_images_np = [item for sublist in raw_support_images_flat_np for item in sublist]
                    else:
                        flat_support_images_np = raw_support_images_flat_np
                    actual_support_images_np = [img for img in flat_support_images_np if np.sum(img) > 0] # Filter out padding
                    if actual_support_images_np:
                        processed_support_images_flat = torch.stack([transform(img_np) for img_np in actual_support_images_np]).to(DEVICE)
                        processed_support_images_reshaped = processed_support_images_flat.view(
                            batch_size_actual, max_support_imgs, 
                            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
                        )
                    else:
                        processed_support_images_reshaped = torch.zeros(
                            batch_size_actual, max_support_imgs, 
                            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3],
                            device=DEVICE
                        )
                else:
                    processed_support_images_reshaped = torch.zeros(
                        batch_size_actual, max_support_imgs, 
                        processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3],
                        device=DEVICE
                    )
            else:
                processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_processor.run(
                    raw_query_images_view1_np,
                    raw_query_images_view2_np,
                    raw_support_images_flat_np
                )
                max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual = processed_query_images_view1.shape[0]
                processed_support_images_reshaped = processed_support_images_flat.view(
                    batch_size_actual, max_support_imgs, 
                    processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
                )
            support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
            
            # Apply Mixup/Cutmix
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels) # Labels for view2 not used for mixup
            
            # Zero gradients for both model and discriminator
            optimizer.zero_grad()
            if style_discriminator:
                optimizer_d.zero_grad()

            with record_function("forward_backward_pass"):
                # SAM Optimizer logic (first and second step)
                if isinstance(optimizer, SAM) and cfg['training'].get('use_sam_optimizer', False):
                    # First forward pass for SAM
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs_sam_first = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                                detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                        bongard_logits_sam_first = outputs_sam_first['bongard_logits']
                        
                        # Calculate primary loss for SAM's first step
                        # Pass query_conf1 to the criterion
                        per_sample_bongard_losses_sam_first = model.bongard_criterion(bongard_logits_sam_first, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard_sam_first = (per_sample_bongard_losses_sam_first * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses_sam_first.mean()
                        
                    if cfg['training']['use_amp']:
                        scaler.scale(loss_bongard_sam_first).backward()
                        optimizer.first_step(zero_grad=True)
                    else:
                        loss_bongard_sam_first.backward()
                        optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass for SAM
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, # Use same augmented images
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        attribute_logits = outputs['attribute_logits']
                        relation_logits = outputs['relation_logits']
                        attribute_features = outputs['attribute_features']
                        scene_graphs = outputs['scene_graphs']
                        
                        # Calculate full loss (similar to LitBongard's training_step)
                        # Pass query_conf1 to the criterion
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard = (per_sample_bongard_losses * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses.mean()
                        
                        total_batch_loss = loss_bongard
                        
                        # Attribute Loss
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        num_attribute_losses = 0
                        current_flat_idx = 0
                        for i_img in range(len(scene_graphs)):
                            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
                            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                                gt_obj = None
                                inferred_bbox = inferred_obj.get('bbox_xyxy')
                                if inferred_bbox:
                                    for gt_o in sg_gt['objects']:
                                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                            gt_obj = gt_o
                                            break
                                if gt_obj:
                                    for attr_name in cfg['model']['attribute_classifier_config'].keys():
                                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                                            continue
                                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                                            if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                                    predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                                    loss_attribute += model.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                                    num_attribute_losses += 1
                                current_flat_idx += 1
                        if num_attribute_losses > 0:
                            loss_attribute /= num_attribute_losses
                            total_batch_loss += loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0)
                        
                        # Relation Loss
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        if not model.perception_module.use_scene_gnn and relation_logits.numel() > 0:
                            all_gt_edge_labels_flat = []
                            for b in range(batch_size_actual):
                                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                                num_gt_objects = len(sg_gt['objects'])
                                if num_gt_objects > 1:
                                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                                    for rel in sg_gt['relations']:
                                        subj_id = int(rel['subject_id'].split('_')[1]) # Convert 'obj_0' to 0
                                        obj_id = int(rel['object_id'].split('_')[1])
                                        rel_type = rel['type']
                                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                                    all_gt_edge_labels_flat.append(temp_gt_labels)
                            if all_gt_edge_labels_flat:
                                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                                loss_relation = model.relation_criterion(relation_logits, labels_flat)
                                total_batch_loss += loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                        
                        # Consistency Loss
                        loss_consistency = torch.tensor(0.0, device=DEVICE)
                        if cfg['training']['consistency_loss_weight'] > 0:
                            # Need outputs for view2 to calculate consistency
                            outputs_view2 = model(images_view2_aug, ground_truth_json_strings=query_gts_json_view2,
                                                detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                            if cfg['training']['feature_consistency_weight'] > 0 and attribute_features.numel() > 0 and outputs_view2['attribute_features'].numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(attribute_features, outputs_view2['attribute_features'])
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=scene_graphs,
                                    scene_graphs2=outputs_view2['scene_graphs'],
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency
                        
                        # Distillation Loss
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        if model_idx == -1 and teacher_model is not None and cfg['training']['use_knowledge_distillation']: # Only for student model
                            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                                teacher_models=teacher_model, # Pass the teacher model(s)
                                raw_images_np=raw_query_images_view1_np, # Use view1 for teacher prediction
                                raw_gt_json_strings=query_gts_json_view1,
                                raw_support_images_np=raw_support_images_flat_np,
                                distillation_config=cfg['training']['distillation_config'],
                                dali_image_processor=dali_processor,
                                config=cfg, # Pass the full config
                                detected_bboxes_batch=query_bboxes_view1,
                                detected_masks_batch=query_masks_view1
                            )
                            if teacher_logits_batch.numel() > 0:
                                per_sample_soft_loss, per_sample_hard_loss = model.distillation_criterion(
                                    bongard_logits, teacher_logits_batch, query_labels
                                )
                                if distillation_mask is not None and cfg['training']['distillation_config']['use_mask_distillation']:
                                    masked_soft_loss = per_sample_soft_loss * distillation_mask.to(DEVICE)
                                    masked_hard_loss = per_sample_hard_loss * distillation_mask.to(DEVICE)
                                    loss_distillation = (cfg['training']['distillation_config']['alpha'] * masked_soft_loss + \
                                                         (1. - cfg['training']['distillation_config']['alpha']) * masked_hard_loss).mean()
                                else:
                                    loss_distillation = (cfg['training']['distillation_config']['alpha'] * per_sample_soft_loss + \
                                                         (1. - cfg['training']['distillation_config']['alpha']) * per_sample_hard_loss).mean()
                                total_batch_loss += loss_distillation * cfg['training']['distillation_config'].get('loss_weight', 1.0)
                        
                        # Domain Adaptation Loss
                        loss_domain_adaptation = torch.tensor(0.0, device=DEVICE)
                        if style_discriminator and HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
                            # Get features from the perception module (before the classification head)
                            # This assumes PerceptionModule.forward returns 'attribute_features' or similar global features
                            # Or, if it's a LitBongard, we need to access its internal perception_module's features
                            if isinstance(model, LitBongard):
                                # Access the features from the backbone/feature extractor
                                # This might require modifying PerceptionModule to expose these features
                                # For now, let's assume 'attribute_features' is the global representation
                                # or we can get it directly from the backbone.
                                # If the model is LitBongard, its `outputs` already contain `attribute_features`.
                                features_for_discriminator = attribute_features # Use attribute_features as the domain-agnostic feature
                            else: # If model is just PerceptionModule directly
                                features_for_discriminator = outputs['attribute_features'] # Assuming it exposes this
                            if features_for_discriminator.numel() > 0:
                                # Discriminator wants to classify synthetic as 0
                                synthetic_labels_disc = torch.zeros(features_for_discriminator.shape[0], 1).to(DEVICE)
                                
                                # Train discriminator to distinguish (maximize this loss)
                                # The GRL is applied inside the StyleDiscriminator
                                disc_preds_synthetic = style_discriminator(features_for_discriminator.detach()) # Detach to only train discriminator
                                disc_loss = bce_loss(disc_preds_synthetic, synthetic_labels_disc)
                                
                                # Update discriminator
                                if cfg['training']['use_amp']:
                                    scaler.scale(disc_loss).backward()
                                    scaler.unscale_(optimizer_d)
                                    scaler.step(optimizer_d)
                                else:
                                    disc_loss.backward()
                                    optimizer_d.step()
                                
                                # Train generator (main model) to fool discriminator (minimize this loss)
                                # Synthetic features should be classified as 1 (real) by discriminator
                                real_labels_disc = torch.ones(features_for_discriminator.shape[0], 1).to(DEVICE)
                                # Apply GRL: discriminator sees reversed gradients from this path
                                disc_preds_adversarial = style_discriminator(features_for_discriminator, alpha=cfg['training'].get('grl_alpha', 1.0))
                                loss_domain_adaptation = bce_loss(disc_preds_adversarial, real_labels_disc)
                                total_batch_loss += loss_domain_adaptation * cfg['training'].get('lambda_style', 1.0)
                        
                        current_loss = total_batch_loss.item() # Store final loss for logging
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer) # Unscale gradients before clipping
                    else:
                        total_batch_loss.backward()
                    
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                else: # Standard optimizer path (not SAM)
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        attribute_logits = outputs['attribute_logits']
                        relation_logits = outputs['relation_logits']
                        attribute_features = outputs['attribute_features']
                        scene_graphs = outputs['scene_graphs']
                        
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard = (per_sample_bongard_losses * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses.mean()
                        
                        total_batch_loss = loss_bongard
                        
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        num_attribute_losses = 0
                        current_flat_idx = 0
                        for i_img in range(len(scene_graphs)):
                            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
                            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                                gt_obj = None
                                inferred_bbox = inferred_obj.get('bbox_xyxy')
                                if inferred_bbox:
                                    for gt_o in sg_gt['objects']:
                                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                            gt_obj = gt_o
                                            break
                                if gt_obj:
                                    for attr_name in cfg['model']['attribute_classifier_config'].keys():
                                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                                            continue
                                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                                            if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                                    predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                                    loss_attribute += model.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                                    num_attribute_losses += 1
                                current_flat_idx += 1
                        if num_attribute_losses > 0:
                            loss_attribute /= num_attribute_losses
                            total_batch_loss += loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0)
                        
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        if not model.perception_module.use_scene_gnn and relation_logits.numel() > 0:
                            all_gt_edge_labels_flat = []
                            for b in range(batch_size_actual):
                                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                                num_gt_objects = len(sg_gt['objects'])
                                if num_gt_objects > 1:
                                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                                    for rel in sg_gt['relations']:
                                        subj_id = int(rel['subject_id'].split('_')[1])
                                        obj_id = int(rel['object_id'].split('_')[1])
                                        rel_type = rel['type']
                                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                                    all_gt_edge_labels_flat.append(temp_gt_labels)
                            if all_gt_edge_labels_flat:
                                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                                loss_relation = model.relation_criterion(relation_logits, labels_flat)
                                total_batch_loss += loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                        
                        loss_consistency = torch.tensor(0.0, device=DEVICE)
                        if cfg['training']['consistency_loss_weight'] > 0:
                            outputs_view2 = model(images_view2_aug, ground_truth_json_strings=query_gts_json_view2,
                                                detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                            if cfg['training']['feature_consistency_weight'] > 0 and attribute_features.numel() > 0 and outputs_view2['attribute_features'].numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(attribute_features, outputs_view2['attribute_features'])
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=scene_graphs,
                                    scene_graphs2=outputs_view2['scene_graphs'],
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency
                        
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        # Distillation loss is only for student model, not for main perception training
                        
                        loss_domain_adaptation = torch.tensor(0.0, device=DEVICE)
                        if style_discriminator and HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
                            features_for_discriminator = attribute_features
                            if features_for_discriminator.numel() > 0:
                                synthetic_labels_disc = torch.zeros(features_for_discriminator.shape[0], 1).to(DEVICE)
                                
                                disc_preds_synthetic = style_discriminator(features_for_discriminator.detach())
                                disc_loss = bce_loss(disc_preds_synthetic, synthetic_labels_disc)
                                
                                if cfg['training']['use_amp']:
                                    scaler.scale(disc_loss).backward()
                                    scaler.unscale_(optimizer_d)
                                    scaler.step(optimizer_d)
                                else:
                                    disc_loss.backward()
                                    optimizer_d.step()
                                
                                real_labels_disc = torch.ones(features_for_discriminator.shape[0], 1).to(DEVICE)
                                disc_preds_adversarial = style_discriminator(features_for_discriminator, alpha=cfg['training'].get('grl_alpha', 1.0))
                                loss_domain_adaptation = bce_loss(disc_preds_adversarial, real_labels_disc)
                                total_batch_loss += loss_domain_adaptation * cfg['training'].get('lambda_style', 1.0)
                        
                        current_loss = total_batch_loss.item()
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        total_batch_loss.backward()
                    
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
            
            total_train_loss += current_loss
            
            if scheduler is not None and (isinstance(scheduler, OneCycleLR) or (HAS_GRADUAL_WARMUP and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
                scheduler.step()
            
            if ema_model and cfg['training'].get('use_mean_teacher', False):
                ema_decay = cfg['training']['mean_teacher_config'].get('alpha', 0.99)
                for student_param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            
            if grad_norm_instance:
                losses_for_gradnorm = {
                    'bongard_loss': loss_bongard.item(),
                    'attribute_loss': loss_attribute.item(),
                    'relation_loss': loss_relation.item(),
                    'consistency_loss': loss_consistency.item(),
                    'domain_adaptation_loss': loss_domain_adaptation.item() if style_discriminator else 0.0
                }
                losses_for_gradnorm = {k: v for k, v in losses_for_gradnorm.items() if v != 0.0}
                if losses_for_gradnorm:
                    grad_norm_instance.update_weights(losses_for_gradnorm, list(model.parameters()))
            
            train_loop.set_postfix(loss=current_loss)
            
            predictions = torch.argmax(bongard_logits, dim=1)
            train_correct_predictions += (predictions == query_labels).sum().item()
            train_total_samples += query_labels.size(0)
            
            if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and train_loader.dataset.sampler is not None:
                per_sample_losses = model.bongard_criterion(bongard_logits, query_labels, reduction='none', confidences=query_conf1)
                if hasattr(train_loader.dataset.sampler, 'update_priorities'):
                    train_loader.dataset.sampler.update_priorities(original_indices.cpu().tolist(), per_sample_losses.cpu().tolist())
                else:
                    logger.warning("Sampler does not have update_priorities method. Skipping PER priority update.")
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        if swa_model and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.debug(f"SWA model updated at epoch {epoch+1}.")
        
        val_model = swa_model.module if (swa_model and epoch >= swa_start_epoch) else model
        val_model.eval()
        if style_discriminator:
            style_discriminator.eval()
        total_val_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0
        
        current_epoch_val_logits = []
        current_epoch_val_labels = []

        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(val_loop):
                (raw_query_images_view1_np, _, query_labels_val,
                 query_gts_json_view1_val, _, _, _, _, _,
                 raw_support_images_flat_np_val, support_labels_flat_val, _, _, _, _,
                 query_bboxes_view1_val, query_masks_view1_val,
                 _, _,
                 support_bboxes_flat_val, support_masks_flat_val,
                 query_conf1_val, query_conf2_val, support_confs_flat_val
                ) = batch_val
                
                query_labels_val = query_labels_val.to(DEVICE).long()
                support_labels_flat_val = support_labels_flat_val.to(DEVICE).long()
                query_conf1_val = query_conf1_val.to(DEVICE).float()

                if dali_processor is None or not cfg['data'].get('use_dali', False):
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize(tuple(cfg['data']['image_size'])),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    processed_query_images_view1_val = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                    
                    max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    batch_size_actual_val = processed_query_images_view1_val.shape[0]
                    if raw_support_images_flat_np_val and len(raw_support_images_flat_np_val) > 0:
                        if isinstance(raw_support_images_flat_np_val[0], list):
                            flat_support_images_np_val = [item for sublist in raw_support_images_flat_np_val for item in sublist]
                        else:
                            flat_support_images_np_val = raw_support_images_flat_np_val
                        actual_support_images_np_val = [img for img in flat_support_images_np_val if np.sum(img) > 0]
                        if actual_support_images_np_val:
                            processed_support_images_flat_val = torch.stack([transform(img_np) for img_np in actual_support_images_np_val]).to(DEVICE)
                            processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                                batch_size_actual_val, max_support_imgs_val, 
                                processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                            )
                        else:
                            processed_support_images_reshaped_val = torch.zeros(
                                batch_size_actual_val, max_support_imgs_val, 
                                processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3],
                                device=DEVICE
                            )
                    else:
                        processed_support_images_reshaped_val = torch.zeros(
                            batch_size_actual_val, max_support_imgs_val, 
                            processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3],
                            device=DEVICE
                        )
                else:
                    processed_query_images_view1_val, _, processed_support_images_flat_val = dali_processor.run(
                        raw_query_images_view1_np,
                        [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np),
                        raw_support_images_flat_np_val
                    )
                    max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    batch_size_actual_val = processed_query_images_view1_val.shape[0]
                    processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                        batch_size_actual_val, max_support_imgs_val, 
                        processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                    )
                support_labels_reshaped_val = support_labels_flat_val.view(batch_size_actual_val, max_support_imgs_val)

                outputs_val = val_model(processed_query_images_view1_val, ground_truth_json_strings=query_gts_json_view1_val,
                                                detected_bboxes_batch=query_bboxes_view1_val, detected_masks_batch=query_masks_view1_val,
                                                support_images=processed_support_images_reshaped_val,
                                                support_labels_flat=support_labels_reshaped_val)
                bongard_logits_val = outputs_val['bongard_logits']
                
                loss_val = model.bongard_criterion(bongard_logits_val, query_labels_val, confidences=query_conf1_val).mean()
                total_val_loss += loss_val.item()
                predictions_val = torch.argmax(bongard_logits_val, dim=1)
                val_correct_predictions += (predictions_val == query_labels_val).sum().item()
                val_total_samples += query_labels_val.size(0)
                val_loop.set_postfix(loss=loss_val.item())
                current_epoch_val_logits.extend(bongard_logits_val.cpu().numpy().tolist())
                current_epoch_val_labels.extend(query_labels_val.cpu().numpy().tolist())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_samples
        logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        if scheduler is not None and not (isinstance(scheduler, OneCycleLR) or (HAS_GRADUAL_WARMUP and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
    
    if profiler_enabled:
        prof.stop()
        logger.info("PyTorch Profiler trace saved.")
    
    if swa_model:
        logger.info("Updating BatchNorm for SWA model.")
        pass # Placeholder for update_bn if it's not defined
        swa_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_swa_model.pt")
        torch.save(swa_model.state_dict(), swa_model_path)
        logger.info(f"SWA model saved to: {swa_model_path}")
    
    if os.path.exists(best_model_path):
        final_model_for_quant = LitBongard(cfg).to(DEVICE)
        loaded_state_dict = torch.load(best_model_path, map_location=DEVICE)
        if list(loaded_state_dict.keys())[0].startswith('module.'):
            loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        final_model_for_quant.perception_module.load_state_dict(loaded_state_dict)
        logger.info(f"Loaded best model from {best_model_path} for final evaluation/quantization.")
    else:
        logger.warning(f"No best model checkpoint found at {best_model_path} to load for final evaluation/quantization. Using current model state.")
        final_model_for_quant = model

    if cfg['training']['quantization'].get('qat', False):
        final_model_for_quant = quantize_model_qat(final_model_for_quant, cfg)
        optimized_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_qat_optimized_bongard_model.pth")
        torch.save(final_model_for_quant.state_dict(), optimized_model_path)
        logger.info(f"QAT optimized model saved to: {optimized_model_path}")

    if cfg['training']['quantization'].get('ptq', False):
        final_model_for_quant = quantize_model_ptq(final_model_for_quant, val_loader, cfg)
        optimized_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_ptq_optimized_bongard_model.pth")
        torch.save(final_model_for_quant.state_dict(), optimized_model_path)
        logger.info(f"PTQ optimized model saved to: {optimized_model_path}")
    
    logger.info("--- Main training session finished. ---")
    
    return best_model_path, current_epoch_val_logits, current_epoch_val_labels, {
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'logits': np.array(current_epoch_val_logits),
        'labels': np.array(current_epoch_val_labels)
    }

def train_ensemble(
    num_members: int,
    train_loader: Any,
    val_loader: Any,
    cfg: Dict[str, Any],
    current_rank: int,
    world_size: int,
    is_ddp_initialized: bool
) -> Tuple[List[LitBongard], List[Dict[str, Any]]]:
    """
    Trains an ensemble of `num_members` PerceptionModule models.
    Args:
        num_members (int): Number of ensemble members to train.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        cfg (Dict[str, Any]): Configuration dictionary.
        current_rank (int): Current GPU rank.
        world_size (int): Total number of GPUs.
        is_ddp_initialized (bool): True if DDP is already initialized.
    Returns:
        Tuple[List[LitBongard], List[Dict[str, Any]]]:
            - List of trained ensemble member models.
            - List of best validation metrics for each member.
    """
    logger.info(f"Starting training for an ensemble of {num_members} models.")
    trained_members = []
    all_members_best_metrics = []
    for i in range(num_members):
        logger.info(f"--- Training Ensemble Member {i+1}/{num_members} ---")
        model = LitBongard(cfg).to(DEVICE)
        if is_ddp_initialized:
            model = DDP(model, device_ids=[current_rank], output_device=current_rank, find_unused_parameters=False)
        
        member_best_path, member_val_logits, member_val_labels, member_metrics = \
            _run_single_training_session_ensemble_member( # Renamed to avoid confusion with main training function
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                current_rank=current_rank,
                is_ddp_initialized=is_ddp_initialized,
                model_idx=i, # Pass member index
                cfg=cfg
            )
        
        # Load the best model state dict for the ensemble member
        loaded_model = LitBongard(cfg).to(DEVICE)
        loaded_state_dict = torch.load(member_best_path, map_location=DEVICE)
        if list(loaded_state_dict.keys())[0].startswith('module.'):
            loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        loaded_model.perception_module.load_state_dict(loaded_state_dict)
        loaded_model.eval()
        trained_members.append(loaded_model)
        all_members_best_metrics.append(member_metrics)
        logger.info(f"Ensemble Member {i+1} training completed. Best accuracy: {member_metrics.get('val_accuracy', 0.0):.4f}")
    
    logger.info("All ensemble members trained.")
    return trained_members, all_members_best_metrics

def _run_single_training_session_ensemble_member( # Renamed for clarity
    model: LitBongard,
    train_loader: Any,
    val_loader: Any,
    current_rank: int,
    is_ddp_initialized: bool,
    model_idx: int, # For naming checkpoints (e.g., member_0, member_1, student)
    cfg: Dict[str, Any],
    replay_buffer: Optional[KnowledgeReplayBuffer] = None, # Pass replay buffer directly
    teacher_model: Optional[Union[nn.ModuleList, nn.Module]] = None # For knowledge distillation
) -> Tuple[str, List[float], List[int], Dict[str, Any]]:
    """
    Runs a single training session for an ensemble member or student model.
    This function replaces the previous `run_training_once` for more granular control
    and integration with ensemble/distillation and domain adaptation.
    Args:
        model (LitBongard): The model to train (already moved to DEVICE).
        train_loader: DALI or PyTorch DataLoader for training.
        val_loader: DALI or PyTorch DataLoader for validation.
        current_rank (int): Current GPU rank (0 for single GPU).
        is_ddp_initialized (bool): True if DDP is already initialized.
        model_idx (int): Index of the model (e.g., 0, 1, ..., -1 for student).
        cfg (Dict[str, Any]): The configuration dictionary.
        replay_buffer (Optional[KnowledgeReplayBuffer]): The PER replay buffer instance.
        teacher_model (Optional[Union[nn.ModuleList, nn.Module]]): Teacher model(s) for knowledge distillation.
    Returns:
        Tuple[str, List[float], List[int], Dict[str, Any]]:
            - Path to the best saved model checkpoint.
            - List of validation logits (flattened).
            - List of true validation labels (flattened).
            - Dictionary of best validation metrics.
    """
    set_seed(cfg['training']['seed'] + model_idx) # Use a different seed for each member
    
    # Initialize optimizer and scheduler using centralized functions
    optimizer = get_optimizer(model.parameters(), cfg['training'])
    
    # total_steps is needed for OneCycleLR
    num_epochs = cfg['training']['epochs'] if model_idx != -1 else cfg['training']['distillation_config'].get('student_epochs', cfg['training']['epochs']) # Use student epochs if student
    total_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, cfg['training'], total_steps)
    
    scaler = GradScaler() if cfg['training']['use_amp'] else None
    
    # Define checkpoint path based on model_idx
    model_name_prefix = f"member_{model_idx}" if model_idx != -1 else "student"
    checkpoint_dir = cfg['debug']['save_model_checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_best_model.pt")
    
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stop_patience'],
        delta=cfg['training']['early_stop_delta'],
        verbose=True,
        path=best_model_path # Save to the specific model's path
    )
    logger.info(f"Early Stopping initialized for {model_name_prefix} with patience={early_stopping.patience}, delta={early_stopping.delta}.")
    
    num_bongard_classes = cfg['model']['bongard_head_config']['num_classes']
    mixup_cutmix_augmenter = MixupCutmixAugmenter(cfg['training'], num_bongard_classes)
    
    # Mean Teacher (EMA) model
    ema_model = None
    if cfg['training'].get('use_mean_teacher', False) and model_idx != -1: # Only for teacher members
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info(f"Mean Teacher (EMA) model initialized for {model_name_prefix}.")
    
    # GradNorm
    grad_norm_instance = None
    if cfg['training'].get('use_grad_norm', False):
        initial_task_weights = {
            'bongard_loss': 1.0,
            'attribute_loss': cfg['training'].get('attribute_loss_weight', 1.0),
            'relation_loss': cfg['training'].get('relation_loss_weight', 1.0),
            'consistency_loss': cfg['training'].get('consistency_loss_weight', 1.0),
            'distillation_loss': cfg['training']['distillation_config'].get('loss_weight', 1.0) if model_idx == -1 else 0.0 # Only for student
        }
        initial_task_weights = {k: v for k, v in initial_task_weights.items() if v > 0}
        if initial_task_weights:
            grad_norm_instance = GradNorm(initial_task_weights, cfg['training'].get('grad_norm_alpha', 1.5))
            grad_norm_instance.to(DEVICE)
            logger.info(f"GradNorm initialized for {model_name_prefix}.")
        else:
            logger.warning(f"No active loss components for GradNorm for {model_name_prefix}. GradNorm will not be used.")
    
    # SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    if cfg['training'].get('use_swa', False):
        # Ensure 'swa_config' exists in cfg['training']
        swa_config = cfg['training'].get('swa_config', {})
        swa_start_epoch = swa_config.get('swa_start_epoch', num_epochs // 2)
        if num_epochs > swa_start_epoch:
            swa_model = swa_utils.AveragedModel(model)
            swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=swa_config.get('swa_lr', 0.05))
            logger.info(f"SWA enabled for {model_name_prefix} starting at epoch {swa_start_epoch}.")
        else:
            logger.warning(f"SWA start epoch {swa_start_epoch} is not less than total epochs {num_epochs}. SWA disabled for {model_name_prefix}.")
    best_val_accuracy = 0.0
    
    # Domain Adaptation: Style Discriminator
    style_discriminator = None
    optimizer_d = None
    bce_loss = None
    if HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
        # Assuming feat_dim is available in model config (e.g., from backbone output)
        feat_dim = cfg['model'].get('feat_dim', 576) # Default to MobileNetV3 small output
        style_discriminator = StyleDiscriminator(feat_dim=feat_dim).to(DEVICE)
        optimizer_d = torch.optim.Adam(style_discriminator.parameters(), lr=cfg['training'].get('lr_disc', 1e-4))
        bce_loss = nn.BCELoss()
        logger.info(f"Style Discriminator initialized with feat_dim={feat_dim}, lr_disc={cfg['training'].get('lr_disc', 1e-4)}.")

    # Compile model if enabled (requires PyTorch 2.0+)
    if cfg['training'].get('use_torch_compile', False):
        try:
            model = torch.compile(model)
            logger.info(f"Model for {model_name_prefix} compiled with torch.compile.")
        except Exception as e:
            logger.warning(f"Failed to compile model for {model_name_prefix}: {e}. Continuing without compilation.")

    # Profiler setup
    profiler_enabled = cfg['debug'].get('enable_profiler', False) # Use 'enable_profiler' from debug
    if profiler_enabled:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=cfg['debug'].get('profiler_schedule_wait', 1),
                warmup=cfg['debug'].get('profiler_schedule_warmup', 1),
                active=cfg['debug'].get('profiler_schedule_active', 3),
                repeat=cfg['debug'].get('profiler_schedule_repeat', 1)
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(cfg['debug']['logs_dir'], f"profiler/{model_name_prefix}")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
    else:
        prof = None

    all_val_logits = []
    all_val_labels = []

    for epoch in range(num_epochs):
        if profiler_enabled:
            prof.step() # Step profiler at the start of each epoch
        
        # Set train loader epoch for DistributedSampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        if style_discriminator:
            style_discriminator.train() # Set discriminator to train mode

        total_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train {model_name_prefix}]")
        for batch_idx, batch in enumerate(train_loop):
            # Unpack batch data (now includes confidences)
            (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
             query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
             raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
             tree_indices, is_weights,
             query_bboxes_view1, query_masks_view1,
             query_bboxes_view2, query_masks_view2,
             support_bboxes_flat, support_masks_flat,
             query_conf1, query_conf2, support_confs_flat # Added confidences
            ) = batch
            
            query_labels = query_labels.to(DEVICE).long()
            support_labels_flat = support_labels_flat.to(DEVICE).long()
            query_conf1 = query_conf1.to(DEVICE).float() # Move confidences to device
            
            # DALI Image Processor or torchvision transforms
            dali_processor = getattr(train_loader.dataset, 'dali_image_processor', None) # Access from dataset if DALI is used
            
            if dali_processor is None or not cfg['data'].get('use_dali', False): # Use cfg['data']['use_dali']
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize(tuple(cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(DEVICE)
                
                # Handle support images for main model's forward pass
                max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual = processed_query_images_view1.shape[0]
                
                if raw_support_images_flat_np and len(raw_support_images_flat_np) > 0:
                    if isinstance(raw_support_images_flat_np[0], list): # If it's a list of lists
                        flat_support_images_np = [item for sublist in raw_support_images_flat_np for item in sublist]
                    else:
                        flat_support_images_np = raw_support_images_flat_np
                    actual_support_images_np = [img for img in flat_support_images_np if np.sum(img) > 0] # Filter out padding
                    if actual_support_images_np:
                        processed_support_images_flat = torch.stack([transform(img_np) for img_np in actual_support_images_np]).to(DEVICE)
                        processed_support_images_reshaped = processed_support_images_flat.view(
                            batch_size_actual, max_support_imgs, 
                            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
                        )
                    else:
                        processed_support_images_reshaped = torch.zeros(
                            batch_size_actual, max_support_imgs, 
                            processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3],
                            device=DEVICE
                        )
                else:
                    processed_support_images_reshaped = torch.zeros(
                        batch_size_actual, max_support_imgs, 
                        processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3],
                        device=DEVICE
                    )
            else:
                processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_processor.run(
                    raw_query_images_view1_np,
                    raw_query_images_view2_np,
                    raw_support_images_flat_np
                )
                max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual = processed_query_images_view1.shape[0]
                processed_support_images_reshaped = processed_support_images_flat.view(
                    batch_size_actual, max_support_imgs, 
                    processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
                )
            support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)
            
            # Apply Mixup/Cutmix
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
            images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels) # Labels for view2 not used for mixup
            
            # Zero gradients for both model and discriminator
            optimizer.zero_grad()
            if style_discriminator:
                optimizer_d.zero_grad()

            with record_function("forward_backward_pass"):
                # SAM Optimizer logic (first and second step)
                if isinstance(optimizer, SAM) and cfg['training'].get('use_sam_optimizer', False):
                    # First forward pass for SAM
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs_sam_first = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                                detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                        bongard_logits_sam_first = outputs_sam_first['bongard_logits']
                        
                        # Calculate primary loss for SAM's first step
                        # Pass query_conf1 to the criterion
                        per_sample_bongard_losses_sam_first = model.bongard_criterion(bongard_logits_sam_first, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard_sam_first = (per_sample_bongard_losses_sam_first * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses_sam_first.mean()
                        
                    if cfg['training']['use_amp']:
                        scaler.scale(loss_bongard_sam_first).backward()
                        optimizer.first_step(zero_grad=True)
                    else:
                        loss_bongard_sam_first.backward()
                        optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass for SAM
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, # Use same augmented images
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        attribute_logits = outputs['attribute_logits']
                        relation_logits = outputs['relation_logits']
                        attribute_features = outputs['attribute_features']
                        scene_graphs = outputs['scene_graphs']
                        
                        # Calculate full loss (similar to LitBongard's training_step)
                        # Pass query_conf1 to the criterion
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, query_labels, confidences=query_conf1, reduction='none')
                        loss_bongard = (per_sample_bongard_losses * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses.mean()
                        
                        total_batch_loss = loss_bongard
                        
                        # Attribute Loss
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        num_attribute_losses = 0
                        current_flat_idx = 0
                        for i_img in range(len(scene_graphs)):
                            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                            inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
                            for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                                gt_obj = None
                                inferred_bbox = inferred_obj.get('bbox_xyxy')
                                if inferred_bbox:
                                    for gt_o in sg_gt['objects']:
                                        if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                            gt_obj = gt_o
                                            break
                                if gt_obj:
                                    for attr_name in cfg['model']['attribute_classifier_config'].keys():
                                        if attr_name == 'mlp_dim' or attr_name == 'head_dropout_prob':
                                            continue
                                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                                            if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                                    gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                                    predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                                    loss_attribute += model.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                                    num_attribute_losses += 1
                                current_flat_idx += 1
                        if num_attribute_losses > 0:
                            loss_attribute /= num_attribute_losses
                            total_batch_loss += loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0)
                        
                        # Relation Loss
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        if not model.perception_module.use_scene_gnn and relation_logits.numel() > 0:
                            all_gt_edge_labels_flat = []
                            for b in range(batch_size_actual):
                                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                                num_gt_objects = len(sg_gt['objects'])
                                if num_gt_objects > 1:
                                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                                    for rel in sg_gt['relations']:
                                        subj_id = int(rel['subject_id'].split('_')[1]) # Convert 'obj_0' to 0
                                        obj_id = int(rel['object_id'].split('_')[1])
                                        rel_type = rel['type']
                                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                                    all_gt_edge_labels_flat.append(temp_gt_labels)
                            if all_gt_edge_labels_flat:
                                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                                loss_relation = model.relation_criterion(relation_logits, labels_flat)
                                total_batch_loss += loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                        
                        # Consistency Loss
                        loss_consistency = torch.tensor(0.0, device=DEVICE)
                        if cfg['training']['consistency_loss_weight'] > 0:
                            # Need outputs for view2 to calculate consistency
                            outputs_view2 = model(images_view2_aug, ground_truth_json_strings=query_gts_json_view2,
                                                detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                            if cfg['training']['feature_consistency_weight'] > 0 and attribute_features.numel() > 0 and outputs_view2['attribute_features'].numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(attribute_features, outputs_view2['attribute_features'])
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=scene_graphs,
                                    scene_graphs2=outputs_view2['scene_graphs'],
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency
                        
                        # Distillation Loss
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        if model_idx == -1 and teacher_model is not None and cfg['training']['use_knowledge_distillation']: # Only for student model
                            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                                teacher_models=teacher_model, # Pass the teacher model(s)
                                raw_images_np=raw_query_images_view1_np, # Use view1 for teacher prediction
                                raw_gt_json_strings=query_gts_json_view1,
                                raw_support_images_np=raw_support_images_flat_np,
                                distillation_config=cfg['training']['distillation_config'],
                                dali_image_processor=dali_processor,
                                config=cfg, # Pass the full config
                                detected_bboxes_batch=query_bboxes_view1,
                                detected_masks_batch=query_masks_view1
                            )
                            if teacher_logits_batch.numel() > 0:
                                per_sample_soft_loss, per_sample_hard_loss = model.distillation_criterion(
                                    bongard_logits, teacher_logits_batch, query_labels
                                )
                                if distillation_mask is not None and cfg['training']['distillation_config']['use_mask_distillation']:
                                    masked_soft_loss = per_sample_soft_loss * distillation_mask.to(DEVICE)
                                    masked_hard_loss = per_sample_hard_loss * distillation_mask.to(DEVICE)
                                    loss_distillation = (cfg['training']['distillation_config']['alpha'] * masked_soft_loss + \
                                                         (1. - cfg['training']['distillation_config']['alpha']) * masked_hard_loss).mean()
                                else:
                                    loss_distillation = (cfg['training']['distillation_config']['alpha'] * per_sample_soft_loss + \
                                                         (1. - cfg['training']['distillation_config']['alpha']) * per_sample_hard_loss).mean()
                                total_batch_loss += loss_distillation * cfg['training']['distillation_config'].get('loss_weight', 1.0)
                        
                        # Domain Adaptation Loss
                        loss_domain_adaptation = torch.tensor(0.0, device=DEVICE)
                        if style_discriminator and HAS_DOMAIN_ADAPTATION and cfg['training'].get('use_domain_adaptation', False):
                            # Get features from the perception module (before the classification head)
                            # This assumes PerceptionModule.forward returns 'attribute_features' or similar global features
                            # Or, if it's a LitBongard, we need to access its internal perception_module's features
                            if isinstance(model, LitBongard):
                                # Access the features from the backbone/feature extractor
                                # This might require modifying PerceptionModule to expose these features
                                # For now, let's assume 'attribute_features' is the global representation
                                # or we can get it directly from the backbone.
                                # If the model is LitBongard, its `outputs` already contain `attribute_features`.
                                features_for_discriminator = attribute_features # Use attribute_features as the domain-agnostic feature
                            else: # If model is just PerceptionModule directly
                                features_for_discriminator = outputs['attribute_features'] # Assuming it exposes this
                            if features_for_discriminator.numel() > 0:
                                # Discriminator wants to classify synthetic as 0
                                synthetic_labels_disc = torch.zeros(features_for_discriminator.shape[0], 1).to(DEVICE)
                                
                                # Train discriminator to distinguish (maximize this loss)
                                # The GRL is applied inside the StyleDiscriminator
                                disc_preds_synthetic = style_discriminator(features_for_discriminator.detach()) # Detach to only train discriminator
                                disc_loss = bce_loss(disc_preds_synthetic, synthetic_labels_disc)
                                
                                # Update discriminator
                                if cfg['training']['use_amp']:
                                    scaler.scale(disc_loss).backward()
                                    scaler.unscale_(optimizer_d)
                                    scaler.step(optimizer_d)
                                else:
                                    disc_loss.backward()
                                    optimizer_d.step()
                                
                                # Train generator (main model) to fool discriminator (minimize this loss)
                                # Synthetic features should be classified as 1 (real) by discriminator
                                real_labels_disc = torch.ones(features_for_discriminator.shape[0], 1).to(DEVICE)
                                # Apply GRL: discriminator sees reversed gradients from this path
                                disc_preds_adversarial = style_discriminator(features_for_discriminator, alpha=cfg['training'].get('grl_alpha', 1.0))
                                loss_domain_adaptation = bce_loss(disc_preds_adversarial, real_labels_disc)
                                total_batch_loss += loss_domain_adaptation * cfg['training'].get('lambda_style', 1.0)
                        
                        current_loss = total_batch_loss.item() # Store final loss for logging
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer) # Unscale gradients before clipping
                    else:
                        total_batch_loss.backward()
                    
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
            
            total_train_loss += current_loss
            
            if scheduler is not None and (isinstance(scheduler, OneCycleLR) or (HAS_GRADUAL_WARMUP and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
                scheduler.step()
            
            if ema_model and cfg['training'].get('use_mean_teacher', False):
                ema_decay = cfg['training']['mean_teacher_config'].get('alpha', 0.99)
                for student_param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            
            # GradNorm update (if enabled)
            if grad_norm_instance:
                losses_for_gradnorm = {
                    'bongard_loss': loss_bongard.item(),
                    'attribute_loss': loss_attribute.item(),
                    'relation_loss': loss_relation.item(),
                    'consistency_loss': loss_consistency.item(),
                    'distillation_loss': loss_distillation.item(),
                    'domain_adaptation_loss': loss_domain_adaptation.item() if style_discriminator else 0.0
                }
                # Filter out zero losses if they are not active components
                losses_for_gradnorm = {k: v for k, v in losses_for_gradnorm.items() if v != 0.0}
                if losses_for_gradnorm: # Only update if there are active losses
                    grad_norm_instance.update_weights(losses_for_gradnorm, list(model.parameters()))
            
            train_loop.set_postfix(loss=current_loss)
            
            predictions = torch.argmax(bongard_logits, dim=1)
            train_correct_predictions += (predictions == query_labels).sum().item()
            train_total_samples += query_labels.size(0)
            
            # Update PER priorities if curriculum learning is active
            if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and replay_buffer is not None:
                # Use the per-sample Bongard loss for priority update
                per_sample_losses = model.bongard_criterion(bongard_logits, query_labels, reduction='none', confidences=query_conf1) # Pass confidences here too
                # Ensure tree_indices match per_sample_losses length
                if tree_indices is not None and per_sample_losses.shape[0] == len(tree_indices):
                    async_update_priorities(replay_buffer, tree_indices.cpu().tolist(), per_sample_losses.cpu().tolist(), cfg)
                else:
                    logger.warning("Skipping PER priority update: tree_indices mismatch or not provided.")
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        # SWA update
        if swa_model and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.debug(f"SWA model updated at epoch {epoch+1}.")
        
        # Validation phase
        # Use the base model for validation, or SWA model if active
        val_model = swa_model.module if (swa_model and epoch >= swa_start_epoch) else model
        val_model.eval()
        if style_discriminator:
            style_discriminator.eval() # Set discriminator to eval mode
        total_val_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0
        
        current_epoch_val_logits = []
        current_epoch_val_labels = []

        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation {model_name_prefix}]")
        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(val_loop):
                (raw_query_images_view1_np, _, query_labels_val,
                 query_gts_json_view1_val, _, _, _, _, _,
                 raw_support_images_flat_np_val, support_labels_flat_val, _, _, _, _,
                 query_bboxes_view1_val, query_masks_view1_val,
                 _, _,
                 support_bboxes_flat_val, support_masks_flat_val,
                 query_conf1_val, query_conf2_val, support_confs_flat_val
                ) = batch_val
                
                query_labels_val = query_labels_val.to(DEVICE).long()
                support_labels_flat_val = support_labels_flat_val.to(DEVICE).long()
                query_conf1_val = query_conf1_val.to(DEVICE).float()

                if dali_processor is None or not cfg['data'].get('use_dali', False): # Use cfg['data']['use_dali']
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize(tuple(cfg['data']['image_size'])),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    processed_query_images_view1_val = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                    
                    max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    batch_size_actual_val = processed_query_images_view1_val.shape[0]
                    if raw_support_images_flat_np_val and len(raw_support_images_flat_np_val) > 0:
                        if isinstance(raw_support_images_flat_np_val[0], list):
                            flat_support_images_np_val = [item for sublist in raw_support_images_flat_np_val for item in sublist]
                        else:
                            flat_support_images_np_val = raw_support_images_flat_np_val
                        actual_support_images_np_val = [img for img in flat_support_images_np_val if np.sum(img) > 0]
                        if actual_support_images_np_val:
                            processed_support_images_flat_val = torch.stack([transform(img_np) for img_np in actual_support_images_np_val]).to(DEVICE)
                            processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                                batch_size_actual_val, max_support_imgs_val, 
                                processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                            )
                        else:
                            processed_support_images_reshaped_val = torch.zeros(
                                batch_size_actual_val, max_support_imgs_val, 
                                processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3],
                                device=DEVICE
                            )
                    else:
                        processed_support_images_reshaped_val = torch.zeros(
                            batch_size_actual_val, max_support_imgs_val, 
                            processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3],
                            device=DEVICE
                        )
                else:
                    processed_query_images_view1_val, _, processed_support_images_flat_val = dali_processor.run(
                        raw_query_images_view1_np,
                        [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np),
                        raw_support_images_flat_np_val
                    )
                    max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    batch_size_actual_val = processed_query_images_view1_val.shape[0]
                    processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                        batch_size_actual_val, max_support_imgs_val, 
                        processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                    )
                support_labels_reshaped_val = support_labels_flat_val.view(batch_size_actual_val, max_support_imgs_val)

                outputs_val = val_model(processed_query_images_view1_val, ground_truth_json_strings=query_gts_json_view1_val,
                                                detected_bboxes_batch=query_bboxes_view1_val, detected_masks_batch=query_masks_view1_val,
                                                support_images=processed_support_images_reshaped_val,
                                                support_labels_flat=support_labels_reshaped_val)
                bongard_logits_val = outputs_val['bongard_logits']
                
                loss_val = model.bongard_criterion(bongard_logits_val, query_labels_val, confidences=query_conf1_val).mean()
                total_val_loss += loss_val.item()
                predictions_val = torch.argmax(bongard_logits_val, dim=1)
                val_correct_predictions += (predictions_val == query_labels_val).sum().item()
                val_total_samples += query_labels_val.size(0)
                val_loop.set_postfix(loss=loss_val.item())
                current_epoch_val_logits.extend(bongard_logits_val.cpu().numpy().tolist())
                current_epoch_val_labels.extend(query_labels_val.cpu().numpy().tolist())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_samples
        logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        if scheduler is not None and not (isinstance(scheduler, OneCycleLR) or (HAS_GRADUAL_WARMUP and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered for {model_name_prefix} at epoch {epoch+1}.")
            break
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logger.info(f"New best validation accuracy for {model_name_prefix}: {best_val_accuracy:.4f}")
    
    if profiler_enabled:
        prof.stop()
        logger.info(f"PyTorch Profiler trace saved for {model_name_prefix}.")
    
    if swa_model:
        logger.info(f"Updating BatchNorm for SWA model of {model_name_prefix}.")
        pass # Placeholder for update_bn if it's not defined
        swa_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_swa_model.pt")
        torch.save(swa_model.state_dict(), swa_model_path)
        logger.info(f"SWA model saved to: {swa_model_path}")
    
    if os.path.exists(best_model_path):
        final_model_for_quant = LitBongard(cfg).to(DEVICE)
        loaded_state_dict = torch.load(best_model_path, map_location=DEVICE)
        if list(loaded_state_dict.keys())[0].startswith('module.'):
            loaded_state_dict = {k.replace('module.', ''): v for k, v in loaded_state_dict.items()}
        final_model_for_quant.perception_module.load_state_dict(loaded_state_dict)
        logger.info(f"Loaded best model from {best_model_path} for final evaluation/quantization.")
    else:
        logger.warning(f"No best model checkpoint found at {best_model_path} to load for final evaluation/quantization. Using current model state.")
        final_model_for_quant = model

    if cfg['training']['quantization'].get('qat', False):
        final_model_for_quant = quantize_model_qat(final_model_for_quant, cfg)
        optimized_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_qat_optimized_bongard_model.pth")
        torch.save(final_model_for_quant.state_dict(), optimized_model_path)
        logger.info(f"QAT optimized model saved to: {optimized_model_path}")

    if cfg['training']['quantization'].get('ptq', False):
        final_model_for_quant = quantize_model_ptq(final_model_for_quant, val_loader, cfg)
        optimized_model_path = os.path.join(checkpoint_dir, f"{model_name_prefix}_ptq_optimized_bongard_model.pth")
        torch.save(final_model_for_quant.state_dict(), optimized_model_path)
        logger.info(f"PTQ optimized model saved to: {optimized_model_path}")
    
    logger.info(f"--- Training session for {model_name_prefix} finished. ---")
    
    return best_model_path, current_epoch_val_logits, current_epoch_val_labels, {
        'val_loss': avg_val_loss,
        'val_accuracy': val_accuracy,
        'logits': np.array(current_epoch_val_logits),
        'labels': np.array(current_epoch_val_labels)
    }


def train_distilled_student_orchestrator_combined(
    num_ensemble_members: int,
    train_loader: Any,
    val_loader: Any,
    student_model_config: Dict[str, Any],
    epochs_student: int,
    teacher_ensemble_type: str, # Not directly used in this function, but kept for signature
    train_members: bool, # Flag to indicate if teachers should be trained or loaded
    cfg: Dict[str, Any], # Full config
    current_rank: int,
    world_size: int,
    is_ddp_initialized: bool
) -> Tuple[LitBongard, Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrates the training of teacher ensemble members and then distills knowledge
    to a student model.
    Args:
        num_ensemble_members (int): Number of teacher ensemble members.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        student_model_config (Dict[str, Any]): Configuration for the student model.
        epochs_student (int): Number of epochs to train the student.
        teacher_ensemble_type (str): Type of ensemble ('soft_vote', 'hard_vote', 'stacked').
        train_members (bool): If True, train teachers; otherwise, load them.
        cfg (Dict[str, Any]): Full configuration dictionary.
        current_rank (int): Current GPU rank.
        world_size (int): Total number of GPUs.
        is_ddp_initialized (bool): True if DDP is already initialized.
    Returns:
        Tuple[LitBongard, Dict[str, Any], Dict[str, Any]]:
            - The trained student model.
            - Metrics for the teacher ensemble.
            - Metrics for the student model.
    """
    logger.info("Starting distillation orchestration.")
    teacher_models = []
    teacher_ensemble_metrics = {}
    if train_members:
        logger.info(f"Training {num_ensemble_members} teacher ensemble members.")
        # Create data loaders for teacher training
        dm_teachers = BongardDataModule(cfg)
        train_loader_teachers = dm_teachers.train_dataloader()
        val_loader_teachers = dm_teachers.val_dataloader()

        teacher_models, teacher_metrics_list = train_ensemble(
            num_members=num_ensemble_members,
            train_loader=train_loader_teachers,
            val_loader=val_loader_teachers,
            cfg=cfg,
            current_rank=current_rank,
            world_size=world_size,
            is_ddp_initialized=is_ddp_initialized
        )
        # Aggregate teacher metrics (e.g., average accuracy)
        if teacher_metrics_list:
            avg_teacher_acc = sum(m.get('val_accuracy', 0.0) for m in teacher_metrics_list) / len(teacher_metrics_list)
            teacher_ensemble_metrics = {'accuracy': avg_teacher_acc, 'individual_metrics': teacher_metrics_list}
            logger.info(f"Teacher ensemble training completed. Average accuracy: {avg_teacher_acc:.4f}")
    else:
        logger.info(f"Loading {num_ensemble_members} pre-trained teacher ensemble members.")
        # Create dummy data loader for loading models that might need it for BN update
        dm_dummy = BongardDataModule(cfg)
        val_loader_dummy = dm_dummy.val_dataloader()

        for i in range(num_ensemble_members):
            member_path = os.path.join(cfg['debug']['save_model_checkpoints'], f"member_{i}_best_model.pt")
            if os.path.exists(member_path):
                loaded_model, _ = load_trained_model(
                    model_path=member_path,
                    cfg=cfg,
                    current_rank=current_rank,
                    is_ddp_initialized=is_ddp_initialized,
                    val_loader=val_loader_dummy # Pass val_loader for model init if needed
                )
                loaded_model.eval()
                teacher_models.append(loaded_model)
            else:
                logger.error(f"Teacher model member not found at {member_path}. Cannot proceed with distillation.")
                return None, {}, {}
        logger.info("All teacher ensemble members loaded.")

    # Initialize student model
    logger.info("Initializing student model for distillation.")
    student_model = LitBongard(student_model_config).to(DEVICE)
    if is_ddp_initialized:
        student_model = DDP(student_model, device_ids=[current_rank], output_device=current_rank, find_unused_parameters=False)

    logger.info("Starting student model distillation training.")
    # Create data loaders for student training
    dm_student = BongardDataModule(cfg)
    train_loader_student = dm_student.train_dataloader()
    val_loader_student = dm_student.val_dataloader()

    student_best_path, student_val_logits, student_val_labels, student_metrics = \
        _run_single_training_session_ensemble_member( # Use the member training function for student
            model=student_model,
            train_loader=train_loader_student,
            val_loader=val_loader_student,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=-1, # Indicate this is the student model
            cfg=cfg,
            teacher_model=teacher_models # Pass the trained teacher models
        )
    
    # Load the best student model
    final_student_model, _ = load_trained_model(
        model_path=student_best_path,
        cfg=cfg,
        current_rank=current_rank,
        is_ddp_initialized=is_ddp_initialized,
        val_loader=val_loader_student
    )
    final_student_model.eval()
    logger.info(f"Student model distillation completed. Best accuracy: {student_metrics.get('val_accuracy', 0.0):.4f}")
    return final_student_model, teacher_ensemble_metrics, student_metrics

# Function to load a trained model (PerceptionModule)
def load_trained_model(
    model_path: str, 
    cfg: Dict[str, Any], 
    current_rank: int, 
    is_ddp_initialized: bool,
    val_loader: Any = None # Optional: for BN update if needed
) -> Tuple[Union[PerceptionModule, LitBongard], Dict[str, Any]]:
    """
    Loads a pre-trained PerceptionModule model.
    Args:
        model_path (str): Path to the model checkpoint.
        cfg (Dict[str, Any]): Configuration dictionary.
        current_rank (int): Current GPU rank.
        is_ddp_initialized (bool): True if DDP is already initialized.
        val_loader (Any, optional): DataLoader for validation, used for BatchNorm update if needed.
    Returns:
        Tuple[Union[PerceptionModule, LitBongard], Dict[str, Any]]:
            - Loaded model (PerceptionModule or LitBongard).
            - Loaded metrics (if available in checkpoint).
    """
    logger.info(f"Loading model from {model_path}...")
    # Initialize the full LitBongard model as it contains PerceptionModule
    model = LitBongard(cfg).to(DEVICE)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # If the state_dict was saved from a DDP model, remove 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # If the state_dict is from a raw PerceptionModule, load it directly into model.perception_module
    # If it's from a LitBongard, it should have keys like 'perception_module.backbone.0.weight'
    # Attempt to load, allowing for missing keys if the checkpoint is only for the backbone
    model.perception_module.load_state_dict(state_dict, strict=False)
    
    logger.info("Model loaded successfully.")
    model.eval() # Set to evaluation mode
    
    # If DDP is initialized, wrap the model
    if is_ddp_initialized:
        model = DDP(model, device_ids=[current_rank], output_device=current_rank, find_unused_parameters=False)
    
    # If using SWA and loading an SWA model, you might need to update BN layers
    # This is a placeholder for that logic if needed.
    # if cfg['training'].get('use_swa', False) and 'swa_model' in model_path:
    #     logger.info("Updating BatchNorm for loaded SWA model...")
    #     update_bn(val_loader, model, device=DEVICE) 
    
    return model, {} # Return empty metrics for now, as they are not typically saved in state_dict
