# Folder: bongard_solver/
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

# Import for pruning
import torch.nn.utils.prune as prune

# Import configuration
from config import (
    CONFIG, DEVICE, HAS_WANDB, HAS_TORCH_QUANTIZATION,
    ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
    ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
    RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD
)

# Import PyTorch Lightning modules
import pytorch_lightning as pl
from models import LitBongard, LitSimCLR, PerceptionModule
from data import BongardDataModule, get_loader # Changed from create_dataloader to get_loader

# Import losses (assuming these are used)
from losses import LabelSmoothingCrossEntropy, DistillationLoss, FeatureConsistencyLoss, SymbolicConsistencyLoss

# Import for torchvision.transforms.v2 for MixUp/CutMix
try:
    import torchvision.transforms.v2 as T_v2
    HAS_TORCHVISION_V2 = True
    logger = logging.getLogger(__name__)
    logger.info("torchvision.transforms.v2 found and enabled.")
except ImportError:
    HAS_TORCHVISION_V2 = False
    logger = logging.getLogger(__name__)
    logger.warning("torchvision.transforms.v2 not found. MixUp/CutMix will be disabled.")

# Import SAM optimizer
try:
    from sam import SAM
    HAS_SAM_OPTIMIZER = True
except ImportError:
    HAS_SAM_OPTIMIZER = False
    logger = logging.getLogger(__name__)
    logger.warning("SAM optimizer not found. SAM optimizer functionality will be disabled.")

# Import advanced optimizers (if available)
try:
    from ranger_adabelief import RangerAdaBelief
    HAS_RANGER = True
except ImportError:
    HAS_RANGER = False
    logger = logging.getLogger(__name__)
    logger.warning("RangerAdaBelief not found. Ranger optimizer will be disabled.")

try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    HAS_LION = False
    logger = logging.getLogger(__name__)
    logger.warning("Lion optimizer not found. Lion optimizer will be disabled.")

# Import GradNorm
try:
    from grad_norm import GradNorm
    HAS_GRAD_NORM = True
    logger = logging.getLogger(__name__)
    logger.info("GradNorm found and enabled.")
except ImportError:
    HAS_GRAD_NORM = False
    logger = logging.getLogger(__name__)
    logger.warning("GradNorm not found. GradNorm functionality will be disabled.")

# Import GradualWarmupScheduler and CosineAnnealingWarmRestarts (for 7.1)
try:
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    from warmup_scheduler import GradualWarmupScheduler # Assuming this is available
    HAS_GRADUAL_WARMUP = True
    logger = logging.getLogger(__name__)
    logger.info("GradualWarmupScheduler and CosineAnnealingWarmRestarts found and enabled.")
except ImportError:
    HAS_GRADUAL_WARMUP = False
    logger = logging.getLogger(__name__)
    logger.warning("GradualWarmupScheduler or CosineAnnealingWarmRestarts not found. Warmup cosine scheduler will be disabled.")


logger = logging.getLogger(__name__)

# --- Helper functions (from previous context or to be defined) ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For PyTorch Lightning, it's often better to set seed via `pl.seed_everything(seed)`
    pl.seed_everything(seed)
    logger.info(f"Random seed set to {seed}")

# Dummy async_update_priorities for PER
def async_update_priorities(replay_buffer, indices, losses, cfg):
    """Dummy function for asynchronous priority update."""
    # In a real scenario, this would involve a separate thread or process
    # updating the replay buffer's priorities.
    # For now, just log a message.
    logger.debug(f"Async update priorities called for {len(indices)} samples.")
    # Example: replay_buffer.update_priorities(indices, losses)
    # For now, directly call update for demonstration
    replay_buffer.update_priorities(indices, losses) # Removed cfg from here, as it's not used in the buffer's update_priorities

# Dummy QAT/PTQ functions (actual implementations might be in prune_quantize.py)
def quantize_model_qat(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """Dummy function for Quantization Aware Training (QAT)."""
    logger.info("Performing dummy Quantization Aware Training (QAT).")
    # In a real scenario, this would involve:
    # 1. Fusion of modules (e.g., Conv + BatchNorm + ReLU)
    # 2. Insertion of FakeQuantize modules
    # 3. Training loop with QAT
    # 4. Conversion to a quantized model
    return model

def quantize_model_ptq(model: nn.Module, val_loader: DataLoader, cfg: Dict[str, Any]) -> nn.Module:
    """Dummy function for Post-Training Quantization (PTQ)."""
    logger.info("Performing dummy Post-Training Quantization (PTQ).")
    # In a real scenario, this would involve:
    # 1. Setting up quantization configurations (e.g., QConfig)
    # 2. Fusing modules
    # 3. Calibrating the model with a representative dataset (val_loader)
    # 4. Converting to a quantized model
    return model

# Early Stopping class (moved here for self-containment as requested)
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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Mixup/CutMix Augmenter (moved here for self-containment)
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
            return self.aug(images, labels)
        return images, F.one_hot(labels, num_classes=self.num_classes).float() # Return one-hot for consistency if no mixup

# Ensemble Teacher Logits (from previous context)
def _get_ensemble_teacher_logits(
    teacher_models: nn.ModuleList,
    raw_images_np: List[np.ndarray],
    raw_gt_json_strings: List[bytes],
    raw_support_images_np: List[np.ndarray],
    distillation_config: Dict[str, Any],
    dali_image_processor: Any = None # DALI processor from datamodule
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Helper function to get ensemble teacher logits for distillation.
    """
    if not teacher_models:
        logger.warning("No teacher models provided for distillation. Returning empty logits.")
        return torch.empty(0), None
    all_teacher_logits = []
    all_distillation_masks = [] # For per-sample masking
    for teacher_model in teacher_models:
        teacher_model.eval() # Set teacher to evaluation mode
        with torch.no_grad():
            # Process images using DALI or torchvision transforms
            if dali_image_processor is None:
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((CONFIG['data']['image_size'], CONFIG['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_images = torch.stack([transform(img_np) for img_np in raw_images_np]).to(DEVICE)
                processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_np]).to(DEVICE)
            else:
                # DALI returns processed tensors
                processed_images, _, processed_support_images_flat = dali_image_processor.run(
                    raw_images_np,
                    [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_images_np), # Dummy view2
                    raw_support_images_np
                )
            
            # Reshape flattened support images back to (B, N_support, C, H, W)
            max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_images.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_images.shape[1], processed_images.shape[2], processed_images.shape[3]
            )
            # Teacher forward pass
            # Note: Teacher models also need ground_truth_json_strings and support_labels_flat
            # for their internal PerceptionModule to build scene graphs and compute bongard_logits.
            # We'll pass dummy labels for support_labels_flat if not available, as teacher doesn't need them for loss.
            teacher_outputs = teacher_model.perception_module(
                processed_images,
                ground_truth_json_strings=raw_gt_json_strings,
                support_images=processed_support_images_reshaped,
                support_labels_flat=torch.zeros_like(processed_support_images_reshaped[:,:,0,0,0], dtype=torch.long) # Dummy labels
            )
            teacher_logits = teacher_outputs['bongard_logits']
            
            # Per-sample masking for distillation
            current_distillation_mask = torch.ones(batch_size_actual, dtype=torch.float, device=DEVICE)
            if distillation_config.get('use_mask_distillation', False):
                # Example masking: mask out samples with low confidence from teacher
                teacher_probs = F.softmax(teacher_logits / distillation_config['temperature'], dim=-1)
                max_teacher_probs, _ = torch.max(teacher_probs, dim=-1)
                current_distillation_mask = (max_teacher_probs > distillation_config.get('mask_threshold', 0.8)).float()
            
            all_teacher_logits.append(teacher_logits)
            all_distillation_masks.append(current_distillation_mask)
    # Average logits from all teachers
    if all_teacher_logits:
        ensemble_logits = torch.stack(all_teacher_logits, dim=0).mean(dim=0) # (B, num_classes)
        # Combine masks (e.g., logical AND if any teacher masks out, it's masked out)
        final_distillation_mask = torch.stack(all_distillation_masks, dim=0).prod(dim=0)
        return ensemble_logits, final_distillation_mask
    return torch.empty(0), None

# --- Main Training Function ---
def run_training_once(cfg: Dict[str, Any], epochs: int = None, trial: Optional[Any] = None) -> float:
    """
    Runs a single training session for the Bongard solver.
    This function is used for both standard training and HPO trials.
    Args:
        cfg (Dict[str, Any]): The configuration dictionary.
        epochs (int, optional): Number of epochs to train. If None, uses cfg['training']['epochs'].
        trial (optuna.trial.Trial, optional): Optuna trial object for pruning.
    Returns:
        float: The final validation accuracy achieved.
    """
    set_seed(cfg['training']['seed']) # Ensure reproducibility for each trial
    # Initialize model
    model = LitBongard(cfg).to(DEVICE)
    # Initialize data modules and loaders
    data_module = BongardDataModule(cfg)
    data_module.setup(stage='fit') # Call setup to prepare data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Optimizer and Scheduler (manual instantiation for this loop)
    # This part needs to match configure_optimizers in LitBongard
    lr = cfg['training']['learning_rate']
    weight_decay = cfg['training']['weight_decay']
    optimizer_name = cfg['training'].get('optimizer', 'AdamW')
    
    # Initialize base optimizer
    base_optimizer = None
    if optimizer_name == 'ranger':
        # Assuming RangerAdaBelief is imported from torch_optimizer (if HAS_RANGER)
        if HAS_RANGER:
            base_optimizer = RangerAdaBelief(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            logger.warning("Ranger optimizer requested but not available. Falling back to AdamW.")
            base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'lion':
        # Assuming Lion is imported from lion_pytorch (if HAS_LION)
        if HAS_LION:
            base_optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            logger.warning("Lion optimizer requested but not available. Falling back to AdamW.")
            base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: # Default to AdamW
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Wrap with SAM if configured
    optimizer = base_optimizer
    if cfg['training'].get('optimizer', 'AdamW') == 'sam' and HAS_SAM_OPTIMIZER:
        optimizer = SAM(base_optimizer, rho=cfg['training'].get('sam_rho', 0.05))
        logger.info("SAM optimizer initialized for manual training loop.")
    else:
        if cfg['training'].get('optimizer', 'AdamW') == 'sam':
            logger.warning("SAM optimizer requested but `sam.SAM` not found. Proceeding without SAM.")

    # Scheduler setup
    scheduler = None
    scheduler_name = cfg['training'].get('scheduler', 'CosineAnnealingLR')
    if scheduler_name == 'OneCycleLR':
        if len(train_loader) > 0:
            steps_per_epoch = len(train_loader)
        else:
            logger.warning("Train loader is empty, cannot calculate steps_per_epoch for OneCycleLR. Using default.")
            steps_per_epoch = 1000 # Fallback
        scheduler = OneCycleLR(
            optimizer.base_optimizer if (isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER) else optimizer,
            max_lr=cfg['training']['scheduler_config']['OneCycleLR'].get('max_lr', 1e-3),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs if epochs is not None else cfg['training']['epochs'],
            pct_start=cfg['training']['scheduler_config']['OneCycleLR'].get('pct_start', 0.3),
            div_factor=cfg['training']['scheduler_config']['OneCycleLR'].get('div_factor', 25),
            final_div_factor=cfg['training']['scheduler_config']['OneCycleLR'].get('final_div_factor', 1e4)
        )
        logger.info("OneCycleLR scheduler initialized.")
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer.base_optimizer if (isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER) else optimizer,
            T_max=epochs if epochs is not None else cfg['training']['epochs'],
            eta_min=cfg['training']['scheduler_config']['CosineAnnealingLR'].get('eta_min', 1e-6)
        )
        logger.info("CosineAnnealingLR scheduler initialized.")
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer.base_optimizer if (isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER) else optimizer,
            mode=cfg['training']['scheduler_config']['ReduceLROnPlateau'].get('mode', 'min'),
            factor=cfg['training']['scheduler_config']['ReduceLROnPlateau'].get('factor', 0.1),
            patience=cfg['training']['scheduler_config']['ReduceLROnPlateau'].get('patience', 5)
        )
        logger.info("ReduceLROnPlateau scheduler initialized.")
    elif scheduler_name == 'warmup_cosine': # Added for 7.1
        if HAS_GRADUAL_WARMUP:
            scheduler_cos = CosineAnnealingWarmRestarts(
                optimizer.base_optimizer if (isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER) else optimizer, 
                T_0=cfg['training']['scheduler_config']['warmup_cosine'].get('T_0', 10), 
                T_mult=cfg['training']['scheduler_config']['warmup_cosine'].get('T_mult', 2), 
                eta_min=cfg['training']['scheduler_config']['warmup_cosine'].get('eta_min', 1e-6)
            )
            scheduler = GradualWarmupScheduler(
                optimizer.base_optimizer if (isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER) else optimizer, 
                multiplier=cfg['training']['scheduler_config']['warmup_cosine'].get('multiplier', 1.0), 
                total_epoch=cfg['training']['scheduler_config']['warmup_cosine'].get('warmup_epochs', 5), 
                after_scheduler=scheduler_cos
            )
            logger.info("warmup_cosine scheduler initialized.")
        else:
            logger.warning("warmup_cosine scheduler requested but GradualWarmupScheduler not found. Falling back to no scheduler.")
    else:
        logger.warning(f"Scheduler '{scheduler_name}' not recognized. No scheduler will be used.")

    scaler = GradScaler() if cfg['training']['use_amp'] else None

    # Early Stopping
    early_stop_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_model_early_stop.pt")
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stop_patience'],
        delta=cfg['training']['early_stop_delta'],
        verbose=True,
        path=early_stop_path
    )
    logger.info(f"Early Stopping initialized with patience={early_stopping.patience}, delta={early_stopping.delta}.")

    # MixUp/CutMix Augmenter
    num_bongard_classes = cfg['model']['bongard_head_config']['num_classes']
    mixup_cutmix_augmenter = MixupCutmixAugmenter(cfg['training'], num_bongard_classes)

    # Mean Teacher (EMA)
    ema_model = None
    if cfg['training'].get('use_mean_teacher', False):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info("Mean Teacher (EMA) model initialized.")

    # GradNorm (if enabled)
    grad_norm = None
    if cfg['training'].get('use_grad_norm', False) and HAS_GRAD_NORM:
        # Assuming GradNorm needs task weights or a similar setup
        # For a single-task Bongard problem, it might just monitor overall gradients
        # or if there are multiple loss components, it can balance them.
        # Here, we'll assume it's set up to balance bongard_loss with other potential losses
        # or just monitor overall gradient norm.
        # Initial task weights (e.g., all 1.0)
        initial_task_weights = {'bongard_loss': 1.0} # Add other loss names if applicable
        grad_norm = GradNorm(model, initial_task_weights, cfg['training']['grad_norm_alpha'], DEVICE)
        logger.info("GradNorm initialized.")

    best_val_accuracy = 0.0
    num_epochs = epochs if epochs is not None else cfg['training']['epochs']

    # PyTorch Profiler setup
    # The schedule means: wait for 1 step, then warmup for 1 step, then active for 3 steps, then repeat 2 times.
    # This will profile steps 2,3,4 and 7,8,9.
    # The trace will be saved to cfg.paths.logs_dir (e.g., './logs/').
    # Make sure this directory exists and is writable.
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.paths.logs_dir),
        record_shapes=True,
        profile_memory=True, # Enable memory profiling
        with_stack=True # Record stack information
    )
    prof.start() # Start the profiler

    for epoch in range(num_epochs):
        # 8.2 Hyperband Pruner Callbacks
        if trial: # Check if trial object is provided (for Optuna HPO)
            trial.report(best_val_accuracy, step=epoch) # Report current best accuracy to Optuna
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                raise optuna.TrialPruned()

        model.train()
        total_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        # Wrap train_loader with tqdm for progress bar
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_loop):
            # Unpack batch data
            (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
             query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
             raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
             tree_indices, is_weights) = batch
            
            # Move labels to device
            query_labels = query_labels.to(DEVICE).long()
            support_labels_flat = support_labels_flat.to(DEVICE).long()

            # DALI Image Processor
            # Assuming dali_processor is correctly initialized and available from data_module
            # If using PyTorch DataLoader, dali_processor will be None, and transforms are applied.
            dali_processor = getattr(data_module, 'dali_train_loader', None) # Access the DALI loader directly if it's a DALIGenericIterator
            
            # If not using DALI, apply PyTorch transforms manually
            if not (cfg['data']['use_dali'] and HAS_DALI): # Check if DALI is enabled and available
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_query_images_view1 = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                processed_query_images_view2 = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(DEVICE)
                processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np]).to(DEVICE)
            else:
                # If DALI is used, the batch items are already tensors on the correct device
                # and processed by the DALI pipeline.
                processed_query_images_view1 = batch['query_img1'].to(DEVICE)
                processed_query_images_view2 = batch['query_img2'].to(DEVICE)
                processed_support_images_flat = batch['padded_support_imgs'].to(DEVICE)
                # For DALI, other items like query_gts_json_view1 might still be on CPU or need specific handling
                # based on how they are passed through the DALI pipeline.
                # Assuming they are passed as raw data and handled by the model's forward.


            # Reshape flattened support images back to (B, N_support, C, H, W)
            max_support_imgs = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_query_images_view1.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
            )
            # Reshape flattened support labels back to (B, N_support)
            support_labels_reshaped = support_labels_flat.view(batch_size_actual, max_support_imgs)

            # Apply Mixup/CutMix
            images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)

            # PyTorch Profiler: Mark model inference section
            with record_function("model_inference"):
                # SAM Optimizer first step
                if isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER:
                    # First forward-backward pass
                    with autocast(enabled=cfg['training']['use_amp']) if cfg['training']['use_amp'] else torch.no_grad():
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        # Calculate loss (only Bongard loss for simplicity in SAM step)
                        loss = model.bongard_criterion(bongard_logits, labels_mixed).mean() # Use mean for SAM step
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(loss).backward()
                        optimizer.first_step(zero_grad=True)
                    else:
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                    
                    # Second forward-backward pass
                    with autocast(enabled=cfg['training']['use_amp']) if cfg['training']['use_amp'] else torch.no_grad():
                        outputs2 = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits2 = outputs2['bongard_logits']
                        # Calculate total loss for the second step
                        total_batch_loss = model.bongard_criterion(bongard_logits2, labels_mixed).mean() # Use mean for SAM step
                        # Add other losses if needed, similar to LitBongard's training_step
                        # For simplicity in this `run_training_once`, we'll just use bongard_loss for SAM.
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer) # Unscale before clipping
                    else:
                        total_batch_loss.backward()
                    
                    # Gradient Clipping before optimizer.second_step
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.second_step(zero_grad=True) # zero_grad=True for SAM's second step
                    
                    current_loss = total_batch_loss.item()
                else: # Standard optimizer
                    optimizer.zero_grad()
                    with autocast(enabled=cfg['training']['use_amp']) if cfg['training']['use_amp'] else torch.no_grad():
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        
                        # Calculate full loss (similar to LitBongard's training_step)
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, labels_mixed).sum(dim=1) # sum for MixUp/CutMix
                        loss_bongard = per_sample_bongard_losses.mean() # Mean for batch
                        # Placeholder for attribute and relation losses (simplified for run_training_once)
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        
                        # Feature Consistency Loss
                        if cfg['training'].get('feature_consistency_alpha', 0) > 0:
                            _, _, aggregated_outputs_view2 = model(processed_query_images_view2, query_gts_json_view2, processed_support_images_reshaped, support_labels_reshaped)
                            feature_consistency_criterion = FeatureConsistencyLoss(loss_type=cfg['training']['feature_consistency_loss_type'])
                            consistency_loss_features = feature_consistency_criterion(
                                outputs['image_features_student'],
                                aggregated_outputs_view2['image_features_student']
                            )
                            loss_bongard += cfg['training']['feature_consistency_alpha'] * consistency_loss_features
                        
                        # Symbolic Consistency Loss
                        if cfg['training'].get('symbolic_consistency_alpha', 0) > 0:
                            symbolic_cons_loss_fn = SymbolicConsistencyLoss(
                                all_bongard_rules=ALL_BONGARD_RULES, # Pass rules if needed
                                config=cfg
                            )
                            # This needs actual scene graphs from both views
                            # For simplicity, passing dummy inputs if not fully integrated
                            symbolic_consistency_loss = symbolic_cons_loss_fn(
                                outputs['extracted_scene_graphs'],
                                outputs['extracted_scene_graphs'] # Use same for both views as dummy
                            )
                            loss_bongard += cfg['training']['symbolic_consistency_alpha'] * symbolic_consistency_loss
                        
                        # Knowledge Distillation (if teacher model provided) - assuming teacher is set up in LitBongard
                        if cfg['training'].get('use_knowledge_distillation', False) and hasattr(model, 'teacher_model') and model.teacher_model is not None:
                            dist_criterion = DistillationLoss(
                                temperature=cfg['training']['distillation_temperature'],
                                alpha=cfg['training']['distillation_alpha'],
                                reduction='mean'
                            )
                            # Get teacher logits (assuming teacher_model is a single model or ensemble)
                            # This needs to be done carefully to match teacher's forward signature
                            # For now, assuming model.teacher_model can directly process images_view1_aug
                            with torch.no_grad():
                                teacher_logits, _, _ = model.teacher_model(processed_query_images_view1, query_gts_json_view1, processed_support_images_reshaped, support_labels_reshaped)
                            
                            dist_loss = dist_criterion(
                                student_logits=bongard_logits,
                                teacher_logits=teacher_logits,
                                target_labels=query_labels # Hard labels for the hard component
                            )
                            total_batch_loss = dist_loss # Distillation loss replaces main loss if enabled (soft target only)
                        else:
                            total_batch_loss = loss_bongard + loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0) + \
                                               loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                    
                    if cfg['training']['use_amp']:
                        scaler.scale(total_batch_loss).backward()
                        scaler.unscale_(optimizer) # Unscale before clipping
                    else:
                        total_batch_loss.backward()
                    
                    # Gradient Clipping
                    if cfg['training'].get('max_grad_norm', 0.0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    
                    if cfg['training']['use_amp']:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    current_loss = total_batch_loss.item()
            
            total_train_loss += current_loss
            
            # Scheduler step per batch (for OneCycleLR)
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            elif scheduler is not None and isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts):
                # For GradualWarmupScheduler with CosineAnnealingWarmRestarts, step the wrapper
                scheduler.step()
            
            # Update EMA Teacher
            if ema_model and cfg['training'].get('use_mean_teacher', False):
                ema_decay = cfg['training']['mean_teacher_config'].get('alpha', 0.99)
                for student_param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            
            # Update GradNorm weights
            if grad_norm and HAS_GRAD_NORM:
                # Assuming GradNorm.backward() was called earlier or it hooks into the loss.
                # Here, we update its weights after optimizer step.
                # This needs to be integrated with how GradNorm computes its loss.
                # For simple logging, we can just access its weights.
                pass # GradNorm logic is more complex, handled within LitBongard if integrated there.
            
            # Update progress bar
            train_loop.set_postfix(loss=current_loss)
            
            # Calculate accuracy for logging
            predictions = torch.argmax(bongard_logits, dim=1)
            train_correct_predictions += (predictions == query_labels).sum().item()
            train_total_samples += query_labels.size(0)

            # Update replay buffer priorities (for PER)
            if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and hasattr(data_module.train_dataset, 'replay_buffer'):
                # Get per-sample losses for priority update
                per_sample_losses = model.bongard_criterion(bongard_logits, query_labels, reduction='none')
                data_module.train_dataset.replay_buffer.update_priorities(original_indices.cpu().tolist(), per_sample_losses.cpu().tolist()) # Removed cfg
            
            # PyTorch Profiler: Step the profiler after each batch
            prof.step()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")

        # Log GradNorm Weights to TensorBoard
        if grad_norm and HAS_GRAD_NORM and hasattr(model, 'logger') and hasattr(model.logger, 'experiment'):
            # Assuming model has a logger with a TensorBoard experiment
            writer = model.logger.experiment # Access the TensorBoard writer
            for name, w in grad_norm.weights_.items():
                writer.add_scalar(f"GradNorm/{name}", w, epoch)
            logger.info(f"GradNorm weights logged for epoch {epoch+1}.")

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0
        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(val_loop):
                (raw_query_images_view1_np, _, query_labels_val,
                 query_gts_json_view1_val, _, _, _, _, _,
                 raw_support_images_flat_np_val, support_labels_flat_val, _, _, _, _) = batch_val
                
                query_labels_val = query_labels_val.to(DEVICE).long()
                support_labels_flat_val = support_labels_flat_val.to(DEVICE).long()

                # If not using DALI, apply PyTorch transforms manually
                if not (cfg['data']['use_dali'] and HAS_DALI):
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    processed_query_images_view1_val = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                    processed_support_images_flat_val = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np_val]).to(DEVICE)
                else:
                    processed_query_images_view1_val = batch_val['query_img1'].to(DEVICE)
                    processed_support_images_flat_val = batch_val['padded_support_imgs'].to(DEVICE)

                max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual_val = processed_query_images_view1_val.shape[0]
                processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                    batch_size_actual_val, max_support_imgs_val, 
                    processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                )
                support_labels_reshaped_val = support_labels_flat_val.view(batch_size_actual_val, max_support_imgs_val)

                outputs_val = model(processed_query_images_view1_val, ground_truth_json_strings=query_gts_json_view1_val,
                                    support_images=processed_support_images_reshaped_val,
                                    support_labels_flat=support_labels_reshaped_val)
                bongard_logits_val = outputs_val['bongard_logits']
                
                loss_val = model.bongard_criterion(bongard_logits_val, query_labels_val).mean()
                total_val_loss += loss_val.item()
                predictions_val = torch.argmax(bongard_logits_val, dim=1)
                val_correct_predictions += (predictions_val == query_labels_val).sum().item()
                val_total_samples += query_labels_val.size(0)
                val_loop.set_postfix(loss=loss_val.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_samples
        logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

        # Scheduler step per epoch (for CosineAnnealingLR, ReduceLROnPlateau)
        if scheduler is not None and not isinstance(scheduler, OneCycleLR) and not (isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Early Stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
        
        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_bongard_model.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved to {model_save_path} with accuracy: {best_val_accuracy:.4f}")

    # Stop the profiler after the training loop
    prof.stop()
    logger.info("PyTorch Profiler trace saved.")

    # Load best model for potential QAT/PTQ
    if os.path.exists(early_stop_path):
        model.load_state_dict(torch.load(early_stop_path))
        logger.info(f"Loaded best model from {early_stop_path} for final evaluation/quantization.")
    elif os.path.exists(os.path.join(cfg['debug']['save_model_checkpoints'], "best_bongard_model.pt")):
        model.load_state_dict(torch.load(os.path.join(cfg['debug']['save_model_checkpoints'], "best_bongard_model.pt")))
        logger.info("Loaded best model (from best_bongard_model.pt) for final evaluation/quantization.")
    else:
        logger.warning("No best model checkpoint found to load for final evaluation/quantization.")

    # QAT/PTQ Hooks
    if cfg['quantization']['qat']:
        model = quantize_model_qat(model, cfg)
        optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "qat_optimized_bongard_model.pth")
        torch.save(model.state_dict(), optimized_model_path)
        logger.info(f"QAT optimized model saved to: {optimized_model_path}")
    if cfg['quantization']['ptq']:
        model = quantize_model_ptq(model, val_loader, cfg) # Pass val_loader for calibration
        optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "ptq_optimized_bongard_model.pth")
        torch.save(model.state_dict(), optimized_model_path)
        logger.info(f"PTQ optimized model saved to: {optimized_model_path}")

    logger.info("--- Training Pipeline finished. ---")
    return best_val_accuracy

# Dummy functions for pruning (actual implementations might be in prune_quantize.py)
def compute_layer_sensitivity(model: nn.Module, val_loader: DataLoader, dali_image_processor: Any, current_rank: int) -> Dict[str, float]:
    logger.info("Performing dummy layer sensitivity computation.")
    # Return dummy sensitivities for demonstration
    sensitivities = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d)):
            sensitivities[name] = random.uniform(0.1, 1.0) # Dummy sensitivity
    return sensitivities

def apply_structured_pruning(
    model: nn.Module,
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    dali_image_processor: Any,
    current_rank: int,
    is_ddp_initialized: bool,
    sensitivity_scores: Optional[Dict[str, float]] = None,
    groups: int = 1 # Added groups
) -> nn.Module:
    logger.info(f"Performing dummy structured pruning with groups={groups}.")
    # Dummy pruning logic: apply some pruning to a few layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if random.random() < cfg['training']['pruning'].get('pruning_target_layers_ratio', 0.5):
                prune.random_unstructured(module, name="weight", amount=cfg['training']['pruning'].get('amount', 0.2))
                logger.info(f"Dummy pruning applied to {name} with amount {cfg['training']['pruning'].get('amount', 0.2)}")
    return model

# Main execution block (for direct script execution)
if __name__ == "__main__":
    # Example usage:
    # Ensure your config.yaml has relevant sections for training, model, etc.
    # For a minimal run, you might need:
    # training:
    #   seed: 42
    #   epochs: 10
    #   learning_rate: 1e-4
    #   weight_decay: 1e-5
    #   optimizer: AdamW
    #   scheduler: CosineAnnealingLR
    #   scheduler_config:
    #     CosineAnnealingLR:
    #       eta_min: 1e-6
    #     warmup_cosine: # Added for 7.1
    #       T_0: 10
    #       T_mult: 2
    #       eta_min: 1e-6
    #       multiplier: 1.0
    #       warmup_epochs: 5
    #   use_amp: False
    #   early_stop_patience: 5
    #   early_stop_delta: 0.001
    #   use_mixup_cutmix: False
    #   use_mean_teacher: False
    #   max_grad_norm: 0.0
    #   use_knowledge_distillation: False
    #   consistency_loss_weight: 0.0
    #   feature_consistency_weight: 0.0
    #   symbolic_consistency_weight: 0.0
    #   use_grad_norm: False # Added for 8.1
    #   grad_norm_alpha: 0.1 # Added for 8.1
    # model:
    #   backbone: mobilenet_v3_small
    #   pretrained: True
    #   attribute_classifier_config:
    #     shape: 4
    #     color: 6
    #     fill: 4
    #     size: 3
    #     orientation: 2
    #     texture: 2
    #     mlp_dim: 256 # Added for AttributeClassifier head
    #     head_dropout_prob: 0.3 # Added for AttributeClassifier head
    #   relation_gnn_config:
    #     hidden_dim: 256
    #     num_layers: 2
    #     num_relations: 11
    #     dropout_prob: 0.1
    #     use_edge_features: False
    #   bongard_head_config:
    #     hidden_dim: 256
    #     num_classes: 2
    #     dropout_prob: 0.3
    #     attn_dim: 256 # Added for BongardHead FiLM
    #   simclr_config:
    #     temperature: 0.07
    #     projection_dim: 128
    #     mlp_hidden_size: 512
    #     pretrain_epochs: 50
    #     use_moco: False # Added for MoCo
    #     head_layers: 4 # Added for SimCLR projection head
    #     moco_k: 65536 # Added for MoCo
    #     moco_m: 0.999 # Added for MoCo
    #   use_dropblock: False
    #   dropblock_config:
    #     block_size: 7
    #     drop_prob: 0.1
    #   use_stochastic_depth: False
    #   stochastic_depth_p: 0.1
    # few_shot:
    #   enable: False
    #   k_shot: 5
    #   n_way: 2
    # data:
    #   image_size: 128
    #   dataloader_workers: 4
    #   synthetic_data_config:
    #     max_support_images_per_problem: 5
    #     num_train_problems: 100
    #     num_val_problems: 20
    #     min_objects_per_image: 1
    #     max_objects_per_image: 5
    #     min_support_images_per_problem: 2
    #     max_support_images_per_problem: 5
    # object_detector:
    #   use_yolo: False
    #   fine_tune: False
    #   yolo_pretrained: 'yolov8n.pt'
    #   train_data: 'data/objects/train'
    #   val_data: 'data/objects/val'
    #   epochs: 50
    #   batch_size: 16
    #   yolo_conf_threshold: 0.25
    #   yolo_iou_threshold: 0.45
    # segmentation:
    #   use_sam: False
    #   sam_model_type: 'vit_b'
    #   sam_checkpoint_path: 'sam_vit_b_01ec64.pth'
    #   sam_pred_iou_thresh: 0.86
    # debug:
    #   save_model_checkpoints: './checkpoints'
    #   ray_tune_dir: './ray_results'
    # quantization:
    #   qat: False
    #   ptq: False
    # HAS_WANDB: False
    # replay: # Added for 6.1
    #   anneal_epochs: 10
    #   alpha_start: 0.6
    #   alpha_end: 0.0
    #   beta_start: 0.4
    #   beta_end: 1.0
    cfg = CONFIG # Load the default config from config.py
    # Example: Override some config values for a quick test run
    cfg['training']['epochs'] = 2
    cfg['debug']['save_model_checkpoints'] = './temp_checkpoints'
    os.makedirs(cfg['debug']['save_model_checkpoints'], exist_ok=True)
    # For `run_training_once` to work without a full HPO setup,
    # we need to ensure the `trial` object is `None` or a dummy.
    # When called from hpo.py, `trial` will be provided.
    final_accuracy = run_training_once(cfg, trial=None) 
    print(f"Training finished. Final validation accuracy: {final_accuracy:.4f}")
