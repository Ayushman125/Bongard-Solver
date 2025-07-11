# Folder: bongard_solver/
# File: training.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # Keep optim for custom optimizers
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast # For AMP
import os
import logging
import random
import numpy as np
import copy # For deepcopy for Mean Teacher
import json # For handling scene graph bytes
import threading # For async updates

from typing import List, Dict, Any, Tuple, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Conditional imports for optimizers
try:
    from torch_optimizer import SAM
    HAS_SAM_OPTIMIZER = True
except ImportError:
    HAS_SAM_OPTIMIZER = False

try:
    from ranger_adabelief import RangerAdaBelief
    HAS_RANGER = True
except ImportError:
    HAS_RANGER = False

try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    HAS_LION = False

# Conditional imports for data augmentation (AutoAugment, RandAugment)
try:
    # Assuming autoaugment is installed via git+https://github.com/google/automl.git#egg=autoaugment
    from autoaugment import ImageNetPolicy, CIFAR10Policy
    HAS_AUTOAUGMENT = True
except ImportError:
    HAS_AUTOAUGMENT = False

from torchvision import transforms as T
from torchvision.transforms import RandomErasing

# Import configuration
from config import CONFIG, DEVICE, IMAGENET_MEAN, IMAGENET_STD, RELATION_MAP, \
                   ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP, \
                   ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP, \
                   HAS_TORCH_QUANTIZATION, HAS_WANDB # Import HAS_TORCH_QUANTIZATION and HAS_WANDB

# Import data loading components
from data import (
    BongardSyntheticDataset, RealBongardDataset,
    CurriculumSampler, build_dali_image_processor,
    BongardGenerator, get_dataloader, custom_collate_fn
)
from replay_buffer import KnowledgeReplayBuffer # For curriculum learning / PER

# Import model components
from models import LitBongard, BongardSolverEnsemble, PerceptionModule, LitSimCLR # All models from models.py

# Import loss functions
from losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss, NTXentLoss

# Import utils functions
from utils import set_seed, _calculate_iou, make_edge_index_map, async_update_priorities # async_update_priorities from utils

# Import bongard_rules and symbolic_engine for SymbolicConsistencyLoss
try:
    from bongard_rules import ALL_BONGARD_RULES, BongardRule
    from symbolic_engine import SymbolicEngine
    HAS_SYMBOLIC_ENGINE_DEPS = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import bongard_rules or symbolic_engine. SymbolicConsistencyLoss will be limited.")
    HAS_SYMBOLIC_ENGINE_DEPS = False
    # Dummy classes if not found, to prevent errors
    class SymbolicEngine:
        def __init__(self, *args, **kwargs): pass
        def find_best_rules(self, *args, **kwargs): return []
    class BongardRule:
        def __init__(self, name="dummy_rule"): self.name = name
    ALL_BONGARD_RULES = []

# Import pruning specific modules
try:
    import torch.nn.utils.prune as prune
    HAS_TORCH_PRUNE = True
except ImportError:
    HAS_TORCH_PRUNE = False
    logger.warning("torch.nn.utils.prune not found. Pruning functionalities will be disabled.")

# Import quantization specific modules
if HAS_TORCH_QUANTIZATION:
    try:
        import torch.quantization as tq
    except ImportError:
        logger.warning("PyTorch Quantization not found. QAT/PTQ will be disabled.")
        HAS_TORCH_QUANTIZATION = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Helper for Mixup/CutMix ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    W = x.size(2)
    H = x.size(3)

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class MixupCutmixAugmenter:
    """
    Applies Mixup or CutMix augmentation to a batch of images and labels.
    """
    def __init__(self, config: Dict[str, Any], num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.mixup_alpha = config.get('mixup_alpha', 0.8)
        self.cutmix_alpha = config.get('cutmix_alpha', 1.0)
        self.mixup_prob = config.get('mixup_prob', 0.5)
        self.cutmix_prob = config.get('cutmix_prob', 0.5)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        logger.info(f"MixupCutmixAugmenter initialized. Mixup alpha: {self.mixup_alpha}, CutMix alpha: {self.cutmix_alpha}")

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        do_mixup = random.random() < self.mixup_prob
        do_cutmix = random.random() < self.cutmix_prob

        # Ensure only one is chosen if both are true
        if do_mixup and do_cutmix:
            if random.random() < 0.5:
                do_cutmix = False
            else:
                do_mixup = False

        if do_mixup:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, self.mixup_alpha, images.device)
            # Apply label smoothing to the mixed labels
            one_hot_y_a = F.one_hot(y_a, num_classes=self.num_classes).float().to(labels.device)
            one_hot_y_b = F.one_hot(y_b, num_classes=self.num_classes).float().to(labels.device)
            mixed_labels = lam * one_hot_y_a + (1 - lam) * one_hot_y_b
            smoothed_labels = mixed_labels * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            return mixed_images, smoothed_labels
        elif do_cutmix:
            mixed_images, y_a, y_b, lam = cutmix_data(images, labels, self.cutmix_alpha, images.device)
            # Apply label smoothing to the mixed labels
            one_hot_y_a = F.one_hot(y_a, num_classes=self.num_classes).float().to(labels.device)
            one_hot_y_b = F.one_hot(y_b, num_classes=self.num_classes).float().to(labels.device)
            mixed_labels = lam * one_hot_y_a + (1 - lam) * one_hot_y_b
            smoothed_labels = mixed_labels * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            return mixed_images, smoothed_labels
        else:
            # If no augmentation, apply label smoothing to original labels
            one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(labels.device)
            smoothed_labels = one_hot_labels * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            return images, smoothed_labels

# --- Quantization Functions ---
def quantize_model_qat(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """
    Prepares a model for Quantization Aware Training (QAT).
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.error("PyTorch Quantization is not available. Cannot perform QAT.")
        return model
    logger.info("Applying Quantization Aware Training (QAT)...")
    
    # Fuse modules before QAT
    # This requires specific fuse_model methods implemented in your PerceptionModule
    # or its sub-modules (e.g., AttributeModel's backbone).
    # For a ResNet backbone, fusion is typically done on the Conv-BN-ReLU blocks.
    # If your model structure doesn't have a generic `fuse_model` method,
    # you'd need to manually fuse or apply QAT to individual layers.
    # For a PyTorch Lightning module, you usually prepare the underlying `nn.Module`.
    
    # Ensure the model is on CPU for fusion
    model.cpu()

    # If the model has a custom fusion method (e.g., in PerceptionModule or AttributeModel)
    if hasattr(model, 'fuse_modules'): # Assuming a top-level fuse_modules method
        model.fuse_modules()
    else:
        logger.warning("Model does not have a 'fuse_modules' method. Attempting generic fusion.")
        # Attempt to apply fusions to common patterns if possible
        # This is a generic attempt; specific fusion might be needed per model architecture.
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                try:
                    torch.quantization.fuse_modules(module, inplace=True)
                except Exception as e:
                    logger.debug(f"Could not fuse module {name}: {e}")
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                # If there's a BN or ReLU immediately after, try to fuse
                pass # This is best handled by explicit fusion patterns

    model.qconfig = tq.get_default_qat_qconfig('fbgemm') # fbgemm for server-side, qnnpack for mobile
    tq.prepare_qat(model, inplace=True)
    
    logger.info("Model prepared for QAT. Remember to fine-tune the model with QAT enabled.")
    return model

def quantize_model_ptq(model: nn.Module, calibration_loader: Any, dali_image_processor: Any, config: Dict[str, Any]) -> nn.Module:
    """
    Applies Post-Training Quantization (PTQ) to a model.
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.error("PyTorch Quantization is not available. Cannot perform PTQ.")
        return model
    logger.info("Applying Post-Training Quantization (PTQ)...")
    
    # Ensure model is on CPU for fusion and calibration
    model.cpu()

    # Fuse modules before PTQ
    if hasattr(model, 'fuse_modules'):
        model.fuse_modules()
    else:
        logger.warning("Model does not have a 'fuse_modules' method. Attempting generic fusion.")
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                try:
                    torch.quantization.fuse_modules(module, inplace=True)
                except Exception as e:
                    logger.debug(f"Could not fuse module {name}: {e}")

    model.qconfig = tq.get_default_qconfig('fbgemm') # fbgemm for server-side, qnnpack for mobile
    tq.prepare(model, inplace=True)
    
    logger.info("PTQ: Performing calibration step...")
    # Calibration: run a few batches through the prepared model
    model.eval() # Set to eval mode for calibration
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(calibration_loader), total=len(calibration_loader), desc="PTQ Calibration"):
            # Unpack raw NumPy arrays from collate_fn output
            raw_query_images_view1_np = batch_data[0]
            query_gts_json_view1 = batch_data[3]
            raw_support_images_flat_np = batch_data[9]
            
            # Process raw images through DALI Image Processor
            processed_query_images_view1, _, processed_support_images_flat = dali_image_processor.run(
                raw_query_images_view1_np,
                [np.zeros_like(raw_query_images_view1_np[0])] * len(raw_query_images_view1_np), # Dummy view2
                raw_support_images_flat_np
            )
            # Reshape flattened support images back to (B, N_support, C, H, W)
            max_support_imgs = config['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_query_images_view1.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
            )
            
            # Run forward pass for calibration.
            _ = model(processed_query_images_view1.to(DEVICE), 
                           ground_truth_json_strings=query_gts_json_view1, 
                           support_images=processed_support_images_reshaped.to(DEVICE))
            
            if batch_idx >= config['training']['quantization'].get('calibration_batches', 10):
                break
    
    tq.convert(model, inplace=True)
    logger.info("Model quantized using Post-Training Quantization (PTQ).")
    return model

# --- Pruning Functions ---
def compute_layer_sensitivity(model: nn.Module, val_loader: Any, dali_image_processor: Any, current_rank: int = 0) -> Dict[str, float]:
    """
    Computes the sensitivity of each relevant layer to pruning by zeroing out its weights
    and measuring the increase in validation loss.
    
    Args:
        model (nn.Module): The model to analyze (should be LitBongard or its underlying PerceptionModule).
        val_loader (Any): DataLoader for validation data.
        dali_image_processor (Any): The DALIImageProcessor instance.
        current_rank (int): Current GPU rank for DDP.
    Returns:
        Dict[str, float]: A dictionary mapping layer names to their sensitivity scores (loss increase).
    """
    if current_rank != 0: # Only compute on rank 0 to avoid redundant computation
        return {}
    if not HAS_TORCH_PRUNE:
        logger.warning("torch.nn.utils.prune not available. Skipping sensitivity analysis.")
        return {}

    logger.info("Starting layer sensitivity analysis for pruning...")
    sensitivities = {}
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get base loss
    base_metrics = _validate_model(model, val_loader, dali_image_processor, current_rank, False, limit_batches=20) # Limit for speed
    base_loss = base_metrics['loss']
    logger.info(f"Base validation loss for sensitivity analysis: {base_loss:.4f}")
    
    # Access the underlying PerceptionModule if model is LitBongard
    perception_module = model.perception_module if isinstance(model, pl.LightningModule) else model
    
    # Collect convolutional layers to test (focus on backbone layers)
    conv_layers = []
    # Assuming AttributeModel's feature_extractor contains the Conv2d layers
    if hasattr(perception_module, 'attribute_model') and hasattr(perception_module.attribute_model, 'feature_extractor'):
        # Iterate through named modules of the feature_extractor
        for name, module in perception_module.attribute_model.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d) and 'downsample' not in name and 'classifier' not in name and 'head' not in name:
                conv_layers.append((f"perception_module.attribute_model.feature_extractor.{name}", module))
    
    if not conv_layers:
        logger.warning("No suitable Conv2d layers found for sensitivity analysis in AttributeModel feature_extractor.")
        return {}

    for name, module in tqdm(conv_layers, desc="Computing layer sensitivity"):
        orig_weights = module.weight.data.clone()
        
        # Zero out weights
        module.weight.data.zero_()
        
        # Evaluate model with zeroed weights
        pruned_metrics = _validate_model(model, val_loader, dali_image_processor, current_rank, False, limit_batches=10) # Limit for speed
        pruned_loss = pruned_metrics['loss']
        
        # Calculate sensitivity as increase in loss
        sensitivity_score = pruned_loss - base_loss
        sensitivities[name] = sensitivity_score
        
        # Restore original weights
        module.weight.data.copy_(orig_weights)
    
    # Sort sensitivities for logging
    sorted_sensitivities = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    logger.info("Layer sensitivities (loss increase, higher means more sensitive):")
    for name, sens in sorted_sensitivities:
        logger.info(f"  {name}: {sens:.4f}")
    logger.info("Sensitivity analysis completed.")
    return sensitivities

def apply_structured_pruning(
    model: nn.Module,
    config: Dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    dali_image_processor: Any, # Pass DALI processor
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    sensitivity_scores: Optional[Dict[str, float]] = None
) -> nn.Module:
    """
    Applies structured pruning to the model and fine-tunes it iteratively.
    """
    if not HAS_TORCH_PRUNE:
        logger.warning("torch.nn.utils.prune not available. Skipping pruning.")
        return model

    pruning_config = config['training']['pruning']
    pruning_method = pruning_config['method']
    pruning_amount = pruning_config['amount']
    pruning_iterations = pruning_config['iterations']
    fine_tune_epochs_per_iter = pruning_config['fine_tune_epochs_per_iter']
    fine_tune_lr = config['training']['learning_rate']
    use_sensitivity_analysis = pruning_config['use_sensitivity']
    pruning_target_layers_ratio = pruning_config.get('pruning_target_layers_ratio', 0.5)
    
    logger.info(f"Applying structured pruning using {pruning_method} with amount {pruning_amount} "
                f"over {pruning_iterations} iterations, fine-tuning for {fine_tune_epochs_per_iter} epochs each.")
    
    # Access the underlying PerceptionModule if model is LitBongard
    perception_module = model.perception_module if isinstance(model, pl.LightningModule) else model

    # Identify parameters to prune (focus on AttributeModel's feature_extractor)
    parameters_to_prune = []
    all_conv_layers = []
    if hasattr(perception_module, 'attribute_model') and hasattr(perception_module.attribute_model, 'feature_extractor'):
        for name, module in perception_module.attribute_model.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d) and 'downsample' not in name and 'classifier' not in name and 'head' not in name:
                all_conv_layers.append((f"perception_module.attribute_model.feature_extractor.{name}", module))
    
    if not all_conv_layers:
        logger.warning("No suitable Conv2d layers found for structured pruning. Skipping pruning.")
        return model

    if use_sensitivity_analysis and sensitivity_scores:
        # Sort layers by sensitivity (ascending, so least sensitive are first)
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1])
        
        # Select a ratio of layers to prune
        num_layers_to_prune = max(1, int(len(sorted_layers) * pruning_target_layers_ratio))
        target_layer_names = [name for name, _ in sorted_layers[:num_layers_to_prune]]
        logger.info(f"Sensitivity analysis: Targeting {num_layers_to_prune} least sensitive layers for pruning: {target_layer_names}")
        
        for name, module in all_conv_layers:
            if name in target_layer_names:
                parameters_to_prune.append((module, 'weight'))
    else:
        logger.info("Sensitivity analysis not used or scores not available. Pruning all Conv2d layers.")
        for name, module in all_conv_layers:
            parameters_to_prune.append((module, 'weight'))
    
    if not parameters_to_prune:
        logger.warning("No parameters selected for pruning. Skipping pruning.")
        return model

    for i in range(pruning_iterations):
        logger.info(f"Pruning iteration {i+1}/{pruning_iterations}...")
        
        # Apply pruning method
        for module, name in parameters_to_prune:
            if pruning_method == 'l1_unstructured':
                prune.l1_unstructured(module, name=name, amount=pruning_amount)
            elif pruning_method == 'random_unstructured':
                prune.random_unstructured(module, name=name, amount=pruning_amount)
            elif pruning_method == 'ln_structured':
                # Ensure dim is correct for structured pruning (e.g., dim=0 for output channels)
                # For Conv2d, dim=0 prunes output channels, dim=1 prunes input channels
                # L2 norm for filters (dim=0) is common for structured pruning
                prune.ln_structured(module, name=name, amount=pruning_amount, n=2, dim=0) 
            else:
                raise ValueError(f"Unsupported pruning method: {pruning_method}")
        
        # Log sparsity
        total_zero_weights = 0
        total_weights = 0
        for module, name in parameters_to_prune:
            total_zero_weights += torch.sum(getattr(module, name) == 0).item()
            total_weights += getattr(module, name).numel()
        overall_sparsity = 100. * float(total_zero_weights) / float(total_weights)
        logger.info(f"Pruning applied. Overall sparsity: {overall_sparsity:.2f}%")
        logger.info(f"Fine-tuning after pruning iteration {i+1}...")
        
        # Fine-tune the model using a simplified training loop
        fine_tune_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            dali_image_processor=dali_image_processor,
            epochs=fine_tune_epochs_per_iter,
            learning_rate=fine_tune_lr,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized
        )
        logger.info(f"Fine-tuning after pruning iteration {i+1} completed.")
        
        # Remove pruning reparameterization to make weights permanent for next iteration
        for module, name in parameters_to_prune:
            prune.remove_pruning(module, name)
        logger.info(f"Pruning reparameterization removed after iteration {i+1}.")
    logger.info("Structured pruning process completed.")
    return model

def fine_tune_model(
    model: Union[PerceptionModule, pl.LightningModule, nn.Module], # Can be LitBongard or PerceptionModule
    train_loader: Any,
    val_loader: Any,
    dali_image_processor: Any, # Pass DALI processor
    epochs: int,
    learning_rate: float,
    current_rank: int,
    is_ddp_initialized: bool
):
    """
    Helper function to fine-tune the model after pruning or for QAT.
    This is a simplified training loop, not a full PyTorch Lightning Trainer.
    """
    # If model is LitBongard, get its underlying module for optimizer
    if isinstance(model, pl.LightningModule):
        params_to_optimize = model.parameters()
    else: # Direct nn.Module or DDP wrapped
        params_to_optimize = model.parameters()

    optimizer = optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=CONFIG['training']['weight_decay'])
    
    # Use CrossEntropyLoss for Bongard classification. If labels are soft (Mixup/Cutmix), use KLDivLoss.
    bongard_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    scaler = GradScaler() if CONFIG['training']['use_amp'] else None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Set epoch for sampler if it has a set_epoch method (e.g., DistributedSampler, CurriculumSampler)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Fine-tune Epoch {epoch+1}/{epochs} (Rank {current_rank})", disable=(current_rank != 0))
        
        for batch_idx, batch_data in pbar:
            # Unpack raw NumPy arrays and other data from collate_fn output
            raw_query_images_view1_np = batch_data[0]
            query_labels = batch_data[2].long().to(DEVICE)
            query_gts_json_view1 = batch_data[3]
            raw_support_images_flat_np = batch_data[9]
            
            # Process raw images through DALI Image Processor
            processed_query_images_view1, _, processed_support_images_flat = dali_image_processor.run(
                raw_query_images_view1_np,
                [np.zeros_like(raw_query_images_view1_np[0])] * len(raw_query_images_view1_np), # Dummy view2
                raw_support_images_flat_np
            )
            
            # Reshape flattened support images back to (B, N_support, C, H, W)
            max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_query_images_view1.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
            )
            
            optimizer.zero_grad()
            with autocast(enabled=CONFIG['training']['use_amp']):
                # For fine-tuning, we only care about bongard_logits
                outputs = model(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                support_images=processed_support_images_reshaped)
                bongard_logits = outputs['bongard_logits']
                loss = bongard_criterion(bongard_logits, query_labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}", lr=optimizer.param_groups[0]['lr'])
        
        val_metrics = _validate_model(model, val_loader, dali_image_processor, current_rank, is_ddp_initialized)
        if current_rank == 0:
            logger.info(f"  Fine-tune Epoch {epoch+1} Val Accuracy: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")
            if HAS_WANDB and CONFIG['training']['use_wandb']:
                import wandb
                wandb.log({
                    f"fine_tune/loss": total_loss / num_batches,
                    f"fine_tune/val_accuracy": val_metrics['accuracy'],
                    f"fine_tune/val_loss": val_metrics['loss'],
                    "fine_tune_epoch": epoch
                })

# --- Main Training Function (Orchestrator for Lightning) ---
def _run_supervised_training_lightning(rank: int, world_size: int, cfg: Dict[str, Any]):
    """
    Main function for each DDP training process using PyTorch Lightning.
    This function sets up the data loaders, model, trainer, and initiates training.
    """
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        logger.info(f"Rank {rank} / {world_size} initialized for Lightning Trainer.")
    
    set_seed(cfg['training']['seed'] + rank) # Set different seed for each rank
    logger.info(f"Process {rank}: Starting Lightning supervised training session.")

    # --- Data Loading ---
    train_loader = get_dataloader(cfg, is_train=True, rank=rank, world_size=world_size)
    val_loader   = get_dataloader(cfg, is_train=False, rank=rank, world_size=world_size)
    
    logger.info(f"Process {rank}: Train DataLoader created with batch size {cfg['training']['batch_size']}.")
    logger.info(f"Process {rank}: Validation DataLoader created.")

    # --- Model Initialization ---
    model = LitBongard(cfg)
    logger.info(f"Process {rank}: LitBongard model initialized.")

    # --- Knowledge Distillation: Teacher Models ---
    # Load or train teacher models if knowledge distillation is enabled
    teacher_models_list = []
    if cfg['training']['use_knowledge_distillation']:
        logger.info("Knowledge Distillation enabled. Setting up teacher models.")
        if cfg['ensemble']['train_members']:
            logger.info("Training teacher models (ensemble members).")
            # For simplicity, we'll train teachers sequentially here or assume they are pre-trained.
            # In a real scenario, you might train them in parallel or load from checkpoints.
            # For this context, we will load them if paths are provided, otherwise assume they are trained.
            # If no paths, we'll just use the current model as a dummy teacher for now.
            if cfg['ensemble']['teacher_model_paths']:
                for i, path in enumerate(cfg['ensemble']['teacher_model_paths']):
                    teacher_m = PerceptionModule(cfg)
                    try:
                        teacher_m.load_state_dict(torch.load(path, map_location=DEVICE)['model_state_dict'])
                        teacher_models_list.append(teacher_m.to(DEVICE))
                        logger.info(f"Loaded teacher model from {path}")
                    except Exception as e:
                        logger.error(f"Failed to load teacher model from {path}: {e}. Skipping this teacher.")
            else:
                logger.warning("No teacher model paths provided. Using current student model as a dummy teacher for distillation.")
                # If no pre-trained teachers, use a copy of the student as a dummy teacher for demonstration
                # In a real scenario, this would be a separate, pre-trained model.
                teacher_models_list.append(copy.deepcopy(model.perception_module).to(DEVICE))
        else:
            logger.info("Loading pre-trained teacher models.")
            if cfg['ensemble']['teacher_model_paths']:
                for i, path in enumerate(cfg['ensemble']['teacher_model_paths']):
                    teacher_m = PerceptionModule(cfg)
                    try:
                        teacher_m.load_state_dict(torch.load(path, map_location=DEVICE)['model_state_dict'])
                        teacher_models_list.append(teacher_m.to(DEVICE))
                        logger.info(f"Loaded teacher model from {path}")
                    except Exception as e:
                        logger.error(f"Failed to load teacher model from {path}: {e}. Skipping this teacher.")
            else:
                logger.error("Knowledge distillation enabled but no teacher models to load and 'train_members' is false.")
    
    # Pass teacher models to LitBongard if distillation is enabled
    if cfg['training']['use_knowledge_distillation']:
        model.teacher_models = nn.ModuleList(teacher_models_list) # Assign to LitBongard instance

    # --- Callbacks ---
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg['debug']['save_model_checkpoints'],
            filename='best_bongard_model',
            monitor='val/loss', # Monitor validation loss
            mode='min',
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    if cfg['training'].get('early_stop_patience', 0) > 0:
        callbacks.append(EarlyStopping(
            monitor='val/loss', # Monitor validation loss
            patience=cfg['training']['early_stop_patience'],
            mode='min'
        ))
        logger.info(f"Early Stopping enabled with patience: {cfg['training']['early_stop_patience']}.")

    # --- PyTorch Lightning Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg['training']['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None, # Use 1 device per process
        precision=16 if cfg['training']['use_amp'] else 32,
        logger=TensorBoardLogger("lightning_logs/", name="bongard_solver"),
        callbacks=callbacks,
        strategy="ddp" if world_size > 1 else "auto", # Use DDP strategy for multi-GPU
        log_every_n_steps=cfg['training']['log_interval_batches'],
        val_check_interval=cfg['training']['validation_frequency_batches'] if val_loader else 1.0, # Validate every N batches or every epoch
        gradient_clip_val=cfg['training'].get('max_grad_norm', 0.0) # Gradient clipping
    )
    logger.info(f"Process {rank}: PyTorch Lightning Trainer initialized.")

    # --- Fit the model ---
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Process {rank}: Training finished.")

    if world_size > 1:
        dist.destroy_process_group()
        logger.info(f"Process {rank}: Distributed process group destroyed.")

# --- Main execution (orchestrates the pipeline) ---
def run_training_pipeline(cfg: Dict[str, Any]):
    """
    Main entry point for the Bongard Solver training pipeline.
    Handles SimCLR pretraining and then supervised Bongard training.
    """
    logger.info("--- Starting Bongard Solver Training Pipeline ---")

    # 1. SimCLR Pretraining (if enabled)
    if cfg['model']['simclr_config']['enabled'] and cfg['model']['simclr_config']['pretrain_epochs'] > 0:
        logger.info("Initiating SimCLR pretraining.")
        run_simclr_pretraining(cfg)
        logger.info("SimCLR pretraining completed.")
        # After pretraining, the best SimCLR model weights are saved.
        # The LitBongard model's backbone can then be initialized with these weights
        # if the checkpoint loading logic is set up to do so.
        # For now, we assume this is handled implicitly or by manual loading.

    # 2. Supervised Bongard Training
    logger.info("Initiating supervised Bongard problem training.")
    world_size = torch.cuda.device_count()
    if world_size > 1:
        logger.info(f"Found {world_size} GPUs. Starting distributed training.")
        mp.spawn(_run_supervised_training_lightning, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        logger.info("Running on a single GPU/CPU.")
        _run_supervised_training_lightning(0, 1, cfg)
    
    logger.info("--- Bongard Solver Training Pipeline finished. ---")

# This function is used by Optuna for HPO
def run_training_once(cfg: Dict[str, Any], epochs: int) -> float:
    """
    Runs a single training session for HPO.
    Returns validation accuracy.
    """
    original_epochs = cfg['training']['epochs']
    cfg['training']['epochs'] = epochs # Temporarily set epochs for HPO trial
    
    # Set up a minimal trainer for HPO
    model = LitBongard(cfg)
    train_loader = get_dataloader(cfg, is_train=True, rank=0, world_size=1)
    val_loader = get_dataloader(cfg, is_train=False, rank=0, world_size=1)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False, # Disable logging for HPO trials
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[
            EarlyStopping(monitor='val/loss', patience=2, mode='min') # Shorter patience for HPO
        ]
    )
    trainer.fit(model, train_loader, val_loader)
    
    # Get best validation accuracy from trainer.callback_metrics
    # This assumes val/accuracy is logged
    val_acc = trainer.callback_metrics.get("val/accuracy", torch.tensor(0.0)).item()
    
    cfg['training']['epochs'] = original_epochs # Restore original epochs
    return val_acc

# --- Validation Function (used by pruning and sensitivity analysis) ---
def _validate_model(
    model: Union[PerceptionModule, pl.LightningModule, nn.Module],
    val_loader: Any,
    dali_image_processor: Any, # Pass DALI processor
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    limit_batches: Optional[int] = None # For sensitivity analysis
) -> Dict[str, float]:
    """
    Validates a single model (PerceptionModule or LitBongard).
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Loss functions for validation
    bongard_criterion = nn.CrossEntropyLoss(reduction='mean')
    attribute_criterion = nn.CrossEntropyLoss(reduction='mean')
    relation_criterion = nn.CrossEntropyLoss(reduction='mean')
    MAX_GNN_OBJECTS = CONFIG['model']['MAX_GNN_OBJECTS']
    E_max = MAX_GNN_OBJECTS * (MAX_GNN_OBJECTS - 1)

    # Set epoch for sampler (if using DistributedSampler or CurriculumSampler)
    if hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(0) # For validation, epoch doesn't change, but set_epoch might do other setup

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                    desc=f"Validating (Rank {current_rank})", disable=(current_rank != 0))
        for batch_idx, batch_data in pbar:
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            
            # Unpack raw NumPy arrays and other data from collate_fn output
            raw_query_images_view1_np = batch_data[0]
            query_labels = batch_data[2].long().to(DEVICE)
            query_gts_json_view1 = batch_data[3]
            raw_support_images_flat_np = batch_data[9]
            
            # Process raw images through DALI Image Processor
            processed_query_images_view1, _, processed_support_images_flat = dali_image_processor.run(
                raw_query_images_view1_np,
                [np.zeros_like(raw_query_images_view1_np[0])] * len(raw_query_images_view1_np), # Dummy view2 for DALI
                raw_support_images_flat_np
            )
            # Reshape flattened support images back to (B, N_support, C, H, W)
            max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_query_images_view1.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
            )
            
            # If `model` is LitBongard, call its forward, otherwise call PerceptionModule directly
            if isinstance(model, pl.LightningModule):
                outputs = model(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                support_images=processed_support_images_reshaped)
            else: # Assume it's PerceptionModule or DDP(PerceptionModule)
                outputs = model(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                support_images=processed_support_images_reshaped)

            bongard_logits = outputs['bongard_logits']
            attribute_logits = outputs['attribute_logits']
            relation_logits = outputs['relation_logits']
            scene_graphs = outputs['scene_graphs']

            # Bongard Loss
            loss_bongard = bongard_criterion(bongard_logits, query_labels)
            
            # Attribute Loss (similar to training, but simplified for validation)
            loss_attribute = torch.tensor(0.0, device=DEVICE)
            num_attribute_losses = 0
            current_flat_idx = 0
            for i_img in range(len(scene_graphs)):
                sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                inferred_objects_for_img = scene_graphs[i_img].get('objects', [])
                for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                    gt_obj = None
                    inferred_bbox = inferred_obj.get('bbox')
                    if inferred_bbox:
                        for gt_o in sg_gt['objects']:
                            if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                gt_obj = gt_o
                                break
                    if gt_obj:
                        for attr_name, num_classes in CONFIG['model']['attribute_classifier_config'].items():
                            if attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
                                if current_flat_idx < attribute_logits[attr_name].shape[0]:
                                    attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                    if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                        gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                        predicted_logits = attribute_logits[attr_name][current_flat_idx].unsqueeze(0)
                                        loss_attribute += attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                        num_attribute_losses += 1
                    current_flat_idx += 1
            
            if num_attribute_losses > 0:
                loss_attribute /= num_attribute_losses
            
            # Relation Loss
            loss_relation = torch.tensor(0.0, device=DEVICE)
            if relation_logits.numel() > 0:
                B_current, _, R = relation_logits.shape
                gt_edge_labels = torch.full((B_current, E_max), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                for b in range(B_current):
                    sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                    num_gt_objects = len(sg_gt['objects'])
                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                    for rel in sg_gt['relations']:
                        subj_id = rel['subject_id']
                        obj_id = rel['object_id']
                        rel_type = rel['type']
                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                            gt_edge_labels[b, edge_idx_flat] = RELATION_MAP[rel_type]
                logits_flat = relation_logits.view(-1, R)
                labels_flat = gt_edge_labels.view(-1)
                loss_relation = relation_criterion(logits_flat, labels_flat)

            # Total Loss
            current_batch_loss = loss_bongard + loss_attribute + CONFIG['training']['relation_loss_weight'] * loss_relation
            total_loss += current_batch_loss.item()

            # Accuracy
            predictions = torch.argmax(bongard_logits, dim=1)
            correct_predictions += (predictions == query_labels).sum().item()
            total_samples += query_labels.size(0)
            pbar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}", acc=f"{correct_predictions / total_samples:.4f}")

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples

    # If using DDP, gather metrics from all ranks
    if is_ddp_initialized:
        gathered_loss = [torch.tensor(0.0, device=DEVICE) for _ in range(dist.get_world_size())]
        gathered_correct = [torch.tensor(0.0, device=DEVICE) for _ in range(dist.get_world_size())]
        gathered_total = [torch.tensor(0.0, device=DEVICE) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_loss, torch.tensor(total_loss, device=DEVICE))
        dist.all_gather(gathered_correct, torch.tensor(correct_predictions, device=DEVICE))
        dist.all_gather(gathered_total, torch.tensor(total_samples, device=DEVICE))
        
        total_loss = sum(l.item() for l in gathered_loss)
        correct_predictions = sum(c.item() for c in gathered_correct)
        total_samples = sum(t.item() for t in gathered_total)
        
        avg_loss = total_loss / (len(val_loader) * dist.get_world_size())
        accuracy = correct_predictions / total_samples
    
    return {'loss': avg_loss, 'accuracy': accuracy}

