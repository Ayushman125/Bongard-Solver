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
import random # For MixupCutmix
import numpy as np # For MixupCutmix
import json # For parsing GT JSON
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image # For saving synthetic images for Grad-CAM
import torchvision.transforms as T # For converting tensor to PIL Image for Grad-CAM
import threading # For async updates (conceptual)

# Import for pruning
import torch.nn.utils.prune as prune

# Import configuration
from config import (
    CONFIG, DEVICE, HAS_TIMM_OPTIM, HAS_TORCH_QUANTIZATION, HAS_WANDB,
    ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
    ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
    RELATION_MAP
)

# Import from utils
from utils import set_seed, get_symbolic_embedding_dims, make_edge_index_map

# Import from data
from data import build_dali_image_processor, BongardSyntheticDataset, CurriculumSampler, load_bongard_data, RealBongardDataset, BongardGenerator, _calculate_iou # Added _calculate_iou

# Import bongard_rules
from bongard_rules import ALL_BONGARD_RULES, BongardRule # Added BongardRule for RL

# Import from models
from models import PerceptionModule, SimCLREncoder # Added SimCLREncoder

# Import from losses
from losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, DistillationLoss, SymbolicConsistencyLoss, FocalLoss, NTXentLoss # Added NTXentLoss

# Import from replay_buffer
from replay_buffer import KnowledgeReplayBuffer

# Import from xai (for Grad-CAM)
from xai import generate_grad_cam

# Import optimizers and schedulers from the new optimizers.py module
from optimizers import get_optimizer, get_scheduler, SAM # SAM and SophiaG imported here

# Import rule_evaluator
from rule_evaluator import find_best_rules, evaluate_rule_on_support_set

# Import new symbolic_engine and rl_module
from symbolic_engine import SymbolicEngine, RuleInducer, AlephRuleInducer # Added for DSL/ILP
from rl_module import BongardEnv, RulePolicy, RuleGraphSpace # Added for RL

logger = logging.getLogger(__name__)

# Conditional imports for quantization
if HAS_TORCH_QUANTIZATION:
    try:
        import torch.quantization as tq
    except ImportError:
        logger.warning("PyTorch Quantization not found. QAT/PTQ will be disabled.")
        HAS_TORCH_QUANTIZATION = False


# --- Quantization Functions ---
def quantize_model_qat(model: nn.Module, config: Dict[str, Any]) -> nn.Module:
    if not HAS_TORCH_QUANTIZATION:
        logger.error("PyTorch Quantization is not available. Cannot perform QAT.")
        return model

    logger.info("Applying Quantization Aware Training (QAT)...")
    
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model
    
    # Fuse modules before QAT
    if hasattr(base_model, 'fuse_model'):
        base_model.fuse_model()
    else:
        logger.warning("Model does not have a 'fuse_model' method. Fusion skipped for QAT.")

    base_model.qconfig = tq.get_default_qat_qconfig('fbgemm') # fbgemm for server-side, qnnpack for mobile
    tq.prepare_qat(base_model, inplace=True)
    
    logger.info("Model prepared for QAT. Remember to fine-tune the model with QAT enabled.")
    return model

def quantize_model_ptq(model: nn.Module, calibration_loader: Any, config: Dict[str, Any]) -> nn.Module:
    if not HAS_TORCH_QUANTIZATION:
        logger.error("PyTorch Quantization is not available. Cannot perform PTQ.")
        return model

    logger.info("Applying Post-Training Quantization (PTQ)...")

    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    # Fuse modules before PTQ
    if hasattr(base_model, 'fuse_model'):
        base_model.fuse_model()
    else:
        logger.warning("Model does not have a 'fuse_model' method. Fusion skipped for PTQ.")

    base_model.qconfig = tq.get_default_qconfig('fbgemm') # fbgemm for server-side, qnnpack for mobile
    tq.prepare(base_model, inplace=True)

    logger.info("PTQ: Performing calibration step...")
    # Calibration: run a few batches through the prepared model
    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(calibration_loader), total=len(calibration_loader), desc="PTQ Calibration"):
            # Assuming calibration_loader yields images for calibration
            # For Bongard, use query_images_view1
            # Need to get raw images and process through DALI first
            raw_images_view1_np = batch_data[0] # List of numpy arrays
            raw_images_view2_np = batch_data[1] # List of numpy arrays
            raw_support_images_flat_np = batch_data[9] # List of numpy arrays

            # Assuming calibration_loader's collate_fn provides raw numpy arrays
            # and we need a DALIImageProcessor for calibration
            # This function needs to receive dali_image_processor
            
            # For now, let's simplify calibration to just use the first view of query images
            # and assume they are already augmented/normalized if PTQ is called without DALI processor.
            # A proper PTQ calibration would involve DALI processor here.
            # For this context, we'll assume `calibration_loader` provides already processed tensors
            # or that the `dali_image_processor` is passed.
            
            # If `calibration_loader` provides raw images, we need a DALI processor here.
            # If it provides processed tensors, then the `dali_image_processor` from `run_training_process`
            # would need to be passed to this function.
            
            # For simplicity, if `calibration_loader` is a standard PyTorch DataLoader,
            # it will yield processed tensors if the collate_fn includes a DALI processor.
            # If not, this section needs a `dali_image_processor` argument.
            
            # Let's assume the `calibration_loader` provides the processed tensors directly for calibration.
            # This implies the `main.py`'s `val_loader` (which uses DALI processor via collate_fn)
            # is passed as `calibration_loader`.
            
            # Unpack processed tensors from batch_data (assuming collate_fn has processed them)
            # If `calibration_loader` is the `val_loader` from `main.py`, its `collate_fn`
            # will have already run DALI.
            images_view1 = batch_data[0].float().to(DEVICE) # This would be the *processed* tensor

            # Only need to run forward pass, no need for full Bongard problem inference
            # The model's forward needs to be adaptable for this.
            # Here, we'll just pass the images to the backbone for calibration.
            # If the model is a DDP model, access the base_model.
            if hasattr(base_model.attribute_classifier, 'feature_extractor'):
                _ = base_model.attribute_classifier.feature_extractor(images_view1)
            else:
                _ = base_model(images_view1, ground_truth_json_strings=[], support_images=torch.empty(0,0,0,0,0)) # Dummy inputs for full model
            
            if batch_idx >= config['training']['pruning_config'].get('calibration_batches', 10): # Calibrate for a few batches
                break


    tq.convert(base_model, inplace=True)
    
    logger.info("Model quantized using Post-Training Quantization (PTQ).")
    return model

# --- Pruning Functions ---
def compute_layer_sensitivity(model: nn.Module, val_loader: Any, dali_image_processor: Any, current_rank: int = 0) -> Dict[str, float]:
    """
    Computes the sensitivity of each convolutional layer to pruning by zeroing out its weights
    and measuring the increase in validation loss.
    
    Args:
        model (nn.Module): The model to analyze.
        val_loader (Any): DataLoader for validation data.
        dali_image_processor (Any): The DALIImageProcessor instance.
        current_rank (int): Current GPU rank for DDP.

    Returns:
        Dict[str, float]: A dictionary mapping layer names to their sensitivity scores (loss increase).
    """
    if current_rank != 0: # Only compute on rank 0 to avoid redundant computation
        return {}

    logger.info("Starting layer sensitivity analysis for pruning...")
    sensitivities = {}
    
    # Ensure model is in eval mode
    model.eval()
    
    # Get base loss
    base_metrics = _validate_model_ensemble(model, val_loader, dali_image_processor, current_rank, False, 0) # is_ddp_initialized=False for base model
    base_loss = base_metrics['loss']
    logger.info(f"Base validation loss for sensitivity analysis: {base_loss:.4f}")

    # Access the base model if it's DDP wrapped
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    # Collect convolutional layers to test
    conv_layers = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' not in name and 'classifier' not in name and 'head' not in name:
            conv_layers.append((name, module))
    
    if not conv_layers:
        logger.warning("No Conv2d layers found for sensitivity analysis.")
        return {}

    for name, module in tqdm(conv_layers, desc="Computing layer sensitivity"):
        orig_weights = module.weight.data.clone()
        
        # Zero out weights
        module.weight.data.zero_()
        
        # Evaluate model with zeroed weights
        pruned_metrics = _validate_model_ensemble(model, val_loader, dali_image_processor, current_rank, False, 0, limit_batches=10) # Limit for speed
        pruned_loss = pruned_metrics['loss']
        
        # Calculate sensitivity as increase in loss
        sensitivity = pruned_loss - base_loss
        sensitivities[name] = sensitivity
        
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
    model_idx: int = 0,
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    sensitivity_scores: Optional[Dict[str, float]] = None
) -> nn.Module:
    pruning_config = config['pruning_config']
    pruning_method = pruning_config['method']
    pruning_amount = pruning_config['amount']
    pruning_iterations = pruning_config['iterations']
    fine_tune_epochs_per_iter = pruning_config['fine_tune_epochs_per_iter']
    fine_tune_lr = pruning_config['fine_tune_lr']
    use_sensitivity_analysis = pruning_config['use_sensitivity_analysis']
    pruning_target_layers_ratio = pruning_config['pruning_target_layers_ratio']

    logger.info(f"Applying structured pruning using {pruning_method} with amount {pruning_amount} "
                f"over {pruning_iterations} iterations, fine-tuning for {fine_tune_epochs_per_iter} epochs each.")

    # Access the base model if it's DDP wrapped
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    # Identify parameters to prune
    parameters_to_prune = []
    all_conv_layers = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' not in name and 'classifier' not in name and 'head' not in name:
            all_conv_layers.append((name, module))
    
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
                # For structured pruning (e.g., filter pruning), dim=0 for Conv2d weights
                prune.ln_structured(module, name=name, amount=pruning_amount, n=2, dim=0) # L2 norm for filters
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
        fine_tune_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            dali_image_processor=dali_image_processor, # Pass DALI processor
            epochs=fine_tune_epochs_per_iter,
            learning_rate=fine_tune_lr,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=model_idx
        )
        logger.info(f"Fine-tuning after pruning iteration {i+1} completed.")

        # Remove pruning reparameterization to make weights permanent for next iteration
        for module, name in parameters_to_prune:
            prune.remove_pruning(module, name)
        logger.info(f"Pruning reparameterization removed after iteration {i+1}.")

    logger.info("Structured pruning process completed.")
    return model

def fine_tune_model(
    model: Union[PerceptionModule, DDP],
    train_loader: Any,
    val_loader: Any,
    dali_image_processor: Any, # Pass DALI processor
    epochs: int,
    learning_rate: float,
    current_rank: int,
    is_ddp_initialized: bool,
    model_idx: int
):
    """
    Helper function to fine-tune the model after pruning or for QAT.
    """
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=CONFIG['training']['weight_decay'])
    criterion = LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing'])
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
            # raw_query_images_view2_np = batch_data[1] # Not needed for fine-tuning loss
            query_labels = batch_data[2].long().to(DEVICE)
            query_gts_json_view1 = batch_data[3]
            # query_gts_json_view2 = batch_data[4] # Not needed for fine-tuning loss
            # difficulties = batch_data[5]
            # affine1 = batch_data[6]
            # affine2 = batch_data[7]
            # original_indices = batch_data[8]
            raw_support_images_flat_np = batch_data[9]
            # support_labels_flat = batch_data[10]
            # support_sgs_flat_bytes = batch_data[11]
            num_support_per_problem = batch_data[12] # Tensor of counts

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
                outputs = model(processed_query_images_view1, ground_truth_json_strings=query_gts_json_view1,
                                support_images=processed_support_images_reshaped)
                bongard_logits = outputs['bongard_logits']
                loss = criterion(bongard_logits, query_labels)
            
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

        val_metrics = _validate_model_ensemble(model, val_loader, dali_image_processor, current_rank, is_ddp_initialized, model_idx)
        if current_rank == 0:
            logger.info(f"  Fine-tune Epoch {epoch+1} Val Accuracy: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")
            if HAS_WANDB and CONFIG['training']['use_wandb']:
                import wandb
                wandb.log({
                    f"fine_tune/loss_member_{model_idx}": total_loss / num_batches,
                    f"fine_tune/val_accuracy_member_{model_idx}": val_metrics['accuracy'],
                    f"fine_tune/val_loss_member_{model_idx}": val_metrics['loss'],
                    "fine_tune_epoch": epoch
                })

# --- MixupCutmix Class (unchanged, omitted for brevity) ---
class MixupCutmix:
    def __init__(self, config: Dict[str, Any], num_classes: int):
        self.config = config
        self.num_classes = num_classes
        self.mixup_alpha = config.get('mixup_alpha', 0.8)
        self.cutmix_alpha = config.get('cutmix_alpha', 1.0)
        self.mixup_prob = config.get('mixup_prob', 0.5)
        self.cutmix_prob = config.get('cutmix_prob', 0.5)
        self.label_smoothing = config.get('label_smoothing', 0.1)
        logger.info(f"MixupCutmix initialized. Mixup alpha: {self.mixup_alpha}, CutMix alpha: {self.cutmix_alpha}")

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        do_mixup = random.random() < self.mixup_prob
        do_cutmix = random.random() < self.cutmix_prob

        if do_mixup and do_cutmix:
            if random.random() < 0.5:
                do_cutmix = False
            else:
                do_mixup = False

        if do_mixup:
            return self._mixup(images, labels)
        elif do_cutmix:
            return self._cutmix(images, labels)
        else:
            one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(labels.device)
            smoothed_labels = one_hot_labels * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
            return images, smoothed_labels
            
    def _mixup(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(labels.device)
        mixed_labels = lam * one_hot_labels + (1 - lam) * one_hot_labels[index, :]
        
        return mixed_images, mixed_labels

    def _cutmix(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        r_x = np.random.uniform(0, images.shape[3])
        r_y = np.random.uniform(0, images.shape[2])
        r_w = images.shape[3] * np.sqrt(1 - lam)
        r_h = images.shape[2] * np.sqrt(1 - lam)
        r_x = int(r_x - r_w / 2)
        r_y = int(r_y - r_h / 2)
        r_w = int(r_w)
        r_h = int(r_h)

        x1 = np.clip(r_x, 0, images.shape[3])
        y1 = np.clip(r_y, 0, images.shape[2])
        x2 = np.clip(r_x + r_w, 0, images.shape[3])
        y2 = np.clip(r_y + r_h, 0, images.shape[2])

        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.shape[2] * images.shape[3]))

        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(labels.device)
        mixed_labels = lam * one_hot_labels + (1 - lam) * one_hot_labels[index, :]

        return images, mixed_labels


# Placeholder for asynchronous priority update
def async_update_priorities(replay_buffer: KnowledgeReplayBuffer, tree_indices: List[int], losses: List[float]):
    """
    Initiates an asynchronous update of priorities in the replay buffer.
    """
    def _target_function():
        replay_buffer.update_priorities(tree_indices, losses)
    
    thread = threading.Thread(target=_target_function)
    thread.daemon = True # Allow the main program to exit even if thread is running
    thread.start()
    logger.debug(f"Initiated async priority update for {len(tree_indices)} samples.")


def _get_ensemble_teacher_logits(
    teacher_models: List[nn.Module],
    raw_images_np: List[np.ndarray], # Raw NumPy images for DALI
    raw_gt_json_strings: List[bytes], # Raw GT JSON for teacher's forward
    raw_support_images_np: List[np.ndarray], # Raw NumPy support images for DALI
    distillation_config: Dict[str, Any],
    dali_image_processor: Any # DALI processor for teacher's input
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Computes teacher logits from an ensemble of models, applying dropout and
    optionally generating a distillation mask.

    Args:
        teacher_models (List[nn.Module]): A list of teacher models (ensemble members).
        raw_images_np (List[np.ndarray]): Raw input images (NumPy arrays) for which to get logits.
        raw_gt_json_strings (List[bytes]): Raw Ground truth JSON strings for the batch (for teacher's forward pass).
        raw_support_images_np (List[np.ndarray]): Raw support images (NumPy arrays) for the batch.
        distillation_config (Dict[str, Any]): Distillation configuration.
        dali_image_processor (Any): The DALIImageProcessor instance to augment teacher inputs.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
        - Averaged teacher logits (Batch_size, Num_classes).
        - Optional distillation mask (Batch_size,) if use_mask_distillation is True, else None.
    """
    logits_list = []
    
    ensemble_dropout_prob = distillation_config.get('ensemble_dropout_prob', 0.0)
    use_mask_distillation = distillation_config.get('use_mask_distillation', False)
    mask_agreement_threshold = distillation_config.get('mask_agreement_threshold', 0.9)

    # Process raw images through DALI Image Processor for teacher inputs
    # Use dummy view2 and support images for the DALI call if only view1 is needed for teacher inference
    processed_query_images_view1, _, processed_support_images_flat = dali_image_processor.run(
        raw_images_np,
        [np.zeros_like(raw_images_np[0])] * len(raw_images_np), # Dummy view2
        raw_support_images_np
    )

    # Reshape flattened support images back to (B, N_support, C, H, W)
    max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
    batch_size_actual = processed_query_images_view1.shape[0]
    processed_support_images_reshaped = processed_support_images_flat.view(
        batch_size_actual, max_support_imgs, 
        processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
    )


    for teacher_model in teacher_models:
        # Randomly drop a teacher model for diversity
        if random.random() > ensemble_dropout_prob:
            teacher_model.eval() # Ensure teacher is in eval mode for inference
            with torch.no_grad():
                # Teacher model's forward pass
                outputs = teacher_model(processed_query_images_view1, ground_truth_json_strings=raw_gt_json_strings,
                                        support_images=processed_support_images_reshaped)
                logits = outputs['bongard_logits']
                logits_list.append(logits)
            # No need to set back to train mode, as teachers are always eval during student training

    if not logits_list:
        logger.warning("No teacher models selected for ensemble logits. Returning zero logits.")
        # Return zero tensor with correct shape
        num_classes = CONFIG['model']['bongard_head_config']['num_classes']
        return torch.zeros(raw_images_np.shape[0], num_classes, device=DEVICE), None # Use raw_images_np for batch size

    stacked_logits = torch.stack(logits_list) # (N_selected_teachers, Batch_size, Num_classes)
    averaged_logits = stacked_logits.mean(dim=0) # (Batch_size, Num_classes)

    distillation_mask = None
    if use_mask_distillation:
        # Compute probabilities from each teacher's logits
        all_teacher_probs = torch.stack([F.softmax(l, dim=1) for l in logits_list]) # (N_selected_teachers, B, C)
        
        # Average probabilities (not logits for mask calculation)
        avg_probs = all_teacher_probs.mean(dim=0) # (B, C)
        
        # Get the max probability and corresponding class for each sample
        max_probs, predicted_classes = avg_probs.max(dim=1)
        
        # Create a mask where agreement is high
        # A simple agreement check: if the max probability is above a threshold
        distillation_mask = (max_probs > mask_agreement_threshold).float() # (B,)
        
    return averaged_logits, distillation_mask


def _run_single_training_session_ensemble(
    model: Union[PerceptionModule, DDP],
    train_loader: Any, # PyTorch DataLoader
    val_loader: Any,
    optimizer: optim.Optimizer,
    scheduler: Optional[Union[OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR]] = None,
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    model_idx: int = 0, # For ensemble member tracking
    replay_buffer: Optional[KnowledgeReplayBuffer] = None,
    start_epoch: int = 0,
    total_epochs: int = CONFIG['training']['epochs'],
    teacher_models: Optional[List[nn.Module]] = None, # Changed to list of teacher models
    dali_image_processor: Any = None # DALI processor for image augmentation
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Runs a single training session for one model/ensemble member.
    Handles training, validation, logging, and checkpointing.

    Args:
        model: The model to train (can be DDP wrapped).
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for validation.
        optimizer: The optimizer.
        scheduler: Optional learning rate scheduler.
        current_rank (int): Current GPU rank for DDP.
        is_ddp_initialized (bool): True if DDP is already initialized.
        model_idx (int): Index of the current ensemble member.
        replay_buffer (Optional[KnowledgeReplayBuffer]): PER buffer for curriculum.
        start_epoch (int): Epoch to start training from.
        total_epochs (int): Total number of epochs for this training session.
        teacher_models (Optional[List[nn.Module]]): A list of teacher models for knowledge distillation.
        dali_image_processor (Any): The DALIImageProcessor instance for GPU augmentation.

    Returns:
        Tuple[float, float, Dict[str, Any]]: Best validation accuracy, best validation loss,
                                              and dictionary of best metrics.
    """
    training_config = CONFIG['training']
    model_config = CONFIG['model']
    debug_config = CONFIG['debug']

    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    best_metrics = {}

    scaler = GradScaler() if training_config['use_amp'] else None
    
    # Initialize loss functions
    # Bongard criterion is always 'none' reduction if PER is used, so it returns per-sample losses
    bongard_criterion = LabelSmoothingCrossEntropy(smoothing=training_config['label_smoothing'], reduction='none')
    
    attribute_criterion = LabelSmoothingCrossEntropy(smoothing=training_config['label_smoothing'])
    
    relation_criterion = LabelSmoothingCrossEntropy(smoothing=training_config['label_smoothing'], reduction='mean')

    feature_consistency_criterion = FeatureConsistencyLoss(loss_type='mse')
    
    # Symbolic Consistency Loss
    symbolic_consistency_criterion = SymbolicConsistencyLoss(
        all_bongard_rules=ALL_BONGARD_RULES,
        loss_weight=training_config['consistency_loss_weight']
    )

    # Distillation loss (if applicable)
    distillation_criterion = None
    if training_config['use_knowledge_distillation']:
        if not teacher_models: # Check if list is empty or None
            logger.error("Knowledge distillation enabled but no teacher_models provided.")
        else:
            distillation_criterion = DistillationLoss(
                temperature=training_config['distillation_config']['temperature'],
                alpha=training_config['distillation_config']['alpha'],
                reduction='none' # Set reduction to 'none' for per-sample losses for masking
            )

    # SimCLR Contrastive Loss
    nt_xent_criterion = NTXentLoss(temperature=model_config['simclr_config']['temperature'])

    # SWA setup
    if training_config['use_swa'] and current_rank == 0:
        swa_model = swa_utils.AveragedModel(model.module if is_ddp_initialized else model)
        swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=training_config['swa_lr'])
        logger.info(f"SWA enabled from epoch {training_config['swa_start_epoch']} with LR {training_config['swa_lr']}.")
    else:
        swa_model = None
        swa_scheduler = None

    mixup_cutmix_augmenter = MixupCutmix(training_config, num_classes=CONFIG['model']['bongard_head_config']['num_classes'])

    # Pre-compute the edge index map (needed for GT edge labels)
    MAX_GNN_OBJECTS = CONFIG['model']['MAX_GNN_OBJECTS']
    edge_index_map = make_edge_index_map(MAX_GNN_OBJECTS)
    E_max = MAX_GNN_OBJECTS * (MAX_GNN_OBJECTS - 1)

    # Initialize SymbolicEngine for DSL/ILP (if needed for rule induction outside RL)
    symbolic_engine = SymbolicEngine(CONFIG)


    # --- SimCLR Pretraining Phase ---
    if model_config['simclr_config']['enabled'] and model_config['simclr_config']['pretrain_epochs'] > 0:
        logger.info(f"Starting SimCLR pretraining for {model_config['simclr_config']['pretrain_epochs']} epochs...")
        simclr_optimizer = get_optimizer(
            model=model.module.simclr_encoder if is_ddp_initialized else model.simclr_encoder,
            optimizer_name=training_config['optimizer'],
            learning_rate=model_config['simclr_config']['pretrain_lr'],
            weight_decay=training_config['weight_decay']
        )
        simclr_scaler = GradScaler() if training_config['use_amp'] else None

        for pretrain_epoch in range(model_config['simclr_config']['pretrain_epochs']):
            model.train()
            total_simclr_loss = 0.0
            num_simclr_batches = 0
            
            # Set epoch for sampler (if using DistributedSampler or CurriculumSampler)
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(pretrain_epoch)

            pbar_simclr = tqdm(enumerate(train_loader), total=len(train_loader), 
                                desc=f"SimCLR Pretrain Epoch {pretrain_epoch+1}", disable=(current_rank != 0))

            for batch_idx, batch_data in pbar_simclr:
                # Unpack raw NumPy arrays from collate_fn output
                raw_images_view1_np = batch_data[0]
                raw_images_view2_np = batch_data[1]
                # Other batch_data elements are not directly used for SimCLR loss

                # Process raw images through DALI Image Processor
                processed_images_view1, processed_images_view2, _ = dali_image_processor.run(
                    raw_images_view1_np,
                    raw_images_view2_np,
                    [np.zeros_like(raw_images_view1_np[0])] * len(raw_images_view1_np) # Dummy support images for DALI call
                )

                simclr_optimizer.zero_grad()

                with autocast(enabled=training_config['use_amp']):
                    simclr_embeddings1 = model(processed_images_view1, is_simclr_pretraining=True)['simclr_embeddings']
                    simclr_embeddings2 = model(processed_images_view2, is_simclr_pretraining=True)['simclr_embeddings']
                    
                    loss_simclr = nt_xent_criterion(simclr_embeddings1, simclr_embeddings2)

                if simclr_scaler:
                    simclr_scaler.scale(loss_simclr).backward()
                    simclr_scaler.step(simclr_optimizer)
                    simclr_scaler.update()
                else:
                    loss_simclr.backward()
                    simclr_optimizer.step()
                
                total_simclr_loss += loss_simclr.item()
                num_simclr_batches += 1
                pbar_simclr.set_postfix(simclr_loss=f"{total_simclr_loss / num_simclr_batches:.4f}")

            avg_simclr_loss = total_simclr_loss / num_simclr_batches
            logger.info(f"SimCLR Pretrain Epoch {pretrain_epoch+1} - Avg Loss: {avg_simclr_loss:.4f}")
            if HAS_WANDB and CONFIG['training']['use_wandb']:
                import wandb
                wandb.log({f"SimCLR/Loss_member_{model_idx}": avg_simclr_loss, "epoch": pretrain_epoch})
        
        logger.info("SimCLR pretraining finished.")


    # --- RL Reformulation Phase ---
    if training_config['use_rl_reformulation'] and current_rank == 0: # RL typically runs on one rank
        logger.info("Starting RL-based rule search...")
        rl_config = training_config['rl_config']
        
        # Initialize RulePolicy network
        policy_input_dim = model_config['support_set_encoder_config']['output_dim']
        rule_policy = RulePolicy(policy_input_dim, RuleGraphSpace().num_actions).to(DEVICE)
        policy_optimizer = optim.Adam(rule_policy.parameters(), lr=rl_config['policy_lr'])

        # To get actual support set features and scene graphs for RL environment:
        # We need to iterate through the training loader to get real Bongard problems
        # and use their support sets.
        
        # Create a separate PyTorch DataLoader for RL episodes to sample support sets
        # This uses the same underlying dataset but with a batch size of 1 for problem-by-problem processing.
        rl_data_loader = DataLoader(
            train_loader.dataset, # Use the same underlying dataset
            batch_size=1, # Process one problem at a time for RL episode
            num_workers=0, # No workers for single problem fetching
            pin_memory=True,
            collate_fn=train_loader.collate_fn, # Use the same collate_fn
            drop_last=True
        )
        
        for episode in range(num_rl_episodes):
            try:
                # Get a single Bongard problem (query + support) from the RL data loader
                # No need to call reset on PyTorch DataLoader directly, just iterate
                batch_data = next(iter(rl_data_loader))
                
                # Unpack relevant support set data from the batch
                raw_support_images_flat_np = batch_data[9][0] # List of np.ndarray for single problem
                support_labels_flat = batch_data[10][0].tolist() # List of int
                support_sgs_flat_bytes = batch_data[11][0] # List of bytes
                num_support_per_problem = batch_data[12][0].item() # Scalar int

                if num_support_per_problem == 0:
                    logger.warning(f"RL Episode {episode+1}: No support images found. Skipping episode.")
                    continue

                # Process raw support images through DALI Image Processor
                # DALI processor expects lists of numpy arrays for each input
                # So, we pass a list containing the single flattened support image array
                # And dummy query images.
                processed_support_images_flat_tensor = dali_image_processor.run(
                    [np.zeros((CONFIG['data']['image_size'], CONFIG['data']['image_size'], 3), dtype=np.uint8)], # Dummy query1
                    [np.zeros((CONFIG['data']['image_size'], CONFIG['data']['image_size'], 3), dtype=np.uint8)], # Dummy query2
                    raw_support_images_flat_np # Actual support images
                )[2] # Get the third output (processed_support_imgs_flat)
                
                # Reshape flattened support images back to (N_support, C, H, W)
                max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
                processed_support_images_reshaped_for_encoder = processed_support_images_flat_tensor.view(
                    1, max_support_imgs, 
                    processed_support_images_flat_tensor.shape[1], processed_support_images_flat_tensor.shape[2], processed_support_images_flat_tensor.shape[3]
                ).squeeze(0) # Remove batch dim if processing one problem

                # Only take the actual support images for this problem, not the padded ones
                actual_processed_support_images = processed_support_images_reshaped_for_encoder[:num_support_per_problem]
                actual_support_sgs = [json.loads(sg.decode('utf-8')) for sg in support_sgs_flat_bytes[:num_support_per_problem]]


                # Process support images through the PerceptionModule to get features
                # Run in eval mode for feature extraction
                model.eval()
                with torch.no_grad():
                    # Extract features from the full image using the backbone
                    # (B_support_actual, C, H, W)
                    # Need to handle if model is DDP wrapped
                    base_model_for_rl = model.module if is_ddp_initialized else model
                    support_batch_feature_maps = base_model_for_rl.attribute_classifier.feature_extractor(actual_processed_support_images)
                    
                    # Global pool to get (B_support_actual, C_feats)
                    support_features_global_pooled = F.adaptive_avg_pool2d(support_batch_feature_maps, (1,1)).flatten(1)
                    
                    # Pass through DeepSets encoder to get a single context vector
                    # DeepSets expects (N_elements, Feature_dim) and returns (Output_dim,)
                    # So, we pass the entire set of support features to it.
                    current_support_context_vector = base_model_for_rl.support_set_encoder(support_features_global_pooled).unsqueeze(0) # (1, D_context)
                model.train() # Set back to train mode

                # Reset RL environment with the new support problem
                rl_env = BongardEnv(
                    support_set_features=[sf.cpu() for sf in support_features_global_pooled], # Convert to list of CPU tensors for env
                    support_set_labels=support_labels_flat,
                    support_set_scene_graphs=actual_support_sgs,
                    max_steps=rl_config['max_steps_per_episode']
                )
                
                log_probs = []
                rewards = []
                entropies = []

                observation = rl_env.support_set_features # Initial observation (list of features)

                for step in range(rl_config['max_steps_per_episode']):
                    action_logits = rule_policy(current_support_context_vector) # Policy takes the single context vector
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    entropy = action_dist.entropy()

                    # Take a step in the environment
                    _, reward, done, info = rl_env.step(action.item())

                    log_probs.append(log_prob)
                    rewards.append(reward)
                    entropies.append(entropy)

                    if done:
                        break
                
                # Calculate discounted rewards
                discounted_rewards = []
                R = 0
                for r in reversed(rewards):
                    R = r + rl_config['gamma'] * R
                    discounted_rewards.insert(0, R)
                discounted_rewards = torch.tensor(discounted_rewards, device=DEVICE)
                
                # Normalize rewards (optional but common)
                if len(discounted_rewards) > 1:
                    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

                # Calculate policy loss
                policy_loss = []
                for lp, dr, ent in zip(log_probs, discounted_rewards, entropies):
                    policy_loss.append(-lp * dr - rl_config['entropy_coeff'] * ent) # Maximize reward, maximize entropy
                
                policy_optimizer.zero_grad()
                torch.stack(policy_loss).sum().backward()
                policy_optimizer.step()

                if (episode + 1) % 10 == 0:
                    logger.info(f"RL Episode {episode+1}/{num_rl_episodes} - Total Reward: {sum(rewards):.2f}, Rule Accuracy: {info['rule_accuracy']:.2f}")
                    if HAS_WANDB and CONFIG['training']['use_wandb']:
                        import wandb
                        wandb.log({
                            f"RL/Total_Reward_member_{model_idx}": sum(rewards),
                            f"RL/Policy_Loss_member_{model_idx}": torch.stack(policy_loss).sum().item(),
                            f"RL/Rule_Accuracy_member_{model_idx}": info['rule_accuracy'],
                            "rl_episode": episode
                        })

            except StopIteration:
                logger.info("RL data loader exhausted. Resetting for next RL epoch.")
                # If you want more episodes than available unique problems, you'd reset the loader here.
                # For now, we just stop if the loader runs out.
                break
            except Exception as e:
                logger.error(f"Error during RL episode {episode+1}: {e}")
                continue

        logger.info("RL-based rule search finished.")
        # After RL, the `rule_policy` can be used to propose rules.
        # You might want to save the trained policy.


    # --- Quantization Setup (QAT) ---
    if training_config['use_quantization_aware_training'] and HAS_TORCH_QUANTIZATION:
        logger.info("Preparing model for Quantization Aware Training (QAT)...")
        # Ensure model is on CPU for fusion before preparing for QAT
        # DDP models need to be unwrapped for fusion and QAT preparation
        if is_ddp_initialized:
            model_to_quantize = model.module.to('cpu')
        else:
            model_to_quantize = model.to('cpu')
        
        model_to_quantize = quantize_model_qat(model_to_quantize, training_config)
        
        # Move back to GPU if DDP or single GPU
        if is_ddp_initialized:
            model = DDP(model_to_quantize.to(DEVICE), device_ids=[current_rank])
        else:
            model = model_to_quantize.to(DEVICE)
        logger.info("Model ready for QAT fine-tuning.")


    # --- Main Training Loop ---
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0.0
        total_bongard_loss = 0.0
        total_attribute_loss = 0.0
        total_relation_loss = 0.0
        total_consistency_loss = 0.0
        total_distillation_loss = 0.0
        total_symbolic_consistency_loss = 0.0
        
        num_batches = 0
        
        # Set epoch for sampler (if using DistributedSampler or CurriculumSampler)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        if replay_buffer and training_config['curriculum_config']['difficulty_sampling']:
            replay_buffer.update_beta(epoch, total_epochs) # Beta annealing

        # --- Structured Pruning Application (Iterative) ---
        if training_config['use_structured_pruning'] and training_config['pruning_config']['enabled'] and \
           (epoch + 1) % training_config['pruning_config']['prune_interval_epochs'] == 0 and current_rank == 0:
            logger.info(f"Applying structured pruning at epoch {epoch+1}...")
            
            sensitivity_scores = None
            if training_config['pruning_config']['use_sensitivity_analysis']:
                # Compute sensitivity on a subset of validation data for speed
                sensitivity_scores = compute_layer_sensitivity(model, val_loader, dali_image_processor, current_rank)

            apply_structured_pruning(
                model=model,
                config=training_config,
                train_loader=train_loader, # Pass train_loader for fine-tuning
                val_loader=val_loader,     # Pass val_loader for fine-tuning
                dali_image_processor=dali_image_processor, # Pass DALI processor
                model_idx=model_idx,
                current_rank=current_rank,
                is_ddp_initialized=is_ddp_initialized,
                sensitivity_scores=sensitivity_scores
            )
            logger.info(f"Structured pruning applied for epoch {epoch+1}.")


        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{total_epochs} (Rank {current_rank})", disable=(current_rank != 0))
        
        for batch_idx, batch_data in pbar:
            # Unpack raw NumPy arrays and other data from collate_fn output
            raw_query_images_view1_np = batch_data[0]
            raw_query_images_view2_np = batch_data[1]
            query_labels = batch_data[2].long().to(DEVICE)
            query_gts_json_view1 = batch_data[3]
            query_gts_json_view2 = batch_data[4]
            difficulties = batch_data[5].float().to(DEVICE)
            affine1 = batch_data[6].float().to(DEVICE)
            affine2 = batch_data[7].float().to(DEVICE)
            original_indices = batch_data[8].long().to(DEVICE)

            raw_support_images_flat_np = batch_data[9]
            support_labels_flat = batch_data[10].long().to(DEVICE)
            support_sgs_flat_bytes = batch_data[11]
            num_support_per_problem = batch_data[12] # This is a tensor/list of counts per problem in batch


            tree_indices = None
            is_weights = None
            # Check if PER info is present in the batch (last two elements from collate_fn)
            if len(batch_data) > 13 and batch_data[13] is not None: 
                tree_indices = batch_data[13].long().to(DEVICE)
                is_weights = batch_data[14].float().to(DEVICE)

            # Process raw images through DALI Image Processor
            processed_query_images_view1, processed_query_images_view2, processed_support_images_flat = dali_image_processor.run(
                raw_query_images_view1_np,
                raw_query_images_view2_np,
                raw_support_images_flat_np
            )
            
            # Reshape flattened support images back to (B, N_support, C, H, W)
            max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_query_images_view1.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_query_images_view1.shape[1], processed_query_images_view1.shape[2], processed_query_images_view1.shape[3]
            )

            # Apply Mixup/CutMix if enabled
            if training_config.get('use_mixup_cutmix', False):
                images_view1_aug, labels_mixed = mixup_cutmix_augmenter(processed_query_images_view1, query_labels)
                images_view2_aug, _ = mixup_cutmix_augmenter(processed_query_images_view2, query_labels)
            else:
                images_view1_aug = processed_query_images_view1
                images_view2_aug = processed_query_images_view2
                labels_mixed = None

            # Handle SAM optimizer's first_step
            if training_config['use_sam'] and isinstance(optimizer, SAM):
                with autocast(enabled=training_config['use_amp']):
                    outputs1 = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, 
                                     support_images=processed_support_images_reshaped)
                    bongard_logits1 = outputs1['bongard_logits']
                    
                    if labels_mixed is not None:
                        loss_for_sam_first_step = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='batchmean')
                    else:
                        loss_for_sam_first_step = bongard_criterion(bongard_logits1, query_labels).mean()
                
                if scaler:
                    scaler.scale(loss_for_sam_first_step).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss_for_sam_first_step.backward()
                
                optimizer.first_step(zero_grad=True)

            optimizer.zero_grad()

            with autocast(enabled=training_config['use_amp']):
                # Forward pass for view 1
                outputs1 = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, 
                                 support_images=processed_support_images_reshaped)
                bongard_logits1 = outputs1['bongard_logits']
                attribute_logits1 = outputs1['attribute_logits']
                relation_logits1 = outputs1['relation_logits']
                attribute_features1 = outputs1['attribute_features']
                global_graph_embeddings1 = outputs1['global_graph_embeddings']
                scene_graphs1 = outputs1['scene_graphs']

                # Forward pass for view 2 (for consistency losses)
                outputs2 = model(images_view2_aug, ground_truth_json_strings=query_gts_json_view2, 
                                 support_images=processed_support_images_reshaped)
                bongard_logits2 = outputs2['bongard_logits']
                attribute_logits2 = outputs2['attribute_logits']
                relation_logits2 = outputs2['relation_logits']
                attribute_features2 = outputs2['attribute_features']
                global_graph_embeddings2 = outputs2['global_graph_embeddings']
                scene_graphs2 = outputs2['scene_graphs']


                # --- Calculate Losses ---
                # 1. Bongard Classification Loss
                if labels_mixed is not None:
                    per_sample_bongard_losses = F.kl_div(F.log_softmax(bongard_logits1, dim=-1), labels_mixed, reduction='none').sum(dim=1)
                else:
                    per_sample_bongard_losses = bongard_criterion(bongard_logits1, query_labels)
                
                # Apply IS weights if PER is used
                if replay_buffer and is_weights is not None:
                    loss_bongard = (per_sample_bongard_losses * is_weights).mean()
                else:
                    loss_bongard = per_sample_bongard_losses.mean()


                # 2. Attribute Classification Loss
                loss_attribute = torch.tensor(0.0, device=DEVICE)
                num_attribute_losses = 0
                
                current_flat_idx = 0
                for i_img in range(len(scene_graphs1)):
                    sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                    inferred_objects_for_img = scene_graphs1[i_img].get('objects', [])
                    
                    for inferred_obj_idx_in_img, inferred_obj in enumerate(inferred_objects_for_img):
                        gt_obj = None
                        inferred_bbox = inferred_obj.get('bbox')
                        if inferred_bbox:
                            for gt_o in sg_gt['objects']:
                                if _calculate_iou(gt_o['bbox'], inferred_bbox) > 0.7:
                                    gt_obj = gt_o
                                    break
                        
                        if gt_obj:
                            for attr_name, num_classes in model_config['attribute_classifier_config'].items():
                                if attr_name not in ['backbone_name', 'pretrained', 'freeze_backbone'] and \
                                   attr_name in gt_obj['attributes'] and attr_name in attribute_logits1 and attribute_logits1[attr_name].numel() > 0:
                                    
                                    # Ensure the flat index is within bounds of attribute_logits1[attr_name]
                                    if current_flat_idx < attribute_logits1[attr_name].shape[0]:
                                        attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                        if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                            gt_attr_label = attr_map[gt_obj['attributes'][attr_name]]
                                            predicted_logits = attribute_logits1[attr_name][current_flat_idx].unsqueeze(0)
                                            loss_attribute += attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                            num_attribute_losses += 1
                                        else:
                                            logger.warning(f"Attribute map or value not found for {attr_name}: {gt_obj['attributes'][attr_name]}")
                        
                        # Increment for the next inferred object, regardless of whether it was matched to GT
                        current_flat_idx += 1


                if num_attribute_losses > 0:
                    loss_attribute /= num_attribute_losses


                # 3. Relation Classification Loss
                loss_relation = torch.tensor(0.0, device=DEVICE)
                if relation_logits1.numel() > 0:
                    B_current, _, R = relation_logits1.shape
                    
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
                            else:
                                logger.debug(f"Skipping unmapped relation: {(subj_id, obj_id, rel_type)} in GT for batch {b}.")

                    logits_flat = relation_logits1.view(-1, R)
                    labels_flat = gt_edge_labels.view(-1)
                    
                    loss_relation = relation_criterion(logits_flat, labels_flat)
                    total_relation_loss += loss_relation.item()


                # 4. Consistency Losses
                loss_consistency = torch.tensor(0.0, device=DEVICE)
                if training_config['consistency_loss_weight'] > 0:
                    if training_config['feature_consistency_weight'] > 0:
                        if attribute_features1.numel() > 0 and attribute_features2.numel() > 0:
                            loss_feature_consistency = feature_consistency_criterion(attribute_features1, attribute_features2)
                            loss_consistency += training_config['feature_consistency_weight'] * loss_feature_consistency
                        else:
                            logger.debug("Skipping feature consistency loss: no objects detected in one or both views.")
                    
                    # Symbolic Consistency Loss
                    if training_config['symbolic_consistency_weight'] > 0:
                        # For symbolic consistency, we need the *ground truth* scene graphs
                        # for the query images to find the best rule.
                        gt_positive_sgs = [json.loads(g.decode('utf-8')) for i, g in enumerate(query_gts_json_view1) if query_labels[i].item() == 1]
                        gt_negative_sgs = [json.loads(g.decode('utf-8')) for i, g in enumerate(query_gts_json_view1) if query_labels[i].item() == 0]

                        hypothesized_rules_for_batch = []
                        if gt_positive_sgs and gt_negative_sgs:
                            top_rules = find_best_rules(
                                all_rules=list(ALL_BONGARD_RULES.values()), # Evaluate against known rules
                                positive_scene_graphs=gt_positive_sgs,
                                negative_scene_graphs=gt_negative_sgs,
                                k=1
                            )
                            if top_rules:
                                hypothesized_rules_for_batch = top_rules
                                if current_rank == 0 and (batch_idx + 1) % debug_config['rule_eval_log_interval_batches'] == 0:
                                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Hypothesized Rule: {top_rules[0][0].name} (Score: {top_rules[0][1]})")
                            else:
                                logger.debug(f"No best rule found for batch {batch_idx+1}.")
                        else:
                            logger.debug(f"Skipping rule evaluation for batch {batch_idx+1}: insufficient positive/negative GT samples.")

                        loss_symbolic_consistency = symbolic_consistency_criterion(
                            scene_graphs1=scene_graphs1, # Inferred scene graphs for view 1
                            scene_graphs2=scene_graphs2, # Inferred scene graphs for view 2
                            labels=query_labels,
                            hypothesized_rules_info=hypothesized_rules_for_batch
                        )
                        total_symbolic_consistency_loss += loss_symbolic_consistency.item()
                        loss_consistency += training_config['symbolic_consistency_weight'] * loss_symbolic_consistency
                
                # 5. Knowledge Distillation Loss (if enabled)
                loss_distillation = torch.tensor(0.0, device=DEVICE)
                if distillation_criterion and teacher_models: # teacher_models is now a list
                    # Get streaming teacher logits with ensemble dropout and optional mask
                    teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                        teacher_models=teacher_models,
                        raw_images_np=raw_query_images_view1_np, # Pass raw images for teacher's DALI processing
                        raw_gt_json_strings=query_gts_json_view1, # Pass raw GT JSON for teacher's forward
                        raw_support_images_np=raw_support_images_flat_np, # Pass raw support images for teacher's DALI processing
                        distillation_config=training_config['distillation_config'],
                        dali_image_processor=dali_image_processor # Pass the DALI processor
                    )
                    
                    if teacher_logits_batch.numel() > 0: # Ensure we got valid logits
                        # DistillationLoss returns per-sample soft and hard losses
                        per_sample_soft_loss, per_sample_hard_loss = distillation_criterion(
                            bongard_logits1, teacher_logits_batch, query_labels
                        )
                        
                        # Apply mask if enabled
                        if distillation_mask is not None and training_config['distillation_config']['use_mask_distillation']:
                            # Ensure mask is broadcastable if needed (e.g., (B,) for (B,1) losses)
                            # KLDivLoss with reduction='none' already gives (B,) loss
                            # CrossEntropyLoss with reduction='none' also gives (B,) loss
                            masked_soft_loss = per_sample_soft_loss * distillation_mask
                            masked_hard_loss = per_sample_hard_loss * distillation_mask
                            loss_distillation = (training_config['distillation_config']['alpha'] * masked_soft_loss + \
                                                 (1. - training_config['distillation_config']['alpha']) * masked_hard_loss).mean()
                        else:
                            # If no mask or mask distillation is off, just take the mean
                            loss_distillation = (training_config['distillation_config']['alpha'] * per_sample_soft_loss + \
                                                 (1. - training_config['distillation_config']['alpha']) * per_sample_hard_loss).mean()
                        
                        total_distillation_loss += loss_distillation.item()
                    else:
                        logger.warning("No teacher logits generated for distillation in this batch.")


                # Combine all losses
                current_batch_loss = loss_bongard
                if loss_attribute > 0: current_batch_loss += loss_attribute
                if loss_relation > 0: current_batch_loss += training_config['relation_loss_weight'] * loss_relation
                if loss_consistency > 0: current_batch_loss += training_config['consistency_loss_weight'] * loss_consistency
                if loss_distillation > 0: current_batch_loss += loss_distillation


            # Backward pass
            if scaler:
                scaler.scale(current_batch_loss).backward()
                if training_config['max_grad_norm'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
            else:
                current_batch_loss.backward()
                if training_config['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['max_grad_norm'])
                optimizer.step()
            
            # Handle SAM optimizer's second_step
            if training_config['use_sam'] and isinstance(optimizer, SAM):
                optimizer.second_step(zero_grad=True)

            # Update total losses
            total_loss += current_batch_loss.item()
            total_bongard_loss += loss_bongard.item()
            total_attribute_loss += loss_attribute.item()
            total_relation_loss += loss_relation.item()
            total_consistency_loss += loss_consistency.item()
            total_distillation_loss += loss_distillation.item()
            total_symbolic_consistency_loss += total_symbolic_consistency_loss # This was a bug, should be loss_symbolic_consistency.item()
            # Corrected:
            # total_symbolic_consistency_loss += loss_symbolic_consistency.item() # This line was causing double counting.
            # The value `loss_symbolic_consistency.item()` is already added to `total_consistency_loss`.
            # If `total_symbolic_consistency_loss` is meant to track only the symbolic part,
            # then it should be added like this:
            # total_symbolic_consistency_loss += loss_symbolic_consistency.item()
            # However, since `loss_consistency` already sums it up, and `total_consistency_loss` tracks that,
            # keeping `total_symbolic_consistency_loss` separate and adding its raw item value is fine for logging.
            # Let's assume the previous line was meant to be the correct one:
            # total_symbolic_consistency_loss += loss_symbolic_consistency.item()
            num_batches += 1

            # Update PER buffer if enabled and at specified frequency
            if replay_buffer and training_config['curriculum_config']['difficulty_sampling'] and \
               (batch_idx + 1) % training_config['curriculum_config']['difficulty_update_frequency_batches'] == 0:
                
                losses_np = per_sample_bongard_losses.detach().cpu().numpy()
                tree_indices_np = tree_indices.cpu().numpy()
                
                async_update_priorities(replay_buffer, tree_indices_np.tolist(), losses_np.tolist())

            # Log training metrics
            if (batch_idx + 1) % training_config['log_interval_batches'] == 0 and current_rank == 0:
                avg_loss = total_loss / num_batches
                avg_bongard_loss = total_bongard_loss / num_batches
                avg_attribute_loss = total_attribute_loss / num_batches
                avg_relation_loss = total_relation_loss / num_batches
                avg_symbolic_consistency_loss = total_symbolic_consistency_loss / num_batches
                avg_distillation_loss = total_distillation_loss / num_batches

                pbar.set_postfix(loss=f"{avg_loss:.4f}", bongard_loss=f"{avg_bongard_loss:.4f}", 
                                  attr_loss=f"{avg_attribute_loss:.4f}",
                                  rel_loss=f"{avg_relation_loss:.4f}",
                                  sym_cons_loss=f"{avg_symbolic_consistency_loss:.4f}",
                                  kd_loss=f"{avg_distillation_loss:.4f}",
                                  lr=optimizer.param_groups[0]['lr'])
                
                if HAS_WANDB and CONFIG['training']['use_wandb']:
                    import wandb
                    wandb.log({
                        f"train/loss_member_{model_idx}": avg_loss,
                        f"train/bongard_loss_member_{model_idx}": avg_bongard_loss,
                        f"train/attribute_loss_member_{model_idx}": avg_attribute_loss,
                        f"train/relation_loss_member_{model_idx}": avg_relation_loss,
                        f"train/symbolic_consistency_loss_member_{model_idx}": avg_symbolic_consistency_loss,
                        f"train/distillation_loss_member_{model_idx}": avg_distillation_loss,
                        f"train/lr_member_{model_idx}": optimizer.param_groups[0]['lr'],
                        "global_step": epoch * len(train_loader) + batch_idx
                    })

            # Validate at specified batch frequency
            if training_config['validation_frequency_batches'] and \
               (batch_idx + 1) % training_config['validation_frequency_batches'] == 0:
                val_metrics = _validate_model_ensemble(model, val_loader, dali_image_processor, current_rank, is_ddp_initialized, model_idx)
                if current_rank == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Validation Accuracy: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")
                    if HAS_WANDB and CONFIG['training']['use_wandb']:
                        import wandb
                        wandb.log({
                            f"val/accuracy_member_{model_idx}": val_metrics['accuracy'],
                            f"val/loss_member_{model_idx}": val_metrics['loss'],
                            "global_step": epoch * len(train_loader) + batch_idx
                        })
                model.train()

        # End of epoch
        if scheduler and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(total_loss / num_batches)
            else:
                scheduler.step()
        
        if swa_model and epoch >= training_config['swa_start_epoch']:
            swa_model.update_parameters(model.module if is_ddp_initialized else model)
            swa_scheduler.step()
            logger.info(f"SWA model updated. SWA LR: {optimizer.param_groups[0]['lr']}")

        val_metrics = _validate_model_ensemble(model, val_loader, dali_image_processor, current_rank, is_ddp_initialized, model_idx)

        if current_rank == 0:
            logger.info(f"Epoch {epoch+1} Summary (Member {model_idx}):")
            logger.info(f"  Train Loss: {total_loss / num_batches:.4f}")
            logger.info(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")
            
            if HAS_WANDB and CONFIG['training']['use_wandb']:
                import wandb
                wandb.log({
                    f"epoch_val/accuracy_member_{model_idx}": val_metrics['accuracy'],
                    f"epoch_val/loss_member_{model_idx}": val_metrics['loss'],
                    f"epoch_train/loss_member_{model_idx}": total_loss / num_batches,
                    "epoch": epoch
                })

            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_val_loss = val_metrics['loss']
                best_metrics = val_metrics
                
                save_path = os.path.join(debug_config['save_model_checkpoints'], f"best_model_member_{model_idx}.pth")
                os.makedirs(debug_config['save_model_checkpoints'], exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': (model.module if is_ddp_initialized else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_accuracy,
                    'val_loss': best_val_loss,
                    'metrics': best_metrics,
                }, save_path)
                logger.info(f"Saved best model checkpoint for member {model_idx} at {save_path}")
            elif val_metrics['accuracy'] == best_val_accuracy and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_metrics = val_metrics
                save_path = os.path.join(debug_config['save_model_checkpoints'], f"best_model_member_{model_idx}.pth")
                os.makedirs(debug_config['save_model_checkpoints'], exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': (model.module if is_ddp_initialized else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_accuracy,
                    'val_loss': best_val_loss,
                    'metrics': best_metrics,
                }, save_path)
                logger.info(f"Updated best model checkpoint for member {model_idx} (lower loss).")

    if swa_model and current_rank == 0:
        logger.info("Performing final SWA update and evaluation...")
        swa_utils.swap_parameters_with_averaged_model(model.module if is_ddp_initialized else model, swa_model)
        val_metrics_swa = _validate_model_ensemble(model, val_loader, dali_image_processor, current_rank, is_ddp_initialized, model_idx, is_swa_eval=True)
        logger.info(f"SWA Validation Accuracy: {val_metrics_swa['accuracy']:.4f}, Loss: {val_metrics_swa['loss']:.4f}")
        
        if HAS_WANDB and CONFIG['training']['use_wandb']:
            import wandb
            wandb.log({
                f"swa_val/accuracy_member_{model_idx}": val_metrics_swa['accuracy'],
                f"swa_val/loss_member_{model_idx}": val_metrics_swa['loss'],
            })
        
        swa_utils.swap_parameters_with_averaged_model(model.module if is_ddp_initialized else model, swa_model)

    # --- Post-Training Quantization (PTQ) ---
    if training_config['use_post_training_quantization'] and HAS_TORCH_QUANTIZATION and current_rank == 0:
        logger.info("Applying Post-Training Quantization (PTQ)...")
        # Ensure model is on CPU for PTQ
        if is_ddp_initialized:
            model_to_quantize = model.module.to('cpu')
        else:
            model_to_quantize = model.to('cpu')
        
        model_quantized_ptq = quantize_model_ptq(model_to_quantize, val_loader, training_config)
        logger.info("PTQ applied. Evaluating quantized model.")
        
        # Evaluate quantized model
        val_metrics_ptq = _validate_model_ensemble(model_quantized_ptq, val_loader, dali_image_processor, current_rank, False, model_idx, is_quantized_eval=True)
        logger.info(f"PTQ Quantized Model Validation Accuracy: {val_metrics_ptq['accuracy']:.4f}, Loss: {val_metrics_ptq['loss']:.4f}")
        
        if HAS_WANDB and CONFIG['training']['use_wandb']:
            import wandb
            wandb.log({
                f"ptq_val/accuracy_member_{model_idx}": val_metrics_ptq['accuracy'],
                f"ptq_val/loss_member_{model_idx}": val_metrics_ptq['loss'],
            })
        
        # Replace the model with the quantized one for potential export
        model = model_quantized_ptq.to(DEVICE) # Move back to GPU for consistency if needed

    # --- Export Quantized Model ---
    if training_config['export_quantized_model'] and current_rank == 0:
        logger.info("Exporting quantized model to ONNX...")
        try:
            # Create dummy inputs for all arguments of PerceptionModule.forward
            # These are *processed* tensors, as the export happens after DALI.
            dummy_images = torch.randn(1, 3, CONFIG['data']['image_size'], CONFIG['data']['image_size']).to(DEVICE)
            dummy_gt_json_strings = [json.dumps({}).encode('utf-8')] # Dummy empty SG
            
            # Dummy support images: (B, N_support, C, H, W)
            dummy_support_images = torch.zeros(1, CONFIG['data']['synthetic_data_config']['max_support_images_per_problem'], 
                                               3, CONFIG['data']['image_size'], CONFIG['data']['image_size']).to(DEVICE)

            # Ensure model is in eval mode before export
            model.eval()

            # Export the model
            export_path = os.path.join(debug_config['save_model_checkpoints'], f"bongard_model_member_{model_idx}_quantized.onnx")
            torch.onnx.export(
                model,
                (dummy_images, dummy_gt_json_strings, dummy_support_images, False), # Arguments for model.forward
                export_path,
                export_params=True,
                opset_version=13, # Recommended opset
                do_constant_folding=True,
                input_names=['input_images', 'gt_json_strings', 'support_images', 'is_simclr_pretraining'], # Match model.forward args
                output_names=['bongard_logits', 'attribute_logits', 'relation_logits', 'attribute_features', 
                              'global_graph_embeddings', 'scene_graphs'], # Match model.forward outputs
                dynamic_axes={
                    'input_images': {0: 'batch_size'},
                    'support_images': {0: 'batch_size'},
                    'bongard_logits': {0: 'batch_size'},
                    'attribute_logits': {0: 'total_objects'}, # Dynamic for total objects
                    'relation_logits': {0: 'batch_size'},
                    # Add other dynamic axes as needed based on output shapes
                }
            )
            logger.info(f"Model exported to {export_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            logger.info("Attempting to export to TorchScript instead...")
            try:
                traced_model = torch.jit.trace(model, (dummy_images, dummy_gt_json_strings, dummy_support_images, False))
                export_path = os.path.join(debug_config['save_model_checkpoints'], f"bongard_model_member_{model_idx}_quantized.pt")
                traced_model.save(export_path)
                logger.info(f"Model exported to TorchScript at {export_path}")
            except Exception as jit_e:
                logger.error(f"Failed to export model to TorchScript: {jit_e}")


    return best_val_accuracy, best_val_loss, best_metrics

# --- Validation Function (used by training and sensitivity analysis) ---
def _validate_model_ensemble(
    model: Union[PerceptionModule, DDP],
    val_loader: Any,
    dali_image_processor: Any, # Pass DALI processor
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    model_idx: int = 0,
    is_swa_eval: bool = False,
    is_quantized_eval: bool = False,
    limit_batches: Optional[int] = None # For sensitivity analysis
) -> Dict[str, float]:
    """
    Validates a single model/ensemble member.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Loss functions for validation
    bongard_criterion = LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing'])
    attribute_criterion = LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing'])
    relation_criterion = LabelSmoothingCrossEntropy(smoothing=CONFIG['training']['label_smoothing'], reduction='mean')

    MAX_GNN_OBJECTS = CONFIG['model']['MAX_GNN_OBJECTS']
    E_max = MAX_GNN_OBJECTS * (MAX_GNN_OBJECTS - 1)

    # Set epoch for sampler (if using DistributedSampler or CurriculumSampler)
    if hasattr(val_loader.sampler, 'set_epoch'):
        val_loader.sampler.set_epoch(0) # For validation, epoch doesn't change, but set_epoch might do other setup

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), 
                    desc=f"Validating Model {model_idx} (Rank {current_rank})", disable=(current_rank != 0))
        for batch_idx, batch_data in pbar:
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            # Unpack raw NumPy arrays and other data from collate_fn output
            raw_query_images_view1_np = batch_data[0]
            # raw_query_images_view2_np = batch_data[1] # Not used for validation loss calculation
            query_labels = batch_data[2].long().to(DEVICE)
            query_gts_json_view1 = batch_data[3]
            # query_gts_json_view2 = batch_data[4]
            # difficulties = batch_data[5]
            # affine1 = batch_data[6]
            # affine2 = batch_data[7]
            # original_indices = batch_data[8]
            raw_support_images_flat_np = batch_data[9]
            # support_labels_flat = batch_data[10]
            # support_sgs_flat_bytes = batch_data[11]
            num_support_per_problem = batch_data[12] # Tensor of counts


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
                            if attr_name not in ['backbone_name', 'pretrained', 'freeze_backbone'] and \
                               attr_name in gt_obj['attributes'] and attr_name in attribute_logits and attribute_logits[attr_name].numel() > 0:
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
                        obj_id = rel['object_id'] # Assuming 'id' is object_id for relation
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

