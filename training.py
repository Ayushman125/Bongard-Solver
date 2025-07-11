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
import random  # For MixupCutmix
import numpy as np  # For MixupCutmix
import json  # For parsing GT JSON
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image  # For saving synthetic images for Grad-CAM
import torchvision.transforms as T  # For converting tensor to PIL Image for Grad-CAM
import threading  # For async updates (conceptual)
import copy  # For deepcopy for Mean Teacher
# Import for pruning
import torch.nn.utils.prune as prune
# Import configuration
from config import (
    CONFIG, DEVICE, HAS_WANDB, HAS_TORCH_QUANTIZATION, load_config,
    ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
    ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
    RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD
)
# Import models and data modules
from models import LitBongard, LitSimCLR, PerceptionModule
from data import BongardDataModule, get_dataloader, build_dali_image_processor, HardExampleSampler
# Import utils functions
from utils import set_seed, _calculate_iou, make_edge_index_map, async_update_priorities
# Import for early stopping
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
# Imports for MixUp/CutMix
try:
    from torchvision.transforms.v2 import MixUp, CutMix # Requires torchvision >= 0.13
    HAS_TORCHVISION_V2 = True
    logger.info("torchvision.transforms.v2 found for MixUp/CutMix.")
except ImportError:
    HAS_TORCHVISION_V2 = False
    logger.warning("torchvision.transforms.v2 not found. MixUp/CutMix will be disabled.")
# Import for SAM optimizer
try:
    from torch_optimizer import SAM
    HAS_SAM_OPTIMIZER = True
except ImportError:
    HAS_SAM_OPTIMIZER = False
    logger.warning("torch_optimizer (SAM) not found. SAM optimizer will be disabled.")

# Conditional import for quantization
if HAS_TORCH_QUANTIZATION:
    try:
        from torch.quantization import prepare_qat, convert, fuse_modules
        from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver
        from torch.quantization.fake_quantize import FakeQuantize
        logger.info("PyTorch quantization modules found and enabled.")
    except ImportError:
        logger.warning("PyTorch quantization modules not found. Quantization features will not be available.")
        HAS_TORCH_QUANTIZATION = False  # Update config if import fails
else:
    logger.warning("PyTorch quantization is disabled in config. Quantization features will not be available.")

logger = logging.getLogger(__name__)

# --- Helper Functions for Pruning and Quantization ---
def compute_layer_sensitivity(model: nn.Module, val_loader: DataLoader, dali_image_processor: Any, current_rank: int) -> Dict[str, float]:
    """
    Computes sensitivity for each prunable layer by evaluating performance drop.
    This is a simplified version for demonstration.
    """
    logger.info("Computing layer sensitivity (dummy implementation).")
    sensitivity_scores = {}
    # In a real scenario, you'd prune a small amount, evaluate, restore, and repeat.
    # For this dummy, we just assign random scores.
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            sensitivity_scores[name] = random.random() # Dummy sensitivity
    return sensitivity_scores

def apply_structured_pruning(model: nn.Module, cfg: Dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, dali_image_processor: Any, current_rank: int, is_ddp_initialized: bool, sensitivity_scores: Optional[Dict[str, float]] = None) -> nn.Module:
    """
    Applies structured pruning to the model.
    """
    logger.info(f"Applying structured pruning with method: {cfg['training']['pruning']['method']}")
    # This is a placeholder. Real pruning involves iterative pruning and fine-tuning.
    # For demonstration, we'll just apply one-shot pruning.
    
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Apply pruning to 'weight' and 'bias' if they exist
            if hasattr(module, 'weight') and module.weight is not None:
                parameters_to_prune.append((module, 'weight'))
            if hasattr(module, 'bias') and module.bias is not None:
                parameters_to_prune.append((module, 'bias'))
    
    if not parameters_to_prune:
        logger.warning("No prunable parameters found in the model.")
        return model
    
    pruning_method = cfg['training']['pruning']['method']
    amount = cfg['training']['pruning']['amount']
    
    if pruning_method == "ln_structured":
        # Example: L1-norm based structured pruning (e.g., prune rows/columns)
        # This requires more specific implementation based on layer type (e.g., prune.ln_structured)
        # For simplicity, using unstructured as a fallback if structured is complex.
        logger.warning("ln_structured pruning is complex and requires specific module/dim. Falling back to l1_unstructured for demonstration.")
        prune.remove(parameters_to_prune[0][0], parameters_to_prune[0][1]) # Remove previous pruning masks
        prune.l1_unstructured(parameters_to_prune[0][0], name=parameters_to_prune[0][1], amount=amount)
    elif pruning_method == "l1_unstructured":
        for module, name in parameters_to_prune:
            prune.remove(module, name) # Remove previous pruning masks
            prune.l1_unstructured(module, name=name, amount=amount)
    elif pruning_method == "random_unstructured":
        for module, name in parameters_to_prune:
            prune.remove(module, name) # Remove previous pruning masks
            prune.random_unstructured(module, name=name, amount=amount)
    else:
        logger.error(f"Unsupported pruning method: {pruning_method}. No pruning applied.")
        return model
    
    # Remove pruning reparametrization (make it permanent)
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    logger.info(f"Pruning applied: {amount*100:.2f}% using {pruning_method}.")
    return model

def quantize_model_qat(model: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    """
    Prepares model for Quantization Aware Training (QAT).
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.warning("PyTorch quantization not available. Cannot perform QAT.")
        return model
    logger.info("Preparing model for Quantization Aware Training (QAT).")
    # Fuse modules for better quantization performance
    # Example: fuse Conv-BatchNorm-ReLU
    # This requires specific knowledge of your model's architecture.
    # For a generic model, we might skip fusion or use a simplified approach.
    try:
        # Assuming a simple sequential model or common patterns
        # If your model has specific fusion patterns, implement them here.
        # For example: torch.quantization.fuse_modules(model.backbone, [['conv1', 'bn1', 'relu']], inplace=True)
        logger.info("Skipping module fusion for QAT (requires model-specific knowledge).")
    except Exception as e:
        logger.warning(f"Module fusion failed for QAT: {e}. Proceeding without fusion.")
    # Set quantization configuration
    qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') # 'qnnpack' for ARM, 'fbgemm' for x86
    model.train() # Model must be in training mode for QAT
    model.qconfig = qconfig
    torch.quantization.prepare_qat(model, inplace=True)
    logger.info("Model prepared for QAT. Remember to fine-tune the model with QAT enabled.")
    return model

def quantize_model_ptq(model: nn.Module, val_loader: DataLoader, dali_image_processor: Any, cfg: Dict[str, Any]) -> nn.Module:
    """
    Performs Post-Training Quantization (PTQ) on the model.
    """
    if not HAS_TORCH_QUANTIZATION:
        logger.warning("PyTorch quantization not available. Cannot perform PTQ.")
        return model
    logger.info("Performing Post-Training Quantization (PTQ).")
    model.eval() # Model must be in eval mode for PTQ
    # Fuse modules for better quantization performance
    try:
        logger.info("Skipping module fusion for PTQ (requires model-specific knowledge).")
    except Exception as e:
        logger.warning(f"Module fusion failed for PTQ: {e}. Proceeding without fusion.")
    # Set quantization configuration
    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.qconfig = qconfig
    torch.quantization.prepare(model, inplace=True)
    # Calibrate the model
    logger.info("Calibrating model for PTQ...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="PTQ Calibration")):
            if cfg['training']['use_dali'] and dali_image_processor:
                # DALI returns processed tensors
                raw_query_images_view1_np = batch[0][0] # Assuming first item of first element is raw_query_images_view1_np
                raw_query_images_view2_np = batch[0][1]
                raw_support_images_flat_np = batch[0][9] # Assuming this is correct index for support images
                
                processed_query_images_view1, _, _ = dali_image_processor.run(
                    raw_query_images_view1_np,
                    raw_query_images_view2_np, # DALI expects this input, even if not used for calibration
                    raw_support_images_flat_np
                )
                images = processed_query_images_view1
            else:
                # Fallback for PyTorch DataLoader
                # Assuming the first element of the batch is the image tensor
                images_np = batch[0][0] # Get list of numpy arrays for query_img1
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                images = torch.stack([transform(img_np) for img_np in images_np]).to(DEVICE)
            
            # Forward pass to collect statistics
            # The PerceptionModule's forward needs ground_truth_json_strings and support_images
            # even for inference/calibration.
            # We need to extract these from the batch or provide dummies.
            # Assuming batch structure: (query_img1_np_list, query_img2_np_list, query_labels,
            # query_gts_json_view1, query_gts_json_view2, difficulties,
            # affine_matrix_view1, affine_matrix_view2, original_indices,
            # padded_support_imgs_np_list, padded_support_labels_list, padded_support_sgs_bytes_list,
            # num_support_per_problem, tree_indices, is_weights)
            
            # Extract necessary components from the batch for PerceptionModule.forward
            query_gts_json_view1 = batch[0][3] # List of bytes
            support_labels_flat_dummy = torch.zeros(batch[0][10].shape[0], dtype=torch.long, device=DEVICE) # Dummy for support_labels_flat
            
            # For support images, if DALI is used, it's already processed.
            # If PyTorch DataLoader, it's a list of numpy arrays, which needs to be processed.
            # However, the `model` (PerceptionModule) expects tensors for `support_images`.
            # So, if not DALI, we need to process `raw_support_images_flat_np` here.
            
            # This part is a bit tricky as `val_loader` yields raw numpy for DALI,
            # but for PyTorch DataLoader, it yields processed tensors from `custom_collate_fn`.
            # Let's assume `images` is `processed_query_images_view1` (tensor)
            # and we need to get `processed_support_images_reshaped` (tensor)
            # and `query_gts_json_view1` (list of bytes).
            
            # If using PyTorch DataLoader, `batch[0][9]` is `padded_support_imgs_np_list` (list of numpy)
            # and `batch[0][10]` is `padded_support_labels_list` (tensor).
            # We need to process `padded_support_imgs_np_list` into a tensor.
            
            if not (cfg['training']['use_dali'] and dali_image_processor):
                # If not DALI, support images are still raw numpy in the batch
                raw_support_images_flat_np_calib = batch[0][9]
                transform_support = T.Compose([
                    T.ToPILImage(),
                    T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_support_images_flat_calib = torch.stack([transform_support(img_np) for img_np in raw_support_images_flat_np_calib]).to(DEVICE)
                
                # Reshape for PerceptionModule
                batch_size_calib = images.shape[0]
                max_support_imgs_calib = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                processed_support_images_reshaped_calib = processed_support_images_flat_calib.view(
                    batch_size_calib, max_support_imgs_calib,
                    processed_support_images_flat_calib.shape[1], processed_support_images_flat_calib.shape[2], processed_support_images_flat_calib.shape[3]
                )
            else:
                # If DALI, `dali_image_processor.run` already returned `processed_support_images_flat`
                # which needs to be reshaped.
                # In this PTQ calibration loop, we only have `images` (query_img1),
                # so we need to re-run DALI for support images or assume they are passed.
                # Let's assume the `val_loader` is set up to provide the processed support images
                # if DALI is used, or we need to pass them explicitly.
                # For now, let's simplify and assume `model` can handle `None` for support images
                # during calibration if not strictly needed, or we adapt the `val_loader` output.
                
                # Given the `custom_collate_fn` returns `padded_support_imgs_np_list` (numpy)
                # and `padded_support_labels_list` (tensor), and `PerceptionModule` expects tensors,
                # we need to ensure the correct tensor is passed.
                # If DALI is used, the `dali_image_processor.run` in `training_step`
                # provides `processed_support_images_reshaped`.
                # Here in `quantize_model_ptq`, `val_loader` is a `DALIGenericIterator` if DALI is on.
                # The `DALIGenericIterator` output map is `["query_img1", "query_img2", "support_imgs_flat"]`.
                # So `batch` will be a dictionary of tensors.
                
                if cfg['training']['use_dali'] and dali_image_processor:
                    # `batch` is a dict from DALIGenericIterator
                    processed_support_images_flat_dali = batch['support_imgs_flat']
                    batch_size_calib = images.shape[0]
                    max_support_imgs_calib = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                    processed_support_images_reshaped_calib = processed_support_images_flat_dali.view(
                        batch_size_calib, max_support_imgs_calib,
                        processed_support_images_flat_dali.shape[1], processed_support_images_flat_dali.shape[2], processed_support_images_flat_dali.shape[3]
                    )
                else:
                    # This case should be handled by the `if not (cfg['training']['use_dali'] and dali_image_processor)` block
                    processed_support_images_reshaped_calib = None # Fallback if no support images or not processed
            
            _ = model(images,
                      ground_truth_json_strings=query_gts_json_view1,
                      support_images=processed_support_images_reshaped_calib,
                      support_labels_flat=support_labels_flat_dummy)
            if batch_idx >= cfg['training']['quantization'].get('calibration_batches', 10) - 1:
                break
    logger.info("Calibration finished.")
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    logger.info("Model converted to quantized version.")
    return model

# --- Knowledge Distillation Helper (from training.py, now separated) ---
def _get_ensemble_teacher_logits(teacher_models: nn.ModuleList,
                                 raw_images_np: List[np.ndarray],
                                 raw_gt_json_strings: List[bytes],
                                 raw_support_images_np: List[np.ndarray],
                                 distillation_config: Dict[str, Any],
                                 dali_image_processor: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Helper function to get logits from an ensemble of teacher models.
    Applies a mask for distillation if specified in config.
    """
    if not teacher_models:
        logger.warning("No teacher models provided for knowledge distillation.")
        return torch.empty(0), None
    
    all_teacher_logits = []
    for teacher_model in teacher_models:
        teacher_model.eval() # Set teacher to eval mode
        with torch.no_grad():
            # Process images using DALI or torchvision transforms
            if dali_image_processor:
                processed_query_images, _, processed_support_images_flat = dali_image_processor.run(
                    raw_images_np, # View 1 for teacher
                    [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_images_np), # Dummy view 2
                    raw_support_images_np
                )
            else:
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((CONFIG['data']['image_size'], CONFIG['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                processed_query_images = torch.stack([transform(img_np) for img_np in raw_images_np]).to(DEVICE)
                processed_support_images_flat = torch.stack([transform(img_np) for img_np in raw_support_images_np]).to(DEVICE)
            
            # Reshape flattened support images back to (B, N_support, C, H, W)
            batch_size_actual = processed_query_images.shape[0]
            max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_query_images.shape[1], processed_query_images.shape[2], processed_query_images.shape[3]
            )
            
            # Dummy support_labels_flat for teacher's forward pass if needed
            dummy_support_labels_flat = torch.zeros(batch_size_actual * max_support_imgs, dtype=torch.long, device=DEVICE)
            
            teacher_outputs = teacher_model(
                processed_query_images,
                ground_truth_json_strings=raw_gt_json_strings,
                support_images=processed_support_images_reshaped,
                support_labels_flat=dummy_support_labels_flat
            )
            all_teacher_logits.append(teacher_outputs['bongard_logits'])
    
    if not all_teacher_logits:
        return torch.empty(0), None
    
    # Average logits from ensemble
    avg_teacher_logits = torch.stack(all_teacher_logits, dim=0).mean(dim=0)
    
    distillation_mask = None
    if distillation_config.get('use_mask_distillation', False):
        # Example: Create a mask based on problem difficulty or teacher confidence
        # For now, a dummy mask (all ones) or based on a simple heuristic.
        # In a real scenario, this would involve more sophisticated logic, e.g.,
        # only distill from problems where teacher is highly confident, or easy problems.
        distillation_mask = torch.ones(avg_teacher_logits.shape[0], device=DEVICE) # All ones for now
        logger.debug("Distillation mask applied (dummy implementation).")
    return avg_teacher_logits, distillation_mask

# --- Mixup/Cutmix Augmenter (from training.py, now separated) ---
class MixupCutmixAugmenter:
    """
    Applies MixUp or CutMix augmentation.
    """
    def __init__(self, training_config: Dict[str, Any], num_classes: int):
        self.training_config = training_config
        self.num_classes = num_classes
        self.mixup_fn = None
        
        if HAS_TORCHVISION_V2:
            aug_type = training_config.get('mixup_cutmix_type', 'mixup')
            if aug_type == "mixup":
                self.mixup_fn = MixUp(num_classes=num_classes, alpha=training_config.get('mixup_alpha', 0.2))
                logger.info(f"MixUp enabled with alpha={training_config.get('mixup_alpha', 0.2)}")
            elif aug_type == "cutmix":
                self.mixup_fn = CutMix(num_classes=num_classes, alpha=training_config.get('cutmix_alpha', 1.0))
                logger.info(f"CutMix enabled with alpha={training_config.get('cutmix_alpha', 1.0)}")
            else:
                logger.warning(f"Unknown Mixup/CutMix type: {aug_type}. Augmentation disabled.")
        else:
            logger.warning("torchvision.transforms.v2 not available. MixUp/CutMix disabled.")

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mixup_fn:
            # MixUp/CutMix expects labels as (B,) or one-hot (B, num_classes)
            # If labels are (B,), it converts them internally.
            return self.mixup_fn(images, labels)
        return images, F.one_hot(labels, num_classes=self.num_classes).float() # Return one-hot if no mixup

# --- Main Training Loop Function ---
def run_training_once(cfg: Dict[str, Any], epochs: int) -> float:
    """
    Runs a single training session for the Bongard solver.
    Used for both standard training and as a trial function for HPO.
    Returns the final validation accuracy.
    """
    set_seed(cfg['training']['seed'])  # Ensure reproducibility for each trial
    
    # Initialize PyTorch Lightning Trainer
    # For HPO trials, use a simpler trainer without extensive logging/callbacks
    
    # Early Stopping Callback
    early_stop_callback = None
    if cfg['training'].get('use_early_stopping', False):
        early_stop_callback = EarlyStopping( # Using PL's EarlyStopping
            monitor=cfg['training']['early_stopping_monitor'],
            patience=cfg['training']['early_stopping_patience'],
            min_delta=cfg['training']['early_stopping_min_delta'],
            mode=cfg['training']['early_stopping_mode']
        )
        logger.info(f"Early Stopping enabled with monitor='{early_stop_callback.monitor}', patience={early_stop_callback.patience}")

    callbacks = []
    if early_stop_callback:
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu',
        devices=1,  # Single device for HPO trials to simplify
        max_epochs=epochs,
        precision=16 if cfg['training']['use_amp'] else 32,
        logger=False,  # No extensive logging for individual trials, handled by Ray Tune or custom
        enable_checkpointing=False,  # No checkpoints for trials
        enable_progress_bar=False,  # No progress bar for trials
        check_val_every_n_epoch=1,
        callbacks=callbacks, # Add early stopping callback
        # limit_train_batches=0.1, # Use a subset of data for faster trials
        # limit_val_batches=0.1,
    )
    
    # Initialize DataModule
    data_module = BongardDataModule(cfg)
    
    # Initialize Model
    model = LitBongard(cfg)
    
    # Initialize Teacher Models for Knowledge Distillation if enabled
    if cfg['training'].get('use_knowledge_distillation', False):
        teacher_paths = cfg['training']['distillation_config']['teacher_model_paths']
        for path in teacher_paths:
            try:
                teacher_model = PerceptionModule(cfg).to(DEVICE)
                teacher_checkpoint = torch.load(path, map_location=DEVICE)
                # Handle loading from Lightning checkpoint vs raw state_dict
                if 'state_dict' in teacher_checkpoint:
                    teacher_model_state_dict = {k.replace('perception_module.', ''): v for k, v in teacher_checkpoint['state_dict'].items() if k.startswith('perception_module.')}
                    teacher_model.load_state_dict(teacher_model_state_dict)
                else:
                    teacher_model.load_state_dict(teacher_checkpoint)
                teacher_model.eval()
                model.teacher_models.append(teacher_model)
                logger.info(f"Loaded teacher model from: {path}")
            except Exception as e:
                logger.error(f"Failed to load teacher model from {path}: {e}")
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Evaluate final performance on validation set
    results = trainer.validate(model, datamodule=data_module)
    val_accuracy = results[0]['val/accuracy']
    
    logger.info(f"Training finished. Final validation accuracy: {val_accuracy:.4f}")
    
    # QAT/PTQ Routines After Checkpointing (if enabled)
    if cfg['training']['quantization']['qat'] and HAS_TORCH_QUANTIZATION:
        logger.info("Applying QAT conversion after training.")
        model.eval() # Ensure eval mode before conversion
        # Use the model's perception_module for QAT preparation and conversion
        qat_prepared_perception_module = quantize_model_qat(model.perception_module, cfg)
        quantized_qat_model = torch.quantization.convert(qat_prepared_perception_module.eval(), inplace=False)
        logger.info("QAT model converted to quantized model.")
    elif cfg['training']['quantization']['ptq'] and HAS_TORCH_QUANTIZATION:
        logger.info("Applying PTQ after training (if not already done during pruning pipeline).")
        # Ensure the dataloader for calibration is available.
        # For this context, we can reuse the validation loader.
        val_loader_for_ptq = data_module.val_dataloader()
        dali_processor_for_ptq = data_module.dali_image_processor if cfg['training']['use_dali'] else None
        # Use the model's perception_module for PTQ
        quantized_ptq_model = quantize_model_ptq(model.perception_module, val_loader_for_ptq, dali_processor_for_ptq, cfg)
        logger.info("PTQ applied after training.")

    return val_accuracy

# --- Main Training Orchestration Functions ---
def run_training_pipeline(cfg: Dict[str, Any]):
    """
    Orchestrates the main training pipeline using PyTorch Lightning.
    Handles DDP, SimCLR pretraining, and supervised training.
    """
    set_seed(cfg['training']['seed'])
    
    # 1. SimCLR Pretraining (if enabled)
    if cfg['model']['simclr_config']['enabled']:
        logger.info("--- Starting SimCLR Pretraining ---")
        simclr_model = LitSimCLR(cfg)
        simclr_data_module = BongardDataModule(cfg)
        simclr_logger = None
        if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
            simclr_logger = WandbLogger(project="bongard_solver_simclr", name="simclr_pretrain")
        
        simclr_trainer = pl.Trainer(
            accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu',
            devices=cfg['training'].get('devices', 1),
            strategy='ddp' if cfg['training'].get('devices', 1) > 1 else 'auto',
            max_epochs=cfg['model']['simclr_config']['pretrain_epochs'],
            precision=16 if cfg['training']['use_amp'] else 32,
            logger=simclr_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=cfg['debug']['save_model_checkpoints'],
                    filename="simclr_best_model",
                    monitor="simclr_train_loss",
                    mode="min",
                    save_top_k=1
                ),
                LearningRateMonitor(logging_interval='epoch')
            ]
        )
        simclr_trainer.fit(simclr_model, datamodule=simclr_data_module)
        logger.info("SimCLR Pretraining completed.")
        # Load best SimCLR model weights into the main model's backbone
        best_simclr_checkpoint = os.path.join(cfg['debug']['save_model_checkpoints'], "simclr_best_model.ckpt")
        if os.path.exists(best_simclr_checkpoint):
            logger.info(f"Loading best SimCLR checkpoint from {best_simclr_checkpoint}")
            loaded_simclr_model = LitSimCLR.load_from_checkpoint(best_simclr_checkpoint, cfg=cfg)
            
            dummy_perception_module = PerceptionModule(cfg)
            target_backbone = dummy_perception_module.attribute_model.feature_extractor
            
            simclr_backbone_state_dict = {
                k.replace('feature_extractor.', ''): v
                for k, v in loaded_simclr_model.feature_extractor.state_dict().items()
                if k.startswith('feature_extractor.')
            }
            
            target_backbone.load_state_dict(simclr_backbone_state_dict, strict=False)
            logger.info("SimCLR pretrained backbone weights transferred to main model.")
        else:
            logger.warning("No SimCLR checkpoint found. Main model will not use SimCLR pretrained weights.")

    # 2. Supervised Bongard Solver Training
    logger.info("--- Starting Supervised Bongard Solver Training ---")
    model = LitBongard(cfg)
    data_module = BongardDataModule(cfg)
    # Load teacher models for Knowledge Distillation if enabled
    if cfg['training']['use_knowledge_distillation'] and cfg['training']['distillation_config']['teacher_model_paths']:
        logger.info("Loading teacher models for Knowledge Distillation.")
        for teacher_path in cfg['training']['distillation_config']['teacher_model_paths']:
            if os.path.exists(teacher_path):
                try:
                    teacher_model = PerceptionModule(cfg)
                    teacher_checkpoint = torch.load(teacher_path, map_location=DEVICE)
                    if 'state_dict' in teacher_checkpoint:
                        teacher_model_state_dict = {k.replace('perception_module.', ''): v for k, v in teacher_checkpoint['state_dict'].items() if k.startswith('perception_module.')}
                        teacher_model.load_state_dict(teacher_model_state_dict)
                    else:
                        teacher_model.load_state_dict(teacher_checkpoint)
                    teacher_model.eval()
                    model.teacher_models.append(teacher_model)
                    logger.info(f"Loaded teacher model from: {teacher_path}")
                except Exception as e:
                    logger.error(f"Failed to load teacher model from {teacher_path}: {e}")
            else:
                logger.warning(f"Teacher model checkpoint not found: {teacher_path}. Skipping.")
        if not model.teacher_models:
            logger.warning("No valid teacher models loaded. Knowledge Distillation will be disabled.")
            cfg['training']['use_knowledge_distillation'] = False

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg['debug']['save_model_checkpoints'],
            filename="best_bongard_model_{epoch:02d}-{val_accuracy:.4f}",
            monitor="val/accuracy",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # Early Stopping Callback
    if cfg['training'].get('use_early_stopping', False):
        callbacks.append(EarlyStopping(
            monitor=cfg['training']['early_stopping_monitor'],
            patience=cfg['training']['early_stopping_patience'],
            min_delta=cfg['training']['early_stopping_min_delta'],
            mode=cfg['training']['early_stopping_mode']
        ))
        logger.info(f"Early Stopping enabled with monitor='{cfg['training']['early_stopping_monitor']}', patience={cfg['training']['early_stopping_patience']}")

    # Logger
    main_logger = None
    if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
        main_logger = WandbLogger(project="bongard_solver_main", name="supervised_training", log_model='all')
        main_logger.watch(model, log='all', log_freq=cfg['training']['log_interval_batches'])

    trainer = pl.Trainer(
        accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu',
        devices=cfg['training'].get('devices', 1),
        strategy='ddp' if cfg['training'].get('devices', 1) > 1 else 'auto',
        max_epochs=cfg['training']['epochs'],
        precision=16 if cfg['training']['use_amp'] else 32,
        logger=main_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg['training']['log_interval_batches'],
        check_val_every_n_epoch=1,
        # limit_train_batches=100,
        # limit_val_batches=50,
    )
    
    trainer.fit(model, datamodule=data_module)
    logger.info("Supervised Bongard Solver Training completed.")
    
    # After training, load the best model for potential pruning/quantization
    best_model_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None
    final_model = None
    if best_model_path:
        logger.info(f"Loading best model from {best_model_path} for final evaluation/optimization.")
        final_model = LitBongard.load_from_checkpoint(best_model_path, cfg=cfg)
        perception_module_state_dict_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_perception_module.pth")
        torch.save(final_model.perception_module.state_dict(), perception_module_state_dict_path)
        logger.info(f"Best PerceptionModule state_dict saved to {perception_module_state_dict_path}")
        cfg['training']['pruning']['checkpoint'] = perception_module_state_dict_path # Update config for prune_quantize.py
    else:
        logger.warning("No best model checkpoint found after training. Using the last trained model for quantization attempts.")
        final_model = model # Use the last state of the model if no best checkpoint

    # QAT/PTQ invocation
    if HAS_TORCH_QUANTIZATION:
        if cfg['training']['quantization']['qat']:
            logger.info("Initiating Quantization Aware Training (QAT).")
            model_to_quantize_qat = final_model.perception_module if final_model else model.perception_module
            qat_prepared_model = quantize_model_qat(model_to_quantize_qat, cfg)
            # For true QAT, you would now fine-tune `qat_prepared_model` for a few epochs.
            # As per the request, we convert immediately for demonstration.
            quantized_qat_model = torch.quantization.convert(qat_prepared_model.eval(), inplace=False)
            torch.save(quantized_qat_model.state_dict(), os.path.join(cfg['debug']['save_model_checkpoints'], "quantized_qat_model.pth"))
            logger.info("QAT preparation and immediate conversion completed and quantized model saved. (Note: For full QAT benefits, fine-tuning after preparation is recommended.)")
        
        if cfg['training']['quantization']['ptq']:
            logger.info("Initiating Post-Training Quantization (PTQ).")
            val_dataloader_for_ptq = data_module.val_dataloader()
            dali_processor_for_ptq = data_module.dali_image_processor if cfg['training']['use_dali'] else None
            model_to_quantize_ptq = final_model.perception_module if final_model else model.perception_module
            quantized_ptq_model = quantize_model_ptq(model_to_quantize_ptq, val_dataloader_for_ptq, dali_processor_for_ptq, cfg)
            torch.save(quantized_ptq_model.state_dict(), os.path.join(cfg['debug']['save_model_checkpoints'], "quantized_ptq_model.pth"))
            logger.info("PTQ completed and quantized model saved.")
    else:
        logger.warning("PyTorch quantization is not enabled or available. Skipping QAT/PTQ steps.")

# --- Main execution block for direct training (not HPO) ---
if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config.yaml")
    
    # Set up basic logging if not already configured by Lightning
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Starting direct training run.")
    
    # Example: Override epochs for a quick test if needed
    # cfg['training']['epochs'] = 5
    
    # Call the main training pipeline function
    run_training_pipeline(cfg)
    logger.info(f"Direct training run completed.")
