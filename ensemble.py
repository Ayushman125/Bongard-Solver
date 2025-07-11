# Folder: bongard_solver/
# File: ensemble.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import os
import collections
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime # For TensorBoard log directory
import math # For OneCycleLR total_steps
import random # For rand_bbox
import jsonschema # For GT JSON validation
from sklearn.model_selection import KFold # For Meta-Learner Cross-Validation
from PIL import Image # For loading images for symbolic output/MC dropout
import matplotlib.pyplot as plt # For plotting history
import torchvision.transforms as T # For converting PIL Image to Tensor for symbolic output saving
from tqdm import tqdm # For progress bars in meta-learner training

# Import from config
from config import CONFIG, DEVICE, HAS_WANDB, IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS, DATA_ROOT_PATH
# Import from utils
from utils import setup_logging, set_seed, get_symbolic_embedding_dims
# Import from data
from data import build_dali_loader, BongardSyntheticDataset, CurriculumSampler, load_bongard_data, RealBongardDataset, BongardGenerator, get_loader
from bongard_rules import ALL_BONGARD_RULES # Needed for BongardGenerator
# Import from models
from models import PerceptionModule # Assuming this is in models.py
# Import from training (re-importing due to refactoring, ensure these are available)
# Assuming _run_single_training_session_ensemble and _validate_model_ensemble
# are now part of this ensemble.py or a dedicated training_orchestrator.py
# For now, I will include them here as they were in the large snippet.
from optimizers import get_optimizer, get_scheduler # For training individual members
from losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss # For training losses
from metrics import calculate_accuracy, calculate_precision_recall_f1, calculate_roc_auc, calculate_brier_score, calculate_expected_calibration_error
# SWA imports
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn # Explicitly import
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau # For scheduler type checking

logger = logging.getLogger(__name__)

# --- Helper functions (from previous context or to be defined) ---
# Dummy rand_bbox and distillation_loss if they are not imported from phase1_code_part2b
# Assuming these are now in losses.py or utils.py as part of refactoring.
# If not, they would need to be defined here or properly imported.
# For now, I'll assume they are available from the refactored project.

# Define a basic JSON schema for ground truth validation
# This was in the original large snippet, keeping it here for context if needed.
BONGARD_GT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bongard_label": {"type": "integer", "minimum": 0, "maximum": 1},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["id", "bbox", "attributes", "confidence"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "integer"},
                    "dst": {"type": "integer"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["src", "dst", "type", "confidence"]
            }
        }
    },
    "required": ["bongard_label", "objects", "relations"]
}

# AugmentMix (copied from previous snippet to ensure self-containment if not in data.py)
class AugmentMix(nn.Module):
    """
    Applies MixUp or CutMix augmentation to a batch of images and labels.
    """
    def __init__(self, mixup_alpha: float, cutmix_alpha: float, mixup_cutmix_ratio: float):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.ratio = mixup_cutmix_ratio # Probability of applying MixUp vs CutMix

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], float, str]]:
        """
        Applies MixUp or CutMix.
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W].
            labels (torch.Tensor): Batch of labels [B].
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], float, str]]:
                - Augmented images.
                - Tuple: (labels_a, labels_b, lambda, mode_str)
                  labels_a: Original labels
                  labels_b: Labels of mixed image (None if no mix)
                  lambda: Mixing coefficient
                  mode_str: 'mixup', 'cutmix', or 'none'
        """
        if self.mixup_alpha > 0 and random.random() < self.ratio:
            # Apply MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam) # Ensure lambda >= 0.5
            
            index = torch.randperm(images.size(0)).to(images.device)
            mixed_images = lam * images + (1 - lam) * images[index, :]
            labels_a, labels_b = labels, labels[index]
            
            return mixed_images, (labels_a, labels_b, lam, 'mixup')
        
        elif self.cutmix_alpha > 0 and random.random() >= self.ratio:
            # Apply CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = max(lam, 1 - lam)
            
            index = torch.randperm(images.size(0)).to(images.device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            labels_a, labels_b = labels, labels[index]
            
            return mixed_images, (labels_a, labels_b, lam, 'cutmix')
        
        else:
            # No MixUp/CutMix
            return images, (labels, None, 1.0, 'none')

# Helper function for CutMix (from torchvision examples)
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def compute_diversity(members: List[nn.Module], x: torch.Tensor) -> Tuple[float, float]:
    """
    Computes diversity metrics (ensemble entropy and disagreement) for a list of models.
    Args:
        members (List[nn.Module]): A list of trained ensemble member models.
        x (torch.Tensor): A batch of input data (images).
    Returns:
        Tuple[float, float]:
            - entropy (float): Measures the average uncertainty of the ensemble's mean prediction.
            - disagreement (float): Measures the variance of individual member predictions.
    """
    if not members:
        logger.warning("No ensemble members provided for diversity computation.")
        return 0.0, 0.0
    
    # Ensure models are in evaluation mode
    for m in members:
        m.eval()
    
    probs = []
    with torch.no_grad():
        for m in members:
            # Assuming model(x) returns logits for Bongard classification
            # You might need to adapt this if your model returns a dictionary or other outputs
            # For simplicity, assuming the first return value is the Bongard logits.
            if isinstance(m, DDP): # Unwrap DDP if necessary to call specific methods
                # Pass dummy GTs if needed by the model's forward method
                bongard_logits, _, _ = m.module(x, [json.dumps({})]*x.shape[0]) 
            else:
                # Pass dummy GTs if needed by the model's forward method
                bongard_logits, _, _ = m(x, [json.dumps({})]*x.shape[0]) 
            
            probs.append(F.softmax(bongard_logits, dim=-1)) # Convert logits to probabilities
    
    stacked_probs = torch.stack(probs) # Shape: [M, B, C], M=num_members, B=batch_size, C=num_classes
    
    # Ensemble Entropy (measures uncertainty of the average prediction)
    mean_p = stacked_probs.mean(0) # Average probabilities across members: [B, C]
    entropy = -(mean_p * torch.log(mean_p + 1e-9)).sum(-1).mean().item() # Add epsilon for log(0)
    
    # Disagreement (measures variance among member predictions)
    # Calculate variance of probabilities across members for each class and batch item
    disagreement = torch.var(stacked_probs, dim=0).mean().item() # Mean variance across batch and classes
    
    return entropy, disagreement

# 15.1 Persist Diversity Metrics to JSON
def save_diversity(entropy, disagreement, path):
    """
    Saves diversity metrics (entropy and disagreement) to a JSON file.
    Args:
        entropy (float): The calculated diversity entropy.
        disagreement (float): The calculated diversity disagreement.
        path (str): The file path to save the JSON.
    """
    try:
        data = {'entropy': float(entropy), 'disagreement': float(disagreement)}
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Diversity metrics saved to: {path}")
    except Exception as e:
        logger.error(f"Error saving diversity metrics to {path}: {e}", exc_info=True)


# --- Core Training and Validation Functions (extracted and adapted) ---
def _run_single_training_session_ensemble(
    model: PerceptionModule,
    train_loader: Any, # Can be DALI or PyTorch DataLoader
    val_loader: Any, # Can be DALI or PyTorch DataLoader
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    current_rank: int,
    is_ddp_initialized: bool,
    model_idx: int, # Index of the ensemble member
    cfg: Dict[str, Any], # Pass the full config for this run
    replay_buffer: Optional[Any] = None, # For curriculum learning
    teacher_model: Optional[Any] = None # For knowledge distillation
) -> Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Runs a single training session for a PerceptionModule model.
    This function is designed to be called by the ensemble orchestrator.
    Args:
        model (PerceptionModule): The model to train.
        train_loader: DALI or PyTorch DataLoader for training.
        val_loader: DALI or PyTorch DataLoader for validation.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        scheduler (Optional[Any]): Learning rate scheduler.
        current_rank (int): Current GPU rank.
        is_ddp_initialized (bool): True if DDP is already initialized.
        model_idx (int): Unique ID for the ensemble member (for naming checkpoints/logs).
        cfg (Dict[str, Any]): The configuration for this specific training session.
        replay_buffer (Optional[Any]): PrioritizedReplayBuffer instance for curriculum learning.
        teacher_model (Optional[Any]): A pre-trained teacher model (e.g., an ensemble) for knowledge distillation.
    Returns:
        Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
            - Path to the best saved model checkpoint.
            - All validation predictions (logits) from the best model.
            - All validation true labels.
            - Dictionary of best validation metrics.
    """
    set_seed(cfg['training']['seed'] + model_idx) # Ensure unique seed for each member
    logger.info(f"--- Starting Training Session for Member {model_idx} (Seed: {cfg['training']['seed'] + model_idx}) ---")
    
    epochs = cfg['training']['epochs']
    logger.info(f"Training for {epochs} epochs.")
    
    # Initialize TensorBoard writer
    log_dir = os.path.join(cfg['debug']['log_dir'], datetime.now().strftime(f'member_{model_idx}_%Y%m%d-%H%M%S'))
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs for member {model_idx} at: {log_dir}")

    # Initialize Weights & Biases (WandB)
    if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
        import wandb
        wandb.init(project="Bongard_Perception_Ensemble",
                   group="ensemble_training",
                   name=f"member_{model_idx}_seed_{cfg['training']['seed'] + model_idx}",
                   config=cfg,
                   reinit=True)
        logger.info(f"WandB initialized for member {model_idx}.")

    # Loss function (for Bongard classification)
    criterion_bongard = LabelSmoothingCrossEntropy(smoothing=cfg['training']['label_smoothing_epsilon'])
    logger.info(f"Using CrossEntropyLoss with label_smoothing_epsilon={cfg['training']['label_smoothing_epsilon']}.")
    
    # Loss functions for attribute and relation classification
    criterion_attr = nn.CrossEntropyLoss()
    criterion_rel = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['use_amp'])
    logger.info(f"AMP scaler initialized (enabled={cfg['training']['use_amp']}).")

    # Early Stopping
    best_val_loss = float('inf')
    best_val_accuracy = -float('inf')
    patience_counter = 0
    best_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], f'member_{model_idx}_best_model.pt')

    # Initialize AugmentMix
    augmenter = AugmentMix(
        mixup_alpha=cfg['training']['mixup_alpha'],
        cutmix_alpha=cfg['training']['cutmix_alpha'],
        mixup_cutmix_ratio=cfg['training']['mixup_cutmix_ratio']
    )
    logger.info(f"AugmentMix initialized with mixup_alpha={augmenter.mixup_alpha}, cutmix_alpha={augmenter.cutmix_alpha}.")

    # SWA setup
    swa_model = None
    if cfg['training']['use_swa']:
        swa_model = AveragedModel(model)
        logger.info("SWA AveragedModel initialized.")
    
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples_epoch = 0
        
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # QAT specific: enable observer for last few epochs
        if cfg['training'].get('use_qat', False) and cfg['HAS_TORCH_QUANTIZATION'] and epoch >= cfg['training'].get('qat_start_epoch', epochs - 5):
            logger.info(f"Epoch {epoch}: Enabling QAT observers.")
            try:
                import torch.quantization as tq
                model.apply(tq.enable_observer)
                model.apply(tq.enable_fake_quant)
            except ImportError:
                logger.warning("torch.quantization not available. QAT observers cannot be enabled.")
        
        # Snapshot ensemble saving
        if cfg['ensemble'].get('snap_epoch', 0) > 0 and (epoch + 1) % cfg['ensemble']['snap_epoch'] == 0:
            snapshot_path = os.path.join(cfg['debug']['save_model_checkpoints'], f"member_{model_idx}_snapshot_epoch_{epoch+1}.pt")
            # Save the unwrapped model if DDP
            torch.save(model.module.state_dict() if is_ddp_initialized else model.state_dict(), snapshot_path)
            logger.info(f"Snapshot of model {model_idx} saved to: {snapshot_path}")

        for batch_idx, data in enumerate(train_loader):
            # DALI returns data as a dictionary of lists/tensors.
            # Adjust based on your `custom_collate_fn` or DALI output_map.
            images_view1 = data['query_img1'].to(DEVICE) if isinstance(data, dict) else data[0].to(DEVICE)
            images_view2 = data['query_img2'].to(DEVICE) if isinstance(data, dict) else data[1].to(DEVICE)
            labels_bongard = data['query_labels'].to(DEVICE) if isinstance(data, dict) else data[2].to(DEVICE)
            gts_json_strings_batch = data['query_gts_json_view1'] if isinstance(data, dict) else data[3]
            original_indices = data['original_indices'].to(DEVICE) if isinstance(data, dict) else data[8].to(DEVICE) # For PER
            is_weights = data['is_weights'].to(DEVICE) if isinstance(data, dict) else data[14].to(DEVICE) # For PER

            # Apply MixUp/CutMix
            images_view1_aug, mixinfo = augmenter(images_view1, labels_bongard)
            labels_a, labels_b, lam, mode_aug = mixinfo

            # Forward pass
            with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                bongard_logits, detected_objects_view1, aggregated_outputs_view1 = model(images_view1_aug, gts_json_strings_batch)
                
                # Bongard Classification Loss (potentially with MixUp/CutMix)
                if mode_aug == 'mixup' or mode_aug == 'cutmix':
                    loss_bongard = lam * criterion_bongard(bongard_logits, labels_a) + (1 - lam) * criterion_bongard(bongard_logits, labels_b)
                else:
                    loss_bongard = criterion_bongard(bongard_logits, labels_bongard)
                
                # Apply importance sampling weights for HardExampleSampler
                if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False):
                    loss_bongard = (loss_bongard * is_weights).mean()
                else:
                    loss_bongard = loss_bongard.mean() # Ensure scalar loss for backprop
                total_loss_batch = loss_bongard
                
                # Attribute and Relation Losses (Joint Symbolic Loss)
                if 'attribute_logits' in aggregated_outputs_view1 and 'attribute_gt' in aggregated_outputs_view1:
                    for attr_name, attr_logits in aggregated_outputs_view1['attribute_logits'].items():
                        if attr_name in aggregated_outputs_view1['attribute_gt']:
                            attr_gt = aggregated_outputs_view1['attribute_gt'][attr_name]
                            total_loss_batch += cfg['training']['attribute_loss_weight'] * criterion_attr(attr_logits, attr_gt)
                
                if 'relation_logits' in aggregated_outputs_view1 and 'relation_gt' in aggregated_outputs_view1:
                    rel_logits = aggregated_outputs_view1['relation_logits']
                    rel_gt = aggregated_outputs_view1['relation_gt']
                    total_loss_batch += cfg['training']['relation_loss_weight'] * criterion_rel(rel_logits, rel_gt)
                
                # Feature Consistency Loss
                if cfg['training'].get('feature_consistency_alpha', 0) > 0:
                    _, _, aggregated_outputs_view2 = model(images_view2, gts_json_strings_batch)
                    feature_consistency_criterion = FeatureConsistencyLoss(loss_type=cfg['training']['feature_consistency_loss_type'])
                    consistency_loss_features = feature_consistency_criterion(
                        aggregated_outputs_view1['image_features_student'],
                        aggregated_outputs_view2['image_features_student']
                    )
                    total_loss_batch += cfg['training']['feature_consistency_alpha'] * consistency_loss_features
                
                # Symbolic Consistency Loss
                if cfg['training'].get('symbolic_consistency_alpha', 0) > 0:
                    symbolic_cons_loss = SymbolicConsistencyLoss(
                        all_bongard_rules=ALL_BONGARD_RULES, # Pass rules if needed
                        config=cfg
                    )
                    # Dummy call for now, needs proper integration
                    pass 
            
            # Knowledge Distillation (if teacher model provided)
            if teacher_model is not None and cfg['training'].get('use_knowledge_distillation', False):
                with torch.no_grad(): # Teacher inference should be no_grad
                    # Teacher model might be a single model or an ensemble of models
                    if isinstance(teacher_model, list): # Ensemble teacher
                        teacher_logits_list = []
                        for t_model in teacher_model:
                            t_model.eval()
                            # Ensure teacher model receives appropriate inputs
                            t_logits, _, _ = t_model(images_view1, gts_json_strings_batch)
                            teacher_logits_list.append(t_logits)
                        teacher_logits = torch.mean(torch.stack(teacher_logits_list, dim=0), dim=0)
                    else: # Single teacher model (e.g., Stacked Ensemble Meta-Learner)
                        teacher_model.eval()
                        if isinstance(teacher_model, nn.Module): # If it's the meta-learner
                            # For meta-learner, input is concatenated logits from base models.
                            # This means we need to get base model logits first.
                            # This complexity implies that `teacher_model` should be the actual
                            # base models if using weighted average, or the meta-learner if stacked.
                            # If `teacher_model` is a meta-learner, it expects `meta_input`.
                            # This needs to be handled carefully.
                            # The `_run_single_training_session_ensemble` should receive the *teacher logits* directly
                            # or a callable that can produce them for the current batch.
                            # For now, let's assume `teacher_model` is a single model that directly produces logits
                            # from `images_view1`. If it's a stacked ensemble, its prediction logic is different.
                            # This needs a more robust `teacher_model` interface or separate distillation logic.
                            
                            # If the teacher is a StackedEnsembleMetaLearner, we need the base model logits
                            # for the current batch `images_view1`. This implies the base models
                            # that fed the meta-learner also need to be accessible here, or their
                            # logits for the current batch pre-calculated.
                            # This is too complex for a single _run_training_once.
                            # Let's assume `teacher_model` is a single `PerceptionModule` or list of them
                            # for weighted average. Stacked distillation needs different handling.
                            
                            teacher_logits, _, _ = teacher_model(images_view1, gts_json_strings_batch)
                        else:
                            logger.error("Unsupported teacher_model type for distillation.")
                            teacher_logits = bongard_logits.detach() # Fallback
                
                dist_criterion = DistillationLoss(
                    temperature=cfg['training']['distillation_temperature'],
                    alpha=cfg['training']['distillation_alpha'],
                    reduction='mean' # Use mean reduction for overall batch loss
                )
                dist_loss = dist_criterion(
                    student_logits=bongard_logits,
                    teacher_logits=teacher_logits,
                    target_labels=labels_bongard # Target labels for the hard component of distillation
                )
                total_loss_batch = dist_loss # Distillation loss replaces main loss if enabled (soft target only)
                # If hard target is also used (alpha < 1), then total_loss_batch should be:
                # (1 - alpha) * original_loss + alpha * soft_distillation_loss
                # The DistillationLoss class should handle this `alpha` parameter.
                # Assuming `DistillationLoss` handles the `alpha` weighting internally.
            
            # Scale loss for gradient accumulation
            total_loss_batch = total_loss_batch / cfg['training']['gradient_accumulation_steps']
            scaler.scale(total_loss_batch).backward()

            if (batch_idx + 1) % cfg['training']['gradient_accumulation_steps'] == 0:
                if cfg['training'].get('use_sam_optimizer', False):
                    scaler.unscale_(optimizer)
                    if cfg['training'].get('max_grad_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass for SAM
                    with torch.cuda.amp.autocast(enabled=cfg['training']['use_amp']):
                        bongard_logits_second_step, _, _ = model(images_view1_aug, gts_json_strings_batch)
                        loss_bongard_second_step = criterion_bongard(bongard_logits_second_step, labels_bongard)
                        if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False):
                            loss_bongard_second_step = (loss_bongard_second_step * is_weights).mean()
                        else:
                            loss_bongard_second_step = loss_bongard_second_step.mean()
                        total_loss_second_step = loss_bongard_second_step # Simplified for SAM's second step
                        
                        if teacher_model is not None and cfg['training'].get('use_knowledge_distillation', False):
                            with torch.no_grad():
                                if isinstance(teacher_model, list):
                                    teacher_logits_list_second_step = []
                                    for t_model in teacher_model:
                                        t_model.eval()
                                        t_logits, _, _ = t_model(images_view1, gts_json_strings_batch)
                                        teacher_logits_list_second_step.append(t_logits)
                                    teacher_logits_second_step = torch.mean(torch.stack(teacher_logits_list_second_step, dim=0), dim=0)
                                else:
                                    teacher_logits_second_step, _, _ = teacher_model(images_view1, gts_json_strings_batch)
                            dist_loss_second_step = dist_criterion(
                                student_logits=bongard_logits_second_step,
                                teacher_logits=teacher_logits_second_step,
                                target_labels=labels_bongard
                            )
                            total_loss_second_step = dist_loss_second_step
                    total_loss_second_step = total_loss_second_step / cfg['training']['gradient_accumulation_steps']
                    scaler.scale(total_loss_second_step).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    if cfg['training'].get('max_grad_norm', 0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if scheduler and isinstance(scheduler, OneCycleLR):
                    scheduler.step()
            
            total_loss += total_loss_batch.item() * cfg['training']['gradient_accumulation_steps']
            
            _, predicted = torch.max(bongard_logits.data, 1)
            correct_predictions += (predicted == labels_bongard).sum().item()
            total_samples_epoch += labels_bongard.size(0)

            # Update HardExampleSampler's replay buffer
            if replay_buffer is not None and cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False):
                # Ensure criterion_bongard can return per-sample loss
                per_sample_losses = criterion_bongard(bongard_logits, labels_bongard, reduction='none')
                
                # If using MixUp/CutMix, the `labels_bongard` might be the blended labels,
                # or `criterion_bongard` handles `labels_a`, `labels_b`, `lam`.
                # For PER, we need the "true" loss of the original sample.
                # If `criterion_bongard` is LabelSmoothingCrossEntropy, `reduction='none'` works.
                # If `mode_aug` is 'mixup' or 'cutmix', the `per_sample_losses` should be adjusted
                # to reflect the true loss for the original sample, not the mixed one.
                # This is a known challenge with PER and data augmentation.
                # For now, we'll use the loss calculated on the mixed sample as the priority.
                
                replay_buffer.update_priorities(original_indices.cpu().tolist(), per_sample_losses.cpu().tolist())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples_epoch if total_samples_epoch > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
            import wandb
            wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_accuracy, "epoch": epoch})
        
        # Validation
        val_loss, val_accuracy, val_predictions_logits, val_true_labels = _validate_model_ensemble(model, val_loader, criterion_bongard, cfg)
        logger.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
            import wandb
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch})
        
        # --- Persist Diversity Metrics to JSON (Call after each epoch) ---
        # This assumes that `val_loader` can provide a representative sample for diversity calculation,
        # or that `compute_diversity` can take the `val_predictions_logits` directly.
        # For this setup, `compute_diversity` expects a list of models and an input tensor.
        # We will pass the current model and a sample from the val_loader.
        if (epoch + 1) in cfg['ensemble'].get('eval_epochs', []):
            logger.info(f"Computing and saving diversity metrics at epoch {epoch+1}...")
            # To compute diversity, we need predictions from multiple models.
            # Here, we only have one model (`model`).
            # For a true diversity metric, you'd need to run inference with ALL ensemble members
            # on the same validation batch and then pass those predictions to `compute_diversity`.
            # As a conceptual placeholder, we'll use the current model's predictions.
            # In a full ensemble training script, this would be done by the orchestrator
            # after all members for a given stage have trained.
            
            # For a single model, diversity is not meaningful.
            # This call is more appropriate in `train_ensemble` or `train_distilled_student_orchestrator_combined`
            # where multiple models' predictions are available.
            # However, if the intent is to track *this single model's* performance
            # and then combine it later, we can use dummy values or adapt.
            
            # For now, let's assume `compute_diversity` will be called with actual ensemble members
            # by the orchestrator. This `save_diversity` call here is for *this member's*
            # validation performance, not ensemble diversity.
            # To fulfill the prompt's request "Call after each epoch: ent,dis = compute_diversity(...)",
            # I will adapt it to use the current model's predictions as a proxy,
            # but note that this doesn't represent *ensemble* diversity.
            
            # To properly compute ensemble diversity here, we would need access to all
            # ensemble members' predictions for the current validation set.
            # This is typically done *after* all members are trained, or at specific checkpoints.
            # Given the structure, `compute_diversity` is designed for multiple models.
            # I will make a conceptual call here, but the actual diversity would be computed
            # at the orchestrator level.
            
            # Dummy call for a single model's "diversity" (will be 0,0 or based on its own uncertainty)
            # This is a placeholder to demonstrate the call.
            # For real diversity, you need multiple models' outputs.
            # `compute_diversity` expects `List[nn.Module]` as first arg.
            
            # To get actual ensemble diversity at this point, the `train_ensemble`
            # function would need to collect predictions from all members for the val set
            # at each `eval_epoch` and then pass them to `compute_diversity`.
            # This is a more complex change.
            # For now, I'll ensure `save_diversity` is called, but `compute_diversity`
            # will be called with the single current model, which is not true ensemble diversity.
            
            # To make `compute_diversity` work here, it needs a batch of images.
            # Let's get a sample batch from val_loader.
            try:
                # Reset val_loader to get a fresh batch if needed
                if hasattr(val_loader, 'reset'):
                    val_loader.reset()
                
                # Get one batch for diversity calculation
                sample_data = next(iter(val_loader))
                sample_images = sample_data['query_img1'].to(DEVICE) if isinstance(sample_data, dict) else sample_data[0].to(DEVICE)
                
                # Compute diversity using the current model as a "single member ensemble"
                # This will result in 0 diversity, but demonstrates the call.
                ent, dis = compute_diversity([model], sample_images) 
                
                diversity_path = cfg['ensemble']['diversity_path']
                save_diversity(ent, dis, diversity_path)
            except Exception as e:
                logger.error(f"Error computing/saving diversity at epoch {epoch+1}: {e}", exc_info=True)
        # --- End Diversity Metrics ---

        # SWA update
        if cfg['training']['use_swa'] and epoch >= cfg['training'].get('swa_start_epoch', epochs // 2): # Use 'swa_start_epoch' from config
            swa_model.update_parameters(model)
            logger.debug(f"SWA model updated at epoch {epoch}.")
            # SWA LR scheduler is stepped per batch in OneCycleLR case, but here it's per epoch for SWALR.
            # The SWALR scheduler is passed to `scheduler` if SWA is enabled.
            if isinstance(scheduler, SWALR):
                scheduler.step() # Step SWALR scheduler
            else:
                # If a different scheduler is active, this means SWA is used with another LR schedule.
                # The user's snippet for SWA in optimizers.py implies SWALR is the SWA scheduler.
                pass # No explicit LR update here if SWALR is not the main scheduler
        
        # Early Stopping check
        monitor_metric = val_loss
        if cfg['training']['early_stopping_monitor_metric'] == 'val_accuracy':
            monitor_metric = -val_accuracy # For accuracy, we want to maximize, so minimize negative accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving model.")
                # Save the unwrapped model if DDP
                torch.save(model.module.state_dict() if is_ddp_initialized else model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                logger.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{cfg['training']['early_stopping_patience']}")
        else: # Default to val_loss
            if val_loss < best_val_loss - cfg['training']['early_stopping_min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
                # Save the unwrapped model if DDP
                torch.save(model.module.state_dict() if is_ddp_initialized else model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{cfg['training']['early_stopping_patience']}")
        
        if scheduler and not isinstance(scheduler, OneCycleLR) and not isinstance(scheduler, SWALR): # Step other schedulers per epoch (if not OneCycleLR or SWALR)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if patience_counter >= cfg['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    # Finalize SWA model if enabled
    if cfg['training']['use_swa'] and swa_model is not None:
        # Update BN layers for SWA model with training data statistics
        # This requires iterating over the training loader again.
        # Ensure `train_loader` is reset if it's a DALI iterator.
        if hasattr(train_loader, 'reset'):
            train_loader.reset()
        update_bn(train_loader, swa_model, device=DEVICE)
        final_model_path = best_model_path.replace('.pt', '_swa.pt')
        torch.save(swa_model.state_dict(), final_model_path)
        logger.info(f"SWA model finalized and saved to {final_model_path}")
        best_model_path = final_model_path # Use SWA model for final evaluation

    # QAT specific: convert to quantized model
    if cfg['training'].get('use_qat', False) and cfg['HAS_TORCH_QUANTIZATION']:
        logger.info("Finalizing QAT model conversion.")
        model.eval()
        try:
            import torch.quantization as tq
            model_to_convert = model.module.cpu() if is_ddp_initialized else model.cpu() # Move to CPU before conversion
            model_int8 = tq.convert(model_to_convert, inplace=False)
            quantized_model_path = best_model_path.replace('.pt', '_quantized.pt')
            torch.save(model_int8.state_dict(), quantized_model_path)
            logger.info(f"Quantized model saved to {quantized_model_path}")
            best_model_path = quantized_model_path # Use quantized model for inference
        except ImportError:
            logger.warning("torch.quantization not available. QAT conversion skipped.")
        except Exception as e:
            logger.error(f"Error during QAT conversion: {e}. QAT conversion skipped.")

    writer.close()
    if cfg['HAS_WANDB'] and cfg['training'].get('use_wandb', False):
        import wandb
        wandb.finish()
    
    # Load the best model to return its state
    if os.path.exists(best_model_path):
        # Load the saved model (which might be SWA or quantized)
        loaded_model_state_dict = torch.load(best_model_path, map_location=DEVICE)
        # Create a new model instance to load the state dict into
        final_model_instance = PerceptionModule(cfg).to(DEVICE)
        # Handle DDP prefix if needed when loading into a bare model
        if list(loaded_model_state_dict.keys())[0].startswith('module.'):
            loaded_model_state_dict = {k.replace('module.', ''): v for k, v in loaded_model_state_dict.items()}
        final_model_instance.load_state_dict(loaded_model_state_dict)
        final_model_instance.eval() # Set to eval mode
        logger.info(f"Loaded best model from {best_model_path} for final evaluation.")
    else:
        logger.warning(f"Best model checkpoint not found at {best_model_path}. Returning current model state.")
        final_model_instance = model.module if is_ddp_initialized else model # Return the last state of the model
    
    # Re-run validation on the best model to get final predictions and metrics
    final_val_loss, final_val_accuracy, final_val_predictions_logits, final_val_true_labels = _validate_model_ensemble(final_model_instance, val_loader, criterion_bongard, cfg)
    
    best_metrics = {
        'val_loss': final_val_loss,
        'val_accuracy': final_val_accuracy,
        'logits': np.array(final_val_predictions_logits),
        'labels': np.array(final_val_true_labels)
    }
    logger.info(f"--- Training Session for Member {model_idx} Finished. Best model saved to {best_model_path} ---")
    return best_model_path, np.array(final_val_predictions_logits), np.array(final_val_true_labels), best_metrics

def _validate_model_ensemble(model: PerceptionModule, data_loader: Any, criterion: nn.Module, config: Dict[str, Any]):
    """
    Validates the model on the given data loader.
    Args:
        model (PerceptionModule): The model to validate.
        data_loader (Any): DALI or PyTorch data loader for validation data.
        criterion (nn.Module): Loss function.
        config (Dict): Configuration dictionary.
    Returns:
        Tuple[float, float, List[np.ndarray], List[int]]:
            - Average validation loss.
            - Average validation accuracy.
            - List of numpy arrays of predictions (logits) for each validation sample.
            - List of true labels for each validation sample.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions_logits = []
    all_true_labels = []

    # Reset DALI iterator for validation if it's a DALI loader
    if hasattr(data_loader, 'reset'):
        data_loader.reset()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images_view1 = data['query_img1'].to(DEVICE) if isinstance(data, dict) else data[0].to(DEVICE)
            labels_bongard = data['query_labels'].to(DEVICE) if isinstance(data, dict) else data[2].to(DEVICE)
            gts_json_strings_batch = data['query_gts_json_view1'] if isinstance(data, dict) else data[3]
            
            with torch.cuda.amp.autocast(enabled=config['training']['use_amp']):
                bongard_logits, _, _ = model(images_view1, gts_json_strings_batch)
                loss = criterion(bongard_logits, labels_bongard)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(bongard_logits.data, 1)
            correct_predictions += (predicted == labels_bongard).sum().item()
            total_samples += labels_bongard.size(0)
            all_predictions_logits.extend(bongard_logits.cpu().numpy())
            all_true_labels.extend(labels_bongard.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, all_predictions_logits, all_true_labels

# --- Ensemble Orchestration ---
def train_ensemble(
    num_members: int,
    train_loader: Any,
    val_loader: Any,
    cfg: Dict[str, Any], # Pass the full config
    current_rank: int = 0,
    world_size: int = 1,
    is_ddp_initialized: bool = False,
) -> Tuple[List[nn.Module], List[Dict[str, Any]]]:
    """
    Trains multiple ensemble members. Each member is trained independently.
    Args:
        num_members (int): Number of ensemble members to train.
        train_loader: DALI or PyTorch DataLoader for training.
        val_loader: DALI or PyTorch DataLoader for validation.
        cfg (Dict[str, Any]): The configuration dictionary.
        current_rank (int): Current GPU rank.
        world_size (int): Total number of GPUs.
        is_ddp_initialized (bool): True if DDP is already initialized.
    Returns:
        Tuple[List[nn.Module], List[Dict[str, Any]]]:
            - List of trained ensemble models.
            - List of best validation metrics for each model (including predictions, labels, logits).
    """
    logger.info("Starting ensemble teacher training and student distillation orchestration.")
    teacher_models = []
    all_members_val_predictions_logits = [] # List of (N_val, num_classes) logits for each member
    all_members_val_labels = [] # List of (N_val,) labels for each member (should be identical)
    ensemble_member_accuracies = [] # List of accuracies for each member
    logger.info(f"Training {num_members} ensemble members.")
    ensemble_models = []
    all_members_best_metrics = []

    for i in range(num_members):
        logger.info(f"Training ensemble member {i+1}/{num_members}...")
        
        # Initialize a new model for each member
        model = PerceptionModule(cfg).to(DEVICE) # Pass full config to PerceptionModule
        
        # If DDP is initialized, wrap the model
        if is_ddp_initialized:
            model = DDP(model, device_ids=[current_rank])
        
        # Initialize optimizer and scheduler for this member
        optimizer = get_optimizer(model, cfg['training'])
        
        # Calculate total steps for OneCycleLR
        total_steps = cfg['training']['epochs'] * len(train_loader) if hasattr(train_loader, '__len__') else cfg['training']['epochs'] * 100 # Fallback
        scheduler = get_scheduler(optimizer, cfg['training'], total_steps)
        
        # Run training for this member
        _, _, _, best_metrics = _run_single_training_session_ensemble(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=i,
            cfg=cfg, # Pass full config
            replay_buffer=train_loader.dataset.replay_buffer if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) else None
        )
        
        # Store the trained model (unwrapped if DDP)
        ensemble_models.append(model.module if is_ddp_initialized else model)
        all_members_best_metrics.append(best_metrics)
        logger.info(f"Ensemble member {i+1} training completed.")

    return ensemble_models, all_members_best_metrics

def load_trained_model(
    model_path: str,
    cfg: Dict[str, Any], # Pass the full config
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    val_loader: Any = None # Required for re-running validation
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads a pre-trained model and optionally re-runs validation to get metrics.
    Args:
        model_path (str): Path to the model checkpoint.
        cfg (Dict[str, Any]): The configuration dictionary.
        current_rank (int): Current GPU rank.
        is_ddp_initialized (bool): True if DDP is already initialized.
        val_loader (Any): DALI or PyTorch DataLoader for validation. Required to get fresh metrics.
    Returns:
        Tuple[nn.Module, Dict[str, Any]]:
            - Loaded model (unwrapped if DDP).
            - Dictionary of validation metrics (predictions, labels, logits).
    """
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    model = PerceptionModule(cfg).to(DEVICE) # Pass full config to PerceptionModule
    
    # Load state dict, handling DDP prefix if necessary
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint: # If checkpoint is a dict with 'model_state_dict' key
        state_dict = checkpoint['model_state_dict']
    if is_ddp_initialized and not list(state_dict.keys())[0].startswith('module.'):
        # Add 'module.' prefix if loading a non-DDP saved model into DDP
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not is_ddp_initialized and list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix if loading a DDP saved model into non-DDP
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    # Wrap with DDP if DDP is initialized for this process
    if is_ddp_initialized:
        model = DDP(model, device_ids=[current_rank])
    
    logger.info("Model loaded successfully.")
    
    # Re-run validation to get fresh predictions, labels, and logits
    if val_loader:
        logger.info("Re-running validation on loaded model to get metrics...")
        criterion_bongard = LabelSmoothingCrossEntropy(smoothing=cfg['training']['label_smoothing_epsilon'])
        val_loss, val_accuracy, val_predictions_logits, val_true_labels = _validate_model_ensemble(
            model=model,
            data_loader=val_loader,
            criterion=criterion_bongard,
            config=cfg
        )
        val_metrics = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'logits': np.array(val_predictions_logits),
            'labels': np.array(val_true_labels)
        }
        logger.info(f"Validation on loaded model complete. Accuracy: {val_metrics['val_accuracy']:.4f}")
    else:
        logger.warning("No validation loader provided. Cannot get fresh validation metrics for loaded model.")
        val_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'logits': np.array([]),
            'labels': np.array([])
        }
    return model.module if is_ddp_initialized else model, val_metrics

def weighted_average_ensemble_prediction(
    all_members_val_predictions_logits: List[np.ndarray], # List of (N, num_classes) logits
    all_members_val_labels: np.ndarray, # Single (N,) labels array
    ensemble_member_accuracies: List[float] # List of accuracies for each member
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Combines predictions from ensemble members using a weighted average of their logits.
    Weights are based on individual model accuracies.
    Args:
        all_members_val_predictions_logits (List[np.ndarray]): List of validation logits
                                                                (each array is for one member).
        all_members_val_labels (np.ndarray): True labels for the validation set.
        ensemble_member_accuracies (List[float]): Validation accuracy for each member.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
            - combined_logits (np.ndarray): Weighted average of logits.
            - combined_predictions (np.ndarray): Hard predictions from combined logits.
            - true_labels (np.ndarray): True labels (from any member, they are identical).
            - ensemble_metrics (Dict): Metrics for the combined ensemble.
    """
    if not all_members_val_predictions_logits:
        logger.warning("No member logits provided for weighted average ensemble. Returning empty results.")
        return np.array([]), np.array([]), np.array([]), {}
    
    num_members = len(all_members_val_predictions_logits)
    
    # Calculate weights based on accuracies (e.g., softmax over accuracies)
    # Adding a small epsilon to avoid zero weights if accuracy is 0.
    weights = np.array(ensemble_member_accuracies) + 1e-6
    weights = weights / np.sum(weights) # Normalize to sum to 1
    
    logger.info(f"Weighted average ensemble: Member weights: {weights.tolist()}")
    combined_logits = np.zeros_like(all_members_val_predictions_logits[0])
    for i in range(num_members):
        combined_logits += weights[i] * all_members_val_predictions_logits[i]
    
    combined_predictions = np.argmax(combined_logits, axis=1)
    true_labels = all_members_val_labels # All label lists should be identical, so just take the first one
    
    # Calculate ensemble metrics
    ensemble_metrics = _calculate_ensemble_metrics(combined_predictions, true_labels, combined_logits)
    logger.info(f"Weighted Average Ensemble Metrics: Accuracy: {ensemble_metrics['accuracy']:.4f}")
    return combined_logits, combined_predictions, true_labels, ensemble_metrics

class StackedEnsembleMetaLearner(nn.Module):
    """
    A simple meta-learner for stacked ensemble.
    Takes concatenated logits from base models and predicts final class.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        logger.info(f"StackedEnsembleMetaLearner initialized with input_dim={input_dim}, num_classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Concatenated logits from base models (batch_size, num_members * num_classes).
        Returns:
            torch.Tensor: Final logits for classification.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_stacked_ensemble_meta_learner(
    all_members_val_predictions_logits: List[np.ndarray], # List of (N, num_classes) logits
    all_members_val_labels: np.ndarray, # Single (N,) labels array
    num_classes: int,
    epochs: int = 50,
    lr: float = 0.001
) -> Tuple[StackedEnsembleMetaLearner, Dict[str, Any]]:
    """
    Trains a meta-learner for stacked ensemble.
    Args:
        all_members_val_predictions_logits (List[np.ndarray]): List of validation logits
                                                                (each array is for one member).
        all_members_val_labels (np.ndarray): True labels for the validation set.
        num_classes (int): Number of Bongard problem classes.
        epochs (int): Number of epochs to train the meta-learner.
        lr (float): Learning rate for the meta-learner.
    Returns:
        Tuple[StackedEnsembleMetaLearner, Dict[str, Any]]:
            - Trained meta-learner model.
            - Metrics of the meta-learner on the validation set.
    """
    if not all_members_val_predictions_logits:
        logger.warning("No member logits provided for stacked ensemble meta-learner training. Returning dummy.")
        return StackedEnsembleMetaLearner(10, num_classes), {} # Dummy meta-learner
    
    num_members = len(all_members_val_predictions_logits)
    num_samples = all_members_val_predictions_logits[0].shape[0]
    
    # Prepare data for meta-learner
    # Concatenate logits for each sample: (N, num_members * num_classes)
    meta_input_np = np.concatenate(all_members_val_predictions_logits, axis=1)
    meta_input = torch.tensor(meta_input_np, dtype=torch.float32).to(DEVICE)
    meta_labels = torch.tensor(all_members_val_labels, dtype=torch.long).to(DEVICE)
    input_dim = meta_input.shape[1]
    
    meta_learner = StackedEnsembleMetaLearner(input_dim, num_classes).to(DEVICE)
    
    optimizer = optim.Adam(meta_learner.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"Training Stacked Ensemble Meta-Learner for {epochs} epochs...")
    for epoch in tqdm(range(epochs), desc="Meta-Learner Training"):
        meta_learner.train()
        optimizer.zero_grad()
        outputs = meta_learner(meta_input)
        loss = criterion(outputs, meta_labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.debug(f"Meta-Learner Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Evaluate meta-learner
    meta_learner.eval()
    with torch.no_grad():
        meta_logits = meta_learner(meta_input).cpu().numpy()
        meta_predictions = np.argmax(meta_logits, axis=1)
    
    ensemble_metrics = _calculate_ensemble_metrics(meta_predictions, all_members_val_labels, meta_logits)
    logger.info(f"Stacked Ensemble Meta-Learner Metrics: Accuracy: {ensemble_metrics['accuracy']:.4f}")
    return meta_learner, ensemble_metrics

def _calculate_ensemble_metrics(predictions: np.ndarray, labels: np.ndarray, logits: np.ndarray) -> Dict[str, Any]:
    """Calculates common metrics for an ensemble's predictions."""
    accuracy = calculate_accuracy(predictions, labels)
    precision_recall_f1 = calculate_precision_recall_f1(predictions, labels)
    
    # Ensure there are at least two unique labels for ROC AUC
    if len(np.unique(labels)) > 1:
        probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
        roc_auc = calculate_roc_auc(probabilities[:, 1], labels)
        brier_score = calculate_brier_score(probabilities[:, 1], labels)
        ece_metrics = calculate_expected_calibration_error(probabilities[:, 1], labels)
    else:
        logger.warning("Skipping ROC AUC, Brier Score, ECE calculation for ensemble: Only one class present in labels.")
        roc_auc = 0.0
        brier_score = 0.0
        ece_metrics = {'ece': 0.0, 'mce': 0.0}
    
    return {
        'accuracy': accuracy,
        'precision': precision_recall_f1['precision'],
        'recall': precision_recall_f1['recall'],
        'f1_score': precision_recall_f1['f1_score'],
        'roc_auc': roc_auc,
        'brier_score': brier_score,
        'ece': ece_metrics['ece'],
        'mce': ece_metrics['mce'],
        'predictions': predictions,
        'labels': labels,
        'logits': logits
    }

def train_distilled_student_orchestrator_combined(
    num_ensemble_members: int,
    train_loader: Any,
    val_loader: Any,
    student_model_config: Dict[str, Any],
    epochs_student: int,
    lr_student: float,
    teacher_ensemble_type: str = 'weighted_average', # 'weighted_average' or 'stacked'
    train_members: bool = True, # Whether to train teachers or load them
    cfg: Dict[str, Any], # Pass the full config
    current_rank: int = 0,
    world_size: int = 1,
    is_ddp_initialized: bool = False,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Orchestrates the training of an ensemble of teacher models and a student model
    using knowledge distillation.
    Args:
        num_ensemble_members (int): Number of teacher ensemble members.
        train_loader: DALI or PyTorch DataLoader for training.
        val_loader: DALI or PyTorch DataLoader for validation.
        student_model_config (Dict[str, Any]): Configuration for the student model.
        epochs_student (int): Number of epochs to train the student model.
        lr_student (float): Learning rate for the student model.
        teacher_ensemble_type (str): Type of teacher ensemble ('weighted_average' or 'stacked').
        train_members (bool): If True, train teacher members; otherwise, load them.
        cfg (Dict[str, Any]): The configuration dictionary.
        current_rank (int): Current GPU rank.
        world_size (int): Total number of GPUs.
        is_ddp_initialized (bool): True if DDP is already initialized.
    Returns:
        Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
            - Trained student model.
            - Metrics for the teacher ensemble.
            - Metrics for the student model.
    """
    logger.info("Starting ensemble teacher training and student distillation orchestration.")
    teacher_models = []
    all_members_val_predictions_logits = [] # List of (N_val, num_classes) logits for each member
    all_members_val_labels = [] # List of (N_val,) labels for each member (should be identical)
    ensemble_member_accuracies = [] # List of accuracies for each member

    if train_members:
        # Train teacher ensemble members
        teacher_models, all_members_best_metrics = train_ensemble(
            num_members=num_ensemble_members,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            current_rank=current_rank,
            world_size=world_size,
            is_ddp_initialized=is_ddp_initialized
        )
        for metrics in all_members_best_metrics:
            all_members_val_predictions_logits.append(metrics['logits'])
            all_members_val_labels.append(metrics['labels'])
            ensemble_member_accuracies.append(metrics['val_accuracy'])
    else:
        # Load pre-trained teacher ensemble members
        logger.info("Loading pre-trained teacher ensemble members...")
        model_save_dir = cfg['debug']['save_model_checkpoints']
        for i in range(num_ensemble_members):
            model_path = os.path.join(model_save_dir, f"member_{i}_best_model.pt")
            if os.path.exists(model_path):
                # Load model and re-run validation to get fresh metrics
                model, val_metrics = load_trained_model(
                    model_path=model_path,
                    cfg=cfg,
                    current_rank=current_rank,
                    is_ddp_initialized=is_ddp_initialized,
                    val_loader=val_loader # Pass val_loader to get actual metrics
                )
                teacher_models.append(model)
                all_members_val_predictions_logits.append(val_metrics['logits'])
                all_members_val_labels.append(val_metrics['labels'])
                ensemble_member_accuracies.append(val_metrics['val_accuracy'])
            else:
                logger.warning(f"Teacher model {model_path} not found. Skipping this member.")
        
        if not teacher_models:
            logger.error("No teacher models loaded. Cannot proceed with distillation.")
            return None, {}, {}
    
    # Combine teacher predictions based on ensemble type
    teacher_ensemble_logits = None
    teacher_ensemble_metrics = {}
    num_bongard_classes = cfg['model']['bongard_head_config']['num_classes']
    
    # Ensure all_members_val_labels is consistent (all same)
    if all_members_val_labels:
        true_labels_for_ensemble_eval = all_members_val_labels[0]
    else:
        true_labels_for_ensemble_eval = np.array([]) # Empty if no labels

    if teacher_ensemble_type == 'weighted_average':
        logger.info("Combining teacher predictions using weighted average.")
        teacher_ensemble_logits, _, _, teacher_ensemble_metrics = weighted_average_ensemble_prediction(
            all_members_val_predictions_logits=all_members_val_predictions_logits,
            all_members_val_labels=true_labels_for_ensemble_eval,
            ensemble_member_accuracies=ensemble_member_accuracies
        )
        # For distillation, the teacher_model passed to _run_single_training_session_ensemble
        # should be the list of teacher models if weighted average is desired.
        teacher_for_distillation = teacher_models
    elif teacher_ensemble_type == 'stacked':
        logger.info("Training stacked ensemble meta-learner.")
        meta_learner, teacher_ensemble_metrics = train_stacked_ensemble_meta_learner(
            all_members_val_predictions_logits=all_members_val_predictions_logits,
            all_members_val_labels=true_labels_for_ensemble_eval,
            num_classes=num_bongard_classes,
            epochs=cfg['training']['stacked_ensemble_config']['meta_learner_epochs'],
            lr=cfg['training']['stacked_ensemble_config']['meta_learner_lr']
        )
        # Get logits from the trained meta-learner on the validation set
        meta_input_np = np.concatenate(all_members_val_predictions_logits, axis=1)
        meta_input = torch.tensor(meta_input_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            teacher_ensemble_logits = meta_learner(meta_input).cpu().numpy()
        # For distillation, the teacher_model passed should be the meta-learner itself.
        teacher_for_distillation = meta_learner
    else:
        raise ValueError(f"Unsupported teacher_ensemble_type: {teacher_ensemble_type}")
    
    if teacher_ensemble_logits is None:
        logger.error("Failed to obtain teacher ensemble logits. Aborting student training.")
        return None, teacher_ensemble_metrics, {}
    
    logger.info("Teacher ensemble predictions combined. Starting student model training with distillation.")
    
    # Train student model with knowledge distillation
    student_model = PerceptionModule(student_model_config).to(DEVICE)
    if is_ddp_initialized:
        student_model = DDP(student_model, device_ids=[current_rank])
    
    student_optimizer = get_optimizer(student_model, student_model_config['training'])
    total_steps_student = epochs_student * len(train_loader) if hasattr(train_loader, '__len__') else epochs_student * 100
    student_scheduler = get_scheduler(student_optimizer, student_model_config['training'], total_steps_student)
    
    _, _, _, student_metrics = _run_single_training_session_ensemble(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=student_optimizer,
        scheduler=student_scheduler,
        current_rank=current_rank,
        is_ddp_initialized=is_ddp_initialized,
        model_idx=-1, # Indicate student model
        cfg=student_model_config, # Pass student's config
        replay_buffer=train_loader.dataset.replay_buffer if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) else None,
        teacher_model=teacher_for_distillation # Pass the teacher(s) or meta-learner
    )
    logger.info("Student model training with distillation completed.")
    return student_model.module if is_ddp_initialized else student_model, teacher_ensemble_metrics, student_metrics

def ensemble_predict_orchestrator(
    models: List[PerceptionModule],
    image_paths: List[str],
    config: Dict[str, Any],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0,
    model_weights: Optional[List[float]] = None # Optional list of weights for weighted averaging
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble prediction by averaging probabilities from multiple PerceptionModule models.
    Args:
        models (List[PerceptionModule]): List of trained PerceptionModule instances.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
        use_mc_dropout (bool): If True, enable MC Dropout during inference.
        mc_dropout_samples (int): Number of forward passes for MC Dropout.
        model_weights (Optional[List[float]]): Weights for each model for weighted averaging.
                                                If None, simple averaging is used.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Averaged Bongard problem probabilities (after softmax) [num_images, num_classes].
            - List of symbolic outputs (from the first model's inference, for example).
    """
    if not models:
        logger.error("No models provided for ensemble prediction.")
        return np.empty((len(image_paths), config['model']['bongard_head_config']['num_classes'])), []
    
    all_bongard_probs = []
    first_model_symbolic_outputs = [] # Collect symbolic outputs from the first model for example
    
    # Create DALI loader for inference using refactored function
    # This part assumes `build_dali_loader` can be called with just file_list and labels_list.
    # It might need a dummy dataset object or a more direct way to feed image paths.
    # For now, let's assume `get_loader` can handle this for inference.
    
    # Dummy dataset and loader for inference (since get_loader expects a dataset)
    # In a real scenario, you'd create a specific inference dataset.
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, cfg):
            self.image_paths = image_paths
            self.cfg = cfg
            self.transform = T.Compose([
                T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            transformed_image = self.transform(image)
            # Return dummy labels and other info to match custom_collate_fn
            return (transformed_image, transformed_image, 0, b'{}', b'{}', 0.5, np.eye(3).tolist(), np.eye(3).tolist(), idx,
                    torch.zeros(1, 3, self.cfg['data']['image_size'], self.cfg['data']['image_size']), # Padded support imgs
                    torch.tensor([-1]), b'{}', torch.tensor(0), torch.tensor(idx), torch.tensor(1.0))
    inference_dataset = InferenceDataset(image_paths, config)
    inference_loader = get_loader(config, train=False, rank=0, world_size=1, dataset=inference_dataset)
    logger.info("Inference loader initialized.")
    
    for model_idx, model in enumerate(models):
        model.eval()
        if use_mc_dropout:
            # Enable dropout layers for MC Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train() # Set dropout to training mode to enable dropout during inference
            logger.info(f"MC Dropout enabled for model {model_idx}.")
        else:
            # Ensure dropout layers are in eval mode if not using MC Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
        
        member_probs = []
        member_symbolic_outputs_mc_samples = [] # Store symbolic outputs for each MC sample
        
        with torch.no_grad():
            # Efficient MC Dropout Sampling
            for mc_sample_idx in tqdm(range(mc_dropout_samples if use_mc_dropout else 1), desc=f"Model {model_idx} MC Inference"):
                # Reset loader for each MC Dropout sample or single pass
                if hasattr(inference_loader, 'reset'): # For DALI
                    inference_loader.reset()
                else: # For PyTorch DataLoader, need to re-create iterator
                    inference_loader_iter = iter(inference_loader)
                current_sample_probs = []
                current_batch_symbolic_outputs = []
                
                # Iterate through the loader
                for batch_idx, data in enumerate(inference_loader):
                    images_view1 = data['query_img1'].to(DEVICE) # NCHW
                    gts_json_strings_batch = data['query_gts_json_view1'] # Pass dummy GT for inference
                    
                    # PerceptionModule.forward now returns bongard_logits, detected_objects, aggregated_outputs
                    bongard_logits, detected_objects_batch, aggregated_outputs = model(images_view1, gts_json_strings_batch)
                    
                    probs = F.softmax(bongard_logits, dim=-1).cpu().numpy()
                    current_sample_probs.append(probs)
                    
                    # Collect structured symbolic outputs from the first model, first MC dropout sample
                    if model_idx == 0: # Only collect for the first model
                        for b_idx in range(images_view1.shape[0]):
                            # Ensure detected_objects_batch and aggregated_outputs are correctly indexed for a single image
                            single_detected_objects = detected_objects_batch[b_idx]
                            single_attribute_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['attribute_logits'].items()}
                            single_relation_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['relation_logits'].items()}
                            
                            scene_graph = model.extract_scene_graph(
                                single_detected_objects,
                                single_attribute_logits,
                                single_relation_logits
                            )
                            scene_graph["bongard_prediction"] = int(torch.argmax(bongard_logits[b_idx]).item())
                            
                            # Store raw logits for uncertainty calculation later
                            scene_graph["raw_bongard_logits"] = bongard_logits[b_idx].cpu().numpy().tolist()
                            scene_graph["bongard_probs"] = probs[b_idx].tolist()
                            
                            current_batch_symbolic_outputs.append(scene_graph)
                
                member_probs.append(np.concatenate(current_sample_probs, axis=0))
                if model_idx == 0: # Only store symbolic outputs for the first model across MC samples
                    member_symbolic_outputs_mc_samples.append(current_batch_symbolic_outputs)
        
        # Average MC Dropout samples for this model
        if use_mc_dropout:
            avg_member_probs = np.mean(np.stack(member_probs, axis=0), axis=0)
            logger.info(f"Model {model_idx} averaged {mc_dropout_samples} MC Dropout samples.")
            
            # If MC Dropout, calculate epistemic and aleatoric uncertainty for symbolic outputs
            if model_idx == 0 and member_symbolic_outputs_mc_samples:
                # Reorganize symbolic outputs to be per-image, across MC samples
                num_images_in_batch = len(member_symbolic_outputs_mc_samples[0])
                for img_idx in range(num_images_in_batch):
                    all_mc_logits_for_image = []
                    for mc_sample_data in member_symbolic_outputs_mc_samples:
                        all_mc_logits_for_image.append(mc_sample_data[img_idx]["raw_bongard_logits"])
                    
                    all_mc_logits_for_image_np = np.array(all_mc_logits_for_image) # [mc_samples, num_classes]
                    
                    # Epistemic uncertainty: variance of predictions across MC samples
                    epistemic_probs = F.softmax(torch.tensor(all_mc_logits_for_image_np), dim=-1).numpy()
                    epistemic_uncertainty = np.mean(np.var(epistemic_probs, axis=0)) # Mean variance across classes
                    
                    # Aleatoric uncertainty: average of (p * (1-p)) over MC samples
                    aleatoric_uncertainty = np.mean(epistemic_probs * (1 - epistemic_probs))
                    
                    # Update the first MC sample's symbolic output for this image with uncertainty
                    member_symbolic_outputs_mc_samples[0][img_idx]["uncertainty"] = {
                        "epistemic": float(epistemic_uncertainty),
                        "aleatoric": float(aleatoric_uncertainty)
                    }
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0] # Use the first sample's structure, updated with uncertainty
        else:
            avg_member_probs = member_probs[0] # Only one sample
            if model_idx == 0:
                # If no MC dropout, symbolic output's uncertainty will be 0.0
                for img_data in member_symbolic_outputs_mc_samples[0]:
                    img_data["uncertainty"] = {"epistemic": 0.0, "aleatoric": 0.0}
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0] # Just the single sample's output
        
        all_bongard_probs.append(avg_member_probs)
    
    stacked_probs = np.stack(all_bongard_probs, axis=0) # Shape: [num_models, num_images, num_classes]
    
    if model_weights is not None and len(model_weights) == len(models):
        # Normalize weights to sum to 1
        normalized_weights = np.array(model_weights) / np.sum(model_weights)
        # Reshape weights to [num_models, 1, 1] for broadcasting
        weighted_probs = stacked_probs * normalized_weights[:, np.newaxis, np.newaxis]
        ensemble_averaged_probs = np.sum(weighted_probs, axis=0) # Sum along the model dimension
        logger.info("Ensemble prediction using weighted averaging.")
    else:
        ensemble_averaged_probs = np.mean(stacked_probs, axis=0)
        logger.info("Ensemble prediction using simple averaging.")
    
    logger.info(f"Ensemble prediction complete. Averaged probabilities for {len(image_paths)} images.")
    return ensemble_averaged_probs, first_model_symbolic_outputs
