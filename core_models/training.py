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

# Import for pruning (dummy functions, actual implementations in prune_quantize.py)
import torch.nn.utils.prune as prune

# Import configuration (from parent directory)
from ..config import (
    CONFIG, DEVICE, HAS_WANDB, HAS_TORCH_QUANTIZATION,
    ATTRIBUTE_FILL_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_SIZE_MAP,
    ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_TEXTURE_MAP,
    RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD
)

# Import PyTorch Lightning modules
import pytorch_lightning as pl

# Import models (from current directory)
from .models import LitBongard, LitSimCLR, PerceptionModule

# Import data module and loader (from parent directory's src folder)
from ..data import BongardDataModule, get_loader

# Import losses (from current directory)
from .losses import LabelSmoothingCrossEntropy, DistillationLoss, FeatureConsistencyLoss, SymbolicConsistencyLoss, GradNorm

# Import optimizers and schedulers (from current directory)
from .optimizers import get_optimizer, get_scheduler, HAS_SAM_OPTIMIZER, Lion, MADGRAD # Assuming SAM, Lion, MADGRAD are exposed by optimizers.py
try:
    from ranger_adabelief import RangerAdaBelief
    HAS_RANGER = True
except ImportError:
    HAS_RANGER = False
    logger = logging.getLogger(__name__)
    logger.warning("RangerAdaBelief not found. Ranger optimizer will be disabled.")


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

# Import GradualWarmupScheduler and CosineAnnealingWarmRestarts (for 7.1)
# These are now handled by optimizers.py's get_scheduler, so direct import here is not strictly needed.
# Keeping the HAS_GRADUAL_WARMUP flag for conditional logic if needed.
try:
    from warmup_scheduler import GradualWarmupScheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    HAS_GRADUAL_WARMUP = True
except ImportError:
    HAS_GRADUAL_WARMUP = False


logger = logging.getLogger(__name__)

# --- Helper functions ---
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed) # Use PyTorch Lightning's seed_everything
    logger.info(f"Random seed set to {seed}")

# Import async_update_priorities from replay_buffer.py (now in core_models)
from .replay_buffer import KnowledgeReplayBuffer # Import the class directly
# Dummy function for asynchronous priority update for replay buffer
def async_update_priorities(replay_buffer: KnowledgeReplayBuffer, tree_indices: List[int], losses: List[float], cfg: Dict[str, Any]):
    """
    Dummy function for asynchronous priority update.
    This function simulates updating replay buffer priorities.
    In a real scenario, this might involve a separate thread or process.
    """
    # For now, directly call update for demonstration
    # The replay_buffer.update method now takes cfg directly for annealing.
    replay_buffer.update(losses, cfg) 
    logger.debug(f"Async update priorities called for {len(tree_indices)} samples.")


# Dummy QAT/PTQ functions (actual implementations are in prune_quantize.py, which is in the project root)
# These are kept here as placeholders if run_training_once needs to call them directly.
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
            return self.aug(images, labels)
        return images, F.one_hot(labels, num_classes=self.num_classes).float()

# Ensemble Teacher Logits (from previous context)
def _get_ensemble_teacher_logits(
    teacher_models: nn.ModuleList,
    raw_images_np: List[np.ndarray],
    raw_gt_json_strings: List[bytes],
    raw_support_images_np: List[np.ndarray],
    distillation_config: Dict[str, Any],
    dali_image_processor: Any = None,  # DALI processor from datamodule
    # New: Pass detected bboxes and masks for teacher's PerceptionModule
    detected_bboxes_batch: Optional[List[List[List[float]]]] = None,
    detected_masks_batch: Optional[List[List[np.ndarray]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Helper function to get ensemble teacher logits for distillation.
    """
    if not teacher_models:
        logger.warning("No teacher models provided for distillation. Returning empty logits.")
        return torch.empty(0), None
    
    all_teacher_logits = []
    all_distillation_masks = []
    
    for teacher_model in teacher_models:
        teacher_model.eval()
        with torch.no_grad():
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
                processed_images, _, processed_support_images_flat = dali_image_processor.run(
                    raw_images_np,
                    [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_images_np),
                    raw_support_images_np
                )
            
            max_support_imgs = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
            batch_size_actual = processed_images.shape[0]
            processed_support_images_reshaped = processed_support_images_flat.view(
                batch_size_actual, max_support_imgs, 
                processed_images.shape[1], processed_images.shape[2], processed_images.shape[3]
            )
            
            # Teacher forward pass with new bbox/mask arguments
            teacher_outputs = teacher_model.perception_module(
                processed_images,
                ground_truth_json_strings=raw_gt_json_strings,
                detected_bboxes_batch=detected_bboxes_batch, # Pass bboxes
                detected_masks_batch=detected_masks_batch,   # Pass masks
                support_images=processed_support_images_reshaped,
                support_labels_flat=torch.zeros_like(processed_support_images_reshaped[:,:,0,0,0], dtype=torch.long)
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
        ensemble_logits = torch.stack(all_teacher_logits, dim=0).mean(dim=0)
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
    set_seed(cfg['training']['seed'])
    
    model = LitBongard(cfg).to(DEVICE)
    
    data_module = BongardDataModule(cfg)
    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Initialize optimizer and scheduler using centralized functions
    optimizer = get_optimizer(model, cfg['training'])
    
    # total_steps is needed for OneCycleLR
    total_steps = (epochs if epochs is not None else cfg['training']['epochs']) * len(train_loader)
    scheduler = get_scheduler(optimizer, cfg['training'], total_steps)
    
    scaler = GradScaler() if cfg['training']['use_amp'] else None
    
    early_stop_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_model_early_stop.pt")
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stop_patience'],
        delta=cfg['training']['early_stop_delta'],
        verbose=True,
        path=early_stop_path
    )
    logger.info(f"Early Stopping initialized with patience={early_stopping.patience}, delta={early_stopping.delta}.")
    
    num_bongard_classes = cfg['model']['bongard_head_config']['num_classes']
    mixup_cutmix_augmenter = MixupCutmixAugmenter(cfg['training'], num_bongard_classes)
    
    ema_model = None
    if cfg['training'].get('use_mean_teacher', False):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        logger.info("Mean Teacher (EMA) model initialized.")
    
    grad_norm = None
    if cfg['training'].get('use_grad_norm', False): # HAS_GRAD_NORM is checked inside GradNorm class now
        # GradNorm is initialized with initial loss weights and alpha
        # Assuming the main losses are 'bongard_loss', 'attribute_loss', 'relation_loss'
        initial_task_weights = {
            'bongard_loss': 1.0,
            'attribute_loss': cfg['training'].get('attribute_loss_weight', 1.0),
            'relation_loss': cfg['training'].get('relation_loss_weight', 1.0),
            'consistency_loss': cfg['training'].get('consistency_loss_weight', 1.0),
            'distillation_loss': cfg['training'].get('distillation_config', {}).get('loss_weight', 1.0) # Assuming a weight for distillation
        }
        # Filter out tasks with 0 weight if they shouldn't be considered by GradNorm
        initial_task_weights = {k: v for k, v in initial_task_weights.items() if v > 0}

        if initial_task_weights:
            grad_norm = GradNorm(initial_task_weights, cfg['training'].get('grad_norm_alpha', 1.5))
            grad_norm.to(DEVICE)
            logger.info("GradNorm initialized.")
        else:
            logger.warning("No active loss components for GradNorm. GradNorm will not be used.")
            grad_norm = None

    best_val_accuracy = 0.0
    num_epochs = epochs if epochs is not None else cfg['training']['epochs']
    
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg['debug']['logs_dir']), # Use cfg['debug']['logs_dir']
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    prof.start()

    for epoch in range(num_epochs):
        if trial:
            trial.report(best_val_accuracy, step=epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}.")
                # optuna.TrialPruned is not imported, assuming it's handled by HPO script
                raise Exception("TrialPruned") # Replace with optuna.TrialPruned if Optuna is imported here
        
        model.train()
        total_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_loop):
            # Unpack batch data
            (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
             query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
             raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
             tree_indices, is_weights,
             query_bboxes_view1, query_masks_view1, # New: detected bboxes and masks for query images
             query_bboxes_view2, query_masks_view2,
             support_bboxes_flat, support_masks_flat # New: detected bboxes and masks for support images
            ) = batch
            
            query_labels = query_labels.to(DEVICE).long()
            support_labels_flat = support_labels_flat.to(DEVICE).long()
            
            # DALI Image Processor or torchvision transforms
            dali_processor = getattr(data_module, 'dali_image_processor', None) # Access from DataModule
            
            if dali_processor is None or not cfg['training'].get('use_dali', False):
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
            
            with record_function("model_inference"):
                if isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER:
                    # SAM first step
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1,
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits = outputs['bongard_logits']
                        
                        # Use the same loss calculation as LitBongard for consistency
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, labels_mixed, reduction='none')
                        loss_bongard = (per_sample_bongard_losses * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else per_sample_bongard_losses.mean()
                        
                        # For SAM's first step, we just need a scalar loss to compute gradients
                        loss_for_sam_first_step = loss_bongard # Simplification for SAM's first step
                        
                    if cfg['training']['use_amp']:
                        scaler.scale(loss_for_sam_first_step).backward()
                        optimizer.first_step(zero_grad=True)
                    else:
                        loss_for_sam_first_step.backward()
                        optimizer.first_step(zero_grad=True)
                    
                    # SAM second step
                    with autocast(enabled=cfg['training']['use_amp']):
                        outputs2 = model(images_view1_aug, ground_truth_json_strings=query_gts_json_view1, # Use view1_aug for second pass too
                                        detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1,
                                        support_images=processed_support_images_reshaped,
                                        support_labels_flat=support_labels_reshaped)
                        bongard_logits2 = outputs2['bongard_logits']
                        attribute_logits2 = outputs2['attribute_logits']
                        relation_logits2 = outputs2['relation_logits']
                        attribute_features2 = outputs2['attribute_features']
                        scene_graphs2 = outputs2['scene_graphs'] # For consistency loss
                        
                        # Calculate full loss for the second step, similar to LitBongard
                        total_batch_loss = model.bongard_criterion(bongard_logits2, labels_mixed, reduction='none')
                        total_batch_loss = (total_batch_loss * is_weights.to(DEVICE)).mean() if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and is_weights is not None else total_batch_loss.mean()
                        
                        # Attribute Loss
                        loss_attribute = torch.tensor(0.0, device=DEVICE)
                        num_attribute_losses = 0
                        current_flat_idx = 0
                        for i_img in range(len(scene_graphs2)):
                            sg_gt = json.loads(query_gts_json_view1[i_img].decode('utf-8'))
                            inferred_objects_for_img = scene_graphs2[i_img].get('objects', [])
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
                                        if attr_name in gt_obj['attributes'] and attr_name in attribute_logits2 and attribute_logits2[attr_name].numel() > 0:
                                            if current_flat_idx < attribute_logits2[attr_name].shape[0]:
                                                attr_map = globals().get(f"ATTRIBUTE_{attr_name.upper()}_MAP")
                                                if attr_map and gt_obj['attributes'][attr_name] in attr_map:
                                                    gt_attr_label = attr_map[gt_o['attributes'][attr_name]]
                                                    predicted_logits = attribute_logits2[attr_name][current_flat_idx].unsqueeze(0)
                                                    loss_attribute += model.attribute_criterion(predicted_logits, torch.tensor([gt_attr_label], device=DEVICE))
                                                    num_attribute_losses += 1
                                current_flat_idx += 1
                        if num_attribute_losses > 0:
                            loss_attribute /= num_attribute_losses
                            total_batch_loss += loss_attribute * cfg['training'].get('attribute_loss_weight', 1.0)
                        
                        # Relation Loss (simplified, assuming SceneGNN or RelationGNN)
                        loss_relation = torch.tensor(0.0, device=DEVICE)
                        if not model.perception_module.use_scene_gnn and relation_logits2.numel() > 0:
                            all_gt_edge_labels_flat = []
                            for b in range(batch_size_actual):
                                sg_gt = json.loads(query_gts_json_view1[b].decode('utf-8'))
                                num_gt_objects = len(sg_gt['objects'])
                                if num_gt_objects > 1:
                                    edge_map_for_img = make_edge_index_map(num_gt_objects)
                                    temp_gt_labels = torch.full((len(edge_map_for_img),), fill_value=RELATION_MAP['none'], dtype=torch.long, device=DEVICE)
                                    for rel in sg_gt['relations']:
                                        subj_id = rel['subject_id']
                                        obj_id = rel['object_id']
                                        rel_type = rel['type']
                                        if (subj_id, obj_id) in edge_map_for_img and rel_type in RELATION_MAP:
                                            edge_idx_flat = edge_map_for_img[(subj_id, obj_id)]
                                            temp_gt_labels[edge_idx_flat] = RELATION_MAP[rel_type]
                                    all_gt_edge_labels_flat.append(temp_gt_labels)
                            if all_gt_edge_labels_flat:
                                labels_flat = torch.cat(all_gt_edge_labels_flat, dim=0)
                                loss_relation = model.relation_criterion(relation_logits2, labels_flat)
                                total_batch_loss += loss_relation * cfg['training'].get('relation_loss_weight', 1.0)
                        
                        # Consistency Loss
                        loss_consistency = torch.tensor(0.0, device=DEVICE)
                        if cfg['training']['consistency_loss_weight'] > 0:
                            if cfg['training']['feature_consistency_weight'] > 0 and outputs['attribute_features'].numel() > 0 and attribute_features2.numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(outputs['attribute_features'], attribute_features2)
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=outputs['scene_graphs'],
                                    scene_graphs2=scene_graphs2,
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency

                        # Distillation Loss
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        if model.distillation_criterion and cfg['training']['use_knowledge_distillation'] and model.teacher_models:
                            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                                teacher_models=model.teacher_models,
                                raw_images_np=raw_query_images_view1_np,
                                raw_gt_json_strings=query_gts_json_view1,
                                raw_support_images_np=raw_support_images_flat_np,
                                distillation_config=cfg['training']['distillation_config'],
                                dali_image_processor=dali_processor,
                                detected_bboxes_batch=query_bboxes_view1, # Pass for teacher
                                detected_masks_batch=query_masks_view1
                            )
                            if teacher_logits_batch.numel() > 0:
                                per_sample_soft_loss, per_sample_hard_loss = model.distillation_criterion(
                                    bongard_logits2, teacher_logits_batch, query_labels
                                )
                                if distillation_mask is not None and cfg['training']['distillation_config']['use_mask_distillation']:
                                    masked_soft_loss = per_sample_soft_loss * distillation_mask.to(DEVICE)
                                    masked_hard_loss = per_sample_hard_loss * distillation_mask.to(DEVICE)
                                    loss_distillation = (cfg['training']['distillation_config']['alpha'] * masked_soft_loss + \
                                                         (1. - cfg['training']['distillation_config']['alpha']) * masked_hard_loss).mean()
                                else:
                                    loss_distillation = (cfg['training']['distillation_config']['alpha'] * per_sample_soft_loss + \
                                                         (1. - cfg['training']['distillation_config']['alpha']) * per_sample_hard_loss).mean()
                                total_batch_loss += loss_distillation
                        
                        current_loss = total_batch_loss.item() # Store final loss for logging
                        
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
                        optimizer.second_step(zero_grad=True) if isinstance(optimizer, SAM) and HAS_SAM_OPTIMIZER else optimizer.step() # zero_grad for SAM
                    
                else: # Standard optimizer path
                    optimizer.zero_grad()
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

                        # Calculate full loss (similar to LitBongard's training_step)
                        per_sample_bongard_losses = model.bongard_criterion(bongard_logits, labels_mixed, reduction='none')
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
                                        subj_id = rel['subject_id']
                                        obj_id = rel['object_id']
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
                            outputs_view2 = model(processed_query_images_view2, ground_truth_json_strings=query_gts_json_view2,
                                                detected_bboxes_batch=query_bboxes_view2, detected_masks_batch=query_masks_view2,
                                                support_images=processed_support_images_reshaped,
                                                support_labels_flat=support_labels_reshaped)
                            if cfg['training']['feature_consistency_weight'] > 0 and outputs['attribute_features'].numel() > 0 and outputs_view2['attribute_features'].numel() > 0:
                                loss_feature_consistency = model.feature_consistency_criterion(outputs['attribute_features'], outputs_view2['attribute_features'])
                                loss_consistency += cfg['training']['feature_consistency_weight'] * loss_feature_consistency
                            if cfg['training']['symbolic_consistency_weight'] > 0 and model.HAS_SYMBOLIC_CONSISTENCY and model.symbolic_consistency_criterion:
                                loss_symbolic_consistency = model.symbolic_consistency_criterion(
                                    scene_graphs1=outputs['scene_graphs'],
                                    scene_graphs2=outputs_view2['scene_graphs'],
                                    labels=query_labels,
                                    ground_truth_scene_graphs=query_gts_json_view1 # Use view1 GT for symbolic consistency
                                )
                                loss_consistency += cfg['training']['symbolic_consistency_weight'] * loss_symbolic_consistency
                        total_batch_loss += cfg['training']['consistency_loss_weight'] * loss_consistency

                        # Distillation Loss
                        loss_distillation = torch.tensor(0.0, device=DEVICE)
                        if model.distillation_criterion and cfg['training']['use_knowledge_distillation'] and model.teacher_models:
                            teacher_logits_batch, distillation_mask = _get_ensemble_teacher_logits(
                                teacher_models=model.teacher_models,
                                raw_images_np=raw_query_images_view1_np,
                                raw_gt_json_strings=query_gts_json_view1,
                                raw_support_images_np=raw_support_images_flat_np,
                                distillation_config=cfg['training']['distillation_config'],
                                dali_image_processor=dali_processor,
                                detected_bboxes_batch=query_bboxes_view1, # Pass for teacher
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
                                total_batch_loss += loss_distillation

                        current_loss = total_batch_loss.item() # Store final loss for logging
                    
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
            
            if scheduler is not None and (isinstance(scheduler, OneCycleLR) or (isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
                scheduler.step()
            
            if ema_model and cfg['training'].get('use_mean_teacher', False):
                ema_decay = cfg['training']['mean_teacher_config'].get('alpha', 0.99)
                for student_param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(student_param.data * (1 - ema_decay))
            
            if grad_norm: # GradNorm is applied after optimizer step for its own backward pass
                # This needs to be carefully integrated. If GradNorm is part of LitBongard,
                # it's handled in LitBongard's training_step.
                # For this manual loop, we need to manually call its update_weights.
                # This requires breaking down the total_batch_loss into its components.
                # For simplicity, if GradNorm is enabled, we assume LitBongard handles it.
                # If not using LitBongard, you'd need to compute individual losses here and pass them.
                pass # GradNorm logic is more complex, handled within LitBongard if integrated there.
            
            train_loop.set_postfix(loss=current_loss)
            
            predictions = torch.argmax(bongard_logits, dim=1)
            train_correct_predictions += (predictions == query_labels).sum().item()
            train_total_samples += query_labels.size(0)
            
            if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and hasattr(data_module.train_dataset, 'replay_buffer'):
                per_sample_losses = model.bongard_criterion(bongard_logits, query_labels, reduction='none')
                # Pass cfg to replay_buffer.update for annealing
                data_module.train_dataset.replay_buffer.update(per_sample_losses.cpu().tolist(), cfg)
            
            prof.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        
        if grad_norm and hasattr(model, 'logger') and hasattr(model.logger, 'experiment'):
             # This part is for PyTorch Lightning's logger. For manual loop, you'd need a separate TensorBoard writer.
             pass # GradNorm logging is typically handled by LitBongard's logger

        model.eval()
        total_val_loss = 0
        val_correct_predictions = 0
        val_total_samples = 0
        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(val_loop):
                (raw_query_images_view1_np, _, query_labels_val,
                 query_gts_json_view1_val, _, _, _, _, _,
                 raw_support_images_flat_np_val, support_labels_flat_val, _, _, _, _,
                 query_bboxes_view1_val, query_masks_view1_val, # New: detected bboxes and masks for query images
                 _, _,
                 support_bboxes_flat_val, support_masks_flat_val # New: detected bboxes and masks for support images
                ) = batch_val
                
                query_labels_val = query_labels_val.to(DEVICE).long()
                support_labels_flat_val = support_labels_flat_val.to(DEVICE).long()
                
                if dali_processor is None or not cfg['training'].get('use_dali', False):
                    transform = T.Compose([
                        T.ToPILImage(),
                        T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    ])
                    processed_query_images_view1_val = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                    processed_support_images_flat_val = torch.stack([transform(img_np) for img_np in raw_support_images_flat_np_val]).to(DEVICE)
                else:
                    processed_query_images_view1_val, _, processed_support_images_flat_val = dali_processor.run(
                        raw_query_images_view1_np,
                        [np.zeros((1,1,3), dtype=np.uint8)] * len(raw_query_images_view1_np), # Dummy view2
                        raw_support_images_flat_np_val
                    )
                
                max_support_imgs_val = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
                batch_size_actual_val = processed_query_images_view1_val.shape[0]
                processed_support_images_reshaped_val = processed_support_images_flat_val.view(
                    batch_size_actual_val, max_support_imgs_val, 
                    processed_query_images_view1_val.shape[1], processed_query_images_view1_val.shape[2], processed_query_images_view1_val.shape[3]
                )
                support_labels_reshaped_val = support_labels_flat_val.view(batch_size_actual_val, max_support_imgs_val)
                
                outputs_val = model(processed_query_images_view1_val, ground_truth_json_strings=query_gts_json_view1_val,
                                         detected_bboxes_batch=query_bboxes_view1_val, detected_masks_batch=query_masks_view1_val,
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
        
        if scheduler is not None and not (isinstance(scheduler, OneCycleLR) or (isinstance(scheduler, GradualWarmupScheduler) and isinstance(scheduler.after_scheduler, CosineAnnealingWarmRestarts))):
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
            model_save_path = os.path.join(cfg['debug']['save_model_checkpoints'], "best_bongard_model.pt")
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best model saved to {model_save_path} with accuracy: {best_val_accuracy:.4f}")
    
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
    
    if cfg['training']['quantization']['qat']:
        model = quantize_model_qat(model, cfg)
        optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "qat_optimized_bongard_model.pth")
        torch.save(model.state_dict(), optimized_model_path)
        logger.info(f"QAT optimized model saved to: {optimized_model_path}")
    if cfg['training']['quantization']['ptq']:
        model = quantize_model_ptq(model, val_loader, cfg)
        optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "ptq_optimized_bongard_model.pth")
        torch.save(model.state_dict(), optimized_model_path)
        logger.info(f"PTQ optimized model saved to: {optimized_model_path}")
    
    logger.info("--- Training Pipeline finished. ---")
    return best_val_accuracy

# Dummy functions for pruning (actual implementations are in prune_quantize.py)
def compute_layer_sensitivity(model: nn.Module, val_loader: DataLoader, dali_image_processor: Any, current_rank: int) -> Dict[str, float]:
    logger.info("Performing dummy layer sensitivity computation.")
    sensitivities = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d)):
            sensitivities[name] = random.uniform(0.1, 1.0)
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
    groups: int = 1
) -> nn.Module:
    logger.info(f"Performing dummy structured pruning with groups={groups}.")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if random.random() < cfg['training']['pruning'].get('pruning_target_layers_ratio', 0.5):
                prune.random_unstructured(module, name="weight", amount=cfg['training']['pruning'].get('amount', 0.2))
                logger.info(f"Dummy pruning applied to {name} with amount {cfg['training']['pruning'].get('amount', 0.2)}")
    return model

# Main execution block (for direct script execution)
if __name__ == "__main__":
    # Setup basic logging for the script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the default config from config.py
    cfg = CONFIG
    
    # Example: Override some config values for a quick test run
    cfg['training']['epochs'] = 2
    cfg['debug']['save_model_checkpoints'] = './temp_checkpoints'
    cfg['debug']['logs_dir'] = './temp_logs' # Ensure logs_dir is set for profiler
    os.makedirs(cfg['debug']['save_model_checkpoints'], exist_ok=True)
    os.makedirs(cfg['debug']['logs_dir'], exist_ok=True) # Create logs directory
    
    final_accuracy = run_training_once(cfg, trial=None) 
    print(f"Training finished. Final validation accuracy: {final_accuracy:.4f}")
