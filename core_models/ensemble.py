# Folder: bongard_solver/core_models/
# File: ensemble.py
import torch
import torch.nn as nn
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
from pathlib import Path # For save_diversity

# Import from config (assuming config.py is in the project root)
from ..config import CONFIG, DEVICE, HAS_WANDB, IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS, DATA_ROOT_PATH

# Import from utils (assuming utils.py is in src/)
from ..src.utils import setup_logging, set_seed, get_symbolic_embedding_dims # Assuming these are in src/utils.py

# Import from data (assuming data.py is in src/)
from ..data import build_dali_loader, BongardSyntheticDataset, CurriculumSampler, load_bongard_data, RealBongardDataset, BongardGenerator, get_loader

# Import from bongard_rules (assuming bongard_rules.py is in src/)
from ..src.bongard_rules import ALL_BONGARD_RULES # Needed for BongardGenerator

# Import from models (from current directory)
from .models import PerceptionModule

# Import from training (from current directory)
from .training import _run_single_training_session_ensemble, _validate_model_ensemble, _get_ensemble_teacher_logits # Also import _get_ensemble_teacher_logits

# Import from optimizers (from current directory)
from .optimizers import get_optimizer, get_scheduler

# Import from losses (from current directory)
from .losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss

# Import from metrics (from current directory)
from .metrics import calculate_accuracy, calculate_precision_recall_f1, calculate_roc_auc, calculate_brier_score, calculate_expected_calibration_error, plot_cm, log_confusion

# SWA imports
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn # Explicitly import

# For torchvision.transforms.v2 for MixUp/CutMix (if needed here, but usually handled in training.py)
# try:
#     import torchvision.transforms.v2 as T_v2
#     HAS_TORCHVISION_V2 = True
# except ImportError:
#     HAS_TORCHVISION_V2 = False

logger = logging.getLogger(__name__)

# Define a basic JSON schema for ground truth validation
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

# Helper function for CutMix (assuming it's used elsewhere, otherwise remove)
# If MixupCutmixAugmenter from training.py is used, this is not needed here.
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def compute_diversity(
    members: List[nn.Module],
    data_loader: Any, # Now takes a DataLoader
    cfg: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Computes diversity metrics (ensemble entropy and disagreement) for a list of models.
    Args:
        members (List[nn.Module]): A list of trained ensemble member models.
        data_loader (Any): DataLoader for inference (e.g., validation set).
        cfg (Dict[str, Any]): Configuration dictionary.
    Returns:
        Tuple[float, float]:
            - entropy (float): Measures the average uncertainty of the ensemble's mean prediction.
            - disagreement (float): Measures the variance of individual member predictions.
    """
    if not members:
        logger.warning("No ensemble members provided for diversity computation.")
        return 0.0, 0.0
    
    for m in members:
        m.eval()
    
    all_member_probs_per_batch = [] # List of lists, each inner list is (num_images_in_batch, num_classes)
    
    # Reset loader for consistent evaluation
    if hasattr(data_loader, 'reset'):
        data_loader.reset()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images_view1 = data['query_img1'].to(DEVICE) if isinstance(data, dict) else data[0].to(DEVICE)
            # Dummy bboxes and masks for inference
            dummy_bboxes = [[] for _ in range(images_view1.shape[0])]
            dummy_masks = [[] for _ in range(images_view1.shape[0])]
            dummy_gts = [b'{}'] * images_view1.shape[0] # Dummy GT JSONs
            
            batch_member_probs = []
            for m in members:
                # PerceptionModule.forward expects detected_bboxes_batch and detected_masks_batch
                outputs = m(images_view1, ground_truth_json_strings=dummy_gts,
                            detected_bboxes_batch=dummy_bboxes, detected_masks_batch=dummy_masks)
                bongard_logits = outputs['bongard_logits']
                batch_member_probs.append(F.softmax(bongard_logits, dim=-1).cpu().numpy())
            
            all_member_probs_per_batch.append(np.stack(batch_member_probs, axis=0)) # (num_members, batch_size, num_classes)
            
    if not all_member_probs_per_batch:
        logger.warning("No data processed for diversity calculation. Returning 0.0, 0.0.")
        return 0.0, 0.0

    # Concatenate all batches
    stacked_probs = np.concatenate(all_member_probs_per_batch, axis=1) # (num_members, total_images, num_classes)
    
    # Convert to torch tensor for easier calculation
    stacked_probs_torch = torch.tensor(stacked_probs, device=DEVICE) # Shape: [M, N, C]
    
    # Ensemble Entropy (measures uncertainty of the average prediction)
    mean_p = stacked_probs_torch.mean(0) # Average probabilities across members: [N, C]
    entropy = -(mean_p * torch.log(mean_p + 1e-9)).sum(-1).mean().item() # Add epsilon for log(0)
    
    # Disagreement (measures variance among member predictions)
    # Calculate variance of probabilities across members for each class and batch item
    disagreement = torch.var(stacked_probs_torch, dim=0).mean().item() # Mean variance across batch and classes
    
    logger.info(f"Diversity Metrics: Entropy={entropy:.4f}, Disagreement={disagreement:.4f}")
    return entropy, disagreement

def save_diversity(entropy: float, disagreement: float, path: str):
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
    ensemble_models = []
    all_members_best_metrics = []
    logger.info(f"Training {num_members} ensemble members.")
    
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
        # _run_single_training_session_ensemble now returns path, logits, labels, metrics
        best_model_path, val_predictions_logits, val_true_labels, best_metrics = _run_single_training_session_ensemble(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=i,
            cfg=cfg, # Pass full config
            replay_buffer=train_loader.dataset.replay_buffer if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and hasattr(train_loader.dataset, 'replay_buffer') else None
        )
        
        # Load the best model saved by _run_single_training_session_ensemble
        loaded_model = PerceptionModule(cfg).to(DEVICE)
        loaded_model_state_dict = torch.load(best_model_path, map_location=DEVICE)
        # Handle DDP prefix if needed
        if list(loaded_model_state_dict.keys())[0].startswith('module.'):
            loaded_model_state_dict = {k.replace('module.', ''): v for k, v in loaded_model_state_dict.items()}
        loaded_model.load_state_dict(loaded_model_state_dict)
        loaded_model.eval() # Ensure it's in eval mode
        
        ensemble_models.append(loaded_model)
        all_members_best_metrics.append(best_metrics) # This already contains logits and labels
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
        criterion_bongard = LabelSmoothingCrossEntropy(smoothing=cfg['training']['label_smoothing']) # Use correct config key
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
    all_members_val_labels_list = [] # List of (N_val,) labels for each member (should be identical)
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
            all_members_val_labels_list.append(metrics['labels'])
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
                all_members_val_labels_list.append(val_metrics['labels'])
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
    if all_members_val_labels_list:
        true_labels_for_ensemble_eval = all_members_val_labels_list[0]
    else:
        true_labels_for_ensemble_eval = np.array([]) # Empty if no labels

    teacher_for_distillation = None # Initialize teacher for distillation
    
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
        replay_buffer=train_loader.dataset.replay_buffer if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and hasattr(train_loader.dataset, 'replay_buffer') else None,
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
        # Return empty arrays with correct shape if possible, or just empty
        num_classes = config['model']['bongard_head_config']['num_classes'] if 'model' in config and 'bongard_head_config' in config['model'] else 2
        return np.empty((len(image_paths), num_classes)), []
    
    all_bongard_probs = []
    first_model_symbolic_outputs = [] # Collect symbolic outputs from the first model for example
    
    # Dummy dataset and loader for inference (since get_loader expects a dataset)
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
            
            # Dummy bounding boxes and masks (empty lists for no detections)
            dummy_bboxes = []
            dummy_masks = []
            
            # Return tuple matching custom_collate_fn structure
            return (transformed_image, transformed_image, 0, b'{}', b'{}', 0.5, np.eye(3).tolist(), np.eye(3).tolist(), idx,
                    torch.zeros(1, 3, self.cfg['data']['image_size'], self.cfg['data']['image_size']), # Padded support imgs
                    torch.tensor([-1]), b'{}', torch.tensor(0), torch.tensor(idx), torch.tensor(1.0),
                    dummy_bboxes, dummy_masks, # query_bboxes_view1, query_masks_view1
                    dummy_bboxes, dummy_masks, # query_bboxes_view2, query_masks_view2
                    dummy_bboxes, dummy_masks) # support_bboxes_flat, support_masks_flat
            
    inference_dataset = InferenceDataset(image_paths, config)
    inference_loader = get_loader(config, train=False, rank=0, world_size=1, dataset=inference_dataset)
    logger.info("Inference loader initialized.")
    
    for model_idx, model in enumerate(models):
        model.eval()
        if use_mc_dropout:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
            logger.info(f"MC Dropout enabled for model {model_idx}.")
        else:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
        
        member_probs = []
        member_symbolic_outputs_mc_samples = []
        
        with torch.no_grad():
            for mc_sample_idx in tqdm(range(mc_dropout_samples if use_mc_dropout else 1), desc=f"Model {model_idx} MC Inference"):
                if hasattr(inference_loader, 'reset'):
                    inference_loader.reset()
                
                current_sample_probs = []
                current_batch_symbolic_outputs = []
                
                for batch_idx, data in enumerate(inference_loader):
                    # Extract processed images, and dummy bboxes/masks for PerceptionModule
                    images_view1 = data[0].to(DEVICE) # Assuming data[0] is the processed image
                    # The get_loader + InferenceDataset now returns the full tuple, extract bboxes/masks from it
                    query_bboxes_view1 = data[15] # Index 15 for query_bboxes_view1
                    query_masks_view1 = data[16]  # Index 16 for query_masks_view1
                    
                    dummy_gts_json_strings = [b'{}'] * images_view1.shape[0]
                    
                    outputs = model(images_view1, ground_truth_json_strings=dummy_gts_json_strings,
                                    detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1)
                    
                    bongard_logits = outputs['bongard_logits']
                    scene_graphs = outputs['scene_graphs'] # Get scene graphs directly from PerceptionModule output
                    
                    probs = F.softmax(bongard_logits, dim=-1).cpu().numpy()
                    current_sample_probs.append(probs)
                    
                    if model_idx == 0:
                        for b_idx in range(images_view1.shape[0]):
                            current_sg = scene_graphs[b_idx]
                            current_sg["bongard_prediction"] = int(torch.argmax(bongard_logits[b_idx]).item())
                            current_sg["raw_bongard_logits"] = bongard_logits[b_idx].cpu().numpy().tolist()
                            current_sg["bongard_probs"] = probs[b_idx].tolist()
                            current_batch_symbolic_outputs.append(current_sg)
                
                member_probs.append(np.concatenate(current_sample_probs, axis=0))
                if model_idx == 0:
                    member_symbolic_outputs_mc_samples.append(current_batch_symbolic_outputs)
        
        if use_mc_dropout:
            avg_member_probs = np.mean(np.stack(member_probs, axis=0), axis=0)
            logger.info(f"Model {model_idx} averaged {mc_dropout_samples} MC Dropout samples.")
            
            if model_idx == 0 and member_symbolic_outputs_mc_samples:
                num_images_in_batch = len(member_symbolic_outputs_mc_samples[0])
                for img_idx in range(num_images_in_batch):
                    all_mc_logits_for_image = []
                    for mc_sample_data in member_symbolic_outputs_mc_samples:
                        all_mc_logits_for_image.append(mc_sample_data[img_idx]["raw_bongard_logits"])
                    
                    all_mc_logits_for_image_np = np.array(all_mc_logits_for_image)
                    
                    epistemic_probs = F.softmax(torch.tensor(all_mc_logits_for_image_np), dim=-1).numpy()
                    epistemic_uncertainty = np.mean(np.var(epistemic_probs, axis=0))
                    aleatoric_uncertainty = np.mean(epistemic_probs * (1 - epistemic_probs))
                    
                    member_symbolic_outputs_mc_samples[0][img_idx]["uncertainty"] = {
                        "epistemic": float(epistemic_uncertainty),
                        "aleatoric": float(aleatoric_uncertainty)
                    }
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0]
        else:
            avg_member_probs = member_probs[0]
            if model_idx == 0:
                for img_data in member_symbolic_outputs_mc_samples[0]:
                    img_data["uncertainty"] = {"epistemic": 0.0, "aleatoric": 0.0}
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0]
        
        all_bongard_probs.append(avg_member_probs)
    
    stacked_probs = np.stack(all_bongard_probs, axis=0)
    
    if model_weights is not None and len(model_weights) == len(models):
        normalized_weights = np.array(model_weights) / np.sum(model_weights)
        weighted_probs = stacked_probs * normalized_weights[:, np.newaxis, np.newaxis]
        ensemble_averaged_probs = np.sum(weighted_probs, axis=0)
        logger.info("Ensemble prediction using weighted averaging.")
    else:
        ensemble_averaged_probs = np.mean(stacked_probs, axis=0)
        logger.info("Ensemble prediction using simple averaging.")
    
    logger.info(f"Ensemble prediction complete. Averaged probabilities for {len(image_paths)} images.")
    return ensemble_averaged_probs, first_model_symbolic_outputs
