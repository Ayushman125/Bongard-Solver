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
from tqdm import tqdm # For progress bars

# Import from config (assuming config.py is in the project root)
try:
    from ..config import CONFIG, DEVICE, HAS_WANDB, IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS, DATA_ROOT_PATH
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import config. Using dummy CONFIG, DEVICE, etc.")
    # Dummy CONFIG, DEVICE, etc. for standalone execution
    CONFIG = {}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_WANDB = False
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]
    NUM_CHANNELS = 3
    DATA_ROOT_PATH = "./data"

# Import from utils (assuming utils.py is in src/)
try:
    from ..src.utils import setup_logging, set_seed, get_symbolic_embedding_dims # Assuming these are in src/utils.py
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import utils.py. Some utility functions will be dummy.")
    def setup_logging(log_dir, rank): pass
    def set_seed(seed): pass
    def get_symbolic_embedding_dims(config): return 128 # Dummy

# Import from data (assuming data.py is in src/)
try:
    from ..data import build_dali_loader, BongardSyntheticDataset, CurriculumSampler, load_bongard_data, RealBongardDataset, BongardGenerator, get_loader, custom_collate_fn, BongardDataModule
    HAS_DATA_MODULE = True
except ImportError:
    HAS_DATA_MODULE = False
    logger = logging.getLogger(__name__)
    logger.warning("Could not import data module components. Data loading will be limited.")
    # Dummy classes/functions for data if not available
    class BongardDataModule:
        def __init__(self, *args, **kwargs): pass
        def setup(self, stage): pass
        def train_dataloader(self): return []
        def val_dataloader(self): return []
    def build_dali_loader(*args, **kwargs): return None
    def get_loader(*args, **kwargs): return []
    def custom_collate_fn(batch): return batch # Passthrough
    class BongardSyntheticDataset:
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 0
        def __getitem__(self, idx): return None
    class RealBongardDataset:
        def __init__(self, *args, **kwargs): pass
        def __len__(self): return 0
        def __getitem__(self, idx): return None
    class BongardGenerator:
        def __init__(self, *args, **kwargs): pass
        def generate_problem(self): return None
    class CurriculumSampler:
        def __init__(self, *args, **kwargs): pass
        def __iter__(self): return iter([])
        def __len__(self): return 0

# Import from bongard_rules (assuming bongard_rules.py is in src/)
try:
    from ..src.bongard_rules import ALL_BONGARD_RULES # Needed for BongardGenerator
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import bongard_rules.py. Bongard rule related functionalities will be limited.")
    ALL_BONGARD_RULES = {}

# Import from models (from current directory)
from .models import PerceptionModule, LitBongard # Import LitBongard for student training

# Import from training (from current directory)
# These functions are assumed to be in training.py
try:
    from .training import _run_single_training_session_ensemble, _validate_model_ensemble, _get_ensemble_teacher_logits
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import training utilities. Ensemble training will be limited.")
    def _run_single_training_session_ensemble(*args, **kwargs):
        logger.error("Dummy _run_single_training_session_ensemble called.")
        return "", [], [], {} # Dummy return
    def _validate_model_ensemble(*args, **kwargs):
        logger.error("Dummy _validate_model_ensemble called.")
        return 0.0, 0.0, [], [] # Dummy return
    def _get_ensemble_teacher_logits(*args, **kwargs):
        logger.error("Dummy _get_ensemble_teacher_logits called.")
        return torch.empty(0), None # Dummy return

# Import from optimizers (from current directory)
from .optimizers import get_optimizer, get_scheduler

# Import from losses (from current directory)
from .losses import LabelSmoothingCrossEntropy, FeatureConsistencyLoss, SymbolicConsistencyLoss, DistillationLoss

# Import from metrics (from current directory)
from .metrics import classification_accuracy, calculate_precision_recall_f1, calculate_roc_auc, calculate_brier_score, calculate_expected_calibration_error, plot_cm, log_confusion

# SWA imports
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn # Explicitly import

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
    
    all_member_probs_per_batch = [] # List of numpy arrays, each (num_members, batch_size, num_classes)
    
    # Reset loader for consistent evaluation
    if hasattr(data_loader, 'reset'):
        data_loader.reset()
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader, desc="Computing Diversity")):
            # Extract processed images, and dummy bboxes/masks for PerceptionModule
            # Assuming data is a tuple from custom_collate_fn
            # Ensure data is moved to the correct device
            images_view1 = data[0]
            if isinstance(images_view1, np.ndarray): # Convert numpy to tensor if not already
                images_view1 = torch.from_numpy(images_view1).to(DEVICE)
            else:
                images_view1 = images_view1.to(DEVICE)

            query_bboxes_view1 = data[15] # Index 15 for query_bboxes_view1
            query_masks_view1 = data[16] # Index 16 for query_masks_view1
            dummy_gts_json_strings = [b'{}'] * images_view1.shape[0] # Dummy GT JSONs
            
            batch_member_probs = []
            for m in members:
                # PerceptionModule.forward expects detected_bboxes_batch and detected_masks_batch
                outputs = m(images_view1, ground_truth_json_strings=dummy_gts_json_strings,
                            detected_bboxes_batch=query_bboxes_view1, detected_masks_batch=query_masks_view1)
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
    # Add epsilon for log(0) and handle cases where mean_p might have very small values
    entropy = -(mean_p * torch.log(mean_p + 1e-9)).sum(-1).mean().item() 
    
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
    logger.info("Starting ensemble teacher training.")
    ensemble_models = []
    all_members_best_metrics = []
    logger.info(f"Training {num_members} ensemble members.")
    
    for i in range(num_members):
        logger.info(f"Training ensemble member {i+1}/{num_members}...")
        
        # Initialize a new model (Lightning Module) for each member
        # LitBongard wraps PerceptionModule
        model = LitBongard(cfg).to(DEVICE)
        
        # If DDP is initialized, wrap the model
        if is_ddp_initialized:
            model = DDP(model, device_ids=[current_rank])
        
        # Run training for this member using the _run_single_training_session_ensemble utility
        best_model_path, val_predictions_logits, val_true_labels, best_metrics = _run_single_training_session_ensemble(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=i,
            cfg=cfg, # Pass full config
            replay_buffer=train_loader.dataset.replay_buffer if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and hasattr(train_loader.dataset, 'replay_buffer') else None
        )
        
        # Load the best model saved by _run_single_training_session_ensemble
        # Create a new LitBongard instance and load state_dict
        loaded_model = LitBongard(cfg).to(DEVICE)
        loaded_model_state_dict = torch.load(best_model_path, map_location=DEVICE)
        
        # Handle DDP prefix if needed (remove 'module.' if loading a DDP-saved model into non-DDP LitBongard)
        if list(loaded_model_state_dict.keys())[0].startswith('module.'):
            loaded_model_state_dict = {k.replace('module.', ''): v for k, v in loaded_model_state_dict.items()}
        
        loaded_model.load_state_dict(loaded_model_state_dict)
        loaded_model.eval() # Ensure it's in eval mode
        
        # If the loaded model is a DDP wrapper, unwrap it before appending
        if isinstance(loaded_model, DDP):
            ensemble_models.append(loaded_model.module)
        else:
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
    
    model = LitBongard(cfg).to(DEVICE) # Initialize LitBongard
    
    # Load state dict, handling DDP prefix if necessary
    state_dict = checkpoint
    if 'state_dict' in checkpoint: # PyTorch Lightning saves under 'state_dict'
        state_dict = checkpoint['state_dict']
    
    # Remove 'model.' prefix if it's from a LightningModule's state_dict
    state_dict = {k.replace('perception_module.', ''): v for k, v in state_dict.items() if k.startswith('perception_module.')}
    
    # Handle DDP prefix if loading a DDP-saved model into non-DDP, or vice-versa
    if is_ddp_initialized and not list(state_dict.keys())[0].startswith('module.'):
        # Add 'module.' prefix if loading a non-DDP saved model into DDP
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif not is_ddp_initialized and list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix if loading a DDP saved model into non-DDP
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load into the PerceptionModule within LitBongard
    model.perception_module.load_state_dict(state_dict)
    
    # Wrap with DDP if DDP is initialized for this process
    if is_ddp_initialized:
        model = DDP(model, device_ids=[current_rank])
    
    logger.info("Model loaded successfully.")
    
    # Re-run validation to get fresh predictions, labels, and logits
    if val_loader:
        logger.info("Re-running validation on loaded model to get metrics...")
        # _validate_model_ensemble returns val_loss, val_accuracy, val_predictions_logits, val_true_labels
        val_loss, val_accuracy, val_predictions_logits, val_true_labels = _validate_model_ensemble(
            model=model, # Pass the (possibly DDP-wrapped) LitBongard model
            data_loader=val_loader,
            criterion=None, # Criterion is handled internally by LitBongard's validation_step
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
    
    # Return the unwrapped PerceptionModule if DDP was used
    return model.module.perception_module if is_ddp_initialized else model.perception_module, val_metrics

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
    accuracy = classification_accuracy(predictions, labels)
    precision_recall_f1 = calculate_precision_recall_f1(predictions, labels)
    
    # Ensure there are at least two unique labels for ROC AUC
    if len(np.unique(labels)) > 1:
        probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
        # Assuming binary classification, take probabilities of the positive class (index 1)
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
                teacher_models.append(model) # Append the unwrapped PerceptionModule
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
    teacher_for_distillation = None # Initialize teacher for distillation (can be list of models or meta-learner)
    
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
    # The student model is also a LitBongard instance
    student_model = LitBongard(student_model_config).to(DEVICE)
    if is_ddp_initialized:
        student_model = DDP(student_model, device_ids=[current_rank])
    
    # _run_single_training_session_ensemble now accepts `teacher_model`
    _, _, _, student_metrics = _run_single_training_session_ensemble(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        current_rank=current_rank,
        is_ddp_initialized=is_ddp_initialized,
        model_idx=-1, # Indicate student model
        cfg=student_model_config, # Pass student's config
        replay_buffer=train_loader.dataset.replay_buffer if cfg['training'].get('curriculum_learning', False) and cfg['training']['curriculum_config'].get('difficulty_sampling', False) and hasattr(train_loader.dataset, 'replay_buffer') else None,
        teacher_model=teacher_for_distillation # Pass the teacher(s) or meta-learner
    )
    logger.info("Student model training with distillation completed.")
    # Return the unwrapped PerceptionModule of the student model
    return student_model.module.perception_module if is_ddp_initialized else student_model.perception_module, teacher_ensemble_metrics, student_metrics

class EnsembleSolver(nn.Module):
    """
    Combines multiple trained Bongard problem solvers (PerceptionModules)
    to make a final prediction using various voting strategies.
    """
    def __init__(self, models: List[PerceptionModule], ensemble_type: str = 'soft_vote'):
        """
        Args:
            models (List[PerceptionModule]): A list of trained PerceptionModule instances.
            ensemble_type (str): The voting strategy to use ('soft_vote', 'hard_vote', 'weighted_soft_vote').
        """
        if not models:
            raise ValueError("EnsembleSolver requires at least one model.")
        self.models = nn.ModuleList(models) # Use nn.ModuleList to register models if EnsembleSolver is part of a larger nn.Module
        self.ensemble_type = ensemble_type
        for model in self.models:
            model.eval() # Ensure all models are in evaluation mode
        logger.info(f"EnsembleSolver initialized with {len(models)} members and type '{ensemble_type}'.")

    def predict(self, 
                images: torch.Tensor, # Batch of images (B, C, H, W)
                detected_bboxes_batch: List[List[List[float]]], # List of lists of bboxes per image in batch
                detected_masks_batch: List[List[np.ndarray]], # List of lists of masks per image in batch
                support_images: torch.Tensor = None, # (B, N_support, C, H, W) for few-shot
                support_labels_flat: torch.Tensor = None, # (B, N_support) for few-shot
                model_weights: Optional[List[float]] = None # Weights for weighted_soft_vote
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combines predictions from ensemble members using the specified voting strategy.
        Args:
            images (torch.Tensor): Batch of input images (B, C, H, W).
            detected_bboxes_batch (List[List[List[float]]]): Pre-detected bounding boxes for each image in the batch.
            detected_masks_batch (List[List[np.ndarray]]): Pre-detected binary masks for each image in the batch.
            support_images (torch.Tensor): Batch of support images for few-shot learning (B, N_support, C, H, W).
            support_labels_flat (torch.Tensor): Flattened support labels (B, N_support) for few-shot.
            model_weights (Optional[List[float]]): Weights for each model for weighted averaging (only for 'weighted_soft_vote').
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ensemble_predictions (torch.Tensor): Hard predictions (class indices) (B,).
                - ensemble_probabilities (torch.Tensor): Averaged probabilities (B, num_classes).
        """
        all_logits = []
        
        with torch.no_grad():
            for model in self.models:
                # Ensure ground_truth_json_strings is provided, even if dummy, for PerceptionModule's forward
                dummy_gts_json_strings = [b'{}'] * images.shape[0]
                outputs = model(images, ground_truth_json_strings=dummy_gts_json_strings,
                                detected_bboxes_batch=detected_bboxes_batch, detected_masks_batch=detected_masks_batch,
                                support_images=support_images, support_labels_flat=support_labels_flat)
                all_logits.append(outputs['bongard_logits'])
        
        # Stack logits: (num_models, B, num_classes)
        stacked_logits = torch.stack(all_logits, dim=0)
        
        if self.ensemble_type == 'soft_vote':
            # Average probabilities (softmax over logits, then average)
            ensemble_probabilities = F.softmax(stacked_logits, dim=-1).mean(dim=0)
            ensemble_predictions = torch.argmax(ensemble_probabilities, dim=-1)
        elif self.ensemble_type == 'hard_vote':
            # Get hard predictions from each model, then majority vote
            individual_predictions = torch.argmax(stacked_logits, dim=-1) # (num_models, B)
            
            # Perform majority voting. Mode returns (values, counts)
            # If there's a tie, `mode` returns the smallest value.
            ensemble_predictions = torch.mode(individual_predictions, dim=0).values
            
            # To get ensemble probabilities for hard vote, you could convert the hard predictions
            # back to one-hot and then average, or just return zeros/ones.
            # For simplicity, we'll return average of individual model's probabilities.
            ensemble_probabilities = F.softmax(stacked_logits, dim=-1).mean(dim=0)
            
        elif self.ensemble_type == 'weighted_soft_vote':
            if model_weights is None or len(model_weights) != len(self.models):
                logger.warning("Model weights not provided or mismatch for weighted_soft_vote. Falling back to soft_vote.")
                return self.predict(images, detected_bboxes_batch, detected_masks_batch, support_images, support_labels_flat, None) # Recursive call with soft_vote
            
            weights_tensor = torch.tensor(model_weights, dtype=torch.float32, device=images.device).unsqueeze(-1).unsqueeze(-1) # (num_models, 1, 1)
            weighted_probs = F.softmax(stacked_logits, dim=-1) * weights_tensor
            ensemble_probabilities = weighted_probs.sum(dim=0) # Sum across models
            ensemble_predictions = torch.argmax(ensemble_probabilities, dim=-1)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
            
        return ensemble_predictions, ensemble_probabilities
