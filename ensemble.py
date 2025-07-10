# Folder: bongard_solver/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import os
import collections
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional, Union

# Import from config
from config import CONFIG, DEVICE, HAS_WANDB

# Import from utils
from utils import setup_logging, set_seed, get_symbolic_embedding_dims # Assuming these are in utils.py

# Import from data
from data import build_dali_loader, BongardSyntheticDataset, CurriculumSampler, load_bongard_data, RealBongardDataset, BongardGenerator # Assuming these are in data.py
from bongard_rules import ALL_BONGARD_RULES # Needed for BongardGenerator

# Import from models
from models import PerceptionModule # Assuming this is in models.py

# Import from training
from training import _run_single_training_session_ensemble, _validate_model_ensemble # Assuming these are in training.py

# Import from metrics
from metrics import calculate_accuracy, calculate_precision_recall_f1, calculate_roc_auc, calculate_brier_score, calculate_expected_calibration_error # Assuming these are in metrics.py

logger = logging.getLogger(__name__)


def train_ensemble(
    num_members: int,
    train_loader: Any,
    val_loader: Any,
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
        current_rank (int): Current GPU rank.
        world_size (int): Total number of GPUs.
        is_ddp_initialized (bool): True if DDP is already initialized.

    Returns:
        Tuple[List[nn.Module], List[Dict[str, Any]]]:
            - List of trained ensemble models.
            - List of best validation metrics for each model (including predictions, labels, logits).
    """
    ensemble_models = []
    all_members_best_metrics = []

    for i in range(num_members):
        logger.info(f"Training ensemble member {i+1}/{num_members}...")
        
        # Get symbolic embedding dimensions
        symbolic_embedding_dims = get_symbolic_embedding_dims()
        
        # Initialize a new model for each member
        model = PerceptionModule(CONFIG, symbolic_embedding_dims).to(DEVICE) # Ensure model is on correct device
        
        # If DDP is initialized, wrap the model
        if is_ddp_initialized:
            model = DDP(model, device_ids=[current_rank])

        # Initialize optimizer and scheduler for this member
        from optimizers import get_optimizer, get_scheduler
        optimizer = get_optimizer(model, CONFIG['training'])
        
        # Calculate total steps for OneCycleLR
        total_steps = CONFIG['training']['epochs'] * len(train_loader) if hasattr(train_loader, '__len__') else CONFIG['training']['epochs'] * 100 # Fallback
        scheduler = get_scheduler(optimizer, CONFIG['training'], total_steps)

        # Run training for this member
        _, _, best_metrics = _run_single_training_session_ensemble(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=i,
            replay_buffer=train_loader.external_source.dataset.curriculum_sampler.replay_buffer if CONFIG['training']['use_prioritized_replay'] else None # Pass replay buffer
        )
        
        # Store the trained model (unwrapped if DDP)
        ensemble_models.append(model.module if is_ddp_initialized else model)
        all_members_best_metrics.append(best_metrics)

        logger.info(f"Ensemble member {i+1} training completed.")

    return ensemble_models, all_members_best_metrics


def load_trained_model(
    model_path: str,
    current_rank: int = 0,
    is_ddp_initialized: bool = False,
    val_loader: Any = None # Required for re-running validation
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads a pre-trained model and optionally re-runs validation to get metrics.

    Args:
        model_path (str): Path to the model checkpoint.
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
    
    symbolic_embedding_dims = get_symbolic_embedding_dims()
    model = PerceptionModule(CONFIG, symbolic_embedding_dims).to(DEVICE)
    
    # Load state dict, handling DDP prefix if necessary
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
        val_metrics = _validate_model_ensemble(
            model=model,
            val_loader=val_loader,
            current_rank=current_rank,
            is_ddp_initialized=is_ddp_initialized,
            model_idx=0 # Dummy index for loaded model
        )
        logger.info(f"Validation on loaded model complete. Accuracy: {val_metrics['accuracy']:.4f}")
    else:
        logger.warning("No validation loader provided. Cannot get fresh validation metrics for loaded model.")
        val_metrics = {
            'predictions': np.array([]),
            'labels': np.array([]),
            'logits': np.array([]),
            'accuracy': 0.0,
            'loss': float('inf')
        }

    return model.module if is_ddp_initialized else model, val_metrics


def weighted_average_ensemble_prediction(
    all_members_val_predictions_logits: List[np.ndarray], # List of (N, num_classes) logits
    all_members_val_labels: List[np.ndarray], # List of (N,) labels
    ensemble_member_accuracies: List[float] # List of accuracies for each member
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Combines predictions from ensemble members using a weighted average of their logits.
    Weights are based on individual model accuracies.

    Args:
        all_members_val_predictions_logits (List[np.ndarray]): List of validation logits
                                                                (each array is for one member).
        all_members_val_labels (List[np.ndarray]): List of validation labels (should be identical).
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
    true_labels = all_members_val_labels[0] # All label lists should be identical

    # Calculate ensemble metrics
    ensemble_metrics = _calculate_ensemble_metrics(combined_predictions, true_labels, combined_logits)
    logger.info(f"Weighted Average Ensemble Metrics: Accuracy: {ensemble_metrics['accuracy']:.4f}")

    return combined_logits, combined_predictions, true_labels, ensemble_metrics


class StackedEnsembleMetaLearner(nn.Module):
    """
    A simple meta-learner for stacked ensemble.
    Takes concatenated logits from base models and predicts final class.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim // 2, num_classes)
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
    train_members: bool = True, # Whether to train members or load them
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
            current_rank=current_rank,
            world_size=world_size,
            is_ddp_initialized=is_ddp_initialized
        )
        for metrics in all_members_best_metrics:
            all_members_val_predictions_logits.append(metrics['logits'])
            all_members_val_labels.append(metrics['labels'])
            ensemble_member_accuracies.append(metrics['accuracy'])
    else:
        # Load pre-trained teacher ensemble members
        logger.info("Loading pre-trained teacher ensemble members...")
        model_save_dir = CONFIG['debug']['save_model_checkpoints']
        for i in range(num_ensemble_members):
            model_path = os.path.join(model_save_dir, f"best_model_member_{i}.pth")
            if os.path.exists(model_path):
                # Load model and re-run validation to get fresh metrics
                model, val_metrics = load_trained_model(
                    model_path=model_path,
                    current_rank=current_rank,
                    is_ddp_initialized=is_ddp_initialized,
                    val_loader=val_loader # Pass val_loader to get actual metrics
                )
                teacher_models.append(model)
                all_members_val_predictions_logits.append(val_metrics['logits'])
                all_members_val_labels.append(val_metrics['labels'])
                ensemble_member_accuracies.append(val_metrics['accuracy'])
            else:
                logger.warning(f"Teacher model {model_path} not found. Skipping this member.")
        
        if not teacher_models:
            logger.error("No teacher models loaded. Cannot proceed with distillation.")
            return None, {}, {}

    # Combine teacher predictions based on ensemble type
    teacher_ensemble_logits = None
    teacher_ensemble_metrics = {}
    num_bongard_classes = CONFIG['model']['bongard_head_config']['num_classes']

    if teacher_ensemble_type == 'weighted_average':
        logger.info("Combining teacher predictions using weighted average.")
        # Need to ensure all_members_val_labels is consistent (all same)
        if all_members_val_labels:
            true_labels_for_ensemble_eval = all_members_val_labels[0]
        else:
            true_labels_for_ensemble_eval = np.array([]) # Empty if no labels
        
        teacher_ensemble_logits, _, _, teacher_ensemble_metrics = weighted_average_ensemble_prediction(
            all_members_val_predictions_logits=all_members_val_predictions_logits,
            all_members_val_labels=true_labels_for_ensemble_eval,
            ensemble_member_accuracies=ensemble_member_accuracies
        )
    elif teacher_ensemble_type == 'stacked':
        logger.info("Training stacked ensemble meta-learner.")
        # Ensure all_members_val_labels is consistent (all same)
        if all_members_val_labels:
            true_labels_for_ensemble_eval = all_members_val_labels[0]
        else:
            true_labels_for_ensemble_eval = np.array([]) # Empty if no labels

        meta_learner, teacher_ensemble_metrics = train_stacked_ensemble_meta_learner(
            all_members_val_predictions_logits=all_members_val_predictions_logits,
            all_members_val_labels=true_labels_for_ensemble_eval,
            num_classes=num_bongard_classes,
            epochs=CONFIG['training']['stacked_ensemble_config']['meta_learner_epochs'],
            lr=CONFIG['training']['stacked_ensemble_config']['meta_learner_lr']
        )
        # Get logits from the trained meta-learner on the validation set
        meta_input_np = np.concatenate(all_members_val_predictions_logits, axis=1)
        meta_input = torch.tensor(meta_input_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            teacher_ensemble_logits = meta_learner(meta_input).cpu().numpy()
    else:
        raise ValueError(f"Unsupported teacher_ensemble_type: {teacher_ensemble_type}")

    if teacher_ensemble_logits is None:
        logger.error("Failed to obtain teacher ensemble logits. Aborting student training.")
        return None, teacher_ensemble_metrics, {}

    logger.info("Teacher ensemble predictions combined. Starting student model training with distillation.")

    # Train student model with knowledge distillation
    symbolic_embedding_dims = get_symbolic_embedding_dims()
    student_model = PerceptionModule(student_model_config, symbolic_embedding_dims).to(DEVICE)
    if is_ddp_initialized:
        student_model = DDP(student_model, device_ids=[current_rank])

    from optimizers import get_optimizer, get_scheduler
    student_optimizer = get_optimizer(student_model, {'optimizer': CONFIG['training']['optimizer'], 'learning_rate': lr_student, 'sam_rho': CONFIG['training']['sam_rho']})
    total_steps_student = epochs_student * len(train_loader) if hasattr(train_loader, '__len__') else epochs_student * 100
    student_scheduler = get_scheduler(student_optimizer, {'scheduler': 'OneCycleLR', 'scheduler_config': {'OneCycleLR': {'max_lr': lr_student * 10, 'pct_start': 0.3, 'anneal_strategy': 'cos'}}, 'epochs': epochs_student}, total_steps_student)

    # Pass teacher logits to the training loop via CONFIG or directly
    # For simplicity, let's assume `_run_single_training_session_ensemble` can accept `teacher_logits`
    # or that it's set in a global config for distillation.
    # A cleaner way is to create a custom distillation training loop or pass it as an argument.
    # For now, we'll pass it via a modified CONFIG for the student training session.
    
    # Create a temporary config for student training to include teacher_logits
    student_training_config_temp = CONFIG['training'].copy()
    student_training_config_temp['use_knowledge_distillation'] = True
    # Store the full teacher_ensemble_logits in the config. The training loop will index into it.
    student_training_config_temp['teacher_logits'] = torch.tensor(teacher_ensemble_logits, dtype=torch.float32).to(DEVICE) 
    
    # Temporarily override CONFIG for student training
    original_config_training = CONFIG['training']
    CONFIG['training'] = student_training_config_temp

    _, _, student_metrics = _run_single_training_session_ensemble(
        model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=student_optimizer,
        scheduler=student_scheduler,
        current_rank=current_rank,
        is_ddp_initialized=is_ddp_initialized,
        model_idx=-1, # Indicate student model
        replay_buffer=train_loader.external_source.dataset.curriculum_sampler.replay_buffer if CONFIG['training']['use_prioritized_replay'] else None
    )

    # Restore original CONFIG
    CONFIG['training'] = original_config_training

    logger.info("Student model training with distillation completed.")

    return student_model.module if is_ddp_initialized else student_model, teacher_ensemble_metrics, student_metrics

