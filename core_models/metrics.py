# Folder: bongard_solver/core_models/
# File: metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, brier_score_loss,
    confusion_matrix, f1_score # Explicitly import f1_score for rule_match_f1
)
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import torch # For softmax in ROC AUC and for box_iou
import torch.nn.functional as F # For softmax

# 12.1 mAP for Object Detection
try:
    from torchvision.ops import box_iou
    HAS_TORCHVISION_OPS = True
except ImportError:
    HAS_TORCHVISION_OPS = False
    logging.getLogger(__name__).warning("torchvision.ops not found. mAP for object detection will be unavailable.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def classification_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the classification accuracy score.
    Args:
        predictions (np.ndarray): Predicted class labels.
        labels (np.ndarray): True class labels.
    Returns:
        float: Accuracy score.
    """
    if len(predictions) == 0 or len(labels) == 0:
        logger.warning("No predictions or labels provided for accuracy calculation. Returning 0.0.")
        return 0.0
    if predictions.shape != labels.shape:
        logger.error(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}. Cannot calculate accuracy.")
        return 0.0
    acc = accuracy_score(labels, predictions)
    logger.debug(f"Classification Accuracy: {acc:.4f}")
    return acc

def calculate_precision_recall_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculates precision, recall, and F1-score.
    Args:
        predictions (np.ndarray): Predicted class labels.
        labels (np.ndarray): True class labels.
        average (str): Type of averaging to perform ('micro', 'macro', 'weighted', None).
    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and f1-score.
    """
    if len(predictions) == 0 or len(labels) == 0:
        logger.warning("No predictions or labels provided for precision/recall/f1 calculation. Returning zeros.")
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    if predictions.shape != labels.shape:
        logger.error(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}. Cannot calculate precision/recall/f1.")
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Handle cases where there's only one class present in labels or predictions
    unique_labels = np.unique(labels)
    unique_preds = np.unique(predictions)

    if len(unique_labels) < 2 and len(unique_preds) < 2:
        logger.warning("Only one class present in labels and predictions for precision/recall/f1. Metrics might be ill-defined.")
        # If all predictions match all labels and there's only one class, P/R/F1 can be 1.0
        if np.array_equal(predictions, labels):
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
        else:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    elif len(unique_labels) < 2:
        logger.warning("Only one class present in true labels for precision/recall/f1. Metrics might be ill-defined.")
        # If only one true class, and predictions vary, precision/recall/f1 might be 0 or 1 depending on 'average'
        # Set zero_division=0 to explicitly handle this.
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average=average, zero_division=0)
        metrics = {'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1)}
        logger.debug(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f} (average={average})")
        return metrics

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    metrics = {'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1)}
    logger.debug(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f} (average={average})")
    return metrics

def calculate_roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    Assumes binary classification where probabilities are for the positive class (label 1).
    Args:
        probabilities (np.ndarray): Predicted probabilities for the positive class.
        labels (np.ndarray): True binary labels (0 or 1).
    Returns:
        float: ROC AUC score.
    """
    if len(probabilities) == 0 or len(labels) == 0:
        logger.warning("No probabilities or labels provided for ROC AUC calculation. Returning 0.5.")
        return 0.5
    if probabilities.shape[0] != labels.shape[0]:
        logger.error(f"Shape mismatch: probabilities {probabilities.shape} vs labels {labels.shape}. Cannot calculate ROC AUC.")
        return 0.5

    # Check if there are at least two unique labels for ROC AUC
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present in true labels for ROC AUC calculation. Returning 0.5.")
        return 0.5
    try:
        roc_auc = roc_auc_score(labels, probabilities)
        logger.debug(f"ROC AUC: {roc_auc:.4f}")
        return roc_auc
    except ValueError as e:
        logger.warning(f"Could not calculate ROC AUC: {e}. Returning 0.5.")
        return 0.5

def calculate_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Brier score, a measure of the accuracy of probabilistic predictions.
    Lower Brier score is better.
    Args:
        probabilities (np.ndarray): Predicted probabilities for the positive class.
        labels (np.ndarray): True binary labels (0 or 1).
    Returns:
        float: Brier score.
    """
    if len(probabilities) == 0 or len(labels) == 0:
        logger.warning("No probabilities or labels provided for Brier score calculation. Returning 0.0.")
        return 0.0
    if probabilities.shape[0] != labels.shape[0]:
        logger.error(f"Shape mismatch: probabilities {probabilities.shape} vs labels {labels.shape}. Cannot calculate Brier score.")
        return 0.0
    brier = brier_score_loss(labels, probabilities)
    logger.debug(f"Brier Score: {brier:.4f}")
    return brier

def calculate_expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Calculates the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    ECE measures the difference between confidence and accuracy.
    Args:
        probabilities (np.ndarray): Predicted probabilities for the positive class.
        labels (np.ndarray): True binary labels (0 or 1).
        num_bins (int): Number of bins for calibration.
    Returns:
        Dict[str, float]: Dictionary containing ECE and MCE.
    """
    if len(probabilities) == 0 or len(labels) == 0:
        logger.warning("No probabilities or labels provided for ECE calculation. Returning zeros.")
        return {'ece': 0.0, 'mce': 0.0}
    if probabilities.shape[0] != labels.shape[0]:
        logger.error(f"Shape mismatch: probabilities {probabilities.shape} vs labels {labels.shape}. Cannot calculate ECE.")
        return {'ece': 0.0, 'mce': 0.0}

    bins = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        
        # Select samples that fall into the current confidence bin
        in_bin = (probabilities > lower_bound) & (probabilities <= upper_bound)
        
        if np.sum(in_bin) > 0:
            bin_accuracies[i] = np.mean(labels[in_bin])
            bin_confidences[i] = np.mean(probabilities[in_bin])
            bin_counts[i] = np.sum(in_bin)

    ece = 0.0
    mce = 0.0
    total_samples = len(probabilities)

    if total_samples == 0: # Avoid division by zero if no samples
        return {'ece': 0.0, 'mce': 0.0}

    for i in range(num_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * np.abs(bin_accuracies[i] - bin_confidences[i])
            mce = max(mce, np.abs(bin_accuracies[i] - bin_confidences[i]))
            
    metrics = {'ece': float(ece), 'mce': float(mce)}
    logger.debug(f"ECE: {ece:.4f}, MCE: {mce:.4f}")
    return metrics

def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
):
    """
    Plots a reliability diagram to visualize model calibration.
    Args:
        probabilities (np.ndarray): Predicted probabilities for the positive class.
        labels (np.ndarray): True binary labels (0 or 1).
        num_bins (int): Number of bins for calibration.
        title (str): Title of the plot.
        save_path (Optional[str]): Path to save the plot. If None, displays the plot.
    """
    if len(probabilities) == 0 or len(labels) == 0:
        logger.warning("No probabilities or labels provided for reliability diagram. Skipping plot.")
        return
    if probabilities.shape[0] != labels.shape[0]:
        logger.error(f"Shape mismatch: probabilities {probabilities.shape} vs labels {labels.shape}. Cannot plot reliability diagram.")
        return

    bins = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        in_bin = (probabilities > lower_bound) & (probabilities <= upper_bound)
        
        if np.sum(in_bin) > 0:
            bin_accuracies[i] = np.mean(labels[in_bin])
            bin_confidences[i] = np.mean(probabilities[in_bin])
            bin_counts[i] = np.sum(in_bin)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Plot only bins that have data points
    valid_bins = bin_counts > 0
    plt.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 's-', color='blue', label='Model Calibration')
    
    plt.xlabel("Confidence (Mean Predicted Probability)")
    plt.ylabel("Accuracy (Fraction of Positives)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Reliability diagram saved to {save_path}")
    else:
        plt.show()
    plt.close()

def detection_map(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh: float = 0.5) -> float:
    """
    Computes Average Precision (AP) for object detection.
    This is a simplified implementation for a single class and assumes inputs are
    already processed (e.g., non-max suppression applied to predictions).
    For full mAP@0.5:0.95, a more robust implementation is needed (e.g., from pycocotools).
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes (N, 4) in (x1, y1, x2, y2) format.
        pred_scores (torch.Tensor): Confidence scores for predicted boxes (N,).
        gt_boxes (torch.Tensor): Ground truth bounding boxes (M, 4) in (x1, y1, x2, y2) format.
        iou_thresh (float): IoU threshold to consider a detection a True Positive.
    Returns:
        float: Average Precision (AP) for the given class and IoU threshold.
    """
    if not HAS_TORCHVISION_OPS:
        logger.warning("torchvision.ops not available. Cannot compute detection mAP. Returning 0.0.")
        return 0.0
    
    # Handle empty predictions or ground truths
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 1.0 # Perfect score if both are empty (no objects, correctly predicted no objects)
    if pred_boxes.numel() == 0 and gt_boxes.numel() > 0:
        return 0.0 # No predictions but there are ground truths
    if pred_boxes.numel() > 0 and gt_boxes.numel() == 0:
        # Many false positives, so AP should be low.
        # A simple way: if any predictions, and no GT, then 0.
        return 0.0 
    
    # Sort predictions by confidence in descending order
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]
    
    # Keep track of which ground truth boxes have been matched
    matched_gts = torch.zeros(num_gts, dtype=torch.bool, device=gt_boxes.device)
    
    true_positives = torch.zeros(num_preds, dtype=torch.bool, device=pred_boxes.device)
    false_positives = torch.zeros(num_preds, dtype=torch.bool, device=pred_boxes.device)

    for i in range(num_preds):
        pred_box = pred_boxes[i].unsqueeze(0) # Make it (1, 4)
        
        # Calculate IoU between the current prediction and all ground truth boxes
        ious = box_iou(pred_box, gt_boxes).squeeze(0) # Result is (num_gts,)
        
        if ious.numel() > 0: # Ensure there are actual IoU values
            best_iou, best_gt_idx = ious.max(dim=0) # Find the best matching GT
            
            # If best IoU is above threshold and the GT box hasn't been matched yet
            if best_iou >= iou_thresh and not matched_gts[best_gt_idx]:
                true_positives[i] = True
                matched_gts[best_gt_idx] = True # Mark this GT as matched
            else:
                false_positives[i] = True
        else:
            false_positives[i] = True # No GT boxes to compare with, so it's a false positive
    
    # Calculate cumulative true positives and false positives
    tp_cumsum = torch.cumsum(true_positives, dim=0).float()
    fp_cumsum = torch.cumsum(false_positives, dim=0).float()
    
    # Calculate precision and recall curves
    recall = tp_cumsum / (num_gts + 1e-6) # Add epsilon to avoid division by zero
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Add (0,1) and (1,0) points for interpolation (start and end of PR curve)
    precision = torch.cat((torch.tensor([1.0], device=precision.device), precision))
    recall = torch.cat((torch.tensor([0.0], device=recall.device), recall))
    
    # Compute Average Precision (AP) using numerical integration (trapz)
    # This is a common way to approximate the area under the PR curve.
    average_precision = torch.trapz(precision, recall).item()
    logger.debug(f"Detection mAP (IoU={iou_thresh}): {average_precision:.4f}")
    return average_precision

def plot_cm(cm: np.ndarray, class_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Plots a confusion matrix and returns it as a numpy array (image).
    Args:
        cm (np.ndarray): The confusion matrix.
        class_names (Optional[List[str]]): List of class names for labels.
    Returns:
        np.ndarray: A numpy array representing the confusion matrix plot (RGB image).
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    num_classes = cm.shape[0]
    tick_marks = np.arange(num_classes)
    if class_names:
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    else:
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    fig = plt.gcf()
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img_array

def log_confusion(preds: np.ndarray, targets: np.ndarray, writer: Any, epoch: int, class_names: Optional[List[str]] = None):
    """
    Computes and logs a confusion matrix to TensorBoard.
    Args:
        preds (np.ndarray): Predicted labels.
        targets (np.ndarray): True labels.
        writer (Any): TensorBoard SummaryWriter instance.
        epoch (int): Current epoch number.
        class_names (Optional[List[str]]): List of class names for the confusion matrix plot.
    """
    if len(preds) == 0 or len(targets) == 0:
        logger.warning("No predictions or targets provided for confusion matrix logging. Skipping.")
        return
    
    cm = confusion_matrix(targets, preds)
    cm_image = plot_cm(cm, class_names)
    writer.add_image('ConfusionMatrix', cm_image.transpose(2, 0, 1), epoch)
    logger.info(f"Confusion matrix logged for epoch {epoch}.")

# --- New: Rule Matching F1 Score ---
def rule_match_f1(
    predicted_rules: List[Any], # List of BongardRule objects or similar
    ground_truth_rules: List[Any], # List of BongardRule objects or similar
    match_threshold: float = 0.8 # Similarity threshold for considering two rules a match
) -> Dict[str, float]:
    """
    Calculates F1-score for rule matching.
    This metric requires a way to compare two rules for similarity.
    For demonstration, we'll use a dummy similarity based on rule names/descriptions.
    In a real system, this would involve comparing ASTs or logical forms.
    
    Args:
        predicted_rules (List[Any]): List of rules predicted by the solver.
        ground_truth_rules (List[Any]): List of true rules for the problem.
        match_threshold (float): Similarity threshold to consider a predicted rule
                                 a match for a ground truth rule.
    Returns:
        Dict[str, float]: Dictionary containing precision, recall, and F1-score for rule matching.
    """
    if not predicted_rules and not ground_truth_rules:
        logger.info("No predicted or ground truth rules. Returning perfect F1 (1.0).")
        return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
    if not predicted_rules:
        logger.info("No predicted rules. Returning F1 (0.0).")
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    if not ground_truth_rules:
        logger.info("No ground truth rules. Returning F1 (0.0).")
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Dummy rule similarity function for demonstration
    def _rule_similarity(rule1: Any, rule2: Any) -> float:
        """
        Calculates a dummy similarity between two rules.
        In a real scenario, this would compare ASTs, logical forms, or evaluate
        them on a set of examples.
        For now, let's assume rules have a 'name' attribute.
        """
        if hasattr(rule1, 'name') and hasattr(rule2, 'name'):
            # Simple string matching for demo
            return 1.0 if rule1.name == rule2.name else 0.0
        return 0.0

    # Create a binary matching matrix
    # match_matrix[i][j] = 1 if predicted_rules[i] matches ground_truth_rules[j]
    match_matrix = np.zeros((len(predicted_rules), len(ground_truth_rules)), dtype=np.bool_)

    for i, pred_rule in enumerate(predicted_rules):
        for j, gt_rule in enumerate(ground_truth_rules):
            if _rule_similarity(pred_rule, gt_rule) >= match_threshold:
                match_matrix[i, j] = True

    # Calculate True Positives, False Positives, False Negatives
    # A predicted rule is a TP if it matches at least one GT rule.
    # A GT rule is covered if at least one predicted rule matches it.
    
    # Each predicted rule should match at most one GT rule for a clean count.
    # For simplicity, if a predicted rule matches multiple GTs, we count it as 1 TP.
    # And if a GT rule is matched by multiple predictions, it's still 1 covered GT.

    # True Positives: number of ground truth rules that are correctly matched by at least one predicted rule
    # (i.e., columns in match_matrix that have at least one True)
    tp = np.sum(np.any(match_matrix, axis=0))

    # False Positives: number of predicted rules that do not match any ground truth rule
    # (i.e., rows in match_matrix that are all False)
    fp = np.sum(np.all(~match_matrix, axis=1))

    # False Negatives: number of ground truth rules that are not matched by any predicted rule
    # (i.e., columns in match_matrix that are all False)
    fn = np.sum(np.all(~match_matrix, axis=0))

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    logger.debug(f"Rule Match F1: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")
    return metrics

