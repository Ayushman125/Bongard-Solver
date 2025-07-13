# Folder: bongard_solver/
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, brier_score_loss,
    confusion_matrix
)
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import os # Added for saving plots

# 12.1 mAP for Object Detection
try:
    from torchvision.ops import box_iou
    HAS_TORCHVISION_OPS = True
except ImportError:
    HAS_TORCHVISION_OPS = False
    logging.getLogger(__name__).warning("torchvision.ops not found. mAP for object detection will be unavailable.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the accuracy score.
    Args:
        predictions (np.ndarray): Predicted class labels.
        labels (np.ndarray): True class labels.
    Returns:
        float: Accuracy score.
    """
    if len(predictions) == 0:
        logger.warning("No predictions provided for accuracy calculation. Returning 0.0.")
        return 0.0
    acc = accuracy_score(labels, predictions)
    logger.debug(f"Accuracy: {acc:.4f}")
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
    if len(predictions) == 0:
        logger.warning("No predictions provided for precision/recall/f1 calculation. Returning zeros.")
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    metrics = {'precision': precision, 'recall': recall, 'f1_score': f1}
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
    if len(probabilities) == 0 or len(np.unique(labels)) < 2:
        logger.warning("Insufficient data or only one class present for ROC AUC calculation. Returning 0.5.")
        return 0.5  # Default for random classifier
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
    if len(probabilities) == 0:
        logger.warning("No probabilities provided for Brier score calculation. Returning 0.0.")
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
    if len(probabilities) == 0:
        logger.warning("No probabilities provided for ECE calculation. Returning zeros.")
        return {'ece': 0.0, 'mce': 0.0}
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        
        # Find samples whose probabilities fall into this bin
        in_bin = (probabilities > lower_bound) & (probabilities <= upper_bound)
        
        if np.sum(in_bin) > 0:
            bin_accuracies[i] = np.mean(labels[in_bin])
            bin_confidences[i] = np.mean(probabilities[in_bin])
            bin_counts[i] = np.sum(in_bin)
    ece = 0.0
    mce = 0.0
    total_samples = len(probabilities)
    for i in range(num_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * np.abs(bin_accuracies[i] - bin_confidences[i])
            mce = max(mce, np.abs(bin_accuracies[i] - bin_confidences[i]))
            
    metrics = {'ece': ece, 'mce': mce}
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
    if len(probabilities) == 0:
        logger.warning("No probabilities provided for reliability diagram. Skipping plot.")
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
    plt.plot(bin_confidences[bin_counts > 0], bin_accuracies[bin_counts > 0], 's-', color='blue', label='Model Calibration')
    
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

# 12.1 mAP for Object Detection
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

    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 1.0 # Perfect score if no predictions and no ground truths
    if pred_boxes.numel() == 0 and gt_boxes.numel() > 0:
        return 0.0 # No predictions but ground truths exist
    if pred_boxes.numel() > 0 and gt_boxes.numel() == 0:
        return 0.0 # Predictions exist but no ground truths

    # Sort predictions by score in descending order
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]

    # Keep track of matched ground truths to avoid double counting
    matched_gts = torch.zeros(num_gts, dtype=torch.bool)
    
    true_positives = torch.zeros(num_preds, dtype=torch.bool)
    false_positives = torch.zeros(num_preds, dtype=torch.bool)

    for i in range(num_preds):
        pred_box = pred_boxes[i].unsqueeze(0) # Add batch dim for box_iou
        
        # Calculate IoU with all ground truth boxes
        ious = box_iou(pred_box, gt_boxes).squeeze(0) # Remove batch dim
        
        # Find the best matching ground truth box
        if ious.numel() > 0:
            best_iou, best_gt_idx = ious.max(dim=0)
            
            if best_iou >= iou_thresh and not matched_gts[best_gt_idx]:
                true_positives[i] = True
                matched_gts[best_gt_idx] = True
            else:
                false_positives[i] = True
        else:
            false_positives[i] = True # No ground truths to match against

    # Calculate cumulative true positives and false positives
    tp_cumsum = torch.cumsum(true_positives, dim=0)
    fp_cumsum = torch.cumsum(false_positives, dim=0)

    # Calculate precision and recall
    recall = tp_cumsum / (num_gts + 1e-6) # Add epsilon to avoid division by zero
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Compute Average Precision using the 11-point interpolation method (or all-points)
    # For simplicity, we'll use the all-points method (area under PR curve)
    # This requires interpolating precision for recall values.
    # A more robust implementation would use VOC/COCO AP calculation.
    
    # Append (0,0) and (1,0) to precision-recall curve
    precision = torch.cat((torch.tensor([1.0]), precision))
    recall = torch.cat((torch.tensor([0.0]), recall))

    # Compute the area under the curve using numerical integration (trapezoidal rule)
    average_precision = torch.trapz(precision, recall).item()

    logger.debug(f"Detection mAP (IoU={iou_thresh}): {average_precision:.4f}")
    return average_precision

# 12.2 Confusion Matrix Logging
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

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    # Convert plot to image array
    fig = plt.gcf()
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) # Close the figure to free memory
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
    # TensorBoard expects NCHW for images, so permute HWC to CHW
    writer.add_image('ConfusionMatrix', cm_image.transpose(2, 0, 1), epoch)
    logger.info(f"Confusion matrix logged for epoch {epoch}.")

