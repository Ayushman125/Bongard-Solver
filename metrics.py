# Folder: bongard_solver/

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, brier_score_loss,
    confusion_matrix
)
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)

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
        return 0.5 # Default for random classifier

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

