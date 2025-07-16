# Folder: bongard_solver/src/
# File: validate.py

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Import metrics from core_models
try:
    from core_models.metrics import classification_accuracy, rule_match_f1, calculate_f1_scores
    HAS_METRICS = True
except ImportError:
    logging.warning("core_models.metrics not found. Using dummy metric functions.")
    HAS_METRICS = False
    def classification_accuracy(y_true: List[int], y_pred: List[int]) -> float:
        logger.warning("Using dummy classification_accuracy.")
        return np.mean(np.array(y_true) == np.array(y_pred))
    def rule_match_f1(gt_rules: List[str], inferred_rules: List[str]) -> float:
        logger.warning("Using dummy rule_match_f1.")
        # Simple dummy F1 for demonstration
        matches = sum(1 for r_gt in gt_rules for r_inf in inferred_rules if r_gt == r_inf)
        if len(gt_rules) == 0 and len(inferred_rules) == 0: return 1.0
        precision = matches / len(inferred_rules) if len(inferred_rules) > 0 else 0.0
        recall = matches / len(gt_rules) if len(gt_rules) > 0 else 0.0
        if precision + recall == 0: return 0.0
        return 2 * (precision * recall) / (precision + recall)
    def calculate_f1_scores(y_true, y_pred, labels=None, average='binary'):
        logger.warning("Using dummy calculate_f1_scores.")
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, labels=labels, average=average, zero_division=0)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_validation(
    predicted_labels: List[int],
    true_labels: List[int],
    predicted_rules: Optional[List[str]] = None,
    ground_truth_rules: Optional[List[str]] = None,
    problem_ids: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Runs a comprehensive validation process, calculating various metrics.
    Args:
        predicted_labels (List[int]): List of predicted class labels (e.g., 0 or 1).
        true_labels (List[int]): List of true class labels.
        predicted_rules (Optional[List[str]]): List of inferred DSL rules.
        ground_truth_rules (Optional[List[str]]): List of ground truth DSL rules.
        problem_ids (Optional[List[Any]]): Optional list of problem identifiers for logging.
    Returns:
        Dict[str, Any]: A dictionary containing various computed metrics.
    """
    metrics = {}

    # 1. Classification Accuracy
    if predicted_labels and true_labels:
        acc = classification_accuracy(true_labels, predicted_labels)
        metrics['classification_accuracy'] = acc
        logger.info(f"Classification Accuracy: {acc:.4f}")
    else:
        logger.warning("Skipping classification accuracy: Labels not provided.")

    # 2. Rule Match F1 Score (if rules are provided)
    if predicted_rules is not None and ground_truth_rules is not None:
        f1 = rule_match_f1(ground_truth_rules, predicted_rules)
        metrics['rule_match_f1'] = f1
        logger.info(f"Rule Match F1 Score: {f1:.4f}")
    else:
        logger.warning("Skipping rule match F1 score: Rules not provided.")

    # 3. Detailed F1, Precision, Recall (for classification)
    if predicted_labels and true_labels and HAS_METRICS:
        try:
            # Assuming binary classification (0, 1)
            f1_binary = calculate_f1_scores(true_labels, predicted_labels, average='binary')
            precision_binary, recall_binary, _, _ = calculate_f1_scores(true_labels, predicted_labels, average='binary', output_dict=True) # output_dict for precision/recall
            
            metrics['f1_binary'] = f1_binary
            metrics['precision_binary'] = precision_binary
            metrics['recall_binary'] = recall_binary

            logger.info(f"Binary F1 Score: {f1_binary:.4f}")
            logger.info(f"Binary Precision: {precision_binary:.4f}")
            logger.info(f"Binary Recall: {recall_binary:.4f}")

            # You can also compute per-class F1 if needed
            # f1_per_class = calculate_f1_scores(true_labels, predicted_labels, average=None)
            # metrics['f1_per_class'] = f1_per_class.tolist()

        except Exception as e:
            logger.error(f"Error calculating detailed classification metrics: {e}", exc_info=True)

    # 4. Misclassified Problems (for debugging/analysis)
    if predicted_labels and true_labels and problem_ids:
        misclassified_problems = []
        for i, (pred, true) in enumerate(zip(predicted_labels, true_labels)):
            if pred != true:
                misclassified_problems.append({
                    'id': problem_ids[i],
                    'predicted': pred,
                    'true': true,
                    'gt_rule': ground_truth_rules[i] if ground_truth_rules else 'N/A',
                    'inferred_rule': predicted_rules[i] if predicted_rules else 'N/A'
                })
        metrics['misclassified_problems'] = misclassified_problems
        logger.info(f"Found {len(misclassified_problems)} misclassified problems.")

    return metrics

if __name__ == '__main__':
    logger.info("Running validate.py example.")

    # Example 1: Basic Classification
    true_labels_cls = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    predicted_labels_cls = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    problem_ids_cls = [f"P{i:03d}" for i in range(len(true_labels_cls))]

    logger.info("\n--- Classification Validation Example ---")
    results_cls = run_validation(predicted_labels_cls, true_labels_cls, problem_ids=problem_ids_cls)
    logger.info(f"Classification Results: {results_cls}")

    # Example 2: Rule Matching
    ground_truth_rules_rule = ["SHAPE(TRIANGLE)", "COLOR(RED)", "CONTAINS(CIRCLE,SQUARE)"]
    inferred_rules_rule = ["SHAPE(TRIANGLE)", "COLOR(BLUE)", "CONTAINS(CIRCLE,SQUARE)", "SIZE(SMALL)"] # One correct, one incorrect, one extra

    logger.info("\n--- Rule Matching Validation Example ---")
    results_rule = run_validation(
        predicted_labels=[], true_labels=[], # No classification labels for this example
        predicted_rules=inferred_rules_rule,
        ground_truth_rules=ground_truth_rules_rule
    )
    logger.info(f"Rule Matching Results: {results_rule}")

    # Example 3: Combined
    logger.info("\n--- Combined Validation Example ---")
    results_combined = run_validation(
        predicted_labels=[1, 0, 1, 1, 0],
        true_labels=[1, 0, 0, 1, 1],
        predicted_rules=["SHAPE(CIRCLE)", "COLOR(RED)", "SIZE(SMALL)", "SHAPE(SQUARE)", "FILL(SOLID)"],
        ground_truth_rules=["SHAPE(CIRCLE)", "COLOR(BLUE)", "SIZE(LARGE)", "SHAPE(SQUARE)", "FILL(OUTLINED)"],
        problem_ids=["Prob_A", "Prob_B", "Prob_C", "Prob_D", "Prob_E"]
    )
    logger.info(f"Combined Results: {results_combined}")

