"""
Enhanced validation metrics for the Bongard hybrid generator system.
Integrated from legacy validation with improvements for current architecture.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sklearn metrics, fallback to simple implementations
try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        f1_score, precision_score, recall_score
    )
    HAS_SKLEARN = True
except ImportError:
    logger.warning("sklearn not available. Using simple metric implementations.")
    HAS_SKLEARN = False

def classification_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate classification accuracy."""
    if HAS_SKLEARN:
        return accuracy_score(y_true, y_pred)
    else:
        return float(np.mean(np.array(y_true) == np.array(y_pred)))

def rule_match_f1(gt_rules: List[str], inferred_rules: List[str]) -> float:
    """Calculate F1 score for rule matching."""
    if not gt_rules and not inferred_rules:
        return 1.0
    
    # Convert to sets for comparison
    gt_set = set(gt_rules)
    inf_set = set(inferred_rules)
    
    # Calculate intersection
    matches = len(gt_set.intersection(inf_set))
    
    # Calculate precision and recall
    precision = matches / len(inf_set) if len(inf_set) > 0 else 0.0
    recall = matches / len(gt_set) if len(gt_set) > 0 else 0.0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

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
        predicted_labels: List of predicted class labels (e.g., 0 or 1).
        true_labels: List of true class labels.
        predicted_rules: List of inferred DSL rules (optional).
        ground_truth_rules: List of ground truth DSL rules (optional).
        problem_ids: Optional list of problem identifiers for logging.
        
    Returns:
        Dictionary containing various computed metrics.
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
    if predicted_labels and true_labels and len(set(true_labels)) > 1:
        try:
            if HAS_SKLEARN:
                # Use sklearn for detailed metrics
                precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
                    true_labels, predicted_labels, average='binary', zero_division=0
                )
                
                metrics['f1_binary'] = f1_binary
                metrics['precision_binary'] = precision_binary
                metrics['recall_binary'] = recall_binary
                
                # Multi-class metrics
                f1_macro = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
                f1_micro = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
                
                metrics['f1_macro'] = f1_macro
                metrics['f1_micro'] = f1_micro
                
                logger.info(f"F1 Binary: {f1_binary:.4f}")
                logger.info(f"Precision Binary: {precision_binary:.4f}")
                logger.info(f"Recall Binary: {recall_binary:.4f}")
                logger.info(f"F1 Macro: {f1_macro:.4f}")
                logger.info(f"F1 Micro: {f1_micro:.4f}")
            else:
                # Simple binary F1 calculation
                true_pos = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
                false_pos = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)
                false_neg = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics['f1_binary'] = f1
                metrics['precision_binary'] = precision
                metrics['recall_binary'] = recall
                
                logger.info(f"F1 Binary (simple): {f1:.4f}")
                logger.info(f"Precision Binary (simple): {precision:.4f}")
                logger.info(f"Recall Binary (simple): {recall:.4f}")
                
        except Exception as e:
            logger.error(f"Error calculating detailed metrics: {e}")

    # 4. Per-problem breakdown (if problem IDs provided)
    if problem_ids and len(problem_ids) == len(predicted_labels):
        problem_metrics = []
        for i, pid in enumerate(problem_ids):
            problem_acc = 1.0 if predicted_labels[i] == true_labels[i] else 0.0
            problem_metrics.append({
                'problem_id': pid,
                'predicted': predicted_labels[i],
                'true': true_labels[i],
                'correct': problem_acc == 1.0
            })
        
        metrics['per_problem'] = problem_metrics
        correct_count = sum(1 for pm in problem_metrics if pm['correct'])
        logger.info(f"Per-problem accuracy: {correct_count}/{len(problem_metrics)} "
                   f"({correct_count/len(problem_metrics):.4f})")

    # 5. Summary statistics
    if predicted_labels and true_labels:
        metrics['total_samples'] = len(predicted_labels)
        metrics['positive_samples'] = sum(true_labels)
        metrics['predicted_positive'] = sum(predicted_labels)
        
        logger.info(f"Total samples: {len(predicted_labels)}")
        logger.info(f"Positive samples: {sum(true_labels)}")
        logger.info(f"Predicted positive: {sum(predicted_labels)}")

    return metrics

def validate_generator_quality(
    generated_scenes: List[Dict[str, Any]],
    rules: List[str],
    expected_split: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Validate the quality of generated scenes from hybrid generator.
    
    Args:
        generated_scenes: List of generated scene dictionaries
        rules: List of rule descriptions used
        expected_split: Expected distribution (e.g., {'cp': 0.7, 'ga': 0.3})
        
    Returns:
        Dictionary of quality metrics
    """
    quality_metrics = {}
    
    if not generated_scenes:
        logger.warning("No generated scenes provided for quality validation.")
        return quality_metrics
    
    # Basic counts
    total_scenes = len(generated_scenes)
    quality_metrics['total_scenes'] = total_scenes
    
    # Check for generator method distribution if available
    cp_scenes = sum(1 for scene in generated_scenes 
                   if scene.get('generator_method') == 'cp_sat')
    ga_scenes = sum(1 for scene in generated_scenes 
                   if scene.get('generator_method') == 'genetic')
    
    if cp_scenes > 0 or ga_scenes > 0:
        quality_metrics['cp_sat_scenes'] = cp_scenes
        quality_metrics['genetic_scenes'] = ga_scenes
        quality_metrics['cp_ratio'] = cp_scenes / total_scenes if total_scenes > 0 else 0
        quality_metrics['ga_ratio'] = ga_scenes / total_scenes if total_scenes > 0 else 0
        
        logger.info(f"Generator split: CP-SAT {cp_scenes}/{total_scenes} "
                   f"({cp_scenes/total_scenes:.3f}), "
                   f"Genetic {ga_scenes}/{total_scenes} ({ga_scenes/total_scenes:.3f})")
        
        # Check against expected split
        if expected_split:
            cp_expected = expected_split.get('cp', 0.5)
            ga_expected = expected_split.get('ga', 0.5)
            cp_actual = cp_scenes / total_scenes if total_scenes > 0 else 0
            ga_actual = ga_scenes / total_scenes if total_scenes > 0 else 0
            
            cp_error = abs(cp_actual - cp_expected)
            ga_error = abs(ga_actual - ga_expected)
            
            quality_metrics['cp_split_error'] = cp_error
            quality_metrics['ga_split_error'] = ga_error
            quality_metrics['split_within_tolerance'] = cp_error < 0.1 and ga_error < 0.1
    
    # Rule distribution
    rule_counts = {}
    for scene in generated_scenes:
        rule = scene.get('rule', 'unknown')
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    quality_metrics['rule_distribution'] = rule_counts
    quality_metrics['unique_rules'] = len(rule_counts)
    
    if rules:
        coverage = len(rule_counts) / len(rules) if len(rules) > 0 else 0
        quality_metrics['rule_coverage'] = coverage
        logger.info(f"Rule coverage: {len(rule_counts)}/{len(rules)} ({coverage:.3f})")
    
    # Scene diversity metrics
    unique_scenes = set()
    for scene in generated_scenes:
        # Create a simplified fingerprint for diversity check
        if 'objects' in scene:
            fingerprint = tuple(sorted([
                (obj.get('shape', ''), obj.get('color', ''), obj.get('size', 0))
                for obj in scene['objects']
            ]))
            unique_scenes.add(fingerprint)
    
    diversity = len(unique_scenes) / total_scenes if total_scenes > 0 else 0
    quality_metrics['scene_diversity'] = diversity
    quality_metrics['unique_scenes'] = len(unique_scenes)
    
    logger.info(f"Scene diversity: {len(unique_scenes)}/{total_scenes} ({diversity:.3f})")
    
    return quality_metrics

class ValidationSuite:
    """Comprehensive validation suite for the Bongard generator system."""
    
    def __init__(self):
        self.results = {}
    
    def run_all_validations(self, 
                          predicted_labels: List[int] = None,
                          true_labels: List[int] = None,
                          generated_scenes: List[Dict[str, Any]] = None,
                          rules: List[str] = None) -> Dict[str, bool]:
        """Run all available validations and return success status."""
        
        validation_results = {}
        
        # Basic validation
        validation_results['basic'] = self._basic_validation()
        
        # Classification validation
        if predicted_labels and true_labels:
            validation_results['classification'] = self._classification_validation(
                predicted_labels, true_labels
            )
        else:
            validation_results['classification'] = True  # Skip if no data
        
        # Generator quality validation
        if generated_scenes:
            validation_results['generator_quality'] = self._generator_quality_validation(
                generated_scenes, rules
            )
        else:
            validation_results['generator_quality'] = True  # Skip if no data
        
        # Hybrid validation
        validation_results['hybrid'] = self._hybrid_validation(generated_scenes)
        
        self.results = validation_results
        return validation_results
    
    def _basic_validation(self) -> bool:
        """Basic system validation."""
        try:
            # Test imports
            from .hybrid_sampler import HybridSampler
            from .rule_loader import get_all_rules
            from .config_loader import get_sampler_config
            return True
        except Exception as e:
            logger.error(f"Basic validation failed: {e}")
            return False
    
    def _classification_validation(self, predicted: List[int], true: List[int]) -> bool:
        """Validate classification performance."""
        try:
            accuracy = classification_accuracy(true, predicted)
            return accuracy > 0.5  # Basic threshold
        except Exception as e:
            logger.error(f"Classification validation failed: {e}")
            return False
    
    def _generator_quality_validation(self, scenes: List[Dict], rules: List[str]) -> bool:
        """Validate generator quality."""
        try:
            metrics = validate_generator_quality(scenes, rules)
            diversity = metrics.get('scene_diversity', 0)
            coverage = metrics.get('rule_coverage', 0)
            return diversity > 0.1 and coverage > 0.1  # Basic thresholds
        except Exception as e:
            logger.error(f"Generator quality validation failed: {e}")
            return False
    
    def _hybrid_validation(self, scenes: List[Dict]) -> bool:
        """Validate hybrid generator functionality."""
        try:
            if not scenes:
                return True  # Skip if no scenes
            
            # Check if we have both CP-SAT and genetic scenes
            cp_count = sum(1 for s in scenes if s.get('generator_method') == 'cp_sat')
            ga_count = sum(1 for s in scenes if s.get('generator_method') == 'genetic')
            
            # Either both methods used or method not tracked (both OK)
            return (cp_count > 0 and ga_count > 0) or (cp_count == 0 and ga_count == 0)
        except Exception as e:
            logger.error(f"Hybrid validation failed: {e}")
            return False
    
    def print_validation_report(self):
        """Print a comprehensive validation report."""
        if not self.results:
            print("No validation results available. Run run_all_validations() first.")
            return
        
        print("=" * 60)
        print("BONGARD GENERATOR VALIDATION REPORT")
        print("=" * 60)
        
        all_passed = True
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.upper():<20} {status}")
            if not result:
                all_passed = False
        
        print("-" * 60)
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "⚠ SOME TESTS FAILED"
        print(f"OVERALL STATUS: {overall_status}")
        print("=" * 60)
