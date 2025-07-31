import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import pickle
from pathlib import Path

class AdaptivePredicateThresholds:
    """
    Dynamic learning system for predicate thresholds based on data distributions
    and batch profiling with exponential moving averages and confidence intervals.
    """
    def __init__(self, 
                 history_size: int = 1000,
                 confidence_level: float = 0.95,
                 adaptation_rate: float = 0.1,
                 min_samples: int = 50):
        self.history_size = history_size
        self.confidence_level = confidence_level
        self.adaptation_rate = adaptation_rate
        self.min_samples = min_samples
        self.predicate_data = defaultdict(lambda: {
            'values': deque(maxlen=history_size),
            'success_rates': deque(maxlen=history_size),
            'thresholds': {'current': None, 'optimal': None},
            'statistics': {'mean': 0, 'std': 0, 'confidence_interval': (0, 1)}
        })
        self.performance_history = defaultdict(list)
        self.batch_stats = []
        logging.info("AdaptivePredicateThresholds initialized")
    def update_predicate_data(self, 
                            predicate: str, 
                            value: float, 
                            success: bool,
                            image_type: str = "default") -> None:
        data = self.predicate_data[predicate]
        data['values'].append(value)
        data['success_rates'].append(1.0 if success else 0.0)
        if len(data['values']) >= self.min_samples:
            self._update_statistics(predicate)
            self._update_optimal_threshold(predicate)
        self.performance_history[f"{predicate}_{image_type}"].append({
            'value': value,
            'success': success,
            'timestamp': len(self.batch_stats)
        })
    def _update_statistics(self, predicate: str) -> None:
        data = self.predicate_data[predicate]
        values = np.array(data['values'])
        mean = float(np.mean(values))
        std = float(np.std(values))
        from scipy import stats
        confidence_interval = stats.t.interval(
            self.confidence_level,
            len(values) - 1,
            loc=mean,
            scale=stats.sem(values)
        )
        if data['statistics']['mean'] != 0:
            data['statistics']['mean'] = (
                (1 - self.adaptation_rate) * data['statistics']['mean'] +
                self.adaptation_rate * mean
            )
            data['statistics']['std'] = (
                (1 - self.adaptation_rate) * data['statistics']['std'] +
                self.adaptation_rate * std
            )
        else:
            data['statistics']['mean'] = mean
            data['statistics']['std'] = std
        data['statistics']['confidence_interval'] = confidence_interval
    def _update_optimal_threshold(self, predicate: str) -> None:
        data = self.predicate_data[predicate]
        values = np.array(data['values'])
        success_labels = np.array(data['success_rates'])
        if len(np.unique(success_labels)) < 2:
            return
        optimal = float(np.percentile(values, 50))
        data['thresholds']['optimal'] = optimal
        if data['thresholds']['current'] is not None:
            data['thresholds']['current'] = (
                (1 - self.adaptation_rate) * data['thresholds']['current'] +
                self.adaptation_rate * optimal
            )
        else:
            data['thresholds']['current'] = optimal
    def get_dynamic_threshold(self, predicate: str, default: float = 0.5, image_type: str = "default") -> float:
        data = self.predicate_data[predicate]
        if (data['thresholds']['current'] is not None and 
            len(data['values']) >= self.min_samples):
            return float(data['thresholds']['current'])
        type_specific_key = f"{predicate}_{image_type}"
        if type_specific_key in self.performance_history:
            type_data = self.performance_history[type_specific_key]
            if len(type_data) >= self.min_samples:
                values = [d['value'] for d in type_data]
                return float(np.percentile(values, 50))
        return default
    def get_orientation_tolerances(self, image_characteristics: Dict) -> Dict[str, float]:
        base_tolerances = {
            'parallel': 7.0,
            'perpendicular': 7.0,
            'alignment': 3.0
        }
        edge_density = image_characteristics.get('edge_density', 0.1)
        shape_complexity = image_characteristics.get('shape_complexity', 0.5)
        noise_level = image_characteristics.get('noise_level', 0.1)
        complexity_factor = 1.0 + (shape_complexity * 0.5)
        noise_factor = 1.0 + (noise_level * 0.3)
        edge_factor = 0.8 + (edge_density * 0.4)
        adapted_tolerances = {}
        for key, base_tol in base_tolerances.items():
            adapted_tol = base_tol * complexity_factor * noise_factor * edge_factor
            learned_tol = self.get_dynamic_threshold(f"orientation_{key}", adapted_tol)
            adapted_tolerances[key] = max(1.0, min(15.0, learned_tol))
        return adapted_tolerances
    def save_learned_thresholds(self, filepath: str) -> None:
        serializable_data = {}
        for predicate, data in self.predicate_data.items():
            serializable_data[predicate] = {
                'thresholds': data['thresholds'],
                'statistics': data['statistics'],
                'sample_count': len(data['values'])
            }
        save_data = {
            'predicate_data': serializable_data,
            'performance_history': dict(self.performance_history),
            'config': {
                'history_size': self.history_size,
                'confidence_level': self.confidence_level,
                'adaptation_rate': self.adaptation_rate,
                'min_samples': self.min_samples
            }
        }
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        logging.info(f"Saved learned thresholds to {filepath}")
    def load_learned_thresholds(self, filepath: str) -> None:
        if not Path(filepath).exists():
            logging.warning(f"Threshold file not found: {filepath}")
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            for predicate, pred_data in data['predicate_data'].items():
                self.predicate_data[predicate]['thresholds'] = pred_data['thresholds']
                self.predicate_data[predicate]['statistics'] = pred_data['statistics']
            self.performance_history = defaultdict(list, data.get('performance_history', {}))
            logging.info(f"Loaded learned thresholds from {filepath}")
        except Exception as e:
            logging.error(f"Failed to load thresholds from {filepath}: {e}")
def create_adaptive_predicate_functions(adaptive_thresholds: AdaptivePredicateThresholds,
                                      image_characteristics: Dict) -> Dict:
    tolerances = adaptive_thresholds.get_orientation_tolerances(image_characteristics)
    def adaptive_parallel(a, b):
        tol = tolerances['parallel']
        return abs(a.get('orientation', 0) - b.get('orientation', 0)) < tol or \
               abs(abs(a.get('orientation', 0) - b.get('orientation', 0)) - 180) < tol
    def adaptive_perpendicular(a, b):
        tol = tolerances['perpendicular']
        return abs(abs(a.get('orientation', 0) - b.get('orientation', 0)) - 90) < tol
    def adaptive_proximal(a, b):
        distance = np.linalg.norm(
            np.array([a.get('cx', 0), a.get('cy', 0)]) - 
            np.array([b.get('cx', 0), b.get('cy', 0)])
        )
        threshold = adaptive_thresholds.get_dynamic_threshold('proximal_distance', 50.0)
        return distance < threshold
    return {
        'parallel_to': adaptive_parallel,
        'perpendicular_to': adaptive_perpendicular,
        'proximal_to': adaptive_proximal,
    }
