import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from collections import defaultdict, deque
import logging
from datetime import datetime
import json
from pathlib import Path

class ConceptDriftVisualizer:
    def __init__(self,
                 window_size: int = 100,
                 overlap_ratio: float = 0.5,
                 drift_threshold: float = 0.1,
                 visualization_dir: str = "visualizations/drift"):
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.drift_threshold = drift_threshold
        self.viz_dir = Path(visualization_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_history = defaultdict(deque)
        self.performance_history = defaultdict(deque)
        self.drift_points = defaultdict(list)
        plt.style.use('seaborn-v0_8')
        self.color_palette = sns.color_palette("husl", 10)
        logging.info(f"ConceptDriftVisualizer initialized with window_size={window_size}")
    def add_threshold_data(self,
                          predicate: str,
                          threshold_value: float,
                          performance_score: float,
                          timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now()
        self.threshold_history[predicate].append({
            'timestamp': timestamp,
            'threshold': threshold_value,
            'performance': performance_score
        })
        max_history = self.window_size * 10
        if len(self.threshold_history[predicate]) > max_history:
            self.threshold_history[predicate].popleft()
    def detect_drift(self, predicate: str) -> List[Dict[str, Any]]:
        history = list(self.threshold_history[predicate])
        if len(history) < self.window_size * 2:
            return []
        drift_points = []
        for i in range(self.window_size, len(history) - self.window_size, int(self.window_size * self.overlap_ratio)):
            window1 = [h['threshold'] for h in history[i - self.window_size:i]]
            window2 = [h['threshold'] for h in history[i:i + self.window_size]]
            drift_detected, drift_metrics = self._statistical_drift_test(window1, window2)
            if drift_detected:
                drift_points.append({
                    'timestamp': history[i]['timestamp'],
                    'drift_type': drift_metrics['type'],
                    'magnitude': drift_metrics['magnitude'],
                    'confidence': drift_metrics['confidence']
                })
        self.drift_points[predicate] = drift_points
        return drift_points
    def _statistical_drift_test(self, window1, window2) -> Tuple[bool, Dict[str, Any]]:
        from scipy import stats
        mean1, std1 = np.mean(window1), np.std(window1)
        mean2, std2 = np.mean(window2), np.std(window2)
        pooled_std = np.sqrt(((len(window1) - 1) * std1**2 + (len(window2) - 1) * std2**2) / 
                            (len(window1) + len(window2) - 2))
        cohens_d = abs(mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        t_stat, p_value = stats.ttest_ind(window1, window2)
        ks_stat, ks_p_value = stats.ks_2samp(window1, window2)
        drift_detected = (
            cohens_d > self.drift_threshold and  # Effect size threshold
            p_value < 0.05 and ks_p_value < 0.05
        )
        if drift_detected:
            if mean2 > mean1:
                drift_type = "increasing"
            else:
                drift_type = "decreasing"
        else:
            drift_type = "stable"
        drift_metrics = {
            'magnitude': cohens_d,
            'type': drift_type,
            'confidence': 1 - min(p_value, ks_p_value),
            'mean_before': mean1,
            'mean_after': mean2,
            'std_before': std1,
            'std_after': std2,
            't_statistic': t_stat,
            'p_value': p_value,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value
        }
        return drift_detected, drift_metrics
    def visualize_threshold_evolution(self,
                                    predicates: List[str],
                                    save_path: Optional[str] = None) -> str:
        n_predicates = len(predicates)
        fig, axes = plt.subplots(n_predicates, 2, figsize=(15, 4 * n_predicates))
        if n_predicates == 1:
            axes = axes.reshape(1, -1)
        for i, predicate in enumerate(predicates):
            history = list(self.threshold_history[predicate])
            if not history:
                continue
            timestamps = [point['timestamp'] for point in history]
            thresholds = [point['threshold'] for point in history]
            performances = [point['performance'] for point in history]
            ax1 = axes[i, 0]
            ax1.plot(timestamps, thresholds, 'b-', linewidth=2, alpha=0.7, label='Threshold')
            drift_points = self.drift_points.get(predicate, [])
            for drift in drift_points:
                ax1.axvline(x=drift['timestamp'], color='red', linestyle='--', alpha=0.7)
                ax1.text(drift['timestamp'], max(thresholds) * 0.9, 
                        f"Drift\n({drift['drift_type']})", 
                        rotation=90, ha='right', va='top', fontsize=8)
            ax1.set_title(f'{predicate}: Threshold Evolution')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Threshold Value')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax2 = axes[i, 1]
            scatter = ax2.scatter(thresholds, performances, 
                                c=range(len(thresholds)), 
                                cmap='viridis', alpha=0.6)
            if len(thresholds) > 1:
                z = np.polyfit(thresholds, performances, 1)
                p = np.poly1d(z)
                ax2.plot(sorted(thresholds), p(sorted(thresholds)), 
                        "r--", alpha=0.8, linewidth=2)
            ax2.set_title(f'{predicate}: Performance vs Threshold')
            ax2.set_xlabel('Threshold Value')
            ax2.set_ylabel('Performance Score')
            ax2.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Time Progression')
        plt.tight_layout()
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.viz_dir / f"threshold_evolution_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Threshold evolution visualization saved to: {save_path}")
        return str(save_path)
    def generate_drift_report(self, predicates: List[str], save_path: Optional[str] = None) -> str:
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': {
                'window_size': self.window_size,
                'overlap_ratio': self.overlap_ratio,
                'drift_threshold': self.drift_threshold
            },
            'predicates': {}
        }
        for predicate in predicates:
            drift_points = self.detect_drift(predicate)
            history = list(self.threshold_history[predicate])
            if history:
                thresholds = [point['threshold'] for point in history]
                performances = [point['performance'] for point in history]
                predicate_report = {
                    'total_data_points': len(history),
                    'drift_points_detected': len(drift_points),
                    'drift_rate': len(drift_points) / max(len(history) / self.window_size, 1),
                    'threshold_statistics': {
                        'mean': float(np.mean(thresholds)),
                        'std': float(np.std(thresholds)),
                        'min': float(np.min(thresholds)),
                        'max': float(np.max(thresholds)),
                        'range': float(np.max(thresholds) - np.min(thresholds))
                    },
                    'performance_statistics': {
                        'mean': float(np.mean(performances)),
                        'std': float(np.std(performances)),
                        'correlation_with_threshold': float(np.corrcoef(thresholds, performances)[0, 1]) if len(thresholds) > 1 else 0.0
                    },
                    'drift_points': [
                        {
                            'timestamp': drift['timestamp'].isoformat(),
                            'drift_type': drift['drift_type'],
                            'magnitude': drift['magnitude'],
                            'confidence': drift['confidence']
                        }
                        for drift in drift_points
                    ]
                }
                report['predicates'][predicate] = predicate_report
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.viz_dir / f"drift_report_{timestamp}.json"
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logging.info(f"Drift analysis report saved to: {save_path}")
        return str(save_path)
