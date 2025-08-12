"""
quality_monitor.py
Centralized quality scoring, error tracking, and statistics for Bongard Solver.
"""
import logging

class QualityMonitor:
    def __init__(self):
        self.scores = []
        self.errors = []
        self.stats = {}

    def log_quality(self, feature_name, quality_info):
        self.scores.append((feature_name, quality_info))
        if quality_info.get('degenerate', False):
            self.errors.append((feature_name, quality_info))
            if quality_info.get('zero_area', False):
                logging.warning(f"[QualityMonitor] Zero-area detected for {feature_name}: {quality_info}")
            if quality_info.get('qhull_failure', False):
                logging.error(f"[QualityMonitor] QHull failure for {feature_name}: {quality_info}")
        logging.info(f"[QualityMonitor] {feature_name}: {quality_info}")

    def get_stats(self):
        self.stats = {
            'total': len(self.scores),
            'degenerate': sum(1 for _, q in self.scores if q.get('degenerate', False)),
            'average_quality': sum(q.get('quality', 0.0) for _, q in self.scores) / max(1, len(self.scores))
        }
        return self.stats

quality_monitor = QualityMonitor()
