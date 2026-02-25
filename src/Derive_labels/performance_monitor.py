import time
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.history = []

    def log(self, problem_id, concepts, metrics):
        self.history.append({
            'problem_id': problem_id,
            'concepts': concepts,
            'metrics': metrics,
            'timestamp': time.time()
        })

    def report(self, last_n=100):
        recent = self.history[-last_n:]
        avg_metrics = {}
        for metric_name in ['consistency', 'diversity', 'context_dependence', 'analogy_quality']:
            values = [h['metrics'].get(metric_name, 0) for h in recent]
            avg_metrics[metric_name] = np.mean(values) if values else 0
        return {
            'average_metrics': avg_metrics,
            'total_problems_processed': len(self.history),
            'performance_trend': avg_metrics
        }

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
