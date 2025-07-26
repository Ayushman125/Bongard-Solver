import torch
from sklearn.model_selection import StratifiedKFold
try:
    import optuna
except ImportError:
    optuna = None
import numpy as np
import time

from .utils import MaskType, classify_mask

class AutomatedBongardOptimizer:
    """
    Automated parameter optimizer for Bongard image augmentation pipelines.
    Uses Bayesian optimization and cross-validation to calibrate augmentation and QA parameters before data generation.
    """
    def __init__(self, augmentor, n_trials=50, n_folds=5, random_state=42):
        self.augmentor = augmentor
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state
        self.optimal_config = None

    def _generate_search_space(self, validation_samples):
        # Analyze mask density, variance, and robust statistics to set parameter bounds
        mask_densities = [mask.sum().item() / mask.numel() for mask in validation_samples['masks']]
        pixvars = [np.var(img.cpu().numpy()) for img in validation_samples['images']]
        outlier_stats = [np.std(mask.cpu().numpy()) for mask in validation_samples['masks']]
        search_space = {
            'morphology_range': (max(1, int(np.percentile(mask_densities, 10)*512)), min(64, int(np.percentile(mask_densities, 90)*512))),
            'pixvar_min': (max(0.0005, float(np.percentile(pixvars, 5))), min(0.01, float(np.percentile(pixvars, 95)))),
            'outlier_sigma': (1.5, 4.0),
            'edge_overlap_min': (0.01, 0.15)
        }
        return search_space

    def _objective(self, trial, validation_samples):
        # Sample parameters
        morph_size = trial.suggest_int('morph_size', *self.search_space['morphology_range'])
        pixvar_min = trial.suggest_float('pixvar_min', *self.search_space['pixvar_min'])
        outlier_sigma = trial.suggest_float('outlier_sigma', *self.search_space['outlier_sigma'])
        edge_overlap_min = trial.suggest_float('edge_overlap_min', *self.search_space['edge_overlap_min'])
        # Set parameters in augmentor
        self.augmentor.QA_THRESHOLDS['pixvar_min'] = pixvar_min
        self.augmentor.QA_THRESHOLDS['outlier_sigma'] = outlier_sigma
        self.augmentor.QA_THRESHOLDS['edge_overlap_min'] = edge_overlap_min
        # Use stratified k-fold cross-validation
        masks = validation_samples['masks']
        images = validation_samples['images']
        mask_types = [classify_mask(mask) for mask in masks]
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        qa_scores = []
        speed_scores = []
        cost_scores = []
        for train_idx, test_idx in skf.split(images, mask_types):
            imgs_fold = [images[i] for i in test_idx]
            masks_fold = [masks[i] for i in test_idx]
            # Simulate augmentation and QA
            t0 = time.time()
            results = self.augmentor.augment_batch(
                torch.stack(imgs_fold),
                geometries=None,
                batch_idx=0
            )
            t1 = time.time()
            # QA pass rate
            qa_pass = self.augmentor.metric_tracker.get('qa_pass_rate', 0.0)
            qa_scores.append(qa_pass)
            # Speed (images/sec)
            speed = len(imgs_fold) / (t1-t0+1e-6)
            speed_scores.append(speed)
            # Resource cost (memory, time)
            cost_scores.append((t1-t0))
        # Multi-objective weighted score
        quality = np.mean(qa_scores)
        speed = np.mean(speed_scores)
        cost = np.mean(cost_scores)
        score = 0.4*quality + 0.3*speed/100 + 0.3*(1.0/(cost+1e-6))
        trial.set_user_attr('qa_scores', qa_scores)
        trial.set_user_attr('speed_scores', speed_scores)
        trial.set_user_attr('cost_scores', cost_scores)
        return score

    def auto_optimize_before_generation(self, validation_samples):
        self.search_space = self._generate_search_space(validation_samples)
        if optuna is None:
            raise ImportError("Optuna is required for Bayesian optimization. Please install optuna.")
        
        study = optuna.create_study(direction='maximize')
        skf_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        mask_types = [classify_mask(m) for m in validation_samples['masks']]
        
        def nested_objective(trial):
            # Suggest parameters
            params = {
                'morph_size': trial.suggest_int('morph_size', *self.search_space['morphology_range']),
                'pixvar_min': trial.suggest_float('pixvar_min', *self.search_space['pixvar_min']),
                'edge_overlap_min': trial.suggest_float('edge_overlap_min', *self.search_space['edge_overlap_min'])
            }
            self.augmentor.QA_THRESHOLDS.update(params)
            scores = []
            for train_idx, test_idx in skf_outer.split(validation_samples['images'], mask_types):
                imgs = torch.stack([validation_samples['images'][i] for i in test_idx])
                geoms = [validation_samples['geometries'][i] for i in test_idx] if 'geometries' in validation_samples else None
                result = self.augmentor.augment_batch(imgs, geometries=geoms, batch_idx=0)
                fail_rate = self.augmentor.metrics.store.get('fail_rate', [0.0])[-1]
                scores.append(1.0 - fail_rate)
            return float(np.mean(scores))
        
        study.optimize(nested_objective, n_trials=self.n_trials)
        self.optimal_config = study.best_params
        self.augmentor.QA_THRESHOLDS.update(self.optimal_config)
        return self.optimal_config

    def generate_optimized_data(self, full_dataset, batch_size=32):
        # Run data generation with optimized parameters
        results = []
        for i in range(0, len(full_dataset['images']), batch_size):
            batch_imgs = full_dataset['images'][i:i+batch_size]
            batch_geoms = full_dataset.get('geometries', None)
            batch_result = self.augmentor.augment_batch(
                torch.stack(batch_imgs),
                geometries=batch_geoms,
                batch_idx=i//batch_size
            )
            results.append(batch_result)
        return results

    def select_best_outlier_method(self, validation_samples):
        from scipy.stats import iqr
        from sklearn.ensemble import IsolationForest
        # Try MAD, IQR, Isolation Forest and select best for outlier detection
        mask_stats = [mask.sum().item() for mask in validation_samples['masks']]
        mad_score = np.median(np.abs(mask_stats - np.median(mask_stats)))
        iqr_score = iqr(mask_stats)
        iso_forest = IsolationForest(random_state=self.random_state)
        iso_labels = iso_forest.fit_predict(np.array(mask_stats).reshape(-1,1))
        iso_score = np.mean(iso_labels == 1)
        # Select method with lowest false positive rate
        scores = {'MAD': mad_score, 'IQR': iqr_score, 'IsolationForest': iso_score}
        best_method = min(scores, key=scores.get)
        self.augmentor.outlier_method = best_method
        return best_method

class QAParameterOptimizer:
    def __init__(self, augmentor, validation_data, n_trials: int = 50):
        self.augmentor = augmentor
        self.validation_data = validation_data
        self.n_trials = n_trials

    def _objective(self, trial):
        # Sample thresholds
        pixvar_min = trial.suggest_float('pixvar_min', 0.0005, 0.02)
        edge_min   = trial.suggest_float('edge_overlap_min', 0.01, 0.2)
        self.augmentor.QA_THRESHOLDS.update({
            'pixvar_min': pixvar_min,
            'edge_overlap_min': edge_min
        })
        # Evaluate QA pass rate on a small validation batch
        result = self.augmentor.augment_batch(
            self.validation_data['images'],
            geometries=self.validation_data['geometries'],
            batch_idx=0
        )
        # Assume augmentor.metrics stores last fail_rate
        fail_rate = self.augmentor.metrics.store.get('fail_rate', [0.0])[-1]
        return 1.0 - fail_rate  # maximize pass rate

    def optimize(self):
        if optuna is None:
            print("[WARNING] Optuna not available for QA parameter optimization")
            return {}
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.n_trials)
        best = study.best_params
        self.augmentor.QA_THRESHOLDS.update(best)
        return best
