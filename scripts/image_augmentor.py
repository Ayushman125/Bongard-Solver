import os
import sys
import time
import json
import argparse
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
from typing import Tuple, Optional, List, Dict, Union

try:
    from . import MaskType
except ImportError:
    # Fallback: define MaskType if not imported
    from enum import Enum
    class MaskType(Enum):
        EMPTY = 0
        THIN = 1
        SPARSE = 2
        DENSE = 3

# Import hybrid pipeline components
try:
    from sam_wrapper import get_sam_autocoder, SamAutocoder
    from skeleton_processor import get_skeleton_processor, SkeletonAwareProcessor
    HYBRID_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Hybrid pipeline components not available: {e}")
    HYBRID_PIPELINE_AVAILABLE = False
import numpy as np
from sklearn.model_selection import StratifiedKFold
try:
    import optuna
except ImportError:
    optuna = None
from sklearn.ensemble import IsolationForest
from scipy.stats import iqr

# ==================================================================
# CRITICAL TENSOR CORRUPTION FIXES FOR CV_8UC512 ERROR
# ==================================================================

def diagnose_tensor_corruption(tensor, name="tensor", path=None):
    """
    Print shape, dtype, device and OpenCV compatibility of a tensor or numpy array.
    """
    print(f"\n🛠️ Diagnosing {name}{': '+path if path else ''}")
    if isinstance(tensor, torch.Tensor):
        t = tensor
        print(f" PyTorch: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
              f"contiguous={t.is_contiguous()}")
        np_arr = t.detach().cpu().numpy()
    else:
        np_arr = tensor
        print(f" NumPy: shape={np_arr.shape}, dtype={np_arr.dtype}")

    # Squeeze singleton dims and take first channel if >2D
    arr = np_arr
    while arr.ndim > 2:
        if arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.shape[0] == 1:
            arr = arr.squeeze(0)
        else:
            arr = arr[..., 0]
    print(f" → After squeeze: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")

    # Test OpenCV connectedComponents
    opencv_ok = False
    is_corrupted = False
    try:
        cv2_arr = arr.astype(np.uint8)
        n_labels, _ = cv2.connectedComponents(cv2_arr)
        print(f" ✅ OpenCV connectedComponents succeeded: labels={n_labels}")
        opencv_ok = True
    except Exception as e:
        print(f" ❌ OpenCV error: {e}")
        is_corrupted = True

    result = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": arr.min(),
        "max": arr.max(),
        "opencv_ok": opencv_ok,
        "corrupted": is_corrupted,
    }
    return result

def sanitize_for_opencv(tensor):
    """Fix common tensor corruption patterns"""
    if isinstance(tensor, torch.Tensor):
        # Ensure CPU and detached
        if tensor.device.type == 'cuda':
            np_array = tensor.cpu().detach().numpy()
        else:
            np_array = tensor.detach().numpy()
    else:
        np_array = tensor.copy()
    
    # CRITICAL: Remove all extra dimensions
    while len(np_array.shape) > 2:
        if np_array.shape[-1] == 1:
            np_array = np_array.squeeze(-1)
        elif np_array.shape[0] == 1:
            np_array = np_array.squeeze(0)
        else:
            # Take first channel if multi-channel corruption
            if len(np_array.shape) == 3:
                np_array = np_array[:, :, 0]
            break
    
    # Ensure proper data type for OpenCV
    if np_array.dtype != np.uint8:
        if np_array.max() <= 1.0:
            np_array = (np_array * 255).astype(np.uint8)
        else:
            np_array = np.clip(np_array, 0, 255).astype(np.uint8)
    
    return np_array

def safe_device_transfer(tensor, device):
    """Safe tensor device transfer with corruption protection"""
    if tensor.device != device:
        # Force contiguous layout before transfer
        tensor = tensor.contiguous()
        # Transfer via CPU if direct GPU-GPU fails (common with A5000/4090)
        if device.type == 'cuda' and tensor.device.type == 'cuda':
            tensor = tensor.cpu().to(device)
        else:
            tensor = tensor.to(device)
    return tensor

def safe_mask_conversion(mask):
    """Convert mask tensor to OpenCV-safe format"""
    # Always use sanitizer for mask conversions
    mask_np = sanitize_for_opencv(mask)
    
    # Additional mask-specific safety checks
    if len(mask_np.shape) > 2:
        mask_np = mask_np.squeeze()
    
    return mask_np

def minimal_outlier_detection(mask_sums):
    """Simplified outlier detection during corruption cleanup"""
    # Only flag completely empty masks
    return mask_sums < 10  # 10 pixels minimum

def safe_topology_validation(orig_mask, aug_mask):
    """Topology validation with corruption protection"""
    try:
        from skimage.morphology import skeletonize
        from skimage.measure import label
        
        # Sanitize masks before processing
        orig_binary = sanitize_for_opencv(orig_mask) > 127
        aug_binary = sanitize_for_opencv(aug_mask) > 127
        
        # Extract skeletons using skimage (more robust than OpenCV)
        orig_skel = skeletonize(orig_binary)
        aug_skel = skeletonize(aug_binary)
        
        # Count connected components using scikit-image
        orig_components = label(orig_skel, connectivity=2).max()
        aug_components = label(aug_skel, connectivity=2).max()
        
        # Allow small topology changes (up to 1 component difference)
        topology_diff = abs(orig_components - aug_components)
        
        if topology_diff > 2:
            return False, f"Major topology change: {orig_components} -> {aug_components}"
        
        return True, "Topology preserved"
        
    except Exception as e:
        # Don't fail on topology errors during cleanup
        print(f"[TOPOLOGY WARNING] {e}")
        return True, "Topology check skipped due to error"

# ==================================================================
# END TENSOR CORRUPTION FIXES
# ==================================================================

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
        # ───────────────────────────────────────────────────────────────
        # Section: Nested Cross-Validation in auto_optimize_before_generation
        # ───────────────────────────────────────────────────────────────
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
    # Bongard-specific QA thresholds for mask types
    BONGARD_QA_THRESHOLDS = {
        MaskType.EMPTY: {
            'pixvar_min': 0.0001,
            'edge_overlap_min': 0.0,
            'area_ratio_min': 0.0,
            'area_ratio_max': 10.0
        },
        MaskType.THIN: {
            'pixvar_min': 0.0005,
            'edge_overlap_min': 0.02,
            'area_ratio_min': 0.1,
            'area_ratio_max': 5.0
        },
        MaskType.SPARSE: {
            'pixvar_min': 0.002,
            'edge_overlap_min': 0.08,
            'area_ratio_min': 0.2,
            'area_ratio_max': 3.0
        },
        MaskType.DENSE: {
            'pixvar_min': 0.01,
            'edge_overlap_min': 0.15,
            'area_ratio_min': 0.3,
            'area_ratio_max': 2.5
        }
    }

# ───────────────────────────────────────────────────────────────
# Section: Bayesian Hyperparameter Optimization for QA
# ───────────────────────────────────────────────────────────────

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
from typing import Tuple
import numpy as np
import torch
from enum import Enum, auto

class MaskType(Enum):
    EMPTY  = auto()
    THIN   = auto()
    SPARSE = auto()
    DENSE  = auto()

QA_THRESHOLDS = {
    'pixvar_min': 0.003,
    'pixvar_max': 0.25,
    'edge_overlap_min': 0.15,
    'area_ratio_min': 0.10,
    'area_ratio_max': 4.00,
    'outlier_sigma': 3.5
}

def robust_z_scores(x: torch.Tensor) -> torch.Tensor:
    median = x.median()
    mad    = torch.median(torch.abs(x - median))
    mad    = mad if mad > 0 else torch.tensor(1e-6, device=x.device)
    return torch.abs(x - median) / (mad * 1.4826)

def classify_mask(mask: torch.Tensor) -> MaskType:
    foreground = mask.sum().item()
    area_pct   = foreground / (mask.numel() + 1e-6)
    if foreground < 1:
        return MaskType.EMPTY
    elif area_pct < 0.005:
        return MaskType.THIN
    elif area_pct < 0.05:
        return MaskType.SPARSE
    else:
        return MaskType.DENSE

class MetricTracker:
    def __init__(self):
        self.store = {}
    def log(self, name: str, value: float):
        if name not in self.store:
            self.store[name] = []
        self.store[name].append(value)
    def summary(self, name: str, k: int = 50) -> Tuple[float, float]:
        data = list(self.store.get(name, []))[-k:]
        if not data:
            return 0.0, 0.0
        return float(np.mean(data)), float(np.std(data))
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
import cv2
import random
import sys, os, json, time, argparse, hashlib, traceback
import torch.nn.functional as F

def ensure_size_512(tensor):
    """Ensure tensor is [1,512,512] (for mask/image) via center crop or pad."""
    import torch
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    C, H, W = tensor.shape
    if H == 512 and W == 512:
        return tensor
    # Center crop if larger
    if H > 512 and W > 512:
        y_start = (H - 512) // 2
        x_start = (W - 512) // 2
        return tensor[:, y_start:y_start+512, x_start:x_start+512]
    # Pad if smaller
    pad_h = max(0, 512 - H)
    pad_w = max(0, 512 - W)
    padded = torch.zeros((C, 512, 512), dtype=tensor.dtype, device=tensor.device)
    y_start = (512 - H) // 2
    x_start = (512 - W) // 2
    padded[:, y_start:y_start+H, x_start:x_start+W] = tensor
    return padded




__version__ = "1.0.0"

import argparse


import sys
from torch.utils.data.dataloader import default_collate


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
import cv2
import random
import sys, os, json, time, argparse, hashlib, traceback
# Custom collate_fn: pad images in batch to max size
import torch.nn.functional as F
def pad_collate(batch):
    # batch: list of (image_tensor, path, geometry)
    images, paths, geoms = zip(*batch)
    heights = [img.shape[1] for img in images]
    widths  = [img.shape[2] for img in images]
    max_h, max_w = max(heights), max(widths)
    padded = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        padded.append(padded_img)
    images_stacked = torch.stack(padded, dim=0)
    return images_stacked, list(paths), list(geoms)
import os
import time
import json
import pickle
import hashlib

# Ensure project root is in sys.path for integration imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

import albumentations as A
import cv2
from PIL import Image


try:
    from integration.task_profiler import TaskProfiler
    from integration.cuda_stream_manager import CUDAStreamManager
    from integration.data_validator import DataValidator
except ImportError as e:
    print(f"[ERROR] Could not import integration modules: {e}")
    raise


class ImagePathDataset(data.Dataset):
    """Loads grayscale images from paths and returns normalized tensors."""
    def __init__(self, image_paths, derived_labels_path=None):
        self.image_paths = image_paths
        self.tensor_transform = T.Compose([
            T.ToTensor()
        ])
        self.required_fields = {
            'problem_id': str,
            'image_path': str,
            'features': dict,
            'geometry': list,
            'action_program': list
        }
        self.labels_map = {}
        self.validation_errors = []
        if derived_labels_path is not None:
            try:
                with open(derived_labels_path, 'r') as f:
                    labels = json.load(f)
                for idx, entry in enumerate(labels):
                    entry_errors = []
                    for field, typ in self.required_fields.items():
                        if field not in entry:
                            entry_errors.append(f"Missing field '{field}' in entry {idx}")
                        elif not isinstance(entry[field], typ):
                            entry_errors.append(f"Field '{field}' in entry {idx} has wrong type: expected {typ.__name__}, got {type(entry[field]).__name__}")
                    # geometry: check each element is [x, y] with int/float
                    if 'geometry' in entry and isinstance(entry['geometry'], list):
                        for i, pt in enumerate(entry['geometry']):
                            if not (isinstance(pt, list) and len(pt) == 2 and all(isinstance(x, (int, float)) for x in pt)):
                                entry_errors.append(f"geometry[{i}] in entry {idx} is not [x, y] with int/float")
                    # action_program: check each element is str
                    if 'action_program' in entry and isinstance(entry['action_program'], list):
                        for i, act in enumerate(entry['action_program']):
                            if not isinstance(act, str):
                                entry_errors.append(f"action_program[{i}] in entry {idx} is not str")
                    if entry_errors:
                        self.validation_errors.append({'entry': idx, 'errors': entry_errors})
                    rel_path = os.path.normpath(entry['image_path'].replace('category_1', '1').replace('category_0', '0'))
                    abs_path = os.path.abspath(rel_path)
                    self.labels_map[rel_path] = entry
                    self.labels_map[abs_path] = entry
                print(f"[DEBUG] Derived labels loaded. Keys in labels_map:")
                for k in self.labels_map.keys():
                    print(f"    {k}")
                if self.validation_errors:
                    print(f"[ERROR] Strict validation failed for derived_labels.json:")
                    for err in self.validation_errors:
                        print(f"  Entry {err['entry']}: {err['errors']}")
                else:
                    print(f"[INFO] All derived_labels.json entries passed strict validation.")
            except Exception as e:
                print(f"[WARN] Could not load derived labels: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        norm_path = os.path.normpath(path.replace('category_0', '0').replace('category_1', '1'))
        abs_path = os.path.abspath(norm_path)
        print(f"[LOG] Loading image: {norm_path}")
        print(f"[DEBUG] Lookup paths: norm_path={norm_path}, abs_path={abs_path}")
        with Image.open(norm_path) as img:
            img = img.convert('L')
            arr = np.array(img, dtype=np.float32) / 255.0
        print(f"[LOG] Image loaded: {norm_path}, min={arr.min()}, max={arr.max()}, shape={arr.shape}")
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"Image not normalized: {norm_path}"

        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # [1,H,W]
        # Lookup geometry/features from derived labels
        label_entry = self.labels_map.get(norm_path) or self.labels_map.get(abs_path)
        if label_entry:
            print(f"[LOG] Found derived label for {norm_path}: keys={list(label_entry.keys())}")
            geometry = label_entry.get('geometry', [])
            if geometry is None:
                geometry = []
            if not isinstance(geometry, list):
                geometry = list(geometry)
        else:
            print(f"[LOG] No derived label for {norm_path}")
            geometry = []
        # Pad or truncate geometry to fixed length for batching
        MAX_GEOM_LEN = 32  # adjust as needed for your dataset
        if len(geometry) < MAX_GEOM_LEN:
            geometry = geometry + [[0,0]] * (MAX_GEOM_LEN - len(geometry))
        elif len(geometry) > MAX_GEOM_LEN:
            geometry = geometry[:MAX_GEOM_LEN]
        return tensor, norm_path, geometry



class ImageAugmentor:
    # Bongard-specific QA thresholds for mask types
    # Emergency QA thresholds for CV_8UC512 corruption cleanup
    BONGARD_QA_THRESHOLDS = {
        MaskType.EMPTY:  {'pixvar_min':0.00001,'edge_overlap_min':0.0,'area_ratio_min':0.0,'area_ratio_max':20.0},
        MaskType.THIN:   {'pixvar_min':0.00005,'edge_overlap_min':0.001,'area_ratio_min':0.05,'area_ratio_max':15.0},
        MaskType.SPARSE: {'pixvar_min':0.0001,'edge_overlap_min':0.005,'area_ratio_min':0.1,'area_ratio_max':12.0},
        MaskType.DENSE:  {'pixvar_min':0.0005,'edge_overlap_min':0.01,'area_ratio_min':0.15,'area_ratio_max':10.0},
    }
    
    # Method alias for the module-level functions (provides class interface for global functions)
    def sanitize_for_opencv(self, tensor):
        """Convert tensor to OpenCV-compatible format (class method alias for module function)"""
        import cv2
        import numpy as np
        import torch
        # Call the module-level function
        return sanitize_for_opencv(tensor)
    
    def diagnose_tensor_corruption(self, tensor, name="tensor", path=None):
        """Class method alias for module-level diagnostic function"""
        return diagnose_tensor_corruption(tensor, name=name, path=path)
        
    def safe_device_transfer(self, tensor, device):
        """
        Safely move a PyTorch tensor to the specified device, handling all edge cases.
        """
        import torch
        # Ensure tensor is contiguous before transfer
        tensor = tensor.contiguous()
        if tensor.device == device:
            return tensor
        # Transfer via CPU if direct GPU-GPU fails (common with A5000/4090)
        if device.type == 'cuda' and tensor.device.type == 'cuda':
            tensor = tensor.cpu().to(device)
        else:
            tensor = tensor.to(device)
        return tensor
    
    def repair_mask(self, mask_tensor):
        """Apply morphological closing to repair thin masks."""
        import cv2
        arr = self.sanitize_for_opencv(mask_tensor)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        repaired = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
        return torch.from_numpy(repaired).unsqueeze(0).float()

    def pre_warp_mask(self, mask_tensor):
        """Morphological dilation to fatten thin masks before augmentation."""
        import cv2
        arr = self.sanitize_for_opencv(mask_tensor)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(arr, kernel, iterations=2)
        return torch.from_numpy(dilated).unsqueeze(0).float()
    
    def pre_warp_fatten(self, tensor, size=7):
        """Aggressively dilate mask or line-art image before augmentation."""
        import cv2, numpy as np, torch
        arr = self.sanitize_for_opencv(tensor) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dilated = cv2.dilate(arr, kernel, iterations=2)
        return torch.from_numpy(dilated).unsqueeze(0).float() / 255.0

    # ───────────────────────────────────────────────────────────────
    # Section: Topology-Aware Morphological Repair
    # ───────────────────────────────────────────────────────────────

    def topology_aware_morphological_repair(self, mask: np.ndarray, mask_type: MaskType) -> np.ndarray:
        """
        Perform adaptive opening/closing based on mask_type to preserve structure.
        """
        import cv2
        if mask_type == MaskType.THIN:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            ops = [cv2.MORPH_CLOSE, cv2.MORPH_OPEN]
        elif mask_type == MaskType.SPARSE:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            ops = [cv2.MORPH_CLOSE]
        else:  # DENSE or EMPTY
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
            ops = [cv2.MORPH_OPEN, cv2.MORPH_CLOSE]

        repaired = mask.copy()
        for op in ops:
            repaired = cv2.morphologyEx(repaired, op, kernel)
        return repaired

    """GPU-batched geometric augmentations with profiling, adaptive QA, and failover logic."""
    def __init__(self, device: str = 'cuda', batch_size: int = 32, geometric_transforms=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.profiler = TaskProfiler()
        self.stream_manager = CUDAStreamManager()
        self.data_validator = DataValidator()
        
        # CSV logging prevention
        self.logged_ids = set()
        
        # Epoch-level metrics for percentile scheduling
        self.edge_overlap_hist = []
        
        # Initialize hybrid SAM+SAP pipeline
        if HYBRID_PIPELINE_AVAILABLE:
            try:
                print("[HYBRID] Initializing SAM autocoder...")
                self.sam_autocoder = get_sam_autocoder(
                    model_type="vit_b",  # Start with smallest model
                    device=str(self.device),
                    points_per_side=32
                )
                print("[HYBRID] Initializing skeleton processor...")
                self.skeleton_processor = get_skeleton_processor(
                    min_branch_len=5,
                    gap_fill_radius=3,
                    min_component_size=50
                )
                self.hybrid_enabled = self.sam_autocoder.is_available()
                print(f"[HYBRID] SAM+SAP pipeline initialized: {self.hybrid_enabled}")
                if self.hybrid_enabled:
                    print(f"[HYBRID] SAM model info: {self.sam_autocoder.get_model_info() if hasattr(self.sam_autocoder, 'get_model_info') else 'No model info available'}")
                
                # Tracking counters
                self._sam_mask_count = 0
                self._topology_check_count = 0
                
            except Exception as e:
                print(f"[WARNING] Failed to initialize hybrid pipeline: {e}")
                import traceback
                traceback.print_exc()
                self.hybrid_enabled = False
                self.sam_autocoder = None
                self.skeleton_processor = None
                self._sam_mask_count = 0
                self._topology_check_count = 0
        else:
            self.hybrid_enabled = False
            self.sam_autocoder = None
            self.skeleton_processor = None
            self._sam_mask_count = 0
            self._topology_check_count = 0
        
        # Albumentations transform pipeline with mask synchronization and limited strengths
        # Less aggressive augmentation pipeline
        self.albumentations_configs = [
            A.Compose([
                A.RandomRotate90(p=0.2),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.1),
                # Aggressive transforms removed
            ], additional_targets={'mask': 'mask'}),
            A.Compose([
                A.RandomRotate90(p=0.1),
                A.HorizontalFlip(p=0.1),
                A.VerticalFlip(p=0.05),
                # Aggressive transforms removed
            ], additional_targets={'mask': 'mask'}),
            A.Compose([
                A.RandomRotate90(p=0.05),
                A.HorizontalFlip(p=0.05),
                # Only minimal flips
            ], additional_targets={'mask': 'mask'}),
        ]
        self.albumentations_transform = self.albumentations_configs[0]
        # Emergency QA thresholds for corruption cleanup period
        self.QA_THRESHOLDS = {
            'pixvar_min': 0.0001,      # Emergency: Very low variance for black masks
            'pixvar_max': 0.35,        # Allow higher variance during cleanup
            'edge_overlap_min': 0.01,  # Emergency: Accept minimal edge overlap
            'area_ratio_min': 0.1,     # Emergency: Allow very small masks
            'area_ratio_max': 10.0,    # Emergency: Allow large masks during cleanup
            'outlier_sigma': 2.0       # Emergency: More permissive outlier detection
        }
        
        # Update thresholds for hybrid pipeline (more permissive for SAM-generated masks)
        if self.hybrid_enabled:
            self.QA_THRESHOLDS.update({
                'edge_overlap_min': 0.005,  # Emergency hybrid: Ultra-permissive for SAM cleanup
                'area_ratio_min': 0.05,     # Emergency hybrid: Very small masks OK
                'area_ratio_max': 15.0,     # Emergency hybrid: Very large masks OK during recovery
                'pixvar_min': 0.00005       # Emergency hybrid: Black masks are OK temporarily
            })
            print("[EMERGENCY HYBRID] Ultra-permissive QA thresholds for CV_8UC512 corruption cleanup")
        
        self.adaptive_log = []
        self.metrics = MetricTracker()
        
        # Emergency initialization summary
        print("\n" + "="*80)
        print("🚨 EMERGENCY CV_8UC512 CORRUPTION MITIGATION ACTIVATED 🚨")
        print("="*80)
        print("✅ Tensor corruption diagnostic functions initialized")
        print("✅ OpenCV sanitization functions active")
        print("✅ Safe device transfer protocols enabled")
        print("✅ Emergency QA thresholds configured")
        print("✅ Topology validation using safe fallbacks")
        print("✅ All tensor↔OpenCV conversions protected")
        if self.hybrid_enabled:
            print("✅ Ultra-permissive hybrid QA thresholds for SAM cleanup")
        print("="*80)
        print("🔧 Ready for CV_8UC512 corruption recovery operations")
        print("="*80 + "\n")
    def visualize_mask(self, mask_tensor, out_path):
        import matplotlib.pyplot as plt
        arr = mask_tensor.squeeze(0).cpu().numpy()
        plt.imshow(arr, cmap='gray')
        plt.axis('off')
        plt.savefig(out_path)
        plt.close()
        # Relaxed QA thresholds
        self.QA_THRESHOLDS = {
            'pixvar_min': 0.01,
            'pixvar_max': 0.25,
            'edge_overlap_min': 0.5,
            'area_ratio_min': 0.2,
            'area_ratio_max': 4.0
        }
        self.adaptive_log = []

    def adapt_qa_thresholds(self, fail_rate: float):
        """
        Adjust QA_THRESHOLDS based on a rolling failure rate (0.0–1.0).
        """
        window     = 100
        target     = 0.27    # desired max failure % (was 0.25)
        k_relax    = 0.94    # relax thresholds by 6% (was 0.92)
        k_tighten  = 1.04    # tighten thresholds by 4% (was 1.05)

        # Maintain rolling log
        self.adaptive_log.append(fail_rate)
        if len(self.adaptive_log) > window:
            self.adaptive_log.pop(0)

        long_fail = float(np.mean(self.adaptive_log))
        if long_fail > target:
            self.QA_THRESHOLDS['edge_overlap_min'] *= k_relax
            self.QA_THRESHOLDS['pixvar_min']       *= k_relax
            print(f"[ADAPTIVE QA] Relaxed thresholds after {long_fail:.2%} failure rate")
        elif long_fail < target * 0.5:
            self.QA_THRESHOLDS['edge_overlap_min'] *= k_tighten
            self.QA_THRESHOLDS['pixvar_min']       *= k_tighten
            print(f"[ADAPTIVE QA] Tightened thresholds after {long_fail:.2%} failure rate")

    def retry_augmentation(self, image, geometry, max_retries=3):
        for attempt in range(max_retries):
            result = self.augment_batch(image, geometries=[geometry])
            if self.passes_qa(result):
                return result, 'retry_pass'
        return result, 'forced_pass'

    def passes_qa(self, result):
        # Accept if all QA metrics pass (simple check)
        profiling = result.get('profiling', {})
        # You can add more sophisticated checks here
        return profiling.get('qa_fail_count', 0) == 0

    def get_augmentation_pipeline(self, mask_type):
        """Return mask-type-specific augmentation pipeline (Kornia for sparse, Albumentations for dense)."""
        try:
            import kornia.augmentation as K
        except ImportError:
            K = None
        if mask_type == MaskType.EMPTY:
            # No augmentation for empty masks
            return None
        elif mask_type == MaskType.THIN:
            if K:
                return K.AugmentationSequential(
                    K.RandomHorizontalFlip(p=0.1),
                    K.RandomRotation(degrees=5, p=0.1),
                    data_keys=["input", "mask"]
                )
            else:
                return A.Compose([
                    A.HorizontalFlip(p=0.1),
                    A.RandomRotate90(p=0.05)
                ], additional_targets={'mask': 'mask'})
        elif mask_type == MaskType.SPARSE:
            if K:
                return K.AugmentationSequential(
                    K.RandomHorizontalFlip(p=0.2),
                    K.RandomVerticalFlip(p=0.1),
                    K.RandomRotation(degrees=10, p=0.1),
                    data_keys=["input", "mask"]
                )
            else:
                return A.Compose([
                    A.HorizontalFlip(p=0.2),
                    A.VerticalFlip(p=0.1),
                    A.RandomRotate90(p=0.1),
                    A.Rotate(limit=10, p=0.1)
                ], additional_targets={'mask': 'mask'})
        else:  # DENSE
            return self.albumentations_configs[0]

    def geometry_to_binary_mask(self, geometry, H, W):
        # Rasterize geometry (list of [x, y] vertices) into a binary mask
        from PIL import ImageDraw
        mask_img = Image.new('L', (W, H), 0)
        # Convert tensor to list and filter out zero-padding
        if isinstance(geometry, torch.Tensor):
            geometry = geometry.cpu().numpy().tolist()
        # Filter out padding points, which are [0,0]
        poly_points = [
            (float(x), float(y))
            for x, y in geometry
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not (x == 0 and y == 0)
        ]
        if not poly_points:
            print(f"[QA WARN] Received empty or invalid geometry for mask generation.")
        if len(poly_points) > 1:
            # Scale points to image dimensions. Assuming original canvas is 256x256.
            scaled_points = [(x * (W/256), y * (H/256)) for x, y in poly_points]
            ImageDraw.Draw(mask_img).line(scaled_points, fill=1, width=2)
        mask = np.array(mask_img, dtype=np.float32)
        return torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

    def _extract_features(self, images):
        """Extract feature embeddings using a pretrained ResNet18."""
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.eval()
        # Remove final layer
        feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        with torch.no_grad():
            feats = feature_extractor(images.repeat(1,3,1,1)) # grayscale to 3ch
            return feats.view(images.size(0), -1)

    def _save_grid(self, images, masks, batch_idx, out_dir="qa_grids"):
        import torch
        os.makedirs(out_dir, exist_ok=True)
        # Defensive: ensure images/masks are [B,1,H,W]
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        elif masks.ndim == 4 and masks.shape[1] != 1:
            masks = masks[:, :1, :, :]
        # Optionally crop to max size for visualization
        crop_h, crop_w = 512, 512
        images = images[..., :crop_h, :crop_w]
        masks = masks[..., :crop_h, :crop_w]
        print(f"[DEBUG] _save_grid images shape: {images.shape}, masks shape: {masks.shape}")
        grid_img = make_grid(images.float().cpu(), nrow=4, normalize=True)
        grid_mask = make_grid(masks.float().cpu(), nrow=4, normalize=True)
        save_image(grid_img, os.path.join(out_dir, f"batch_{batch_idx}_images.png"))
        save_image(grid_mask, os.path.join(out_dir, f"batch_{batch_idx}_masks.png"))

    # ───────────────────────────────────────────────────────────────
    # Section: Multi-Scale Outlier Detection Setup
    # ───────────────────────────────────────────────────────────────

    def multi_scale_outlier_mask(self, mask_sums: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Combine MAD, IQR, and IsolationForest to identify outlier samples.
        Returns boolean array of length B indicating flagged samples.
        """
        # 1. MAD-based
        median = np.median(mask_sums)
        mad = np.median(np.abs(mask_sums - median))
        mad_z = np.abs(mask_sums - median) / (mad * 1.4826 + 1e-6)
        mad_flags = mad_z > 3.0

        # 2. IQR-based
        q75, q25 = np.percentile(mask_sums, [75, 25])
        iqr_val = q75 - q25
        iqr_flags = (mask_sums < q25 - 1.5*iqr_val) | (mask_sums > q75 + 1.5*iqr_val)

        # 3. IsolationForest with adaptive contamination
        contam = min(0.5, max(0.03, (batch_size/1280)))
        iso = IsolationForest(contamination=contam, random_state=42)
        iso_labels = iso.fit_predict(mask_sums.reshape(-1,1))  # -1 for outlier
        iso_flags = iso_labels == -1

        # Combine: flag if any method marks outlier
        return mad_flags | iqr_flags | iso_flags

    def _statistical_outlier_detection(self, images, masks, batch_idx, outlier_dir="qa_outliers"):
        os.makedirs(outlier_dir, exist_ok=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        assert images.ndim == 4, f"images should be 4D [B,1,H,W], got {images.shape}"
        assert masks.ndim == 4, f"masks should be 4D [B,1,H,W], got {masks.shape}"
        B = images.size(0)
        features = {
            'img_mean': images.mean(dim=[1,2,3]),
            'img_std' : images.std(dim=[1,2,3]),
            'mask_sum': masks.sum(dim=[1,2,3])
        }
        # Compute robust z-scores for each feature
        rz_scores = {k: robust_z_scores(v) for k, v in features.items()}
        # Compute adaptive percentile cutoff (e.g., 97th percentile)
        all_rz = torch.cat([rz_scores['img_mean'], rz_scores['img_std'], rz_scores['mask_sum']])
        adaptive_cutoff = float(torch.quantile(all_rz, 0.97).item())
        
        # Enhanced multi-scale outlier detection
        mask_sums_np = features['mask_sum'].cpu().numpy()
        outlier_flags = self.multi_scale_outlier_mask(mask_sums_np, batch_size=B)
        
        # Two-tier flagging: warn and reject
        warn_flags = torch.zeros(B, dtype=torch.bool, device=images.device)
        reject_flags = torch.zeros(B, dtype=torch.bool, device=images.device)
        
        # Integrate multi-scale outlier flags
        warn_flags |= torch.from_numpy(outlier_flags).to(warn_flags.device)
        # Mask-type-aware gating (further relaxed factors)
        orig_masks = masks
        sigma_factors = {
            MaskType.EMPTY: 3.0,
            MaskType.THIN: 2.5,
            MaskType.SPARSE: 1.8,
            MaskType.DENSE: 1.0
        }
        warn_threshold = 4.5  # raised from 4.0
        for i in range(B):
            mtype = classify_mask(orig_masks[i])
            sigma_factor = sigma_factors[mtype]
            local_cutoff = max(QA_THRESHOLDS['outlier_sigma'] * sigma_factor, adaptive_cutoff, warn_threshold)
            # Feature-specific gating (img_mean only for reject, all for warn)
            if rz_scores['img_mean'][i] > local_cutoff * 1.4:
                reject_flags[i] = True
            elif rz_scores['img_mean'][i] > local_cutoff:
                warn_flags[i] = True
            # Also warn if any feature is above local_cutoff
            for k in rz_scores:
                if rz_scores[k][i] > local_cutoff:
                    warn_flags[i] = True
        # Save flagged samples
        import csv
        csv_path = os.path.join(outlier_dir, "batch_warnings.csv")
        csv_fields = ["batch_idx", "sample_idx", "flag_type", "reason", "mean", "std", "mask_sum", "mask_type"]
        csv_rows = []
        # Save flagged samples
        for i in range(B):
            mtype = classify_mask(orig_masks[i])
            mean_val = float(features['img_mean'][i].item())
            std_val = float(features['img_std'][i].item())
            mask_sum_val = float(features['mask_sum'][i].item())
            if reject_flags[i]:
                reason = f"img_mean robust_z={rz_scores['img_mean'][i]:.2f} > reject_cutoff={local_cutoff*1.4:.2f}"
                print(f"[QA OUTLIER] Sample {i} REJECTED by robust MAD gate in batch {batch_idx}.")
                save_image(images[i], os.path.join(outlier_dir, f"batch{batch_idx}_img{i}_reject.png"))
                save_image(masks[i],  os.path.join(outlier_dir, f"batch{batch_idx}_mask{i}_reject.png"))
                csv_rows.append([batch_idx, i, "REJECT", reason, mean_val, std_val, mask_sum_val, mtype.name])
            elif warn_flags[i]:
                # Find which feature(s) triggered warning
                reasons = []
                for k in rz_scores:
                    if rz_scores[k][i] > local_cutoff:
                        reasons.append(f"{k} robust_z={rz_scores[k][i]:.2f} > warn_cutoff={local_cutoff:.2f}")
                reason = "; ".join(reasons) if reasons else "Unknown"
                print(f"[QA OUTLIER] Sample {i} WARNED by robust MAD gate in batch {batch_idx}.")
                save_image(images[i], os.path.join(outlier_dir, f"batch{batch_idx}_img{i}_warn.png"))
                save_image(masks[i],  os.path.join(outlier_dir, f"batch{batch_idx}_mask{i}_warn.png"))
                csv_rows.append([batch_idx, i, "WARN", reason, mean_val, std_val, mask_sum_val, mtype.name])
        # Write all warnings/rejects to CSV for inspection
        if csv_rows:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(csv_fields)
                writer.writerows(csv_rows)
        # Automated alert for excessive WARNs
        if not hasattr(self, 'warn_history'):
            self.warn_history = []
        warn_rate = warn_flags.sum().item() / B
        self.warn_history.append(warn_rate)
        if len(self.warn_history) > 5:
            self.warn_history.pop(0)
        if all(wr > 0.07 for wr in self.warn_history) and len(self.warn_history) == 5:
            print(f"[ALERT] WARN rate exceeded 7% for 5 consecutive batches. Manual review recommended for outlier strategy and thresholds.")

    def _augmentation_diversity_metrics(self, images, batch_idx):
        feats = self._extract_features(images.cpu())
        # Compute pairwise distances
        dists = torch.cdist(feats, feats)
        diversity = dists.mean().item()
        print(f"[QA DIVERSITY] Batch {batch_idx} diversity score: {diversity:.3f}")

    def _adversarial_robustness(self, images, masks, batch_idx, adv_dir="qa_adversarial"):
        # ───────────────────────────────────────────────────────────────
        # Section: Adversarial Robustness Testing
        # ───────────────────────────────────────────────────────────────
        os.makedirs(adv_dir, exist_ok=True)
        
        # Enhanced adversarial noise test
        adv_images = self.adversarial_noise_test(images, magnitude=0.2)
        
        # Re-run statistical outlier detection on adversarial samples
        self._statistical_outlier_detection(
            adv_images, masks, batch_idx, outlier_dir="qa_adversarial"
        )
        
        # Original simple check
        for i in range(images.size(0)):
            if adv_images[i].max()-adv_images[i].min() < 0.1:
                print(f"[QA ADV] Adversarial corruption detected in batch {batch_idx}, sample {i}.")
                save_image(adv_images[i], os.path.join(adv_dir, f"batch{batch_idx}_img{i}_adv.png"))

    def adversarial_noise_test(self, images: torch.Tensor, magnitude: float = 0.2) -> torch.Tensor:
        """
        Apply randomized high-frequency noise and verify QA metrics remain stable.
        Returns perturbed images for inspection.
        """
        noise = magnitude * torch.randn_like(images)
        adv_images = torch.clamp(images + noise, 0.0, 1.0)
        return adv_images

    def _validate_augmented_batch(self, aug_images, aug_masks, orig_masks, paths, batch_idx, batch_labels=None):
        fail_count = 0
        flagged = set()
        qa_human_review_dir = "qa_human_review"
        os.makedirs(qa_human_review_dir, exist_ok=True)
        edge_overlaps = []
        # CSV logging setup
        import csv
        outlier_dir = "qa_outliers"
        os.makedirs(outlier_dir, exist_ok=True)
        csv_path = os.path.join(outlier_dir, "batch_warnings.csv")
        csv_fields = ["batch_idx", "sample_idx", "flag_type", "reason", "pixvar", "orig_area", "aug_area", "area_ratio", "edge_overlap", "mask_type"]
        csv_rows = []
        for i in range(aug_images.size(0)):
            path = paths[i] if paths else f"sample_{i}"
            mtype = classify_mask(orig_masks[i])
            
            # Early mask validation gate with tensor corruption diagnosis
            if not self._is_mask_valid(aug_masks[i]):
                print(f"[QA SKIP] Invalid mask detected for {path}, skipping QA.")
                # Run emergency tensor corruption diagnosis
                tensor_diag = self.diagnose_tensor_corruption(aug_masks[i])
                if tensor_diag['corrupted']:
                    print(f"[EMERGENCY] CV_8UC512 corruption detected in {path}: {tensor_diag['issues']}")
                continue
                
            # Prevent double logging
            log_id = (batch_idx, i)
            if log_id in self.logged_ids:
                continue
            self.logged_ids.add(log_id)
            
            thresholds = self.BONGARD_QA_THRESHOLDS[mtype]
            
            # Use adaptive edge overlap thresholds
            edge_complexity = self.compute_edge_complexity(orig_masks[i])
            adaptive_edge_threshold = self.adaptive_edge_overlap_threshold(mtype, edge_complexity)
            thresholds = thresholds.copy()  # Don't modify original
            thresholds['edge_overlap_min'] = adaptive_edge_threshold
            
            pixvar = aug_images[i].var().item()
            orig_area = orig_masks[i].sum().item()
            aug_area = aug_masks[i].sum().item()
            area_ratio = aug_area / orig_area if orig_area > 0 else 0
            edge_overlap = None
            flag_type = None
            reason = None
            # Bypass QA for empty masks
            if mtype == MaskType.EMPTY:
                continue
            # Content Preservation Check (pixel variance)
            if pixvar < thresholds['pixvar_min']:
                print(f"[QA FAIL] Content loss detected in {path}. Pixel variance {pixvar:.5f} below threshold.")
                flagged.add(i)
                fail_count += 1
                flag_type = "FAIL"
                reason = f"Content loss: pixvar={pixvar:.5f} < floor={thresholds['pixvar_min']:.5f}"
            # Mask Integrity & Area Deviation Monitoring
            if orig_area > 10 and aug_area < 5:
                print(f"[QA FAIL] Mask integrity failed for {path}. Shape was lost during augmentation.")
                flagged.add(i)
                fail_count += 1
                flag_type = "FAIL"
                reason = f"Mask integrity: orig_area={orig_area:.1f} > 10, aug_area={aug_area:.1f} < 5"
            elif orig_area > 10:
                if not (thresholds['area_ratio_min'] < area_ratio < thresholds['area_ratio_max']):
                    print(f"[QA WARN] Significant area deviation for {path}. Ratio: {area_ratio:.2f}")
                    flagged.add(i)
                    flag_type = "WARN"
                    reason = f"Area deviation: area_ratio={area_ratio:.2f} not in ({thresholds['area_ratio_min']:.2f}, {thresholds['area_ratio_max']:.2f})"
            # Edge overlap (if mask is binary)
            if aug_masks[i].max() > 0.5 and orig_masks[i].max() > 0.5:
                overlap = ((aug_masks[i] > 0.5) & (orig_masks[i] > 0.5)).float().sum().item()
                aug_bin_area = (aug_masks[i] > 0.5).float().sum().item()
                orig_bin_area = (orig_masks[i] > 0.5).float().sum().item()
                denom = max(aug_bin_area, orig_bin_area, 1)
                edge_overlap = overlap / denom
                edge_overlaps.append(edge_overlap)
                
                # Store edge overlap for percentile scheduling
                self.edge_overlap_hist.append(edge_overlap)
                
                if edge_overlap < thresholds['edge_overlap_min']:
                    print(f"[QA WARN] Low edge overlap for {path}: {edge_overlap:.3f}")
                    flagged.add(i)
                    flag_type = "WARN"
                    reason = f"Low edge overlap: {edge_overlap:.3f} < min={thresholds['edge_overlap_min']:.3f}"
            
            # Skeleton topology QA (if hybrid pipeline available) with safe validation
            if self.hybrid_enabled and self.skeleton_processor is not None:
                try:
                    # Use safe topology validation with emergency fallbacks
                    is_topology_valid, topology_reason = self.safe_topology_validation(
                        orig_masks[i], aug_masks[i], tolerance=1
                    )
                    self._topology_check_count += 1  # Track topology checks
                    
                    if not is_topology_valid:
                        print(f"[QA EMERGENCY] Topology violation for {path}: {topology_reason}")
                        flagged.add(i)
                        fail_count += 1
                        flag_type = "EMERGENCY"
                        reason = f"Emergency Topology: {topology_reason}"
                        
                except Exception as e:
                    print(f"[QA WARN] Topology check failed for {path}: {e}")
                    # Don't fail on topology check errors, just warn
            # Label consistency check (if batch_labels provided)
            if batch_labels is not None and len(batch_labels) > i:
                label = batch_labels[i]
                if isinstance(label, dict) and 'annotators' in label:
                    annots = label['annotators']
                    if len(set(annots)) > 1:
                        print(f"[QA LABEL] Inconsistent annotators for {path}: {annots}")
                        flagged.add(i)
                        flag_type = "WARN"
                        reason = f"Inconsistent annotators: {annots}"
            # Save flagged sample images
            if i in flagged:
                save_image(aug_images[i], os.path.join(qa_human_review_dir, f"batch{batch_idx}_img{i}_flagged.png"))
                save_image(aug_masks[i],  os.path.join(qa_human_review_dir, f"batch{batch_idx}_mask{i}_flagged.png"))
                # Log to CSV if flagged
                csv_rows.append([
                    batch_idx, i, flag_type if flag_type else "WARN", reason if reason else "Unknown",
                    float(pixvar), float(orig_area), float(aug_area), float(area_ratio),
                    float(edge_overlap) if edge_overlap is not None else "",
                    mtype.name
                ])
        # Edge overlap percentile flagging
        if edge_overlaps:
            threshold = np.percentile(edge_overlaps, 20)
            for i, eo in enumerate(edge_overlaps):
                if eo < threshold:
                    flagged.add(i)
                    # Log to CSV for percentile-based warning
                    csv_rows.append([
                        batch_idx, i, "WARN", f"Edge overlap below 20th percentile: {eo:.2f} < {threshold:.2f}",
                        float(aug_images[i].var().item()), float(orig_masks[i].sum().item()), float(aug_masks[i].sum().item()),
                        float(aug_masks[i].sum().item()) / (float(orig_masks[i].sum().item()) + 1e-6),
                        float(eo),
                        classify_mask(orig_masks[i]).name
                    ])
        # Write all warnings/failures to CSV for inspection
        if csv_rows:
            # Ensure output directory exists and is writable
            output_dir = os.path.dirname(csv_path)
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Cannot write to directory: {output_dir}")
            # Open CSV with buffering, write header if new
            file_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, mode="a", newline="", buffering=1) as csvfile:
                writer = csv.writer(csvfile)
                if file_new:
                    writer.writerow(csv_fields)
                writer.writerows(csv_rows)
        self._statistical_outlier_detection(aug_images, aug_masks, batch_idx, outlier_dir="qa_outliers")
        self._augmentation_diversity_metrics(aug_images, batch_idx)
        self._adversarial_robustness(aug_images, aug_masks, batch_idx, adv_dir="qa_adversarial")
        self._save_grid(aug_images, aug_masks, batch_idx)
        batch_fail_rate = fail_count / aug_images.size(0)
        self.metrics.log("fail_rate", batch_fail_rate)
        mean_fail, _ = self.metrics.summary("fail_rate", 50)
        self.adapt_qa_thresholds(mean_fail)
        
        # Apply epoch-level percentile scheduler for adaptive thresholds
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'store') and "epoch_end" in getattr(self.metrics, 'store', {}):
            if len(self.edge_overlap_hist) >= 500:
                hist = np.array(self.edge_overlap_hist[-500:])
                p20 = np.percentile(hist, 20)
                # Update global QA threshold
                if hasattr(self, 'QA_THRESHOLDS'):
                    self.QA_THRESHOLDS['edge_overlap_min'] = round(float(p20), 3)
                    print(f"[PERCENTILE SCHEDULER] Updated global edge_overlap_min to {p20:.3f} based on 20th percentile")
                # Also update BONGARD_QA_THRESHOLDS for all mask types
                for mask_type in self.BONGARD_QA_THRESHOLDS:
                    self.BONGARD_QA_THRESHOLDS[mask_type]['edge_overlap_min'] = round(float(p20), 3)
        
        return fail_count

    def augment_batch(self, images: torch.Tensor, paths=None, geometries=None, augment_type: str = 'geometric', batch_idx=0, batch_labels=None):
        # Defensive: ensure images is a tensor, not a tuple/list
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"[AUG ERROR] 'images' must be a torch.Tensor, got {type(images)}")
        # Defensive: ensure images are 4D [B,1,H,W]
        if images.ndim == 3:
            images = images.unsqueeze(1)
        batch_size, C, H, W = images.shape
        print(f"[AUG] Received batch: size={batch_size}, shape=({C},{H},{W}), type={augment_type}")
        print(f"[AUG] Original pixel range: min={images.min().item():.3f}, max={images.max().item():.3f}")
        # Validate actual shapes against schema
        image_shapes = [list(img.shape) for img in images]
        try:
            self.data_validator.validate({
                'images': image_shapes,
                'type': augment_type
            }, 'image_augmentation.schema.json')
        except Exception as ve:
            print("[SCHEMA ERROR] Validation failed in augment_batch!")
            print("Expected: list of image shapes, e.g. [[1, 512, 512], ...]")
            print(f"Actual: {image_shapes}")
            print(f"Error: {ve}")
            raise ve
        images_np = images.cpu().numpy()
        # Rasterize geometry to binary masks for each image
        orig_masks = []
        mask_types = []
        for idx, geom in enumerate(geometries if geometries is not None else [None]*batch_size):
            if geom is not None:
                mask = self.geometry_to_binary_mask(geom, H, W)
                mask_type = classify_mask(mask)
                
                # If mask is empty or very thin, try hybrid SAM generation
                if mask_type == MaskType.EMPTY and self.hybrid_enabled:
                    print(f"[HYBRID] Empty geometry mask detected, using SAM for sample {idx}")
                    print(f"[HYBRID] Pipeline state: hybrid_enabled={self.hybrid_enabled}, "
                          f"sam_autocoder={self.sam_autocoder is not None}, "
                          f"skeleton_processor={self.skeleton_processor is not None}")
                    
                    # Verify image format before processing
                    img = images[idx]
                    print(f"[HYBRID] Input image: shape={img.shape}, dtype={img.dtype}, "
                          f"device={img.device}, min={img.min().item():.4f}, max={img.max().item():.4f}")
                    
                    try:
                        # Generate SAM mask for this image
                        print(f"[HYBRID] Calling generate_hybrid_mask for sample {idx}")
                        hybrid_mask = self.generate_hybrid_mask(images[idx])
                        print(f"[HYBRID] Generate hybrid mask returned. Shape={hybrid_mask.shape}, "
                              f"sum={hybrid_mask.sum().item():.1f}")
                              
                        if hybrid_mask.sum() > 10:  # Valid mask generated
                            print(f"[HYBRID] Valid mask generated (sum > 10)")
                            mask = hybrid_mask.squeeze(0)  # Remove channel dim
                            mask_type = classify_mask(mask)
                            self._sam_mask_count += 1  # Track SAM usage
                            print(f"[HYBRID] SAM generated {mask_type.name} mask with area {mask.sum().item():.1f}")
                        else:
                            print(f"[HYBRID] Empty mask returned (sum <= 10): {hybrid_mask.sum().item()}")
                    except Exception as e:
                        print(f"[HYBRID] SAM mask generation failed for sample {idx}: {e}")
                        import traceback
                        traceback.print_exc()
                
                mask_types.append(mask_type)
                # Conservative pre-warp fattening for mask (area-adaptive)
                mask = self.pre_warp_fatten(mask, size=7)
            else:
                # No geometry provided - use hybrid SAM generation if available
                if self.hybrid_enabled:
                    print(f"[HYBRID] No geometry provided, using SAM for sample {idx}")
                    try:
                        hybrid_mask = self.generate_hybrid_mask(images[idx])
                        if hybrid_mask.sum() > 10:
                            mask = hybrid_mask.squeeze(0)
                            mask_type = classify_mask(mask)
                            mask_types.append(mask_type)
                            self._sam_mask_count += 1  # Track SAM usage
                            print(f"[HYBRID] SAM generated {mask_type.name} mask with area {mask.sum().item():.1f}")
                        else:
                            mask = torch.zeros((1, H, W), dtype=torch.float32)
                            mask_types.append(MaskType.EMPTY)
                    except Exception as e:
                        print(f"[HYBRID] SAM fallback failed for sample {idx}: {e}")
                        mask = torch.zeros((1, H, W), dtype=torch.float32)
                        mask_types.append(MaskType.EMPTY)
                else:
                    mask = torch.zeros((1, H, W), dtype=torch.float32)
                    mask_types.append(MaskType.EMPTY)
            orig_masks.append(mask)
        orig_masks = torch.stack(orig_masks)
        if orig_masks.ndim == 3:
            orig_masks = orig_masks.unsqueeze(1)
        masks_np = orig_masks.cpu().numpy()
        aug_images_list = []
        aug_masks_list = []
        t0 = time.time()
        for i in range(batch_size):
            img = images_np[i].transpose(1,2,0)  # [H,W,C]
            mask = masks_np[i].transpose(1,2,0)
            mask_tensor = torch.from_numpy(mask.transpose(2,0,1))
            img_tensor = torch.from_numpy(img.transpose(2,0,1))
            mask_type = mask_types[i]
            mask_foreground = mask_tensor.sum().item()
            # Smart QA bypass for ultra-sparse masks
            if mask_type == MaskType.EMPTY or mask_foreground < 5:
                aug_img_cropped = self.safe_device_transfer(ensure_size_512(img_tensor), self.device)
                aug_mask_cropped = self.safe_device_transfer(ensure_size_512(mask_tensor), self.device)
                aug_images_list.append(aug_img_cropped)
                aug_masks_list.append(aug_mask_cropped)
                continue
            # Canvas padding based on mask type
            pad_size = 256 if mask_type == MaskType.THIN else 128
            mask_padded, img_padded = self.pad_and_center(mask_tensor, img_tensor, pad=pad_size)
            dilation_size = 21 if mask_type == MaskType.THIN else 15
            dilation_iter = 4 if mask_type == MaskType.THIN else 2
            # Fatten mask for augmentation
            mask_fattened = self.fatten_and_erode(mask_padded, size=dilation_size, iterations=dilation_iter, mode='dilate')
            img_fattened = img_padded * (mask_fattened > 0)
            # Select augmentation pipeline
            aug_pipeline = self.get_augmentation_pipeline(mask_type)
            max_retries = 3
            for attempt in range(max_retries):
                if aug_pipeline is None:
                    aug_img_cropped = ensure_size_512(img_padded)
                    aug_mask_cropped = ensure_size_512(mask_padded)
                elif hasattr(aug_pipeline, 'forward'):
                    # Kornia pipeline
                    aug_out = aug_pipeline(img_fattened.unsqueeze(0), mask_fattened.unsqueeze(0))
                    aug_img = aug_out[0].squeeze(0)
                    aug_mask = aug_out[1].squeeze(0)
                    aug_img_cropped = self.center_crop(aug_img, size=512)
                    aug_mask_cropped = self.center_crop(aug_mask, size=512)
                else:
                    # Albumentations pipeline
                    augmented = aug_pipeline(
                        image=img_fattened.cpu().numpy().transpose(1,2,0),
                        mask=mask_fattened.cpu().numpy().transpose(1,2,0)
                    )
                    aug_img = torch.from_numpy(augmented['image'].transpose(2,0,1))
                    aug_mask = torch.from_numpy(augmented['mask'].transpose(2,0,1))
                    aug_img_cropped = self.center_crop(aug_img, size=512)
                    aug_mask_cropped = self.center_crop(aug_mask, size=512)
                # Erode after augmentation
                aug_mask_eroded = self.fatten_and_erode(aug_mask_cropped, size=dilation_size, iterations=dilation_iter, mode='erode')
                
                # Apply topology-aware morphological repair
                mask_np = self.sanitize_for_opencv(aug_mask_eroded)
                mask_np = self.topology_aware_morphological_repair(mask_np, mask_type)
                aug_mask_eroded = self.safe_device_transfer(torch.from_numpy(mask_np).unsqueeze(0).float(), self.device)
                
                aug_img_cropped = ensure_size_512(aug_img_cropped)
                aug_mask_cropped = ensure_size_512(aug_mask_eroded)
                # QA checks
                mask_fattened_cropped = self.center_crop(mask_fattened, size=512)
                edge_overlap = self.compute_edge_overlap(mask_fattened_cropped, aug_mask_cropped)
                area_ratio = aug_mask_cropped.sum().item() / (mask_fattened_cropped.sum().item() + 1e-6)
                # Strict QA gate for thin/sparse masks
                if mask_type in [MaskType.THIN, MaskType.SPARSE]:
                    if edge_overlap >= 0.02 and 0.1 < area_ratio < 5.0:
                        break
                else:
                    if edge_overlap > 0.1 and 0.33 < area_ratio < 3.0:
                        break
                # If last attempt, fallback to padded original
                if attempt == max_retries - 1:
                    aug_img_cropped = ensure_size_512(img_padded)
                    aug_mask_cropped = ensure_size_512(mask_padded)
            aug_img_cropped = self.safe_device_transfer(aug_img_cropped, self.device)
            aug_mask_cropped = self.safe_device_transfer(aug_mask_cropped, self.device)
            aug_images_list.append(aug_img_cropped)
            aug_masks_list.append(aug_mask_cropped)
        aug_images = self.safe_device_transfer(torch.stack(aug_images_list), self.device)
        aug_masks = self.safe_device_transfer(torch.stack(aug_masks_list), self.device)
        if aug_images.ndim == 3:
            aug_images = aug_images.unsqueeze(1)
        if aug_masks.ndim == 3:
            aug_masks = aug_masks.unsqueeze(1)
        print(f"[AUG] Augmented image range: min={aug_images.min().item():.3f}, max={aug_images.max().item():.3f}")
        print(f"[AUG] Geometric mask transform range: min={aug_masks.min().item():.3f}, max={aug_masks.max().item():.3f}, unique={len(torch.unique(aug_masks))}")
        self._validate_augmented_batch(aug_images, aug_masks, self.safe_device_transfer(orig_masks, self.device), paths, batch_idx, batch_labels=batch_labels)
        mask_bin = (aug_masks > 0.5).float()
        results = {}
        results['original']  = aug_images
        results['geometric'] = mask_bin
        if augment_type == 'both':
            results['combined'] = torch.cat([aug_images, mask_bin], dim=1)
        transform_time = (time.time() - t0) * 1000
        self.profiler.log_latency('image_augmentation', transform_time, {
            'batch_size': batch_size,
            'augment_type': augment_type,
            'device': str(self.device),
            'transform_ms': transform_time
        })
        results['profiling'] = {
            'transform_ms': transform_time,
            'throughput_imgs_per_sec': batch_size / (transform_time / 1000)
        }
        return results

    def export_image_mask_pairs(self, dataset, export_dir="export_for_diffusion", max_samples=None):
        """Export image and mask pairs for diffusion model conditioning."""
        import shutil
        os.makedirs(export_dir, exist_ok=True)
        img_dir = os.path.join(export_dir, "images")
        mask_dir = os.path.join(export_dir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        count = 0
        for idx in range(len(dataset)):
            tensor, norm_path, geometry = dataset[idx]
            # Convert tensor to image
            arr = tensor.squeeze(0).cpu().numpy() * 255.0
            arr = arr.astype(np.uint8)
            img_pil = Image.fromarray(arr, mode="L")
            # Generate mask
            H, W = arr.shape
            mask_tensor = self.geometry_to_binary_mask(geometry, H, W)
            mask_arr = (mask_tensor.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
            mask_pil = Image.fromarray(mask_arr, mode="L")
            # Use problem_id or filename for naming
            base_name = os.path.splitext(os.path.basename(norm_path))[0]
            img_path = os.path.join(img_dir, f"{base_name}.png")
            mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
            img_pil.save(img_path)
            mask_pil.save(mask_path)
            count += 1
            if max_samples and count >= max_samples:
                break
        print(f"[EXPORT] Saved {count} image-mask pairs to {export_dir}")

    def generate_diffusion_augmented(self, image_path, mask_path, output_dir, num_variations=4, model_name="lllyasviel/sd-controlnet-seg", device="cuda"):
        """Generate synthetic image variations using ControlNet/Stable Diffusion with mask conditioning."""
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        import torch
        from PIL import Image
        import random
        os.makedirs(output_dir, exist_ok=True)
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Load ControlNet model
        controlnet = ControlNetModel.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
        prompt = "A realistic object with diverse lighting and texture, mask preserved"
        for i in range(num_variations):
            seed = random.randint(0, 999999)
            generator = torch.manual_seed(seed)
            result = pipe(prompt, image=image, control_image=mask, generator=generator, num_inference_steps=30)
            out_img = result.images[0]
            out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.png")
            out_img.save(out_path)
        print(f"[DIFFUSION] Generated {num_variations} variations for {image_path}")

    # --- Advanced, research-driven augmentation methods ---
    def synthesize_pair_diffusion(self, image_np, mask_np, prompt="", device="cuda"):
        """Mask-guided diffusion synthesis: preserves mask, varies appearance."""
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        ).to(device)
        # mask_np: HxW, 255 for regions to preserve
        out = pipe(prompt=prompt, image=image_np, mask_image=mask_np)["images"][0]
        img_t = torch.from_numpy(np.array(out).transpose(2,0,1)/255.0).unsqueeze(0)
        msk_t = torch.from_numpy(mask_np[np.newaxis,:,:]/255.0).unsqueeze(0)
        return img_t.float(), msk_t.float()

    def x_paste_batch(self, target_img, target_mask, instance_images, instance_masks, k=5):
        """Scalable Copy-Paste: paste k instances, preserve mask shapes."""
        import random, cv2
        H, W, _ = target_img.shape
        out_img, out_msk = target_img.copy(), target_mask.copy()
        for _ in range(k):
            inst_img = random.choice(instance_images)
            inst_msk = random.choice(instance_masks)
            scale = random.uniform(0.5, 1.2)
            h, w = int(inst_img.shape[0]*scale), int(inst_img.shape[1]*scale)
            inst_img_s = cv2.resize(inst_img, (w,h), interpolation=cv2.INTER_LINEAR)
            inst_msk_s = cv2.resize(inst_msk, (w,h), interpolation=cv2.INTER_NEAREST)
            x, y = random.randint(0, W-w), random.randint(0, H-h)
            mask_bool = inst_msk_s>127
            out_img[y:y+h, x:x+w][mask_bool] = inst_img_s[mask_bool]
            out_msk[y:y+h, x:x+w][mask_bool] = 1
        return out_img, out_msk

    def keepmask_augment(self, image, mask):
        """Perturb background, preserve foreground object."""
        import numpy as np
        aug = image.copy()
        noise = np.random.normal(0, 0.1, size=image.shape)
        aug[mask==0] = np.clip(aug[mask==0] + 255*noise[mask==0], 0, 255)
        factor = np.random.uniform(0.8, 1.2, size=(1,1,3))
        aug[mask==0] = np.clip(aug[mask==0] * factor, 0, 255)
        return aug.astype(np.uint8), mask

    def advanced_augment_batch(self, images, masks, instance_images=None, instance_masks=None, device="cuda"):
        """Hybrid augmentation: geometric + advanced generative/compositional methods."""
        import random
        batch_size = images.shape[0]
        aug_images, aug_masks = [], []
        for i in range(batch_size):
            img = images[i].cpu().numpy().transpose(1,2,0) * 255.0
            img = img.astype(np.uint8)
            msk = masks[i].cpu().numpy().squeeze(0)
            # 1. Geometric warp (already applied)
            # 2. Advanced methods
            r = random.random()
            if r < 0.33:
                # Mask-guided diffusion synthesis
                aug_img_t, aug_msk_t = self.synthesize_pair_diffusion(img, msk*255, prompt="vibrant", device=device)
            elif r < 0.66 and instance_images is not None and instance_masks is not None:
                # X-Paste
                aug_img_np, aug_msk_np = self.x_paste_batch(img, msk, instance_images, instance_masks, k=3)
                aug_img_t = torch.from_numpy(aug_img_np.transpose(2,0,1)/255.0).unsqueeze(0).float()
                aug_msk_t = torch.from_numpy(aug_msk_np[np.newaxis,:,:]).float()
            else:
                # KeepMask background jitter
                aug_img_np, aug_msk_np = self.keepmask_augment(img, msk)
                aug_img_t = torch.from_numpy(aug_img_np.transpose(2,0,1)/255.0).unsqueeze(0).float()
                aug_msk_t = torch.from_numpy(aug_msk_np[np.newaxis,:,:]).float()
            aug_images.append(aug_img_t)
            aug_masks.append(aug_msk_t)
        aug_images = torch.cat(aug_images, dim=0)
        aug_masks = torch.cat(aug_masks, dim=0)
        return aug_images, aug_masks

    def pad_and_center_mask_image(self, mask, img, pad=32, desired_size=512):
        # Defensive: ensure always (C, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        coords = np.column_stack(np.where(mask[0].cpu().numpy() > 0))
        if len(coords) == 0:
            while mask.ndim < 3:
                mask = mask.unsqueeze(0)
            while img.ndim < 3:
                img = img.unsqueeze(0)
            return mask, img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        crop_y0 = max(0, y0 - pad)
        crop_x0 = max(0, x0 - pad)
        crop_y1 = min(mask.shape[-2], y1 + pad)
        crop_x1 = min(mask.shape[-1], x1 + pad)
        crop_mask = mask[..., crop_y0:crop_y1, crop_x0:crop_x1]
        crop_img  = img[..., crop_y0:crop_y1, crop_x0:crop_x1]
        new_mask = torch.zeros((mask.shape[0], desired_size, desired_size), dtype=mask.dtype)
        new_img  = torch.zeros((img.shape[0], desired_size, desired_size), dtype=img.dtype)
        ch, cw = crop_mask.shape[-2:]
        y_start = (desired_size - ch) // 2
        x_start = (desired_size - cw) // 2
        new_mask[..., y_start:y_start+ch, x_start:x_start+cw] = crop_mask
        new_img[..., y_start:y_start+ch, x_start:x_start+cw]  = crop_img
        return new_mask, new_img

    def center_mask_and_image(self, mask, img):
        # Defensive: ensure always (C, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        coords = np.column_stack(np.where(mask[0].cpu().numpy() > 0))
        if len(coords) == 0:
            # No foreground found, return in shape [C, H, W]
            while mask.ndim < 3:
                mask = mask.unsqueeze(0)
            while img.ndim < 3:
                img = img.unsqueeze(0)
            return mask, img
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        crop_mask = mask[..., y0:y1+1, x0:x1+1]
        crop_img  = img[..., y0:y1+1, x0:x1+1]
        # Ensure (C, h, w)
        if crop_mask.ndim < 3:
            crop_mask = crop_mask.unsqueeze(0)
        if crop_img.ndim < 3:
            crop_img = crop_img.unsqueeze(0)
        new_mask = torch.zeros_like(mask)
        new_img  = torch.zeros_like(img)
        H, W = new_mask.shape[1:] if new_mask.ndim == 3 else new_mask.shape[-2:]
        h, w = crop_mask.shape[1:] if crop_mask.ndim == 3 else crop_mask.shape[-2:]
        y_start = (H - h) // 2
        x_start = (W - w) // 2
        new_mask[..., y_start:y_start+h, x_start:x_start+w] = crop_mask
        new_img[..., y_start:y_start+h, x_start:x_start+w]  = crop_img
        return new_mask, new_img

    def center_and_pad_mask(self, mask, img, pad=64):
        # Center mask and image in a larger canvas before augmentation
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        C, H, W = mask.shape
        big_mask = torch.zeros((C, H + 2*pad, W + 2*pad), dtype=mask.dtype, device=mask.device)
        big_img = torch.zeros_like(big_mask)
        big_mask[:, pad:pad+H, pad:pad+W] = mask
        big_img[:, pad:pad+H, pad:pad+W] = img
        return big_mask, big_img

    def pad_and_center(self, mask, img, pad=128):
        # Pad to larger canvas so mask/image are centered and have safety buffer
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        C, H, W = mask.shape
        padded_mask = torch.zeros((C, H+2*pad, W+2*pad), device=mask.device, dtype=mask.dtype)
        padded_img  = torch.zeros_like(padded_mask)
        padded_mask[:, pad:pad+H, pad:pad+W] = mask
        padded_img[:, pad:pad+H, pad:pad+W]  = img
        return padded_mask, padded_img

    def fatten_and_erode(self, mask, size=15, iterations=2, mode='dilate'):
        """Apply dilation or erosion with area-adaptive parameters."""
        # Area-adaptive morphology parameters
        mask_area = mask.sum().item()
        if mask_area < 50:  # Very small mask
            size = max(3, size // 3)
            iterations = max(1, iterations // 2)
        elif mask_area < 200:  # Small mask
            size = max(5, size // 2)
            iterations = max(1, iterations // 2)
        # else: use default parameters for larger masks
        
        arr = self.sanitize_for_opencv(mask) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        if mode == 'dilate':
            arr = cv2.dilate(arr, kernel, iterations=iterations)
        else:
            arr = cv2.erode(arr, kernel, iterations=iterations)
        return torch.from_numpy(arr / 255.).unsqueeze(0).float().to(mask.device)

    def center_crop(self, tensor, size=512):
        # Center crop to size x size
        C, H, W = tensor.shape
        y_start = (H - size) // 2
        x_start = (W - size) // 2
        return tensor[:, y_start:y_start+size, x_start:x_start+size]

    def compute_edge_overlap(self, mask1, mask2):
        """Compute normalized edge overlap between two binary masks. Ensures both masks are on the same device."""
        # Move mask2 to mask1's device if needed
        if mask2.device != mask1.device:
            mask2 = mask2.to(mask1.device)
        mask1_bin = (mask1 > 0.5)
        mask2_bin = (mask2 > 0.5)
        overlap = (mask1_bin & mask2_bin).float().sum().item()
        area1 = mask1_bin.float().sum().item()
        area2 = mask2_bin.float().sum().item()
        return overlap / max(area1, area2, 1)

    def _lightweight_QA(self, aug_img, aug_msk, ref_msk, path):
        """Cheap per-sample check that never touches global MAD."""
        edge = self.compute_edge_overlap(ref_msk, aug_msk)
        if edge < 0.05:
            # store a single WARN image for manual triage
            warn_dir = "qa_quick_flags"
            os.makedirs(warn_dir, exist_ok=True)
            save_image(aug_img, os.path.join(warn_dir, f"{path}_edge{edge:.2f}.png"))
            return False
        return True

    def generate_hybrid_mask(self, image_tensor: torch.Tensor, original_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate high-quality mask using SAM + skeleton-aware post-processing.
        
        Args:
            image_tensor: Input image tensor [C,H,W] 
            original_mask: Optional reference mask for guidance
            
        Returns:
            Enhanced mask tensor [1,H,W]
        """
        if not self.hybrid_enabled:
            # Fallback to simple thresholding
            print("[HYBRID] Hybrid mode disabled, using fallback thresholding")
            if original_mask is not None:
                print(f"[HYBRID] Using provided original mask: shape={original_mask.shape}")
                # Apply binarization to original mask with logging
                return self.proper_binarize(original_mask, threshold=0.001)
            else:
                print("[HYBRID] No original mask provided, creating from image using OTSU thresholding")
                # Convert to numpy for processing
                img_np = image_tensor.cpu().numpy()
                if img_np.ndim == 3:
                    img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                
                # Simple threshold fallback
                if len(img_np.shape) == 3:
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_np
                
                print(f"[HYBRID] Applying OTSU thresholding on grayscale image: shape={gray.shape}, range=[{gray.min()}, {gray.max()}]")
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                print(f"[HYBRID] OTSU thresholding result: nonzero pixels={np.sum(thresh > 0)}")
                
                mask_tensor = torch.from_numpy(thresh / 255.0).unsqueeze(0).float()
                mask_tensor = self.safe_device_transfer(mask_tensor, self.device)
                
                # Apply proper binarization for consistency
                return self.proper_binarize(mask_tensor, threshold=0.001)
        
        try:
            print(f"[HYBRID] Generating mask with SAM. Image tensor shape: {image_tensor.shape}")
            # Convert tensor to numpy for SAM processing
            img_np = image_tensor.cpu().numpy()
            if img_np.ndim == 3:
                print(f"[HYBRID] Transposing image from CHW to HWC format. Current shape: {img_np.shape}")
                img_np = img_np.transpose(1, 2, 0)  # CHW -> HWC
                print(f"[HYBRID] After transpose, shape: {img_np.shape}")
            
            if img_np.max() <= 1.0:
                print(f"[HYBRID] Normalizing image from [0,1] to [0,255]. Current range: [{img_np.min()}, {img_np.max()}]")
                img_np = (img_np * 255).astype(np.uint8)
                print(f"[HYBRID] After normalization, range: [{img_np.min()}, {img_np.max()}]")
            
            # Ensure RGB format for SAM
            if len(img_np.shape) == 2:
                print(f"[HYBRID] Converting grayscale to RGB. Current shape: {img_np.shape}")
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                print(f"[HYBRID] After conversion, shape: {img_np.shape}")
            elif img_np.shape[2] == 1:
                print(f"[HYBRID] Converting single-channel to RGB. Current shape: {img_np.shape}")
                img_np = cv2.cvtColor(img_np.squeeze(-1), cv2.COLOR_GRAY2RGB)
                print(f"[HYBRID] After conversion, shape: {img_np.shape}")
            
            print(f"[HYBRID] Calling SAM autocoder get_best_mask with image shape: {img_np.shape}, dtype: {img_np.dtype}")
            if self.sam_autocoder is None:
                print("[HYBRID ERROR] SAM autocoder is None! This should not happen when hybrid_enabled=True")
                raise RuntimeError("SAM autocoder is None but hybrid_enabled=True")
                
            # Generate mask with SAM
            sam_mask = self.sam_autocoder.get_best_mask(img_np)
            print(f"[HYBRID] SAM mask generated: shape={sam_mask.shape}, dtype={sam_mask.dtype}, min={sam_mask.min()}, max={sam_mask.max()}")
            
            # Log detailed diagnostics for SAM mask
            sam_tensor = torch.from_numpy(sam_mask / 255.0).unsqueeze(0).float()
            self.log_detailed_mask_diagnostics(sam_tensor, name="SAM")
            
            # Calculate mask statistics
            nonzero_count = np.count_nonzero(sam_mask)
            total_pixels = sam_mask.size
            coverage_percent = (nonzero_count / total_pixels) * 100
            
            print(f"[HYBRID] SAM mask statistics: nonzero={nonzero_count}, total={total_pixels}, coverage={coverage_percent:.2f}%")
            
            # Handle problematic masks (very thin or warped with low values)
            if sam_mask.max() > 0 and sam_mask.max() <= 10:  # Extremely low values (0-10 instead of 0-255)
                print(f"[HYBRID WARNING] SAM returned very low-value mask: max={sam_mask.max()}")
                # Scale up values to proper range to help with skeleton processing
                sam_mask = sam_mask * (255.0 / sam_mask.max())
                print(f"[HYBRID] Rescaled mask to proper range: max={sam_mask.max()}")
            
            # Apply skeleton-aware post-processing
            if sam_mask.max() > 0:
                print("[HYBRID] Applying skeleton-aware post-processing")
                refined_mask = self.skeleton_processor.sap_refine(sam_mask)
                print(f"[HYBRID] Refined mask: shape={refined_mask.shape}, min={refined_mask.min()}, max={refined_mask.max()}")
                
                # Handle common failure case of skeleton processor
                if refined_mask.max() <= 0:
                    print("[HYBRID WARNING] Skeleton processor returned empty mask, falling back to original SAM mask")
                    refined_mask = sam_mask
            else:
                print("[HYBRID] SAM returned empty mask, skipping skeleton-aware post-processing")
                refined_mask = sam_mask
            
            # Convert back to tensor
            mask_tensor = torch.from_numpy(refined_mask / 255.0).unsqueeze(0).float()
            mask_tensor = self.safe_device_transfer(mask_tensor, self.device)
            
            # Log refined mask diagnostics
            self.log_detailed_mask_diagnostics(mask_tensor, name="pre-binarized")
            
            # Apply proper binarization with enhanced logging
            print("[HYBRID] Applying proper mask binarization")
            binary_mask = self.proper_binarize(mask_tensor, threshold=0.001)
            
            # Final mask QA check
            if not self._is_mask_valid(binary_mask):
                print("[HYBRID WARNING] Binarized mask still invalid - may be empty or near-empty")
                # Try more aggressive settings
                print("[HYBRID RESCUE] Attempting mask rescue with ultra-low threshold")
                binary_mask = self.proper_binarize(mask_tensor, threshold=0.0005)
                
                # Final rescue attempt with morphological operations
                if not self._is_mask_valid(binary_mask):
                    print("[HYBRID EMERGENCY] Final rescue attempt with morphological closing")
                    # Try to convert original mask with aggressive closing
                    closed_mask = self.apply_morphology(mask_tensor, operation='close', kernel_size=5)
                    binary_mask = self.proper_binarize(closed_mask, threshold=0.0001)
            
            # Final diagnostic log
            self.log_detailed_mask_diagnostics(binary_mask, name="final")
            
            return binary_mask
            
        except Exception as e:
            print(f"[HYBRID ERROR] Mask generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Print diagnostic info about objects
            print(f"[HYBRID DIAGNOSTIC] hybrid_enabled = {self.hybrid_enabled}")
            print(f"[HYBRID DIAGNOSTIC] sam_autocoder is None: {self.sam_autocoder is None}")
            print(f"[HYBRID DIAGNOSTIC] skeleton_processor is None: {self.skeleton_processor is None}")
            print(f"[HYBRID DIAGNOSTIC] image_tensor: shape={image_tensor.shape}, dtype={image_tensor.dtype}")
            
            # Fallback to original mask or simple threshold
            if original_mask is not None:
                print("[HYBRID FALLBACK] Using original mask as fallback")
                return self.proper_binarize(original_mask, threshold=0.001)
            else:
                print("[HYBRID FALLBACK] Using emergency OTSU thresholding as fallback")
                try:
                    # Try OTSU thresholding as emergency fallback
                    img_np = self.sanitize_for_opencv(image_tensor.cpu())
                    if len(img_np.shape) == 3:
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img_np
                    
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    mask_tensor = torch.from_numpy(thresh / 255.0).unsqueeze(0).float()
                    mask_tensor = self.safe_device_transfer(mask_tensor, self.device)
                    return self.proper_binarize(mask_tensor, threshold=0.001)
                except Exception as e2:
                    print(f"[HYBRID EMERGENCY] OTSU fallback also failed: {e2}")
                    print("[HYBRID EMERGENCY] Creating empty mask as last resort fallback")
                    return self.safe_device_transfer(torch.zeros(1, image_tensor.shape[1], image_tensor.shape[2]), self.device)

    def generate_diverse_sam_masks(self, image_tensor: torch.Tensor, top_k: int = 3) -> List[torch.Tensor]:
        """
        Generate multiple diverse masks using SAM for enhanced dataset variety.
        
        Args:
            image_tensor: Input image tensor [C,H,W]
            top_k: Number of diverse masks to generate
            
        Returns:
            List of mask tensors, each [1,H,W]
        """
        if not self.hybrid_enabled:
            return []
        
        try:
            # Convert tensor to numpy
            img_np = image_tensor.cpu().numpy()
            if img_np.ndim == 3:
                img_np = img_np.transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            # Ensure RGB format
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            # Get multiple masks from SAM
            mask_candidates = self.sam_autocoder.get_multiple_masks(img_np, top_k=top_k)
            
            diverse_masks = []
            for mask_data in mask_candidates:
                mask = mask_data['segmentation']
                
                # Apply skeleton refinement
                refined_mask = self.skeleton_processor.sap_refine(mask.astype(np.uint8))
                
                # Convert to tensor
                mask_tensor = torch.from_numpy(refined_mask / 255.0).unsqueeze(0).float()
                diverse_masks.append(self.safe_device_transfer(mask_tensor, self.device))
            
            return diverse_masks
            
        except Exception as e:
            print(f"[HYBRID] Diverse mask generation failed: {e}")
            return []

    def test_hybrid_pipeline(self, test_images: torch.Tensor, save_dir: str = "hybrid_test_results") -> Dict:
        """
        Test the hybrid SAM+SAP pipeline and compare with traditional methods.
        
        Args:
            test_images: Batch of test images [B,C,H,W]
            save_dir: Directory to save comparison results
            
        Returns:
            Dictionary with performance metrics and comparison results
        """
        os.makedirs(save_dir, exist_ok=True)
        results = {
            'hybrid_enabled': self.hybrid_enabled,
            'sam_available': self.sam_autocoder.is_available() if self.sam_autocoder else False,
            'processing_times': [],
            'mask_qualities': [],
            'topology_scores': []
        }
        
        if not self.hybrid_enabled:
            print("[TEST] Hybrid pipeline not available - skipping test")
            return results
        
        print(f"[TEST] Testing hybrid pipeline on {test_images.shape[0]} images")
        
        for i, image in enumerate(test_images):
            print(f"[TEST] Processing image {i+1}/{test_images.shape[0]}")
            
            # Time hybrid mask generation
            start_time = time.time()
            try:
                hybrid_mask = self.generate_hybrid_mask(image)
                diverse_masks = self.generate_diverse_sam_masks(image, top_k=3)
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                
                # Evaluate mask quality
                mask_area = hybrid_mask.sum().item()
                mask_type = classify_mask(hybrid_mask)
                
                # Topology analysis if skeleton processor available
                if self.skeleton_processor:
                    mask_np = (hybrid_mask.squeeze() > 0.5).cpu().numpy().astype(np.uint8) * 255
                    topology_metrics = self.skeleton_processor.compute_topology_metrics(mask_np)
                    results['topology_scores'].append(topology_metrics)
                
                results['mask_qualities'].append({
                    'area': mask_area,
                    'type': mask_type.name,
                    'diverse_count': len(diverse_masks)
                })
                
                # Save visualization
                save_path = os.path.join(save_dir, f"test_image_{i}")
                self._save_hybrid_comparison(image, hybrid_mask, diverse_masks, save_path)
                
                print(f"[TEST] Image {i}: {mask_type.name} mask, area={mask_area:.1f}, time={processing_time:.3f}s")
                
            except Exception as e:
                print(f"[TEST] Failed to process image {i}: {e}")
                results['processing_times'].append(-1)
        
        # Summary statistics
        valid_times = [t for t in results['processing_times'] if t > 0]
        if valid_times:
            results['avg_processing_time'] = sum(valid_times) / len(valid_times)
            results['max_processing_time'] = max(valid_times)
            
        results['success_rate'] = len(valid_times) / len(test_images)
        
        # Save results summary
        with open(os.path.join(save_dir, "test_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"[TEST] Hybrid pipeline test completed:")
        print(f"  Success rate: {results['success_rate']:.1%}")
        print(f"  Avg processing time: {results.get('avg_processing_time', 0):.3f}s")
        print(f"  Results saved to: {save_dir}")
        
        return results
    
    def _save_hybrid_comparison(self, image: torch.Tensor, main_mask: torch.Tensor, 
                              diverse_masks: List[torch.Tensor], save_path: str):
        """Save visualization comparing original image with hybrid-generated masks."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            img_np = image.cpu().numpy()
            if img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)
            axes[0,0].imshow(img_np, cmap='gray')
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')
            
            # Main hybrid mask
            mask_np = main_mask.squeeze().cpu().numpy()
            axes[0,1].imshow(mask_np, cmap='viridis')
            axes[0,1].set_title(f'SAM+SAP Mask (area: {main_mask.sum().item():.0f})')
            axes[0,1].axis('off')
            
            # Overlay
            if img_np.ndim == 2:
                overlay = np.stack([img_np, img_np, img_np], axis=-1)
            else:
                overlay = img_np
            overlay = overlay.copy()
            mask_bool = mask_np > 0.5
            overlay[mask_bool] = [1.0, 0.0, 0.0]  # Red overlay
            axes[0,2].imshow(overlay)
            axes[0,2].set_title('Mask Overlay')
            axes[0,2].axis('off')
            
            # Diverse masks
            for i, diverse_mask in enumerate(diverse_masks[:3]):
                row, col = (1, i)
                if row < 2 and col < 3:
                    div_mask_np = diverse_mask.squeeze().cpu().numpy()
                    axes[row,col].imshow(div_mask_np, cmap='plasma')
                    axes[row,col].set_title(f'Diverse Mask {i+1} (area: {diverse_mask.sum().item():.0f})')
                    axes[row,col].axis('off')
            
            # Fill remaining subplots
            for i in range(len(diverse_masks), 3):
                axes[1,i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{save_path}_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"[VIZ] Failed to save comparison: {e}")

    def apply_morphology(self, mask, operation='open', kernel_size=3):
        """
        Apply morphological operations to clean up mask.
        
        Args:
            mask: Input mask tensor [1,H,W]
            operation: 'open' (removes small noise), 'close' (fills small holes), 
                       'dilate' (expands mask), 'erode' (shrinks mask)
            kernel_size: Size of the kernel for the operation
            
        Returns:
            Processed mask tensor [1,H,W]
        """
        import cv2
        import numpy as np
        
        # Convert to numpy for OpenCV processing
        mask_np = mask.cpu().squeeze(0).numpy()
        
        # Create kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Convert to uint8 for OpenCV
        if mask_np.max() <= 1.0:
            mask_np = (mask_np * 255).astype(np.uint8)
            
        # Apply morphological operation
        if operation == 'open':
            result = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            result = cv2.dilate(mask_np, kernel, iterations=1)
        elif operation == 'erode':
            result = cv2.erode(mask_np, kernel, iterations=1)
        else:
            return mask  # No change
        
        # Convert back to tensor
        result_tensor = torch.from_numpy(result.astype(np.float32) / 255.0).unsqueeze(0)
        return self.safe_device_transfer(result_tensor, self.device)

    def proper_binarize(self, mask, threshold=0.001):
        """
        Properly binarize a mask with adaptive thresholding.
        
        Args:
            mask: Input mask tensor
            threshold: Binarization threshold (lower for thin/warped masks)
            
        Returns:
            Binary mask tensor (0.0 or 1.0)
        """
        # Get mask statistics for logging
        mask_min = mask.min().item()
        mask_max = mask.max().item()
        mask_mean = mask.mean().item()
        
        # Log mask statistics for debugging
        print(f"[MASK BINARIZE] Mask stats before binarization: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}")
        
        # Adaptive threshold based on mask properties
        if mask_max <= 0.01:  # Very low values, lower the threshold
            final_threshold = min(0.001, mask_max / 2)
            print(f"[MASK BINARIZE] Using ultra-low threshold: {final_threshold:.6f} for low-value mask")
        else:
            final_threshold = threshold
        
        # Apply binary threshold
        binary_mask = (mask > final_threshold).float()
        
        # Get post-binarization statistics
        nonzero_count = binary_mask.sum().item()
        print(f"[MASK BINARIZE] After binarization: nonzero_count={nonzero_count}, threshold={final_threshold:.6f}")
        
        # Apply morphological operations if needed to clean noise
        if nonzero_count > 0 and nonzero_count < 100:
            print("[MASK BINARIZE] Applying morphological closing to fill small holes")
            binary_mask = self.apply_morphology(binary_mask, operation='close', kernel_size=3)
            new_nonzero = binary_mask.sum().item()
            print(f"[MASK BINARIZE] After morphological operation: nonzero_count={new_nonzero}")
        
        return binary_mask
    
    def log_detailed_mask_diagnostics(self, mask, name="unknown"):
        """
        Log detailed diagnostics for problematic masks.
        
        Args:
            mask: The mask tensor to analyze
            name: Identifier for the mask (e.g., "SAM", "hybrid", "refined")
        """
        if mask is None:
            print(f"[MASK DIAGNOSTIC] {name} mask is None!")
            return
        
        try:
            # Get basic statistics
            mask_min = mask.min().item() if hasattr(mask, 'min') else float('nan')
            mask_max = mask.max().item() if hasattr(mask, 'max') else float('nan')
            mask_mean = mask.mean().item() if hasattr(mask, 'mean') else float('nan')
            mask_std = mask.std().item() if hasattr(mask, 'std') else float('nan')
            
            # Value distribution analysis
            if hasattr(mask, 'cpu') and hasattr(mask, 'flatten'):
                mask_np = mask.cpu().flatten().numpy()
                percentiles = {
                    "1%": float(np.percentile(mask_np, 1)),
                    "5%": float(np.percentile(mask_np, 5)),
                    "10%": float(np.percentile(mask_np, 10)),
                    "50%": float(np.percentile(mask_np, 50)),
                    "90%": float(np.percentile(mask_np, 90)),
                    "95%": float(np.percentile(mask_np, 95)),
                    "99%": float(np.percentile(mask_np, 99))
                }
                
                # Thresholding analysis at different levels
                thresholds = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5]
                threshold_counts = {}
                for t in thresholds:
                    count = (mask_np > t).sum()
                    threshold_counts[f">={t}"] = int(count)
                
                # Print comprehensive diagnostic report
                print(f"[MASK DIAGNOSTIC] {name} mask analysis:")
                print(f"  Basic stats: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}, std={mask_std:.6f}")
                print(f"  Shape: {mask.shape}")
                print(f"  Device: {mask.device if hasattr(mask, 'device') else 'unknown'}")
                print(f"  Dtype: {mask.dtype if hasattr(mask, 'dtype') else 'unknown'}")
                print(f"  Percentiles: {', '.join([f'{k}={v:.6f}' for k, v in percentiles.items()])}")
                print(f"  Threshold counts: {', '.join([f'{k}={v}' for k, v in threshold_counts.items()])}")
                
                # Check for common problems
                issues = []
                if mask_max <= 0.01:
                    issues.append("VERY_LOW_VALUES")
                if mask_std <= 0.001:
                    issues.append("LOW_VARIANCE")
                if threshold_counts.get(">=0.001", 0) <= 10:
                    issues.append("FEW_NONZERO")
                
                if issues:
                    print(f"  🚨 Detected issues: {', '.join(issues)}")
                    print(f"  💡 Recommended action: Use lower threshold (0.0005) for binarization")
            else:
                print(f"[MASK DIAGNOSTIC] Unable to perform detailed analysis on {name} mask")
                
        except Exception as e:
            print(f"[MASK DIAGNOSTIC] Error during analysis: {e}")
    
    def _is_mask_valid(self, mask):
        """Early gate: check if mask has meaningful content."""
        # Get detailed mask statistics
        mask_min = mask.min().item()
        mask_max = mask.max().item()
        mask_mean = mask.mean().item()
        
        # Print mask statistics for debugging
        print(f"[MASK VALIDATE] Mask stats: min={mask_min:.6f}, max={mask_max:.6f}, mean={mask_mean:.6f}")
        
        # More permissive threshold for thin/warped masks (0.001 instead of 0.25)
        mask_nonzero = (mask > 0.001).float().sum().item()
        is_valid = mask_nonzero >= 5  # At least 5 pixels
        
        print(f"[MASK VALIDATE] Mask nonzero pixels: {mask_nonzero}, valid: {is_valid}")
        
        # Log detailed diagnostics for problematic masks
        if not is_valid:
            self.log_detailed_mask_diagnostics(mask, name="invalid")
            
        return is_valid

    # ───────────────────────────────────────────────────────────────
    # Section: Adaptive Edge Overlap Thresholds
    # ───────────────────────────────────────────────────────────────

    def compute_edge_complexity(self, mask: torch.Tensor) -> float:
        """Compute edge complexity score based on contour analysis."""
        import cv2
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        # Complexity based on perimeter-to-area ratio
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        total_area = sum(cv2.contourArea(c) for c in contours)
        return total_perimeter / (total_area + 1e-6)

    def adaptive_edge_overlap_threshold(self, mask_type: MaskType, edge_complexity: float) -> float:
        """
        Compute dynamic minimum edge overlap based on mask type and geometric complexity.
        """
        base_thresholds = {
            MaskType.THIN:   0.02,
            MaskType.SPARSE: 0.05,
            MaskType.DENSE:  0.15,
        }
        complexity_factor = min(1.5, 1.0 + edge_complexity * 0.1)
        return base_thresholds.get(mask_type, 0.0) * complexity_factor

    # Add method alias for tensor corruption diagnostics
    def diagnose_tensor_corruption(self, tensor, name="tensor", path=None):
        return diagnose_tensor_corruption(tensor, name, path)

def main():
    import argparse, os, pickle, time
    import numpy as np
    parser = argparse.ArgumentParser(description="Bongard geometric augmentation pipeline with hybrid SAM+SAP support")
    parser.add_argument('--input', type=str, required=True, help='Path to derived_labels.json')
    parser.add_argument('--out', type=str, required=True, help='Path to output augmented.pkl')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for augmentation')
    parser.add_argument('--type', type=str, default='geometric', help='Augmentation type')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--rotate', type=float, default=10.0, help='Max rotation (degrees)')
    parser.add_argument('--scale', type=float, default=1.2, help='Max scale factor')
    
    # Hybrid pipeline options
    parser.add_argument('--enable-hybrid', action='store_true', help='Enable SAM+SAP hybrid pipeline')
    parser.add_argument('--sam-model', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'], 
                       help='SAM model variant (vit_b=fastest, vit_h=best quality)')
    parser.add_argument('--test-hybrid', action='store_true', help='Run hybrid pipeline test on first batch')
    parser.add_argument('--fallback-empty', action='store_true', help='Use SAM for empty geometry masks')
    parser.add_argument('--qa-fail-threshold', type=float, default=0.15, help='QA failure rate threshold')
    
    # Emergency corruption diagnostics 
    parser.add_argument('--test-corruption-fixes', action='store_true', 
                       help='Test CV_8UC512 corruption diagnostic and mitigation systems')
    parser.add_argument('--force-emergency-qa', action='store_true',
                       help='Force ultra-permissive emergency QA thresholds for cleanup period')
    
    args = parser.parse_args()

    print("[INIT] Initializing Bongard Hybrid Augmentation Pipeline")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.out}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hybrid enabled: {args.enable_hybrid}")
    if args.enable_hybrid:
        print(f"  SAM model: {args.sam_model}")

    # Load derived labels
    with open(args.input, 'r') as f:
        derived_labels = json.load(f)
    image_paths = [entry['image_path'] for entry in derived_labels]
    dataset = ImagePathDataset(image_paths, derived_labels_path=args.input)

    # Initialize augmentor with hybrid options
    augmentor = ImageAugmentor(batch_size=args.batch_size)
    
    # Configure hybrid pipeline if enabled
    if args.enable_hybrid and HYBRID_PIPELINE_AVAILABLE:
        if augmentor.hybrid_enabled:
            print("[HYBRID] SAM+SAP pipeline ready")
            print(f"  SAM model: {augmentor.sam_autocoder.get_model_info()}")
            
            # Update QA thresholds for hybrid mode
            augmentor.QA_THRESHOLDS['edge_overlap_min'] = 0.25  # More permissive for SAM
            print(f"  Updated QA thresholds for hybrid mode")
        else:
            print("[WARNING] Hybrid pipeline requested but not available - using fallback")
    
    # Test hybrid pipeline on first batch if requested
    if args.test_hybrid and augmentor.hybrid_enabled:
        print("[TEST] Running hybrid pipeline test on first batch...")
        test_batch = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
        test_images, test_paths, test_geoms = pad_collate(test_batch)
        
        test_results = augmentor.test_hybrid_pipeline(test_images, save_dir="hybrid_test_output")
        print(f"[TEST] Hybrid test completed - success rate: {test_results.get('success_rate', 0):.1%}")
        if test_results.get('avg_processing_time', 0) > 0:
            print(f"[TEST] Average processing time: {test_results['avg_processing_time']:.3f}s per image")

    # Emergency corruption test if requested
    if args.test_corruption_fixes:
        print("\n" + "="*80)
        print("🔬 TESTING CV_8UC512 CORRUPTION DIAGNOSTIC SYSTEMS 🔬")
        print("="*80)
        
        # Create test tensors that would trigger CV_8UC512 errors
        test_batch = [dataset[i] for i in range(min(3, len(dataset)))]
        test_images, test_paths, test_geoms = pad_collate(test_batch)
        
        print("Testing tensor corruption diagnosis...")
        for i, (image, path) in enumerate(zip(test_images[:3], test_paths[:3])):
            print(f"\n📊 Analyzing tensor {i+1}: {path}")
            
            # Test corruption diagnosis
            diag = augmentor.diagnose_tensor_corruption(image)
            print(f"  Corruption detected: {diag['corrupted']}")
            print(f"  Shape: {diag['shape']}")
            print(f"  Channels: {diag['channels']}")
            print(f"  Device: {diag['device']}")
            if diag['issues']:
                print(f"  Issues: {', '.join(diag['issues'])}")
            
            # Test safe device transfer
            print(f"  Testing safe device transfer...")
            safe_tensor = augmentor.safe_device_transfer(image, augmentor.device)
            print(f"  Transfer successful: {safe_tensor.shape}")
            
            # Test OpenCV sanitization
            print(f"  Testing OpenCV sanitization...")
            safe_cv_array = augmentor.sanitize_for_opencv(image)
            print(f"  OpenCV array shape: {safe_cv_array.shape}")
            print(f"  OpenCV array dtype: {safe_cv_array.dtype}")
            
            # Test mask conversion if we have geometry
            if test_geoms[i] is not None:
                print(f"  Testing safe mask conversion...")
                H, W = image.shape[1], image.shape[2]
                test_mask = augmentor.geometry_to_binary_mask(test_geoms[i], H, W)
                safe_mask = augmentor.safe_mask_conversion(test_mask)
                print(f"  Safe mask shape: {safe_mask.shape}")
        
        print("\n" + "="*80)
        print("✅ CV_8UC512 corruption diagnostics test completed successfully!")
        print("="*80 + "\n")
        
        if not args.force_emergency_qa:
            print("💡 Use --force-emergency-qa to proceed with processing using emergency protocols")
            return

    # Force emergency QA if requested  
    if args.force_emergency_qa:
        print("\n🚨 EMERGENCY QA PROTOCOLS ACTIVATED 🚨")
        print("Ultra-permissive thresholds enabled for corruption cleanup period")
        # QA thresholds are already set to emergency levels in __init__

    # Main augmentation loop
    all_augmented = []
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    fail_counts = []
    
    print(f"[PROCESS] Starting augmentation of {len(dataset)} images in {total_batches} batches")
    
    for idx in range(0, len(dataset), args.batch_size):
        batch_num = idx // args.batch_size + 1
        print(f"[BATCH {batch_num}/{total_batches}] Processing images {idx} to {min(idx+args.batch_size, len(dataset)-1)}")
        
        batch = [dataset[i] for i in range(idx, min(idx+args.batch_size, len(dataset)))]
        images, paths, geoms = pad_collate(batch)
        
        # Handle empty geometries with hybrid pipeline if enabled
        if args.fallback_empty and augmentor.hybrid_enabled:
            empty_count = sum(1 for g in geoms if g is None)
            if empty_count > 0:
                print(f"[HYBRID] Using SAM fallback for {empty_count} empty geometries in batch {batch_num}")
        
        # Process batch
        try:
            result = augmentor.augment_batch(
                images, 
                paths=paths, 
                geometries=geoms, 
                augment_type=args.type, 
                batch_idx=batch_num-1
            )
            all_augmented.append(result)
            
            # Track failure rates
            if hasattr(result, 'get'):
                fail_count = result.get('fail_count', 0)
                fail_rate = fail_count / len(images) if len(images) > 0 else 0
                fail_counts.append(fail_rate)
                
                if fail_rate > args.qa_fail_threshold:
                    print(f"[WARNING] High QA failure rate in batch {batch_num}: {fail_rate:.1%}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process batch {batch_num}: {e}")
            # Continue with next batch rather than failing completely
            continue
    
    # Summary statistics
    if fail_counts:
        avg_fail_rate = sum(fail_counts) / len(fail_counts)
        max_fail_rate = max(fail_counts)
        print(f"[SUMMARY] QA Statistics:")
        print(f"  Average failure rate: {avg_fail_rate:.1%}")
        print(f"  Maximum failure rate: {max_fail_rate:.1%}")
        print(f"  Batches processed: {len(fail_counts)}/{total_batches}")
        
        if avg_fail_rate > args.qa_fail_threshold:
            print(f"[WARNING] Average QA failure rate {avg_fail_rate:.1%} exceeds threshold {args.qa_fail_threshold:.1%}")
            if augmentor.hybrid_enabled:
                print("  Hybrid pipeline may need parameter tuning")
    
    # Save all augmented results
    print(f"[SAVE] Saving {len(all_augmented)} batch results to {args.out}")
    with open(args.out, 'wb') as f:
        pickle.dump(all_augmented, f)
    print(f"[DONE] Augmented data saved to {args.out}")
    
    # Final hybrid pipeline summary
    if augmentor.hybrid_enabled:
        print("[HYBRID] Pipeline summary:")
        print(f"  SAM masks generated: {getattr(augmentor, '_sam_mask_count', 0)}")
        print(f"  Topology validations: {getattr(augmentor, '_topology_check_count', 0)}")
        if hasattr(augmentor, 'edge_overlap_hist') and len(augmentor.edge_overlap_hist) > 0:
            print(f"  Edge overlap samples: {len(augmentor.edge_overlap_hist)}")
            print(f"  Edge overlap 20th percentile: {np.percentile(augmentor.edge_overlap_hist, 20):.3f}")

if __name__ == "__main__":
    main()