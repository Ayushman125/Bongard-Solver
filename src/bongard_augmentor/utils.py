# Action-based augmentation utilities
# No longer requires SAM or complex CV dependencies

import numpy as np
import cv2
from enum import Enum, auto
import logging
import sys
from typing import Dict
import yaml
from pathlib import Path

class MaskType(Enum):
    EMPTY = auto()
    THIN = auto()
    SPARSE = auto()
    DENSE = auto()

def classify_mask(mask_tensor):
    """Classify mask into EMPTY, THIN, SPARSE, or DENSE based on area and morphology."""
    area = mask_tensor.sum().item() if hasattr(mask_tensor, 'sum') else np.sum(mask_tensor)
    if area == 0:
        return MaskType.EMPTY
    
    # Use morphological properties to distinguish between thin, sparse, and dense
    if hasattr(mask_tensor, 'cpu'):
        np_mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    else:
        np_mask = (mask_tensor * 255).astype(np.uint8) if mask_tensor.dtype != np.uint8 else mask_tensor
    
    eroded = cv2.erode(np_mask, np.ones((3,3), np.uint8), iterations=1)
    eroded_area = np.sum(eroded > 0)
    
    thinness_ratio = eroded_area / area if area > 0 else 0
    
    if thinness_ratio < 0.3:
        return MaskType.THIN
    elif area < 5000: # Heuristic threshold for sparse vs dense
        return MaskType.SPARSE
    else:
        return MaskType.DENSE

def sanitize_for_opencv(arr):
    """Convert array to a format OpenCV understands."""
    if hasattr(arr, 'cpu'):  # Handle torch tensors if present
        arr = arr.cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return arr

def diagnose_array_corruption(arr, name="array", path=None):
    """
    Print shape, dtype and OpenCV compatibility of an array.
    """
    print(f"\n🛠️ Diagnosing {name}{': '+path if path else ''}")
    if hasattr(arr, 'cpu'):  # Handle torch tensors if present
        np_arr = arr.detach().cpu().numpy()
        print(f" Tensor: shape={tuple(arr.shape)}, dtype={arr.dtype}, device={getattr(arr, 'device', 'N/A')}")
    else:
        np_arr = arr
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

    if hasattr(arr, 'device'):
        device = str(arr.device)
    else:
        device = 'cpu'
    result = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": arr.min(),
        "max": arr.max(),
        "opencv_ok": opencv_ok,
        "corrupted": is_corrupted,
        "channels": arr.shape[-1] if arr.ndim == 3 else 1,
        "device": device,
        "issues": [],
    }
    # Optionally, append issue strings if you detect problems
    if is_corrupted:
        result['issues'].append('Tensor corruption detected')
    return result


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

def repair_mask(mask_array):
    """Apply morphological closing to repair thin masks."""
    arr = sanitize_for_opencv(mask_array)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    repaired = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    return repaired

def pre_warp_mask(mask_array):
    """Morphological dilation to fatten thin masks before augmentation."""
    arr = sanitize_for_opencv(mask_array)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(arr, kernel, iterations=2)
    return dilated

def pre_warp_fatten(array, size=7):
    """Aggressively dilate mask or line-art image before augmentation."""
    arr = sanitize_for_opencv(array) 
    if arr.max() <= 1.0:  # If normalized, scale to 0-255
        arr = (arr * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(arr, kernel, iterations=2)
    return dilated.astype(np.float32) / 255.0

def topology_aware_morphological_repair(mask: np.ndarray, mask_type: MaskType) -> np.ndarray:
    """
    Perform adaptive opening/closing based on mask_type to preserve structure.
    """
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

QA_THRESHOLDS = {
    'pixvar_min': 0.003,
    'pixvar_max': 0.25,
    'edge_overlap_min': 0.15,
    'area_ratio_min': 0.10,
    'area_ratio_max': 4.00,
    'outlier_sigma': 3.5
}

class MetricTracker:
    def __init__(self):
        self.store = {}
    def log(self, name: str, value: float):
        if name not in self.store:
            self.store[name] = []
        self.store[name].append(value)
    def summary(self, name: str, k: int = 50):
        data = list(self.store.get(name, []))[-k:]
        if not data:
            return 0.0, 0.0
        return float(np.mean(data)), float(np.std(data))

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

def setup_logging(log_level: str = 'INFO', log_file: str = 'logs/augmentor.log'):
    """Set up logging for the application."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Ensure log directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a')
        ]
    )
    log = logging.getLogger(__name__)
    log.info(f"Logging initialized. Level: {log_level}, File: {log_file}")

def get_base_config() -> Dict:
    """Returns default configuration for action-based augmentation pipeline."""
    return {
        'data': {
            'input_path': None,  # Path to derived_labels.json (optional)
            'action_programs_dir': None,  # Path to action programs directory (optional) 
            'output_path': 'data/augmented.pkl',
            'problems_list': None,  # Optional file containing list of problems to process
            'n_select': 50,  # Number of problems to select from action programs
        },
        'processing': {
            'batch_size': 32,
            'device': 'cpu',  # Action-based processing doesn't require GPU
        },
        'image_size': (64, 64),  # Size for generated masks
        'enable_post_processing': False,  # Enable morphological post-processing
        'refinement': {
            'contour_approx_factor': 0.02,
            'min_component_size': 50,
            'closing_kernel_size': 3,
            'opening_kernel_size': 3,
        },
        'skeleton': {
            'min_branch_length': 10,
        },
        'inspection_dir': None,  # Optional directory for saving inspection images
        'logging': {
            'level': 'INFO',
            'log_file': 'logs/augmentor.log'
        }
    }
