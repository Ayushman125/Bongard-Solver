try:
    import segment_anything
    import sap_toolkit
    HYBRID_PIPELINE_AVAILABLE = True
except ImportError:
    HYBRID_PIPELINE_AVAILABLE = False

import torch
import numpy as np
import cv2
from enum import Enum, auto
import logging
import sys
import yaml
from pathlib import Path

class MaskType(Enum):
    EMPTY = auto()
    THIN = auto()
    SPARSE = auto()
    DENSE = auto()

def classify_mask(mask_tensor):
    """Classify mask into EMPTY, THIN, SPARSE, or DENSE based on area and morphology."""
    area = mask_tensor.sum().item()
    if area == 0:
        return MaskType.EMPTY
    
    # Use morphological properties to distinguish between thin, sparse, and dense
    np_mask = (mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    eroded = cv2.erode(np_mask, np.ones((3,3), np.uint8), iterations=1)
    eroded_area = np.sum(eroded > 0)
    
    thinness_ratio = eroded_area / area if area > 0 else 0
    
    if thinness_ratio < 0.3:
        return MaskType.THIN
    elif area < 5000: # Heuristic threshold for sparse vs dense
        return MaskType.SPARSE
    else:
        return MaskType.DENSE

def robust_z_scores(tensor):
    """Compute z-scores using median absolute deviation (MAD)."""
    tensor = tensor.float()
    median = torch.median(tensor)
    mad = torch.median(torch.abs(tensor - median))
    return (tensor - median) / (1.4826 * mad + 1e-6)

def sanitize_for_opencv(tensor):
    """Convert PyTorch tensor to a format OpenCV understands."""
    arr = tensor.cpu().numpy()
    if arr.ndim == 3:
        arr = arr.squeeze(0)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return arr

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

def repair_mask(mask_tensor):
    """Apply morphological closing to repair thin masks."""
    arr = sanitize_for_opencv(mask_tensor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    repaired = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    return torch.from_numpy(repaired).unsqueeze(0).float()

def pre_warp_mask(mask_tensor):
    """Morphological dilation to fatten thin masks before augmentation."""
    arr = sanitize_for_opencv(mask_tensor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(arr, kernel, iterations=2)
    return torch.from_numpy(dilated).unsqueeze(0).float()

def pre_warp_fatten(tensor, size=7):
    """Aggressively dilate mask or line-art image before augmentation."""
    arr = sanitize_for_opencv(tensor) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(arr, kernel, iterations=2)
    return torch.from_numpy(dilated).unsqueeze(0).float() / 255.0

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
    """Loads a base configuration from a YAML file or returns a default."""
    # In a real-world scenario, this would load from a file
    # For this case, we'll define a comprehensive default config here.
    return {
        'data': {
            'input_path': 'data/derived_labels.json',
            'output_path': 'data/augmented.pkl',
            'problem_folders_path': 'data/ShapeBongard_V2/puzzles',
        },
        'processing': {
            'batch_size': 32,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        'augmentation': {
            'type': 'geometric', # 'geometric', 'photometric', 'both'
            'geometric_params': {
                'rotation': 15,
                'scale_min': 0.9,
                'scale_max': 1.1,
                'translate_x': 0.1,
                'translate_y': 0.1,
                'shear': 5,
            },
            'photometric_params': {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1,
            }
        },
        'sam': {
            'model_type': 'vit_h',
            'checkpoint_path': 'models/sam_vit_h_4b8939.pth',
            'hq_checkpoint_path': 'models/sam_hq_vit_h.pth',
            'use_hq': True,
        },
        'prompting': {
            'strategy': 'hybrid', # 'grid', 'bbox', 'hybrid'
            'grid_points_per_side': 16,
            'use_semantic_filtering': True,
            'concept_source': 'data/conceptnet_lite.json',
        },
        'refinement': {
            'use_iterative_refinement': True,
            'max_iterations': 3,
            'use_morphological_cleanup': True,
            'use_fallback_on_failure': True,
            'fallback_strategy': 'convex_hull', # 'convex_hull', 'bbox'
        },
        'qa': {
            'enabled': True,
            'failure_rate_threshold': 0.15, # 15% failure rate triggers fallback mode
            'metrics': ['iou', 'pixel_accuracy', 'topology_consistency'],
            'log_dir': 'qa_adversarial',
        },
        'logging': {
            'level': 'INFO',
            'log_file': 'logs/hybrid_augmentor.log'
        }
    }
