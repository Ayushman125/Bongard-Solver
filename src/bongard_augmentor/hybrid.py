# ------------------------------------------------------------------
# Hybrid Augmentation Pipeline Components
# Consolidates SAM and Skeleton-Aware Processing
# ------------------------------------------------------------------

import os
import torch
import numpy as np
import cv2
from typing import Optional, Dict, List, Union, Tuple
import logging
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, remove_small_objects

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

# ==================================================================
# Section: Segment Anything Model (SAM) Wrapper
# ==================================================================

class SAMAutoCoder:
    """
    Professional SAM wrapper for prompt-free mask generation.
    Handles model loading, caching, and graceful fallbacks.
    """
    
    def __init__(self, 
                 model_type: str = "vit_b", 
                 ckpt_path: Optional[str] = None,
                 device: str = "cuda", 
                 points_per_side: int = 32,
                 cache_dir: str = "~/.cache/sam"):
        """
        Initialize SAM autocoder with professional error handling.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        
        self.sam = None
        self.generator = None
        self.is_initialized = False
        
        if SAM_AVAILABLE:
            try:
                self._initialize_sam(ckpt_path, points_per_side)
            except Exception as e:
                logging.error(f"SAM initialization failed: {e}")
                self.is_initialized = False
        else:
            logging.warning("SAM not available - using fallback mode")
    
    def _initialize_sam(self, ckpt_path: Optional[str], points_per_side: int):
        """Initialize SAM model with checkpoint management."""
        if ckpt_path is None:
            ckpt_path = self._get_or_download_checkpoint()
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {ckpt_path}")
        
        self.sam = sam_model_registry[self.model_type](checkpoint=ckpt_path)
        self.sam.to(self.device)
        
        self.generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100
        )
        
        self.is_initialized = True
        logging.info(f"SAM {self.model_type} initialized successfully on {self.device}")
    
    def _get_or_download_checkpoint(self) -> str:
        """Get checkpoint path, downloading if necessary."""
        filename = f"sam_{self.model_type}_checkpoint.pth"
        ckpt_path = self.cache_dir / filename
        
        if not ckpt_path.exists():
            logging.info(f"Downloading SAM {self.model_type} checkpoint...")
            try:
                import requests
                url = self.checkpoint_urls[self.model_type]
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(ckpt_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logging.info(f"Downloaded SAM checkpoint to {ckpt_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download SAM checkpoint: {e}")
        
        return str(ckpt_path)
    
    def get_best_mask(self, img: np.ndarray) -> np.ndarray:
        """Generate highest quality mask from image."""
        if not self.is_initialized:
            return self._fallback_mask(img)
        
        try:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            masks = self.generator.generate(img)
            
            if not masks:
                logging.warning("SAM generated no masks, using fallback")
                return self._fallback_mask(img)
            
            best_mask = max(masks, key=lambda x: x['stability_score'])
            return best_mask['segmentation'].astype(np.uint8)
            
        except Exception as e:
            logging.error(f"SAM mask generation failed: {e}")
            return self._fallback_mask(img)
    
    def get_multiple_masks(self, img: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Generate multiple high-quality masks for diversity."""
        if not self.is_initialized:
            fallback = self._fallback_mask(img)
            return [{'segmentation': fallback, 'stability_score': 0.5, 'area': fallback.sum()}]
        
        try:
            masks = self.generator.generate(img)
            if not masks:
                return []
            
            sorted_masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
            return sorted_masks[:top_k]
            
        except Exception as e:
            logging.error(f"Multiple mask generation failed: {e}")
            return []
    
    def _fallback_mask(self, img: np.ndarray) -> np.ndarray:
        """Generate simple threshold-based mask as fallback."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.astype(np.uint8)
    
    def is_available(self) -> bool:
        return self.is_initialized
    
    def get_model_info(self) -> Dict:
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'initialized': self.is_initialized,
            'sam_available': SAM_AVAILABLE,
            'cache_dir': str(self.cache_dir)
        }

# ==================================================================
# Section: Skeleton-Aware Post-processing (SAP)
# ==================================================================

class SkeletonAwareProcessor:
    """
    Professional skeleton-aware post-processing for mask refinement.
    """
    
    def __init__(self, 
                 min_branch_len: int = 5,
                 gap_fill_radius: int = 3,
                 min_component_size: int = 50):
        self.min_branch_len = min_branch_len
        self.gap_fill_radius = gap_fill_radius  
        self.min_component_size = min_component_size
        
        self.has_ximgproc = hasattr(cv2, 'ximgproc')
        if not self.has_ximgproc:
            logging.warning("cv2.ximgproc not available - using skimage thinning fallback")
    
    def sap_refine(self, mask_bin: np.ndarray) -> np.ndarray:
        """Complete skeleton-aware post-processing pipeline."""
        if mask_bin.max() == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)
        
        try:
            mask_clean = self._clean_input_mask(mask_bin)
            if mask_clean.max() == 0: return mask_clean

            skeleton = self._extract_skeleton(mask_clean)
            if skeleton.max() == 0: return mask_clean

            skeleton_repaired = self._repair_skeleton_gaps(skeleton)
            if skeleton_repaired.max() == 0: return skeleton

            mask_restored = self._restore_mask_width(skeleton_repaired, mask_clean)
            if mask_restored.max() == 0: return mask_clean
            
            return mask_restored.astype(np.uint8)
            
        except Exception as e:
            logging.error(f"SAP refinement failed: {e}", exc_info=True)
            fallback_mask = mask_bin.copy()
            if fallback_mask.max() <= 1:
                fallback_mask = (fallback_mask * 255)
            return fallback_mask.astype(np.uint8)
    
    def _clean_input_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean and normalize input mask."""
        if mask.max() <= 1:
            mask_bin = (mask * 255).astype(np.uint8)
        else:
            mask_bin = (mask > 127).astype(np.uint8) * 255
        
        if mask_bin.max() == 0: return mask_bin
            
        nonzero_count = np.count_nonzero(mask_bin)
        if nonzero_count < 10: return mask_bin
        
        try:
            mask_bin = remove_small_objects(
                mask_bin > 0, min_size=min(self.min_component_size, nonzero_count // 4)
            ).astype(np.uint8) * 255
        except Exception as e:
            print(f"[SAP] Small object removal failed: {e}, skipping")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
        
        return mask_bin
    
    def _extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """Extract 1-pixel skeleton."""
        mask_bool = mask > 127
        
        if self.has_ximgproc:
            try:
                return cv2.ximgproc.thinning(mask.astype(np.uint8), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            except Exception as e:
                logging.warning(f"OpenCV thinning failed: {e}, using skimage")
        
        return (skeletonize(mask_bool) * 255).astype(np.uint8)
    
    def _repair_skeleton_gaps(self, skeleton: np.ndarray) -> np.ndarray:
        """Connects nearby endpoints within gap_fill_radius."""
        skeleton_repaired = skeleton.copy()
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(skeleton, -1, kernel) - skeleton
        endpoints = ((neighbor_count == 255) & (skeleton == 255))
        
        if not np.any(endpoints): return skeleton_repaired
        
        endpoint_coords = np.column_stack(np.where(endpoints))
        
        for i, (y1, x1) in enumerate(endpoint_coords):
            for j, (y2, x2) in enumerate(endpoint_coords[i+1:], i+1):
                distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                if distance <= self.gap_fill_radius:
                    cv2.line(skeleton_repaired, (x1, y1), (x2, y2), 255, 1)
        
        return skeleton_repaired
    
    def _restore_mask_width(self, skeleton: np.ndarray, original_mask: np.ndarray) -> np.ndarray:
        """Restore mask width using distance transform."""
        if skeleton.max() == 0: return original_mask
        
        original_bool = original_mask > 127
        if not np.any(original_bool): return skeleton
        
        try:
            distance_map = distance_transform_edt(original_bool)
        except Exception as e:
            return skeleton
        
        skeleton_points = skeleton > 127
        if not np.any(skeleton_points): return original_mask
        
        skeleton_distances = distance_map[skeleton_points]
        if len(skeleton_distances) == 0: return original_mask
        
        median_radius = np.median(skeleton_distances)
        final_radius = min(max(1, int(median_radius)), 7)
        
        kernel_size = final_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        try:
            restored_mask = cv2.dilate(skeleton, kernel, iterations=1)
        except Exception as e:
            return skeleton
        
        return cv2.bitwise_and(restored_mask, original_mask)

# ==================================================================
# Section: Global Getters
# ==================================================================

_sam_instance = None
_sap_instance = None

def get_sam_autocoder(**kwargs) -> SAMAutoCoder:
    """Get global SAM instance with lazy initialization."""
    global _sam_instance
    if _sam_instance is None:
        _sam_instance = SAMAutoCoder(**kwargs)
    return _sam_instance

def get_skeleton_processor(**kwargs) -> SkeletonAwareProcessor:
    """Get global skeleton processor with lazy initialization."""
    global _sap_instance
    if _sap_instance is None:
        _sap_instance = SkeletonAwareProcessor(**kwargs)
    return _sap_instance
