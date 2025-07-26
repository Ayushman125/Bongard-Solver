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

# ==================================================================
# Section: Advanced Mask Refinement
# ==================================================================

class MaskRefiner:
    """
    An advanced, professional-grade mask refinement pipeline.
    Uses a two-stage process: contour-based simplification and morphological cleaning.
    This approach is more robust than skeletonization for preserving object shape.
    """
    def __init__(self, 
                 contour_approx_factor: float = 0.005,
                 min_component_size: int = 50,
                 closing_kernel_size: int = 5,
                 opening_kernel_size: int = 3):
        self.contour_approx_factor = contour_approx_factor
        self.min_component_size = min_component_size
        self.closing_kernel_size = closing_kernel_size
        self.opening_kernel_size = opening_kernel_size

    def refine(self, mask_bin: np.ndarray) -> np.ndarray:
        if not isinstance(mask_bin, np.ndarray) or mask_bin.max() == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)
        try:
            # Ensure mask is in the correct format (binary 0 or 255)
            if mask_bin.max() <= 1:
                mask_bin = (mask_bin * 255).astype(np.uint8)
            else:
                mask_bin = (mask_bin > 127).astype(np.uint8) * 255

            # Stage 1: Contour-based Refinement
            refined_mask = self._contour_refinement(mask_bin)
            if refined_mask.max() == 0:
                return refined_mask

            # Stage 2: Morphological Cleaning
            cleaned_mask = self._morphological_cleaning(refined_mask)
            return cleaned_mask.astype(np.uint8)
        except Exception as e:
            logging.error(f"Mask refinement pipeline failed: {e}", exc_info=True)
            return (mask_bin > 127).astype(np.uint8) * 255

    def _contour_refinement(self, mask: np.ndarray) -> np.ndarray:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask, dtype=np.uint8)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_component_size:
            return np.zeros_like(mask, dtype=np.uint8)
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = perimeter * self.contour_approx_factor
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        refined_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(refined_mask, [approx_contour], -1, (255), thickness=cv2.FILLED)
        return refined_mask

    def _morphological_cleaning(self, mask: np.ndarray) -> np.ndarray:
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   (self.closing_kernel_size, self.closing_kernel_size))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                   (self.opening_kernel_size, self.opening_kernel_size))
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, opening_kernel)
        return opened_mask

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


# ==================================================================
# Section: Global Getters
# ==================================================================

# ==================================================================
# Section: Global Getters
# ==================================================================

_sam_instance = None
_refiner_instance = None

def get_sam_autocoder(**kwargs) -> SAMAutoCoder:
    """Get global SAM instance with lazy initialization."""
    global _sam_instance
    if _sam_instance is None:
        _sam_instance = SAMAutoCoder(**kwargs)
    return _sam_instance

def get_mask_refiner(**kwargs) -> MaskRefiner:
    """Get global mask refiner instance with lazy initialization."""
    global _refiner_instance
    if _refiner_instance is None:
        _refiner_instance = MaskRefiner(**kwargs)
    return _refiner_instance
