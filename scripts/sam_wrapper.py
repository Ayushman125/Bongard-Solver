# ------------------------------------------------------------------
# Segment Anything Model (SAM) prompt-free mask generator
# Professional implementation with error handling and caching
# ------------------------------------------------------------------

import os
import torch
import numpy as np
import cv2
from typing import Optional, Dict, List, Union
import logging
from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

class SamAutocoder:
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
        
        Args:
            model_type: SAM model variant ('vit_b', 'vit_l', 'vit_h')
            ckpt_path: Path to checkpoint, auto-downloads if None
            device: Computation device
            points_per_side: Grid density for mask generation
            cache_dir: Directory for model caching
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model checkpoint mapping
        self.checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }
        
        self.sam = None
        self.generator = None
        self.is_initialized = False
        
        # Initialize if SAM is available
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
        
        # Load SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=ckpt_path)
        self.sam.to(self.device)
        
        # Configure automatic mask generator
        self.generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100  # Filter tiny artifacts
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
        """
        Generate highest quality mask from image.
        
        Args:
            img: Input image as numpy array [H,W,3] or [H,W]
            
        Returns:
            Binary mask as numpy array [H,W] uint8
        """
        if not self.is_initialized:
            return self._fallback_mask(img)
        
        try:
            # Ensure RGB format
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Generate masks
            masks = self.generator.generate(img)
            
            if not masks:
                logging.warning("SAM generated no masks, using fallback")
                return self._fallback_mask(img)
            
            # Select best mask by stability score
            best_mask = max(masks, key=lambda x: x['stability_score'])
            return best_mask['segmentation'].astype(np.uint8)
            
        except Exception as e:
            logging.error(f"SAM mask generation failed: {e}")
            return self._fallback_mask(img)
    
    def get_multiple_masks(self, img: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Generate multiple high-quality masks for diversity.
        
        Args:
            img: Input image as numpy array
            top_k: Number of top masks to return
            
        Returns:
            List of mask dictionaries with metadata
        """
        if not self.is_initialized:
            fallback = self._fallback_mask(img)
            return [{'segmentation': fallback, 'stability_score': 0.5, 'area': fallback.sum()}]
        
        try:
            masks = self.generator.generate(img)
            if not masks:
                return []
            
            # Sort by stability score and return top_k
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
        
        # Otsu thresholding as basic segmentation
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.astype(np.uint8)
    
    def is_available(self) -> bool:
        """Check if SAM is properly initialized."""
        return self.is_initialized
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'initialized': self.is_initialized,
            'sam_available': SAM_AVAILABLE,
            'cache_dir': str(self.cache_dir)
        }


class SamMaskCache:
    """Simple LRU cache for SAM-generated masks to improve performance."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
    
    def _hash_image(self, img: np.ndarray) -> str:
        """Create hash key for image."""
        return str(hash(img.tobytes()))
    
    def get(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Get cached mask if available."""
        key = self._hash_image(img)
        if key in self.cache:
            # Move to end (most recent)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, img: np.ndarray, mask: np.ndarray):
        """Cache mask for image."""
        key = self._hash_image(img)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = mask.copy()
        if key not in self.access_order:
            self.access_order.append(key)


# Global singleton for efficiency
_sam_instance = None

def get_sam_autocoder(**kwargs) -> SamAutocoder:
    """Get global SAM instance with lazy initialization."""
    global _sam_instance
    if _sam_instance is None:
        _sam_instance = SamAutocoder(**kwargs)
    return _sam_instance
