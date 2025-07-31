import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

class RealFeatureExtractor:
    """
    Professional feature extraction system using SAM image encoder, CLIP vision transformer,
    and masked feature extraction for high-quality object representations.
    """
    def __init__(self, 
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 sam_encoder_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_features: bool = True):
        self.device = device
        self.cache_features = cache_features
        self.feature_cache = {}
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        self.sam_encoder = None
        if sam_encoder_path and Path(sam_encoder_path).exists():
            try:
                from segment_anything import sam_model_registry
                sam = sam_model_registry["vit_h"](checkpoint=sam_encoder_path)
                self.sam_encoder = sam.image_encoder.to(device)
                self.sam_encoder.eval()
                logging.info("SAM image encoder loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load SAM encoder: {e}")
        # For openai/clip-vit-base-patch32, the feature dim is 768
        # SAM encoder outputs 256-dim features per mask (not 1280)
        self.feature_fusion = FeatureFusionNetwork(
            clip_dim=768,
            sam_dim=256 if self.sam_encoder else 0,
            output_dim=384
        ).to(device)
        logging.info(f"RealFeatureExtractor initialized on {device}")
    def extract_object_features(self, 
                               image: np.ndarray, 
                               mask: np.ndarray, 
                               object_id: str) -> torch.Tensor:
        cache_key = f"{object_id}_{hash(mask.tobytes())}"
        if self.cache_features and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        with torch.no_grad():
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            clip_features = self._extract_clip_features_masked(image, mask)
            sam_features = None
            if self.sam_encoder is not None:
                sam_features = self._extract_sam_features_masked(image, mask)
            fused_features = self.feature_fusion(clip_features, sam_features)
            if self.cache_features:
                self.feature_cache[cache_key] = fused_features
            return fused_features
    def _extract_clip_features_masked(self, 
                                    image: np.ndarray, 
                                    mask: np.ndarray) -> torch.Tensor:
        from PIL import Image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        vision_outputs = self.clip_model.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        early_features = hidden_states[6]
        mid_features = hidden_states[9]
        late_features = hidden_states[12]
        # For ViT-B/32, patch grid is 7x7 (49 patches)
        patch_grid_size = 7
        mask_resized = cv2.resize(mask.astype(np.uint8), (patch_grid_size, patch_grid_size), interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).float().to(self.device)
        mask_tensor = mask_tensor.view(1, -1, 1)  # [1, 49, 1]
        def masked_pool(features, mask_weights):
            patch_features = features[:, 1:, :]  # [1, 49, C]
            weighted_features = patch_features * mask_weights  # [1, 49, C]
            mask_sum = mask_weights.sum(dim=1, keepdim=True) + 1e-6
            pooled = weighted_features.sum(dim=1) / mask_sum.squeeze(-1)
            return pooled
        early_pooled = masked_pool(early_features, mask_tensor)
        mid_pooled = masked_pool(mid_features, mask_tensor)
        late_pooled = masked_pool(late_features, mask_tensor)
        clip_features = torch.cat([early_pooled, mid_pooled, late_pooled], dim=-1)
        return clip_features.squeeze(0)
    def _extract_sam_features_masked(self, 
                                   image: np.ndarray, 
                                   mask: np.ndarray) -> torch.Tensor:
        image_sam = cv2.resize(image, (1024, 1024))
        mask_sam = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image_sam).permute(2, 0, 1).float().to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        sam_features = self.sam_encoder(image_tensor)
        mask_sam_tensor = torch.from_numpy(mask_sam).float().to(self.device)
        mask_sam_resized = torch.nn.functional.interpolate(
            mask_sam_tensor.unsqueeze(0).unsqueeze(0), 
            size=(64, 64), 
            mode='nearest'
        ).squeeze()
        mask_weights = mask_sam_resized.view(1, 1, 64, 64)
        masked_features = sam_features * mask_weights
        pooled_features = masked_features.sum(dim=(2, 3)) / (mask_weights.sum(dim=(2, 3)) + 1e-6)
        return pooled_features.squeeze(0)

class FeatureFusionNetwork(nn.Module):
    def __init__(self, clip_dim: int, sam_dim: int, output_dim: int):
        super().__init__()
        self.clip_dim = clip_dim * 3
        self.sam_dim = sam_dim
        self.output_dim = output_dim
        self.clip_proj = nn.Sequential(
            nn.Linear(self.clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        if sam_dim > 0:
            self.sam_proj = nn.Sequential(
                nn.Linear(sam_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            )
            fusion_input_dim = 256 + 128
        else:
            self.sam_proj = None
            fusion_input_dim = 256
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    def forward(self, clip_features: torch.Tensor, sam_features: Optional[torch.Tensor]) -> torch.Tensor:
        clip_proj = self.clip_proj(clip_features)
        if sam_features is not None and self.sam_proj is not None:
            sam_proj = self.sam_proj(sam_features)
            combined = torch.cat([clip_proj, sam_proj], dim=-1)
        else:
            combined = clip_proj
        output = self.fusion(combined)
        return output
