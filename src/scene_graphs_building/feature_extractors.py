
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import os
import joblib
from src.scene_graphs_building.memory_efficient_cache import MemoryEfficientFeatureCache

class RealFeatureExtractor:
    """
    SOTA feature extraction: lazy SAM, ROI-crop, CLIP pooling, geometric+programmatic fusion, disk-backed cache, fallback for tiny masks.
    """
    def __init__(self, 
                 clip_model_name: str = "openai/clip-vit-base-patch32",
                 sam_encoder_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 cache_features: bool = True,
                 use_sam: bool = False):
        self.device = device
        self.cache_features = cache_features
        self.use_sam = use_sam
        cache_dir = os.environ.get("HF_HOME", "./model_cache")
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name, cache_dir=cache_dir)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name, cache_dir=cache_dir).to(device)
        self.clip_model.eval()
        self.sam_encoder = None
        if use_sam and sam_encoder_path and Path(sam_encoder_path).exists():
            try:
                from segment_anything import sam_model_registry
                sam = sam_model_registry["vit_h"](checkpoint=sam_encoder_path)
                self.sam_encoder = sam.image_encoder.to(device)
                self.sam_encoder.eval()
                logging.info("SAM image encoder loaded successfully")
            except Exception as e:
                logging.warning(f"Failed to load SAM encoder: {e}")
        self.feature_fusion = FeatureFusionNetwork(
            clip_dim=768,
            sam_dim=256 if self.sam_encoder else 0,
            output_dim=384
        ).to(device)
        self.disk_cache = MemoryEfficientFeatureCache()
        logging.info(f"RealFeatureExtractor initialized on {device}, use_sam={use_sam}")

    def extract_object_features(self, 
                               image: np.ndarray, 
                               mask: np.ndarray, 
                               object_id: str,
                               problem_id: str = None,
                               node_attrs: dict = None) -> torch.Tensor:
        # Key by (problem_id, object_id, mask_hash)
        mask_hash = str(hash(mask.tobytes()))
        cache_key = f"{problem_id or ''}_{object_id}_{mask_hash}"
        cached = self.disk_cache.load_features(cache_key, device=self.device) if self.cache_features else None
        if cached is not None:
            return cached
        with torch.no_grad():
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            # ROI-crop to stroke bounding box
            bbox = self._get_bbox_from_mask(mask, pad=8)
            crop_img, crop_mask = self._crop_to_bbox(image, mask, bbox)
            # CLIP features: use only mid/late layers for efficiency
            clip_features, mask_coverage = self._extract_clip_features_masked(crop_img, crop_mask, return_coverage=True)
            # Fallback: if mask covers <5% of patches, use global image CLIP embedding
            if mask_coverage < 0.05:
                clip_features, _ = self._extract_clip_features_masked(image, np.ones_like(mask), return_coverage=True)
            sam_features = None
            if self.use_sam and self.sam_encoder is not None:
                sam_features = self._extract_sam_features_masked(crop_img, crop_mask)
            fused_features = self.feature_fusion(clip_features, sam_features)
            # Append normalized geometric/programmatic features
            if node_attrs is not None:
                norm_feats = self._normalize_and_concat_attrs(node_attrs)
                fused_features = torch.cat([fused_features, norm_feats], dim=-1)
            if self.cache_features:
                self.disk_cache.store_features(cache_key, fused_features, metadata={"object_id": object_id, "problem_id": problem_id or '', "mask_hash": mask_hash})
            return fused_features

    def _get_bbox_from_mask(self, mask, pad=8):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, mask.shape[1], mask.shape[0]]
        x0, x1 = max(xs.min()-pad, 0), min(xs.max()+pad, mask.shape[1])
        y0, y1 = max(ys.min()-pad, 0), min(ys.max()+pad, mask.shape[0])
        return [x0, y0, x1, y1]

    def _crop_to_bbox(self, image, mask, bbox):
        x0, y0, x1, y1 = bbox
        crop_img = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        return crop_img, crop_mask

    def _extract_clip_features_masked(self, image: np.ndarray, mask: np.ndarray, return_coverage=False) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        from PIL import Image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        inputs = self.clip_processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        vision_outputs = self.clip_model.vision_model(pixel_values, output_hidden_states=True)
        hidden_states = vision_outputs.hidden_states
        # Only use mid/late layers for efficiency
        mid_features = hidden_states[9]
        late_features = hidden_states[12]
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
        mid_pooled = masked_pool(mid_features, mask_tensor)
        late_pooled = masked_pool(late_features, mask_tensor)
        clip_features = torch.cat([mid_pooled, late_pooled], dim=-1)
        mask_coverage = mask_tensor.sum().item() / (patch_grid_size * patch_grid_size)
        if return_coverage:
            return clip_features.squeeze(0), mask_coverage
        return clip_features.squeeze(0)

    def _extract_sam_features_masked(self, image: np.ndarray, mask: np.ndarray) -> torch.Tensor:
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

    def _normalize_and_concat_attrs(self, node_attrs: dict) -> torch.Tensor:
        # Normalize and one-hot encode geometric/programmatic features
        length = float(node_attrs.get('length', 0) or 0) / 128.0
        orientation = float(node_attrs.get('orientation', 0) or 0) / 360.0
        centroid = node_attrs.get('centroid', [0, 0])
        cx = float(centroid[0]) / 128.0 if len(centroid) > 0 else 0.0
        cy = float(centroid[1]) / 128.0 if len(centroid) > 1 else 0.0
        turn_direction = node_attrs.get('turn_direction', None)
        turn_onehot = [0, 0, 0]
        if turn_direction == 'left':
            turn_onehot[0] = 1
        elif turn_direction == 'right':
            turn_onehot[1] = 1
        elif turn_direction == 'none':
            turn_onehot[2] = 1
        arr = [length, orientation, cx, cy] + turn_onehot
        return torch.tensor(arr, dtype=torch.float32, device=self.device)
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
