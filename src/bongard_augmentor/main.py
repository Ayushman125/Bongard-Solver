import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import kornia.augmentation as K
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import time
import random
import json
import traceback
from enum import Enum, auto
from collections import deque
import csv
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple

from .utils import (
    HYBRID_PIPELINE_AVAILABLE,
    MaskType,
    classify_mask,
    robust_z_scores,
    sanitize_for_opencv,
    diagnose_tensor_corruption,
    safe_device_transfer,
    repair_mask,
    pre_warp_mask,
    pre_warp_fatten,
    topology_aware_morphological_repair
)
if HYBRID_PIPELINE_AVAILABLE:
    from .hybrid import SAMAutoCoder, get_mask_refiner

class ImageAugmentor:
    """GPU-batched geometric augmentations with profiling, adaptive QA, and failover logic."""
    def __init__(self, device: str = 'cuda', batch_size: int = 32, geometric_transforms=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.geometric_transforms = geometric_transforms
        self.profiler = TaskProfiler()
        self.metrics = TaskProfiler()  # Fix for AttributeError
        self.adaptive_log = deque(maxlen=20)
        self.qa_thresholds = self.BONGARD_QA_THRESHOLDS.copy()
        self.hybrid_enabled = False
        self.sam_autocoder = None
        self.mask_refiner = None
        self.diffusion_enabled = False

    BONGARD_QA_THRESHOLDS = {
        MaskType.EMPTY:  {'pixvar_min':0.00001,'edge_overlap_min':0.0,'area_ratio_min':0.0,'area_ratio_max':20.0},
        MaskType.THIN:   {'pixvar_min':0.00005,'edge_overlap_min':0.001,'area_ratio_min':0.05,'area_ratio_max':15.0},
        MaskType.SPARSE: {'pixvar_min':0.0001,'edge_overlap_min':0.005,'area_ratio_min':0.1,'area_ratio_max':12.0},
        MaskType.DENSE:  {'pixvar_min':0.0005,'edge_overlap_min':0.01,'area_ratio_min':0.15,'area_ratio_max':10.0},
    }

    def initialize_hybrid_pipeline(self, sam_model_type: str = 'vit_h', enable_refiner: bool = True, enable_diffusion: bool = False):
        if HYBRID_PIPELINE_AVAILABLE:
            self.sam_autocoder = SAMAutoCoder(model_type=sam_model_type, device=self.device)
            if enable_refiner:
                self.mask_refiner = get_mask_refiner()
            self.hybrid_enabled = True
            print("[HYBRID] SAM and MaskRefiner initialized successfully.")
        else:
            print("[HYBRID] Hybrid pipeline dependencies not found. Disabling.")
            self.hybrid_enabled = False
        self.diffusion_enabled = enable_diffusion
        if self.diffusion_enabled:
            print("[DIFFUSION] Diffusion-based augmentation is enabled.")

    def safe_mask_conversion(self, mask, refine: bool = False):
        """
        Safely convert mask between torch tensor and numpy array for OpenCV.
        Handles device, dtype, and value range. Enhanced to recover faint masks.
        """
        import numpy as np
        try:
            import torch
        except ImportError:
            torch = None

        # If mask is a torch tensor, move to cpu and convert to numpy
        if torch is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        # Squeeze to 2D if needed
        while mask.ndim > 2:
            mask = mask.squeeze()
        
        # Handle different mask value ranges more intelligently
        if mask.dtype != np.uint8:
            if mask.max() <= 0.01:
                # Very low-value masks - use adaptive threshold to recover structure
                if mask.max() > 0:
                    threshold = mask.max() / 2  # Use half of max as threshold
                    mask = (mask > threshold).astype(np.uint8) * 255
                    print(f"[MASK RECOVERY] Low-value mask detected (max={mask.max():.6f}), using adaptive threshold={threshold:.6f}")
                else:
                    mask = np.zeros_like(mask, dtype=np.uint8)
            elif mask.max() <= 1.0:
                # Standard float mask [0,1] -> [0,255] with gentle threshold
                mask = (mask > 0.001).astype(np.uint8) * 255  # Gentle threshold for thin structures
            else:
                # Already in [0,255] range or higher
                mask = (mask > 127).astype(np.uint8) * 255
        
        # Optionally refine mask using MaskRefiner
        if refine and self.mask_refiner is not None:
            mask = self.mask_refiner.refine(mask)
        return mask

    def visualize_mask(self, mask_tensor, out_path):
        import matplotlib.pyplot as plt
        arr = mask_tensor.squeeze(0).cpu().numpy()
        plt.imshow(arr, cmap='gray')
        plt.axis('off')
        plt.savefig(out_path)
        plt.close()

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
            for mtype in self.qa_thresholds:
                self.qa_thresholds[mtype]['edge_overlap_min'] *= k_relax
                self.qa_thresholds[mtype]['pixvar_min'] *= k_relax
            print(f"[ADAPTIVE QA] Relaxed thresholds after {long_fail:.2%} failure rate")
        elif long_fail < target * 0.5:
            for mtype in self.qa_thresholds:
                self.qa_thresholds[mtype]['edge_overlap_min'] *= k_tighten
                self.qa_thresholds[mtype]['pixvar_min'] *= k_tighten
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

    def generate_diffusion_augmented(self, images, masks, batch_idx, num_variations=2):
        """
        Generate variations using a fine-tuned diffusion model (ControlNet).
        """
        if not self.diffusion_enabled:
            return images.repeat(num_variations, 1, 1, 1), masks.repeat(num_variations, 1, 1, 1)

        try:
            from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
            
            # Lazy load model
            if not hasattr(self, 'controlnet_pipe'):
                controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
                self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
                )
                self.controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(self.controlnet_pipe.scheduler.config)
                self.controlnet_pipe.to(self.device)

            all_augmented_images = []
            for i in range(images.size(0)):
                # Prepare conditioning image (canny edge from mask)
                mask_np = self.safe_mask_conversion(masks[i])
                canny_image = cv2.Canny(mask_np, 100, 200)
                canny_image = Image.fromarray(canny_image).convert("RGB")

                prompt = "a simple abstract shape, black and white, minimalist"
                
                with torch.autocast("cuda"):
                    augmented_imgs = self.controlnet_pipe(
                        [prompt] * num_variations,
                        num_inference_steps=20,
                        generator=torch.manual_seed(i), 
                        image=canny_image
                    ).images
                
                # Convert back to tensor
                for aug_img in augmented_imgs:
                    aug_tensor = ToTensorV2()(image=np.array(aug_img.convert("L")))['image'].unsqueeze(0)
                    all_augmented_images.append(aug_tensor)

            return torch.cat(all_augmented_images, dim=0), masks.repeat(num_variations, 1, 1, 1)

        except Exception as e:
            print(f"[DIFFUSION ERROR] Failed: {e}")
            traceback.print_exc()
            return images.repeat(num_variations, 1, 1, 1), masks.repeat(num_variations, 1, 1, 1)

    def synthesize_pair_diffusion(self, positive_img, negative_img, positive_mask, negative_mask, batch_idx):
        """
        (Placeholder) Synthesize a new image pair using diffusion, guided by masks.
        """
        print(f"[DIFFUSION] Placeholder for synthesizing new pair for batch {batch_idx}.")
        # This would involve a more complex diffusion process, possibly with textual inversion
        # or other conditioning to respect the positive/negative concept.
        return positive_img.clone(), negative_img.clone()

    def x_paste_batch(self, images, masks, batch_idx):
        """
        Advanced paste that preserves the mask's relative scale and position.
        """
        B, C, H, W = images.shape
        output_images = torch.zeros_like(images)
        output_masks = torch.zeros_like(masks)

        for i in range(B):
            # Use the current image and mask as the object to paste
            obj = images[i]
            obj_mask = masks[i]
            obj_h, obj_w = obj.shape[-2:]

            # Create a new blank image and mask
            new_img = torch.zeros_like(obj)
            new_mask = torch.zeros_like(obj_mask)

            # Determine a random position to paste the object
            if H > obj_h:
                new_y = random.randint(0, H - obj_h)
            else:
                new_y = 0
            
            if W > obj_w:
                new_x = random.randint(0, W - obj_w)
            else:
                new_x = 0

            # Paste the object and mask
            new_img[:, new_y:new_y+obj_h, new_x:new_x+obj_w] = obj
            new_mask[:, new_y:new_y+obj_h, new_x:new_x+obj_w] = obj_mask
            
            output_images[i] = new_img
            output_masks[i] = new_mask

        return output_images, output_masks

    def keepmask_augment(self, images, masks, batch_idx):
        """
        Augment content *around* the mask, keeping the mask itself static.
        Uses inpainting to fill the area after removing the masked object.
        """
        if not self.diffusion_enabled:
            print("[KEEPMASK] Diffusion disabled, skipping.")
            return images, masks

        try:
            from diffusers import StableDiffusionInpaintPipeline
            import torch

            if not hasattr(self, 'inpaint_pipe'):
                self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16,
                )
                self.inpaint_pipe.to(self.device)

            augmented_images = []
            for i in range(images.size(0)):
                img_pil = Image.fromarray(self.safe_mask_conversion(images[i])).convert("RGB")
                mask_pil = Image.fromarray(self.safe_mask_conversion(masks[i])).convert("RGB")
                
                prompt = "minimalist background, abstract, simple"
                
                with torch.autocast("cuda"):
                    inpainted_img = self.inpaint_pipe(
                        prompt=prompt,
                        image=img_pil,
                        mask_image=mask_pil,
                        num_inference_steps=20,
                        generator=torch.manual_seed(batch_idx * 100 + i)
                    ).images[0]

                # Convert back to tensor
                inpainted_tensor = ToTensorV2()(image=np.array(inpainted_img.convert("L")))['image']
                
                # Combine inpainted background with original foreground
                final_img = torch.where(masks[i] > 0.5, images[i], inpainted_tensor.to(self.device))
                augmented_images.append(final_img.unsqueeze(0))

            return torch.cat(augmented_images, dim=0), masks
        except Exception as e:
            print(f"[KEEPMASK ERROR] Failed: {e}")
            traceback.print_exc()
            return images, masks

    def get_augmentation_pipeline(self, mask_type):
        """Return mask-type-specific augmentation pipeline."""
        # Always use Albumentations for now since we have issues with Kornia
        if mask_type == MaskType.EMPTY:
            # No augmentation for empty masks
            return None
        elif mask_type == MaskType.THIN:
            return A.Compose([
                A.HorizontalFlip(p=0.1),
                A.RandomRotate90(p=0.05),
                A.Rotate(limit=5, p=0.1)
            ], additional_targets={'mask': 'mask'})
        elif mask_type == MaskType.SPARSE:
            return A.Compose([
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.1),
                A.Rotate(limit=10, p=0.1)
            ], additional_targets={'mask': 'mask'})
        else:  # DENSE
            return A.Compose([
                A.RandomRotate90(p=0.2),
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=15, p=0.1)
            ], additional_targets={'mask': 'mask'})

    def geometry_to_binary_mask(self, geometry, H, W):
        # Defensive: flatten, parse, and validate geometry
        from PIL import ImageDraw
        mask_img = Image.new('L', (W, H), 0)
        # Debug print removed to prevent terminal flooding
        # Parse string geometry if needed
        if isinstance(geometry, str):
            try:
                geometry = json.loads(geometry)
            except Exception:
                print(f"[QA WARN] Could not parse geometry string: {geometry}")
                geometry = []
        if isinstance(geometry, torch.Tensor):
            geometry = geometry.cpu().numpy().tolist()
        # Flatten if single list
        if geometry and all(isinstance(v, (int, float)) for v in geometry):
            geometry = [geometry[i:i+2] for i in range(0, len(geometry), 2) if i+1 < len(geometry)]
        # Remove zero points and non-numeric
        poly_points = []
        for pt in geometry:
            if (isinstance(pt, (list, tuple)) and len(pt) == 2 and
                all(isinstance(coord, (int, float)) for coord in pt) and not (pt[0] == 0 and pt[1] == 0)):
                poly_points.append((float(pt[0]), float(pt[1])))
        # If still invalid, fallback to hybrid mask
        if not poly_points or len(poly_points) < 2:
            print(f"[QA WARN] Invalid geometry, using hybrid mask fallback: {geometry}")
            if getattr(self, 'hybrid_enabled', False) and hasattr(self, 'generate_hybrid_mask'):
                # Use SAM/SAP hybrid mask
                return self.generate_hybrid_mask(torch.zeros((1, H, W)))
            else:
                return torch.zeros((1, H, W))
        # Normalize geometry coordinates to image size
        max_x = max(x for x, y in poly_points)
        max_y = max(y for x, y in poly_points)
        # Always scale geometry to image dimensions to fit the canvas.
        # This handles cases where geometry is normalized (e.g., to [0,1]) 
        # or in a different coordinate space than the image.
        scale_x = (W / max_x) if max_x > 0 else 1.0
        scale_y = (H / max_y) if max_y > 0 else 1.0
        norm_points = [(x * scale_x, y * scale_y) for x, y in poly_points]
        ImageDraw.Draw(mask_img).line(norm_points, fill=1, width=2)
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
            local_cutoff = max(self.qa_thresholds.get('outlier_sigma', 3.5) * sigma_factor, adaptive_cutoff, warn_threshold)
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

    def _augmentation_diversity_metrics(self, images, batch_idx):
        feats = self._extract_features(images.cpu())
        # Compute pairwise distances
        dists = torch.cdist(feats, feats)
        diversity = dists.mean().item()
        print(f"[QA DIVERSITY] Batch {batch_idx} diversity score: {diversity:.3f}")

    def _adversarial_robustness(self, images, masks, batch_idx, adv_dir="qa_adversarial"):
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
            
            aug_img = aug_images[i]
            aug_mask = aug_masks[i]
            orig_mask = orig_masks[i]
            
            # Basic QA checks
            pixvar = aug_img.var().item()
            orig_area = orig_mask.sum().item()
            aug_area = aug_mask.sum().item()
            area_ratio = aug_area / (orig_area + 1e-6)
            edge_overlap = self.compute_edge_overlap(orig_mask, aug_mask)
            edge_overlaps.append(edge_overlap)
            
            mask_type = classify_mask(orig_mask)
            thresholds = self.qa_thresholds.get(mask_type, self.qa_thresholds[MaskType.DENSE])

            reasons = []
            if pixvar < thresholds['pixvar_min']: reasons.append(f"low_pixvar({pixvar:.4f})")
            if not (thresholds['area_ratio_min'] < area_ratio < thresholds['area_ratio_max']):
                reasons.append(f"bad_area_ratio({area_ratio:.2f})")
            if edge_overlap < thresholds['edge_overlap_min']:
                reasons.append(f"low_edge_overlap({edge_overlap:.3f})")

            if reasons:
                fail_count += 1
                flagged.add(i)
                reason_str = ", ".join(reasons)
                csv_rows.append([batch_idx, i, "FAIL", reason_str, pixvar, orig_area, aug_area, area_ratio, edge_overlap, mask_type.name])
                # Save failing examples for review
                save_image(aug_img, os.path.join(qa_human_review_dir, f"batch{batch_idx}_sample{i}_fail_img.png"))
                save_image(aug_mask, os.path.join(qa_human_review_dir, f"batch{batch_idx}_sample{i}_fail_mask.png"))

        # Statistical outlier detection on the whole batch
        self._statistical_outlier_detection(aug_images, aug_masks, batch_idx)
        
        # Log to CSV
        if csv_rows:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(csv_fields)
                writer.writerows(csv_rows)
        
        # Update adaptive QA
        fail_rate = fail_count / aug_images.size(0)
        self.adapt_qa_thresholds(fail_rate)
        
        # Log metrics (safe no-op)
        # If TaskProfiler does not have log, skip or print
        if hasattr(self.metrics, 'log') and callable(self.metrics.log):
            self.metrics.log('qa_fail_rate', fail_rate)
            self.metrics.log('edge_overlap', np.mean(edge_overlaps))
        
        return fail_count

    def augment_batch(self, images, paths, geometries, augment_type='both', batch_idx=0):
        """
        Apply geometric augmentations to a batch of images and masks.
        Handles both Kornia (GPU) and Albumentations (CPU) pipelines.
        """
        t0 = time.time()
        batch_size = images.size(0)
        images = safe_device_transfer(images, self.device)
        
        # 1. Generate initial masks from geometry or hybrid pipeline
        orig_masks_list = []
        batch_labels = [] # For storing derived labels if available
        
        for i in range(batch_size):
            geom = geometries[i] if geometries else []
            path = paths[i] if paths else None
            
            # Check for derived labels from dataset
            if hasattr(self, 'dataset') and self.dataset and hasattr(self.dataset, 'labels_map') and self.dataset.labels_map:
                label_entry = self.dataset.labels_map.get(os.path.normpath(path))
                if label_entry:
                    batch_labels.append(label_entry)

            # Use hybrid pipeline if enabled and geometry is missing
            if self.hybrid_enabled and (not geom or all(p == [0,0] for p in geom)):
                print(f"[HYBRID] Using SAM for empty geometry in sample {i} of batch {batch_idx}")
                mask_tensor = self.generate_hybrid_mask(images[i])
            else:
                # Fallback to geometry rasterization
                mask_tensor = self.geometry_to_binary_mask(geom, images.shape[2], images.shape[3])
            
            orig_masks_list.append(safe_device_transfer(mask_tensor, self.device))

        orig_masks = torch.stack(orig_masks_list)
        
        # 2. Pre-augmentation processing (padding, centering)
        padded_images_list, padded_masks_list = [], []
        for i in range(batch_size):
            # Pad and center to create buffer for transformations
            padded_img, padded_mask = self.pad_and_center(images[i], orig_masks[i], pad=128)
            padded_images_list.append(padded_img)
            padded_masks_list.append(padded_mask)
        
        images = torch.stack(padded_images_list)
        orig_masks = torch.stack(padded_masks_list)

        # 3. Apply augmentations
        aug_images_list, aug_masks_list = [], []
        for i in range(batch_size):
            img = images[i]
            mask = orig_masks[i]
            mask_type = classify_mask(mask)
            
            # Get augmentation pipeline based on mask type
            transform = self.get_augmentation_pipeline(mask_type)
            
            if transform:
                # Apply Albumentations transform
                img_np = img.squeeze(0).cpu().numpy()
                mask_np = mask.squeeze(0).cpu().numpy()
                
                # Ensure correct format for Albumentations
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
                
                augmented = transform(image=img_np, mask=mask_np)
                aug_img = torch.from_numpy(augmented['image']).unsqueeze(0).float() / 255.0
                aug_mask = torch.from_numpy(augmented['mask']).unsqueeze(0).float()
            else:
                # No augmentation for empty masks
                aug_img, aug_mask = img, mask

            # Center crop back to original size
            aug_img_cropped = self.center_crop(aug_img, size=512)
            aug_mask_cropped = self.center_crop(aug_mask, size=512)
            
            aug_images_list.append(aug_img_cropped)
            aug_masks_list.append(aug_mask_cropped)

        aug_images = safe_device_transfer(torch.stack(aug_images_list), self.device)
        aug_masks = safe_device_transfer(torch.stack(aug_masks_list), self.device)

        # 4. Post-augmentation validation and binarization
        self._validate_augmented_batch(aug_images, aug_masks, safe_device_transfer(orig_masks, self.device), paths, batch_idx, batch_labels=batch_labels)
        
        # Gentle binarization
        mask_bin = (aug_masks > 0.001).float()

        # 5. Prepare results dictionary
        results = {
            'original': aug_images,
            'geometric': mask_bin,
            'profiling': {
                'transform_ms': (time.time() - t0) * 1000,
                'throughput_imgs_per_sec': batch_size / ((time.time() - t0) + 1e-9)
            }
        }
        if augment_type == 'both':
            results['combined'] = torch.cat([aug_images, mask_bin], dim=1)
            
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

    def x_paste_batch(self, target_img_np, target_mask_np, instance_images, instance_masks, k=5):
        """Scalable Copy-Paste: paste k instances, preserve mask shapes."""
        H, W, _ = target_img_np.shape
        out_img, out_msk = target_img_np.copy(), target_mask_np.copy()
        
        for _ in range(k):
            # Randomly select an instance to paste
            idx = random.randint(0, len(instance_images) - 1)
            inst_img_tensor = instance_images[idx]
            inst_msk_tensor = instance_masks[idx]

            # Convert tensors to numpy for processing
            inst_img = inst_img_tensor.squeeze(0).cpu().numpy()
            inst_msk = inst_msk_tensor.squeeze(0).cpu().numpy()

            # Randomly scale the instance
            scale = random.uniform(0.5, 1.2)
            h, w = int(inst_img.shape[0] * scale), int(inst_img.shape[1] * scale)
            
            if h == 0 or w == 0: continue # Skip if scaled to nothing

            resized_img = cv2.resize(inst_img, (w, h), interpolation=cv2.INTER_AREA)
            resized_msk = cv2.resize(inst_msk, (w, h), interpolation=cv2.INTER_NEAREST)

            # Randomly choose a paste location
            if H > h:
                y_start = random.randint(0, H - h)
            else:
                y_start = 0
            
            if W > w:
                x_start = random.randint(0, W - w)
            else:
                x_start = 0

            # Paste the instance
            out_img[y_start:y_start+h, x_start:x_start+w] = resized_img
            out_msk[y_start:y_start+h, x_start:x_start+w] = resized_msk

        return out_img, out_msk

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
        
        # Get dimensions from mask and image separately
        C_mask, H, W = mask.shape
        C_img = img.shape[0]  # Image may have different channel count
        
        # Create padded tensors with correct channel dimensions
        padded_mask = torch.zeros((C_mask, H+2*pad, W+2*pad), device=mask.device, dtype=mask.dtype)
        padded_img = torch.zeros((C_img, H+2*pad, W+2*pad), device=img.device, dtype=img.dtype)
        
        # Fill the padded tensors
        padded_mask[:, pad:pad+H, pad:pad+W] = mask
        padded_img[:, pad:pad+H, pad:pad+W] = img
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
        
        arr = sanitize_for_opencv(mask) * 255
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
        """Compute normalized edge overlap between two binary masks. Ensures both masks are on the same device and size."""
        # Move mask2 to mask1's device if needed
        if mask2.device != mask1.device:
            mask2 = mask2.to(mask1.device)
        # Ensure both masks are the same size
        H1, W1 = mask1.shape[-2:]
        H2, W2 = mask2.shape[-2:]
        target_size = min(H1, H2, W1, W2, 512)  # Use 512 or smallest
        def center_crop(tensor, size):
            C, H, W = tensor.shape
            y_start = (H - size) // 2
            x_start = (W - size) // 2
            return tensor[:, y_start:y_start+size, x_start:x_start+size]
        mask1_cropped = center_crop(mask1, target_size)
        mask2_cropped = center_crop(mask2, target_size)
        mask1_bin = (mask1_cropped > 0.5)
        mask2_bin = (mask2_cropped > 0.5)
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
        Generate high-quality mask using SAM + advanced mask refinement.
        """
        if not self.hybrid_enabled or self.sam_autocoder is None:
            # Fallback to simple thresholding if hybrid is disabled or not initialized
            if original_mask is not None:
                return (original_mask > 0.5).float()
            else:
                img_np = self.sanitize_for_opencv(image_tensor)
                _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return torch.from_numpy(thresh / 255.0).unsqueeze(0).float().to(self.device)

        try:
            img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            
            if len(img_np.shape) == 2 or img_np.shape[2] == 1:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            sam_mask = self.sam_autocoder.get_best_mask(img_np)
            # Use advanced mask refiner if available
            if self.mask_refiner and sam_mask.max() > 0:
                refined_mask = self.mask_refiner.refine(sam_mask)
            else:
                refined_mask = sam_mask
            mask_tensor = torch.from_numpy(refined_mask / 255.0).unsqueeze(0).float()
            return safe_device_transfer(mask_tensor, self.device)

        except Exception as e:
            print(f"[HYBRID ERROR] Mask generation failed: {e}")
            traceback.print_exc()
            if original_mask is not None:
                return (original_mask > 0.5).float()
            return torch.zeros(1, image_tensor.shape[1], image_tensor.shape[2], device=self.device)

    def generate_diverse_sam_masks(self, image_tensor: torch.Tensor, top_k: int = 3) -> List[torch.Tensor]:
        # ... (implementation to be moved)
        pass

    def test_hybrid_pipeline(self, test_images: torch.Tensor, save_dir: str = "hybrid_test_results") -> Dict:
        # ... (implementation to be moved)
        pass
    
    def _save_hybrid_comparison(self, image: torch.Tensor, main_mask: torch.Tensor, 
                              diverse_masks: List[torch.Tensor], save_path: str):
        # ... (implementation to be moved)
        pass

    def apply_morphology(self, mask, operation='open', kernel_size=3):
        # ... (implementation to be moved)
        pass

    # - _lightweight_QA
    # - generate_hybrid_mask
    # - generate_diverse_sam_masks
    # - test_hybrid_pipeline
    # - _save_hybrid_comparison
    # - apply_morphology
    # - test_corruption_mitigation
    # - proper_binarize
    # - _is_mask_valid
    # - log_detailed_mask_diagnostics
    
class TaskProfiler:
    def __init__(self, log_file="logs/profiler.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_latency(self, task_name, latency_ms, metadata=None):
        log_entry = {
            'timestamp': time.time(),
            'task': task_name,
            'latency_ms': latency_ms,
            'metadata': metadata or {}
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
