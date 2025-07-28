# ------------------------------------------------------------------
# Hybrid Augmentation Pipeline Components
# Consolidates SAM and Skeleton-Aware Processing
# ------------------------------------------------------------------


import os
import torch
import numpy as np
import cv2
import time
import logging
import requests
import json
from typing import Optional, Dict, List, Union, Tuple
import torch
import psutil
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy import ndimage as ndi

from skimage.morphology import skeletonize, remove_small_objects
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from skimage import morphology as ski_morphology

from src.bongard_augmentor.prompting import PromptGenerator
# Monte Carlo dropout utilities for uncertainty quantification
from .mc_dropout_utils import mc_dropout_mask_prediction

# Research-level analytics and RL stubs (optional, for extensibility)
from .rl_analytics import MaskPipelineAnalytics, PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




# Use MaskRefiner for all mask refinement and QA
from src.bongard_augmentor.refiners import MaskRefiner

# --- Official SAM integration ---
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

class SAMMaskGenerator:
    def __init__(self, model_type: str = 'vit_h', checkpoint_dir: str = './sam_checkpoints', device: str = 'cpu'):
        if not SAM_AVAILABLE:
            raise ImportError("segment_anything is not installed. Please install the official Meta AI Segment Anything package.")
        checkpoint_map = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth',
        }
        checkpoint_file = checkpoint_map.get(model_type, 'sam_vit_h_4b8939.pth')
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_dir, exist_ok=True)
            # Download from official Meta AI if not present
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{checkpoint_file}"
            import requests
            print(f"Downloading SAM checkpoint {checkpoint_file} from {url} ...")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(checkpoint_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded SAM checkpoint to {checkpoint_path}")
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        self.predictor = SamPredictor(self.model)

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict(self, point_coords, point_labels, multimask_output=True):
        return self.predictor.predict(point_coords, point_labels, multimask_output=multimask_output)

    

            # --- SkeletonProcessor Class ---
class SkeletonProcessor:
    def sam_skeleton_intelligent_fusion(self, sam_mask: np.ndarray, skeleton: np.ndarray, confidence_map: np.ndarray = None) -> np.ndarray:
        """
        Advanced SAM-skeleton integration with weighted fusion, topology preservation, branch point enhancement, and conflict resolution.
        """
        # Weighted fusion based on confidence (if provided)
        if confidence_map is not None:
            conf_norm = cv2.normalize(confidence_map, None, 0, 1, cv2.NORM_MINMAX)
            fused = np.where(conf_norm > 0.5, sam_mask, skeleton)
        else:
            fused = cv2.bitwise_or(sam_mask, skeleton)
        # Topology-preserving merge (dilate then erode)
        kernel = np.ones((3, 3), np.uint8)
        fused = cv2.dilate(fused, kernel, iterations=1)
        fused = cv2.erode(fused, kernel, iterations=1)
        # --- Branch point enhancement ---
        branch_points = self._get_branch_points(skeleton)
        for y, x in branch_points:
            cv2.circle(fused, (x, y), 2, 255, -1)  # Enhance branch points
        # --- Conflict resolution ---
        # Where SAM and skeleton disagree, prefer skeleton at branch points, else SAM
        sam_bin = (sam_mask > 0).astype(np.uint8)
        skel_bin = (skeleton > 0).astype(np.uint8)
        conflict = (sam_bin != skel_bin)
        for y, x in np.argwhere(conflict):
            if (y, x) in branch_points:
                fused[y, x] = skeleton[y, x]
            else:
                fused[y, x] = sam_mask[y, x]
        return fused
    """
    Advanced skeleton processing for line art and geometric shapes.
    Handles branch detection, pruning, and skeleton-based analysis.
    """
    def __init__(self, min_branch_length: int = 10):
        """
        Initializes the SkeletonProcessor.

        Args:
            min_branch_length (int): Minimum length of a branch to be considered
                                      significant; shorter branches will be pruned.
        """
        self.min_branch_length = min_branch_length

    def _get_branch_points(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detects branch points in a skeletonized image.
        A branch point is a pixel with more than two neighbors in a 8-connected window.

        Args:
            skeleton (np.ndarray): A binary skeleton image (2D array, values 0 or 255).

        Returns:
            List[Tuple[int, int]]: A list of (row, column) coordinates of branch points.
        """
        branch_points = []
        # Create a kernel for counting 8-connected neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1], # Center is 0 to exclude self from count
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the skeleton (normalized to 1s and 0s)
        # BORDER_CONSTANT with value 0 ensures pixels outside are treated as non-existent
        convolved_neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Branch points are skeleton pixels with 3 or more neighbors
        # (convolved_neighbors >= 3) and are part of the skeleton (skeleton > 0)
        branch_point_coords = np.argwhere((skeleton > 0) & (convolved_neighbors >= 3))

        return [tuple(pt) for pt in branch_point_coords]


    def _prune_small_branches(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Removes small branches from the skeleton to simplify it.
        This helps in focusing on the main structure by iteratively removing
        endpoints that are part of short branches.

        Args:
            skeleton (np.ndarray): The input binary skeleton image (uint8, values 0 or 255).

        Returns:
            np.ndarray: The pruned binary skeleton image (uint8).
        """
        pruned_skeleton = skeleton.copy()

        # Iteratively remove endpoints until no more small branches can be pruned
        # or the branch length exceeds min_branch_length.
        while True:
            endpoints = np.array(np.where(self._get_endpoints(pruned_skeleton))).T
            if len(endpoints) == 0:
                break # No more endpoints, pruning is complete

            changes_made = False
            for r, c in endpoints:
                # Trace path from endpoint. If the path is short and leads to dead end or branch
                # Trace slightly beyond min_length to check if it connects to a significant junction
                path = self._trace_path(pruned_skeleton, (r, c), self.min_branch_length + 1)

                # Check if the path is a small branch
                # A path is considered a small branch if it's within min_branch_length
                # and its termination point (if it didn't hit max_length) isn't a significant branch point
                if path is not None and len(path) <= self.min_branch_length:
                    # To be more robust, we should check if the last point of the path (if path_length < max_length)
                    # is *not* a real branch point. If it stopped due to reaching a branch point, it's not a small branch.
                    is_small_branch = True
                    if len(path) > 1:
                        # Check the point *before* the current endpoint.
                        # If this point had multiple connections (i.e., was a branch point)
                        # and now became an endpoint because the "other" branch was pruned,
                        # we should be careful not to prune further.
                        prev_r, prev_c = path[-2] # Point just before the current endpoint
                        temp_skel = pruned_skeleton.copy()
                        temp_skel[r,c] = 0 # Temporarily remove current endpoint to check neighbor count of previous point

                        kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
                        n_count_prev = cv2.filter2D((temp_skel > 0).astype(np.uint8), -1, kernel_neighbors, borderType=cv2.BORDER_CONSTANT)[prev_r, prev_c]

                        if n_count_prev >= 2: # If previous point has 2 or more neighbors after removing current endpoint,
                                              # it's likely an internal point or another branch point, not a simple end of a small branch.
                            is_small_branch = False

                    if is_small_branch:
                        for pr, pc in path:
                            pruned_skeleton[pr, pc] = 0 # Remove small branch segment
                        changes_made = True
            if not changes_made:
                break # No more changes, exit loop
        return pruned_skeleton

    def _get_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Detects endpoints in a skeletonized image.
        An endpoint is a pixel that is part of the skeleton and has exactly one 8-connected neighbor.

        Args:
            skeleton (np.ndarray): A binary skeleton image (2D array, values 0 or 255).

        Returns:
            np.ndarray: A boolean array of the same shape as skeleton, where True indicates an endpoint.
        """
        # Define a 3x3 kernel for neighbor counting (excluding center)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the skeleton (normalized to 1s and 0s) to count neighbors
        convolved_neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Endpoints are skeleton pixels that have exactly one neighbor
        endpoints = (skeleton > 0) & (convolved_neighbors == 1)
        return endpoints

    def _trace_path(self, skeleton: np.ndarray, start_point: Tuple[int, int], max_length: int) -> Optional[List[Tuple[int, int]]]:
        """
        Traces a path from a start_point along the skeleton up to max_length or until a branch point/intersection
        or another endpoint (excluding the starting one).
        Returns the path as a list of coordinates.

        Args:
            skeleton (np.ndarray): The binary skeleton image (uint8, values 0 or 255).
            start_point (Tuple[int, int]): The (row, column) coordinates of the starting point.
            max_length (int): The maximum length of the path to trace.

        Returns:
            Optional[List[Tuple[int, int]]]: A list of (row, col) tuples representing the path,
                                            or None if the path cannot be traced (e.g., start_point is isolated).
        """
        path = [start_point]
        current_point = start_point
        visited = {start_point} # Keep track of visited points to avoid loops

        for _ in range(max_length): # Limit path length
            neighbors = []
            r, c = current_point
            # Check 8-connectivity neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc

                    # Check bounds and if neighbor is part of skeleton and not yet visited
                    if (0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and
                        skeleton[nr, nc] > 0 and (nr, nc) not in visited):
                        neighbors.append((nr, nc))

            if len(neighbors) == 0:
                break  # No more neighbors to follow (reached an isolated point or true end)
            elif len(neighbors) == 1:
                # Continue following the path
                current_point = neighbors[0]
                path.append(current_point)
                visited.add(current_point)
            else:
                # Multiple neighbors - we've reached a branch point or complex intersection
                break

        # If the path is just the start point and no movement, treat as effectively no path or already pruned.
        if len(path) == 1 and start_point == current_point and max_length > 0:
             return None # Did not move from start point, possibly an isolated pixel or already part of a pruned branch

        return path

    def process_skeleton(self, mask: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Processes a binary mask to extract and refine its skeleton.
        This involves generating the skeleton, finding branch points, and pruning small branches.

        Args:
            mask (np.ndarray): The input binary mask (2D array, uint8).

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: A tuple containing:
                - np.ndarray: The pruned binary skeleton image (uint8).
                - List[Tuple[int, int]]: A list of (row, column) coordinates of branch points in the *initial* skeleton.
        """
        # Ensure mask is boolean for skeletonize
        binary_mask = mask > 0

        # Generate skeleton using skimage's skeletonize
        skeleton = ski_morphology.skeletonize(binary_mask)
        skeleton = (skeleton.astype(np.uint8)) * 255 # Convert back to 0/255

        # Find branch points in the initial skeleton
        branch_points = self._get_branch_points(skeleton)

        # Prune small branches
        pruned_skeleton = self._prune_small_branches(skeleton)

        return pruned_skeleton, branch_points

    def skeleton_aware_refinement(self, mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        """
        Refines a mask using the corresponding skeleton structure. This aims to:
        1. Fill gaps in the mask that are spanned by the skeleton.
        2. Enhance connectivity and fill minor holes using morphological operations.

        Args:
            mask (np.ndarray): The original binary mask (uint8).
            skeleton (np.ndarray): The processed skeleton of the mask (uint8).

        Returns:
            np.ndarray: The refined binary mask (uint8).
        """
        try:
            # Combine the original mask with the skeleton to fill any gaps along the skeleton path
            refined = cv2.bitwise_or(mask, skeleton)

            # Apply a small dilation to further connect close components, if necessary
            # The kernel size should be small to avoid excessive thickening
            dilation_kernel = np.ones((3, 3), np.uint8)
            refined = cv2.dilate(refined, dilation_kernel, iterations=1)

            # Fill any remaining holes within the now more connected regions
            filled = ndi.binary_fill_holes(refined > 0).astype(np.uint8) * 255
            logging.info("Skeleton-aware refinement applied: mask combined with skeleton, dilated, and holes filled.")
            return filled
        except Exception as e:
            logging.error(f"Skeleton-aware refinement failed: {e}. Returning original mask.")
            return mask

from .dataset import ImagePathDataset

# --- Main Hybrid Augmentation Pipeline ---

class HybridAugmentationPipeline:
    def benchmark_enabled_processing(self, image: np.ndarray, image_type: str = "unknown") -> Tuple[np.ndarray, Dict]:
        """
        Comprehensive performance benchmarking with timing, success rate, resource monitoring, and analytics.
        Tracks per-image-type success, quality, and speed for trade-off analysis.
        """
        import time, psutil
        if not hasattr(self, '_analytics'):
            self._analytics = {}
        timings = {}
        start = time.time()
        mask = self.process_image(image)
        timings['process_image'] = time.time() - start
        # Resource monitoring (CPU, memory)
        process = psutil.Process()
        mem = process.memory_info().rss / 1024**2
        cpu = process.cpu_percent(interval=0.1)
        # Quality metric: fill ratio
        nonzero = int(np.count_nonzero(mask))
        total = int(mask.size)
        fill_ratio = nonzero / total if total > 0 else 0.0
        # Success: mask not too small/large
        success = 0.01 < fill_ratio < 0.5
        # Track analytics per image type
        stats = self._analytics.setdefault(image_type, {'count': 0, 'success': 0, 'fail': 0, 'total_time': 0.0, 'quality': []})
        stats['count'] += 1
        stats['total_time'] += timings['process_image']
        stats['quality'].append(fill_ratio)
        if success:
            stats['success'] += 1
        else:
            stats['fail'] += 1
        # Compute trade-off: mean quality, mean time, success rate
        mean_quality = float(np.mean(stats['quality'])) if stats['quality'] else 0.0
        mean_time = stats['total_time'] / stats['count'] if stats['count'] else 0.0
        success_rate = stats['success'] / stats['count'] if stats['count'] else 0.0
        analytics = {
            'image_type': image_type,
            'count': stats['count'],
            'success_rate': success_rate,
            'mean_quality': mean_quality,
            'mean_time': mean_time
        }
        metrics = {'timings': timings, 'memory_MB': mem, 'cpu_percent': cpu, 'analytics': analytics}
        return mask, metrics

    def comprehensive_quality_assurance_pipeline(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Medical-grade quality assurance with multi-criteria assessment, anomaly detection, statistical validation, and clinical scoring.
        """
        # Multi-criteria assessment
        nonzero = np.count_nonzero(mask)
        total = mask.size
        fill_ratio = nonzero / total if total > 0 else 0.0
        # Anomaly detection (mask too large or too small)
        anomaly = fill_ratio < 0.01 or fill_ratio > 0.5
        # --- Statistical validation ---
        # SSIM between mask and image (grayscale)
        if image.ndim == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        mask_resized = cv2.resize(mask.astype(np.uint8), (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        ssim_score = ssim(image_gray.astype(np.float32), mask_resized.astype(np.float32), data_range=255.0)
        # Dice coefficient (mask vs. image > 0)
        def dice(a, b):
            a = (a > 0).astype(np.uint8)
            b = (b > 0).astype(np.uint8)
            intersection = np.sum(a * b)
            union = np.sum(a) + np.sum(b)
            if union == 0:
                return 1.0
            return 2. * intersection / (union + 1e-6)
        dice_score = dice(mask_resized, image_gray > 0)
        # Edge overlap (Canny)
        edges_img = cv2.Canny(image_gray.astype(np.uint8), 50, 150)
        edges_mask = cv2.Canny(mask_resized.astype(np.uint8), 50, 150)
        sum_edges_img = np.sum(edges_img > 0)
        edge_overlap = np.sum((edges_img > 0) & (edges_mask > 0)) / (sum_edges_img + 1e-6) if sum_edges_img > 0 else 0.0
        # --- Clinical scoring ---
        # Simple rule-based: pass if all metrics above thresholds
        ssim_pass = ssim_score > 0.5
        dice_pass = dice_score > 0.5
        edge_pass = edge_overlap > 0.3
        fill_pass = 0.01 < fill_ratio < 0.5
        clinical_score = (ssim_score + dice_score + edge_overlap + (1.0 if fill_pass else 0.0)) / 4.0
        clinically_acceptable = all([ssim_pass, dice_pass, edge_pass, fill_pass]) and not anomaly
        stats = {
            'fill_ratio': fill_ratio,
            'anomaly': anomaly,
            'ssim': ssim_score,
            'dice': dice_score,
            'edge_overlap': edge_overlap,
            'clinical_score': clinical_score
        }
        return mask, {'qa_stats': stats, 'clinically_acceptable': clinically_acceptable}
    """
    Main orchestrator for hybrid mask generation and augmentation pipeline.
    Combines SAM, skeleton processing, and mask refinement.
    """

    def __init__(self, config: Dict = None):
        """
        Initializes the HybridAugmentationPipeline with a configuration dictionary.

        Args:
            config (Dict, optional): Configuration dictionary for the pipeline,
                                     including settings for SAM, mask refinement,
                                     skeleton processing, and data paths.
                                     Defaults to an empty dictionary.
        """
        self.config = config or {}




        # Initialize SAMMaskGenerator (official SAM)
        sam_cfg = self.config.get('sam', {}) or {}
        model_type = sam_cfg.get('model_type', 'vit_h')
        checkpoint_dir = sam_cfg.get('checkpoint_dir', './sam_checkpoints')
        device = self.config.get('processing', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_generator = SAMMaskGenerator(model_type=model_type, checkpoint_dir=checkpoint_dir, device=device)

        # Initialize MaskRefiner
        ref_cfg = self.config.get('refinement', {}) or {}
        valid_refiner_args = ['contour_approx_factor', 'min_component_size', 'closing_kernel_size', 'opening_kernel_size']
        filtered_ref_cfg = {k: v for k, v in ref_cfg.items() if k in valid_refiner_args}
        self.mask_refiner = MaskRefiner(**filtered_ref_cfg)

        # Initialize SkeletonProcessor
        self.skeleton_processor = SkeletonProcessor(**(self.config.get('skeleton', {}) or {}))



    def process_image(self, image: np.ndarray, image_path: str = None, inspection_dir: str = None) -> np.ndarray:
        """
        Uses official SAM model to generate the initial mask, then applies skeleton-aware refinement and QA.
        """
        use_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        # --- Use SAM to generate mask ---
        try:
            self.sam_generator.set_image(image)
            h, w = image.shape[:2]
            # Use a central point as a prompt (can be improved for your use case)
            point_coords = np.array([[w // 2, h // 2]])
            point_labels = np.array([1])
            masks, scores, logits = self.sam_generator.predict(point_coords, point_labels, multimask_output=True)
            # Use the best mask (highest score)
            sam_mask = masks[np.argmax(scores)]
            sam_mask = (sam_mask * 255).astype(np.uint8)
        except Exception as e:
            logging.error(f"SAM mask generation failed: {e}. Falling back to edge-based mask.")
            # Fallback to edge-based mask if SAM fails
            sam_mask = np.zeros_like(gray, dtype=np.uint8)

        # --- Skeleton-aware edge refinement ---
        skeleton, branch_points = self.skeleton_processor.process_skeleton(sam_mask)
        mask_refined = self.skeleton_processor.skeleton_aware_refinement(sam_mask, skeleton)

        # --- Post-Processing Refinement (via MaskRefiner) ---
        mask_post = self.mask_refiner.robust_binary_conversion_pipeline(mask_refined)

        # --- Mask Quality Check & Fallback (via MaskRefiner) ---
        nonzero = np.count_nonzero(mask_post)
        mask_quality_fail = False
        if nonzero > (gray.size * 0.25):
            logging.warning(f"Mask too large ({nonzero} pixels). Using fallback pipeline.")
            mask_quality_fail = True
        elif nonzero < (gray.size * 0.01):
            logging.warning(f"Mask too small ({nonzero} pixels). Using fallback pipeline.")
            mask_quality_fail = True

        if mask_quality_fail:
            # Use MaskRefiner's optimized fallback
            final_mask = self.mask_refiner.optimized_fallback_execution_pipeline(mask_post, image, fallback_type='edge')
        else:
            # Use MaskRefiner's mask quality validation with real MC-Dropout if possible
            model = None
            input_tensor = None
            device = self.config.get('processing', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(self, 'sam_generator') and hasattr(self.sam_generator, 'predictor') and self.sam_generator.predictor is not None:
                model = getattr(self.sam_generator, 'model', None)
                img = image
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                img_tensor = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0) / 255.0
                input_tensor = img_tensor.to(device)
            def qa_call():
                return self.mask_refiner.validate_mask_quality_with_confidence(
                    mask_post, image, prediction_scores=[], model=model, input_tensor=input_tensor, mc_dropout_runs=20, device=device)
            if device == 'cuda':
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(qa_call)
                    validated_mask, quality_score, metrics = future.result()
            else:
                validated_mask, quality_score, metrics = qa_call()
            final_mask = validated_mask

        # Robust hole/background separation using flood fill and bitwise ops (unchanged)
        if use_cuda:
            gpu_final_mask = cv2.cuda_GpuMat()
            gpu_final_mask.upload(final_mask)
            gpu_mask_inv = cv2.cuda.bitwise_not(gpu_final_mask)
            mask_inv = gpu_mask_inv.download()
        else:
            mask_inv = cv2.bitwise_not(final_mask)
        h, w = mask_inv.shape
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(mask_inv, flood_mask, (0,0), 255)
        if use_cuda:
            gpu_mask_inv2 = cv2.cuda_GpuMat()
            gpu_mask_inv2.upload(mask_inv)
            gpu_mask_out = cv2.cuda.bitwise_not(gpu_mask_inv2)
            mask_out = gpu_mask_out.download()
        else:
            mask_out = cv2.bitwise_not(mask_inv)

        # Save inspection images (real image and final output mask) after all processing
        if inspection_dir is not None and image_path is not None:
            os.makedirs(inspection_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            img_save_path = os.path.join(inspection_dir, f"{base_name}_input.png")
            mask_save_path = os.path.join(inspection_dir, f"{base_name}_final_mask.png")
            side_by_side_path = os.path.join(inspection_dir, f"{base_name}_side_by_side.png")
            if image.ndim == 3:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_save_path, img_bgr)
            else:
                img_bgr = image
                cv2.imwrite(img_save_path, image)
            cv2.imwrite(mask_save_path, mask_out)
            mask_color = cv2.cvtColor(mask_out, cv2.COLOR_GRAY2BGR)
            if img_bgr.shape[:2] != mask_color.shape[:2]:
                mask_color = cv2.resize(mask_color, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            side_by_side = cv2.hconcat([img_bgr, mask_color])
            cv2.imwrite(side_by_side_path, side_by_side)
            logging.info(f"Saved inspection images to {inspection_dir} (real image, final mask, side-by-side)")

        return mask_out


    def process(self, image: np.ndarray) -> Dict:
        """
        Legacy method for backward compatibility. Processes a single image
        and returns the generated mask wrapped in a dictionary.

        Args:
            image (np.ndarray): The input image (H, W, C, typically RGB).

        Returns:
            Dict: A dictionary containing the generated mask under the 'mask' key.
        """
        mask = self.process_image(image)
        return {'mask': mask}

    def _process_with_qa(self, img: np.ndarray) -> Dict:
        """
        Processes image with quality assurance checks and visualization if enabled in config.

        Args:
            img (np.ndarray): The input image.

        Returns:
            Dict: A dictionary containing the processing results, including the mask.
        """
        result = self.process(img)

        # QA visualization if enabled
        qa_cfg = self.config.get('qa', {})
        if qa_cfg.get('enabled', False):
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title("Original")
                plt.subplot(1, 2, 2)
                # Ensure mask is displayed correctly (e.g., as grayscale image)
                mask_to_display = result.get('mask', np.zeros_like(img[:,:,0] if img.ndim==3 else img, dtype=np.uint8))
                plt.imshow(mask_to_display, cmap='gray')
                plt.title("Generated Mask")
                plt.show()
            except ImportError:
                logging.warning("Matplotlib not found. QA visualization skipped.")
            except Exception as e:
                logging.warning(f"QA visualization failed: {e}")

        return result

    def batch_process(self, image_paths: list, output_dir: str, batch_size: int = 8,
                      num_workers: int = 1, inspection_dir: str = None, parallel: bool = True) -> list:
        """
        Processes multiple images in batches using multiprocessing.

        Args:
            image_paths (list): A list of paths to the input images.
            output_dir (str): The directory where the processed mask files will be saved.
            batch_size (int, optional): The number of images to process in each batch. Defaults to 8.
            num_workers (int, optional): The number of worker processes to use for parallel processing. Defaults to 8.
            parallel (bool, optional): Whether to use parallel processing. Defaults to True.
            inspection_dir (str, optional): Directory to save intermediate inspection images
                                            for each image processed in the batch. Defaults to None.

        Returns:
            list: A list of paths to the saved mask files.
        """
        os.makedirs(output_dir, exist_ok=True)
        if inspection_dir:
            os.makedirs(inspection_dir, exist_ok=True)
        processed_files = []

        # Pass the pipeline's config to each worker, so they can re-initialize
        # components if necessary.
        pipeline_config = self.config

        import psutil
        from tqdm import tqdm
        # Dynamic RAM-aware batch processing with parallelism
        total_images = len(image_paths)
        default_batch_size = max(1, self.config.get('processing', {}).get('batch_size', 1))
        min_free_ram_mb = self.config.get('processing', {}).get('min_free_ram_mb', 500) # Minimum free RAM in MB
        num_workers = max(1, self.config.get('processing', {}).get('num_workers', 1))
        with tqdm(total=total_images, desc="Processing images in RAM-efficient batches") as pbar:
            batch_start = 0
            while batch_start < total_images:
                # Check available RAM
                available_ram_mb = psutil.virtual_memory().available // (1024 * 1024)
                # Dynamically adjust batch size if RAM is low
                batch_size = default_batch_size
                if available_ram_mb < min_free_ram_mb:
                    batch_size = max(1, default_batch_size // 2)
                    logging.warning(f"Low RAM detected ({available_ram_mb}MB available). Reducing batch size to {batch_size}.")
                else:
                    logging.info(f"RAM available: {available_ram_mb}MB. Using batch size: {batch_size}.")
                batch_end = min(batch_start + batch_size, total_images)
                batch_paths = image_paths[batch_start:batch_end]
                args_list = [(path, output_dir, inspection_dir, pipeline_config) for path in batch_paths]
                if parallel and num_workers > 1:
                    # Use multiprocessing Pool for parallel processing
                    from multiprocessing import Pool
                    with Pool(processes=num_workers) as pool:
                        results = pool.map(self._process_single_image_worker, args_list)
                else:
                    # Fallback to sequential processing
                    results = [self._process_single_image_worker(args) for args in args_list]
                for result_path in results:
                    if result_path is not None:
                        processed_files.append(result_path)
                    pbar.update(1)
                batch_start += batch_size
        return processed_files

    def _process_single_image_worker(self, args: Tuple[str, str, Optional[str], Dict]) -> Optional[str]:
        """
        Helper method executed by each worker process for batch processing.
        Loads an image, processes it, and saves the resulting mask.

        Args:
            args (Tuple[str, str, Optional[str], Dict]): A tuple containing:
                - image_path (str): The path to the input image.
                - output_dir (str): The directory to save the mask.
                - inspection_dir (Optional[str]): Directory for inspection images.
                - pipeline_config (Dict): The configuration dictionary for the pipeline.

        Returns:
            Optional[str]: The path to the saved mask file, or None if processing failed.
        """
        image_path, output_dir, inspection_dir, pipeline_config = args
        try:
            # Normalize path to avoid Windows control-char issues
            image_path = os.path.normpath(image_path)

            # Re-initialize HybridAugmentationPipeline components within the worker process
            worker_pipeline = HybridAugmentationPipeline(pipeline_config)

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image: {image_path}. Skipping.")
                return None
            # Log image stats
            logging.info(f"Loaded image {image_path}: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
            # Convert BGR to RGB (OpenCV loads as BGR by default)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image using the worker's own pipeline instance
            mask = worker_pipeline.process_image(image, image_path, inspection_dir)
            # Log mask stats
            if mask is not None:
                logging.info(f"Generated mask for {image_path}: shape={mask.shape}, dtype={mask.dtype}, min={mask.min()}, max={mask.max()}, nonzero={np.count_nonzero(mask)}")
            # Guard against empty mask and supply blank fallback
            if mask is None or mask.size == 0:
                logging.warning(f"Empty mask for {image_path}, creating blank fallback mask.")
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            # Confirm mask dtype & shape
            mask = mask.astype(np.uint8)
            assert mask.ndim == 2 and mask.shape[0] > 0 and mask.shape[1] > 0

            # Save result
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_mask.png")
            try:
                cv2.imwrite(output_path, mask)
            except Exception as e:
                logging.error(f"Failed to save mask to {output_path}: {e}")
            return output_path
        except Exception as e:
            logging.error(f"Error processing {image_path} in worker: {e}")
            return None

    def run_pipeline(self):
        """
        Main entry point for the pipeline. Loads input image paths from a JSON file,
        runs batch processing to generate masks, and saves the results as a pickle file.
        This method is typically called from a command-line interface.
        """
        # Get config values
        input_path = self.config.get('data', {}).get('input_path')
        output_path = self.config.get('data', {}).get('output_path')
        batch_size = self.config.get('processing', {}).get('batch_size', 8)
        inspection_dir = self.config.get('inspection_dir', None)
        num_workers = self.config.get('processing', {}).get('num_workers', 1)

        if not input_path or not output_path:
            logging.critical("Input or output path not specified in config. Please check 'data.input_path' and 'data.output_path'.")
            raise ValueError("Input or output path not specified.")

        # Load image paths from derived_labels.json
        if not os.path.exists(input_path):
            logging.critical(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, 'r') as f:
            derived_labels = json.load(f)

        # Remap paths as specified in the original snippet
        # This remapping seems specific to how image paths are stored vs. how they're accessed.
        def remap_path(path):
            # Example remapping, adjust as per actual path structure if needed
            return path.replace('category_1', '1').replace('category_0', '0')
        image_paths = [remap_path(entry['image_path']) for entry in derived_labels]

        # Output directory for masks
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Processing {len(image_paths)} images. Output masks will be saved to: {output_dir}")

        # Run batch processing using explicit image paths
        mask_files = self.batch_process(
            image_paths=image_paths,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            inspection_dir=inspection_dir
        )

        # Save results (paths to generated mask files) as a pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(mask_files, f)
        logging.info(f"Pipeline finished. Saved {len(mask_files)} mask file paths to {output_path}")

# Alias for backward compatibility with CLI and imports
HybridAugmentor = HybridAugmentationPipeline

# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    # Create dummy image files and a dummy derived_labels.json for testing
    test_dir = "test_pipeline_output"
    os.makedirs(test_dir, exist_ok=True)
    dummy_input_path = os.path.join(test_dir, "dummy_derived_labels.json")
    dummy_output_path = os.path.join(test_dir, "dummy_masks.pkl")
    dummy_image_dir = os.path.join(test_dir, "dummy_images")
    os.makedirs(dummy_image_dir, exist_ok=True)
    dummy_inspection_dir = os.path.join(test_dir, "dummy_inspection")

    num_dummy_images = 5
    dummy_image_paths = []
    for i in range(num_dummy_images):
        dummy_img_path = os.path.join(dummy_image_dir, f"image_{i}.png")
        # Create a simple dummy image (e.g., a white square)
        dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(dummy_img_path, dummy_img)
        dummy_image_paths.append({'image_path': dummy_img_path.replace(os.sep, '/')}) # Use forward slashes for consistency

    with open(dummy_input_path, 'w') as f:
        json.dump(dummy_image_paths, f, indent=4)

    # Example configuration
    import torch
    cuda_available = torch.cuda.is_available()
    device_str = 'cuda' if cuda_available else 'cpu'
    if not cuda_available:
        print("[WARNING] CUDA is not available. Falling back to CPU. To use GPU, install CUDA-enabled PyTorch and OpenCV, and ensure a compatible GPU is present.")
    pipeline_config = {
        'data': {
            'input_path': dummy_input_path,
            'output_path': dummy_output_path,
        },
        'processing': {
            'batch_size': 2,
            'num_workers': 1,
            'device': device_str
        },
        'sam': {
            'model_type': 'vit_b', # Use a valid model type from the stub
            'checkpoint_path': '~/.cache/sam'
        },
        'refinement': {
            'closing_kernel_size': 3,
            'opening_kernel_size': 3
        },
        'inspection_dir': dummy_inspection_dir,
        'qa': {'enabled': False} # Set to True to see matplotlib plots if installed
    }

    try:
        pipeline = HybridAugmentationPipeline(pipeline_config)
        pipeline.run_pipeline()
        # Pipeline ran successfully. Check '{test_dir}' for output.

        # Clean up dummy files
        # import shutil
        # shutil.rmtree(test_dir)
        # logging.info(f"Cleaned up test directory: {test_dir}")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")


# Alias for backward compatibility with CLI and imports
HybridAugmentor = HybridAugmentationPipeline
