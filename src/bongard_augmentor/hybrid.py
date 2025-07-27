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
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import SAM components, set SAM_AVAILABLE flag
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
    logging.info("SAM library imported successfully.")
except ImportError:
    logging.warning("Segment Anything Model (SAM) library not found. SAM functionalities will be disabled.")
    SAM_AVAILABLE = False
except Exception as e:
    logging.error(f"Error importing SAM library: {e}. SAM functionalities will be disabled.")
    SAM_AVAILABLE = False


# ==================================================================
# Section: Advanced Mask Refinement
# ==================================================================

class MaskRefiner:
    """
    An advanced, professional-grade mask refinement and generation pipeline.
    Uses a two-stage process for refinement: contour-based simplification and morphological cleaning.
    For initial mask generation, it leverages an ensemble of Segment Anything Models (SAM)
    with adaptive prompting, and provides a robust fallback stack if SAM prediction fails.
    This approach is more robust than simple skeletonization for preserving object shape.
    """
    def __init__(self,
                 contour_approx_factor: float = 0.005,
                 min_component_size: int = 50,
                 closing_kernel_size: int = 5,
                 opening_kernel_size: int = 3,
                 sam_model_types: Optional[List[str]] = None,
                 sam_cache_dir: Optional[str] = None):
        """
        Initializes the MaskRefiner with parameters for mask processing and SAM integration.

        Args:
            contour_approx_factor (float): Factor for epsilon in contour approximation.
                                            Smaller values result in more precise contours.
            min_component_size (int): Minimum size (in pixels) for connected components to be kept
                                    after morphological operations.
            closing_kernel_size (int): Size of the kernel for morphological closing.
            opening_kernel_size (int): Size of the kernel for morphological opening.
            sam_model_types (Optional[List[str]]): List of SAM model types to load (e.g., ['vit_h', 'vit_l']).
                                                    If None, defaults to ['vit_h'].
            sam_cache_dir (Optional[str]): Directory to cache SAM checkpoints. Defaults to './sam_checkpoints'.
        """
        self.contour_approx_factor = contour_approx_factor
        self.min_component_size = min_component_size
        self.closing_kernel_size = closing_kernel_size
        self.opening_kernel_size = opening_kernel_size

        # Precompute distance threshold for skeleton grow (though not directly used in provided code)
        self.grow_threshold = opening_kernel_size

        # SAM-related attributes
        self.models = {}
        self.predictors = {}
        self.is_initialized = False
        self.device = "cuda" if SAM_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(sam_cache_dir if sam_cache_dir else "./sam_checkpoints")
        self.ensemble_weights = None # Will be set during SAM initialization

        # Initialize SAM models if available
        if SAM_AVAILABLE:
            self._initialize_sam_models(sam_model_types)
        else:
            logging.warning("SAM is not available, mask generation will rely solely on fallback methods.")

    def _initialize_sam_models(self, model_types: Optional[List[str]] = None):
        """
        Initializes and loads SAM models and predictors. Downloads checkpoints if necessary.

        Args:
            model_types (Optional[List[str]]): List of SAM model types to load.
                                                Defaults to ['vit_h'] if None.
        """
        if not SAM_AVAILABLE:
            logging.warning("SAM library not imported, skipping SAM model initialization.")
            return

        if model_types is None:
            model_types = ['vit_h'] # Default to the largest model if not specified

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_map = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth',
        }
        
        # Default ensemble weights (can be adjusted based on model performance)
        # Assuming equal weight for now, but could be set per model type if needed
        self.ensemble_weights = [1.0] * len(model_types)

        loaded_count = 0
        for i, model_type in enumerate(model_types):
            checkpoint_file = checkpoint_map.get(model_type)
            if not checkpoint_file:
                logging.warning(f"Unknown SAM model type: {model_type}. Skipping.")
                continue

            target_path = self.cache_dir / checkpoint_file
            checkpoint_path = None

            if target_path.exists():
                checkpoint_path = target_path
                logging.info(f"Found SAM checkpoint {checkpoint_file} at {checkpoint_path}.")
            else:
                logging.info(f"SAM checkpoint {checkpoint_file} not found locally. Attempting download...")
                checkpoint_urls = {
                    'sam_vit_h_4b8939.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                    'sam_vit_l_0b3195.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                    'sam_vit_b_01ec64.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                }
                url = checkpoint_urls.get(checkpoint_file)
                if url:
                    try:
                        import requests
                        logging.info(f"Downloading SAM checkpoint {checkpoint_file} from {url} ...")
                        r = requests.get(url, stream=True)
                        r.raise_for_status()
                        with open(target_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        checkpoint_path = target_path
                        logging.info(f"Downloaded SAM checkpoint to {checkpoint_path}")
                    except Exception as e:
                        logging.error(f"Failed to download {checkpoint_file}: {e}")
                else:
                    logging.error(f"No download URL found for checkpoint: {checkpoint_file}")

            if checkpoint_path is None:
                logging.error(f"SAM checkpoint not found and could not be downloaded for {model_type}. "
                              f"Tried: {checkpoint_file}")
                continue

            try:
                # Load model
                if sam_model_registry and model_type in sam_model_registry:
                    sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(self.device)
                    self.models[model_type] = sam_model
                    # Create predictor
                    predictor = SamPredictor(sam_model)
                    self.predictors[model_type] = predictor
                    loaded_count += 1
                    logging.info(f"Loaded SAM model: {model_type} from {checkpoint_path}")
                else:
                    logging.error(f"SAM model registry does not contain '{model_type}'.")
            except Exception as e:
                logging.error(f"Failed to load SAM model {model_type}: {e}")

        self.is_initialized = loaded_count > 0
        if self.is_initialized:
            logging.info(f"SAM ensemble initialized with {loaded_count} model(s): {list(self.models.keys())}")
            # Adjust ensemble weights if the actual number of loaded models is different
            self.ensemble_weights = [1.0] * loaded_count
        else:
            logging.error("No SAM models loaded. SAM functionalities will be disabled.")

    def _contour_refinement(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Refines a binary mask using contour approximation.
        Simplifies contours and fills resulting shapes, smoothing out jagged edges.

        Args:
            mask_bin (np.ndarray): Input binary mask (2D array, uint8, values 0 or 255).

        Returns:
            np.ndarray: Refined binary mask.
        """
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.debug("No contours found for refinement.")
            return np.zeros_like(mask_bin, dtype=np.uint8)

        mask_refined = np.zeros_like(mask_bin, dtype=np.uint8)
        for cnt in contours:
            epsilon = self.contour_approx_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(mask_refined, [approx], -1, 255, -1)
        logging.debug(f"Contour refinement completed. Original non-zero: {np.count_nonzero(mask_bin)}, refined non-zero: {np.count_nonzero(mask_refined)}")
        return mask_refined

    def _fallback_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generates a simple fallback binary mask using Otsu thresholding.
        Used when advanced SAM or other methods fail.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            np.ndarray: A binary mask (uint8).
        """
        logging.info("Using simple Otsu thresholding as fallback mask.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def _auto_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Attempts to generate an automatic mask using SAM's automatic mask generation.
        This method is a placeholder as actual SAM auto-masking requires specific SAM tooling
        that is not part of the basic `SamPredictor`.
        In a real implementation, this would involve `SamAutomaticMaskGenerator`.

        Args:
            image (np.ndarray): The input image (H, W, C).

        Returns:
            Optional[np.ndarray]: The generated binary mask (uint8) if successful, otherwise None.
        """
        if not SAM_AVAILABLE or not self.is_initialized:
            logging.warning("SAM not available or not initialized for auto-mask generation.")
            return None

        try:
            # Placeholder for SAM's automatic mask generation.
            # Real implementation would use segment_anything.SamAutomaticMaskGenerator
            # For this example, we'll return None as the functionality is complex and not fully provided.
            logging.debug("SAM auto-mask generation is a placeholder. Returning None.")
            return None # Replace with actual SAM auto-mask generation logic
        except Exception as e:
            logging.error(f"Auto mask generation failed: {e}")
            return None

    def _generate_masks_single_model(self, image: np.ndarray, prompts: List[Dict], model_name: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generates masks using a single SAM model with given prompts.

        Args:
            image (np.ndarray): The input image (H, W, C).
            prompts (List[Dict]): A list of prompt dictionaries, each containing
                                    'point_coords' (np.ndarray) and 'point_labels' (np.ndarray).
            model_name (str): The name of the SAM model to use (e.g., 'vit_h').

        Returns:
            Tuple[List[np.ndarray], List[float]]: A tuple containing a list of generated
                                                binary masks and a list of their corresponding scores.
        """
        if not SAM_AVAILABLE or not self.is_initialized or model_name not in self.predictors:
            logging.warning(f"SAM not available or model '{model_name}' not initialized. Cannot generate masks.")
            return [], []

        predictor = self.predictors[model_name]
        masks = []
        scores = []

        try:
            predictor.set_image(image)
            for prompt in prompts:
                point_coords = prompt.get('point_coords')
                point_labels = prompt.get('point_labels')

                if point_coords is None or point_labels is None:
                    logging.warning(f"Skipping malformed prompt: {prompt}")
                    continue

                # SAM prediction can return multiple masks per prompt; taking the best one (highest score)
                # This assumes prompt points are for a single object.
                predicted_masks, predicted_scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True # Allow multiple masks for the same prompt point
                )

                if predicted_masks.size > 0:
                    best_mask_idx = np.argmax(predicted_scores)
                    masks.append((predicted_masks[best_mask_idx] * 255).astype(np.uint8))
                    scores.append(predicted_scores[best_mask_idx])
                    logging.debug(f"Generated mask with {model_name} for prompt. Score: {predicted_scores[best_mask_idx]:.4f}")
                else:
                    logging.debug(f"No mask generated by {model_name} for current prompt.")

            return masks, scores
        except Exception as e:
            logging.error(f"Single model mask generation failed for {model_name}: {e}")
            return [], []

    def generate_adaptive_prompts(self, image: np.ndarray) -> List[Dict]:
        """
        Generates diverse, content-aware prompts for robust mask generation with SAM.
        Includes edge-based, multi-scale grid, and center/corner prompts to cover various
        object types and image compositions.

        Args:
            image (np.ndarray): The input image (H, W, C).

        Returns:
            List[Dict]: A list of prompt dictionaries, suitable for SAM's predict method.
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: # Handle RGBA to RGB conversion
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        h, w = image.shape[:2]
        prompts = []

        # 1. Edge-based prompting: Sample points along prominent edges
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_points_coords = np.column_stack(np.where(edges > 0)) # (row, col) -> (y, x)

        if len(edge_points_coords) > 0:
            # Sample edge points strategically (e.g., 5 points)
            n_edge_samples = min(5, len(edge_points_coords))
            if n_edge_samples > 0:
                # Use np.random.default_rng for modern random number generation
                rng = np.random.default_rng()
                edge_indices = rng.choice(len(edge_points_coords), n_edge_samples, replace=False)
                for idx in edge_indices:
                    y, x = edge_points_coords[idx]
                    prompts.append({
                        'point_coords': np.array([[x, y]]), # SAM expects (x, y)
                        'point_labels': np.array([1]) # Foreground point
                    })
        logging.debug(f"Generated {len(prompts)} edge-based prompts.")

        # 2. Multi-scale grid prompting: Cover the image with points at different densities
        for scale in [0.3, 0.5, 0.7]:
            # Ensure grid_size is at least 1 and does not exceed image dimensions
            grid_step = max(1, int(min(h, w) * scale))
            for i in range(grid_step // 2, h, grid_step):
                for j in range(grid_step // 2, w, grid_step):
                    if i < h and j < w: # Ensure points are within bounds
                        prompts.append({
                            'point_coords': np.array([[j, i]]),
                            'point_labels': np.array([1])
                        })
        logging.debug(f"Generated {len(prompts)} total prompts after grid-based.")

        # 3. Center and corner prompts: Ensure basic coverage
        center_prompts = [
            {'point_coords': np.array([[w // 2, h // 2]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[w // 4, h // 4]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[3 * w // 4, h // 4]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[w // 4, 3 * h // 4]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[3 * w // 4, 3 * h // 4]]), 'point_labels': np.array([1])},
        ]
        prompts.extend(center_prompts)
        logging.debug(f"Generated {len(prompts)} total prompts after center/corner.")

        # Limit to prevent excessive computation for a large number of prompts
        # A reasonable limit might be around 15-30 prompts for performance.
        return prompts[:30]

    def generate_ensemble_masks(self, image: np.ndarray, use_ensemble: bool = True) -> np.ndarray:
        """
        Generates masks using an ensemble of SAM models with adaptive prompting.
        Includes a robust fallback stack if SAM prediction fails or produces no masks.

        Args:
            image (np.ndarray): The input image (H, W, C).
            use_ensemble (bool): Whether to use the ensemble of models or just the first one.

        Returns:
            np.ndarray: The final combined binary mask (H, W).
        """
        if not self.is_initialized:
            logging.warning("SAM models not initialized. Falling back to simple thresholding.")
            return self._fallback_mask(image)

        # Ensure image is in correct format (RGB) for SAM
        if image.ndim == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image

        try:
            # Thin-line art detection: consider using SAM's automatic mask generator if few edges
            # This is a heuristic and might need tuning.
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            if edges.sum() < 200: # Heuristic for thin-line art (very few strong edges)
                logging.info("Detected potential thin-line art. Attempting SAM auto-mask (placeholder).")
                auto_mask = self._auto_mask(image_rgb)
                if auto_mask is not None and auto_mask.max() > 0:
                    logging.info("Successfully generated mask using SAM auto-mask (placeholder).")
                    return auto_mask
                else:
                    logging.warning("SAM auto-mask failed or produced no mask for thin-line art. Proceeding with prompts.")

            # Generate adaptive prompts
            prompts = self.generate_adaptive_prompts(image_rgb)

            if not prompts:
                logging.warning("No adaptive prompts generated. Falling back to simple thresholding.")
                return self._fallback_mask(image_rgb)

            if not use_ensemble or len(self.models) == 1:
                # Use only the first available model if ensemble is disabled or only one model exists
                model_name = list(self.models.keys())[0]
                masks, scores = self._generate_masks_single_model(image_rgb, prompts, model_name)
                if masks:
                    best_mask_idx = np.argmax(scores)
                    logging.info(f"Generated mask using single model {model_name}.")
                    return masks[best_mask_idx]
                else:
                    logging.warning(f"Single model {model_name} produced no masks for the given prompts.")
                    return self._run_advanced_fallback_stack(image_rgb)

            # Ensemble prediction
            all_masks = []
            all_scores = []

            # Ensure ensemble_weights matches the number of actual loaded models
            current_model_names = list(self.models.keys())
            if not self.ensemble_weights or len(self.ensemble_weights) != len(current_model_names):
                logging.warning("Ensemble weights not correctly set or mismatch with loaded models. Using equal weights.")
                self.ensemble_weights = [1.0] * len(current_model_names)

            for i, model_name in enumerate(current_model_names):
                weight = self.ensemble_weights[i]
                masks, scores = self._generate_masks_single_model(image_rgb, prompts, model_name)
                if len(masks) > 0:
                    # Weight scores by model confidence
                    weighted_scores = [score * weight for score in scores]
                    all_masks.extend(masks)
                    all_scores.extend(weighted_scores)
                    logging.info(f"[MASK] Ensemble model {model_name} produced {len(masks)} mask(s).")
                else:
                    logging.info(f"[MASK] Ensemble model {model_name} produced no masks for current prompts.")

            if not all_masks:
                logging.warning("No masks generated by ensemble. Using advanced professional fallback stack.")
                return self._run_advanced_fallback_stack(image_rgb)

            # Combine masks: Select the mask with the highest overall weighted score.
            best_overall_mask_idx = np.argmax(all_scores)
            final_mask = all_masks[best_overall_mask_idx]
            logging.info(f"Ensemble successfully generated a mask with overall best score.")
            return final_mask

        except Exception as e:
            logging.error(f"Ensemble mask generation failed: {e}. Falling back to simple thresholding.")
            return self._fallback_mask(image_rgb)

    def _potrace_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Performs a Potrace-like contour fill using OpenCV. This method finds contours
        and then fills them to create a solid mask, useful for vectorization-style fill.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The filled binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting Potrace-like contour fill.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary, dtype=np.uint8)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, -1)
        nonzero = np.count_nonzero(mask)
        logging.info(f"Potrace-like fill produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _skeleton_graph_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates a skeleton-aware mask. It skeletonizes the binary image and converts it
        back to a mask. Useful for preserving thin structures.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The skeleton-based binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting skeleton-aware mask generation.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        # Apply Otsu thresholding for robustness before skeletonization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        skeleton = skeletonize(binary > 0)
        skeleton_mask = (skeleton.astype(np.uint8)) * 255
        nonzero = np.count_nonzero(skeleton_mask)
        logging.info(f"Skeleton mask has {nonzero} nonzero pixels.")
        return skeleton_mask if nonzero > 0 else None

    def _deep_lineart_segmentation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Uses adaptive thresholding as a "deep segmentation" fallback,
        suitable for images with varying illumination.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The adaptively thresholded binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting adaptive thresholding for lineart segmentation.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2) # Block size 11, C=2
        nonzero = np.count_nonzero(mask)
        logging.info(f"Adaptive thresholding produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _edge_watershed_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Performs edge-aware watershed segmentation using OpenCV and scikit-image.
        This method is good for segmenting objects with well-defined edges.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The watershed-segmented binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting edge-aware watershed segmentation.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)

        # Ensure markers are not empty to prevent watershed from crashing on some inputs
        distance = ndi.distance_transform_edt(edges == 0)
        # Use a percentile to get a robust threshold for local maxima
        local_maxi_threshold = np.percentile(distance, 99.5) # Adjusted percentile
        local_maxi = (distance > local_maxi_threshold).astype(np.uint8)

        # Label markers for watershed
        markers, num_markers = ndi.label(local_maxi)
        if num_markers == 0:
            logging.warning("No markers found for watershed, returning None.")
            return None

        # Watershed requires a 3-channel image
        image_3channel = np.stack([gray] * 3, axis=-1)
        labels = cv2.watershed(image_3channel, markers)
        # Pixels with -1 are boundaries, 0 is background. We want foreground regions (>0).
        mask = (labels > 0).astype(np.uint8) * 255
        nonzero = np.count_nonzero(mask)
        logging.info(f"Watershed produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _hed_crf_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Uses Canny edge detection followed by a dilation as a HED-like fallback.
        Aims to connect broken edge lines to form a more complete mask.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The edge-based binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting HED/CRF-like edge fill (Canny + Dilation).")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 100, 200) # Canny parameters can be tuned
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1) # Dilate to close small gaps
        nonzero = np.count_nonzero(mask)
        logging.info(f"HED/CRF-like fill produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _learnable_morph_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies morphological closing and opening operations, followed by Otsu thresholding.
        Acts as a "learnable morph" fallback by applying standard morphological operations.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The morphologically processed binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting morphological closing/opening.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        kernel = np.ones((self.closing_kernel_size, self.closing_kernel_size), np.uint8)
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        _, mask = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        nonzero = np.count_nonzero(mask)
        logging.info(f"Morphological ops produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _pre_thicken(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies dilation to thicken thin lines in the input image.
        Useful as a pre-processing step for other mask generation methods.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The thickened image (uint8) if successful, otherwise None.
        """
        logging.info("Attempting pre-thicken (dilation) operation.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        kernel = np.ones((3,3), np.uint8)
        thick = cv2.dilate(gray, kernel, iterations=1)
        nonzero = np.count_nonzero(thick)
        logging.info(f"Pre-thicken produced mask with {nonzero} nonzero pixels.")
        return thick if nonzero > 0 else None

    def _polygon_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a polygon mask by finding contours and approximating them.
        Similar to `_contour_refinement` but used here as a standalone fallback.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The polygon-filled binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting polygon mask extraction.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary, dtype=np.uint8)
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True) # Fixed epsilon for this fallback
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(mask, [approx], -1, 255, -1)
        nonzero = np.count_nonzero(mask)
        logging.info(f"Polygon mask produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _mser_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects Maximally Stable Extremal Regions (MSER) using OpenCV and forms a mask
        from their convex hulls. Useful for detecting blobs or text regions.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The MSER-based binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting MSER region detection.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        mask = np.zeros_like(gray, dtype=np.uint8)
        for region in regions:
            # Reshape region to (N, 1, 2) as required by cv2.convexHull
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], -1, 255, -1)
        nonzero = np.count_nonzero(mask)
        logging.info(f"MSER produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _grabcut_refine(self, image: np.ndarray, initial_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Refines an initial mask using the GrabCut algorithm. This method is interactive
        and uses a graph-cut approach to segment the foreground from the background.

        Args:
            image (np.ndarray): The input image (H, W, C).
            initial_mask (np.ndarray): An initial binary mask (uint8) where foreground pixels are >0.

        Returns:
            Optional[np.ndarray]: The GrabCut refined binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting GrabCut refinement.")
        if image.ndim == 2: # GrabCut expects 3-channel image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: # Convert RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        mask = np.zeros(image.shape[:2], np.uint8)
        # Initialize GrabCut mask: Probable foreground (2) where initial_mask is >0
        mask[initial_mask > 0] = cv2.GC_PR_FGD
        # Remaining pixels are assumed to be probable background (0)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        # Define a rectangle around the entire image (or the region of interest)
        rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2) # Margins of 1 pixel

        try:
            # Run GrabCut for 5 iterations
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            # Extract final mask: foreground (1) and probable foreground (3) pixels
            result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            nonzero = np.count_nonzero(result_mask)
            logging.info(f"GrabCut produced mask with {nonzero} nonzero pixels.")
            return result_mask if nonzero > 0 else None
        except Exception as e:
            logging.error(f"GrabCut failed: {e}")
            return None

    def _multi_scale_threshold(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Applies a combination of adaptive and Otsu thresholding to create a mask.
        Aims to capture details across different intensity ranges.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).

        Returns:
            Optional[np.ndarray]: The combined binary mask (uint8) if successful, otherwise None.
        """
        logging.info("Attempting multi-scale thresholding.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image

        # Adaptive thresholding (mean)
        mask1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        # Otsu thresholding
        _, mask2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Combine them with a bitwise OR
        mask = cv2.bitwise_or(mask1, mask2)
        nonzero = np.count_nonzero(mask)
        logging.info(f"Multi-scale thresholding produced mask with {nonzero} nonzero pixels.")
        return mask if nonzero > 0 else None

    def _run_advanced_fallback_stack(self, image: np.ndarray, image_path: Optional[str] = None, inspection_dir: Optional[str] = None) -> np.ndarray:
        """
        Executes a sequence of advanced fallback masking techniques if SAM fails to produce
        a satisfactory mask. This stack is ordered by increasing computational complexity
        and diverse segmentation approaches.

        Args:
            image (np.ndarray): The input image (H, W, C or H, W).
            image_path (Optional[str]): Original path to the image, used for saving inspection outputs.
            inspection_dir (Optional[str]): Directory to save inspection images and masks.

        Returns:
            np.ndarray: The final combined binary mask (H, W).
        """
        logging.info("Running advanced professional fallback stack for mask generation.")

        # Ensure image is 3-channel for methods that expect it
        image_for_fallbacks = image
        if image.ndim == 2:
            image_for_fallbacks = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_for_fallbacks = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 1. Potrace fallback (contour filling)
        mask_potrace = self._potrace_fill(image_for_fallbacks)
        if mask_potrace is not None and mask_potrace.max() > 0:
            logging.info("Potrace fallback successful.")
            final_mask = mask_potrace
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_potrace")
            return final_mask

        # 2. Skeleton-graph closure (skeletonization)
        mask_skel = self._skeleton_graph_fill(image_for_fallbacks)
        if mask_skel is not None and mask_skel.max() > 0:
            logging.info("Skeleton-graph fallback successful.")
            final_mask = mask_skel
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_skeleton")
            return final_mask

        # 3. Deep segmentation (adaptive thresholding)
        mask_deep = self._deep_lineart_segmentation(image_for_fallbacks)
        if mask_deep is not None and mask_deep.max() > 0:
            logging.info("Deep lineart segmentation fallback successful.")
            final_mask = mask_deep
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_deep_seg")
            return final_mask

        # 4. Edge-aware watershed
        mask_ws = self._edge_watershed_fill(image_for_fallbacks)
        if mask_ws is not None and mask_ws.max() > 0:
            logging.info("Edge-aware watershed fallback successful.")
            final_mask = mask_ws
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_watershed")
            return final_mask

        # 5. HED/CRF-like edge fill (Canny + Dilation)
        mask_hed = self._hed_crf_fill(image_for_fallbacks)
        if mask_hed is not None and mask_hed.max() > 0:
            logging.info("HED/CRF-like fallback successful.")
            final_mask = mask_hed
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_hed_crf")
            return final_mask

        # 6. Learnable morph ops (closing and opening)
        mask_morph = self._learnable_morph_fill(image_for_fallbacks)
        if mask_morph is not None and mask_morph.max() > 0:
            logging.info("Learnable morph ops fallback successful.")
            final_mask = mask_morph
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_morph")
            return final_mask

        # 7. Classic fallback stack (Thicken -> Polygon -> MSER -> GrabCut)
        logging.info("Attempting classic multi-step fallback.")
        thick = self._pre_thicken(image_for_fallbacks)
        if thick is not None:
            poly = self._polygon_mask(thick)
            mser = self._mser_mask(thick)

            combined = None
            if poly is not None and mser is not None:
                combined = cv2.bitwise_or(poly, mser)
            elif poly is not None:
                combined = poly
            elif mser is not None:
                combined = mser

            if combined is not None and combined.max() > 0:
                grab = self._grabcut_refine(image_for_fallbacks, combined)
                if grab is not None and grab.max() > 0:
                    logging.info("Classic fallback stack with GrabCut successful.")
                    final_mask = grab
                    if inspection_dir and image_path:
                        self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_grabcut")
                    return final_mask

        # Final fallback: multi-scale thresholding
        final_fallback_mask = self._multi_scale_threshold(image_for_fallbacks)
        if final_fallback_mask is not None and final_fallback_mask.max() > 0:
            logging.info("Final multi-scale thresholding fallback successful.")
            final_mask = final_fallback_mask
            if inspection_dir and image_path:
                self._save_inspection_output(image_for_fallbacks, final_mask, image_path, inspection_dir, "_final_thresh")
            return final_mask

        logging.error("[MASK] All advanced fallbacks failed. Returning a blank mask as a last resort.")
        return np.zeros(image.shape[:2], dtype=np.uint8) # Return a blank mask if all else fails

    def _save_inspection_output(self, image: np.ndarray, mask: np.ndarray, original_path: str, save_dir: str, suffix: str):
        """
        Helper function to save inspection output (original image and generated mask).

        Args:
            image (np.ndarray): The original input image.
            mask (np.ndarray): The generated mask.
            original_path (str): The path to the original image.
            save_dir (str): Directory to save the inspection output.
            suffix (str): Suffix to append to the filename (e.g., '_potrace').
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            img_save_path = os.path.join(save_dir, f"{base_name}_input{suffix}.png")
            mask_save_path = os.path.join(save_dir, f"{base_name}_mask{suffix}.png")
            
            cv2.imwrite(img_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # Convert back to BGR for OpenCV
            cv2.imwrite(mask_save_path, mask)
            logging.info(f"Saved inspection image to {img_save_path} and mask to {mask_save_path}")
        except Exception as e:
            logging.error(f"Failed to save inspection output: {e}")


# --- SkeletonProcessor Class ---

class SkeletonProcessor:
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
        A branch point is a pixel with more than two neighbors in a 3x3 window.

        Args:
            skeleton (np.ndarray): A binary skeleton image (2D array, values 0 or 255).

        Returns:
            List[Tuple[int, int]]: A list of (row, column) coordinates of branch points.
        """
        branch_points = []
        # Define a 3x3 kernel for neighbor counting
        # For a skeleton pixel, sum of neighbors in 8-connectivity.
        # A branch point will have 3 or more neighbors.
        
        # Using convolution to count neighbors is more efficient than iterating pixels.
        # A pixel (r, c) is a branch point if skeleton[r,c] is 255 and its 8-connected neighbors
        # sum up to 3*255 or more (excluding itself).
        # We can use hit-or-miss transform or simply neighbor sum.
        
        # A simpler way to count neighbors for skeletonized images (thin, single-pixel lines)
        # is to sum the 8-connected neighbors.
        # A pixel (r,c) is a branch point if it's part of the skeleton AND has >= 3 neighbors.
        
        # Create a kernel for counting 8-connected neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1], # Center is 0 to exclude self from count
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the skeleton (normalized to 1s and 0s)
        convolved_neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Branch points are skeleton pixels with 3 or more neighbors
        # (convolved_neighbors >= 3) and are part of the skeleton (skeleton > 0)
        branch_point_coords = np.argwhere((skeleton > 0) & (convolved_neighbors >= 3))
        
        return [tuple(pt) for pt in branch_point_coords]


    def _prune_small_branches(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Removes small branches from the skeleton to simplify it.
        This helps in focusing on the main structure.

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
                path = self._trace_path(pruned_skeleton, (r, c), self.min_branch_length + 1) # Trace slightly beyond min_length

                # Check if the path is a small branch (ends without reaching a significant intersection)
                if path is not None and len(path) <= self.min_branch_length:
                    # Verify if the end of the path is truly an endpoint or a branch point
                    # For a true small branch, the path should ideally end without connecting to a major junction
                    
                    # Simple check: if the path is short and its last point isn't a known branch point
                    # (more robust would involve graph analysis)
                    is_small_branch = True
                    if len(path) > 1: # If path has more than just the start point
                        last_point_is_branch = False
                        # Check if the point before the last one has more than 2 neighbors in the original skeleton
                        # (this implies it was an internal point that became an endpoint after pruning)
                        if len(path) > 1:
                            prev_r, prev_c = path[-2]
                            temp_skel = pruned_skeleton.copy()
                            temp_skel[r,c] = 0 # Temporarily remove current endpoint to check neighbor count of previous
                            # Count neighbors of the point just before the current endpoint
                            kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
                            n_count_prev = cv2.filter2D((temp_skel > 0).astype(np.uint8), -1, kernel_neighbors, borderType=cv2.BORDER_CONSTANT)[prev_r, prev_c]
                            if n_count_prev >= 3: # If the point before was a branch point, don't prune
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
                                            or None if the path cannot be traced.
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
             return None # Did not move from start point
        
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
        
        # Generate skeleton
        # thin() can be used for more iterative thinning, skeletonize() is usually sufficient.
        skeleton = skeletonize(binary_mask)
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

# --- Pipeline Integration ---

# Wrapper class for backward compatibility
# Note: This class depends on 'SAMMaskGenerator' which is not defined in the provided snippets.
class PromptGenerator:
    """
    Wrapper class for backward compatibility with PromptGenerator.
    This class is intended to generate adaptive prompts for a Segment Anything Model (SAM).
    It acts as an interface to an underlying SAM mask generation utility.
    """
    def __init__(self):
        """
        Initializes the PromptGenerator.
        Requires SAMMaskGenerator to be available in the environment.
        """
        # Assuming SAMMaskGenerator exists and is importable/accessible
        # from wherever this class is intended to be used.
        # Placeholder for actual SAMMaskGenerator instantiation.
        # This will raise a NameError if SAMMaskGenerator is not defined.
        try:
            # from your_sam_module import SAMMaskGenerator # Example import
            self.sam_generator = SAMMaskGenerator() # This line would typically be uncommented
        except NameError:
            logging.error("SAMMaskGenerator is not defined. PromptGenerator cannot function without it.")
            self.sam_generator = None
        except Exception as e:
            logging.error(f"Error initializing SAMMaskGenerator: {e}")
            self.sam_generator = None

    def generate_prompts(self, image: np.ndarray) -> List[Dict]:
        """
        Generates prompts for SAM based on the input image.
        This method delegates the call to an internal SAM mask generator.

        Args:
            image (np.ndarray): The input image for which to generate prompts (H, W, C).

        Returns:
            List[Dict]: A list of prompt dictionaries suitable for SAM.
        """
        if self.sam_generator:
            return self.sam_generator.generate_adaptive_prompts(image)
        else:
            logging.warning("SAMMaskGenerator not initialized. Cannot generate prompts.")
            return []

# --- HybridAugmentationPipeline Class ---
# Note: Only a partial snippet of this class was provided.
# The full class definition and other methods are assumed to exist elsewhere.
class HybridAugmentationPipeline:
    """
    A placeholder for the HybridAugmentationPipeline class.
    Only a partial snippet of its internal logic was provided.
    This class likely combines various image processing and augmentation
    techniques, possibly including mask refinement and skeleton processing.
    """
    def __init__(self, min_branch_length: int = 10, **kwargs):
        # Placeholder for actual initialization logic
        self.min_branch_length = min_branch_length
        logging.info("HybridAugmentationPipeline initialized (partial definition).")

    def some_method_containing_snippet(self, original_mask: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
        """
        A placeholder method demonstrating a snippet of the HybridAugmentationPipeline's logic.
        This part seems to be involved in growing a skeleton and combining it with an original mask.

        Args:
            original_mask (np.ndarray): The initial binary mask.
            distance_map (np.ndarray): A distance transform map related to the skeleton.

        Returns:
            np.ndarray: A refined mask after combining with grown skeleton and morphological operations.
        """
        # The following lines were provided as a snippet.
        # They appear to be part of a method that refines a mask using skeleton information.
        grown_skeleton = (distance_map <= self.min_branch_length // 2).astype(np.uint8) * 255
        logging.debug(f"Grown skeleton by {self.min_branch_length // 2} pixels.")

        # Combine the grown skeleton with the original mask
        # This helps fill gaps while retaining the original mask's overall shape
        refined_mask = cv2.bitwise_or(original_mask, grown_skeleton)

        # Apply a small closing operation to ensure connectivity and fill minor holes
        kernel_size = max(3, self.min_branch_length // 4)
        closing_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
        logging.debug(f"Applied closing with kernel size {kernel_size}.")

        # Fill any remaining holes
        refined_mask = ndi.binary_fill_holes(refined_mask > 0).astype(np.uint8) * 255
        logging.debug("Filled holes in the refined mask.")

        return refined_mask




# ==================================================================
# Section: Hybrid Augmentation Pipeline Orchestrator (Example Usage)
# ==================================================================

class HybridAugmentationPipeline:
    """
    Orchestrates the mask generation and refinement process using
    SAM and skeleton-aware processing.
    """
    def __init__(self, config: Dict):
        """
        Initializes the HybridAugmentationPipeline with a config dict.
        """
        # Only use valid SAM model types
        valid_models = ['vit_h', 'vit_l', 'vit_b']
        model_type = config.get('sam', {}).get('model_type', 'vit_h')
        sam_models = [model_type] if model_type in valid_models else ['vit_h']
        self.sam_generator = SAMMaskGenerator(
            models=sam_models,
            cache_dir=config.get('sam', {}).get('checkpoint_path', '~/.cache/sam'),
            device=config.get('processing', {}).get('device', 'cuda'),
            ensemble_weights=None
        )
        # Only pass valid MaskRefiner args
        ref_cfg = config.get('refinement', {}) or {}
        valid_refiner_args = ['contour_approx_factor', 'min_component_size', 'closing_kernel_size', 'opening_kernel_size']
        filtered_ref_cfg = {k: v for k, v in ref_cfg.items() if k in valid_refiner_args}
        self.mask_refiner = MaskRefiner(**filtered_ref_cfg)
        self.skeleton_processor = SkeletonProcessor(**(config.get('skeleton', {}) or {}))
        self.config = config
        logging.info("Hybrid Augmentation Pipeline initialized.")
    def run_pipeline(self):
        """
        Loads images, processes them, and saves results using ImagePathDataset.
        """
        import pickle
        from .dataset import ImagePathDataset
        input_path = self.config['data']['input_path']
        output_path = self.config['data']['output_path']
        batch_size = self.config['processing']['batch_size']
        # Load derived_labels.json
        with open(input_path, 'r') as f:
            derived_labels = json.load(f)
        image_paths = [entry['image_path'] for entry in derived_labels]
        dataset = ImagePathDataset(image_paths, derived_labels_path=input_path)
        all_results = []
        inspection_dir = self.config.get('inspection_dir', None)
        for idx in range(0, len(dataset), batch_size):
            batch_indices = list(range(idx, min(idx+batch_size, len(dataset))))
            batch = [dataset[i] for i in batch_indices]
            images = [item[0] for item in batch if item[0] is not None]
            paths = [item[1] for item in batch if item[1] is not None]
            geometries = [item[2] for item in batch if item[2] is not None]
            logging.info(f"Processing batch {idx//batch_size+1}: indices {batch_indices}")
            logging.info(f"Batch image paths: {paths}")
            if not images:
                logging.warning(f"Batch {idx//batch_size+1} is empty after filtering. Raw batch: {batch}")
                continue
            batch_results = []
            for img, path, geom in zip(images, paths, geometries):
                logging.info(f"Processing image: {path}")
                # Convert tensor to numpy image
                np_img = img.squeeze().cpu().numpy()
                if np_img.max() <= 1.0:
                    np_img = (np_img * 255).astype(np.uint8)
                if np_img.ndim == 2:
                    np_img = np.stack([np_img]*3, axis=-1)
                mask = self.process_image(np_img, image_path=path, inspection_dir=inspection_dir)
                batch_results.append({'image_path': path, 'geometry': geom, 'mask': mask})
            all_results.append(batch_results)
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
        logging.info(f"Saved {len(all_results)} batches to {output_path}")

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Processes an input image to generate and refine a robust mask, with object-by-object segmentation,
        topology-aware prompting, hole-punching, and aggressive thinning for skeletons.
        """
        import matplotlib.pyplot as plt
        from skimage.morphology import thin, skeletonize
        logging.info("Starting object-by-object mask generation for the input image.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        # Step 1: Detect all objects (outer contours)
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            hierarchy = hierarchy[0]
        object_masks = []
        for i, cnt in enumerate(contours):
            # Only process outer contours (objects)
            if hierarchy is not None and hierarchy[i][3] != -1:
                continue
            mask_obj = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(mask_obj, [cnt], -1, 255, -1)
            # Generate topology-aware prompts for this object
            prompts = []
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                prompts.append({'point_coords': np.array([[cx, cy]]), 'point_labels': np.array([1])})
            # Add negative prompts for holes inside this object
            if hierarchy is not None:
                child_idx = hierarchy[i][2]
                while child_idx != -1:
                    hole_cnt = contours[child_idx]
                    M_hole = cv2.moments(hole_cnt)
                    if M_hole['m00'] > 0:
                        hx = int(M_hole['m10']/M_hole['m00'])
                        hy = int(M_hole['m01']/M_hole['m00'])
                        prompts.append({'point_coords': np.array([[hx, hy]]), 'point_labels': np.array([0])})
                    child_idx = hierarchy[child_idx][0]
            # Run SAM for this object
            obj_mask = None
            try:
                masks, scores = self.sam_generator._generate_masks_single_model(image, prompts, list(self.sam_generator.models.keys())[0])
                if masks:
                    obj_mask = masks[np.argmax(scores)]
                else:
                    obj_mask = mask_obj
                object_masks.append(obj_mask)
            except Exception as e:
                logging.error(f"Error in SAM mask generation for object {i}: {e}")
                obj_mask = mask_obj
                object_masks.append(obj_mask)
        # ...existing code...

    def _process_with_qa(self, img: np.ndarray) -> Dict:
        """Process image with quality assurance checks and visualization."""
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
                plt.imshow(result.get('mask', np.zeros_like(img[:,:,0])), cmap='gray')
                plt.title("Generated Mask")
                plt.show()
            except Exception as e:
                logging.warning(f"QA visualization failed: {e}")
        
        return result

# --- HybridAugmentationPipeline Class ---

class HybridAugmentationPipeline:
    """
    Main orchestrator for hybrid mask generation and augmentation pipeline.
    Combines SAM, skeleton processing, and mask refinement.
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.sam_generator = SAMMaskGenerator()
        self.mask_refiner = MaskRefiner()
        self.skeleton_processor = SkeletonProcessor()
        logging.info("HybridAugmentationPipeline initialized")

    def process_image(self, image: np.ndarray, image_path: str = None, inspection_dir: str = None) -> np.ndarray:
        """
        Process a single image to generate high-quality mask.
        
        Args:
            image (np.ndarray): Input image
            image_path (str): Path to original image for naming inspection outputs
            inspection_dir (str): Directory to save inspection images
            
        Returns:
            np.ndarray: Generated binary mask
        """
        try:
            # Generate mask using SAM
            mask = self.sam_generator.generate_ensemble_masks(image)
            
            # Refine mask
            refined_mask = self.mask_refiner.refine(mask)
            
            # Save inspection images if requested
            if inspection_dir is not None and image_path is not None:
                import os
                os.makedirs(inspection_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                img_save_path = os.path.join(inspection_dir, f"{base_name}_input.png")
                mask_save_path = os.path.join(inspection_dir, f"{base_name}_mask.png")
                
                if image.ndim == 3:
                    cv2.imwrite(img_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(img_save_path, image)
                cv2.imwrite(mask_save_path, refined_mask)
                logging.info(f"Saved inspection image to {img_save_path} and mask to {mask_save_path}")
            
            return refined_mask
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            # Fallback to simple thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask

    def process(self, image: np.ndarray) -> Dict:
        """
        Legacy method for backward compatibility.
        """
        mask = self.process_image(image)
        return {'mask': mask}

    def batch_process(self, image_paths: list, output_dir: str, batch_size: int = 8, 
                     num_workers: int = 4, inspection_dir: str = None) -> list:
        """
        Process multiple images in batches with multiprocessing, using explicit image paths.
        """
        import os
        from multiprocessing import Pool
        from tqdm import tqdm
        os.makedirs(output_dir, exist_ok=True)
        if inspection_dir:
            os.makedirs(inspection_dir, exist_ok=True)
        processed_files = []
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(image_paths), desc="Processing images") as pbar:
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i+batch_size]
                    batch_args = [(path, output_dir, inspection_dir) for path in batch_paths]
                    batch_results = pool.map(self._process_single_image, batch_args)
                    processed_files.extend(batch_results)
                    pbar.update(len(batch_paths))
        return processed_files

    def _process_single_image(self, args):
        """Helper method for batch processing."""
        image_path, output_dir, inspection_dir = args
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image: {image_path}")
                return None
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process image
            mask = self.process_image(image, image_path, inspection_dir)
            # Check if mask is blank (all zeros)
            if mask is not None and (mask == 0).all():
                logging.warning(f"Mask for {image_path} is blank (all zeros). Possible error in mask generation.")
            # Save result
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(output_path, mask)
            return output_path
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            return None

    def run_pipeline(self):
        """
        Main entry point for CLI. Loads input paths, runs batch processing, and saves results.
        """
        import json
        import pickle
        import os
        # Get config values
        input_path = self.config.get('data', {}).get('input_path')
        output_path = self.config.get('data', {}).get('output_path')
        batch_size = self.config.get('processing', {}).get('batch_size', 8)
        inspection_dir = self.config.get('inspection_dir', None)
        num_workers = self.config.get('processing', {}).get('num_workers', 4)

        if not input_path or not output_path:
            logging.critical("Input or output path not specified in config.")
            raise ValueError("Input or output path not specified.")

        # Load image paths from derived_labels.json
        with open(input_path, 'r') as f:
            derived_labels = json.load(f)
        def remap_path(path):
            return path.replace('category_1', '1').replace('category_0', '0')
        image_paths = [remap_path(entry['image_path']) for entry in derived_labels]

        # Output directory for masks
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Run batch processing using explicit image paths
        mask_files = self.batch_process(
            image_paths=image_paths,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            inspection_dir=inspection_dir
        )

        # Save results as a pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(mask_files, f)
        logging.info(f"Saved {len(mask_files)} mask files to {output_path}")

# Create alias for backward compatibility
HybridAugmentor = HybridAugmentationPipeline
