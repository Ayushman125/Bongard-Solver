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
        # Precompute distance threshold for skeleton grow
        self.grow_threshold = opening_kernel_size

    def _contour_refinement(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Refines a binary mask using contour approximation.
        Simplifies contours and fills resulting shapes.
        """
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask_bin, dtype=np.uint8)

        refined_mask = np.zeros_like(mask_bin, dtype=np.uint8)
        for contour in contours:
            epsilon = self.contour_approx_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(refined_mask, [approx], -1, 255, -1)  # Fill the approximated contour
        return refined_mask

    def _morphological_cleaning(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Applies morphological operations (closing, opening) to clean the mask.
        """
        if self.closing_kernel_size > 0:
            closing_kernel = np.ones((self.closing_kernel_size, self.closing_kernel_size), np.uint8)
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, closing_kernel)
        if self.opening_kernel_size > 0:
            opening_kernel = np.ones((self.opening_kernel_size, self.opening_kernel_size), np.uint8)
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, opening_kernel)
        # Remove small objects (noise)
        cleaned_mask = remove_small_objects(mask_bin.astype(bool), min_size=self.min_component_size).astype(np.uint8) * 255
        return cleaned_mask

    def _skeleton_grow(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Grows a skeletonized mask based on a distance transform,
        useful for very thin or sparse masks.
        """
        if mask_bin.max() == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)

        skeleton = skeletonize(mask_bin > 0)
        if not np.any(skeleton):
            return np.zeros_like(mask_bin, dtype=np.uint8)

        distance_map = distance_transform_edt(~skeleton)
        grown_mask = (distance_map <= self.grow_threshold).astype(np.uint8) * 255
        return grown_mask

    def refine(self, mask_bin: np.ndarray) -> np.ndarray:
        """
        Main refinement method. Applies contour-based refinement, morphological cleaning,
        and hole filling. Handles thin-line masks by growing their skeleton.
        """
        if not isinstance(mask_bin, np.ndarray) or mask_bin.max() == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)
        try:
            # Ensure mask is in the correct format (binary 0 or 255)
            if mask_bin.max() <= 1:
                mask_bin = (mask_bin * 255).astype(np.uint8)
            else:
                mask_bin = (mask_bin > 127).astype(np.uint8) * 255

            # Detect thin-line masks and grow skeleton if too sparse
            if np.sum((mask_bin > 127).astype(np.uint8)) < self.min_component_size * 2:
                logging.info("Detected thin-line mask, applying skeleton grow.")
                return self._skeleton_grow(mask_bin)

            # Stage 1: Contour-based Refinement
            refined_mask = self._contour_refinement(mask_bin)
            if refined_mask.max() == 0:
                logging.warning("Contour refinement resulted in an empty mask.")
                return refined_mask

            # Stage 2: Morphological Cleaning + hole-fill
            cleaned = self._morphological_cleaning(refined_mask)
            filled = ndi.binary_fill_holes(cleaned > 0)
            return (filled.astype(np.uint8) * 255)
        except Exception as e:
            logging.error(f"Mask refinement failed: {e}")
            return np.zeros_like(mask_bin, dtype=np.uint8)


# ==================================================================
# Section: SAM-based Mask Generation
# ==================================================================

class SAMMaskGenerator:
    """
    Generates masks using an ensemble of Segment Anything Models (SAM)
    with adaptive prompting and a robust fallback stack.
    """
    def __init__(self,
                 models: List[str] = ['vit_h', 'vit_b'],
                 cache_dir: str = "~/.cache/sam",
                 device: str = "cuda",
                 ensemble_weights: Optional[List[float]] = None):
        """
        Initializes the SAMMaskGenerator.

        Args:
            models (List[str]): List of SAM model types to use for the ensemble
                                 (e.g., 'vit_h', 'vit_l', 'vit_b').
            cache_dir (str): Directory to cache downloaded SAM checkpoints.
            device (str): Device to run SAM models on ('cuda' or 'cpu').
            ensemble_weights (Optional[List[float]]): Weights for each model in the ensemble.
                                                      If None, defaults to [0.6, 0.4] for 'vit_h', 'vit_b'.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Default ensemble weights (vit_h gets higher weight if present)
        self.ensemble_weights = ensemble_weights or [0.6, 0.4]

        self.models = {}      # Stores loaded SAM models
        self.predictors = {}  # Stores SamPredictor instances for each model
        self.is_initialized = False # Flag to check if SAM models were successfully loaded

        if SAM_AVAILABLE:
            try:
                self._initialize_ensemble(models)
            except Exception as e:
                logging.error(f"SAM Ensemble initialization failed: {e}")
                self.is_initialized = False
        else:
            logging.warning("SAM not available - ensemble and SAM-specific functionalities disabled.")

    def _initialize_ensemble(self, model_types: List[str]):
        """
        Initializes multiple SAM models for ensemble prediction.
        Downloads checkpoints if they don't exist locally.

        Args:
            model_types (List[str]): List of SAM model types to initialize.
        """
        checkpoint_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }

        for model_type in model_types:
            if model_type not in checkpoint_urls:
                logging.warning(f"Unknown SAM model type: {model_type}. Skipping.")
                continue
            try:
                # Download checkpoint if needed
                ckpt_path = self._get_checkpoint(model_type, checkpoint_urls[model_type])

                # Load model
                sam_model = sam_model_registry[model_type](checkpoint=ckpt_path)
                sam_model.to(self.device)

                self.models[model_type] = sam_model
                self.predictors[model_type] = SamPredictor(sam_model)
                logging.info(f"Loaded SAM {model_type} successfully on {self.device}.")

            except Exception as e:
                logging.error(f"Failed to load SAM {model_type}: {e}")

        if self.models:
            self.is_initialized = True
            logging.info(f"SAM Ensemble initialized with {len(self.models)} models.")
        else:
            logging.error("No SAM models could be initialized.")

    def _get_checkpoint(self, model_type: str, url: str) -> str:
        """
        Downloads a SAM checkpoint file if it does not already exist in the cache directory.

        Args:
            model_type (str): The type of SAM model (e.g., 'vit_h').
            url (str): The URL to download the checkpoint from.

        Returns:
            str: The local path to the downloaded checkpoint file.

        Raises:
            RuntimeError: If the download fails.
        """
        filename = f"sam_{model_type}_checkpoint.pth"
        ckpt_path = self.cache_dir / filename

        if not ckpt_path.exists():
            logging.info(f"Downloading SAM {model_type} checkpoint from {url}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

                with open(ckpt_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"Downloaded SAM checkpoint to {ckpt_path}.")
            except Exception as e:
                raise RuntimeError(f"Failed to download SAM checkpoint from {url}: {e}")

        return str(ckpt_path)

    def _generate_masks_single_model(self, image: np.ndarray, prompts: List[Dict], model_name: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generates masks using a single SAM model for a given set of prompts.

        Args:
            image (np.ndarray): The input image (H, W, 3) in RGB format.
            prompts (List[Dict]): A list of prompt dictionaries, each containing
                                   'point_coords' and 'point_labels'.
            model_name (str): The name of the SAM model to use.

        Returns:
            Tuple[List[np.ndarray], List[float]]: A tuple containing a list of generated
                                                   masks and their corresponding confidence scores.
        """
        masks = []
        scores = []
        predictor = self.predictors.get(model_name)
        if predictor is None:
            logging.error(f"Predictor for model '{model_name}' not found.")
            return [], []

        try:
            predictor.set_image(image)
        except Exception as e:
            logging.error(f"Failed to set image for SAM predictor ({model_name}): {e}")
            return [], []

        for prompt in prompts:
            try:
                mask, score, _ = predictor.predict(
                    point_coords=prompt['point_coords'],
                    point_labels=prompt['point_labels'],
                    multimask_output=True
                )
                masks.extend([m.astype(np.uint8) * 255 for m in mask])
                scores.extend(score.tolist())
            except Exception as e:
                logging.warning(f"SAM prediction failed for prompt using model {model_name}: {e}")
        return masks, scores

    def _auto_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generates a mask using SAM's automatic mask generator,
        suitable for thin-line art or general object detection.

        Args:
            image (np.ndarray): The input image (H, W, 3) in RGB format.

        Returns:
            Optional[np.ndarray]: The generated binary mask (H, W) or None if generation fails.
        """
        if SAM_AVAILABLE and self.models:
            try:
                # Use the first available model for automatic mask generation
                model_for_auto = self.models[list(self.models.keys())[0]]
                mask_gen = SamAutomaticMaskGenerator(model_for_auto)
                masks = mask_gen.generate(image)
                if masks:
                    # Return the largest mask by area
                    best_mask = max(masks, key=lambda m: m.get('area', 0))
                    logging.info(f"SAM auto mask generated with area: {best_mask.get('area', 0)}")
                    return best_mask['segmentation'].astype(np.uint8) * 255
            except Exception as e:
                logging.warning(f"SAM automatic mask generation failed: {e}")
        return None

    def _fallback_mask(self, image: np.ndarray) -> np.ndarray:
        """
        A simple fallback mask generation method using Otsu's thresholding
        if SAM or other advanced methods fail.

        Args:
            image (np.ndarray): The input image (H, W, C).

        Returns:
            np.ndarray: A binary mask (H, W).
        """
        logging.info("Using simple thresholding as fallback mask generation.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def generate_adaptive_prompts(self, image: np.ndarray) -> List[Dict]:
        """
        Generate diverse, content-aware prompts for robust mask generation.
        Includes edge-based, multi-scale grid, and center/corner prompts.

        Args:
            image (np.ndarray): The input image (H, W, C).

        Returns:
            List[Dict]: A list of prompt dictionaries.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape[:2]
        prompts = []

        # 1. Edge-based prompting
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))

        if len(edge_points) > 0:
            # Sample edge points strategically
            n_edge_samples = min(5, len(edge_points))
            edge_indices = np.random.choice(len(edge_points), n_edge_samples, replace=False)
            for idx in edge_indices:
                y, x = edge_points[idx]
                prompts.append({
                    'point_coords': np.array([[x, y]]),
                    'point_labels': np.array([1])
                })
        logging.debug(f"Generated {len(prompts)} edge-based prompts.")

        # 2. Multi-scale grid prompting
        for scale in [0.3, 0.5, 0.7]:
            grid_size = max(32, int(min(h, w) * scale))
            for i in range(grid_size // 2, h, grid_size):
                for j in range(grid_size // 2, w, grid_size):
                    if i < h and j < w:
                        prompts.append({
                            'point_coords': np.array([[j, i]]),
                            'point_labels': np.array([1])
                        })
        logging.debug(f"Generated {len(prompts)} total prompts after grid-based.")

        # 3. Center and corner prompts
        center_prompts = [
            {'point_coords': np.array([[w // 2, h // 2]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[w // 4, h // 4]]), 'point_labels': np.array([1])},
            {'point_coords': np.array([[3 * w // 4, 3 * h // 4]]), 'point_labels': np.array([1])},
        ]
        prompts.extend(center_prompts)
        logging.debug(f"Generated {len(prompts)} total prompts after center/corner.")

        return prompts[:15]  # Limit to prevent excessive computation

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
            logging.warning("SAM models not initialized, falling back to simple thresholding.")
            return self._fallback_mask(image)

        try:
            # Thin-line art detection: use SAM's automatic mask generator if few edges
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            if SAM_AVAILABLE and edges.sum() < 200: # Heuristic for thin-line art
                auto_mask = self._auto_mask(image)
                if auto_mask is not None and auto_mask.max() > 0:
                    logging.info("Detected thin-line art, successfully generated mask using SAM auto-mask.")
                    return auto_mask
                else:
                    logging.warning("SAM auto-mask failed for thin-line art, proceeding with prompts.")

            # Ensure image is in correct format (RGB)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Generate adaptive prompts
            prompts = self.generate_adaptive_prompts(image)

            if not prompts:
                logging.warning("No adaptive prompts generated, falling back to simple thresholding.")
                return self._fallback_mask(image)

            if not use_ensemble or len(self.models) == 1:
                # Use only the first available model if ensemble is disabled or only one model exists
                model_name = list(self.models.keys())[0]
                masks, scores = self._generate_masks_single_model(image, prompts, model_name)
                if masks:
                    # Select the mask with the highest score
                    best_mask_idx = np.argmax(scores)
                    logging.info(f"Generated mask using single model {model_name}.")
                    return masks[best_mask_idx]
                else:
                    logging.warning(f"Single model {model_name} produced no masks.")
                    return self._run_advanced_fallback_stack(image)

            # Ensemble prediction
            all_masks = []
            all_scores = []

            for model_name, weight in zip(self.models.keys(), self.ensemble_weights):
                masks, scores = self._generate_masks_single_model(image, prompts, model_name)
                if len(masks) > 0:
                    # Weight scores by model confidence
                    weighted_scores = [score * weight for score in scores]
                    all_masks.extend(masks)
                    all_scores.extend(weighted_scores)
                    logging.info(f"[MASK] Ensemble model {model_name} produced {len(masks)} mask(s).")
                else:
                    logging.info(f"[MASK] Ensemble model {model_name} produced no masks for current prompts.")

            if not all_masks:
                logging.warning("No masks generated by ensemble, using advanced professional fallback stack.")
                return self._run_advanced_fallback_stack(image)

            # Combine masks: For simplicity, we'll take the mask with the highest overall score.
            # A more advanced ensemble might involve voting or weighted averaging of masks.
            best_overall_mask_idx = np.argmax(all_scores)
            final_mask = all_masks[best_overall_mask_idx]
            logging.info(f"Ensemble successfully generated a mask with overall best score.")
            return final_mask

        except Exception as e:
            logging.error(f"Ensemble mask generation failed: {e}. Falling back to simple thresholding.")
            return self._fallback_mask(image)

    # --- Advanced fallback stack methods (placeholders) ---
    # These methods would contain more sophisticated image processing techniques
    # to generate a mask if SAM fails. They are currently placeholders.
    def _potrace_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement Potrace vectorization and fill for mask generation.
        Requires an external library like 'potrace' or a custom implementation.
        """
        logging.debug("Attempting Potrace fill (placeholder).")
        # Example: Convert image to binary, then to SVG path, then fill.
        # This would be a complex implementation involving external tools or libraries.
        return None

    def _skeleton_graph_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement skeleton graph closure for mask generation.
        Could use libraries like 'sknw' (skeleton network) to build a graph
        from the skeleton and close loops.
        """
        logging.debug("Attempting skeleton graph fill (placeholder).")
        # Example: skeletonize -> build graph -> find closed loops -> fill.
        return None

    def _deep_lineart_segmentation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement deep learning-based line art segmentation (e.g., U-Net/Pix2Pix).
        This would involve loading and running a pre-trained deep learning model.
        """
        logging.debug("Attempting deep lineart segmentation (placeholder).")
        # Example: Load a PyTorch/TensorFlow model, preprocess image, run inference.
        return None

    def _edge_watershed_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement edge-aware watershed segmentation.
        Uses image gradients/edges to define 'basins' for segmentation.
        """
        logging.debug("Attempting edge-watershed fill (placeholder).")
        # Example: Canny edges -> distance transform -> local maxima -> watershed.
        return None

    def _hed_crf_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement HED (Holistically-Nested Edge Detection) + CRF (Conditional Random Field) fill.
        HED extracts rich edges, CRF refines segmentation based on these edges.
        """
        logging.debug("Attempting HED/CRF fill (placeholder).")
        # Example: Run HED model -> apply CRF.
        return None

    def _learnable_morph_fill(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement learnable morphological operations (e.g., MorphNet).
        This would involve a neural network trained to apply optimal morphological transformations.
        """
        logging.debug("Attempting learnable morph fill (placeholder).")
        # Example: Load a MorphNet model, apply operations.
        return None

    def _pre_thicken(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement a pre-thicken operation for very thin lines.
        Can help make features more robust for subsequent classical image processing.
        """
        logging.debug("Attempting pre-thicken (placeholder).")
        # Example: Apply a small dilation or custom thickening filter.
        return None

    def _polygon_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement polygon mask extraction from image features.
        Could involve contour detection and simplification to polygons.
        """
        logging.debug("Attempting polygon mask extraction (placeholder).")
        # Example: Find contours, approximate polygons, fill.
        return None

    def _mser_mask(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement MSER (Maximally Stable Extremal Regions) mask extraction.
        Useful for detecting text or distinct blobs in an image.
        """
        logging.debug("Attempting MSER mask extraction (placeholder).")
        # Example: cv2.MSER.detect() -> convert regions to mask.
        return None

    def _grabcut_refine(self, image: np.ndarray, initial_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement GrabCut refinement.
        Requires an initial mask or bounding box, then iteratively refines it.
        """
        logging.debug("Attempting GrabCut refinement (placeholder).")
        # Example: cv2.grabCut() with initial mask.
        return None

    def _multi_scale_threshold(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Placeholder: Implement multi-scale thresholding.
        Applies thresholding at different scales/resolutions and combines results.
        """
        logging.debug("Attempting multi-scale thresholding (placeholder).")
        # Example: Apply adaptive thresholding or Otsu at different image sizes.
        return None

    def _run_advanced_fallback_stack(self, image: np.ndarray) -> np.ndarray:
        """
        Executes a sequence of advanced fallback masking techniques if SAM fails.
        This stack is ordered by increasing computational complexity/reliance on external models.
        """
        logging.info("Running advanced professional fallback stack for mask generation.")
        
        # 1. Vector-trace + fill (Potrace)
        mask_potrace = self._potrace_fill(image)
        if mask_potrace is not None and mask_potrace.max() > 0:
            logging.info("[MASK] Potrace fallback produced a non-empty mask.")
            return mask_potrace
        else:
            logging.debug("[MASK] Potrace fallback failed or produced empty mask.")

        # 2. Skeleton-graph closure (sknw)
        mask_skel = self._skeleton_graph_fill(image)
        if mask_skel is not None and mask_skel.max() > 0:
            logging.info("[MASK] Skeleton-graph fallback produced a non-empty mask.")
            return mask_skel
        else:
            logging.debug("[MASK] Skeleton-graph fallback failed or produced empty mask.")

        # 3. Deep segmentation (U-Net/Pix2Pix)
        mask_deep = self._deep_lineart_segmentation(image)
        if mask_deep is not None and mask_deep.max() > 0:
            logging.info("[MASK] Deep segmentation fallback produced a non-empty mask.")
            return mask_deep
        else:
            logging.debug("[MASK] Deep segmentation fallback failed or produced empty mask.")

        # 4. Edge-aware watershed
        mask_ws = self._edge_watershed_fill(image)
        if mask_ws is not None and mask_ws.max() > 0:
            logging.info("[MASK] Edge-watershed fallback produced a non-empty mask.")
            return mask_ws
        else:
            logging.debug("[MASK] Edge-watershed fallback failed or produced empty mask.")

        # 5. HED/SE edge detector + CRF
        mask_hed = self._hed_crf_fill(image)
        if mask_hed is not None and mask_hed.max() > 0:
            logging.info("[MASK] HED/CRF fallback produced a non-empty mask.")
            return mask_hed
        else:
            logging.debug("[MASK] HED/CRF fallback failed or produced empty mask.")

        # 6. Learnable morph ops (MorphNet)
        mask_morph = self._learnable_morph_fill(image)
        if mask_morph is not None and mask_morph.max() > 0:
            logging.info("[MASK] Learnable morph fallback produced a non-empty mask.")
            return mask_morph
        else:
            logging.debug("[MASK] Learnable morph fallback failed or produced empty mask.")

        # 7. Classic fallback stack (existing, more traditional CV methods)
        logging.info("[MASK] Attempting classic computer vision fallback stack.")
        thick = self._pre_thicken(image)
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
            grab = self._grabcut_refine(image, combined)
            if grab is not None and grab.max() > 0:
                logging.info("[MASK] Classic grabcut fallback produced a non-empty mask.")
                return grab
            else:
                logging.debug("[MASK] Classic grabcut fallback failed or produced empty mask.")
        else:
            logging.debug("[MASK] Polygon/MSER combination failed or produced empty mask.")

        # Final fallback: simple multi-scale thresholding
        final_fallback_mask = self._multi_scale_threshold(image)
        if final_fallback_mask is not None and final_fallback_mask.max() > 0:
            logging.info("[MASK] Multi-scale thresholding fallback produced a non-empty mask.")
            return final_fallback_mask
        else:
            logging.error("[MASK] All advanced fallbacks failed. Returning a blank mask or simple threshold.")
            return self._fallback_mask(image) # Fallback to the most basic method
# ==================================================================
# Section: Skeleton-Aware Processing
# ==================================================================

class SkeletonProcessor:
    """
    A class for processing skeletons extracted from binary masks,
    including branch point detection and skeleton-aware refinement.
    """
    def __init__(self, min_branch_length: int = 10):
        self.min_branch_length = min_branch_length

    def _get_branch_points(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detects branch points in a skeletonized image.
        A branch point is a pixel with more than two neighbors in a 3x3 window.
        """
        branch_points = []
        # Define a 3x3 kernel for neighbor counting
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1], # Center pixel weighted to easily identify it
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the skeleton with the kernel
        # The value at each pixel will be the sum of its neighbors + 10 if it's a skeleton pixel
        convolved = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Iterate through the convolved image to find branch points
        # A pixel is a branch point if it's part of the skeleton (value >= 10)
        # and has more than 2 neighbors (sum of neighbors > 2)
        # For a skeleton pixel (value 10), if convolved value is > 12, it has >2 neighbors.
        # (10 for itself + 3 for 3 neighbors = 13)
        # (10 for itself + 4 for 4 neighbors = 14)
        # (10 for itself + 5 for 5 neighbors = 15) etc.
        # So, if convolved[i,j] - 10 > 2, it's a branch point.
        for r, c in np.argwhere(skeleton):
            # Extract 3x3 neighborhood
            neighborhood = skeleton[max(0, r-1):min(skeleton.shape[0], r+2),
                                    max(0, c-1):min(skeleton.shape[1], c+2)]
            # Count active neighbors (excluding the center pixel itself)
            num_neighbors = np.sum(neighborhood) - neighborhood[min(1, r):min(2, r+1), min(1, c):min(2, c+1)] # Adjust for boundary
            if num_neighbors > 2:
                branch_points.append((r, c))
        return branch_points


    def _prune_small_branches(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Removes small branches from the skeleton to simplify it.
        This helps in focusing on the main structure.
        """
        # Iterate and remove end points until no more small branches can be pruned
        # This is a simplified iterative pruning; more robust methods exist (e.g., using graph theory)
        pruned_skeleton = skeleton.copy()
        while True:
            endpoints = np.array(np.where(self._get_endpoints(pruned_skeleton))).T
            if len(endpoints) == 0:
                break

            changes_made = False
            for r, c in endpoints:
                # Find path from endpoint to nearest branch point or another endpoint
                # This is a simplified approach; a full graph traversal would be more robust
                path = self._trace_path(pruned_skeleton, (r, c), self.min_branch_length)
                if path is not None and len(path) < self.min_branch_length:
                    for pr, pc in path:
                        pruned_skeleton[pr, pc] = 0 # Remove small branch
                    changes_made = True
            if not changes_made:
                break
        return pruned_skeleton

    def _get_endpoints(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Detects endpoints in a skeletonized image.
        An endpoint is a pixel with exactly one neighbor in a 3x3 window.
        """
        # Define a 3x3 kernel for neighbor counting (excluding center)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the skeleton with the kernel
        convolved = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Endpoints are skeleton pixels with exactly one neighbor
        endpoints = (skeleton > 0) & (convolved == 1)
        return endpoints

    def _trace_path(self, skeleton: np.ndarray, start_point: Tuple[int, int], max_length: int) -> Optional[List[Tuple[int, int]]]:
        """
        Traces a path from a start_point along the skeleton up to max_length or until a branch point/intersection.
        Returns the path as a list of coordinates.
        """
        path = [start_point]
        current_point = start_point
        visited = {start_point} # Keep track of visited points to avoid loops

        for _ in range(max_length): # Limit path length to avoid infinite loops on complex skeletons
            neighbors = []
            r, c = current_point
            # Check 8-connectivity neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and skeleton[nr, nc] > 0:
                        if (nr, nc) not in visited:
                            neighbors.append((nr, nc))

            if len(neighbors) == 1: # Continue along a single path
                current_point = neighbors[0]
                path.append(current_point)
                visited.add(current_point)
            elif len(neighbors) == 0: # End of a branch
                break
            else: # Branch point or intersection
                break
        return path

    def process_skeleton(self, mask_bin: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Performs skeletonization, prunes small branches, and detects branch points.

        Args:
            mask_bin (np.ndarray): Input binary mask (0 or 255).

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: A tuple containing the
                                                       pruned skeleton and a list of branch points.
        """
        if mask_bin.max() == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8), []

        # Ensure mask is boolean for skeletonize
        binary_mask = mask_bin > 0
        skeleton = skeletonize(binary_mask)

        # Prune small branches
        pruned_skeleton = self._prune_small_branches(skeleton)

        # Detect branch points on the pruned skeleton
        branch_points = self._get_branch_points(pruned_skeleton)

        return (pruned_skeleton.astype(np.uint8) * 255), branch_points

    def skeleton_aware_refinement(self, original_mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        """
        Refines the original mask by ensuring connectivity and shape preservation
        based on the skeleton. This can involve growing the skeleton or
        using it as a guide for morphological operations.

        Args:
            original_mask (np.ndarray): The initial binary mask.
            skeleton (np.ndarray): The processed skeleton of the mask.

        Returns:
            np.ndarray: The skeleton-aware refined mask.
        """
        if skeleton.max() == 0:
            logging.warning("Empty skeleton provided for skeleton-aware refinement. Returning original mask.")
            return original_mask

        # Grow the skeleton to create a thicker, connected structure
        # Use distance transform from the skeleton
        distance_map = distance_transform_edt(~skeleton)
        # Threshold the distance map to grow the skeleton
        # The threshold can be a parameter or dynamically determined
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
    def __init__(self, sam_models: List[str] = ['vit_h', 'vit_b'],
                 sam_cache_dir: str = "~/.cache/sam",
                 sam_device: str = "cuda",
                 sam_ensemble_weights: Optional[List[float]] = None,
                 mask_refiner_params: Optional[Dict] = None,
                 skeleton_processor_params: Optional[Dict] = None):
        """
        Initializes the HybridAugmentationPipeline with configurable components.
        """
        self.sam_generator = SAMMaskGenerator(
            models=sam_models,
            cache_dir=sam_cache_dir,
            device=sam_device,
            ensemble_weights=sam_ensemble_weights
        )
        self.mask_refiner = MaskRefiner(**(mask_refiner_params or {}))
        self.skeleton_processor = SkeletonProcessor(**(skeleton_processor_params or {}))
        logging.info("Hybrid Augmentation Pipeline initialized.")

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Processes an input image to generate and refine a robust mask.

        Args:
            image (np.ndarray): The input image (H, W, C).

        Returns:
            np.ndarray: The final refined binary mask.
        """
        logging.info("Starting mask generation for the input image.")
        # Step 1: Generate initial mask using SAM ensemble
        initial_mask = self.sam_generator.generate_ensemble_masks(image)
        logging.info(f"Initial mask generated. Max pixel value: {initial_mask.max()}")

        if initial_mask.max() == 0:
            logging.warning("Initial SAM mask is empty. Skipping further refinement steps.")
            return initial_mask

        # Step 2: Refine the mask using general mask refinement techniques
        refined_mask_stage1 = self.mask_refiner.refine(initial_mask)
        logging.info(f"Mask refined (Stage 1). Max pixel value: {refined_mask_stage1.max()}")

        if refined_mask_stage1.max() == 0:
            logging.warning("Mask refinement (Stage 1) resulted in an empty mask. Returning empty mask.")
            return refined_mask_stage1

        # Step 3: Process skeleton and apply skeleton-aware refinement
        # Ensure the mask is binary (0 or 255) before skeletonization
        skeleton, branch_points = self.skeleton_processor.process_skeleton(refined_mask_stage1)
        logging.info(f"Skeleton processed. Found {len(branch_points)} branch points.")

        final_mask = self.skeleton_processor.skeleton_aware_refinement(refined_mask_stage1, skeleton)
        logging.info(f"Final mask generated (Skeleton-aware refinement). Max pixel value: {final_mask.max()}")

        return final_mask


# Alias for CLI compatibility
HybridAugmentor = HybridAugmentationPipeline
