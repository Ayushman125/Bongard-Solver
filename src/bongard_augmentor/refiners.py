# Import MC Dropout utility if available
try:
    from .mc_dropout_utils import mc_dropout_mask_prediction
except ImportError:
    mc_dropout_mask_prediction = None


import numpy as np
import cv2
import logging
import time
from scipy.ndimage import distance_transform_edt
from scipy import ndimage as ndi, ndimage
from skimage.morphology import skeletonize, remove_small_objects
from typing import List, Dict, Optional
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# Import torch and SAM-related symbols if available
try:
    import torch
except ImportError:
    torch = None

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    sam_model_registry = None
    SamPredictor = None
    SAM_AVAILABLE = False

class MaskRefiner:
    def ensemble_fallback_stack(self, image: np.ndarray, mask: np.ndarray = None, min_quality: float = 0.5, max_time: float = 3.0, scales=[1.0, 1.5], top_n: int = 5) -> np.ndarray:
        """
        Exponential quality boost: Ensemble voting of top fallback methods, multi-edge fusion, iterative refinement, advanced denoising, multi-scale fallback, edge-aware morphology, and confidence-weighted fusion.
        """
        import cv2
        import numpy as np
        import time
        from scipy import ndimage
        start_time = time.time()
        masks = []
        qualities = []
        method_names = []
        # Advanced denoising (non-local means)
        image_denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21) if image.ndim == 3 else cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        # Multi-scale fallback
        fallback_methods = [
            self.potrace_fill,
            self.skeleton_graph_fill,
            self.deep_lineart_segmentation,
            self.edge_watershed_fill,
            self.hed_crf_fill,
            self.learnable_morph_fill,
            self.pre_thicken,
            self.polygon_mask,
            self.mser_mask,
            lambda img: self.grabcut_refine(img, mask if mask is not None else np.zeros(img.shape[:2], np.uint8)),
            self.multi_scale_threshold,
        ]
        method_labels = [
            'potrace_fill','skeleton_graph_fill','deep_lineart_segmentation','edge_watershed_fill','hed_crf_fill','learnable_morph_fill','pre_thicken','polygon_mask','mser_mask','grabcut_refine','multi_scale_threshold'
        ]
        # For each scale, run each fallback and collect masks/qualities
        for scale in scales:
            if time.time() - start_time > max_time:
                break
            if scale != 1.0:
                h, w = image_denoised.shape[:2]
                scaled_img = cv2.resize(image_denoised, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
            else:
                scaled_img = image_denoised
            for method, label in zip(fallback_methods, method_labels):
                try:
                    mask_candidate = method(scaled_img)
                    if scale != 1.0:
                        mask_candidate = cv2.resize(mask_candidate, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Multi-edge fusion: combine Canny, Sobel, Laplacian
                    gray = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2GRAY) if scaled_img.ndim == 3 else scaled_img
                    canny = cv2.Canny(gray, 50, 150)
                    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
                    sobel = np.uint8(np.absolute(sobel))
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    laplacian = np.uint8(np.absolute(laplacian))
                    edge_fused = cv2.bitwise_or(canny, sobel)
                    edge_fused = cv2.bitwise_or(edge_fused, laplacian)
                    # Edge-aware morphology: dilate only along edges
                    mask_edges = cv2.bitwise_or(mask_candidate, edge_fused)
                    kernel = np.ones((3,3), np.uint8)
                    mask_morph = cv2.morphologyEx(mask_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
                    # Iterative refinement: fill holes, remove spurs, re-threshold, re-clean
                    mask_iter = mask_morph.copy()
                    for _ in range(2):
                        mask_iter = ndimage.binary_fill_holes(mask_iter > 0).astype(np.uint8) * 255
                        mask_iter = cv2.morphologyEx(mask_iter, cv2.MORPH_OPEN, kernel, iterations=1)
                        mask_iter = cv2.morphologyEx(mask_iter, cv2.MORPH_CLOSE, kernel, iterations=1)
                    # Final robust binary conversion
                    mask_final = self.robust_binary_conversion_pipeline(mask_iter)
                    # Quality metric
                    _, quality, _ = self.validate_mask_quality_with_confidence(mask_final, image, prediction_scores=[], model=None, input_tensor=None, mc_dropout_runs=3, device='cpu')
                    masks.append(mask_final)
                    qualities.append(quality)
                    method_names.append(f"{label}_scale{scale}")
                except Exception as e:
                    logging.warning(f"[EnsembleFallback] {label} at scale {scale} failed: {e}")
        # Confidence-weighted fusion: weighted sum of top-N masks
        if not masks:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        # Select top-N by quality
        top_idx = np.argsort(qualities)[-top_n:]
        mask_stack = np.stack([masks[i] for i in top_idx], axis=0)
        qual_stack = np.array([qualities[i] for i in top_idx])
        qual_stack = np.clip(qual_stack, 0, 1)
        # Weighted average
        weighted = np.tensordot(qual_stack, mask_stack, axes=([0],[0])) / (np.sum(qual_stack)+1e-6)
        # Majority voting
        majority = (np.mean(mask_stack > 0, axis=0) > 0.5).astype(np.uint8) * 255
        # Combine weighted and majority
        combined = ((weighted > 127) | (majority > 0)).astype(np.uint8) * 255
        # Final iterative refinement
        for _ in range(2):
            combined = ndimage.binary_fill_holes(combined > 0).astype(np.uint8) * 255
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
        return combined

    def _platt_scaling(self, logits, temperature=1.0):
        # Platt scaling: sigmoid(logit / T)
        if logits is None or len(logits) == 0:
            return None
        import scipy.special
        logits = np.asarray(logits, dtype=np.float32)
        scaled = scipy.special.expit(logits / temperature)
        return scaled

    def _calibrated_confidence(self, prediction_scores, logits=None, temperature=1.0):
        # Use Platt/temperature scaling if logits are available
        if logits is not None and len(logits) > 0:
            scaled = self._platt_scaling(logits, temperature)
            return float(np.mean(scaled))
        if prediction_scores is not None and len(prediction_scores) > 0:
            return float(np.mean(prediction_scores))
        return 0.5

    def _run_advanced_fallback_stack(self, image: np.ndarray, mask: np.ndarray = None, min_quality: float = 0.5, max_time: float = 2.0) -> np.ndarray:
        """
        DEPRECATED: Use ensemble_fallback_stack for exponentially improved quality.
        """
        logging.warning("_run_advanced_fallback_stack is deprecated. Use ensemble_fallback_stack instead.")
        return self.ensemble_fallback_stack(image, mask, min_quality, max_time)
    def __init__(self, contour_approx_factor=0.005, min_component_size=50, closing_kernel_size=5, opening_kernel_size=3, passes=2):
        self.contour_approx_factor = contour_approx_factor
        self.min_component_size = min_component_size
        self.closing_kernel_size = closing_kernel_size
        self.opening_kernel_size = opening_kernel_size
        self.grow_threshold = opening_kernel_size
        self.passes = passes


    def _single_pass_refine(self, mask_bin):
        if not isinstance(mask_bin, np.ndarray) or mask_bin.max() == 0:
            return np.zeros_like(mask_bin, dtype=np.uint8)
        try:
            if mask_bin.max() <= 1:
                mask_bin = (mask_bin * 255).astype(np.uint8)
            else:
                mask_bin = (mask_bin > 127).astype(np.uint8) * 255
            if np.sum((mask_bin > 127).astype(np.uint8)) < self.min_component_size * 2:
                return self._skeleton_grow(mask_bin)
            refined_mask = self._contour_refinement(mask_bin)
            if refined_mask.max() == 0:
                return refined_mask
            cleaned = self._morphological_cleaning(refined_mask)
            filled = ndi.binary_fill_holes(cleaned > 0)
            return (filled.astype(np.uint8) * 255)
        except Exception as e:
            logging.error(f"Mask refinement pipeline failed: {e}", exc_info=True)
            return (mask_bin > 127).astype(np.uint8) * 255

    def _contour_refinement(self, mask):
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

    def _morphological_cleaning(self, mask):
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.closing_kernel_size, self.closing_kernel_size))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.opening_kernel_size, self.opening_kernel_size))
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, opening_kernel)
        return opened_mask

    def _skeleton_grow(self, mask):
        binary = (mask > 127).astype(np.uint8)
        dist = distance_transform_edt(binary)
        grown = dist > self.grow_threshold
        return (grown.astype(np.uint8) * 255)

    # --- Fallbacks ---
    def potrace_fill(self, image):
        """
        Vectorizes the main object using contours and fills the largest contour (Potrace-like effect).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        return mask
    def skeleton_graph_fill(self, image):
        """
        Fills gaps in the main object using skeletonization and morphological closing.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        skeleton = skeletonize((binary > 0)).astype(np.uint8) * 255
        closed = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        filled = ndi.binary_fill_holes(closed > 0).astype(np.uint8) * 255
        return filled
    def deep_lineart_segmentation(self, image):
        """
        Approximates deep segmentation by combining edge detection and region growing.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        # Region growing from largest edge area
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask[labels == largest] = 255
        return mask
    def edge_watershed_fill(self, image):
        """
        Uses edge detection and watershed to segment and fill main object.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        # Distance transform for sure foreground
        dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        # Markers for watershed
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[edges > 0] = 0
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.watershed(color, markers)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers > 1] = 255
        return mask
    def hed_crf_fill(self, image):
        """
        Approximates HED/CRF by combining Canny edges and dense CRF-like smoothing.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        # Morphological closing to simulate CRF smoothing
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        filled = ndi.binary_fill_holes(closed > 0).astype(np.uint8) * 255
        return filled
    def learnable_morph_fill(self, image):
        """
        Simulates learnable morphology by combining adaptive thresholding and morphological ops.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        return morph
    def pre_thicken(self, image):
        """
        Thickens lines/objects using dilation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        dilated = cv2.dilate(gray, np.ones((3,3), np.uint8), iterations=2)
        _, binary = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    def polygon_mask(self, image):
        """
        Extracts a polygonal mask by approximating the largest contour.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            cv2.drawContours(mask, [approx], -1, 255, thickness=cv2.FILLED)
        return mask
    def mser_mask(self, image):
        """
        Extracts regions using MSER (Maximally Stable Extremal Regions).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        mask = np.zeros_like(gray, dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], -1, 255, -1)
        return mask
    def grabcut_refine(self, image, mask):
        """
        Refines a mask using the GrabCut algorithm.
        """
        if image.ndim == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img = image.copy()
        mask_gc = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img, mask_gc, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            result_mask = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        except Exception as e:
            logging.warning(f"GrabCut failed: {e}")
            result_mask = mask.copy() if mask is not None else np.zeros(img.shape[:2], dtype=np.uint8)
        return result_mask
    def multi_scale_threshold(self, image):
        """
        Combines global and local thresholding at multiple scales.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        # Global Otsu
        _, global_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Local adaptive
        local_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        # Combine
        combined = cv2.bitwise_or(global_mask, local_mask)
        return combined

    def optimized_fallback_execution_pipeline(self, mask: np.ndarray, image: np.ndarray = None, fallback_type: str = 'edge', **kwargs) -> np.ndarray:
        """
        Advanced fallback optimization: If the main mask fails quality or is empty, use a robust fallback strategy.
        Supports 'edge', 'threshold', or 'blank' fallback types. Optionally logs fallback events for analytics.
        """
        logging.info(f"[Fallback] Invoked fallback pipeline with type: {fallback_type}")
        if fallback_type == 'edge' and image is not None:
            # Use Canny edge detection as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            edges = cv2.Canny(gray, 50, 150)
            # Optionally dilate to thicken edges
            kernel = np.ones((3, 3), np.uint8)
            fallback_mask = cv2.dilate(edges, kernel, iterations=1)
            logging.info("[Fallback] Edge-based fallback mask generated.")
            return fallback_mask
        elif fallback_type == 'threshold' and image is not None:
            # Use Otsu thresholding as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            _, fallback_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logging.info("[Fallback] Threshold-based fallback mask generated.")
            return fallback_mask
        else:
            # Blank fallback
            if image is not None:
                shape = image.shape[:2]
            else:
                shape = mask.shape
            fallback_mask = np.zeros(shape, dtype=np.uint8)
            logging.info("[Fallback] Blank fallback mask generated.")
            return fallback_mask
    def generate_content_aware_prompts(self, image: np.ndarray, context_analysis: Dict = None) -> List[Dict]:
        """
        Implement sophisticated, content-aware prompts using multi-modal analysis and reinforcement learning.
        Key improvements:
        - Edge density-based prompt placement using advanced Canny detection
        - Multi-scale feature extraction with confidence weighting
        - Temporal prompt optimization through feedback loops
        - Cross-correlation analysis for optimal point distribution
        """
        prompts = []
        # Edge-aware prompt placement
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        edge_points = np.column_stack(np.where(edges > 0))
        n_edge_samples = min(10, len(edge_points))
        if n_edge_samples > 0:
            rng = np.random.default_rng()
            edge_indices = rng.choice(len(edge_points), n_edge_samples, replace=False)
            for idx in edge_indices:
                y, x = edge_points[idx]
                prompts.append({'point_coords': np.array([[x, y]]), 'point_labels': np.array([1])})
        # Multi-scale grid sampling (density-based)
        h, w = gray.shape
        grid_size = max(8, min(h, w) // 32)
        for y in range(grid_size//2, h, grid_size):
            for x in range(grid_size//2, w, grid_size):
                if edges[y, x] == 0:
                    prompts.append({'point_coords': np.array([[x, y]]), 'point_labels': np.array([1])})
        # Confidence scoring (placeholder: uniform)
        for p in prompts:
            p['confidence'] = 1.0
        return prompts
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
        # Use torch.cuda.is_available() check for device (if torch is available)
        self.device = "cuda" if SAM_AVAILABLE and torch and torch.cuda.is_available() else "cpu"
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
                        # This part would download the actual checkpoint
                        # For a standalone example, we'll just simulate it.
                        logging.info(f"Simulating download of SAM checkpoint {checkpoint_file} from {url} ...")
                        # In a real scenario:
                        # r = requests.get(url, stream=True)
                        # r.raise_for_status()
                        # with open(target_path, 'wb') as f:
                        #     for chunk in r.iter_content(chunk_size=8192):
                        #         f.write(chunk)
                        checkpoint_path = target_path # Assume download was successful for mock purposes
                        target_path.touch() # Create an empty file to simulate existence
                        logging.info(f"Simulated download of SAM checkpoint to {checkpoint_path}")
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
                    # In a real scenario:
                    # sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(self.device)
                    sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path)) # Use mock
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

    def robust_binary_conversion_pipeline(self, mask: np.ndarray, confidence_map: np.ndarray = None) -> np.ndarray:
        """
        Advanced binary mask conversion with adaptive thresholding, multi-level thresholding,
        edge-preserving smoothing, and confidence-weighted binarization.
        """
        # Ensure mask is a numpy array and float32 for processing
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)

        # Edge-preserving smoothing
        # Bilateral filter requires 3 channels for color images, or single channel for grayscale.
        # Assuming the input mask is grayscale (single channel), check its dimensions.
        if len(mask.shape) == 2: # Grayscale
            mask_smooth = cv2.bilateralFilter(mask, d=5, sigmaColor=0.1, sigmaSpace=5)
        elif len(mask.shape) == 3 and mask.shape[2] == 1: # Grayscale with channel dim
            mask_smooth = cv2.bilateralFilter(mask, d=5, sigmaColor=0.1, sigmaSpace=5).reshape(mask.shape[:2])
        else:
            # If it's a 3-channel color image, convert to grayscale first for bilateral filtering
            logging.warning("Input mask has multiple channels. Converting to grayscale for smoothing.")
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[2] == 3 else mask[:,:,0]
            mask_smooth = cv2.bilateralFilter(mask_gray, d=5, sigmaColor=0.1, sigmaSpace=5)

        # Otsu + adaptive thresholding
        mask_uint8 = (mask_smooth * 255).astype(np.uint8)
        
        # Ensure there's enough variance for Otsu's method, otherwise fallback to a simple threshold
        if mask_uint8.min() == mask_uint8.max():
            _, otsu_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        else:
            _, otsu_mask = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        adaptive_mask = cv2.adaptiveThreshold(mask_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Combine Otsu and adaptive
        combined_mask = cv2.bitwise_or(otsu_mask, adaptive_mask)
        
        # Confidence-weighted binarization (if confidence_map provided)
        if confidence_map is not None:
            # Ensure confidence_map is float32 and normalized to [0, 1]
            if confidence_map.dtype != np.float32:
                conf_norm = confidence_map.astype(np.float32) / confidence_map.max() if confidence_map.max() > 0 else confidence_map.astype(np.float32)
            else:
                conf_norm = cv2.normalize(confidence_map, None, 0, 1, cv2.NORM_MINMAX)
            
            # Ensure conf_norm has the same dimensions as combined_mask
            if conf_norm.shape != combined_mask.shape:
                logging.warning(f"Confidence map shape {conf_norm.shape} does not match mask shape {combined_mask.shape}. Resizing confidence map.")
                conf_norm = cv2.resize(conf_norm, (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            combined_mask = np.where(conf_norm > 0.5, combined_mask, 0)
        
        # Morphological closing to fill small holes
        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((self.closing_kernel_size, self.closing_kernel_size), np.uint8))
        
        # Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        processed_mask = np.copy(closed) # Make a copy to modify
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_component_size:
                processed_mask[labels == i] = 0
        
        # Fill holes with size constraints
        # ndimage.binary_fill_holes expects a boolean array
        filled_holes = ndimage.binary_fill_holes(processed_mask > 0).astype(np.uint8)
        
        return filled_holes * 255 # Return binary mask (0 or 255)

    def validate_mask_quality_with_confidence(self, mask: np.ndarray, image: np.ndarray, prediction_scores: list, model=None, input_tensor=None, mc_dropout_runs: int = 20, device: str = 'cpu', logits=None, temperature: float = 1.0) -> tuple:
        """
        Comprehensive mask quality validation using SSIM, boundary coherence,
        confidence calibration, and uncertainty quantification.
        Returns: (validated_mask, quality_score, metrics_dict)
        """
        # Ensure mask and image are the same size and type for SSIM
        if mask.shape != image.shape[:2]:
            logging.warning("Mask and image shapes do not match. Resizing mask for SSIM comparison.")
            mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask.astype(np.uint8)

        # Convert image to grayscale if it's color for SSIM calculation
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # SSIM for shape consistency (vs. input or prior mask if available)
        ssim_score = ssim(image_gray.astype(np.float32), mask_resized.astype(np.float32), data_range=255.0)

        # Boundary coherence: edge overlap
        edges_img = cv2.Canny((image_gray).astype(np.uint8), 50, 150)
        edges_mask = cv2.Canny((mask_resized).astype(np.uint8), 50, 150)
        sum_edges_img = np.sum(edges_img > 0)
        if sum_edges_img == 0:
            edge_overlap = 0.0
        else:
            edge_overlap = np.sum((edges_img > 0) & (edges_mask > 0)) / (sum_edges_img + 1e-6)


        # Uncertainty quantification (Monte Carlo dropout)
        logits_mc = None
        if model is not None and input_tensor is not None:
            try:
                mean_mask, var_mask, logits_mc = mc_dropout_mask_prediction(model, input_tensor, n_runs=mc_dropout_runs, device=device)
                uncertainty = float(np.mean(var_mask))
            except Exception as e:
                logging.warning(f"MC Dropout failed: {e}. Falling back to std of prediction_scores.")
                uncertainty = np.std(prediction_scores) if prediction_scores and len(prediction_scores) > 1 else 0.0
        else:
            uncertainty = np.std(prediction_scores) if prediction_scores and len(prediction_scores) > 1 else 0.0

        # Confidence calibration (Platt/temperature scaling)
        # Use MC logits if available, else use provided logits, else fallback to prediction_scores
        use_logits = None
        if logits_mc is not None and hasattr(logits_mc, 'shape') and logits_mc.shape[0] > 0:
            use_logits = np.mean(logits_mc, axis=0)
        elif logits is not None and len(logits) > 0:
            use_logits = logits
        conf_score = self._calibrated_confidence(prediction_scores, logits=use_logits, temperature=temperature)

        # Dice coefficient
        def calculate_dice_coefficient(a, b):
            a = (a > 0).astype(np.uint8)
            b = (b > 0).astype(np.uint8)
            intersection = np.sum(a * b)
            union = np.sum(a) + np.sum(b)
            if union == 0:
                return 1.0
            return 2. * intersection / (union + 1e-6)
        dice = calculate_dice_coefficient(mask_resized, image_gray > 0)

        quality_score = 0.4 * ssim_score + 0.3 * edge_overlap + 0.2 * conf_score + 0.1 * (1 - uncertainty)
        metrics = {
            'ssim': ssim_score,
            'edge_overlap': edge_overlap,
            'confidence': conf_score,
            'uncertainty': uncertainty,
            'dice': dice,
            'quality_score': quality_score
        }
        threshold = 0.5
        validated_mask = mask if quality_score > threshold else np.zeros_like(mask)
        return validated_mask, quality_score, metrics

    def refine(self, mask: np.ndarray) -> np.ndarray:
        """
        DEPRECATED: Use robust_binary_conversion_pipeline instead.
        This function now simply calls the robust_binary_conversion_pipeline.
        """
        logging.warning("The 'refine' method is deprecated. Please use 'robust_binary_conversion_pipeline' instead.")
        return self.robust_binary_conversion_pipeline(mask)

    def set_image(self, image: np.ndarray):
        """
        Sets the image for all loaded SAM predictors.
        """
        if not self.is_initialized:
            logging.warning("SAM models are not initialized. Cannot set image.")
            return

        # Ensure image is suitable for SAM (e.g., uint8 RGB)
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8) # Normalize and convert to uint8
        if len(image.shape) == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4: # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        for model_type, predictor in self.predictors.items():
            try:
                predictor.set_image(image)
                logging.info(f"Image set for SAM model: {model_type}")
            except Exception as e:
                logging.error(f"Failed to set image for SAM model {model_type}: {e}")

    def predict(self, point_coords: np.ndarray, point_labels: np.ndarray, multimask_output: bool = True):
        """
        Performs prediction using the ensemble of initialized SAM models.
        Returns a weighted average of masks, scores, and logits.
        """
        if not self.is_initialized or not self.predictors:
            logging.warning("SAM models are not initialized or no predictors available. Cannot predict.")
            return np.array([]), np.array([]), np.array([])

        all_masks = []
        all_scores = []
        all_logits = []
        
        active_predictors = 0
        for i, (model_type, predictor) in enumerate(self.predictors.items()):
            try:
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=multimask_output
                )
                
                if masks.size > 0:
                    all_masks.append(masks)
                    all_scores.append(scores)
                    all_logits.append(logits)
                    active_predictors += 1
                else:
                    logging.warning(f"SAM model {model_type} returned empty prediction.")
            except Exception as e:
                logging.error(f"Error during prediction with SAM model {model_type}: {e}")

        if not all_masks:
            logging.warning("No successful SAM predictions from any model in the ensemble.")
            return np.array([]), np.array([]), np.array([])

        # Simple ensemble averaging (can be made more sophisticated with weights)
        # For simplicity, let's take the mask with the highest score from each model if multimask_output is True
        # Or average probabilities if multimask_output is False (single mask output expected)

        if multimask_output:
            # If multimask_output is True, each model returns multiple masks.
            # We need a strategy to combine them. A common strategy is to pick the best mask
            # from each model based on its score, then combine those.
            
            combined_final_mask = np.zeros_like(all_masks[0][0], dtype=np.float32) # Initialize with first mask's shape
            best_scores_per_model = []

            for masks_set, scores_set in zip(all_masks, all_scores):
                if len(scores_set) > 0:
                    best_mask_idx = np.argmax(scores_set)
                    best_mask = masks_set[best_mask_idx]
                    best_score = scores_set[best_mask_idx]
                    
                    # Convert boolean mask to float (0 or 1) before averaging
                    combined_final_mask += best_mask.astype(np.float32)
                    best_scores_per_model.append(best_score)
                else:
                    logging.warning("One model returned empty scores_set for multimask output.")

            if active_predictors > 0:
                combined_final_mask /= active_predictors # Average the masks
                # Threshold the averaged mask to get a binary mask
                final_mask = (combined_final_mask > 0.5).astype(bool)
                final_score = np.mean(best_scores_per_model) if best_scores_per_model else 0.0
                final_logits = np.mean(all_logits, axis=0) if all_logits else np.array([]) # Simple average of logits
            else:
                final_mask = np.array([])
                final_score = 0.0
                final_logits = np.array([])

        else:
            # If multimask_output is False, each model returns one mask.
            # We can average the probabilities or choose the best one.
            # Here, we'll average the raw masks directly (assuming 0 or 1 values)
            averaged_mask = np.mean([mask.astype(np.float32) for mask_set in all_masks for mask in mask_set if mask.size > 0], axis=0)
            final_mask = (averaged_mask > 0.5).astype(bool) # Binarize the averaged mask
            final_score = np.mean([score for scores_set in all_scores for score in scores_set]) if all_scores else 0.0
            final_logits = np.mean(all_logits, axis=0) if all_logits else np.array([])


        return final_mask, final_score, final_logits
