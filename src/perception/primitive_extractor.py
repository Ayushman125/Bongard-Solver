# --- Exported model loader for external use ---
import sys
import os
import logging
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter, defaultdict
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms

# --- Path setup ---
# This block ensures that the root directory of the project is in the Python path,
# allowing for relative imports of modules like 'config' and 'core_models'.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- External Dependencies Import ---
# Initialize CONFIG and PerceptionModule to None. They will be assigned if imports succeed.
CONFIG = None
PerceptionModule = None

try:
    from config import CONFIG
    from core_models.models import PerceptionModule
    logger.info("Successfully imported config and core_models.models.")
except ImportError as e:
    logger.error(f"Failed to import external dependencies (config or core_models): {e}. "
                 "Please ensure 'config.py' and 'core_models/models.py' are correctly set up "
                 "in your project and accessible via the Python path. CNN inference will not be available.")


# Determine the device for PyTorch tensors (CPU for general compatibility)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Attribute maps for index-to-name mapping ---
# These maps define the possible values for each object attribute and their
# corresponding integer indices used by the CNN model.
ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2, 'polygon': 3, 'unknown_shape': 4}
ATTRIBUTE_COLOR_MAP = {'red': 0, 'green': 1, 'blue': 2, 'black': 3, 'unknown_color': 4}
ATTRIBUTE_FILL_MAP = {'solid': 0, 'outlined': 1, 'striped': 2, 'dotted': 3, 'unknown_fill': 4}
ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2, 'unknown_size': 3}
ATTRIBUTE_ORIENTATION_MAP = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'unknown_orientation': 4}
ATTRIBUTE_TEXTURE_MAP = {'smooth': 0, 'rough': 1, 'unknown_texture': 2}

def _invert_map(m: Dict[str, int]) -> Dict[int, str]:
    """Inverts a dictionary mapping string names to integer indices."""
    return {v: k for k, v in m.items()}

# Inverted maps for converting model output indices back to attribute names.
attribute_maps_inv = {
    'shape': _invert_map(ATTRIBUTE_SHAPE_MAP),
    'color': _invert_map(ATTRIBUTE_COLOR_MAP),
    'fill': _invert_map(ATTRIBUTE_FILL_MAP),
    'size': _invert_map(ATTRIBUTE_SIZE_MAP),
    'orientation': _invert_map(ATTRIBUTE_ORIENTATION_MAP),
    'texture': _invert_map(ATTRIBUTE_TEXTURE_MAP),
}

# --- Canonical Preprocessing and TTA Transforms ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Determine image_size based on CONFIG availability
image_size = (224, 224)
if CONFIG:
    image_size = tuple(CONFIG.get('data', {}).get('image_size', (224, 224)))

# Canonical preprocess: Resize, ToTensor, Normalize
PREPROCESS = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Canonical PIL-based TTA (configurable)
# This transform applies various augmentations to the input image for Test-Time Augmentation.
tta_cfg = {}
if CONFIG:
    tta_cfg = CONFIG.get('tta', {})

TTA_PIL = transforms.Compose([
    transforms.RandomResizedCrop(
        tta_cfg.get('resize', 224),
        scale=tuple(tta_cfg.get('random_resized_crop', {'scale': (0.8, 1.0)})['scale'])
    ),
    transforms.RandomHorizontalFlip(tta_cfg.get('horizontal_flip_p', 0.5)),
    transforms.ColorJitter(**tta_cfg.get('color_jitter', {'brightness': 0.2, 'contrast': 0.2})),
])

# --- Model import and global instance ---

# --- Model import and global instance ---
# This block attempts to initialize the PerceptionModule only if it was successfully imported
# and CONFIG is available. The global MODEL is only set by explicit loader calls.
MODEL = None
if PerceptionModule is not None and CONFIG is not None:
    try:
        _model = PerceptionModule(CONFIG).to(DEVICE)
        _model.eval()
        MODEL = _model
        logger.info("PerceptionModule initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PerceptionModule: {e}. CNN inference will not be available.")
else:
    logger.warning("PerceptionModule or CONFIG not available due to import errors. CNN inference will not be available.")


# --- Explicit model loader for fine-tuning or evaluation ---
def initialize_perception_model(checkpoint_path: str, device: torch.device, config: dict = None) -> Optional[nn.Module]:
    """
    Loads a PerceptionModule from a checkpoint for fine-tuning or evaluation.
    Sets the global MODEL for use by extract_cnn_features, etc.
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth file)
        device (torch.device): Device to load the model onto
        config (dict, optional): Config dict. If None, uses global CONFIG.
    Returns:
        nn.Module or None: The loaded PerceptionModule, or None if failed.
    """
    global MODEL
    if config is None:
        config = CONFIG
    if PerceptionModule is None or config is None:
        logger.error("Cannot initialize PerceptionModule: missing class or config.")
        return None
    try:
        model = PerceptionModule(config).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        MODEL = model
        logger.info(f"PerceptionModule loaded from {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load PerceptionModule from {checkpoint_path}: {e}")
        return None


def load_perception_model() -> nn.Module:
    """
    Returns the initialized PerceptionModule (MODEL) for use in other modules.
    If the model failed to initialize, returns None.
    """
    return MODEL


def single_inference(img_pil: Image.Image, model: Optional[nn.Module] = None) -> Dict[str, Tuple[str, float]]:
    """
    Performs a single inference pass on a cropped object image using the CNN model.
    This function is a helper for `extract_cnn_features` to enable Test-Time Augmentation (TTA).

    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
        model (Optional[nn.Module]): The CNN model to use for inference. If None,
                                      the globally loaded `MODEL` is used.

    Returns:
        Dict[str, Tuple[str, float]]: A dictionary mapping attribute type (e.g., 'shape')
                                      to a tuple of (predicted_value_name, confidence).
    """
    # Use the provided model instance, or fall back to the global MODEL.
    current_model = model if model is not None else MODEL

    if current_model is None:
        logger.error("CNN model not available for single inference.")
        # Return 'unknown' for all expected attributes if the model is not loaded.
        return {attr_type: (f"unknown_{attr_type}", 0.0) for attr_type in attribute_maps_inv.keys()}

    # Preprocess the PIL image: resize, convert to tensor, and normalize.
    # `unsqueeze(0)` adds a batch dimension (B, C, H, W) for a single image.
    input_tensor = PREPROCESS(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Provide dummy inputs for parameters typically used during training
        # but required by the `PerceptionModule`'s forward method signature.
        # These are placeholders as the current task is only attribute extraction
        # on a pre-cropped object.
        dummy_gts_json_strings = [b'{}'] # Empty JSON string for ground truth
        H, W = input_tensor.shape[2:]
        # Full-image bbox as a (1,4) Tensor, indicating the object covers the whole input.
        full_bbox = torch.tensor([0, 0, W, H], dtype=torch.float32, device=DEVICE)
        dummy_bboxes_batch = [full_bbox]
        # Full-mask covering the entire image, indicating the object is fully visible.
        dummy_masks_batch = [torch.ones(1, 1, H, W, dtype=torch.float32, device=DEVICE)]

        # Perform the forward pass through the model.
        model_output = current_model(
            images=input_tensor,
            ground_truth_json_strings=dummy_gts_json_strings,
            detected_bboxes_batch=dummy_bboxes_batch,
            detected_masks_batch=dummy_masks_batch,
            support_images=None,
            support_labels_flat=None,
            is_simclr_pretraining=False
        )

        # Extract attribute logits from the model's output.
        attribute_logits_dict = model_output.get('attribute_logits', {})
        extracted_features: Dict[str, Tuple[str, float]] = {}

        # Process the logits for each attribute type (e.g., 'shape', 'color').
        for attr_type, logits in attribute_logits_dict.items():
            if logits.numel() == 0:
                # If no logits are returned for an attribute, assign unknown.
                extracted_features[attr_type] = (f"unknown_{attr_type}", 0.0)
                continue
            # Apply softmax to convert logits to probabilities, then get the most confident prediction.
            probs = torch.softmax(logits, dim=1).squeeze(0) # Squeeze to remove batch dimension
            conf, idx = probs.max(0) # Get max probability and its index
            predicted_idx = idx.item()
            # Map the predicted index back to the attribute name using the inverted map.
            attr_name = attribute_maps_inv.get(attr_type, {}).get(predicted_idx, f"unknown_{attr_type}")
            extracted_features[attr_type] = (attr_name, conf.item())

    return extracted_features


def extract_shape_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts the shape of an object from a PIL image using classical computer vision techniques.
    It identifies basic shapes like triangles, squares, circles, and general polygons
    by analyzing contours and their properties (number of vertices, aspect ratio, circularity).

    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.

    Returns:
        Tuple[str, float]: A tuple containing the predicted shape name (string)
                           and its confidence score (float between 0 and 1).
    """
    # Convert the PIL image to a grayscale NumPy array.
    if img_pil.mode != 'L':
        gray = np.array(img_pil.convert('L'))
    else:
        gray = np.array(img_pil)

    # Apply Otsu's thresholding to convert the grayscale image into a binary image.
    # `cv2.THRESH_BINARY_INV` inverts the thresholding, so objects (typically darker)
    # become white (255) and the background becomes black (0).
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the binary image.
    # `cv2.RETR_EXTERNAL` retrieves only the extreme outer contours.
    # `cv2.CHAIN_APPROX_SIMPLE` compresses horizontal, vertical, and diagonal segments
    # by leaving only their end points.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no contours are found, it means no distinct object was detected.
        logger.debug("No contours found for shape extraction.")
        return "unknown_shape", 0.0

    # Find the largest contour by area, assuming it represents the main object.
    main_contour = max(contours, key=cv2.contourArea)

    # Approximate the polygon of the main contour.
    # `epsilon` is the maximum distance from the contour to the approximated polygon.
    # A smaller epsilon means a more precise approximation, potentially more vertices.
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    num_vertices = len(approx)  # Number of vertices in the approximated polygon.
    shape = "polygon" # Default to polygon if not specifically matched
    confidence = 0.0

    # Calculate the area and perimeter of the main contour for more robust shape analysis.
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)

    if num_vertices == 3:
        shape = "triangle"
        # Confidence for a triangle is based on the ratio of its area to the area of its convex hull.
        # A compact, well-defined triangle will have this ratio close to 1.0.
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            confidence = area / hull_area
        else:
            confidence = 0.5  # Default confidence if hull area is zero (e.g., very small contour).
    elif num_vertices == 4:
        # For shapes with 4 vertices, check if it's a rectangle or square.
        x, y, w, h = cv2.boundingRect(main_contour)  # Get the bounding rectangle.
        aspect_ratio = float(w) / h
        rect_area = w * h

        # Check the aspect ratio to distinguish between squares and general quadrilaterals.
        # A square has an aspect ratio close to 1.
        if 0.8 <= aspect_ratio <= 1.2 and rect_area > 0:
            shape = "square"
            # Confidence based on how much of the bounding rectangle's area is filled by the contour.
            # A perfectly filled square would have this ratio close to 1.0.
            confidence = area / rect_area
        else:
            shape = "polygon" # More general than "quadrilateral" for this context
            confidence = 0.7  # General quadrilateral, assigned a slightly lower default confidence.
    elif num_vertices > 4:
        # For shapes with more than 4 vertices, check for circularity.
        (x, y), radius = cv2.minEnclosingCircle(main_contour)  # Get the minimum enclosing circle.
        circle_area = np.pi * (radius ** 2)
        if circle_area > 0:
            # Circularity is the ratio of the contour's area to the area of its minimum enclosing circle.
            # A perfect circle will have this ratio close to 1.0.
            circularity = area / circle_area
            if circularity > 0.8:  # Threshold for high circularity.
                shape = "circle"
                confidence = circularity
            else:
                shape = "polygon"  # If not highly circular, it's a generic polygon.
                confidence = 0.5  # Lower confidence for generic polygons.
        else:
            # If circle_area is zero (e.g., extremely small contour), default to polygon.
            shape = "polygon"
            confidence = 0.5

    # Ensure the confidence score is clamped between 0 and 1.
    confidence = max(0.0, min(1.0, confidence))

    logger.debug(f"CV Extracted Shape: {shape} with confidence {confidence:.4f}")
    return shape, confidence


def extract_fill_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts the fill type (solid, outlined, striped, dotted) of an object
    from a PIL image using classical computer vision techniques.
    It analyzes the pixel intensity distribution within the object's masked area.

    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.

    Returns:
        Tuple[str, float]: A tuple containing the predicted fill type name (string)
                           and its confidence score (float between 0 and 1).
    """
    # Convert the PIL image to a grayscale NumPy array.
    if img_pil.mode != 'L':
        gray = np.array(img_pil.convert('L'))
    else:
        gray = np.array(img_pil)

    # Apply Otsu's thresholding to create a binary mask of the object.
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no contours are found, return 'unknown_fill'.
        logger.debug("No contours found for fill extraction.")
        return "unknown_fill", 0.0

    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    # Draw the largest contour onto a new mask image, filled with white (255).
    cv2.drawContours(mask, [main_contour], -1, 255, cv2.FILLED)

    # Extract pixel values within the masked area.
    masked_pixels = gray[mask == 255]
    if masked_pixels.size == 0:
        # If no pixels are within the mask (e.g., very small object), return 'unknown_fill'.
        logger.debug("No pixels found within the mask for fill extraction.")
        return "unknown_fill", 0.0
    mean_val = np.mean(masked_pixels)  # Calculate the mean pixel intensity within the object.

    # Analyze internal texture for patterned fills (striped/dotted).
    # Erode the mask to get a smaller "inner" region of the object. This helps
    # to exclude edge pixels which might have varying intensities due to anti-aliasing
    # and focus on the internal pattern.
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    inner_pixels = gray[eroded_mask == 255]

    fill_type = "unknown_fill"
    confidence = 0.0

    if inner_pixels.size > 0:
        std_dev = np.std(inner_pixels)  # Calculate the standard deviation of pixel intensities in the inner region.

        # Heuristics to determine fill type:
        # 1. Solid: Low mean intensity (dark object) and low standard deviation (uniform pixels).
        if mean_val < 100 and std_dev < 30:
            fill_type = "solid"
            # Confidence increases as the object's interior is darker and more uniform.
            confidence = 1.0 - (mean_val / 255.0) # Scale confidence based on darkness (0=dark, 1=light)
            confidence = max(0.0, min(1.0, confidence * (1 - std_dev / 50.0))) # Penalize for std_dev
        # 2. Outlined: High mean intensity (light interior, likely background showing through)
        #    and low standard deviation (uniform interior).
        elif mean_val > 150 and std_dev < 30:
            fill_type = "outlined"
            # Confidence increases as the interior is lighter and more uniform.
            confidence = (mean_val - 150.0) / 105.0 # Scale confidence based on brightness
            confidence = max(0.0, min(1.0, confidence * (1 - std_dev / 50.0))) # Penalize for std_dev
        # 3. Striped/Dotted (Patterned): High standard deviation suggests significant
        #    variation in pixel values, indicating a pattern.
        elif std_dev > 50:
            # For simplicity, and without more advanced pattern recognition (e.g., FFT for stripes,
            # blob detection for dots), we randomly assign 'striped' or 'dotted' for demonstration.
            # In a real system, more sophisticated image analysis would be needed here.
            fill_type = "striped" if random.random() > 0.5 else "dotted"
            # Confidence scales with the standard deviation (higher std_dev, higher confidence in pattern).
            confidence = min(1.0, std_dev / 100.0) # Max confidence at std_dev 100

    # Ensure the confidence score is clamped between 0 and 1.
    confidence = max(0.0, min(1.0, confidence))

    logger.debug(f"CV Extracted Fill: {fill_type} with confidence {confidence:.4f}")
    return fill_type, confidence


def extract_cnn_features(img_pil: Image.Image) -> Tuple[Dict[str, Tuple[str, float]], float]:
    """
    Extracts features (shape, color, fill, etc.) and their confidences
    using a pre-trained CNN model (PerceptionModule) combined with
    Test-Time Augmentation (TTA). TTA involves performing inference on
    multiple augmented versions of the input image and aggregating the results.

    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.

    Returns:
        Tuple[Dict[str, Tuple[str, float]], float]:
            - A dictionary mapping feature type (e.g., 'shape') to a tuple of (predicted_value_name, confidence).
            - The average confidence across all features.
    """
    # Inspect the original PIL crop
    logger.debug(f"INPUT PIL image: type={type(img_pil)}, mode={getattr(img_pil, 'mode', None)}, size={getattr(img_pil, 'size', None)}")

    # If the model is not loaded, return a tuple matching the normal return type.
    if MODEL is None:
        logger.error("CNN model not available for inference.")
        # Return dummy 'unknown' features with 0.0 confidence.
        dummy = {t: (f"unknown_{t}", 0.0) for t in attribute_maps_inv}
        return dummy, 0.0

    # --- Robust TTA and Preprocessing ---
    crops_pil = [img_pil] # Start with the original image
    tta_enabled = CONFIG.get('tta', {}).get('enabled', True) if CONFIG else False # Check CONFIG for TTA
    if tta_enabled:
        try:
            # Apply Test-Time Augmentation to create an augmented version of the image.
            crops_pil.append(TTA_PIL(img_pil))
        except Exception as e:
            logger.warning(f"TTA error: {e}. Using only original crop.")

    # Preprocess all PIL images (original + augmented) into PyTorch tensors.
    crops_tensor = [PREPROCESS(p).to(DEVICE) for p in crops_pil]
    # Stack the tensors to form a batch (Batch_size, Channels, Height, Width).
    inp = torch.stack(crops_tensor, dim=0)
    logger.debug(f"extract_cnn_features → inp.shape={inp.shape}, device={inp.device}")

    # Prepare dummy detection inputs so the model actually runs its attribute heads.
    # The PerceptionModule expects these even if object detection isn't the primary goal here.
    B, C, H, W = inp.shape
    # Create bounding boxes that cover the entire image for each item in the batch.
    full_bbox = torch.tensor([[0, 0, W, H]], dtype=torch.float32, device=DEVICE)
    dummy_bboxes = [full_bbox.clone() for _ in range(B)]
    # Create masks that cover the entire image for each item in the batch.
    full_mask = torch.ones((1, 1, H, W), dtype=torch.float32, device=DEVICE)
    dummy_masks = [full_mask.clone() for _ in range(B)]
    logger.debug(f"extract_cnn_features → bboxes[0].shape={dummy_bboxes[0].shape}, mask.shape={dummy_masks[0].shape}")

    with torch.no_grad(): # Disable gradient calculation for inference to save memory and speed up computation.
        model_output = MODEL(
            images=inp,
            ground_truth_json_strings=[b'{}'] * B, # Dummy ground truth
            detected_bboxes_batch=dummy_bboxes,
            detected_masks_batch=dummy_masks,
            support_images=None,
            support_labels_flat=None,
            is_simclr_pretraining=False
        )
    attribute_logits_dict = model_output.get('attribute_logits', {})

    # For each attribute, aggregate predictions across the TTA batch.
    # This involves averaging confidences and potentially taking a majority vote for the class.
    final_feats, avg_conf_list = {}, []
    for attr, logits in attribute_logits_dict.items():
        logger.debug(f"HEAD {attr}: logits={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        probs = torch.softmax(logits, dim=1) # Convert logits to probabilities
        max_vals, max_idxs = probs.max(dim=1) # Get the max probability and its index for each TTA crop

        # For the final prediction, we take the prediction from the first crop (original image)
        # or could implement a more sophisticated voting mechanism across TTA crops.
        # Here, `choice` is the predicted index from the first (original) image in the batch.
        choice = max_idxs[0].item()

        # The confidence is the average of the maximum probabilities across all TTA crops for this attribute.
        conf = max_vals.float().mean().item()
        attr_name = attribute_maps_inv.get(attr, {}).get(choice, f"unknown_{attr}")
        final_feats[attr] = (attr_name, conf)
        avg_conf_list.append(conf)

    # Calculate the overall average confidence across all attributes.
    avg_confidence = float(sum(avg_conf_list) / len(avg_conf_list)) if avg_conf_list else 0.0
    logger.debug(f"FINAL features -> {final_feats}")
    logger.debug(f"FINAL avg_conf -> {avg_confidence:.4f}")
    return final_feats, avg_confidence


if __name__ == '__main__':
    # Configure basic logging for the example execution.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running primitive_extractor.py example.")

    # Define image size for dummy images.
    img_size = 224

    # --- Test 1: Filled Red Triangle ---
    logger.info("\n--- Creating and Testing a Filled Red Triangle ---")
    # Create a new PIL image with a gray background.
    dummy_img_pil_triangle = Image.new('RGB', (img_size, img_size), (200, 200, 200))
    draw_triangle = ImageDraw.Draw(dummy_img_pil_triangle)

    # Draw a filled red triangle on the dummy image.
    pts_triangle = [(112, 50), (62, 150), (162, 150)]
    draw_triangle.polygon(pts_triangle, fill=(255, 0, 0), outline=(0, 0, 0))

    # Test classical CV extractors on the triangle image.
    logger.info("\n--- Testing Classical CV Extraction (Triangle) ---")
    shape_t, shape_conf_t = extract_shape_conf(dummy_img_pil_triangle)
    fill_t, fill_conf_t = extract_fill_conf(dummy_img_pil_triangle)
    logger.info(f"CV Extraction: Shape='{shape_t}' ({shape_conf_t:.4f}), Fill='{fill_t}' ({fill_conf_t:.4f})")

    # Test CNN feature extractor with TTA on the triangle image.
    logger.info("\n--- Testing CNN Feature Extraction (Triangle) with TTA ---")
    cnn_features_t, cnn_avg_conf_t = extract_cnn_features(dummy_img_pil_triangle)
    logger.info(f"CNN Extraction Results (with TTA, Avg Conf: {cnn_avg_conf_t:.4f}):")
    for feat_type, (value, conf) in cnn_features_t.items():
        logger.info(f"     {feat_type}: '{value}' ({conf:.4f})")

    # --- Test 2: Outlined Blue Circle ---
    logger.info("\n--- Creating and Testing an Outlined Blue Circle ---")
    # Create a new PIL image with a white background for the circle.
    dummy_img_pil_circle = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw_circle = ImageDraw.Draw(dummy_img_pil_circle)
    # Draw an outlined blue circle.
    draw_circle.ellipse([50, 50, 150, 150], outline=(0, 0, 255), width=5)

    # Test classical CV extractors on the circle image.
    logger.info("\n--- Testing Classical CV Extraction (Circle) ---")
    shape_c, shape_conf_c = extract_shape_conf(dummy_img_pil_circle)
    fill_c, fill_conf_c = extract_fill_conf(dummy_img_pil_circle)
    logger.info(f"CV Extraction: Shape='{shape_c}' ({shape_conf_c:.4f}), Fill='{fill_c}' ({fill_conf_c:.4f})")

    # Test CNN feature extractor with TTA on the circle image.
    logger.info("\n--- Testing CNN Feature Extraction (Circle) with TTA ---")
    cnn_features_c, cnn_avg_conf_c = extract_cnn_features(dummy_img_pil_circle)
    logger.info(f"CNN Extraction Results (with TTA, Avg Conf: {cnn_avg_conf_c:.4f}):")
    for feat_type, (value, conf) in cnn_features_c.items():
        logger.info(f"     {feat_type}: '{value}' ({conf:.4f})")

    logger.info("\nPrimitive extractor example finished.")
