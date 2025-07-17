import sys
import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter, defaultdict
import random

# --- Path setup ---
# This block ensures that the root directory of the project is in the Python path,
# allowing for relative imports of modules like 'config' and 'core_models'.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Import configuration and attribute maps ---
# Always import config from core_models.training_args (dataclass), and all other constants from config.py
try:
    from core_models.training_args import config
    from config import (
        ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP,
        ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP,
        IMAGENET_MEAN, IMAGENET_STD, DEVICE
    )
    from core_models.models import BongardPerceptionModel, PerceptionModule
    try:
        from utils.augment import augment_image
    except ImportError:
        logging.warning("utils.augment not found, TTA functionality may be limited.")
        augment_image = None
except ImportError as e:
    print(f"Error importing core modules or config: {e}")
    raise

# --- Inverse attribute maps for index-to-name mapping ---
def _invert_map(m: Dict[str, int]) -> Dict[int, str]:
    """
    Inverts a dictionary, swapping keys and values. This is used to map
    numerical attribute indices (from model predictions) back to human-readable
    attribute names (e.g., 0 -> 'circle').
    """
    return {v: k for k, v in m.items()}

# Create inverse maps for all defined attribute types.
attribute_maps_inv = {
    'shape': _invert_map(ATTRIBUTE_SHAPE_MAP),
    'color': _invert_map(ATTRIBUTE_COLOR_MAP),
    'fill': _invert_map(ATTRIBUTE_FILL_MAP),
    'size': _invert_map(ATTRIBUTE_SIZE_MAP),
    'orientation': _invert_map(ATTRIBUTE_ORIENTATION_MAP),
    'texture': _invert_map(ATTRIBUTE_TEXTURE_MAP),
}

# --- Logger Setup ---
# Configure the logger for this module to output informational messages.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # show debug messages
# Add a stream handler if no handlers are already configured,
# ensuring logs are printed to standard output.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Preprocessing transform for CNN input ---
# This torchvision transform pipeline prepares PIL images for input into the CNN.
# It resizes images to a standard size and normalizes pixel values using ImageNet statistics.
preprocess_transform = T.Compose([
    T.Resize(tuple(getattr(config.data, 'image_size', (224, 224)))),
    T.ToTensor(),  # Converts PIL Image to PyTorch Tensor (HWC to CHW, 0-255 to 0.0-1.0)
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalizes with ImageNet stats
])

# --- Global Model Initialization ---
# This section attempts to load the pre-trained perception model.
# The model is loaded once globally to avoid repeated loading overhead.
MODEL: Optional[nn.Module] = None
try:
    _model = PerceptionModule(config).to(DEVICE)
    # Try both best and last checkpoints
    candidates = []
    best_cp = getattr(config, 'best_model_path', None)
    if best_cp:
        candidates.append(best_cp)
    last_cp = getattr(config, 'last_model_path', None)
    if last_cp:
        candidates.append(last_cp)
    if not last_cp or last_cp != "checkpoints/bongard_perception_last.pth":
        candidates.append("checkpoints/bongard_perception_last.pth")

    for cp in candidates:
        if cp and os.path.exists(cp):
            logger.info(f"Loading PerceptionModel checkpoint from {cp}")
            state = torch.load(cp, map_location=DEVICE)
            # Handle possible 'perception_module.' prefix in keys
            if isinstance(state, dict) and any(k.startswith('perception_module.') for k in state.keys()):
                new_state = {k.replace('perception_module.', ''): v for k, v in state.items()}
                _model.load_state_dict(new_state, strict=False)
            elif isinstance(state, dict) and 'perception_module' in state:
                _model.load_state_dict(state['perception_module'], strict=True)
            else:
                _model.load_state_dict(state, strict=False)
            _model.eval()
            MODEL = _model
            break

    if MODEL is None:
        logger.warning("No perception checkpoint found; MODEL remains None.")
    else:
        logger.info("PerceptionModel loaded successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PerceptionModel: {e}. CNN inference will not be available.")
    # If model loading fails, MODEL remains None, and subsequent functions
    # will handle this by returning 'unknown' attributes.


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
        logger.error("CNN model not available for inference.")
        # Return 'unknown' for all expected attributes if the model is not loaded.
        # Return as a tuple of (dict, ) to match the expected tuple return in extract_cnn_features fallback.
        return {attr_type: (f"unknown_{attr_type}", 0.0) for attr_type in attribute_maps_inv.keys()}

    # Preprocess the PIL image: resize, convert to tensor, and normalize.
    # `unsqueeze(0)` adds a batch dimension (B, C, H, W).
    input_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():  # Disable gradient calculation for efficient inference.
        # Provide dummy inputs for parameters typically used during training
        # but required by the `PerceptionModule`'s forward method signature.
        dummy_gts_json_strings = [b'{}']  # Empty JSON string as bytes
        # Empty tensors for detected bounding boxes and masks.
        dummy_bboxes_batch = [torch.empty(0, 4, dtype=torch.float32, device=DEVICE)]
        dummy_masks_batch = [torch.empty(0, 1, input_tensor.shape[2], input_tensor.shape[3], dtype=torch.float32, device=DEVICE)]

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
                # If no logits are present for an attribute, mark it as unknown.
                extracted_features[attr_type] = (f"unknown_{attr_type}", 0.0)
                continue

            # Apply softmax to convert logits to probabilities, then get the most confident prediction.
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = probs.max(0)  # `conf` is the confidence, `idx` is the predicted class index.
            predicted_idx = idx.item()

            # Map the predicted numerical index back to its human-readable attribute name.
            attr_name = attribute_maps_inv.get(attr_type, {}).get(predicted_idx, f"unknown_{attr_type}")
            extracted_features[attr_type] = (attr_name, conf.item())

    return extracted_features


def extract_shape_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts the shape of an object from a PIL image using classical computer vision techniques.
    It identifies basic shapes like triangles, squares, circles, and general polygons
    by analyzing contours and their properties.

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
        return "unknown", 0.0

    # Find the largest contour by area, assuming it represents the main object.
    main_contour = max(contours, key=cv2.contourArea)

    # Approximate the polygon of the main contour.
    # `epsilon` is the maximum distance from the contour to the approximated polygon.
    # A smaller epsilon means a more precise approximation, potentially more vertices.
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)

    num_vertices = len(approx)  # Number of vertices in the approximated polygon.
    shape = "other"
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
        if 0.8 <= aspect_ratio <= 1.2 and rect_area > 0:  # Aspect ratio close to 1 for squares.
            shape = "square"
            # Confidence based on how much of the bounding rectangle's area is filled by the contour.
            confidence = area / rect_area
        else:
            shape = "quadrilateral"
            confidence = 0.7  # General quadrilateral, assigned a slightly lower default confidence.
    elif num_vertices > 4:
        # For shapes with more than 4 vertices, check for circularity.
        (x, y), radius = cv2.minEnclosingCircle(main_contour)  # Get the minimum enclosing circle.
        circle_area = np.pi * (radius ** 2)
        if circle_area > 0:
            circularity = area / circle_area  # Ratio of contour area to enclosing circle area.
            if circularity > 0.8:  # Threshold for high circularity.
                shape = "circle"
                confidence = circularity
            else:
                shape = "polygon"  # If not highly circular, it's a generic polygon.
                confidence = 0.5  # Lower confidence for generic polygons.

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
        return "unknown_fill", 0.0

    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    # Draw the largest contour onto a new mask image, filled with white (255).
    cv2.drawContours(mask, [main_contour], -1, 255, cv2.FILLED)

    # Extract pixel values within the masked area.
    masked_pixels = gray[mask == 255]
    if masked_pixels.size == 0:
        # If no pixels are within the mask (e.g., very small object), return 'unknown_fill'.
        return "unknown_fill", 0.0
    mean_val = np.mean(masked_pixels)  # Calculate the mean pixel intensity within the object.

    # Analyze internal texture for patterned fills (striped/dotted).
    # Erode the mask to get a smaller "inner" region of the object. This helps
    # to exclude edge pixels which might have varying intensities due to anti-aliasing.
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    inner_pixels = gray[eroded_mask == 255]

    fill_type = "unknown_fill"
    confidence = 0.0

    if inner_pixels.size > 0:
        std_dev = np.std(inner_pixels)  # Calculate the standard deviation of pixel intensities in the inner region.

        # Heuristics to determine fill type:
        # 1. Solid: Low mean intensity (dark object) and low standard deviation (uniform).
        if mean_val < 100 and std_dev < 30:
            fill_type = "solid"
            # Confidence increases as the object gets darker and more uniform.
            confidence = 1.0 - (mean_val / 100.0)
        # 2. Outlined: High mean intensity (light interior, likely background showing through)
        #    and low standard deviation (uniform interior).
        elif mean_val > 150 and std_dev < 30:
            fill_type = "outlined"
            # Confidence increases as the interior gets lighter and more uniform.
            confidence = (mean_val - 150.0) / 105.0
        # 3. Striped/Dotted (Patterned): High standard deviation suggests significant
        #    variation in pixel values, indicating a pattern.
        elif std_dev > 50:
            # For simplicity, and without more advanced pattern recognition (e.g., FFT for stripes,
            # blob detection for dots), we randomly assign 'striped' or 'dotted' for demonstration.
            fill_type = "striped" if random.random() > 0.5 else "dotted"
            # Confidence scales with the standard deviation (higher std_dev, higher confidence in pattern).
            confidence = min(1.0, std_dev / 100.0)

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
        dummy = {t: (f"unknown_{t}", 0.0) for t in attribute_maps_inv}
        return dummy, 0.0

    crops_pil = [img_pil]
    if augment_image:
        try:
            aug_np = augment_image(img_pil)
            logger.debug(f"augment_image → type={type(aug_np)}, shape={getattr(aug_np,'shape', None)}, dtype={getattr(aug_np,'dtype', None)}, strides={getattr(aug_np,'strides', None)}")
            aug_np = np.ascontiguousarray(aug_np)
            logger.debug(f"After ascontiguousarray: strides={aug_np.strides}")
            aug_t = torch.from_numpy(aug_np)
            logger.debug(f"torch.from_numpy → type={type(aug_t)}, shape={aug_t.shape}, dtype={aug_t.dtype}")
            if aug_t.ndim == 3 and aug_t.shape[2] == 3:
                aug_t = aug_t.permute(2, 0, 1)
                logger.debug(f"Permuted to C×H×W: shape={aug_t.shape}")
            aug_t = aug_t.float().to(DEVICE)
            logger.debug(f"Moved to device {DEVICE}: shape={aug_t.shape}, dtype={aug_t.dtype}")
            # For now, keep the PIL path for compatibility:
            img_for_pil = aug_t.cpu().clamp(0,255).byte().permute(1,2,0).numpy() if aug_t.max() > 1.5 else (aug_t.cpu() * 255).clamp(0,255).byte().permute(1,2,0).numpy()
            crops_pil.append(Image.fromarray(img_for_pil))
        except Exception as e:
            logger.warning(f"TTA error: {e}. Skipping augmentation.")

    all_votes: Dict[str, List[str]] = defaultdict(list)
    all_confs: Dict[str, List[float]] = defaultdict(list)


    for c in crops_pil:
        logger.debug(f"FEEDING to single_inference: type={type(c)}, mode={getattr(c, 'mode', None)}, size={getattr(c, 'size', None)}")
        inference_results = single_inference(c)
        for attr_type, (val, conf) in inference_results.items():
            all_votes[attr_type].append(val)
            all_confs[attr_type].append(conf)

    final_extracted_features: Dict[str, Tuple[str, float]] = {}

    # Aggregate the results for each attribute type using majority voting and average confidence.
    total_conf = 0.0
    n_attr = 0
    for attr_type in all_votes.keys():
        votes = all_votes[attr_type]
        confs = all_confs[attr_type]

        if not votes:
            # If no votes were collected for an attribute, mark it as unknown.
            final_extracted_features[attr_type] = (f"unknown_{attr_type}", 0.0)
            continue

        # Determine the most common predicted value (majority vote).
        most_common_val = Counter(votes).most_common(1)[0][0]

        # Calculate the average confidence only for those predictions that match the majority value.
        matching_confs = [c for v, c in zip(votes, confs) if v == most_common_val]
        avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.0

        final_extracted_features[attr_type] = (most_common_val, avg_conf)
        total_conf += avg_conf
        n_attr += 1
        logger.debug(f"CNN Extracted {attr_type.capitalize()} (TTA): {most_common_val} with confidence {avg_conf:.4f}")

    # Compute average confidence across all attributes
    avg_confidence = total_conf / n_attr if n_attr > 0 else 0.0
    logger.debug(f"FINAL features → {final_extracted_features}")
    logger.debug(f"FINAL avg_conf → {avg_confidence:.4f}")
    return final_extracted_features, avg_confidence


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
    # This will use the dummy model if a real one isn't loaded due to path/config issues.
    logger.info("\n--- Testing CNN Feature Extraction (Triangle) with TTA ---")
    cnn_features_t = extract_cnn_features(dummy_img_pil_triangle)
    logger.info(f"CNN Extraction Results (with TTA):")
    for feat_type, (value, conf) in cnn_features_t.items():
        logger.info(f"    {feat_type}: '{value}' ({conf:.4f})")

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
    cnn_features_c = extract_cnn_features(dummy_img_pil_circle)
    logger.info(f"CNN Extraction Results (with TTA):")
    for feat_type, (value, conf) in cnn_features_c.items():
        logger.info(f"    {feat_type}: '{value}' ({conf:.4f})")
