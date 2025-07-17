# --- Imports (must be at the very top) ---
import sys
import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms



# --- Canonical Preprocessing and TTA Transforms ---
try:
    from core_models.training_args import get_config
    config = get_config()
except ImportError:
    from config import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Canonical preprocess: Resize, ToTensor, Normalize
PREPROCESS = transforms.Compose([
    transforms.Resize(tuple(getattr(config.data, 'image_size', (224, 224)))),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Canonical PIL-based TTA (configurable)
TTA_PIL = transforms.Compose([
    transforms.RandomResizedCrop(
        getattr(config, 'tta', {}).get('resize', 224),
        scale=tuple(getattr(config, 'tta', {}).get('random_resized_crop', {'scale': (0.8, 1.0)})['scale'])
    ),
    transforms.RandomHorizontalFlip(getattr(config, 'tta', {}).get('horizontal_flip_p', 0.5)),
    transforms.ColorJitter(**getattr(config, 'tta', {}).get('color_jitter', {'brightness': 0.2, 'contrast': 0.2})),
])
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter, defaultdict
import random

# --- Path setup ---
# This block ensures that the root directory of the project is in the Python path,
# allowing for relative imports of modules like 'config' and 'core_models'.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Dummy Configuration and Model Definitions (for independent execution) ---
# In a real setup, these would be imported from external files.
# These dummy classes allow the provided code to run without external dependencies.

class Config:
    """
    A dummy configuration class to simulate the 'config' object.
    It provides necessary attributes like image_size and attribute_heads
    that the PerceptionModule expects.
    """
    def __init__(self):
        self.data = type('data', (object,), {'image_size': (224, 224)})()
        self.best_model_path = None
        self.last_model_path = None
        # Define dummy attribute heads with example number of classes
        self.model = type('model', (object,), {
            'attribute_heads': {
                'shape': 5,          # e.g., circle, square, triangle, polygon, unknown
                'color': 5,          # e.g., red, green, blue, black, unknown
                'fill': 4,           # e.g., solid, outlined, striped, dotted
                'size': 3,           # e.g., small, medium, large
                'orientation': 4,    # e.g., up, down, left, right
                'texture': 2,        # e.g., smooth, rough
            }
        })()

# Initialize a dummy config instance
config = Config()

# Define dummy ImageNet mean and std for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Determine the device for PyTorch tensors (CPU for general compatibility)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PerceptionModule(nn.Module):
    """
    A dummy PerceptionModule class to simulate the actual model's behavior.
    It takes a config object and simulates attribute prediction by returning
    random logits for each attribute head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Simulate attribute heads with simple linear layers
        self.attribute_heads = nn.ModuleDict()
        # The input feature size (10) is arbitrary for this dummy model
        dummy_feature_size = 10
        for attr_type, num_classes in config.model.attribute_heads.items():
            self.attribute_heads[attr_type] = nn.Linear(dummy_feature_size, num_classes)

    def forward(self, images, ground_truth_json_strings, detected_bboxes_batch,
                detected_masks_batch, support_images, support_labels_flat,
                is_simclr_pretraining):
        """
        Simulates the forward pass of the PerceptionModule.
        Generates random features and passes them through dummy attribute heads.
        """
        B = images.shape[0]  # Batch size
        # Simulate extracted features for each image in the batch
        dummy_features = torch.randn(B, 10).to(images.device)

        attribute_logits = {}
        for attr_type, head in self.attribute_heads.items():
            attribute_logits[attr_type] = head(dummy_features)

        # In a real model, this would also return other outputs like detection results,
        # but for attribute extraction, only 'attribute_logits' is relevant here.
        return {'attribute_logits': attribute_logits}

# Dummy BongardPerceptionModel (if needed, but PerceptionModule is the one directly used)
class BongardPerceptionModel(nn.Module):
    """
    A dummy BongardPerceptionModel, primarily to satisfy imports if any part
    of the system expects it. It simply wraps the PerceptionModule.
    """
    def __init__(self, config):
        super().__init__()
        self.perception_module = PerceptionModule(config)

    def forward(self, *args, **kwargs):
        return self.perception_module(*args, **kwargs)

# Dummy augment_image function if utils.augment is not available
augment_image = None
try:
    from utils.augment import augment_image
except ImportError:
    logging.warning("utils.augment not found, TTA functionality may be limited. Using dummy augment_image.")
    # Define a dummy augment_image that just returns the original image
    def augment_image(img_pil: Image.Image) -> Image.Image:
        return img_pil


# --- Inverse attribute maps for index-to-name mapping ---
# These maps are crucial for converting numerical model predictions back
# to human-readable attribute names (e.g., 0 -> 'circle').
# For this runnable example, we define simple dummy maps.
ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2, 'polygon': 3, 'unknown_shape': 4}
ATTRIBUTE_COLOR_MAP = {'red': 0, 'green': 1, 'blue': 2, 'black': 3, 'unknown_color': 4}
ATTRIBUTE_FILL_MAP = {'solid': 0, 'outlined': 1, 'striped': 2, 'dotted': 3}
ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
ATTRIBUTE_ORIENTATION_MAP = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
ATTRIBUTE_TEXTURE_MAP = {'smooth': 0, 'rough': 1}


def _invert_map(m: Dict[str, int]) -> Dict[int, str]:
    """
    Inverts a dictionary, swapping keys and values.
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
logger.setLevel(logging.DEBUG)  # Show debug messages
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
    # In a real scenario, checkpoint loading logic would go here.
    # For this runnable example, we just instantiate the dummy model.
    # The original code had complex checkpoint loading logic. For this
    # self-contained example, we assume the model is ready after instantiation.
    _model.eval() # Set to evaluation mode
    MODEL = _model
    logger.info("PerceptionModule (dummy) initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize PerceptionModule: {e}. CNN inference will not be available.")
    # If model loading fails, MODEL remains None, and subsequent functions
    # will handle this by returning 'unknown' attributes.

def load_perception_model() -> Optional[nn.Module]:
    """
    Loads the PerceptionModule and its weights if not already loaded. Returns the model instance.
    This function is primarily for external use.
    """
    global MODEL
    if MODEL is None:
        try:
            _model = PerceptionModule(config).to(DEVICE)
            # Simplified loading: In a real scenario, this would load actual weights.
            # For this dummy setup, we just instantiate and set to eval mode.
            _model.eval()
            MODEL = _model
            logger.info("PerceptionModule (dummy) loaded successfully via load_perception_model.")
        except Exception as e:
            logger.error(f"Failed to initialize PerceptionModule in load_perception_model: {e}.")
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
    # `unsqueeze(0)` adds a batch dimension (B, C, H, W).
    input_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Provide dummy inputs for parameters typically used during training
        # but required by the `PerceptionModule`'s forward method signature.
        dummy_gts_json_strings = [b'{}']
        # Provide a full-image bbox so attribute_model sees an object
        H, W = input_tensor.shape[2:]
        # full-image bbox as a (1,4) Tensor
        full_bbox = torch.tensor([0, 0, W, H], dtype=torch.float32, device=DEVICE)
        dummy_bboxes_batch = [full_bbox]
        # Full-mask covering the entire image
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
                extracted_features[attr_type] = (f"unknown_{attr_type}", 0.0)
                continue
            # Apply softmax to convert logits to probabilities, then get the most confident prediction.
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = probs.max(0)
            predicted_idx = idx.item()
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
        if 0.8 <= aspect_ratio <= 1.2 and rect_area > 0:  # Aspect ratio close to 1 for squares.
            shape = "square"
            # Confidence based on how much of the bounding rectangle's area is filled by the contour.
            confidence = area / rect_area
        else:
            shape = "polygon" # More general than "quadrilateral" for this context
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
            confidence = 1.0 - (mean_val / 100.0) # Scale confidence based on mean_val
        # 2. Outlined: High mean intensity (light interior, likely background showing through)
        #    and low standard deviation (uniform interior).
        elif mean_val > 150 and std_dev < 30:
            fill_type = "outlined"
            # Confidence increases as the interior gets lighter and more uniform.
            confidence = (mean_val - 150.0) / 105.0 # Scale confidence based on mean_val
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

    # --- Robust TTA and Preprocessing ---
    crops_pil = [img_pil]
    tta_enabled = getattr(config, 'tta', {}).get('enabled', True)
    if tta_enabled:
        try:
            crops_pil.append(TTA_PIL(img_pil))
        except Exception as e:
            logger.warning(f"TTA error: {e}. Using only original crop.")

    crops_tensor = [PREPROCESS(p).to(DEVICE) for p in crops_pil]
    inp = torch.stack(crops_tensor, dim=0)  # (B,3,H,W)
    logger.debug(f"extract_cnn_features → inp.shape={inp.shape}, device={inp.device}")

    # Prepare dummy detection inputs so the model actually runs its heads
    B, C, H, W = inp.shape
    full_bbox = torch.tensor([[0, 0, W, H]], dtype=torch.float32, device=DEVICE)
    dummy_bboxes = [full_bbox.clone() for _ in range(B)]
    full_mask = torch.ones((1, 1, H, W), dtype=torch.float32, device=DEVICE)
    dummy_masks = [full_mask.clone() for _ in range(B)]
    logger.debug(f"extract_cnn_features → bboxes[0].shape={dummy_bboxes[0].shape}, mask.shape={dummy_masks[0].shape}")

    # DEBUG: verify we really have B boxes of shape (1,4)
    # logger.debug(f"extract_cnn_features -> inp.shape={inp.shape}, dtype={inp.dtype}, device={inp.device}")
    # logger.debug(f"extract_cnn_features -> dummy_bboxes_batch[0]={dummy_bboxes_batch[0]}, shape={dummy_bboxes_batch[0].shape}")
    # logger.debug(f"extract_cnn_features -> dummy_masks_batch[0]=Tensor shape{dummy_masks_batch[0].shape}")


    with torch.no_grad():
        model_output = MODEL(
            images=inp,
            ground_truth_json_strings=[b'{}'] * B,
            detected_bboxes_batch=dummy_bboxes,
            detected_masks_batch=dummy_masks,
            support_images=None,
            support_labels_flat=None,
            is_simclr_pretraining=False
        )
    attribute_logits_dict = model_output.get('attribute_logits', {})

    # For each attribute, aggregate predictions across TTA batch
    final_feats, avg_conf_list = {}, []
    for attr, logits in attribute_logits_dict.items():
        logger.debug(f"HEAD {attr}: logits={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        probs = torch.softmax(logits, dim=1)
        max_vals, max_idxs = probs.max(dim=1)
        choice = max_idxs[0].item()
        conf = max_vals.float().mean().item()
        attr_name = attribute_maps_inv.get(attr, {}).get(choice, f"unknown_{attr}")
        final_feats[attr] = (attr_name, conf)
        avg_conf_list.append(conf)

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
