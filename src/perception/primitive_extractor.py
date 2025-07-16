# Folder: bongard_solver/src/perception/
# File: primitive_extractor.py

import logging
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Tuple, Dict, Any, Optional

# Import global configuration and model components
try:
    from config import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE, YOLO_CLASS_MAP
    from core_models.models import PerceptionModule # Assuming PerceptionModule is in core_models/models.py
except ImportError as e:
    logging.error(f"Failed to import from config or core_models.models: {e}. Using dummy values/classes.")
    CONFIG = {'model': {'object_detector_model_path': None, 'detection_confidence_threshold': 0.1, 'image_size': [224, 224]}, 'data': {'image_size': [224, 224]}}
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device('cpu')
    YOLO_CLASS_MAP = {0: 'circle', 1: 'square', 2: 'triangle'}
    class PerceptionModule(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            logging.warning("Dummy PerceptionModule used in primitive_extractor.")
            self.cfg = cfg
            self.linear = nn.Linear(3, len(YOLO_CLASS_MAP)) # Dummy output
        def forward(self, x):
            # Simulate a simple output based on random data
            return {'bongard_logits': torch.randn(x.shape[0], len(YOLO_CLASS_MAP))}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Preprocessing transform for CNN model input
preprocess_transform = T.Compose([
    T.Resize(CONFIG['data']['image_size']), # Resize to model's expected input size
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Global variable for the loaded CNN model
_CNN_MODEL: Optional[PerceptionModule] = None

def _load_cnn_model():
    """Loads the CNN model once."""
    global _CNN_MODEL
    if _CNN_MODEL is None:
        model_path = CONFIG['model'].get('object_detector_model_path')
        if model_path and os.path.exists(model_path):
            try:
                # Assuming PerceptionModule is your main model for feature extraction
                _CNN_MODEL = PerceptionModule(CONFIG).to(DEVICE)
                # Load state dict, but handle if it's a YOLO model or a PyTorch model
                if model_path.endswith('.pt') or model_path.endswith('.pth'):
                    # This is for a custom PyTorch model
                    state_dict = torch.load(model_path, map_location=DEVICE)
                    _CNN_MODEL.load_state_dict(state_dict, strict=False) # strict=False for partial loads
                    logger.info(f"Loaded PerceptionModule from {model_path}.")
                else:
                    logger.warning(f"Model path {model_path} is not a .pt/.pth file. Assuming it's a dummy or handled externally.")
                _CNN_MODEL.eval()
            except Exception as e:
                logger.error(f"Error loading CNN model from {model_path}: {e}", exc_info=True)
                _CNN_MODEL = PerceptionModule(CONFIG).to(DEVICE) # Fallback to dummy
                _CNN_MODEL.eval()
        else:
            logger.warning(f"CNN model path '{model_path}' not found or not specified. Using dummy PerceptionModule.")
            _CNN_MODEL = PerceptionModule(CONFIG).to(DEVICE) # Use dummy if path not found
            _CNN_MODEL.eval()
    return _CNN_MODEL

def extract_shape_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts shape and confidence using classical computer vision.
    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
    Returns:
        Tuple[str, float]: (shape_name, confidence_score)
    """
    gray = np.array(img_pil.convert('L')) # Convert to grayscale numpy array
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "unknown", 0.0

    # Find the largest contour (assuming it's the main object)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    num_vertices = len(approx)
    
    shape = "other"
    confidence = 0.0

    if num_vertices == 3:
        shape = "triangle"
        confidence = 1.0 - (abs(cv2.contourArea(main_contour) - (0.5 * (approx[1][0][0] - approx[0][0][0]) * (approx[2][0][1] - approx[0][0][1]))) / (cv2.contourArea(main_contour) + 1e-6)) # Simple area-based confidence
    elif num_vertices == 4:
        # Check if it's a rectangle/square
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w)/h
        if 0.9 <= aspect_ratio <= 1.1:
            shape = "square"
        else:
            shape = "quadrilateral"
        confidence = 1.0 - (abs(cv2.contourArea(main_contour) - (w*h)) / (cv2.contourArea(main_contour) + 1e-6))
    elif num_vertices > 4:
        # Check circularity
        area = cv2.contourArea(main_contour)
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        circle_area = np.pi * (radius ** 2)
        if circle_area > 0:
            circularity = area / circle_area
            if circularity > 0.8: # Threshold for circularity
                shape = "circle"
                confidence = circularity
            else:
                shape = "polygon" # Generic polygon
                confidence = 0.5 # Lower confidence for generic
    
    # Refine confidence based on how well the contour matches its ideal shape
    # This is a very simplified confidence. A real system would use more robust metrics.
    if shape != "other":
        confidence = max(0.1, min(1.0, confidence)) # Clamp between 0.1 and 1.0
    
    logger.debug(f"CV Extracted Shape: {shape} with confidence {confidence:.4f}")
    return shape, confidence

def extract_fill_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts fill type (filled/outlined) and confidence using classical CV.
    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
    Returns:
        Tuple[str, float]: (fill_type, confidence_score)
    """
    gray = np.array(img_pil.convert('L'))
    
    # Calculate mean pixel value of the object area (assuming object is dark on light background)
    # A simple way: find the object contour, then calculate mean inside it.
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "unknown", 0.0
    
    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [main_contour], -1, 255, cv2.FILLED)
    
    # Calculate mean pixel value within the masked area
    masked_pixels = gray[mask == 255]
    if masked_pixels.size == 0:
        return "unknown", 0.0

    mean_val = np.mean(masked_pixels)
    
    fill_type = "outlined"
    confidence = 0.0
    
    # Assuming dark objects on light background:
    # A low mean value indicates a filled (dark) object.
    # A high mean value (close to background) indicates an outlined object.
    if mean_val < 100: # Threshold for "filled" (dark interior)
        fill_type = "solid"
        confidence = 1.0 - (mean_val / 100.0) # Higher confidence for darker
    elif mean_val > 150: # Threshold for "outlined" (light interior)
        fill_type = "outlined"
        confidence = (mean_val - 150.0) / 105.0 # Higher confidence for lighter
    else:
        fill_type = "unknown_fill" # Ambiguous
        confidence = 0.0
    
    confidence = max(0.0, min(1.0, confidence)) # Clamp confidence
    
    logger.debug(f"CV Extracted Fill: {fill_type} with confidence {confidence:.4f}")
    return fill_type, confidence

def extract_cnn_features(img_pil: Image.Image) -> Dict[str, Tuple[str, float]]:
    """
    Extracts features (shape, color, fill, etc.) and their confidences
    using a pre-trained CNN model (PerceptionModule).
    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
    Returns:
        Dict[str, Tuple[str, float]]: A dictionary mapping feature type (e.g., 'shape')
                                      to a tuple of (predicted_value_name, confidence).
    """
    model = _load_cnn_model()
    
    # Preprocess image for the CNN
    input_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE) # Add batch dim, move to device

    with torch.no_grad():
        # Assuming PerceptionModule's forward returns a dictionary of logits
        # for different attributes, and potentially a main classification output.
        # Example: {'shape_logits': tensor, 'color_logits': tensor, ...}
        model_output = model(input_tensor)
        
        extracted_features: Dict[str, Tuple[str, float]] = {}

        # Process shape
        if 'shape_logits' in model_output:
            shape_logits = model_output['shape_logits']
            shape_probs = torch.softmax(shape_logits, dim=1).squeeze(0)
            conf, idx = shape_probs.max(0)
            predicted_shape_idx = idx.item()
            predicted_shape_name = YOLO_CLASS_MAP.get(predicted_shape_idx, "unknown_shape")
            extracted_features['shape'] = (predicted_shape_name, conf.item())
            logger.debug(f"CNN Extracted Shape: {predicted_shape_name} with confidence {conf.item():.4f}")
        else:
            logger.warning("Shape logits not found in CNN model output.")
            extracted_features['shape'] = ("unknown_shape", 0.0)

        # Process fill (example, assuming a 'fill_logits' output)
        if 'fill_logits' in model_output:
            fill_logits = model_output['fill_logits']
            fill_probs = torch.softmax(fill_logits, dim=1).squeeze(0)
            conf, idx = fill_probs.max(0)
            # You need a mapping for fill indices to names, similar to YOLO_CLASS_MAP
            # Example: FILL_MAP = {0: 'solid', 1: 'outlined'}
            FILL_MAP = {0: 'solid', 1: 'outlined', 2: 'striped', 3: 'dotted'} # Example from config.py
            predicted_fill_name = FILL_MAP.get(idx.item(), "unknown_fill")
            extracted_features['fill'] = (predicted_fill_name, conf.item())
            logger.debug(f"CNN Extracted Fill: {predicted_fill_name} with confidence {conf.item():.4f}")
        else:
            extracted_features['fill'] = ("unknown_fill", 0.0)

        # Add more features (color, size, orientation, texture, etc.)
        # This requires your PerceptionModule to output logits for these attributes
        # and corresponding mapping dictionaries (e.g., ATTRIBUTE_COLOR_MAP_INV).
        # For demonstration, let's add a dummy color.
        extracted_features['color'] = (random.choice(['red', 'blue', 'green']), random.random()) # Dummy color
        extracted_features['size'] = (random.choice(['small', 'medium', 'large']), random.random()) # Dummy size
        extracted_features['position_h'] = (random.choice(['left', 'center_h', 'right']), random.random()) # Dummy pos
        extracted_features['position_v'] = (random.choice(['top', 'center_v', 'bottom']), random.random()) # Dummy pos
        extracted_features['orientation'] = (random.choice(['upright', 'rotated_90']), random.random()) # Dummy orient
        extracted_features['texture'] = (random.choice(['flat', 'rough']), random.random()) # Dummy texture

    return extracted_features

if __name__ == '__main__':
    logger.info("Running primitive_extractor.py example.")

    # Create a dummy image for testing
    img_size = 224
    dummy_img_pil = Image.new('RGB', (img_size, img_size), (200, 200, 200)) # Gray background
    draw = ImageDraw.Draw(dummy_img_pil)
    
    # Draw a filled red triangle
    pts_triangle = [(112, 50), (62, 150), (162, 150)]
    draw.polygon(pts_triangle, fill=(255, 0, 0), outline=(0, 0, 0))

    # Test classical CV extractors
    shape, shape_conf = extract_shape_conf(dummy_img_pil)
    fill, fill_conf = extract_fill_conf(dummy_img_pil)
    logger.info(f"CV Extraction: Shape='{shape}' ({shape_conf:.4f}), Fill='{fill}' ({fill_conf:.4f})")

    # Test CNN feature extractor (will use dummy model if real not loaded)
    cnn_features = extract_cnn_features(dummy_img_pil)
    logger.info(f"CNN Extraction Results:")
    for feat_type, (value, conf) in cnn_features.items():
        logger.info(f"  {feat_type}: '{value}' ({conf:.4f})")

