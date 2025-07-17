
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --- Import config and attribute maps first ---
try:
    from config import (
        ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP,
        ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP,
        CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE
    )
except ImportError as e:
    print(f"Error importing config: {e}")
    raise

# --- Inverse attribute maps for index-to-name mapping ---
def _invert_map(m):
    return {v: k for k, v in m.items()}

attribute_maps_inv = {
    'shape': _invert_map(ATTRIBUTE_SHAPE_MAP),
    'color': _invert_map(ATTRIBUTE_COLOR_MAP),
    'fill': _invert_map(ATTRIBUTE_FILL_MAP),
    'size': _invert_map(ATTRIBUTE_SIZE_MAP),
    'orientation': _invert_map(ATTRIBUTE_ORIENTATION_MAP),
    'texture': _invert_map(ATTRIBUTE_TEXTURE_MAP),
}


# --- Expose a global MODEL for import (after all dependencies are defined) ---
import torch
try:
    from core_models.models import BongardPerceptionModel
    from core_models.training_args import Config
    _config = Config()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = None
    DEVICE = _device
except ImportError as e:
    print(f"Error importing core_models: {e}")
    print("Make sure the src directory is in your Python path")
    raise

def load_model():
    global MODEL
    _model = BongardPerceptionModel().to(_device)
    _model.load_state_dict(torch.load(_config.best_model_path, map_location=_device))
    _model.eval()
    MODEL = _model
# Folder: bongard_solver/src/perception/
# File: primitive_extractor.py

import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw # Added ImageDraw for dummy image creation
import torch
import torch.nn as nn
import torchvision.transforms as T
from typing import Tuple, Dict, Any, Optional, List
from collections import Counter  # For TTA majority vote
import random
import collections

###############################
# Logger Setup
###############################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
 # Import global configuration and model components


from config import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE
from config import ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP
from config import ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP
from core_models.models import BongardPerceptionModel, PerceptionModule   # Import both
try:
    from utils.augment import augment_image  # Import augment_image for TTA
except ImportError:
    logger.warning("utils.augment not found, TTA functionality may be limited")

# --- Preprocessing transform for CNN input ---
preprocess_transform = T.Compose([
    T.Resize(tuple(CONFIG['data']['image_size']) if 'data' in CONFIG and 'image_size' in CONFIG['data'] else (224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])



def extract_cnn_feature(img: Image.Image):
    # Test-Time Augmentation: original + one augment
    crops = [img]
    try:
        from utils.augment import augment_image
        aug = augment_image(img)
        if isinstance(aug, torch.Tensor):
            # Denormalize and convert to PIL
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            aug = aug * std + mean
            aug = torch.clamp(aug * 255, 0, 255).byte()
            aug_img = Image.fromarray(aug.permute(1,2,0).cpu().numpy())
            crops.append(aug_img)
        elif isinstance(aug, Image.Image):
            crops.append(aug)
    except Exception:
        pass
    votes, confs = [], []
    for c in crops:
        result = single_inference(c)
        # Prefer 'bongard' if present, else first key
        if 'bongard' in result:
            v, c_conf = result['bongard']
        else:
            first_attr = next(iter(result))
            v, c_conf = result[first_attr]
        votes.append(v)
        confs.append(c_conf)
    from collections import Counter
    val = Counter(votes).most_common(1)[0][0]
    avg_conf = sum(c for v, c in zip(votes, confs) if v == val) / votes.count(val)
    return val, avg_conf




def single_inference(img_pil: Image.Image, model=None) -> Dict[str, Tuple[str, float]]:
    """
    Performs a single inference pass on a cropped object image using the CNN model.
    This function is a helper for extract_cnn_features to enable TTA.
    """
    if model is None:
        # Always instantiate the real model directly
        model = PerceptionModule(CONFIG).to(DEVICE)
        model.eval()
        # Optionally, load weights here if needed
    input_tensor = preprocess_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dummy_gts_json_strings = [b'{}']
        dummy_bboxes_batch = [[]]
        dummy_masks_batch = [[]]
        model_output = model(
            images=input_tensor,
            ground_truth_json_strings=dummy_gts_json_strings,
            detected_bboxes_batch=dummy_bboxes_batch,
            detected_masks_batch=dummy_masks_batch,
            support_images=None,
            support_labels_flat=None,
            is_simclr_pretraining=False
        )
        attribute_logits_dict = model_output.get('attribute_logits', {})
        extracted_features: Dict[str, Tuple[str, float]] = {}
        for attr_type, logits in attribute_logits_dict.items():
            if logits.numel() == 0:
                extracted_features[attr_type] = (f"unknown_{attr_type}", 0.0)
                continue
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf, idx = probs.max(0)
            predicted_idx = idx.item()
            attr_name = attribute_maps_inv.get(attr_type, {}).get(predicted_idx, f"unknown_{attr_type}")
            extracted_features[attr_type] = (attr_name, conf.item())
    return extracted_features

def extract_shape_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts shape and confidence using classical computer vision.
    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
    Returns:
        Tuple[str, float]: (shape_name, confidence_score)
    """
    if img_pil.mode != 'L':   # Convert to grayscale if not already
        gray = np.array(img_pil.convert('L'))
    else:
        gray = np.array(img_pil)
    
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
    
    # Calculate area and perimeter for more robust shape analysis
    area = cv2.contourArea(main_contour)
    perimeter = cv2.arcLength(main_contour, True)
    
    if num_vertices == 3:
        shape = "triangle"
        # Confidence based on area ratio to convex hull or ideal triangle
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            confidence = area / hull_area   # How compact is the triangle
        else:
            confidence = 0.5   # Default if area is zero
    elif num_vertices == 4:
        # Check if it's a rectangle/square
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w)/h
        rect_area = w * h
        
        if 0.8 <= aspect_ratio <= 1.2 and rect_area > 0:   # Check aspect ratio for square-like
            shape = "square"
            confidence = area / rect_area   # How much of the bounding rect is filled
        else:
            shape = "quadrilateral"
            confidence = 0.7   # General quadrilateral, lower confidence
    elif num_vertices > 4:
        # Check circularity
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        circle_area = np.pi * (radius ** 2)
        if circle_area > 0:
            circularity = area / circle_area
            if circularity > 0.8:   # Threshold for circularity
                shape = "circle"
                confidence = circularity
            else:
                shape = "polygon"   # Generic polygon
                confidence = 0.5   # Lower confidence for generic
    
    # Ensure confidence is within [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    logger.debug(f"CV Extracted Shape: {shape} with confidence {confidence:.4f}")
    return shape, confidence

def extract_fill_conf(img_pil: Image.Image) -> Tuple[str, float]:
    """
    Extracts fill type (filled/outlined/striped/dotted) and confidence using classical CV.
    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
    Returns:
        Tuple[str, float]: (fill_type, confidence_score)
    """
    if img_pil.mode != 'L':
        gray = np.array(img_pil.convert('L'))
    else:
        gray = np.array(img_pil)
    
    # Apply thresholding to get a binary mask of the object
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
    
    # Analyze internal texture for striped/dotted
    # Erode the object to get the "inner" part, then check variance or patterns
    kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    inner_pixels = gray[eroded_mask == 255]
    
    fill_type = "unknown_fill"
    confidence = 0.0
    
    if inner_pixels.size > 0:
        std_dev = np.std(inner_pixels)
        
        # Heuristic for fill type:
        # Solid: low std dev, low mean (dark object)
        # Outlined: high mean (light interior), possibly higher std dev at edges
        # Striped/Dotted: high std dev (variation in pixel values)
        
        if mean_val < 100 and std_dev < 30:   # Dark and uniform
            fill_type = "solid"
            confidence = 1.0 - (mean_val / 100.0)   # Higher confidence for darker solid
        elif mean_val > 150 and std_dev < 30:   # Light and uniform (likely background showing through outline)
            fill_type = "outlined"
            confidence = (mean_val - 150.0) / 105.0   # Higher confidence for lighter outlined
        elif std_dev > 50:   # High variance suggests pattern
            # Further analysis needed to distinguish striped vs dotted
            # For simplicity, let's just use a generic "patterned" for now
            # A more advanced method would use FFT or Hough lines for stripes, blob detection for dots
            fill_type = "striped" if random.random() > 0.5 else "dotted"   # Randomly assign for demo
            confidence = min(1.0, std_dev / 100.0)   # Scale std dev to confidence
        
    confidence = max(0.0, min(1.0, confidence))   # Clamp confidence
    
    logger.debug(f"CV Extracted Fill: {fill_type} with confidence {confidence:.4f}")
    return fill_type, confidence

def extract_cnn_features(img_pil: Image.Image) -> Dict[str, Tuple[str, float]]:
    """
    Extracts features (shape, color, fill, etc.) and their confidences
    using a pre-trained CNN model (PerceptionModule) with Test-Time Augmentation (TTA).
    Args:
        img_pil (PIL.Image.Image): Cropped image of a single object.
    Returns:
        Dict[str, Tuple[str, float]]: A dictionary mapping feature type (e.g., 'shape')
                                      to a tuple of (predicted_value_name, confidence).
    """
    # Test-time augmentations
    # augment_image expects PIL Image and returns a PyTorch tensor (C, H, W)
    # We need to convert this tensor back to a PIL Image for single_inference if it expects PIL.
    # single_inference expects PIL Image, so we convert augmented tensor back to PIL.
    # Note: augment_image returns a normalized tensor. We need to denormalize for PIL.
    
    # Original image (PIL)
    crops_pil = [img_pil]
    
    # Augmented image (convert to numpy, augment, then convert back to PIL for consistency)
    # The augment_image function from utils.augment.py returns a normalized torch.Tensor
    # We need to reverse the normalization and convert to PIL for single_inference if it expects PIL.
    # Or, modify single_inference to accept a tensor. Let's modify single_inference for simplicity.
    
    # Let's assume augment_image returns a PIL Image for this context, or handle conversion here.
    # Re-reading utils/augment.py, `augment_image` returns a `torch.Tensor`.
    # So, we need to adapt `single_inference` to take a `torch.Tensor` or convert here.
    # The snippet implies `single_inference` takes PIL, so convert back from tensor.
    
    # Create an augmented version as a PIL Image
    # This requires denormalizing the tensor and converting to numpy then PIL
    
    # Convert original PIL to tensor for augmentation pipeline, then augment
    img_np_orig = np.array(img_pil)
    augmented_tensor = augment_image(Image.fromarray(img_np_orig)) # augment_image expects PIL
    
    # Denormalize the augmented tensor and convert to PIL Image
    # This assumes IMAGENET_MEAN and IMAGENET_STD are available globally or passed.
    # If using the default augment.py, the tensor is normalized.
    # To convert back to PIL (0-255 range), we need to reverse normalization.
    
    # Denormalization logic (assuming 3 channels, float tensor)
    mean_tensor = torch.tensor(IMAGENET_MEAN, device=DEVICE).view(3, 1, 1)
    std_tensor = torch.tensor(IMAGENET_STD, device=DEVICE).view(3, 1, 1)
    
    # Ensure augmented_tensor is on the correct device for operations
    augmented_tensor_denorm = augmented_tensor.to(DEVICE) * std_tensor + mean_tensor
    augmented_tensor_denorm = torch.clamp(augmented_tensor_denorm * 255, 0, 255).byte() # Scale to 0-255, convert to byte
    
    # Convert back to PIL Image (CHW to HWC for numpy, then PIL)
    augmented_img_pil = Image.fromarray(augmented_tensor_denorm.permute(1, 2, 0).cpu().numpy())
    
    crops_pil.append(augmented_img_pil)
    
    # Dictionary to store votes and confidences for each attribute type
    # { 'shape': [val1, val2], 'color': [val1, val2], ... }
    all_votes: Dict[str, List[str]] = collections.defaultdict(list)
    all_confs: Dict[str, List[float]] = collections.defaultdict(list)
    
    for c in crops_pil: # Iterate through PIL images
        # Perform single inference for each crop
        inference_results = single_inference(c) # single_inference expects PIL
        for attr_type, (val, conf) in inference_results.items():
            all_votes[attr_type].append(val)
            all_confs[attr_type].append(conf)
            
    final_extracted_features: Dict[str, Tuple[str, float]] = {}
    for attr_type in all_votes.keys():
        votes = all_votes[attr_type]
        confs = all_confs[attr_type]
        
        if not votes:
            final_extracted_features[attr_type] = (f"unknown_{attr_type}", 0.0)
            continue
        
        # Majority vote on values
        most_common_val = Counter(votes).most_common(1)[0][0]
        
        # Average confidence among those votes that match the majority value
        matching_confs = [c for v, c in zip(votes, confs) if v == most_common_val]
        avg_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.0
        
        final_extracted_features[attr_type] = (most_common_val, avg_conf)
        logger.debug(f"CNN Extracted {attr_type.capitalize()} (TTA): {most_common_val} with confidence {avg_conf:.4f}")
    
    return final_extracted_features



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running primitive_extractor.py example.")
    
    # Create a dummy image for testing
    img_size = 224
    dummy_img_pil = Image.new('RGB', (img_size, img_size), (200, 200, 200))   # Gray background
    draw = ImageDraw.Draw(dummy_img_pil)
    
    # Draw a filled red triangle
    pts_triangle = [(112, 50), (62, 150), (162, 150)]
    draw.polygon(pts_triangle, fill=(255, 0, 0), outline=(0, 0, 0))

    # Test classical CV extractors
    logger.info("\n--- Testing Classical CV Extraction ---")
    shape, shape_conf = extract_shape_conf(dummy_img_pil)
    fill, fill_conf = extract_fill_conf(dummy_img_pil)
    logger.info(f"CV Extraction: Shape='{shape}' ({shape_conf:.4f}), Fill='{fill}' ({fill_conf:.4f})")

    # Test CNN feature extractor with TTA (will use dummy model if real not loaded)
    logger.info("\n--- Testing CNN Feature Extraction with TTA ---")
    cnn_features = extract_cnn_features(dummy_img_pil)
    logger.info(f"CNN Extraction Results (with TTA):")
    for feat_type, (value, conf) in cnn_features.items():
        logger.info(f"   {feat_type}: '{value}' ({conf:.4f})")

    # Draw an outlined blue circle
    dummy_img_pil_circle = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw_circle = ImageDraw.Draw(dummy_img_pil_circle)
    draw_circle.ellipse([50, 50, 150, 150], outline=(0, 0, 255), width=5)   # Outlined blue circle

    logger.info("\n--- Testing Classical CV Extraction (Circle) ---")
    shape_c, shape_conf_c = extract_shape_conf(dummy_img_pil_circle)
    fill_c, fill_conf_c = extract_fill_conf(dummy_img_pil_circle)
    logger.info(f"CV Extraction: Shape='{shape_c}' ({shape_conf_c:.4f}), Fill='{fill_c}' ({fill_conf_c:.4f})")

    logger.info("\n--- Testing CNN Feature Extraction (Circle) with TTA ---")
    cnn_features_c = extract_cnn_features(dummy_img_pil_circle)
    logger.info(f"CNN Extraction Results (with TTA):")
    for feat_type, (value, conf) in cnn_features_c.items():
        logger.info(f"   {feat_type}: '{value}' ({conf:.4f})")
