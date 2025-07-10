import os
import random
import json
import logging
import numpy as np
from pathlib import Path
from collections import Counter
import math # For math.ceil

import cv2
import yaml
import torch
from perlin_noise import PerlinNoise

# --- DALI Imports ---
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    HAS_DALI = True
    dali_logger = logging.getLogger("dali_logger")
    dali_logger.setLevel(logging.INFO)
    if not dali_logger.handlers:
        dali_logger.addHandler(logging.StreamHandler())
    dali_logger.info("NVIDIA DALI found and imported.")
except ImportError:
    HAS_DALI = False
    dali_logger = logging.getLogger("dali_logger")
    dali_logger.setLevel(logging.WARNING)
    if not dali_logger.handlers:
        dali_logger.addHandler(logging.StreamHandler())
    dali_logger.warning("NVIDIA DALI not found. Falling back to PyTorch DataLoader only if implemented.")
# --------------------

# Configure main logger for my_data_utils
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- YOLO Classes and IDs ---
YOLO_CLASSES = {0: 'object', 1: 'shape', 2: 'relation'} 
CLASS_ID = {name: idx for idx, name in YOLO_CLASSES.items()}


# --- Helper Functions (Full Implementations) ---

def detect_contours(image_gray, block_size, C, min_area, max_contours_to_find, morphological_ops=None):
    """
    Detects contours in a grayscale image using adaptive thresholding,
    with optional morphological cleanup.
    Args:
        image_gray (np.array): Grayscale image (H, W).
        block_size (int): Size of a pixel neighborhood that is used to calculate a threshold value. (Odd number, e.g., 11)
        C (int): Constant subtracted from the mean or weighted mean. (e.g., 2)
        min_area (int): Minimum contour area to be considered.
        max_contours_to_find (int): Maximum number of contours to return, sorted by area.
        morphological_ops (str, optional): Type of morphological operation ('open', 'close', None).
    Returns:
        list: A list of detected contours (OpenCV format).
    """
    if image_gray is None or image_gray.size == 0:
        logger.warning("Input image for contour detection is empty or None.")
        return []

    if block_size % 2 == 0:
        block_size += 1

    thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, C)
    
    # NEW: Optional morphological operations
    if morphological_ops:
        kernel = np.ones((3,3), np.uint8) # Small kernel for cleanup
        if morphological_ops == 'open':
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        elif morphological_ops == 'close':
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:max_contours_to_find]

    return filtered_contours

def cnt_to_yolo(contour, img_width, img_height, min_contour_area=0):
    """
    Converts an OpenCV contour to YOLO bounding box format (class_id center_x center_y width height).
    Args:
        contour (np.array): An OpenCV contour.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.
        min_contour_area (int): Minimum contour area to process.
    Returns:
        str: YOLO format string "class_id center_x center_y width height" or None if invalid.
    """
    if cv2.contourArea(contour) < min_contour_area:
        return None

    x, y, w, h = cv2.boundingRect(contour)

    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height

    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 <= norm_width <= 1 and 0 <= norm_height <= 1):
        logger.warning(f"Invalid YOLO box coordinates calculated for contour: ({center_x}, {center_y}, {norm_width}, {norm_height}). Skipping.")
        return None

    class_id = CLASS_ID.get('object', 0) 

    return f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"


def extract_attrs(contour, image_color, img_size_categories=(0.01, 0.05, 0.15)):
    """
    Extracts attributes from a contour, including shape type, size category, and orientation.
    Args:
        contour (np.array): An OpenCV contour.
        image_color (np.array): The original color image (H, W, 3).
        img_size_categories (tuple): Area thresholds for 'small', 'medium', 'large' relative to total image area.
    Returns:
        dict: Dictionary of extracted attributes.
    """
    if contour is None or image_color is None:
        return {}

    attrs = {}
    
    x, y, w, h = cv2.boundingRect(contour)
    attrs['bbox'] = [x, y, w, h]
    
    area = cv2.contourArea(contour)
    attrs['area'] = area
    
    if h > 0:
        attrs['aspect_ratio'] = w / h
    else:
        attrs['aspect_ratio'] = 0.0

    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        attrs['centroid'] = [cX, cY]
    else:
        attrs['centroid'] = [0, 0]

    mask = np.zeros(image_color.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    mean_color = cv2.mean(image_color, mask=mask)[:3]
    attrs['color_bgr'] = [int(c) for c in mean_color]

    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * attrs['area'] / (perimeter * perimeter)
        attrs['circularity'] = circularity
    else:
        attrs['circularity'] = 0.0
    
    # NEW: Shape type detection
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)
    attrs['num_vertices'] = num_vertices

    if num_vertices == 3:
        attrs['shape_type'] = 'triangle'
    elif num_vertices == 4:
        # Check for rectangle-like shapes (aspect ratio close to 1, angles close to 90)
        # This is a simplification; a more robust check would involve angle analysis
        if 0.8 <= attrs['aspect_ratio'] <= 1.2 and attrs['circularity'] > 0.6: # High circularity for square-like
             attrs['shape_type'] = 'square'
        elif 0.5 <= attrs['aspect_ratio'] <= 2.0:
            attrs['shape_type'] = 'rectangle'
        else:
            attrs['shape_type'] = 'quadrilateral'
    elif num_vertices > 4 and attrs['circularity'] > 0.8: # High circularity for circle-like
        attrs['shape_type'] = 'circle'
    elif num_vertices > 4:
        attrs['shape_type'] = 'polygon'
    else:
        attrs['shape_type'] = 'other' # For complex or irregular shapes

    # NEW: Size category
    total_image_area = image_color.shape[0] * image_color.shape[1]
    relative_area = area / total_image_area if total_image_area > 0 else 0
    if relative_area < img_size_categories[0]:
        attrs['size_category'] = 'tiny'
    elif relative_area < img_size_categories[1]:
        attrs['size_category'] = 'small'
    elif relative_area < img_size_categories[2]:
        attrs['size_category'] = 'medium'
    else:
        attrs['size_category'] = 'large'

    # NEW: Orientation (angle of minimum area rectangle)
    if len(contour) >= 5: # minAreaRect requires at least 5 points
        rect = cv2.minAreaRect(contour)
        attrs['orientation'] = rect[2] # Angle in degrees [-90, 0)
    else:
        attrs['orientation'] = 0.0 # Default if not enough points

    # NEW: Complexity (perimeter^2 / area ratio, higher for more complex)
    if area > 0:
        attrs['complexity'] = (perimeter * perimeter) / area
    else:
        attrs['complexity'] = 0.0

    # Fill type (conceptual: requires more info than just contour)
    # For Bongard, if it's binary, it's either filled or not.
    # If the contour is detected as a solid shape, it's 'filled'.
    # If it's just an outline, it's 'outline'.
    # This is hard to determine robustly from just the contour.
    # For now, we assume if a contour is detected, it represents a 'filled' object
    # from the thresholding process.
    # attrs['fill_type'] = 'filled' # Placeholder

    # Stroke width (conceptual: requires original image pixel analysis or drawing info)
    # attrs['stroke_width'] = 'N/A' # Placeholder

    return attrs

def compute_relations(attributes_list):
    """
    Computes simple spatial relations between detected objects, including overlaps and contains.
    Args:
        attributes_list (list): A list of attribute dictionaries for each object.
    Returns:
        list: A list of dictionaries, each representing a relation.
    """
    relations = []
    num_objects = len(attributes_list)

    if num_objects < 2:
        return relations

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1_attrs = attributes_list[i]
            obj2_attrs = attributes_list[j]

            # Bounding box coordinates for overlap/contains checks
            bbox1 = obj1_attrs.get('bbox')
            bbox2 = obj2_attrs.get('bbox')

            if 'centroid' in obj1_attrs and 'centroid' in obj2_attrs:
                cx1, cy1 = obj1_attrs['centroid']
                cx2, cy2 = obj2_attrs['centroid']

                # Positional relations
                if cx1 < cx2 - 10:
                    relations.append({'type': 'left_of', 'object1_idx': i, 'object2_idx': j})
                elif cx1 > cx2 + 10:
                    relations.append({'type': 'right_of', 'object1_idx': i, 'object2_idx': j})
                else:
                    relations.append({'type': 'aligned_horizontally', 'object1_idx': i, 'object2_idx': j})

                if cy1 < cy2 - 10:
                    relations.append({'type': 'above', 'object1_idx': i, 'object2_idx': j})
                elif cy1 > cy2 + 10:
                    relations.append({'type': 'below', 'object1_idx': i, 'object2_idx': j})
                else:
                    relations.append({'type': 'aligned_vertically', 'object1_idx': i, 'object2_idx': j})
            
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if bbox1 and bbox2:
                    # Approximation for "is_close_to" based on bounding box diagonal
                    diag1 = np.sqrt(bbox1[2]**2 + bbox1[3]**2)
                    diag2 = np.sqrt(bbox2[2]**2 + bbox2[3]**2)
                    if dist < (diag1 + diag2) / 2 * 0.75: # If centroids are closer than 75% of average diagonal
                        relations.append({'type': 'is_close_to', 'object1_idx': i, 'object2_idx': j})
            
            # NEW: Overlap and Contains relations
            if bbox1 and bbox2:
                x1, y1, w1, h1 = bbox1
                x2, y2, w2, h2 = bbox2

                # Calculate intersection rectangle
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                intersection_area = x_overlap * y_overlap
                area1 = w1 * h1
                area2 = w2 * h2

                if intersection_area > 0:
                    relations.append({'type': 'overlaps', 'object1_idx': i, 'object2_idx': j, 'overlap_area': intersection_area})

                    # Check for 'contains'
                    # Object 1 contains Object 2
                    if x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2):
                        relations.append({'type': 'contains', 'object1_idx': i, 'object2_idx': j})
                    # Object 2 contains Object 1
                    elif x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                        relations.append({'type': 'contains', 'object1_idx': j, 'object2_idx': i})

    return relations

def random_bg(img_np, bg_paths):
    if not bg_paths:
        logger.warning("No background images found for random_bg. Returning original image.")
        return img_np

    if img_np is None or img_np.size == 0:
        logger.warning("Input image for random_bg is empty or None. Returning empty array.")
        return img_np

    bg_path = random.choice(bg_paths)
    try:
        bg = cv2.imread(str(bg_path))
        if bg is None:
            logger.warning(f"Could not read background image: {bg_path}. Returning original image.")
            return img_np
        
        bg = cv2.resize(bg, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        if bg.shape[-1] == 1:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        
        if img_np.shape[-1] == 1 and bg.shape[-1] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        gray_fg = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if img_np.ndim == 3 else img_np
        _, mask = cv2.threshold(gray_fg, 1, 255, cv2.THRESH_BINARY)
        
        mask_inv = cv2.bitwise_not(mask)

        img_fg = cv2.bitwise_and(img_np, img_np, mask=mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)

        combined_img = cv2.add(img_fg, bg_part)
        return combined_img

    except Exception as e:
        logger.error(f"Error applying random background from {bg_path}: {e}. Returning original image.")
        return img_np

def occlude(img, occlude_p, occlude_max_factor):
    """
    Applies a random rectangular occlusion to the image.
    Args:
        img (np.array): Image to occlude (H, W, C).
        occlude_p (float): Probability of applying occlusion.
        occlude_max_factor (float): Maximum fraction of image dimension for occlusion.
    Returns:
        np.array: Occluded image.
    """
    # occlude_p is handled by the calling function (process_batch_of_images)
    # This function just applies the occlusion if called.

    h, w, _ = img.shape
    mf = occlude_max_factor

    # Calculate random patch dimensions
    ph = int(h * random.uniform(0.05, mf)) # Min 5% to avoid tiny patches
    pw = int(w * random.uniform(0.05, mf))

    if ph <= 0 or pw <= 0:
        logger.debug("Occlusion patch dimensions too small. Skipping occlusion.")
        return img.copy()

    # Calculate random top-left corner for the patch
    y = random.randint(0, h - ph) if h - ph > 0 else 0
    x = random.randint(0, w - pw) if w - pw > 0 else 0

    occluded_img = img.copy()
    # Fill the patch with random noise (or solid color, e.g., black)
    occluded_img[y:y+ph, x:x+pw] = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)

    return occluded_img

def add_clutter(img, num_clutter_patches, clutter_max_factor):
    """
    Adds random clutter patches (small shapes or noise) to the image.
    Args:
        img (np.array): Image to add clutter to (H, W, C).
        num_clutter_patches (int): Number of clutter patches to add.
        clutter_max_factor (float): Max relative size of a clutter patch.
    Returns:
        np.array: Image with clutter.
    """
    cluttered_img = img.copy()
    h, w, _ = img.shape

    for _ in range(num_clutter_patches):
        patch_h = int(h * random.uniform(0.02, clutter_max_factor))
        patch_w = int(w * random.uniform(0.02, clutter_max_factor))

        if patch_h <= 0 or patch_w <= 0:
            continue

        y = random.randint(0, h - patch_h) if h - patch_h > 0 else 0
        x = random.randint(0, w - patch_w) if w - patch_w > 0 else 0  # FIXED: Changed pw to patch_w

        # Randomly choose between noise or simple shape
        if random.random() < 0.5: # Add random noise patch
            cluttered_img[y:y+patch_h, x:x+patch_w] = np.random.randint(0, 255, (patch_h, patch_w, 3), dtype=np.uint8)
        else: # Add a simple shape (circle or rectangle)
            shape_type = random.choice(['circle', 'rectangle'])
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            
            if shape_type == 'circle':
                center = (x + patch_w // 2, y + patch_h // 2)
                radius = min(patch_w, patch_h) // 2
                cv2.circle(cluttered_img, center, radius, color, -1) # -1 for filled
            else: # rectangle
                cv2.rectangle(cluttered_img, (x, y), (x + patch_w, y + patch_h), color, -1)
    
    return cluttered_img


def get_cached_fractal(fractal_type, cache_dir, size, noise_params):
    """
    Generates or loads a cached fractal image (e.g., Perlin noise, Sierpinski, Koch).
    Args:
        fractal_type (str): Type of fractal (e.g., 'perlin', 'sierpinski', 'koch').
        cache_dir (str): Directory to store/load cached fractals.
        size (tuple): (height, width) of the fractal image.
        noise_params (dict): Parameters specific to fractal generation (e.g., octaves, level, seed).
    Returns:
        np.array: Fractal image (grayscale or color, uint8).
    """
    # Ensure seed is part of the cache key for Perlin and other random fractals
    # For Sierpinski/Koch, level is the key parameter.
    seed_key = noise_params.get('seed', 0)
    level_key = noise_params.get('level', 0)

    if fractal_type == 'perlin':
        cache_filename = f"{fractal_type}_{size[0]}x{size[1]}_oct{noise_params.get('octaves',4)}_seed{seed_key}.png"
    elif fractal_type in ['sierpinski', 'koch']:
        cache_filename = f"{fractal_type}_{size[0]}x{size[1]}_level{level_key}_seed{seed_key}.png" # Added seed for consistency
    else:
        cache_filename = f"{fractal_type}_{size[0]}x{size[1]}_seed{seed_key}.png" # Generic fallback

    cache_path = Path(cache_dir) / cache_filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        img = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 4: 
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.ndim == 2: # Ensure 3 channels if it's grayscale for blending
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            logger.debug(f"Loaded cached fractal: {cache_path}")
            return img
        else:
            logger.warning(f"Failed to read cached fractal {cache_path}. Regenerating.")

    logger.info(f"Generating new fractal: {fractal_type} (size={size}, params={noise_params})...")
    generated_img = None

    if fractal_type == 'perlin':
        noise = PerlinNoise(
            octaves=noise_params.get('octaves', 4),
            seed=seed_key # Use the provided seed
        )
        img_array = np.array([[noise([i/size[0], j/size[1]]) for j in range(size[1])] for i in range(size[0])])
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) 
        generated_img = (img_array * 255).astype(np.uint8) # Grayscale
        generated_img = cv2.cvtColor(generated_img, cv2.COLOR_GRAY2BGR) # Convert to BGR for consistency

    elif fractal_type == 'sierpinski':
        generated_img = draw_sierpinski(img_size=size, level=noise_params.get('level', 4), color=(255, 255, 255)) 

    elif fractal_type == 'koch':
        generated_img = draw_koch(img_size=size, level=noise_params.get('level', 3), color=(255, 255, 255)) 

    else:
        logger.warning(f"Unknown fractal type: {fractal_type}. Returning black image.")
        return np.zeros((size[0], size[1], 3), dtype=np.uint8) # Return 3-channel black image

    if generated_img is not None:
        cv2.imwrite(str(cache_path), generated_img)
        logger.info(f"Cached generated fractal to: {cache_path}")
        return generated_img
    else:
        logger.error(f"Failed to generate {fractal_type} image.")
        return np.zeros((size[0], size[1], 3), dtype=np.uint8) # Return 3-channel black image on failure

def draw_sierpinski(img_size=(224, 224), level=4, color=(255, 255, 255)):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    points = np.array([
        [img_size[1] // 2, 0],
        [0, img_size[0] - 1],
        [img_size[1] - 1, img_size[0] - 1]
    ], np.int32)
    
    cv2.fillPoly(img, [points], color)

    def _draw_recursive(p1, p2, p3, level_left):
        if level_left == 0:
            return

        mid12 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        mid23 = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
        mid31 = ((p3[0] + p1[0]) // 2, (p3[1] + p1[1]) // 2)

        inverted_triangle_points = np.array([mid12, mid23, mid31], np.int32)
        cv2.fillPoly(img, [inverted_triangle_points], (0, 0, 0)) # Black for "empty" space

        _draw_recursive(p1, mid12, mid31, level_left - 1)
        _draw_recursive(mid12, p2, mid23, level_left - 1)
        _draw_recursive(mid31, mid23, p3, level_left - 1)

    _draw_recursive(points[0], points[1], points[2], level)

    return img

def draw_koch(img_size=(224, 224), level=3, color=(255, 255, 255)):
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    side_length = min(img_size) * 0.7
    height = side_length * (np.sqrt(3) / 2)
    center_x, center_y = img_size[1] // 2, img_size[0] // 2

    p1 = (int(center_x), int(center_y - height / 2))
    p2 = (int(center_x - side_length / 2), int(center_y + height / 2))
    p3 = (int(center_x + side_length / 2), int(center_y + height / 2))

    def koch_curve(p_start, p_end, n):
        if n == 0:
            yield p_start
            yield p_end
            return

        x1, y1 = p_start
        x2, y2 = p_end

        dx, dy = x2 - x1, y2 - y1

        pA = (x1, y1)
        pB = (x1 + dx / 3, y1 + dy / 3)
        pD = (x1 + 2 * dx / 3, y1 + 2 * dy / 3)
        pE = (x2, y2)

        angle = np.deg2rad(-60)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xC = x1 + dx / 2 + (dx * cos_a - dy * sin_a) / np.sqrt(3)
        yC = y1 + dy / 2 + (dx * sin_a + dy * cos_a) / np.sqrt(3)
        pC = (xC, yC)

        yield from koch_curve(pA, pB, n - 1)
        yield from koch_curve(pB, pC, n - 1)
        yield from koch_curve(pC, pD, n - 1)
        yield from koch_curve(pD, pE, n - 1)
    
    snowflake_points = []
    points_side1 = list(koch_curve(p1, p2, level))
    snowflake_points.extend(points_side1[:-1])

    points_side2 = list(koch_curve(p2, p3, level))
    snowflake_points.extend(points_side2[:-1])

    points_side3 = list(koch_curve(p3, p1, level))
    snowflake_points.extend(points_side3)

    snowflake_points_np = np.array(snowflake_points, np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(img, [snowflake_points_np], color)

    return img

# --- End of Helper Functions ---

# --- Python function for clamping and casting to UINT8 ---
def clamp_to_uint8(x: np.ndarray):
    """
    Clips float values to [0, 255] and casts to uint8.
    This is used as a Python function hook in DALI.
    """
    return np.clip(x, 0.0, 255.0).astype(np.uint8)

# --- Python function for JPEG compression/decompression using OpenCV ---
def jpeg_np(x: np.ndarray, quality_range: tuple):
    """
    Performs JPEG compression and decompression using OpenCV.
    Args:
        x (np.ndarray): Input image (HWC, uint8).
        quality_range (tuple): A tuple (min_quality, max_quality) for random selection.
    Returns:
        np.ndarray: JPEG compressed and decompressed image (HWC, uint8).
    """
    q = int(np.random.uniform(*quality_range))
    # Ensure input is uint8 for imencode
    _, buf = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
# ---------------------------------------------------------


if HAS_DALI:
    class BongardDaliPipeline(Pipeline):
        def __init__(self, file_root, file_list, config, device_id):
            super().__init__(
                batch_size=config['batch_size'],
                num_threads=config['dali_num_threads'], 
                device_id=device_id, # This device_id is passed to the DALI backend
                seed=config['seed'],
                py_num_workers=config.get('dali_py_num_workers', 0),
                prefetch_queue_depth=config['dali_prefetch_queue'], 
                exec_async=True, exec_pipelined=True
            )
            self.config    = config
            self.file_root = str(file_root)
            self.file_list = str(file_list)

            # Initialize a counter for debug logging within Python functions
            self.debug_log_limit = 2 # Log only for the first 2 batches
            self.debug_log_counter = 0

            dali_logger.info(f"DALI Pipeline Initialized: batch_size={config['batch_size']}, threads={config['dali_num_threads']}, device_id={device_id}")
            dali_logger.info(f"DALI Reader configured with: file_root='{self.file_root}', file_list='{self.file_list}'")

            # Determine DALI operator device based on config and actual device_id
            # If force_cpu_dali is True, always use CPU for DALI ops
            if config.get('force_cpu_dali', False):
                self.dali_op_device = "cpu"
                self.decode_device = "cpu"
                dali_logger.warning("DALI operations are forced to CPU as per configuration.")
            else:
                self.dali_op_device = "gpu" if device_id != -1 else "cpu"
                self.decode_device = "mixed" if device_id != -1 else "cpu" 

            dali_logger.info(f"[NVML Check] DALI pipeline device_id={self.device_id}, decode device='{self.decode_device}'")


        def define_graph(self):
            images_raw_data, labels = fn.readers.file( 
                file_root     = self.file_root,
                file_list     = self.file_list,
                random_shuffle= True,
                name           = "Reader"
            )
            dali_logger.info(f"[DALI define_graph] Reader outputs (DataNodes): images_raw_data={images_raw_data}, labels={labels}") 
            dali_logger.info(f"[DALI] Reader instantiated with: file_root={self.file_root}, file_list={self.file_list}") 

            decoded_images = fn.decoders.image(
                images_raw_data, 
                device=self.decode_device, 
                output_type=types.RGB
            )
            # dali_logger.info(f"[DALI define_graph] Decoded images (DataNode after decode): {decoded_images}") # Removed per-batch log
            # dali_logger.info(f"[DALI] Applied decoder") # Removed per-batch log
            
            # If decode was on CPU but ops are on GPU, copy to GPU
            if self.decode_device == "cpu" and self.dali_op_device == "gpu":
                # dali_logger.info(f"[DALI] Moving decoded_images from CPU to GPU before GPU-bound resize.") # Removed per-batch log
                decoded_images = fn.copy(decoded_images, device="gpu") 
            # dali_logger.info(f"[DALI] decoded_images after potential GPU copy: {decoded_images}") # Removed per-batch log


            resized_images = fn.resize(
                decoded_images, 
                resize_x=self.config['image_size'][1], # Use keyword arg
                resize_y=self.config['image_size'][0], # Use keyword arg
                interp_type=types.INTERP_LINEAR,
                device=self.dali_op_device 
            )
            # dali_logger.info(f"[DALI define_graph] Resized images (DataNode after resize): {resized_images}") # Removed per-batch log
            # dali_logger.info(f"[DALI] Resized images node: {resized_images}") # Removed per-batch log
            # dali_logger.info(f"[DALI] Resize called with: image tensor={decoded_images}, resize_x={self.config['image_size'][1]}, resize_y={self.config['image_size'][0]}, device={self.dali_op_device}") # Removed per-batch log

            imgs_augmented = resized_images 

            # Elastic Transform / Warp Affine Fallback
            if self.config.get('elastic_p', 0.0) > 0:
                # dali_logger.info(f"[DALI] coin_flip probability for elastic warp: {self.config['elastic_p']}") # Removed per-batch log
                coin_flip_elastic = fn.random.coin_flip(probability=self.config['elastic_p'])
                # dali_logger.info(f"[DALI] coin_flip DataNode for elastic: {coin_flip_elastic}") # Removed per-batch log

                # dali_logger.info(f"[DALI DEBUG] Before Elastic/Warp: Shape={fn.shapes(imgs_augmented)}, Dtype={fn.cast(imgs_augmented, dtype=types.FLOAT)}") # Removed per-batch log
                if hasattr(fn, 'elastic_transform'):
                    # dali_logger.info("Using DALI elastic_transform (available).") # Removed per-batch log
                    e_transformed = fn.elastic_transform(
                        imgs_augmented, 
                        alpha=(self.config['elastic']['alpha'], self.config['elastic']['alpha']),
                        sigma=(self.config['elastic']['sigma'], self.config['elastic']['sigma']),
                        interp_type=types.INTERP_LINEAR,
                        device=self.dali_op_device
                    )
                else:
                    # dali_logger.warning("DALI fn.elastic_transform not available. Using fn.warp_affine as fallback for elastic distortion.") # Removed per-batch log
                    angle = fn.random.uniform(range=(-5.0, 5.0)) 
                    affine_matrix = fn.transforms.rotation(
                        angle=angle,
                        device="cpu" 
                    )
                    # dali_logger.info(f"[DALI] Affine matrix device (should be CPU): {affine_matrix.device}") # Removed per-batch log
                    
                    e_transformed = fn.warp_affine(
                        imgs_augmented,
                        matrix=affine_matrix, 
                        interp_type=types.INTERP_LINEAR,
                        fill_value=0, 
                        device=self.dali_op_device
                    )

                # dali_logger.info(f"[DALI] Input tensor before elastic/warp_affine: {imgs_augmented}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Output tensor after elastic/warp_affine: {e_transformed}") # Removed per-batch log
                
                # The result of this arithmetic operation is float.
                imgs_augmented = fn.cast(coin_flip_elastic, dtype=types.FLOAT) * e_transformed + \
                                 (1 - fn.cast(coin_flip_elastic, dtype=types.FLOAT)) * imgs_augmented
                # dali_logger.info(f"[DALI DEBUG] After Elastic/Warp (conditional): Shape={fn.shapes(imgs_augmented)}, Dtype={fn.cast(imgs_augmented, dtype=types.FLOAT)}") # Removed per-batch log
                # dali_logger.info(f"[DALI define_graph] Images after Elastic Transform/Warp Affine (DataNode): {imgs_augmented}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Augmented image node after conditional selection (elastic): {imgs_augmented}") # Removed per-batch log


            # Gaussian Blur
            if self.config.get('phot_blur_p', 0.0) > 0:
                # dali_logger.info(f"[DALI] coin_flip probability for gaussian_blur: {self.config['phot_blur_p']}") # Removed per-batch log
                coin_flip_blur = fn.random.coin_flip(probability=self.config['phot_blur_p'])
                # dali_logger.info(f"[DALI] coin_flip DataNode for blur: {coin_flip_blur}") # Removed per-batch log

                # dali_logger.info(f"[DALI DEBUG] Before Gaussian Blur: Shape={fn.shapes(imgs_augmented)}, Dtype={fn.cast(imgs_augmented, dtype=types.FLOAT)}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Applying gaussian_blur with sigma={self.config['phot_blur']}, device={self.dali_op_device}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Input tensor before gaussian_blur: {imgs_augmented}") # Removed per-batch log
                b_blurred = fn.gaussian_blur(
                    imgs_augmented,
                    sigma=tuple(self.config['phot_blur']),
                    device=self.dali_op_device
                )
                # dali_logger.info(f"[DALI] Output tensor after gaussian_blur: {b_blurred}") # Removed per-batch log
                
                # The result of this arithmetic operation is float.
                imgs_augmented = fn.cast(coin_flip_blur, dtype=types.FLOAT) * b_blurred + \
                                 (1 - fn.cast(coin_flip_blur, dtype=types.FLOAT)) * imgs_augmented
                # dali_logger.info(f"[DALI DEBUG] After Gaussian Blur (conditional): Shape={fn.shapes(imgs_augmented)}, Dtype={fn.cast(imgs_augmented, dtype=types.FLOAT)}") # Removed per-batch log
                # dali_logger.info(f"[DALI define_graph] Images after Gaussian Blur (DataNode): {imgs_augmented}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Augmented image node after conditional selection (blur): {imgs_augmented}") # Removed per-batch log


            # --- Debug Hook 1: Inspect imgs_augmented before JPEG processing ---
            def inspect_imgs_augmented(img_np):
                if self.debug_log_counter < self.debug_log_limit:
                    dali_logger.info(f">>> DEBUG HOOK (Pre-JPEG aug): Dtype={img_np.dtype}, Shape={img_np.shape}, Min={img_np.min()}, Max={img_np.max()}")
                return img_np

            imgs_augmented_hooked = fn.python_function(
                imgs_augmented,
                function=inspect_imgs_augmented,
                device="cpu", # Run on CPU to access NumPy array directly
                num_outputs=1
            )
            # --- End Debug Hook 1 ---


            # UPDATED: JPEG Compression Distortion using OpenCV Fallback
            if self.config.get('jpeg_p', 0.0) > 0:
                # dali_logger.info(f"[DALI] coin_flip probability for jpeg_compression_distortion: {self.config['jpeg_p']}") # Removed per-batch log
                coin_flip_jpeg = fn.random.coin_flip(probability=self.config['jpeg_p'])
                # dali_logger.info(f"[DALI] coin_flip DataNode for JPEG: {coin_flip_jpeg}") # Removed per-batch log

                # dali_logger.info(f"[DALI] Applying JPEG distortion via OpenCV fallback with quality={self.config['jpeg_q']}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Input tensor before JPEG distortion (OpenCV): {imgs_augmented_hooked}") # Removed per-batch log
                
                # Use Python function to clamp and cast to UINT8
                imgs_uint8 = fn.python_function(
                    imgs_augmented_hooked,
                    function=clamp_to_uint8,
                    device="cpu",        # runs on CPU so you can use numpy
                    num_outputs=1
                )
                # Force materialization before passing to jpeg_np (optional but safe)
                jpeg_ready = fn.copy(imgs_uint8) 
                
                # dali_logger.info(f"[DALI] Tensor *immediately before* JPEG distortion (OpenCV, after clamp, cast to UINT8, and copy): {jpeg_ready}") # Removed per-batch log
                # dali_logger.info(f"[DALI DEBUG] Before JPEG Distortion (OpenCV): Shape={fn.shapes(jpeg_ready)}, Dtype={fn.cast(jpeg_ready, dtype=types.FLOAT)}") # Removed per-batch log

                # --- Debug Hook 2: Inspect jpeg_ready (input to jpeg_np) ---
                def inspect_jpeg_ready(img_np):
                    if self.debug_log_counter < self.debug_log_limit:
                        dali_logger.info(f">>> DEBUG HOOK (Post-Clamp-Cast-Copy Pre-OpenCV JPEG): Dtype={img_np.dtype}, Shape={img_np.shape}, Min={img_np.min()}, Max={img_np.max()}")
                    return img_np

                jpeg_ready_hooked = fn.python_function(
                    jpeg_ready,
                    function=inspect_jpeg_ready,
                    device="cpu", # Run on CPU to access NumPy array directly
                    num_outputs=1
                )
                # --- End Debug Hook 2 ---

                # Perform JPEG compression/decompression using OpenCV via python_function
                j_compressed_uint8 = fn.python_function(
                    jpeg_ready_hooked, 
                    function=lambda x: jpeg_np(x, self.config['jpeg_q']), 
                    device="cpu", # OpenCV operations run on CPU
                    num_outputs=1
                )

                # dali_logger.info(f"[DALI] Output tensor after JPEG distortion (OpenCV): {j_compressed_uint8}") # Removed per-batch log
                # dali_logger.info(f"[DALI DEBUG] After JPEG Distortion (OpenCV): Shape={fn.shapes(j_compressed_uint8)}, Dtype={fn.cast(j_compressed_uint8, dtype=types.FLOAT)}") # Removed per-batch log
                
                # Convert j_compressed_uint8 back to float for the rest of your pipeline:
                j_float = fn.cast(j_compressed_uint8, dtype=types.FLOAT)
                imgs_augmented = fn.cast(coin_flip_jpeg, dtype=types.FLOAT) * j_float + \
                                 (1 - fn.cast(coin_flip_jpeg, dtype=types.FLOAT)) * imgs_augmented
                # dali_logger.info(f"[DALI define_graph] Images after JPEG Compression Distortion (DataNode): {imgs_augmented}") # Removed per-batch log
                # dali_logger.info(f"[DALI] Augmented image node after conditional selection (jpeg): {imgs_augmented}") # Removed per-batch log

            # Increment the counter after all conditional logging for this batch
            self.debug_log_counter += 1

            # Normalize and change layout to NCHW float [0,1]
            # dali_logger.info(f"[DALI DEBUG] Before CropMirrorNormalize: Shape={fn.shapes(imgs_augmented)}, Dtype={fn.cast(imgs_augmented, dtype=types.FLOAT)}") # Removed per-batch log
            final_images = fn.crop_mirror_normalize(
                    imgs_augmented,
                    dtype=types.FLOAT,
                    output_layout=types.NCHW,
                    mean=[0.0,0.0,0.0],
                    std=[255.0,255.0,255.0])
            # dali_logger.info(f"[DALI define_graph] Final images before output (DataNode): {final_images}") # Removed per-batch log
            # dali_logger.info(f"[DALI] Final normalized output: {final_images}") # Removed per-batch log
            # dali_logger.info(f"[DALI DEBUG] Final Output: Shape={fn.shapes(final_images)}, Dtype={fn.cast(final_images, dtype=types.FLOAT)}") # Removed per-batch log


            return final_images, labels
