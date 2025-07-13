import os
import random
import json
import logging
import math
from pathlib import Path
from collections import Counter

import cv2
import yaml
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from perlin_noise import PerlinNoise

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import AugMix

from .config_loader import CONFIG
from .simgan_refiner import SimGANRefiner
from timm.data.auto_augment import rand_augment_transform

ra_transform = rand_augment_transform(
    config_str='rand-m9-mstd0.5-inc1', # example policy
    hparams=None
)
# apply `ra_transform(image=img_np)['image']` before ToTensorV2

# ----------------- LOGGER SETUP -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------- SIMGAN REFINER INIT --------------
device = CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu")
simgan_refiner = SimGANRefiner().to(device)
logger.info(f"SimGAN Refiner initialized on {device}")

# --- Augmentation Pipeline ---
_alb_pipeline = A.Compose([
  A.Resize(640,640),
  A.RandomBrightnessContrast(p=0.5),
  A.MotionBlur(blur_limit=7, p=0.3),
  A.GaussianNoise(var_limit=(10,50), p=0.3),
  A.HueSaturationValue(p=0.3),
  A.RandomShadow(p=0.3),
  A.RandomSnow(p=0.2),
  ToTensorV2(),
])



# 2) TorchVision’s AugMix
_torch_augmix = AugMix(
    severity=CONFIG.get("augmix_severity", 3),
    mixture_width=CONFIG.get("augmix_mixture_width", 3),
    chain_depth=CONFIG.get("augmix_chain_depth", -1),
    alpha=CONFIG.get("augmix_alpha", 1.0),
    all_ops=True
)

def transform_image(image_np: np.ndarray) -> torch.Tensor:
    """
    Applies Albumentations → ToTensorV2 → TorchVision AugMix.
    Input: H×W×C uint8 image array
    Output: C×H×W float tensor normalized [0,1]
    """
    # Albumentations: returns {'image': Tensor[C,H,W]}
    result = _alb_pipeline(image=image_np)
    img = result["image"]

    # AugMix: in-place on Tensor
    img = _torch_augmix(img)
    return img

# -------------- SimGAN REFINING -----------------
def apply_simgan(image: np.ndarray) -> np.ndarray:
    """
    Refine a NumPy H×W×C image via SimGAN.
    Returns a refined H×W×C uint8 array.
    """
    # To float tensor [B=1,C,H,W] in [0,1]
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        refined = simgan_refiner(tensor)
    refined = refined.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (refined * 255).astype(np.uint8)

# ---------------- NVIDIA DALI --------------------
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    HAS_DALI = True
    dali_logger = logging.getLogger("dali")
    dali_logger.setLevel(logging.ERROR)
    dali_logger.addHandler(logging.StreamHandler())
    dali_logger.info("NVIDIA DALI is available")

except ImportError:
    HAS_DALI = False
    logging.getLogger("dali").warning("NVIDIA DALI not found; fallback to PyTorch DataLoader")

# Example: build a simple DALI pipeline (customize per your data)
def build_dali_pipeline(batch_size, num_threads, device_id, data_dir, img_size):
    assert HAS_DALI, "DALI not installed"
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            name="Reader"
        )
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=img_size, resize_y=img_size)
        pipe.set_outputs(images, labels)
    return pipe

# --------------- PyTorch DataLoader ---------------
from torch.utils.data import Dataset, DataLoader

class BongardDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.files = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img_np = np.array(img)
        if self.transform:
            return self.transform(img_np)
        return torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

def get_pytorch_loader(image_dir, batch_size, shuffle=True, num_workers=4):
    ds = BongardDataset(image_dir, transform=transform_image)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

# --- Initialize translators once (using CONFIG from main) ---
# simgan_gen = SimGANGenerator(CONFIG['simgan']['path']) if CONFIG.get('simgan', {}).get('enabled') else None
# cyclegan_gen = CycleGANGenerator(CONFIG['cyclegan']['path']) if CONFIG.get('cyclegan', {}).get('enabled') else None
# These are now handled by the apply_simgan and apply_cyclegan functions,
# and the models themselves are initialized once globally if needed.
# For now, we'll keep the stubs for SimGANGenerator/CycleGANGenerator if they are used elsewhere.
# If they are only used for apply_simgan/apply_cyclegan, they can be removed.
# Assuming SimGANGenerator and CycleGANGenerator are still needed for other parts of the pipeline,
# but their direct usage for translation will be replaced by apply_simgan/apply_cyclegan.
# For the purpose of this task, we will comment out the initialization of simgan_gen and cyclegan_gen
# as apply_simgan function will be used directly.

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
    
    if morphological_ops:
        kernel = np.ones((3,3), np.uint8)
        if morphological_ops == 'open':
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        elif morphological_ops == 'close':
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:max_contours_to_find]
    return filtered_contours

def cnt_to_yolo(contour, img_width, img_height, class_id, min_contour_area=0):
    """
    Converts an OpenCV contour to YOLO bounding box format (class_id center_x center_y width height).
    Args:
        contour (np.array): An OpenCV contour.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.
        class_id (int): The specific class ID for this contour's shape type.
        min_contour_area (int): Minimum contour area to process.
    Returns:
        tuple: (class_id, center_x, center_y, width, height) or None if invalid.
    """
    if cv2.contourArea(contour) < min_contour_area:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    center_x = (x + w / 2) / img_width
    center_y = (y + h / 2) / img_height
    norm_width = w / img_width
    norm_height = h / img_height
    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 <= norm_width <= 1 and 0 <= norm_height <= 1):
        return None
    return (class_id, center_x, center_y, norm_width, norm_height)

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
    
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)
    attrs['num_vertices'] = num_vertices
    total_image_area = image_color.shape[0] * image_color.shape[1]
    relative_area = area / total_image_area if total_image_area > 0 else 0
    
    # UPDATED: Shape type detection for 'line' and 'dot'
    # Heuristics for 'dot'
    if relative_area < 0.001 and attrs['circularity'] > 0.7 and 0.7 < attrs['aspect_ratio'] < 1.3:
        attrs['shape_type'] = 'dot'
    # Heuristics for 'line'
    elif (attrs['aspect_ratio'] > 5 or attrs['aspect_ratio'] < 0.2) and relative_area < 0.01 and attrs['circularity'] < 0.5 and num_vertices <= 2:
        attrs['shape_type'] = 'line'
    elif num_vertices == 3:
        attrs['shape_type'] = 'triangle'
    elif num_vertices == 4:
        if 0.8 <= attrs['aspect_ratio'] <= 1.2 and attrs['circularity'] > 0.6:
            attrs['shape_type'] = 'square'
        elif 0.5 <= attrs['aspect_ratio'] <= 2.0:
            attrs['shape_type'] = 'rectangle'
        else:
            attrs['shape_type'] = 'polygon' # Fallback for irregular quadrilaterals
    elif num_vertices > 4 and attrs['circularity'] > 0.8:
        attrs['shape_type'] = 'circle'
    elif num_vertices > 4:
        attrs['shape_type'] = 'polygon'
    else:
        attrs['shape_type'] = 'other' # For complex or irregular shapes
    if relative_area < img_size_categories[0]:
        attrs['size_category'] = 'tiny'
    elif relative_area < img_size_categories[1]:
        attrs['size_category'] = 'small'
    elif relative_area < img_size_categories[2]:
        attrs['size_category'] = 'medium'
    else:
        attrs['size_category'] = 'large'
    if len(contour) >= 5:
        rect = cv2.minAreaRect(contour)
        attrs['orientation'] = rect[2]
    else:
        attrs['orientation'] = 0.0
    if area > 0:
        attrs['complexity'] = (perimeter * perimeter) / area
    else:
        attrs['complexity'] = 0.0
    return attrs

def compute_relations(attributes_list):
    """
    Computes simple spatial relations between detected objects, including overlaps and contains.
    """
    relations = []
    num_objects = len(attributes_list)
    if num_objects < 2:
        return relations
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1_attrs = attributes_list[i]
            obj2_attrs = attributes_list[j]
            bbox1 = obj1_attrs.get('bbox')
            bbox2 = obj2_attrs.get('bbox')
            if 'centroid' in obj1_attrs and 'centroid' in obj2_attrs:
                cx1, cy1 = obj1_attrs['centroid']
                cx2, cy2 = obj2_attrs['centroid']
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
                    diag1 = np.sqrt(bbox1[2]**2 + bbox1[3]**2)
                    diag2 = np.sqrt(bbox2[2]**2 + bbox2[3]**2)
                    if dist < (diag1 + diag2) / 2 * 0.75:
                        relations.append({'type': 'is_close_to', 'object1_idx': i, 'object2_idx': j})
            
            if bbox1 and bbox2:
                x1, y1, w1, h1 = bbox1
                x2, y2, w2, h2 = bbox2
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                intersection_area = x_overlap * y_overlap
                if intersection_area > 0:
                    relations.append({'type': 'overlaps', 'object1_idx': i, 'object2_idx': j, 'overlap_area': intersection_area})
                    if x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2):
                        relations.append({'type': 'contains', 'object1_idx': i, 'object2_idx': j})
                    elif x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                        relations.append({'type': 'contains', 'object1_idx': j, 'object2_idx': i})
    return relations

def random_bg(img_np, bg_paths):
    """
    Applies a random background image to the foreground objects in img_np.
    Assumes img_np has a white background that needs to be replaced.
    Args:
        img_np (np.ndarray): Foreground image (HWC, uint8, RGB or BGR).
        bg_paths (list): List of paths to background images.
    Returns:
        np.ndarray: Image with random background applied.
    """
    if not bg_paths:
        return img_np
    if img_np is None or img_np.size == 0:
        return img_np
    bg_path = random.choice(bg_paths)
    try:
        bg = cv2.imread(str(bg_path))
        if bg is None:
            logger.warning(f"Failed to load background image: {bg_path}. Returning original image.")
            return img_np
        
        bg = cv2.resize(bg, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        if bg.shape[-1] == 1:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        
        # Ensure both are 3-channel for bitwise operations
        if img_np.ndim == 2: # if grayscale, convert to BGR
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # Create a mask from the foreground (assuming white background)
        # Convert to grayscale to create a binary mask
        gray_fg = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        # Invert the mask if the foreground is black on white, or vice versa
        # Assuming objects are dark on a light background, threshold to get objects as white
        _, mask = cv2.threshold(gray_fg, 240, 255, cv2.THRESH_BINARY_INV) # Objects are > 0, background is 0
        mask_inv = cv2.bitwise_not(mask) # Invert mask for background region
        # Extract foreground and background parts
        img_fg = cv2.bitwise_and(img_np, img_np, mask=mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
        
        combined_img = cv2.add(img_fg, bg_part)
        return combined_img
    except Exception as e:
        logger.error(f"Error applying random background from {bg_path}: {e}. Returning original image.")
        return img_np

def occlude(img, occlude_p, occlude_max_factor):
    """
    Applies random rectangular occlusion to an image.
    Args:
        img (np.ndarray): Input image (HWC, uint8).
        occlude_p (float): Probability of applying occlusion (not used here, assumed to be handled by DALI).
        occlude_max_factor (float): Maximum fraction of image dimension for occlusion patch size.
    Returns:
        np.ndarray: Image with occlusion.
    """
    h, w, _ = img.shape
    mf = occlude_max_factor
    ph = int(h * random.uniform(0.05, mf))
    pw = int(w * random.uniform(0.05, mf))
    if ph <= 0 or pw <= 0:
        return img.copy()
    y = random.randint(0, h - ph) if h - ph > 0 else 0
    x = random.randint(0, w - pw) if w - pw > 0 else 0
    occluded_img = img.copy()
    occluded_img[y:y+ph, x:x+pw] = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)
    return occluded_img

def add_clutter(img, num_clutter_patches, clutter_max_factor):
    """
    Adds random clutter (rectangles or circles) to an image.
    Args:
        img (np.ndarray): Input image (HWC, uint8).
        num_clutter_patches (int): Number of clutter patches to add.
        clutter_max_factor (float): Maximum fraction of image dimension for clutter patch size.
    Returns:
        np.ndarray: Image with clutter.
    """
    cluttered_img = img.copy()
    h, w, _ = img.shape
    for _ in range(num_clutter_patches):
        patch_h = int(h * random.uniform(0.02, clutter_max_factor))
        patch_w = int(w * random.uniform(0.02, clutter_max_factor))
        if patch_h <= 0 or patch_w <= 0:
            continue
        y = random.randint(0, h - patch_h) if h - patch_h > 0 else 0
        x = random.randint(0, w - patch_w) if w - pw > 0 else 0
        
        if random.random() < 0.5: # Random color block
            cluttered_img[y:y+patch_h, x:x+pw] = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)
        else: # Random basic shape
            shape_type = random.choice(['circle', 'rectangle'])
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            
            if shape_type == 'circle':
                center = (x + patch_w // 2, y + patch_h // 2)
                radius = min(patch_w, patch_h) // 2
                cv2.circle(cluttered_img, center, radius, color, -1)
            else:
                cv2.rectangle(cluttered_img, (x, y), (x + patch_w, y + patch_h), color, -1)
    return cluttered_img

def get_cached_fractal(fractal_type, cache_dir, size, noise_params):
    seed_key = noise_params.get('seed', 0)
    level_key = noise_params.get('level', 0)
    if fractal_type == 'perlin':
        cache_filename = f"{fractal_type}_{size[0]}x{size[1]}_oct{noise_params.get('octaves',4)}_seed{seed_key}.png"
    elif fractal_type in ['sierpinski', 'koch']:
        cache_filename = f"{fractal_type}_{size[0]}x{size[1]}_level{level_key}_seed{seed_key}.png"
    else:
        cache_filename = f"{fractal_type}_{size[0]}x{size[1]}_seed{seed_key}.png"
    
    cache_path = Path(cache_dir) / cache_filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        img = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.ndim == 2:
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
            seed=seed_key
        )
        img_array = np.array([[noise([i/size[0], j/size[1]]) for j in range(size[1])] for i in range(size[0])])
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        generated_img = (img_array * 255).astype(np.uint8)
        generated_img = cv2.cvtColor(generated_img, cv2.COLOR_GRAY2BGR)
    elif fractal_type == 'sierpinski':
        generated_img = draw_sierpinski(img_size=size, level=noise_params.get('level', 4), color=(255, 255, 255))
    elif fractal_type == 'koch':
        generated_img = draw_koch(img_size=size, level=noise_params.get('level', 3), color=(255, 255, 255))
    else:
        logger.warning(f"Unknown fractal type: {fractal_type}. Returning black image.")
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)
    if generated_img is not None:
        cv2.imwrite(str(cache_path), generated_img)
        logger.info(f"Cached generated fractal to: {cache_path}")
        return generated_img
    else:
        logger.error(f"Failed to generate {fractal_type} image.")
        return np.zeros((size[0], size[1], 3), dtype=np.uint8)

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
        cv2.fillPoly(img, [inverted_triangle_points], (0, 0, 0)) # Fill with black to "remove"
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
    Perform JPEG compression and decompression using OpenCV.
    Args:
        x (np.ndarray): Input image (HWC, uint8).
        quality_range (tuple): A tuple (min_quality, max_quality) for random selection.
    Returns:
        np.ndarray: JPEG compressed and decompressed image (HWC, uint8).
    """
    q = int(np.random.uniform(*quality_range))
    _, buf = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# --- MixUp / CutMix Helper Functions (for DALI Python Function) ---
def mixup_np(img1, labels1, img2, labels2, alpha):
    """
    Applies MixUp augmentation to two images and their labels.
    Args:
        img1 (np.ndarray): First image (HWC, uint8).
        labels1 (np.ndarray): Labels for first image (N, 5).
        img2 (np.ndarray): Second image (HWC, uint8).
        labels2 (np.ndarray): Labels for second image (N, 5).
        alpha (float): MixUp alpha parameter.
    Returns:
        tuple: (mixed_image, mixed_labels)
    """
    lam = np.random.beta(alpha, alpha)
    mixed_img = (lam * img1 + (1 - lam) * img2).astype(np.uint8)
    mixed_labels = np.concatenate((labels1, labels2), axis=0)
    return mixed_img, mixed_labels

def cutmix_np(img1, labels1, img2, labels2, alpha):
    """
    Applies CutMix augmentation to two images and their labels.
    Args:
        img1 (np.ndarray): First image (HWC, uint8).
        labels1 (np.ndarray): Labels for first image (N, 5).
        img2 (np.ndarray): Second image (HWC, uint8).
        labels2 (np.ndarray): Labels for second image (N, 5).
        alpha (float): CutMix alpha parameter.
    Returns:
        tuple: (cutmixed_image, cutmixed_labels)
    """
    lam = np.random.beta(alpha, alpha)
    H, W, _ = img1.shape
    
    cx = np.random.uniform(0, W)
    cy = np.random.uniform(0, H)
    w_cut = W * np.sqrt(1 - lam)
    h_cut = H * np.sqrt(1 - lam)
    
    x1 = int(np.round(max(0, cx - w_cut / 2)))
    y1 = int(np.round(max(0, cy - h_cut / 2)))
    x2 = int(np.round(min(W, cx + w_cut / 2)))
    y2 = int(np.round(min(H, cy + h_cut / 2)))
    cutmixed_img = img1.copy()
    cutmixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    # Adjust labels: keep labels from img1 that are outside the cut-out
    # and labels from img2 that are inside the cut-out.
    labels1_px = labels1.copy()
    if labels1_px.shape[0] > 0:
        labels1_px[:, 1] = labels1_px[:, 1] * W # cx
        labels1_px[:, 2] = labels1_px[:, 2] * H # cy
        labels1_px[:, 3] = labels1_px[:, 3] * W # w
        labels1_px[:, 4] = labels1_px[:, 4] * H # h
    filtered_labels1 = []
    for label in labels1_px:
        _, cx, cy, w_box, h_box = label
        box_x1, box_y1 = cx - w_box / 2, cy - h_box / 2
        box_x2, box_y2 = cx + w_box / 2, cy + h_box / 2
        
        if not (box_x2 > x1 and box_x1 < x2 and box_y2 > y1 and box_y1 < y2):
            filtered_labels1.append(label)
    
    labels2_px = labels2.copy()
    if labels2_px.shape[0] > 0:
        labels2_px[:, 1] = labels2_px[:, 1] * W # cx
        labels2_px[:, 2] = labels2_px[:, 2] * H # cy
        labels2_px[:, 3] = labels2_px[:, 3] * W # w
        labels2_px[:, 4] = labels2_px[:, 4] * H # h
    filtered_labels2 = []
    for label in labels2_px:
        _, cx, cy, w_box, h_box = label
        box_x1, box_y1 = cx - w_box / 2, cy - h_box / 2
        box_x2, box_y2 = cx + w_box / 2, cy + h_box / 2
        if (box_x1 >= x1 and box_y1 >= y1 and box_x2 <= x2 and box_y2 <= y2):
            filtered_labels2.append(label)
    combined_labels_px = np.array(filtered_labels1 + filtered_labels2, dtype=np.float32)
    if combined_labels_px.shape[0] > 0:
        combined_labels_px[:, 1] = combined_labels_px[:, 1] / W
        combined_labels_px[:, 2] = combined_labels_px[:, 2] / H
        combined_labels_px[:, 3] = combined_labels_px[:, 3] / W
        combined_labels_px[:, 4] = combined_labels_px[:, 4] / H
    
    return cutmixed_img, combined_labels_px

# --- Integrated DALI Python Function for Annotation and Difficulty ---
def _process_image_and_annotate_dali(image_np, config_dict, class_id_map, yolo_classes_list, difficulty_weights):
    """
    Processes a single image (already augmented by DALI) to detect contours,
    extract attributes, compute relations, and calculate a difficulty score.
    This function is designed to be called by DALI's fn.python_function.
    Args:
        image_np (np.ndarray): The augmented image from DALI (HWC, float32 or uint8).
        config_dict (dict): A dictionary containing necessary configuration parameters
                            (e.g., cnt_block, cnt_C, min_area, max_cnt, morphological_ops).
        class_id_map (dict): Mapping from class name to ID (e.g., {'circle': 0}).
        yolo_classes_list (list): List of YOLO class names.
        difficulty_weights (dict): Weights for difficulty scoring.
    Returns:
        tuple: (yolo_labels_array, annotations_json_string, difficulty_score_float)
    """
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    img_height, img_width, _ = image_np.shape
    image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    contours = detect_contours(
        image_gray=image_gray,
        block_size=config_dict['cnt_block'],
        C=config_dict['cnt_C'],
        min_area=config_dict['min_area'],
        max_contours_to_find=config_dict['max_cnt'],
        morphological_ops=config_dict['morphological_ops']
    )
    yolo_labels = []
    attributes_list = []
    for contour in contours:
        attrs = extract_attrs(contour, image_np)
        shape_type = attrs.get('shape_type')
        if shape_type in class_id_map:
            class_id = class_id_map[shape_type]
            yolo_box = cnt_to_yolo(contour, img_width, img_height, class_id, config_dict['min_area'])
            if yolo_box:
                yolo_labels.append(yolo_box)
                attributes_list.append(attrs)
        else:
            logger.debug(f"Skipping contour with unknown shape_type: {shape_type}")
    relations = compute_relations(attributes_list)
    num_objects = len(attributes_list)
    avg_complexity = np.mean([a.get('complexity', 0) for a in attributes_list]) if attributes_list else 0
    num_relations = len(relations)
    difficulty_score = (
        difficulty_weights.get('num_objects', 0) * num_objects +
        difficulty_weights.get('avg_complexity', 0) * avg_complexity +
        difficulty_weights.get('num_relations', 0) * num_relations
    )
    
    max_possible_difficulty = (
        difficulty_weights.get('num_objects', 0) * config_dict['max_cnt'] +
        difficulty_weights.get('avg_complexity', 0) * 100 + # Assuming max complexity is around 100
        difficulty_weights.get('num_relations', 0) * (config_dict['max_cnt'] * (config_dict['max_cnt'] - 1) / 2)
    )
    difficulty_score = difficulty_score / max_possible_difficulty if max_possible_difficulty > 0 else 0.0
    difficulty_score = np.clip(difficulty_score, 0.0, 1.0)
    yolo_labels_np = np.array(yolo_labels, dtype=np.float32) if yolo_labels else np.zeros((0, 5), dtype=np.float32)
    annotations_dict = {
        'objects': attributes_list,
        'relations': relations,
        'difficulty_score': float(difficulty_score)
    }
    annotations_json_string = json.dumps(annotations_dict)
    return yolo_labels_np, np.array(annotations_json_string, dtype=object), np.array(difficulty_score, dtype=np.float32)

# --- DALI Pipeline Definition (UPDATED FOR NEW AUGMENTATIONS & MIXUP/CUTMIX) ---
if HAS_DALI:
    class BongardDaliPipeline(Pipeline):
        def __init__(self, file_root, file_list, config, device_id, is_training=True):
            super().__init__(
                batch_size=config['dali_batch_size'],
                num_threads=config['dali_num_threads'],
                device_id=device_id,
                seed=config['seed'],
                py_num_workers=config.get('dali_py_num_workers', 0),
                prefetch_queue_depth=config['dali_prefetch_queue'],
                exec_async=True, exec_pipelined=True
            )
            self.config = config
            self.file_root = str(file_root)
            self.file_list = str(file_list)
            self.is_training = is_training
            # Dynamic Tuning & Memory Preallocation
            logger.info(f"DALI threads: num_threads={config['dali_num_threads']}, py_workers={config['dali_py_num_workers']}")
            os.environ['DALI_BUFFER_GROWTH_FACTOR'] = str(self.config.get('dali_buffer_growth_factor', 2.0))
            os.environ['DALI_HOST_BUFFER_SHRINK_THRESHOLD'] = str(self.config.get('dali_host_buffer_shrink_threshold', 0.1))
            dali_logger.info(f"DALI environment variables set: DALI_BUFFER_GROWTH_FACTOR={os.environ['DALI_BUFFER_GROWTH_FACTOR']}, DALI_HOST_BUFFER_SHRINK_THRESHOLD={os.environ['DALI_HOST_BUFFER_SHRINK_THRESHOLD']}")
            dali_logger.info(f"DALI Pipeline Initialized: batch_size={config['dali_batch_size']}, threads={config['dali_num_threads']}, device_id={device_id}")
            dali_logger.info(f"DALI Reader configured with: file_root='{self.file_root}', file_list='{self.file_list}'")
            if config.get('force_cpu_dali', False):
                self.dali_op_device = "cpu"
                self.decode_device = "cpu"
                dali_logger.warning("DALI operations are forced to CPU as per configuration.")
            else:
                self.dali_op_device = "gpu" if device_id != -1 else "cpu"
                self.decode_device = "mixed" if device_id != -1 else "cpu"
            dali_logger.info(f"[NVML Check] DALI pipeline device_id={self.device_id}, decode device='{self.decode_device}'")
            self.dali_py_func_args = {
                'config_dict': {
                    'cnt_block': self.config['cnt_block'],
                    'cnt_C': self.config['cnt_C'],
                    'min_area': self.config['min_area'],
                    'max_cnt': self.config['max_cnt'],
                    'morphological_ops': self.config['morphological_ops'],
                },
                'difficulty_weights': DIFFICULTY_WEIGHTS,
                'class_id_map': CLASS_ID,
                'yolo_classes_list': YOLO_CLASSES_LIST,
            }
            if self.device_id != -1 and not self.config['force_cpu_dali']:
                gpu_mem_bytes = self.config.get('dali_gpu_memory_bytes')
                if gpu_mem_bytes:
                    backend.PreallocateDeviceMemory(gpu_mem_bytes, self.device_id)
                    dali_logger.info(f"Preallocated {gpu_mem_bytes / (1024*1024):.2f} MB of DALI GPU memory on device {self.device_id}.")
                
                pinned_mem_bytes = self.config.get('dali_pinned_memory_bytes')
                if pinned_mem_bytes:
                    backend.PreallocatePinnedMemory(pinned_mem_bytes)
                    dali_logger.info(f"Preallocated {pinned_mem_bytes / (1024*1024):.2f} MB of DALI pinned host memory.")
            else:
                dali_logger.info("DALI memory preallocation skipped (running on CPU or GPU disabled).")
        def define_graph(self):
            # Read image paths from the file_list
            jpegs, _ = fn.readers.file(
                file_root=self.file_root,
                file_list=self.file_list,
                random_shuffle=self.is_training,
                name="Reader1",
                # The labels output of readers.file will be the filename string
                labels=True
            )
            # Decode and resize first image stream
            decoded_images = fn.decoders.image(jpegs, device=self.decode_device, output_type=types.RGB)
            if self.decode_device == "cpu" and self.dali_op_device == "gpu":
                decoded_images = fn.copy(decoded_images, device="gpu")
            resized_images = fn.resize(
                decoded_images,
                resize_x=self.config['image_size'][1],
                resize_y=self.config['image_size'][0],
                interp_type=types.INTERP_LINEAR,
                device=self.dali_op_device
            )
            imgs_augmented = resized_images
            # Apply base augmentations (elastic, blur, jpeg) and NEW DALI augmentations
            if self.is_training:
                # Elastic Transform (if available) or Affine (fallback)
                if self.config.get('elastic_p', 0.0) > 0:
                    coin_flip_elastic = fn.random.coin_flip(probability=self.config['elastic_p'])
                    if hasattr(fn, 'elastic_transform'):
                        e_transformed = fn.elastic_transform(
                            imgs_augmented, alpha=(self.config['elastic']['alpha'], self.config['elastic']['alpha']),
                            sigma=(self.config['elastic']['sigma'], self.config['elastic']['sigma']),
                            interp_type=types.INTERP_LINEAR, device=self.dali_op_device)
                    else: # Fallback to affine if elastic_transform not available
                        angle = fn.random.uniform(range=(-5.0, 5.0))
                        affine_matrix = fn.transforms.rotation(angle=angle, device="cpu")
                        e_transformed = fn.warp_affine(imgs_augmented, matrix=affine_matrix,
                            interp_type=types.INTERP_LINEAR, fill_value=0, device=self.dali_op_device)
                    imgs_augmented = fn.cast(coin_flip_elastic, dtype=types.FLOAT) * e_transformed + \
                                   (1 - fn.cast(coin_flip_elastic, dtype=types.FLOAT)) * imgs_augmented
                
                # Gaussian Blur
                if self.config.get('phot_blur_p', 0.0) > 0:
                    coin_flip_blur = fn.random.coin_flip(probability=self.config['phot_blur_p'])
                    b_blurred = fn.gaussian_blur(imgs_augmented, sigma=tuple(self.config['phot_blur']), device=self.dali_op_device)
                    imgs_augmented = fn.cast(coin_flip_blur, dtype=types.FLOAT) * b_blurred + \
                                   (1 - fn.cast(coin_flip_blur, dtype=types.FLOAT)) * imgs_augmented
                
                # JPEG Compression
                if self.config.get('jpeg_p', 0.0) > 0:
                    coin_flip_jpeg = fn.random.coin_flip(probability=self.config['jpeg_p'])
                    imgs_augmented_cpu_for_jpeg = fn.copy(imgs_augmented, device="cpu")
                    imgs_uint8 = fn.python_function(imgs_augmented_cpu_for_jpeg, function=clamp_to_uint8, device="cpu", num_outputs=1)
                    jpeg_ready = imgs_uint8
                    j_compressed_uint8 = fn.python_function(jpeg_ready, function=lambda x: jpeg_np(x, self.config['jpeg_q']), device="cpu", num_outputs=1)
                    if self.dali_op_device == "gpu":
                        j_compressed_uint8_gpu = fn.copy(j_compressed_uint8, device="gpu")
                    else:
                        j_compressed_uint8_gpu = j_compressed_uint8
                    j_float = fn.cast(j_compressed_uint8_gpu, dtype=types.FLOAT)
                    imgs_augmented = fn.cast(coin_flip_jpeg, dtype=types.FLOAT) * j_float + \
                                   (1 - fn.cast(coin_flip_jpeg, dtype=types.FLOAT)) * imgs_augmented
                # NEW DALI Geometric Augmentations
                # Rotation
                coin_rot = fn.random.coin_flip(probability=self.config['dali_rotation_p'])
                rot = fn.transforms.rotation(imgs_augmented, angle=fn.random.uniform(range=(-self.config['dali_rotation_degrees'], self.config['dali_rotation_degrees'])), device=self.dali_op_device)
                imgs_augmented = fn.cast(coin_rot, types.FLOAT) * rot + (1-fn.cast(coin_rot, types.FLOAT))*imgs_augmented
                # Translation
                # DALI's deformable_transform is for flow fields, for simple translation, use affine or crop.
                # Using `crop` with random anchor for a simple translation effect.
                # A more precise DALI translation would involve `fn.transforms.translation` or `fn.warp_affine`.
                coin_trans = fn.random.coin_flip(probability=self.config['dali_translation_p'])  # Define coin_trans
                tx = fn.random.uniform(range=(-self.config['dali_translation_range'][0], self.config['dali_translation_range'][0])) * self.config['image_size'][1]
                ty = fn.random.uniform(range=(-self.config['dali_translation_range'][1], self.config['dali_translation_range'][1])) * self.config['image_size'][0]
                translation_matrix = fn.transforms.translation(offset=(tx, ty), device="cpu")
                translated_imgs = fn.warp_affine(imgs_augmented, matrix=translation_matrix,
                                                 interp_type=types.INTERP_LINEAR, fill_value=0, device=self.dali_op_device)
                imgs_augmented = fn.cast(coin_trans, types.FLOAT) * translated_imgs + imgs_augmented * (1 - fn.cast(coin_trans, types.FLOAT))
                # Shear
                coin_shear = fn.random.coin_flip(probability=self.config['dali_shear_p'])
                shear_angle_x = fn.random.uniform(range=(-self.config['dali_shear_range'][0], self.config['dali_shear_range'][0]))
                shear_angle_y = fn.random.uniform(range=(-self.config['dali_shear_range'][1], self.config['dali_shear_range'][1]))
                shear_matrix = fn.transforms.shear(angle=(shear_angle_x, shear_angle_y), device="cpu")
                sheared = fn.warp_affine(imgs_augmented, matrix=shear_matrix,
                                         interp_type=types.INTERP_LINEAR, fill_value=0, device=self.dali_op_device)
                imgs_augmented = fn.cast(coin_shear, types.FLOAT)*sheared + (1-fn.cast(coin_shear, types.FLOAT))*imgs_augmented
                # NEW DALI Color Augmentations
                # HSV
                coin_hsv = fn.random.coin_flip(probability=self.config['dali_hsv_p'])
                hsv = fn.hsv(
                    imgs_augmented,
                    hue=(self.config['dali_hsv_h_range'][0], self.config['dali_hsv_h_range'][1]),
                    saturation=(self.config['dali_hsv_s_range'][0], self.config['dali_hsv_s_range'][1]),
                    value=(self.config['dali_hsv_v_range'][0], self.config['dali_hsv_v_range'][1]),
                    device=self.dali_op_device
                )
                imgs_augmented = fn.cast(coin_hsv, types.FLOAT)*hsv + (1-fn.cast(coin_hsv, types.FLOAT))*imgs_augmented
                # NEW DALI Noise Augmentations
                # Gaussian Noise
                coin_gn = fn.random.coin_flip(probability=self.config['dali_gaussian_noise_p'])
                noise = fn.noise.gaussian(imgs_augmented, stddev=self.config['dali_gaussian_noise_std'], device=self.dali_op_device)
                imgs_augmented = fn.cast(coin_gn, types.FLOAT)*noise + (1-fn.cast(coin_gn, types.FLOAT))*imgs_augmented
                # Salt & Pepper
                coin_sp = fn.random.coin_flip(probability=self.config['dali_salt_pepper_p'])
                sp = fn.noise.salt_and_pepper(imgs_augmented, amount=0.02, device=self.dali_op_device)
                imgs_augmented = fn.cast(coin_sp, types.FLOAT)*sp + (1-fn.cast(coin_sp, types.FLOAT))*imgs_augmented
                # Integrate Python Augmentations into DALI (occlude, add_clutter)
                # Ensure images are uint8 for these OpenCV-based functions
                imgs_augmented_cpu_uint8 = fn.python_function(fn.copy(imgs_augmented, device="cpu"), function=clamp_to_uint8, device="cpu", num_outputs=1)
                # Occlusion
                coin_occl = fn.random.coin_flip(probability=self.config['occlude_p'])
                occluded = fn.python_function(imgs_augmented_cpu_uint8, function=lambda x: occlude(x, self.config['occlude_p'], self.config['occlude_max']), device="cpu", num_outputs=1)
                # Cast back to float and move to original device if needed
                if self.dali_op_device == "gpu":
                    occluded = fn.copy(occluded, device="gpu")
                imgs_augmented = fn.cast(coin_occl, types.FLOAT) * fn.cast(occluded, types.FLOAT) + imgs_augmented * (1 - fn.cast(coin_occl, types.FLOAT))
                # Add Clutter
                coin_clutter = fn.random.coin_flip(probability=self.config['add_clutter_p'])
                # Re-copy to CPU and clamp to uint8 after previous augmentation, before applying clutter
                imgs_augmented_cpu_uint8_for_clutter = fn.python_function(fn.copy(imgs_augmented, device="cpu"), function=clamp_to_uint8, device="cpu", num_outputs=1)
                cluttered = fn.python_function(imgs_augmented_cpu_uint8_for_clutter, function=lambda x: add_clutter(x, self.config['num_clutter_patches'], self.config['clutter_max_factor']), device="cpu", num_outputs=1)
                # Cast back to float and move to original device if needed
                if self.dali_op_device == "gpu":
                    cluttered = fn.copy(cluttered, device="gpu")
                imgs_augmented = fn.cast(coin_clutter, types.FLOAT) * fn.cast(cluttered, types.FLOAT) + imgs_augmented * (1 - fn.cast(coin_clutter, types.FLOAT))
                # --- SimGAN / CycleGAN Integration ---
                # These need to be applied as Python functions if they are not native DALI ops
                # Assuming simgan_gen and cyclegan_gen are initialized globally in my_data_utils.py
                # and are callable with a numpy array, returning a numpy array.
                if CONFIG.get('simgan', {}).get('enabled'):
                    # Convert to torch tensor, apply SimGAN, convert back to numpy
                    # The apply_simgan function expects HWC numpy array and returns HWC torch tensor.
                    # We need to ensure the DALI python_function receives and returns numpy arrays.
                    # The apply_simgan function is updated to handle the conversion to/from torch.Tensor
                    # and move to/from GPU internally.
                    simgan_translated_np = fn.python_function(
                        imgs_augmented,
                        function=lambda x: apply_simgan(x).numpy(), # apply_simgan now returns HWC numpy
                        device="cpu", # This python function runs on CPU
                        num_outputs=1
                    )
                    if self.dali_op_device == "gpu":
                        simgan_translated_np = fn.copy(simgan_translated_np, device="gpu")
                    imgs_augmented = simgan_translated_np # Update imgs_augmented with the refined image
                    logger.debug("Applied SimGAN translation in DALI pipeline.")

                if CONFIG.get('cyclegan', {}).get('enabled'):
                    imgs_augmented_cpu_uint8_for_cyclegan = fn.python_function(fn.copy(imgs_augmented, device="cpu"), function=clamp_to_uint8, device="cpu", num_outputs=1)
                    cyclegan_translated = fn.python_function(imgs_augmented_cpu_uint8_for_cyclegan, function=lambda x: cyclegan_gen.translate(x), device="cpu", num_outputs=1)
                    if self.dali_op_device == "gpu":
                        cyclegan_translated = fn.copy(cyclegan_translated, device="gpu")
                    imgs_augmented = fn.cast(cyclegan_translated, types.FLOAT)
                    logger.debug("Applied CycleGAN translation in DALI pipeline.")
            # Get initial annotations and labels for both image streams from precomputed files
            # This requires passing the raw image path (from `jpegs` output of reader) to the python function
            # and then loading the corresponding JSON annotation file inside the python function.
            # The `_process_image_and_annotate_dali` function is designed to take an image_np, not a path.
            # So, we need to load the raw annotations here, or pass the path and load in a new function.
            # Let's create a helper Python function to load precomputed annotations
            # This function will be called with the image path string (label from reader)
            # and return the YOLO labels and annotation JSON string.
            def _load_precomputed_annotations_np(image_path_str, raw_annotations_dir_str, class_id_map):
                image_stem = Path(image_path_str.decode('utf-8')).stem # Decode bytes string
                anno_path = Path(raw_annotations_dir_str.decode('utf-8')) / (image_stem + '.json')
                
                if not anno_path.exists():
                    logger.warning(f"Precomputed annotation not found for {image_stem}. Returning empty.")
                    return np.zeros((0,5), dtype=np.float32), json.dumps({})
                try:
                    with open(anno_path, 'r') as f:
                        anno_data = json.load(f)
                    yolo_labels = np.array(anno_data.get('yolo_labels', []), dtype=np.float32)
                    # The annotation JSON for the mixed image will be re-generated by _process_image_and_annotate_dali
                    # So here we just return the raw annotation JSON for consistency if needed, but primarily labels.
                    return yolo_labels, json.dumps(anno_data)
                except Exception as e:
                    logger.error(f"Error loading precomputed annotation for {image_stem}: {e}")
                    return np.zeros((0,5), dtype=np.float32), json.dumps({})
            # Get labels and annotations for first image stream
            yolo_labels1_precomputed, annotations_json_string1_precomputed = fn.python_function(
                jpegs, # Pass the image path string from the reader
                function=lambda path_str: _load_precomputed_annotations_np(
                    path_str,
                    str(self.config['raw_annotations_dir']).encode('utf-8'), # Pass as bytes
                    CLASS_ID
                ),
                num_outputs=2,
                device="cpu",
                output_layouts=["F", "F"] # Flat array for labels, Flat for JSON string
            )
            # To implement MixUp/CutMix, we need a second image and its labels.
            # We'll use a second reader to get an independent sample.
            jpegs2, _ = fn.readers.file(
                file_root=self.file_root,
                file_list=self.file_list,
                random_shuffle=self.is_training,
                name="Reader2", # Unique name for the second reader
                labels=True
            )
            decoded_images2 = fn.decoders.image(jpegs2, device=self.decode_device, output_type=types.RGB)
            if self.decode_device == "cpu" and self.dali_op_device == "gpu":
                decoded_images2 = fn.copy(decoded_images2, device="gpu")
            resized_images2 = fn.resize(
                decoded_images2,
                resize_x=self.config['image_size'][1],
                resize_y=self.config['image_size'][0],
                interp_type=types.INTERP_LINEAR,
                device=self.dali_op_device
            )
            imgs_augmented2 = resized_images2
            # Apply base augmentations to the second image stream as well for consistency
            if self.is_training:
                # Elastic Transform (if available) or Affine (fallback)
                if self.config.get('elastic_p', 0.0) > 0:
                    coin_flip_elastic2 = fn.random.coin_flip(probability=self.config['elastic_p'])
                    if hasattr(fn, 'elastic_transform'):
                        e_transformed2 = fn.elastic_transform(
                            imgs_augmented2, alpha=(self.config['elastic']['alpha'], self.config['elastic']['alpha']),
                            sigma=(self.config['elastic']['sigma'], self.config['elastic']['sigma']),
                            interp_type=types.INTERP_LINEAR, device=self.dali_op_device)
                    else:
                        angle2 = fn.random.uniform(range=(-5.0, 5.0))
                        affine_matrix2 = fn.transforms.rotation(angle=angle2, device="cpu")
                        e_transformed2 = fn.warp_affine(imgs_augmented2, matrix=affine_matrix2,
                            interp_type=types.INTERP_LINEAR, fill_value=0, device=self.dali_op_device)
                    imgs_augmented2 = fn.cast(coin_flip_elastic2, dtype=types.FLOAT) * e_transformed2 + \
                                   (1 - fn.cast(coin_flip_elastic2, dtype=types.FLOAT)) * imgs_augmented2
                
                # Gaussian Blur
                if self.config.get('phot_blur_p', 0.0) > 0:
                    coin_flip_blur2 = fn.random.coin_flip(probability=self.config['phot_blur_p'])
                    b_blurred2 = fn.gaussian_blur(imgs_augmented2, sigma=tuple(self.config['phot_blur']), device=self.dali_op_device)
                    imgs_augmented2 = fn.cast(coin_flip_blur2, dtype=types.FLOAT) * b_blurred2 + \
                                   (1 - fn.cast(coin_flip_blur2, dtype=types.FLOAT)) * imgs_augmented2
                
                # JPEG Compression
                if self.config.get('jpeg_p', 0.0) > 0:
                    coin_flip_jpeg2 = fn.random.coin_flip(probability=self.config['jpeg_p'])
                    imgs_augmented_cpu_for_jpeg2 = fn.copy(imgs_augmented2, device="cpu")
                    imgs_uint82 = fn.python_function(imgs_augmented_cpu_for_jpeg2, function=clamp_to_uint8, device="cpu", num_outputs=1)
                    jpeg_ready2 = imgs_uint82
                    j_compressed_uint82 = fn.python_function(jpeg_ready2, function=lambda x: jpeg_np(x, self.config['jpeg_q']), device="cpu", num_outputs=1)
                    if self.dali_op_device == "gpu":
                        j_compressed_uint8_gpu2 = fn.copy(j_compressed_uint82, device="gpu")
                    else:
                        j_compressed_uint8_gpu2 = j_compressed_uint82
                    j_float2 = fn.cast(j_compressed_uint8_gpu2, dtype=types.FLOAT)
                    imgs_augmented2 = fn.cast(coin_flip_jpeg2, dtype=types.FLOAT) * j_float2 + \
                                   (1 - fn.cast(coin_flip_jpeg2, dtype=types.FLOAT)) * imgs_augmented2
                # NEW DALI Geometric Augmentations for second stream
                coin_rot2 = fn.random.coin_flip(probability=self.config['dali_rotation_p'])
                rot2 = fn.transforms.rotation(imgs_augmented2, angle=fn.random.uniform(range=(-self.config['dali_rotation_degrees'], self.config['dali_rotation_degrees'])), device=self.dali_op_device)
                imgs_augmented2 = fn.cast(coin_rot2, types.FLOAT) * rot2 + (1-fn.cast(coin_rot2, types.FLOAT))*imgs_augmented2
                coin_trans2 = fn.random.coin_flip(probability=self.config['dali_translation_p'])
                tx2 = fn.random.uniform(range=(-self.config['dali_translation_range'][0], self.config['dali_translation_range'][0])) * self.config['image_size'][1]
                ty2 = fn.random.uniform(range=(-self.config['dali_translation_range'][1], self.config['dali_translation_range'][1])) * self.config['image_size'][0]
                translation_matrix2 = fn.transforms.translation(offset=(tx2, ty2), device="cpu")
                translated_imgs2 = fn.warp_affine(imgs_augmented2, matrix=translation_matrix2,
                                                 interp_type=types.INTERP_LINEAR, fill_value=0, device=self.dali_op_device)
                imgs_augmented2 = fn.cast(coin_trans2, types.FLOAT) * translated_imgs2 + imgs_augmented2 * (1 - fn.cast(coin_trans2, types.FLOAT))
                coin_shear2 = fn.random.coin_flip(probability=self.config['dali_shear_p'])
                shear_angle_x2 = fn.random.uniform(range=(-self.config['dali_shear_range'][0], self.config['dali_shear_range'][0]))
                shear_angle_y2 = fn.random.uniform(range=(-self.config['dali_shear_range'][1], self.config['dali_shear_range'][1]))
                shear_matrix2 = fn.transforms.shear(angle=(shear_angle_x2, shear_angle_y2), device="cpu")
                sheared2 = fn.warp_affine(imgs_augmented2, matrix=shear_matrix2,
                                         interp_type=types.INTERP_LINEAR, fill_value=0, device=self.dali_op_device)
                imgs_augmented2 = fn.cast(coin_shear2, types.FLOAT)*sheared2 + (1-fn.cast(coin_shear2, types.FLOAT))*imgs_augmented2
                # NEW DALI Color Augmentations for second stream
                coin_hsv2 = fn.random.coin_flip(probability=self.config['dali_hsv_p'])
                hsv2 = fn.hsv(
                    imgs_augmented2,
                    hue=(self.config['dali_hsv_h_range'][0], self.config['dali_hsv_h_range'][1]),
                    saturation=(self.config['dali_hsv_s_range'][0], self.config['dali_hsv_s_range'][1]),
                    value=(self.config['dali_hsv_v_range'][0], self.config['dali_hsv_v_range'][1]),
                    device=self.dali_op_device
                )
                imgs_augmented2 = fn.cast(coin_hsv2, types.FLOAT)*hsv2 + (1-fn.cast(coin_hsv2, types.FLOAT))*imgs_augmented2
                # NEW DALI Noise Augmentations for second stream
                coin_gn2 = fn.random.coin_flip(probability=self.config['dali_gaussian_noise_p'])
                noise2 = fn.noise.gaussian(imgs_augmented2, stddev=self.config['dali_gaussian_noise_std'], device=self.dali_op_device)
                imgs_augmented2 = fn.cast(coin_gn2, types.FLOAT)*noise2 + (1-fn.cast(coin_gn2, types.FLOAT))*imgs_augmented2
                coin_sp2 = fn.random.coin_flip(probability=self.config['dali_salt_pepper_p'])
                sp2 = fn.noise.salt_and_pepper(imgs_augmented2, amount=0.02, device=self.dali_op_device)
                imgs_augmented2 = fn.cast(coin_sp2, types.FLOAT)*sp2 + (1-fn.cast(coin_sp2, types.FLOAT))*imgs_augmented2
                # Integrate Python Augmentations into DALI (occlude, add_clutter) for second stream
                imgs_augmented_cpu_uint8_2 = fn.python_function(fn.copy(imgs_augmented2, device="cpu"), function=clamp_to_uint8, device="cpu", num_outputs=1)
                coin_occl2 = fn.random.coin_flip(probability=self.config['occlude_p'])
                occluded2 = fn.python_function(imgs_augmented_cpu_uint8_2, function=lambda x: occlude(x, self.config['occlude_p'], self.config['occlude_max']), device="cpu", num_outputs=1)
                if self.dali_op_device == "gpu":
                    occluded2 = fn.copy(occluded2, device="gpu")
                imgs_augmented2 = fn.cast(coin_occl2, types.FLOAT) * fn.cast(occluded2, types.FLOAT) + imgs_augmented2 * (1 - fn.cast(coin_occl2, types.FLOAT))
                coin_clutter2 = fn.random.coin_flip(probability=self.config['add_clutter_p'])
                imgs_augmented_cpu_uint8_for_clutter2 = fn.python_function(fn.copy(imgs_augmented2, device="cpu"), function=clamp_to_uint8, device="cpu", num_outputs=1)
                cluttered2 = fn.python_function(imgs_augmented_cpu_uint8_for_clutter2, function=lambda x: add_clutter(x, self.config['num_clutter_patches'], self.config['clutter_max_factor']), device="cpu", num_outputs=1)
                if self.dali_op_device == "gpu":
                    cluttered2 = fn.copy(cluttered2, device="gpu")
                imgs_augmented2 = fn.cast(coin_clutter2, types.FLOAT) * fn.cast(cluttered2, types.FLOAT) + imgs_augmented2 * (1 - fn.cast(coin_clutter2, types.FLOAT))
                # --- SimGAN / CycleGAN Integration for second stream ---
                if CONFIG.get('simgan', {}).get('enabled'):
                    simgan_translated_np2 = fn.python_function(
                        imgs_augmented2,
                        function=lambda x: apply_simgan(x).numpy(),
                        device="cpu",
                        num_outputs=1
                    )
                    if self.dali_op_device == "gpu":
                        simgan_translated_np2 = fn.copy(simgan_translated_np2, device="gpu")
                    imgs_augmented2 = simgan_translated_np2
                    logger.debug("Applied SimGAN translation to second stream in DALI pipeline.")

                if CONFIG.get('cyclegan', {}).get('enabled'):
                    imgs_augmented_cpu_uint8_for_cyclegan2 = fn.python_function(fn.copy(imgs_augmented2, device="cpu"), function=clamp_to_uint8, device="cpu", num_outputs=1)
                    cyclegan_translated2 = fn.python_function(imgs_augmented_cpu_uint8_for_cyclegan2, function=lambda x: cyclegan_gen.translate(x), device="cpu", num_outputs=1)
                    if self.dali_op_device == "gpu":
                        cyclegan_translated2 = fn.copy(cyclegan_translated2, device="gpu")
                    imgs_augmented2 = fn.cast(cyclegan_translated2, types.FLOAT)
                    logger.debug("Applied CycleGAN translation to second stream in DALI pipeline.")
            # Generate lambda for MixUp/CutMix
            mix_lam = fn.random.uniform(range=(self.config['mixup_alpha'], 1.0 - self.config['mixup_alpha']))
            mix_type_choice = fn.random.coin_flip(probability=0.5) # 0 for MixUp, 1 for CutMix
            # Apply MixUp/CutMix using a python function
            # This function will take two images, two sets of labels, and the lambda.
            # It will return the mixed image and the merged labels.
            # It needs to be on CPU for numpy/OpenCV operations.
            
            # Ensure images are float for mixing, then convert back to uint8 after mixing
            imgs_augmented_float1 = fn.cast(imgs_augmented, dtype=types.FLOAT)
            imgs_augmented_float2 = fn.cast(imgs_augmented2, dtype=types.FLOAT)
            mixed_image_raw, mixed_labels_raw = fn.python_function(
                imgs_augmented_float1, yolo_labels1_precomputed, imgs_augmented_float2, yolo_labels2_precomputed, mix_lam, mix_type_choice,
                function=lambda img1, labels1, img2, labels2, lam_val, mix_type_choice_val:
                    mixup_np(img1, labels1, img2, labels2, lam_val) if mix_type_choice_val == 0 else cutmix_np(img1, labels1, img2, labels2, lam_val),
                num_outputs=2,
                device="cpu",
                output_layouts=["HWC", "F"] # HWC for image, F for flat array of labels
            )
            
            if self.dali_op_device == "gpu":
                mixed_image_raw = fn.copy(mixed_image_raw, device="gpu")
            
            # Re-process the mixed image for annotations and difficulty score
            # This ensures the attributes and relations are re-calculated based on the mixed image.
            mixed_image_cpu_for_annotation = fn.python_function(mixed_image_raw, function=clamp_to_uint8, device="cpu", num_outputs=1)
            mixed_yolo_labels_out, mixed_annotations_json_string_out, mixed_difficulty_score_out = fn.python_function(
                mixed_image_cpu_for_annotation,
                function=_process_image_and_annotate_dali,
                num_outputs=3,
                device="cpu",
                function_args=[self.dali_py_func_args['config_dict'], self.dali_py_func_args['class_id_map'],
                               self.dali_py_func_args['yolo_classes_list'], self.dali_py_func_args['difficulty_weights']]
            )
            final_images = fn.crop_mirror_normalize(
                mixed_image_raw,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[0.0,0.0,0.0],
                std=[255.0,255.0,255.0]
            )
            # Return the processed image, YOLO labels (from mixup/cutmix), annotations JSON, and difficulty score
            return final_images, mixed_yolo_labels_out, mixed_annotations_json_string_out, mixed_difficulty_score_out
