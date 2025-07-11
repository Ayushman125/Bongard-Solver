import os
import glob
import random
import json
import logging
from pathlib import Path
from copy import deepcopy
import functools  # Import functools for partial
import cv2
import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise
import albumentations as A
# Import RandomAugMix from the local file
from albumentations_augmix import RandomAugMix 
# Add torchvision.transforms imports here for the worker processes
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy
# Import Numba for JIT compilation
from numba import njit, prange # prange for parallel loops if needed

# Import the new refactored modules
from fast_props import extract_props
from numba_relations import compute_relations_typed, compute_difficulty_typed

# Configure logging for pipeline_workers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Note: CONFIG and CLASS_ID are not directly imported here.
# They are passed as arguments from the main script.

# --- GLOBAL CONSTANTS (should be consistent with pipeline_workers) ---
YOLO_CLASSES_LIST = [
    'circle', 'square', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'octagon', 'polygon'
]
CLASS_ID = {cls_name: i for i, cls_name in enumerate(YOLO_CLASSES_LIST)}

# ─── UTILITIES FOR TUNER (copied from main for self-containment) ──────────────────────────────────────────────────────
# @njit is not suitable for this function due to dictionary operations and logging
def class_entropy(counts):
    """Calculates entropy for class distribution (higher is better for balance)."""
    total = sum(counts.values())
    if total == 0: return 0.0
    probs = [v / total for v in counts.values() if v > 0]
    return -sum(p * np.log2(p) for p in probs)

# This function is now defined here, at the top-level of this module.
def _detect_labels_and_difficulty_wrapper(args):
    """
    Wrapper for detect_labels_and_difficulty to be used with multiprocessing.Pool.
    """
    path, cfg, class_id_map, yolo_classes_list = args  # Added yolo_classes_list
    return detect_labels_and_difficulty(path, cfg, class_id_map, yolo_classes_list)

# Refactored detect_labels_and_difficulty to use fast_props and numba_relations
def detect_labels_and_difficulty(path, cfg, class_id_map, yolo_classes_list):  # Accept class_id_map and yolo_classes_list
    """
    Detects labels and estimates difficulty for a given image path.
    Uses scikit-image for property extraction and Numba for relations/difficulty.
    """
    try:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) # Convert Path to string
        if gray is None:
            logger.warning(f"Could not read image for difficulty estimation: {path}. Skipping.")
            return [], 0.0, [], np.empty((0,3), dtype=np.int32) # Return empty lists/arrays on failure
        
        # Resize to a consistent size for property extraction
        H_target, W_target = cfg['image_size']
        if gray.shape[0] != H_target or gray.shape[1] != W_target:
            gray = cv2.resize(gray, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
        
        # Use fast_props to extract bounding boxes and areas
        boxes, areas = extract_props(
            gray,
            cfg['min_area'],
            cfg['max_cnt']
        )
        
        n_objects = boxes.shape[0]
        if n_objects == 0:
            return [], 0.0, [], np.empty((0,3), dtype=np.int32) # Return empty lists/arrays if no objects

        # Build YOLO labels from extracted boxes
        labels = []
        # Note: Shape detection is simplified here. In a full pipeline,
        # you'd re-integrate more sophisticated shape classification if needed.
        # For now, we'll use a dummy class_id (e.g., 0 for 'circle' or 'polygon')
        # or you can add logic to infer shape from properties if `skimage.measure.regionprops`
        # provides enough information (e.g., `prop.eccentricity`, `prop.perimeter`).
        # For this refactor, we'll just use the bounding box and a placeholder class_id.
        
        # Placeholder for object attributes that were previously extracted by `extract_attrs`
        # This will be used to populate the 'objects' list in the annotation JSON.
        objects_attrs_for_json = []

        for i in range(n_objects):
            x, y, w, h = boxes[i]
            area = areas[i]

            cx = (x + w / 2) / W_target
            cy = (y + h / 2) / H_target
            nw = w / W_target
            nh = h / H_target
            
            # Use a default class_id if actual shape detection is not performed here
            # For demonstration, let's use a dummy class ID, e.g., 0 for 'polygon'
            dummy_class_id = class_id_map.get('polygon', 0) # Fallback to 0 if 'polygon' not found
            
            labels.append(f"{dummy_class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # Reconstruct basic object attributes for JSON output
            objects_attrs_for_json.append({
                'class_type': YOLO_CLASSES_LIST[dummy_class_id], # Placeholder
                'area': float(area),
                'box': [float(x), float(y), float(w), float(h)],
                # Other attributes like fill_type, stroke_width, orientation, complexity
                # would need to be derived from `prop` or re-computed if necessary.
                # For this refactor, we're focusing on the numeric core.
                'complexity': (w + h) / (area**0.5) if area > 0 else 0.0 # Simple complexity metric
            })


        # Prepare data for Numba-optimized relations and difficulty computation
        xs, ys, ws, hs = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        
        # Compute relations using Numba
        relations = compute_relations_typed(xs, ys, ws, hs)
        
        # Compute average complexity (using area as a proxy for now)
        avg_complexity = np.mean(areas) if n_objects > 0 else 0.0

        # Get difficulty weights
        w0 = cfg['difficulty_weights'].get('num_objects', 0.0)
        w1 = cfg['difficulty_weights'].get('avg_complexity', 0.0)
        w2 = cfg['difficulty_weights'].get('num_relations', 0.0)
        
        # Compute difficulty score using Numba
        difficulty = compute_difficulty_typed(n_objects, float(avg_complexity), relations.shape[0], w0, w1, w2)
        
        # Return labels and difficulty score, along with the detailed object attributes for saving
        return labels, float(difficulty), objects_attrs_for_json, relations # Added objects_attrs_for_json and relations

    except Exception as e:
        logger.warning(f"Failed to process {path} for difficulty estimation: {e}")
        return [], 0.0, [], np.empty((0,3), dtype=np.int32) # Return empty lists/arrays on failure

# ─── CONTOUR & YOLO ENCODING ─────────────────────────────────────────────────
# This function is now mostly superseded by `extract_props` for core object detection.
# It might still be used for specific cases or if `extract_props` is not fully integrated everywhere.
# For now, keep it as it might be called from other parts not yet refactored.
def detect_contours_tuned(gray_img, block_size, C, min_area, max_cnt, morphological_ops):
    """
    Detects contours in a grayscale image using adaptive thresholding and morphological operations.
    Uses parameters passed from the tuner.
    """
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    blur = cv2.GaussianBlur(gray_img, (5,5), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block_size, C
    )
    k = np.ones((3,3), np.uint8)
    # Morphological cleanup
    if morphological_ops == 'close':
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    elif morphological_ops == 'open':
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    
    cnts, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Filter by min_area and then sort and truncate
    filtered_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]
    return sorted(filtered_cnts, key=cv2.contourArea, reverse=True)[:max_cnt]

# @njit is generally not suitable here due to cv2.boundingRect, cv2.arcLength, cv2.approxPolyDP
# and dictionary operations (class_id_map.get).
# This function's role is largely replaced by `extract_props` for core attribute extraction.
# Keep it if it's used elsewhere for specific YOLO conversion needs.
def cnt_to_yolo(cnt, W, H, class_id_map, min_area_threshold):  # Accept class_id_map and min_area_threshold
    """
    Converts an OpenCV contour to YOLO bounding box format and extracts shape type.
    Returns (shape_type, yolo_string, bbox_tuple, contour_object).
    """
    area = cv2.contourArea(cnt) # Define area here
    if area < min_area_threshold: return None  # Use passed threshold
    
    x, y, w, h = cv2.boundingRect(cnt)
    peri = cv2.arcLength(cnt, True)
    
    # Shape detection based on vertices and circularity
    epsilon = 0.02 * peri # Use a more standard epsilon
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)
    
    shape_type = 'polygon'
    if num_vertices == 3: shape_type = 'triangle'
    elif num_vertices == 4:
        aspect_ratio = w / float(h)
        # Refined check for square vs rectangle
        if 0.85 <= aspect_ratio <= 1.15: # and check for approximate right angles if possible
            shape_type = 'square' 
        else:
            shape_type = 'rectangle'
    elif num_vertices == 5: shape_type = 'pentagon'
    elif num_vertices == 6: shape_type = 'hexagon'
    elif num_vertices == 8: shape_type = 'octagon'
    else:
        circularity = 4 * np.pi * area / (peri * peri) if peri else 0
        if circularity > 0.7: shape_type = 'circle' # High circularity for circle
    
    class_id = class_id_map.get(shape_type)  # Use passed class_id_map
    if class_id is None: 
        # logger.debug(f"Shape '{shape_type}' not in YOLO_CLASSES. Skipping contour.")
        return None # Return None if class_id is not found
    
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    nw, nh = w / W, h / H
    
    # Ensure normalized coordinates are within [0, 1]
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= nw <= 1 and 0 <= nh <= 1):
        return None

    return shape_type, f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}", (x,y,w,h), cnt

# ─── ATTRIBUTE & RELATIONS ───────────────────────────────────────────────────
# Removed this function as its functionality is replaced by `extract_props` and direct attribute derivation.
# def extract_attrs(shape_type, cnt, gray_img, bbox, W, H):
#     ...

# Removed this function as its functionality is replaced by `compute_relations_typed`.
# def compute_relations(objects_attrs_list):
#     ...

# The `compute_difficulty_score` is now `compute_difficulty_typed` from `numba_relations.py`

# ─── AUGMENTERS ───────────────────────────────────────────────────────────────
# Custom JPEG compression function using OpenCV
# @njit is not suitable here due to cv2 operations
def _apply_jpeg_compression(image, quality_range, **kwargs):  # Added **kwargs to accept additional arguments
    """Applies JPEG compression using OpenCV with a random quality from the given range."""
    # Ensure image is in BGR format for cv2.imencode
    if image.ndim == 3 and image.shape[2] == 3:  # Assuming RGB input from Albumentations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image  # Grayscale or already BGR
    
    quality = random.randint(quality_range[0], quality_range[1])
    # Encode as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
    
    # Handle potential failure of cv2.imdecode (e.g., if encimg is empty)
    decoded_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    if decoded_img is None:
        logger.warning("Failed to decode JPEG compressed image. Returning original image.")
        return image  # Return original image if decoding fails
    # Convert back to RGB for consistency with other transforms if original was RGB
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    return decoded_img

# @njit is not suitable for make_augmenters due to creating Albumentations/Torchvision objects
def make_augmenters(cfg):
    """Initializes Albumentations and Torchvision augmentations."""
    alb_transform = A.Compose([
        A.ElasticTransform(**cfg['elastic'], p=cfg['elastic_p'], interpolation=cv2.INTER_LINEAR),
        A.GaussianBlur(blur_limit=cfg['phot_blur'], p=cfg['phot_blur_p']),
        # Use functools.partial to wrap _apply_jpeg_compression for A.Lambda
        A.Lambda(
            image=functools.partial(_apply_jpeg_compression, quality_range=cfg['jpeg_q']),
            name='CustomJPEGCompression',
            p=cfg['jpeg_p']
        ),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))  # Added clip=True
    
    # RandomAugMix might need specific handling if it's a custom Albumentations transform
    # Ensure it's compatible with A.Compose
    augmix_transform = A.Compose([
        RandomAugMix(p=cfg['augmix_p'])
    ])
    
    rand_augment = RandAugment(num_ops=cfg['rand_nops'], magnitude=cfg['rand_mag'])
    auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)
    return alb_transform, augmix_transform, rand_augment, auto_augment

# ─── BACKGROUNDS & TEXTURES ───────────────────────────────────────────────────
# @njit is not suitable here due to cv2 operations and glob
def random_bg(img, bg_root):
    """Applies a random background image."""
    bg_paths = glob.glob(f"{bg_root}/*.png") + \
               glob.glob(f"{bg_root}/*.jpg")
    if not bg_paths:
        return img.copy()
    bg_path = random.choice(bg_paths)
    try:
        bg = cv2.imread(bg_path)
        if bg is None:
            return img.copy()
        
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if bg.shape[-1] == 1:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        
        gray_fg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        _, mask = cv2.threshold(gray_fg, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img_fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
        combined_img = cv2.add(img_fg, bg_part)
        return combined_img
    except Exception as e:
        logger.error(f"Error applying random background from {bg_path}: {e}. Returning original image.")
        return img.copy()

@njit(cache=True, parallel=True) # Parallelize outer loop for Perlin noise generation
def perlin_bg(W, H, perlin_noise_func):
    """
    Generates a Perlin noise background using a Numba-compatible Perlin noise function.
    The perlin_noise_func must be a Numba-jitted function itself or a direct Python callable
    that Numba can compile in object mode.
    For best performance with @njit(nopython=True), perlin_noise_func should also be @njit.
    """
    img = np.zeros((H, W), np.float32)
    for y in prange(H): # Use prange for parallel loop
        for x in range(W):
            img[y, x] = perlin_noise_func([y / H, x / W]) # Call the passed Numba-compatible function
    
    # Normalize to 0-255
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8) # Handle flat noise case
        
    # Numba does not handle cv2.cvtColor directly in nopython mode.
    # Conversion to BGR will happen outside or in a separate step.
    # For now, return grayscale.
    return img

# ─── COPY-PASTE, OCCLUDE, CLUTTER ─────────────────────────────────────────────
# @njit is not suitable due to cv2 operations, random.randint/uniform, and list/dict manipulations
def copy_paste(main_img, main_yolo_labels, main_objects_attrs, source_img, source_yolo_labels, source_objects_attrs):
    """
    Performs Copy-Paste augmentation.
    Copies a random object from source_img to main_img.
    Requires object masks/contours for precise pasting. This is a simplified bbox-based copy.
    """
    if not source_objects_attrs:
        return main_img, main_yolo_labels, main_objects_attrs
    H, W, _ = main_img.shape
    
    chosen_obj_idx = random.randint(0, len(source_objects_attrs) - 1)
    chosen_obj_attr = source_objects_attrs[chosen_obj_idx]
    chosen_obj_yolo = source_yolo_labels[chosen_obj_idx]
    
    sx, sy, sw, sh = chosen_obj_attr['box']
    
    if sw <= 0 or sh <= 0 or sx < 0 or sy < 0 or sx+sw > source_img.shape[1] or sy+sh > source_img.shape[0]:
        logger.warning(f"Invalid source object bbox for copy-paste: {chosen_obj_attr['box']}. Skipping.")
        return main_img, main_yolo_labels, main_objects_attrs
    
    object_crop = source_img[sy:sy+sh, sx:sx+sw]
    
    if W - sw <= 0 or H - sh <= 0:
        logger.warning(f"Main image too small for object of size {sw}x{sh}. Skipping copy-paste.")
        return main_img, main_yolo_labels, main_objects_attrs
    
    paste_x = random.randint(0, W - sw)
    paste_y = random.randint(0, H - sh)
    
    gray_object_crop = cv2.cvtColor(object_crop, cv2.COLOR_BGR2GRAY) if object_crop.ndim == 3 else object_crop
    _, object_mask = cv2.threshold(gray_object_crop, 1, 255, cv2.THRESH_BINARY)
    object_mask_inv = cv2.bitwise_not(object_mask)
    
    main_bg_roi = cv2.bitwise_and(main_img[paste_y:paste_y+sh, paste_x:paste_x+sw], 
                                  main_img[paste_y:paste_y+sh, paste_x:paste_x+sw], 
                                  mask=object_mask_inv)
    
    object_fg_roi = cv2.bitwise_and(object_crop, object_crop, mask=object_mask)
    combined_roi = cv2.add(main_bg_roi, object_fg_roi)
    
    main_img[paste_y:paste_y+sh, paste_x:paste_x+sw] = combined_roi
    
    new_cx = (paste_x + sw / 2) / W
    new_cy = (paste_y + sh / 2) / H
    new_nw = sw / W
    new_nh = sh / H
    
    new_yolo_label = f"{chosen_obj_yolo.split(' ')[0]} {new_cx:.6f} {new_cy:.6f} {new_nw:.6f} {new_nh:.6f}"
    main_yolo_labels.append(new_yolo_label)
    
    new_obj_attr = deepcopy(chosen_obj_attr)
    new_obj_attr['box'] = (paste_x, paste_y, sw, sh)
    main_objects_attrs.append(new_obj_attr)
    
    return main_img, main_yolo_labels, main_objects_attrs

# @njit is not suitable due to cv2 operations and random.randint/uniform
def occlude_image(img, occlude_p, occlude_max):
    """Applies a random rectangular occlusion to the image."""
    if random.random() > occlude_p: return img.copy()
    h,w,_=img.shape; mf=occlude_max
    ph,pw=int(h*random.uniform(0.05,mf)), int(w*random.uniform(0.05,mf))
    if ph <= 0 or pw <= 0:
        return img.copy()
    y,x=random.randint(0,h-ph), random.randint(0,w-pw)
    img_copy = img.copy()
    img_copy[y:y+ph, x:x+pw]=np.random.randint(0,255,(ph,pw,3),dtype=np.uint8)
    return img_copy

# @njit is not suitable due to cv2 operations and random.randint/uniform
def add_clutter(img, add_clutter_p, num_clutter_patches, clutter_max_factor):
    """
    Adds random clutter patches (small shapes or noise) to the image.
    """
    if random.random() > add_clutter_p: return img.copy()
    cluttered_img = img.copy()
    h, w, _ = img.shape
    for _ in range(random.randint(1, num_clutter_patches)):
        patch_h = int(h * random.uniform(0.02, clutter_max_factor))
        patch_w = int(w * random.uniform(0.02, clutter_max_factor))
        if patch_h <= 0 or patch_w <= 0:
            continue
        y = random.randint(0, h - patch_h) if h - patch_h > 0 else 0
        x = random.randint(0, w - patch_w) if w - patch_w > 0 else 0 
        if random.random() < 0.5:
            cluttered_img[y:y+patch_h, x:x+patch_w] = np.random.randint(0, 255, (ph, pw, 3), dtype=np.uint8)
        else:
            shape_type = random.choice(['circle', 'rectangle'])
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            
            if shape_type == 'circle':
                center = (x + patch_w // 2, y + patch_h // 2)
                radius = min(patch_w, patch_h) // 2
                cv2.circle(cluttered_img, center, radius, color, -1)
            else:
                cv2.rectangle(cluttered_img, (x, y), (x + patch_w, y + patch_h), color, -1)
    
    return cluttered_img

# ─── MIXUP & CUTMIX ──────────────────────────────────────────────────────────
# @njit is not suitable due to string splitting/joining and list appending
def mixup(img1, yolo1, img2, yolo2, mixup_alpha):
    """Applies MixUp augmentation."""
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    mixed_img = ((img1 * lam + img2 * (1 - lam))).astype(np.uint8)
    mixed_yolo = yolo1 + yolo2
    return mixed_img, mixed_yolo

# @njit is not suitable due to string splitting/joining and list appending
def cutmix(img1, yolo1, img2, yolo2):
    """Applies CutMix augmentation."""
    H, W, _ = img1.shape
    cut_w = int(W * random.uniform(0.1, 0.5))
    cut_h = int(H * random.uniform(0.1, 0.5))
    cut_x = random.randint(0, W - cut_w)
    cut_y = random.randint(0, H - cut_h)
    cutmixed_img = img1.copy()
    cutmixed_img[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w] = img2[cut_y:cut_y+cut_h, cut_x:cut_x+cut_w]
    cutmixed_yolo = yolo1.copy()
    for ann_str in yolo2:
        parts = list(map(float, ann_str.split()))
        class_id, cx, cy, nw, nh = parts
        
        obj_x = int((cx - nw / 2) * W)
        obj_y = int((cy - nh / 2) * H)
        obj_w = int(nw * W)
        obj_h = int(nh * H)
        if cut_x < obj_x + obj_w / 2 < cut_x + cut_w and \
           cut_y < obj_y + obj_h / 2 < cut_y + cut_h:
            cutmixed_yolo.append(ann_str)
    
    return cutmixed_img, cutmixed_yolo

# ─── PARALLEL DATA GENERATION WORKER ──────────────────────────────────────────
# This function must be at the top-level of the module for multiprocessing to pickle it.
# @njit is not suitable here due to cv2 operations, PIL, random, and file I/O
def _process_single_image_for_generation(args):
    """
    Worker function for multiprocessing data generation.
    Generates all augmented variants for a single image and saves them.
    Returns a list of difficulty scores for the generated variants.
    """
    p, split_name_for_temp_dir, current_config_dict, perlin_noise_seed, class_id_map = args  # Unpack arguments, include class_id_map
    
    # Re-initialize PerlinNoise instance within the worker process
    # This avoids pickling issues with global objects.
    perlin_instance = PerlinNoise(octaves=current_config_dict.get('fract_depth', 4), seed=perlin_noise_seed) # Use config for octaves
    
    # Initialize augmenters within the worker process using the passed config dict
    # This ensures each worker has its own instances.
    alb_transform, augmix_transform, rand_augment, auto_augment = make_augmenters(current_config_dict)
    
    stem = Path(p).stem
    # Output to the temporary directory structure
    output_root_temp = Path(current_config_dict['temp_generated_data_dir'])
    
    # Ensure base temporary directories exist for this worker
    temp_images_dir = output_root_temp / 'images' / split_name_for_temp_dir
    temp_labels_dir = output_root_temp / 'labels' / split_name_for_temp_dir
    temp_annotations_dir = output_root_temp / 'annotations' / split_name_for_temp_dir
    
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)
    temp_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    local_difficulty_scores = []
    
    try:  # Comprehensive try-except block for all image processing and augmentation
        img_raw = cv2.imread(str(p), cv2.IMREAD_COLOR)  # Convert Path to string for cv2.imread
        if img_raw is None:
            logger.warning(f"Worker could not read raw image: {p}. Skipping.")
            return []
        
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        H, W, _ = img_raw.shape
        
        # Call the refactored detect_labels_and_difficulty
        # It now returns labels, difficulty, objects_attrs_for_json, and relations (typed numpy array)
        initial_yolo_labels, initial_difficulty, initial_objects_attrs_for_json, initial_relations_typed = \
            detect_labels_and_difficulty(p, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        
        # Convert typed relations back to list of lists for JSON serialization if needed
        # Or store as is if JSON can handle numpy arrays (it generally can't directly)
        initial_relations_list = initial_relations_typed.tolist() if initial_relations_typed.size > 0 else []

        variants_data = []
        
        # --- Base Variant (Original Image) ---
        current_image = img_raw.copy()
        current_yolo_labels = deepcopy(initial_yolo_labels)
        current_objects_attrs = deepcopy(initial_objects_attrs_for_json)
        current_relations = deepcopy(initial_relations_list)
        current_difficulty = initial_difficulty
        variants_data.append((current_image, current_yolo_labels, current_objects_attrs, current_relations, current_difficulty))
        
        # --- Albumentations (Elastic, Blur, JPEG) ---
        try:
            # Prepare bboxes and class_labels for Albumentations
            alb_bboxes = []
            alb_class_labels = []
            for l in initial_yolo_labels:
                parts = list(map(float, l.split()))
                alb_class_labels.append(int(parts[0]))
                alb_bboxes.append(parts[1:]) # cx, cy, w, h
            
            augmented_alb = alb_transform(image=img_raw, bboxes=alb_bboxes, class_labels=alb_class_labels)
            
            transformed_yolo_labels = []
            for bbox, class_label in zip(augmented_alb['bboxes'], augmented_alb['class_labels']):
                cx, cy, w, h = bbox
                new_w = max(0.0, w)
                new_h = max(0.0, h)
                if new_w == 0: new_w = 1e-6
                if new_h == 0: new_h = 1e-6
                transformed_yolo_labels.append(f"{int(class_label)} {cx:.6f} {cy:.6f} {new_w:.6f} {new_h:.6f}")
            
            # For augmented images, re-calculate difficulty and attributes based on the new image
            # This involves calling `detect_labels_and_difficulty` again on the augmented image.
            # This might be computationally intensive if done for every variant.
            # A more optimized approach might track changes or apply transforms to attributes directly.
            
            # Save the augmented image to a temporary path, then run detection on it
            temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
            cv2.imwrite(str(temp_aug_path), cv2.cvtColor(augmented_alb['image'], cv2.COLOR_RGB2BGR))
            
            aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
                detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
            
            os.remove(temp_aug_path) # Clean up temp file

            variants_data.append((augmented_alb['image'], aug_labels, aug_objects_attrs, 
                                  aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        except ValueError as e:
            logger.warning(f"Albumentations transform failed for {p} (Elastic/JPEG): {e}. Skipping this variant.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during Albumentations transform for {p}: {e}. Skipping this variant.")
        
        # --- Albumentations AugMix ---
        augmented_augmix = augmix_transform(image=img_raw)['image']
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(augmented_augmix, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((augmented_augmix, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Torchvision RandAugment (requires PIL Image) ---
        pil_img = Image.fromarray(img_raw)
        augmented_ra = np.array(rand_augment(pil_img))
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(augmented_ra, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((augmented_ra, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Torchvision AutoAugment (requires PIL Image) ---
        augmented_aa = np.array(auto_augment(pil_img))
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(augmented_aa, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((augmented_aa, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Random Background ---
        bg_img = random_bg(img_raw, current_config_dict['background_root'])
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((bg_img, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Perlin Noise Background ---
        # Pass the PerlinNoise instance's callable method to the Numba-jitted function
        @njit(cache=True)
        def _numba_perlin_noise_callable(coords):
            return perlin_instance(coords)

        perlin_img_gray = perlin_bg(W, H, _numba_perlin_noise_callable)
        perlin_img = cv2.cvtColor(perlin_img_gray, cv2.COLOR_GRAY2BGR) # Convert to BGR after Numba
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(perlin_img, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((perlin_img, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Occlusion ---
        occluded_img = occlude_image(img_raw, current_config_dict['occlude_p'], current_config_dict['occlude_max'])
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(occluded_img, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((occluded_img, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Clutter ---
        cluttered_img = add_clutter(img_raw, current_config_dict['add_clutter_p'], current_config_dict['num_clutter_patches'], current_config_dict['clutter_max_factor'])
        temp_aug_path = Path(current_config_dict['temp_generated_data_dir']) / "temp_aug_img.png"
        cv2.imwrite(str(temp_aug_path), cv2.cvtColor(cluttered_img, cv2.COLOR_RGB2BGR))
        aug_labels, aug_difficulty, aug_objects_attrs, aug_relations_typed = \
            detect_labels_and_difficulty(temp_aug_path, current_config_dict, class_id_map, YOLO_CLASSES_LIST)
        os.remove(temp_aug_path)
        variants_data.append((cluttered_img, aug_labels, aug_objects_attrs, 
                              aug_relations_typed.tolist() if aug_relations_typed.size > 0 else [], aug_difficulty))
        
        # --- Fill Contour Variation (Removed for now as it relies on raw contours) ---
        # This augmentation would require re-extracting contours or passing them through,
        # which complicates the simplified `extract_props` approach.
        # If needed, it would require a more complex refactor to get contours from `skimage.measure`.
        
        # Save all generated variants to the temporary directory structure
        for i, (img_variant, yolo_labels_variant, objects_attrs_variant, relations_variant, difficulty_score_variant) in enumerate(variants_data):
            variant_stem = f"{stem}_v{i}" if i > 0 else stem
            
            # Construct paths using the pre-created temporary base directories
            temp_images_dir = Path(current_config_dict['temp_generated_data_dir']) / 'images' / split_name_for_temp_dir
            temp_labels_dir = Path(current_config_dict['temp_generated_data_dir']) / 'labels' / split_name_for_temp_dir
            temp_annotations_dir = Path(current_config_dict['temp_generated_data_dir']) / 'annotations' / split_name_for_temp_dir

            variant_img_path = temp_images_dir / f"{variant_stem}.png"
            variant_label_path = temp_labels_dir / f"{variant_stem}.txt"
            variant_anno_path = temp_annotations_dir / f"{variant_stem}.json"
            
            # Ensure image is in BGR format before saving with cv2.imwrite
            if img_variant.ndim == 3 and img_variant.shape[2] == 3:
                img_variant_bgr = cv2.cvtColor(img_variant, cv2.COLOR_RGB2BGR)
            else:
                img_variant_bgr = img_variant # Already grayscale or BGR
            
            cv2.imwrite(str(variant_img_path), img_variant_bgr)
            
            with open(variant_label_path, 'w') as f:
                # Always write, even if empty, to ensure file existence
                f.write("\n".join(yolo_labels_variant))
            
            annotation_data = {
                'original_filename': str(p),  # Convert PosixPath to string here
                'variant_id': i,
                'image_size': img_variant.shape[:2],
                'objects': objects_attrs_variant, # This now comes from the simplified detection
                'relations': relations_variant, # This is now the list from typed Numba output
                'difficulty_score': difficulty_score_variant
            }
            with open(variant_anno_path, 'w') as f:
                json.dump(annotation_data, f, indent=4)
            
            # Store difficulty score with its temporary path relative to output_root_temp
            local_difficulty_scores.append({
                'filename': str(variant_img_path.relative_to(output_root_temp)), 
                'split': split_name_for_temp_dir,  # This is 'temp'
                'difficulty_score': difficulty_score_variant
            })
        
        return local_difficulty_scores
    except Exception as e:
        # Catch any unexpected errors during the processing of this specific image
        logger.error(f"Error processing image {p} in worker: {e}", exc_info=True)
        return []  # Return an empty list of scores for this failed image

# ─── Wrapper for multiprocessing pool to save annotations (MOVED HERE) ───────────────────
# Removed @njit: This function uses OpenCV calls and Python dictionaries, not Numba-compatible in nopython mode.
def _detect_labels_and_difficulty_wrapper_for_save(args):
    path, cfg, class_id_map, yolo_classes_list = args
    
    # Call the refactored detect_labels_and_difficulty
    labels, difficulty_score, objects_attrs_for_json, relations_typed = \
        detect_labels_and_difficulty(path, cfg, class_id_map, yolo_classes_list)

    # Convert typed relations back to list of lists for JSON serialization
    relations_list = relations_typed.tolist() if relations_typed.size > 0 else []

    try:
        img_raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_raw is None:
            logger.warning(f"Could not read image for re-annotation: {path}. Skipping saving annotation.")
            return False # Indicate failure
        H, W, _ = img_raw.shape # Use original image dimensions

        annotation_data = {
            'original_filename': str(path),
            'image_size': [H, W],
            'yolo_labels_raw': labels, # Use the labels from detect_labels_and_difficulty
            'objects': objects_attrs_for_json, # Use the objects_attrs from detect_labels_and_difficulty
            'relations': relations_list, # Use the relations from detect_labels_and_difficulty
            'difficulty_score': difficulty_score
        }
        
        # Save annotation to a JSON file in the raw_annotations_dir
        anno_filename = Path(path).stem + '.json'
        anno_path = Path(cfg['raw_annotations_dir']) / anno_filename
        with open(anno_path, 'w') as f:
            json.dump(annotation_data, f, indent=4)
        
        return True # Indicate success
    except Exception as e:
        logger.error(f"Error saving annotation for {path}: {e}", exc_info=True)
        return False # Indicate failure
