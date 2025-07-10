# pipeline_workers.py
import os
import glob
import random
import json
import logging
from pathlib import Path
from copy import deepcopy
import functools # Import functools for partial

import cv2
import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise

import albumentations as A
# Import RandomAugMix from the local file
from albumentations_augmix import RandomAugMix 
# Add torchvision.transforms imports here for the worker processes
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy


# Note: CONFIG and CLASS_ID are not directly imported here.
# They are passed as arguments from the main script.

# ─── UTILITIES FOR TUNER (copied from main for self-containment) ──────────────────────────────────────────────────────
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
    path, cfg, class_id_map, yolo_classes_list = args # Added yolo_classes_list
    return detect_labels_and_difficulty(path, cfg, class_id_map, yolo_classes_list)

def detect_labels_and_difficulty(path, cfg, class_id_map, yolo_classes_list): # Accept class_id_map and yolo_classes_list
    """
    Detects labels and estimates difficulty for a given image path.
    Used by the hyperparameter tuner.
    """
    try:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            logging.getLogger().warning(f"Could not read image for difficulty estimation: {path}. Skipping.")
            return [], 0.0
        
        # Resize to a consistent size for contour detection during tuning
        H_target, W_target = cfg['image_size']
        if gray.shape[0] != H_target or gray.shape[1] != W_target:
            gray = cv2.resize(gray, (W_target, H_target), interpolation=cv2.INTER_LINEAR)

        # Use temporary config for contour detection
        temp_cnt_block = cfg.get('cnt_block')
        temp_cnt_C = cfg.get('cnt_C')
        temp_min_area = cfg.get('min_area')
        temp_max_cnt = cfg.get('max_cnt')
        temp_morph_ops = cfg.get('morphological_ops')

        cnts = detect_contours_tuned(gray, temp_cnt_block, temp_cnt_C, temp_min_area, temp_max_cnt, temp_morph_ops)
        
        labels = []
        objects_attrs = []
        
        for c in cnts:
            # Pass min_area from cfg to cnt_to_yolo
            result = cnt_to_yolo(c, gray.shape[1], gray.shape[0], class_id_map, cfg['min_area'])
            if result:
                shape, yolo_box, bbox, contour_obj = result
                labels.append(class_id_map.get(shape))
                objects_attrs.append(extract_attrs(shape, c, gray, bbox, gray.shape[1], gray.shape[0]))
        
        # Calculate difficulty components
        num_objects = len(objects_attrs)
        avg_complexity = sum(attr.get('complexity', 0) for attr in objects_attrs) / num_objects if num_objects > 0 else 0
        relations = compute_relations(objects_attrs)
        num_relations = len(relations)

        difficulty_score = (num_objects * cfg['difficulty_weights']['num_objects'] +
                            avg_complexity * cfg['difficulty_weights']['avg_complexity'] +
                            num_relations * cfg['difficulty_weights']['num_relations'])
        
        return labels, difficulty_score
    except Exception as e:
        logging.getLogger().warning(f"Failed to process {path} for difficulty estimation: {e}")
        return [], 0.0

# ─── CONTOUR & YOLO ENCODING ─────────────────────────────────────────────────
def detect_contours_tuned(gray_img, block_size, C, min_area, max_cnt, morphological_ops):
    """
    Detects contours in a grayscale image using adaptive thresholding and morphological operations.
    Uses parameters passed from the tuner.
    """
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
    return sorted(cnts, key=cv2.contourArea, reverse=True)[:max_cnt]

def cnt_to_yolo(cnt, W, H, class_id_map, min_area_threshold): # Accept class_id_map and min_area_threshold
    """
    Converts an OpenCV contour to YOLO bounding box format and extracts shape type.
    Returns (shape_type, yolo_string, bbox_tuple, contour_object).
    """
    area = cv2.contourArea(cnt)
    if area < min_area_threshold: return None # Use passed threshold
    
    x, y, w, h = cv2.boundingRect(cnt)
    peri = cv2.arcLength(cnt, True)
    
    # Shape detection based on vertices and circularity
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    num_vertices = len(approx)
    
    shape_type = 'polygon'
    if num_vertices == 3: shape_type = 'triangle'
    elif num_vertices == 4:
        aspect_ratio = w / float(h)
        shape_type = 'square' if 0.85 <= aspect_ratio <= 1.15 else 'rectangle'
    elif num_vertices == 5: shape_type = 'pentagon'
    elif num_vertices == 6: shape_type = 'hexagon'
    elif num_vertices == 8: shape_type = 'octagon'
    else:
        circularity = 4 * np.pi * area / (peri * peri) if peri else 0
        if circularity > 0.7: shape_type = 'circle'
    
    class_id = class_id_map.get(shape_type) # Use passed class_id_map
    if class_id is None: 
        logging.getLogger().debug(f"Shape '{shape_type}' not in YOLO_CLASSES. Skipping contour.")
        return None
    
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    nw, nh = w / W, h / H
    
    return shape_type, f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}", (x,y,w,h), cnt

# ─── ATTRIBUTE & RELATIONS ───────────────────────────────────────────────────
def extract_attrs(shape_type, cnt, gray_img, bbox, W, H):
    """
    Extracts attributes for a single object.
    """
    mask = np.zeros_like(gray_img)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    
    # Fill type
    filled_area_ratio = mask.sum() / (255 * bbox[2] * bbox[3]) if bbox[2]*bbox[3] > 0 else 0
    fill_type = 'outline' if filled_area_ratio < 0.2 else 'solid' if filled_area_ratio > 0.8 else 'hollow'
    
    # Stroke width (estimated)
    dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    stroke_width = float(np.median(dist_transform[dist_transform > 0])) if dist_transform.max() > 0 else 0
    
    # Size category
    area_ratio = (bbox[2] * bbox[3]) / (W * H)
    size_category = 'tiny' if area_ratio < 0.01 else \
                    'small' if area_ratio < 0.05 else \
                    'medium' if area_ratio < 0.2 else \
                    'large' if area_ratio < 0.5 else 'full_frame'
    
    # Orientation
    angle = cv2.minAreaRect(cnt)[-1]
    orientation = f"{int(round(angle/45)*45)%360}°"
    
    # Complexity (e.g., number of contour points, or perimeter/area ratio)
    complexity = len(cnt)
    
    return dict(
        class_type=shape_type,
        fill_type=fill_type,
        stroke_width=stroke_width,
        size_category=size_category,
        orientation=orientation,
        complexity=complexity,
        box=bbox
    )

def compute_relations(objects_attrs):
    """
    Computes spatial relations between objects based on their bounding boxes.
    """
    relations = []
    num_objects = len(objects_attrs)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1 = objects_attrs[i]
            obj2 = objects_attrs[j]

            x1, y1, w1, h1 = obj1['box']
            x2, y2, w2, h2 = obj2['box']

            if x1 <= x2 and y1 <= y2 and (x1 + w1) >= (x2 + w2) and (y1 + h1) >= (y2 + h2):
                relations.append({'type': 'contains', 'object1_idx': i, 'object2_idx': j})
            elif x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                relations.append({'type': 'inside', 'object1_idx': i, 'object2_idx': j})
            else:
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                if x_overlap * y_overlap > 0:
                    relations.append({'type': 'overlaps', 'object1_idx': i, 'object2_idx': j})
                else:
                    if (x1 + w1) < x2: relations.append({'type': 'left_of', 'object1_idx': i, 'object2_idx': j})
                    elif (x2 + w2) < x1: relations.append({'type': 'right_of', 'object1_idx': i, 'object2_idx': j})
                    
                    if (y1 + h1) < y2: relations.append({'type': 'above', 'object1_idx': i, 'object2_idx': j})
                    elif (y2 + h2) < y1: relations.append({'type': 'below', 'object1_idx': i, 'object2_idx': j})
    return relations

def compute_difficulty_score(attributes, relations, config):
    """
    Calculates a difficulty score for an image based on object attributes and relations.
    """
    score = 0.0
    weights = config['difficulty_weights']

    num_objects = len(attributes)
    score += weights.get('num_objects', 0) * num_objects

    if num_objects > 0:
        avg_complexity = sum(attr.get('complexity', 0) for attr in attributes) / num_objects
        normalized_complexity = min(avg_complexity / 100.0, 1.0)
        score += weights.get('avg_complexity', 0) * normalized_complexity
    
    num_relations = len(relations)
    score += weights.get('num_relations', 0) * num_relations

    return score

# ─── AUGMENTERS ───────────────────────────────────────────────────────────────
# Custom JPEG compression function using OpenCV
def _apply_jpeg_compression(image, quality_range, **kwargs): # Added **kwargs to accept additional arguments
    """Applies JPEG compression using OpenCV with a random quality from the given range."""
    # Ensure image is in BGR format for cv2.imencode
    if image.ndim == 3 and image.shape[2] == 3: # Assuming RGB input from Albumentations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image # Grayscale or already BGR
    
    quality = random.randint(quality_range[0], quality_range[1])
    # Encode as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image_bgr, encode_param)
    
    # Handle potential failure of cv2.imdecode (e.g., if encimg is empty)
    decoded_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    if decoded_img is None:
        logging.getLogger().warning("Failed to decode JPEG compressed image. Returning original image.")
        return image # Return original image if decoding fails

    # Convert back to RGB for consistency with other transforms if original was RGB
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
    return decoded_img


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
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True)) # Added clip=True

    augmix_transform = A.Compose([
        RandomAugMix(p=cfg['augmix_p'])
    ])

    rand_augment = RandAugment(num_ops=cfg['rand_nops'], magnitude=cfg['rand_mag'])
    auto_augment = AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

    return alb_transform, augmix_transform, rand_augment, auto_augment

# ─── BACKGROUNDS & TEXTURES ───────────────────────────────────────────────────
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
        logging.getLogger().error(f"Error applying random background from {bg_path}: {e}. Returning original image.")
        return img.copy()

def perlin_bg(W, H, perlin_instance):
    """Generates a Perlin noise background."""
    img = np.zeros((H,W), np.float32)
    for y in range(H):
        for x in range(W):
            img[y,x] = perlin_instance([y/H, x/W])
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# ─── COPY-PASTE, OCCLUDE, CLUTTER ─────────────────────────────────────────────
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
        logging.getLogger().warning(f"Invalid source object bbox for copy-paste: {chosen_obj_attr['box']}. Skipping.")
        return main_img, main_yolo_labels, main_objects_attrs

    object_crop = source_img[sy:sy+sh, sx:sx+sw]

    if W - sw <= 0 or H - sh <= 0:
        logging.getLogger().warning(f"Main image too small for object of size {sw}x{sh}. Skipping copy-paste.")
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
        # Corrected: Use patch_w instead of pw
        x = random.randint(0, w - patch_w) if w - patch_w > 0 else 0 

        if random.random() < 0.5:
            cluttered_img[y:y+patch_h, x:x+patch_w] = np.random.randint(0, 255, (patch_h, patch_w, 3), dtype=np.uint8)
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
def mixup(img1, yolo1, img2, yolo2, mixup_alpha):
    """Applies MixUp augmentation."""
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    mixed_img = ((img1 * lam + img2 * (1 - lam))).astype(np.uint8)
    mixed_yolo = yolo1 + yolo2
    return mixed_img, mixed_yolo

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
def _process_single_image_for_generation(args):
    """
    Worker function for multiprocessing data generation.
    Generates all augmented variants for a single image and saves them.
    Returns a list of difficulty scores for the generated variants.
    """
    p, split_name_for_temp_dir, current_config_dict, perlin_noise_seed, class_id_map = args # Unpack arguments, include class_id_map
    
    # Re-initialize PerlinNoise instance within the worker process
    # This avoids pickling issues with global objects.
    perlin_instance = PerlinNoise(octaves=4, seed=perlin_noise_seed)

    # Initialize augmenters within the worker process using the passed config dict
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

    try: # Comprehensive try-except block for all image processing and augmentation
        img_raw = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_raw is None:
            logging.getLogger().warning(f"Worker could not read raw image: {p}. Skipping.")
            return []
        
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        H, W, _ = img_raw.shape
        
        gray_img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
        
        contours = detect_contours_tuned(gray_img, current_config_dict['cnt_block'], current_config_dict['cnt_C'],
                                         current_config_dict['min_area'], current_config_dict['max_cnt'], current_config_dict['morphological_ops'])
        initial_yolo_labels = []
        initial_objects_attrs = []
        initial_contours_objects = []
        
        for c in contours:
            result = cnt_to_yolo(c, W, H, class_id_map, current_config_dict['min_area']) # Pass min_area
            if result:
                shape, yolo_box_str, bbox_tuple, contour_obj = result
                initial_yolo_labels.append(yolo_box_str)
                initial_objects_attrs.append(extract_attrs(shape, c, gray_img, bbox_tuple, W, H))
                initial_contours_objects.append(contour_obj)

        # Removed: if not initial_yolo_labels: return []
        # This ensures that an image and its (potentially empty) label file are always saved.

        initial_relations = compute_relations(initial_objects_attrs)

        variants_data = []

        # --- Base Variant (Original Image) ---
        current_image = img_raw.copy()
        current_yolo_labels = deepcopy(initial_yolo_labels)
        current_objects_attrs = deepcopy(initial_objects_attrs)
        current_relations = deepcopy(initial_relations)
        current_difficulty = compute_difficulty_score(current_objects_attrs, current_relations, current_config_dict)
        variants_data.append((current_image, current_yolo_labels, current_objects_attrs, current_relations, current_difficulty))

        # --- Albumentations (Elastic, Blur, JPEG) ---
        try:
            # Correctly unpack bboxes and class_labels from the augmented result
            augmented_alb = alb_transform(image=img_raw, bboxes=[list(map(float, l.split()[1:])) for l in initial_yolo_labels], class_labels=[int(l.split()[0]) for l in initial_yolo_labels])
            
            transformed_yolo_labels = []
            for bbox, class_label in zip(augmented_alb['bboxes'], augmented_alb['class_labels']):
                # bbox is (cx, cy, w, h) in normalized [0, 1] format
                cx, cy, w, h = bbox

                new_w = max(0.0, w)
                new_h = max(0.0, h)

                # If box collapsed to zero, make it a tiny box to avoid division by zero later
                if new_w == 0: new_w = 1e-6
                if new_h == 0: new_h = 1e-6

                transformed_yolo_labels.append(f"{int(class_label)} {cx:.6f} {cy:.6f} {new_w:.6f} {new_h:.6f}")
            
            variants_data.append((augmented_alb['image'], transformed_yolo_labels, deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))
        except ValueError as e:
            logging.getLogger().warning(f"Albumentations transform failed for {p} (Elastic/JPEG): {e}. Skipping this variant.")
        except Exception as e:
            logging.getLogger().error(f"An unexpected error occurred during Albumentations transform for {p}: {e}. Skipping this variant.")


        # --- Albumentations AugMix ---
        augmented_augmix = augmix_transform(image=img_raw)['image']
        variants_data.append((augmented_augmix, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Torchvision RandAugment (requires PIL Image) ---
        pil_img = Image.fromarray(img_raw)
        augmented_ra = np.array(rand_augment(pil_img))
        variants_data.append((augmented_ra, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Torchvision AutoAugment (requires PIL Image) ---
        augmented_aa = np.array(auto_augment(pil_img))
        variants_data.append((augmented_aa, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Random Background ---
        bg_img = random_bg(img_raw, current_config_dict['background_root'])
        variants_data.append((bg_img, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Perlin Noise Background ---
        perlin_img = perlin_bg(W, H, perlin_instance)
        variants_data.append((perlin_img, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Occlusion ---
        occluded_img = occlude_image(img_raw, current_config_dict['occlude_p'], current_config_dict['occlude_max'])
        variants_data.append((occluded_img, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Clutter ---
        cluttered_img = add_clutter(img_raw, current_config_dict['add_clutter_p'], current_config_dict['num_clutter_patches'], current_config_dict['clutter_max_factor'])
        variants_data.append((cluttered_img, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))

        # --- Fill Contour Variation (augmentation) ---
        if random.random() < current_config_dict['fill_contour_p']:
            filled_img = img_raw.copy()
            for c_obj in initial_contours_objects:
                mask_for_color = np.zeros(filled_img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_for_color, [c_obj], -1, 255, cv2.FILLED)
                mean_color_of_object = cv2.mean(filled_img, mask=mask_for_color)[:3]
                fill_color = (int(mean_color_of_object[0]), int(mean_color_of_object[1]), int(mean_color_of_object[2]))
                cv2.drawContours(filled_img, [c_obj], -1, fill_color, cv2.FILLED)
            variants_data.append((filled_img, deepcopy(initial_yolo_labels), deepcopy(initial_objects_attrs), deepcopy(initial_relations), compute_difficulty_score(initial_objects_attrs, initial_relations, current_config_dict)))
        
        # Save all generated variants to the temporary directory structure
        for i, (img_variant, yolo_labels_variant, objects_attrs_variant, relations_variant, difficulty_score_variant) in enumerate(variants_data):
            variant_stem = f"{stem}_v{i}" if i > 0 else stem
            
            # Construct paths using the pre-created temporary base directories
            variant_img_path = temp_images_dir / f"{variant_stem}.png"
            variant_label_path = temp_labels_dir / f"{variant_stem}.txt"
            variant_anno_path = temp_annotations_dir / f"{variant_stem}.json"

            cv2.imwrite(str(variant_img_path), cv2.cvtColor(img_variant, cv2.COLOR_RGB2BGR))
            
            with open(variant_label_path, 'w') as f:
                # Always write, even if empty, to ensure file existence
                f.write("\n".join(yolo_labels_variant))
            
            annotation_data = {
                'original_filename': p,
                'variant_id': i,
                'image_size': img_variant.shape[:2],
                'objects': objects_attrs_variant,
                'relations': relations_variant,
                'difficulty_score': difficulty_score_variant
            }
            with open(variant_anno_path, 'w') as f:
                json.dump(annotation_data, f, indent=4)
            
            # Store difficulty score with its temporary path relative to output_root_temp
            local_difficulty_scores.append({
                'filename': str(variant_img_path.relative_to(output_root_temp)), 
                'split': split_name_for_temp_dir, # This is 'temp'
                'difficulty_score': difficulty_score_variant
            })
        
        return local_difficulty_scores
    except Exception as e:
        # Catch any unexpected errors during the processing of this specific image
        logging.getLogger().error(f"Error processing image {p} in worker: {e}", exc_info=True)
        return [] # Return an empty list of scores for this failed image
