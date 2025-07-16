import os
import json
import shutil
import random
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
from PIL import Image
import yaml
from pathlib import Path
import logging
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallel processing

# Import PyTorch for GAN and AMP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.cuda import amp # For Automatic Mixed Precision
    TORCH_AVAILABLE = True
    logging.info("PyTorch found and imported.")
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not found. GAN and mixed precision functionalities will be limited.")

# Local imports (assuming these are in the same project structure)
from .logger import setup_logging, log_detection
from .metrics import Evaluator # Assuming Evaluator is in metrics.py
from .gan_model import GANModel # Import the new GANModel class

# Import EmbedDetector, auto_label, pseudo_labeling, active_learning_step from auto_labeling.py
try:
    from .auto_labeling import EmbedDetector, auto_label, pseudo_labeling, active_learning_step, map_owlvit_label_to_dataset_class_id, load_class_map
    # Also import SamPredictor and sam_model_registry if auto_labeling uses them
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    EmbedDetector = None
    auto_label = None
    pseudo_labeling = None
    active_learning_step = None
    map_owlvit_label_to_dataset_class_id = None
    load_class_map = None
    SamPredictor = None
    sam_model_registry = None
    logging.warning("auto_labeling module or SAM not found. OWL-ViT and related functionalities will be limited.")

# Import split_dataset from split_dataset.py
try:
    from .split_dataset import split_dataset
except ImportError:
    split_dataset = None
    logging.warning("split_dataset module not found. Dataset splitting will be unavailable.")

# Import DALI pipeline functions from dali_pipeline.py
try:
    from .dali_pipeline import prepare_dataset as dali_prepare_dataset
    DALI_AVAILABLE = True
except ImportError:
    dali_prepare_dataset = None
    DALI_AVAILABLE = False
    logging.warning("dali_pipeline module not found. DALI pipelines will be unavailable.")


# --- Utility: Letterbox and Directory Creation ---
def letterbox(img, target_size, color=(114,114,114)):
    """
    Resizes and pads an image to a target square size, maintaining aspect ratio.
    Args:
        img (np.array): Input image.
        target_size (int): Desired square size (e.g., 640).
        color (tuple): Border color (RGB).
    Returns:
        tuple: (padded_img, scale, (pad_left, pad_top))
    """
    h, w = img.shape[:2]
    scale = min(target_size/h, target_size/w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w, pad_h = target_size - new_w, target_size - new_h
    top, bottom = pad_h//2, pad_h - pad_h//2
    left, right = pad_w//2, pad_w - pad_w//2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (left, top)

def make_dirs(path):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)

# --- Copy-Paste Synthesis Utilities ---
def load_object_masks(mask_dir):
    """
    Loads object masks from a directory.
    Returns list of (class_name_str, mask_array).
    """
    objects = []
    if not os.path.exists(mask_dir):
        logging.warning(f"Object mask directory not found: {mask_dir}. Skipping mask loading.")
        return objects
    
    try:
        for fname in os.listdir(mask_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')): # Allow image formats for masks
                mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE) # Load as grayscale
                if mask is not None:
                    # Use stem for class name, assuming filename is class_name.png
                    objects.append((Path(fname).stem, mask)) 
                else:
                    logging.warning(f"Could not load mask: {fname}")
    except Exception as e:
        logging.error(f"Failed to load object masks from {mask_dir}: {e}")
    return objects

def paste_objects(bg_image, bg_labels, objects_with_masks, class_names, max_pastes=3):
    """
    Pastes random objects onto a background image using their masks.
    Assumes objects are white where mask is non-zero for blending, or uses the mask as intensity.
    Args:
        bg_image (np.array): Background image (H, W, 3).
        bg_labels (list): List of existing labels on the background in YOLO format (class_id, cx, cy, w, h).
        objects_with_masks (list): List of (class_name_str, mask_array) tuples.
        class_names (list): List of all class names to map class_name_str to class_id.
        max_pastes (int): Maximum number of objects to paste.
    Returns:
        tuple: (new_image (np.array), updated_labels (list))
    """
    h_bg, w_bg, _ = bg_image.shape
    new_labels = [list(map(float, l)) if isinstance(l, list) else l for l in bg_labels] # Ensure labels are mutable lists of floats
    result_image = bg_image.copy()

    if not objects_with_masks:
        logging.warning("No object masks available for copy-paste synthesis. Returning original image.")
        return result_image, new_labels

    num_objects_to_paste = random.randint(1, max_pastes)
    
    for _ in range(num_objects_to_paste):
        try:
            obj_name_str, obj_mask_full = random.choice(objects_with_masks)
            
            # Find bounding box of the non-zero part of the mask
            y_idxs, x_idxs = np.where(obj_mask_full > 0)
            if len(x_idxs) == 0 or len(y_idxs) == 0:
                logging.warning(f"Empty mask for object: {obj_name_str}. Skipping paste.")
                continue
            
            x_min_obj, y_min_obj = np.min(x_idxs), np.min(y_idxs)
            x_max_obj, y_max_obj = np.max(x_idxs), np.max(y_idxs)
            
            obj_w_orig = x_max_obj - x_min_obj + 1
            obj_h_orig = y_max_obj - y_min_obj + 1

            # Extract the relevant part of the mask (object itself, assuming mask is the object)
            # If mask is just binary, we'd need to define object color. Let's assume mask is grayscale intensity.
            obj_roi = obj_mask_full[y_min_obj:y_max_obj+1, x_min_obj:x_max_obj+1]
            
            # Resize object mask to a random scale
            scale_factor = random.uniform(0.5, 1.5) 
            new_obj_w = int(obj_w_orig * scale_factor)
            new_obj_h = int(obj_h_orig * scale_factor)

            if new_obj_w <= 0 or new_obj_h <= 0:
                logging.warning(f"Scaled object size is zero or negative for {obj_name_str}. Skipping paste.")
                continue

            obj_roi_resized = cv2.resize(obj_roi, (new_obj_w, new_obj_h), interpolation=cv2.INTER_LINEAR)
            alpha_channel = (obj_roi_resized.astype(np.float32) / 255.0) # Normalize mask intensity to [0, 1] as alpha

            # Determine random paste location
            if new_obj_w >= w_bg or new_obj_h >= h_bg:
                logging.warning(f"Object {obj_name_str} ({new_obj_w}x{new_obj_h}) is larger than or equal to background ({w_bg}x{h_bg}). Skipping paste.")
                continue

            paste_x = random.randint(0, w_bg - new_obj_w)
            paste_y = random.randint(0, h_bg - new_obj_h)

            # Extract background region where object will be pasted
            bg_region = result_image[paste_y:paste_y+new_obj_h, paste_x:paste_x+new_obj_w]

            # Convert grayscale object ROI to 3 channels (assuming it should be white or original color)
            # If obj_roi_resized is meant to be the object's pixel values, then use it directly.
            # If it's a binary mask, we can make the object white.
            object_image_rgb = cv2.cvtColor(obj_roi_resized, cv2.COLOR_GRAY2BGR) # Convert to 3 channels
            
            # Blending formula: result = alpha * foreground + (1 - alpha) * background
            # Ensure alpha_channel is (H, W) or (H, W, 1) and broadcastable
            alpha_3_channel = np.expand_dims(alpha_channel, axis=-1) # (H, W, 1)

            blended_region = (alpha_3_channel * object_image_rgb + (1 - alpha_3_channel) * bg_region).astype(np.uint8)
            result_image[paste_y:paste_y+new_obj_h, paste_x:paste_x+new_obj_w] = blended_region

            # Update labels
            class_id = -1
            try:
                class_id = class_names.index(obj_name_str)
            except ValueError:
                logging.warning(f"Class name '{obj_name_str}' not found in class_names list. Skipping label for this object.")
                continue

            # Convert pasted object's pixel coordinates to YOLO format
            cx = (paste_x + new_obj_w / 2) / w_bg
            cy = (paste_y + new_obj_h / 2) / h_bg
            w_norm = new_obj_w / w_bg
            h_norm = new_obj_h / h_bg
            
            # Ensure normalized coordinates are within [0, 1]
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            w_norm = np.clip(w_norm, 0.0, 1.0)
            h_norm = np.clip(h_norm, 0.0, 1.0)

            new_labels.append([float(class_id), cx, cy, w_norm, h_norm])

        except Exception as e:
            logging.error(f"Error during single object paste in copy-paste synthesis: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next object, don't stop the whole process

    return result_image, new_labels

def generate_synthetic_data_copy_paste(
    background_image_paths: list, 
    background_label_paths: list, 
    objects_with_masks: list, 
    class_names: list, 
    n_samples: int, 
    out_img_dir: str, 
    out_lbl_dir: str, 
    max_pastes: int = 3,
    num_workers: int = 1
):
    """
    Generates synthetic images using copy-paste synthesis.
    Args:
        background_image_paths (list): Paths to images to use as backgrounds.
        background_label_paths (list): Paths to labels for background images.
        objects_with_masks (list): List of (class_name_str, mask_array) tuples for objects to paste.
        class_names (list): List of all class names for mapping.
        n_samples (int): Number of synthetic samples to generate.
        out_img_dir (str): Output directory for generated images.
        out_lbl_dir (str): Output directory for generated labels.
        max_pastes (int): Maximum number of objects to paste per image.
        num_workers (int): Number of parallel workers for processing.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    logging.info(f"Generating {n_samples} synthetic samples using copy-paste synthesis into {out_img_dir} and {out_lbl_dir}...")

    if not background_image_paths or not objects_with_masks:
        logging.warning("Insufficient background images or object masks for copy-paste synthesis. Skipping.")
        return

    # Create a map for quick lookup of background labels
    bg_label_map = {Path(p).stem: p for p in background_label_paths}

    # Prepare arguments for parallel processing
    tasks = []
    for i in range(n_samples):
        # Choose a random background image and its labels
        bg_img_path = random.choice(background_image_paths)
        bg_stem = Path(bg_img_path).stem
        bg_lbl_path = bg_label_map.get(bg_stem)

        if not bg_lbl_path or not os.path.exists(bg_lbl_path):
            logging.warning(f"No label file found for background image {bg_img_path}. Skipping this background.")
            continue
        
        # Output paths for the new synthetic image and label
        output_img_path = os.path.join(out_img_dir, f'cp_synth_{i:05d}.jpg')
        output_lbl_path = os.path.join(out_lbl_dir, f'cp_synth_{i:05d}.txt')

        tasks.append((bg_img_path, bg_lbl_path, objects_with_masks, class_names, max_pastes, output_img_path, output_lbl_path))

    if not tasks:
        logging.warning("No valid tasks generated for copy-paste synthesis. Check background images and labels.")
        return

    def _process_single_copy_paste_task(task_args):
        bg_img_path, bg_lbl_path, objects_with_masks_worker, class_names_worker, max_pastes_worker, output_img_path, output_lbl_path = task_args
        
        try:
            bg_img = cv2.imread(bg_img_path)
            if bg_img is None:
                logging.warning(f"Worker: Could not read background image {bg_img_path}. Skipping.")
                return False

            bg_labels = []
            with open(bg_lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        bg_labels.append(list(map(float, parts))) # Ensure labels are lists of floats

            augmented_img, augmented_labels = paste_objects(
                bg_image=bg_img,
                bg_labels=bg_labels,
                objects_with_masks=objects_with_masks_worker,
                class_names=class_names_worker,
                max_pastes=max_pastes_worker
            )
            
            cv2.imwrite(output_img_path, augmented_img)
            with open(output_lbl_path, 'w') as f:
                for label_data in augmented_labels:
                    f.write(f"{int(label_data[0])} {label_data[1]:.6f} {label_data[2]:.6f} {label_data[3]:.6f} {label_data[4]:.6f}\n")
            return True
        except Exception as e:
            logging.error(f"Worker: Error processing copy-paste task for {bg_img_path}: {e}")
            import traceback
            traceback.print_exc()
            return False

    processed_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_single_copy_paste_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copy-Paste Synthesis"):
            if future.result():
                processed_count += 1
    
    logging.info(f"Copy-paste synthesis complete. Successfully generated {processed_count} samples.")


# --- Data Validation and Statistics ---
def check_bbox_integrity(img_w, img_h, x_center, y_center, w, h):
    """
    Checks if a YOLO format bounding box is valid and within image bounds.
    Normalized coordinates are expected [0, 1].
    """
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return False # Basic range check

    # Convert to absolute pixel coordinates for more precise boundary check
    x1_abs = (x_center - w/2) * img_w
    y1_abs = (y_center - h/2) * img_h
    x2_abs = (x_center + w/2) * img_w
    y2_abs = (y_center + h/2) * img_h

    # Check if the box corners are within image dimensions
    return 0 <= x1_abs < x2_abs <= img_w and 0 <= y1_abs <= y2_abs <= img_h # Fixed y2_abs check

def validate_image_and_labels(img_path, lbl_path):
    """
    Validates an image and its corresponding YOLO label file.
    Checks if image exists, can be read, and if bounding boxes in label file are valid.
    """
    img = cv2.imread(img_path)
    if img is None:
        logging.warning(f"Could not read image: {img_path}")
        return False
    h, w = img.shape[:2]

    try:
        if not os.path.exists(lbl_path):
            logging.warning(f"Label file not found for {img_path}: {lbl_path}")
            return False

        with open(lbl_path, 'r') as f:
            lines = f.read().strip().splitlines()

        if not lines:
            # logging.info(f"Label file is empty for {img_path}: {lbl_path}")
            return True # An empty label file is valid if there are no objects

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                logging.warning(f"Malformed label line in {lbl_path}: '{line}'")
                return False
            try:
                cls_id, x, y, bw, bh = map(float, parts)
            except ValueError:
                logging.warning(f"Invalid numeric value in label line in {lbl_path}: '{line}'")
                return False

            if not check_bbox_integrity(w, h, x, y, bw, bh):
                logging.warning(f"Invalid bounding box in {lbl_path}: [{x}, {y}, {bw}, {bh}] for image size ({w}, {h})")
                return False
        return True
    except Exception as e:
        logging.error(f"Error validating labels for {img_path} ({lbl_path}): {e}")
        return False

def filter_valid_samples(images, labels, num_workers: int = 1):
    """
    Filters out image-label pairs that fail validation using a ThreadPoolExecutor.
    """
    valid_imgs = []
    valid_lbls = []
    
    if not images or not labels:
        return [], []

    logging.info(f"Validating {len(images)} samples with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(validate_image_and_labels, img, lbl): (img, lbl)
                   for img, lbl in zip(images, labels)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating Samples"):
            is_valid = future.result() # validate_image_and_labels returns a boolean
            if is_valid:
                valid_imgs.append(futures[future]) # Get the (img, lbl) tuple back
                valid_lbls.append(futures[future][1]) # Get the label path
    
    # Reconstruct valid_imgs and valid_lbls from the successful futures
    final_valid_imgs = [item[0] for item in valid_imgs]
    final_valid_lbls = [item[1] for item in valid_imgs] # Corrected to use the stored label path

    logging.info(f"Filtered {len(images) - len(final_valid_imgs)} invalid samples. {len(final_valid_imgs)} valid samples remaining.")
    return final_valid_imgs, final_valid_lbls


def compute_class_balance(lbl_paths, n_classes):
    """Computes the class distribution across label files."""
    counts = Counter()
    for lp in lbl_paths:
        try:
            with open(lp, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        counts[cls_id] += 1
        except Exception as e:
            logging.warning(f"Could not read class from label file {lp}: {e}")
            continue
    total = sum(counts.values())
    balance = {c: counts[c]/total if total > 0 else 0 for c in range(n_classes)}
    return balance

def _compute_single_image_mean_std(img_path):
    """Internal helper for parallel mean/std computation."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = img.astype(np.float32)/255.0 # Normalize to [0, 1]
    return img.mean(axis=(0,1)), img.std(axis=(0,1))

def compute_mean_std(image_paths, num_samples=500, num_workers: int = 1):
    """Computes mean and standard deviation of image pixel values using a ThreadPoolExecutor."""
    sampled = random.sample(image_paths, min(num_samples, len(image_paths)))
    
    means_list = []
    stds_list = []
    
    logging.info(f"Computing mean/std on {len(sampled)} samples with {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_compute_single_image_mean_std, img_path): img_path
                   for img_path in sampled}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Mean/Std"):
            result = future.result()
            if result is not None:
                mean, std = result
                means_list.append(mean)
                stds_list.append(std)
    
    if means_list:
        mean = np.mean(means_list, axis=0)
        std = np.mean(stds_list, axis=0) # Average stds, or compute overall std from all pixels
    else:
        mean = np.zeros(3)
        std = np.zeros(3)
    
    return mean.tolist(), std.tolist()

# --- Test-Time Augmentation (TTA) ---
def test_time_augmentation(image, model, tta_transforms=None, detection_prompts=None):
    """
    Applies TTA (e.g., horizontal flip, scale) and potentially averages predictions.
    This is a placeholder; actual implementation depends on the model's prediction interface.
    Args:
        image (np.array or PIL.Image): Input image.
        model: Trained detection model with a .detect(image) method.
        tta_transforms (list): List of transformation functions to apply.
        detection_prompts (list): Prompts for the detection model.
    Returns:
        list: List of predictions from each augmented view.
    """
    if model is None or detection_prompts is None:
        logging.warning("Model or detection prompts not provided for Test-Time Augmentation. Skipping TTA.")
        return []

    if tta_transforms is None:
        # Default TTA: original and horizontal flip
        tta_transforms = [lambda x: x, lambda x: cv2.flip(x, 1) if isinstance(x, np.ndarray) else x.transpose(Image.FLIP_LEFT_RIGHT)]

    all_preds = []
    for t in tta_transforms:
        try:
            aug_img = t(image)
            # Ensure the augmented image is in the format expected by model.detect
            if isinstance(image, Image.Image) and isinstance(aug_img, np.ndarray):
                aug_img = Image.fromarray(aug_img)
            elif isinstance(image, np.ndarray) and isinstance(aug_img, Image.Image):
                aug_img = Image.fromarray(aug_img) # Convert to PIL Image for OWL-ViT

            # Use autocast for mixed precision if CUDA is available
            with amp.autocast() if TORCH_AVAILABLE and model.device == 'cuda' else torch.no_grad():
                preds = model.detect(aug_img, prompts=detection_prompts) # Assuming OWL-ViT detect interface
            all_preds.append(preds)
        except Exception as e:
            logging.error(f"Error during TTA application: {e}")
            continue
    logging.info(f"Test-Time Augmentation applied, collected {len(all_preds)} sets of predictions.")
    return all_preds

# --- Synthetic Data Generation ---
def generate_synthetic_data(generator_model_instance: GANModel, n_samples: int, batch_size: int, 
                            out_img_dir: str, out_lbl_dir: str,
                            image_size: tuple = (640, 640), class_names: list = None,
                            embed_detector_instance: EmbedDetector = None, detection_prompts: list = None,
                            class_map: dict = None, detection_threshold: float = 0.3):
    """
    Generates synthetic images using a generative model and pseudo-labels them using an object detector.
    Args:
        generator_model_instance: An instantiated GANModel with a `generate_images_batch()` method.
        n_samples (int): Total number of synthetic samples to generate.
        batch_size (int): Number of images to generate/process in each batch.
        out_img_dir (str): Output directory for generated images.
        out_lbl_dir (str): Output directory for generated labels.
        image_size (tuple): (width, height) for generated images.
        class_names (list): List of class names for the dataset.
        embed_detector_instance (EmbedDetector): An initialized OWL-ViT EmbedDetector for pseudo-labeling.
        detection_prompts (list): Prompts for the EmbedDetector.
        class_map (dict): Class mapping for EmbedDetector.
        detection_threshold (float): Confidence threshold for EmbedDetector pseudo-labels.
    """
    if generator_model_instance is None:
        logging.warning("[WARN] No generator model instance provided for synthetic data. Skipping generation.")
        return

    if embed_detector_instance is None:
        logging.warning("[WARN] No EmbedDetector instance provided for pseudo-labeling synthetic data. Labels will be empty.")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    logging.info(f"Generating {n_samples} synthetic samples in batches of {batch_size} into {out_img_dir} and {out_lbl_dir}...")

    num_batches = (n_samples + batch_size - 1) // batch_size
    generated_count = 0

    for batch_idx in tqdm(range(num_batches), desc="Generating & Pseudo-labeling Synthetic Data"):
        current_batch_size = min(batch_size, n_samples - generated_count)
        if current_batch_size <= 0:
            break

        try:
            # 1. Generate a batch of images from the GAN
            # Use autocast for mixed precision if CUDA is available
            with amp.autocast() if TORCH_AVAILABLE and generator_model_instance.device == 'cuda' else torch.no_grad():
                gen_img_pil_batch = generator_model_instance.generate_images_batch(
                    batch_size=current_batch_size,
                    image_size=image_size
                ) 

            # 2. Pseudo-labeling the synthetic image batch using EmbedDetector
            pseudo_labels_batch = []
            if embed_detector_instance and detection_prompts and class_names and class_map:
                try:
                    # NOTE: embed_detector_instance.detect currently takes a single image.
                    # For true batch inference, EmbedDetector.detect would need to be modified
                    # to accept a list of PIL Images and return batched results.
                    # For now, we loop through the batch for detection.
                    # This is still faster due to GPU operations on the generated images.
                    current_batch_pseudo_labels = []
                    for gen_img_pil in gen_img_pil_batch:
                        img_w, img_h = gen_img_pil.size
                        with amp.autocast() if TORCH_AVAILABLE and embed_detector_instance.device == 'cuda' else torch.no_grad():
                            boxes_xyxy, scores, pred_label_indices, _ = embed_detector_instance.detect(gen_img_pil, detection_prompts)
                        
                        single_image_labels = []
                        for bbox_xyxy, score, owlvit_label_index in zip(boxes_xyxy, scores, pred_label_indices):
                            if score >= detection_threshold:
                                owlvit_prompt_text = detection_prompts[owlvit_label_index]
                                mapped_class_id = map_owlvit_label_to_dataset_class_id(
                                    owlvit_label_index, owlvit_prompt_text, class_map, class_names
                                )
                                
                                if mapped_class_id != -1: # Only include if a valid class mapping exists
                                    # Convert bbox_xyxy to YOLO format
                                    x1, y1, x2, y2 = bbox_xyxy
                                    cx = ((x1 + x2) / 2) / img_w
                                    cy = ((y1 + y2) / 2) / img_h
                                    w = (x2 - x1) / img_w
                                    h = (y2 - y1) / img_h
                                    
                                    # Clip to [0,1] to ensure valid YOLO coordinates
                                    cx = np.clip(cx, 0.0, 1.0)
                                    cy = np.clip(cy, 0.0, 1.0)
                                    w = np.clip(w, 0.0, 1.0)
                                    h = np.clip(h, 0.0, 1.0)

                                    single_image_labels.append([mapped_class_id, cx, cy, w, h])
                        current_batch_pseudo_labels.append(single_image_labels)
                    pseudo_labels_batch = current_batch_pseudo_labels
                    logging.debug(f"Pseudo-labeled batch {batch_idx} with objects.")
                except Exception as e:
                    logging.error(f"Error pseudo-labeling synthetic image batch {batch_idx} with EmbedDetector: {e}")
                    pseudo_labels_batch = [[] for _ in range(current_batch_size)] # Fallback to empty labels
            else:
                logging.warning(f"EmbedDetector not available for synthetic data. Generating empty label files for batch {batch_idx}.")
                pseudo_labels_batch = [[] for _ in range(current_batch_size)]


            # 3. Save images and their pseudo-labels
            for k in range(current_batch_size):
                img_path = os.path.join(out_img_dir, f'synth_{generated_count + k:05d}.jpg')
                lbl_path = os.path.join(out_lbl_dir, f'synth_{generated_count + k:05d}.txt')
                
                gen_img_pil_batch[k].save(img_path)
                with open(lbl_path, 'w') as f:
                    for obj_data in pseudo_labels_batch[k]:
                        f.write(f"{int(obj_data[0])} {obj_data[1]:.6f} {obj_data[2]:.6f} {obj_data[3]:.6f} {obj_data[4]:.6f}\n")
            
            generated_count += current_batch_size

        except Exception as e:
            logging.error(f"Critical error during synthetic data generation batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            break # Stop if a critical error occurs

    logging.info(f"[INFO] Generated and pseudo-labeled {generated_count} synthetic samples in {out_img_dir} and {out_lbl_dir}.")

# --- Hard Negative Mining ---
def _process_single_hnm(img_path, lbl_path, model, detection_prompts, threshold):
    """Internal helper for parallel hard negative mining."""
    try:
        image_pil = Image.open(img_path).convert('RGB')
        # Get predictions from the model
        # Use autocast for mixed precision if CUDA is available
        with amp.autocast() if TORCH_AVAILABLE and model.device == 'cuda' else torch.no_grad():
            boxes, scores, _, _ = model.detect(image_pil, detection_prompts)

        # Load ground truth boxes
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append(list(map(float, parts[1:]))) # YOLO format: cx, cy, w, h

        # Identify false positives (predictions with high confidence but no good GT match)
        fp_count = 0
        if len(boxes) > 0 and len(gt_boxes) > 0:
            h_img, w_img = image_pil.height, image_pil.width
            gt_boxes_xyxy = []
            for gt_cx, gt_cy, gt_w, gt_h in gt_boxes:
                gt_x1 = (gt_cx - gt_w/2) * w_img
                gt_y1 = (gt_cy - gt_h/2) * h_img
                gt_x2 = (gt_cx + gt_w/2) * w_img
                gt_y2 = (gt_cy + gt_h/2) * h_img
                gt_boxes_xyxy.append([gt_x1, gt_y1, gt_x2, gt_y2])

            for i, pred_box in enumerate(boxes):
                if scores[i] >= threshold: # Consider only confident predictions
                    matched = False
                    for gt_box in gt_boxes_xyxy:
                        if _iou_xyxy(pred_box, gt_box) > 0.5: # IoU threshold for a match
                            matched = True
                            break
                    if not matched:
                        fp_count += 1
        elif len(boxes) > 0 and len(gt_boxes) == 0:
            fp_count = sum(1 for s in scores if s >= threshold)

        if fp_count > 0 or (len(scores) > 0 and np.max(scores) < 0.5):
            return img_path # Identified as hard negative
        return None
    except Exception as e:
        logging.error(f"Error processing {img_path} for hard negative mining: {e}")
        return None

def hard_negative_mining(image_paths, label_paths, model, detection_prompts: list, threshold=0.3, num_workers: int = 1):
    """
    Finds hard negatives (false positives or high uncertainty detections) for curriculum learning.
    Uses ThreadPoolExecutor for parallel processing.
    """
    if model is None or detection_prompts is None:
        logging.warning("Model or detection prompts not provided for hard negative mining. Skipping.")
        return []

    hard_negatives = []
    logging.info(f"Starting hard negative mining with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_hnm, img, lbl, model, detection_prompts, threshold): img
                   for img, lbl in zip(image_paths, label_paths)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Hard Negative Mining"):
            hn_img_path = future.result()
            if hn_img_path:
                hard_negatives.append(hn_img_path)

    logging.info(f"[INFO] Found {len(hard_negatives)} hard negatives.")
    return hard_negatives

def _iou_xyxy(boxA, boxB):
    """Calculates IoU for boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = float(boxA_area + boxB_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


# --- Label Noise Detection and Correction ---
def _yolo_iou(boxA, boxB):
    """
    Calculates IoU for boxes in YOLO format (cx, cy, w, h).
    """
    def to_corners(x, y, w, h):
        x1 = x - w/2; y1 = y - h/2; x2 = x + w/2; y2 = y + h/2
        return x1, y1, x2, y2

    a1, a2, a3, a4 = to_corners(*boxA)
    b1, b2, b3, b4 = to_corners(*boxB)

    xi1, yi1 = max(a1, b1), max(a2, b2)
    xi2, yi2 = min(a3, b3), min(a4, b4)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    areaA = (a3 - a1) * (a4 - a2)
    areaB = (b3 - b1) * (b4 - b2)

    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def _detect_single_label_noise(lbl_path, iou_thresh, min_area):
    """Internal helper for parallel label noise detection."""
    bboxes = []
    issues_found = []
    try:
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            return lbl_path, issues_found # Empty file, no noise to detect

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                issues_found.append('Malformed line')
                continue
            try:
                cls_id, x, y, w, h = map(float, parts)
            except ValueError:
                issues_found.append('Invalid numeric value')
                continue

            bbox_yolo = (x, y, w, h)
            bboxes.append(bbox_yolo)

            if w * h < min_area:
                issues_found.append('Tiny box')

        # Check for duplicate boxes (high IoU)
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = _yolo_iou(bboxes[i], bboxes[j])
                if iou > iou_thresh:
                    issues_found.append('Duplicate/overlapping boxes')

    except Exception as e:
        issues_found.append(f'Processing error: {e}')
    return lbl_path, issues_found

def detect_label_noise(label_paths, iou_thresh=0.7, min_area=0.001, num_workers: int = 1):
    """
    Detects label noise: duplicate boxes, tiny boxes, and potentially other issues.
    Uses ThreadPoolExecutor for parallel processing.
    """
    issues = defaultdict(list)
    logging.info(f"Detecting label noise with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_detect_single_label_noise, lbl_path, iou_thresh, min_area): lbl_path
                   for lbl_path in label_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting Label Noise"):
            lbl_path, issues_found = future.result()
            if issues_found:
                issues[lbl_path].extend(issues_found)
    
    logging.info(f"Label noise detection complete. Found issues in {len(issues)} files.")
    return dict(issues)

def _audit_single_label_file(lbl_path, image_path_map):
    """Internal helper for parallel data quality audit."""
    outliers_found = []
    consistency_issues_found = []

    # Outlier detection (e.g., extreme bbox sizes or counts)
    bboxes = []
    try:
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, x, y, w, h = map(float, parts)
                    bboxes.append((x, y, w, h))
        if bboxes:
            areas = [w*h for (_, _, w, h) in bboxes]
            if np.max(areas) > 0.5:
                outliers_found.append('Very large bbox detected')
            if np.min(areas) < 0.0001:
                outliers_found.append('Very small bbox detected')
            if len(bboxes) > 50:
                outliers_found.append('Excessive number of objects')
    except Exception as e:
        outliers_found.append(f'Processing error: {e}')

    # Consistency checks
    img_id = os.path.basename(lbl_path).split('.')[0]
    if img_id not in image_path_map:
        consistency_issues_found.append('Corresponding image missing')
    else:
        img_path = image_path_map[img_id]
        if not validate_image_and_labels(img_path, lbl_path): # validate_image_and_labels returns (bool)
            consistency_issues_found.append('Bounding box integrity or format issues')
    
    return lbl_path, outliers_found, consistency_issues_found

def audit_data_quality(label_paths, image_paths=None, num_workers: int = 1):
    """
    Automated data quality audit: outlier detection, duplicate removal, annotation error visualization.
    Uses ThreadPoolExecutor for parallel processing.
    """
    logging.info(f"Running comprehensive data quality audit with {num_workers} workers...")
    audit_report = {}

    # 1. Label noise detection (already parallelized)
    label_noise_issues = detect_label_noise(label_paths, num_workers=num_workers)
    audit_report['label_noise'] = label_noise_issues

    outliers = defaultdict(list)
    consistency_issues = defaultdict(list)
    
    image_path_map = {os.path.basename(p).split('.')[0]: p for p in image_paths} if image_paths else {}

    if label_paths:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_audit_single_label_file, lbl_path, image_path_map): lbl_path
                       for lbl_path in label_paths}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Auditing Data Quality"):
                lbl_path, outliers_found, consistency_issues_found = future.result()
                if outliers_found:
                    outliers[lbl_path].extend(outliers_found)
                if consistency_issues_found:
                    consistency_issues[lbl_path].extend(consistency_issues_found)

    audit_report['outliers'] = dict(outliers)
    audit_report['consistency_checks'] = dict(consistency_issues)

    logging.info("Data quality audit complete.")
    return audit_report

def _correct_single_label(label_path: str, image_path: str, model: EmbedDetector, 
                          detection_prompts: list, class_names: list, class_map: dict = None, 
                          threshold: float = 0.5, human_review: bool = False):
    """Internal helper for parallel label correction."""
    try:
        image_pil = Image.open(image_path).convert('RGB')
        h_img, w_img = image_pil.height, image_pil.width

        with amp.autocast() if TORCH_AVAILABLE and model.device == 'cuda' else torch.no_grad():
            pred_boxes_xyxy, pred_scores, pred_label_indices, _ = model.detect(image_pil, detection_prompts)

        gt_labels_yolo = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_labels_yolo.append(list(map(float, parts)))

        corrected_labels_yolo = []
        gt_boxes_xyxy = []
        for gt_cls, gt_cx, gt_cy, gt_w, gt_h in gt_labels_yolo:
            gt_x1 = (gt_cx - gt_w/2) * w_img
            gt_y1 = (gt_cy - gt_h/2) * h_img
            gt_x2 = (gt_cx + gt_w/2) * w_img
            gt_y2 = (gt_cy + gt_h/2) * h_img
            gt_boxes_xyxy.append([gt_x1, gt_y1, gt_x2, gt_y2])

        matched_gt_indices = set()
        for i, pred_box_xyxy in enumerate(pred_boxes_xyxy):
            if pred_scores[i] >= threshold:
                owlvit_label_index = pred_label_indices[i]
                owlvit_prompt_text = detection_prompts[owlvit_label_index]
                mapped_class_id = map_owlvit_label_to_dataset_class_id(
                    owlvit_label_index, owlvit_prompt_text, class_map or {}, class_names
                )
                
                if mapped_class_id == -1:
                    continue

                is_new_detection = True
                for j, gt_box_xyxy in enumerate(gt_boxes_xyxy):
                    if _iou_xyxy(pred_box_xyxy, gt_box_xyxy) > 0.7:
                        matched_gt_indices.add(j)
                        is_new_detection = False
                        break
                if is_new_detection:
                    x1, y1, x2, y2 = pred_box_xyxy
                    cx = np.clip(((x1 + x2) / 2) / w_img, 0.0, 1.0)
                    cy = np.clip(((y1 + y2) / 2) / h_img, 0.0, 1.0)
                    bw = np.clip((x2 - x1) / w_img, 0.0, 1.0)
                    bh = np.clip((y2 - y1) / h_img, 0.0, 1.0) # Corrected h to h_img

                    corrected_labels_yolo.append([mapped_class_id, cx, cy, bw, bh])

        for j, gt_label in enumerate(gt_labels_yolo):
            if j not in matched_gt_indices:
                corrected_labels_yolo.append(gt_label)

        # Save corrected labels
        with open(label_path, 'w') as f:
            for label_data in corrected_labels_yolo:
                f.write(f"{int(label_data[0])} {label_data[1]:.6f} {label_data[2]:.6f} {label_data[3]:.6f} {label_data[4]:.6f}\n")
        return label_path, True # Indicate success
    except Exception as e:
        logging.error(f"Error during label correction for {label_path}: {e}")
        return label_path, False # Indicate failure

def correct_labels_with_model(label_paths_to_correct: list, image_paths_map: dict, model: EmbedDetector, 
                              detection_prompts: list, class_names: list, class_map: dict = None, 
                              threshold: float = 0.5, human_review: bool = False, num_workers: int = 1):
    """
    Semi-automatic label correction using model predictions and optional human review.
    Uses ThreadPoolExecutor for parallel processing.
    """
    if model is None:
        logging.warning("Model not provided for label correction. Skipping correction.")
        return []

    logging.info(f"Attempting to correct {len(label_paths_to_correct)} labels with {num_workers} workers...")
    corrected_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_correct_single_label, lbl_path, image_paths_map.get(os.path.basename(lbl_path).replace('.txt', '')), 
                                   model, detection_prompts, class_names, class_map, threshold, human_review): lbl_path
                   for lbl_path in label_paths_to_correct if image_paths_map.get(os.path.basename(lbl_path).replace('.txt', ''))}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Correcting Labels"):
            lbl_path, success = future.result()
            if success:
                corrected_count += 1
    
    if human_review:
        logging.info(f"[HUMAN_REVIEW_REQUIRED] Manual review indicated for corrected labels. A UI/tool would be needed here.")

    logging.info(f"Label correction attempted. Successfully corrected {corrected_count} files.")
    return [lp for lp in label_paths_to_correct if os.path.exists(lp) and validate_image_and_labels(image_paths_map.get(os.path.basename(lp).replace('.txt', '')), lp)]


# --- Curriculum Learning ---
def curriculum_sampling(image_paths, label_paths, model: EmbedDetector = None, detection_prompts: list = None, 
                        class_names: list = None, class_map: dict = None, stages: int = 3):
    """
    Progressive sampling: start with easy samples, add harder ones (rare classes, hard negatives).
    Args:
        image_paths (list): List of paths to images.
        label_paths (list): List of paths to corresponding YOLO label files.
        model (EmbedDetector, optional): Trained detection model for hard negative mining.
        detection_prompts (list, optional): Prompts for the detection model. Required if model is used.
        class_names (list, optional): List of your dataset's class names. Required if model is used.
        class_map (dict, optional): Class mapping.
        stages (int): Number of curriculum stages.
    Returns:
        tuple: (list of selected image paths, list of selected label paths)
    """
    logging.info(f"[INFO] Curriculum sampling: {stages} stages.")

    all_selected_imgs = []
    all_selected_lbls = []

    # Stage 1: Easy samples (e.g., large boxes, common classes, few objects)
    easy_imgs, easy_lbls = [], []
    for img, lbl in zip(image_paths, label_paths):
        try:
            with open(lbl, 'r') as f:
                lines = f.readlines()
            
            if not lines: # Empty label file is considered easy (no objects)
                easy_imgs.append(img)
                easy_lbls.append(lbl)
                continue

            is_easy = False
            num_objects = len(lines)
            if num_objects < 5: # Few objects
                is_easy = True
            
            # Check for large boxes
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5: continue # Skip malformed lines
                _, _, _, w, h = map(float, parts)
                if w > 0.2 and h > 0.2: # Example: boxes larger than 20% of image
                    is_easy = True
                    break
            if is_easy and img not in all_selected_imgs:
                easy_imgs.append(img)
                easy_lbls.append(lbl)
        except Exception as e:
            logging.warning(f"Error reading label for easy sample check {lbl}: {e}")
            continue
    logging.info(f"Stage 1 (Easy Samples): {len(easy_imgs)} selected.")
    all_selected_imgs.extend(easy_imgs)
    all_selected_lbls.extend(easy_lbls)

    # Stage 2: Add rare classes
    class_counts = Counter()
    for lbl in label_paths:
        try:
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and len(parts) == 5: # Ensure valid line
                        class_counts[int(parts[0])] += 1
        except Exception as e:
            logging.warning(f"Error reading label for class counting {lbl}: {e}")
            continue

    total_objects = sum(class_counts.values())
    # Define rare classes as those with less than X% of total objects
    rare_class_threshold = 0.01 # 1% of total objects
    rare_classes = [cls_id for cls_id, count in class_counts.items() if total_objects > 0 and count / total_objects < rare_class_threshold]
    logging.info(f"Rare classes identified (less than {rare_class_threshold*100}% of total objects): {rare_classes}")

    rare_imgs, rare_lbls = [], []
    for img, lbl in zip(image_paths, label_paths):
        try:
            with open(lbl, 'r') as f:
                lines = f.readlines()
            has_rare_class = False
            for line in lines:
                parts = line.strip().split()
                if parts and len(parts) == 5 and int(parts[0]) in rare_classes:
                    has_rare_class = True
                    break
            if has_rare_class and img not in all_selected_imgs: # Avoid duplicates
                rare_imgs.append(img)
                rare_lbls.append(lbl)
        except Exception as e:
            logging.warning(f"Error reading label for rare class check {lbl}: {e}")
            continue
    logging.info(f"Stage 2 (Rare Classes): {len(rare_imgs)} selected.")
    all_selected_imgs.extend(rare_imgs)
    all_selected_lbls.extend(rare_lbls)

    # Stage 3 (if applicable): Add hard negatives (requires a model)
    if model is not None and detection_prompts is not None:
        logging.info("Stage 3 (Hard Negatives): Identifying hard negatives...")
        hard_imgs = hard_negative_mining(image_paths, label_paths, model, detection_prompts, threshold=model.threshold)
        # Filter out images already selected and ensure corresponding labels exist
        hard_lbls = [lbl for img, lbl in zip(image_paths, label_paths) if img in hard_imgs and img not in all_selected_imgs]
        logging.info(f"Stage 3 (Hard Negatives): {len(hard_imgs)} selected.")
        all_selected_imgs.extend(hard_imgs)
        all_selected_lbls.extend(hard_lbls)
    else:
        logging.warning("Skipping Stage 3 (Hard Negatives): Model or detection prompts not provided.")

    # Stage 4 (optional): Meta-data driven sampling
    # Prioritize images with diverse object sizes, relations, or other meta-data
    # This requires `compute_metadata` and potentially `generate_relation_graph` etc.
    # For simplicity, this remains a conceptual stage.
    # from metadata_logger import compute_metadata # Assuming this is available
    # meta_scores = []
    # for img, lbl in zip(image_paths, label_paths):
    #     labels_data = [] # Load labels in a format compute_metadata expects
    #     # ... populate labels_data ...
    #     meta = compute_metadata(img, labels_data)
    #     score = meta.get('obj_count', 0) + (0.5 - abs(meta.get('avg_bbox_area', 0.25)-0.25)) # Example score
    #     meta_scores.append((score, img, lbl))
    # meta_scores.sort(reverse=True)
    # meta_imgs = [img for _, img, _ in meta_scores[:20] if img not in all_selected_imgs] # Top 20 diverse
    # meta_lbls = [lbl for _, _, lbl in meta_scores[:20] if lbl not in all_selected_lbls]
    # logging.info(f"Stage 4 (Meta-data driven): {len(meta_imgs)} selected.")
    # all_selected_imgs.extend(meta_imgs)
    # all_selected_lbls.extend(meta_lbls) # Corrected from original

    # Remove duplicates while preserving order (if order matters)
    unique_pairs = []
    seen_imgs = set()
    for img, lbl in zip(all_selected_imgs, all_selected_lbls):
        if img not in seen_imgs:
            unique_pairs.append((img, lbl))
            seen_imgs.add(img)

    final_imgs, final_lbls = zip(*unique_pairs) if unique_pairs else ([], [])

    logging.info(f"[INFO] Curriculum sampling: {len(final_imgs)} unique samples selected across all stages.")
    return list(final_imgs), list(final_lbls)

# --- Error analysis utility ---
def _analyze_single_error(img_path, lbl_path, model, detection_prompts):
    """Internal helper for parallel error analysis."""
    report = {'fp': 0, 'fn': 0, 'low_conf': 0}
    try:
        image_pil = Image.open(img_path).convert('RGB')
        h_img, w_img = image_pil.height, image_pil.width

        with amp.autocast() if TORCH_AVAILABLE and model.device == 'cuda' else torch.no_grad():
            pred_boxes_xyxy, pred_scores, _, _ = model.detect(image_pil, detection_prompts)

        gt_boxes_yolo = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes_yolo.append(list(map(float, parts[1:])))

        gt_boxes_xyxy = []
        for gt_cx, gt_cy, gt_w, gt_h in gt_boxes_yolo:
            gt_x1 = (gt_cx - gt_w/2) * w_img
            gt_y1 = (gt_cy - gt_h/2) * h_img
            gt_x2 = (gt_cx + gt_w/2) * w_img
            gt_y2 = (gt_cy + gt_h/2) * h_img
            gt_boxes_xyxy.append([gt_x1, gt_y1, gt_x2, gt_y2])

        matched_preds = [False] * len(pred_boxes_xyxy)
        matched_gts = [False] * len(gt_boxes_xyxy)

        for i, pred_box in enumerate(pred_boxes_xyxy):
            if pred_scores[i] < 0.3:
                report['low_conf'] += 1
                continue

            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes_xyxy):
                current_iou = _iou_xyxy(pred_box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = j

            if best_iou > 0.5:
                matched_preds[i] = True
                matched_gts[best_gt_idx] = True
            else:
                report['fp'] += 1

        for j, gt_box in enumerate(gt_boxes_xyxy):
            if not matched_gts[j]:
                report['fn'] += 1
        return img_path, report
    except Exception as e:
        logging.error(f"Error during error analysis for {img_path}: {e}")
        report['processing_error'] = str(e)
        return img_path, report

def error_analysis(image_paths, label_paths, model: EmbedDetector, detection_prompts: list, num_workers: int = 1):
    """
    Runs error analysis: finds FP, FN, low-confidence, and label mismatches.
    Uses ThreadPoolExecutor for parallel processing.
    """
    if model is None or detection_prompts is None:
        logging.warning("Model or detection prompts not provided for error analysis. Skipping.")
        return {}

    logging.info(f"Running error analysis with {num_workers} workers...")
    error_report = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_analyze_single_error, img, lbl, model, detection_prompts): img
                   for img, lbl in zip(image_paths, label_paths)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Error Analysis"):
            img_path, report_data = future.result()
            error_report[img_path] = report_data

    logging.info("[INFO] Error analysis complete.")
    return error_report

def _continuous_validation_single(img_path, lbl_path):
    """Internal helper for parallel continuous validation."""
    errors_found = []
    if not os.path.exists(img_path):
        errors_found.append('Image file missing')
    if not os.path.exists(lbl_path):
        errors_found.append('Label file missing')
    if os.path.exists(img_path) and os.path.exists(lbl_path):
        if not validate_image_and_labels(img_path, lbl_path):
            errors_found.append('Bounding box integrity or format issues')
    return img_path, errors_found

def continuous_validation(image_paths, label_paths, step_name="validation", num_workers: int = 1):
    """
    Automated validation after each pipeline step, with error reports.
    Uses ThreadPoolExecutor for parallel processing.
    """
    logging.info(f"[VALIDATION] Running continuous validation for step: {step_name} with {num_workers} workers...")
    errors = defaultdict(list)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_continuous_validation_single, img, lbl): img
                   for img, lbl in zip(image_paths, label_paths)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Validating {step_name}"):
            img_path, errors_found = future.result()
            if errors_found:
                errors[img_path].extend(errors_found)

    logging.info(f"[VALIDATION] {step_name}: {len(errors)} images found with issues.")
    try:
        import mlflow
        mlflow.log_dict(dict(errors), f"validation_errors_{step_name}.json")
    except ImportError:
        pass
    except Exception as e:
        logging.error(f"MLflow logging failed for validation step '{step_name}': {e}")
    return dict(errors)

# --- Main Dataset Processing Logic ---
def ensure_dirs(base_dir):
    """Creates necessary subdirectories for YOLO dataset structure."""
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val', 'images/test', 'labels/test']: # Added test splits
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

def parse_classes(classes_file):
    """Reads class names from a file."""
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Classes file not found: {classes_file}")
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def parse_annotations(ann_path):
    """
    Parses annotation data from a JSON file.
    Expected format: {image_id: {"boxes": [[xmin, ymin, xmax, ymax, class_name], ...], "width": int, "height": int}}
    """
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    with open(ann_path) as f:
        data = json.load(f)
    return data

def convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """Converts bounding box from absolute (xmin, ymin, xmax, ymax) to normalized YOLO (cx, cy, w, h)."""
    x_c = ((xmin + xmax) / 2) / img_w
    y_c = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # Clip to [0,1] to handle potential floating point errors or out-of-bounds annotations
    x_c, y_c, w, h = [max(0.0, min(1.0, v)) for v in [x_c, y_c, w, h]]
    return x_c, y_c, w, h

def write_label_file(label_path, objects):
    """Writes YOLO format label data to a file."""
    with open(label_path, 'w') as f:
        for obj in objects:
            # Ensure obj has 'class_idx', 'x_c', 'y_c', 'w', 'h'
            class_idx = obj.get('class_idx', 0)
            x_c = obj.get('x_c', 0.5)
            y_c = obj.get('y_c', 0.5)
            w = obj.get('w', 0.1)
            h = obj.get('h', 0.1)
            f.write(f"{class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def split_filenames(filenames, split_ratio, seed=42):
    """Splits a list of filenames into train and validation sets."""
    random.seed(seed)
    random.shuffle(filenames)
    n_train = int(len(filenames) * split_ratio)
    return filenames[:n_train], filenames[n_train:]

def generate_yaml(output_dir, class_names):
    """Generates the data.yaml file for YOLO training."""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_content = {
        'train': os.path.abspath(os.path.join(output_dir, 'images/train')),
        'val': os.path.abspath(os.path.join(output_dir, 'images/val')),
        'test': os.path.abspath(os.path.join(output_dir, 'images/test')), # Added test path
        'nc': len(class_names),
        'names': class_names
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    logging.info(f"Generated data.yaml at {yaml_path}")

def generate_classes_file(output_dir, class_names):
    """Generates a classes.names file."""
    classes_file_path = os.path.join(output_dir, 'classes.names')
    with open(classes_file_path, 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    logging.info(f"Generated classes.names at {classes_file_path}")


def process_dataset(args):
    """
    Processes the raw dataset into YOLO format, performs splitting, and generates metadata.
    This function orchestrates the entire dataset preparation pipeline.
    """
    logging.info("Starting dataset processing...")

    cfg = {} # Initialize empty config
    if getattr(args, 'config', None) and os.path.exists(args.config):
        try:
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            setup_logging(cfg.get('logging', {})) # Ensure logging is set up from config
            logging.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logging.error(f"Error loading config from {args.config}: {e}. Proceeding with CLI defaults.")
    else:
        logging.warning("No config file provided or found. Proceeding with default arguments.")

    # Override config values with CLI arguments if provided
    # This ensures CLI args takes precedence over config file
    cfg['force_rerun'] = getattr(args, 'force_rerun', cfg.get('force_rerun', False))
    
    # Model Config
    cfg.setdefault('model', {})
    cfg['model']['name'] = getattr(args, 'owlvit_model', cfg['model'].get('name', 'google/owlvit-base-patch32'))
    cfg['model']['device'] = getattr(args, 'device', cfg['model'].get('device', 'cpu'))
    cfg['model']['detection_threshold'] = getattr(args, 'min_score', cfg['model'].get('detection_threshold', 0.3))
    cfg['model']['detection_prompts'] = getattr(args, 'prompts', cfg['model'].get('detection_prompts', ["object"])) # Ensure prompts are here
    cfg['model']['class_map_path'] = getattr(args, 'class_map_path', cfg['model'].get('class_map_path'))

    # Paths Config
    cfg.setdefault('paths', {})
    cfg['paths']['raw_images'] = getattr(args, 'input_dir', cfg['paths'].get('raw_images'))
    cfg['paths']['output_dir'] = getattr(args, 'output_dir', cfg['paths'].get('output_dir'))
    cfg['paths']['annotations'] = getattr(args, 'annotations', cfg['paths'].get('annotations'))
    cfg['paths']['classes_file'] = getattr(args, 'classes_file', cfg['paths'].get('classes_file'))
    cfg['paths']['raw_labels_output'] = getattr(args, 'raw_labels_output_dir', cfg['paths'].get('raw_labels_output', os.path.join(cfg['paths']['output_dir'], 'initial_labels')))
    cfg['paths']['synthetic_data_base'] = cfg['paths'].get('synthetic_data_base', os.path.join(cfg['paths']['output_dir'], 'synthetic_data'))


    # Split Config
    cfg.setdefault('split', {})
    cfg['split']['train_size'] = getattr(args, 'split_ratio', cfg['split'].get('train_size', 0.8))
    cfg['split']['val_size'] = cfg['split'].get('val_size', 0.1) # Default to 0.1 if not in config
    cfg['split']['test_size'] = cfg['split'].get('test_size', 0.1) # Default to 0.1 if not in config
    cfg['split']['stratified'] = cfg['split'].get('stratified', True)
    cfg['split']['n_splits'] = cfg['split'].get('n_splits', None) # For k-fold

    # General Config
    cfg['seed'] = getattr(args, 'seed', cfg.get('seed', 42))
    cfg.setdefault('resources', {})
    cfg['resources']['num_workers'] = getattr(args, 'num_workers', cfg['resources'].get('num_workers', 4))

    # Feature Flags (nested under their respective sections)
    cfg.setdefault('auto_labeling', {})
    cfg['auto_labeling']['enabled'] = getattr(args, 'auto_labeling_enabled', cfg['auto_labeling'].get('enabled', False))
    cfg['auto_labeling']['sam_model_type'] = cfg['auto_labeling'].get('sam_model_type', 'vit_b')
    cfg['auto_labeling']['sam_checkpoint_path'] = cfg['auto_labeling'].get('sam_checkpoint_path')


    cfg.setdefault('pseudo_labeling', {})
    cfg['pseudo_labeling']['enabled'] = getattr(args, 'pseudo_labeling_enabled', cfg['pseudo_labeling'].get('enabled', False))

    cfg.setdefault('active_learning', {})
    cfg['active_learning']['enabled'] = getattr(args, 'active_learning_enabled', cfg['active_learning'].get('enabled', False))

    cfg.setdefault('label_correction', {})
    cfg['label_correction']['enabled'] = getattr(args, 'label_correction_enabled', cfg['label_correction'].get('enabled', False))

    cfg.setdefault('class_balancing', {})
    cfg['class_balancing']['enabled'] = getattr(args, 'class_balancing_enabled', cfg['class_balancing'].get('enabled', False))
    cfg.setdefault('smote', {}) # SMOTE is now a sub-flag of class_balancing conceptually
    cfg['smote']['enabled'] = getattr(args, 'smote_enabled', cfg['smote'].get('enabled', False))


    cfg.setdefault('curriculum_learning', {})
    cfg['curriculum_learning']['enabled'] = getattr(args, 'curriculum_learning_enabled', cfg['curriculum_learning'].get('enabled', False))

    cfg.setdefault('hard_negative_mining', {})
    cfg['hard_negative_mining']['enabled'] = getattr(args, 'hard_negative_mining_enabled', cfg['hard_negative_mining'].get('enabled', False))

    cfg.setdefault('synthetic_data', {})
    cfg['synthetic_data']['enabled'] = getattr(args, 'synthetic_data_enabled', cfg['synthetic_data'].get('enabled', False))
    cfg['synthetic_data']['n_samples'] = cfg['synthetic_data'].get('n_samples', 0)
    cfg['synthetic_data']['batch_size'] = cfg['synthetic_data'].get('batch_size', 16) # Added batch_size
    cfg['synthetic_data']['image_size'] = cfg['synthetic_data'].get('image_size', [640, 640])
    cfg['synthetic_data']['model_config'] = cfg['synthetic_data'].get('model_config', {})

    cfg.setdefault('copy_paste_synthesis', {})
    cfg['copy_paste_synthesis']['enabled'] = getattr(args, 'copy_paste_enabled', cfg['copy_paste_synthesis'].get('enabled', False))
    cfg['copy_paste_synthesis']['n_samples'] = cfg['copy_paste_synthesis'].get('n_samples', 0)
    cfg['copy_paste_synthesis']['mask_dir'] = getattr(args, 'mask_dir', cfg['copy_paste_synthesis'].get('mask_dir'))
    cfg['copy_paste_synthesis']['max_pastes'] = cfg['copy_paste_synthesis'].get('max_pastes', 3)
    cfg['copy_paste_synthesis']['background_source'] = cfg['copy_paste_synthesis'].get('background_source', 'real_data') # 'real_data' or 'generated_data'


    cfg.setdefault('self_training', {}) # Added self_training config
    cfg['self_training']['enabled'] = cfg['self_training'].get('enabled', False)
    cfg['self_training']['cycles'] = cfg['self_training'].get('cycles', 1)


    cfg.setdefault('test_time_augmentation', {})
    cfg['test_time_augmentation']['enable'] = getattr(args, 'test_time_augmentation_enabled', cfg['test_time_augmentation'].get('enable', False))

    cfg.setdefault('error_analysis', {})
    cfg['error_analysis']['enable'] = getattr(args, 'error_analysis_enabled', cfg['error_analysis'].get('enable', False))

    cfg.setdefault('visualization', {})
    cfg['visualization']['enable'] = getattr(args, 'visualization_enabled', cfg['visualization'].get('enable', False))

    cfg.setdefault('mlflow', {})
    cfg['mlflow']['enable'] = getattr(args, 'mlflow_enabled', cfg['mlflow'].get('enable', False))

    cfg.setdefault('dali', {})
    cfg['dali']['enabled'] = getattr(args, 'dali_enabled', cfg['dali'].get('enabled', False))


    # Set up logging based on the (potentially updated) config
    setup_logging(cfg.get('logging', {}))

    logging.info("Starting dataset preparation pipeline...")
    # logging.info(f"Effective Configuration: {yaml.dump(cfg, indent=2)}") # Log the effective config

    mlflow_enabled = cfg.get('mlflow', {}).get('enable', False)
    if mlflow_enabled:
        try:
            import mlflow
            mlflow.set_tracking_uri(cfg['mlflow'].get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(cfg['mlflow'].get('experiment', 'dataset-prep'))
            mlflow.start_run(run_name="dataset-prep-pipeline")
            mlflow.log_params({k: v for k, v in vars(args).items() if v is not None}) # Log CLI args
            mlflow.log_dict(cfg, "final_config.yaml") # Log the effective config
        except ImportError:
            logging.warning("MLflow is enabled in config but mlflow library is not installed.")
            mlflow_enabled = False
        except Exception as e:
            logging.error(f"Error setting up MLflow: {e}. Disabling MLflow.")
            mlflow_enabled = False

    # Initialize EmbedDetector and SAM Predictor for potential use in hooks
    embed_detector_instance = None
    sam_predictor_instance = None
    
    # Load class names and class map
    class_names = parse_classes(cfg['paths']['classes_file'])
    class_map_data = {}
    if cfg['model'].get('class_map_path'):
        if load_class_map:
            try:
                class_map_data = load_class_map(cfg['model']['class_map_path'])
            except Exception as e:
                logging.error(f"Failed to load class map: {e}. Proceeding without class mapping.")
        else:
            logging.warning("`load_class_map` function not available. Cannot load class mapping.")

    # Initialize EmbedDetector if any model-dependent features are enabled
    if EmbedDetector and (
        cfg['auto_labeling'].get('enabled', False) or 
        cfg['pseudo_labeling'].get('enabled', False) or 
        cfg['active_learning'].get('enabled', False) or 
        cfg['label_correction'].get('enabled', False) or 
        cfg['hard_negative_mining'].get('enabled', False) or 
        cfg['curriculum_learning'].get('enabled', False) or 
        cfg['error_analysis'].get('enabled', False) or 
        cfg['test_time_augmentation'].get('enable', False) or
        cfg['synthetic_data'].get('enabled', False) or # Synthetic data now uses detector for pseudo-labeling
        cfg['copy_paste_synthesis'].get('enabled', False) # Copy-paste needs detector for pseudo-labeling if no GT
    ):
        try:
            embed_detector_instance = EmbedDetector(cfg['model'])
            logging.info("EmbedDetector (OWL-ViT) initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize EmbedDetector: {e}. OWL-ViT dependent steps will be skipped.")
            embed_detector_instance = None
    elif EmbedDetector is None:
        logging.warning("EmbedDetector not available. OWL-ViT dependent steps will be skipped.")

    # Initialize SAM Predictor only if auto_labeling is explicitly enabled
    if cfg.get('auto_labeling', {}).get('enabled', False):
        if SamPredictor and sam_model_registry:
            try:
                sam_model_type = cfg['auto_labeling'].get('sam_model_type', 'vit_b')
                sam_checkpoint_path = cfg['auto_labeling'].get('sam_checkpoint_path')
                
                if not sam_checkpoint_path or not os.path.exists(sam_checkpoint_path):
                    logging.critical(f"SAM checkpoint not found at {sam_checkpoint_path}. Auto-labeling will not work.")
                    sam_predictor_instance = None
                else:
                    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
                    sam_model.to(device=cfg['model']['device']) # Use same device as OWL-ViT
                    sam_predictor_instance = SamPredictor(sam_model)
                    logging.info("SAM Predictor initialized.")
            except Exception as e:
                logging.critical(f"Failed to initialize SAM Predictor: {e}. Auto-labeling will not work.")
                import traceback
                traceback.print_exc()
                sam_predictor_instance = None
        else:
            logging.critical("SAM (SamPredictor or sam_model_registry) is not available. Auto-labeling requires SAM.")

    # Ensure output directories exist
    ensure_dirs(cfg['paths']['output_dir'])
    os.makedirs(cfg['paths']['synthetic_data_base'], exist_ok=True) # Ensure base synthetic dir exists

    # 1. Initial Auto labeling (if enabled in config and no raw labels provided)
    raw_images_dir = cfg["paths"]["raw_images"]
    initial_labels_dir = cfg['paths']['raw_labels_output']
    
    # Checkpoint for initial auto-labeling
    auto_labeling_skipped = False
    if not cfg['force_rerun'] and os.path.exists(initial_labels_dir) and any(f.endswith('.txt') for f in os.listdir(initial_labels_dir)):
        logging.info(f"Skipping initial auto-labeling: Output directory '{initial_labels_dir}' already contains labels.")
        auto_labeling_skipped = True
    elif auto_label and cfg.get('auto_labeling', {}).get('enabled', False) and embed_detector_instance and sam_predictor_instance:
        logging.info("Running initial auto-labeling on raw images...")
        os.makedirs(initial_labels_dir, exist_ok=True)
        unlabeled_images = [os.path.join(raw_images_dir, f) for f in os.listdir(raw_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_path in tqdm(unlabeled_images, desc="Initial Auto-labeling"):
            output_lbl_path = os.path.join(initial_labels_dir, os.path.basename(img_path).replace(os.path.splitext(img_path)[1], '.txt'))
            auto_label(
                image_path=img_path,
                output_label_path=output_lbl_path,
                embed_detector=embed_detector_instance,
                sam_predictor_instance=sam_predictor_instance,
                detection_prompts=cfg['model'].get('detection_prompts', ["object"]),
                class_names=class_names,
                class_map=class_map_data,
                min_score=cfg['model'].get('detection_threshold', 0.3)
            )
    else:
        logging.info("Initial auto-labeling skipped (either not enabled, models unavailable, or checkpointed).")
        auto_labeling_skipped = True # Mark as skipped if conditions not met for running

    # Determine the source for labels for the rest of the pipeline
    # Prioritize auto-generated labels if they exist, otherwise use raw_labels from config
    source_labels_dir = initial_labels_dir if os.path.exists(initial_labels_dir) and any(f.endswith('.txt') for f in os.listdir(initial_labels_dir)) else cfg["paths"].get("raw_labels")
    
    if not source_labels_dir or not os.path.exists(source_labels_dir):
        logging.critical(f"No source labels directory found. Please provide raw_labels path or enable auto_labeling. Exiting.")
        if mlflow_enabled: mlflow.end_run()
        return

    # Load object masks once if copy-paste is enabled
    objects_for_copy_paste = []
    if cfg.get('copy_paste_synthesis', {}).get('enabled', False):
        mask_dir = cfg['copy_paste_synthesis'].get('mask_dir')
        if mask_dir:
            objects_for_copy_paste = load_object_masks(mask_dir)
            if not objects_for_copy_paste:
                logging.warning(f"No objects loaded from {mask_dir} for copy-paste. Disabling copy-paste synthesis.")
                cfg['copy_paste_synthesis']['enabled'] = False # Disable if no masks
        else:
            logging.warning("Copy-paste synthesis enabled but 'mask_dir' not provided. Disabling.")
            cfg['copy_paste_synthesis']['enabled'] = False


    # --- Iterative Self-Training Loop ---
    self_training_enabled = cfg.get('self_training', {}).get('enabled', False)
    self_training_cycles = cfg.get('self_training', {}).get('cycles', 1)

    # Initialize current dataset paths before the loop
    current_img_paths = sorted([os.path.join(raw_images_dir, f) for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    current_lbl_paths = sorted([os.path.join(source_labels_dir, f) for f in os.listdir(source_labels_dir) if f.endswith('.txt')])
    
    # Sync initial real data
    img_stems = {os.path.splitext(os.path.basename(p))[0] for p in current_img_paths}
    lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in current_lbl_paths}
    common_stems = list(img_stems.intersection(lbl_stems))
    current_img_paths = [p for p in current_img_paths if os.path.splitext(os.path.basename(p))[0] in common_stems]
    current_lbl_paths = [p for p in current_lbl_paths if os.path.splitext(os.path.basename(p))[0] in common_stems]
    current_img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
    current_lbl_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

    if not current_img_paths or not current_lbl_paths:
        logging.error("No synced images or labels found after initial processing. Exiting pipeline.")
        if mlflow_enabled: mlflow.end_run()
        return
    logging.info(f"Starting pipeline with {len(current_img_paths)} synced initial image-label pairs.")


    for cycle in range(self_training_cycles):
        logging.info(f"\n--- Starting Self-Training Cycle {cycle + 1}/{self_training_cycles} ---")

        # --- Re-collect all current dataset paths (real + synthetic from previous cycles) ---
        # This ensures the current_img_paths and current_lbl_paths reflect the full dataset
        # including any synthetic data generated in previous cycles.
        all_current_images = []
        all_current_labels = []

        # Add real images and their labels
        all_current_images.extend(sorted([os.path.join(raw_images_dir, f) for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))
        all_current_labels.extend(sorted([os.path.join(source_labels_dir, f) for f in os.listdir(source_labels_dir) if f.endswith('.txt')]))

        # Add synthetic images and labels from all previous cycles (if they exist)
        # Include GAN and Copy-Paste synthetic data
        for prev_cycle in range(cycle + 1): 
            gan_synthetic_img_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_images_cycle_{prev_cycle}')
            gan_synthetic_lbl_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_labels_cycle_{prev_cycle}')
            if os.path.exists(gan_synthetic_img_dir) and os.path.exists(gan_synthetic_lbl_dir):
                all_current_images.extend(sorted([os.path.join(gan_synthetic_img_dir, f) for f in os.listdir(gan_synthetic_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))
                all_current_labels.extend(sorted([os.path.join(gan_synthetic_lbl_dir, f) for f in os.listdir(gan_synthetic_lbl_dir) if f.endswith('.txt')]))

            cp_synthetic_img_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'cp_images_cycle_{prev_cycle}')
            cp_synthetic_lbl_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'cp_labels_cycle_{prev_cycle}')
            if os.path.exists(cp_synthetic_img_dir) and os.path.exists(cp_synthetic_lbl_dir):
                all_current_images.extend(sorted([os.path.join(cp_synthetic_img_dir, f) for f in os.listdir(cp_synthetic_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))
                all_current_labels.extend(sorted([os.path.join(cp_synthetic_lbl_dir, f) for f in os.listdir(cp_synthetic_lbl_dir) if f.endswith('.txt')]))
        
        # Re-sync and sort the combined list
        img_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_current_images}
        lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_current_labels}
        common_stems = list(img_stems.intersection(lbl_stems))
        current_img_paths = [p for p in all_current_images if os.path.splitext(os.path.basename(p))[0] in common_stems]
        current_lbl_paths = [p for p in all_current_labels if os.path.splitext(os.path.basename(p))[0] in common_stems]
        current_img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        current_lbl_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
        
        logging.info(f"Dataset size for Cycle {cycle+1} (real + synthetic): {len(current_img_paths)} images.")


        # 2. Validate current dataset
        current_img_paths, current_lbl_paths = filter_valid_samples(current_img_paths, current_lbl_paths, num_workers=cfg['resources']['num_workers'])
        continuous_validation(current_img_paths, current_lbl_paths, step_name=f"cycle_{cycle+1}_initial_validation", num_workers=cfg['resources']['num_workers'])

        # 3. Data quality audit
        audit_report = audit_data_quality(current_lbl_paths, current_img_paths, num_workers=cfg['resources']['num_workers'])
        logging.info(f"[AUDIT] Data quality report (Cycle {cycle+1}): {audit_report}")
        if mlflow_enabled:
            try:
                mlflow.log_dict(audit_report, f"data_quality_audit_cycle_{cycle+1}.json")
            except Exception as e:
                logging.error(f"MLflow logging for data quality audit failed: {e}")

        # 4. Label noise detection and correction
        label_issues = detect_label_noise(current_lbl_paths, num_workers=cfg['resources']['num_workers'])
        if label_issues:
            logging.warning(f"[WARN] Label noise detected in {len(label_issues)} files (Cycle {cycle+1}).")
            if mlflow_enabled:
                try:
                    mlflow.log_dict(label_issues, f"label_noise_detected_cycle_{cycle+1}.json")
                except Exception as e:
                    logging.error(f"MLflow logging for label noise failed: {e}")

            if cfg.get('label_correction', {}).get('enabled', False) and embed_detector_instance and correct_labels_with_model:
                logging.info(f"Attempting label correction (Cycle {cycle+1})...")
                correction_prompts = cfg['label_correction'].get('prompts', cfg['model'].get('detection_prompts', ["object"]))
                
                # Create a map from image stem to full path for parallel correction
                current_img_path_map = {os.path.basename(p).replace(os.path.splitext(p)[1], ''): p for p in current_img_paths}
                
                corrected_label_paths = correct_labels_with_model(
                    label_paths_to_correct=list(label_issues.keys()), # Only correct files with issues
                    image_paths_map=current_img_path_map, 
                    model=embed_detector_instance, 
                    detection_prompts=correction_prompts,
                    class_names=class_names,
                    class_map=class_map_data,
                    threshold=cfg['label_correction'].get('threshold', 0.5), 
                    human_review=cfg['label_correction'].get('human_review', False),
                    num_workers=cfg['resources']['num_workers']
                )
                # After correction, re-filter the main lists to ensure only valid ones remain
                current_img_paths, current_lbl_paths = filter_valid_samples(current_img_paths, current_lbl_paths, num_workers=cfg['resources']['num_workers'])
                logging.info(f"Label correction attempted (Cycle {cycle+1}). Re-validating dataset.")
            else:
                initial_count = len(current_lbl_paths)
                problematic_lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in label_issues.keys()}
                
                current_lbl_paths = [l for l in current_lbl_paths if os.path.splitext(os.path.basename(l))[0] not in problematic_lbl_stems]
                current_img_paths = [p for p in current_img_paths if os.path.splitext(os.path.basename(p))[0] not in problematic_lbl_stems]
                logging.info(f"Removed {initial_count - len(current_lbl_paths)} problematic labels (Cycle {cycle+1}).")

        continuous_validation(current_img_paths, current_lbl_paths, step_name=f"cycle_{cycle+1}_post_label_correction", num_workers=cfg['resources']['num_workers'])

        # 5. Class balancing (oversample rare classes)
        if cfg.get('class_balancing', {}).get('enabled', False):
            logging.info(f"Applying class balancing (oversampling) (Cycle {cycle+1})...")
            from .augmentations import class_balanced_oversample, smote_oversample
            current_img_paths, current_lbl_paths = class_balanced_oversample(current_img_paths, current_lbl_paths, min_count=cfg['class_balancing'].get("min_class_count", 100))
            
            if cfg.get('smote', {}).get('enabled', False) and smote_oversample:
                logging.info(f"Applying SMOTE oversampling (conceptual) (Cycle {cycle+1})...")
                smote_oversample(current_img_paths, current_lbl_paths, min_count=cfg['smote'].get("min_class_count", 100))
            
            continuous_validation(current_img_paths, current_lbl_paths, step_name=f"cycle_{cycle+1}_post_oversampling", num_workers=cfg['resources']['num_workers'])

        # 6. Curriculum learning (progressive/meta-data sampling)
        if cfg.get('curriculum_learning', {}).get('enabled', False) and embed_detector_instance and curriculum_sampling:
            logging.info(f"Applying curriculum learning sampling (Cycle {cycle+1})...")
            curriculum_prompts = cfg['curriculum_learning'].get('prompts', cfg['model'].get('detection_prompts', ["object"]))
            current_img_paths, current_lbl_paths = curriculum_sampling(
                current_img_paths, current_lbl_paths, 
                model=embed_detector_instance, 
                detection_prompts=curriculum_prompts,
                class_names=class_names,
                class_map=class_map_data,
                stages=cfg['curriculum_learning'].get('stages', 3)
            )
            continuous_validation(current_img_paths, current_lbl_paths, step_name=f"cycle_{cycle+1}_post_curriculum", num_workers=cfg['resources']['num_workers'])
        elif cfg.get('curriculum_learning', {}).get('enabled', False):
            logging.warning(f"Curriculum learning enabled but EmbedDetector or curriculum_sampling not available. Skipping (Cycle {cycle+1}).")

        # 7. Hard negative mining (active retraining)
        if cfg.get('hard_negative_mining', {}).get('enabled', False) and embed_detector_instance and hard_negative_mining:
            logging.info(f"Performing hard negative mining (Cycle {cycle+1})...")
            hnm_prompts = cfg['hard_negative_mining'].get('prompts', cfg['model'].get('detection_prompts', ["object"]))
            hard_negatives = hard_negative_mining(
                current_img_paths, current_lbl_paths, 
                model=embed_detector_instance, 
                detection_prompts=hnm_prompts,
                threshold=cfg['hard_negative_mining'].get('threshold', 0.3),
                num_workers=cfg['resources']['num_workers']
            )
            if cfg['hard_negative_mining'].get('add_to_train', False):
                logging.info(f"Adding {len(hard_negatives)} hard negatives to dataset (Cycle {cycle+1}).")
                # For hard negatives, we assume their labels exist or are generated by the model.
                # Here, we'll just add the paths if they weren't already in the dataset.
                newly_added_hn_imgs = []
                newly_added_hn_lbls = []
                current_stems = {os.path.splitext(os.path.basename(p))[0] for p in current_img_paths}

                for hn_img_path in hard_negatives:
                    hn_stem = os.path.splitext(os.path.basename(hn_img_path))[0]
                    if hn_stem not in current_stems:
                        hn_lbl_path = os.path.join(source_labels_dir, hn_stem + '.txt') # Assume labels are in source_labels_dir
                        if os.path.exists(hn_lbl_path):
                            newly_added_hn_imgs.append(hn_img_path)
                            newly_added_hn_lbls.append(hn_lbl_path)
                        else:
                            logging.warning(f"Hard negative image {hn_img_path} found, but no corresponding label at {hn_lbl_path}. Skipping addition.")
                
                current_img_paths.extend(newly_added_hn_imgs)
                current_lbl_paths.extend(newly_added_hn_lbls)
                
                # Re-sort to ensure consistency
                current_img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
                current_lbl_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

                logging.info(f"Total samples after adding hard negatives: {len(current_img_paths)} (Cycle {cycle+1})")
            continuous_validation(current_img_paths, current_lbl_paths, step_name=f"cycle_{cycle+1}_post_hard_negative", num_workers=cfg['resources']['num_workers'])
        elif cfg.get('hard_negative_mining', {}).get('enabled', False):
            logging.warning(f"Hard negative mining enabled but EmbedDetector or hard_negative_mining not available. Skipping (Cycle {cycle+1}).")


        # 8. Synthetic data generation (GAN/Diffusion or Copy-Paste) - within self-training loop
        if cfg.get('synthetic_data', {}).get('enabled', False) or cfg.get('copy_paste_synthesis', {}).get('enabled', False):
            logging.info(f"Generating synthetic data (Cycle {cycle+1})...")
            
            synthetic_data_generated_this_cycle = False

            # --- GAN-based Synthetic Data Generation ---
            if cfg.get('synthetic_data', {}).get('enabled', False):
                synthetic_img_dir_gan = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_images_cycle_{cycle}')
                synthetic_lbl_dir_gan = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_labels_cycle_{cycle}')

                if not cfg['force_rerun'] and os.path.exists(synthetic_img_dir_gan) and any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(synthetic_img_dir_gan)):
                    logging.info(f"Skipping GAN synthetic data generation: Output directory '{synthetic_img_dir_gan}' already contains data.")
                elif TORCH_AVAILABLE and GANModel:
                    try:
                        gan_model_config = cfg['synthetic_data'].get('model_config', {})
                        gan_model_config['image_size'] = cfg['synthetic_data'].get('image_size', [640, 640])
                        gan_model_config['device'] = cfg['model'].get('device', 'cpu')
                        gan_model_config['num_classes'] = len(class_names)

                        generator_model = GANModel(gan_model_config)
                        
                        os.makedirs(synthetic_img_dir_gan, exist_ok=True)
                        os.makedirs(synthetic_lbl_dir_gan, exist_ok=True)

                        generate_synthetic_data(
                            generator_model_instance=generator_model, 
                            n_samples=cfg['synthetic_data'].get('n_samples', 0), 
                            batch_size=cfg['synthetic_data'].get('batch_size', 16),
                            out_img_dir=synthetic_img_dir_gan,
                            out_lbl_dir=synthetic_lbl_dir_gan,
                            image_size=cfg['synthetic_data'].get('image_size', (640, 640)),
                            class_names=class_names,
                            embed_detector_instance=embed_detector_instance, # Pass the detector for pseudo-labeling
                            detection_prompts=cfg['model'].get('detection_prompts', ["object"]),
                            class_map=class_map_data,
                            detection_threshold=cfg['model'].get('detection_threshold', 0.3)
                        )
                        synthetic_data_generated_this_cycle = True
                    except Exception as e:
                        logging.error(f"GAN synthetic data generation failed (Cycle {cycle+1}): {e}. Skipping.")
                        import traceback
                        traceback.print_exc()
                else:
                    logging.warning(f"PyTorch or GANModel not available. Skipping GAN synthetic data generation (Cycle {cycle+1}).")

            # --- Copy-Paste Synthesis ---
            if cfg.get('copy_paste_synthesis', {}).get('enabled', False) and objects_for_copy_paste:
                synthetic_img_dir_cp = os.path.join(cfg['paths']['synthetic_data_base'], f'cp_images_cycle_{cycle}')
                synthetic_lbl_dir_cp = os.path.join(cfg['paths']['synthetic_data_base'], f'cp_labels_cycle_{cycle}')
                
                if not cfg['force_rerun'] and os.path.exists(synthetic_img_dir_cp) and any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(synthetic_img_dir_cp)):
                    logging.info(f"Skipping Copy-Paste synthetic data generation: Output directory '{synthetic_img_dir_cp}' already contains data.")
                else:
                    background_source = cfg['copy_paste_synthesis'].get('background_source', 'real_data')
                    bg_img_paths_for_cp = []
                    bg_lbl_paths_for_cp = []

                    if background_source == 'real_data':
                        bg_img_paths_for_cp = sorted([os.path.join(raw_images_dir, f) for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                        bg_lbl_paths_for_cp = sorted([os.path.join(source_labels_dir, f) for f in os.listdir(source_labels_dir) if f.endswith('.txt')])
                        # Ensure background images and labels are synced
                        bg_stems = {os.path.splitext(os.path.basename(p))[0] for p in bg_img_paths_for_cp}
                        bg_lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in bg_lbl_paths_for_cp}
                        common_bg_stems = list(bg_stems.intersection(bg_lbl_stems))
                        bg_img_paths_for_cp = [p for p in bg_img_paths_for_cp if os.path.splitext(os.path.basename(p))[0] in common_bg_stems]
                        bg_lbl_paths_for_cp = [p for p in bg_lbl_paths_for_cp if os.path.splitext(os.path.basename(p))[0] in common_bg_stems]
                        bg_img_paths_for_cp.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
                        bg_lbl_paths_for_cp.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

                    elif background_source == 'generated_data':
                        # This would imply using GAN-generated images as backgrounds for copy-paste
                        # For now, this is conceptual. You'd need to ensure GAN-generated data is available.
                        logging.warning("Using 'generated_data' as background source for copy-paste is conceptual and not fully implemented.")
                        # You would collect paths from synthetic_img_dir_gan and synthetic_lbl_dir_gan here
                        gan_synthetic_img_dir_current = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_images_cycle_{cycle}')
                        gan_synthetic_lbl_dir_current = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_labels_cycle_{cycle}')
                        if os.path.exists(gan_synthetic_img_dir_current) and os.path.exists(gan_synthetic_lbl_dir_current):
                            bg_img_paths_for_cp = sorted([os.path.join(gan_synthetic_img_dir_current, f) for f in os.listdir(gan_synthetic_img_dir_current) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                            bg_lbl_paths_for_cp = sorted([os.path.join(gan_synthetic_lbl_dir_current, f) for f in os.listdir(gan_synthetic_lbl_dir_current) if f.endswith('.txt')])
                            # Re-sync them
                            bg_stems = {os.path.splitext(os.path.basename(p))[0] for p in bg_img_paths_for_cp}
                            bg_lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in bg_lbl_paths_for_cp}
                            common_bg_stems = list(bg_stems.intersection(bg_lbl_stems))
                            bg_img_paths_for_cp = [p for p in bg_img_paths_for_cp if os.path.splitext(os.path.basename(p))[0] in common_bg_stems]
                            bg_lbl_paths_for_cp = [p for p in bg_lbl_paths_for_cp if os.path.splitext(os.path.basename(p))[0] in common_bg_stems]
                            bg_img_paths_for_cp.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
                            bg_lbl_paths_for_cp.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
                        else:
                            logging.warning(f"GAN-generated data not found for background source 'generated_data' (Cycle {cycle+1}). Skipping copy-paste.")
                            bg_img_paths_for_cp = []
                            bg_lbl_paths_for_cp = []

                    else:
                        logging.error(f"Unknown background source for copy-paste synthesis: {background_source}. Skipping.")
                        bg_img_paths_for_cp = []
                        bg_lbl_paths_for_cp = []

                    if bg_img_paths_for_cp:
                        generate_synthetic_data_copy_paste(
                            background_image_paths=bg_img_paths_for_cp,
                            background_label_paths=bg_lbl_paths_for_cp,
                            objects_with_masks=objects_for_copy_paste,
                            class_names=class_names,
                            n_samples=cfg['copy_paste_synthesis'].get('n_samples', 0),
                            out_img_dir=synthetic_img_dir_cp,
                            out_lbl_dir=synthetic_lbl_dir_cp,
                            max_pastes=cfg['copy_paste_synthesis'].get('max_pastes', 3),
                            num_workers=cfg['resources']['num_workers']
                        )
                        synthetic_data_generated_this_cycle = True
                    else:
                        logging.warning("No valid background images found for copy-paste synthesis. Skipping.")
            else:
                logging.info(f"Copy-paste synthesis not enabled or no masks available (Cycle {cycle+1}).")

            # After synthetic data generation (or loading from checkpoint), re-collect current_img_paths/lbl_paths
            # to include the new synthetic data for the current cycle's processing.
            if synthetic_data_generated_this_cycle:
                all_current_images = []
                all_current_labels = []

                all_current_images.extend(sorted([os.path.join(raw_images_dir, f) for f in os.listdir(raw_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))
                all_current_labels.extend(sorted([os.path.join(source_labels_dir, f) for f in os.listdir(source_labels_dir) if f.endswith('.txt')]))

                # Add GAN synthetic data
                gan_synthetic_img_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_images_cycle_{cycle}')
                gan_synthetic_lbl_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'gan_labels_cycle_{cycle}')
                if os.path.exists(gan_synthetic_img_dir):
                    all_current_images.extend(sorted([os.path.join(gan_synthetic_img_dir, f) for f in os.listdir(gan_synthetic_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))
                    all_current_labels.extend(sorted([os.path.join(gan_synthetic_lbl_dir, f) for f in os.listdir(gan_synthetic_lbl_dir) if f.endswith('.txt')]))

                # Add Copy-Paste synthetic data
                cp_synthetic_img_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'cp_images_cycle_{cycle}')
                cp_synthetic_lbl_dir = os.path.join(cfg['paths']['synthetic_data_base'], f'cp_labels_cycle_{cycle}')
                if os.path.exists(cp_synthetic_img_dir):
                    all_current_images.extend(sorted([os.path.join(cp_synthetic_img_dir, f) for f in os.listdir(cp_synthetic_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]))
                    all_current_labels.extend(sorted([os.path.join(cp_synthetic_lbl_dir, f) for f in os.listdir(cp_synthetic_lbl_dir) if f.endswith('.txt')]))

                # Re-sync and sort the combined list
                img_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_current_images}
                lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in all_current_labels}
                common_stems = list(img_stems.intersection(lbl_stems))
                current_img_paths = [p for p in all_current_images if os.path.splitext(os.path.basename(p))[0] in common_stems]
                current_lbl_paths = [p for p in all_current_labels if os.path.splitext(os.path.basename(p))[0] in common_stems]
                current_img_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
                current_lbl_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

                logging.info(f"Total samples after synthetic data (current cycle): {len(current_img_paths)} (Cycle {cycle+1})")
                continuous_validation(current_img_paths, current_lbl_paths, step_name=f"cycle_{cycle+1}_post_synthetic", num_workers=cfg['resources']['num_workers'])
            else:
                logging.info(f"No new synthetic data generated this cycle (Cycle {cycle+1}). Dataset size remains: {len(current_img_paths)}")
        else:
            logging.info(f"Synthetic data generation not enabled (Cycle {cycle+1}).")


        # --- Conceptual: Retrain EmbedDetector on the augmented dataset ---
        if self_training_enabled and cycle < self_training_cycles -1: # Don't retrain after the last cycle
            logging.info(f"[CONCEPTUAL_STEP] Retraining EmbedDetector on the augmented dataset (Cycle {cycle+1}).")
            logging.info("In a real pipeline, you would trigger your model training script here.")
            logging.info(f"The training dataset for this cycle would be: {len(current_img_paths)} images with labels (including synthetic and corrected data).")
            # After training, you would ideally reload/update `embed_detector_instance` with the new weights.
            # Example: embed_detector_instance.load_weights("path/to/new/best_model.pt")
            # For this script, we assume the detector's performance conceptually improves.
        elif self_training_enabled and cycle == self_training_cycles - 1:
            logging.info(f"[INFO] Final self-training cycle complete. No further retraining needed.")


    # 9. Stats (Final)
    balance = compute_class_balance(current_lbl_paths, len(class_names))
    mean, std = compute_mean_std(current_img_paths, num_workers=cfg['resources']['num_workers'])
    logging.info(f"Final Class balance: {balance}")
    logging.info(f"Final Dataset mean: {mean}, std: {std}")
    if mlflow_enabled:
        try:
            mlflow.log_dict(balance, "final_class_balance.json")
            mlflow.log_dict({"mean": mean, "std": std}, "final_mean_std.json")
        except Exception as e:
            logging.error(f"MLflow logging for final stats failed: {e}")

    # 10. Split (Final)
    logging.info("Performing final dataset splitting...")
    if split_dataset:
        train_ratio = cfg['split'].get('train_size', 0.8)
        val_ratio = cfg['split'].get('val_size', 0.1)
        test_ratio = cfg['split'].get('test_size', 0.1)

        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            logging.warning(f"Split ratios do not sum to 1.0 (sum={total_ratio}). Adjusting test ratio.")
            test_ratio = 1.0 - (train_ratio + val_ratio)
            if test_ratio < 0: test_ratio = 0
        
        pseudo_label_hook = None
        if pseudo_labeling and cfg.get('pseudo_labeling', {}).get('enabled', False):
            pseudo_label_hook = lambda img_dir, out_label_dir: pseudo_labeling(
                img_dir=img_dir, embed_detector=embed_detector_instance, out_label_dir=out_label_dir,
                detection_prompts=cfg['model'].get('detection_prompts', ["object"]), class_names=class_names,
                class_map=class_map_data, min_score=cfg['model'].get('detection_threshold', 0.3)
            )

        active_learning_hook = None
        if active_learning_step and cfg.get('active_learning', {}).get('enabled', False):
            active_learning_hook = lambda unlabeled_img_dir, selection_count: active_learning_step(
                unlabeled_img_dir=unlabeled_img_dir, embed_detector=embed_detector_instance,
                detection_prompts=cfg['model'].get('detection_prompts', ["object"]), class_names=class_names,
                class_map=class_map_data, selection_count=selection_count
            )

        split_dataset(
            image_paths=current_img_paths, # Use the accumulated paths
            label_paths=current_lbl_paths, # Use the accumulated paths
            splits=(train_ratio, val_ratio, test_ratio),
            out_dir=cfg['paths']['output_dir'],
            pseudo_label_func=pseudo_label_hook,
            active_learning_func=active_learning_hook,
            stratified=cfg['split'].get('stratified', True),
            kfolds=cfg['split'].get('n_splits', None),
            seed=cfg['seed']
        )
        logging.info("Final dataset splitting complete.")
    else:
        logging.error("split_dataset function not available. Skipping final dataset splitting.")
        if mlflow_enabled: mlflow.end_run()
        return

    # Generate YOLO data.yaml and classes.names after splitting
    generate_yaml(cfg['paths']['output_dir'], class_names)
    generate_classes_file(cfg['paths']['output_dir'], class_names)

    # --- Metadata Logging ---
    def collect_stats(processed_dir, out_path_prefix):
        """Collects and logs dataset statistics."""
        images = []
        labels = []
        for split in ['train', 'val', 'test']:
            split_img_dir = os.path.join(processed_dir, 'images', split)
            split_lbl_dir = os.path.join(processed_dir, 'labels', split)
            if os.path.exists(split_img_dir):
                images.extend([os.path.join(split_img_dir, f) for f in os.listdir(split_img_dir) if f.lower().endswith(('.jpg', '.png'))])
            if os.path.exists(split_lbl_dir):
                labels.extend([os.path.join(split_lbl_dir, f) for f in os.listdir(split_lbl_dir) if f.endswith('.txt')])

        total_imgs = len(images)
        total_boxes = 0
        areas = []
        for lbl in labels:
            try:
                with open(lbl, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 5:
                            _, x, y, w, h = map(float, parts)
                            areas.append(w * h)
                            total_boxes += 1
            except Exception as e:
                logging.warning(f"Error reading label file for stats {lbl}: {e}")
                continue

        stats = {
            "total_images": total_imgs,
            "total_boxes": total_boxes,
            "min_box_area": min(areas) if areas else 0,
            "max_box_area": max(areas) if areas else 0,
            "avg_box_area": float(np.mean(areas)) if areas else 0
        }
        json_path = out_path_prefix + '.json'
        csv_path = out_path_prefix + '.csv'

        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Generated JSON stats at {json_path}")

        try:
            import csv
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(stats.keys())
                writer.writerow(stats.values())
            logging.info(f"Generated CSV stats at {csv_path}")
        except ImportError:
            logging.warning("csv module not available for CSV stats export.")

    collect_stats(cfg['paths']['output_dir'], os.path.join(cfg['paths']['output_dir'], 'metadata_full'))

    # --- Advanced Augmentation (Albumentations, MixUp, Mosaic) ---
    try:
        from .augmentations import make_pipeline, augment_dataset as augment_dataset_func
        
        if cfg.get('augmentation', {}).get('enabled', False):
            logging.info("Applying Albumentations-based augmentation...")
            pipeline = make_pipeline(cfg.get('augment', {})) # Pass 'augment' section for specific params
            if pipeline:
                train_img_dir = os.path.join(cfg['paths']['output_dir'], 'images', 'train')
                train_lbl_dir = os.path.join(cfg['paths']['output_dir'], 'labels', 'train')
                out_img_aug = os.path.join(cfg['paths']['output_dir'], 'images', 'train_augmented')
                out_lbl_aug = os.path.join(cfg['paths']['output_dir'], 'labels', 'train_augmented')
                
                augment_dataset_func(
                    img_dir=train_img_dir,
                    lbl_dir=train_lbl_dir,
                    out_img_dir=out_img_aug,
                    out_lbl_dir=out_lbl_aug,
                    pipeline=pipeline,
                    num_workers=cfg['resources']['num_workers']
                )
                logging.info(f"Augmented data saved to {out_img_aug} and {out_lbl_aug}")
            else:
                logging.warning("Augmentation pipeline could not be created. Skipping augmentation.")

    except ImportError:
        logging.warning("Albumentations or required augmentation functions not found. Skipping advanced augmentation.")
    except Exception as e:
        logging.error(f"Albumentations augmentation step failed: {e}")
        import traceback
        traceback.print_exc()

    # --- DALI Integration (Conceptual) ---
    if DALI_AVAILABLE and cfg.get('dali', {}).get('enabled', False):
        logging.info("Attempting DALI pipeline integration (conceptual)...")
        if dali_prepare_dataset:
            try:
                # This would typically involve passing paths to DALI for efficient loading
                # and potentially DALI-specific augmentations.
                dali_prepare_dataset(cfg['paths']['output_dir'], cfg['dali'])
                logging.info("DALI pipeline conceptually integrated and run.")
            except Exception as e:
                logging.error(f"DALI pipeline preparation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logging.warning("DALI is enabled in config but `dali_prepare_dataset` function is not available.")
    else:
        logging.info("DALI integration not enabled or DALI not available.")


    # 11. Test-Time Augmentation (TTA) for validation (example usage)
    if cfg.get('test_time_augmentation', {}).get('enable', False) and embed_detector_instance:
        logging.info("Running Test-Time Augmentation (TTA)...")
        val_img_dir = os.path.join(cfg['paths']['output_dir'], 'images', 'val')
        val_lbl_dir = os.path.join(cfg['paths']['output_dir'], 'labels', 'val')
        
        sample_img_paths = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))])
        sample_lbl_paths = sorted([os.path.join(val_lbl_dir, f) for f in os.listdir(val_lbl_dir) if f.endswith('.txt')])
        
        if sample_img_paths and sample_lbl_paths:
            sample_img_path = sample_img_paths[0]
            sample_labels = []
            try:
                with open(sample_lbl_paths[0], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            sample_labels.append(list(map(float, parts)))
            except Exception as e:
                logging.warning(f"Could not read sample label for TTA: {e}")

            image_for_tta = Image.open(sample_img_path).convert('RGB')
            tta_results = test_time_augmentation(
                image=image_for_tta, 
                model=embed_detector_instance, 
                detection_prompts=cfg['model'].get('detection_prompts', ["object"])
            )
            logging.info(f"TTA performed on sample image {sample_img_path}. Results collected.")
        else:
            logging.warning("No validation images/labels found for TTA example.")
    elif cfg.get('test_time_augmentation', {}).get('enable', False):
        logging.warning("TTA enabled but EmbedDetector not available. Skipping.")

    # 12. Error analysis
    if cfg.get('error_analysis', {}).get('enable', False) and embed_detector_instance:
        logging.info("Running error analysis...")
        val_img_dir = os.path.join(cfg['paths']['output_dir'], 'images', 'val')
        val_lbl_dir = os.path.join(cfg['paths']['output_dir'], 'labels', 'val')
        
        val_img_paths = sorted([os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png'))])
        val_lbl_paths = sorted([os.path.join(val_lbl_dir, f) for f in os.listdir(val_lbl_dir) if f.endswith('.txt')])

        if val_img_paths and val_lbl_paths:
            error_report = error_analysis(
                val_img_paths, val_lbl_paths, 
                model=embed_detector_instance, 
                detection_prompts=cfg['model'].get('detection_prompts', ["object"]),
                num_workers=cfg['resources']['num_workers']
            )
            logging.info(f"Error analysis complete. Report: {error_report}")
            if mlflow_enabled:
                try:
                    mlflow.log_dict(error_report, "error_analysis_report.json")
                except Exception as e:
                    logging.error(f"MLflow logging for error analysis failed: {e}")
        else:
            logging.warning("No validation images/labels found for error analysis.")
    elif cfg.get('error_analysis', {}).get('enable', False):
        logging.warning("Error analysis enabled but EmbedDetector not available. Skipping.")

    # 13. Visualization
    if cfg.get('visualization', {}).get('enabled', False): # Use 'enabled' for consistency
        logging.info("Generating visualizations...")
        try:
            from .augmentations import visualize_augmentations # Import here to avoid circular
            train_img_dir = os.path.join(cfg['paths']['output_dir'], 'images', 'train_augmented')
            train_lbl_dir = os.path.join(cfg['paths']['output_dir'], 'labels', 'train_augmented')
            
            if os.path.exists(train_img_dir) and os.path.exists(train_lbl_dir):
                sample_img = next((os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.png'))), None)
                if sample_img:
                    sample_lbl = os.path.join(train_lbl_dir, os.path.basename(sample_img).replace(os.path.splitext(sample_img)[1], '.txt'))
                    if os.path.exists(sample_lbl):
                        visualize_augmentations(
                            image_path=sample_img, 
                            label_path=sample_lbl, 
                            out_dir=os.path.join(cfg['paths']['output_dir'], 'visualizations'), 
                            n=cfg['visualization'].get('sample_count', 5),
                            cfg=cfg
                        )
                        logging.info("Augmentation visualizations generated.")
                    else:
                        logging.warning(f"No label found for sample image {sample_img} for visualization.")
                else:
                    logging.warning("No augmented images found for visualization.")
            else:
                logging.warning("Augmented training directories not found for visualization.")
        except ImportError:
            logging.warning("visualize_augmentations function not found. Skipping visualization.")
        except Exception as e:
            logging.error(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()


    logging.info("All advanced dataset preparation steps completed.")
    if mlflow_enabled:
        mlflow.end_run()
