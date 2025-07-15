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

# Local imports (assuming these are in the same project structure)
from .logger import setup_logging, log_detection
# from .embedding_model import EmbedDetector # Assuming EmbedDetector is in auto_labeling or a dedicated model file
from .metrics import Evaluator
# Import EmbedDetector from auto_labeling if it's placed there
try:
    from .auto_labeling import EmbedDetector
except ImportError:
    EmbedDetector = None
    logging.warning("EmbedDetector not found. OWL-ViT functionality will be limited.")


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

# --- Data Validation and Statistics (from main_pipeline context) ---
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
    return 0 <= x1_abs < x2_abs <= img_w and 0 <= y1_abs < y2_abs <= img_h

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

def filter_valid_samples(images, labels):
    """
    Filters out image-label pairs that fail validation.
    """
    valid_imgs = []
    valid_lbls = []
    for img, lbl in tqdm(zip(images, labels), total=len(images), desc="Validating Samples"):
        if validate_image_and_labels(img, lbl):
            valid_imgs.append(img)
            valid_lbls.append(lbl)
    logging.info(f"Filtered {len(images) - len(valid_imgs)} invalid samples. {len(valid_imgs)} valid samples remaining.")
    return valid_imgs, valid_lbls

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

def compute_mean_std(image_paths, num_samples=500):
    """Computes mean and standard deviation of image pixel values."""
    sampled = random.sample(image_paths, min(num_samples, len(image_paths)))
    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0
    for p in sampled:
        img = cv2.imread(p)
        if img is None:
            continue
        img = img.astype(np.float32)/255.0 # Normalize to [0, 1]
        mean += img.mean(axis=(0,1))
        std += img.std(axis=(0,1))
        count += 1
    if count > 0:
        mean /= count
        std /= count
    return mean.tolist(), std.tolist()

# --- Test-Time Augmentation (TTA) ---
def test_time_augmentation(image, model, tta_transforms=None):
    """
    Applies TTA (e.g., horizontal flip, scale) and potentially averages predictions.
    This is a placeholder; actual implementation depends on the model's prediction interface.
    Args:
        image (np.array or PIL.Image): Input image.
        model: Trained detection model with a .detect(image) method.
        tta_transforms (list): List of transformation functions to apply.
    Returns:
        list: List of predictions from each augmented view.
    """
    if model is None:
        logging.warning("Model not provided for Test-Time Augmentation. Skipping TTA.")
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
                aug_img = np.array(aug_img)

            # Assuming model.detect takes PIL Image and prompts
            # If your model.detect takes numpy array, adjust this.
            preds = model.detect(aug_img, prompts=["object"]) # Assuming OWL-ViT detect interface
            all_preds.append(preds)
        except Exception as e:
            logging.error(f"Error during TTA application: {e}")
            continue
    logging.info(f"Test-Time Augmentation applied, collected {len(all_preds)} sets of predictions.")
    return all_preds

# --- Synthetic Data Generation ---
def generate_synthetic_data(generator_model, n_samples, out_dir):
    """
    Generates synthetic images and labels using a generator model (e.g., GAN, Diffusion).
    This is a placeholder; actual implementation depends on the generator model interface.
    Args:
        generator_model: An instantiated generative model with a `generate()` method.
                         Expected to return (PIL.Image, list of dicts for labels).
                         Labels dict should contain 'class_idx', 'x_c', 'y_c', 'w', 'h'.
        n_samples (int): Number of synthetic samples to generate.
        out_dir (str): Output directory for generated images and labels.
    """
    if generator_model is None:
        logging.warning("[WARN] No generator model provided for synthetic data. Skipping generation.")
        return

    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Generating {n_samples} synthetic samples into {out_dir}...")

    for i in tqdm(range(n_samples), desc="Generating synthetic data"):
        try:
            # Example: generator_model.generate() returns PIL Image and list of label dicts
            img, label_data = generator_model.generate() # Assuming this interface

            img_path = os.path.join(out_dir, f'synth_{i:05d}.jpg')
            lbl_path = os.path.join(out_dir, f'synth_{i:05d}.txt')

            img.save(img_path)
            with open(lbl_path, 'w') as f:
                for obj in label_data:
                    # Ensure obj has required keys and values are floats/ints
                    class_idx = obj.get('class_idx', 0)
                    x_c = obj.get('x_c', 0.5)
                    y_c = obj.get('y_c', 0.5)
                    w = obj.get('w', 0.1)
                    h = obj.get('h', 0.1)
                    f.write(f"{class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        except Exception as e:
            logging.error(f"Error generating synthetic sample {i}: {e}")
            continue
    logging.info(f"[INFO] Generated {n_samples} synthetic samples in {out_dir}")

# --- Hard Negative Mining ---
def hard_negative_mining(image_paths, label_paths, model, threshold=0.3):
    """
    Finds hard negatives (false positives or high uncertainty detections) for curriculum learning.
    Args:
        image_paths (list): List of paths to images.
        label_paths (list): List of paths to corresponding YOLO label files.
        model: Trained detection model with a .detect(image, prompts) method (e.g., EmbedDetector).
        threshold (float): Confidence threshold for considering a detection.
    Returns:
        list: List of image paths identified as hard negatives.
    """
    if model is None:
        logging.warning("Model not provided for hard negative mining. Skipping.")
        return []

    hard_negatives = []
    logging.info("Starting hard negative mining...")

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Hard Negative Mining"):
        try:
            image_pil = Image.open(img_path).convert('RGB')
            # Get predictions from the model
            boxes, scores, _, _ = model.detect(image_pil, prompts=["object"]) # Assuming "object" is a general prompt

            # Load ground truth boxes
            gt_boxes = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_boxes.append(list(map(float, parts[1:]))) # YOLO format: cx, cy, w, h

            # Identify false positives (predictions with high confidence but no good GT match)
            # Or identify images where the model is highly uncertain (e.g., low max score)
            fp_count = 0
            if len(boxes) > 0 and len(gt_boxes) > 0:
                # Simple IoU matching for FP detection
                # Convert YOLO GT to xyxy for IoU check against model's xyxy output
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
                # If there are predictions but no ground truth, all confident predictions are FPs
                fp_count = sum(1 for s in scores if s >= threshold)

            # Define a heuristic for hard negative: e.g., many FPs or low overall confidence
            if fp_count > 0 or (len(scores) > 0 and np.max(scores) < 0.5): # Example heuristic
                hard_negatives.append(img_path)

        except Exception as e:
            logging.error(f"Error processing {img_path} for hard negative mining: {e}")
            continue

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

def detect_label_noise(label_paths, iou_thresh=0.7, min_area=0.001):
    """
    Detects label noise: duplicate boxes, tiny boxes, and potentially other issues.
    Args:
        label_paths (list): List of paths to YOLO label files.
        iou_thresh (float): IoU threshold for considering boxes as duplicates.
        min_area (float): Minimum normalized area for a valid bounding box.
    Returns:
        dict: A dictionary where keys are label file paths and values are lists of issues found.
    """
    issues = defaultdict(list)
    logging.info("Detecting label noise...")

    for lbl_path in tqdm(label_paths, desc="Detecting Label Noise"):
        bboxes = []
        try:
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                continue # Empty file, no noise to detect

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    issues[lbl_path].append('Malformed line')
                    continue
                try:
                    cls_id, x, y, w, h = map(float, parts)
                except ValueError:
                    issues[lbl_path].append('Invalid numeric value')
                    continue

                bbox_yolo = (x, y, w, h)
                bboxes.append(bbox_yolo)

                if w * h < min_area:
                    issues[lbl_path].append('Tiny box')

            # Check for duplicate boxes (high IoU)
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    iou = _yolo_iou(bboxes[i], bboxes[j])
                    if iou > iou_thresh:
                        issues[lbl_path].append('Duplicate/overlapping boxes')

        except Exception as e:
            logging.error(f"Error processing {lbl_path} for noise detection: {e}")
            issues[lbl_path].append(f'Processing error: {e}')
    logging.info(f"Label noise detection complete. Found issues in {len(issues)} files.")
    return dict(issues) # Convert defaultdict to dict for final return

def audit_data_quality(label_paths, image_paths=None):
    """
    Automated data quality audit: outlier detection, duplicate removal, annotation error visualization.
    This is a high-level function that calls other detection functions.
    Args:
        label_paths (list): List of paths to YOLO label files.
        image_paths (list, optional): List of paths to corresponding images. Not directly used for audit but useful context.
    Returns:
        dict: A report containing various data quality issues.
    """
    logging.info("Running comprehensive data quality audit...")
    audit_report = {}

    # 1. Label noise detection
    label_noise_issues = detect_label_noise(label_paths)
    audit_report['label_noise'] = label_noise_issues

    # 2. Outlier detection (e.g., extreme bbox sizes or counts)
    outliers = defaultdict(list)
    for lbl_path in tqdm(label_paths, desc="Detecting Outliers"):
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
                # Flag images with very large or very small boxes
                if np.max(areas) > 0.5: # Example: box covers more than 50% of image
                    outliers[lbl_path].append('Very large bbox detected')
                if np.min(areas) < 0.0001: # Example: box covers less than 0.01% of image
                    outliers[lbl_path].append('Very small bbox detected')
                # Flag images with an unusually high number of objects
                if len(bboxes) > 50: # Example: more than 50 objects
                    outliers[lbl_path].append('Excessive number of objects')
        except Exception as e:
            logging.error(f"Error during outlier detection for {lbl_path}: {e}")
            outliers[lbl_path].append(f'Processing error: {e}')
    audit_report['outliers'] = dict(outliers)

    # 3. Consistency checks (e.g., image exists for label, label exists for image, bbox within image bounds)
    consistency_issues = defaultdict(list)
    if image_paths: # Only run if image paths are provided
        image_path_map = {os.path.basename(p).split('.')[0]: p for p in image_paths}
        for lbl_path in tqdm(label_paths, desc="Checking Consistency"):
            img_id = os.path.basename(lbl_path).split('.')[0]
            if img_id not in image_path_map:
                consistency_issues[lbl_path].append('Corresponding image missing')
            else:
                # Detailed bbox integrity check using image dimensions
                img_path = image_path_map[img_id]
                if not validate_image_and_labels(img_path, lbl_path):
                    # validate_image_and_labels already logs specific warnings
                    consistency_issues[lbl_path].append('Bounding box integrity issues (see warnings above)')
    audit_report['consistency_checks'] = dict(consistency_issues)

    logging.info("Data quality audit complete.")
    return audit_report

def correct_labels_with_model(label_path, image_path, model, threshold=0.5, human_review=False):
    """
    Semi-automatic label correction using model predictions and optional human review.
    Args:
        label_path (str): Path to the YOLO label file to correct.
        image_path (str): Path to the corresponding image.
        model: Trained detection model with a .detect(image, prompts) method (e.g., EmbedDetector).
        threshold (float): Confidence threshold for model predictions to be considered.
        human_review (bool): If True, a human review step is indicated (placeholder).
    Returns:
        list: List of corrected labels (YOLO format: class_id, cx, cy, w, h).
    """
    if model is None:
        logging.warning("Model not provided for label correction. Skipping correction for " + label_path)
        return []

    logging.info(f"Attempting to correct labels for {label_path} using model...")
    try:
        image_pil = Image.open(image_path).convert('RGB')
        h_img, w_img = image_pil.height, image_pil.width

        # Get model predictions
        pred_boxes_xyxy, pred_scores, pred_label_indices, _ = model.detect(image_pil, prompts=["object"]) # General prompt

        # Load existing ground truth labels
        gt_labels_yolo = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_labels_yolo.append(list(map(float, parts))) # [class_id, cx, cy, w, h]

        corrected_labels_yolo = []
        # Strategy: Keep high-confidence model predictions and merge with existing GT if no strong match
        # This is a simplified strategy; a robust system would involve more complex NMS/matching

        # Convert GT YOLO to XYXY for IoU comparison
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
                is_new_detection = True
                for j, gt_box_xyxy in enumerate(gt_boxes_xyxy):
                    if _iou_xyxy(pred_box_xyxy, gt_box_xyxy) > 0.7: # High IoU means it's likely a match
                        # If a prediction matches an existing GT, we can choose to keep GT or replace with prediction
                        # For simplicity, we'll mark GT as matched and consider the prediction for addition
                        matched_gt_indices.add(j)
                        is_new_detection = False
                        break
                if is_new_detection:
                    # Convert new prediction to YOLO format
                    x1, y1, x2, y2 = pred_box_xyxy
                    cx = ((x1 + x2) / 2) / w_img
                    cy = ((y1 + y2) / 2) / h_img
                    bw = (x2 - x1) / w_img
                    bh = (y2 - y1) / h
                    class_id = int(pred_label_indices[i]) # Using OWL-ViT's label index as class ID
                    corrected_labels_yolo.append([class_id, cx, cy, bw, bh])

        # Add back any GT labels that were not matched by a confident prediction
        for j, gt_label in enumerate(gt_labels_yolo):
            if j not in matched_gt_indices:
                corrected_labels_yolo.append(gt_label)

        # Optionally: human review step (placeholder)
        if human_review:
            logging.info(f"[REVIEW] Please manually review corrected labels for {label_path}.")
            # In a real application, this would involve displaying the image with new/old labels
            # and providing an interface for human correction.

        # Save corrected labels
        with open(label_path, 'w') as f:
            for label_data in corrected_labels_yolo:
                f.write(f"{int(label_data[0])} {label_data[1]:.6f} {label_data[2]:.6f} {label_data[3]:.6f} {label_data[4]:.6f}\n")
        logging.info(f"Corrected labels saved to {label_path}. Total objects: {len(corrected_labels_yolo)}")
        return corrected_labels_yolo
    except Exception as e:
        logging.error(f"Error during label correction for {label_path}: {e}")
        return []

# --- Curriculum Learning ---
def curriculum_sampling(image_paths, label_paths, model=None, stages=3):
    """
    Progressive sampling: start with easy samples, add harder ones (rare classes, hard negatives).
    Args:
        image_paths (list): List of paths to images.
        label_paths (list): List of paths to corresponding YOLO label files.
        model: Optional trained detection model (e.g., EmbedDetector) for hard negative mining.
        stages (int): Number of curriculum stages.
    Returns:
        tuple: (list of selected image paths, list of selected label paths)
    """
    logging.info(f"[INFO] Curriculum sampling: {stages} stages.")

    all_selected_imgs = []
    all_selected_lbls = []

    # Stage 1: Easy samples (e.g., large boxes, common classes)
    easy_imgs, easy_lbls = [], []
    for img, lbl in zip(image_paths, label_paths):
        try:
            with open(lbl, 'r') as f:
                lines = f.readlines()
            if not lines: continue # Skip empty label files for easy samples

            is_easy = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, _, _, w, h = map(float, parts)
                    if w > 0.2 and h > 0.2: # Example: boxes larger than 20% of image
                        is_easy = True
                        break
            if is_easy:
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
                    if parts:
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
                if parts and int(parts[0]) in rare_classes:
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
    if model is not None:
        logging.info("Stage 3 (Hard Negatives): Identifying hard negatives...")
        hard_imgs = hard_negative_mining(image_paths, label_paths, model)
        # Filter out images already selected and ensure corresponding labels exist
        hard_lbls = [lbl for img, lbl in zip(image_paths, label_paths) if img in hard_imgs and img not in all_selected_imgs]
        logging.info(f"Stage 3 (Hard Negatives): {len(hard_imgs)} selected.")
        all_selected_imgs.extend(hard_imgs)
        all_selected_lbls.extend(hard_lbls)
    else:
        logging.warning("Skipping Stage 3 (Hard Negatives): Model not provided.")

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
    # all_selected_lbls.extend(meta_lbls)


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
def error_analysis(image_paths, label_paths, model):
    """
    Runs error analysis: finds FP, FN, low-confidence, and label mismatches.
    Args:
        image_paths (list): List of paths to images.
        label_paths (list): List of paths to corresponding YOLO label files.
        model: Trained detection model with a .detect(image, prompts) method (e.g., EmbedDetector).
    Returns:
        dict: A report containing error counts per image.
    """
    if model is None:
        logging.warning("Model not provided for error analysis. Skipping.")
        return {}

    logging.info("[INFO] Running error analysis...")
    error_report = defaultdict(lambda: {'fp': 0, 'fn': 0, 'low_conf': 0})

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc="Error Analysis"):
        try:
            image_pil = Image.open(img_path).convert('RGB')
            h_img, w_img = image_pil.height, image_pil.width

            # Get model predictions
            pred_boxes_xyxy, pred_scores, _, _ = model.detect(image_pil, prompts=["object"])

            # Load ground truth boxes (YOLO format)
            gt_boxes_yolo = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_boxes_yolo.append(list(map(float, parts[1:]))) # cx, cy, w, h

            # Convert GT YOLO to XYXY for IoU comparison
            gt_boxes_xyxy = []
            for gt_cx, gt_cy, gt_w, gt_h in gt_boxes_yolo:
                gt_x1 = (gt_cx - gt_w/2) * w_img
                gt_y1 = (gt_cy - gt_h/2) * h_img
                gt_x2 = (gt_cx + gt_w/2) * w_img
                gt_y2 = (gt_cy + gt_h/2) * h_img
                gt_boxes_xyxy.append([gt_x1, gt_y1, gt_x2, gt_y2])

            # False Positives (FP): Prediction with high confidence, no corresponding GT
            # False Negatives (FN): GT with no corresponding prediction
            # Low Confidence: Prediction below a certain threshold

            matched_preds = [False] * len(pred_boxes_xyxy)
            matched_gts = [False] * len(gt_boxes_xyxy)

            # Match predictions to ground truths
            for i, pred_box in enumerate(pred_boxes_xyxy):
                if pred_scores[i] < 0.3: # Low confidence threshold for counting
                    error_report[img_path]['low_conf'] += 1
                    continue

                best_iou = 0
                best_gt_idx = -1
                for j, gt_box in enumerate(gt_boxes_xyxy):
                    current_iou = _iou_xyxy(pred_box, gt_box)
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_gt_idx = j

                if best_iou > 0.5: # If a good match is found
                    matched_preds[i] = True
                    matched_gts[best_gt_idx] = True
                else: # No good GT match for this confident prediction
                    error_report[img_path]['fp'] += 1

            # Count False Negatives
            for j, gt_box in enumerate(gt_boxes_xyxy):
                if not matched_gts[j]:
                    error_report[img_path]['fn'] += 1

        except Exception as e:
            logging.error(f"Error during error analysis for {img_path}: {e}")
            # Mark as error in report
            error_report[img_path]['processing_error'] = str(e)
            continue

    logging.info("[INFO] Error analysis complete.")
    return dict(error_report)

def continuous_validation(image_paths, label_paths, step_name="validation"):
    """
    Automated validation after each pipeline step, with error reports.
    Args:
        image_paths (list): List of paths to images.
        label_paths (list): List of paths to corresponding YOLO label files.
        step_name (str): Name of the current pipeline step for logging.
    Returns:
        dict: A dictionary of images with validation issues.
    """
    logging.info(f"[VALIDATION] Running continuous validation for step: {step_name}...")
    errors = defaultdict(list)

    for img_path, lbl_path in tqdm(zip(image_paths, label_paths), total=len(image_paths), desc=f"Validating {step_name}"):
        # Check file existence
        if not os.path.exists(img_path):
            errors[img_path].append('Image file missing')
        if not os.path.exists(lbl_path):
            errors[img_path].append('Label file missing')

        # If both exist, perform detailed label validation
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            if not validate_image_and_labels(img_path, lbl_path):
                # validate_image_and_labels already logs specific warnings
                errors[img_path].append('Bounding box integrity or format issues')

    logging.info(f"[VALIDATION] {step_name}: {len(errors)} images found with issues.")
    try: # Assuming mlflow is imported at the top level of the main script
        import mlflow
        mlflow.log_dict(dict(errors), f"validation_errors_{step_name}.json")
    except ImportError:
        pass # MLflow not available
    except Exception as e:
        logging.error(f"MLflow logging failed for validation step '{step_name}': {e}")
    return dict(errors)

# --- Main Dataset Processing Logic (from main_pipeline context) ---
def parse_args():
    """Parses command-line arguments for the data preparation utility."""
    parser = argparse.ArgumentParser(description="YOLO Data Preparation Utility")
    parser.add_argument('--input-dir', required=True, help='Directory with raw images')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--annotations', required=True, help='Path to annotation file (JSON, CSV, or XML)')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--classes-file', required=True, help='Path to class names file (one class per line)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml (optional)')
    parser.add_argument('--use-owlvit', action='store_true', help='Use OWL-ViT detection pipeline')
    return parser.parse_args()

def ensure_dirs(base_dir):
    """Creates necessary subdirectories for YOLO dataset structure."""
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
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
    This function combines elements from the original "main_pipeline" and "data_preparation_utils"
    sections of the user's provided document.
    """
    logging.info("Starting dataset processing...")

    cfg = None
    if getattr(args, 'config', None) and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        setup_logging(cfg.get('logging', {})) # Ensure logging is set up from config
        logging.info(f"Loaded configuration from {args.config}")
    else:
        logging.warning("No config file provided or found. Proceeding with default arguments.")

    # If use_owlvit is true and EmbedDetector is available, run OWL-ViT detection pipeline
    if getattr(args, 'use_owlvit', False) and EmbedDetector is not None:
        logging.info("Using OWL-ViT detection pipeline for initial labeling.")
        if not cfg or 'model' not in cfg:
            logging.error("Cannot use OWL-ViT without model configuration in config.yaml.")
            return

        detector = EmbedDetector(cfg['model'])
        evaluator = Evaluator(cfg['metrics']) # Assuming Evaluator is available

        images_dir = Path(args.input_dir) # Use input-dir from CLI args for raw images
        output_dir = Path(args.output_dir) # Use output-dir from CLI args for processed data
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for images and labels within the output_dir
        output_images_base = output_dir / "images"
        output_labels_base = output_dir / "labels"
        output_images_base.mkdir(parents=True, exist_ok=True)
        output_labels_base.mkdir(parents=True, exist_ok=True)


        for img_path in tqdm(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")), desc="OWL-ViT Labeling"):
            image = Image.open(img_path).convert("RGB")
            # Prompts should ideally come from config or be passed as an argument
            prompts = cfg['model'].get('detection_prompts', ["object"])
            boxes, scores, labels, embeddings = detector.detect(image, prompts)

            # Write YOLO-format label file
            label_file = output_labels_base / f"{img_path.stem}.txt" # Save labels to output_dir/labels
            with open(label_file, 'w') as f:
                for box, score, label_idx in zip(boxes, scores, labels):
                    # Filter by score if needed
                    if score < detector.threshold: # Use detector's threshold
                        continue
                    x0,y0,x1,y1 = box
                    width, height = image.size
                    cx, cy = (x0 + x1)/2/width, (y0 + y1)/2/height
                    w, h = (x1 - x0)/width, (y1 - y0)/height
                    # Use the label_idx from OWL-ViT directly as class ID
                    f.write(f"{int(label_idx)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            # Copy original image to output_dir/images
            output_image_path = output_images_base / f"{img_path.name}"
            shutil.copy(img_path, output_image_path)

            # Log each detection (if logging is configured)
            for idx, (b, s, lb, emb) in enumerate(zip(boxes, scores, labels, embeddings)):
                if s >= detector.threshold:
                    log_detection(
                        image_path=str(img_path),
                        box=b.tolist(),
                        score=float(s),
                        label=int(lb),
                        embedding=emb.tolist() if emb is not None else []
                    )
            evaluator.update(gt_boxes=None, pred_boxes=boxes, scores=scores) # Update evaluator with predictions
        evaluator.finalize()
        logging.info("[INFO] OWL-ViT detection and YOLO label generation complete.")
        # After OWL-ViT labeling, the rest of the pipeline can proceed with these generated labels
        # The subsequent splitting and YAML generation will use these newly created files.

    # --- Classic YOLO data prep logic (enhanced) ---
    # This part assumes labels are already in YOLO format (either pre-existing or generated by OWL-ViT)
    # It focuses on organizing into train/val splits and generating YAML/class files.
    ensure_dirs(args.output_dir) # Ensure train/val subdirs are ready

    class_names = parse_classes(args.classes_file)
    class_to_idx = {name: i for i, name in enumerate(class_names)} # Not directly used here, but good for mapping

    # Use the labels generated by OWL-ViT or existing ones in output_dir/labels
    lbl_dir_for_split = os.path.join(args.output_dir, 'labels')
    if not os.path.exists(lbl_dir_for_split):
        logging.error(f"Label directory for splitting not found: {lbl_dir_for_split}. Please ensure labels are generated or copied.")
        return

    all_label_files = [f for f in os.listdir(lbl_dir_for_split) if f.endswith('.txt')]
    if not all_label_files:
        logging.warning(f"No label files found in {lbl_dir_for_split}. Skipping splitting and YAML generation.")
        return

    # Collect image IDs corresponding to label files
    all_image_ids = [f.replace('.txt', '.jpg') if os.path.exists(os.path.join(args.output_dir, 'images', f.replace('.txt', '.jpg'))) else \
                     f.replace('.txt', '.png') if os.path.exists(os.path.join(args.output_dir, 'images', f.replace('.txt', '.png'))) else \
                     None for f in all_label_files]
    all_image_ids = [f for f in all_image_ids if f is not None] # Filter out missing images

    if not all_image_ids:
        logging.error("No corresponding images found for label files. Cannot perform split.")
        return

    # --- Stratified Split ---
    # This section needs to be robust and use the `split_dataset` function from `split_dataset.py`
    # and not re-implement the splitting logic.
    logging.info("Performing dataset splitting (train/val/test)...")

    # The `split_dataset` function expects a label directory and will handle image copying.
    # We need to ensure `split_dataset` can access the images from `args.output_dir/images`
    # and labels from `args.output_dir/labels`.
    # `split_dataset` expects a list of label paths.
    full_label_paths = [os.path.join(lbl_dir_for_split, f) for f in all_label_files]
    full_image_paths = [os.path.join(os.path.join(args.output_dir, 'images'), img_id) for img_id in all_image_ids]

    # Use the `split_dataset` function from `split_dataset.py`
    # It will create train/val/test directories under `args.output_dir`
    # and copy images/labels into them.
    # The `splits` argument should come from config or default.
    train_ratio = cfg.get('split', {}).get('train_size', 0.8) if cfg else args.split_ratio
    val_ratio = cfg.get('split', {}).get('val_size', 0.1) if cfg else 0.1 # Assuming 0.1 default for val
    test_ratio = cfg.get('split', {}).get('test_size', 0.1) if cfg else 0.1 # Assuming 0.1 default for test

    # Sum of ratios might not be exactly 1.0, adjust test_ratio if necessary
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01): # Allow for floating point inaccuracies
        logging.warning(f"Split ratios do not sum to 1.0 (sum={total_ratio}). Adjusting test ratio.")
        test_ratio = 1.0 - (train_ratio + val_ratio)
        if test_ratio < 0: test_ratio = 0

    # Pass the actual function references for pseudo_labeling and active_learning_step
    # These functions are defined in auto_labeling.py
    try:
        from .auto_labeling import pseudo_labeling, active_learning_step
        pseudo_label_hook = pseudo_labeling if cfg and cfg.get('pseudo_labeling', {}).get('enabled', False) else None
        active_learning_hook = active_learning_step if cfg and cfg.get('active_learning', {}).get('enabled', False) else None
    except ImportError:
        logging.warning("auto_labeling module not found. Pseudo-labeling and active learning hooks will be disabled.")
        pseudo_label_hook = None
        active_learning_hook = None

    try:
        from .split_dataset import split_dataset as actual_split_dataset
        actual_split_dataset(
            label_dir=lbl_dir_for_split, # Directory containing all labels to be split
            splits=(train_ratio, val_ratio, test_ratio),
            out_dir=args.output_dir, # Output root for train/val/test folders
            pseudo_label_func=pseudo_label_hook,
            active_learning_func=active_learning_hook,
            stratified=cfg.get('split', {}).get('stratified', True) if cfg else True,
            kfolds=cfg.get('split', {}).get('n_splits', None) if cfg else None,
            seed=args.seed
        )
        logging.info("Dataset splitting complete.")
    except ImportError:
        logging.error("split_dataset module not found. Skipping dataset splitting.")
        return # Cannot proceed without splitting

    # Generate YOLO data.yaml and classes.names after splitting
    generate_yaml(args.output_dir, class_names)
    generate_classes_file(args.output_dir, class_names)

    # --- Metadata Logging ---
    def collect_stats(processed_dir, out_path_prefix):
        """Collects and logs dataset statistics."""
        images = []
        labels = []
        for split in ['train', 'val', 'test']:
            split_img_dir = os.path.join(processed_dir, split, 'images')
            split_lbl_dir = os.path.join(processed_dir, split, 'labels')
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

    collect_stats(args.output_dir, os.path.join(args.output_dir, 'metadata_full'))

    # --- Advanced Augmentation (Albumentations, MixUp, Mosaic) ---
    # This section should ideally be handled by `apply_augmentations` from `augmentations.py`
    # and potentially integrated into a data loader, rather than a separate script step.
    # The current code snippet here is complex and might be better as an example or part of a DALI pipeline.
    # For now, I'll keep the structure but comment out the multiprocessing part if it's not fully functional.
    try:
        from .augmentations import make_pipeline, augment_dataset as augment_dataset_func
        
        logging.info("Attempting Albumentations-based augmentation (example)...")
        if cfg and cfg.get('augmentation', {}).get('enabled', False):
            pipeline = make_pipeline(cfg.get('augmentation', {}))
            if pipeline:
                train_img_dir = os.path.join(args.output_dir, 'train', 'images')
                train_lbl_dir = os.path.join(args.output_dir, 'train', 'labels')
                out_img_aug = train_img_dir + "_aug"; out_lbl_aug = train_lbl_dir + "_aug"
                
                augment_dataset_func(
                    img_dir=train_img_dir,
                    lbl_dir=train_lbl_dir,
                    out_img_dir=out_img_aug,
                    out_lbl_dir=out_lbl_aug,
                    pipeline=pipeline,
                    num_workers=getattr(args, 'num_workers', 4)
                )
                logging.info(f"Augmented data saved to {out_img_aug} and {out_lbl_aug}")
            else:
                logging.warning("Augmentation pipeline could not be created. Skipping augmentation.")

    except ImportError:
        logging.warning("Albumentations or required augmentation functions not found. Skipping advanced augmentation.")
    except Exception as e:
        logging.error(f"Albumentations augmentation step failed: {e}")

    logging.info("Data preparation complete! Ready for YOLO training.")

# --- CLI entry point (for data_preparation_utils.py specific calls) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="YOLO Data Preparation Utility Functions")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for process_dataset (main pipeline)
    process_parser = subparsers.add_parser('process_dataset', help='Run the full dataset processing pipeline')
    process_parser.add_argument('--input-dir', required=True, help='Directory with raw images')
    process_parser.add_argument('--output-dir', required=True, help='Output directory for YOLO dataset')
    process_parser.add_argument('--annotations', required=True, help='Path to annotation file (JSON, CSV, or XML)')
    process_parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    process_parser.add_argument('--classes-file', required=True, help='Path to class names file (one class per line)')
    process_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    process_parser.add_argument('--config', type=str, default=None, help='Path to config.yaml (optional)')
    process_parser.add_argument('--use-owlvit', action='store_true', help='Use OWL-ViT detection pipeline for initial labeling')
    process_parser.add_argument('--device', type=str, default='cpu', help='Device for OWL-ViT (e.g., "cuda", "cpu")')
    process_parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for multiprocessing (e.g., augmentation)')
    process_parser.set_defaults(func='run_process_dataset')

    # Subparser for visualize_augmentations (from augmentations.py, but useful here)
    aug_viz_parser = subparsers.add_parser('visualize_aug', help='Visualize augmentations on a sample image')
    aug_viz_parser.add_argument("--img", required=True, help="Path to image")
    aug_viz_parser.add_argument("--lbl", required=True, help="Path to label txt")
    aug_viz_parser.add_argument("--out", default="aug_preview", help="Output dir for preview images")
    aug_viz_parser.add_argument("--n", type=int, default=5, help="Number of samples to visualize")
    aug_viz_parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional, for augmentation settings)")
    aug_viz_parser.set_defaults(func='run_visualize_aug')

    # Subparser for collect_stats
    stats_parser = subparsers.add_parser('collect_stats', help='Collect and log dataset statistics')
    stats_parser.add_argument('--data_dir', required=True, help='Root directory of the processed dataset (e.g., output_dir)')
    stats_parser.add_argument('--output_prefix', default='metadata', help='Prefix for output metadata files (JSON/CSV)')
    stats_parser.set_defaults(func='run_collect_stats')

    args = parser.parse_args()

    if args.command == 'run_process_dataset':
        process_dataset(args)
    elif args.command == 'run_visualize_aug':
        # Need to load config for get_train_transforms if it depends on it
        cfg_for_aug = None
        if args.config and os.path.exists(args.config):
            with open(args.config) as f:
                cfg_for_aug = yaml.safe_load(f)
        # Call the visualize_augmentations function (assuming it's imported or defined)
        # It's currently in augmentations.py, so this would require importing it
        # For self-containment in this file, it would need to be defined here.
        # For now, it assumes it's imported from .augmentations
        try:
            from .augmentations import visualize_augmentations as viz_aug_func
            viz_aug_func(args.img, args.lbl, args.out, args.n, cfg=cfg_for_aug)
        except ImportError:
            logging.error("visualize_augmentations function not found in augmentations module. Cannot run visualization.")
    elif args.command == 'run_collect_stats':
        collect_stats(args.data_dir, os.path.join(args.data_dir, args.output_prefix))
    else:
        parser.print_help()
