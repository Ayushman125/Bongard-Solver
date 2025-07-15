import numpy as np
import cv2
def test_time_augmentation(image, model, tta_transforms=None):
    """
    Applies TTA (horizontal flip, scale, etc.) and averages predictions.
    """
    import numpy as np
    if tta_transforms is None:
        tta_transforms = [lambda x: x, lambda x: np.fliplr(x)]
    all_preds = []
    for t in tta_transforms:
        aug_img = t(image)
        preds = model.predict(aug_img)
        all_preds.append(preds)
    # Simple average (customize as needed)
    # Here, just return all predictions for further processing
    return all_preds

def generate_synthetic_data(generator_model, n_samples, out_dir):
    """
    Generates synthetic images and labels using a generator model (GAN, domain randomization, etc.)
    """
    """
    Supports GAN/Diffusion-based generation if specified in config.
    """
    import importlib
    if generator_model is None:
        print("[WARN] No generator model provided for synthetic data.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_samples):
        # If config specifies GAN/Diffusion, use it
        if hasattr(generator_model, 'generate_gan'):
            img, label = generator_model.generate_gan()
        elif hasattr(generator_model, 'generate_diffusion'):
            img, label = generator_model.generate_diffusion()
        else:
            img, label = generator_model.generate()
        img_path = os.path.join(out_dir, f'synth_{i:05d}.jpg')
        lbl_path = os.path.join(out_dir, f'synth_{i:05d}.txt')
        img.save(img_path)
        with open(lbl_path, 'w') as f:
            for obj in label:
                f.write(f"{obj['class_idx']} {obj['x_c']:.6f} {obj['y_c']:.6f} {obj['w']:.6f} {obj['h']:.6f}\n")
    print(f"[INFO] Generated {n_samples} synthetic samples in {out_dir}")

def hard_negative_mining(image_paths, label_paths, model, threshold=0.3):
    """
    Finds hard negatives (false positives) for curriculum learning.
    Returns list of image paths with high model uncertainty or errors.
    """
    import numpy as np
    hard_negatives = []
    for img, lbl in zip(image_paths, label_paths):
        preds = model.predict(img)
        fp_count = sum(1 for p in preds if p[-1] < threshold)
        if fp_count > 0:
            hard_negatives.append(img)
    print(f"[INFO] Found {len(hard_negatives)} hard negatives.")
    # Optionally: retrain model on hard negatives (active retraining)
    # if hasattr(model, 'retrain_on_hard_negatives'):
    #     model.retrain_on_hard_negatives(hard_negatives)
    return hard_negatives
def detect_label_noise(label_paths, iou_thresh=0.7, min_area=0.001):
    pass
def audit_data_quality(label_paths, image_paths=None):
    pass
    """
    Automated data quality audit: outlier detection, duplicate removal, annotation error visualization.
    Returns dict of issues and optionally saves visualizations.
    """
    issues = detect_label_noise(label_paths)
    # Outlier detection: flag images with extreme bbox sizes or counts
    outliers = {}
    for lbl in label_paths:
        bboxes = []
        for line in open(lbl):
            parts = line.strip().split()
            if len(parts) == 5:
                _, x, y, w, h = map(float, parts)
                bboxes.append((x, y, w, h))
        if bboxes:
            areas = [w*h for (_, _, w, h) in bboxes]
            if np.max(areas) > 0.5 or np.min(areas) < 0.0001:
                outliers[lbl] = 'Extreme bbox area'
            if len(bboxes) > 50:
                outliers[lbl] = 'Too many objects'
    # Optionally: visualize annotation errors
    # ...existing code...
    return {'label_noise': issues, 'outliers': outliers}
    """
    Detects label noise: duplicate boxes, tiny boxes, and outliers.
    Returns a dict of file -> issues.
    """
    issues = {}
    for lbl in label_paths:
        bboxes = []
        for line in open(lbl):
            parts = line.strip().split()
            if len(parts) != 5:
                issues.setdefault(lbl, []).append('Malformed line')
                continue
            _, x, y, w, h = map(float, parts)
            bboxes.append((x, y, w, h))
            if w*h < min_area:
                issues.setdefault(lbl, []).append('Tiny box')
        # Check for duplicate boxes (high IoU)
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                iou = _yolo_iou(bboxes[i], bboxes[j])
                if iou > iou_thresh:
                    issues.setdefault(lbl, []).append('Duplicate/overlapping boxes')
    return issues

def _yolo_iou(boxA, boxB):
    # box: (x, y, w, h) in YOLO format
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
    return inter_area / union_area if union_area > 0 else 0

def denoise_bboxes(label_path, iou_thresh=0.7, min_area=0.001):
    pass
def correct_labels_with_model(label_path, image_path, model, threshold=0.5, human_review=False):
    pass
    """
    Semi-automatic label correction using model predictions and optional human review.
    """
    img = cv2.imread(image_path)
    preds = model.predict(img)
    # Compare predictions to existing labels
    with open(label_path) as f:
        gt_lines = f.readlines()
    gt_boxes = [list(map(float, l.strip().split()[1:])) for l in gt_lines if len(l.strip().split()) == 5]
    corrected = []
    for pred in preds:
        # Example correction logic: replace gt with pred if IoU < threshold
        corrected.append(pred)
    # Optionally: human review step
    if human_review:
        print("[REVIEW] Please check corrected labels.")
    # Save corrected labels
    with open(label_path, 'w') as f:
        for box in corrected:
            f.write(' '.join(map(str, box)) + '\n')
    return corrected
    # (Removed stray code. If needed, move to a function.)

def check_label_consistency(image_path, label_path):
    """
    Checks if all boxes are within image bounds and class indices are valid.
    Returns list of issues.
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return ['Image not found']
    h, w = img.shape[:2]
    issues = []
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) != 5:
            issues.append('Malformed line')
            continue
        cls, x, y, bw, bh = map(float, parts)
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            issues.append('Box out of [0,1] range')
        # Check if box is within image bounds
        x1 = (x - bw/2) * w
        y1 = (y - bh/2) * h
        x2 = (x + bw/2) * w
        y2 = (y + bh/2) * h
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            issues.append('Box out of image bounds')
    return issues

import os
import json
import shutil
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import yaml
from pathlib import Path
from logger import setup_logging, log_detection
from embedding_model import EmbedDetector
from metrics import Evaluator

def parse_args():
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
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

def parse_classes(classes_file):
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def parse_annotations(ann_path):
    # Only JSON supported for now; extend as needed
    with open(ann_path) as f:
        data = json.load(f)
    # Expecting: {image_id: {"boxes": [[xmin, ymin, xmax, ymax, class_name], ...], "width": int, "height": int}}
    return data

def convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    x_c = ((xmin + xmax) / 2) / img_w
    y_c = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # Clip to [0,1]
    x_c, y_c, w, h = [max(0, min(1, v)) for v in [x_c, y_c, w, h]]
    return x_c, y_c, w, h

def write_label_file(label_path, objects):
    with open(label_path, 'w') as f:
        for obj in objects:
            f.write(f"{obj['class_idx']} {obj['x_c']:.6f} {obj['y_c']:.6f} {obj['w']:.6f} {obj['h']:.6f}\n")

def split_filenames(filenames, split_ratio, seed=42):
    random.seed(seed)
    random.shuffle(filenames)
    n_train = int(len(filenames) * split_ratio)
    return filenames[:n_train], filenames[n_train:]

def generate_yaml(output_dir, class_names):
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_content = {
        'train': os.path.abspath(os.path.join(output_dir, 'images/train')),
        'val': os.path.abspath(os.path.join(output_dir, 'images/val')),
        'nc': len(class_names),
        'names': class_names
    }
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

def generate_classes_file(output_dir, class_names):
    with open(os.path.join(output_dir, 'classes.names'), 'w') as f:
        for name in class_names:
            f.write(name + '\n')

def process_dataset(args):
    # If config is provided, load YAML config and set up logging/OWL-ViT/metrics
    if getattr(args, 'config', None):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        setup_logging(cfg['logging'])
        if getattr(args, 'use_owlvit', False):
            detector = EmbedDetector(cfg['model'])
            evaluator = Evaluator(cfg['metrics'])
            # Use OWL-ViT detection for all images in images_dir
            images_dir = Path(cfg['data']['images_dir'])
            output_dir = Path(cfg['data']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            for img_path in images_dir.glob("*.jpg"):
                image = Image.open(img_path).convert("RGB")
                prompts = ["object"]  # replace with your class prompts
                boxes, scores, labels, embeddings = detector.detect(image, prompts)
                # Write YOLO-format label file
                label_file = output_dir / f"{img_path.stem}.txt"
                with open(label_file, 'w') as f:
                    for box, score, label in zip(boxes, scores, labels):
                        x0,y0,x1,y1 = box
                        width, height = image.size
                        cx, cy = (x0 + x1)/2/width, (y0 + y1)/2/height
                        w, h = (x1 - x0)/width, (y1 - y0)/height
                        f.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                # Log each detection
                for idx, (b, s, lb, emb) in enumerate(zip(boxes, scores, labels, embeddings)):
                    log_detection(
                        image_path=str(img_path),
                        box=b.tolist(),
                        score=float(s),
                        label=int(lb),
                        embedding=emb.tolist()
                    )
                evaluator.update(gt_boxes=None, pred_boxes=boxes, scores=scores)
            evaluator.finalize()
            print("[INFO] OWL-ViT detection and YOLO label generation complete.")
            return
    # --- Classic YOLO data prep logic (preserved) ---
    ensure_dirs(args.output_dir)
    class_names = parse_classes(args.classes_file)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    ann_data = parse_annotations(args.annotations)
    image_ids = list(ann_data.keys())
    train_ids, val_ids = split_filenames(image_ids, args.split_ratio, args.seed)
    stats = defaultdict(int)
    for split, ids in [('train', train_ids), ('val', val_ids)]:
        for img_id in tqdm(ids, desc=f'Processing {split} set'):
            entry = ann_data[img_id]
            img_path = os.path.join(args.input_dir, img_id)
    """
    Applies TTA (horizontal flip, scale, etc.) and averages predictions.
    """
    import numpy as np
    if tta_transforms is None:
        tta_transforms = [lambda x: x, lambda x: np.fliplr(x), lambda x: np.flipud(x)]
    all_preds = []
    for t in tta_transforms:
        aug_img = t(image)
        preds = model.predict(aug_img)
        all_preds.append(preds)
    # Aggregate predictions (majority vote or mean)
    # For detection, return all for further NMS/aggregation
    print(f"[INFO] TTA applied with {len(tta_transforms)} transforms.")
    return all_preds
    """
    Checks if all boxes are within image bounds and class indices are valid.
    Returns list of issues.
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return ['Image not found']
    h, w = img.shape[:2]
    issues = []
    for line in open(label_path):
        parts = line.strip().split()
        if len(parts) != 5:
            issues.append('Malformed line')
            continue
        cls, x, y, bw, bh = map(float, parts)
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            issues.append('Box out of [0,1] range')
        # Check if box is within image bounds
        x1 = (x - bw/2) * w
        y1 = (y - bh/2) * h
        x2 = (x + bw/2) * w
        y2 = (y + bh/2) * h
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            issues.append('Box out of image bounds')
    if not issues:
        print(f"[INFO] Label consistency check passed for {label_path}")
    else:
        print(f"[WARN] Label consistency issues in {label_path}: {issues}")
    return issues
# --- Curriculum learning utility ---
def curriculum_sampling(image_paths, label_paths, model=None, stages=3):
    """
    Progressive sampling: start with easy samples, add harder ones (rare classes, hard negatives).
    """
    print(f"[INFO] Curriculum sampling: {stages} stages.")
    # Stage 1: easy samples (large boxes, common classes)
    easy_imgs, easy_lbls = [], []
    for img, lbl in zip(image_paths, label_paths):
        for line in open(lbl):
            parts = line.strip().split()
            if len(parts) == 5:
                _, _, _, w, h = map(float, parts)
                if w > 0.2 and h > 0.2:
                    easy_imgs.append(img)
                    easy_lbls.append(lbl)
                    break
    # Stage 2: add rare classes
    from collections import Counter
    class_counts = Counter()
    for lbl in label_paths:
        for line in open(lbl):
            class_counts[int(line.split()[0])] += 1
    rare_classes = [cls for cls, cnt in class_counts.items() if cnt < 20]
    rare_imgs, rare_lbls = [], []
    for img, lbl in zip(image_paths, label_paths):
        for line in open(lbl):
            if int(line.split()[0]) in rare_classes:
                rare_imgs.append(img)
                rare_lbls.append(lbl)
                break
    # Stage 3: add hard negatives (if model provided)
    hard_imgs, hard_lbls = [], []
    if model is not None:
        hard_imgs = hard_negative_mining(image_paths, label_paths, model)
        hard_lbls = [lbl for img, lbl in zip(image_paths, label_paths) if img in hard_imgs]
    # Meta-data driven sampling: prioritize images with diverse object sizes, relations
    from metadata_logger import compute_metadata
    meta_scores = []
    for img, lbl in zip(image_paths, label_paths):
        labels = []
        for line in open(lbl):
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append(list(map(float, parts)))
        meta = compute_metadata(img, labels)
        # Score: prefer images with high obj_count and avg_bbox_area in mid-range
        score = meta['obj_count'] + (0.5 - abs(meta['avg_bbox_area']-0.25))
        meta_scores.append((score, img, lbl))
    meta_scores.sort(reverse=True)
    meta_imgs = [img for _, img, _ in meta_scores[:20]]
    meta_lbls = [lbl for _, _, lbl in meta_scores[:20]]
    # Combine all
    all_imgs = easy_imgs + rare_imgs + hard_imgs + meta_imgs
    all_lbls = easy_lbls + rare_lbls + hard_lbls + meta_lbls
    print(f"[INFO] Curriculum sampling: {len(all_imgs)} samples selected.")
    return all_imgs, all_lbls
# --- Error analysis utility ---
def error_analysis(image_paths, label_paths, model):
    pass
def continuous_validation(image_paths, label_paths, step_name="validation"):
    pass
    """
    Automated validation after each pipeline step, with error reports.
    """
    errors = {}
    for img, lbl in zip(image_paths, label_paths):
        # Example: check label consistency (implement as needed)
        issues = []
        if not os.path.exists(img) or not os.path.exists(lbl):
            issues.append('Missing file')
        # Add more checks as needed
        if issues:
            errors[img] = issues
    print(f"[VALIDATION] {step_name}: {len(errors)} errors found.")
    # Optionally: log to MLflow
    try:
        import mlflow
        mlflow.log_dict(errors, f"validation_{step_name}.json")
    except ImportError:
        pass
    return errors
    """
    Runs error analysis: finds FP, FN, low-confidence, and label mismatches.
    """
    print("[INFO] Running error analysis...")
    error_report = {}
    for img, lbl in zip(image_paths, label_paths):
        preds = model.predict(img)
        # Compare preds to GT labels (simple IoU matching)
        gt_boxes = [list(map(float, l.strip().split()[1:])) for l in open(lbl)]
        fp, fn, low_conf = 0, 0, 0
        for pred in preds:
            cls, x, y, w, h, conf = pred
            if conf < 0.3:
                low_conf += 1
            # Simple matching: if no GT box with IoU > 0.5, count as FP
            if not any(_yolo_iou((x, y, w, h), gt) > 0.5 for gt in gt_boxes):
                fp += 1
        for gt in gt_boxes:
            if not any(_yolo_iou(gt, (p[1], p[2], p[3], p[4])) > 0.5 for p in preds):
                fn += 1
        error_report[img] = {'fp': fp, 'fn': fn, 'low_conf': low_conf}
    print("[INFO] Error analysis complete.")
    return error_report
    print("\n--- Summary ---")
    print(f"Total images processed: {len(image_ids)}")
    print(f"Total labels written: {sum(stats.values())}")
    for cname in class_names:
        print(f"  {cname}: {stats[cname]}")
    # Check for label/image mismatches
    for split, ids in [('train', train_ids), ('val', val_ids)]:
        img_dir = os.path.join(args.output_dir, f'images/{split}')
        lbl_dir = os.path.join(args.output_dir, f'labels/{split}')
        for fname in os.listdir(img_dir):
            lbl_file = os.path.join(lbl_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt'))
            if not os.path.exists(lbl_file):
                print(f"[WARN] No label file for image: {fname}")
    print("Data preparation complete.")
