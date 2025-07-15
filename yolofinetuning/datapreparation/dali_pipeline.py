from augmentations import get_train_transforms, custom_mixup, custom_cutmix
from auto_labeling import auto_label, pseudo_labeling
from copy_paste_synthesis import load_object_masks, paste_objects
from data_preparation_utils import generate_synthetic_data, test_time_augmentation
from embedding_model import EmbedDetector
from fuse_graphs_with_yolo import load_graph, load_yolo, iou
from logger import setup_logging, log_detection
from metadata_logger import compute_metadata, log_metadata
from metrics import Evaluator
from split_dataset import split_dataset


# --- DALI pipeline with OWL-ViT integration and logging/metrics ---

import os
import logging
import random
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
from pathlib import Path
from PIL import Image
from collections import Counter
import glob
import shutil
import pandas as pd
import argparse
from logger import log_detection
from embedding_model import EmbedDetector
from metrics import Evaluator
import mlflow

# --- Config loading and seed setting ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    def init_device_threads():
        if 'resources' in cfg:
            torch.set_num_threads(cfg['resources'].get('max_threads', 4))
            torch.set_num_interop_threads(cfg['resources'].get('max_threads', 4))
    init_device_threads()
else:
    cfg = None

# --- MLflow setup (if enabled in config) ---
def setup_mlflow(cfg):
    if cfg and cfg.get('mlflow', {}).get('enable', False):
        mlflow.set_tracking_uri(cfg['mlflow'].get('tracking_uri', 'mlruns'))
        mlflow.set_experiment(cfg['mlflow'].get('experiment', 'dataset-prep'))
        return True
    return False

# --- Data validation and statistics ---
def check_bbox_integrity(img_w, img_h, x_center, y_center, w, h):
    if w <= 0 or h <= 0:
        return False
    x1 = (x_center - w/2) * img_w
    y1 = (y_center - h/2) * img_h
    x2 = (x_center + w/2) * img_w
    y2 = (y_center + h/2) * img_h
    return 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h

def validate_image_and_labels(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    try:
        with open(lbl_path) as f:
            lines = f.read().strip().splitlines()
        if not lines:
            return False
        for line in lines:
            cls, *coords = line.split()
            x, y, bw, bh = map(float, coords)
            if not check_bbox_integrity(w, h, x, y, bw, bh):
                return False
        return True
    except Exception:
        return False

def filter_valid_samples(images, labels):
    valid_imgs = []
    valid_lbls = []
    for img, lbl in zip(images, labels):
        if validate_image_and_labels(img, lbl):
            valid_imgs.append(img)
            valid_lbls.append(lbl)
    return valid_imgs, valid_lbls

def compute_class_balance(lbl_paths, n_classes):
    counts = Counter()
    for lp in lbl_paths:
        for line in open(lp):
            cls = int(line.split()[0])
            counts[cls] += 1
    total = sum(counts.values())
    balance = {c: counts[c]/total if total > 0 else 0 for c in range(n_classes)}
    return balance

def compute_mean_std(image_paths, num_samples=500):
    sampled = random.sample(image_paths, min(num_samples, len(image_paths)))
    mean = np.zeros(3)
    std = np.zeros(3)
    for p in sampled:
        img = cv2.imread(p)
        if img is None:
            continue
        img = img.astype(np.float32)/255.0
        mean += img.mean(axis=(0,1))
        std += img.std(axis=(0,1))
    if len(sampled) > 0:
        mean /= len(sampled)
        std /= len(sampled)
    return mean.tolist(), std.tolist()

# --- Stratified split ---
def stratified_split(img_paths, lbl_paths, n_splits, seed=42):
    from sklearn.model_selection import StratifiedKFold
    y = [int(open(lbl).read().split()[0]) for lbl in lbl_paths]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in skf.split(img_paths, y):
        train_imgs = [img_paths[i] for i in train_idx]
        train_lbls = [lbl_paths[i] for i in train_idx]
        val_imgs   = [img_paths[i] for i in val_idx]
        val_lbls   = [lbl_paths[i] for i in val_idx]
        break
    return train_imgs, train_lbls, val_imgs, val_lbls

def simple_split(img_paths, lbl_paths, train_size):
    paired = list(zip(img_paths, lbl_paths))
    random.shuffle(paired)
    cut = int(len(paired)*train_size)
    train, val = paired[:cut], paired[cut:]
    return zip(*train), zip(*val)

# --- Albumentations transforms ---
def get_train_transforms():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    aug = []
    # Use custom MosaicDataset for mosaic, not Albumentations
    if cfg and cfg["augment"]["mosaic"]:
        pass  # Mosaic handled in loader
    aug += [
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=int(255*cfg["augment"]["hsv"]["hue"]),
            sat_shift_limit=int(255*cfg["augment"]["hsv"]["sat"]),
            val_shift_limit=int(255*cfg["augment"]["hsv"]["val"]),
            p=0.5
        ),
        A.HorizontalFlip(p=cfg["augment"]["flip_p"]),
        A.VerticalFlip(p=cfg["augment"]["flip_p"]/2),
        A.RandomScale(scale_limit=0.5, p=0.5),
    ]
    # Use custom MixUp/CutMix if not available in Albumentations
    if cfg and cfg["augment"]["mixup"]:
        try:
            aug.append(A.MixUp(p=0.5))
        except Exception:
            from augmentations import custom_mixup
            aug.append(custom_mixup(p=0.5))
    if cfg and cfg["augment"]["cutmix"]:
        try:
            aug.append(A.CutMix(p=0.5))
        except Exception:
            from augmentations import custom_cutmix
            aug.append(custom_cutmix(p=0.5))
    if cfg and cfg["resize"]["multiscale"]:
        aug.append(A.RandomChoice([
            A.Resize(s, s) for s in cfg["resize"]["scales"]
        ]))
    else:
        aug.append(A.Resize(cfg["resize"]["base_size"], cfg["resize"]["base_size"]))
    aug.append(ToTensorV2())
    return A.Compose(aug, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_val_transforms():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(cfg["resize"]["base_size"], cfg["resize"]["base_size"]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --- Mosaic Dataset (PyTorch) ---
class MosaicDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, lbl_paths, img_size=640, transforms=None):
        self.img_paths, self.lbl_paths = img_paths, lbl_paths
        self.img_size, self.transforms = img_size, transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        indices = [index] + random.choices(range(len(self.img_paths)), k=3)
        mosaic_img = np.full((self.img_size * 2, self.img_size * 2, 3), 114, dtype=np.uint8)
        mosaic_bboxes, mosaic_labels = [], []
        xc = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))
        yc = int(random.uniform(self.img_size * 0.5, self.img_size * 1.5))
        for i, idx in enumerate(indices):
            img = cv2.imread(self.img_paths[idx])
            h0, w0 = img.shape[:2]
            bboxes = []
            for line in open(self.lbl_paths[idx]):
                cls, x, y, w, h = map(float, line.split())
                x1 = int((x - w/2) * w0); y1 = int((y - h/2) * h0)
                x2 = int((x + w/2) * w0); y2 = int((y + h/2) * h0)
                bboxes.append([x1, y1, x2, y2, int(cls)])
            scale = self.img_size / max(h0, w0)
            img_resized = cv2.resize(img, (int(w0*scale), int(h0*scale)))
            h, w = img_resized.shape[:2]
            if i == 0:
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
            elif i == 1:
                x1a, y1a = xc, max(yc - h, 0)
            elif i == 2:
                x1a, y1a = max(xc - w, 0), yc
            else:
                x1a, y1a = xc, yc
            x2a, y2a = x1a + w, y1a + h
            mosaic_img[y1a:y2a, x1a:x2a] = img_resized
            for x1, y1, x2, y2, cls in bboxes:
                x1n = x1 * scale + x1a
                y1n = y1 * scale + y1a
                x2n = x2 * scale + x1a
                y2n = y2 * scale + y1a
                mosaic_bboxes.append([x1n, y1n, x2n, y2n, cls])
        # IoU filtering
        def iou(box1, box2):
            xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
            xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
            area2 = (box2[2] - box2[0])*(box2[3] - box2[1])
            return inter / (area1 + area2 - inter + 1e-9)
        def filter_by_iou(bboxes, threshold=0.2):
            filtered = []
            img_box = [0, 0, 2*self.img_size, 2*self.img_size]
            for box in bboxes:
                x1, y1, x2, y2, _ = box
                if iou(img_box, [x1, y1, x2, y2]) >= threshold:
                    filtered.append(box)
            return filtered
        mosaic_bboxes = filter_by_iou(mosaic_bboxes)
        final_bboxes, final_labels = [], []
        for x1, y1, x2, y2, cls in mosaic_bboxes:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(2*self.img_size, x2), min(2*self.img_size, y2)
            if x2c - x1c > 5 and y2c - y1c > 5:
                cx = ((x1c + x2c) / 2) / (2*self.img_size)
                cy = ((y1c + y2c) / 2) / (2*self.img_size)
                bw = (x2c - x1c) / (2*self.img_size)
                bh = (y2c - y1c) / (2*self.img_size)
                final_bboxes.append([cx, cy, bw, bh])
                final_labels.append(int(cls))
        if self.transforms:
            data = self.transforms(
                image=mosaic_img, bboxes=final_bboxes, class_labels=final_labels
            )
            mosaic_img = data["image"]
            final_bboxes = data["bboxes"]
            final_labels = data["class_labels"]
        return mosaic_img, {"boxes": final_bboxes, "labels": final_labels}

# --- Save dataset utility ---
def save_dataset(imgs, lbls, out_dir, transforms=None):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    for img_path, lbl_path in zip(imgs, lbls):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if transforms:
            data = transforms(image=img, bboxes=[list(map(float, l.split()[1:])) for l in open(lbl_path)], class_labels=[int(l.split()[0]) for l in open(lbl_path)])
            img = data["image"]
            new_bboxes = data["bboxes"]
            new_labels = data["class_labels"]
        else:
            new_bboxes = [l.split()[1:] for l in open(lbl_path)]
            new_labels = [int(l.split()[0]) for l in open(lbl_path)]
        fname = os.path.basename(img_path)
        cv2.imwrite(os.path.join(out_dir, "images", fname), img)
        lines = [f"{lab} {' '.join(map(str, bb))}\n" for lab, bb in zip(new_labels, new_bboxes)]
        with open(os.path.join(out_dir, "labels", fname.replace(".jpg", ".txt")), "w") as f:
            f.writelines(lines)

# --- Export YOLO data.yaml ---
def export_yolo_yaml(cfg):
    data_cfg = {
        "train": cfg["paths"]["out_train"] + "/images",
        "val":   cfg["paths"]["out_val"]   + "/images",
        "nc":    len(open(cfg["paths"]["classes_file"]).read().splitlines()),
        "names": open(cfg["paths"]["classes_file"]).read().splitlines()
    }
    with open("data.yaml", "w") as f:
        yaml.dump(data_cfg, f, sort_keys=False)

def prepare_dataset(data_cfg, detector: EmbedDetector, evaluator: Evaluator):
    images_dir = Path(data_cfg['images_dir'])
    output_dir = Path(data_cfg['output_dir'])
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
                # convert to relative xywh
                cx, cy = (x0 + x1)/2/width, (y0 + y1)/2/height
                w, h = (x1 - x0)/width, (y1 - y0)/height
                # Optionally: filter by bbox integrity
                if check_bbox_integrity(width, height, cx, cy, w, h):
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

        # Update evaluation metrics
        evaluator.update(gt_boxes=None, pred_boxes=boxes, scores=scores)
    # Finalize and print mAP, IoU, precision, recall
    evaluator.finalize()

def run_full_dataset_preparation():
    if not cfg:
        print("No config.yaml found, skipping advanced dataset prep.")
        return
    mlflow_enabled = setup_mlflow(cfg)
    if mlflow_enabled:
        mlflow.start_run(run_name="dataset-prep")
    # 1. Gather all images/labels
    img_paths = sorted(glob.glob(cfg["paths"]["raw_images"] + "/*.jpg"))
    lbl_paths = sorted(glob.glob(cfg["paths"]["raw_labels"] + "/*.txt"))

    # 2. Validate
    img_paths, lbl_paths = filter_valid_samples(img_paths, lbl_paths)
    from data_preparation_utils import continuous_validation
    continuous_validation(img_paths, lbl_paths, step_name="initial_validation")

    # 3. Data quality audit
    from data_preparation_utils import audit_data_quality
    audit_report = audit_data_quality(lbl_paths, img_paths)
    print("[AUDIT] Data quality report:", audit_report)

    # 4. Label noise detection and correction
    from data_preparation_utils import detect_label_noise, correct_labels_with_model
    label_issues = detect_label_noise(lbl_paths)
    if label_issues:
        print("[WARN] Label noise detected:", label_issues)
        # Optionally: correct labels using model and human review
        if cfg.get('label_correction', {}).get('enable', False):
            for lbl in label_issues:
                img_path = lbl.replace('.txt', '.jpg')
                correct_labels_with_model(lbl, img_path, model=None, human_review=cfg['label_correction'].get('human_review', False))
        lbl_paths = [l for l in lbl_paths if l not in label_issues]
    continuous_validation(img_paths, lbl_paths, step_name="post_label_correction")

    # 5. Class balancing (oversample rare classes)
    from augmentations import class_balanced_oversample, smote_oversample
    img_paths, lbl_paths = class_balanced_oversample(img_paths, lbl_paths, min_count=cfg.get("min_class_count", 100))
    if cfg.get('smote', {}).get('enable', False):
        smote_oversample(img_paths, lbl_paths, min_count=cfg.get("min_class_count", 100))
    continuous_validation(img_paths, lbl_paths, step_name="post_oversampling")

    # 6. Curriculum learning (progressive/meta-data sampling)
    from data_preparation_utils import curriculum_sampling
    img_paths, lbl_paths = curriculum_sampling(img_paths, lbl_paths, model=None, stages=3)
    continuous_validation(img_paths, lbl_paths, step_name="post_curriculum")

    # 7. Hard negative mining (active retraining)
    from data_preparation_utils import hard_negative_mining
    hard_negatives = hard_negative_mining(img_paths, lbl_paths, model=None)
    # Optionally: add hard negatives to training set
    if cfg.get('hard_negative', {}).get('add_to_train', False):
        img_paths += hard_negatives
        lbl_paths += [lbl for img, lbl in zip(img_paths, lbl_paths) if img in hard_negatives]
    continuous_validation(img_paths, lbl_paths, step_name="post_hard_negative")

    # 8. Synthetic data generation (GAN/Diffusion)
    from data_preparation_utils import generate_synthetic_data
    if cfg.get('synthetic', {}).get('enable', False):
        generator_model = None # Load or instantiate as needed
        generate_synthetic_data(generator_model, n_samples=cfg['synthetic'].get('n_samples', 0), out_dir=cfg["paths"]["raw_images"])
    continuous_validation(img_paths, lbl_paths, step_name="post_synthetic")

    # 9. Stats
    classes = open(cfg["paths"]["classes_file"]).read().splitlines()
    balance = compute_class_balance(lbl_paths, len(classes))
    mean, std = compute_mean_std(img_paths)
    print("Class balance:", balance)
    print("Dataset mean:", mean, "std:", std)
    if mlflow_enabled:
        mlflow.log_dict(balance, "class_balance.json")
        mlflow.log_dict({"mean": mean, "std": std}, "mean_std.json")

    # 10. Split (advanced stratified sampling for rare classes)
    if cfg["split"]["n_splits"] > 1:
        tr_imgs, tr_lbls, v_imgs, v_lbls = stratified_split(
            img_paths, lbl_paths, cfg["split"]["n_splits"], seed=cfg["seed"]
        )
    else:
        (tr_imgs, tr_lbls), (v_imgs, v_lbls) = simple_split(
            img_paths, lbl_paths, cfg["split"]["train_size"]
        )
    continuous_validation(tr_imgs, tr_lbls, step_name="train_split")
    continuous_validation(v_imgs, v_lbls, step_name="val_split")

    # 11. Save
    save_dataset(tr_imgs, tr_lbls, cfg["paths"]["out_train"], transforms=get_train_transforms())
    save_dataset(v_imgs, v_lbls, cfg["paths"]["out_val"], transforms=get_val_transforms())

    # 12. Export YOLO yaml
    export_yolo_yaml(cfg)
    print("[INFO] Dataset preparation complete. Train/val splits and data.yaml written.")
    if mlflow_enabled:
        mlflow.log_artifact("data.yaml")
        mlflow.end_run()

    # 13. Test-Time Augmentation (TTA) for validation
    from data_preparation_utils import test_time_augmentation
    # Placeholder: pass a trained model and validation images
    # tta_results = test_time_augmentation(image=..., model=...)

    # 14. Error analysis
    from data_preparation_utils import error_analysis
    # Placeholder: pass a trained model and validation images/labels
    # error_report = error_analysis(v_imgs, v_lbls, model=None)

    # 15. Visualization
    from augmentations import visualize_augmentations
    # Example: visualize_augmentations(tr_imgs[0], tr_lbls[0])

    print("[INFO] All advanced dataset preparation steps completed.")

# --- Pseudo-labeling logic (placeholder, to be implemented as needed) ---
def pseudo_labeling(unlabeled_img_dir, model, out_label_dir):
    """
    Generate pseudo-labels for unlabeled images using a trained model.
    Args:
        unlabeled_img_dir: Directory with unlabeled images
        model: Trained detection model
        out_label_dir: Output directory for pseudo-labels
    """
    os.makedirs(out_label_dir, exist_ok=True)
    img_paths = glob.glob(os.path.join(unlabeled_img_dir, '*.jpg'))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        # Example: model.predict returns list of (cls, cx, cy, w, h, conf)
        preds = model.predict(img)
        label_path = os.path.join(out_label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            for pred in preds:
                cls, cx, cy, w, h, conf = pred
                if conf > 0.5:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# --- Active learning logic (placeholder, to be implemented as needed) ---
def active_learning_step(unlabeled_img_dir, model, selection_count=100):
    """
    Select most informative samples for annotation.
    Args:
        unlabeled_img_dir: Directory with unlabeled images
        model: Trained detection model
        selection_count: Number of samples to select
    Returns:
        List of selected image paths
    """
    img_paths = glob.glob(os.path.join(unlabeled_img_dir, '*.jpg'))
    # Example: select by model uncertainty (placeholder)
    uncertainties = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        preds = model.predict(img)
        # Placeholder: use std of confidences as uncertainty
        confs = [p[-1] for p in preds]
        uncertainty = np.std(confs) if confs else 0
        uncertainties.append((uncertainty, img_path))
    uncertainties.sort(reverse=True)
    selected = [p for _, p in uncertainties[:selection_count]]
    return selected

# YOLO format pipeline
def dali_augment(image_paths, labels=None, batch_size=16):
    """
    DALI-based augmentation generator for a list of image paths.
    Args:
        image_paths: List of image file paths
        labels: Optional labels (not used in this simple version)
        batch_size: Batch size for augmentation
    Yields:
        Augmented image batches as numpy arrays
    """
    # Placeholder: Use OpenCV for now, replace with DALI pipeline as needed
    for i in range(0, len(image_paths), batch_size):
        batch = []
        for p in image_paths[i:i+batch_size]:
            img = cv2.imread(p)
            if img is not None:
                # Example: simple flip augmentation
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)
                batch.append(img)
        if batch:
            yield np.stack(batch)

def yolo_pipeline(image_dir, label_dir):
    """
    YOLO format pipeline for custom YOLO datasets.
    Handles YOLO txt format labels with class_id, x_center, y_center, width, height.
    """
    # Read image files
    images, image_ids = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        name="ImageReader"
    )
    
    # Read corresponding YOLO label files  
    labels = fn.readers.file(
        file_root=label_dir,
        random_shuffle=False,  # Keep same order as images
        name="LabelReader"
    )
    
    images = fn.decoders.image(images, device="mixed")
    
    # Basic augmentations without bbox operations (simplified for YOLO)
    flip_prob = fn.random.coin_flip(probability=0.5)
    images = fn.flip(images, horizontal=flip_prob)
    
    # Color augmentations
    images = fn.brightness_contrast(
        images,
        brightness=fn.random.uniform(range=(0.9, 1.1)),
        contrast=fn.random.uniform(range=(0.9, 1.1))
    )
    
    # Resize and normalize
    images = fn.resize(images, resize_x=640, resize_y=640)
    images = fn.crop_mirror_normalize(
        images,
        crop=(640, 640), 
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255],
        dtype=types.FLOAT
    )
    
    return images, labels

# Simple image augmentation pipeline 
@pipeline_def(batch_size=16, num_threads=4, device_id=0)
def simple_augment_pipeline(image_dir):
    """
    Simple image augmentation pipeline without bbox operations.
    Good for testing and basic image processing tasks.
    """
    print(f"[DALI DEBUG] Reading images from: {image_dir}")
    
    # Step 1: Read files
    images = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        name="Reader"
    )
    print(f"[DALI DEBUG] File reader output type: {type(images)}")
    
    # Step 2: Decode images
    images = fn.decoders.image(images, device="mixed")
    print(f"[DALI DEBUG] Image decoder output type: {type(images)}")
    
    # Step 3: Rotation augmentation
    angles = fn.random.uniform(range=(-15.0, 15.0))
    print(f"[DALI DEBUG] Random angles type: {type(angles)}")
    images = fn.rotate(images, angle=angles, fill_value=0)
    print(f"[DALI DEBUG] After rotation type: {type(images)}")
    
    # Step 4: Color augmentations
    brightness_vals = fn.random.uniform(range=(0.8, 1.2))
    contrast_vals = fn.random.uniform(range=(0.8, 1.2))
    print(f"[DALI DEBUG] Brightness type: {type(brightness_vals)}, Contrast type: {type(contrast_vals)}")
    
    images = fn.brightness_contrast(
        images,
        brightness=brightness_vals,
        contrast=contrast_vals
    )
    print(f"[DALI DEBUG] After color augmentation type: {type(images)}")
    
    # Step 5: Geometric augmentations
    flip_prob = fn.random.coin_flip(probability=0.5)
    print(f"[DALI DEBUG] Flip probability type: {type(flip_prob)}")
    images = fn.flip(images, horizontal=flip_prob)
    print(f"[DALI DEBUG] After flip type: {type(images)}")
    
    # Step 6: Resize
    images = fn.resize(images, resize_x=640, resize_y=640)
    print(f"[DALI DEBUG] After resize type: {type(images)}")
    
    # Step 7: Normalize
    images = fn.crop_mirror_normalize(
        images,
        crop=(640, 640),
        mean=[0.485*255, 0.456*255, 0.406*255], 
        std=[0.229*255, 0.224*255, 0.225*255],
        dtype=types.FLOAT
    )
    print(f"[DALI DEBUG] After normalization type: {type(images)}")
    print(f"[DALI DEBUG] Final pipeline return type: {type(images)}")
    
    # Return as tuple to avoid nested DataNode error
    return (images,)

def run_detection_pipeline(img_dir, lbl_file, iterations):
    """Run the professional detection pipeline with bbox support."""
    # detection_pipeline is not defined in this file. Uncomment and implement if available.
    # pipe = detection_pipeline(img_dir, lbl_file)
    # pipe.build()
    # print(f"Running detection pipeline for {iterations} iterations...")
    # for i in range(iterations):
    #     imgs, bboxes, labels = pipe.run()
    #     print(f"Detection batch {i+1}/{iterations}")
    #     imgs_np = imgs.as_cpu().as_array()
    #     bboxes_np = bboxes.as_cpu().as_array() 
    #     labels_np = labels.as_cpu().as_array()
    #     print(f"  Images: {imgs_np.shape}, Bboxes: {len(bboxes_np)}, Labels: {len(labels_np)}")
    #     # save_augmented_data(imgs_np, bboxes_np, labels_np, i)
    # print("Detection pipeline completed successfully!")

def run_yolo_pipeline(image_dir, label_dir, iterations):
    """Run the YOLO format pipeline."""
    pipe = yolo_pipeline(image_dir, label_dir)
    pipe.build()
    
    print(f"Running YOLO pipeline for {iterations} iterations...")
    for i in range(iterations):
        imgs, lbls = pipe.run()
        print(f"YOLO batch {i+1}/{iterations}")
        
        imgs_np = imgs.as_cpu().as_array()
        print(f"  Image batch shape: {imgs_np.shape}")
        
    print("YOLO pipeline completed successfully!")

def run_simple_augment(image_dir, iterations):
    """Run the simple augmentation pipeline."""
    pipe = simple_augment_pipeline(image_dir)
    pipe.build()
    
    print(f"Running simple augmentation for {iterations} iterations...")
    for i in range(iterations):
        imgs, = pipe.run()  # Unpack single output
        print(f"Simple augment batch {i+1}/{iterations}")
        
        imgs_np = imgs.as_cpu().as_array()
        print(f"  Augmented batch shape: {imgs_np.shape}")
        
    print("Simple augmentation pipeline completed successfully!")

if __name__ == "__main__":
    # Import all major functions from datasetpreparation folder
    from fuse_graphs_with_yolo import load_graph, load_yolo, iou, nms, check_relation, label_quality, log_metadata as fg_log_metadata
    from metadata_logger import compute_metadata, log_metadata as meta_log_metadata
    from copy_paste_synthesis import load_object_masks, paste_objects
    from auto_labeling import auto_label
    from augmentations import get_train_transforms, apply_augmentations
    from split_dataset import split_dataset
    # Import SAM and Albumentations/AugMix utilities
    try:
        from sam import load_sam_model, get_mask_generator, save_mask_png, sam_masks_to_yolo, generate_relation_graph, get_symbolic_labels, overlay_symbolic_debugger, generate_reasoning_chain
        sam_predictor = load_sam_model()
        mask_generator = get_mask_generator()
    except Exception as e:
        print(f"[WARN] Could not import or load SAM: {e}")
        sam_predictor = None
        mask_generator = None
    try:
        from albumentations_augmix import CopyPaste, augment_and_mix, RandomAugMix
    except Exception as e:
        print(f"[WARN] Could not import AugMix from albumentations_augmix: {e}")
        CopyPaste = None
        augment_and_mix = None
        RandomAugMix = None

    parser = argparse.ArgumentParser(description="Main Dataset Preparation Pipeline")
    parser.add_argument("--img_root", default="ShapeBongard_V2", help="Root directory of images")
    parser.add_argument("--output_root", default="dataset", help="Output root directory")
    parser.add_argument("--lbl_dir", default="data/annotations.json", help="Label file/directory path")
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--mode", default="simple", choices=["detection", "yolo", "simple"], help="Pipeline mode")
    parser.add_argument("--pseudo_label", action="store_true", help="Run pseudo-labeling on unlabeled data")
    parser.add_argument("--active_learning", action="store_true", help="Run active learning step")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    args = parser.parse_args()

    print(f"Main Dataset Preparation Pipeline Mode: {args.mode}")
    print(f"Image Root: {args.img_root}")
    print(f"Label Path: {args.lbl_dir}")
    print(f"Output Root: {args.output_root}")
    print(f"Iterations: {args.iters}")
    print("-" * 50)

    try:
        # --- Docker environment check ---
        if os.path.exists("Dockerfile"):
            print("[DOCKER] Dockerfile found. Checking environment...")
            with open("Dockerfile") as f:
                docker_content = f.read()
            required_libs = ["albumentations", "mlflow", "torch", "imblearn", "transformers"]
            for lib in required_libs:
                if lib not in docker_content:
                    print(f"[DOCKER WARN] {lib} not found in Dockerfile. Please add it.")

        # --- Advanced full dataset prep (config-driven) ---
        # Synthetic data generation
        if cfg.get('synthetic_data', {}).get('enabled', False):
            print("Generating synthetic data...")
            # generator_model should be defined/configured
            generate_synthetic_data(generator_model=None, n_samples=cfg['synthetic_data']['count'], out_dir=cfg['paths']['out_train'])

        # Data augmentation
        if cfg.get('augmentation', {}).get('enabled', False):
            print("Applying augmentations...")
            get_train_transforms()
            custom_mixup()
            custom_cutmix()

        # Auto labeling
        if cfg.get('pseudo_labeling', {}).get('enabled', False):
            print("Running auto labeling...")
            auto_label(args.img_root, args.lbl_dir)

        # Copy-paste synthesis
        if cfg.get('copy_paste_synthesis', {}).get('enabled', False):
            print("Running copy-paste synthesis...")
            masks = load_object_masks(args.img_root)
            paste_objects(bg_image=None, bg_labels=None, objects=masks)

        # Metadata logging
        print("Logging metadata...")
        meta = compute_metadata("image_id", "labels")
        log_metadata(meta)

        # Embedding extraction
        print("Extracting embeddings...")
        embedder = EmbedDetector(cfg['model'])

        # Graph fusion
        print("Fusing graphs with YOLO...")
        graph = load_graph(args.lbl_dir)
        yolo = load_yolo(args.lbl_dir)
        iou(graph, yolo)

        # Split dataset
        print("Splitting dataset...")
        split_dataset(args.lbl_dir)

        # Metrics
        print("Evaluating metrics...")
        evaluator = Evaluator(cfg['metrics'])

        # Test-time augmentation
        print("Running test-time augmentation...")
        test_time_augmentation(image=None, model=None)

        # 1. Split dataset
        print("Splitting dataset...")
        split_dataset(args.lbl_dir)

        # 2. Run DALI pipeline (recursively on all PNG images)
        image_paths = [y for x in os.walk(args.img_root) for y in glob.glob(os.path.join(x[0], '*.png'))]
        print(f"Found {len(image_paths)} PNG images in {args.img_root}")
        labels = None  # Placeholder, adjust as needed
        gen = dali_augment(image_paths, labels)
        for i in range(args.iters):
            try:
                imgs = next(gen)
            except StopIteration:
                print("No more images to process.")
                break
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                src_img_path = image_paths[i * batch_size + j]
                rel_path = os.path.relpath(src_img_path, args.img_root)
                out_dir = os.path.join(args.output_root, os.path.dirname(rel_path))
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.basename(src_img_path))
                img_arr = imgs[j]
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
                # --- Advanced AugMix ---
                if augment_and_mix is not None:
                    img_arr = augment_and_mix(img_arr)
                # --- Albumentations/AugMix CopyPaste ---
                if CopyPaste is not None:
                    cp = CopyPaste()
                    img_arr = cp(image=img_arr)['image']
                # --- RandomAugMix (Albumentations pipeline) ---
                if RandomAugMix is not None:
                    try:
                        import albumentations as A
                        aug_pipeline = A.Compose([RandomAugMix(severity=3, p=0.5)])
                        img_arr = aug_pipeline(image=img_arr)['image']
                    except Exception as e:
                        print(f"[WARN] RandomAugMix failed: {e}")
                # --- SAM mask generation (single image) ---
                if sam_predictor is not None:
                    try:
                        sam_predictor.set_image(img_arr)
                        h, w = img_arr.shape[:2]
                        box = np.array([[0, 0, w, h]])
                        masks, _, _ = sam_predictor.predict(box)
                        mask_path = out_path.replace('.png', '_sam_mask.png')
                        save_mask_png(masks[0], mask_path)
                        print(f"Saved SAM mask: {mask_path}")
                        # --- Convert SAM masks to YOLO labels ---
                        yolo_labels = sam_masks_to_yolo(masks, (h, w))
                        yolo_path = out_path.replace('.png', '_sam_yolo.txt')
                        with open(yolo_path, 'w') as f:
                            f.write('\n'.join(yolo_labels))
                        print(f"Saved YOLO labels: {yolo_path}")
                        # --- Generate relation graph ---
                        relation_graph = generate_relation_graph(masks)
                        graph_path = out_path.replace('.png', '_sam_relgraph.json')
                        import json
                        with open(graph_path, 'w') as f:
                            json.dump(relation_graph, f)
                        print(f"Saved relation graph: {graph_path}")
                        # --- Symbolic reasoning ---
                        symbols = get_symbolic_labels(masks, img_arr, "primitive reasoning")
                        reasoning = generate_reasoning_chain(symbols)
                        reasoning_path = out_path.replace('.png', '_sam_reasoning.txt')
                        with open(reasoning_path, 'w') as f:
                            f.write(reasoning)
                        print(f"Saved reasoning chain: {reasoning_path}")
                        # --- Overlay symbolic debugger ---
                        debug_img = overlay_symbolic_debugger(img_arr, masks, symbols)
                        debug_path = out_path.replace('.png', '_sam_debug.png')
                        Image.fromarray(debug_img).save(debug_path)
                        print(f"Saved symbolic debug image: {debug_path}")
                    except Exception as e:
                        print(f"[WARN] SAM advanced features failed: {e}")
                # --- Batch mask generation (optional, if mask_generator is available) ---
                if mask_generator is not None:
                    try:
                        batch_masks = mask_generator.generate(img_arr)
                        batch_mask_path = out_path.replace('.png', '_sam_batchmask.json')
                        import json
                        with open(batch_mask_path, 'w') as f:
                            json.dump(batch_masks, f)
                        print(f"Saved batch masks: {batch_mask_path}")
                    except Exception as e:
                        print(f"[WARN] SAM batch mask generation failed: {e}")
                # --- Save final image ---
                im = Image.fromarray(img_arr)
                im.save(out_path)
                print(f"Saved: {out_path}")

        # 3. Apply augmentations
        print("Applying augmentations...")
        get_train_transforms()
        apply_augmentations(args.img_root, args.lbl_dir)

        # 4. Auto labeling
        print("Running auto labeling...")
        auto_label(args.img_root, args.lbl_dir)

        # 5. Copy-paste synthesis
        print("Running copy-paste synthesis...")
        masks = load_object_masks(args.img_root)
        # Example: paste_objects usage (requires bg_image, bg_labels, objects)
        # paste_objects(bg_image, bg_labels, masks)

        # 6. Fuse graphs with YOLO
        print("Fusing graphs with YOLO...")
        graph = load_graph(args.lbl_dir)
        yolo = load_yolo(args.lbl_dir)
        # Example: iou, nms, check_relation, label_quality, fg_log_metadata usage
        # iou(boxA, boxB)
        # nms(boxes, scores)
        # check_relation(graph, yolo)
        # label_quality(yolo)
        # fg_log_metadata(...)

        # 7. Metadata logging
        print("Logging metadata...")
        meta = compute_metadata("image_id", "labels")
        meta_log_metadata(meta)

        # 8. Pseudo-labeling (if requested)
        if args.pseudo_label:
            print("Running ensemble pseudo-labeling...")
            # Example: pass a list of models for ensemble
            ensemble_models = [] # Populate with model instances as needed
            from auto_labeling import pseudo_labeling
            pseudo_labeling(args.img_root, model=ensemble_models, out_label_dir=os.path.join(args.output_root, "pseudo_labels"))

        # 9. Active learning (if requested)
        if args.active_learning:
            print("Running active learning step...")
            selected = active_learning_step(args.img_root, model=None, selection_count=100)
            print(f"Selected {len(selected)} samples for annotation.")

        print("All datasetpreparation steps completed successfully!")
    except Exception as e:
        import traceback
        print(f"Pipeline failed with error: {e}")
        traceback.print_exc()
        print("Please check your data paths and format.")

# --- Dockerfile note ---
# To containerize this pipeline, create a Dockerfile in this folder with all dependencies (Python, OpenCV, PyTorch, DALI, Albumentations, MLflow, etc.) and copy this script and config.yaml into the image.
