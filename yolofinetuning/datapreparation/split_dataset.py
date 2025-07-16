import cv2
import numpy as np
import logging
import torch
import os
import json
import random
import shutil
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.transforms import functional as TF
from collections import Counter # Added for class_balanced_oversample
from .metadata_logger import compute_metadata, log_metadata # Fixed to relative import for module execution

try:
    import mlflow
except ImportError:
    mlflow = None

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split # Added for split_dataset
except ImportError:
    StratifiedKFold = None
    train_test_split = None
    logging.warning("scikit-learn not installed. Stratified splitting and K-Fold will not be available.")

try:
    from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, \
        HueSaturationValue, RandomScale, RandomChoice, Resize
    from albumentations.pytorch import ToTensorV2
    try:
        from albumentations import MixUp, CutMix, Mosaic
    except ImportError:
        MixUp = None
        CutMix = None
        Mosaic = None
except ImportError:
    Compose = None
    RandomBrightnessContrast = None
    HorizontalFlip = None
    HueSaturationValue = None
    RandomScale = None
    RandomChoice = None
    Resize = None
    ToTensorV2 = None
    MixUp = None
    CutMix = None
    Mosaic = None
    logging.warning("Albumentations not installed. Augmentations will be limited.")


# --- Custom/fallback implementations ---
def custom_mixup(p=0.5):
    def _mixup(image, bboxes, class_labels):
        # Simple mixup: blend with random noise
        if np.random.rand() < p:
            alpha = 0.5
            noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image = cv2.addWeighted(image, alpha, noise, 1-alpha, 0)
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
    return _mixup

def custom_cutmix(p=0.5):
    def _cutmix(image, bboxes, class_labels):
        # Simple cutmix: cut and paste a random patch
        if np.random.rand() < p:
            h, w = image.shape[:2]
            x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
            x2, y2 = x1 + w//4, y1 + h//4
            patch = image[y1:y2, x1:x2].copy()
            image[0:y2-y1, 0:x2-x1] = patch
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
    return _cutmix

def custom_mosaic(p=0.5):
    def _mosaic(image, bboxes, class_labels):
        # Simple mosaic: tile image with itself
        if np.random.rand() < p:
            h, w = image.shape[:2]
            new_img = np.zeros_like(image)
            new_img[:h//2, :w//2] = image[:h//2, :w//2]
            new_img[h//2:, w//2:] = image[h//2:, w//2:]
            image = new_img
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
    return _mosaic

# --- CopyPaste and AugMix (if available) ---
try:
    from albumentations_augmix import CopyPaste, augment_and_mix, RandomAugMix
except ImportError:
    CopyPaste = None
    augment_and_mix = None
    RandomAugMix = None
    logging.warning("albumentations_augmix not installed. CopyPaste and AugMix will not be available.")


def get_train_transforms(cfg=None):
    aug = []
    if Compose is None:
        raise ImportError("Albumentations is not installed. Cannot create transforms.")
    # Standard augmentations
    aug.append(RandomBrightnessContrast(p=0.5))
    aug.append(HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5))
    aug.append(HorizontalFlip(p=0.5))
    aug.append(RandomScale(scale_limit=0.5, p=0.5))
    # Advanced photometric, spatial, adversarial augmentations
    try:
        from albumentations import RandomCrop, ShiftScaleRotate, Perspective, RGBShift, Blur, GaussNoise, CLAHE, RandomGamma, CoarseDropout, ElasticTransform, GridDistortion, RandomErasing, ColorJitter
        aug += [
            RandomCrop(height=512, width=512, p=0.3),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
            Perspective(scale=(0.05,0.1), p=0.2),
            RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
            Blur(blur_limit=3, p=0.2),
            GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            CLAHE(p=0.1),
            RandomGamma(p=0.1),
            CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            ElasticTransform(p=0.2),
            GridDistortion(p=0.2),
            ColorJitter(p=0.2),
            RandomErasing(p=0.2)
        ]
        # Adversarial augmentation placeholder (requires external lib)
        if cfg and cfg.get('augment', {}).get('adversarial', False):
            try:
                from adversarial_aug import AdversarialAugmentation
                aug.append(AdversarialAugmentation(p=0.2))
            except Exception:
                logging.warning("AdversarialAugmentation not found. Skipping.")
                pass
    except Exception:
        logging.warning("Some advanced Albumentations transforms not found. Skipping them.")
        pass
    # Sample-mix augmentations
    if Mosaic is not None:
        aug.append(Mosaic(p=0.5))
    else:
        aug.append(custom_mosaic(p=0.5))
    if MixUp is not None:
        aug.append(MixUp(p=0.5))
    else:
        aug.append(custom_mixup(p=0.5))
    if CutMix is not None:
        aug.append(CutMix(p=0.5))
    else:
        aug.append(custom_cutmix(p=0.5))
    # CopyPaste and AugMix
    if CopyPaste is not None:
        aug.append(CopyPaste(p=0.5))
    if RandomAugMix is not None:
        aug.append(RandomAugMix(severity=3, p=0.5))
    # Resize
    if cfg and cfg.get("resize", {}).get("multiscale", False):
        scales = cfg["resize"]["scales"]
        aug.append(RandomChoice([Resize(s, s) for s in scales]))
    else:
        base_size = 640 if not cfg else cfg["resize"]["base_size"]
        aug.append(Resize(base_size, base_size))
    aug.append(ToTensorV2())
    return Compose(aug, bbox_params={'format': 'yolo', 'label_fields': ['class_labels']})

# --- Class balancing utility ---
def class_balanced_oversample(image_paths, label_paths, min_count=100):
    """
    Oversample rare classes to ensure class balance in dataset.
    """
    class_counts = Counter()
    for lbl in label_paths:
        try:
            with open(lbl, 'r') as f:
                for line in f:
                    class_counts[int(line.split()[0])] += 1
        except Exception as e:
            logging.error(f"Error reading label file {lbl}: {e}")
            continue

    new_imgs, new_lbls = list(image_paths), list(label_paths)
    for cls, cnt in class_counts.items():
        if cnt < min_count:
            needed = min_count - cnt
            candidates = [i for i,l in enumerate(label_paths) if any(int(line.split()[0])==cls for line in open(l))]
            for _ in range(needed):
                idx = random.choice(candidates)
                new_imgs.append(image_paths[idx])
                new_lbls.append(label_paths[idx])
    return new_imgs, new_lbls

def smote_oversample(image_paths, label_paths, min_count=100):
    """
    SMOTE-based synthetic oversampling for minority classes.
    """
    try:
        from imblearn.over_sampling import SMOTE
        import pandas as pd
        # Prepare features for SMOTE (flatten bboxes)
        X, y = [], []
        for lbl in label_paths:
            try:
                with open(lbl, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            y.append(int(parts[0]))
                            X.append(list(map(float, parts[1:])))
            except Exception as e:
                logging.error(f"Error reading label file {lbl} for SMOTE: {e}")
                continue

        if not X:
            logging.warning("No valid label data for SMOTE. Skipping oversampling.")
            return

        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        # Optionally: create synthetic images/labels for new samples
        # This part would require a generative model and is beyond simple SMOTE.
        # For now, we just indicate completion.
        print(f"[INFO] SMOTE oversampling completed: {len(X_res)} samples.")
    except ImportError:
        logging.warning("imblearn not installed. SMOTE oversampling will not be available.")
    except Exception as e:
        print(f"[WARN] SMOTE oversampling failed: {e}")

# --- Visual inspection utility ---
def visualize_augmentations(image_path, label_path, out_dir="aug_preview", n=5, cfg=None):
    """
    Save n augmented samples for visual inspection.
    """
    import cv2, os
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image: {image_path}")
        return

    bboxes = []
    class_labels = []
    try:
        with open(label_path, 'r') as f:
            for l in f:
                parts = l.split()
                if len(parts) > 1: # Ensure there are bbox coordinates
                    bboxes.append(list(map(float, parts[1:])))
                    class_labels.append(int(parts[0]))
    except Exception as e:
        logging.error(f"Error reading label file {label_path}: {e}")
        return

    aug = get_train_transforms(cfg)
    for i in range(n):
        data = aug(image=img, bboxes=bboxes, class_labels=class_labels)
        aug_img = data["image"].permute(1,2,0).cpu().numpy()
        aug_img = (aug_img * 255).astype('uint8') if aug_img.max()<=1.0 else aug_img
        for bb, cl in zip(data["bboxes"], data["class_labels"]):
            x, y, w, h = bb
            h_img, w_img = aug_img.shape[:2]
            x1 = int((x-w/2)*w_img); y1 = int((y-h/2)*h_img)
            x2 = int((x+w/2)*w_img); y2 = int((y+h/2)*h_img)
            cv2.rectangle(aug_img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(aug_img, str(cl), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imwrite(os.path.join(out_dir, f"aug_{i}.jpg"), aug_img)


def apply_augmentations(image_path, labels, cfg=None):
    """
    image_path: str
    labels: list of [class, x_center, y_center, w, h]
    cfg: config dict (optional)
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image for augmentation: {image_path}")
        return None, None, None

    bboxes = [tuple(l[1:]) for l in labels]
    class_labels = [int(l[0]) for l in labels]
    aug = get_train_transforms(cfg)
    result = aug(image=image, bboxes=bboxes, class_labels=class_labels)
    # Logging
    logging.info(f"Applied augmentations to {image_path}")
    # Optionally: log to MLflow if enabled
    if mlflow:
        try:
            mlflow.log_param("aug_image", image_path)
        except Exception as e:
            logging.error(f"MLflow logging failed: {e}")

    return result['image'], result['bboxes'], result['class_labels']

class EmbedDetector:
    """
    OWL-ViT detection and embedding extraction utility.
    Usage:
        det = EmbedDetector(model_cfg)
        boxes, scores, labels, embeddings = det.detect(image, prompts)
    """
    def __init__(self, model_cfg):
        try:
            self.device = torch.device(model_cfg['device'])
            self.processor = OwlViTProcessor.from_pretrained(model_cfg['name'])
            self.model = OwlViTForObjectDetection.from_pretrained(model_cfg['name'])
            self.model.to(self.device).eval()
            self.threshold = model_cfg['detection_threshold']
            self.max_queries = model_cfg['max_queries']
        except Exception as e:
            print(f"[ERROR] Failed to load OWL-ViT model: {e}")
            self.processor = None
            self.model = None
            self.device = None

    def detect(self, image, prompts):
        try:
            if self.model is None:
                raise RuntimeError("OWL-ViT model not loaded. Cannot perform detection.")

            """
            Detect objects and extract embeddings.
            Args:
                image: PIL.Image
                prompts: list of str
            Returns:
                boxes, scores, labels, embeddings
            """
            inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # Post-process detections to get xyxy boxes and scores
            target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.threshold
            )[0]
            boxes = results['boxes'].cpu().numpy()         # [[x0,y0,x1,y1], ...]
            scores = results['scores'].cpu().numpy()       # [0.8, 0.5, ...]
            labels = results['labels'].cpu().numpy()       # [class_id, ...]
            embeddings = outputs.last_hidden_state.cpu().numpy() if hasattr(outputs, 'last_hidden_state') else None
            # Extract CLIP embeddings for each crop
            clip_embeddings = []
            for box in boxes:
                x0, y0, x1, y1 = [int(v) for v in box]
                crop = TF.crop(image, y0, x0, y1-y0, x1-x0)
                pix = self.processor.images_processor(crop, return_tensors="pt").pixel_values.to(self.device)
                clip_outputs = self.model.clip.vision_model(pix)
                clip_embeddings.append(clip_outputs.pooler_output.cpu().detach().numpy()[0])

            return boxes, scores, labels, clip_embeddings
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return [], [], [], None


def split_dataset(label_dir, splits=(0.8, 0.1, 0.1), out_dir='splits', pseudo_label_func=None, active_learning_func=None, stratified=True, kfolds=None, seed=42):
    """
    Splits dataset into train/val/test or k-folds. Supports stratified splitting by class distribution.
    Args:
        label_dir: directory with label txt files
        splits: tuple, e.g. (0.8,0.1,0.1)
        out_dir: output directory
        pseudo_label_func: optional hook (function that takes label_dir, model, out_label_dir)
        active_learning_func: optional hook (function that takes label_dir, model, selection_count)
        stratified: if True, split by class distribution
        kfolds: if int, perform k-fold cross-validation
        seed: random seed
    """
    if not os.path.exists(label_dir):
        logging.error(f"Label directory not found: {label_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    random.seed(seed)
    random.shuffle(files)

    # Get main class for each file
    def get_main_class(lblfile):
        counts = {}
        try:
            with open(os.path.join(label_dir, lblfile), 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        c = int(parts[0])
                        counts[c] = counts.get(c,0)+1
        except Exception as e:
            logging.warning(f"Could not read main class from {lblfile}: {e}")
        return max(counts, key=counts.get) if counts else -1

    y = [get_main_class(f) for f in files]

    # Error handling wrapper for hooks
    def safe_run_hook(fn, *args, **kwargs):
        import traceback
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in hook {fn.__name__}: {e}")
            traceback.print_exc()

    # KFold logic
    if kfolds:
        if StratifiedKFold is None:
            logging.error("scikit-learn not installed, cannot perform K-Fold cross-validation.")
            return

        skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(files, y)):
            for phase, idxs in zip(['train','val'], [train_idx, val_idx]):
                fold_dir = os.path.join(out_dir, f'fold_{fold}')
                os.makedirs(fold_dir, exist_ok=True)
                with open(os.path.join(fold_dir, f'{phase}.txt'),'w') as fout:
                    for i in idxs:
                        fname = files[i]
                        img_id = fname.split('.txt')[0]
                        labels = []
                        try:
                            with open(os.path.join(label_dir, fname), 'r') as f_read:
                                for line in f_read:
                                    parts = line.strip().split()
                                    labels.append(list(map(float, parts)))
                        except Exception as e:
                            logging.warning(f"Could not read labels for {fname}: {e}")
                            continue

                        meta = compute_metadata(img_id, labels)
                        log_metadata(meta)
                        fout.write(img_id + '\n') # Write image ID to split file
                        # Copy the actual label file to the split directory
                        shutil.copy(os.path.join(label_dir, fname), os.path.join(fold_dir, fname))

                        logging.info(f"Fold {fold} {phase}: {img_id}")
                        if mlflow:
                            try:
                                mlflow.log_param(f"fold{fold}_{phase}_img", img_id)
                            except Exception as e:
                                logging.error(f"MLflow logging failed for fold {fold} {phase}: {e}")
        return

    # Stratified or Simple split logic
    if stratified:
        if train_test_split is None:
            logging.error("scikit-learn not installed, cannot perform stratified splitting.")
            # Fallback to simple split if stratified is requested but not available
            logging.info("Falling back to simple (non-stratified) split.")
            stratified = False # Disable stratified for the next block

    if stratified:
        # First split: separate out test set
        train_val_files, test_files, train_val_y, test_y = train_test_split(
            files, y, test_size=splits[2], stratify=y, random_state=seed
        )
        # Second split: separate train and validation from the remaining
        val_ratio_from_train_val = splits[1] / (splits[0] + splits[1])
        train_files, val_files, _, _ = train_test_split(
            train_val_files, train_val_y, test_size=val_ratio_from_train_val,
            stratify=train_val_y, random_state=seed
        )
        split_dict = {'train': train_files, 'val': val_files, 'test': test_files}
    else: # Simple (non-stratified) split logic
        n = len(files)
        train_end_idx = int(n * splits[0])
        val_end_idx = int(n * (splits[0] + splits[1]))
        split_dict = {
            'train': files[:train_end_idx],
            'val': files[train_end_idx:val_end_idx],
            'test': files[val_end_idx:]
        }

    # Write files for all splits (common logic for both stratified and simple)
    for phase, subset in split_dict.items():
        split_dir = os.path.join(out_dir, phase) # Use phase directly for directory name
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'{phase}.txt'),'w') as fout_list: # List of image IDs
            for fname in subset:
                img_id = fname.split('.txt')[0]
                labels = []
                # Read labels from original label_dir
                try:
                    with open(os.path.join(label_dir, fname), 'r') as f_read:
                        for line in f_read:
                            parts = line.strip().split()
                            labels.append(list(map(float, parts)))
                except Exception as e:
                    logging.warning(f"Could not read labels for {fname}: {e}")
                    continue

                meta = compute_metadata(img_id, labels)
                log_metadata(meta)
                fout_list.write(img_id + '\n') # Write image ID to split list file
                # Copy the actual label file to the new split directory
                shutil.copy(os.path.join(label_dir, fname), os.path.join(split_dir, fname))

                logging.info(f"{phase}: {img_id}")
                if mlflow:
                    try:
                        mlflow.log_param(f"{phase}_img", img_id)
                    except Exception as e:
                        logging.error(f"MLflow logging failed for {phase}: {e}")

    # Pseudo-labeling hook
    if pseudo_label_func is not None:
        logging.info("Running pseudo-labeling hook...")
        safe_run_hook(pseudo_label_func, label_dir, model=None, out_label_dir=os.path.join(out_dir, "pseudo_labels"))
    # Active learning hook
    if active_learning_func is not None:
        logging.info("Running active learning hook...")
        selected = safe_run_hook(active_learning_func, label_dir, model=None, selection_count=100)
        if selected:
            logging.info(f"Active learning selected: {selected}")
            if mlflow:
                try:
                    mlflow.log_param("active_learning_selected", selected)
                except Exception as e:
                    logging.error(f"MLflow logging failed for active learning: {e}")


# --- CLI entry point ---
if __name__ == "__main__":
    import argparse
    from PIL import Image

    # Main parser for selecting utility
    parser = argparse.ArgumentParser(description="Dataset Preparation Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for EmbedDetector (OWL-ViT)
    owlvit_parser = subparsers.add_parser('owlvit', help='Run OWL-ViT detection utility')
    owlvit_parser.add_argument('--img', required=True, help='Path to image')
    owlvit_parser.add_argument('--prompt', required=True, nargs='+', help='Detection prompts')
    owlvit_parser.add_argument('--device', default='cpu', help='Device to run OWL-ViT on (e.g., "cuda" or "cpu")')
    owlvit_parser.add_argument('--model', default='google/owlvit-base-patch32', help='OWL-ViT model name')
    owlvit_parser.set_defaults(func='run_owlvit')

    # Subparser for Visualize Augmentations
    aug_viz_parser = subparsers.add_parser('visualize_aug', help='Visualize augmentations')
    aug_viz_parser.add_argument("--img", required=True, help="Path to image")
    aug_viz_parser.add_argument("--lbl", required=True, help="Path to label txt")
    aug_viz_parser.add_argument("--out", default="aug_preview", help="Output dir")
    aug_viz_parser.add_argument("--n", type=int, default=5, help="Number of samples to visualize")
    aug_viz_parser.set_defaults(func='run_visualize_aug')

    # Subparser for Split Dataset
    split_parser = subparsers.add_parser('split_data', help='Split dataset into train/val/test or k-folds')
    split_parser.add_argument('--label_dir', required=True, help='Directory with label txt files')
    split_parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio')
    split_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    split_parser.add_argument('--test_ratio', type=float, default=0.1, help='Test split ratio')
    split_parser.add_argument('--out_dir', default='splits', help='Output directory for splits')
    split_parser.add_argument('--no_stratified', action='store_false', dest='stratified', help='Disable stratified splitting')
    split_parser.add_argument('--kfolds', type=int, help='Perform K-Fold cross-validation with specified folds')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    split_parser.set_defaults(func='run_split_data')


    args = parser.parse_args()

    if args.command == 'run_owlvit':
        cfg = {'device': args.device, 'name': args.model, 'detection_threshold': 0.3, 'max_queries': 10}
        det = EmbedDetector(cfg)
        image = Image.open(args.img).convert('RGB')
        boxes, scores, labels, embeddings = det.detect(image, args.prompt)
        print('Boxes:', boxes)
        print('Scores:', scores)
        print('Labels:', labels)
        if embeddings is not None:
            print('Embeddings shape:', embeddings[0].shape if embeddings else 'N/A')

    elif args.command == 'run_visualize_aug':
        visualize_augmentations(args.img, args.lbl, args.out, args.n)
        print(f"Augmented images saved to {args.out}")

    elif args.command == 'run_split_data':
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if not (0.99 <= total_ratio <= 1.01): # Allow for floating point inaccuracies
            logging.warning(f"Split ratios do not sum to 1.0 (sum={total_ratio}). Adjusting test ratio.")
            args.test_ratio = 1.0 - (args.train_ratio + args.val_ratio)
            if args.test_ratio < 0: args.test_ratio = 0 # Ensure non-negative

        # Placeholder for pseudo_label_func and active_learning_func
        # These would typically be imported from other modules if they exist
        pseudo_label_hook = None
        active_learning_hook = None

        split_dataset(
            label_dir=args.label_dir,
            splits=(args.train_ratio, args.val_ratio, args.test_ratio),
            out_dir=args.out_dir,
            pseudo_label_func=pseudo_label_hook,
            active_learning_func=active_learning_hook,
            stratified=args.stratified,
            kfolds=args.kfolds,
            seed=args.seed
        )
        print(f"Dataset splitting complete. Splits saved to {args.out_dir}")
    else:
        parser.print_help()
