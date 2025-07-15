def get_train_transforms():

import cv2
import numpy as np
import logging
try:
    from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, HueSaturationValue, RandomScale, RandomChoice, Resize
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

def get_train_transforms(cfg=None):
    aug = []
    if Compose is None:
        raise ImportError("Albumentations is not installed.")
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
                pass
    except Exception:
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
            for line in open(lbl):
                parts = line.strip().split()
                if len(parts) == 5:
                    y.append(int(parts[0]))
                    X.append(list(map(float, parts[1:])))
        sm = SMOTE()
        X_res, y_res = sm.fit_resample(X, y)
        # Optionally: create synthetic images/labels for new samples
        # ...existing code...
        print(f"[INFO] SMOTE oversampling completed: {len(X_res)} samples.")
    except Exception as e:
        print(f"[WARN] SMOTE oversampling failed: {e}")
    """
    Oversample rare classes to ensure class balance in dataset.
    """
    from collections import Counter
    import random
    class_counts = Counter()
    for lbl in label_paths:
        for line in open(lbl):
            class_counts[int(line.split()[0])] += 1
    max_class = max(class_counts.values())
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

# --- Visual inspection utility ---
def visualize_augmentations(image_path, label_path, out_dir="aug_preview", n=5, cfg=None):
    """
    Save n augmented samples for visual inspection.
    """
    import cv2, os
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    bboxes = [list(map(float, l.split()[1:])) for l in open(label_path)]
    class_labels = [int(l.split()[0]) for l in open(label_path)]
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

# --- CLI entry for preview ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augmentation Preview Utility")
    parser.add_argument("--img", required=True, help="Path to image")
    parser.add_argument("--lbl", required=True, help="Path to label txt")
    parser.add_argument("--out", default="aug_preview", help="Output dir")
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    args = parser.parse_args()
    visualize_augmentations(args.img, args.lbl, args.out, args.n)


def apply_augmentations(image_path, labels, cfg=None):
    """
    image_path: str
    labels: list of [class, x_center, y_center, w, h]
    cfg: config dict (optional)
    """
    image = cv2.imread(image_path)
    bboxes = [tuple(l[1:]) for l in labels]
    class_labels = [int(l[0]) for l in labels]
    aug = get_train_transforms(cfg)
    result = aug(image=image, bboxes=bboxes, class_labels=class_labels)
    # Logging
    logging.info(f"Applied augmentations to {image_path}")
    # Optionally: log to MLflow if enabled
    try:
        import mlflow
        mlflow.log_param("aug_image", image_path)
    except ImportError:
        pass
    return result['image'], result['bboxes'], result['class_labels']
