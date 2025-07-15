import cv2
import numpy as np
import logging
import torch
import os # Added for os.path operations
import random # Added for random.choice in oversample
from tqdm import tqdm # Added for progress bars
from collections import Counter # Added for class_balanced_oversample
from PIL import Image # Required for OWL-ViT's image input (if EmbedDetector is used here)

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
    """
    Returns an Albumentations Compose object for training data augmentation.
    Args:
        cfg (dict, optional): Configuration dictionary for augmentation settings.
    Returns:
        albumentations.Compose: Composed augmentation pipeline.
    Raises:
        ImportError: If Albumentations is not installed.
    """
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

def get_val_transforms(cfg=None):
    """
    Returns an Albumentations Compose object for validation data.
    Args:
        cfg (dict, optional): Configuration dictionary for resize settings.
    Returns:
        albumentations.Compose: Composed augmentation pipeline for validation.
    Raises:
        ImportError: If Albumentations is not installed.
    """
    if Compose is None:
        raise ImportError("Albumentations is not installed. Cannot create transforms.")
    
    base_size = 640 if not cfg else cfg["resize"]["base_size"]
    return Compose([
        Resize(base_size, base_size),
        ToTensorV2()
    ], bbox_params={'format': 'yolo', 'label_fields': ['class_labels']})


# --- Class balancing utility ---
def class_balanced_oversample(image_paths: list, label_paths: list, min_count: int = 100):
    """
    Oversample rare classes to ensure class balance in dataset.
    Args:
        image_paths (list): List of paths to image files.
        label_paths (list): List of paths to corresponding label files.
        min_count (int): Minimum desired count for each class.
    Returns:
        tuple: (new_image_paths, new_label_paths) with oversampled data.
    """
    class_counts = Counter()
    for lbl in label_paths:
        try:
            with open(lbl, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_counts[int(parts[0])] += 1
        except Exception as e:
            logging.error(f"Error reading label file {lbl} for class balancing: {e}")
            continue

    if not class_counts:
        logging.warning("No class counts found for oversampling. Returning original paths.")
        return list(image_paths), list(label_paths)

    new_imgs, new_lbls = list(image_paths), list(label_paths)
    
    # Identify images that contain each class
    class_to_image_indices = defaultdict(list)
    for i, lbl_path in enumerate(label_paths):
        try:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        class_to_image_indices[class_id].append(i)
        except Exception as e:
            logging.error(f"Error reading label file {lbl_path} for class-to-image mapping: {e}")
            continue

    for cls_id, current_count in class_counts.items():
        if current_count < min_count:
            needed = min_count - current_count
            candidates_indices = class_to_image_indices[cls_id]
            if not candidates_indices:
                logging.warning(f"Class {cls_id} needs oversampling but no images contain it. Skipping.")
                continue

            for _ in range(needed):
                idx_to_copy = random.choice(candidates_indices)
                new_imgs.append(image_paths[idx_to_copy])
                new_lbls.append(label_paths[idx_to_copy])
    
    logging.info(f"Class balancing complete. Original samples: {len(image_paths)}, Oversampled samples: {len(new_imgs)}")
    return new_imgs, new_lbls

def smote_oversample(image_paths: list, label_paths: list, min_count: int = 100):
    """
    SMOTE-based synthetic oversampling for minority classes.
    This is a conceptual placeholder, as SMOTE typically works on feature vectors,
    not directly on images/labels for object detection.
    Args:
        image_paths (list): List of paths to image files.
        label_paths (list): List of paths to corresponding label files.
        min_count (int): Minimum desired count for each class.
    """
    try:
        from imblearn.over_sampling import SMOTE
        import pandas as pd
        
        logging.warning("SMOTE for object detection is complex and often requires feature extraction. This is a simplified placeholder.")
        
        # Prepare features for SMOTE (flatten bboxes and class IDs)
        # This is a very simplistic representation; a real implementation would need
        # more sophisticated feature engineering (e.g., embeddings, image features).
        X_features, y_classes = [], []
        for lbl in label_paths:
            try:
                with open(lbl, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            y_classes.append(int(parts[0]))
                            # Flatten bbox coords as features
                            X_features.append(list(map(float, parts[1:])))
            except Exception as e:
                logging.error(f"Error reading label file {lbl} for SMOTE: {e}")
                continue

        if not X_features:
            logging.warning("No valid label data for SMOTE. Skipping oversampling.")
            return

        # Convert to numpy arrays
        X_features = np.array(X_features)
        y_classes = np.array(y_classes)
        
        # Filter for classes that need oversampling based on min_count
        unique_classes, counts = np.unique(y_classes, return_counts=True)
        minority_classes = unique_classes[counts < min_count]
        
        if len(minority_classes) == 0:
            logging.info("No minority classes found for SMOTE oversampling.")
            return

        # Create a mapping for SMOTE sampling strategy
        sampling_strategy = {cls: min_count for cls in minority_classes}
        
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        
        # Perform SMOTE resampling
        X_res, y_res = sm.fit_resample(X_features, y_classes)
        
        logging.info(f"[INFO] SMOTE oversampling completed. Original samples (bboxes): {len(X_features)}, Resampled samples (bboxes): {len(X_res)}")
        # Note: SMOTE generates synthetic *features*. Creating corresponding synthetic *images*
        # and *label files* from these features is a complex task and not handled here.
        # This function primarily indicates the feature-level oversampling.
        
    except ImportError:
        logging.warning("imblearn not installed. SMOTE oversampling will not be available.")
    except Exception as e:
        logging.error(f"[WARN] SMOTE oversampling failed: {e}")

# --- Visual inspection utility ---
def visualize_augmentations(image_path: str, label_path: str, out_dir: str = "aug_preview", n: int = 5, cfg=None):
    """
    Saves 'n' augmented samples for visual inspection.
    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the corresponding label text file (YOLO format).
        out_dir (str): Output directory to save augmented images.
        n (int): Number of augmented samples to generate and save.
        cfg (dict, optional): Configuration dictionary for augmentation settings.
    """
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image for visualization: {image_path}")
        return

    bboxes = []
    class_labels = []
    try:
        with open(label_path, 'r') as f:
            for l in f:
                parts = l.split()
                if len(parts) == 5: # class_id cx cy w h
                    class_labels.append(int(parts[0]))
                    bboxes.append(list(map(float, parts[1:])))
                else:
                    logging.warning(f"Malformed label line in {label_path} for visualization: {l.strip()}")
    except FileNotFoundError:
        logging.error(f"Label file not found for visualization: {label_path}")
        return
    except Exception as e:
        logging.error(f"Error reading label file {label_path} for visualization: {e}")
        return

    aug = get_train_transforms(cfg)
    if aug is None:
        logging.error("Failed to get augmentation transforms. Cannot visualize.")
        return

    logging.info(f"Generating {n} augmented previews for {image_path}...")
    for i in tqdm(range(n), desc="Generating Augmentation Previews"):
        try:
            data = aug(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = data["image"].permute(1,2,0).cpu().numpy() # Convert from Tensor to numpy (HWC)
            aug_img = (aug_img * 255).astype('uint8') if aug_img.max() <= 1.0 else aug_img # Scale to 0-255 if normalized

            # Draw bounding boxes on the augmented image
            h_img, w_img = aug_img.shape[:2]
            for bb, cl in zip(data["bboxes"], data["class_labels"]):
                # Convert YOLO (cx, cy, w, h) to pixel (x1, y1, x2, y2)
                x, y, w, h = bb
                x1 = int((x - w/2) * w_img)
                y1 = int((y - h/2) * h_img)
                x2 = int((x + w/2) * w_img)
                y2 = int((y + h/2) * h_img)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)

                if x2 > x1 and y2 > y1: # Draw only if valid box
                    cv2.rectangle(aug_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
                    cv2.putText(aug_img, str(cl), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Class label

            cv2.imwrite(os.path.join(out_dir, f"aug_preview_{os.path.basename(image_path).split('.')[0]}_{i}.jpg"), aug_img)
        except Exception as e:
            logging.error(f"Error generating augmented preview {i} for {image_path}: {e}")

def apply_augmentations(image_path: str, labels: list, cfg=None):
    """
    Applies training augmentations to a single image and its labels.
    Args:
        image_path (str): Path to the image file.
        labels (list): List of labels in YOLO format: [class_id, cx, cy, w, h].
        cfg (dict, optional): Configuration dictionary for augmentation settings.
    Returns:
        tuple: (augmented_image (torch.Tensor), augmented_bboxes (list), augmented_class_labels (list))
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Could not read image for augmentation: {image_path}")
        return None, None, None

    bboxes_yolo = [tuple(l[1:]) for l in labels]
    class_labels = [int(l[0]) for l in labels]
    
    aug = get_train_transforms(cfg)
    if aug is None:
        logging.error("Failed to get augmentation transforms. Cannot apply augmentations.")
        return None, None, None

    try:
        result = aug(image=image, bboxes=bboxes_yolo, class_labels=class_labels)
        logging.info(f"Applied augmentations to {image_path}")
        
        # Optionally: log to MLflow if enabled
        try:
            import mlflow
            mlflow.log_param("aug_image", image_path)
        except ImportError:
            pass # MLflow not installed
        except Exception as e:
            logging.error(f"MLflow logging failed for augmentation: {e}")
            
        return result['image'], result['bboxes'], result['class_labels']
    except Exception as e:
        logging.error(f"Error applying augmentations to {image_path}: {e}")
        return None, None, None

# --- Advanced Augmentation Pipeline (make_pipeline, augment_dataset) ---
def make_pipeline(cfg: dict):
    """
    Creates an Albumentations augmentation pipeline based on configuration.
    Args:
        cfg (dict): Configuration dictionary with augmentation settings.
    Returns:
        albumentations.Compose: The composed augmentation pipeline.
    """
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        transforms = []
        
        # HSV jitter (BrightnessContrast, HueSaturationValue are already added in get_train_transforms)
        # If cfg['hsv_jitter'] is intended to control brightness/contrast, it's redundant with RandomBrightnessContrast
        # Assuming 'hsv_jitter' here means additional color jitter
        if cfg.get('hsv_jitter', 0) > 0:
            transforms.append(A.ColorJitter(
                brightness=cfg.get('brightness_limit', 0.2),
                contrast=cfg.get('contrast_limit', 0.2),
                saturation=cfg.get('sat_limit', 0.2),
                hue=cfg.get('hue_limit', 0.1),
                p=0.5
            ))
        
        if cfg.get('flip_prob', 0.5) > 0:
            transforms.append(A.HorizontalFlip(p=cfg.get('flip_prob', 0.5)))
        
        if cfg.get('apply_mosaic', False):
            if hasattr(A, 'Mosaic'): # Newer Albumentations might have A.Mosaic
                transforms.insert(0, A.Mosaic(p=1.0))
            elif hasattr(A, 'MosaicAlpha'): # Older version
                transforms.insert(0, A.MosaicAlpha(p=1.0))
            else:
                logging.warning("Albumentations Mosaic transform not found. Skipping mosaic augmentation.")
        
        if cfg.get('apply_mixup', False):
            if hasattr(A, 'MixUp'):
                transforms.append(A.MixUp(p=1.0))
            else:
                logging.warning("Albumentations MixUp transform not found. Skipping mixup augmentation.")
        
        # Add ToTensorV2 at the end if not already part of the pipeline
        if ToTensorV2 not in [type(t) for t in transforms]:
             transforms.append(ToTensorV2())

        # Filter out None transforms
        transforms = [t for t in transforms if t is not None]

        return A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    except ImportError:
        logging.error("Albumentations not installed. Cannot create advanced augmentation pipeline.")
        return None
    except Exception as e:
        logging.error(f"Error creating Albumentations pipeline: {e}")
        return None

def augment_dataset(img_dir: str, lbl_dir: str, out_img_dir: str, out_lbl_dir: str, pipeline, num_workers: int = 4):
    """
    Applies augmentation pipeline to a dataset in parallel using multiprocessing.
    Args:
        img_dir (str): Input directory containing images.
        lbl_dir (str): Input directory containing YOLO label files.
        out_img_dir (str): Output directory to save augmented images.
        out_lbl_dir (str): Output directory to save augmented label files.
        pipeline: An Albumentations Compose pipeline.
        num_workers (int): Number of parallel processes to use.
    """
    from multiprocessing import Pool
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    args_list = []
    for f_name in img_files:
        img_path = os.path.join(img_dir, f_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(f_name)[0] + '.txt')
        
        if os.path.exists(lbl_path):
            args_list.append((img_path, lbl_path, out_img_dir, out_lbl_dir, pipeline))
        else:
            logging.warning(f"Skipping {img_path}: no corresponding label file {lbl_path}")
            # Optionally, copy original image/label if no augmentation is applied
            # shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
            # if os.path.exists(lbl_path): shutil.copy(lbl_path, os.path.join(out_lbl_dir, os.path.basename(lbl_path)))

    if not args_list:
        logging.info("No image-label pairs found for augmentation.")
        return

    def augment_pair_worker(args_tuple):
        img_path, lbl_path, out_img_dir_worker, out_lbl_dir_worker, aug_pipeline_worker = args_tuple
        try:
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Worker: Could not read image {img_path}. Skipping.")
                return

            boxes, class_labels = [], []
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        boxes.append(list(map(float, parts[1:])))
            
            if not boxes:
                # If no boxes, just copy the image and an empty label file (if it was empty)
                shutil.copy(img_path, os.path.join(out_img_dir_worker, os.path.basename(img_path)))
                with open(os.path.join(out_lbl_dir_worker, os.path.basename(lbl_path)), 'w') as f:
                    pass # Create empty label file
                return

            data = aug_pipeline_worker(image=img, bboxes=boxes, class_labels=class_labels)
            
            aug_img = data['image']
            # If ToTensorV2 was applied, convert back to numpy for saving
            if isinstance(aug_img, torch.Tensor):
                aug_img = aug_img.permute(1,2,0).cpu().numpy() # Convert from Tensor to numpy (HWC)
                aug_img = (aug_img * 255).astype('uint8') if aug_img.max() <= 1.0 else aug_img # Scale to 0-255 if normalized

            new_boxes = data['bboxes']
            new_class_labels = data['class_labels']
            
            base_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(out_img_dir_worker, base_name), aug_img)
            with open(os.path.join(out_lbl_dir_worker, os.path.splitext(base_name)[0]+'.txt'),'w') as f:
                for c, b in zip(new_class_labels, new_boxes):
                    f.write(f"{c} {' '.join(map(str,b))}\n")
        except Exception as e:
            logging.error(f"Worker: Error augmenting {img_path}: {e}")
            import traceback
            traceback.print_exc()

    logging.info(f"Starting parallel augmentation with {num_workers} workers...")
    with Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(augment_pair_worker, args_list),
                    total=len(args_list), desc="Augmenting Dataset"))
    logging.info("Parallel augmentation complete.")

# --- Error Handling Wrapper ---
def safe_run(fn, *args, **kwargs):
    """
    A wrapper to safely run a function, catching and logging exceptions.
    Args:
        fn (callable): The function to run.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
    Returns:
        Any: The result of the function call, or None if an error occurred.
    """
    import traceback
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {fn.__name__}: {e}")
        traceback.print_exc()
        return None

# The `if __name__ == "__main__":` block for `visualize_augmentations`
# is removed from here as it will be handled by the `cli_main.py` Typer app.
