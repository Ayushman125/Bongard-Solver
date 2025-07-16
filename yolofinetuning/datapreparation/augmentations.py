import cv2
import numpy as np
import logging
import torch
import os
import random
from tqdm import tqdm
from collections import Counter, defaultdict # Added defaultdict for SMOTE helper
import shutil # Added for augment_dataset to copy if no boxes
from PIL import Image # For visualize_augmentations and apply_augmentations

try:
    from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, \
        HueSaturationValue, RandomScale, RandomChoice, Resize
    from albumentations.pytorch import ToTensorV2
    # Specific Albumentations transforms that might require newer versions or separate installs
    try:
        from albumentations import MixUp, CutMix, Mosaic # These might be A.MixUp, A.CutMix, A.Mosaic in newer versions
    except ImportError:
        MixUp = None
        CutMix = None
        Mosaic = None
        logging.warning("Albumentations MixUp, CutMix, or Mosaic not found. Using custom fallbacks.")

    # Check for albucore version compatibility if specific issues arise
    # This is a runtime check, not an import error.
    # import albucore
    # if albucore.__version__ != '0.0.24':
    #     logging.warning(f"Albucore version mismatch: Expected 0.0.24, got {albucore.__version__}. This might cause issues with Albumentations 2.0.8.")

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
    logging.error("Albumentations is not installed. Augmentations functionality will be severely limited.")


# --- Custom/fallback implementations for MixUp, CutMix, Mosaic ---
# These are simple conceptual implementations, not full replacements for Albumentations' versions.
def custom_mixup(p=0.5):
    """Simple conceptual mixup augmentation."""
    def _mixup(image, bboxes, class_labels):
        if np.random.rand() < p:
            alpha = 0.5 # Blend factor
            # Simple blend with random noise or another image (if available)
            # For a real mixup, you'd mix with another image from the batch/dataset.
            noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
            image = cv2.addWeighted(image, alpha, noise, 1 - alpha, 0)
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
    return _mixup

def custom_cutmix(p=0.5):
    """Simple conceptual cutmix augmentation."""
    def _cutmix(image, bboxes, class_labels):
        if np.random.rand() < p:
            h, w = image.shape[:2]
            # Define a random patch to cut
            patch_w, patch_h = w // 4, h // 4
            x1 = np.random.randint(0, w - patch_w)
            y1 = np.random.randint(0, h - patch_h)
            x2, y2 = x1 + patch_w, y1 + patch_h

            # Create a random patch (e.g., from noise or another image)
            # For a real cutmix, you'd paste a patch from another image.
            patch = np.random.randint(0, 256, (patch_h, patch_w, image.shape[2]), dtype=np.uint8)
            
            # Paste the patch onto the image
            image[y1:y2, x1:x2] = patch
            # Bounding box labels would also need to be adjusted for cutmix, which is complex.
            # This simple version doesn't handle bbox changes.
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
    return _cutmix

def custom_mosaic(p=0.5):
    """Simple conceptual mosaic augmentation."""
    def _mosaic(image, bboxes, class_labels):
        if np.random.rand() < p:
            h, w = image.shape[:2]
            # Create a new image by tiling the current image (or parts of it)
            # For a real mosaic, you'd combine 4 images.
            new_img = np.zeros_like(image)
            new_img[:h//2, :w//2] = image[:h//2, :w//2]
            new_img[:h//2, w//2:] = image[:h//2, w//2:]
            new_img[h//2:, :w//2] = image[h//2:, :w//2]
            new_img[h//2:, w//2:] = image[h//2:, w//2:]
            image = new_img
            # Bounding box labels would also need to be adjusted for mosaic, which is complex.
            # This simple version doesn't handle bbox changes.
        return {"image": image, "bboxes": bboxes, "class_labels": class_labels}
    return _mosaic

# --- CopyPaste and AugMix (if available) ---
try:
    # Assuming these are from 'albumentations_augmix' or similar external libs
    from albumentations_augmix import CopyPaste, RandomAugMix
    # augment_and_mix is a function, not a transform, so it's not in the Compose list
except ImportError:
    CopyPaste = None
    RandomAugMix = None
    logging.warning("albumentations_augmix not installed. CopyPaste and RandomAugMix will not be available.")


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
    # Configurable parameters for these basic transforms
    aug.append(RandomBrightnessContrast(
        brightness_limit=cfg.get('augment', {}).get('brightness_limit', 0.2),
        contrast_limit=cfg.get('augment', {}).get('contrast_limit', 0.2),
        p=cfg.get('augment', {}).get('brightness_contrast_prob', 0.5)
    ))
    aug.append(HueSaturationValue(
        hue_shift_limit=cfg.get('augment', {}).get('hue_shift_limit', 10),
        sat_shift_limit=cfg.get('augment', {}).get('sat_shift_limit', 20),
        val_shift_limit=cfg.get('augment', {}).get('val_shift_limit', 10),
        p=cfg.get('augment', {}).get('hsv_prob', 0.5)
    ))
    aug.append(HorizontalFlip(p=cfg.get('augment', {}).get('horizontal_flip_prob', 0.5)))
    aug.append(RandomScale(
        scale_limit=cfg.get('augment', {}).get('scale_limit', 0.5),
        p=cfg.get('augment', {}).get('random_scale_prob', 0.5)
    ))
    
    # Advanced photometric, spatial, adversarial augmentations (configurable via config)
    try:
        from albumentations import RandomCrop, ShiftScaleRotate, Perspective, RGBShift, Blur, GaussNoise, CLAHE, RandomGamma, CoarseDropout, ElasticTransform, GridDistortion, RandomErasing, ColorJitter
        
        if cfg.get('augment', {}).get('random_crop_enabled', True):
            aug.append(RandomCrop(
                height=cfg.get('augment', {}).get('random_crop_height', 512),
                width=cfg.get('augment', {}).get('random_crop_width', 512),
                p=cfg.get('augment', {}).get('random_crop_prob', 0.3)
            ))
        if cfg.get('augment', {}).get('shift_scale_rotate_enabled', True):
            aug.append(ShiftScaleRotate(
                shift_limit=cfg.get('augment', {}).get('shift_limit', 0.1),
                scale_limit=cfg.get('augment', {}).get('scale_limit_ssr', 0.2), # Differentiate from RandomScale
                rotate_limit=cfg.get('augment', {}).get('rotate_limit', 20),
                p=cfg.get('augment', {}).get('shift_scale_rotate_prob', 0.5)
            ))
        if cfg.get('augment', {}).get('perspective_enabled', True):
            aug.append(Perspective(
                scale=cfg.get('augment', {}).get('perspective_scale', (0.05,0.1)),
                p=cfg.get('augment', {}).get('perspective_prob', 0.2)
            ))
        if cfg.get('augment', {}).get('rgb_shift_enabled', True):
            aug.append(RGBShift(
                r_shift_limit=cfg.get('augment', {}).get('r_shift_limit', 15),
                g_shift_limit=cfg.get('augment', {}).get('g_shift_limit', 15),
                b_shift_limit=cfg.get('augment', {}).get('b_shift_limit', 15),
                p=cfg.get('augment', {}).get('rgb_shift_prob', 0.3)
            ))
        if cfg.get('augment', {}).get('blur_enabled', True):
            aug.append(Blur(
                blur_limit=cfg.get('augment', {}).get('blur_limit', 3),
                p=cfg.get('augment', {}).get('blur_prob', 0.2)
            ))
        if cfg.get('augment', {}).get('gauss_noise_enabled', True):
            aug.append(GaussNoise(
                var_limit=cfg.get('augment', {}).get('gauss_noise_var_limit', (10.0, 50.0)),
                p=cfg.get('augment', {}).get('gauss_noise_prob', 0.2)
            ))
        if cfg.get('augment', {}).get('clahe_enabled', True):
            aug.append(CLAHE(p=cfg.get('augment', {}).get('clahe_prob', 0.1)))
        if cfg.get('augment', {}).get('random_gamma_enabled', True):
            aug.append(RandomGamma(p=cfg.get('augment', {}).get('random_gamma_prob', 0.1)))
        if cfg.get('augment', {}).get('coarse_dropout_enabled', True):
            aug.append(CoarseDropout(
                max_holes=cfg.get('augment', {}).get('coarse_dropout_max_holes', 8),
                max_height=cfg.get('augment', {}).get('coarse_dropout_max_height', 32),
                max_width=cfg.get('augment', {}).get('coarse_dropout_max_width', 32),
                p=cfg.get('augment', {}).get('coarse_dropout_prob', 0.3)
            ))
        if cfg.get('augment', {}).get('elastic_transform_enabled', True):
            aug.append(ElasticTransform(p=cfg.get('augment', {}).get('elastic_transform_prob', 0.2)))
        if cfg.get('augment', {}).get('grid_distortion_enabled', True):
            aug.append(GridDistortion(p=cfg.get('augment', {}).get('grid_distortion_prob', 0.2)))
        if cfg.get('augment', {}).get('color_jitter_enabled', True):
            aug.append(ColorJitter(p=cfg.get('augment', {}).get('color_jitter_prob', 0.2)))
        if cfg.get('augment', {}).get('random_erasing_enabled', True):
            aug.append(RandomErasing(p=cfg.get('augment', {}).get('random_erasing_prob', 0.2)))

        # Adversarial augmentation placeholder (requires external lib)
        if cfg.get('augment', {}).get('adversarial_enabled', False):
            try:
                from adversarial_aug import AdversarialAugmentation
                aug.append(AdversarialAugmentation(p=cfg.get('augment', {}).get('adversarial_prob', 0.2)))
            except ImportError:
                logging.warning("AdversarialAugmentation not found. Skipping.")
            except Exception as e:
                logging.error(f"Error with AdversarialAugmentation: {e}. Skipping.")

    except ImportError:
        logging.warning("Some advanced Albumentations transforms not found. Skipping them.")
    except Exception as e:
        logging.error(f"Error during advanced Albumentations transforms setup: {e}. Skipping.")
        pass # Continue with basic transforms

    # Sample-mix augmentations (configurable probability)
    mosaic_prob = cfg.get('augment', {}).get('mosaic_prob', 0.5)
    mixup_prob = cfg.get('augment', {}).get('mixup_prob', 0.5)
    cutmix_prob = cfg.get('augment', {}).get('cutmix_prob', 0.5)

    if cfg.get('augment', {}).get('mosaic_enabled', True):
        if Mosaic is not None:
            aug.append(Mosaic(p=mosaic_prob))
        elif mosaic_prob > 0: # Only warn/add custom if enabled and prob > 0
            aug.append(custom_mosaic(p=mosaic_prob))
            logging.warning("Albumentations Mosaic not found, using custom_mosaic.")

    if cfg.get('augment', {}).get('mixup_enabled', True):
        if MixUp is not None:
            aug.append(MixUp(p=mixup_prob))
        elif mixup_prob > 0:
            aug.append(custom_mixup(p=mixup_prob))
            logging.warning("Albumentations MixUp not found, using custom_mixup.")

    if cfg.get('augment', {}).get('cutmix_enabled', True):
        if CutMix is not None:
            aug.append(CutMix(p=cutmix_prob))
        elif cutmix_prob > 0:
            aug.append(custom_cutmix(p=cutmix_prob))
            logging.warning("Albumentations CutMix not found, using custom_cutmix.")
    
    # CopyPaste and AugMix (configurable probability)
    copypaste_prob = cfg.get('augment', {}).get('copypaste_prob', 0.5)
    random_augmix_prob = cfg.get('augment', {}).get('random_augmix_prob', 0.5)

    if cfg.get('augment', {}).get('copypaste_enabled', True):
        if CopyPaste is not None:
            aug.append(CopyPaste(p=copypaste_prob))
        elif copypaste_prob > 0:
            logging.warning("Albumentations CopyPaste not found. Skipping CopyPaste augmentation.")
    
    if cfg.get('augment', {}).get('random_augmix_enabled', True):
        if RandomAugMix is not None:
            aug.append(RandomAugMix(
                severity=cfg.get('augment', {}).get('random_augmix_severity', 3),
                p=random_augmix_prob
            ))
        elif random_augmix_prob > 0:
            logging.warning("Albumentations RandomAugMix not found. Skipping RandomAugMix augmentation.")

    # Resize (configurable via config)
    if cfg and cfg.get("resize", {}).get("multiscale", False):
        scales = cfg["resize"].get("scales", [640])
        aug.append(RandomChoice([Resize(s, s) for s in scales]))
    else:
        base_size = cfg.get("resize", {}).get("base_size", 640) if cfg else 640
        aug.append(Resize(base_size, base_size))
    
    aug.append(ToTensorV2())
    
    # Filter out None entries from the augmentation list
    aug = [t for t in aug if t is not None]

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
    
    base_size = cfg.get("resize", {}).get("base_size", 640) if cfg else 640
    return Compose([
        Resize(base_size, base_size),
        ToTensorV2()
    ], bbox_params={'format': 'yolo', 'label_fields': ['class_labels']})


# --- Class balancing utility ---
def class_balanced_oversample(image_paths: list, label_paths: list, min_count: int = 100):
    """
    Oversample rare classes by duplicating samples to ensure class balance in dataset.
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

def smote_oversample(image_paths: list, label_paths: list, min_count: int = 100, feature_extractor=None):
    """
    SMOTE-based synthetic oversampling for minority classes.
    This implementation is a conceptual placeholder for object detection,
    as direct SMOTE on raw bounding box coordinates is usually not effective.
    It demonstrates the idea of oversampling based on extracted features.
    
    Args:
        image_paths (list): List of paths to image files.
        label_paths (list): List of paths to corresponding label files.
        min_count (int): Minimum desired count for each class.
        feature_extractor (callable, optional): A function that takes (image_path, labels)
                                                and returns a list of feature vectors for each object.
                                                If None, uses simple bbox coordinates as features.
    Returns:
        tuple: (original_image_paths, original_label_paths) - SMOTE doesn't generate new images/labels directly,
               it operates on features. This function primarily logs the SMOTE process.
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        logging.info("Attempting SMOTE oversampling (conceptual for object detection)...")
        logging.warning("Direct SMOTE on bounding box coordinates is often not ideal for object detection. "
                        "Consider using advanced feature extraction or generative models for true synthetic data.")
        
        X_features, y_classes, original_indices = [], [], []
        
        # Collect features and classes for each object across all images
        object_idx = 0
        for i, lbl_path in enumerate(label_paths):
            try:
                with open(lbl_path, 'r') as f:
                    current_image_labels = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox_coords = list(map(float, parts[1:]))
                            current_image_labels.append((class_id, bbox_coords))
                            
                            y_classes.append(class_id)
                            # If no feature_extractor, use bbox coords directly
                            X_features.append(bbox_coords) 
                            original_indices.append(i) # Keep track of original image index
                        else:
                            logging.warning(f"Skipping malformed label line in {lbl_path}: '{line.strip()}'")
            except FileNotFoundError:
                logging.warning(f"Label file not found for SMOTE: {lbl_path}. Skipping.")
            except Exception as e:
                logging.error(f"Error reading label file {lbl_path} for SMOTE: {e}")
                continue

        if not X_features:
            logging.warning("No valid object data found for SMOTE. Skipping oversampling.")
            return image_paths, label_paths

        X_features_np = np.array(X_features)
        y_classes_np = np.array(y_classes)
        
        # Identify minority classes based on min_count
        unique_classes, counts = np.unique(y_classes_np, return_counts=True)
        minority_classes_to_oversample = {cls: min_count for cls, count in zip(unique_classes, counts) if count < min_count}
        
        if not minority_classes_to_oversample:
            logging.info("No minority classes found to oversample with SMOTE.")
            return image_paths, label_paths

        logging.info(f"Minority classes for SMOTE: {minority_classes_to_oversample}")
        
        sm = SMOTE(sampling_strategy=minority_classes_to_oversample, random_state=42)
        
        # Perform SMOTE resampling on the features
        X_resampled, y_resampled = sm.fit_resample(X_features_np, y_classes_np)
        
        logging.info(f"SMOTE completed. Original objects: {len(X_features)}, Resampled objects: {len(X_resampled)}")
        
        # IMPORTANT: SMOTE generates synthetic *features* (bbox coordinates in this simple case).
        # It DOES NOT generate new images or full label files.
        # To use these synthetic features, you would typically:
        # 1. Train a generative model (e.g., GAN/Diffusion) conditioned on these features.
        # 2. Or, use these features to augment an existing dataset by "pasting" objects
        #    onto existing images, which requires careful implementation.
        # For this pipeline, we just demonstrate the feature-level oversampling.
        
        # The function returns the original image/label paths, as SMOTE doesn't directly add new files.
        return image_paths, label_paths 
        
    except ImportError:
        logging.warning("imblearn (scikit-learn-contrib) not installed. SMOTE oversampling will not be available.")
        return image_paths, label_paths
    except Exception as e:
        logging.error(f"[WARN] SMOTE oversampling failed: {e}")
        return image_paths, label_paths

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
               Returns (None, None, None) on error.
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
        albumentations.Compose: The composed augmentation pipeline. Returns None on error.
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
    from multiprocessing import Pool # Import Pool locally to avoid global issues

    if pipeline is None:
        logging.error("Augmentation pipeline is None. Cannot augment dataset.")
        return

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
            logging.warning(f"Skipping {img_path}: no corresponding label file {lbl_path}. Copying original.")
            # If no label file, just copy the original image and create an empty label file
            try:
                shutil.copy(img_path, os.path.join(out_img_dir, os.path.basename(img_path)))
                with open(os.path.join(out_lbl_dir, os.path.basename(lbl_path)), 'w') as f:
                    pass # Create empty label file
            except Exception as e:
                logging.error(f"Error copying original {img_path} or creating empty label: {e}")

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
            
            # Even if no boxes, we can still apply image-only augmentations
            # The pipeline should handle empty bboxes gracefully
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
            
            # Write augmented labels
            with open(os.path.join(out_lbl_dir_worker, os.path.splitext(base_name)[0]+'.txt'),'w') as f:
                for c, b in zip(new_class_labels, new_boxes):
                    # Ensure bounding box coordinates are formatted correctly
                    f.write(f"{c} {' '.join([f'{val:.6f}' for val in b])}\n") # Format to 6 decimal places
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
