import cv2
import numpy as np
import logging
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from torchvision.transforms import functional as TF

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
    from PIL import Image
    parser = argparse.ArgumentParser(description="OWL-ViT detection utility")
    parser.add_argument('--img', required=True, help='Path to image')
    parser.add_argument('--prompt', required=True, nargs='+', help='Detection prompts')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', default='google/owlvit-base-patch32')
    args = parser.parse_args()
    cfg = {'device': args.device, 'name': args.model, 'detection_threshold': 0.3, 'max_queries': 10}
    det = EmbedDetector(cfg)
    image = Image.open(args.img).convert('RGB')
    boxes, scores, labels, embeddings = det.detect(image, args.prompt)
    print('Boxes:', boxes)
    print('Scores:', scores)
    print('Labels:', labels)


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

    def detect(self, image, prompts):
        try:
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
