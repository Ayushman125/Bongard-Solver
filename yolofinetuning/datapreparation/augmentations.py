import cv2
import numpy as np
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip,
    MixUp, CutMix, Mosaic
)
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    # Compose standard + sample-mix augmentations
    return Compose([
        RandomBrightnessContrast(p=0.5),
        HorizontalFlip(p=0.5),
        Mosaic(p=0.5),
        MixUp(p=0.5),
        CutMix(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'yolo', 'label_fields': ['class_labels']})

def apply_augmentations(image_path, labels):
    """
    image_path: str
    labels: list of [class, x_center, y_center, w, h]
    """
    image = cv2.imread(image_path)
    bboxes = [tuple(l[1:]) for l in labels]
    class_labels = [int(l[0]) for l in labels]

    aug = get_train_transforms()
    result = aug(image=image, bboxes=bboxes, class_labels=class_labels)

    return result['image'], result['bboxes'], result['class_labels']
