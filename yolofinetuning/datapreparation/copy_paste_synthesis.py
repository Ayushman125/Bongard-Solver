import cv2
import json
import os
import numpy as np
from random import choice, randint

def load_object_masks(mask_dir):
    objects = []
    for fname in os.listdir(mask_dir):
        if fname.endswith('.png'):
            mask = cv2.imread(os.path.join(mask_dir, fname), 0)
            objects.append((fname.split('.png')[0], mask))
    return objects

def paste_objects(bg_image, bg_labels, objects, max_pastes=3):
    h, w, _ = bg_image.shape
    new_labels = bg_labels.copy()
    result = bg_image.copy()

    for _ in range(randint(1, max_pastes)):
        name, mask = choice(objects)
        y_idxs, x_idxs = np.where(mask > 0)
        x0, y0 = min(x_idxs), min(y_idxs)
        x1, y1 = max(x_idxs), max(y_idxs)
        obj = mask[y0:y1, x0:x1]
        roi = bg_image[y0:y1, x0:x1]
        alpha = (obj > 0).astype(np.uint8) * 255
        for c in range(3):
            roi[:,:,c] = roi[:,:,c] * (1 - alpha/255) + bg_image[y0:y1, x0:x1, c] * (alpha/255)
        tx, ty = randint(0, w - (x1-x0)), randint(0, h - (y1-y0))
        result[ty:ty+(y1-y0), tx:tx+(x1-x0)] = roi
        x_c = (tx + (x1-x0)/2) / w
        y_c = (ty + (y1-y0)/2) / h
        new_labels.append([int(name), x_c, y_c, (x1-x0)/w, (y1-y0)/h])
    return result, new_labels
