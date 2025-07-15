import cv2
import json
import os
import numpy as np
from random import choice, randint

def load_object_masks(mask_dir):
    """
    Loads object masks from a directory.
    Returns list of (class_name, mask_array).
    """
    objects = []
    try:
        for fname in os.listdir(mask_dir):
            if fname.endswith('.png'):
                mask = cv2.imread(os.path.join(mask_dir, fname), 0)
                if mask is not None:
                    objects.append((fname.split('.png')[0], mask))
                else:
                    print(f"[WARN] Could not load mask: {fname}")
    except Exception as e:
        print(f"[ERROR] Failed to load object masks: {e}")
    return objects

def paste_objects(bg_image, bg_labels, objects, max_pastes=3):
    """
    Paste random objects onto a background image.
    Returns new image and updated labels.
    """
    try:
        h, w, _ = bg_image.shape
        new_labels = bg_labels.copy()
        result = bg_image.copy()
        for _ in range(randint(1, max_pastes)):
            name, mask = choice(objects)
            y_idxs, x_idxs = np.where(mask > 0)
            if len(x_idxs) == 0 or len(y_idxs) == 0:
                print(f"[WARN] Empty mask for object: {name}")
                continue
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
        return result, new_labels
    except Exception as e:
        print(f"[ERROR] Copy-paste synthesis failed: {e}")
        return bg_image, bg_labels
        y_c = (ty + (y1-y0)/2) / h
        new_labels.append([int(name), x_c, y_c, (x1-x0)/w, (y1-y0)/h])
    return result, new_labels

# --- CLI entry point ---
if __name__ == "__main__":
    import argparse
    import cv2
    parser = argparse.ArgumentParser(description="Copy-paste synthesis utility")
    parser.add_argument('--bg', required=True, help='Background image path')
    parser.add_argument('--lbl', required=True, help='Background label txt')
    parser.add_argument('--mask_dir', required=True, help='Object mask directory')
    parser.add_argument('--out_img', required=True, help='Output image path')
    parser.add_argument('--out_lbl', required=True, help='Output label txt')
    args = parser.parse_args()
    bg_img = cv2.imread(args.bg)
    bg_lbls = [list(map(float, l.strip().split())) for l in open(args.lbl)]
    objects = load_object_masks(args.mask_dir)
    out_img, out_lbls = paste_objects(bg_img, bg_lbls, objects)
    cv2.imwrite(args.out_img, out_img)
    with open(args.out_lbl, 'w') as f:
        for l in out_lbls:
            f.write(' '.join(map(str, l))+'\n')
