from typing import List

import numpy as np
from PIL import Image


def load_image_array(image_path: str, image_size: int = 64) -> np.ndarray:
    img = Image.open(image_path).convert("L").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def extract_image_features(image_path: str, image_size: int = 64) -> np.ndarray:
    arr = load_image_array(image_path=image_path, image_size=image_size)
    ink = 1.0 - arr

    eps = 1e-8
    ink_sum = float(ink.sum()) + eps
    h, w = ink.shape

    ys, xs = np.mgrid[0:h, 0:w]
    xs_norm = xs / max(w - 1, 1)
    ys_norm = ys / max(h - 1, 1)

    centroid_x = float((ink * xs_norm).sum() / ink_sum)
    centroid_y = float((ink * ys_norm).sum() / ink_sum)
    spread_x = float(np.sqrt(((ink * ((xs_norm - centroid_x) ** 2)).sum() / ink_sum) + eps))
    spread_y = float(np.sqrt(((ink * ((ys_norm - centroid_y) ** 2)).sum() / ink_sum) + eps))

    mirror_lr = np.fliplr(arr)
    mirror_ud = np.flipud(arr)
    h_sym = float(np.mean(np.abs(arr - mirror_lr)))
    v_sym = float(np.mean(np.abs(arr - mirror_ud)))

    dx = np.diff(arr, axis=1)
    dy = np.diff(arr, axis=0)
    edge_strength = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    q1 = float(ink[: h // 2, : w // 2].sum() / ink_sum)
    q2 = float(ink[: h // 2, w // 2 :].sum() / ink_sum)
    q3 = float(ink[h // 2 :, : w // 2].sum() / ink_sum)
    q4 = float(ink[h // 2 :, w // 2 :].sum() / ink_sum)

    features: List[float] = [
        float(ink.mean()),
        centroid_x,
        centroid_y,
        spread_x,
        spread_y,
        h_sym,
        v_sym,
        edge_strength,
        q1,
        q2,
        q3,
        q4,
    ]

    return np.asarray(features, dtype=np.float32)
