
# -----------------------------------------------------------
# Make sure we can import both:
#   1) core_models   (living at project root)
#   2) anything under src/
# -----------------------------------------------------------

import os, sys
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)     # for core_models/
sys.path.insert(0, SRC_ROOT)      # for src/data, src/perception, src/utils

#!/usr/bin/env python3


import os, time, glob, logging
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

import torch
from torchvision.transforms.functional import to_tensor


from core_models.training_args import config
from core_models.training      import train_perception_with_buffer, fine_tune_perception
from src.data.generator        import LogoGenerator
from src.perception.primitive_extractor import extract_cnn_features, MODEL, load_perception_model


# 1) If no checkpoint yet, train model first (only if not already present)
def checkpoint_exists():
    return os.path.exists(config.best_model_path) or os.path.exists(config.last_model_path)

if not checkpoint_exists():
    print(f">>> No checkpoint found at {config.best_model_path} or {config.last_model_path}\n>>> Training Phase-1 perception model...")
    train_perception_with_buffer(config)
    print(">>> Training done. Now proceeding to validation.")
else:
    print(f"Checkpoint found at {config.best_model_path if os.path.exists(config.best_model_path) else config.last_model_path}. Skipping training.")


# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("validate_phase1")

# Ensure checkpoints exist and resume if interrupted
Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
ckpt = config.best_model_path if os.path.exists(config.best_model_path) else config.last_model_path
if os.path.exists(ckpt):
    logger.info(f"Loading model checkpoint: {ckpt}")
    # Model is loaded automatically in primitive_extractor; no need to call load_model()
else:
    logger.error(f"No checkpoint found at {ckpt} after attempted training. Model may be uninitialized!")


import functools
@functools.lru_cache(maxsize=2)
def build_synth_holdout(n=None, cache_path="synth_holdout.npz"):
    n = n or config.synth_holdout_count
    if os.path.exists(cache_path):
        logger.info(f"Loading cached synthetic holdout from {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)
        imgs = [Image.fromarray(x) for x in arr['imgs']]
        labels = arr['labels'].tolist()
        return imgs, labels
    logger.info(f"Generating {n} synthetic holdout samples...")
    gen = LogoGenerator(config.img_size, config.textures_dir)
    imgs, labels = [], []
    for _ in tqdm(range(n), desc="Synth Holdout Generation"):
        feat, val = gen.sample_rule()
        pos, neg, _ = gen.sample(feat, val)
        for img in pos + neg:
            imgs.append(img)
            labels.append(MODEL.class_names.index(val))
    # Save cache
    arr_imgs = np.stack([np.array(img) for img in imgs])
    np.savez_compressed(cache_path, imgs=arr_imgs, labels=np.array(labels))
    logger.info(f"Saved synthetic holdout to {cache_path}")
    return imgs, labels


@functools.lru_cache(maxsize=2)
def load_real_holdout(root=None, cache_path="real_holdout.npz"):
    root = root or config.real_holdout_root
    if not os.path.isdir(root) or not os.listdir(root):
        logger.warning(f"No files in real holdout dir {root}, skipping real validation.")
        return None, None
    if os.path.exists(cache_path):
        logger.info(f"Loading cached real holdout from {cache_path}")
        arr = np.load(cache_path, allow_pickle=True)
        imgs = [Image.fromarray(x) for x in arr['imgs']]
        labels = arr['labels'].tolist()
        return imgs, labels
    logger.info(f"Loading real holdout images from {root}")
    imgs, labels = [], []
    for prob in tqdm(sorted(os.listdir(root)), desc="Real Holdout Problems"):
        for lbl_folder in ["0","1"]:
            files = glob.glob(f"{root}/{prob}/images/{lbl_folder}/*")
            for fn in tqdm(files, desc=f"{prob}/{lbl_folder}", leave=False):
                img = Image.open(fn).convert("L")
                imgs.append(img)
                labels.append(int(lbl_folder))
    if not imgs:
        logger.warning(f"No images found in real holdout dir {root}, skipping real validation.")
        return None, None
    arr_imgs = np.stack([np.array(img) for img in imgs])
    np.savez_compressed(cache_path, imgs=arr_imgs, labels=np.array(labels))
    logger.info(f"Saved real holdout to {cache_path}")
    return imgs, labels


def eval_set(imgs, labels):
    preds, confs = [], []
    logger.info(f"Evaluating {len(imgs)} images...")
    # Fallback class names if MODEL is None or missing class_names
    fallback_class_names = ['triangle', 'quadrilateral', 'filled', 'outlined']
    class_names = getattr(MODEL, 'class_names', fallback_class_names)
    for img in tqdm(imgs, desc="Inferencing"):
        result = extract_cnn_features(img)
        if isinstance(result, tuple) and len(result) >= 2:
            v, c = result[:2]
        else:
            raise ValueError(f"Unexpected output from extract_cnn_features: {result}")
        try:
            preds.append(class_names.index(v))
        except ValueError:
            # If v is not in class_names, append a default index (e.g., 0)
            preds.append(0)
        confs.append(c)
    acc = accuracy_score(labels, preds)
    logger.info(f"Accuracy: {acc:.4f}")
    prob_true, prob_pred = calibration_curve(
        [int(p==t) for p,t in zip(preds, labels)],
        confs, n_bins=config.validate_bins
    )
    return acc, prob_pred, prob_true


def plot_calibration(x, y, title):
    logger.info(f"Plotting calibration curve: {title}")
    plt.plot(x, y, "o-")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.title(title); plt.xlabel("Predicted Confidence"); plt.ylabel("Empirical Accuracy")
    plt.show()


def online_finetune_test(imgs, labels):
    from core_models.replay_buffer import ReplayBuffer
    # Ensure MODEL is loaded
    model = load_perception_model()
    if model is None:
        raise RuntimeError("PerceptionModel failed to load—cannot fine-tune")
    buffer = ReplayBuffer(capacity=len(imgs))
    logger.info("Preparing buffer for online fine-tuning...")
    for img, lbl in tqdm(list(zip(imgs, labels)), desc="Buffering", leave=False):
        buffer.push(to_tensor(img), lbl)
    logger.info("Evaluating before fine-tuning...")
    pre_acc, *_ = eval_set(imgs, labels)
    logger.info("Starting online fine-tuning...")
    start = time.time()
    fine_tune_perception(model, buffer, config)
    dur = time.time() - start
    logger.info(f"Fine-tuning completed in {dur:.1f}s")
    logger.info("Evaluating after fine-tuning...")
    post_acc, *_ = eval_set(imgs, labels)
    return pre_acc, post_acc, dur


if __name__ == "__main__":
    # Determine synthetic holdout cache path from config if present
    synth_cache = None
    if hasattr(config, 'data') and hasattr(config.data, 'synthetic_data_config'):
        synth_cache = config.data.synthetic_data_config.get('holdout_cache', None)
    if synth_cache is None:
        synth_cache = 'synth_holdout.npz'

    if hasattr(config.data, 'use_synthetic_data') and config.data.use_synthetic_data:
        logger.info("==== Phase 1 Validation: Synthetic Holdout ====")
        s_imgs, s_lbls = build_synth_holdout(cache_path=synth_cache)
        s_acc, s_pp, s_pt = eval_set(s_imgs, s_lbls)
        logger.info(f"Synth Acc: {s_acc:.4f}")
        plot_calibration(s_pp, s_pt, "Synthetic Calibration")
    else:
        logger.info("==== Phase 1 Validation: Real Holdout ====")
        r_imgs, r_lbls = load_real_holdout()
        if r_imgs is not None:
            r_acc, r_pp, r_pt = eval_set(r_imgs, r_lbls)
            logger.info(f"Real Acc: {r_acc:.4f}")
            plot_calibration(r_pp, r_pt, "Real Calibration")
        else:
            logger.info("No real holdout data found. Skipping real validation.")

    logger.info("==== Phase 1: Online Fine-tune Test (first 100 synth) ====")
    # Use synthetic for fine-tune if available, else real
    if 's_imgs' in locals() and 's_lbls' in locals():
        pre, post, t = online_finetune_test(s_imgs[:100], s_lbls[:100])
    elif 'r_imgs' in locals() and r_imgs is not None:
        pre, post, t = online_finetune_test(r_imgs[:100], r_lbls[:100])
    else:
        logger.info("No data available for online fine-tune test.")
        pre = post = t = None
    if pre is not None:
        logger.info(f"Online FT: Pre {pre:.4f} → Post {post:.4f} in {t:.1f}s")
