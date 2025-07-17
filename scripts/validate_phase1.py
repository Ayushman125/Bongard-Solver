
# -----------------------------------------------------------
# Make sure we can import both:
#   1) core_models   (living at project root)
#   2) anything under src/
# -----------------------------------------------------------
import os, sys
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
src_dir      = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

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

from src.data.generator import LogoGenerator
from src.perception.primitive_extractor import extract_cnn_feature, load_model, MODEL
from core_models.training import fine_tune_perception, train_perception_with_buffer

# Use Config class and instantiate config
from core_models.training_args import Config
config = Config()


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
    load_model()
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
        v, c = extract_cnn_feature(img)
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
    buffer = ReplayBuffer(capacity=len(imgs))
    logger.info("Preparing buffer for online fine-tuning...")
    for img, lbl in tqdm(list(zip(imgs, labels)), desc="Buffering", leave=False):
        buffer.push(to_tensor(img), lbl)
    logger.info("Evaluating before fine-tuning...")
    pre_acc, *_ = eval_set(imgs, labels)
    logger.info("Starting online fine-tuning...")
    start = time.time()
    fine_tune_perception(MODEL, buffer, config)
    dur = time.time() - start
    logger.info(f"Fine-tuning completed in {dur:.1f}s")
    logger.info("Evaluating after fine-tuning...")
    post_acc, *_ = eval_set(imgs, labels)
    return pre_acc, post_acc, dur


if __name__ == "__main__":
    logger.info("==== Phase 1 Validation: Synthetic Holdout ====")
    s_imgs, s_lbls = build_synth_holdout()
    s_acc, s_pp, s_pt = eval_set(s_imgs, s_lbls)
    logger.info(f"Synth Acc: {s_acc:.4f}")
    plot_calibration(s_pp, s_pt, "Synthetic Calibration")

    logger.info("==== Phase 1 Validation: Real Holdout ====")
    r_imgs, r_lbls = load_real_holdout()
    r_acc, r_pp, r_pt = eval_set(r_imgs, r_lbls)
    logger.info(f"Real Acc: {r_acc:.4f}")
    plot_calibration(r_pp, r_pt, "Real Calibration")

    logger.info("==== Phase 1: Online Fine-tune Test (first 100 synth) ====")
    pre, post, t = online_finetune_test(s_imgs[:100], s_lbls[:100])
    logger.info(f"Online FT: Pre {pre:.4f} → Post {post:.4f} in {t:.1f}s")
