import sys
import os
# Ensure repo root is on sys.path so 'data' is importable
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

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
from src.perception.primitive_extractor import extract_cnn_feature, MODEL
from core_models.training import fine_tune_perception
from core_models.training_args import Config
config = Config()


# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("validate_phase1")

# Ensure checkpoints exist
Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
ckpt = config.best_model_path if os.path.exists(config.best_model_path) else config.last_model_path
if os.path.exists(ckpt):
    logger.info(f"Loading model checkpoint: {ckpt}")
    MODEL.load_state_dict(torch.load(ckpt, map_location=config.device))
else:
    logger.warning(f"No checkpoint found at {ckpt}, model may be uninitialized!")


def build_synth_holdout(n=config.synth_holdout_count, cache_path="synth_holdout.npz"):
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


def load_real_holdout(root=config.real_holdout_root, cache_path="real_holdout.npz"):
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
    for img in tqdm(imgs, desc="Inferencing"):
        v, c = extract_cnn_feature(img)
        preds.append(MODEL.class_names.index(v))
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
