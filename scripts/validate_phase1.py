
#!/usr/bin/env python3
import os, time, glob
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, calibration_curve

import torch
from torchvision.transforms.functional import to_tensor

from src.data.generator import LogoGenerator
from src.perception.primitive_extractor import extract_cnn_feature, MODEL
from core_models.training import fine_tune_perception
from core_models.training_args import config

# Ensure checkpoints exist
Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
ckpt = config.best_model_path if os.path.exists(config.best_model_path) else config.last_model_path
MODEL.load_state_dict(torch.load(ckpt, map_location=config.device))

def build_synth_holdout(n=config.synth_holdout_count):
    gen = LogoGenerator(size=config.img_size, bg_textures_dir=config.textures_dir)
    imgs, labels = [], []
    for _ in range(n):
        feat, val = gen.sample_rule()
        pos, neg, _ = gen.sample(feat, val)
        for img in pos + neg:
            imgs.append(img)
            labels.append(MODEL.class_names.index(val))
    return imgs, labels

def load_real_holdout(root=config.real_holdout_root):
    imgs, labels = [], []
    for prob in sorted(os.listdir(root)):
        for lbl_folder in ["0","1"]:
            for fn in glob.glob(f"{root}/{prob}/images/{lbl_folder}/*"):
                img = Image.open(fn).convert("L")
                imgs.append(img)
                labels.append(int(lbl_folder))
    return imgs, labels

def eval_set(imgs, labels):
    preds, confs = [], []
    for img in tqdm(imgs, desc="Inferencing"):
        v, c = extract_cnn_feature(img)
        preds.append(MODEL.class_names.index(v))
        confs.append(c)
    acc = accuracy_score(labels, preds)
    prob_true, prob_pred = calibration_curve(
        [int(p==t) for p,t in zip(preds, labels)],
        confs, n_bins=config.validate_bins
    )
    return acc, prob_pred, prob_true

def plot_calibration(x, y, title):
    plt.plot(x, y, "o-")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.title(title); plt.xlabel("Predicted Confidence"); plt.ylabel("Empirical Accuracy")
    plt.show()

def online_finetune_test(imgs, labels):
    from core_models.replay_buffer import ReplayBuffer
    buffer = ReplayBuffer(capacity=len(imgs))
    for img, lbl in zip(imgs, labels):
        buffer.push(to_tensor(img), lbl)
    pre_acc, *_ = eval_set(imgs, labels)
    start = time.time()
    fine_tune_perception(MODEL, buffer, config)
    dur = time.time() - start
    post_acc, *_ = eval_set(imgs, labels)
    return pre_acc, post_acc, dur

if __name__ == "__main__":
    # Synthetic
    s_imgs, s_lbls = build_synth_holdout()
    s_acc, s_pp, s_pt = eval_set(s_imgs, s_lbls)
    print(f"Synth Acc: {s_acc:.4f}")
    plot_calibration(s_pp, s_pt, "Synthetic Calibration")

    # Real
    r_imgs, r_lbls = load_real_holdout()
    r_acc, r_pp, r_pt = eval_set(r_imgs, r_lbls)
    print(f"Real Acc: {r_acc:.4f}")
    plot_calibration(r_pp, r_pt, "Real Calibration")

    # Online fine-tune
    pre, post, t = online_finetune_test(s_imgs[:100], s_lbls[:100])
    print(f"Online FT: Pre {pre:.4f}  Post {post:.4f} in {t:.1f}s")
