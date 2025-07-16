#!/usr/bin/env python3
import os, glob, time
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score, calibration_curve
from torchvision import transforms

import torch
from torchvision.transforms.functional import to_tensor


from src.data.generator import LogoGenerator
from src.perception.primitive_extractor import extract_cnn_feature, MODEL
from src.data.bongardlogo_dataset import BongardLogoDataset
from core_models.training_args import Config

# Load config
config = Config()

# Create checkpoints dir if missing
Path(config.debug.save_model_checkpoints).mkdir(exist_ok=True, parents=True)

# Load latest/best model
ckpt = os.path.join(config.debug.save_model_checkpoints, 'best_perception_model.pt')
if not os.path.exists(ckpt):
    ckpt = os.path.join(config.debug.save_model_checkpoints, 'bongard_perception_last.pth')
MODEL.load_state_dict(torch.load(ckpt, map_location=config.device))
MODEL.eval()


# --- Real Bongard-LOGO Validation: Per-Type Accuracy ---
def get_split_type_from_path(path):
    # path: .../ShapeBongard_V2/{hd,bd,ff}/.../images/{0,1}/img.png
    parts = path.replace('\\', '/').split('/')
    if 'ShapeBongard_V2' in parts:
        idx = parts.index('ShapeBongard_V2')
        if idx+1 < len(parts):
            return parts[idx+1]
    return 'unknown'

def validate_real_bongardlogo():
    print("\n--- Validating on real Bongard-LOGO images (ShapeBongard_V2) ---")
    dataset = BongardLogoDataset(root_dir='data/Bongard-LOGO', img_size=128)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    results_by_type = {'hd': [], 'bd': [], 'ff': []}
    all_preds, all_labels = [], []
    for batch_imgs, batch_labels in tqdm(loader, desc="Bongard-LOGO Eval"):
        for i in range(batch_imgs.size(0)):
            img = batch_imgs[i]
            label = int(batch_labels[i].item())
            # Find split type for this sample
            path, _ = dataset.samples[i]
            split_type = get_split_type_from_path(path)
            # Run perception model (extract_cnn_feature expects PIL Image)
            pil_img = transforms.ToPILImage()(img.cpu()).convert('L')
            pred = None
            try:
                feats = extract_cnn_feature(pil_img)
                # Use shape or fill or whatever is most relevant; here just use shape if present
                if 'shape' in feats:
                    pred = feats['shape'][0]
                else:
                    pred = -1
            except Exception as e:
                print(f"Error extracting features: {e}")
                pred = -1
            results_by_type.setdefault(split_type, []).append((pred, label))
            all_preds.append(pred)
            all_labels.append(label)
    # Compute per-type accuracy
    for split_type in ['hd', 'bd', 'ff']:
        preds, labels = zip(*results_by_type[split_type]) if results_by_type[split_type] else ([],[])
        if preds:
            acc = accuracy_score(labels, preds)
            print(f"Accuracy for {split_type}: {acc:.3f} ({len(preds)} samples)")
        else:
            print(f"No samples for {split_type}.")
    print("Overall accuracy:", accuracy_score(all_labels, all_preds))

if __name__ == "__main__":
    validate_real_bongardlogo()
