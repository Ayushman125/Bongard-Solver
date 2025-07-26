"""
Visualize outputs from all advanced mask techniques for a sample image, and ensure training and inference happen in one run.
"""
import cv2
import numpy as np
import os
from src.bongard_augmentor.hybrid import AdvancedSAMEnsemble, BongardSegmentationTrainer

# Path to sample image (user should set this)
SAMPLE_IMAGE_PATH = "data/sample_bongard.png"

# Train deep segmentation model if not already trained
trainer = BongardSegmentationTrainer()
if not trainer.load_trained_model():
    print("[INFO] No trained model found. Starting training...")
    trainer.train(epochs=10, batch_size=8, lr=1e-3)
    print("[INFO] Training complete.")
else:
    print("[INFO] Trained model loaded.")

# Load sample image
image = cv2.imread(SAMPLE_IMAGE_PATH, cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

# Run all advanced mask techniques
ensemble = AdvancedSAMEnsemble()
results = ensemble.generate_all_advanced_masks(image)

# Save and display all outputs
os.makedirs("mask_outputs", exist_ok=True)
for name, mask in results.items():
    if mask is not None:
        out_path = f"mask_outputs/{name}.png"
        cv2.imwrite(out_path, mask)
        print(f"[RESULT] {name} mask saved to {out_path}")
    else:
        print(f"[RESULT] {name} mask is empty or failed.")

print("[INFO] All mask outputs visualized and saved.")
