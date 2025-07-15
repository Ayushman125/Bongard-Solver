
# --- Required imports ---
import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
import torch
import logging
try:
    from groundingdino import build_gdino
    from sam import predictor
except ImportError:
    build_gdino = None
    predictor = None

# --- Auto-labeling function ---
def auto_label(image_path, output_label_path, min_score=0.3):
    """
    Auto-labels an image using GroundingDINO and SAM.
    Args:
        image_path: path to image
        output_label_path: path to save YOLO label
        min_score: minimum detection score
    """
    if build_gdino is None or predictor is None:
        print("[ERROR] Required models not available for auto-labeling.")
        return
    gdino = build_gdino(checkpoint='groundingdino_swint_ogc.pth')
    gdino.eval()
    image = cv2.imread(image_path)
    with torch.no_grad():
        boxes, scores, labels = gdino.predict(image, caption="logo primitive")
    predictor.set_image(image)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=boxes,
        multimask_output=False
    )
    h, w, _ = image.shape
    with open(output_label_path, 'w') as f:
        for box, score, label in zip(boxes, scores, labels):
            if score < min_score:
                continue
            x1, y1, x2, y2 = box
            x_c = ((x1 + x2) / 2) / w
            y_c = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{label} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

# --- Logging and MLflow ---
import logging
try:
    import mlflow
except ImportError:
    mlflow = None

def pseudo_labeling(unlabeled_img_dir, model, out_label_dir, min_score=0.3):
    """
    Generate pseudo-labels for unlabeled images using a trained model.
    Args:
        unlabeled_img_dir: Directory with unlabeled images
        model: Trained detection model
        out_label_dir: Output directory for pseudo-labels
        min_score: minimum detection score
    """
    import os, glob
    os.makedirs(out_label_dir, exist_ok=True)
def pseudo_labeling(img_dir, model, out_label_dir):
    # --- Model ensemble: list of models ---
    if not isinstance(model, list):
        model = [model]
    os.makedirs(out_label_dir, exist_ok=True)
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        preds = []
        for m in model:
            if m is None:
                continue
            # --- Use model to predict labels ---
            pred = m.predict(img)
            preds.append(pred)
        # --- Consensus filtering ---
        if len(preds) == 0:
            continue
        consensus = get_consensus_labels(preds)
        out_path = os.path.join(out_label_dir, os.path.basename(img_path).replace('.png', '.txt'))
        with open(out_path, 'w') as f:
            f.write('\n'.join(consensus))
        print(f"Saved pseudo-labels: {out_path}")

def get_consensus_labels(preds):
    # --- Majority voting ---
    all_labels = [label for pred in preds for label in pred]
    label_counts = Counter(all_labels)
    consensus = [label for label, count in label_counts.items() if count >= (len(preds) // 2 + 1)]
    return consensus

# Helper IoU for ensemble
def _iou(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB
    xi1, yi1 = max(xA1, xB1), max(yA1, yB1)
    xi2, yi2 = min(xA2, xB2), min(yA2, yB2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    areaA = (xA2 - xA1) * (yA2 - yA1)
    areaB = (xB2 - xB1) * (yB2 - yB1)
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0


# --- Active learning utility ---
def active_learning_step(unlabeled_img_dir, model, selection_count=100):
    """
    Select most informative samples for annotation.
    Args:
        unlabeled_img_dir: Directory with unlabeled images
        model: Trained detection model
        selection_count: Number of samples to select
    Returns:
        List of selected image paths
    """
    import os, glob
    import numpy as np
    img_paths = glob.glob(os.path.join(unlabeled_img_dir, '*.jpg'))
    uncertainties = []
    for img_path in img_paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        with torch.no_grad():
            boxes, scores, labels = model.predict(image)
        uncertainty = np.std(scores) if len(scores) > 0 else 0
        uncertainties.append((uncertainty, img_path))
    uncertainties.sort(reverse=True)
    selected = [p for _, p in uncertainties[:selection_count]]
    logging.info(f"Active learning selected: {selected}")
    if mlflow:
        mlflow.log_param("active_learning_selected", selected)
    return selected

# --- CLI entry point ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto labeling utility")
    parser.add_argument('--img', help='Path to image')
    parser.add_argument('--out', help='Path to output label')
    parser.add_argument('--min_score', type=float, default=0.3)
    args = parser.parse_args()
    if args.img and args.out:
        auto_label(args.img, args.out, args.min_score)
