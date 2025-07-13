import torch
from groundingdino.models import build_model as build_gdino
from detectron2.engine import DefaultPredictor
from segment_anything import sam_model_registry, SamPredictor
import cv2
import json

gdino = build_gdino(checkpoint='groundingdino_swint_ogc.pth')
gdino.eval()

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
predictor = SamPredictor(sam)

def auto_label(image_path, output_label_path, min_score=0.3):
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
