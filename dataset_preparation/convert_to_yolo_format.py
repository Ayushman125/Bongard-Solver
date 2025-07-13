import json
import cv2
from pathlib import Path

EPS = 1e-6

def xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert Pascal VOC (xmin,ymin,xmax,ymax) → YOLO (xc,yc,w,h) normalized [0,1].
    """
    w = max(xmax - xmin, EPS)
    h = max(ymax - ymin, EPS)
    xc = xmin + w / 2
    yc = ymin + h / 2
    return xc / img_w, yc / img_h, w / img_w, h / img_h

def convert(json_dir: str, img_dir: str, out_label_dir: str):
    json_dir = Path(json_dir)
    img_dir  = Path(img_dir)
    out_dir  = Path(out_label_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ann_fp in json_dir.glob("*.json"):
        data = json.load(ann_fp.open())
        img_id = ann_fp.stem
        img_fp = img_dir / f"{img_id}.jpg"
        if not img_fp.exists():
            print(f"⚠️  Missing image for {img_id}, skipping")
            continue

        img = cv2.imread(str(img_fp))
        h, w = img.shape[:2]
        lines = []

        for obj in data.get("objects", []):
            cls_id = obj["class_id"]
            xmin, ymin, xmax, ymax = obj["bbox"]  # [x_min,y_min,x_max,y_max]
            xc, yc, bw, bh = xyxy_to_yolo(xmin, ymin, xmax, ymax, w, h)
            # filter out‐of‐bounds or degenerate boxes
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and bw > EPS and bh > EPS and bw <= 1 and bh <= 1):
                continue
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        (out_dir / f"{img_id}.txt").write_text("\n".join(lines))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir",      default="raw/annotations")
    p.add_argument("--img_dir",       default="raw/images")
    p.add_argument("--out_label_dir", default="labels/yolo")
    args = p.parse_args()
    convert(args.json_dir, args.img_dir, args.out_label_dir)
