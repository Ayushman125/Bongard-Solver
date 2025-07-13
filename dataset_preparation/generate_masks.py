import json
import cv2
import numpy as np
from pathlib import Path

def generate_box_masks(json_dir: str,
                       img_dir: str,
                       out_mask_dir: str,
                       target_size=(640, 640)):
    """
    For each JSON:
      - Create one-channel PNG mask where pixel=object_id+1
      - Background=0
      - Resizes mask to `target_size`
    """
    json_dir = Path(json_dir)
    img_dir  = Path(img_dir)
    out_dir  = Path(out_mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ann_fp in json_dir.glob("*.json"):
        data   = json.load(ann_fp.open())
        img_id = ann_fp.stem
        img_fp = img_dir / f"{img_id}.jpg"
        if not img_fp.exists(): continue

        img = cv2.imread(str(img_fp))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for idx, obj in enumerate(data.get("objects", []), start=1):
            xmin, ymin, xmax, ymax = map(int, obj["bbox"])
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), color=idx, thickness=-1)

        # resize & save
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / f"{img_id}.png"), mask_resized)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir",     default="raw/annotations")
    p.add_argument("--img_dir",      default="raw/images")
    p.add_argument("--out_mask_dir", default="masks")
    args = p.parse_args()
    generate_box_masks(args.json_dir, args.img_dir, args.out_mask_dir)
