import os, json
import numpy as np
from PIL import Image
# This script assumes you have the bongard-logo repository cloned
# and its dependencies installed.
# from bongard.data.datasets import BongardLogoProblem
# from bongard.renderer import BongardLogoRenderer
from pycocotools import mask as mask_utils


# --- Updated for real Bongard-LOGO and ShapeBongard_V2 structure ---

# Path to Bongard-LOGO repo (for program/mask extraction)
BONGARD_LOGO_REPO = "data/Bongard-LOGO"
# Path to real images (ShapeBongard_V2)
SHAPE_BONGARD_ROOT = "ShapeBongard_V2"
# Output paths
OUTPUT_COCO   = "data/nvlabs/annotations_coco.json"
IMG_OUT_DIR   = "data/nvlabs/images"
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_COCO), exist_ok=True)

# COCO skeleton
images, annotations = [], []
ann_id = 1
CATEGORIES = [
  {"id":1,"name":"circle"},
  {"id":2,"name":"square"},
  {"id":3,"name":"triangle"},
  {"id":4,"name":"pentagon"},
  {"id":5,"name":"star"},
  # ... add more categories as needed
]

print("Starting COCO annotation extraction for ShapeBongard_V2 data.")
print(f"BONGARD_LOGO_REPO: {os.path.abspath(BONGARD_LOGO_REPO)}")
print(f"SHAPE_BONGARD_ROOT: {os.path.abspath(SHAPE_BONGARD_ROOT)}")
print(f"OUTPUT_COCO: {os.path.abspath(OUTPUT_COCO)}")
print(f"IMG_OUT_DIR: {os.path.abspath(IMG_OUT_DIR)}")

# --- Main extraction loop ---
try:
    from bongard.data.datasets import BongardLogoProblem
    from bongard.renderer import BongardLogoRenderer
except ImportError:
    print("\nERROR: bongard-logo library not found.")
    print("Please clone https://github.com/NVlabs/Bongard-LOGO into data/Bongard-LOGO and install its requirements.")
    exit()

renderer = BongardLogoRenderer(canvas_size=128)


# Loop over all splits in ShapeBongard_V2
for split in ["bd", "hd", "ff"]:
    split_images_dir = os.path.join(SHAPE_BONGARD_ROOT, split, "images")
    if not os.path.exists(split_images_dir):
        print(f"Split not found: {split_images_dir}")
        continue
    # Each split/images contains many task folders
    for task_folder in sorted(os.listdir(split_images_dir)):
        task_path = os.path.join(split_images_dir, task_folder)
        if not os.path.isdir(task_path):
            continue
        # Each task folder contains 0/ and 1/ subfolders
        for side, label in [("1", 1), ("0", 0)]:
            side_path = os.path.join(task_path, side)
            if not os.path.exists(side_path):
                continue
            img_files = sorted([f for f in os.listdir(side_path) if f.endswith('.png')])
            # Load Bongard-LOGO program for this task
            logo_task_path = os.path.join(BONGARD_LOGO_REPO, task_folder)
            try:
                prob = BongardLogoProblem.load_from_folder(logo_task_path)
            except Exception as e:
                print(f"Could not load Bongard-LOGO program for {task_folder}: {e}")
                continue
            for idx, fname in enumerate(img_files):
                img_path = os.path.join(side_path, fname)
                # Defensive: check if program exists for this index
                try:
                    prog = prob.programs[side][idx]
                except Exception as e:
                    print(f"No program for {task_folder} {side} idx {idx}: {e}")
                    continue
                # Render image + per-shape masks
                img, masks, shape_types = renderer.render_with_masks(prog)
                out_name = f"{split}_{task_folder}_{side}_{idx:02d}.png"
                img.save(os.path.join(IMG_OUT_DIR, out_name))

                images.append({
                    "id": ann_id,
                    "width": img.width,
                    "height": img.height,
                    "file_name": out_name
                })

                # Turn each mask into a COCO annotation
                for m, shape in zip(masks, shape_types):
                    rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                    bbox = mask_utils.toBbox(rle).tolist()
                    cat_id = next((c["id"] for c in CATEGORIES if c["name"] == shape), 0)
                    annotations.append({
                        "id": ann_id,
                        "image_id": ann_id,
                        "category_id": cat_id,
                        "segmentation": rle,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    ann_id += 1

# Write COCO JSON
with open(OUTPUT_COCO, "w") as f:
    json.dump({
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }, f, indent=2)

print(f"✅ Extracted COCO annotations for ShapeBongard_V2 data to {OUTPUT_COCO}")

