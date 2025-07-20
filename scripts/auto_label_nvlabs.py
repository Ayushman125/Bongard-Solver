import os, json
import numpy as np
from PIL import Image
# This script assumes you have the bongard-logo repository cloned
# and its dependencies installed.
# from bongard.data.datasets import BongardLogoProblem
# from bongard.renderer import BongardLogoRenderer
from pycocotools import mask as mask_utils

# Adjust these paths to your checkout
# The user mentioned the data is in ShapeBongard_V2
NVLABS_ROOT   = "ShapeBongard_V2" 
OUTPUT_COCO   = "data/nvlabs/annotations_coco.json"
IMG_OUT_DIR   = "data/nvlabs/images"
os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_COCO), exist_ok=True)


# Build COCO skeleton
images, annotations = [], []
ann_id = 1
CATEGORIES = [
  {"id":1,"name":"circle"},
  {"id":2,"name":"square"},
  {"id":3,"name":"triangle"},
  {"id":4,"name":"pentagon"},
  {"id":5,"name":"star"},
  # … add more categories as needed
]

print("Starting COCO annotation extraction for NVlabs data.")
print(f"Please ensure the Bongard-LOGO repository is available and its dependencies are installed.")
print("The following paths are configured:")
print(f"NVLABS_ROOT: {os.path.abspath(NVLABS_ROOT)}")
print(f"OUTPUT_COCO: {os.path.abspath(OUTPUT_COCO)}")
print(f"IMG_OUT_DIR: {os.path.abspath(IMG_OUT_DIR)}")

# The following part is commented out because it requires the bongard-logo library,
# which may not be installed. The user can uncomment and run this script once the
# environment is set up.

# try:
#     from bongard.data.datasets import BongardLogoProblem
#     from bongard.renderer import BongardLogoRenderer
# except ImportError:
#     print("\nERROR: bongard-logo library not found.")
#     print("Please clone https://github.com/NVlabs/Bongard-LOGO and install its requirements.")
#     exit()


# renderer = BongardLogoRenderer(canvas_size=128)

# for pid in sorted(os.listdir(NVLABS_ROOT)):
#     prob_dir = os.path.join(NVLABS_ROOT, pid)
#     if not os.path.isdir(prob_dir):
#         continue
#     try:
#         prob = BongardLogoProblem.load_from_folder(prob_dir)
#     except Exception as e:
#         print(f"Could not load problem from {prob_dir}: {e}")
#         continue

#     for side,label in [("1",1), ("0",0)]:
#         side_dir = os.path.join(prob_dir, side)
#         if not os.path.exists(side_dir):
#             continue
#         for idx, fname in enumerate(sorted(os.listdir(side_dir))):
#             img_id = int(pid)*100 + (label*10 + idx)
#             prog   = prob.programs[side][idx]
            
#             # Render image + per-shape masks
#             img, masks, shape_types = renderer.render_with_masks(prog)
#             out_name = f"{pid}_{side}_{idx:02d}.png"
#             img.save(os.path.join(IMG_OUT_DIR, out_name))

#             images.append({
#                 "id": img_id,
#                 "width": img.width,
#                 "height": img.height,
#                 "file_name": out_name
#             })

#             # Turn each mask into a COCO annotation
#             for m,shape in zip(masks, shape_types):
#                 rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
#                 bbox = mask_utils.toBbox(rle).tolist()
#                 annotations.append({
#                     "id": ann_id,
#                     "image_id": img_id,
#                     "category_id": next(c["id"] for c in CATEGORIES if c["name"]==shape),
#                     "segmentation": rle,
#                     "bbox": bbox,
#                     "iscrowd": 0
#                 })
#                 ann_id += 1

# # Write COCO JSON
# with open(OUTPUT_COCO,"w") as f:
#     json.dump({
#         "images": images,
#         "annotations": annotations,
#         "categories": CATEGORIES
#     }, f, indent=2)

# print(f"✅ Extracted COCO annotations for NVlabs data to {OUTPUT_COCO}")

