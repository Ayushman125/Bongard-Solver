import os
import json
from pathlib import Path
from src.bongard_generator.sampler import BongardSampler
from src.bongard_rules import ALL_BONGARD_RULES

# You may need to adjust config loading as per your pipeline
cfg = None  # TODO: Replace with actual config loading
output_dir = "exported_scenes"
os.makedirs(output_dir, exist_ok=True)

sampler = BongardSampler(cfg)
for idx, (rule_obj, label) in enumerate(sampler):
    img, objs = sampler.pop_scene(rule_obj, label)  # your interface
    # Save image
    img_path = os.path.join(output_dir, f"{idx:05d}.png")
    img.save(img_path)

    # Build metadata
    meta = {
        "rule": rule_obj.description,
        "label": label,
        "objects": objs  # list of dicts: shape, color, x,y,w,h, etc.
    }
    meta_path = img_path.replace(".png", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
