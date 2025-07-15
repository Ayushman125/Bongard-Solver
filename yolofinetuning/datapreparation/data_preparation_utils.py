
import os
import json
import shutil
import random
import argparse
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import yaml
from pathlib import Path
from logger import setup_logging, log_detection
from embedding_model import EmbedDetector
from metrics import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Data Preparation Utility")
    parser.add_argument('--input-dir', required=True, help='Directory with raw images')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--annotations', required=True, help='Path to annotation file (JSON, CSV, or XML)')
    parser.add_argument('--split-ratio', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--classes-file', required=True, help='Path to class names file (one class per line)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml (optional)')
    parser.add_argument('--use-owlvit', action='store_true', help='Use OWL-ViT detection pipeline')
    return parser.parse_args()

def ensure_dirs(base_dir):
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

def parse_classes(classes_file):
    with open(classes_file) as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def parse_annotations(ann_path):
    # Only JSON supported for now; extend as needed
    with open(ann_path) as f:
        data = json.load(f)
    # Expecting: {image_id: {"boxes": [[xmin, ymin, xmax, ymax, class_name], ...], "width": int, "height": int}}
    return data

def convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    x_c = ((xmin + xmax) / 2) / img_w
    y_c = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    # Clip to [0,1]
    x_c, y_c, w, h = [max(0, min(1, v)) for v in [x_c, y_c, w, h]]
    return x_c, y_c, w, h

def write_label_file(label_path, objects):
    with open(label_path, 'w') as f:
        for obj in objects:
            f.write(f"{obj['class_idx']} {obj['x_c']:.6f} {obj['y_c']:.6f} {obj['w']:.6f} {obj['h']:.6f}\n")

def split_filenames(filenames, split_ratio, seed=42):
    random.seed(seed)
    random.shuffle(filenames)
    n_train = int(len(filenames) * split_ratio)
    return filenames[:n_train], filenames[n_train:]

def generate_yaml(output_dir, class_names):
    yaml_path = os.path.join(output_dir, 'data.yaml')
    yaml_content = {
        'train': os.path.abspath(os.path.join(output_dir, 'images/train')),
        'val': os.path.abspath(os.path.join(output_dir, 'images/val')),
        'nc': len(class_names),
        'names': class_names
    }
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

def generate_classes_file(output_dir, class_names):
    with open(os.path.join(output_dir, 'classes.names'), 'w') as f:
        for name in class_names:
            f.write(name + '\n')

def process_dataset(args):
    # If config is provided, load YAML config and set up logging/OWL-ViT/metrics
    if getattr(args, 'config', None):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        setup_logging(cfg['logging'])
        if getattr(args, 'use_owlvit', False):
            detector = EmbedDetector(cfg['model'])
            evaluator = Evaluator(cfg['metrics'])
            # Use OWL-ViT detection for all images in images_dir
            images_dir = Path(cfg['data']['images_dir'])
            output_dir = Path(cfg['data']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            for img_path in images_dir.glob("*.jpg"):
                image = Image.open(img_path).convert("RGB")
                prompts = ["object"]  # replace with your class prompts
                boxes, scores, labels, embeddings = detector.detect(image, prompts)
                # Write YOLO-format label file
                label_file = output_dir / f"{img_path.stem}.txt"
                with open(label_file, 'w') as f:
                    for box, score, label in zip(boxes, scores, labels):
                        x0,y0,x1,y1 = box
                        width, height = image.size
                        cx, cy = (x0 + x1)/2/width, (y0 + y1)/2/height
                        w, h = (x1 - x0)/width, (y1 - y0)/height
                        f.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                # Log each detection
                for idx, (b, s, lb, emb) in enumerate(zip(boxes, scores, labels, embeddings)):
                    log_detection(
                        image_path=str(img_path),
                        box=b.tolist(),
                        score=float(s),
                        label=int(lb),
                        embedding=emb.tolist()
                    )
                evaluator.update(gt_boxes=None, pred_boxes=boxes, scores=scores)
            evaluator.finalize()
            print("[INFO] OWL-ViT detection and YOLO label generation complete.")
            return
    # --- Classic YOLO data prep logic (preserved) ---
    ensure_dirs(args.output_dir)
    class_names = parse_classes(args.classes_file)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    ann_data = parse_annotations(args.annotations)
    image_ids = list(ann_data.keys())
    train_ids, val_ids = split_filenames(image_ids, args.split_ratio, args.seed)
    stats = defaultdict(int)
    for split, ids in [('train', train_ids), ('val', val_ids)]:
        for img_id in tqdm(ids, desc=f'Processing {split} set'):
            entry = ann_data[img_id]
            img_path = os.path.join(args.input_dir, img_id)
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"[WARN] Could not open image {img_id}: {e}")
                continue
            # Copy image
            out_img_path = os.path.join(args.output_dir, f'images/{split}', img_id)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            shutil.copy(img_path, out_img_path)
            # Prepare label objects
            objects = []
            for box in entry.get('boxes', []):
                if len(box) < 5:
                    continue
                xmin, ymin, xmax, ymax, class_name = box
                if class_name not in class_to_idx:
                    continue
                # Filter degenerate boxes
                if xmax <= xmin or ymax <= ymin:
                    continue
                # Discard very small boxes
                if (xmax - xmin) < 2 or (ymax - ymin) < 2:
                    continue
                x_c, y_c, w, h = convert_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
                # Discard out-of-bounds or tiny boxes
                if w <= 0 or h <= 0 or w > 1 or h > 1:
                    continue
                obj = {
                    'class_idx': class_to_idx[class_name],
                    'x_c': x_c, 'y_c': y_c, 'w': w, 'h': h
                }
                objects.append(obj)
                stats[class_name] += 1
            # Write label file
            label_path = os.path.join(args.output_dir, f'labels/{split}', img_id.replace('.jpg', '.txt').replace('.png', '.txt'))
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            write_label_file(label_path, objects)
    generate_yaml(args.output_dir, class_names)
    generate_classes_file(args.output_dir, class_names)
    print("\n--- Summary ---")
    print(f"Total images processed: {len(image_ids)}")
    print(f"Total labels written: {sum(stats.values())}")
    for cname in class_names:
        print(f"  {cname}: {stats[cname]}")
    # Check for label/image mismatches
    for split, ids in [('train', train_ids), ('val', val_ids)]:
        img_dir = os.path.join(args.output_dir, f'images/{split}')
        lbl_dir = os.path.join(args.output_dir, f'labels/{split}')
        for fname in os.listdir(img_dir):
            lbl_file = os.path.join(lbl_dir, fname.replace('.jpg', '.txt').replace('.png', '.txt'))
            if not os.path.exists(lbl_file):
                print(f"[WARN] No label file for image: {fname}")
    print("Data preparation complete.")
