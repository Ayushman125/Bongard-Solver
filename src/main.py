import argparse
import json
import logging
import numpy as np
import cv2
from bongard_augmentor.hybrid import HybridAugmentor

def load_images_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    images = []
    for entry in data:
        img_path = entry.get('image_path')
        if img_path:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def save_results(results, out_path):
    import pickle
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)

def main():
    parser = argparse.ArgumentParser(description='Bongard Augmentation Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    parser.add_argument('--out', type=str, required=True, help='Output pickle file')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel workers')
    parser.add_argument('--rotate', type=float, default=0.0, help='Rotation angle')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--type', type=str, default='geometric', help='Augmentation type')
    parser.add_argument('--force-emergency-qa', action='store_true', help='Force emergency QA')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = {
        'models': ['vit_h', 'vit_b'],
        'type': args.type,
        'rotate': args.rotate,
        'scale': args.scale,
        'batch_size': args.batch_size,
        'force_emergency_qa': args.force_emergency_qa
    }
    augmentor = HybridAugmentor(config)
    images = load_images_from_json(args.input)
    results = []
    for i in range(0, len(images), args.batch_size):
        batch = images[i:i+args.batch_size]
        batch_results = augmentor.process_batch(batch)
        results.extend(batch_results)
    save_results(results, args.out)
    logging.info(f"Saved augmented results to {args.out}")

if __name__ == '__main__':
    main()
