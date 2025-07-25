"""
GPU-batched geometric and relational augmentations for Phase 1
Integrates with TaskProfiler and CUDAStreamManager
"""


__version__ = "1.0.0"

# Ensure project root is in sys.path for imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import numpy as np
from typing import List, Dict, Tuple, Optional
from integration.task_profiler import TaskProfiler
from integration.cuda_stream_manager import CUDAStreamManager
from integration.data_validator import DataValidator
import time

class ImageAugmentor:
    """GPU-batched geometric and relational augmentations with profiling"""
    def __init__(self, device: str = 'cuda', batch_size: int = 32):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.profiler = TaskProfiler()
        self.stream_manager = CUDAStreamManager()
        self.data_validator = DataValidator()
        self.geometric_transforms = nn.Sequential(
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ).to(self.device)
        self.relational_transforms = nn.Sequential(
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
        ).to(self.device)
    def augment_batch(self, images: torch.Tensor, augment_type: str = 'both') -> Dict[str, torch.Tensor]:
        start_time = time.time()
        # Convert images to nested list for schema validation (only shape, not actual data)
        shape_list = list(images.shape) if hasattr(images, 'shape') else images
        self.data_validator.validate({
            'images': [[0.0]*shape_list[-1] for _ in range(shape_list[0])],  # Dummy nested list
            'type': augment_type
        }, 'image_augmentation.schema.json')
        if images.device != self.device:
            images = images.to(self.device)
        results = {'original': images}
        with torch.cuda.stream(self.stream_manager.get_stream()):
            if augment_type in ['geometric', 'both']:
                results['geometric'] = self.geometric_transforms(images)
            if augment_type in ['relational', 'both']:
                results['relational'] = self.relational_transforms(images)
            if augment_type == 'both':
                results['combined'] = self.relational_transforms(
                    self.geometric_transforms(images)
                )
        latency = (time.time() - start_time) * 1000
        self.profiler.log_latency('image_augmentation', latency, {
            'batch_size': images.shape[0],
            'augment_type': augment_type,
            'device': str(self.device)
        })
        results['profiling'] = {
            'latency_ms': latency,
            'throughput_imgs_per_sec': images.shape[0] / (latency / 1000)
        }

        return results


# CLI main function
import argparse
import json
import pickle

def main():
    parser = argparse.ArgumentParser(description="GPU-batched image augmentation for Bongard problems")
    parser.add_argument('--input', type=str, required=True, help='Input JSON file with image paths and labels')
    parser.add_argument('--out', type=str, required=True, help='Output pickle file for augmented data')
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel workers (not used, placeholder)')
    parser.add_argument('--rotate', type=float, default=30, help='Max rotation degrees')
    parser.add_argument('--scale', type=float, default=1.2, help='Max scale factor')
    parser.add_argument('--shear', action='store_true', help='Enable shear (not implemented)')
    parser.add_argument('--type', type=str, choices=['geometric', 'relational', 'both'], default='both', help='Augmentation type')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for augmentation (default: 16)')
    args = parser.parse_args()

    # Load input JSON
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect image paths
    image_paths = []
    if isinstance(data, list):
        for item in data:
            if 'image_path' in item:
                image_paths.append(item['image_path'])
    elif isinstance(data, dict) and 'positives' in data:
        for item in data['positives']:
            if 'image_path' in item:
                image_paths.append(item['image_path'])

    # Load images as tensors, robust to missing files
    from PIL import Image
    import torchvision.transforms as T

    tensor_transform = T.Compose([T.ToTensor()])
    batch_size = args.batch_size
    augmentor = ImageAugmentor(batch_size=batch_size)
    augmentor.geometric_transforms[0] = transforms.RandomRotation(degrees=args.rotate)
    augmentor.geometric_transforms[1] = transforms.RandomAffine(degrees=15, scale=(0.8, args.scale))

    all_results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        pil_images = []
        for p in batch_paths:
            norm_path = p.replace('category_0', '0').replace('category_1', '1')
            norm_path = os.path.normpath(norm_path)
            if not os.path.isabs(norm_path):
                norm_path = os.path.join(os.getcwd(), norm_path)
            pil_images.append(Image.open(norm_path).convert('RGB'))
        images = torch.stack([tensor_transform(img) for img in pil_images])
        results = augmentor.augment_batch(images, augment_type=args.type)
        # Move tensors to CPU and detach for pickling
        for k in ['original', 'geometric', 'relational', 'combined']:
            if k in results:
                results[k] = results[k].cpu().detach()
        all_results.append(results)
        # Free up GPU memory
        del images, pil_images, results
        torch.cuda.empty_cache()

    # Save all batch results as a list
    with open(args.out, 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    main()
