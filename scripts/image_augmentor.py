import torch.utils.data as data
import kornia.augmentation as K

class ImagePathDataset(data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        norm_path = path.replace('category_0', '0').replace('category_1', '1')
        norm_path = os.path.normpath(norm_path)
        if not os.path.isabs(norm_path):
            norm_path = os.path.join(os.getcwd(), norm_path)
        img = Image.open(norm_path).convert('L')
        arr = self.tensor_transform(img)
        # Ensure float32 for Kornia
        return arr.float()
"""
GPU-batched geometric augmentations for Phase 1
Integrates TaskProfiler and CUDAStreamManager
"""

__version__ = "1.0.0"

import sys
import os
import hashlib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import argparse
import json
import pickle


import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
import torchvision.transforms as T

# Prepare transform for loading grayscale images (global for multiprocessing)
tensor_transform = T.Compose([
    T.ToTensor(),                   # yields [1, H, W]
    T.Normalize(mean=[0.5], std=[0.5])
])

def process_batch(batch_paths, augment_type, device, batch_size):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    tensor_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    images = []
    for path in batch_paths:
        norm_path = path.replace('category_0', '0').replace('category_1', '1')
        norm_path = os.path.normpath(norm_path)
        if not os.path.isabs(norm_path):
            norm_path = os.path.join(os.getcwd(), norm_path)
        img = Image.open(norm_path).convert('L')
        images.append(tensor_transform(img))
    images = torch.stack(images)
    # Create a new ImageAugmentor in each worker
    from integration.task_profiler import TaskProfiler
    from integration.cuda_stream_manager import CUDAStreamManager
    from integration.data_validator import DataValidator
    augmentor = ImageAugmentor(device=device, batch_size=batch_size)
    results = augmentor.augment_batch(images, augment_type=augment_type)
    # Move tensors to CPU and detach for pickling
    for k in ['original', 'geometric', 'combined']:
        if k in results:
            results[k] = results[k].cpu().detach()
    return results
from typing import Dict

from integration.task_profiler import TaskProfiler
from integration.cuda_stream_manager import CUDAStreamManager
from integration.data_validator import DataValidator


class ImageAugmentor:
    """GPU-batched geometric augmentations with profiling and strict binary masks using Kornia"""
    def __init__(self, device: str = 'cuda', batch_size: int = 32, geometric_transforms=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.profiler = TaskProfiler()
        self.stream_manager = CUDAStreamManager()
        self.data_validator = DataValidator()

        # Use Kornia for fast GPU augmentations
        if geometric_transforms is None:
            geometric_transforms = [
                K.RandomRotation(degrees=30),
                K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.3),
                K.RandomPerspective(distortion_scale=0.2, p=0.3),
            ]
        self.geometric_transforms = nn.Sequential(*geometric_transforms)

    def augment_batch(
        self,
        images: torch.Tensor,
        augment_type: str = 'geometric'
    ) -> Dict[str, torch.Tensor]:
        start_time = time.time()

        # Schema validation
        self.data_validator.validate(
            {
                'images': images.tolist(),    # actual image data as nested lists
                'type': augment_type
            },
            'image_augmentation.schema.json'
        )

        # Move to device
        if images.device != self.device:
            images = images.to(self.device, non_blocking=True)

        results = {}
        results['original'] = images.clone()

        # Apply transforms in a dedicated CUDA stream, ensure transforms run on CUDA
        stream = self.stream_manager.get_stream()
        with torch.cuda.stream(stream):
            if augment_type in ['geometric', 'both']:
                # Ensure input is float32 and on CUDA
                images = images.float().to(self.device, non_blocking=True)
                # Kornia expects [B, C, H, W], grayscale is [B, 1, H, W]
                mask = self.geometric_transforms(images)
                results['geometric'] = mask

            if augment_type == 'both':
                results['combined'] = torch.cat([
                    results['original'], results['geometric']
                ], dim=1)

        # End profiling
        latency = (time.time() - start_time) * 1000
        self.profiler.log_latency('image_augmentation', latency, {
            'batch_size': images.shape[0],
            'augment_type': augment_type,
            'device': str(self.device),
            'latency_ms': latency
        })
        results['profiling'] = {
            'latency_ms': latency,
            'throughput_imgs_per_sec': images.shape[0] / (latency / 1000)
        }

        return results


def compute_hash(input_path, params_dict):
    """Compute a hash of the input file and augmentation parameters."""
    h = hashlib.sha256()
    with open(input_path, 'rb') as f:
        h.update(f.read())
    for k in sorted(params_dict.keys()):
        h.update(str(k).encode())
        h.update(str(params_dict[k]).encode())
    return h.hexdigest()

def cache_valid(cache_path, hash_path, current_hash):
    """Check if cache and hash file exist and match current hash."""
    if not os.path.exists(cache_path) or not os.path.exists(hash_path):
        return False
    with open(hash_path, 'r') as f:
        cached_hash = f.read().strip()
    return cached_hash == current_hash

def save_hash(hash_path, hash_val):
    with open(hash_path, 'w') as f:
        f.write(hash_val)

def profile_optimal_batch_size(dataset, augmentor, args):
    """Profile and select the optimal batch size for throughput without OOM."""
    import gc
    candidate_sizes = [8, 16, 32, 64, 128]
    best_size = candidate_sizes[0]
    best_throughput = 0
    print("[INFO] Profiling batch sizes for optimal throughput...")
    for size in candidate_sizes:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=size,
            shuffle=False,
            num_workers=args.parallel,
            pin_memory=True,
            persistent_workers=True
        )
        try:
            images = next(iter(loader))
            results = augmentor.augment_batch(images, augment_type=args.type)
            throughput = results['profiling']['throughput_imgs_per_sec']
            print(f"Batch size {size}: {throughput:.2f} imgs/sec")
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"Batch size {size} caused OOM, skipping.")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"Batch size {size} failed: {e}")
    print(f"[INFO] Selected optimal batch size: {best_size}")
    return best_size

def main():
    parser = argparse.ArgumentParser(
        description="GPU-batched image augmentation for Bongard problems"
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input JSON file with image paths and labels'
    )
    parser.add_argument(
        '--out', type=str, required=True,
        help='Output pickle file for augmented data'
    )
    parser.add_argument(
        '--parallel', type=int, default=8,
        help='Number of parallel workers (placeholder)'
    )
    parser.add_argument(
        '--rotate', type=float, default=30,
        help='Max rotation degrees'
    )
    parser.add_argument(
        '--scale', type=float, default=1.2,
        help='Max scale factor'
    )
    parser.add_argument(
        '--type', type=str,
        choices=['geometric', 'both'],
        default='geometric',
        help='Augmentation type'
    )
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size for augmentation'
    )

    args = parser.parse_args()

    # Hash input and parameters for cache validation
    params_dict = {
        'rotate': args.rotate,
        'scale': args.scale,
        'type': args.type,
        'batch_size': args.batch_size,
        'parallel': args.parallel,
        'input': args.input
    }
    current_hash = compute_hash(args.input, params_dict)
    hash_path = args.out + '.hash'

    # Check cache validity
    if cache_valid(args.out, hash_path, current_hash):
        print(f"[INFO] Cache valid for {args.out}. Skipping augmentation.")
        return

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

    geometric_transforms = [
        K.RandomRotation(degrees=args.rotate),
        K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, args.scale)),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.3),
        K.RandomPerspective(distortion_scale=0.2, p=0.3),
    ]
    augmentor = ImageAugmentor(batch_size=args.batch_size, geometric_transforms=geometric_transforms)

    # Use DataLoader for efficient batch loading
    dataset = ImagePathDataset(image_paths)

    # Profile and tune batch size if requested
    if args.batch_size == 0:
        batch_size = profile_optimal_batch_size(dataset, augmentor, args)
    else:
        batch_size = args.batch_size

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.parallel,
        pin_memory=True,
        persistent_workers=True
    )

    all_results = []
    for images in loader:
        results = augmentor.augment_batch(images, augment_type=args.type)
        all_results.append(results)
        torch.cuda.empty_cache()

    # Save augmented results and hash
    with open(args.out, 'wb') as f:
        pickle.dump(all_results, f)
    save_hash(hash_path, current_hash)


if __name__ == "__main__":
    main()
