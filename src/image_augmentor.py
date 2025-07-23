import torch
import kornia.augmentation as K
from integration.task_profiler import TaskProfiler
from integration.adaptive_scheduler import AdaptiveScheduler

class ImageAugmentor:
    """
    GPU-batched geometric & stroke augmentations.
    Uses Kornia to perform random rotations, scalings, shears, and
    brush-stroke noise in a single fused pipeline.
    """
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.transform = K.RandomChoice(
            [
                K.RandomRotation(degrees=30.0, p=0.5),
                K.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2), p=0.5),
                K.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                K.RandomErasing(scale=(0.02, 0.1), p=0.3),
            ]
        ).to(self.device)
        self.profiler = TaskProfiler()
        self.scheduler = AdaptiveScheduler()

    def augment_batch(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, C, H, W) tensor on self.device
        returns: augmented (B, C, H, W)
        """
        with self.scheduler.allocate('image_augmentor'), \
             self.profiler.profile('augment_batch'):
            # Kornia expects floats in [0,1]
            imgs = imgs.float() / 255.0
            out = self.transform(imgs)
            return (out * 255.0).to(torch.uint8)
