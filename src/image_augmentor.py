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
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self._float_buf = None
        self.transform = K.RandomChoice(
            [
                K.RandomRotation(30.0, p=0.5),
                K.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2), p=0.5),
                K.ColorJitter(0.2, 0.2, 0.2, p=0.3),
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
        imgs = imgs.to(self.device, non_blocking=True)
        B, C, H, W = imgs.shape
        if self._float_buf is None or self._float_buf.shape != imgs.shape:
            self._float_buf = torch.empty_like(imgs, dtype=torch.float32)
        with self.scheduler.allocate('ImageAugmentor'), \
             self.profiler.profile('augment_batch'):
            imgs_f = imgs.to(self._float_buf).float().div_(255.0)
            out_f = self.transform(imgs_f)
            out = (out_f.mul_(255.0).clamp_(0,255).to(torch.uint8))
        return out
