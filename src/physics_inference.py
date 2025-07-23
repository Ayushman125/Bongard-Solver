import torch
import numpy as np
from integration.task_profiler import TaskProfiler
from integration.adaptive_scheduler import AdaptiveScheduler
from src.commonsense_kb import CommonsenseKB

class PhysicsInference:
    """
    Batched center-of-mass, inertia tensor, and affordance estimates,
    plus commonsense predicate lookups.
    """
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device)
        self.profiler = TaskProfiler()
        self.scheduler = AdaptiveScheduler()
        self.kb = CommonsenseKB()
        self._cache = {}

    def _get_coords(self, H: int, W: int):
        key = (H, W)
        if key not in self._cache:
            inds = torch.arange(H * W, device=self.device)
            rows = (inds // W).float()
            cols = (inds %  W).float()
            self._cache[key] = (rows, cols)
        return self._cache[key]

    def compute_proxies(self, masks: torch.Tensor) -> dict:
        """
        masks: (B, 1, H, W) boolean tensor on self.device
        returns dict of:
          - com: (B,2) float tensor
          - inertia: (B,2,2) float tensor
          - affordance_preds: List[B] of KB query results
        """
        B, _, H, W = masks.shape
        masks = masks.to(self.device).float()
        with self.scheduler.allocate('PhysicsInference'), \
             self.profiler.profile('compute_proxies'):
            rows, cols = self._get_coords(H, W)
            flattened = masks.view(B, -1)
            mass = flattened.sum(dim=1, keepdim=True)
            mass_safe = mass.clone().clamp_min_(1.0)
            com_x = (flattened * rows).sum(dim=1, keepdim=True).div_(mass_safe)
            com_y = (flattened * cols).sum(dim=1, keepdim=True).div_(mass_safe)
            com = torch.cat([com_x, com_y], dim=1)
            dx = rows.unsqueeze(0) - com_x
            dy = cols.unsqueeze(0) - com_y
            Ixx = (flattened * dy**2).sum(dim=1)
            Iyy = (flattened * dx**2).sum(dim=1)
            Ixy = -(flattened * dx * dy).sum(dim=1)
            inertia = torch.stack([
                torch.stack([Ixx, Ixy], dim=1),
                torch.stack([Ixy, Iyy], dim=1)
            ], dim=1)
        affordances = [
            self.kb.query('support', com[b].cpu().tolist())
            for b in range(B)
        ]
        return {'com': com, 'inertia': inertia, 'affordances': affordances}
