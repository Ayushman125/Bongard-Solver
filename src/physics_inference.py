import torch
import numpy as np
from integration.task_profiler import TaskProfiler
from src.commonsense_kb import CommonsenseKB

class PhysicsInference:
    """
    Batched center-of-mass, inertia tensor, and affordance estimates,
    plus commonsense predicate lookups.
    """
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.profiler = TaskProfiler()
        self.kb = CommonsenseKB()  # loads ConceptNet-lite

    def compute_proxies(self, masks: torch.Tensor) -> dict:
        """
        masks: (B, 1, H, W) boolean tensor on self.device
        returns dict of:
          - com: (B,2) float tensor
          - inertia: (B,2,2) float tensor
          - affordance_preds: List[B] of KB query results
        """
        B, _, H, W = masks.shape
        with self.profiler.profile('compute_proxies'):
            # flatten height/width dims
            coords = masks.flatten(2)  # (B,1, H*W)
            inds = torch.arange(H*W, device=self.device)
            row = (inds // W).float()
            col = (inds %  W).float()

            mass = coords.sum(dim=-1)                           # (B,1)
            # center of mass
            com_x = (coords * row).sum(dim=-1) / mass
            com_y = (coords * col).sum(dim=-1) / mass
            com = torch.stack([com_x, com_y], dim=1)            # (B,2)

            # inertia about COM
            dx = row.unsqueeze(0) - com_x.unsqueeze(-1)         # (B,H*W)
            dy = col.unsqueeze(0) - com_y.unsqueeze(-1)         # (B,H*W)
            Ixx = (coords * dy**2).sum(-1)
            Iyy = (coords * dx**2).sum(-1)
            Ixy = -(coords * dx * dy).sum(-1)
            inertia = torch.stack([torch.stack([Ixx,Ixy],-1),
                                   torch.stack([Ixy,Iyy],-1)], dim=1)  # (B,2,2)

        # commonsense predicates for each object
        affordance_preds = []
        for b in range(B):
            affordance_preds.append(
                self.kb.query('support', com[b].cpu().tolist())
            )

        return {
            'com': com,
            'inertia': inertia,
            'affordances': affordance_preds
        }
