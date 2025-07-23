import torch
import numpy as np
from integration.task_profiler import TaskProfiler

class CrossDomainReasoner:
    """
    Fuses COM/inertia proxies with commonsense predicate weights
    to produce a confidence score for each object.
    """
    def __init__(self):
        self.profiler = TaskProfiler()

    def infer(self, proxies: dict) -> torch.Tensor:
        """
        proxies: dict as returned by PhysicsInference.compute_proxies()
        returns: (B,) confidence scores in [0,1]
        """
        com = proxies['com']             # (B,2)
        inertia = proxies['inertia']     # (B,2,2)
        affordances = proxies['affordances']  # list of dicts

        with self.profiler.profile('cross_domain_infer'):
            # simple heuristic: weight inverse inertia trace + avg affordance weight
            trace = inertia[:,0,0] + inertia[:,1,1]    # (B,)
            inv_trace = trace.add_(1e-6).reciprocal_()
            kb_scores = torch.tensor(
                [float(np.mean(list(a.values()))) if a else 0.0 for a in affordances],
                device=inv_trace.device
            )
            score = torch.sigmoid(inv_trace + kb_scores)
        return score
