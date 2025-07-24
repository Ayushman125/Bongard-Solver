import random
import copy
import numpy as np

class EvoPerturber:
    def __init__(self, scorer, max_iter=200, seed=None, alpha=1.0, beta=0.2):
        self.scorer = scorer
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def mutate_logo(self, logo_prog):
        # Example: jitter numeric arguments in LOGO code
        prog = copy.deepcopy(logo_prog)
        for cmd in prog.commands:
            if hasattr(cmd, 'param'):
                cmd.param += np.random.uniform(-5, 5)
        return prog

    def fitness(self, orig_prog, mutated_prog):
        flip = self.scorer.is_flip(mutated_prog)
        geom_diff = self.scorer.geom_distance(orig_prog, mutated_prog)
        return self.alpha * flip - self.beta * geom_diff
    
    def search(self, logo_prog):
        best_prog = logo_prog
        best_score = -np.inf
        for _ in range(self.max_iter):
            cand = self.mutate_logo(logo_prog)
            score = self.fitness(logo_prog, cand)
            if score > best_score:
                best_prog, best_score = cand, score
            if self.scorer.is_flip(cand):
                return cand  # Early exit on label flip
        return best_prog if best_score > 0 else None
