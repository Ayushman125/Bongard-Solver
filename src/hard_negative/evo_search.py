import random
import copy
import numpy as np

class EvoPerturber:

    def perturb(self, base_cmds):
        # Randomly mutate the program using one of the grammar rules
        from src.data_pipeline.logo_mutator import mutate
        return mutate(base_cmds)

    def score(self, prog):
        # Use the scorer's concept confidence as the score
        if hasattr(self.scorer, 'predict_concept_confidence'):
            return self.scorer.predict_concept_confidence(prog)
        return 0.0
    def __init__(self, scorer, max_iter=200, seed=None, alpha=1.0, beta=0.2):
        self.scorer = scorer
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def search(self, base_cmds):
        import logging
        best_prog, best_score, stagnation = None, float('-inf'), 0
        for i in range(self.max_iter):
            cand = self.perturb(base_cmds)
            score = self.score(cand)
            if score > best_score:
                best_prog, best_score, stagnation = cand, score, 0
            else:
                stagnation += 1
            if self.scorer.is_flip(cand):
                logging.debug("EvoPerturber.search: flip at iter %d", i)
                return cand
            if stagnation >= 50:
                logging.debug("EvoPerturber.search: stagnated at iter %d", i)
                break
        return best_prog
