"""
BanditModel for adaptive rule selection in MetaController.
Simple epsilon-greedy implementation for demonstration.
"""
import random

class BanditModel:
    def __init__(self, arms, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.stats = {arm: {'attempts': 0, 'successes': 0} for arm in arms}

    def update(self, arm, reward):
        self.stats[arm]['attempts'] += 1
        self.stats[arm]['successes'] += reward

    def sample(self, all_rules, batch_size):
        # Epsilon-greedy: with probability epsilon, pick random; else pick best
        if random.random() < self.epsilon:
            return random.sample(all_rules, batch_size)
        # Otherwise, pick rules with lowest success rate
        sorted_rules = sorted(all_rules, key=lambda r: self._success_rate(r.name))
        return sorted_rules[:batch_size]

    def _success_rate(self, arm):
        s = self.stats[arm]
        if s['attempts'] == 0:
            return 0.0
        return s['successes'] / s['attempts']
