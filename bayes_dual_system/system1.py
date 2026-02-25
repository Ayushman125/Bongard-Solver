import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .features import extract_predicates
from .types import ExampleItem


@dataclass
class ClassStats:
    n: int
    true_counts: Dict[str, int]


class BayesianSystem1:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self._stats = {
            1: ClassStats(n=0, true_counts={}),
            0: ClassStats(n=0, true_counts={}),
        }
        self._class_prior = {1: 0.5, 0: 0.5}

    def fit(self, support_pos: Iterable[ExampleItem], support_neg: Iterable[ExampleItem]) -> None:
        self._stats = {
            1: ClassStats(n=0, true_counts={}),
            0: ClassStats(n=0, true_counts={}),
        }
        self._ingest_items(support_pos, 1)
        self._ingest_items(support_neg, 0)

        total = self._stats[1].n + self._stats[0].n
        if total > 0:
            self._class_prior[1] = self._stats[1].n / total
            self._class_prior[0] = self._stats[0].n / total

    def update(self, item: ExampleItem, true_label: int) -> None:
        self._ingest_item(item, true_label)

    def predict_proba(self, item: ExampleItem) -> Tuple[float, float]:
        predicates = extract_predicates(item.program)
        logp_pos = math.log(self._class_prior[1] + 1e-12)
        logp_neg = math.log(self._class_prior[0] + 1e-12)

        for key, value in predicates.items():
            p_feat_pos = self._posterior_predictive_prob(key, value, 1)
            p_feat_neg = self._posterior_predictive_prob(key, value, 0)
            logp_pos += math.log(max(p_feat_pos, 1e-12))
            logp_neg += math.log(max(p_feat_neg, 1e-12))

        p_pos = 1.0 / (1.0 + math.exp(logp_neg - logp_pos))
        confidence = abs(p_pos - 0.5) * 2.0
        return p_pos, confidence

    def _posterior_predictive_prob(self, key: str, value: bool, label: int) -> float:
        stats = self._stats[label]
        count_true = stats.true_counts.get(key, 0)
        count_false = stats.n - count_true

        if value:
            num = self.alpha + count_true
            den = self.alpha + self.beta + count_true + count_false
            return num / den

        num = self.beta + count_false
        den = self.alpha + self.beta + count_true + count_false
        return num / den

    def _ingest_items(self, items: Iterable[ExampleItem], label: int) -> None:
        for item in items:
            self._ingest_item(item, label)

    def _ingest_item(self, item: ExampleItem, label: int) -> None:
        predicates = extract_predicates(item.program)
        stats = self._stats[label]
        stats.n += 1
        for key, value in predicates.items():
            if value:
                stats.true_counts[key] = stats.true_counts.get(key, 0) + 1
