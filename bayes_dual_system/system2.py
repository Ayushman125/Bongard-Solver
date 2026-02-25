import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .features import extract_predicates
from .types import ExampleItem


Literal = Tuple[str, bool]


@dataclass
class Hypothesis:
    literals: Tuple[Literal, ...]
    log_posterior_unnorm: float


@dataclass
class PosteriorSummary:
    top_hypothesis: Tuple[Literal, ...]
    top_weight: float


class BayesianSystem2:
    def __init__(self, epsilon: float = 0.08, length_penalty: float = 0.8, max_rule_len: int = 2) -> None:
        self.epsilon = epsilon
        self.length_penalty = length_penalty
        self.max_rule_len = max_rule_len
        self.max_candidates = 24
        self.min_discriminative_gap = 0.15
        self._posterior_pos: List[Tuple[Hypothesis, float]] = []
        self._posterior_neg: List[Tuple[Hypothesis, float]] = []

    def fit(self, support_pos: Sequence[ExampleItem], support_neg: Sequence[ExampleItem]) -> None:
        support = list(support_pos) + list(support_neg)
        labels = [1] * len(support_pos) + [0] * len(support_neg)
        pred_maps = [extract_predicates(item.program) for item in support]

        self._posterior_pos = self._build_posterior(pred_maps, labels)
        inverted_labels = [1 - label for label in labels]
        self._posterior_neg = self._build_posterior(pred_maps, inverted_labels)

    def predict_proba(self, item: ExampleItem) -> float:
        p_pos, _, _ = self.predict_with_trace(item)
        return p_pos

    def predict_with_trace(self, item: ExampleItem) -> Tuple[float, PosteriorSummary, PosteriorSummary]:
        pred_map = extract_predicates(item.program)
        p_supports_pos = self._posterior_expectation(self._posterior_pos, pred_map)
        p_supports_neg = self._posterior_expectation(self._posterior_neg, pred_map)

        p_pos_unnorm = max(p_supports_pos, 1e-12)
        p_neg_unnorm = max(p_supports_neg, 1e-12)
        p_pos = p_pos_unnorm / (p_pos_unnorm + p_neg_unnorm)

        return p_pos, self._posterior_summary(self._posterior_pos), self._posterior_summary(self._posterior_neg)

    def top_hypothesis(self) -> Tuple[Literal, ...]:
        if not self._posterior_pos:
            return tuple()
        best = max(self._posterior_pos, key=lambda x: x[1])
        return best[0].literals

    def top_negative_hypothesis(self) -> Tuple[Literal, ...]:
        if not self._posterior_neg:
            return tuple()
        best = max(self._posterior_neg, key=lambda x: x[1])
        return best[0].literals

    def _build_posterior(
        self,
        pred_maps: Sequence[Dict[str, bool]],
        labels: Sequence[int],
    ) -> List[Tuple[Hypothesis, float]]:
        candidate_literals = self._build_candidate_literals(pred_maps, labels)
        hyps = self._enumerate_hypotheses(candidate_literals)

        scored: List[Hypothesis] = []
        for literals in hyps:
            log_like = self._log_likelihood(literals, pred_maps, labels)
            log_prior = -self.length_penalty * len(literals)
            scored.append(Hypothesis(literals=literals, log_posterior_unnorm=log_like + log_prior))

        if not scored:
            scored = [Hypothesis(literals=tuple(), log_posterior_unnorm=0.0)]

        max_log = max(h.log_posterior_unnorm for h in scored)
        weights = [math.exp(h.log_posterior_unnorm - max_log) for h in scored]
        z = sum(weights)
        return list(zip(scored, [w / z for w in weights]))

    def _posterior_expectation(
        self,
        posterior: List[Tuple[Hypothesis, float]],
        pred_map: Dict[str, bool],
    ) -> float:
        expectation = 0.0
        for hypothesis, weight in posterior:
            pred = 1.0 if self._matches(hypothesis.literals, pred_map) else 0.0
            expectation += weight * pred
        return expectation

    def _posterior_summary(self, posterior: List[Tuple[Hypothesis, float]]) -> PosteriorSummary:
        if not posterior:
            return PosteriorSummary(top_hypothesis=tuple(), top_weight=0.0)
        hypothesis, weight = max(posterior, key=lambda x: x[1])
        return PosteriorSummary(top_hypothesis=hypothesis.literals, top_weight=weight)

    def _build_candidate_literals(self, pred_maps: Sequence[Dict[str, bool]], labels: Sequence[int]) -> List[Literal]:
        keys = sorted({k for pred_map in pred_maps for k in pred_map.keys()})
        scored_candidates: List[Tuple[float, Literal]] = []

        n_pos = sum(1 for label in labels if label == 1)
        n_neg = max(len(labels) - n_pos, 1)

        for key in keys:
            for value in [True, False]:
                pos_hits = 0
                neg_hits = 0
                for pred_map, label in zip(pred_maps, labels):
                    is_match = pred_map.get(key, False) == value
                    if not is_match:
                        continue
                    if label == 1:
                        pos_hits += 1
                    else:
                        neg_hits += 1

                pos_rate = pos_hits / max(n_pos, 1)
                neg_rate = neg_hits / n_neg
                gap = abs(pos_rate - neg_rate)

                if gap >= self.min_discriminative_gap:
                    scored_candidates.append((gap, (key, value)))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [literal for _, literal in scored_candidates[: self.max_candidates]]

    def _enumerate_hypotheses(self, literals: Sequence[Literal]) -> List[Tuple[Literal, ...]]:
        hyps: List[Tuple[Literal, ...]] = [tuple()]
        for size in range(1, self.max_rule_len + 1):
            hyps.extend(tuple(combo) for combo in itertools.combinations(literals, size))
        return hyps

    def _log_likelihood(
        self,
        literals: Tuple[Literal, ...],
        pred_maps: Sequence[Dict[str, bool]],
        labels: Sequence[int],
    ) -> float:
        log_like = 0.0
        for pred_map, label in zip(pred_maps, labels):
            pred = 1 if self._matches(literals, pred_map) else 0
            prob = (1.0 - self.epsilon) if pred == label else self.epsilon
            log_like += math.log(max(prob, 1e-12))
        return log_like

    @staticmethod
    def _matches(literals: Iterable[Literal], pred_map: Dict[str, bool]) -> bool:
        for key, expected_value in literals:
            if pred_map.get(key, False) != expected_value:
                return False
        return True
