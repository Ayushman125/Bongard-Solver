import math
from dataclasses import dataclass
from typing import Any, Dict, List

from .system1 import BayesianSystem1
from .system2 import BayesianSystem2
from .types import Episode


@dataclass
class PredictionResult:
    p_pos: float
    predicted_label: int
    used_system2: bool
    system1_confidence: float
    system1_p_pos: float
    system2_p_pos: float
    system1_weight: float
    system2_weight: float
    trace: Dict[str, Any]


class DualSystemBayesianSolver:
    def __init__(
        self,
        system1_confidence_threshold: float = 0.35,
        learning_rate: float = 2.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        epsilon: float = 0.08,
        length_penalty: float = 0.8,
        max_rule_len: int = 2,
    ) -> None:
        self.system1_confidence_threshold = system1_confidence_threshold
        self.system1 = BayesianSystem1(alpha=alpha, beta=beta)
        self.system2 = BayesianSystem2(
            epsilon=epsilon,
            length_penalty=length_penalty,
            max_rule_len=max_rule_len,
        )
        self.learning_rate = learning_rate
        self.system1_weight = 0.5
        self.system2_weight = 0.5

    def fit_episode(self, episode: Episode) -> None:
        self.system1.fit(episode.support_pos, episode.support_neg)
        self.system2.fit(episode.support_pos, episode.support_neg)

    def predict_item(self, item) -> PredictionResult:
        p1, c1 = self.system1.predict_proba(item)
        p2, pos_summary, neg_summary = self.system2.predict_with_trace(item)

        if c1 >= self.system1_confidence_threshold:
            p = p1
            used_system2 = False
            w1 = 1.0
            w2 = 0.0
        else:
            w1 = self.system1_weight
            w2 = self.system2_weight
            p = w1 * p1 + w2 * p2
            used_system2 = True

        pred_label = 1 if p >= 0.5 else 0
        return PredictionResult(
            p_pos=p,
            predicted_label=pred_label,
            used_system2=used_system2,
            system1_confidence=c1,
            system1_p_pos=p1,
            system2_p_pos=p2,
            system1_weight=w1,
            system2_weight=w2,
            trace={
                "system2_top_positive_hypothesis": pos_summary.top_hypothesis,
                "system2_top_positive_weight": pos_summary.top_weight,
                "system2_top_negative_hypothesis": neg_summary.top_hypothesis,
                "system2_top_negative_weight": neg_summary.top_weight,
            },
        )

    def update_arbitration_weights(self, true_label: int, prediction: PredictionResult) -> Dict[str, float]:
        y = float(true_label)

        p1 = min(max(prediction.system1_p_pos, 1e-6), 1 - 1e-6)
        p2 = min(max(prediction.system2_p_pos, 1e-6), 1 - 1e-6)

        loss1 = -((y * _safe_log(p1)) + ((1.0 - y) * _safe_log(1.0 - p1)))
        loss2 = -((y * _safe_log(p2)) + ((1.0 - y) * _safe_log(1.0 - p2)))

        old_w1 = self.system1_weight
        old_w2 = self.system2_weight

        self.system1_weight *= _safe_exp(-self.learning_rate * loss1)
        self.system2_weight *= _safe_exp(-self.learning_rate * loss2)

        norm = self.system1_weight + self.system2_weight
        if norm <= 0.0:
            self.system1_weight = 0.5
            self.system2_weight = 0.5
        else:
            self.system1_weight /= norm
            self.system2_weight /= norm

        return {
            "true_label": y,
            "loss1": loss1,
            "loss2": loss2,
            "old_w1": old_w1,
            "old_w2": old_w2,
            "new_w1": self.system1_weight,
            "new_w2": self.system2_weight,
        }

    def evaluate_episode(self, episode: Episode) -> Dict[str, float]:
        self.fit_episode(episode)

        pos_result = self.predict_item(episode.query_pos)
        neg_result = self.predict_item(episode.query_neg)

        concept_margin = pos_result.p_pos - neg_result.p_pos
        concept_correct = 1.0 if concept_margin > 0.0 else 0.0
        concept_tie = 1.0 if abs(concept_margin) < 1e-12 else 0.0

        pos_correct = 1 if pos_result.predicted_label == 1 else 0
        neg_correct = 1 if neg_result.predicted_label == 0 else 0

        return {
            "concept_correct": concept_correct,
            "concept_margin": concept_margin,
            "concept_tie": concept_tie,
            "query_pos_correct": float(pos_correct),
            "query_neg_correct": float(neg_correct),
            "query_pair_accuracy": float((pos_correct + neg_correct) / 2.0),
            "used_system2_pos": float(1 if pos_result.used_system2 else 0),
            "used_system2_neg": float(1 if neg_result.used_system2 else 0),
            "system1_weight_final": self.system1_weight,
            "system2_weight_final": self.system2_weight,
            "query_pos_p1": pos_result.system1_p_pos,
            "query_pos_p2": pos_result.system2_p_pos,
            "query_pos_combined": pos_result.p_pos,
            "query_neg_p1": neg_result.system1_p_pos,
            "query_neg_p2": neg_result.system2_p_pos,
            "query_neg_combined": neg_result.p_pos,
            "query_pos_trace": pos_result.trace,
            "query_neg_trace": neg_result.trace,
        }


def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-12))


def _safe_exp(x: float) -> float:
    if x > 50:
        x = 50
    if x < -50:
        x = -50
    return math.exp(x)
