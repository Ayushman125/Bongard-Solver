import logging
import math
from dataclasses import dataclass
from typing import Any, Dict

from .system1_nn import NeuralSystem1, TrainConfig
from .system2_bayes_image import BayesianImageSystem2
from .types import Episode

logger = logging.getLogger(__name__)


@dataclass
class HybridPredictionResult:
    p_combined: float
    p1: float
    p2: float
    system1_confidence: float
    used_system2: bool
    w1: float
    w2: float
    trace: Dict[str, Any]


class HybridDualSystemSolver:
    def __init__(
        self,
        image_size: int = 64,
        nn_epochs: int = 60,
        nn_lr: float = 1e-3,
        nn_weight_decay: float = 1e-4,
        system1_confidence_threshold: float = 0.35,
        reliability_lr: float = 1.0,
        seed: int = 0,
        pretrained_model = None,
        use_pretrained: bool = False,
        use_programs: bool = True,
        arbitration_strategy: str = "always_blend",  # Improvement #3: choose strategy
        conflict_threshold: float = 0.3,  # Threshold for conflict-based gating
        max_system2_weight: float = 0.995,  # Prevent total S2 saturation
    ) -> None:
        self.system1_confidence_threshold = system1_confidence_threshold
        self.reliability_lr = reliability_lr
        self.use_pretrained = use_pretrained
        self.arbitration_strategy = arbitration_strategy  # "always_blend" or "conflict_based"
        self.conflict_threshold = conflict_threshold
        self.max_system2_weight = max(0.5, min(0.999, max_system2_weight))

        self.system1 = NeuralSystem1(
            TrainConfig(
                image_size=image_size,
                epochs=nn_epochs,
                lr=nn_lr,
                weight_decay=nn_weight_decay,
                seed=seed,
            ),
            pretrained_model=pretrained_model,
        )
        self.system2 = BayesianImageSystem2(image_size=image_size, use_programs=use_programs)

        self.system1_weight = 0.5
        self.system2_weight = 0.5

    def fit_episode(self, episode: Episode) -> None:
        logger.debug(f"[HYBRID] Fitting episode {episode.task_id}")
        logger.debug(f"[HYBRID] System1 (NN) fitting...")
        self.system1.fit(episode.support_pos, episode.support_neg, freeze_backbone=self.use_pretrained)
        logger.debug(f"[HYBRID] System2 (Bayesian) fitting...")
        self.system2.fit(episode.support_pos, episode.support_neg)
        logger.debug(f"[HYBRID] Episode fitting complete")

    def predict_item(self, item) -> HybridPredictionResult:
        p1, c1 = self.system1.predict_proba(item)
        p2, s2_trace = self.system2.predict_proba(item)
        
        logger.debug(f"[HYBRID] Predict: S1_conf={c1:.4f}, S1_p={p1:.4f}, S2_p={p2:.4f}")

        # Choose arbitration strategy
        if self.arbitration_strategy == "conflict_based":
            # Improvement #3: Conflict-based gating
            # If S1 and S2 disagree significantly, trust S2 (better calibrated)
            # Otherwise, blend both systems
            disagreement = abs(p1 - p2)
            
            if disagreement > self.conflict_threshold:
                # High conflict → trust S2 more (it's better calibrated with programs)
                p = p2
                used_system2 = True
                w1, w2 = 0.0, 1.0
                logger.debug(f"[HYBRID] Conflict-based: disagreement={disagreement:.4f} > {self.conflict_threshold:.4f}, using S2")
            else:
                # Low conflict → blend
                w1, w2 = self.system1_weight, self.system2_weight
                p = w1 * p1 + w2 * p2
                used_system2 = True
                logger.debug(f"[HYBRID] Conflict-based: disagreement={disagreement:.4f} <= {self.conflict_threshold:.4f}, blending")
        else:
            # FIX 1: Always blend - don't bypass S2 based on S1 confidence
            # Reason: S1 is overconfident (95%+ conf) but inaccurate (48%)
            #         S2 is well-calibrated and accurate (70% with programs)
            #         Confidence bypass prevents S2 from being used
            # Reference: THEORETICAL_ANALYSIS.md Section 6 - Fix 1
            w1, w2 = self.system1_weight, self.system2_weight
            p = w1 * p1 + w2 * p2
            used_system2 = True
            logger.debug(f"[HYBRID] Always blend S1+S2: {w1:.4f}*{p1:.4f} + {w2:.4f}*{p2:.4f} = {p:.4f}")
        
        # Old logic (caused 48% accuracy by trusting overconfident S1):
        # if c1 >= self.system1_confidence_threshold:
        #     p = p1; used_system2 = False; w1, w2 = 1.0, 0.0
        # else:
        #     w1, w2 = self.system1_weight, self.system2_weight
        #     p = w1 * p1 + w2 * p2; used_system2 = True

        return HybridPredictionResult(
            p_combined=float(p),
            p1=float(p1),
            p2=float(p2),
            system1_confidence=float(c1),
            used_system2=used_system2,
            w1=float(w1),
            w2=float(w2),
            trace={
                "system2_logp_pos": s2_trace["logp_pos"],
                "system2_logp_neg": s2_trace["logp_neg"],
                "system2_logp_margin": s2_trace["logp_margin"],
                "system2_rule_count": s2_trace.get("rule_count", 0),
                "system2_top_rule": s2_trace.get("top_rule", "n/a"),
                "feature_type": s2_trace.get("feature_type", "unknown"),
            },
        )

    def evaluate_episode(self, episode: Episode) -> Dict[str, Any]:
        logger.debug(f"\n[EPISODE] Evaluating {episode.task_id}")
        self.fit_episode(episode)
        
        logger.debug(f"[EPISODE] Predicting POS query...")
        pos_result = self.predict_item(episode.query_pos)
        
        logger.debug(f"[EPISODE] Predicting NEG query...")
        neg_result = self.predict_item(episode.query_neg)

        concept_margin = pos_result.p_combined - neg_result.p_combined
        concept_correct = 1.0 if concept_margin > 0.0 else 0.0
        concept_tie = 1.0 if abs(concept_margin) < 1e-12 else 0.0
        
        logger.debug(f"[EPISODE] Concept margin: {concept_margin:.6f} (correct={int(concept_correct)})")

        weight_update = self._update_reliability_weights(pos_result, neg_result)
        logger.debug(f"[EPISODE] Weight update: {weight_update}")

        return {
            "concept_correct": concept_correct,
            "concept_margin": float(concept_margin),
            "concept_tie": concept_tie,
            "query_pos_p1": pos_result.p1,
            "query_pos_p2": pos_result.p2,
            "query_pos_combined": pos_result.p_combined,
            "query_neg_p1": neg_result.p1,
            "query_neg_p2": neg_result.p2,
            "query_neg_combined": neg_result.p_combined,
            "used_system2_pos": float(1 if pos_result.used_system2 else 0),
            "used_system2_neg": float(1 if neg_result.used_system2 else 0),
            "query_pos_trace": pos_result.trace,
            "query_neg_trace": neg_result.trace,
            "system1_weight_final": self.system1_weight,
            "system2_weight_final": self.system2_weight,
            "weight_update": weight_update,
        }

    def _update_reliability_weights(self, pos_result: HybridPredictionResult, neg_result: HybridPredictionResult) -> Dict[str, float]:
        """
        Update system weights based on CORRECTNESS and DISAGREEMENT, not just loss.
        
        Key insight: Confidence from loss is backward.
        - System 1 (CNN on 12 examples) gives high confidence but is often wrong
        - System 2 (Bayesian) gives lower confidence but is more honest about uncertainty
        
        Solution: Reward System 2 when:
          1. S2 is correct AND S1 is wrong (S2 corrects S1)
          2. S2 disagrees with S1 (hedging against overconfidence)
        
        This flips the weight update to be correctness-driven, not loss-driven.
        """
        old_w1, old_w2 = self.system1_weight, self.system2_weight
        
        # Evaluate correctness independently
        s1_correct = (pos_result.p1 > neg_result.p1)  # Did S1 predict pos > neg?
        s2_correct = (pos_result.p2 > neg_result.p2)  # Did S2 predict pos > neg?
        
        # Measure disagreement (margin difference)
        s1_margin = pos_result.p1 - neg_result.p1
        s2_margin = pos_result.p2 - neg_result.p2
        disagreement = abs(s1_margin - s2_margin)
        
        # Reward/penalize based on correctness
        w1_adjustment = 1.0
        w2_adjustment = 1.0
        
        # Case 1: S1 correct, S2 correct → both are good, but trust S1 slightly more
        if s1_correct and s2_correct:
            w1_adjustment = math.exp(0.3 * self.reliability_lr)
            w2_adjustment = math.exp(0.2 * self.reliability_lr)
        
        # Case 2: S1 correct, S2 wrong → S1 is better at this task
        elif s1_correct and not s2_correct:
            w1_adjustment = math.exp(0.4 * self.reliability_lr)
            w2_adjustment = math.exp(-0.2 * self.reliability_lr)
        
        # Case 3: S1 wrong, S2 correct → **CRITICAL**: S2 corrected S1, reward heavily
        elif not s1_correct and s2_correct:
            w1_adjustment = math.exp(-0.4 * self.reliability_lr)
            w2_adjustment = math.exp(0.5 * self.reliability_lr)  # Strong reward for correction
            logger.debug(f"[WEIGHT UPDATE] S2 CORRECTED S1! Upweighting System 2")
        
        # Case 4: Both wrong → penalize both, but based on how wrong
        else:
            s1_loss = 0.5 * (_bce_loss(pos_result.p1, 1.0) + _bce_loss(neg_result.p1, 0.0))
            s2_loss = 0.5 * (_bce_loss(pos_result.p2, 1.0) + _bce_loss(neg_result.p2, 0.0))
            
            # Penalize the one with higher loss more heavily
            w1_adjustment = math.exp(-self.reliability_lr * min(s1_loss, 0.5))
            w2_adjustment = math.exp(-self.reliability_lr * min(s2_loss, 0.5))
        
        # Apply adjustments
        self.system1_weight *= w1_adjustment
        self.system2_weight *= w2_adjustment
        
        # Normalize
        norm = self.system1_weight + self.system2_weight
        if norm <= 0 or not (norm > 0):  # Check for NaN or 0
            self.system1_weight, self.system2_weight = 0.5, 0.5
        else:
            self.system1_weight /= norm
            self.system2_weight /= norm

        # Cap S2 weight to keep S1 contributing (helps abstract/generalization cases)
        if self.system2_weight > self.max_system2_weight:
            self.system2_weight = self.max_system2_weight
            self.system1_weight = 1.0 - self.max_system2_weight
        
        return {
            "s1_correct": float(1.0 if s1_correct else 0.0),
            "s2_correct": float(1.0 if s2_correct else 0.0),
            "s1_margin": float(s1_margin),
            "s2_margin": float(s2_margin),
            "disagreement": float(disagreement),
            "w1_adjustment": float(w1_adjustment),
            "w2_adjustment": float(w2_adjustment),
            "old_w1": float(old_w1),
            "old_w2": float(old_w2),
            "new_w1": float(self.system1_weight),
            "new_w2": float(self.system2_weight),
        }


def _bce_loss(p: float, y: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -((y * math.log(p)) + ((1.0 - y) * math.log(1.0 - p)))
