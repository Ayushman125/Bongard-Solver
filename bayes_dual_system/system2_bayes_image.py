import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from .image_utils import extract_image_features
from .program_utils import extract_program_features, compute_sequence_features, find_consensus_program
from .types import ExampleItem


class BayesianImageSystem2:
    def __init__(
        self,
        image_size: int = 64,
        var_floor: float = 1e-3,
        prior_strength: float = 1.0,
        max_rules: int = 24,
        beta_prior: float = 1.0,
        max_primitive_rules: int = 12,
        use_programs: bool = True,
    ) -> None:
        self.image_size = image_size
        self.var_floor = var_floor
        self.prior_strength = prior_strength
        self.max_rules = max_rules
        self.beta_prior = beta_prior
        self.max_primitive_rules = max_primitive_rules
        self.use_programs = use_programs

        self._mu = {0: None, 1: None}
        self._var = {0: None, 1: None}
        self._prior = {0: 0.5, 1: 0.5}
        self._rules: List[RuleHypothesis] = []
        self._using_program_features = False
        
        # Improvement #2: Store consensus programs for sequence matching
        self._consensus_pos_program: Optional[Any] = None
        self._consensus_neg_program: Optional[Any] = None
        self._use_sequence_matching = True  # Enable sequence features

    def _extract_features(self, item: ExampleItem, include_sequence_features: bool = False) -> np.ndarray:
        """
        Extract features from ExampleItem, prioritizing LOGO programs.
        
        Strategy:
        1. Try program features first (symbolic, aligns with concepts)
        2. Optionally add sequence matching features (Improvement #2)
        3. Fallback to image features if programs unavailable
        4. Track which feature type we're using for interpretability
        
        Improvement #2: Sequence matching for better free-form shape recognition.
        Adds LCS, edit distance, prefix/suffix matching features.
        
        Args:
            item: Example item with image and/or program
            include_sequence_features: If True, append sequence similarity features
        
        Returns:
            Feature vector (16-dim symbolic + 12-dim sequence = 28-dim, or 16-dim, or image features)
        """
        if self.use_programs and hasattr(item, 'program') and item.program:
            prog_features = extract_program_features(item.program)
            if prog_features is not None:
                self._using_program_features = True
                
                # Improvement #2: Add sequence matching features
                if include_sequence_features and self._use_sequence_matching:
                    seq_features_pos = np.zeros(6, dtype=np.float32)
                    seq_features_neg = np.zeros(6, dtype=np.float32)
                    
                    if self._consensus_pos_program is not None:
                        seq_features_pos = compute_sequence_features(item.program, self._consensus_pos_program)
                    
                    if self._consensus_neg_program is not None:
                        seq_features_neg = compute_sequence_features(item.program, self._consensus_neg_program)
                    
                    # Concatenate: [symbolic(16) + seq_vs_pos(6) + seq_vs_neg(6)] = 28-dim
                    return np.concatenate([prog_features, seq_features_pos, seq_features_neg])
                
                return prog_features
        
        # Fallback to image features
        return extract_image_features(item.image_path, self.image_size)

    def fit(self, support_pos: Iterable[ExampleItem], support_neg: Iterable[ExampleItem]) -> None:
        # Improvement #2: Store consensus programs for sequence matching
        support_pos_list = list(support_pos)
        support_neg_list = list(support_neg)
        
        if self.use_programs and self._use_sequence_matching:
            # Find consensus programs from support sets
            pos_programs = [item.program for item in support_pos_list if hasattr(item, 'program') and item.program]
            neg_programs = [item.program for item in support_neg_list if hasattr(item, 'program') and item.program]

            self._consensus_pos_program = find_consensus_program(pos_programs)
            self._consensus_neg_program = find_consensus_program(neg_programs)
        
        # Extract features (programs if available, else pixels)
        # IMPORTANT: fit and predict must use the same feature space.
        x_pos = np.asarray(
            [self._extract_features(i, include_sequence_features=self._use_sequence_matching) for i in support_pos_list],
            dtype=np.float32,
        )
        x_neg = np.asarray(
            [self._extract_features(i, include_sequence_features=self._use_sequence_matching) for i in support_neg_list],
            dtype=np.float32,
        )

        self._mu[1], self._var[1] = self._bayes_diag_stats(x_pos)
        self._mu[0], self._var[0] = self._bayes_diag_stats(x_neg)

        n_pos, n_neg = max(len(x_pos), 1), max(len(x_neg), 1)
        n_total = n_pos + n_neg
        self._prior[1] = n_pos / n_total
        self._prior[0] = n_neg / n_total

        self._rules = self._fit_rules(x_pos, x_neg)

    def predict_proba(self, item: ExampleItem) -> Tuple[float, dict]:
        # Improvement #2: Use sequence features during prediction
        x = self._extract_features(item, include_sequence_features=self._use_sequence_matching)
        
        if self._rules:
            logp_pos = math.log(self._prior[1] + 1e-12)
            logp_neg = math.log(self._prior[0] + 1e-12)

            for rule in self._rules:
                obs = 1.0 if rule.fires(x) else 0.0
                p_pos = rule.theta_pos if obs > 0.5 else (1.0 - rule.theta_pos)
                p_neg = rule.theta_neg if obs > 0.5 else (1.0 - rule.theta_neg)
                logp_pos += rule.weight * math.log(max(p_pos, 1e-12))
                logp_neg += rule.weight * math.log(max(p_neg, 1e-12))
        else:
            logp_pos = np.log(self._prior[1] + 1e-12) + self._log_diag_gaussian(x, self._mu[1], self._var[1])
            logp_neg = np.log(self._prior[0] + 1e-12) + self._log_diag_gaussian(x, self._mu[0], self._var[0])

        p_pos = 1.0 / (1.0 + np.exp(logp_neg - logp_pos))
        top_rule = self._rules[0].name if self._rules else "gaussian_fallback"
        feature_type = "program" if self._using_program_features else "image"
        trace = {
            "logp_pos": float(logp_pos),
            "logp_neg": float(logp_neg),
            "logp_margin": float(logp_pos - logp_neg),
            "rule_count": int(len(self._rules)),
            "top_rule": top_rule,
            "feature_type": feature_type,
        }
        return float(p_pos), trace

    def _fit_rules(self, x_pos: np.ndarray, x_neg: np.ndarray) -> List["RuleHypothesis"]:
        d = x_pos.shape[1]
        alpha = self.beta_prior
        primitive_candidates: List[RuleHypothesis] = []

        mean_pos = x_pos.mean(axis=0)
        mean_neg = x_neg.mean(axis=0)

        for j in range(d):
            direction = 1 if mean_pos[j] >= mean_neg[j] else -1
            threshold = 0.5 * float(mean_pos[j] + mean_neg[j])

            pos_fire = self._unary_fire(x_pos[:, j], threshold, direction)
            neg_fire = self._unary_fire(x_neg[:, j], threshold, direction)
            primitive_candidates.append(self._make_rule(
                name=f"f{j}{'>=' if direction > 0 else '<='}{threshold:.3f}",
                kind="unary",
                f1=j,
                f2=-1,
                threshold=threshold,
                direction=direction,
                pos_fire=pos_fire,
                neg_fire=neg_fire,
                alpha=alpha,
            ))

        for j in range(d):
            for k in range(j + 1, d):
                diff_pos = x_pos[:, j] - x_pos[:, k]
                diff_neg = x_neg[:, j] - x_neg[:, k]
                mean_diff_pos = float(diff_pos.mean())
                mean_diff_neg = float(diff_neg.mean())
                direction = 1 if mean_diff_pos >= mean_diff_neg else -1
                threshold = 0.5 * (mean_diff_pos + mean_diff_neg)

                pos_fire = self._unary_fire(diff_pos, threshold, direction)
                neg_fire = self._unary_fire(diff_neg, threshold, direction)
                primitive_candidates.append(self._make_rule(
                    name=f"(f{j}-f{k}){'>=' if direction > 0 else '<='}{threshold:.3f}",
                    kind="pair",
                    f1=j,
                    f2=k,
                    threshold=threshold,
                    direction=direction,
                    pos_fire=pos_fire,
                    neg_fire=neg_fire,
                    alpha=alpha,
                ))

        primitive_candidates.sort(key=lambda r: (r.discriminative_score, r.support_accuracy), reverse=True)
        top_primitives = primitive_candidates[: min(self.max_primitive_rules, len(primitive_candidates))]

        composite_candidates: List[RuleHypothesis] = []
        for i in range(len(top_primitives)):
            for j in range(i + 1, len(top_primitives)):
                left = top_primitives[i]
                right = top_primitives[j]

                left_pos = self._rule_fire_batch(left, x_pos)
                left_neg = self._rule_fire_batch(left, x_neg)
                right_pos = self._rule_fire_batch(right, x_pos)
                right_neg = self._rule_fire_batch(right, x_neg)

                and_pos = np.minimum(left_pos, right_pos)
                and_neg = np.minimum(left_neg, right_neg)
                composite_candidates.append(
                    self._make_composite_rule(
                        name=f"({left.name} AND {right.name})",
                        op="and",
                        left=left,
                        right=right,
                        pos_fire=and_pos,
                        neg_fire=and_neg,
                        alpha=alpha,
                    )
                )

                or_pos = np.maximum(left_pos, right_pos)
                or_neg = np.maximum(left_neg, right_neg)
                composite_candidates.append(
                    self._make_composite_rule(
                        name=f"({left.name} OR {right.name})",
                        op="or",
                        left=left,
                        right=right,
                        pos_fire=or_pos,
                        neg_fire=or_neg,
                        alpha=alpha,
                    )
                )

        all_candidates = top_primitives + composite_candidates
        all_candidates.sort(key=lambda r: (r.discriminative_score, r.support_accuracy), reverse=True)
        selected = all_candidates[: self.max_rules]
        
        if selected:
            total = sum(max(r.discriminative_score, 1e-6) for r in selected)
            for r in selected:
                r.weight = max(r.discriminative_score, 1e-6) / total
        return selected

    @staticmethod
    def _rule_fire_batch(rule: "RuleHypothesis", x: np.ndarray) -> np.ndarray:
        if rule.op == "and":
            return np.minimum(
                BayesianImageSystem2._rule_fire_batch(rule.left, x),
                BayesianImageSystem2._rule_fire_batch(rule.right, x),
            )
        if rule.op == "or":
            return np.maximum(
                BayesianImageSystem2._rule_fire_batch(rule.left, x),
                BayesianImageSystem2._rule_fire_batch(rule.right, x),
            )

        if rule.kind == "pair":
            values = x[:, rule.f1] - x[:, rule.f2]
        else:
            values = x[:, rule.f1]

        if rule.direction > 0:
            return (values >= rule.threshold).astype(np.float32)
        return (values <= rule.threshold).astype(np.float32)

    @staticmethod
    def _unary_fire(values: np.ndarray, threshold: float, direction: int) -> np.ndarray:
        if direction > 0:
            return (values >= threshold).astype(np.float32)
        return (values <= threshold).astype(np.float32)

    @staticmethod
    def _make_rule(
        name: str,
        kind: str,
        f1: int,
        f2: int,
        threshold: float,
        direction: int,
        pos_fire: np.ndarray,
        neg_fire: np.ndarray,
        alpha: float,
    ) -> "RuleHypothesis":
        n_pos = max(len(pos_fire), 1)
        n_neg = max(len(neg_fire), 1)

        pos_count = float(pos_fire.sum())
        neg_count = float(neg_fire.sum())

        theta_pos = (pos_count + alpha) / (n_pos + 2.0 * alpha)
        theta_neg = (neg_count + alpha) / (n_neg + 2.0 * alpha)

        pos_acc = np.maximum(pos_fire, 1.0 - pos_fire).mean()
        neg_acc = np.maximum(neg_fire, 1.0 - neg_fire).mean()
        support_accuracy = float(0.5 * (pos_acc + neg_acc))

        discriminative_score = abs(theta_pos - theta_neg)
        return RuleHypothesis(
            name=name,
            op="atom",
            kind=kind,
            f1=f1,
            f2=f2,
            threshold=float(threshold),
            direction=int(direction),
            theta_pos=float(theta_pos),
            theta_neg=float(theta_neg),
            support_accuracy=float(support_accuracy),
            discriminative_score=float(discriminative_score),
            weight=0.0,
            left=None,
            right=None,
        )

    @staticmethod
    def _make_composite_rule(
        name: str,
        op: str,
        left: "RuleHypothesis",
        right: "RuleHypothesis",
        pos_fire: np.ndarray,
        neg_fire: np.ndarray,
        alpha: float,
    ) -> "RuleHypothesis":
        n_pos = max(len(pos_fire), 1)
        n_neg = max(len(neg_fire), 1)

        pos_count = float(pos_fire.sum())
        neg_count = float(neg_fire.sum())

        theta_pos = (pos_count + alpha) / (n_pos + 2.0 * alpha)
        theta_neg = (neg_count + alpha) / (n_neg + 2.0 * alpha)

        pos_acc = np.maximum(pos_fire, 1.0 - pos_fire).mean()
        neg_acc = np.maximum(neg_fire, 1.0 - neg_fire).mean()
        support_accuracy = float(0.5 * (pos_acc + neg_acc))

        discriminative_score = abs(theta_pos - theta_neg)
        return RuleHypothesis(
            name=name,
            op=op,
            kind="composite",
            f1=-1,
            f2=-1,
            threshold=0.0,
            direction=1,
            theta_pos=float(theta_pos),
            theta_neg=float(theta_neg),
            support_accuracy=float(support_accuracy),
            discriminative_score=float(discriminative_score),
            weight=0.0,
            left=left,
            right=right,
        )

    def _bayes_diag_stats(self, x: np.ndarray):
        d = x.shape[1]
        prior_mu = np.zeros(d, dtype=np.float32)
        prior_var = np.ones(d, dtype=np.float32)

        sample_mu = x.mean(axis=0)
        if len(x) > 1:
            sample_var = x.var(axis=0)
        else:
            sample_var = np.ones(d, dtype=np.float32)

        k0 = self.prior_strength
        n = float(len(x))
        mu_post = (k0 * prior_mu + n * sample_mu) / (k0 + n)
        var_post = ((k0 * prior_var) + (n * sample_var)) / (k0 + n)
        var_post = np.maximum(var_post, self.var_floor)
        return mu_post, var_post

    @staticmethod
    def _log_diag_gaussian(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
        return float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x - mu) ** 2) / var))


@dataclass
class RuleHypothesis:
    name: str
    op: str
    kind: str
    f1: int
    f2: int
    threshold: float
    direction: int
    theta_pos: float
    theta_neg: float
    support_accuracy: float
    discriminative_score: float
    weight: float
    left: Optional["RuleHypothesis"]
    right: Optional["RuleHypothesis"]

    def fires(self, x: np.ndarray) -> bool:
        if self.op == "and":
            return self.left.fires(x) and self.right.fires(x)
        if self.op == "or":
            return self.left.fires(x) or self.right.fires(x)

        if self.kind == "pair":
            value = float(x[self.f1] - x[self.f2])
        else:
            value = float(x[self.f1])

        if self.direction > 0:
            return value >= self.threshold
        return value <= self.threshold
