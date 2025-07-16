# Folder: bongard_solver/core_models/
# File: losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import random # For dummy SymbolicEngine

logger = logging.getLogger(__name__)

# Import SymbolicEngine and BongardRule for SymbolicConsistencyLoss
try:
    from src.symbolic_engine import SymbolicEngine
    from src.bongard_rules import BongardRule
    HAS_SYMBOLIC_ENGINE = True
    logger.info("SymbolicEngine and BongardRule found for SymbolicConsistencyLoss.")
except ImportError:
    HAS_SYMBOLIC_ENGINE = False
    logger.warning("SymbolicEngine or BongardRule not found. SymbolicConsistencyLoss will be a dummy loss.")
    # Dummy classes for standalone testing
    class SymbolicEngine:
        def __init__(self, *args, **kwargs): pass
        def evaluate_scene_graph_against_rule(self, scene_graph: Dict[str, Any], rule: Any) -> bool:
            # Dummy evaluation: always consistent for positive, inconsistent for negative
            if hasattr(rule, 'name') and rule.name.startswith("dummy_pos"): return True
            if hasattr(rule, 'name') and rule.name.startswith("dummy_neg"): return False
            return random.random() > 0.5 # Random for other rules
    class BongardRule:
        def __init__(self, name: str = "dummy_rule"): self.name = name

# --- Adaptive Loss Weighting (GradNorm) ---
class GradNorm(nn.Module):
    """
    GradNorm: Adaptive Loss Weighting for Multi-Task Learning.
    Dynamically adjusts loss weights to balance gradients from different tasks.
    Reference: https://arxiv.org/abs/1711.02257
    """
    def __init__(self, initial_loss_weights: Dict[str, float], alpha: float = 1.5):
        super().__init__()
        self.log_loss_weights = nn.Parameter(torch.log(torch.tensor([initial_loss_weights[k] for k in initial_loss_weights.keys()], dtype=torch.float32)))
        self.task_names = list(initial_loss_weights.keys())
        self.alpha = alpha
        self.initial_losses = {name: None for name in self.task_names}
        logger.info(f"GradNorm initialized with tasks: {self.task_names}, alpha={self.alpha}")

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates weighted losses and updates the loss weights.
        Args:
            losses (Dict[str, torch.Tensor]): A dictionary of scalar loss values for each task.
        Returns:
            Dict[str, torch.Tensor]: A dictionary of dynamically weighted loss values.
        """
        if not all(name in losses for name in self.task_names):
            raise ValueError(f"GradNorm: Missing losses for some tasks. Expected: {self.task_names}, Got: {list(losses.keys())}")
        
        weights = torch.exp(self.log_loss_weights)
        
        if any(self.initial_losses[name] is None for name in self.task_names):
            with torch.no_grad():
                for i, name in enumerate(self.task_names):
                    if self.initial_losses[name] is None:
                        self.initial_losses[name] = losses[name].item()
                        logger.debug(f"GradNorm: Initial loss for {name}: {self.initial_losses[name]:.4f}")
        
        weighted_losses = {}
        for i, name in enumerate(self.task_names):
            weighted_losses[name] = weights[i] * losses[name]
        
        return weighted_losses

    def update_weights(self, losses: Dict[str, torch.Tensor], shared_parameters: List[torch.Tensor]):
        """
        Computes gradient norms and updates the loss weights.
        This method should be called after the main backward pass of the total loss.
        
        Args:
            losses (Dict[str, torch.Tensor]): A dictionary of scalar loss values for each task.
                                             These should be the *unweighted* losses from the current iteration.
            shared_parameters (List[torch.Tensor]): A list of parameters from the shared layers
                                                   (e.g., backbone) of the model.
        """
        if not self.training:
            logger.warning("GradNorm.update_weights called when not in training mode. Skipping weight update.")
            return
        if not shared_parameters:
            logger.warning("GradNorm.update_weights: No shared parameters provided. Cannot compute gradient norms for weight update.")
            return
        
        if any(self.initial_losses[name] is None for name in self.task_names):
            logger.warning("GradNorm.update_weights: Initial losses not set. Skipping weight update.")
            return
        
        L_hat = {}
        for name in self.task_names:
            if self.initial_losses[name] != 0:
                L_hat[name] = losses[name] / self.initial_losses[name]
            else:
                L_hat[name] = losses[name]
        
        avg_L_hat = sum(L_hat.values()) / len(self.task_names)
        
        grad_norms = []
        for i, name in enumerate(self.task_names):
            grads = torch.autograd.grad(
                losses[name],
                shared_parameters,
                retain_graph=True,
                allow_unused=True
            )
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads if g is not None]))
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        G_bar_t = grad_norms.mean()
        
        if G_bar_t == 0:
            logger.warning("GradNorm: Average gradient norm is zero. Skipping weight update.")
            return
        
        target_grad_norms = G_bar_t * (torch.stack([L_hat[name] for name in self.task_names]) / avg_L_hat).pow(self.alpha)
        
        grad_norm_loss = F.l1_loss(grad_norms, target_grad_norms, reduction='mean')
        
        self.zero_grad()
        grad_norm_loss.backward()
        
        with torch.no_grad():
            self.log_loss_weights.grad /= self.log_loss_weights.grad.norm() + 1e-8
            self.log_loss_weights.data.add_(-self.log_loss_weights.grad * 0.001)
        
        logger.debug(f"GradNorm: Updated weights: {torch.exp(self.log_loss_weights).tolist()}")

# --- Other Loss Functions ---
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Model predictions (logits) (N, C).
            target (torch.Tensor): Ground truth labels (N).
        Returns:
            torch.Tensor: Scalar loss.
        """
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class FeatureConsistencyLoss(nn.Module):
    """
    Measures consistency between feature embeddings of two views of the same problem.
    """
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'cosine':
            self.loss_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
        else:
            raise ValueError(f"Unsupported loss_type for FeatureConsistencyLoss: {loss_type}")
        self.loss_type = loss_type

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features1 (torch.Tensor): Features from view 1 (N, D).
            features2 (torch.Tensor): Features from view 2 (N, D).
        Returns:
            torch.Tensor: Scalar consistency loss.
        """
        if features1.numel() == 0 or features2.numel() == 0:
            return torch.tensor(0.0, device=features1.device)
        
        if features1.shape != features2.shape:
            logger.warning(f"FeatureConsistencyLoss: Feature shapes mismatch ({features1.shape} vs {features2.shape}). Returning 0 loss.")
            return torch.tensor(0.0, device=features1.device)
        return self.loss_fn(features1, features2)

class SymbolicConsistencyLoss(nn.Module):
    """
    Encourages consistency between inferred scene graphs and ground truth rules.
    This loss evaluates how well the inferred scene graphs from two views of a Bongard problem
    align with a hypothesized rule and the problem's ground truth label.
    """
    def __init__(self, all_bongard_rules: Dict[str, Any], loss_weight: float = 1.0, config: Optional[Dict[str, Any]] = None, symbolic_engine: Optional[Any] = None):
        super().__init__()
        self.all_bongard_rules = all_bongard_rules
        self.loss_weight = loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        
        self.symbolic_engine = symbolic_engine
        if self.symbolic_engine is None and HAS_SYMBOLIC_ENGINE and config:
            try:
                self.symbolic_engine = SymbolicEngine(config)
                logger.info("SymbolicConsistencyLoss: SymbolicEngine initialized.")
            except Exception as e:
                logger.error(f"SymbolicConsistencyLoss: Failed to initialize SymbolicEngine: {e}. Symbolic consistency will be limited.")
        elif not HAS_SYMBOLIC_ENGINE:
            logger.warning("SymbolicConsistencyLoss: SymbolicEngine not available. Symbolic consistency will be a dummy loss.")
        
        logger.info(f"SymbolicConsistencyLoss initialized with weight: {loss_weight}")

    def forward(self,
                scene_graphs1: List[Dict[str, Any]],
                scene_graphs2: List[Dict[str, Any]],
                labels: torch.Tensor,
                ground_truth_scene_graphs: List[bytes]
                ) -> torch.Tensor:
        """
        Calculates a symbolic consistency loss.
        It evaluates the inferred scene graphs against the hypothesized rules
        and the ground truth labels.
        Args:
            scene_graphs1 (List[Dict[str, Any]]): Inferred scene graphs for view 1 (list of dicts).
            scene_graphs2 (List[Dict[str, Any]]): Inferred scene graphs for view 2 (list of dicts).
            labels (torch.Tensor): Ground truth labels for the Bongard problems (0 or 1).
            ground_truth_scene_graphs (List[bytes]): List of ground truth scene graph JSON strings (bytes).
                                                     Used for training objectives and rule induction.
        Returns:
            torch.Tensor: Scalar symbolic consistency loss.
        """
        if not self.symbolic_engine:
            logger.debug("SymbolicConsistencyLoss: SymbolicEngine not available. Returning 0.0.")
            return torch.tensor(0.0, device=labels.device)
        
        total_symbolic_loss = torch.tensor(0.0, device=labels.device)
        num_evaluated_problems = 0
        
        hypothesized_rule = None
        if self.all_bongard_rules:
            hypothesized_rule = BongardRule(name="dummy_pos_rule")
        if hypothesized_rule is None:
            logger.warning("SymbolicConsistencyLoss: No hypothesized rule available. Returning 0.0.")
            return torch.tensor(0.0, device=labels.device)

        for i in range(len(labels)):
            gt_label = labels[i].item()
            inferred_sg_view1 = scene_graphs1[i]
            inferred_sg_view2 = scene_graphs2[i]
            
            is_consistent_view1 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view1, hypothesized_rule)
            is_consistent_view2 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view2, hypothesized_rule)
            
            target_consistency_view1 = 1.0 if gt_label == 1 else 0.0
            target_consistency_view2 = 1.0 if gt_label == 1 else 0.0
            
            predicted_consistency_view1 = torch.tensor(float(is_consistent_view1), device=labels.device).unsqueeze(0)
            predicted_consistency_view2 = torch.tensor(float(is_consistent_view2), device=labels.device).unsqueeze(0)
            
            logit_view1 = predicted_consistency_view1 * 20.0 - 10.0
            logit_view2 = predicted_consistency_view2 * 20.0 - 10.0
            target_tensor_view1 = torch.tensor(target_consistency_view1, device=labels.device).unsqueeze(0)
            target_tensor_view2 = torch.tensor(target_consistency_view2, device=labels.device).unsqueeze(0)
            
            loss_view1 = self.bce_loss(logit_view1, target_tensor_view1)
            loss_view2 = self.bce_loss(logit_view2, target_tensor_view2)
            
            total_symbolic_loss += (loss_view1 + loss_view2) / 2.0
            num_evaluated_problems += 1
        
        if num_evaluated_problems > 0:
            return total_symbolic_loss / num_evaluated_problems * self.loss_weight
        else:
            return torch.tensor(0.0, device=labels.device)

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining soft targets from teacher and hard targets from ground truth.
    Modified to support per-sample reduction.
    """
    def __init__(self, temperature: float, alpha: float, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, target_labels: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            student_logits (torch.Tensor): Logits from the student model (N, C).
            teacher_logits (torch.Tensor): Logits from the teacher model (N, C).
            target_labels (torch.Tensor): Ground truth labels (N).
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If reduction is 'batchmean' or 'sum': Scalar combined loss.
            - If reduction is 'none': Tuple of (per_sample_soft_loss, per_sample_hard_loss).
        """
        soft_targets_loss = self.kl_div_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature * self.temperature)
        
        hard_targets_loss = F.cross_entropy(student_logits, target_labels, reduction=self.reduction)
        
        if self.reduction == 'none':
            return soft_targets_loss, hard_targets_loss
        else:
            combined_loss = self.alpha * soft_targets_loss + (1. - self.alpha) * hard_targets_loss
            return combined_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits (N, C).
            targets (torch.Tensor): Ground truth labels (N).
        Returns:
            torch.Tensor: Scalar loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.alpha is not None:
            alpha_tensor = torch.full_like(targets, self.alpha, dtype=torch.float32)
            alpha_tensor[targets == 0] = (1 - self.alpha)
            focal_loss = alpha_tensor * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'none':
            return focal_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i (torch.Tensor): Projected features from view 1 (N, D).
            z_j (torch.Tensor): Projected features from view 2 (N, D).
        Returns:
            torch.Tensor: Scalar NT-Xent loss.
        """
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = torch.matmul(representations, representations.T)
        
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        similarity_matrix = similarity_matrix / self.temperature
        
        labels = torch.cat([torch.arange(batch_size, device=similarity_matrix.device) + batch_size,
                             torch.arange(batch_size, device=similarity_matrix.device)], dim=0)
        
        loss = self.criterion(similarity_matrix, labels)
        
        return loss
