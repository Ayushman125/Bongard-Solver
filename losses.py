# Folder: bongard_solver/
# File: losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Import SymbolicEngine and BongardRule for SymbolicConsistencyLoss
try:
    from symbolic_engine import SymbolicEngine
    from bongard_rules import BongardRule
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
            # In a real scenario, this would apply the rule's logic to the scene graph
            if rule.name.startswith("dummy_pos"): return True
            if rule.name.startswith("dummy_neg"): return False
            return random.random() > 0.5 # Random for other rules
    class BongardRule:
        def __init__(self, name: str = "dummy_rule"): self.name = name

# --- Loss Functions ---
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
            return loss  # Return per-sample loss
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
        
        # If the number of objects detected in each view is different,
        # we need a more sophisticated matching (e.g., Hungarian algorithm based on IoU).
        # For now, if shapes don't match, return 0 loss.
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
    def __init__(self, all_bongard_rules: Dict[str, Any], loss_weight: float = 1.0, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.all_bongard_rules = all_bongard_rules
        self.loss_weight = loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean') # For binary consistency prediction
        
        self.symbolic_engine = None
        if HAS_SYMBOLIC_ENGINE and config:
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
                labels: torch.Tensor,  # Ground truth labels for Bongard problem (0 or 1)
                hypothesized_rules_info: List[Tuple[Any, float]]  # List of (BongardRule, score)
                ) -> torch.Tensor:
        """
        Calculates a symbolic consistency loss.
        It evaluates the inferred scene graphs against the hypothesized rules
        and the ground truth labels.

        Args:
            scene_graphs1 (List[Dict[str, Any]]): Inferred scene graphs for view 1 (list of dicts).
            scene_graphs2 (List[Dict[str, Any]]): Inferred scene graphs for view 2 (list of dicts).
            labels (torch.Tensor): Ground truth labels for the Bongard problems (0 or 1).
            hypothesized_rules_info (List[Tuple[Any, float]]): List of (BongardRule, score)
                                                                representing the best rules found for the batch.
        Returns:
            torch.Tensor: Scalar symbolic consistency loss.
        """
        if not hypothesized_rules_info or not self.symbolic_engine:
            logger.debug("SymbolicConsistencyLoss: No hypothesized rules or SymbolicEngine not available. Returning 0.0.")
            return torch.tensor(0.0, device=labels.device)

        total_symbolic_loss = torch.tensor(0.0, device=labels.device)
        num_evaluated_problems = 0

        # Get the top hypothesized rule for this batch
        # Assuming hypothesized_rules_info contains (BongardRule_instance, score)
        best_hypothesized_rule: BongardRule = hypothesized_rules_info[0][0]
        
        # Iterate through each problem in the batch
        # We assume scene_graphs1, scene_graphs2, labels are aligned per problem.
        for i in range(len(labels)):
            gt_label = labels[i].item() # 0 or 1
            inferred_sg_view1 = scene_graphs1[i]
            inferred_sg_view2 = scene_graphs2[i]

            # Evaluate each inferred scene graph against the best hypothesized rule
            # The SymbolicEngine returns True if the scene graph satisfies the rule, False otherwise.
            # Convert inferred scene graph masks (list of lists) to numpy arrays for SymbolicEngine if needed
            # (assuming SymbolicEngine can handle JSON-like scene graph dicts directly)

            # Check consistency for View 1
            is_consistent_view1 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view1, best_hypothesized_rule)
            
            # Check consistency for View 2
            is_consistent_view2 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view2, best_hypothesized_rule)

            # Define the target for the consistency loss
            # If gt_label is 1 (positive problem), both views should be consistent with the rule.
            # If gt_label is 0 (negative problem), one view should be consistent, the other inconsistent.
            # This is a simplification; a negative problem means the rule does NOT hold for the pair.
            # For symbolic consistency, we want the *inferred* properties to align with the *ground truth* label
            # given the *hypothesized* rule.

            # Let's define the target for each view:
            # If GT label is 1 (positive problem): target for both views is 1 (consistent)
            # If GT label is 0 (negative problem): target for both views is 0 (inconsistent)
            # This simplifies the logic, assuming the rule itself explains the positive/negative nature.

            target_consistency_view1 = 1.0 if gt_label == 1 else 0.0
            target_consistency_view2 = 1.0 if gt_label == 1 else 0.0

            # Convert boolean consistency to float tensor for loss calculation
            predicted_consistency_view1 = torch.tensor(float(is_consistent_view1), device=labels.device).unsqueeze(0)
            predicted_consistency_view2 = torch.tensor(float(is_consistent_view2), device=labels.device).unsqueeze(0)
            
            # Create a dummy "logit" for BCEWithLogitsLoss.
            # If predicted_consistency is 1.0, logit should be high positive (e.g., 10.0)
            # If predicted_consistency is 0.0, logit should be high negative (e.g., -10.0)
            # This allows BCEWithLogitsLoss to work without needing actual network outputs.
            logit_view1 = predicted_consistency_view1 * 20.0 - 10.0 # Maps 0 to -10, 1 to 10
            logit_view2 = predicted_consistency_view2 * 20.0 - 10.0

            target_tensor_view1 = torch.tensor(target_consistency_view1, device=labels.device).unsqueeze(0)
            target_tensor_view2 = torch.tensor(target_consistency_view2, device=labels.device).unsqueeze(0)

            # Compute BCEWithLogitsLoss for each view
            loss_view1 = self.bce_loss(logit_view1, target_tensor_view1)
            loss_view2 = self.bce_loss(logit_view2, target_tensor_view2)
            
            total_symbolic_loss += (loss_view1 + loss_view2) / 2.0 # Average loss for the problem
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
        # KLDivLoss expects log-probabilities for input and probabilities for target
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
        # Soft targets loss (KL divergence)
        # Scale by T^2 as per distillation paper
        soft_targets_loss = self.kl_div_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature * self.temperature)
        
        # Hard targets loss (standard cross-entropy)
        hard_targets_loss = F.cross_entropy(student_logits, target_labels, reduction=self.reduction)
        
        if self.reduction == 'none':
            return soft_targets_loss, hard_targets_loss
        else:
            # Combined loss
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
            # Assuming targets are class indices for CrossEntropyLoss
            # Convert targets to one-hot for alpha weighting if alpha is per-class
            # For binary classification, alpha can be a scalar or (alpha_pos, alpha_neg)
            # Here, assuming alpha is a scalar weight for the positive class.
            # If targets are class indices, we need to apply alpha based on the target class.
            # This implementation assumes a binary target (0 or 1) for simplicity with alpha.
            # For multi-class, alpha should be a tensor of shape (C,) or a more complex mapping.
            # For now, let's assume alpha applies to the positive class and (1-alpha) to negative.
            # This part might need refinement based on exact use case.
            
            # If alpha is a scalar, apply it to the loss where target is 1
            # and (1-alpha) where target is 0.
            # This requires `targets` to be 0 or 1.
            alpha_tensor = torch.full_like(targets, self.alpha, dtype=torch.float32)
            alpha_tensor[targets == 0] = (1 - self.alpha) # Assuming 0 is negative class
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
        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate z_i and z_j to form a 2N batch
        representations = torch.cat([z_i, z_j], dim=0)  # (2N, D)
        
        # Compute similarity matrix for the 2N batch
        similarity_matrix = torch.matmul(representations, representations.T)  # (2N, 2N)
        
        # Mask out self-similarity (diagonal elements)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))  # Set self-similarity to -inf
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Labels: For each row `i`, the positive sample is `i+N` if `i < N`, or `i-N` if `i >= N`.
        labels = torch.cat([torch.arange(batch_size, device=similarity_matrix.device) + batch_size,
                             torch.arange(batch_size, device=similarity_matrix.device)], dim=0)
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

