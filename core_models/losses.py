# Folder: bongard_solver/core_models/
# File: losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import random  # For dummy SymbolicEngine
import json # For parsing ground truth scene graphs

logger = logging.getLogger(__name__)

# Import SymbolicEngine and BongardRule for SymbolicConsistencyLoss
HAS_SYMBOLIC_ENGINE = False
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
            if isinstance(rule, BongardRule) and rule.name.startswith("dummy_pos"): return True
            if isinstance(rule, BongardRule) and rule.name.startswith("dummy_neg"): return False
            return random.random() > 0.5  # Random for other rules
    class BongardRule:
        def __init__(self, name: str = "dummy_rule", description: str = "A dummy rule", program_ast: List = [], logical_facts: List = [], is_positive_rule: bool = True):
            self.name = name
            self.description = description
            self.program_ast = program_ast
            self.logical_facts = logical_facts
            self.is_positive_rule = is_positive_rule

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
        self.initial_losses = {name: None for name in self.task_names} # Store initial loss values
        logger.info(f"GradNorm initialized with tasks: {self.task_names}, alpha={self.alpha}")

    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates weighted losses. This method is called in the forward pass to get weighted losses.
        Args:
            losses (Dict[str, torch.Tensor]): A dictionary of scalar loss values for each task.
        Returns:
            Dict[str, torch.Tensor]: A dictionary of dynamically weighted loss values.
        """
        if not all(name in losses for name in self.task_names):
            raise ValueError(f"GradNorm: Missing losses for some tasks. Expected: {self.task_names}, Got: {list(losses.keys())}")
        
        weights = torch.exp(self.log_loss_weights)
        
        # Initialize initial_losses on the first forward pass if they are None
        if any(self.initial_losses[name] is None for name in self.task_names):
            with torch.no_grad():
                for i, name in enumerate(self.task_names):
                    if self.initial_losses[name] is None:
                        self.initial_losses[name] = losses[name].item() # Store scalar value
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
        
        L_hat = {} # Inverse training rates
        for name in self.task_names:
            if self.initial_losses[name] != 0:
                L_hat[name] = losses[name] / self.initial_losses[name]
            else:
                L_hat[name] = losses[name] # Avoid division by zero, use current loss directly if initial was zero
        
        # Average inverse training rate
        avg_L_hat = sum(L_hat.values()) / len(self.task_names)
        
        grad_norms = []
        # Calculate gradient norm for each task's loss with respect to shared parameters
        for i, name in enumerate(self.task_names):
            # Ensure retain_graph=True if you need to compute gradients multiple times
            # for the same graph (e.g., for each task's loss).
            grads = torch.autograd.grad(
                losses[name],
                shared_parameters,
                retain_graph=True, # Keep graph for subsequent grad computations
                allow_unused=True # Allow if some parameters don't receive gradients from this loss
            )
            # Filter out None gradients and concatenate
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads if g is not None]))
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        G_bar_t = grad_norms.mean() # Average gradient norm
        
        if G_bar_t == 0:
            logger.warning("GradNorm: Average gradient norm is zero. Skipping weight update.")
            return
        
        # Calculate target gradient norms
        target_grad_norms = G_bar_t * (torch.stack([L_hat[name] for name in self.task_names]) / avg_L_hat).pow(self.alpha)
        
        # Loss for updating weights (L1 loss between actual and target gradient norms)
        grad_norm_loss = F.l1_loss(grad_norms, target_grad_norms, reduction='mean')
        
        # Backward pass for the GradNorm loss to update log_loss_weights
        self.zero_grad() # Zero gradients for GradNorm parameters
        grad_norm_loss.backward()
        
        # Manual update of log_loss_weights (can also use an optimizer for self.log_loss_weights)
        with torch.no_grad():
            # Normalize gradient to prevent exploding/vanishing updates
            if self.log_loss_weights.grad is not None:
                self.log_loss_weights.grad /= (self.log_loss_weights.grad.norm() + 1e-8) # Add epsilon for stability
                self.log_loss_weights.data.add_(-self.log_loss_weights.grad * 0.001) # Small learning rate for weights
        
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
        
        # NLL loss for the true labels
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        # Loss for the smoothed part (uniform distribution)
        smooth_loss = -logprobs.mean(dim=-1)
        
        # Combine NLL and smooth loss
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss # Return per-sample loss
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
            # Cosine similarity loss: 1 - cosine_similarity (0 for identical, 2 for opposite)
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
        # Handle empty tensors gracefully
        if features1.numel() == 0 or features2.numel() == 0:
            logger.debug("FeatureConsistencyLoss: One or both feature tensors are empty. Returning 0 loss.")
            return torch.tensor(0.0, device=features1.device)
        
        # Ensure shapes match for element-wise comparison
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
        self.all_bongard_rules = all_bongard_rules # Dictionary of all possible Bongard rules
        self.loss_weight = loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean') # Use BCEWithLogitsLoss for binary consistency
        
        self.symbolic_engine = symbolic_engine
        # If SymbolicEngine is not provided, try to initialize it
        if self.symbolic_engine is None and HAS_SYMBOLIC_ENGINE and config:
            try:
                self.symbolic_engine = SymbolicEngine(config) # Pass config to SymbolicEngine
                logger.info("SymbolicConsistencyLoss: SymbolicEngine initialized.")
            except Exception as e:
                logger.error(f"SymbolicConsistencyLoss: Failed to initialize SymbolicEngine: {e}. Symbolic consistency will be limited.")
        elif not HAS_SYMBOLIC_ENGINE:
            logger.warning("SymbolicConsistencyLoss: SymbolicEngine not available. Symbolic consistency will be a dummy loss.")
        
        logger.info(f"SymbolicConsistencyLoss initialized with weight: {loss_weight}")

    def forward(self,
                scene_graphs1: List[Dict[str, Any]],
                scene_graphs2: List[Dict[str, Any]],
                labels: torch.Tensor, # Ground truth labels for the Bongard problems (0 or 1)
                ground_truth_scene_graphs: List[bytes] # List of ground truth scene graph JSON strings (bytes)
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
        
        # In a real scenario, you'd get the 'hypothesized_rule' from the symbolic engine's
        # current best rule or from a rule proposed by the ILP/RL module.
        # For this loss, we need *a* rule to evaluate consistency against.
        # Let's assume for now we pick a 'ground truth' rule if available, or a dummy.
        
        # This part is crucial: how do you get the 'hypothesized_rule'?
        # For training, if you have ground truth rules, you can use them.
        # Or, if you are training the symbolic engine, it might propose rules.
        
        # For demonstration, let's try to get a rule from the ground truth scene graphs
        # or use a dummy.
        hypothesized_rule = None
        if self.all_bongard_rules:
            # Attempt to find a rule that matches the positive examples
            # This is a very simplified heuristic for demonstration.
            # In practice, this would involve a more sophisticated rule induction from GT.
            for rule_name, rule_obj in self.all_bongard_rules.items():
                # For simplicity, assume the first positive rule is the one we "hypothesize"
                # This needs to be smarter, e.g., based on the current problem's GT rule.
                if rule_obj.is_positive_rule:
                    hypothesized_rule = rule_obj
                    break
            if hypothesized_rule is None: # Fallback if no positive rule found in ALL_BONGARD_RULES
                hypothesized_rule = BongardRule(name="dummy_pos_rule", description="Dummy positive rule")
                logger.warning("SymbolicConsistencyLoss: No positive rule found in ALL_BONGARD_RULES. Using dummy positive rule.")
        else:
            hypothesized_rule = BongardRule(name="dummy_rule", description="Generic dummy rule")
            logger.warning("SymbolicConsistencyLoss: all_bongard_rules is empty. Using generic dummy rule.")

        if hypothesized_rule is None:
            logger.warning("SymbolicConsistencyLoss: No hypothesized rule available. Returning 0.0.")
            return torch.tensor(0.0, device=labels.device)

        for i in range(len(labels)):
            gt_label = labels[i].item() # 0 for negative, 1 for positive
            inferred_sg_view1 = scene_graphs1[i]
            inferred_sg_view2 = scene_graphs2[i]
            
            # Evaluate consistency of inferred scene graphs with the hypothesized rule
            # The `evaluate_scene_graph_against_rule` returns a boolean (True/False)
            is_consistent_view1 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view1, hypothesized_rule)
            is_consistent_view2 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view2, hypothesized_rule)
            
            # Target consistency: 1.0 if problem is positive, 0.0 if negative
            target_consistency_view1 = 1.0 if gt_label == 1 else 0.0
            target_consistency_view2 = 1.0 if gt_label == 1 else 0.0
            
            # Convert boolean consistency to a float tensor
            predicted_consistency_view1 = torch.tensor(float(is_consistent_view1), device=labels.device).unsqueeze(0)
            predicted_consistency_view2 = torch.tensor(float(is_consistent_view2), device=labels.device).unsqueeze(0)
            
            # Convert to logits for BCEWithLogitsLoss (e.g., scale 0-1 to -inf to +inf)
            # A simple scaling: 0.0 -> -10.0, 1.0 -> +10.0 (or higher for sharper gradient)
            logit_view1 = predicted_consistency_view1 * 20.0 - 10.0 # Maps 0 to -10, 1 to 10
            logit_view2 = predicted_consistency_view2 * 20.0 - 10.0
            
            target_tensor_view1 = torch.tensor(target_consistency_view1, device=labels.device).unsqueeze(0)
            target_tensor_view2 = torch.tensor(target_consistency_view2, device=labels.device).unsqueeze(0)
            
            # Calculate BCE loss for each view
            loss_view1 = self.bce_loss(logit_view1, target_tensor_view1)
            loss_view2 = self.bce_loss(logit_view2, target_tensor_view2)
            
            total_symbolic_loss += (loss_view1 + loss_view2) / 2.0 # Average loss for the problem
            num_evaluated_problems += 1
        
        if num_evaluated_problems > 0:
            return total_symbolic_loss / num_evaluated_problems * self.loss_weight
        else:
            return torch.tensor(0.0, device=labels.device) # Return 0 if no problems were evaluated

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining soft targets from teacher and hard targets from ground truth.
    Modified to support per-sample reduction.
    """
    def __init__(self, temperature: float, alpha: float, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha # Weight for soft targets vs. hard targets
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction) # KLDivLoss for soft targets
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
        # Scale logits by temperature and apply softmax/log_softmax
        soft_targets_loss = self.kl_div_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature * self.temperature) # Scale by T^2 as per original paper
        
        # Hard targets loss (Cross-Entropy)
        hard_targets_loss = F.cross_entropy(student_logits, target_labels, reduction=self.reduction)
        
        if self.reduction == 'none':
            return soft_targets_loss, hard_targets_loss # Return per-sample losses
        else:
            # Combine soft and hard losses
            combined_loss = self.alpha * soft_targets_loss + (1. - self.alpha) * hard_targets_loss
            return combined_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # Weighting factor for positive/negative classes
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits (N, C).
            targets (torch.Tensor): Ground truth labels (N).
        Returns:
            torch.Tensor: Scalar loss.
        """
        # Calculate standard Cross-Entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)
        
        # Calculate focal term
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            # Create a tensor of alpha values based on target labels
            alpha_tensor = torch.full_like(targets, self.alpha, dtype=torch.float32)
            alpha_tensor[targets == 0] = (1 - self.alpha) # Assuming binary classification (0 or 1)
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
        
        # Normalize embeddings to unit sphere
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate z_i and z_j to form a 2N batch
        representations = torch.cat([z_i, z_j], dim=0)    # (2N, D)
        
        # Compute similarity matrix for the 2N batch
        # (2N, D) @ (D, 2N) -> (2N, 2N)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Mask out self-similarities (diagonal elements)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf')) # Set diagonal to -inf
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create labels for contrastive loss
        # The positive pairs are (i, j) and (j, i) where i and j are augmented views of the same image.
        # For a batch of N samples, the i-th sample in z_i corresponds to the (i+N)-th sample in representations.
        # So, the positive pair for index `k` (from 0 to N-1) in `z_i` is `k+N` in `representations`.
        # The positive pair for index `k` (from 0 to N-1) in `z_j` is `k` in `representations`.
        labels = torch.cat([torch.arange(batch_size, device=similarity_matrix.device) + batch_size, # Labels for z_i (positives are in z_j part)
                            torch.arange(batch_size, device=similarity_matrix.device)], dim=0) # Labels for z_j (positives are in z_i part)
        
        # Compute Cross-Entropy Loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

class ContrastiveLoss(nn.Module):
    """
    Standard Contrastive Loss (e.g., for Siamese networks).
    Penalizes dissimilar pairs within margin, and similar pairs if distance is too high.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output1 (torch.Tensor): Features from the first input (N, D).
            output2 (torch.Tensor): Features from the second input (N, D).
            label (torch.Tensor): Binary label (N,) where 1 indicates similar, 0 indicates dissimilar.
        Returns:
            torch.Tensor: Scalar loss.
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    Ensures that an anchor is closer to a positive than to a negative by a margin.
    """
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__()
        self.margin = margin
        self.p = p # Power for Minkowski distance

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor (torch.Tensor): Anchor features (N, D).
            positive (torch.Tensor): Positive features (N, D).
            negative (torch.Tensor): Negative features (N, D).
        Returns:
            torch.Tensor: Scalar loss.
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=self.p)
        distance_negative = F.pairwise_distance(anchor, negative, p=self.p)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CrossEntropyWithConf(nn.Module):
    """
    Cross-Entropy Loss that can incorporate a confidence score for each sample.
    Higher confidence samples contribute more to the loss.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # Always compute per-sample first

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, confidences: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits (N, C).
            targets (torch.Tensor): Ground truth labels (N).
            confidences (Optional[torch.Tensor]): Confidence scores for each sample (N,).
                                                  If None, behaves like standard CrossEntropyLoss.
        Returns:
            torch.Tensor: Scalar loss.
        """
        per_sample_loss = self.ce_loss(inputs, targets)
        
        if confidences is not None:
            if confidences.shape != per_sample_loss.shape:
                raise ValueError(f"Confidence tensor shape {confidences.shape} must match per-sample loss shape {per_sample_loss.shape}.")
            weighted_loss = per_sample_loss * confidences
        else:
            weighted_loss = per_sample_loss
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        elif self.reduction == 'none':
            return weighted_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")

class MixupCutmix(nn.Module):
    """
    Dummy Mixup/Cutmix module. This is typically implemented as a data augmentation
    strategy within the DataLoader or as a transform, not a loss function.
    However, if it modifies labels to soft targets, it might be used in conjunction
    with a loss.
    
    This class is primarily a placeholder to acknowledge the request.
    The actual implementation for Mixup/Cutmix is usually in `training.py` or `data.py`.
    """
    def __init__(self):
        super().__init__()
        logger.warning("MixupCutmix class is a placeholder. Actual Mixup/Cutmix logic should be implemented in data augmentation.")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This forward method would typically apply Mixup/Cutmix to inputs and targets.
        For a dummy, it just returns them unchanged.
        """
        logger.debug("Dummy MixupCutmix forward pass. Returning inputs and targets unchanged.")
        return inputs, targets

