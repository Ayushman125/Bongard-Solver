# Folder: bongard_solver/

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

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
        # Ensure features have the same number of elements
        if features1.numel() == 0 or features2.numel() == 0:
            return torch.tensor(0.0, device=features1.device)
        
        # If the number of objects detected in each view is different,
        # we need a more sophisticated matching (e.g., Hungarian algorithm based on IoU).
        # For now, we assume a 1:1 correspondence for simplicity,
        # or that the batch_size for these features is the total number of objects,
        # and we're comparing corresponding objects.
        # A more robust implementation would match objects by ID or spatial proximity.
        
        # For now, assuming features1 and features2 correspond to the same set of objects,
        # which would be true if they come from the same batch of images (e.g., from two views of the same problem).
        # If the number of objects is different, we need to handle it.
        # Let's assume the batching ensures `features1.shape[0]` == `features2.shape[0]`
        # if they correspond to the same set of detected objects.
        
        # If shapes are different, we need to decide how to handle.
        # For now, if shapes don't match, return 0 loss.
        if features1.shape != features2.shape:
            logger.warning(f"FeatureConsistencyLoss: Feature shapes mismatch ({features1.shape} vs {features2.shape}). Returning 0 loss.")
            return torch.tensor(0.0, device=features1.device)


        return self.loss_fn(features1, features2)


class SymbolicConsistencyLoss(nn.Module):
    """
    Encourages consistency between inferred scene graphs and ground truth rules.
    This loss is conceptual and would involve comparing inferred scene graphs against
    the hypothesized rules for the problem.
    """
    def __init__(self, all_bongard_rules: Dict[str, Any], loss_weight: float = 1.0):
        super().__init__()
        self.all_bongard_rules = all_bongard_rules
        self.loss_weight = loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        logger.info(f"SymbolicConsistencyLoss initialized with weight: {loss_weight}")

    def forward(self,
                scene_graphs1: List[Dict[str, Any]],
                scene_graphs2: List[Dict[str, Any]],
                labels: torch.Tensor, # Ground truth labels for Bongard problem (0 or 1)
                hypothesized_rules_info: List[Tuple[Any, float]] # List of (BongardRule, score)
                ) -> torch.Tensor:
        """
        Calculates a symbolic consistency loss.
        This is a placeholder and needs a sophisticated symbolic reasoning engine
        to evaluate inferred scene graphs against rules.

        Args:
            scene_graphs1 (List[Dict[str, Any]]): Inferred scene graphs for view 1.
            scene_graphs2 (List[Dict[str, Any]]): Inferred scene graphs for view 2.
            labels (torch.Tensor): Ground truth labels for the Bongard problems (0 or 1).
            hypothesized_rules_info (List[Tuple[Any, float]]): List of (BongardRule, score)
                                                                representing the best rules found for the batch.
        Returns:
            torch.Tensor: Scalar symbolic consistency loss.
        """
        if not hypothesized_rules_info:
            return torch.tensor(0.0, device=labels.device) # No rules to evaluate against

        total_symbolic_loss = torch.tensor(0.0, device=labels.device)
        num_evaluated_problems = 0

        # Iterate through each problem in the batch
        # We assume scene_graphs1, scene_graphs2, labels are aligned per problem.
        for i in range(len(labels)):
            gt_label = labels[i].item()
            inferred_sg_view1 = scene_graphs1[i]
            inferred_sg_view2 = scene_graphs2[i]

            # For each problem, evaluate how well the inferred scene graphs
            # match the hypothesized rule.
            # This is a simplified logic. A real symbolic engine would be needed.
            
            # Example: Check if the inferred scene graph of view 1 has objects,
            # and if the rule is 'all_circles', check if all objects in inferred_sg_view1
            # are predicted as circles.
            
            # This part needs actual symbolic reasoning logic.
            # For a placeholder, let's just make it a dummy loss that encourages
            # the inferred SGs to be "consistent" with the GT label given some rule.
            
            # This loss is best implemented when the symbolic engine is fully functional.
            # For now, it will return 0.0 to avoid errors.
            
            # This loss is meant to guide the perception module to produce scene graphs
            # that are "interpretable" and "consistent" with the underlying rules.
            
            pass # Placeholder for actual symbolic consistency logic

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
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets) # Assuming binary or one-hot targets
            focal_loss = alpha_t * focal_loss

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

        # Compute cosine similarity between all pairs
        # (N, D) @ (D, N) -> (N, N)
        sim_matrix = torch.matmul(z_i, z_j.T) # Positive pairs are on the diagonal

        # Combine positive and negative pairs
        # The diagonal elements are positive pairs (z_i[k] vs z_j[k])
        # All other elements are negative pairs
        
        # Create labels for contrastive loss
        # The positive pair for z_i[k] is z_j[k] (index k)
        # The positive pair for z_j[k] is z_i[k] (index k)
        
        # We need to create a similarity matrix for (z_i, z_j) and (z_j, z_i)
        # Concatenate z_i and z_j to form a 2N batch
        representations = torch.cat([z_i, z_j], dim=0) # (2N, D)
        
        # Compute similarity matrix for the 2N batch
        similarity_matrix = torch.matmul(representations, representations.T) # (2N, 2N)
        
        # Mask out self-similarity (diagonal elements)
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf')) # Set self-similarity to -inf
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Labels: For each row `i`, the positive sample is `i+N` if `i < N`, or `i-N` if `i >= N`.
        # This means the positive pairs are (0, N), (1, N+1), ..., (N-1, 2N-1) and vice versa.
        # So, for row `k`, the positive index is `k+N` if `k < N`, else `k-N`.
        
        # Create `labels` tensor for CrossEntropyLoss
        # For batch_size N, we have 2N samples.
        # The positive pair for sample `i` is `i + N` if `i < N`, and `i - N` if `i >= N`.
        # The target for row `i` in the similarity matrix is the column index of its positive pair.
        labels = torch.cat([torch.arange(batch_size, device=similarity_matrix.device) + batch_size,
                            torch.arange(batch_size, device=similarity_matrix.device)], dim=0)
        
        # Compute loss
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

