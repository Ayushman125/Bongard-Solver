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
            if hasattr(rule, 'name') and rule.name.startswith("dummy_pos"): return True
            if hasattr(rule, 'name') and rule.name.startswith("dummy_neg"): return False
            return random.random() > 0.5  # Random for other rules
    class BongardRule:
        def __init__(self, name: str = "dummy_rule"):  self .name = name

# --- Adaptive Loss Weighting (GradNorm) ---
class GradNorm(nn.Module):
    """
    GradNorm: Adaptive Loss Weighting for Multi-Task Learning.
    Dynamically adjusts loss weights to balance gradients from different tasks.
    Reference: https://arxiv.org/abs/1711.02257
    """
    def __init__(self, initial_loss_weights: Dict[str, float], alpha: float = 1.5):
        super().__init__()
        # Initial loss weights for each task
        self.log_loss_weights = nn.Parameter(torch.log(torch.tensor([initial_loss_weights[k] for k in initial_loss_weights.keys()], dtype=torch.float32)))
        self.task_names = list(initial_loss_weights.keys())
        self.alpha = alpha # Balancing coefficient

        # Store initial loss values (L_0) for normalization
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
        # Ensure all expected losses are present
        if not all(name in losses for name in self.task_names):
            raise ValueError(f"GradNorm: Missing losses for some tasks. Expected: {self.task_names}, Got: {list(losses.keys())}")

        # Convert log weights to actual weights
        weights = torch.exp(self.log_loss_weights)
        
        # Store initial losses on the first iteration
        if any(self.initial_losses[name] is None for name in self.task_names):
            with torch.no_grad():
                for i, name in enumerate(self.task_names):
                    if self.initial_losses[name] is None:
                        self.initial_losses[name] = losses[name].item() # Use .item() for scalar value
                        logger.debug(f"GradNorm: Initial loss for {name}: {self.initial_losses[name]:.4f}")
        
        # Calculate weighted losses
        weighted_losses = {}
        for i, name in enumerate(self.task_names):
            weighted_losses[name] = weights[i] * losses[name]
        
        # Calculate total loss for backpropagation (sum of weighted losses)
        # This is the value that will be used for the overall optimization step.
        total_loss_for_backward = sum(weighted_losses.values())

        # If not in training mode, or if not tracking gradients, just return weighted losses
        if not self.training:
            return weighted_losses
        
        # Calculate GradNorm only during training and if gradients are enabled
        if total_loss_for_backward.grad_fn is None:
            logger.warning("GradNorm: Total loss has no grad_fn. Skipping gradient-based weight update.")
            return weighted_losses

        # Compute gradients of total loss w.r.t. the last shared layer's parameters
        # This requires `total_loss_for_backward` to be computed *before* this call
        # and its graph to be retained for `autograd.grad`.
        
        # Find a shared layer (e.g., the backbone's last layer or a shared trunk)
        # This part requires knowledge of the model architecture.
        # For a general `GradNorm` class, the user would typically pass
        # the shared parameters or the last shared layer's output.
        # As a placeholder, we'll assume the gradients are computed w.r.t. the weights themselves
        # or a dummy shared parameter if a real one isn't passed.
        
        # A common approach is to compute gradients w.r.t. the output of the *shared backbone*.
        # Since this GradNorm is a standalone loss module, it doesn't have direct access
        # to the model's internal layers.
        # The prompt implies `gn(loss_dict)` is called, and then `loss_total = sum(weighted.values())`
        # is used for backward. This means `GradNorm` itself doesn't do the `backward()`.
        
        # To make GradNorm work, we need to compute gradients of each weighted loss
        # with respect to the *shared parameters* of the network.
        # This is typically done by passing the relevant shared layer's output
        # or parameters to GradNorm, or by calling `autograd.grad` externally.
        
        # Let's modify GradNorm to take `shared_parameters` or `shared_features`
        # during its `forward` call, or assume it's called *after* the main backward pass
        # for the individual losses, and then it adjusts weights for the *next* iteration.
        
        # The prompt shows:
        # loss_dict = {'cls': loss_cls, 'attr': loss_attr, 'rel': loss_rel}
        # weighted = gn(loss_dict)
        # loss_total = sum(weighted.values())
        # This implies `gn(loss_dict)` should return the final weighted losses,
        # and the `backward()` is called on `loss_total`.
        # For GradNorm to work, `autograd.grad` needs to be called on the individual
        # losses with respect to the shared layer's parameters.
        
        # Let's adjust the `GradNorm` class to be more self-contained, assuming
        # it receives the gradients from the overall backward pass.
        # This is tricky because `autograd.grad` needs to be called *before* `optimizer.step()`.
        
        # Standard GradNorm implementation requires:
        # 1. Compute L_i (individual losses)
        # 2. Compute L_total = sum(w_i * L_i)
        # 3. Perform backward pass on L_total to get gradients for shared parameters.
        # 4. In a separate step, calculate gradient magnitudes for each task.
        # 5. Update w_i based on these magnitudes.
        
        # For PyTorch Lightning or manual loop, the `backward()` is usually called on the `total_loss`.
        # GradNorm's weight update is a meta-optimization.
        
        # Let's implement it such that `GradNorm.forward` returns the weighted losses,
        # and the `backward()` is called on their sum.
        # The weight adjustment logic needs access to gradients.
        # This is usually done by hooking into the backward pass or by `autograd.grad`.

        # For simplicity and to match the prompt's usage, `GradNorm` will compute
        # the gradient magnitudes *internally* using `autograd.grad` on the `total_loss_for_backward`.
        # This requires `retain_graph=True` for individual losses if they are part of a larger graph.
        
        # Calculate the L_hat (normalized loss) for each task
        L_hat = {}
        for name in self.task_names:
            if self.initial_losses[name] is not None and self.initial_losses[name] != 0:
                L_hat[name] = losses[name] / self.initial_losses[name]
            else:
                L_hat[name] = losses[name] # Avoid division by zero, or handle as appropriate
        
        # Calculate the average loss ratio
        avg_L_hat = sum(L_hat.values()) / len(self.task_names)

        # Compute gradient magnitudes
        # This requires a shared set of parameters.
        # In a typical model, this would be the parameters of the backbone/encoder.
        # Since GradNorm is a standalone module, it needs a reference to these parameters.
        # The most robust way is to pass the shared parameters to GradNorm's constructor
        # or `forward` method.
        
        # As per the prompt, GradNorm is instantiated with `loss_weights` only.
        # This implies it needs to infer or be provided the shared parameters.
        # A common pattern is to pass the `model` or its shared part to `GradNorm.update_weights`.
        
        # For now, I will add a placeholder for `shared_params` that would be passed.
        # In `training.py`, you would need to pass `model.shared_backbone_params` or similar.
        
        # Dummy shared parameters for `autograd.grad` if not explicitly passed
        # In a real setup, `shared_params` would be `list(model.shared_layer.parameters())`
        # or `model.shared_backbone.parameters()`.
        # For this `losses.py` file, we cannot assume access to the model.
        # So, the `GradNorm` class itself cannot directly compute `autograd.grad`
        # without being given the `shared_parameters`.
        
        # The instruction "In each iteration: weighted = gn(loss_dict)" implies `gn` is called
        # before `backward()`. This means `GradNorm` cannot update weights based on gradients
        # of the *current* `loss_total` because those gradients aren't computed yet.
        # It must update weights for the *next* iteration.
        
        # Let's adjust: `GradNorm` will compute gradients w.r.t. its *own* `log_loss_weights`
        # and then update them. This is a meta-optimizer.
        
        # The GradNorm paper computes gradients of each loss w.r.t. the *shared* layer.
        # This means `autograd.grad` needs to be called on `losses[name]` and `shared_parameters`.
        
        # I will add a `compute_and_update_weights` method to `GradNorm` that takes
        # the individual losses and the shared layer's parameters, and this method
        # would be called *after* `backward()` of the main loss, or within a custom
        # training loop/Lightning hook.
        
        # For the `forward` method as per the prompt, it simply returns weighted losses.
        # The weight update logic needs to be external or in a separate call.
        
        # Let's make `GradNorm` store the individual losses and then have a separate `update_weights` method.
        # This aligns better with how `autograd.grad` is typically used.
        
        # The `forward` method will just apply current weights.
        # The `update_weights` method will calculate gradient norms and adjust `self.log_loss_weights`.
        
        # No change to `forward` based on the prompt's usage.
        # The actual GradNorm update logic will be in `update_weights`.
        
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

        # Ensure initial losses are set
        if any(self.initial_losses[name] is None for name in self.task_names):
            logger.warning("GradNorm.update_weights: Initial losses not set. Skipping weight update.")
            return

        # Calculate L_hat (normalized loss) for each task
        L_hat = {}
        for name in self.task_names:
            if self.initial_losses[name] != 0:
                L_hat[name] = losses[name] / self.initial_losses[name]
            else:
                L_hat[name] = losses[name] # Handle division by zero

        # Calculate the average loss ratio
        avg_L_hat = sum(L_hat.values()) / len(self.task_names)

        # Compute gradient magnitudes for each task
        # We need gradients of each individual loss w.r.t. the shared parameters.
        # `autograd.grad` can compute this.
        
        # Ensure gradients are enabled for shared parameters
        for param in shared_parameters:
            if not param.requires_grad:
                logger.warning("GradNorm: Shared parameter does not require grad. Cannot compute gradient norm.")
                return

        grad_norms = []
        for i, name in enumerate(self.task_names):
            # Compute gradient of the current loss w.r.t. shared parameters
            # `create_graph=True` is needed if we want to backprop through the weight updates
            # (i.e., if GradNorm itself is part of the computation graph for another meta-loss).
            # For standard GradNorm, `create_graph=False` is fine.
            # `retain_graph=True` is crucial if `losses[name]` is part of a larger graph
            # and other `autograd.grad` calls or `backward()` calls will follow.
            
            # If losses are already detached, this will fail.
            # Assume `losses` are still part of the computation graph.
            
            grads = torch.autograd.grad(
                losses[name],
                shared_parameters,
                retain_graph=True, # Retain graph for subsequent `autograd.grad` calls
                allow_unused=True # Allow if a loss doesn't affect all shared params
            )
            # Flatten gradients and compute L2 norm
            # Filter out None gradients if allow_unused=True
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads if g is not None]))
            grad_norms.append(grad_norm)
        
        # Convert to tensor
        grad_norms = torch.stack(grad_norms)

        # Calculate inverse training rates (r_i(t))
        # r_i(t) = G_i(t) / G_bar(t) where G_bar(t) is the average gradient norm
        # G_bar_t = grad_norms.mean() # This is the average of the gradient magnitudes
        
        # The paper uses G_bar(t) as the average of the *initial* gradient norms,
        # or the average of the current gradient norms.
        # Let's use the average of the current gradient norms.
        G_bar_t = grad_norms.mean()
        
        if G_bar_t == 0:
            logger.warning("GradNorm: Average gradient norm is zero. Skipping weight update.")
            return

        # Calculate the relative inverse training rates
        inverse_training_rates = grad_norms / G_bar_t

        # Calculate the constant term C_hat (target for normalized gradient magnitudes)
        # C_hat = (avg_L_hat * weights.sum())**(1/alpha)
        # The paper's formulation for C_hat is more complex, often related to the initial total loss.
        # A simpler interpretation is to target `avg_L_hat` for each task's normalized gradient.
        
        # The paper's target for each task's gradient magnitude is:
        # G_i_target = G_bar_t * (L_hat_i)**alpha / (avg_L_hat)**alpha
        # This is simplified: G_i_target = G_bar_t * (L_hat_i / avg_L_hat)**alpha
        
        # The loss for the weights is:
        # L_grad = sum( |G_i(t) - G_i_target| )
        
        # The paper's actual loss for weights is:
        # L_grad = sum( |G_i(t) - G_bar(t) * (L_hat_i / avg_L_hat)**alpha | )
        # And the gradients are taken w.r.t. `log_loss_weights`.
        
        # For simplicity, we can define a target for each normalized gradient
        # and then compute a loss on `log_loss_weights` to move towards this target.
        
        # The actual update rule for `log_loss_weights` is usually:
        # w_i = w_i * (G_bar_t / G_i(t))**beta * (L_hat_i / avg_L_hat)**gamma
        # where beta and gamma are hyperparameters.
        
        # A more direct approach as in the paper:
        # Define a loss for the weights themselves, and backpropagate through it.
        # This requires `log_loss_weights` to be `nn.Parameter`.
        
        # Calculate the gradient loss
        # This loss aims to make inverse_training_rates closer to avg_L_hat
        # The paper defines L_grad as:
        # L_grad = sum_{i} |G_W(i) - G_W_avg * (L_i_hat / L_avg_hat)^alpha |
        # where G_W(i) is the gradient norm of L_i w.r.t. shared weights.
        
        # Let's use a simpler formulation for the target gradient magnitude for each task:
        # Target_grad_norm_i = G_bar_t * (L_hat[name] / avg_L_hat).pow(self.alpha)
        # This is the target value for `grad_norms[i]`.
        
        # The loss for `log_loss_weights` is then a difference between `grad_norms` and `Target_grad_norm`.
        # This loss is then backpropagated to `log_loss_weights`.
        
        # Compute the target gradient magnitudes
        target_grad_norms = G_bar_t * (torch.stack([L_hat[name] for name in self.task_names]) / avg_L_hat).pow(self.alpha)

        # Compute the GradNorm loss (e.g., L1 difference between actual and target gradient norms)
        grad_norm_loss = F.l1_loss(grad_norms, target_grad_norms, reduction='mean')

        # Backpropagate this loss to adjust the `log_loss_weights`
        # This is a meta-gradient step.
        self.zero_grad() # Zero gradients for GradNorm's own parameters
        grad_norm_loss.backward()
        
        # Update the log_loss_weights using an optimizer (e.g., Adam)
        # GradNorm usually has its own internal optimizer for its weights.
        # For simplicity, we'll use a manual update or assume an external optimizer.
        # If no external optimizer, we can do a simple gradient descent step.
        
        # A common practice is to use a separate optimizer for GradNorm's weights.
        # Let's assume this is handled externally or use a simple step.
        
        # For now, we will simply apply the gradients to `log_loss_weights`
        # if they were computed. This requires `log_loss_weights.grad` to be populated.
        
        # The `grad_norm_loss.backward()` will populate `self.log_loss_weights.grad`.
        # Then, a small learning rate can be used to update them.
        
        with torch.no_grad():
            self.log_loss_weights.grad /= self.log_loss_weights.grad.norm() + 1e-8 # Normalize gradients
            self.log_loss_weights.data.add_(-self.log_loss_weights.grad * 0.001) # Small learning rate
        
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
    def __init__(self, all_bongard_rules: Dict[str, Any], loss_weight: float = 1.0, config: Optional[Dict[str, Any]] = None, symbolic_engine: Optional[Any] = None):
        super().__init__()
        self.all_bongard_rules = all_bongard_rules
        self.loss_weight = loss_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # For binary consistency prediction
        
        self.symbolic_engine = symbolic_engine # Use provided engine or initialize
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
                labels: torch.Tensor,  # Ground truth labels for Bongard problem (0 or 1)
                ground_truth_scene_graphs: List[bytes] # Raw GT JSON strings for rule induction
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
        
        # For symbolic consistency, we need to infer rules from the ground truth scene graphs
        # or use a pre-defined set if the problem is fixed.
        # Assuming `ground_truth_scene_graphs` contains the GT for the current batch.
        
        # In a real scenario, you'd use `self.symbolic_engine.induce_rules` here
        # or fetch the correct rules for the current problems.
        # For this dummy implementation, let's assume we have a single "best" rule
        # that should apply to positive examples and not to negative ones.
        
        # This part is highly dependent on how `symbolic_engine.induce_rules` works
        # and how rules are associated with problems.
        # For now, let's use a dummy rule or assume `all_bongard_rules` is properly populated.
        
        hypothesized_rule = None
        if self.all_bongard_rules:
            # In a real system, you'd select the most relevant rule
            # or induce one from the GT. For now, pick a dummy.
            hypothesized_rule = BongardRule(name="dummy_pos_rule") # Assume a dummy positive rule

        if hypothesized_rule is None:
            logger.warning("SymbolicConsistencyLoss: No hypothesized rule available. Returning 0.0.")
            return torch.tensor(0.0, device=labels.device)

        # Iterate through each problem in the batch
        for i in range(len(labels)):
            gt_label = labels[i].item()  # 0 or 1
            inferred_sg_view1 = scene_graphs1[i]
            inferred_sg_view2 = scene_graphs2[i]

            # Evaluate each inferred scene graph against the hypothesized rule
            is_consistent_view1 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view1, hypothesized_rule)
            is_consistent_view2 = self.symbolic_engine.evaluate_scene_graph_against_rule(inferred_sg_view2, hypothesized_rule)

            # Define the target for the consistency loss
            # If gt_label is 1 (positive problem), both views should ideally be consistent with the rule.
            # If gt_label is 0 (negative problem), both views should ideally be inconsistent with the rule.
            # This is a simplification based on the problem being a binary classification.
            target_consistency_view1 = 1.0 if gt_label == 1 else 0.0
            target_consistency_view2 = 1.0 if gt_label == 1 else 0.0

            # Convert boolean consistency to float tensor for loss calculation
            predicted_consistency_view1 = torch.tensor(float(is_consistent_view1), device=labels.device).unsqueeze(0)
            predicted_consistency_view2 = torch.tensor(float(is_consistent_view2), device=labels.device).unsqueeze(0)
            
            # Create a dummy "logit" for BCEWithLogitsLoss.
            # If predicted_consistency is 1.0, logit should be high positive (e.g., 10.0)
            # If predicted_consistency is 0.0, logit should be high negative (e.g., -10.0)
            logit_view1 = predicted_consistency_view1 * 20.0 - 10.0  # Maps 0 to -10, 1 to 10
            logit_view2 = predicted_consistency_view2 * 20.0 - 10.0
            target_tensor_view1 = torch.tensor(target_consistency_view1, device=labels.device).unsqueeze(0)
            target_tensor_view2 = torch.tensor(target_consistency_view2, device=labels.device).unsqueeze(0)
            
            # Compute BCEWithLogitsLoss for each view
            loss_view1 = self.bce_loss(logit_view1, target_tensor_view1)
            loss_view2 = self.bce_loss(logit_view2, target_tensor_view2)
            
            total_symbolic_loss += (loss_view1 + loss_view2) / 2.0  # Average loss for the problem
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
            alpha_tensor = torch.full_like(targets, self.alpha, dtype=torch.float32)
            alpha_tensor[targets == 0] = (1 - self.alpha)  # Assuming 0 is negative class
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
