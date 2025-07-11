# Folder: bongard_solver/
# File: bongard_ensemble.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

# Import PerceptionModule if it's a separate file
try:
    from models import PerceptionModule
    logger.info("PerceptionModule imported for BongardEnsemble.")
except ImportError:
    logger.error("Could not import PerceptionModule. BongardEnsemble will use a dummy model.")
    class PerceptionModule(nn.Module):
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.linear = nn.Linear(10, 2) # Dummy linear layer
            self.config = config
        def forward(self, images: torch.Tensor, ground_truth_json_strings: Optional[List[str]] = None):
            # Dummy forward pass returning random logits, empty detected objects, and empty aggregated outputs
            return torch.randn(images.shape[0], 2), [], {}
        def extract_scene_graph(self, detected_objects, attribute_logits, relation_logits):
            return {"objects": [], "relations": [], "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0}}


class BongardEnsemble(nn.Module):
    """
    A PyTorch Module representing a Bongard problem solving ensemble.
    It holds multiple `PerceptionModule` instances as members and combines their predictions.
    """
    def __init__(self, member_models: List[PerceptionModule], ensemble_weights: Optional[List[float]] = None):
        super().__init__()
        if not member_models:
            raise ValueError("BongardEnsemble must be initialized with at least one member model.")
        
        self.members = nn.ModuleList(member_models)
        
        if ensemble_weights is not None:
            if len(ensemble_weights) != len(member_models):
                raise ValueError("Number of ensemble weights must match the number of member models.")
            self.weights = torch.tensor(ensemble_weights, dtype=torch.float32)
            # Ensure weights are on the same device as models later
        else:
            # Default to equal weights if none provided
            self.weights = torch.ones(len(member_models), dtype=torch.float32)
        
        # Move weights to the device of the first model (assuming all models are on the same device)
        if self.members:
            self.weights = self.weights.to(next(self.members[0].parameters()).device)
        
        # Normalize weights if they are not already
        if self.weights.sum() != 0:
            self.weights = self.weights / self.weights.sum()
        else:
            # Handle case where all weights are zero (e.g., if input weights were all zeros)
            self.weights = torch.ones(len(member_models), dtype=torch.float32) / len(member_models)
        
        self.num_classes = member_models[0].config['model']['bongard_head_config']['num_classes']
        
        print(f"BongardEnsemble initialized with {len(self.members)} members.")
        print(f"Ensemble weights: {self.weights.tolist()}")

    def forward(self, images: torch.Tensor, ground_truth_json_strings: Optional[List[str]] = None) -> torch.Tensor:
        """
        Performs a forward pass through all ensemble members and combines their predictions
        using weighted voting.
        Args:
            images (torch.Tensor): Input image batch (B, C, H, W).
            ground_truth_json_strings (Optional[List[str]]): List of ground truth JSON strings
                                                             for scene graph extraction.
        Returns:
            torch.Tensor: Final ensemble predictions (hard labels) (B,).
        """
        all_member_logits = []
        
        # Ensure all members are in evaluation mode
        for member in self.members:
            member.eval()
        
        with torch.no_grad():
            for member in self.members:
                # Each member model's forward pass
                # Assuming PerceptionModule.forward returns (bongard_logits, detected_objects, aggregated_outputs)
                bongard_logits, _, _ = member(images, ground_truth_json_strings)
                all_member_logits.append(bongard_logits)
            
            # Stack logits from all members: (num_members, B, num_classes)
            stacked_logits = torch.stack(all_member_logits, dim=0)
            
            # Convert logits to probabilities using softmax
            # probs: (num_members, B, num_classes)
            probs = F.softmax(stacked_logits, dim=-1)
            
            # Reshape weights for broadcasting: (num_members, 1, 1)
            # weights_reshaped: (num_members, 1, 1)
            weights_reshaped = self.weights[:, None, None]
            
            # Weighted sum of probabilities
            # final_probs: (B, num_classes)
            final_probs = (weights_reshaped * probs).sum(dim=0) / self.weights.sum()
            
            # Get hard predictions (class with highest probability)
            final_predictions = final_probs.argmax(dim=-1)
            
            return final_predictions

    def get_member_logits_and_probs(self, images: torch.Tensor, ground_truth_json_strings: Optional[List[str]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns raw logits and probabilities from each ensemble member.
        Useful for analysis or further processing (e.g., calibration).
        Args:
            images (torch.Tensor): Input image batch (B, C, H, W).
            ground_truth_json_strings (Optional[List[str]]): List of ground truth JSON strings.
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - List of logits from each member (each tensor: B, num_classes).
                - List of probabilities from each member (each tensor: B, num_classes).
        """
        all_member_logits = []
        all_member_probs = []
        
        for member in self.members:
            member.eval()
            with torch.no_grad():
                bongard_logits, _, _ = member(images, ground_truth_json_strings)
                all_member_logits.append(bongard_logits)
                all_member_probs.append(F.softmax(bongard_logits, dim=-1))
        return all_member_logits, all_member_probs

