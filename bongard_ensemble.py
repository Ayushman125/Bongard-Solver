# Folder: bongard_solver/
# File: bongard_ensemble.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import logging # Added for logging
from pathlib import Path # Added for path manipulation
import random # For dummy data generation
import yaml # For loading config if needed
import json # For loading config if needed

# Import PerceptionModule if it's a separate file
try:
    from models import PerceptionModule
    logger = logging.getLogger(__name__) # Initialize logger
    logger.info("PerceptionModule imported for BongardEnsemble.")
except ImportError:
    logger = logging.getLogger(__name__) # Initialize logger even if import fails
    logger.error("Could not import PerceptionModule. BongardEnsemble will use a dummy model.")
    class PerceptionModule(nn.Module):
        def __init__(self, config: Dict[str, Any]):
            super().__init__()
            self.linear = nn.Linear(10, 2)  # Dummy linear layer
            self.config = config
        def forward(self, images: torch.Tensor, ground_truth_json_strings: Optional[List[str]] = None):
            # Dummy forward pass returning random logits, empty detected objects, and empty aggregated outputs
            return torch.randn(images.shape[0], 2), [], {}
        def extract_scene_graph(self, detected_objects, attribute_logits, relation_logits):
            return {"objects": [], "relations": [], "uncertainty": {"epistemic": 0.0, "aleatoric": 0.0}}

# Import diversity functions from ensemble.py
try:
    from ensemble import compute_diversity, save_diversity
    logger.info("Diversity functions (compute_diversity, save_diversity) imported from ensemble.py.")
except ImportError:
    logger.error("Could not import diversity functions from ensemble.py. Dummy functions will be used.")
    def compute_diversity(predictions_list):
        logger.warning("Using dummy compute_diversity.")
        return 0.0, 0.0
    def save_diversity(entropy, disagreement, path):
        logger.warning(f"Using dummy save_diversity. Would have saved entropy={entropy}, disagreement={disagreement} to {path}")

# --- CONFIGURATION (simplified for ensemble context) ---
# In a real scenario, this might load from a central config or be passed.
CONFIG = {
    'output_root': './datasets/bongard_objects',
    'model_save_dir': './runs/train/yolov8_bongard',
    'model_name': 'yolov8n.pt',
    'yolo_img_size': 224,
    'yolo_batch_size': 16,
    'yolo_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'yolo_epochs': 5, # Added for simulation
    'ensemble': {
        'num_models': 3, # Number of models in the ensemble (conceptual)
        'diversity_path': './runs/ensemble/diversity_metrics.json', # Path to save diversity metrics
        'eval_epochs': [1, 3, 5], # Epochs at which to evaluate diversity
    },
    'model': { # Dummy model config for PerceptionModule initialization
        'bongard_head_config': {
            'num_classes': 2 # Assuming 2 classes for Bongard problems (True/False)
        }
    }
}

def load_data_yaml(data_root):
    """Loads the data.yaml file."""
    data_yaml_path = Path(data_root) / 'data.yaml'
    if not data_yaml_path.exists():
        logger.error(f"data.yaml not found at {data_yaml_path}. Please ensure data generation is complete.")
        return None
    with open(data_yaml_path, 'r') as f:
        return yaml.safe_load(f)

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
        
        logger.info(f"BongardEnsemble initialized with {len(self.members)} members.")
        logger.info(f"Ensemble weights: {self.weights.tolist()}")

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

def run_ensemble_training_and_diversity_evaluation():
    logger.info("Starting Bongard Ensemble Training and Diversity Evaluation...")

    # Load data.yaml (conceptual, replace with actual data loading)
    # For this simulation, we don't need the full data.yaml content,
    # but in a real setup, it would define paths for YOLO.
    # data_yaml = load_data_yaml(CONFIG['output_root'])
    # if data_yaml is None:
    #     return

    # Ensure output directory for diversity metrics exists
    diversity_output_path = Path(CONFIG['ensemble']['diversity_path'])
    diversity_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Conceptual ensemble training loop
    # In a real scenario, you would train multiple models or load pre-trained ones.
    # For this example, we'll simulate training multiple models and evaluating diversity
    # at specified epochs.

    # Dummy models for demonstration
    ensemble_members = []
    for i in range(CONFIG['ensemble']['num_models']):
        # Each member needs a config to initialize PerceptionModule
        member_config = CONFIG.copy() # Use a copy to avoid modifying global CONFIG
        # You might adjust member_config for each model if they have different architectures/hyperparams
        ensemble_members.append(PerceptionModule(member_config).to(CONFIG['yolo_device']))
        logger.info(f"Initialized dummy ensemble member {i+1} on device: {CONFIG['yolo_device']}")

    # Simulate epoch-based training and diversity calculation
    all_diversity_results = []

    for epoch in range(1, CONFIG['yolo_epochs'] + 1):
        logger.info(f"--- Simulating Epoch {epoch} ---")

        # Simulate model training for one epoch for each member
        # In a real scenario, this would be model.train(...) for each member.
        # For this simulation, we'll just log and generate dummy predictions.
        logger.info(f"Simulating training for epoch {epoch} for {CONFIG['ensemble']['num_models']} members...")
        
        # Simulate predictions from multiple "ensemble members"
        # In a real scenario, you'd run each `ensemble_members[i]` through a validation loader
        # to get its predictions.
        predictions_from_ensemble_members = []
        for member_idx, member_model in enumerate(ensemble_members):
            # Simulate a batch of images for prediction
            dummy_images = torch.randn(CONFIG['yolo_batch_size'], 3, CONFIG['yolo_img_size'], CONFIG['yolo_img_size']).to(CONFIG['yolo_device'])
            
            # Get logits from the dummy model
            # Assuming dummy PerceptionModule.forward returns (bongard_logits, detected_objects, aggregated_outputs)
            bongard_logits, _, _ = member_model(dummy_images)
            
            # Convert logits to probabilities
            member_probs = F.softmax(bongard_logits, dim=-1)
            
            # For `compute_diversity`, we need a list of predictions.
            # Here, we'll just use the raw probabilities as a stand-in for "predictions".
            predictions_from_ensemble_members.append(member_probs) # Append the tensor of probabilities

        # Check if this epoch is in the diversity evaluation list
        if epoch in CONFIG['ensemble']['eval_epochs']:
            logger.info(f"Evaluating diversity at epoch {epoch}...")
            # Call compute_diversity from ensemble.py
            # `compute_diversity` expects `List[nn.Module]` and a batch of input images.
            # We will pass the actual `ensemble_members` list and a dummy batch.
            dummy_images_for_diversity = torch.randn(CONFIG['yolo_batch_size'], 3, CONFIG['yolo_img_size'], CONFIG['yolo_img_size']).to(CONFIG['yolo_device'])
            ent, dis = compute_diversity(ensemble_members, dummy_images_for_diversity)
            
            # Store results
            epoch_diversity_data = {
                'epoch': epoch,
                'entropy': ent,
                'disagreement': dis
            }
            all_diversity_results.append(epoch_diversity_data)
            logger.info(f"Epoch {epoch} Diversity: Entropy={ent:.4f}, Disagreement={dis:.4f}")

            # Save diversity metrics after each evaluation epoch
            # Note: The `save_diversity` function overwrites the file each time.
            # If you want to accumulate, you'd need to read the existing JSON, append, and then write.
            # For this example, it demonstrates the call.
            save_diversity(ent, dis, str(diversity_output_path))
            
    logger.info("Bongard Ensemble Training and Diversity Evaluation finished.")
    logger.info(f"All diversity results: {all_diversity_results}")

if __name__ == '__main__':
    # Configure main logger for bongard_ensemble.py
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    run_ensemble_training_and_diversity_evaluation()
