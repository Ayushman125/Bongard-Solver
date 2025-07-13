import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import CONFIG from main.py for global access
from main import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NeuralNMS(nn.Module):
    def __init__(self):
        """
        Implements a simple Neural NMS module using a Multi-Layer Perceptron (MLP).
        It learns to predict a suppression score for each bounding box.
        """
        super().__init__()
        
        # Ensure CONFIG is accessible. If running standalone, provide a dummy.
        current_config = CONFIG if 'CONFIG' in globals() else {
            'learned_nms': {
                'enabled': True,
                'hidden_dim': 64,
                'num_layers': 2,
                'score_threshold': 0.05,
                'iou_threshold': 0.5 # Not directly used in NeuralNMS forward, but for overall NMS
            }
        }
        # Input to the MLP: typically 5 features (x1, y1, x2, y2, score)
        input_dim = 5 
        hidden_dim = current_config['learned_nms']['hidden_dim']
        num_layers = current_config['learned_nms']['num_layers']
        
        layers = []
        dim = input_dim
        for i in range(num_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True)]
            dim = hidden_dim
        
        # Output: a single score per box (logit for probability of being kept)
        layers.append(nn.Linear(dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Store the score threshold for filtering
        self.score_threshold = current_config['learned_nms']['score_threshold']
        logger.info(f"NeuralNMS initialized: hidden_dim={hidden_dim}, num_layers={num_layers}, score_threshold={self.score_threshold}")

    def forward(self, boxes):
        """
        Forward pass for Neural NMS.
        Args:
            boxes (torch.Tensor): Input bounding boxes. Shape: (N, 5) where N is number of boxes,
                                  and 5 features are typically (x1, y1, x2, y2, score).
        Returns:
            torch.Tensor: Filtered bounding boxes after applying learned suppression.
                          Shape: (M, 5) where M <= N.
        """
        if boxes is None or boxes.numel() == 0:
            logger.debug("NeuralNMS received empty input boxes. Returning empty tensor.")
            # Ensure the returned tensor is on the same device as inputs if possible
            device = boxes.device if boxes is not None else 'cpu'
            return torch.empty(0, 5, device=device)

        # Predict a suppression score for each box
        # The output is logits, apply sigmoid to get probabilities (0 to 1)
        # Higher score means higher probability of being kept.
        scores = self.net(boxes).sigmoid().squeeze(-1) # Shape: (N,)
        
        # Filter boxes based on the learned score threshold
        # Keep boxes where the learned score is greater than the configured threshold
        kept_indices = (scores > self.score_threshold).nonzero(as_tuple=True)[0]
        
        filtered_boxes = boxes[kept_indices]
        
        logger.debug(f"NeuralNMS forward pass: Input boxes={boxes.shape[0]}, Kept boxes={filtered_boxes.shape[0]}")
        return filtered_boxes

# Global instance of NeuralNMS (lazy initialization)
_neural_nms_model = None

def apply_neural_nms(boxes):
    """
    Applies the Neural NMS filtering to a set of bounding boxes.
    This function manages the lazy initialization of the NeuralNMS model.
    Args:
        boxes (torch.Tensor): Input bounding boxes. Shape: (N, 5) where N is number of boxes,
                              and 5 features are typically (x1, y1, x2, y2, score).
    Returns:
        torch.Tensor: Filtered bounding boxes.
    """
    global _neural_nms_model
    
    # Ensure CONFIG is accessible. If running standalone, provide a dummy.
    current_config = CONFIG if 'CONFIG' in globals() else {
        'learned_nms': {
            'enabled': True,
            'hidden_dim': 64,
            'num_layers': 2,
            'score_threshold': 0.05,
            'iou_threshold': 0.5
        }
    }
    if not current_config['learned_nms'].get('enabled', False):
        logger.info("Neural NMS is disabled in CONFIG. Skipping application.")
        return boxes # Return original boxes if disabled

    if _neural_nms_model is None:
        logger.info("Initializing NeuralNMS model (first call).")
        _neural_nms_model = NeuralNMS().to(boxes.device)
        # In a real scenario, you would load pre-trained weights for NeuralNMS here
        # _neural_nms_model.load_state_dict(torch.load('path/to/neural_nms_weights.pth'))
        _neural_nms_model.eval() # Set to evaluation mode

    logger.debug("Applying Neural NMS filter.")
    with torch.no_grad():
        return _neural_nms_model(boxes)

if __name__ == '__main__':
    # Example usage for testing the neural_nms module
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Dummy CONFIG for testing
    dummy_config_for_test = {
        'learned_nms': {
            'enabled': True,
            'hidden_dim': 64,
            'num_layers': 2,
            'score_threshold': 0.5, # Adjust threshold for testing
            'iou_threshold': 0.5
        }
    }
    # Temporarily set global CONFIG for testing if it's not already set
    if 'CONFIG' not in globals():
        global CONFIG
        CONFIG = dummy_config_for_test
    else: # If CONFIG exists, update it for the test
        CONFIG.update(dummy_config_for_test)

    print("\n--- Testing NeuralNMS ---")
    
    # Dummy bounding boxes: [x1, y1, x2, y2, score]
    # Scores are what the NMS will operate on, higher score means more likely to be kept.
    # Let's create some boxes, some with high scores, some with low.
    dummy_boxes = torch.tensor([
        [10, 10, 50, 50, 0.95],   # High score, should be kept
        [12, 12, 52, 52, 0.90],   # High score, overlaps with first, might be kept if threshold is low
        [100, 100, 150, 150, 0.80],  # High score, should be kept
        [102, 102, 152, 152, 0.75],  # High score, overlaps
        [200, 200, 250, 250, 0.30],  # Low score, should be removed if threshold > 0.3
        [205, 205, 255, 255, 0.25],  # Very low score, should be removed
        [300, 300, 350, 350, 0.60]   # Medium score, should be kept
    ], dtype=torch.float32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_boxes = dummy_boxes.to(device)

    print(f"Original boxes:\n{dummy_boxes}")
    
    filtered_boxes = apply_neural_nms(dummy_boxes)
    
    print(f"\nFiltered boxes (threshold={CONFIG['learned_nms']['score_threshold']}):\n{filtered_boxes}")
    print(f"Number of original boxes: {dummy_boxes.shape[0]}")
    print(f"Number of filtered boxes: {filtered_boxes.shape[0]}")

    # Test with Neural NMS disabled
    print("\n--- Testing NeuralNMS (Disabled in CONFIG) ---")
    CONFIG['learned_nms']['enabled'] = False
    filtered_boxes_disabled = apply_neural_nms(dummy_boxes)
    print(f"Filtered boxes (Neural NMS disabled - should be original):\n{filtered_boxes_disabled}")
    print(f"Number of filtered boxes (Neural NMS disabled): {filtered_boxes_disabled.shape[0]}")
