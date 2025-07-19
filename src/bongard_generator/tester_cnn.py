"""
Neural tester for semantic verification of generated Bongard scenes.
Provides confidence scores for rule compliance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class TesterCNN(nn.Module):
    """Lightweight CNN for rule verification and confidence scoring."""
    
    def __init__(self, num_rules: int = 16, input_size: int = 128):
        super().__init__()
        self.num_rules = num_rules
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_rules)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Input: [batch_size, 1, 128, 128]
        x = F.relu(self.conv1(x))  # [batch, 32, 128, 128]
        x = self.pool(x)           # [batch, 32, 64, 64]
        
        x = F.relu(self.conv2(x))  # [batch, 64, 64, 64]
        x = self.pool(x)           # [batch, 64, 32, 32]
        
        x = F.relu(self.conv3(x))  # [batch, 128, 32, 32]
        x = self.pool(x)           # [batch, 128, 16, 16]
        
        x = self.adaptive_pool(x)  # [batch, 128, 4, 4]
        x = x.view(x.size(0), -1)  # [batch, 128*4*4]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict_confidence(self, image: np.ndarray, rule_idx: int) -> float:
        """Predict confidence for a specific rule."""
        # Convert image to tensor
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            logits = self.forward(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence = probabilities[0, rule_idx].item()
        
        return confidence
    
    def predict_rule(self, image: np.ndarray) -> Tuple[int, float]:
        """Predict the most likely rule and its confidence."""
        # Convert image to tensor
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            logits = self.forward(tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_rule = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_rule].item()
        
        return predicted_rule, confidence

class MockTesterCNN:
    """Mock tester for when we don't have a trained model yet."""
    
    def __init__(self, num_rules: int = 16):
        self.num_rules = num_rules
        self.rule_patterns = {
            'shape': ['circle', 'triangle', 'square'],
            'fill': ['solid', 'hollow', 'striped'],
            'count': [1, 2, 3, 4],
            'relation': ['near', 'far', 'overlap', 'inside']
        }
    
    def predict_confidence(self, image: np.ndarray, rule_desc: str) -> float:
        """Mock confidence prediction based on simple heuristics."""
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('L'))
        
        # Simple heuristics based on image properties
        height, width = image.shape[:2]
        total_pixels = height * width
        black_pixels = np.sum(image < 128)  # Count dark pixels
        
        # Coverage ratio
        coverage = black_pixels / total_pixels
        
        # Base confidence starts at 0.5
        confidence = 0.5
        
        # Adjust based on coverage (good scenes have moderate coverage)
        if 0.1 <= coverage <= 0.4:
            confidence += 0.3
        elif coverage < 0.05:  # Too sparse
            confidence -= 0.2
        elif coverage > 0.6:   # Too dense
            confidence -= 0.2
        
        # Add some randomness for diversity
        confidence += random.uniform(-0.1, 0.1)
        
        # Ensure confidence is in [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def predict_rule(self, image: np.ndarray) -> Tuple[int, float]:
        """Mock rule prediction."""
        rule_idx = random.randint(0, self.num_rules - 1)
        confidence = self.predict_confidence(image, f"rule_{rule_idx}")
        return rule_idx, confidence

def create_tester_model(model_path: str = None, num_rules: int = 16) -> TesterCNN:
    """Create and optionally load a tester model."""
    model = TesterCNN(num_rules=num_rules)
    # Use device from config if available
    device = getattr(model, 'device', 'cpu')
    if model_path and Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded tester model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            logger.info("Using randomly initialized model")
    else:
        logger.info("Using randomly initialized tester model")
    model.eval()
    return model
