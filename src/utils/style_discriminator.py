# Folder: bongard_solver/src/utils/
# File: style_discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class StyleDiscriminator(nn.Module):
    """
    A neural network discriminator used for domain adaptation (e.g., in GAN-like setups).
    It learns to distinguish between features from different domains (e.g., synthetic vs. real).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256], dropout_prob: float = 0.3):
        """
        Initializes the StyleDiscriminator.
        Args:
            input_dim (int): The dimension of the input features (e.g., feature vector from a backbone).
            hidden_dims (List[int]): A list of integers specifying the sizes of hidden layers.
            dropout_prob (float): Dropout probability applied after each hidden layer.
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2)) # Using LeakyReLU as common in GANs
            layers.append(nn.Dropout(dropout_prob))
            current_dim = h_dim
        
        # Output layer: a single logit for binary classification (real/fake, or domain A/domain B)
        layers.append(nn.Linear(current_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        logger.info(f"StyleDiscriminator initialized with input_dim={input_dim}, hidden_dims={hidden_dims}, dropout_prob={dropout_prob}.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the discriminator.
        Args:
            features (torch.Tensor): Input feature tensor (batch_size, input_dim).
        Returns:
            torch.Tensor: A tensor of raw logits (batch_size, 1) indicating the
                          discriminator's score for each input feature.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0) # Add batch dimension if single feature vector
        
        if features.shape[1] != self.model[0].in_features:
            logger.error(f"Input feature dimension mismatch. Expected {self.model[0].in_features}, got {features.shape[1]}.")
            raise ValueError("Input feature dimension mismatch for StyleDiscriminator.")

        return self.model(features)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running style_discriminator.py example.")

    input_dim = 1024 # Example feature dimension
    hidden_dims = [512, 256]
    dropout_prob = 0.3

    discriminator = StyleDiscriminator(input_dim, hidden_dims, dropout_prob)
    
    # Create some dummy feature vectors
    batch_size = 32
    dummy_synthetic_features = torch.randn(batch_size, input_dim) # Features from synthetic domain
    dummy_real_features = torch.randn(batch_size, input_dim) # Features from real domain

    logger.info(f"Dummy synthetic features shape: {dummy_synthetic_features.shape}")
    logger.info(f"Dummy real features shape: {dummy_real_features.shape}")

    # Forward pass
    synthetic_logits = discriminator(dummy_synthetic_features)
    real_logits = discriminator(dummy_real_features)

    logger.info(f"Synthetic logits shape: {synthetic_logits.shape}")
    logger.info(f"Real logits shape: {real_logits.shape}")

    # Example of how to use the output in a training loop:
    # For adversarial training, you'd typically use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Labels: 1 for real, 0 for synthetic
    real_labels = torch.ones_like(real_logits)
    synthetic_labels = torch.zeros_like(synthetic_logits)

    # Discriminator's loss (trying to correctly classify)
    loss_real = criterion(real_logits, real_labels)
    loss_synthetic = criterion(synthetic_logits, synthetic_labels)
    discriminator_loss = loss_real + loss_synthetic

    logger.info(f"Discriminator loss for real features: {loss_real.item():.4f}")
    logger.info(f"Discriminator loss for synthetic features: {loss_synthetic.item():.4f}")
    logger.info(f"Total Discriminator loss: {discriminator_loss.item():.4f}")

    # For the generator (feature extractor), its goal is to fool the discriminator.
    # So, it would try to make synthetic features look 'real'.
    # generator_loss = criterion(synthetic_logits, real_labels) # Target real_labels for synthetic features
    # logger.info(f"Generator's adversarial loss (trying to fool discriminator): {generator_loss.item():.4f}")

    # You would then perform backward() and optimizer.step() for the discriminator
    # and separately for the generator/feature extractor.
