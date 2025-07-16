# Folder: bongard_solver/core_models/
# File: style_disc.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

# Import the Gradient Reversal Layer
from .grad_rev import grad_reverse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StyleDiscriminator(nn.Module):
    """
    A neural network designed to discriminate between features from different domains.
    This discriminator is used in domain adaptation to encourage the feature extractor
    (e.g., the backbone of the perception model) to produce domain-invariant features.
    
    It applies a Gradient Reversal Layer (GRL) before its classification head,
    so that during training, the feature extractor is optimized to *fool* the discriminator
    (i.e., make its features indistinguishable across domains), while the discriminator
    is optimized to *distinguish* between domains.
    """
    def __init__(self, feat_dim: int = 576):
        """
        Initializes the StyleDiscriminator.
        
        Args:
            feat_dim (int): The dimensionality of the input features from the feature extractor.
                            This should match the output dimension of the backbone/feature_extractor
                            of your perception model (e.g., 576 for MobileNetV3-Small).
        """
        super().__init__()
        
        # Define the discriminator network
        # It's typically a simple MLP that takes features and outputs a single logit
        # indicating the domain (e.g., 0 for synthetic, 1 for real).
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256), # First hidden layer
            nn.BatchNorm1d(256),      # Batch Normalization for stability
            nn.ReLU(),                # Activation function
            nn.Dropout(0.5),          # Dropout for regularization
            
            nn.Linear(256, 64),       # Second hidden layer
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 1)          # Output layer: single logit for binary classification
        )
        
        logger.info(f"StyleDiscriminator initialized with input feature dimension: {feat_dim}")

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass of the StyleDiscriminator.
        
        Args:
            x (torch.Tensor): Input features from the feature extractor (batch_size, feat_dim).
            alpha (float): The alpha parameter for the Gradient Reversal Layer.
                           This controls the strength of the gradient reversal.
                           Defaults to 1.0.
        Returns:
            torch.Tensor: The sigmoid output (probabilities) indicating the domain (batch_size, 1).
        """
        # Apply the Gradient Reversal Layer
        # The gradients flowing back through this point will be multiplied by -alpha.
        x = grad_reverse(x, alpha)
        
        # Pass the (gradient-reversed) features through the discriminator network
        logits = self.net(x)
        
        # Apply sigmoid to get probabilities (for BCEWithLogitsLoss, this is often done externally)
        # However, if BCEWithLogitsLoss is used, the sigmoid is integrated for numerical stability.
        # For this example, we return sigmoid output as per the integration snippet in the prompt.
        return torch.sigmoid(logits)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Running style_disc.py example.")

    # Define a dummy feature dimension
    feat_dim = 128
    
    # Instantiate the discriminator
    discriminator = StyleDiscriminator(feat_dim=feat_dim)
    logger.info(f"Discriminator model structure:\n{discriminator}")

    # Create dummy synthetic features (e.g., from a synthetic image)
    # These features are expected to be classified as domain 0 (synthetic)
    synthetic_features = torch.randn(32, feat_dim, requires_grad=True)
    
    # Create dummy real features (e.g., from a real image)
    # These features are expected to be classified as domain 1 (real)
    real_features = torch.randn(32, feat_dim, requires_grad=True)

    # Define a binary cross-entropy loss
    bce_loss = nn.BCELoss()

    # --- Training loop simulation ---
    num_iterations = 100
    learning_rate = 0.01
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    # Simulate training for a few iterations
    for i in range(num_iterations):
        optimizer_disc.zero_grad()

        # Discriminator's objective: classify synthetic as 0, real as 1
        # For synthetic features, we want the discriminator to predict 0.
        # We detach synthetic_features here to only train the discriminator on this path.
        preds_synthetic = discriminator(synthetic_features.detach(), alpha=1.0)
        loss_disc_synthetic = bce_loss(preds_synthetic, torch.zeros_like(preds_synthetic))

        # For real features, we want the discriminator to predict 1.
        preds_real = discriminator(real_features.detach(), alpha=1.0)
        loss_disc_real = bce_loss(preds_real, torch.ones_like(preds_real))

        total_disc_loss = loss_disc_synthetic + loss_disc_real
        
        total_disc_loss.backward()
        optimizer_disc.step()

        if (i + 1) % 10 == 0:
            logger.info(f"Iteration {i+1}: Discriminator Loss = {total_disc_loss.item():.4f}")
            # Check predictions
            avg_pred_synthetic = preds_synthetic.mean().item()
            avg_pred_real = preds_real.mean().item()
            logger.info(f"  Avg Pred Synthetic: {avg_pred_synthetic:.4f}, Avg Pred Real: {avg_pred_real:.4f}")

    logger.info("\nSimulating feature extractor training (adversarial part).")
    # Now simulate the feature extractor trying to fool the discriminator
    # The feature extractor's loss will be `bce_loss(discriminator(features), 1)`
    # because it wants the discriminator to classify its synthetic features as real (1).
    
    # Reset gradients for synthetic_features (which would come from the feature extractor)
    synthetic_features.grad = None
    
    # The `alpha` in `grad_reverse` ensures the gradients for the feature extractor
    # are reversed. So, when the feature extractor minimizes `bce_loss(..., 1)`,
    # the GRL makes it effectively maximize `bce_loss(..., 0)`.
    
    # Simulate one step of feature extractor training
    optimizer_feat_extractor = torch.optim.Adam([synthetic_features], lr=learning_rate)
    optimizer_feat_extractor.zero_grad()

    # Feature extractor's objective: make synthetic features look real (target 1)
    # The GRL inside the discriminator will reverse this gradient,
    # effectively making the feature extractor learn domain-invariant features.
    preds_adversarial = discriminator(synthetic_features, alpha=1.0) # alpha for GRL
    loss_feat_extractor = bce_loss(preds_adversarial, torch.ones_like(preds_adversarial))
    
    loss_feat_extractor.backward()
    optimizer_feat_extractor.step()

    logger.info(f"Feature Extractor Loss (trying to fool discriminator): {loss_feat_extractor.item():.4f}")
    logger.info(f"Avg Pred Synthetic (after adversarial update): {discriminator(synthetic_features.detach()).mean().item():.4f}")
    logger.info("If the feature extractor loss decreases and synthetic predictions move towards 0.5, it's working.")

