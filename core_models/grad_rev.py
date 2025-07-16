# Folder: bongard_solver/core_models/
# File: grad_rev.py
import torch
from torch.autograd import Function
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GradReverse(Function):
    """
    Implementation of the Gradient Reversal Layer (GRL).
    During the forward pass, this layer acts as an identity function.
    During the backward pass, it multiplies the gradient by a negative scalar `alpha`.
    This is commonly used in domain adaptation techniques like Domain-Adversarial Training
    of Neural Networks (DANN) to encourage domain-invariant feature learning.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Forward pass: Returns the input tensor unchanged.
        Stores `alpha` in `ctx` for use in the backward pass.
        
        Args:
            ctx: Context object to save information for backward pass.
            x (torch.Tensor): The input tensor.
            alpha (float): The scalar value by which to multiply the gradients
                           in the backward pass. Typically positive.
        Returns:
            torch.Tensor: The input tensor `x`.
        """
        ctx.alpha = alpha
        # logger.debug(f"GradReverse: Forward pass. Alpha: {alpha}")
        return x.view_as(x) # Return a view to maintain tensor properties

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass: Multiplies the incoming gradient by -alpha.
        
        Args:
            ctx: Context object containing saved information from forward pass.
            grad_output (torch.Tensor): The gradient tensor from the subsequent layer.
        Returns:
            Tuple[torch.Tensor, None]: The reversed gradient for `x`, and None for `alpha`
                                       (as `alpha` is not a learnable parameter).
        """
        # logger.debug(f"GradReverse: Backward pass. Grad_output norm: {grad_output.norm().item():.4f}, Alpha: {ctx.alpha}")
        return grad_output.neg() * ctx.alpha, None

# Convenience function to apply the Gradient Reversal Layer
def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Applies the Gradient Reversal Layer to the input tensor.
    
    Args:
        x (torch.Tensor): The input tensor.
        alpha (float): The scalar value by which to multiply the gradients
                       in the backward pass.
    Returns:
        torch.Tensor: The output tensor from the GRL.
    """
    return GradReverse.apply(x, alpha)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Running grad_rev.py example.")

    # Create a dummy tensor
    x = torch.randn(10, 5, requires_grad=True)
    logger.info(f"Original tensor x:\n{x}")

    # Apply GradReverse with alpha = 1.0
    alpha_val = 1.0
    y = grad_reverse(x, alpha_val)
    logger.info(f"Output of grad_reverse (should be identical to x):\n{y}")

    # Define a dummy loss (e.g., sum of squares)
    loss = (y ** 2).sum()
    logger.info(f"Dummy loss: {loss.item():.4f}")

    # Perform backward pass
    loss.backward()

    # Check gradients of x
    # The gradient should be -2 * x * alpha
    expected_grad = -2 * x.detach() * alpha_val
    logger.info(f"Gradient of x after backward pass (expected -2*x*alpha):\n{x.grad}")
    logger.info(f"Expected gradient of x:\n{expected_grad}")

    # Verify if gradients are approximately equal
    if torch.allclose(x.grad, expected_grad):
        logger.info("Gradient Reversal Layer works correctly (gradients are reversed and scaled).")
    else:
        logger.error("Gradient Reversal Layer test failed: Gradients do not match expected values.")
        logger.error(f"Difference: {(x.grad - expected_grad).abs().max().item()}")

    # Test with a different alpha
    x2 = torch.randn(5, 3, requires_grad=True)
    alpha_val_2 = 0.5
    y2 = grad_reverse(x2, alpha_val_2)
    loss2 = y2.mean() # Simple mean loss
    loss2.backward()
    expected_grad2 = -torch.ones_like(x2) * (alpha_val_2 / (x2.numel())) # For mean, grad is 1/N * -alpha
    logger.info(f"\nTesting with alpha={alpha_val_2}. Gradient of x2:\n{x2.grad}")
    logger.info(f"Expected gradient of x2 (should be -alpha/N):\n{expected_grad2}")
    if torch.allclose(x2.grad, expected_grad2):
        logger.info("Gradient Reversal Layer works correctly with different alpha.")
    else:
        logger.error("Gradient Reversal Layer test failed with different alpha.")
        logger.error(f"Difference: {(x2.grad - expected_grad2).abs().max().item()}")

