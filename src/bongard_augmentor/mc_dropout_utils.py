"""
Monte Carlo Dropout Utilities for Uncertainty Quantification in Mask Prediction.
"""
import torch
import numpy as np
import logging

def enable_dropout(model):
    """
    Function to enable the dropout layers during test-time
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_mask_prediction(model, input_tensor, n_runs=20, device='cpu'):
    """
    Run Monte Carlo dropout inference for mask prediction.
    Args:
        model: PyTorch model with dropout layers.
        input_tensor: Input tensor for the model (batched or single image).
        n_runs: Number of stochastic forward passes.
        device: Device to run inference on.
    Returns:
        mean_mask: Mean of predicted masks (numpy array)
        var_mask: Variance of predicted masks (numpy array)
        all_masks: All predicted masks (numpy array, shape: [n_runs, ...])
    """
    model.eval()
    enable_dropout(model)
    all_masks = []
    with torch.no_grad():
        for _ in range(n_runs):
            output = model(input_tensor.to(device))
            # Assume output is a mask probability map (B, 1, H, W) or (B, H, W)
            if isinstance(output, (tuple, list)):
                output = output[0]
            mask_prob = torch.sigmoid(output) if output.shape[1] == 1 else output
            mask_np = mask_prob.detach().cpu().numpy()
            all_masks.append(mask_np)
    all_masks = np.stack(all_masks, axis=0) # (n_runs, B, 1, H, W) or (n_runs, ...)
    mean_mask = np.mean(all_masks, axis=0)
    var_mask = np.var(all_masks, axis=0)
    return mean_mask, var_mask, all_masks
