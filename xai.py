# Folder: bongard_solver/
# File: xai.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from PIL import Image
from typing import Dict, Any, Optional, Union, Callable

# Import torchvision transforms for consistent preprocessing/denormalization
from torchvision import transforms as T

# Import Grad-CAM library components
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
    logger = logging.getLogger(__name__) # Ensure logger is initialized before use
    logger.info("pytorch-grad-cam library found and enabled.")
except ImportError:
    HAS_GRAD_CAM = False
    logger = logging.getLogger(__name__) # Ensure logger is initialized before use
    logger.warning("pytorch-grad-cam library not found. Grad-CAM visualization will be disabled.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define ImageNet normalization parameters (assuming your model was trained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define a denormalization transform for visualization
# This transform takes a (C, H, W) tensor, denormalizes it,
# converts to (H, W, C) numpy array in [0, 1] range.
denormalize_transform = T.Compose([
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in IMAGENET_STD]),  # Undo std
    T.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1.0, 1.0, 1.0]),  # Undo mean
    T.ToPILImage(),  # Convert to PIL Image
    T.ToTensor(),  # Convert back to tensor (float, [0,1], C,H,W)
    lambda x: x.permute(1, 2, 0).cpu().numpy()  # Permute to HWC and convert to numpy
])

def generate_grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,  # Expected: (1, C, H, W) normalized tensor
    target_layer: nn.Module,
    target_category: Optional[int],  # Class index to explain
    image_path: str,  # Path to save the CAM overlay image
    save_dir: str = './grad_cam_outputs',
    file_suffix: str = '',
    use_grad_cam_plus_plus: bool = False,
    reshape_transform: Optional[Callable] = None  # For ViT-like models
):
    """
    Generates and saves a Grad-CAM visualization for a given input.
    Args:
        model (nn.Module): The PyTorch model.
        input_tensor (torch.Tensor): The preprocessed input image tensor (1, C, H, W).
        target_layer (nn.Module): The convolutional layer to target for CAM.
        target_category (Optional[int]): The index of the class for which to generate the CAM.
                                         If None, the highest predicted class will be used.
        image_path (str): The original image path (used for filename reference, not loading).
        save_dir (str): Directory to save the Grad-CAM output.
        file_suffix (str): Suffix to add to the saved filename (e.g., '_misclassified').
        use_grad_cam_plus_plus (bool): If True, use GradCAM++ instead of standard GradCAM.
        reshape_transform (Optional[Callable]): A function to reshape activations for non-CNN models (e.g., ViT).
    """
    if not HAS_GRAD_CAM:
        logger.warning("Grad-CAM generation skipped: pytorch-grad-cam library not available.")
        return
    if target_layer is None:
        logger.warning("Grad-CAM generation skipped: target_layer is None.")
        return
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Initialize CAM object
    if use_grad_cam_plus_plus:
        cam_instance = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda,
                                       reshape_transform=reshape_transform)
    else:
        cam_instance = GradCAM(model=model, target_layers=[target_layer], use_cuda=input_tensor.is_cuda,
                               reshape_transform=reshape_transform)
    
    # Define targets for CAM (e.g., the predicted class)
    targets = None
    if target_category is not None:
        targets = [ClassifierOutputTarget(target_category)]
    else:
        # If no target category is specified, use the predicted class
        with torch.no_grad():
            output = model(input_tensor)
            # Assuming the model outputs logits for the main classification task
            if isinstance(output, dict) and 'bongard_logits' in output:
                predicted_class = output['bongard_logits'].argmax(dim=1).item()
            elif isinstance(output, torch.Tensor):
                predicted_class = output.argmax(dim=1).item()
            else:
                logger.error("Model output format not recognized for Grad-CAM target selection.")
                return
            targets = [ClassifierOutputTarget(predicted_class)]
    
    # Compute the grayscale CAM heatmap
    # The `input_tensor` is already prepared (normalized, on device)
    grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets)[0]  # [0] because it returns a batch
    
    # Prepare the original image for overlay
    # Convert the input_tensor (normalized, CHW) back to HWC numpy array in [0,1]
    # for show_cam_on_image.
    # We use the denormalize_transform defined above.
    
    # Clone to avoid modifying original tensor and move to CPU if on GPU
    img_for_overlay_tensor = input_tensor.squeeze(0).cpu()  # Remove batch dim, move to CPU
    img_for_overlay_np = denormalize_transform(img_for_overlay_tensor)
    
    # Ensure the numpy array is in the correct type and range for show_cam_on_image
    # show_cam_on_image expects float32 in [0,1]
    img_for_overlay_np = np.float32(img_for_overlay_np)
    
    # Overlay the CAM heatmap on the image
    cam_overlay = show_cam_on_image(img_for_overlay_np, grayscale_cam, use_rgb=True)
    
    # Save the output image
    os.makedirs(save_dir, exist_ok=True)
    
    # Use the original image_path to derive a filename, but save in save_dir
    base_filename = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
    output_filename = f"{base_filename}_cam{file_suffix}.png"
    output_path = os.path.join(save_dir, output_filename)
    Image.fromarray(cam_overlay).save(output_path)
    logger.info(f"Grad-CAM saved to: {output_path}")

def attention_rollout(attn_mats: list[torch.Tensor], discard_ratio: float = 0.9) -> torch.Tensor:
    """
    Computes attention rollout for Transformer-based models.
    Reference: Abnar & Zuidema, “Quantifying Attention Flow in Transformers,” EMNLP 2020.
    
    Args:
        attn_mats (list[torch.Tensor]): A list of attention matrices from each layer.
                                        Each tensor in the list should be of shape [B, heads, N, N],
                                        where B is batch size, heads is number of attention heads,
                                        N is the sequence length (e.g., num_patches + 1 for CLS token).
        discard_ratio (float): The ratio of attention weights to discard (set to zero)
                               before summing, to focus on stronger connections.
                               A value of 0.9 means the weakest 90% of connections are discarded.
    Returns:
        torch.Tensor: The attention rollout matrix of shape [B, N, N], representing
                      the aggregated attention flow from the input tokens to all other tokens.
    """
    if not attn_mats:
        logger.warning("No attention matrices provided for attention rollout.")
        return torch.empty(0) # Return an empty tensor or handle as appropriate

    # Initialize the result with an identity matrix, representing direct connections.
    # Shape of result will be [B, N, N]
    batch_size, _, seq_len, _ = attn_mats[0].shape
    result = torch.eye(seq_len, device=attn_mats[0].device).unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, N]

    for mat in attn_mats:
        # Average attention weights across heads for each layer
        avg_heads = mat.mean(dim=1)  # [B, N, N]

        # Apply discard ratio: set weakest attention weights to zero
        # Flatten for quantile calculation per batch item
        flat = avg_heads.flatten(start_dim=-2) # [B, N*N]
        # Calculate quantile threshold for each item in the batch
        # unsqueeze(-1) to make it broadcastable with avg_heads
        threshold = torch.quantile(flat, discard_ratio, dim=-1).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]
        
        # Create a mask for values below the threshold
        mask = avg_heads < threshold
        # Apply the mask: set values below threshold to 0
        layer_attn_processed = avg_heads.clone()
        layer_attn_processed[mask] = 0

        # Add identity matrix to allow self-connections and direct paths
        # This ensures that even if a token doesn't attend to others, its own importance propagates.
        layer_attn_processed = layer_attn_processed + torch.eye(seq_len, device=mat.device).unsqueeze(0)

        # Normalize the attention weights so that each row sums to 1
        # This is crucial for proper propagation of attention flow
        layer_attn_processed = layer_attn_processed / layer_attn_processed.sum(dim=-1, keepdim=True)

        # Multiply with the cumulative attention flow
        # This propagates the attention from previous layers to the current layer
        result = result @ layer_attn_processed
        
    return result  # [B, N, N]
