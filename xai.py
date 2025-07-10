# Folder: bongard_solver/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from PIL import Image
from typing import Dict, Any, Optional, Union

# Import torchvision transforms for consistent preprocessing/denormalization
from torchvision import transforms as T

# Import Grad-CAM library components
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
    logger.info("pytorch-grad-cam library found and enabled.")
except ImportError:
    logger.warning("pytorch-grad-cam library not found. Grad-CAM visualization will be disabled.")
    HAS_GRAD_CAM = False

logger = logging.getLogger(__name__)

# Define ImageNet normalization parameters (assuming your model was trained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define a denormalization transform for visualization
# This transform takes a (C, H, W) tensor, denormalizes it,
# converts to (H, W, C) numpy array in [0, 1] range.
denormalize_transform = T.Compose([
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in IMAGENET_STD]), # Undo std
    T.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1.0, 1.0, 1.0]), # Undo mean
    T.ToPILImage(), # Convert to PIL Image
    T.ToTensor(), # Convert back to tensor (float, [0,1], C,H,W)
    lambda x: x.permute(1, 2, 0).cpu().numpy() # Permute to HWC and convert to numpy
])


def generate_grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor, # Expected: (1, C, H, W) normalized tensor
    target_layer: nn.Module,
    target_category: Optional[int], # Class index to explain
    image_path: str, # Path to save the CAM overlay image
    save_dir: str = './grad_cam_outputs',
    file_suffix: str = '',
    use_grad_cam_plus_plus: bool = False,
    reshape_transform: Optional[Any] = None # For ViT-like models
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
        reshape_transform (Optional[Any]): A function to reshape activations for non-CNN models (e.g., ViT).
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
    grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets)[0] # [0] because it returns a batch

    # Prepare the original image for overlay
    # Convert the input_tensor (normalized, CHW) back to HWC numpy array in [0,1]
    # for show_cam_on_image.
    # We use the denormalize_transform defined above.
    
    # Clone to avoid modifying original tensor and move to CPU if on GPU
    img_for_overlay_tensor = input_tensor.squeeze(0).cpu() # Remove batch dim, move to CPU
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

