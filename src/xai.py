# Folder: bongard_solver/src/
# File: xai.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from typing import Dict, Any, Optional, Union, Callable, List

# Import torchvision transforms for consistent preprocessing/denormalization
from torchvision import transforms as T

# Import ImageNet normalization parameters from the project root config
try:
    # Use the 0-1 range for visualization as PIL/numpy expect this for display
    from config import IMAGENET_MEAN_0_1, IMAGENET_STD_0_1
    IMAGENET_MEAN = IMAGENET_MEAN_0_1
    IMAGENET_STD = IMAGENET_STD_0_1
except ImportError:
    logger.error("Could not import IMAGENET_MEAN_0_1, IMAGENET_STD_0_1 from config.py. XAI visualizations might use incorrect normalization.")
    # Fallback to standard ImageNet values (0-1 range) if config is not accessible
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# Import Grad-CAM library components
HAS_GRAD_CAM = False
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
    logger = logging.getLogger(__name__)  # Ensure logger is initialized before use
    logger.info("pytorch-grad-cam library found and enabled.")
except ImportError:
    logger = logging.getLogger(__name__)  # Ensure logger is initialized before use
    logger.warning("pytorch-grad-cam library not found. Grad-CAM visualization will be disabled.")

# Captum is required for Integrated Gradients
HAS_CAPTUM = False
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    logger.warning("Captum not found. GNN XAI functionality will be unavailable.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define a denormalization transform for visualization
# This transform takes a (C, H, W) tensor, denormalizes it,
# converts to (H, W, C) numpy array in [0, 1] range.
denormalize_transform = T.Compose([
    T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in IMAGENET_STD]),   # Undo std
    T.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1.0, 1.0, 1.0]),   # Undo mean
    T.ToPILImage(),   # Convert to PIL Image
    T.ToTensor(),     # Convert back to tensor (float, [0,1], C,H,W)
    lambda x: x.permute(1, 2, 0).cpu().numpy()      # Permute to HWC and convert to numpy
])

def generate_grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,      # Expected: (1, C, H, W) normalized tensor
    target_layer: nn.Module,
    target_category: Optional[int],      # Class index to explain
    image_path: str,     # Path to save the CAM overlay image
    save_dir: str = './grad_cam_outputs',
    file_suffix: str = '',
    use_grad_cam_plus_plus: bool = False,
    reshape_transform: Optional[Callable] = None      # For ViT-like models
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
    grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets)[0]    # [0] because it returns a batch
    
    # Prepare the original image for overlay
    # Convert the input_tensor (normalized, CHW) back to HWC numpy array in [0,1]
    # for show_cam_on_image.
    # We use the denormalize_transform defined above.
    
    # Clone to avoid modifying original tensor and move to CPU if on GPU
    img_for_overlay_tensor = input_tensor.squeeze(0).cpu()      # Remove batch dim, move to CPU
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

# 14.1 Integrated Gradients for GNN Node Embeddings
def explain_gnn(model: nn.Module, node_feats: torch.Tensor, edge_index: torch.Tensor, target_idx: Union[int, Tuple[int, int]]) -> Optional[torch.Tensor]:
    """
    Applies Integrated Gradients to a GNN model to explain node embeddings.
    This is a conceptual implementation and requires a compatible GNN model
    and data structure.
    Args:
        model (torch.nn.Module): The GNN model.
        node_feats (torch.Tensor): Node features.
        edge_index (torch.Tensor): Edge index (adjacency list).
        target_idx (Union[int, Tuple[int, int]]): The target for attribution.
                                                    If int, it's a node index for node-level output.
                                                    If tuple (src_node, dst_node), it's an edge for edge-level output.
                                                    This should correspond to an index in the model's output if
                                                    the model outputs scores per edge/node, or a way to select a scalar output.
    Returns:
        torch.Tensor: Attributions for node features.
    """
    if not HAS_CAPTUM:
        logger.error("Captum is not installed. Cannot perform GNN XAI.")
        return None
    
    logger.info(f"Applying Integrated Gradients for GNN explanation on target: {target_idx}")
    try:
        ig = IntegratedGradients(model)
        
        # Integrated Gradients requires the target to be a scalar output.
        # If your GNN outputs a tensor, you need to specify which element to attribute.
        # For a target_edge, this often means selecting the score for that specific edge.
        # This example assumes the model's forward pass can handle (node_feats, edge_index)
        # and that `target_idx` can be used as a target index for the output.
        # You might need to adapt this based on your actual GNN model's output structure.
        
        # Dummy target function if the model's output is complex
        def forward_with_target_selection(node_feats_input, edge_index_input):
            # This is a placeholder. Adapt this to your actual GNN model's forward pass
            # and how you extract the scalar value you want to explain.
            model_output = model(node_feats_input, edge_index_input)
            
            if isinstance(target_idx, int):  # Explaining a node's output
                if model_output.ndim > 1:  # e.g., [num_nodes, num_classes]
                    # Assuming we want to explain the highest predicted class for this node
                    return model_output[target_idx].max()
                else:  # e.g., [num_nodes]
                    return model_output[target_idx]
            elif isinstance(target_idx, tuple) and len(target_idx) == 2:  # Explaining an edge's output
                # This requires your GNN to output something specific to edges.
                # For example, if it outputs [num_edges, num_relation_types]
                # You'd need to find the index of the target_edge and select its score.
                logger.warning("Edge-level explanation for GNNs is complex and requires specific model output structure.")
                return model_output.sum()  # Fallback to sum for dummy
            else:
                raise ValueError(f"Unsupported target_idx type: {type(target_idx)}")

        # Make node_feats require gradient for attribution
        node_feats_requires_grad = node_feats.clone().detach().requires_grad_(True)
        attributions, delta = ig.attribute(
            inputs=(node_feats_requires_grad, edge_index),
            target=forward_with_target_selection,  # Pass the function that selects scalar output
            return_convergence_delta=True
        )
        logger.info("Integrated Gradients attributions computed.")
        logger.debug(f"Convergence Delta: {delta}")
        return attributions
    except Exception as e:
        logger.error(f"Error during GNN explanation with Integrated Gradients: {e}", exc_info=True)
        return None

def rollout_attention(attns: List[torch.Tensor], discard_ratio: float = 0.9) -> torch.Tensor:
    """
    Computes attention rollout for Transformer-based models.
    Reference: Abnar & Zuidema, “Quantifying Attention Flow in Transformers,” EMNLP 2020.
    
    Args:
        attns (list[torch.Tensor]): A list of attention matrices from each layer.
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
    if not attns:
        logger.warning("No attention matrices provided for attention rollout.")
        return torch.empty(0)  # Return an empty tensor or handle as appropriate
    
    # Initialize the result with an identity matrix, representing direct connections.
    # Shape of result will be [B, N, N]
    batch_size, _, seq_len, _ = attns[0].shape
    result = torch.eye(seq_len, device=attns[0].device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, N]
    for a in attns:  # Iterate through each layer's attention matrix
        # Average attention weights across heads for each layer
        avg = a.mean(dim=1)    # [B, N, N]
        
        # Apply discard ratio: set weakest attention weights to zero
        # Flatten for quantile calculation per batch item
        flat = avg.flatten(start_dim=-2)  # [B, N*N]
        # Calculate quantile threshold for each item in the batch
        # unsqueeze(-1) to make it broadcastable with avg
        threshold = torch.quantile(flat, discard_ratio, dim=-1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        
        # Create a mask for values below the threshold
        mask = avg < threshold
        # Apply the mask: set values below threshold to 0
        avg[mask] = 0
        
        # Add identity matrix to allow self-connections and direct paths
        # This ensures that even if a token doesn't attend to others, its own importance propagates.
        layer_attn_processed = avg + torch.eye(seq_len, device=a.device).unsqueeze(0)
        
        # Normalize the attention weights so that each row sums to 1
        # This is crucial for proper propagation of attention flow
        # Handle potential division by zero if a row sums to zero after thresholding
        row_sums = layer_attn_processed.sum(dim=-1, keepdim=True)
        layer_attn_processed = layer_attn_processed / (row_sums + 1e-8)  # Add small epsilon for stability
        
        # Multiply with the cumulative attention flow
        # This propagates the attention from previous layers to the current layer
        result = result @ layer_attn_processed
        
    return result      # [B, N, N]

if __name__ == '__main__':
    # Example Usage (conceptual - replace with your actual GNN model and data)
    class SimpleGNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim) 
        def forward(self, node_feats, edge_index):
            # Simplified: just a linear layer. A real GNN would use graph convolutions.
            # This example assumes it outputs a scalar for each node.
            return self.linear(node_feats).squeeze(-1)  # Output a scalar for each node

    # Create dummy data
    input_dim = 5
    output_dim = 1
    dummy_model = SimpleGNN(input_dim, output_dim)
    dummy_node_feats = torch.randn(10, input_dim, requires_grad=True)  # 10 nodes, 5 features
    dummy_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Dummy edges
    
    # Target node for explanation (e.g., explaining the output for node 0)
    target_node_idx = 0 
    attributions = explain_gnn(dummy_model, dummy_node_feats, dummy_edge_index, target_node_idx)
    if attributions is not None:
        logger.info(f"Attributions for node {target_node_idx}: {attributions}")
    else:
        logger.info("GNN attributions could not be computed.")

    # Example for Attention Rollout
    logger.info("\n--- Attention Rollout Example ---")
    # Simulate attention matrices from 3 layers, 4 heads, sequence length 10 (1 CLS + 9 patches)
    dummy_attns = [
        torch.randn(1, 4, 10, 10),  # Layer 1
        torch.randn(1, 4, 10, 10),  # Layer 2
        torch.randn(1, 4, 10, 10)   # Layer 3
    ]
    # Normalize dummy attention matrices (each row sums to 1)
    dummy_attns = [F.softmax(attn, dim=-1) for attn in dummy_attns]
    rollout_map = rollout_attention(dummy_attns, discard_ratio=0.5)
    if rollout_map.numel() > 0:
        logger.info(f"Attention Rollout Map shape: {rollout_map.shape}")
        # Example: Attention from CLS token (index 0) to all other tokens
        logger.info(f"Attention from CLS token to other tokens:\n{rollout_map[0, 0, 1:]}")
    else:
        logger.info("Attention Rollout map is empty.")

