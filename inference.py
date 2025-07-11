# Folder: bongard_solver/
# File: inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple

# Assuming these imports are available in your project structure
from config import load_config, DEVICE
from models import PerceptionModule, LitBongard # Your model classes
from data import get_dataloader, build_dali_image_processor # For data loading and preprocessing
from data import BongardSyntheticDataset, RealBongardDataset, BongardGenerator
from bongard_rules import ALL_BONGARD_RULES

# Import XAI functions
from xai import generate_grad_cam, attention_rollout # Import the attention_rollout function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define ImageNet normalization parameters (assuming your model was trained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Define a preprocessing transform for inference
preprocess_transform = T.Compose([
    T.Resize((224, 224)), # Resize to model's expected input size
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_model_for_inference(cfg: Dict[str, Any], checkpoint_path: str) -> nn.Module:
    """
    Loads a trained model for inference.
    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        checkpoint_path (str): Path to the model checkpoint.
    Returns:
        nn.Module: The loaded model in evaluation mode.
    """
    model = PerceptionModule(cfg).to(DEVICE)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if 'state_dict' in checkpoint:
                # Remove 'perception_module.' prefix if loading into bare PerceptionModule
                model_state_dict = {k.replace('perception_module.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('perception_module.')}
                model.load_state_dict(model_state_dict)
                logger.info(f"Loaded PerceptionModule state_dict from Lightning checkpoint: {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded raw model state_dict from: {checkpoint_path}")
            model.eval() # Set to eval mode
        except Exception as e:
            logger.error(f"Error loading model checkpoint from {checkpoint_path}: {e}. Returning uninitialized model.")
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Returning uninitialized model.")
    return model

def preprocess_image(image_path: str, image_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Loads and preprocesses a single image for model inference.
    Args:
        image_path (str): Path to the image file.
        image_size (Tuple[int, int]): Target size for resizing (height, width).
    Returns:
        torch.Tensor: Preprocessed image tensor (1, C, H, W).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Create a dynamic preprocess transform based on the desired image_size
        dynamic_preprocess_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        input_tensor = dynamic_preprocess_transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension and move to device
        return input_tensor
    except FileNotFoundError:
        logger.error(f"Image not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_bongard_problem(model: nn.Module, cfg: Dict[str, Any], problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs inference on a single Bongard problem.
    Args:
        model (nn.Module): The loaded Bongard solver model.
        cfg (Dict[str, Any]): Configuration dictionary.
        problem_data (Dict[str, Any]): Dictionary containing paths to positive and negative images.
                                        e.g., {'positive': ['pos1.png', 'pos2.png'], 'negative': ['neg1.png', 'neg2.png']}
    Returns:
        Dict[str, Any]: Prediction results, including logits and predicted class.
    """
    model.eval() # Ensure model is in evaluation mode

    positive_tensors = []
    for img_path in problem_data['positive']:
        tensor = preprocess_image(img_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
        if tensor is not None:
            positive_tensors.append(tensor)
    
    negative_tensors = []
    for img_path in problem_data['negative']:
        tensor = preprocess_image(img_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
        if tensor is not None:
            negative_tensors.append(tensor)

    if not positive_tensors or not negative_tensors:
        logger.error("Failed to load all images for the Bongard problem.")
        return {"error": "Failed to load images"}

    # Concatenate all positive and negative images into single tensors for batch processing
    # Assuming your model expects a batch of positive and a batch of negative images
    positive_batch = torch.cat(positive_tensors, dim=0) # [num_pos, C, H, W]
    negative_batch = torch.cat(negative_tensors, dim=0) # [num_neg, C, H, W]

    with torch.no_grad():
        # The model's forward pass for a Bongard problem should accept these batches
        # and return logits for the Bongard problem (e.g., 2 classes: True/False)
        # Adjust this call based on your LitBongard or PerceptionModule's forward method
        # For LitBongard, you might call model.forward_inference(positive_batch, negative_batch)
        # For PerceptionModule, you might process images separately and then combine features.
        
        # Example for a LitBongard-like model:
        if isinstance(model, LitBongard):
            output = model.forward_inference(positive_batch, negative_batch)
            logits = output['bongard_logits'] if isinstance(output, dict) else output # Assuming logits are directly returned or in a dict
        elif isinstance(model, PerceptionModule):
            # If PerceptionModule, you'd process images to get features then combine them
            # This is a simplified example; actual logic depends on your model's architecture
            pos_features = model(positive_batch)
            neg_features = model(negative_batch)
            # Then combine features and pass through a final classification head
            # For demonstration, let's assume a simple mean pooling and a linear layer
            combined_features = torch.cat([pos_features.mean(dim=0), neg_features.mean(dim=0)], dim=0).unsqueeze(0)
            # You'll need a classification head here, or integrate this into your PerceptionModule
            # For now, let's just return dummy logits
            logits = torch.randn(1, 2).to(DEVICE) # Dummy logits
            logger.warning("PerceptionModule inference path needs a proper classification head implementation.")
        else:
            logger.error("Model type not supported for Bongard problem inference.")
            return {"error": "Unsupported model type"}

        probabilities = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    return {
        "logits": logits.cpu().numpy().tolist(),
        "probabilities": probabilities.cpu().numpy().tolist(),
        "predicted_class": predicted_class # 0 or 1, representing negative or positive Bongard solution
    }

def visualize_attention_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image_path: str,
    save_dir: str = './attention_rollout_outputs',
    file_suffix: str = '',
    discard_ratio: float = 0.9
):
    """
    Generates and saves an attention rollout heatmap for Transformer-based models.
    Assumes the model has a method `get_attention_weights(input_tensor)`
    that returns a list of attention matrices.
    
    Args:
        model (nn.Module): The Transformer-based model.
        input_tensor (torch.Tensor): Preprocessed input image tensor (1, C, H, W).
        image_path (str): Original image path for filename reference.
        save_dir (str): Directory to save the heatmap.
        file_suffix (str): Suffix for the saved filename.
        discard_ratio (float): Discard ratio for attention_rollout.
    """
    model.eval()
    with torch.no_grad():
        # Assuming the model's forward pass or a specific method returns attention weights
        # This part needs to be adapted to how your model exposes attention weights.
        # For example, if your model is a ViT, it might have a method like:
        # `attn_weights = model.get_attention_weights(input_tensor)`
        
        # Dummy attention weights for demonstration if model.get_attention_weights is not implemented
        # In a real scenario, you'd get these from your actual model.
        # Example: if your model has a list of transformer layers, and each layer has an attention module.
        # This requires modifying your model to expose these.
        
        # For a ViT-like model, the input_tensor might first be passed through a patch embedding layer
        # to get tokens, then these tokens pass through transformer blocks.
        # The attention_rollout expects attention matrices from each layer.
        
        # Example of how you might get attention weights if your model is a custom ViT:
        # If your model has a `forward_with_attention` method or hooks for attention
        
        # Placeholder: You need to implement `model.get_attention_weights(input_tensor)`
        # in your model definition (e.g., PerceptionModule if it contains a Transformer)
        # This method should return a list of attention tensors, e.g., from each Transformer block.
        
        # For demonstration, let's assume `model` directly has a `get_attention_weights` method
        # that returns a list of [B, heads, N, N] tensors.
        
        # If your model is a LitBongard, you might need to access its internal perception_module
        # and then call a method on that module to get attention weights.
        
        attn_weights = []
        if hasattr(model, 'get_attention_weights') and callable(model.get_attention_weights):
            attn_weights = model.get_attention_weights(input_tensor)
        else:
            logger.warning("Model does not have 'get_attention_weights' method. Cannot perform attention rollout.")
            return

        if not attn_weights:
            logger.warning("No attention weights obtained from the model. Skipping attention rollout visualization.")
            return

        rollout_matrix = attention_rollout(attn_weights, discard_ratio=discard_ratio) # [B, N, N]

        if rollout_matrix.numel() == 0:
            logger.warning("Attention rollout matrix is empty. Skipping visualization.")
            return

        # Assuming we are interested in the attention from the CLS token (first token, index 0)
        # to all other tokens (patches).
        # The CLS token usually aggregates global information.
        # If your model doesn't use a CLS token, you might average across all tokens or pick a specific one.
        
        # For a single image (batch size 1), take the first item
        # And take the first row (CLS token's attention to all patches)
        # Assuming N is (num_patches + 1) where index 0 is CLS token.
        # If no CLS token, you might average `rollout_matrix.mean(dim=1)` for global importance.
        
        # If the model uses a CLS token, the first row of rollout_matrix[0]
        # represents the importance of each patch for the overall classification.
        # If there's no CLS token, you might sum or average over a relevant dimension.
        
        # Example for a ViT-like model with CLS token:
        # The attention map for the image is usually derived from the CLS token's attention to patches.
        # The first token (index 0) is typically the CLS token.
        # The remaining N-1 tokens correspond to image patches.
        
        # Reshape the attention scores to a 2D heatmap
        # Assuming square patches for simplicity, e.g., 14x14 patches for 224x224 image
        # N = num_patches + 1 => num_patches = N - 1
        # Side length of patch grid = sqrt(num_patches)
        
        if rollout_matrix.shape[1] > 1: # Ensure there are patches beyond CLS token
            # Extract attention from CLS token to patches
            # Assuming CLS token is at index 0, patches start from index 1
            # If no CLS token, you might need to adapt this logic, e.g., sum attention to all tokens.
            attention_to_patches = rollout_matrix[0, 0, 1:] # [num_patches]
            
            # Calculate grid size for patches
            num_patches = attention_to_patches.shape[0]
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                logger.warning(f"Number of patches ({num_patches}) is not a perfect square. Heatmap reshaping might be inaccurate.")
            
            # Reshape to a square heatmap
            heatmap = attention_to_patches.reshape(grid_size, grid_size).cpu().numpy()
            
            # Resize heatmap to original image dimensions for visualization
            original_image = Image.open(image_path).convert("RGB")
            original_size = original_image.size # (width, height)
            
            # Resize heatmap to match original image size
            from skimage.transform import resize
            heatmap_resized = resize(heatmap, original_size, anti_aliasing=True)
            
            # Normalize heatmap to [0, 1]
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            
            # Apply a colormap (e.g., jet)
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')
            heatmap_colored = cmap(heatmap_resized)[:, :, :3] # Take RGB channels
            
            # Overlay heatmap on original image
            original_image_np = np.array(original_image) / 255.0 # Normalize to [0, 1]
            
            # Blend the image and heatmap
            alpha = 0.5 # Transparency of the heatmap
            overlay_image = (original_image_np * (1 - alpha)) + (heatmap_colored * alpha)
            overlay_image = (overlay_image * 255).astype(np.uint8) # Convert back to 0-255 range
            
            # Save the output image
            os.makedirs(save_dir, exist_ok=True)
            base_filename = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
            output_filename = f"{base_filename}_attention_rollout{file_suffix}.png"
            output_path = os.path.join(save_dir, output_filename)
            Image.fromarray(overlay_image).save(output_path)
            logger.info(f"Attention Rollout heatmap saved to: {output_path}")
        else:
            logger.warning("Attention rollout matrix has no patches to visualize. Skipping heatmap generation.")

# Main inference function
def run_inference(cfg: Dict[str, Any], model_path: str, problem_data_path: str, visualize_xai: bool = False):
    """
    Runs inference on Bongard problems and optionally generates XAI visualizations.
    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        model_path (str): Path to the trained model checkpoint.
        problem_data_path (str): Path to a JSON file defining Bongard problems for inference.
                                 Format: {'problem_id': {'positive': [...], 'negative': [...]}}
        visualize_xai (bool): If True, generate Grad-CAM and Attention Rollout visualizations.
    """
    logger.info(f"--- Starting Inference with model: {model_path} ---")
    
    model = load_model_for_inference(cfg, model_path)
    if model is None:
        logger.error("Failed to load model. Exiting inference.")
        return

    try:
        with open(problem_data_path, 'r') as f:
            all_problems_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Problem data file not found at {problem_data_path}. Exiting inference.")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {problem_data_path}. Exiting inference.")
        return

    inference_results = {}
    for problem_id, problem_data in all_problems_data.items():
        logger.info(f"Processing Bongard Problem: {problem_id}")
        
        result = predict_bongard_problem(model, cfg, problem_data)
        inference_results[problem_id] = result
        
        if "error" in result:
            logger.error(f"Error for problem {problem_id}: {result['error']}")
            continue

        logger.info(f"Problem {problem_id} - Predicted Class: {result['predicted_class']} (Probabilities: {result['probabilities']})")

        if visualize_xai:
            logger.info(f"Generating XAI visualizations for Problem: {problem_id}")
            # For Grad-CAM, we need a single image and a target layer.
            # Let's pick the first positive image for Grad-CAM example.
            if problem_data['positive']:
                sample_image_path = problem_data['positive'][0]
                input_tensor = preprocess_image(sample_image_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
                if input_tensor is not None:
                    # Find a suitable target layer for Grad-CAM
                    # This depends on your model's architecture.
                    # Example: if your model has a 'features' block with a final conv layer.
                    target_layer = None
                    if hasattr(model, 'perception_module') and isinstance(model.perception_module, nn.Module):
                        # Try to find the last convolutional layer in the perception module
                        for name, module in model.perception_module.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)) and not list(module.children()): # Check if it's a leaf module
                                target_layer = module
                                # You might want to be more specific, e.g., target the last conv block
                                # For a ResNet, it might be model.perception_module.layer4[-1]
                                # For a ViT, it might be the last attention block's output
                        if target_layer is None:
                            logger.warning("Could not find a suitable target layer for Grad-CAM in perception_module. Trying top-level model.")
                            # Fallback to model's last layer if perception_module specific layer not found
                            for name, module in model.named_modules():
                                if isinstance(module, (nn.Conv2d, nn.Linear)) and not list(module.children()):
                                    target_layer = module
                                    break

                    if target_layer:
                        generate_grad_cam(
                            model=model,
                            input_tensor=input_tensor,
                            target_layer=target_layer,
                            target_category=result['predicted_class'], # Explain the predicted class
                            image_path=sample_image_path,
                            save_dir='./xai_outputs/grad_cam',
                            file_suffix=f'_problem_{problem_id}'
                        )
                    else:
                        logger.warning(f"Could not find a suitable target layer for Grad-CAM for problem {problem_id}. Skipping.")
                else:
                    logger.warning(f"Skipping Grad-CAM for problem {problem_id}: Failed to preprocess image {sample_image_path}.")

            # For Attention Rollout, it's typically for Transformer-based models.
            # This requires your model to expose attention weights.
            # Assuming your model's forward pass (or a specific method) collects attention weights.
            # If your model is not a Transformer, this visualization won't apply.
            if cfg['model'].get('use_transformer', False): # Check if transformer is enabled in config
                # For attention rollout, we might want to visualize for a specific image,
                # or perhaps an aggregated view. Let's use the first positive image again.
                if problem_data['positive']:
                    sample_image_path = problem_data['positive'][0]
                    input_tensor = preprocess_image(sample_image_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
                    if input_tensor is not None:
                        visualize_attention_heatmap(
                            model=model,
                            input_tensor=input_tensor,
                            image_path=sample_image_path,
                            save_dir='./xai_outputs/attention_rollout',
                            file_suffix=f'_problem_{problem_id}'
                        )
                    else:
                        logger.warning(f"Skipping Attention Rollout for problem {problem_id}: Failed to preprocess image {sample_image_path}.")
                else:
                    logger.warning(f"Skipping Attention Rollout for problem {problem_id}: No positive images found.")
            else:
                logger.info(f"Skipping Attention Rollout for problem {problem_id}: Transformer not enabled in model config.")

    logger.info("--- Inference completed. ---")
    return inference_results

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run Bongard Solver Inference and XAI.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--problem_data_path", type=str, required=True,
                        help="Path to a JSON file containing Bongard problems for inference.")
    parser.add_argument("--visualize_xai", action="store_true",
                        help="Enable generation of XAI visualizations (Grad-CAM, Attention Rollout).")
    
    args = parser.parse_args()

    # Example usage:
    # python inference.py --config config.yaml --model_path ./checkpoints/optimized_bongard_model.pth --problem_data_path ./data/sample_problems.json --visualize_xai

    # Ensure your config.yaml has:
    # data:
    #   image_size: 224
    # model:
    #   use_transformer: True # Set to True if your model uses a Transformer and you want attention rollout
    #   # ... other model specific configs
    # debug:
    #   save_model_checkpoints: "./checkpoints" # Ensure this path exists

    # Create dummy directories if they don't exist for testing
    os.makedirs("./xai_outputs/grad_cam", exist_ok=True)
    os.makedirs("./xai_outputs/attention_rollout", exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path) or './', exist_ok=True) # Ensure model dir exists

    # Create a dummy problem_data.json for testing if it doesn't exist
    if not os.path.exists(args.problem_data_path):
        logger.warning(f"Dummy problem data file not found at {args.problem_data_path}. Creating a sample.")
        sample_problems = {
            "problem_1": {
                "positive": ["./data/sample_images/pos_1_1.png", "./data/sample_images/pos_1_2.png"],
                "negative": ["./data/sample_images/neg_1_1.png", "./data/sample_images/neg_1_2.png"]
            },
            "problem_2": {
                "positive": ["./data/sample_images/pos_2_1.png"],
                "negative": ["./data/sample_images/neg_2_1.png"]
            }
        }
        os.makedirs("./data/sample_images", exist_ok=True)
        # Create dummy image files for the sample problems
        for problem_id, data in sample_problems.items():
            for img_list in [data['positive'], data['negative']]:
                for img_path in img_list:
                    if not os.path.exists(img_path):
                        try:
                            dummy_img = Image.new('RGB', (224, 224), color = 'red')
                            os.makedirs(os.path.dirname(img_path), exist_ok=True)
                            dummy_img.save(img_path)
                            logger.info(f"Created dummy image: {img_path}")
                        except Exception as e:
                            logger.error(f"Could not create dummy image {img_path}: {e}")

        with open(args.problem_data_path, 'w') as f:
            json.dump(sample_problems, f, indent=4)
        logger.info(f"Created sample problem data at: {args.problem_data_path}")
    
    # Run the inference
    run_inference(load_config(args.config), args.model_path, args.problem_data_path, args.visualize_xai)
