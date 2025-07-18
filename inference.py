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
import argparse
from pathlib import Path

# Assuming these imports are available in your project structure
from config import load_config, DEVICE, IMAGENET_MEAN, IMAGENET_STD # Import IMAGENET_MEAN/STD from config
from core_models.models import PerceptionModule, LitBongard # Updated import path
from src.data import get_dataloader # Only get_dataloader is needed here, others are for training
from src.bongard_rules import ALL_BONGARD_RULES # Updated import path
from src.xai import generate_grad_cam, rollout_attention # Updated import path
from src.utils import _calculate_iou # For IoU calculation
from src.sam_utils import get_masked_crop # For object cropping

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Placeholder for TensorRT imports. TensorRT needs to be installed.
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    logger.warning("TensorRT not found. TensorRT export functionality will be unavailable.")

# Placeholder for Captum (XAI for GNNs) imports. Captum needs to be installed.
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    logger.warning("Captum not found. GNN XAI functionality will be unavailable.")

def load_model_for_inference(cfg: Dict[str, Any], checkpoint_path: str) -> nn.Module:
    """
    Loads a trained model for inference.
    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        checkpoint_path (str): Path to the model checkpoint.
    Returns:
        nn.Module: The loaded model in evaluation mode.
    """
    # Initialize PerceptionModule, as LitBongard wraps it
    model = PerceptionModule(cfg).to(DEVICE)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if 'state_dict' in checkpoint:
                # Remove 'perception_module.' prefix if loading into bare PerceptionModule
                # And handle 'model.' prefix from LightningModule if present
                model_state_dict = {k.replace('perception_module.', '').replace('model.', ''): v 
                                    for k, v in checkpoint['state_dict'].items() 
                                    if k.startswith('perception_module.') or k.startswith('model.')}
                model.load_state_dict(model_state_dict)
                logger.info(f"Loaded PerceptionModule state_dict from Lightning checkpoint: {checkpoint_path}")
            else:
                # If it's a raw PerceptionModule state dict
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded raw model state_dict from: {checkpoint_path}")
            model.eval()  # Set to eval mode
        except Exception as e:
            logger.error(f"Error loading model checkpoint from {checkpoint_path}: {e}. Returning uninitialized model.")
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Returning uninitialized model.")
    return model

def preprocess_image(image_path: str, image_size: Tuple[int, int] = (224, 224)) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Loads and preprocesses a single image for model inference.
    Returns the preprocessed tensor and the original image as a NumPy array.
    Args:
        image_path (str): Path to the image file.
        image_size (Tuple[int, int]): Target size for resizing (height, width).
    Returns:
        Tuple[torch.Tensor, np.ndarray]: Preprocessed image tensor (1, C, H, W) and original image as NumPy array (H, W, C).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        original_image_np = np.array(image) # Keep original for scene graph building
        
        dynamic_preprocess_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        input_tensor = dynamic_preprocess_transform(image).unsqueeze(0).to(DEVICE)
        return input_tensor, original_image_np
    except FileNotFoundError:
        logger.error(f"Image not found at {image_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return None, None

def predict_bongard_problem(model: nn.Module, cfg: Dict[str, Any], problem_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs inference on a single Bongard problem using the PerceptionModule.
    Args:
        model (nn.Module): The loaded Bongard solver model (PerceptionModule).
        cfg (Dict[str, Any]): Configuration dictionary.
        problem_data (Dict[str, Any]): Dictionary containing paths to positive and negative images.
                                        e.g., {'positive': ['pos1.png', 'pos2.png'], 'negative': ['neg1.png', 'neg2.png']}
    Returns:
        Dict[str, Any]: Prediction results, including logits and predicted class.
    """
    model.eval() # Ensure model is in evaluation mode

    # Prepare query images (positive and negative examples are treated as query images here)
    query_tensors = []
    query_original_nps = []
    query_gts_json_strings = [] # Dummy GTs for inference
    
    for img_path in problem_data['positive'] + problem_data['negative']:
        tensor, original_np = preprocess_image(img_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
        if tensor is not None and original_np is not None:
            query_tensors.append(tensor)
            query_original_nps.append(original_np)
            query_gts_json_strings.append(b'{}') # Dummy GT for inference
        else:
            logger.error(f"Failed to load image {img_path} for Bongard problem.")
            return {"error": "Failed to load images"}
    
    if not query_tensors:
        logger.error("No query images loaded for the Bongard problem.")
        return {"error": "No query images"}

    query_batch = torch.cat(query_tensors, dim=0) # [num_query_imgs, C, H, W]

    # Dummy detected bboxes and masks for inference.
    # In a real application, you would run an object detector/segmenter here.
    dummy_detected_bboxes_batch = [[] for _ in range(query_batch.shape[0])]
    dummy_detected_masks_batch = [[] for _ in range(query_batch.shape[0])]

    # Dummy support images and labels for inference (if few-shot is enabled in model)
    # The PerceptionModule expects these, even if empty.
    num_support_per_problem = cfg['data']['synthetic_data_config'].get('max_support_images_per_problem', 0)
    dummy_support_images = torch.zeros(query_batch.shape[0], num_support_per_problem, 3, cfg['data']['image_size'], cfg['data']['image_size']).to(DEVICE)
    dummy_support_labels = torch.zeros(query_batch.shape[0], num_support_per_problem, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        # The PerceptionModule's forward method is designed for a batch of query images
        # and potentially a batch of support images.
        outputs = model(
            images=query_batch,
            ground_truth_json_strings=query_gts_json_strings,
            detected_bboxes_batch=dummy_detected_bboxes_batch,
            detected_masks_batch=dummy_detected_masks_batch,
            support_images=dummy_support_images,
            support_labels_flat=dummy_support_labels
        )
        
        bongard_logits = outputs['bongard_logits']
        # The BongardHead in PerceptionModule already outputs 2 classes (True/False for problem)
        # So, the logits are already for the overall problem.
        
        probabilities = F.softmax(bongard_logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item() # This will be 0 or 1 for the problem itself

    return {
        "logits": bongard_logits.cpu().numpy().tolist(),
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
        attn_weights = []
        # Check if the PerceptionModule's backbone (e.g., AttributeClassifier)
        # or another part of the model has attention weights.
        # This part might need to be more specific to your model's internal structure.
        if hasattr(model, 'attribute_model') and hasattr(model.attribute_model.feature_extractor, 'get_attention_weights') and callable(model.attribute_model.feature_extractor.get_attention_weights):
            attn_weights = model.attribute_model.feature_extractor.get_attention_weights(input_tensor)
        elif hasattr(model, 'get_attention_weights') and callable(model.get_attention_weights):
            attn_weights = model.get_attention_weights(input_tensor)
        else:
            logger.warning("Model or its attribute_model.feature_extractor does not have 'get_attention_weights' method. Cannot perform attention rollout.")
            return
        
        if not attn_weights:
            logger.warning("No attention weights obtained from the model. Skipping attention rollout visualization.")
            return
        
        rollout_matrix = rollout_attention(attn_weights, discard_ratio=discard_ratio) # [B, N, N]
        
        if rollout_matrix.numel() == 0:
            logger.warning("Attention rollout matrix is empty. Skipping visualization.")
            return
        
        if rollout_matrix.shape[1] > 1: # Ensure there are patches beyond CLS token
            attention_to_patches = rollout_matrix[0, 0, 1:] # [num_patches]
            
            num_patches = attention_to_patches.shape[0]
            grid_size = int(np.sqrt(num_patches))
            if grid_size * grid_size != num_patches:
                logger.warning(f"Number of patches ({num_patches}) is not a perfect square. Heatmap reshaping might be inaccurate.")
            
            heatmap = attention_to_patches.reshape(grid_size, grid_size).cpu().numpy()
            
            original_image = Image.open(image_path).convert("RGB")
            original_size = original_image.size # (width, height)
            
            from skimage.transform import resize
            heatmap_resized = resize(heatmap, original_size, anti_aliasing=True)
            
            heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            
            import matplotlib.cm as cm
            cmap = cm.get_cmap('jet')
            heatmap_colored = cmap(heatmap_resized)[:, :, :3] # Take RGB channels
            
            original_image_np = np.array(original_image) / 255.0 # Normalize to [0, 1]
            
            alpha = 0.5 # Transparency of the heatmap
            overlay_image = (original_image_np * (1 - alpha)) + (heatmap_colored * alpha)
            overlay_image = (overlay_image * 255).astype(np.uint8) # Convert back to 0-255 range
            
            os.makedirs(save_dir, exist_ok=True)
            base_filename = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
            output_filename = f"{base_filename}_attention_rollout{file_suffix}.png"
            output_path = os.path.join(save_dir, output_filename)
            Image.fromarray(overlay_image).save(output_path)
            logger.info(f"Attention Rollout heatmap saved to: {output_path}")
        else:
            logger.warning("Attention rollout matrix has no patches to visualize. Skipping heatmap generation.")

def export_onnx(model: nn.Module, dummy_input: torch.Tensor, out_path: str):
    """
    Exports the PyTorch model to ONNX format.
    Args:
        model (nn.Module): The PyTorch model to export.
        dummy_input (torch.Tensor): A dummy input tensor with the expected shape and type.
        out_path (str): The output path for the ONNX file.
    """
    import torch.onnx
    logger.info(f"Exporting model to ONNX at: {out_path}")
    try:
        # The dummy input should match the expected input to PerceptionModule.forward
        # which is (images, gt_json_strings, detected_bboxes_batch, detected_masks_batch, support_images, support_labels_flat)
        # ONNX export typically only handles tensor inputs.
        # You might need to create a wrapper model that only takes image tensors
        # or modify the PerceptionModule.forward for ONNX export.
        # For now, let's assume dummy_input is just the image tensor.
        # If your PerceptionModule has a simpler forward for inference, use that.
        
        # Create dummy inputs for non-image arguments
        dummy_gt_json_strings = [b'{}'] * dummy_input.shape[0]
        dummy_bboxes_batch = [[] for _ in range(dummy_input.shape[0])]
        dummy_masks_batch = [[] for _ in range(dummy_input.shape[0])]
        dummy_support_images = torch.zeros(dummy_input.shape[0], 0, 3, dummy_input.shape[2], dummy_input.shape[3]).to(DEVICE)
        dummy_support_labels = torch.zeros(dummy_input.shape[0], 0, dtype=torch.long).to(DEVICE)

        # Wrap the model for ONNX export if its forward signature is complex
        class ONNXExportWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, images):
                # This simplified forward is for ONNX export
                # It assumes the model can handle dummy bboxes/masks/support internally or they are not used for this path
                # Or, if these are dynamically generated, they won't be part of the ONNX graph.
                # You might need to adjust this based on how your PerceptionModule handles these inputs during inference.
                outputs = self.model(
                    images=images,
                    ground_truth_json_strings=[b'{}'] * images.shape[0], # Dummy
                    detected_bboxes_batch=[[] for _ in range(images.shape[0])], # Dummy
                    detected_masks_batch=[[] for _ in range(images.shape[0])], # Dummy
                    support_images=torch.zeros(images.shape[0], 0, 3, images.shape[2], images.shape[3]).to(images.device), # Dummy
                    support_labels_flat=torch.zeros(images.shape[0], 0, dtype=torch.long).to(images.device) # Dummy
                )
                return outputs['bongard_logits'] # Export only the final logits
        
        wrapped_model = ONNXExportWrapper(model).to(DEVICE)
        
        torch.onnx.export(wrapped_model, dummy_input, out_path,
                          input_names=['input_images'], output_names=['bongard_logits'],
                          dynamic_axes={'input_images':{0:'batch'}, 'bongard_logits':{0:'batch'}},
                          opset_version=11) # Specify opset version for broader compatibility
        logger.info("Model successfully exported to ONNX.")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {e}")

# 13.1 TensorRT Conversion
def export_tensorrt(onnx_path, engine_path):
    """
    Converts an ONNX model to a TensorRT engine.
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path to save the output TensorRT engine.
    """
    if not HAS_TENSORRT:
        logger.error("TensorRT is not installed. Cannot perform ONNX to TensorRT conversion.")
        return
    if not Path(onnx_path).exists():
        logger.error(f"ONNX model not found at: {onnx_path}")
        return
    logger.info(f"Converting ONNX model '{onnx_path}' to TensorRT engine '{engine_path}'...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    try:
        # Use EXPLICIT_BATCH flag for network creation
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            # Parse the ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX model.")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return
            # Configure the builder
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30 # 1GB workspace
            # You might need to set precision (e.g., FP16) based on your needs
            # config.set_flag(trt.BuilderFlag.FP16)
            # Build the engine
            logger.info("Building TensorRT engine...")
            engine = builder.build_engine(network, config)
            if engine is None:
                logger.error("Failed to build TensorRT engine.")
                return
            # Serialize and save the engine
            engine_path_obj = Path(engine_path)
            engine_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(engine_path_obj, 'wb') as f:
                f.write(engine.serialize())
            logger.info(f"TensorRT engine saved to: {engine_path}")
    except Exception as e:
        logger.error(f"An error occurred during TensorRT conversion: {e}", exc_info=True)

# 14.1 Integrated Gradients for GNN Node Embeddings (exposed via --xai-gnn flag)
def explain_gnn(model, node_feats, edge_index, target_edge):
    """
    Applies Integrated Gradients to a GNN model to explain node embeddings.
    This is a conceptual implementation and requires a compatible GNN model
    and data structure.
    Args:
        model (torch.nn.Module): The GNN model.
        node_feats (torch.Tensor): Node features.
        edge_index (torch.Tensor): Edge index (adjacency list).
        target_edge (tuple): The target edge (e.g., (node_idx1, node_idx2)) for attribution.
    Returns:
        torch.Tensor: Attributions for node features.
    """
    if not HAS_CAPTUM:
        logger.error("Captum is not installed. Cannot perform GNN XAI.")
        return None
    
    logger.info(f"Applying Integrated Gradients for GNN explanation on target edge: {target_edge}")
    try:
        ig = IntegratedGradients(model)
        # The 'target' argument for IntegratedGradients.attribute depends on how
        # your GNN model's forward pass returns output and which part you want to explain.
        # For a target edge, you might need to ensure your model's forward pass
        # can output a specific scalar related to that edge (e.g., a prediction score).
        # This is a simplified example.
        attributions, delta = ig.attribute(
            (node_feats, edge_index), 
            target=target_edge, # This needs to be an index or tuple of indices if model output is multi-dimensional
            return_convergence_delta=True
        )
        logger.info("Integrated Gradients attributions computed.")
        return attributions
    except Exception as e:
        logger.error(f"Error during GNN explanation with Integrated Gradients: {e}", exc_info=True)
        return None

# Main inference function
def run_inference(cfg: Dict[str, Any], model_path: str, problem_data_path: str, visualize_xai: bool = False, export_onnx_model: bool = False, export_tensorrt_engine: Optional[str] = None, xai_gnn_enabled: bool = False, gnn_model_path: Optional[str] = None, node_features_path: Optional[str] = None, edge_index_path: Optional[str] = None, target_edge: Optional[Tuple[int, int]] = None):
    """
    Runs inference on Bongard problems and optionally generates XAI visualizations,
    exports to ONNX, exports to TensorRT, or performs GNN XAI.
    Args:
        cfg (Dict[str, Any]): Configuration dictionary.
        model_path (str): Path to the trained model checkpoint.
        problem_data_path (str): Path to a JSON file defining Bongard problems for inference.
                                 Format: {'problem_id': {'positive': [...], 'negative': [...]}}
        visualize_xai (bool): If True, generate XAI visualizations (Grad-CAM, Attention Rollout).
        export_onnx_model (bool): If True, export the model to ONNX format.
        export_tensorrt_engine (Optional[str]): If provided, path to save the TensorRT engine.
        xai_gnn_enabled (bool): If True, enable GNN XAI (Integrated Gradients).
        gnn_model_path (Optional[str]): Path to the GNN model for XAI.
        node_features_path (Optional[str]): Path to node features (e.g., .npy file).
        edge_index_path (Optional[str]): Path to edge index (e.g., .npy file).
        target_edge (Optional[Tuple[int, int]]): Target edge for GNN XAI (e.g., (node_idx1, node_idx2)).
    """
    logger.info(f"--- Starting Inference with model: {model_path} ---")
    
    model = load_model_for_inference(cfg, model_path)
    if model is None:
        logger.error("Failed to load model. Exiting inference.")
        return
    
    # ONNX Export
    if export_onnx_model:
        # Create a dummy input based on expected image size and channels
        dummy_input = torch.randn(1, 3, cfg['data']['image_size'], cfg['data']['image_size']).to(DEVICE)
        onnx_output_path = cfg['onnx']['path'] # Assuming 'onnx' and 'path' are in config
        os.makedirs(os.path.dirname(onnx_output_path) or './', exist_ok=True)
        export_onnx(model, dummy_input, onnx_output_path)
    
    # TensorRT Export
    if export_tensorrt_engine:
        onnx_path_for_trt = cfg['onnx']['path'] if export_onnx_model else None
        if onnx_path_for_trt and Path(onnx_path_for_trt).exists():
            export_tensorrt(onnx_path_for_trt, export_tensorrt_engine)
        else:
            logger.error("Cannot export to TensorRT. ONNX model not available or not exported.")

    # GNN XAI
    if xai_gnn_enabled:
        if not all([gnn_model_path, node_features_path, edge_index_path, target_edge]):
            logger.error("Missing arguments for GNN XAI. Please provide --gnn-model-path, --node-features-path, --edge-index-path, and --target-edge.")
        else:
            try:
                # Placeholder for loading GNN model and data
                # In a real application, you would load your actual GNN model and data here
                # Example: gnn_model = torch.load(gnn_model_path)
                # node_feats = torch.from_numpy(np.load(node_features_path))
                # edge_index = torch.from_numpy(np.load(edge_index_path))
                # For demonstration, creating a dummy GNN model and data
                class DummyGNN(nn.Module):
                    def forward(self, node_feats, edge_index):
                        return node_feats.sum(dim=1) # Example output
                
                dummy_gnn_model = DummyGNN()
                dummy_node_feats = torch.randn(10, 5, requires_grad=True) # 10 nodes, 5 features
                dummy_edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long) # Dummy edge index
                attributions = explain_gnn(dummy_gnn_model, dummy_node_feats, dummy_edge_index, target_edge)
                if attributions is not None:
                    logger.info(f"GNN Attributions: {attributions}")
            except Exception as e:
                logger.error(f"Error preparing for GNN XAI: {e}", exc_info=True)

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
                input_tensor, _ = preprocess_image(sample_image_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
                if input_tensor is not None:
                    # Find a suitable target layer for Grad-CAM
                    target_layer = None
                    if hasattr(model, 'attribute_model') and isinstance(model.attribute_model, nn.Module):
                        # Try to find a conv layer in the backbone of attribute_model
                        for name, module in model.attribute_model.feature_extractor.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)) and not list(module.children()):
                                target_layer = module
                                break
                    if target_layer is None:
                        logger.warning("Could not find a suitable target layer for Grad-CAM in attribute_model. Trying top-level model.")
                        for name, module in model.named_modules():
                            if isinstance(module, (nn.Conv2d, nn.Linear)) and not list(module.children()):
                                target_layer = module
                                break
                    if target_layer:
                        generate_grad_cam(
                            model=model,
                            input_tensor=input_tensor,
                            target_layer=target_layer,
                            target_category=result['predicted_class'],
                            image_path=sample_image_path,
                            save_dir='./xai_outputs/grad_cam',
                            file_suffix=f'_problem_{problem_id}'
                        )
                    else:
                        logger.warning(f"Could not find a suitable target layer for Grad-CAM for problem {problem_id}. Skipping.")
                else:
                    logger.warning(f"Skipping Grad-CAM for problem {problem_id}: Failed to preprocess image {sample_image_path}.")
            
            if cfg['model'].get('use_transformer', False):
                if problem_data['positive']:
                    sample_image_path = problem_data['positive'][0]
                    input_tensor, _ = preprocess_image(sample_image_path, image_size=(cfg['data']['image_size'], cfg['data']['image_size']))
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
    import torchvision.transforms as T # Re-import T for main block usage
    
    parser = argparse.ArgumentParser(description="Run Bongard Solver Inference and XAI.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--problem_data_path", type=str, required=True,
                        help="Path to a JSON file containing Bongard problems for inference.")
    parser.add_argument("--visualize_xai", action="store_true",
                        help="Enable generation of XAI visualizations (Grad-CAM, Attention Rollout).")
    parser.add_argument("--export_onnx", action="store_true",
                        help="Export the model to ONNX format.")
    parser.add_argument("--export_tensorrt", type=str, default=None,
                        help="Path to save the TensorRT engine. Requires --export_onnx to generate ONNX first.")
    parser.add_argument('--xai-gnn', action='store_true', help='Enable GNN XAI (Integrated Gradients).')
    parser.add_argument('--gnn-model-path', type=str, default=None, help='Path to the GNN model for XAI.')
    parser.add_argument('--node-features-path', type=str, default=None, help='Path to node features.')
    parser.add_argument('--edge-index-path', type=str, default=None, help='Path to edge index.')
    parser.add_argument('--target-edge', type=int, nargs=2, default=None, help='Target edge for GNN XAI (e.g., 0 1).')
    args = parser.parse_args()
    
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
                            dummy_img = Image.new('RGB', (cfg['data']['image_size'], cfg['data']['image_size']), color = 'red')
                            os.makedirs(os.path.dirname(img_path), exist_ok=True)
                            dummy_img.save(img_path)
                            logger.info(f"Created dummy image: {img_path}")
                        except Exception as e:
                            logger.error(f"Could not create dummy image {img_path}: {e}")
        with open(args.problem_data_path, 'w') as f:
            json.dump(sample_problems, f, indent=4)
        logger.info(f"Created sample problem data at: {args.problem_data_path}")
    
    # Load config (assuming config.yaml exists or is handled by load_config)
    cfg = load_config(args.config)

    # Run the inference
    run_inference(
        cfg=cfg,
        model_path=args.model_path,
        problem_data_path=args.problem_data_path,
        visualize_xai=args.visualize_xai,
        export_onnx_model=args.export_onnx,
        export_tensorrt_engine=args.export_tensorrt,
        xai_gnn_enabled=args.xai_gnn,
        gnn_model_path=args.gnn_model_path,
        node_features_path=args.node_features_path,
        edge_index_path=args.edge_index_path,
        target_edge=tuple(args.target_edge) if args.target_edge else None
    )
