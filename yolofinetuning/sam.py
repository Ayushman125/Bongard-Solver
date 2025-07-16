# sam.py
import torch
import os
import cv2
import numpy as np
import networkx as nx
from matplotlib import cm
import logging
import json
from torch.cuda.amp import autocast # New import for mixed precision

logger = logging.getLogger(__name__)

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Attempt to import Bongard symbolic fusion. This is crucial for symbolic labeling.
# If bongard.symbolic_fusion is not available, symbolic labeling will be skipped.
HAS_BONGARD_SYMBOLIC_FUSION = False
try:
    from bongard_symbolic_fusion import symbolic_fusion
    HAS_BONGARD_SYMBOLIC_FUSION = True
    logger.info("bongard_symbolic_fusion.symbolic_fusion found. Symbolic labeling enabled.")
except ImportError:
    symbolic_fusion = None
    logger.warning("bongard_symbolic_fusion.symbolic_fusion not found. Symbolic labeling will be skipped.")
except Exception as e:
    symbolic_fusion = None
    logger.warning(f"Error importing bongard_symbolic_fusion.symbolic_fusion: {e}. Symbolic labeling will be skipped.")

def load_sam_model(
    checkpoint_path="weights/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    device=None
):
    """
    Loads the Segment Anything Model (SAM) and returns a SamPredictor instance.

    Args:
        checkpoint_path (str): Path to the SAM checkpoint (.pth file).
        model_type (str): SAM variant to load. Options: "vit_h", "vit_l", "vit_b".
        device (str or torch.device): Device to load the model on. Defaults to CUDA if available.

    Returns:
        SamPredictor: Ready-to-use predictor for mask generation.
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"SAM checkpoint not found at: {checkpoint_path}. Please ensure it is downloaded.")
        return None # Return None if checkpoint is not found

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        predictor = SamPredictor(sam)
        logger.info(f"SAM Predictor loaded successfully on {device} from {checkpoint_path}")
        return predictor
    except Exception as e:
        logger.error(f"Error loading SAM Predictor: {e}")
        return None

def get_mask_generator(
    checkpoint_path="weights/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    device=None,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100, # Requires opencv-python-headless
):
    """
    Loads the Segment Anything Model (SAM) and returns a SamAutomaticMaskGenerator instance.

    Args:
        checkpoint_path (str): Path to the SAM checkpoint (.pth file).
        model_type (str): SAM variant to load. Options: "vit_h", "vit_l", "vit_b".
        device (str or torch.device): Device to load the model on. Defaults to CUDA if available.
        points_per_side (int): The number of points to sample along one side of the image.
        pred_iou_thresh (float): A filtering threshold in [0,1], for masks with low predicted IoU.
        stability_score_thresh (float): A filtering threshold in [0,1], for masks with low stability.
        crop_n_layers (int): If >0, masks are generated at multiple input image crops and combined.
        crop_n_points_downscale_factor (int): The number of points to sample in each layer of a crop.
        min_mask_region_area (int): If >0, postprocessing will remove small disconnected regions and holes.

    Returns:
        SamAutomaticMaskGenerator: Ready-to-use generator for automatic mask generation.
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"SAM checkpoint not found at: {checkpoint_path}. Please ensure it is downloaded.")
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
        logger.info(f"SAM AutomaticMaskGenerator loaded successfully on {device} from {checkpoint_path}")
        return mask_generator
    except Exception as e:
        logger.error(f"Error loading SAM AutomaticMaskGenerator: {e}")
        return None

def save_mask_png(mask_array: np.ndarray, save_path: str, colorize: bool = True):
    """
    Saves a binary mask as a PNG image. Optionally colorizes it.

    Args:
        mask_array (np.ndarray): A boolean or uint8 binary mask (H, W).
        save_path (str): Full path to save the PNG image.
        colorize (bool): If True, applies a colormap for visualization.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if colorize:
        # Convert boolean mask to float for colormap application
        mask_float = mask_array.astype(np.float32) * 255.0 # Scale to 0-255 for colormap
        # Use viridis colormap, get RGB channels, and scale to 0-255 uint8
        color_mask = cm.viridis(mask_float / 255.0)[:, :, :3]
        color_mask = (color_mask * 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    else:
        # Save as grayscale binary mask (0 or 255)
        cv2.imwrite(save_path, (mask_array * 255).astype(np.uint8))
    logger.debug(f"Mask saved to {save_path}")

def sam_masks_to_yolo(masks: list, image_shape: tuple, class_id: int = 0) -> list[str]:
    """
    Converts SAM masks to YOLO-format annotations.

    Args:
        masks (list): List of mask dicts from SAMAutomaticMaskGenerator.
                      Each dict is expected to have a "segmentation" key with a boolean mask.
        image_shape (tuple): (height, width) of original image.
        class_id (int): Numeric label for all masks (default is 0 for "object").
                        This can be overridden if dynamic class IDs are available.

    Returns:
        List[str]: YOLO annotations in [class x_center y_center width height] format (normalized).
    """
    h, w = image_shape
    annotations = []

    for mask_obj in masks:
        # Ensure segmentation is a boolean array
        seg = mask_obj["segmentation"].astype(np.uint8) * 255 # Convert boolean to 0/255 for findContours
        
        # Find contours from the mask
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Get bounding box for each contour
            x, y, bw, bh = cv2.boundingRect(cnt)
            
            # Normalize coordinates
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            
            # Append YOLO annotation string
            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return annotations

def generate_relation_graph(masks: list) -> dict:
    """
    Generates a relational graph (NetworkX node-link data) based on overlap between SAM masks.
    Nodes are mask indices, edges indicate overlap.

    Args:
        masks (list): List of mask dicts from SAMAutomaticMaskGenerator.
                      Each dict is expected to have a "segmentation" key with a boolean mask.

    Returns:
        dict: NetworkX graph in node-link data format.
    """
    G = nx.Graph()
    for i, m1_dict in enumerate(masks):
        G.add_node(i) # Add node for each mask
        mask1 = m1_dict["segmentation"]
        for j, m2_dict in enumerate(masks[i+1:], start=i+1):
            mask2 = m2_dict["segmentation"]
            
            # Check for overlap: intersection area
            intersection_area = np.logical_and(mask1, mask2).sum()
            
            # Define an overlap threshold (e.g., if intersection area is significant)
            # This threshold can be adjusted based on desired sensitivity
            if intersection_area > 0: # Simple check for any overlap
                # You can add more complex conditions here, e.g., IoU > threshold
                G.add_edge(i, j, relation="overlap", intersection_area=int(intersection_area))
    
    return nx.readwrite.json_graph.node_link_data(G)

def get_symbolic_labels(masks: list, image: np.ndarray, program_text: str) -> list:
    """
    Applies Bongard symbolic reasoning on SAM masks to extract primitive types,
    attributes, and relations.

    Args:
        masks (list): List of mask dicts from SAMAutomaticMaskGenerator.
                      Each dict is expected to have a "segmentation" key with a boolean mask.
        image (np.ndarray): The original image (H, W, C).
        program_text (str): The symbolic rule/program text associated with the image.

    Returns:
        List[dict]: A list of symbolic annotations, one dictionary per object/mask.
                    Each dict contains 'type', 'attributes', 'relations', etc.
    """
    if not HAS_BONGARD_SYMBOLIC_FUSION:
        logger.warning("Bongard symbolic_fusion not available. Skipping symbolic label generation.")
        return [{"error": "symbolic_fusion not available"} for _ in masks]

    try:
        with autocast(): # Apply mixed precision to symbolic fusion if it uses PyTorch models
            # Assuming symbolic_fusion takes a list of masks (boolean arrays), the image,
            # and the program text to derive symbolic properties.
            # The exact signature of symbolic_fusion might need adjustment based on your bongard library.
            symbolic_annotations = symbolic_fusion(
                [m["segmentation"] for m in masks], # Pass boolean masks
                image,
                program_text
            )
        return symbolic_annotations
    except Exception as e:
        logger.error(f"Error during symbolic label generation: {e}")
        return [{"error": str(e)} for _ in masks]

def overlay_symbolic_debugger(image: np.ndarray, masks_list: list[np.ndarray], symbols: list[dict], font_scale: float = 0.5) -> np.ndarray:
    """
    Overlays segmented masks with their class labels/primitive types, attributes,
    and spatial relation arrows for debugging.

    Args:
        image (np.ndarray): The original image (H, W, C).
        masks_list (list[np.ndarray]): List of boolean segmentation arrays for each object.
        symbols (list[dict]): List of symbolic annotations for each object.
        font_scale (float): Scale for text rendering.

    Returns:
        np.ndarray: Image with symbolic overlays for debugging.
    """
    debug_image = image.copy()
    
    # Ensure image is in BGR format for OpenCV drawing
    if debug_image.ndim == 3 and debug_image.shape[2] == 3 and debug_image.dtype == np.uint8:
        # Already BGR or can be treated as such
        pass
    elif debug_image.ndim == 2: # Grayscale
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
    elif debug_image.ndim == 3 and debug_image.shape[2] == 4: # RGBA
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGBA2BGR)

    # Convert symbols to a dictionary for easier lookup by index
    symbol_map = {i: sym for i, sym in enumerate(symbols)}

    for i, mask in enumerate(masks_list):
        if mask.sum() == 0: # Skip empty masks
            continue

        # Convert boolean mask to uint8 for OpenCV operations
        mask_uint8 = mask.astype(np.uint8) * 255

        # Compute centroid of mask for text and arrow origin
        ys, xs = np.where(mask)
        if len(xs) == 0: continue # Should be caught by mask.sum() == 0, but as a safeguard
        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        # Draw bounding box (green)
        x, y, w, h = cv2.boundingRect(mask_uint8)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label (blue text)
        info = symbol_map.get(i, {})
        primitive_type = info.get('type', 'unknown')
        attributes = info.get('attributes', [])
        
        label_text = f"{primitive_type}"
        if attributes:
            label_text += f" ({','.join(attributes)})"
        
        # Position text above the bounding box
        text_pos = (x, y - 5)
        cv2.putText(debug_image, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw relation arrows (red)
        relations = info.get("relations", [])
        for rel in relations:
            target_id = rel.get("target_id")
            relation_type = rel.get("relation")

            if target_id is not None and target_id in symbol_map:
                target_mask = masks_list[target_id]
                tys, txs = np.where(target_mask)
                if len(txs) == 0: continue
                tx, ty = int(np.mean(txs)), int(np.mean(tys))

                # Draw arrow from current object to target object
                cv2.arrowedLine(debug_image, (cx, cy), (tx, ty), (0, 0, 255), 1, tipLength=0.2)
                
                # Place relation text near the middle of the arrow
                rel_text_pos = ((cx + tx) // 2, (cy + ty) // 2 - 5)
                cv2.putText(debug_image, relation_type, rel_text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    
    return debug_image

def generate_reasoning_chain(symbols: list[dict]) -> str:
    """
    Generates a human-readable reasoning chain describing each object and its relations.

    Args:
        symbols (list[dict]): List of symbolic annotations for each object.

    Returns:
        str: A multi-line string representing the reasoning chain.
    """
    chain = []
    for i, sym in enumerate(symbols):
        desc = f"Object {i}: {sym.get('type', 'unknown_type')}"
        
        attributes = sym.get('attributes', [])
        if attributes:
            desc += f" with attributes [{', '.join(attributes)}]"
        
        relations = sym.get("relations")
        if relations:
            rel_descriptions = []
            for rel in relations:
                target_id = rel.get("target_id")
                relation_type = rel.get("relation")
                if target_id is not None and relation_type:
                    rel_descriptions.append(f"{relation_type} object {target_id}")
            if rel_descriptions:
                desc += f" -- Relations: {'; '.join(rel_descriptions)}"
        chain.append(desc)
    return "\n".join(chain)

if __name__ == "__main__":
    # Example usage for testing SAM loading and utilities
    # Ensure 'weights' directory exists and checkpoint is there
    os.makedirs("weights", exist_ok=True)
    sam_checkpoint_path = "weights/sam_vit_h_4b8939.pth"
    sam_checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    if not os.path.exists(sam_checkpoint_path):
        logger.info(f"Downloading SAM checkpoint to: {sam_checkpoint_path}")
        try:
            import urllib.request
            urllib.request.urlretrieve(sam_checkpoint_url, sam_checkpoint_path)
            logger.info("SAM checkpoint downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download SAM checkpoint: {e}. Please download manually.")
            exit()

    # Test SamPredictor
    predictor_instance = load_sam_model(checkpoint_path=sam_checkpoint_path)
    if predictor_instance:
        logger.info("SamPredictor loaded. Ready for interactive segmentation.")
        # Example: Load a dummy image and set it
        dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
        with autocast(): # Apply mixed precision
            predictor_instance.set_image(dummy_image)
        # You can now call predictor_instance.predict(...)
    
    # Test SamAutomaticMaskGenerator
    mask_generator_instance = get_mask_generator(checkpoint_path=sam_checkpoint_path)
    if mask_generator_instance:
        logger.info("SamAutomaticMaskGenerator loaded. Ready for automatic segmentation.")
        # Example: Generate masks for a dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        with autocast(): # Apply mixed precision
            masks_raw = mask_generator_instance.generate(dummy_image)
        logger.info(f"Generated {len(masks_raw)} masks automatically.")

        # Extract boolean masks for functions that expect them
        masks_boolean = [m["segmentation"] for m in masks_raw]

        # Test saving masks
        if masks_boolean:
            os.makedirs("output_masks", exist_ok=True)
            save_mask_png(masks_boolean[0], "output_masks/test_mask_colored.png", colorize=True)
            save_mask_png(masks_boolean[0], "output_masks/test_mask_binary.png", colorize=False)
            logger.info("Saved example masks.")

        # Test YOLO conversion
        yolo_labels = sam_masks_to_yolo(masks_raw, dummy_image.shape[:2], class_id=0)
        logger.info(f"Generated {len(yolo_labels)} YOLO labels.")
        if yolo_labels:
            os.makedirs("output_labels", exist_ok=True)
            with open("output_labels/test_yolo.txt", "w") as f:
                f.write("\n".join(yolo_labels))
            logger.info("Saved example YOLO labels.")

        # Test relational graph generation
        relation_graph_data = generate_relation_graph(masks_raw)
        logger.info(f"Generated relational graph with {len(relation_graph_data.get('nodes', []))} nodes and {len(relation_graph_data.get('links', []))} links.")
        if relation_graph_data:
            os.makedirs("output_relations", exist_ok=True)
            with open("output_relations/test_relations.json", "w") as f:
                json.dump(relation_graph_data, f, indent=2)
            logger.info("Saved example relational graph.")
        
        # Test symbolic labels and debugger (requires HAS_BONGARD_SYMBOLIC_FUSION to be True)
        if HAS_BONGARD_SYMBOLIC_FUSION and masks_boolean:
            logger.info("Testing symbolic label generation and debugger...")
            # Dummy program text for testing
            dummy_program_text = "draw circle at (100,100) filled red; draw square at (200,200) open blue; square contains circle"
            symbolic_anns = get_symbolic_labels(masks_raw, dummy_image, dummy_program_text)
            logger.info(f"Generated {len(symbolic_anns)} symbolic annotations.")

            if symbolic_anns and len(masks_boolean) == len(symbolic_anns):
                os.makedirs("output_symbolic", exist_ok=True)
                with open("output_symbolic/test_symbolic.json", "w") as f:
                    json.dump(symbolic_anns, f, indent=2)
                logger.info("Saved example symbolic annotations.")

                # Test visual debugger
                debug_img = overlay_symbolic_debugger(dummy_image, masks_boolean, symbolic_anns)
                os.makedirs("output_debug", exist_ok=True)
                cv2.imwrite("output_debug/test_debug_overlay.png", debug_img)
                logger.info("Saved example debug overlay image.")

                # Test reasoning chain
                reasoning_chain_text = generate_reasoning_chain(symbolic_anns)
                os.makedirs("output_reasoning", exist_ok=True)
                with open("output_reasoning/test_reasoning_chain.txt", "w") as f:
                    f.write(reasoning_chain_text)
                logger.info("Saved example reasoning chain.")
            else:
                logger.warning("Skipping symbolic debugger/reasoning chain test: Mismatch in masks/symbols count or no symbols generated.")

