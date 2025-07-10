# Folder: bongard_solver/
# File: scene_graph_builder.py
import numpy as np
import cv2
import logging
import torch # Import torch for tensor operations
from typing import List, Dict, Any, Tuple, Optional

# Import the new MaskRCNNCropper
from crop_extraction import MaskRCNNCropper

# Import Persistent Homology libraries
try:
    from ripser import ripser
    from persim import PersistenceImager
    HAS_PERSIM = True
    logger = logging.getLogger(__name__)
    logger.info("ripser and persim found and enabled for Persistent Homology.")
except ImportError:
    HAS_PERSIM = False
    logger = logging.getLogger(__name__)
    logger.warning("ripser or persim not found. Persistent Homology features will be disabled.")

# --- Global Configuration (assuming CONFIG is imported or passed) ---
# For standalone execution or testing, define a dummy CONFIG if not imported
try:
    from config import CONFIG, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP, ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP, RELATION_MAP
    # Import _calculate_iou from utils.py
    from utils import _calculate_iou
except ImportError:
    logger.warning("Could not import CONFIG or _calculate_iou from config.py/utils.py. Using dummy configuration for scene_graph_builder.")
    CONFIG = {
        'model': {
            'use_mask_rcnn': True,
            'maskrcnn_conf_thresh': 0.7,
            'use_persistent_homology': True,
            'ph_pixel_thresh': 0.5,
            'ph_imager_pixel_size': 0.1,
            'ph_feature_dim': 64,
            'n_attributes': 5, # Dummy value for AttributeModel
        }
    }
    ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2, 'star': 3}
    ATTRIBUTE_COLOR_MAP = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'black': 4, 'white': 5}
    ATTRIBUTE_FILL_MAP = {'solid': 0, 'hollow': 1, 'striped': 2, 'dotted': 3}
    ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
    ATTRIBUTE_ORIENTATION_MAP = {'upright': 0, 'inverted': 1}
    ATTRIBUTE_TEXTURE_MAP = {'none': 0, 'striped': 1, 'dotted': 2}
    RELATION_MAP = {'none': 0, 'left_of': 1, 'right_of': 2}
    # Dummy _calculate_iou if utils.py cannot be imported
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(box1_area + box2_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0


# Initialize MaskRCNNCropper once
_mask_rcnn_cropper: Optional[MaskRCNNCropper] = None
if CONFIG['model']['use_mask_rcnn']:
    try:
        _mask_rcnn_cropper = MaskRCNNCropper(conf_thresh=CONFIG['model']['maskrcnn_conf_thresh'])
    except Exception as e:
        logger.error(f"Failed to initialize MaskRCNNCropper: {e}. Disabling Mask R-CNN.")
        CONFIG['model']['use_mask_rcnn'] = False

# Initialize PersistenceImager once
_ph_imager: Optional[PersistenceImager] = None
if HAS_PERSIM and CONFIG['model']['use_persistent_homology']:
    try:
        _ph_imager = PersistenceImager(pixel_size=CONFIG['model']['ph_imager_pixel_size'])
        _ph_imager.fit(np.array([[0, 1], [0.5, 1.5]])) # Dummy fit with two points
        logger.info(f"PersistenceImager initialized with pixel_size: {CONFIG['model']['ph_imager_pixel_size']}")
    except Exception as e:
        logger.error(f"Failed to initialize PersistenceImager: {e}. Disabling Persistent Homology.")
        CONFIG['model']['use_persistent_homology'] = False
        HAS_PERSIM = False

# Placeholder for the AttributeModel encoder
# This will be properly initialized once models.py is updated.
_attribute_encoder = None
try:
    # Attempt to import AttributeModel from models.py
    # This import will only work if models.py has been updated with AttributeModel
    from models import AttributeModel
    # Instantiate the encoder part of the AttributeModel
    # Use dummy values if CONFIG is not fully loaded or if AttributeModel expects a different init
    _attribute_encoder = AttributeModel(
        n_attributes=CONFIG['model'].get('n_attributes', 5), # Use n_attributes from config
        backbone_pretrained=CONFIG['model'].get('pretrained', True)
    ).encoder
    logger.info("AttributeModel encoder successfully loaded for scene graph builder.")
except ImportError:
    logger.warning("Could not import AttributeModel from models.py. Feature embeddings will be dummy values.")
    # Define a dummy encoder if AttributeModel is not available
    class DummyEncoder(torch.nn.Module):
        def forward(self, x):
            # Simulate ResNet50 encoder output: [B, 2048, H_out, W_out]
            # Assuming input patch is preprocessed to 3x224x224
            return torch.randn(x.shape[0], 2048, 7, 7) # Example output size for 224x224 input
    _attribute_encoder = DummyEncoder()
    # Define a dummy device if not imported from config
    device = torch.device("cpu") # Default device for dummy encoder


def mask_to_pointcloud(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Converts a binary mask to a 2D point cloud.

    Args:
        mask (np.ndarray): A 2D NumPy array representing the mask (values typically 0-1 or 0-255).
        threshold (float): The threshold to binarize the mask if it's not already binary.

    Returns:
        np.ndarray: A NumPy array of shape (N, 2) where N is the number of points,
                    representing the (row, column) coordinates of activated pixels.
    """
    if mask is None or mask.size == 0:
        logger.warning("Received empty mask for point cloud conversion.")
        return np.array([])
    
    # Ensure mask is binary based on the threshold
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Get coordinates of non-zero pixels (foreground)
    # np.where returns (row_indices, col_indices)
    points = np.column_stack(np.where(binary_mask > 0)).astype(np.float32)
    
    return points

def extract_ph_features(mask: np.ndarray) -> np.ndarray:
    """
    Extracts Persistent Homology features from a binary mask.

    Args:
        mask (np.ndarray): A 2D binary mask of an object.

    Returns:
        np.ndarray: A flattened array of Persistent Homology image features.
                    Returns a zero array of configured dimension if PH is disabled
                    or if the mask is too small/invalid.
    """
    if not HAS_PERSIM or not CONFIG['model']['use_persistent_homology']:
        logger.debug("Persistent Homology is disabled or not available. Returning zero features.")
        return np.zeros((CONFIG['model']['ph_feature_dim'],))

    pts = mask_to_pointcloud(mask, threshold=CONFIG['model']['ph_pixel_thresh'])

    # Check if there are enough points to compute homology
    if len(pts) < 5: # Increased threshold for more meaningful PH
        logger.debug(f"Not enough points ({len(pts)}) in mask for PH feature extraction. Returning zero features.")
        return np.zeros((CONFIG['model']['ph_feature_dim'],))

    try:
        # Compute persistence diagrams using Ripser
        dgms = ripser(pts, maxdim=1)['dgms'] # Compute up to 1-dimensional homology

        if len(dgms) < 2 or dgms[1].shape[0] == 0: # Check if H_1 diagram exists and is not empty
            logger.debug("No 1-dimensional persistence diagram found or it's empty. Returning zero features.")
            return np.zeros((CONFIG['model']['ph_feature_dim'],))

        # Transform the persistence diagram (H_1) into a persistence image
        img = _ph_imager.transform(dgms[1])
        
        # Flatten the image and resize to the target feature dimension
        flattened_img = img.flatten()
        
        # Resize to a fixed dimension if necessary (e.g., using interpolation or padding/truncation)
        target_dim = CONFIG['model']['ph_feature_dim']
        if flattened_img.shape[0] != target_dim:
            if flattened_img.shape[0] > target_dim:
                # Truncate if too long
                features = flattened_img[:target_dim]
                logger.debug(f"Truncated PH features from {flattened_img.shape[0]} to {target_dim}.")
            else:
                # Pad with zeros if too short
                features = np.pad(flattened_img, (0, target_dim - flattened_img.shape[0]), 'constant')
                logger.debug(f"Padded PH features from {flattened_img.shape[0]} to {target_dim}.")
        else:
            features = flattened_img

        return features

    except Exception as e:
        logger.error(f"Error during Persistent Homology feature extraction: {e}. Returning zero features.")
        return np.zeros((CONFIG['model']['ph_feature_dim'],))


def extract_basic_attributes(patch: np.ndarray) -> Dict[str, Any]:
    """
    Extracts basic visual attributes from an image patch.
    This is a placeholder for your existing attribute classification logic.
    For a professional system, this would involve a trained attribute classifier model.

    Args:
        patch (np.ndarray): The cropped image patch of an object.

    Returns:
        Dict[str, Any]: A dictionary of extracted attributes (e.g., shape, color, size).
    """
    if patch is None or patch.size == 0:
        logger.warning("Received empty patch for basic attribute extraction.")
        return {
            'shape': 'unknown', 'color': 'unknown', 'fill': 'unknown',
            'size': 'unknown', 'orientation': 'unknown', 'texture': 'unknown',
            'centroid': [0, 0], 'area': 0
        }

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY) # Binarize the patch

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    attrs = {}
    if contours:
        # Assume the largest contour corresponds to the main object
        cnt = max(contours, key=cv2.contourArea)

        # Area
        area = cv2.contourArea(cnt)
        attrs['area'] = float(area)
        
        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            attrs['centroid'] = [cX, cY]
        else:
            attrs['centroid'] = [patch.shape[1] // 2, patch.shape[0] // 2] # Default to center

        # Placeholder for shape classification (e.g., using moments, aspect ratio, or a trained model)
        # In a real system, this would be a call to attribute_classifier.py
        if area > 1000: # Arbitrary size threshold for 'large'
            attrs['size'] = 'large'
        elif area > 100:
            attrs['size'] = 'medium'
        else:
            attrs['size'] = 'small'

        # Simple shape approximation (very basic, a classifier would be better)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
        num_vertices = len(approx)

        if num_vertices == 3:
            attrs['shape'] = 'triangle'
        elif num_vertices == 4:
            # Check aspect ratio for square vs rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            if 0.9 <= aspect_ratio <= 1.1:
                attrs['shape'] = 'square'
            else:
                attrs['shape'] = 'rectangle' # Add rectangle to your shape map if needed
        elif num_vertices > 4:
            # Check circularity
            area_circle = np.pi * (perimeter / (2 * np.pi))**2
            if area > 0 and area_circle > 0 and (area / area_circle) > 0.8: # Simple circularity check
                attrs['shape'] = 'circle'
            else:
                attrs['shape'] = 'polygon' # Generic polygon
        else:
            attrs['shape'] = 'unknown'
        
        # Placeholder for color (average color of the object)
        mean_color = cv2.mean(patch, mask=binary)
        # Simple color classification (replace with a proper color classifier)
        if mean_color[2] > 200 and mean_color[1] < 100 and mean_color[0] < 100:
            attrs['color'] = 'red'
        elif mean_color[0] > 200 and mean_color[1] < 100 and mean_color[2] < 100:
            attrs['color'] = 'blue'
        elif mean_color[1] > 200 and mean_color[0] < 100 and mean_color[2] < 100:
            attrs['color'] = 'green'
        elif mean_color[0] > 150 and mean_color[1] > 150 and mean_color[2] < 100:
            attrs['color'] = 'yellow'
        elif np.mean(mean_color[:3]) < 50:
            attrs['color'] = 'black'
        elif np.mean(mean_color[:3]) > 200:
            attrs['color'] = 'white'
        else:
            attrs['color'] = 'unknown'

        # Placeholder for fill (e.g., check for internal patterns or just solid/hollow)
        # This is very rudimentary; a dedicated texture/fill classifier is needed.
        if np.sum(binary) / area > 0.95: # If almost all pixels inside contour are active
            attrs['fill'] = 'solid'
        else:
            attrs['fill'] = 'hollow' # Could be hollow or striped/dotted if more complex logic is added
        
        attrs['orientation'] = 'upright' # Default, needs proper orientation detection
        attrs['texture'] = 'none' # Default, needs texture analysis

    else:
        logger.debug("No contours found in patch for basic attribute extraction.")
        # Default attributes if no object is detected in the patch
        attrs = {
            'shape': 'unknown', 'color': 'unknown', 'fill': 'unknown',
            'size': 'unknown', 'orientation': 'unknown', 'texture': 'unknown',
            'centroid': [0, 0], 'area': 0
        }
    
    return attrs

# Function to preprocess a patch for the encoder
def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """
    Preprocesses an image patch (NumPy array HWC) into a PyTorch tensor (CHW)
    ready for the attribute encoder.
    Applies normalization consistent with ImageNet pretrained models.
    """
    if patch is None or patch.size == 0:
        logger.warning("Received empty patch for preprocessing. Returning dummy tensor.")
        # Return a dummy tensor of expected shape for a 224x224 image
        return torch.zeros(3, 224, 224, dtype=torch.float32)

    # Convert HWC to CHW
    patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0 # Convert to float and normalize to [0, 1]

    # Resize to the expected input size for the encoder (e.g., 224x224 for ResNet)
    # This assumes the encoder expects a fixed input size.
    # If the patch is already the correct size, this does nothing.
    current_size = patch_tensor.shape[1:] # (H, W)
    target_size = (CONFIG['data']['image_size'], CONFIG['data']['image_size']) # e.g., (224, 224)

    if current_size != target_size:
        patch_tensor = F.interpolate(patch_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

    # Normalize with ImageNet mean and std
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    normalized_patch = (patch_tensor - mean) / std
    
    return normalized_patch


def build_scene_graph(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Builds a scene graph from an input image by detecting objects,
    extracting their attributes (including Persistent Homology features and learned embeddings),
    and identifying relationships between them.

    Args:
        image (np.ndarray): The input image (H, W, C) in RGB format.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents an object in the scene graph with its
                              attributes and potentially relations.
    """
    if _mask_rcnn_cropper is None:
        logger.error("Mask R-CNN cropper not initialized. Cannot build scene graph.")
        return []
    
    # Check if the attribute encoder is initialized.
    # If models.py hasn't been updated, _attribute_encoder will be a DummyEncoder.
    if _attribute_encoder is None:
        logger.error("Attribute encoder not initialized. Cannot extract feature embeddings. Returning empty scene graph.")
        return []

    logger.info("Starting scene graph construction...")
    
    # 1. Object Detection and Cropping using Mask R-CNN
    patches_with_masks = _mask_rcnn_cropper.segment_shapes(image)
    
    scene_graph = []
    objects_data = [] # To store attributes for relation extraction later

    # Ensure device is available, otherwise default to CPU
    current_device = DEVICE if torch.cuda.is_available() else torch.device("cpu")
    _attribute_encoder.to(current_device) # Move encoder to appropriate device
    _attribute_encoder.eval() # Set encoder to evaluation mode

    for i, (patch, mask, metadata) in enumerate(patches_with_masks):
        logger.debug(f"Processing object {i+1} (bbox: {metadata['bbox']}).")
        
        # Preprocess patch for the encoder
        tensor = preprocess_patch(patch).unsqueeze(0).to(current_device) # Add batch dimension and move to device

        # 2. Extract Feature Embeddings using the AttributeModel's encoder
        with torch.no_grad():
            features = _attribute_encoder(tensor)   # Output: [1, 2048, H_out, W_out]
            # Apply adaptive pooling to get a fixed-size vector
            pooled = torch.adaptive_avg_pool2d(features, 1).view(-1) # Output: [2048]
        
        # 3. Extract Persistent Homology features
        ph = extract_ph_features(mask)    # existing PH logic
        
        # 4. Extract Basic Attributes (e.g., color, shape, size heuristics)
        desc = extract_basic_attributes(patch)
        
        # Combine all extracted features into the object description
        desc.update({
            "embed": pooled.cpu().tolist(), # Convert to list for JSON serialization
            "topo": ph.tolist(),            # Convert to list for JSON serialization
            "bbox": metadata['bbox'],       # Add bounding box to attributes
            "score": metadata['score']      # Add confidence score
        })

        objects_data.append(desc) # Store for relation extraction
        scene_graph.append(desc) # Add to the main scene graph list

    # 5. Relation Extraction (between detected objects)
    # This is a placeholder. You would implement your relation_gnn_config logic here
    # or in a separate module (e.g., relation_extractor.py) and import it.
    # For now, let's add some dummy relations based on centroids.
    for i in range(len(objects_data)):
        obj1 = objects_data[i]
        obj1['relations'] = [] # Initialize relations list for each object

        for j in range(len(objects_data)):
            if i == j:
                continue # Skip self-comparison

            obj2 = objects_data[j]
            
            # Simple spatial relations based on centroids
            cx1, cy1 = obj1['centroid']
            cx2, cy2 = obj2['centroid']

            relation_type = 'none'
            # Determine image dimensions for spatial tolerance
            image_height, image_width, _ = image.shape
            spatial_tolerance_ratio = 0.1 # From utils.py's get_predicted_relation or config

            if cx1 < cx2 - spatial_tolerance_ratio * image_width: # obj1 is significantly to the left of obj2
                relation_type = 'left_of'
            elif cx1 > cx2 + spatial_tolerance_ratio * image_width: # obj1 is significantly to the right of obj2
                relation_type = 'right_of'
            
            if cy1 < cy2 - spatial_tolerance_ratio * image_height: # obj1 is significantly above obj2
                if relation_type == 'none': relation_type = 'above'
                else: relation_type += '_and_above' # Example of combining relations
            elif cy1 > cy2 + spatial_tolerance_ratio * image_height: # obj1 is significantly below obj2
                if relation_type == 'none': relation_type = 'below'
                else: relation_type += '_and_below'

            # Add more complex relations like 'is_close_to', 'aligned_horizontally', 'contains', 'intersects'
            # based on bounding box overlaps or distances.
            # You can reuse _calculate_iou here if needed for 'intersects' or 'contains'
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            iou = _calculate_iou(bbox1, bbox2)
            if iou > 0.1: # Example threshold for intersection
                if relation_type == 'none': relation_type = 'intersects'
                else: relation_type += '_and_intersects'

            if relation_type != 'none':
                obj1['relations'].append({
                    'target_object_idx': j,
                    'type': relation_type
                })
    logger.info(f"Scene graph built with {len(scene_graph)} objects and their attributes/relations.")
    return scene_graph

# Example Usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a dummy image for testing
    test_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(test_image, (100, 100), 40, (255, 255, 255), -1) # White circle
    cv2.rectangle(test_image, (250, 150), (350, 250), (0, 0, 255), -1) # Blue square
    cv2.circle(test_image, (100, 100), 20, (0, 0, 0), -1) # Black circle inside white circle (to create a hole)

    print("Building scene graph for dummy image...")
    scene_graph_result = build_scene_graph(test_image)

    if scene_graph_result:
        print("\n--- Generated Scene Graph ---")
        for i, obj in enumerate(scene_graph_result):
            print(f"Object {i+1}:")
            for attr, value in obj.items():
                if attr == 'topology_features':
                    print(f"  {attr}: [features of length {len(value)}]")
                elif attr == 'relations':
                    print(f"  {attr}: {value}")
                else:
                    print(f"  {attr}: {value}")
    else:
        print("Failed to build scene graph.")

