# src/scene_graph_builder.py

import os
import numpy as np
import cv2 # Import OpenCV for image processing
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

# Assuming these imports are correct based on your file structure
from src.preprocess.crop_extractor import ClassicalCVCropper # Updated import
from src.features.topo_features import TopologicalFeatureExtractor

logger = logging.getLogger(__name__)

class SceneGraphBuilder:
    """
    Builds a scene graph from an input image by detecting objects,
    extracting their features (appearance, topology), and inferring relations.
    This version is adapted to be called by the emergent Workspace.
    """
    def __init__(self, images: List[Any], config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SceneGraphBuilder.
        Args:
            images (List[Any]): A list of image data (e.g., file paths or numpy arrays).
                                This is used to mock objects for the Workspace.
            config (Optional[Dict[str, Any]]): Configuration dictionary for thresholds etc.
        """
        self.config = config if config is not None else {}
        
        # Initialize the cropper with classical CV methods
        self.cropper = ClassicalCVCropper(conf_thresh=self.config.get('cv_contour_conf_thresh', 0.1))
        
        # Initialize the topological feature extractor
        self.topo = TopologicalFeatureExtractor(pixel_thresh=self.config.get('ph_pixel_thresh', 0.5))
        
        # Store the images. In a real scenario, you'd process these to get actual objects.
        self.input_images = images
        
        # Mock object IDs for the Workspace. In a real system, these would come from detection.
        # For the emergent system, we need a list of object IDs that Scouts can iterate over.
        # For simplicity, we'll assume each image contains at least one object, or we create dummy ones.
        self.objects: List[str] = [f"obj_{i}" for i in range(len(images))] if images else ["obj_0"]
        
        # Internal state for problem solving status, used by the emergent loop
        self._solution_found: bool = False
        self._solution: Optional[Any] = None

        # A simple cache for extracted features to avoid re-computation within a single run
        self._feature_cache: Dict[Tuple[str, str], Tuple[Any, float]] = {}
        logger.info(f"SceneGraphBuilder initialized with {len(self.input_images)} input images.")
        logger.debug(f"Mock object IDs: {self.objects}")

    def build_scene_graph(self, image_np: np.ndarray, detected_bboxes: Optional[List[Any]]=None, 
                          detected_masks: Optional[List[Any]]=None, attribute_logits: Optional[Dict[str, Any]]=None, 
                          relation_logits: Optional[Any]=None, graph_embed: Optional[Any]=None) -> List[Dict[str, Any]]:
        """
        Builds a scene graph for a given image.
        This method is less directly used by the emergent loop for feature extraction,
        but it represents the overall scene understanding capability.
        Args:
            image_np (np.ndarray): The input image as a NumPy array.
            detected_bboxes (Optional[List[Any]]): Pre-detected bounding boxes (if available).
            detected_masks (Optional[List[Any]]): Pre-detected masks (if available).
            attribute_logits (Optional[Dict[str, Any]]): Logits for object attributes.
            relation_logits (Optional[Any]): Logits for relations between objects.
            graph_embed (Optional[Any]): Global graph embedding.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an object
                                  with its extracted features.
        """
        logger.info("Building scene graph...")
        
        # Use the cropper to get object patches and masks using classical CV
        # The cropper now returns a list of dictionaries, each with 'id', 'bbox', 'mask', 'patch'
        detected_objects_info = self.cropper.segment_shapes(image_np)
        
        scene_graph_objects = []
        # Update self.objects based on actual detected objects if this is the primary detection path
        self.objects = [f"obj_{obj_info['id']}" for obj_info in detected_objects_info]

        for obj_info in detected_objects_info:
            obj_id = f"obj_{obj_info['id']}"
            patch = obj_info['patch']
            mask = obj_info['mask']
            bbox = obj_info['bbox'] # Bounding box [x1, y1, x2, y2]
            
            # Extract appearance features (color, size, fill, position)
            appearance_features = self.extract_appearance(patch, bbox, image_np.shape[1], image_np.shape[0])
            
            # Extract topological features
            ph_feat = self.topo.extract_features(mask)
            
            # Combine all features for the object
            object_descriptor = {
                'id': obj_id,
                'bbox': bbox,
                'mask': mask, # Store mask if needed for later
                'attributes': {
                    'shape': self.infer_shape_from_contour(obj_info.get('contour')), # Use contour from cropper
                    'color': appearance_features['color'],
                    'size': appearance_features['size'],
                    'fill': appearance_features['fill'],
                    'position_h': appearance_features['position_h'],
                    'position_v': appearance_features['position_v'],
                    'topology_features': ph_feat.tolist(), # Convert numpy array to list for JSON compatibility
                }
            }
            scene_graph_objects.append(object_descriptor)
            logger.debug(f"Added object {obj_id} to scene graph with features: {object_descriptor['attributes']}")

        # You can add logic here to infer relations between scene_graph_objects
        # based on their positions, attributes, etc. (e.g., using utils.get_predicted_relation)
        # For this request, we focus on object-level feature extraction for the emergent system.

        logger.info(f"Scene graph built with {len(scene_graph_objects)} objects.")
        return scene_graph_objects

    def extract_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
        """
        Extracts a specific feature for a given object ID.
        This method is called by the Workspace to get primitive features.
        It should ideally retrieve features from a pre-built scene graph or
        perform on-the-fly extraction.
        Args:
            obj_id (str): The ID of the object (e.g., 'obj_0').
            feat_type (str): The type of feature to extract (e.g., 'shape', 'color').
        Returns:
            Tuple[Any, float]: The extracted feature value and a confidence score (0.0-1.0).
        """
        # Check cache first
        if (obj_id, feat_type) in self._feature_cache:
            logger.debug(f"Returning cached feature for {obj_id}, {feat_type}")
            return self._feature_cache[(obj_id, feat_type)]

        # In a real system, you would have a way to map obj_id back to actual image data
        # or pre-computed features. For this example, we'll mock it based on the object ID.
        # Assuming obj_id is like "obj_0", "obj_1", etc., mapping to an image index.
        try:
            image_idx = int(obj_id.split('_')[1])
            if image_idx >= len(self.input_images):
                raise IndexError("Object ID out of bounds for input images.")
            
            # For simplicity, we'll re-run detection for the specific image to get its objects.
            # In a more optimized system, you'd have a pre-computed list of objects and their features.
            current_image_data = self.input_images[image_idx] # This might be a path or actual image
            
            # If current_image_data is a path, load it. For demo, assume it's a dummy string.
            # For actual feature extraction, you'd need the actual image data (e.g., np.ndarray)
            # Let's assume for this method, we have a way to get the image data.
            # For now, we'll use a dummy image for feature extraction if the input is just a string.
            if isinstance(current_image_data, str):
                # Create a dummy image for feature extraction if only a path/name is given
                temp_image_np = np.zeros((256, 256, 3), dtype=np.uint8)
                # Add a simple shape for the dummy image based on ID
                if image_idx % 3 == 0: # Circle
                    cv2.circle(temp_image_np, (128, 128), 50, (255, 0, 0), -1)
                elif image_idx % 3 == 1: # Square
                    cv2.rectangle(temp_image_np, (78, 78), (178, 178), (0, 255, 0), -1)
                else: # Triangle
                    pts = np.array([[128, 78], [78, 178], [178, 178]], np.int32)
                    cv2.fillPoly(temp_image_np, [pts], (0, 0, 255))
                image_to_process = temp_image_np
            else: # Assume it's already a numpy array
                image_to_process = current_image_data

            detected_objects_info = self.cropper.segment_shapes(image_to_process)
            
            # Find the specific object by ID. In a real system, obj_id would map to a unique detected object.
            # For this mock, we'll just take the first detected object from the image as 'obj_X'.
            if detected_objects_info:
                # Assuming obj_id refers to the primary object in that image for simplicity
                # In a multi-object image, you'd need a more robust mapping.
                obj_info = detected_objects_info[0] # Take the first detected object
                patch = obj_info['patch']
                mask = obj_info['mask']
                bbox = obj_info['bbox']
                
                value = "unknown"
                confidence = 0.5 # Default confidence

                if feat_type == 'shape':
                    value = self.infer_shape_from_contour(obj_info.get('contour'))
                    confidence = 0.9 # High confidence for shape
                elif feat_type == 'color':
                    appearance = self.extract_appearance(patch, bbox, image_to_process.shape[1], image_to_process.shape[0])
                    value = appearance['color']
                    confidence = 0.8
                elif feat_type == 'size':
                    appearance = self.extract_appearance(patch, bbox, image_to_process.shape[1], image_to_process.shape[0])
                    value = appearance['size']
                    confidence = 0.8
                elif feat_type == 'position_h':
                    appearance = self.extract_appearance(patch, bbox, image_to_process.shape[1], image_to_process.shape[0])
                    value = appearance['position_h']
                    confidence = 0.7
                elif feat_type == 'position_v':
                    appearance = self.extract_appearance(patch, bbox, image_to_process.shape[1], image_to_process.shape[0])
                    value = appearance['position_v']
                    confidence = 0.7
                elif feat_type == 'fill':
                    appearance = self.extract_appearance(patch, bbox, image_to_process.shape[1], image_to_process.shape[0])
                    value = appearance['fill']
                    confidence = 0.7
                elif feat_type == 'orientation':
                    # Placeholder for orientation extraction
                    value = 'horizontal' # Dummy
                    confidence = 0.6
                elif feat_type == 'texture':
                    # Placeholder for texture extraction
                    value = 'smooth' # Dummy
                    confidence = 0.6
                else:
                    logger.warning(f"Unknown feature type requested: {feat_type} for {obj_id}")
                    value = "unknown"
                    confidence = 0.1
                
                result = (value, confidence)
                self._feature_cache[(obj_id, feat_type)] = result # Cache the result
                logger.debug(f"Extracted feature for {obj_id}, {feat_type}: {result}")
                return result
            else:
                logger.warning(f"No objects detected in image for {obj_id}. Cannot extract feature {feat_type}.")
                result = ("none", 0.0)
                self._feature_cache[(obj_id, feat_type)] = result
                return result
        except (ValueError, IndexError) as e:
            logger.error(f"Error extracting feature for {obj_id}, {feat_type}: {e}")
            result = ("error", 0.0)
            self._feature_cache[(obj_id, feat_type)] = result
            return result


    def extract_appearance(self, patch: np.ndarray, bbox: List[int], image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Extracts appearance features (color, size, fill, position) from a patch.
        Args:
            patch (np.ndarray): The cropped image patch of the object.
            bbox (List[int]): Bounding box [x1, y1, x2, y2] of the object in the original image.
            image_width (int): Width of the original image.
            image_height (int): Height of the original image.
        Returns:
            Dict[str, Any]: A dictionary of appearance features.
        """
        # --- Color ---
        # Convert patch to HSL for better color analysis (optional, but good practice)
        # For simplicity, let's get the average color in RGB
        if patch.size == 0: # Handle empty patches
            avg_color = (0, 0, 0)
        else:
            avg_color = np.mean(patch.reshape(-1, patch.shape[-1]), axis=0)
        
        # Simple color quantization (e.g., to 'red', 'green', 'blue', 'black', 'white')
        color_name = 'unknown'
        if len(avg_color) >= 3: # Ensure it's an RGB image
            r, g, b = avg_color[0], avg_color[1], avg_color[2]
            if r > 200 and g < 50 and b < 50: color_name = 'red'
            elif r < 50 and g > 200 and b < 50: color_name = 'green'
            elif r < 50 and g < 50 and b > 200: color_name = 'blue'
            elif r < 50 and g < 50 and b < 50: color_name = 'black'
            elif r > 200 and g > 200 and b > 200: color_name = 'white'
            elif r > 150 and g > 150 and b < 50: color_name = 'yellow' # Added yellow
            elif r > 100 and g < 100 and b > 100: color_name = 'purple' # Added purple
            elif r < 100 and g > 100 and b > 100: color_name = 'cyan' # Added cyan


        # --- Size ---
        obj_width = bbox[2] - bbox[0]
        obj_height = bbox[3] - bbox[1]
        obj_area = obj_width * obj_height
        image_area = image_width * image_height
        
        size_ratio = obj_area / image_area
        size_name = 'unknown'
        if size_ratio < 0.01: size_name = 'small'
        elif size_ratio < 0.05: size_name = 'medium'
        else: size_name = 'large'

        # --- Fill ---
        # Simple check: if the mask is mostly solid vs. sparse/outline
        # This is a very basic heuristic. A more robust method would analyze pixel distribution.
        if mask.size == 0:
            fill_name = 'unknown'
        else:
            filled_pixels = np.sum(mask)
            total_pixels_in_bbox = mask.shape[0] * mask.shape[1]
            fill_ratio = filled_pixels / total_pixels_in_bbox if total_pixels_in_bbox > 0 else 0
            fill_name = 'filled' if fill_ratio > 0.7 else 'outlined' # Threshold for "filled"

        # --- Position (Horizontal and Vertical) ---
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        pos_h_name = 'unknown'
        if center_x < image_width / 3: pos_h_name = 'left'
        elif center_x < 2 * image_width / 3: pos_h_name = 'center_h'
        else: pos_h_name = 'right'

        pos_v_name = 'unknown'
        if center_y < image_height / 3: pos_v_name = 'top'
        elif center_y < 2 * image_height / 3: pos_v_name = 'center_v'
        else: pos_v_name = 'bottom'

        return {
            'color': color_name,
            'size': size_name,
            'fill': fill_name,
            'position_h': pos_h_name,
            'position_v': pos_v_name,
        }

    def infer_shape_from_contour(self, contour: Optional[np.ndarray]) -> str:
        """
        Infers the shape name (e.g., 'triangle', 'quadrilateral', 'circle')
        from a given contour.
        Args:
            contour (Optional[np.ndarray]): The contour of the object as detected by OpenCV.
        Returns:
            str: The inferred shape name.
        """
        if contour is None or len(contour) < 3:
            return "unknown"

        # Approximate the polygon
        epsilon = 0.04 * cv2.arcLength(contour, True) # Adjust epsilon as needed
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        if num_vertices == 3:
            return "triangle"
        elif num_vertices == 4:
            # Check for square/rectangle properties (aspect ratio, angle)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.9 <= aspect_ratio <= 1.1: # Allow some tolerance for square
                return "square"
            else:
                return "quadrilateral"
        else:
            # For circles, check circularity
            area = cv2.contourArea(contour)
            if area > 0: # Avoid division by zero
                (x, y), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * (radius ** 2)
                # Compare contour area to bounding circle area
                circularity = area / circle_area
                if circularity > 0.8: # Threshold for circularity
                    return "circle"
            return "other" # Default for complex shapes or many vertices

    def problem_solved(self) -> bool:
        """
        Checks if the Bongard problem has been solved.
        This state is set by the emergent loop when a strong solution is found.
        Returns:
            bool: True if the problem is solved, False otherwise.
        """
        return self._solution_found

    def mark_solution(self, solution: Any):
        """
        Marks the problem as solved and stores the solution.
        Args:
            solution (Any): The solution found by the reasoning pipeline.
        """
        self._solution = solution
        self._solution_found = True
        logger.info(f"SceneGraphBuilder: Problem marked as solved with solution: {solution}")

    def get_solution(self) -> Optional[Any]:
        """
        Retrieves the stored solution.
        Returns:
            Optional[Any]: The solution if found, otherwise None.
        """
        return self._solution

# Dummy function for dataset splitting (not directly part of SceneGraphBuilder logic)
def split_dataset(input_folder, output_folder, train_ratio=0.8):
    """
    Splits images from input_folder into train/val folders in output_folder.
    """
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    np.random.shuffle(images)
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]
    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)
    for img in train_imgs:
        os.rename(os.path.join(input_folder, img), os.path.join(output_folder, 'train', img))
    for img in val_imgs:
        os.rename(os.path.join(input_folder, img), os.path.join(output_folder, 'val', img))

