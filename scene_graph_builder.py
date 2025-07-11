# Folder: bongard_solver/
# File: scene_graph_builder.py
import numpy as np
import json
import logging
import torch  # Import torch for attribute_logits and relation_logits
import torch.nn.functional as F  # For softmax
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt # Added for 3.1

# Import topological features
try:
    from topo_features import TopologicalFeatureExtractor
    HAS_TOPO_FEATURES = True
    logger = logging.getLogger(__name__)
    logger.info("TopologicalFeatureExtractor found for scene_graph_builder.")
except ImportError:
    HAS_TOPO_FEATURES = False
    logger = logging.getLogger(__name__)
    logger.warning("topo_features.py not found. Topological features will not be added to scene graphs.")
    # Dummy class if not found
    class TopologicalFeatureExtractor:
        def __init__(self, *args, **kwargs): pass
        def extract(self, mask): return np.zeros(64)  # Return a dummy array

# Import _calculate_iou from utils
try:
    from utils import _calculate_iou, make_edge_index_map # make_edge_index_map is also used
except ImportError:
    logger.warning("utils.py not found. _calculate_iou and make_edge_index_map will be dummy functions.")
    def _calculate_iou(box1: List[float], box2: List[float]) -> float: return 0.0
    def make_edge_index_map(num_objects: int) -> Dict[Tuple[int, int], int]: return {}

# Import RELATION_MAP from config for reverse mapping
try:
    from config import RELATION_MAP, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, \
                     ATTRIBUTE_FILL_MAP, ATTRIBUTE_SIZE_MAP, ATTRIBUTE_ORIENTATION_MAP, \
                     ATTRIBUTE_TEXTURE_MAP, CONFIG
except ImportError:
    logger.warning("Attribute or Relation MAPs not found in config.py. Using dummy maps.")
    RELATION_MAP = {'none': 0, 'left_of': 1, 'right_of': 2, 'above': 3, 'below': 4,
                    'is_close_to': 5, 'intersects': 6, 'aligned_horizontally': 7,
                    'aligned_vertically': 8, 'inside': 9, 'outside': 10}
    ATTRIBUTE_SHAPE_MAP = {'circle': 0, 'square': 1, 'triangle': 2, 'star': 3}
    ATTRIBUTE_COLOR_MAP = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'black': 4, 'white': 5}
    ATTRIBUTE_FILL_MAP = {'solid': 0, 'hollow': 1, 'striped': 2, 'dotted': 3}
    ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
    ATTRIBUTE_ORIENTATION_MAP = {'upright': 0, 'inverted': 1}
    ATTRIBUTE_TEXTURE_MAP = {'none': 0, 'striped': 1, 'dotted': 2}
    CONFIG = {'model': {'use_persistent_homology': False, 'ph_pixel_thresh': 0.5, 'ph_feature_dim': 64},
              'debug': {'plot_persistence': False}} # Added debug for 3.1

logger = logging.getLogger(__name__)

class SceneGraphBuilder:
    """
    Builds a structured scene graph from detected objects and their attributes.
    Infers spatial and attribute-based relations.
    Incorporates topological features.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.image_size = config['data']['image_size']
        
        # Use the imported attribute maps directly
        self.attribute_maps = {
            'shape': ATTRIBUTE_SHAPE_MAP,
            'color': ATTRIBUTE_COLOR_MAP,
            'fill': ATTRIBUTE_FILL_MAP,
            'size': ATTRIBUTE_SIZE_MAP,
            'orientation': ATTRIBUTE_ORIENTATION_MAP,
            'texture': ATTRIBUTE_TEXTURE_MAP
        }
        self.relation_map = RELATION_MAP  # Use imported RELATION_MAP
        self.reverse_relation_map = {v: k for k, v in self.relation_map.items()}
        
        self.topo_feature_extractor = None
        # Check for 'topo' key in config and its 'enabled' status
        if HAS_TOPO_FEATURES and self.config.get('model', {}).get('use_persistent_homology', False):
            self.topo_feature_extractor = TopologicalFeatureExtractor(
                thr=self.config['model'].get('ph_pixel_thresh', 0.5),
                feature_dim=self.config['model'].get('ph_feature_dim', 64)
            )
            logger.info(f"SceneGraphBuilder: Initialized TopologicalFeatureExtractor with pixel threshold {self.config['model'].get('ph_pixel_thresh', 0.5)} and feature_dim {self.config['model'].get('ph_feature_dim', 64)}.")

    def build_scene_graph(
        self,
        image_np: np.ndarray,  # Original image (H, W, C) for context if needed
        detected_bboxes: List[List[float]],  # xyxy format
        detected_masks: List[np.ndarray],  # Binary numpy masks
        attribute_logits: Dict[str, torch.Tensor],  # Dict of {attr_name: (N_objects, N_classes)}
        relation_logits: torch.Tensor,  # (N_edges, N_relations)
        graph_embed: Optional[torch.Tensor] = None, # Added for graph embedding from RelationGNN
        idx: int = 0 # Added for 3.1 to check for first image in batch
    ) -> Dict[str, Any]:
        """
        Builds a scene graph for a single image.
        
        Args:
            image_np (np.ndarray): The original image (H, W, C).
            detected_bboxes (List[List[float]]): List of bounding boxes (xyxy).
            detected_masks (List[np.ndarray]): List of binary masks.
            attribute_logits (Dict[str, torch.Tensor]): Dictionary of attribute classification logits.
                                            Each tensor is (N_objects, Num_classes_for_attr).
            relation_logits (torch.Tensor): Tensor of relation classification logits (N_edges, Num_relations).
            graph_embed (Optional[torch.Tensor]): Global graph embedding from RelationGNN (1, hidden_dim).
            idx (int): Index of the current image in the batch (for debug plotting).
        Returns:
            Dict[str, Any]: A scene graph dictionary.
        """
        scene_graph = {'objects': [], 'relations': []}
        num_objects = len(detected_bboxes)
        if num_objects == 0:
            return scene_graph  # Return empty scene graph if no objects
        
        # 1. Add Objects with Attributes and Topological Features
        for i in range(num_objects):
            obj_id = i
            bbox = detected_bboxes[i]
            mask = detected_masks[i]
            
            inferred_attributes = {}
            for attr_name, logits_tensor in attribute_logits.items():
                if i < logits_tensor.shape[0]:  # Ensure index is valid for the current object
                    # Check if logits_tensor is not empty for this attribute
                    if logits_tensor.numel() > 0:
                        pred_class_idx = torch.argmax(logits_tensor[i]).item()
                        # Reverse map class index to attribute string
                        # Use self.attribute_maps for reverse lookup
                        attr_str = next((k for k, v in self.attribute_maps[attr_name].items() if v == pred_class_idx), 'unknown')
                        inferred_attributes[attr_name] = attr_str
                    else:
                        inferred_attributes[attr_name] = 'unknown'  # Fallback if no logits
                else:
                    inferred_attributes[attr_name] = 'unknown'  # Fallback if index out of bounds
            
            obj_entry = {
                'id': obj_id,
                'bbox': bbox,
                'mask': mask.tolist(),  # Convert numpy mask to list for JSON serialization
                'attributes': inferred_attributes
            }
            
            # Add topological features
            if self.topo_feature_extractor and HAS_TOPO_FEATURES:
                ph_vector = self.topo_feature_extractor.extract(mask)
                obj_entry['topo_features'] = ph_vector.tolist()  # Convert to list for JSON

                # 3.1 Debug-Mode Persistence Image Plot
                if self.config['debug'].get('plot_persistence', False) and idx == 0:
                    # Assuming topo_feature_extractor can generate a persistence image or similar visual
                    # This is a placeholder as TopologicalFeatureExtractor.extract returns a vector.
                    # If you have a method to get the actual persistence image, use it here.
                    # For now, we'll just plot the mask itself or a dummy image.
                    # A more direct way would be if `topo_features.py` had a `get_persistence_image` method.
                    # For demonstration, let's plot the mask as a grayscale image.
                    # Or, if `ph_vector` can be reshaped into a meaningful 2D image.
                    
                    # For now, let's assume `topo_features.py` provides a method to get the persistence image.
                    # If not, this part needs to be adapted.
                    # As a fallback, we can just plot the mask or a dummy representation.
                    
                    # Dummy persistence image for plotting if not available from topo_features
                    ph_image = np.zeros((64, 64), dtype=np.float32) # Example dummy size
                    if ph_vector.size == ph_image.size:
                        ph_image = ph_vector.reshape(ph_image.shape)
                    else:
                        logger.warning("Topological feature vector size does not match dummy image size for plotting. Plotting dummy image.")
                        ph_image = np.random.rand(64, 64).astype(np.float32) # Random dummy image
                    
                    plt.figure(figsize=(6, 6))
                    plt.imshow(ph_image, cmap='hot')
                    plt.title(f'PH Image for Object {obj_id} (Problem {idx})')
                    plt.colorbar(label='Persistence Value')
                    plt.tight_layout()
                    
                    # Save the plot
                    plot_dir = './debug_plots/persistence_images'
                    os.makedirs(plot_dir, exist_ok=True)
                    plt.savefig(os.path.join(plot_dir, f'problem_{idx}_obj_{obj_id}_ph_image.png'))
                    plt.close()
                    logger.info(f"Persistence image plot saved for problem {idx}, object {obj_id}.")
            
            scene_graph['objects'].append(obj_entry)
        
        # 2. Add Relations
        if num_objects > 1:
            # Reconstruct edge_index based on all-to-all connections (used by GNN)
            # This needs to match how edge_index was created for the GNN in PerceptionModule
            row_indices = torch.arange(num_objects, device=relation_logits.device).repeat_interleave(num_objects)
            col_indices = torch.arange(num_objects, device=relation_logits.device).repeat(num_objects)
            non_self_loop_mask = (row_indices != col_indices)
            edge_index_reconstructed = torch.stack([row_indices[non_self_loop_mask], col_indices[non_self_loop_mask]], dim=0)
            
            # Predict relation types from logits
            if relation_logits.numel() > 0 and relation_logits.shape[0] == edge_index_reconstructed.shape[1]:
                predicted_relation_indices = torch.argmax(relation_logits, dim=1)
                predicted_relation_scores = F.softmax(relation_logits, dim=-1)
                for edge_idx, pred_rel_idx in enumerate(predicted_relation_indices):
                    subj_id = edge_index_reconstructed[0, edge_idx].item()
                    obj_id = edge_index_reconstructed[1, edge_idx].item()
                    
                    # Reverse map relation index to string
                    rel_type = self.reverse_relation_map.get(pred_rel_idx.item(), 'unknown')
                    score = predicted_relation_scores[edge_idx, pred_rel_idx].item()
                    
                    if rel_type != 'none':  # Only add non-null relations
                        scene_graph['relations'].append({
                            'subject_id': subj_id,
                            'object_id': obj_id,
                            'type': rel_type,
                            'score': score
                        })
            else:
                logger.warning("Relation logits are empty or shape mismatch. No relations inferred from GNN output.")
        else:
            logger.debug("Only one object detected. No relations to infer.")
        
        # 3. Add Global Graph Embedding (if provided)
        if graph_embed is not None and graph_embed.numel() > 0:
            scene_graph['global_graph_embedding'] = graph_embed.squeeze(0).tolist()  # Convert (1, D) to (D) list
        
        logger.debug(f"Scene graph built with {len(scene_graph['objects'])} objects and {len(scene_graph['relations'])} relations.")
        return scene_graph

