# Folder: bongard_solver/src/emergent/
# File: workspace_ext.py
import heapq
import logging
from typing import List, Dict, Any, Tuple, Set, Optional
import collections  # For defaultdict
from PIL import Image  # For dummy image in case of no image data
import random  # For dummy feature extraction
import numpy as np # For numpy array handling

# Import emergent modules (assuming these exist or are dummy)
try:
    from src.emergent.codelets import Scout, StrengthTester, Builder, Breaker, Codelet, GroupScout, RuleTester, RuleBuilder, RuleProposer  # Import new codelets
    from src.emergent.concept_net import ConceptNetwork
    from core_models.replay_buffer import ReplayBuffer # Import ReplayBuffer
    from torchvision.transforms.functional import to_tensor # For converting PIL image to tensor
    from core_models.models import PerceptionModule # To get class_names from the loaded model
    from config import CONFIG # To access global config for model and class names
except ImportError:
    logging.warning("Could not import emergent modules or replay_buffer/models/config. Using dummy classes.")
    class Scout: pass
    class StrengthTester: pass
    class Builder: pass
    class Breaker: pass
    class Codelet: pass
    class GroupScout: pass
    class RuleTester: pass
    class RuleBuilder: pass
    class RuleProposer: pass
    class ConceptNetwork:
        def __init__(self, *args, **kwargs): pass
        def get_active_nodes(self, threshold): return {}
        def activate_node(self, name, urgency): pass # Renamed from activate_concept
        def step(self, decay_factor, max_activation): pass
    class ReplayBuffer:
        def __init__(self, capacity): pass
        def push(self, *args): pass
        def sample(self, batch_size): return [], []
        def __len__(self): return 0
    def to_tensor(img): return torch.randn(3, 64, 64) # Dummy tensor
    class PerceptionModule: # Dummy for accessing class_names
        def __init__(self, cfg=None):
            self.attribute_classifier_config = cfg['model']['attribute_classifier_config'] if cfg else {'shape': 5, 'color': 7}
            # Create a dummy class_names mapping for common attributes
            self.class_names = {
                'shape': ['circle', 'square', 'triangle', 'pentagon', 'star'],
                'color': ['red', 'blue', 'green', 'yellow', 'black', 'white']
            }
        def get_class_names(self, attribute_type: str) -> List[str]:
            return self.class_names.get(attribute_type, [])
    CONFIG = {'model': {'attribute_classifier_config': {'shape': 5, 'color': 7}}} # Dummy config

# Import primitive_extractor for feature extraction with confidence
try:
    from src.perception.primitive_extractor import extract_shape_conf, extract_fill_conf, extract_cnn_features, _load_cnn_model
    HAS_PRIMITIVE_EXTRACTOR = True
except ImportError:
    logging.warning("Could not import primitive_extractor.py. Workspace will use dummy feature extraction.")
    HAS_PRIMITIVE_EXTRACTOR = False
    def extract_shape_conf(img): return "dummy_shape", 0.5
    def extract_fill_conf(img): return "dummy_fill", 0.5
    def extract_cnn_features(img): return {"shape": ("dummy_cnn_shape", 0.6), "color": ("dummy_cnn_color", 0.7), "size": ("dummy_cnn_size", 0.5), "fill": ("dummy_cnn_fill", 0.6), "orientation": ("dummy_cnn_orientation", 0.5), "texture": ("dummy_cnn_texture", 0.5)}
    def _load_cnn_model(): return PerceptionModule() # Return dummy model

# Import SceneGraphBuilder
try:
    from src.scene_graph_builder import SceneGraphBuilder
    HAS_SCENE_GRAPH_BUILDER = True
    logger = logging.getLogger(__name__)
    logger.info("src/scene_graph_builder.py found for Workspace.")
except ImportError:
    HAS_SCENE_GRAPH_BUILDER = False
    logger = logging.getLogger(__name__)
    logger.warning("src/scene_graph_builder.py not found. Workspace will use a dummy SceneGraphBuilder.")
    # Dummy SceneGraphBuilder if not found, to allow the rest of the code to run
    class SceneGraphBuilder:
        def __init__(self, images: List[Any], config: Optional[Dict[str, Any]] = None, perception_model=None):
            logger.warning("Dummy SceneGraphBuilder initialized. Feature extraction will be mocked.")
            self.images = images
            # Mock object IDs based on images, or a default if no images
            self.objects = [f"obj_{i}" for i in range(len(images))] if images else ["obj_0"]
            self._solution_found = False  # For problem_solved()
            self._solution = None
            self.config = config if config is not None else {}  # Store config for dummy
            self.current_scene_graph_data = [] # For RuleProposer
            self.perception_model = perception_model # Store the dummy model
        def extract_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
            """Mocks feature extraction."""
            logger.debug(f"Dummy SceneGraphBuilder: Extracting feature '{feat_type}' for '{obj_id}'")
            # Mock some feature values and confidence
            if feat_type == 'shape':
                return random.choice(['circle', 'square', 'triangle']), random.uniform(0.7, 0.9)
            elif feat_type == 'color':
                return random.choice(['red', 'blue', 'green']), random.uniform(0.6, 0.8)
            elif feat_type == 'size':
                return random.choice(['small', 'medium', 'large']), random.uniform(0.7, 0.9)
            elif feat_type == 'position_h':
                return random.choice(['left', 'center_h', 'right']), random.uniform(0.6, 0.8)
            elif feat_type == 'position_v':
                return random.choice(['top', 'center_v', 'bottom']), random.uniform(0.6, 0.8)
            elif feat_type == 'fill':
                return random.choice(['filled', 'outlined']), random.uniform(0.6, 0.8)
            elif feat_type == 'orientation':
                return random.choice(['horizontal', 'vertical']), random.uniform(0.6, 0.8)
            elif feat_type == 'texture':
                return random.choice(['smooth', 'rough']), random.uniform(0.6, 0.8)
            else:
                return "unknown", 0.1
        def problem_solved(self) -> bool:
            """Mocks problem solved status."""
            return self._solution_found
        
        def mark_solution(self, solution: Any):
            """Mocks marking a solution."""
            self._solution = solution
            self._solution_found = True
            logger.info(f"Dummy SceneGraphBuilder: Solution marked: {solution}")
        def get_solution(self) -> Optional[Any]:
            """Mocks getting the solution."""
            return self._solution
        def get_object_image(self, obj_id: str) -> Optional[Any]:
            """Dummy: Returns a dummy image for an object."""
            return Image.new('RGB', (50, 50), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        def build_scene_graph(self, image_np: np.ndarray, **kwargs) -> Dict[str, Any]:
            """Dummy build_scene_graph for mock data generation."""
            # Create a simple mock scene graph for a single image
            mock_objects = []
            if image_np is not None:  # Ensure image_np is not None
                obj_id = f"obj_{random.randint(0, 99)}"  # Random ID
                # Mock some attributes based on random choices
                shape = random.choice(['circle', 'square', 'triangle'])
                color = random.choice(['red', 'blue', 'green'])
                size = random.choice(['small', 'medium', 'large'])
                mock_objects.append({
                    'id': obj_id,
                    'attributes': {'shape': shape, 'color': color, 'size': size},
                    'bbox_xyxy': [0,0,10,10],  # Dummy bbox
                    'centroid': [5,5]  # Dummy centroid
                })
            sg = {'objects': mock_objects, 'relations': []}
            self.current_scene_graph_data.append(sg) # Store for RuleProposer
            return sg
        def crop(self, obj_id: str) -> Image.Image:
            """Dummy crop function for an object."""
            return Image.new('RGB', (64, 64), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        def crop_tensor(self, obj_id: str) -> torch.Tensor:
            """Dummy crop_tensor function for an object."""
            return to_tensor(self.crop(obj_id))


logger = logging.getLogger(__name__)

# Global online buffer for online fine-tuning
# Initialize with capacity from config, or a default if config/replay isn't available
online_buffer_capacity = CONFIG.get('replay', {}).get('buffer_capacity', 30000)
online_buffer = ReplayBuffer(capacity=online_buffer_capacity)
logger.info(f"Global online replay buffer initialized with capacity: {online_buffer_capacity}")


class Workspace:
    """
    The central workspace where emergent perception and cognitive processes occur.
    It hosts codelets, manages features, and interacts with the SceneGraphBuilder
    and ConceptNetwork.
    """
    def __init__(self, images: List[Any], config: Dict[str, Any], perception_model: Optional[Any] = None):
        """
        Initializes the Workspace.
        Args:
            images (List[Any]): A list of image data (e.g., file paths, numpy arrays)
                                 that the SceneGraphBuilder will process.
            config (Dict[str, Any]): The configuration dictionary.
            perception_model (Optional[Any]): The loaded perception model instance.
        """
        self.config = config
        # Pass images and perception_model to SceneGraphBuilder. It will handle initial object detection
        # and provide the list of object IDs.
        self.sg = SceneGraphBuilder(images, config, perception_model)  # Pass config and model to SceneGraphBuilder
        
        # Build initial scene graphs for all images in the problem
        self.scene_graphs: List[Dict[str, Any]] = []
        self.object_ids_per_image: List[List[str]] = []  # Store object IDs per image
        for i, img_data in enumerate(images):
            # Assuming img_data is a numpy array
            sg_for_image = self.sg.build_scene_graph(img_data)
            self.scene_graphs.append(sg_for_image)
            self.object_ids_per_image.append([obj['id'] for obj in sg_for_image['objects']])
        # Flatten list of all object IDs across all images for general access
        self.objects: List[str] = [obj_id for sublist in self.object_ids_per_image for obj_id in sublist]
        
        # Stores confirmed features: obj_id -> {feat_type: (value, confidence)}
        self.features: Dict[str, Dict[str, Tuple[Any, float]]] = collections.defaultdict(dict)
        
        # Tracks proposed features that a Scout has identified: (obj_id, feat_type, value, confidence)
        self.proposed: Set[Tuple[str, str, Any, float]] = set()
        
        # Tracks features that have been successfully 'built' by a Builder: (obj_id, feat_type, value, confidence)
        self.built: Set[Tuple[str, str, Any, float]] = set()
        
        # Stores proposed rule fragments from RuleTester codelets
        # Each fragment is a dict: {'rule_description': str, 'confidence': float, 'source_codelet': str}
        self.current_rule_fragments: List[Dict[str, Any]] = []
        
        # A min-heap (priority queue) of (-urgency, codelet) tuples.
        # Negative urgency ensures higher urgency codelets have lower values and are popped first.
        self.coderack: List[Tuple[float, Codelet]] = []
        
        # Initialize the Concept Network with its configuration
        self.concept_net = ConceptNetwork(self.config.get('slipnet_config', {}))
        
        # Support set data (to be populated by main.py)
        self.support_set_scene_graphs: List[Dict[str, Any]] = []
        self.support_set_labels: List[int] = []

        # Workspace state for logging/visualization
        self.workspace = {
            'query_scene_graph_view1': self.scene_graphs[0] if self.scene_graphs else None,  # First image's SG
            'query_scene_graph_view2': self.scene_graphs[1] if len(self.scene_graphs) > 1 else None,  # Second image's SG
            'current_rule_fragments': [],
            'active_concepts': {},
            'step': 0,
            'coderack_size': 0,
            'built_features_count': 0,
            'proposed_features_count': 0,
        }
        logger.info("Workspace initialized.")
        logger.debug(f"Initial objects in workspace: {self.objects}")

    def post_codelet(self, codelet: Codelet):
        """
        Adds a codelet to the coderack.
        Args:
            codelet (Codelet): The codelet instance to add.
        """
        heapq.heappush(self.coderack, (-codelet.urgency, codelet))
        self.workspace['coderack_size'] = len(self.coderack)  # Update workspace state
        logger.debug(f"Posted codelet: {type(codelet).__name__} (urgency: {codelet.urgency:.4f}). Coderack size: {len(self.coderack)}")

    def run_codelets(self, temperature: float, max_steps: int = 20):
        """
        Runs a specified number of codelets from the coderack.
        Codelets with higher urgency are run first.
        Args:
            temperature (float): The current system temperature, passed to codelets.
            max_steps (int): The maximum number of codelets to run in this cycle.
        """
        logger.info(f"Running up to {max_steps} codelets (current coderack size: {len(self.coderack)})...")
        steps_taken = 0
        while self.coderack and steps_taken < max_steps:
            neg_urgency, codelet = heapq.heappop(self.coderack)
            logger.debug(f"Popped codelet: {type(codelet).__name__} (urgency: {-neg_urgency:.4f})")
            try:
                codelet.run(self, self.concept_net, temperature)
                steps_taken += 1
                self.workspace['step'] += 1  # Increment step counter
            except NotImplementedError:
                logger.error(f"Codelet {type(codelet).__name__} has not implemented the 'run' method.")
            except Exception as e:
                logger.error(f"Error running codelet {type(codelet).__name__}: {e}", exc_info=True)
        self.workspace['coderack_size'] = len(self.coderack)  # Update workspace state
        logger.info(f"Finished running codelets. Steps taken: {steps_taken}. Remaining in coderack: {len(self.coderack)}")

    def primitive_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
        """
        Extracts a primitive feature for an object using the SceneGraphBuilder
        or directly from primitive_extractor.py.
        This is the interface for Scout codelets.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to extract.
        Returns:
            Tuple[Any, float]: The extracted feature value and its confidence.
        """
        # In a real system, obj_id would map to an image crop or bounding box.
        # For now, we'll use a dummy image for extraction if SceneGraphBuilder doesn't provide crops.
        obj_image = self.sg.get_object_image(obj_id)  # SceneGraphBuilder should provide this
        if obj_image is None:
            logger.warning(f"No image available for object {obj_id}. Using dummy feature extraction.")
            # Fallback to dummy extraction if no image is available
            val, conf = self.sg.extract_feature(obj_id, feat_type)  # Use dummy SG extractor
        else:
            val, conf = "unknown", 0.0
            if HAS_PRIMITIVE_EXTRACTOR:
                # Use CNN-based features if configured, otherwise classical CV
                if self.config['model'].get('use_cnn_features', True):  # Default to CNN features
                    cnn_feats = extract_cnn_features(obj_image) # This now includes TTA
                    if feat_type in cnn_feats:
                        val, conf = cnn_feats[feat_type]
                    else:
                        logger.warning(f"CNN features for {feat_type} not available. Falling back to CV.")
                        if feat_type == 'shape': val, conf = extract_shape_conf(obj_image)
                        elif feat_type == 'fill': val, conf = extract_fill_conf(obj_image)
                        # Add more CV-based feature extractions as needed
                        else:
                            logger.warning(f"Unsupported feature type '{feat_type}' for classical CV. Using dummy.")
                            val, conf = self.sg.extract_feature(obj_id, feat_type)  # Fallback to dummy SG extractor
                else:  # Use classical CV features
                    if feat_type == 'shape': val, conf = extract_shape_conf(obj_image)
                    elif feat_type == 'fill': val, conf = extract_fill_conf(obj_image)
                    # Add more CV-based feature extractions as needed
                    else:
                        logger.warning(f"Unsupported feature type '{feat_type}' for classical CV. Using dummy.")
                        val, conf = self.sg.extract_feature(obj_id, feat_type)  # Fallback to dummy SG extractor
            else:
                # Fallback to dummy SG extractor if primitive_extractor is not available
                val, conf = self.sg.extract_feature(obj_id, feat_type)

        # Push crop + pseudo-label for fine-tuning to the global online buffer
        # This assumes 'val' is a class name (string) that can be mapped to an index
        if obj_image is not None and HAS_PRIMITIVE_EXTRACTOR and _load_cnn_model():
            try:
                # Get the class names from the loaded CNN model (PerceptionModule)
                model_instance = _load_cnn_model()
                # Assuming PerceptionModule has a method to get attribute class names
                # Or, you might need a global mapping from config
                
                # For this example, let's assume `val` is directly comparable to a list of class names
                # from the config or a hardcoded list for the specific attribute type.
                
                # A more robust way would be to get the actual class names from the model's attribute head output
                # or from a global mapping like ATTRIBUTE_SHAPE_MAP etc.
                
                # Using the inverse maps defined at the top of primitive_extractor.py
                class_names_for_attr = list(attribute_maps_inv.get(feat_type, {}).values())
                
                if val in class_names_for_attr:
                    label_idx = class_names_for_attr.index(val)
                    crop_tensor = to_tensor(obj_image) # Convert PIL Image to Tensor
                    online_buffer.push(crop_tensor, label_idx)
                    logger.debug(f"Pushed object {obj_id} ({feat_type}: {val}) to online buffer with pseudo-label {label_idx}.")
                else:
                    logger.warning(f"Could not find class name '{val}' for attribute '{feat_type}' in class_names_for_attr. Skipping pushing to buffer.")
            except Exception as e:
                logger.error(f"Error pushing to online buffer for obj {obj_id}, feat {feat_type}: {e}", exc_info=True)


        # Add to proposed set (now includes confidence)
        self.proposed.add((obj_id, feat_type, val, conf))
        self.workspace['proposed_features_count'] = len(self.proposed)  # Update workspace state
        logger.debug(f"Primitive feature extracted: obj={obj_id}, feat={feat_type}, val={val}, conf={conf:.4f}")
        return val, conf

    def confirm_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]: # Now returns (value, score)
        """
        Confirms a feature's consistency or re-evaluates its strength.
        This is the interface for StrengthTester codelets.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to confirm.
        Returns:
            Tuple[Any, float]: The re-extracted feature value and a score (0.0 to 1.0)
                               indicating the confirmation strength.
        """
        # Re-extract the feature to ensure consistency or get an updated confidence.
        obj_image = self.sg.get_object_image(obj_id)
        if obj_image is None:
            logger.warning(f"No image available for object {obj_id}. Cannot re-confirm feature. Returning ('unknown', 0.0).")
            return "unknown", 0.0
        
        re_extracted_val, re_extracted_conf = "unknown", 0.0
        if HAS_PRIMITIVE_EXTRACTOR:
            if self.config['model'].get('use_cnn_features', True):
                cnn_feats = extract_cnn_features(obj_image) # This now includes TTA
                if feat_type in cnn_feats:
                    re_extracted_val, re_extracted_conf = cnn_feats[feat_type]
            else:
                if feat_type == 'shape': re_extracted_val, re_extracted_conf = extract_shape_conf(obj_image)
                elif feat_type == 'fill': re_extracted_val, re_extracted_conf = extract_fill_conf(obj_image)
                # Add more CV-based feature re-extractions
        else:
            re_extracted_val, re_extracted_conf = self.sg.extract_feature(obj_id, feat_type)  # Fallback to dummy SG extractor

        # Find the original proposed feature value to compare against
        original_proposed_val = None
        # original_proposed_conf = 0.0 # Not used for score calculation directly here, but could be.
        for prop_obj_id, prop_feat_type, prop_val, prop_conf in self.proposed:
            if prop_obj_id == obj_id and prop_feat_type == feat_type:
                original_proposed_val = prop_val
                # original_proposed_conf = prop_conf
                break
        
        # Confirmation strength is based on matching value and re-extracted confidence
        score = 0.0
        if original_proposed_val is not None and re_extracted_val == original_proposed_val:
            score = re_extracted_conf  # Use the re-extracted confidence if values match
        
        logger.debug(f"Feature confirmed: obj={obj_id}, feat={feat_type}, re-extracted_val={re_extracted_val}, score={score:.4f}")
        return re_extracted_val, score # Return the value along with the score

    def build_feature(self, obj_id: str, feat_type: str, value: Any, confidence: float): # Added 'value' parameter
        """
        Builds (commits) a feature into the workspace's explicit feature representation.
        This is the interface for Builder codelets.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to build.
            value (Any): The value of the feature.
            confidence (float): The confidence of the feature.
        """
        self.features[obj_id][feat_type] = (value, confidence)  # Store value and confidence
        self.built.add((obj_id, feat_type, value, confidence))  # Add confidence to built set
        self.workspace['built_features_count'] = len(self.built)  # Update workspace state
        logger.info(f"Built feature: obj={obj_id}, feat={feat_type}, val={value}, conf={confidence:.4f}. Total built: {len(self.built)}")

    def is_conflict(self, struct_id: Any, threshold: float = 0.2) -> bool:
        """
        Checks if a given structure is in conflict with existing built structures.
        This is a placeholder and needs concrete implementation based on your
        definition of conflicts (e.g., contradictory features on the same object).
        Args:
            struct_id (Any): The identifier of the structure to check (e.g., a tuple).
            threshold (float): A threshold for conflict detection.
        Returns:
            bool: True if a conflict is detected, False otherwise.
        """
        logger.debug(f"Checking for conflict for structure: {struct_id} (threshold: {threshold})")
        # Example: if struct_id is (obj_id, feat_type, value, confidence)
        # Check if there's an existing conflicting value for the same obj_id and feat_type
        if isinstance(struct_id, tuple) and len(struct_id) == 4:  # Now expecting 4 elements
            obj_id, feat_type, new_val, new_conf = struct_id
            if obj_id in self.features and feat_type in self.features[obj_id]:
                existing_val, existing_conf = self.features[obj_id][feat_type]
                if existing_val != new_val and new_conf > existing_conf + threshold:  # New is different and significantly more confident
                    logger.warning(f"Conflict detected: obj {obj_id}, feat {feat_type}. Existing: {existing_val} ({existing_conf:.2f}), New: {new_val} ({new_conf:.2f})")
                    return True
        # More complex conflict detection logic would go here, e.g., for relations
        return False

    def remove_structure(self, struct_id: Any):
        """
        Removes a structure from the workspace's built representation.
        This is the interface for Breaker codelets.
        Args:
            struct_id (Any): The identifier of the structure to remove.
        """
        if struct_id in self.built:
            self.built.remove(struct_id)
            # Also remove from self.features if it's a simple feature
            if isinstance(struct_id, tuple) and len(struct_id) == 4:  # Now expecting 4 elements
                obj_id, feat_type, _, _ = struct_id
                if obj_id in self.features and feat_type in self.features[obj_id]:
                    del self.features[obj_id][feat_type]
                    if not self.features[obj_id]:  # Remove object entry if no features left
                        del self.features[obj_id]
            self.workspace['built_features_count'] = len(self.built)  # Update workspace state
            logger.info(f"Removed conflicting structure: {struct_id}. Total built: {len(self.built)}")
        else:
            logger.warning(f"Attempted to remove non-existent structure: {struct_id}")

    def structure_coherence(self) -> float:
        """
        Calculates a measure of coherence for the built structures.
        This can be used by the temperature computation.
        Returns:
            float: A coherence score (e.g., ratio of built to proposed features).
        """
        if not self.proposed:
            return 1.0 if not self.built else 0.0  # If nothing proposed, and nothing built, coherence is perfect. If something built without proposal, 0.
        
        # Calculate weighted coherence based on confidence
        total_proposed_confidence = sum(conf for _, _, _, conf in self.proposed)
        total_built_confidence = sum(conf for _, _, _, conf in self.built)
        if total_proposed_confidence == 0:
            return 1.0 if total_built_confidence == 0 else 0.0
        coherence = total_built_confidence / total_proposed_confidence
        logger.debug(f"Structure coherence: {coherence:.4f} (Built conf: {total_built_confidence:.2f}, Proposed conf: {total_proposed_confidence:.2f})")
        return coherence
    
    def get_workspace_snapshot(self) -> Dict[str, Any]:
        """Returns a snapshot of the current workspace state for logging/visualization."""
        # Update dynamic parts of the workspace snapshot
        # Filter rule fragments by a confidence threshold
        self.workspace['current_rule_fragments'] = [
            rf for rf in self.current_rule_fragments
            if rf.get('confidence', 0.0) > self.config.get('slipnet_config', {}).get('activation_threshold', 0.1)  # Example threshold
        ]
        # Log active concepts in the workspace for dashboard visualization
        self.workspace['active_concepts'] = self.concept_net.get_active_nodes(
            threshold=self.config.get('slipnet_config', {}).get('activation_threshold', 0.1)
        )
        self.workspace['coderack_size'] = len(self.coderack)
        self.workspace['built_features_count'] = len(self.built)
        self.workspace['proposed_features_count'] = len(self.proposed)
        
        logger.debug(f"SymbolicEngine: Reasoning step {self.workspace['step']} complete. Active concepts: {self.workspace['active_concepts']}")
        logger.debug(f"SymbolicEngine: Current rule fragments: {len(self.workspace['current_rule_fragments'])}")
        
        return self.workspace

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running workspace_ext.py example.")

    # Dummy config for testing
    dummy_config = {
        'model': {
            'perception_model_path': None, # No real model loaded for this test
            'use_cnn_features': True, # Use CNN features (will use dummy if real not available)
            'attribute_classifier_config': {'shape': 5, 'color': 7, 'size': 3, 'fill': 2, 'orientation': 4, 'texture': 2},
            'relation_gnn_config': {'hidden_dim': 256, 'num_relations': 11},
            'bongard_head_config': {'num_classes': 2, 'hidden_dim': 256}
        },
        'data': {'image_size': [64, 64]},
        'slipnet_config': {'activation_threshold': 0.1}
    }

    # Create dummy images (numpy arrays)
    dummy_images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(2)]

    # Initialize Workspace
    ws = Workspace(dummy_images, dummy_config)
    logger.info(f"Workspace initialized with {len(ws.objects)} objects.")

    # Test posting and running a Scout codelet
    logger.info("\n--- Testing Scout Codelet ---")
    if ws.objects:
        first_obj_id = ws.objects[0]
        ws.post_codelet(Scout(first_obj_id, 'shape', urgency=0.5))
        ws.run_codelets(temperature=0.5, max_steps=1)
        logger.info(f"Proposed features after Scout: {ws.proposed}")
        logger.info(f"Coderack size after Scout: {len(ws.coderack)}")
    else:
        logger.warning("No objects found in dummy images for Scout test.")

    # Test running a StrengthTester codelet (if one was posted by Scout)
    logger.info("\n--- Testing StrengthTester Codelet ---")
    if ws.coderack:
        ws.run_codelets(temperature=0.5, max_steps=1)
        logger.info(f"Built features after StrengthTester: {ws.built}")
        logger.info(f"Coderack size after StrengthTester: {len(ws.coderack)}")

    # Test pushing to online buffer (if a feature was built)
    logger.info("\n--- Testing Online Buffer Integration ---")
    if len(online_buffer) > 0:
        logger.info(f"Online buffer size: {len(online_buffer)}")
        # You can inspect items in the buffer if needed for debugging
        # img_tensor, label_idx = online_buffer.buffer[0]
        # logger.info(f"First item in buffer: img_tensor shape {img_tensor.shape}, label_idx {label_idx}")
    else:
        logger.warning("Online buffer is empty. No features were pushed.")

    # Test structure coherence
    logger.info(f"\nStructure Coherence: {ws.structure_coherence():.4f}")

    # Test workspace snapshot
    snapshot = ws.get_workspace_snapshot()
    logger.info(f"\nWorkspace Snapshot: {snapshot}")
