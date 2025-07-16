# src/emergent/workspace_ext.py

import heapq
import logging
from typing import List, Dict, Any, Tuple, Set, Optional

# Import emergent modules
from emergent.codelets import Scout, StrengthTester, Builder, Breaker, Codelet
from emergent.concept_net import ConceptNetwork

# Assume scene_graph_builder is in the parent directory (src/)
# This import might need adjustment based on your exact project structure.
# If scene_graph_builder is not directly importable, you might need to add
# src/ to your Python path or adjust the import statement.
try:
    from scene_graph_builder import SceneGraphBuilder
    HAS_SCENE_GRAPH_BUILDER = True
    logger = logging.getLogger(__name__)
    logger.info("scene_graph_builder.py found for Workspace.")
except ImportError:
    HAS_SCENE_GRAPH_BUILDER = False
    logger = logging.getLogger(__name__)
    logger.warning("scene_graph_builder.py not found. Workspace will use a dummy SceneGraphBuilder.")
    # Dummy SceneGraphBuilder if not found, to allow the rest of the code to run
    class SceneGraphBuilder:
        def __init__(self, images: List[Any]):
            logger.warning("Dummy SceneGraphBuilder initialized. Feature extraction will be mocked.")
            self.images = images
            self.objects = [f"obj_{i}" for i in range(len(images))] if images else ["obj_0"] # Mock object IDs
            self._solution_found = False # For problem_solved()
            self._solution = None

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

logger = logging.getLogger(__name__)

class Workspace:
    """
    The central workspace where emergent perception and cognitive processes occur.
    It hosts codelets, manages features, and interacts with the SceneGraphBuilder
    and ConceptNetwork.
    """
    def __init__(self, images: List[Any]):
        """
        Initializes the Workspace.
        Args:
            images (List[Any]): A list of image data (e.g., file paths, numpy arrays)
                                that the SceneGraphBuilder will process.
        """
        self.sg = SceneGraphBuilder(images) # Initialize your SceneGraphBuilder
        self.objects: List[str] = self.sg.objects  # List of object IDs from SceneGraphBuilder
        
        # Stores confirmed features: obj_id -> {feat_type: value}
        self.features: Dict[str, Dict[str, Any]] = {}              
        
        # Tracks proposed features that a Scout has identified: (obj_id, feat_type, value)
        self.proposed: Set[Tuple[str, str, Any]] = set()           
        
        # Tracks features that have been successfully 'built' by a Builder: (obj_id, feat_type, value)
        self.built: Set[Tuple[str, str, Any]] = set()              
        
        # A min-heap (priority queue) of (-urgency, codelet) tuples.
        # Negative urgency ensures higher urgency codelets have lower values and are popped first.
        self.coderack: List[Tuple[float, Codelet]] = []            
        
        self.concept_net = ConceptNetwork() # Initialize the Concept Network
        logger.info("Workspace initialized.")
        logger.debug(f"Initial objects in workspace: {self.objects}")

    def post_codelet(self, codelet: Codelet):
        """
        Adds a codelet to the coderack.
        Args:
            codelet (Codelet): The codelet instance to add.
        """
        heapq.heappush(self.coderack, (-codelet.urgency, codelet))
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
            except NotImplementedError:
                logger.error(f"Codelet {type(codelet).__name__} has not implemented the 'run' method.")
            except Exception as e:
                logger.error(f"Error running codelet {type(codelet).__name__}: {e}", exc_info=True)
        logger.info(f"Finished running codelets. Steps taken: {steps_taken}. Remaining in coderack: {len(self.coderack)}")

    def primitive_feature(self, obj_id: str, feat_type: str) -> Tuple[Any, float]:
        """
        Extracts a primitive feature for an object using the SceneGraphBuilder.
        This is the interface for Scout codelets.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to extract.
        Returns:
            Tuple[Any, float]: The extracted feature value and its confidence.
        """
        # This calls into your SceneGraphBuilder's feature extraction logic.
        # Ensure SceneGraphBuilder.extract_feature is implemented to return (value, confidence).
        val, conf = self.sg.extract_feature(obj_id, feat_type)
        # Add to proposed set to track what has been considered
        self.proposed.add((obj_id, feat_type, val))
        logger.debug(f"Primitive feature extracted: obj={obj_id}, feat={feat_type}, val={val}, conf={conf:.2f}")
        return val, conf

    def confirm_feature(self, obj_id: str, feat_type: str) -> float:
        """
        Confirms a feature's consistency or re-evaluates its strength.
        This is the interface for StrengthTester codelets.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to confirm.
        Returns:
            float: A score (0.0 to 1.0) indicating the confirmation strength.
        """
        # Re-extract the feature to ensure consistency or get an updated confidence.
        # In a more complex system, this might involve checking against other
        # features or higher-level structures.
        val_recheck, conf_recheck = self.sg.extract_feature(obj_id, feat_type)
        
        # Simple confirmation: check if the re-extracted value matches a previously proposed one
        # and use the confidence as the score.
        if (obj_id, feat_type, val_recheck) in self.proposed:
            logger.debug(f"Feature confirmed: obj={obj_id}, feat={feat_type}, val={val_recheck}, score={conf_recheck:.2f}")
            return conf_recheck
        else:
            logger.debug(f"Feature not confirmed (value mismatch or not previously proposed): obj={obj_id}, feat={feat_type}")
            return 0.0

    def build_feature(self, obj_id: str, feat_type: str):
        """
        Builds (commits) a feature into the workspace's explicit feature representation.
        This is the interface for Builder codelets.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to build.
        """
        # Get the current value from the SceneGraphBuilder (or from a confirmed source)
        val, _ = self.sg.extract_feature(obj_id, feat_type) # Assuming extract_feature can always give a value
        self.features.setdefault(obj_id, {})[feat_type] = val
        self.built.add((obj_id, feat_type, val))
        logger.info(f"Built feature: obj={obj_id}, feat={feat_type}, val={val}. Total built: {len(self.built)}")

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
        # Example: if struct_id is (obj_id, feat_type, value)
        # Check if there's an existing conflicting value for the same obj_id and feat_type
        if isinstance(struct_id, tuple) and len(struct_id) == 3:
            obj_id, feat_type, new_val = struct_id
            if obj_id in self.features and feat_type in self.features[obj_id]:
                existing_val = self.features[obj_id][feat_type]
                if existing_val != new_val:
                    logger.warning(f"Conflict detected: obj {obj_id}, feat {feat_type}. Existing: {existing_val}, New: {new_val}")
                    return True
        # More complex conflict detection logic would go here
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
            if isinstance(struct_id, tuple) and len(struct_id) == 3:
                obj_id, feat_type, _ = struct_id
                if obj_id in self.features and feat_type in self.features[obj_id]:
                    del self.features[obj_id][feat_type]
                    if not self.features[obj_id]: # Remove object if no features left
                        del self.features[obj_id]
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
            return 1.0 if not self.built else 0.0 # If nothing proposed, and nothing built, coherence is perfect. If something built without proposal, 0.
        coherence = len(self.built) / len(self.proposed)
        logger.debug(f"Structure coherence: {coherence:.4f} (Built: {len(self.built)}, Proposed: {len(self.proposed)})")
        return coherence

