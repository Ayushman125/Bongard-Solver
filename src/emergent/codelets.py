# src/emergent/codelets.py

import random
import logging

logger = logging.getLogger(__name__)

class Codelet:
    """
    Base class for all emergent codelets.
    Codelets are small, independent pieces of code that perform a specific task
    within the emergent perception system. They have an urgency that determines
    their priority in the coderack.
    """
    def __init__(self, urgency: float = 0.1):
        """
        Initializes a Codelet.
        Args:
            urgency (float): A value between 0 and 1 indicating the importance
                             or priority of this codelet. Higher urgency means
                             it's more likely to be run.
        """
        self.urgency = urgency

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Executes the logic of the codelet.
        This method must be overridden by subclasses.
        Args:
            workspace (Workspace): The current workspace containing objects, features, etc.
            concept_net (ConceptNetwork): The concept network to interact with.
            temperature (float): The current system temperature, influencing probabilistic actions.
        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

class Scout(Codelet):
    """
    A Scout codelet proposes a new feature for an object.
    It interacts with the workspace to extract a primitive feature and, based on
    confidence and concept activation, may post a StrengthTester codelet.
    """
    def __init__(self, obj_id: str, feat_type: str, urgency: float = 0.1):
        """
        Initializes a Scout codelet.
        Args:
            obj_id (str): The ID of the object to scout.
            feat_type (str): The type of feature to scout (e.g., 'shape', 'color').
            urgency (float): The initial urgency of this scout.
        """
        super().__init__(urgency)
        self.obj_id = obj_id
        self.feat_type = feat_type
        logger.debug(f"Scout created for obj {obj_id}, feat {feat_type} with urgency {urgency}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the Scout codelet. Proposes a feature and potentially posts a StrengthTester.
        """
        logger.debug(f"Running Scout: obj={self.obj_id}, feat={self.feat_type}")
        # Call into your crop_extraction, topo_features, etc. via workspace.sg
        val, confidence = workspace.primitive_feature(self.obj_id, self.feat_type)
        
        # Calculate strength based on confidence, concept activation, and temperature
        concept_activation = concept_net.get_activation(self.feat_type)
        strength = confidence * concept_activation * temperature
        
        logger.debug(f"Scout result: val={val}, conf={confidence}, concept_act={concept_activation}, strength={strength}")

        # Probabilistically post a StrengthTester based on calculated strength
        if random.random() < strength:
            logger.debug(f"Scout posting StrengthTester for {self.obj_id}, {self.feat_type} with strength {strength}")
            workspace.post_codelet(StrengthTester(self.obj_id, self.feat_type, strength))
        else:
            logger.debug(f"Scout did not post StrengthTester (random check failed or strength too low).")

class StrengthTester(Codelet):
    """
    A StrengthTester codelet confirms the validity of a proposed feature.
    It re-evaluates the feature and, if strong enough, posts a Builder codelet.
    """
    def __init__(self, obj_id: str, feat_type: str, strength: float):
        """
        Initializes a StrengthTester codelet.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature being tested.
            strength (float): The initial strength (urgency) of this tester, often
                              inherited from the proposing Scout.
        """
        super().__init__(strength) # Urgency is based on the strength of the proposal
        self.obj_id = obj_id
        self.feat_type = feat_type
        logger.debug(f"StrengthTester created for obj {obj_id}, feat {feat_type} with strength {strength}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the StrengthTester codelet. Confirms a feature and potentially posts a Builder.
        """
        logger.debug(f"Running StrengthTester: obj={self.obj_id}, feat={self.feat_type}")
        score = workspace.confirm_feature(self.obj_id, self.feat_type)
        logger.debug(f"StrengthTester score for {self.obj_id}, {self.feat_type}: {score}")

        if score > 0.5: # A threshold to consider the feature confirmed
            logger.debug(f"StrengthTester confirmed feature, posting Builder for {self.obj_id}, {self.feat_type} with score {score}")
            workspace.post_codelet(Builder(self.obj_id, self.feat_type, score))
        else:
            logger.debug(f"StrengthTester did not confirm feature (score too low).")

class Builder(Codelet):
    """
    A Builder codelet constructs or 'builds' a feature into the workspace's
    representation and activates the corresponding concept in the Concept Network.
    """
    def __init__(self, obj_id: str, feat_type: str, strength: float):
        """
        Initializes a Builder codelet.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to build.
            strength (float): The strength (urgency) of this builder, typically
                              inherited from the StrengthTester's score.
        """
        super().__init__(strength) # Urgency is based on the confirmed strength
        self.obj_id = obj_id
        self.feat_type = feat_type
        logger.debug(f"Builder created for obj {obj_id}, feat {feat_type} with strength {strength}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the Builder codelet. Builds the feature and activates the concept.
        """
        logger.debug(f"Running Builder: obj={self.obj_id}, feat={self.feat_type}")
        workspace.build_feature(self.obj_id, self.feat_type)
        
        # Activate the concept in the concept network based on this successful build
        concept_net.activate(self.feat_type, self.urgency)
        logger.debug(f"Builder built feature {self.feat_type} for {self.obj_id} and activated concept.")

class Breaker(Codelet):
    """
    A Breaker codelet detects and resolves conflicts or inconsistencies in the
    workspace's built structures.
    """
    def __init__(self, structure_id: Any, urgency: float = 0.1):
        """
        Initializes a Breaker codelet.
        Args:
            structure_id (Any): An identifier for the structure to check for conflicts.
                                 This could be a tuple (obj_id, feat_type, value) or similar.
            urgency (float): The initial urgency of this breaker.
        """
        super().__init__(urgency)
        self.structure_id = structure_id
        logger.debug(f"Breaker created for structure {structure_id} with urgency {urgency}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the Breaker codelet. Checks for conflicts and removes structures if necessary.
        """
        logger.debug(f"Running Breaker: structure={self.structure_id}")
        # Check if the structure is in conflict based on some threshold
        if workspace.is_conflict(self.structure_id, threshold=0.2):
            logger.info(f"Conflict detected for structure {self.structure_id}. Removing it.")
            workspace.remove_structure(self.structure_id)
        else:
            logger.debug(f"No significant conflict detected for structure {self.structure_id}.")

