# Folder: bongard_solver/src/emergent/
# File: codelets.py

import random
import logging
import collections
from typing import List, Dict, Any, Tuple, Set, Optional

logger = logging.getLogger(__name__)

# Import DSL and ILP components for RuleTester/RuleBuilder
try:
    from src.dsl import DSL, ASTToFactsTransducer, ASTNode, Primitive, DSLProgram, DSL_VALUES, DSL_FUNCTIONS
    from src.ilp import RuleInducer
    # from src.rule_evaluator import evaluate_rule_on_support_set # Assuming this exists for RuleTester
    HAS_DSL_ILP = True
except ImportError:
    logger.warning("Could not import DSL/ILP components. RuleTester/RuleBuilder will be dummy.")
    HAS_DSL_ILP = False
    # Dummy classes/functions if imports fail
    class DSL:
        facts = set()
        @classmethod
        def get_facts(cls): return list(cls.facts)
    class ASTToFactsTransducer:
        def convert(self, program): return ["DUMMY_FACT"]
    class ASTNode:
        def __init__(self, primitive, children=None, value=None): self.primitive = primitive; self.children = children or []; self.value = value
    class Primitive:
        def __init__(self, name, func=None, type_signature=None, is_terminal=False): self.name = name; self.func = func; self.type_signature = type_signature; self.is_terminal = is_terminal
    class DSLProgram:
        def __init__(self, root_node): self.root = root_node
    class RuleInducer:
        def generate(self, facts): return []
    # Dummy rule evaluator
    def evaluate_rule_on_support_set(rule_desc: str, support_set_scene_graphs: List[Dict[str, Any]], support_set_labels: List[int]) -> float:
        logger.warning("Using dummy evaluate_rule_on_support_set. Rule accuracy will be random.")
        return random.random() # Random accuracy for dummy


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
        # Call into your primitive_extractor via workspace.primitive_feature
        val, confidence = workspace.primitive_feature(self.obj_id, self.feat_type)
        
        # Calculate strength based on confidence, concept activation, and temperature
        concept_activation = concept_net.get_node_activation(self.feat_type) # Use get_node_activation
        strength = confidence * concept_activation * temperature
        
        logger.debug(f"Scout result: val={val}, conf={confidence:.4f}, concept_act={concept_activation:.4f}, strength={strength:.4f}")
        
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
        super().__init__(strength)  # Urgency is based on the strength of the proposal
        self.obj_id = obj_id
        self.feat_type = feat_type
        logger.debug(f"StrengthTester created for obj {obj_id}, feat {feat_type} with strength {strength}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the StrengthTester codelet. Confirms a feature and potentially posts a Builder.
        """
        logger.debug(f"Running StrengthTester: obj={self.obj_id}, feat={self.feat_type}")
        # The confirm_feature now returns a score based on confidence and consistency
        score = workspace.confirm_feature(self.obj_id, self.feat_type)
        
        # Get the actual value and confidence from the workspace's proposed set
        # This is a bit indirect, but we need the value for the Builder
        val_from_proposed = None
        conf_from_proposed = 0.0
        for prop_obj_id, prop_feat_type, prop_val, prop_conf in workspace.proposed:
            if prop_obj_id == self.obj_id and prop_feat_type == self.feat_type:
                val_from_proposed = prop_val
                conf_from_proposed = prop_conf
                break

        logger.debug(f"StrengthTester score for {self.obj_id}, {self.feat_type}: {score:.4f}")
        
        # Use a threshold for posting a Builder, and pass the confidence
        if score > workspace.config['slipnet_config'].get('activation_threshold', 0.1): # Use a configurable threshold
            if val_from_proposed is not None:
                logger.debug(f"StrengthTester confirmed feature, posting Builder for {self.obj_id}, {self.feat_type} with score {score:.4f}")
                # Pass the confidence to the Builder
                workspace.post_codelet(Builder(self.obj_id, self.feat_type, val_from_proposed, conf_from_proposed, score))
            else:
                logger.warning(f"StrengthTester found no value in proposed set for {self.obj_id}, {self.feat_type}. Cannot post Builder.")
        else:
            logger.debug(f"StrengthTester did not confirm feature (score too low).")

class Builder(Codelet):
    """
    A Builder codelet constructs or 'builds' a feature into the workspace's
    representation and activates the corresponding concept in the Concept Network.
    """
    def __init__(self, obj_id: str, feat_type: str, value: Any, confidence: float, strength: float):
        """
        Initializes a Builder codelet.
        Args:
            obj_id (str): The ID of the object.
            feat_type (str): The type of feature to build.
            value (Any): The value of the feature.
            confidence (float): The confidence associated with this feature.
            strength (float): The strength (urgency) of this builder, typically
                              inherited from the StrengthTester's score.
        """
        super().__init__(strength)  # Urgency is based on the confirmed strength
        self.obj_id = obj_id
        self.feat_type = feat_type
        self.value = value
        self.confidence = confidence
        logger.debug(f"Builder created for obj {obj_id}, feat {feat_type}, val {value}, conf {confidence:.4f} with strength {strength}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the Builder codelet. Builds the feature and activates the concept.
        """
        logger.debug(f"Running Builder: obj={self.obj_id}, feat={self.feat_type}, val={self.value}, conf={self.confidence:.4f}")
        # Pass the confidence to build_feature
        workspace.build_feature(self.obj_id, self.feat_type, self.confidence)
        
        # Activate the concept in the concept network based on this successful build
        # Activate both the feature type and the value concept
        concept_net.activate_node(self.feat_type, self.urgency) # e.g., 'shape'
        concept_net.activate_node(str(self.value), self.urgency) # e.g., 'circle'
        logger.debug(f"Builder built feature {self.feat_type} for {self.obj_id} and activated concepts.")

        # After building a feature, consider posting a GroupScout or RuleTester
        # if enough features are built or certain concepts are highly active.
        if len(workspace.built) >= 2: # If at least two features are built, consider grouping or rule testing
            if random.random() < temperature * 0.3: # Probabilistically post GroupScout
                workspace.post_codelet(GroupScout(urgency=self.urgency * 0.8)) # Lower urgency for subsequent tasks
                logger.debug("Builder posted a GroupScout.")
            
            # Also consider posting a RuleTester if some basic facts are available
            if HAS_DSL_ILP and len(workspace.built) >= 1: # At least one built feature to start proposing rules
                # A very simple rule proposal: "All objects have this shape"
                if self.feat_type == 'shape' and random.random() < temperature * 0.2:
                    rule_desc = f"FORALL(O, SHAPE(O, {str(self.value).upper()}))"
                    workspace.post_codelet(RuleTester(rule_desc, urgency=self.urgency * 0.9))
                    logger.debug(f"Builder posted a RuleTester for: {rule_desc}")


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
                                 This could be a tuple (obj_id, feat_type, value, confidence) or similar.
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
        if workspace.is_conflict(self.structure_id, threshold=workspace.config['debug'].get('conflict_threshold', 0.2)):
            logger.info(f"Conflict detected for structure {self.structure_id}. Removing it.")
            workspace.remove_structure(self.structure_id)
            # Potentially activate 'conflict' concept or related concepts
            concept_net.activate_node('conflict', 0.8)
        else:
            logger.debug(f"No significant conflict detected for structure {self.structure_id}.")

class GroupScout(Codelet):
    """
    A GroupScout codelet attempts to discover emergent groups or clusters of objects
    based on their shared features or spatial relationships.
    """
    def __init__(self, urgency: float = 0.1):
        super().__init__(urgency)
        logger.debug(f"GroupScout created with urgency {urgency}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        logger.debug("Running GroupScout.")
        
        # Get all currently built features
        built_features = list(workspace.built)
        
        if len(built_features) < 2:
            logger.debug("GroupScout: Not enough built features to form a group.")
            return

        # Simple grouping heuristic: objects sharing the same dominant shape or color
        # This is a placeholder; real grouping would involve more sophisticated clustering.
        
        # Group by shape
        shape_groups = collections.defaultdict(list)
        for obj_id, feat_type, value, confidence in built_features:
            if feat_type == 'shape' and confidence > 0.5: # Consider only confident shape detections
                shape_groups[value].append(obj_id)
        
        for shape_val, obj_ids in shape_groups.items():
            if len(obj_ids) >= 2: # Found a group of at least two objects with the same shape
                group_description = f"GROUP(SHAPE={shape_val}, OBJECTS={obj_ids})"
                logger.info(f"GroupScout proposed group: {group_description}")
                # Activate 'group' concept and the specific shape concept
                concept_net.activate_node('group', self.urgency)
                concept_net.activate_node(shape_val, self.urgency)
                
                # Consider posting a RuleTester for this group
                if HAS_DSL_ILP and random.random() < temperature * 0.5:
                    # Example rule proposal: "All objects in this group have this shape"
                    rule_desc = f"FORALL(O, IN_GROUP(O, {group_description}) IMPLIES SHAPE(O, {shape_val.upper()}))"
                    workspace.post_codelet(RuleTester(rule_desc, urgency=self.urgency * 1.2)) # Higher urgency for rule testing
                    logger.debug(f"GroupScout posted a RuleTester for group: {rule_desc}")

        # Add more grouping heuristics (e.g., by color, by proximity, by relation)
        # Example: Group by spatial proximity (requires object coordinates from scene graph)
        # This would require accessing scene graph data from workspace.sg
        # For now, this is conceptual.

class RuleTester(Codelet):
    """
    A RuleTester codelet takes a proposed rule (as a string description or AST)
    and evaluates its plausibility against the current workspace state or a small
    set of examples. If plausible, it may post a RuleBuilder.
    """
    def __init__(self, rule_description: str, urgency: float = 0.1):
        super().__init__(urgency)
        self.rule_description = rule_description
        logger.debug(f"RuleTester created for rule: '{rule_description}' with urgency {urgency}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        logger.debug(f"Running RuleTester for rule: '{self.rule_description}'")
        
        if not HAS_DSL_ILP:
            logger.warning("DSL/ILP components not available. RuleTester cannot evaluate rules.")
            return

        # Evaluate the proposed rule against the current problem's support set
        # This requires the Workspace to have access to the support set's scene graphs and labels.
        # For this example, we'll assume the workspace has a way to get this.
        # In main.py, the RL agent provides support_set_scene_graphs to the environment.
        # Here, we need to mock or pass it down.
        
        # Dummy support set for evaluation within RuleTester (replace with actual data from Workspace)
        # In a real scenario, the Workspace would hold the full problem context.
        dummy_support_sgs = [
            {'image_path': 'dummy_pos1.png', 'scene_graph': [{'id': 'obj_0', 'attributes': {'shape': 'circle'}}]},
            {'image_path': 'dummy_pos2.png', 'scene_graph': [{'id': 'obj_1', 'attributes': {'shape': 'circle'}}]},
            {'image_path': 'dummy_neg1.png', 'scene_graph': [{'id': 'obj_2', 'attributes': {'shape': 'square'}}]},
        ]
        dummy_support_labels = [1, 1, 0] # Example: first two are positive, last is negative

        # This `evaluate_rule_on_support_set` would come from `src/rule_evaluator.py`
        # For now, it's a dummy function.
        # The rule_evaluator expects a BongardRule object, not just a string.
        # We need to parse the rule_description into a BongardRule object or its AST/logical facts.
        
        # Simple parsing of rule_description to a dummy BongardRule for evaluation
        # This needs to be robust, ideally using the DSL parser.
        rule_name = self.rule_description.replace(' ', '_').replace('(', '').replace(')', '')
        # Create a dummy BongardRule object for evaluation
        dummy_program_ast = [{"op": "DUMMY_OP", "args": []}] # Placeholder
        dummy_logical_facts = [f"DUMMY_FACT({rule_name})"] # Placeholder
        
        # Check if BongardRule is available (from bongard_rules.py)
        try:
            from src.bongard_rules import BongardRule
            proposed_rule_obj = BongardRule(
                name=rule_name,
                description=self.rule_description,
                program_ast=dummy_program_ast,
                logical_facts=dummy_logical_facts
            )
        except ImportError:
            logger.warning("BongardRule not available. Cannot create rule object for evaluation.")
            proposed_rule_obj = None # Cannot proceed with evaluation

        rule_accuracy = 0.0
        if proposed_rule_obj:
            # Call the actual rule evaluator (if imported)
            try:
                from src.rule_evaluator import evaluate_rule_on_support_set
                rule_accuracy = evaluate_rule_on_support_set(proposed_rule_obj, dummy_support_sgs, dummy_support_labels)
            except ImportError:
                logger.warning("src.rule_evaluator.py not found. Using dummy rule accuracy.")
                rule_accuracy = random.random() # Dummy accuracy if evaluator not found
            except Exception as e:
                logger.error(f"Error evaluating rule '{self.rule_description}': {e}", exc_info=True)
                rule_accuracy = 0.0

        # Confidence in the rule proposal
        confidence = rule_accuracy # Simple confidence based on accuracy
        
        logger.debug(f"RuleTester evaluated rule '{self.rule_description}' with accuracy {rule_accuracy:.4f} and confidence {confidence:.4f}.")

        # Store the rule fragment (description and confidence) in the workspace
        workspace.current_rule_fragments.append({
            'rule_description': self.rule_description,
            'confidence': confidence,
            'source_codelet': type(self).__name__
        })

        # If the rule is plausible, post a RuleBuilder
        if confidence > workspace.config['slipnet_config'].get('activation_threshold', 0.1): # Configurable threshold
            logger.debug(f"RuleTester confirmed rule, posting RuleBuilder for '{self.rule_description}' with confidence {confidence:.4f}")
            workspace.post_codelet(RuleBuilder(self.rule_description, confidence, urgency=self.urgency * 1.5)) # Higher urgency for building confirmed rules
            # Activate 'rule' concept
            concept_net.activate_node('rule', self.urgency * 1.5)
        else:
            logger.debug(f"RuleTester did not confirm rule (confidence too low).")


class RuleBuilder(Codelet):
    """
    A RuleBuilder codelet takes a confirmed rule proposal and attempts to
    construct a formal BongardRule object, potentially adding it to the DSL library
    or marking it as a solution candidate.
    """
    def __init__(self, rule_description: str, confidence: float, urgency: float = 0.1):
        super().__init__(urgency)
        self.rule_description = rule_description
        self.confidence = confidence
        logger.debug(f"RuleBuilder created for rule: '{rule_description}' with confidence {confidence:.4f} and urgency {urgency}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        logger.debug(f"Running RuleBuilder for rule: '{self.rule_description}'")
        
        if not HAS_DSL_ILP:
            logger.warning("DSL/ILP components not available. RuleBuilder cannot build rules.")
            return

        # Attempt to formally construct the BongardRule object
        # This involves parsing the rule_description into an AST and logical facts.
        # This is where the ILP's AST construction logic would be used.
        
        # Dummy AST construction for now
        root_node = ASTNode(Primitive("DUMMY_RULE_OP"), children=[
            ASTNode(Primitive("DUMMY_ARG1"), value="O"),
            ASTNode(Primitive("DUMMY_ARG2"), value="CIRCLE")
        ])
        program = DSLProgram(root_node)
        
        transducer = ASTToFactsTransducer()
        logical_facts = transducer.convert(program)
        
        # Create the BongardRule object
        try:
            from src.bongard_rules import BongardRule
            built_rule = BongardRule(
                name=self.rule_description.replace(' ', '_').replace('(', '').replace(')', ''),
                description=self.rule_description,
                program_ast=[root_node.to_dict()], # Store AST as a list of dicts
                logical_facts=logical_facts,
                is_positive_rule=True # Assume positive for now
            )
            logger.info(f"RuleBuilder successfully built rule: {built_rule.name}")
            
            # Mark as a potential solution in SceneGraphBuilder if confidence is high
            if self.confidence > workspace.config['slipnet_config'].get('solution_threshold', 0.8): # Configurable solution threshold
                workspace.sg.mark_solution(built_rule.description)
                logger.info(f"RuleBuilder marked solution: {built_rule.description} (Confidence: {self.confidence:.4f})")
                # Activate 'problem' concept
                concept_net.activate_node('problem', self.urgency * 2) # High activation for solved problem
            else:
                logger.debug("RuleBuilder did not mark solution (confidence too low).")

        except ImportError:
            logger.error("BongardRule not available. Cannot create rule object.")
        except Exception as e:
            logger.error(f"Error building rule '{self.rule_description}': {e}", exc_info=True)

