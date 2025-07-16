# Folder: bongard_solver/src/emergent/
# File: codelets.py
import random
import logging
import collections
from typing import List, Dict, Any, Tuple, Set, Optional
logger = logging.getLogger(__name__)

# Import DSL and ILP components for RuleTester/RuleBuilder
HAS_DSL_ILP = False
try:
    from src.dsl import DSL, ASTToFactsTransducer, ASTNode, Primitive, DSLProgram, DSL_VALUES, DSL_FUNCTIONS
    from src.ilp import RuleInducer
    # Assuming src.rule_evaluator.py exists and has evaluate_rule_on_support_set
    from src.rule_evaluator import evaluate_rule_on_support_set
    from src.bongard_rules import BongardRule # Needed for RuleTester/RuleBuilder
    HAS_DSL_ILP = True
except ImportError as e:
    logger.warning(f"Could not import DSL/ILP components or rule_evaluator/BongardRule: {e}. Rule-related codelets will be dummy.")
    # Dummy classes/functions if imports fail
    class DSL:
        facts = set()
        @classmethod
        def get_facts(cls): return list(cls.facts)
    class ASTToFactsTransducer:
        def convert(self, program): return ["DUMMY_FACT"]
    class ASTNode:
        def __init__(self, primitive, children=None, value=None): self.primitive = primitive; self.children = children or []; self.value = value
        def to_dict(self): return {'primitive': self.primitive.name, 'value': self.value, 'children': [c.to_dict() for c in self.children]}
    class Primitive:
        def __init__(self, name, func=None, type_signature=None, is_terminal=False): self.name = name; self.func = func; self.type_signature = type_signature; self.is_terminal = is_terminal
    class DSLProgram:
        def __init__(self, root_node): self.root = root_node
    class RuleInducer:
        def generate(self, facts): return []
    # Dummy rule evaluator
    def evaluate_rule_on_support_set(rule_obj: Any, support_set_scene_graphs: List[Dict[str, Any]], support_set_labels: List[int]) -> float:
        logger.warning("Using dummy evaluate_rule_on_support_set. Rule accuracy will be random.")
        return random.random() # Random accuracy for dummy
    class BongardRule:
        def __init__(self, name: str = "dummy_rule", description: str = "A dummy rule", program_ast: List = [], logical_facts: List = [], is_positive_rule: bool = True):
            self.name = name
            self.description = description
            self.program_ast = program_ast
            self.logical_facts = logical_facts
            self.is_positive_rule = is_positive_rule

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
        super().__init__(strength) # Urgency is based on the strength of the proposal
        self.obj_id = obj_id
        self.feat_type = feat_type
        logger.debug(f"StrengthTester created for obj {obj_id}, feat {feat_type} with strength {strength}")
    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        """
        Runs the StrengthTester codelet. Confirms a feature and potentially posts a Builder.
        """
        logger.debug(f"Running StrengthTester: obj={self.obj_id}, feat={self.feat_type}")
        # The confirm_feature now returns a score based on confidence and consistency
        val, score = workspace.confirm_feature(self.obj_id, self.feat_type) # confirm_feature now returns value and score
        
        logger.debug(f"StrengthTester score for {self.obj_id}, {self.feat_type}: {score:.4f}")
        
        # Use a threshold for posting a Builder, and pass the confidence
        if score > workspace.config['slipnet_config'].get('activation_threshold', 0.1): # Use a configurable threshold
            if val is not None: # val is now returned by confirm_feature
                logger.debug(f"StrengthTester confirmed feature, posting Builder for {self.obj_id}, {self.feat_type} with score {score:.4f}")
                # Pass the confidence (which is part of the score) to the Builder
                workspace.post_codelet(Builder(self.obj_id, self.feat_type, val, score, score)) # Use score as confidence and strength
            else:
                logger.warning(f"StrengthTester found no value after confirming for {self.obj_id}, {self.feat_type}. Cannot post Builder.")
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
        super().__init__(strength) # Urgency is based on the confirmed strength
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
        workspace.build_feature(self.obj_id, self.feat_type, self.value, self.confidence) # build_feature now takes value
        
        # Activate the concept in the concept network based on this successful build
        # Activate both the feature type and the value concept
        concept_net.activate_node(self.feat_type, self.urgency) # e.g., 'shape'
        concept_net.activate_node(str(self.value), self.urgency) # e.g., 'circle'
        logger.debug(f"Builder built feature {self.feat_type} for {self.obj_id} and activated concepts.")
        
        # After building a feature, consider posting a GroupScout or RuleProposer
        if len(workspace.built) >= 2: # If at least two features are built, consider grouping or rule proposing
            if random.random() < temperature * 0.3: # Probabilistically post GroupScout
                workspace.post_codelet(GroupScout(urgency=self.urgency * 0.8)) # Lower urgency for subsequent tasks
                logger.debug("Builder posted a GroupScout.")
            
            # Also consider posting a RuleProposer if some basic facts are available
            # This logic can be more sophisticated, e.g., only after certain types of features are built
            if HAS_DSL_ILP and random.random() < temperature * 0.4: # Probabilistically post RuleProposer
                # The RuleProposer will scan the workspace, so no specific rule is passed here
                workspace.post_codelet(RuleProposer(urgency=self.urgency * 1.0)) # Urgency based on current builder
                logger.debug("Builder posted a RuleProposer.")

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
                
                # Consider posting a RuleProposer for this group
                if HAS_DSL_ILP and random.random() < temperature * 0.5:
                    # The RuleProposer will scan the workspace, so no specific rule is passed here
                    workspace.post_codelet(RuleProposer(urgency=self.urgency * 1.2)) # Higher urgency for rule testing
                    logger.debug(f"GroupScout posted a RuleProposer for group.")
        
        # Add more grouping heuristics (e.g., by color, by proximity, by relation)
        # Example: Group by spatial proximity (requires object coordinates from scene graph)
        # This would require accessing scene graph data from workspace.sg
        # For now, this is conceptual.

class RuleProposer(Codelet):
    """
    A RuleProposer codelet scans the workspace's built features and relations
    to generate candidate rule templates (DSL expressions).
    It then posts RuleTester codelets for these candidates.
    """
    def __init__(self, urgency: float = 0.1):
        super().__init__(urgency)
        logger.debug(f"RuleProposer created with urgency {urgency}")

    def run(self, workspace: 'Workspace', concept_net: 'ConceptNetwork', temperature: float):
        logger.debug("Running RuleProposer.")

        if not HAS_DSL_ILP:
            logger.warning("DSL/ILP components not available. RuleProposer cannot generate rules.")
            return

        # Get all currently built features and relations from the workspace
        built_features = list(workspace.built) # (obj_id, feat_type, value, confidence)
        # Access relations from the scene graph in workspace.sg.current_scene_graph
        # Note: current_scene_graph might be a list of scene graphs if multiple images are processed
        # For simplicity, let's assume we are proposing rules based on the first image's scene graph
        current_scene_graphs = workspace.sg.current_scene_graph_data # This is a list of SGs per image
        
        candidate_rules = set() # Use a set to avoid duplicate rule proposals

        # Rule Proposal Strategy 1: Attribute-based rules (e.g., all objects are 'circle')
        # Iterate through built features and propose simple attribute rules
        for obj_id, feat_type, value, confidence in built_features:
            if confidence > 0.7: # Only propose rules based on high-confidence features
                # Example: FORALL(O, SHAPE(O, CIRCLE))
                rule_desc = f"FORALL(O, {feat_type.upper()}(O, {str(value).upper()}))"
                candidate_rules.add(rule_desc)
                # Example: EXISTS(O, SHAPE(O, CIRCLE))
                rule_desc_exists = f"EXISTS(O, {feat_type.upper()}(O, {str(value).upper()}))"
                candidate_rules.add(rule_desc_exists)

        # Rule Proposal Strategy 2: Relation-based rules (e.g., A is LEFT_OF B)
        if current_scene_graphs:
            for sg_data in current_scene_graphs:
                if 'relations' in sg_data:
                    for rel in sg_data['relations']:
                        subj_id = rel['subject_id']
                        obj_id = rel['object_id']
                        rel_type = rel['type']
                        rel_confidence = rel.get('confidence', 1.0)

                        if rel_confidence > 0.7: # Only propose rules based on high-confidence relations
                            # Example: LEFT_OF(OBJ_0, OBJ_1)
                            rule_desc = f"{rel_type.upper()}({subj_id.upper()}, {obj_id.upper()})"
                            candidate_rules.add(rule_desc)
                            # Example: FORALL(A, B, LEFT_OF(A, B)) (more abstract)
                            rule_desc_forall = f"FORALL(A, B, {rel_type.upper()}(A, B))"
                            candidate_rules.add(rule_desc_forall)
                            # Example: EXISTS(A, B, LEFT_OF(A, B))
                            rule_desc_exists = f"EXISTS(A, B, {rel_type.upper()}(A, B))"
                            candidate_rules.add(rule_desc_exists)

        # Rule Proposal Strategy 3: Combined attribute-relation rules
        # Example: FORALL(O, IF(SHAPE(O, CIRCLE), THEN(COLOR(O, RED))))
        for obj_id, feat_type1, val1, conf1 in built_features:
            if conf1 > 0.7:
                for sg_data in current_scene_graphs:
                    if 'relations' in sg_data:
                        for rel in sg_data['relations']:
                            if rel['subject_id'] == obj_id and rel.get('confidence', 1.0) > 0.7:
                                rule_desc = f"FORALL(O, IF({feat_type1.upper()}(O, {str(val1).upper()}), THEN({rel['type'].upper()}(O, {rel['object_id'].upper()}))))"
                                candidate_rules.add(rule_desc)

        # Post RuleTester codelets for each candidate rule
        if candidate_rules:
            logger.info(f"RuleProposer generated {len(candidate_rules)} candidate rules.")
            for rule_desc in candidate_rules:
                # Urgency for RuleTester can be influenced by temperature and confidence of underlying facts
                # For now, a simple urgency based on RuleProposer's urgency
                workspace.post_codelet(RuleTester(rule_desc, urgency=self.urgency * temperature))
                logger.debug(f"RuleProposer posted RuleTester for: {rule_desc}")
            concept_net.activate_node('rule_proposal', self.urgency) # Activate concept for rule proposal
        else:
            logger.debug("RuleProposer found no strong candidate rules to propose.")


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
        # The workspace should provide access to the support set's scene graphs and labels.
        # This is passed into the solve function in main.py and should be accessible via workspace.
        
        # Access the support set from the workspace
        support_set_scene_graphs = workspace.support_set_scene_graphs # List of dicts
        support_set_labels = workspace.support_set_labels # List of ints
        
        if not support_set_scene_graphs or not support_set_labels:
            logger.warning("RuleTester: No support set data available in workspace for evaluation. Skipping rule test.")
            return

        # Parse the rule_description into a BongardRule object or its AST/logical facts.
        # This is a simplified parsing for demonstration; a full DSL parser would be used.
        rule_name = self.rule_description.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
        
        # Create a dummy BongardRule object for evaluation.
        # In a real system, this would involve parsing the rule_description string
        # into a structured BongardRule object with its AST and logical facts.
        # For now, we'll use a simplified rule structure.
        
        # Attempt to infer is_positive_rule based on rule description keywords
        is_positive_rule = "FORALL" in self.rule_description.upper() or "EXISTS" in self.rule_description.upper()
        
        proposed_rule_obj = BongardRule(
            name=rule_name,
            description=self.rule_description,
            program_ast=[], # Placeholder for actual AST
            logical_facts=[], # Placeholder for actual logical facts
            is_positive_rule=is_positive_rule # Set based on simple heuristic
        )
        
        rule_accuracy = 0.0
        try:
            # Call the actual rule evaluator
            # evaluate_rule_on_support_set expects a BongardRule object
            rule_accuracy = evaluate_rule_on_support_set(proposed_rule_obj, support_set_scene_graphs, support_set_labels)
        except Exception as e:
            logger.error(f"Error evaluating rule '{self.rule_description}': {e}", exc_info=True)
            rule_accuracy = 0.0 # Fallback on error

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
        if confidence > workspace.config['slipnet_config'].get('rule_plausibility_threshold', 0.6): # Configurable threshold
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
        # This is where the DSL parser and ILP's AST construction logic would be used.
        
        # For demonstration, let's use a simplified parsing and AST construction.
        # In a real system, you'd call DSL.parse(self.rule_description)
        
        # Example: Simple parsing for 'FORALL(O, SHAPE(O, CIRCLE))'
        rule_name = self.rule_description.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
        
        # Placeholder for actual AST and logical facts from DSL parsing
        parsed_ast = [] # This would be the result of DSL parsing
        parsed_logical_facts = [] # This would be the result of ASTToFactsTransducer
        
        # Attempt to infer is_positive_rule based on rule description keywords
        is_positive_rule = "FORALL" in self.rule_description.upper() or "EXISTS" in self.rule_description.upper()

        try:
            # Create the BongardRule object
            built_rule = BongardRule(
                name=rule_name,
                description=self.rule_description,
                program_ast=parsed_ast, # Use parsed AST
                logical_facts=parsed_logical_facts, # Use parsed logical facts
                is_positive_rule=is_positive_rule # Set based on simple heuristic
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
        except Exception as e:
            logger.error(f"Error building rule '{self.rule_description}': {e}", exc_info=True)

