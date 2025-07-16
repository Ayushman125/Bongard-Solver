# Folder: bongard_solver/src/
# File: ilp.py

import logging
from typing import List, Dict, Any, Tuple, Optional
import collections
import random

# Import BongardRule and RuleAtom from bongard_rules
try:
    from src.bongard_rules import BongardRule, RuleAtom, ALL_RULE_ATOMS
except ImportError:
    logging.warning("Could not import bongard_rules in ILP. Using dummy BongardRule and RuleAtom.")
    class RuleAtom:
        def __init__(self, name, is_value=False, arity=0):
            self.name = name
            self.is_value = is_value
            self.arity = arity
        def __repr__(self): return self.name
    class BongardRule:
        def __init__(self, name, description, program_ast, logical_facts, is_positive_rule=True):
            self.name = name
            self.description = description
            self.program_ast = program_ast
            self.logical_facts = logical_facts
            self.is_positive_rule = is_positive_rule
        def __repr__(self): return self.name
    ALL_RULE_ATOMS = {
        "SHAPE": RuleAtom("SHAPE", arity=2), "COLOR": RuleAtom("COLOR", arity=2),
        "circle": RuleAtom("circle", is_value=True), "red": RuleAtom("red", is_value=True),
        "AND": RuleAtom("AND", arity=-1), "OR": RuleAtom("OR", arity=-1), "NOT": RuleAtom("NOT", arity=1),
        "FORALL": RuleAtom("FORALL", arity=2), "EXISTS": RuleAtom("EXISTS", arity=2),
        "object_variable": RuleAtom("object_variable", is_value=True),
        "IMPLIES": RuleAtom("IMPLIES", arity=2),
        "GT": RuleAtom("GT", arity=2), "LT": RuleAtom("LT", arity=2), "EQ": RuleAtom("EQ", arity=2),
        "COUNT": RuleAtom("COUNT", arity=2),
        "INT_1": RuleAtom("INT_1", is_value=True), "INT_2": RuleAtom("INT_2", is_value=True), # Added is_value=True
        "INT_3": RuleAtom("INT_3", is_value=True), "INT_4": RuleAtom("INT_4", is_value=True),
        "INT_5": RuleAtom("INT_5", is_value=True),
    }

# Import DSL components for AST construction and fact conversion
try:
    from src.dsl import ASTToFactsTransducer, ASTNode, Primitive, DSLProgram, DSL_VALUES, DSL_FUNCTIONS, DSL
except ImportError:
    logging.warning("Could not import DSL components in ILP. Rule generation will be limited.")
    # Dummy classes if dsl.py is not fully accessible
    class ASTNode:
        def __init__(self, primitive, children=None, value=None): self.primitive = primitive; self.children = children or []; self.value = value
        def to_dict(self): return {"op": self.primitive.name, "args": [c.to_dict() for c in self.children]}
    class Primitive:
        def __init__(self, name, func=None, type_signature=None, is_terminal=False): self.name = name; self.func = func; self.type_signature = type_signature; self.is_terminal = is_terminal
    class DSLProgram:
        def __init__(self, root_node): self.root = root_node
    class ASTToFactsTransducer:
        def convert(self, program): return [f"DUMMY_FACT({program.root.primitive.name})"]
    DSL_VALUES = []
    DSL_FUNCTIONS = []
    class DSL:
        facts = set()
        @classmethod
        def get_facts(cls): return list(cls.facts)

logger = logging.getLogger(__name__)

class RuleInducer:
    """
    Inductive Logic Programming (ILP) module for Bongard problems.
    This simplified ILP system generates candidate rules based on observed facts
    from the DSL. It prioritizes common attributes and relations.
    """
    def __init__(self):
        self.transducer = ASTToFactsTransducer()
        # Map RuleAtom names to DSL Primitive objects for AST construction
        # Ensure primitive_map is populated correctly from DSL_VALUES and DSL_FUNCTIONS
        self.primitive_map = {p.name: p for p in DSL_VALUES + DSL_FUNCTIONS}
        
        # Add RuleAtoms that might not be directly in DSL_VALUES/FUNCTIONS but are used in rule construction
        for atom in ALL_RULE_ATOMS.values():
            if atom.name not in self.primitive_map:
                # Create a dummy Primitive for RuleAtoms not found in DSL_VALUES/FUNCTIONS
                self.primitive_map[atom.name] = Primitive(atom.name, None, "unknown_type", is_terminal=atom.is_value)
                logger.debug(f"ILP: Added dummy Primitive for RuleAtom '{atom.name}' to primitive_map.")

        logger.info("RuleInducer initialized.")

    def generate(self, facts: List[Tuple[str, float]]) -> List[BongardRule]:
        """
        Generates a list of candidate Bongard rules based on the provided DSL facts,
        now including confidence.
        This is a heuristic rule generation process, not a full-fledged ILP solver.
        It looks for patterns in the facts to propose simple rules.
        Args:
            facts (List[Tuple[str, float]]): A list of logical facts (e.g., ("SHAPE(obj_0,circle)", 0.9)).
        Returns:
            List[BongardRule]: A list of proposed BongardRule objects.
        """
        logger.info(f"ILP: Generating rules from {len(facts)} facts.")
        candidate_rules: List[BongardRule] = []

        # Parse facts to extract attributes and relations, considering confidence
        parsed_facts = []
        for fact_str, confidence in facts:
            try:
                # Basic parsing: split by '(' and ',' and ')'
                parts = fact_str.replace('(', ',').replace(')', '').split(',')
                op = parts[0].strip().upper()
                args = [arg.strip() for arg in parts[1:] if arg.strip()]
                parsed_facts.append({'op': op, 'args': args, 'raw': fact_str, 'confidence': confidence})
            except Exception as e:
                logger.warning(f"ILP: Could not parse fact string '{fact_str}': {e}")
                continue

        # Heuristic 1: Look for common attributes across objects
        # attr_type -> value -> {obj_id: confidence}
        attribute_data = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))
        object_ids = set()
        for fact in parsed_facts:
            # Check for attribute facts: e.g., SHAPE(obj_0,circle)
            if len(fact['args']) == 2 and fact['args'][0].startswith('obj_'):
                attr_type = fact['op']
                obj_id = fact['args'][0]
                attr_value = fact['args'][1]
                attribute_data[attr_type][attr_value][obj_id] = fact['confidence']
                object_ids.add(obj_id)
        
        num_objects = len(object_ids)
        if num_objects == 0:
            logger.warning("ILP: No objects found in facts to generate attribute rules.")

        for attr_type, values_data in attribute_data.items():
            for value, obj_confidences in values_data.items():
                # Consider a rule if a high percentage of objects have this attribute with high confidence
                # Sum of confidences for this attribute / number of objects
                avg_confidence_for_attribute = sum(obj_confidences.values()) / num_objects if num_objects > 0 else 0
                
                # If most objects have this attribute with a reasonable average confidence
                if avg_confidence_for_attribute >= 0.7: # Threshold for "common" attribute
                    # Propose a FORALL rule: FORALL(O, attr_type(O, value))
                    try:
                        obj_var_node = ASTNode(self.primitive_map["object_variable"], value="O")
                        
                        value_primitive = self.primitive_map.get(value)
                        if value_primitive:
                            value_node = ASTNode(value_primitive, value=value)
                        else:
                            value_node = ASTNode(Primitive(value, None, "unknown_type", is_terminal=True), value=value)
                            logger.warning(f"ILP: Value '{value}' not found in DSL primitives. Using raw string as terminal.")
                        
                        attr_primitive = self.primitive_map.get(attr_type.lower())
                        if not attr_primitive:
                            logger.warning(f"ILP: Attribute type '{attr_type}' not found in primitive map. Skipping rule.")
                            continue
                        
                        predicate_node = ASTNode(attr_primitive, children=[obj_var_node, value_node])
                        root_node = ASTNode(self.primitive_map["FORALL"], children=[obj_var_node, predicate_node])
                        
                        program = DSLProgram(root_node)
                        logical_facts = self.transducer.convert(program)
                        
                        rule_name = f"FORALL_{attr_type}_{value}"
                        rule_desc = f"All objects have {attr_type.lower()} {value.lower()}"
                        candidate_rules.append(BongardRule(rule_name, rule_desc, [root_node.to_dict()], logical_facts))
                        logger.debug(f"ILP: Proposed rule: {rule_name}")
                    except KeyError as e:
                        logger.warning(f"ILP: Missing primitive for rule construction: {e}. Fact: {attr_type}(O,{value})")
                    except Exception as e:
                        logger.error(f"ILP: Error constructing FORALL rule for {attr_type}(O,{value}): {e}", exc_info=True)

        # Heuristic 2: Look for common relations between objects (if multiple objects exist)
        if num_objects > 1:
            # relation_type -> { (obj1, obj2): confidence }
            relation_data = collections.defaultdict(lambda: collections.defaultdict(float))
            for fact in parsed_facts:
                if len(fact['args']) == 2 and fact['args'][0].startswith('obj_') and fact['args'][1].startswith('obj_'):
                    rel_type = fact['op']
                    obj1, obj2 = fact['args'][0], fact['args'][1]
                    relation_data[rel_type][(obj1, obj2)] = fact['confidence']
            
            for rel_type, pairs_confidences in relation_data.items():
                # Consider a rule if a high percentage of *possible* pairs have this relation
                # with high average confidence.
                num_pairs_with_relation = len(pairs_confidences)
                total_possible_pairs = num_objects * (num_objects - 1) # Directed pairs
                avg_confidence_for_relation = sum(pairs_confidences.values()) / num_pairs_with_relation if num_pairs_with_relation > 0 else 0

                if total_possible_pairs > 0 and \
                   (num_pairs_with_relation / total_possible_pairs) >= 0.2 and \
                   avg_confidence_for_relation >= 0.6: # Threshold for "common" relation
                    try:
                        # Propose EXISTS(O1, EXISTS(O2, rel_type(O1, O2))) - very simplified
                        obj_var1_node = ASTNode(self.primitive_map["object_variable"], value="O1")
                        obj_var2_node = ASTNode(self.primitive_map["object_variable"], value="O2")
                        rel_primitive = self.primitive_map.get(rel_type.lower())
                        if not rel_primitive:
                            logger.warning(f"ILP: Relation type '{rel_type}' not found in primitive map. Skipping rule.")
                            continue
                        rel_predicate_node = ASTNode(rel_primitive, children=[obj_var1_node, obj_var2_node])
                        
                        exists_o2_node = ASTNode(self.primitive_map["EXISTS"], children=[obj_var2_node, rel_predicate_node])
                        root_node = ASTNode(self.primitive_map["EXISTS"], children=[obj_var1_node, exists_o2_node])
                        
                        program = DSLProgram(root_node)
                        logical_facts = self.transducer.convert(program)
                        rule_name = f"EXISTS_REL_{rel_type}"
                        rule_desc = f"There exists a pair of objects with relation {rel_type.lower()}"
                        candidate_rules.append(BongardRule(rule_name, rule_desc, [root_node.to_dict()], logical_facts))
                        logger.debug(f"ILP: Proposed rule: {rule_name}")
                    except KeyError as e:
                        logger.warning(f"ILP: Missing primitive for rule construction: {e}. Relation: {rel_type}")
                    except Exception as e:
                        logger.error(f"ILP: Error constructing EXISTS relation rule for {rel_type}: {e}", exc_info=True)

        # Heuristic 3: Count-based rules (e.g., COUNT(shape, circle) == 3)
        # This requires parsing COUNT facts. For now, assume a simple structure.
        count_data = collections.defaultdict(lambda: collections.defaultdict(int)) # attr_type -> value -> count
        for fact in parsed_facts:
            # Example: COUNT(shape,circle,3) (assuming 3 is the count, not an object ID)
            if fact['op'] == 'COUNT' and len(fact['args']) == 3:
                attr_type = fact['args'][0]
                value = fact['args'][1]
                count_val = int(fact['args'][2]) # Assuming it's an integer
                count_data[attr_type][value] = count_val

        for attr_type, values_counts in count_data.items():
            for value, count_val in values_counts.items():
                # Propose a rule like EQ(COUNT(O, attr_type(O, value)), count_val)
                try:
                    obj_var_node = ASTNode(self.primitive_map["object_variable"], value="O")
                    value_node = ASTNode(self.primitive_map.get(value, Primitive(value, None, "unknown_type", is_terminal=True)), value=value)
                    attr_primitive = self.primitive_map.get(attr_type.lower())
                    if not attr_primitive: continue

                    attr_predicate_node = ASTNode(attr_primitive, children=[obj_var_node, value_node])
                    count_node = ASTNode(self.primitive_map["COUNT"], children=[obj_var_node, attr_predicate_node])
                    
                    # Find the integer primitive for count_val
                    int_primitive_name = f"INT_{count_val}"
                    int_primitive = self.primitive_map.get(int_primitive_name)
                    if not int_primitive:
                        logger.warning(f"ILP: Integer primitive '{int_primitive_name}' not found. Skipping count rule.")
                        continue
                    int_node = ASTNode(int_primitive, value=count_val)

                    root_node = ASTNode(self.primitive_map["EQ"], children=[count_node, int_node])
                    
                    program = DSLProgram(root_node)
                    logical_facts = self.transducer.convert(program)
                    rule_name = f"COUNT_{attr_type}_{value}_EQ_{count_val}"
                    rule_desc = f"The count of objects with {attr_type.lower()} {value.lower()} is {count_val}"
                    candidate_rules.append(BongardRule(rule_name, rule_desc, [root_node.to_dict()], logical_facts))
                    logger.debug(f"ILP: Proposed rule: {rule_name}")
                except KeyError as e:
                    logger.warning(f"ILP: Missing primitive for count rule construction: {e}. Fact: COUNT({attr_type},{value},{count_val})")
                except Exception as e:
                    logger.error(f"ILP: Error constructing count rule: {e}", exc_info=True)


        # Fallback: if no specific rules are found, propose a very generic rule
        if not candidate_rules:
            logger.info("ILP: No specific rules generated. Proposing a generic rule.")
            # Example: A simple rule like "EXISTS(O, SHAPE(O, circle))"
            try:
                obj_var_node = ASTNode(self.primitive_map["object_variable"], value="O")
                circle_node = ASTNode(self.primitive_map["circle"], value="circle")
                shape_predicate_node = ASTNode(self.primitive_map["SHAPE"], children=[obj_var_node, circle_node])
                root_node = ASTNode(self.primitive_map["EXISTS"], children=[obj_var_node, shape_predicate_node])
                program = DSLProgram(root_node)
                logical_facts = self.transducer.convert(program)
                candidate_rules.append(BongardRule(
                    "Generic_Exists_Circle",
                    "There exists at least one circle.",
                    [root_node.to_dict()],
                    logical_facts
                ))
            except Exception as e:
                logger.error(f"ILP: Error constructing generic rule: {e}", exc_info=True)
        
        logger.info(f"ILP: Generated {len(candidate_rules)} candidate rules.")
        return candidate_rules

