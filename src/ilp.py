# Folder: bongard_solver/src/ilp.py

import logging
from typing import List, Dict, Any, Tuple, Optional
import collections
import random

# Import BongardRule and RuleAtom from bongard_rules
try:
    from bongard_rules import BongardRule, RuleAtom, ALL_RULE_ATOMS
except ImportError:
    logging.warning("Could not import bongard_rules in ILP. Using dummy BongardRule and RuleAtom.")
    class RuleAtom:
        def __init__(self, name, is_value=False, arity=0):
            self.name = name
            self.is_value = is_value
            self.arity = arity
        def __repr__(self): return self.name
    class BongardRule:
        def __init__(self, name, description, program_ast, logical_facts):
            self.name = name
            self.description = description
            self.program_ast = program_ast
            self.logical_facts = logical_facts
        def __repr__(self): return self.name
    ALL_RULE_ATOMS = {
        "SHAPE": RuleAtom("SHAPE", arity=2), "COLOR": RuleAtom("COLOR", arity=2),
        "circle": RuleAtom("circle", is_value=True), "red": RuleAtom("red", is_value=True),
        "AND": RuleAtom("AND", arity=-1), "OR": RuleAtom("OR", arity=-1), "NOT": RuleAtom("NOT", arity=1),
        "FORALL": RuleAtom("FORALL", arity=2), "EXISTS": RuleAtom("EXISTS", arity=2),
        "object_variable": RuleAtom("object_variable", is_value=True),
        "IMPLIES": RuleAtom("IMPLIES", arity=2)
    }

# Import DSL components for AST construction and fact conversion
try:
    from dsl import ASTToFactsTransducer, ASTNode, Primitive, DSLProgram, DSL_VALUES, DSL_FUNCTIONS, DSL
except ImportError:
    logging.warning("Could not import DSL components in ILP. Rule generation will be limited.")
    # Dummy classes if dsl.py is not fully accessible
    class ASTNode:
        def __init__(self, primitive, children=None, value=None): self.primitive = primitive; self.children = children or []; self.value = value
        def to_dict(self): return {"op": self.primitive.name, "args": [c.to_dict() for c in self.children]}
    class Primitive:
        def __init__(self, name, func, type_signature, is_terminal=False): self.name = name; self.func = func; self.type_signature = type_signature; self.is_terminal = is_terminal
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
        self.primitive_map = {p.name: p for p in DSL_VALUES + DSL_FUNCTIONS}
        logger.info("RuleInducer initialized.")

    @classmethod
    def generate(cls, facts: List[str]) -> List[BongardRule]:
        """
        Generates a list of candidate Bongard rules based on the provided DSL facts.
        This is a heuristic rule generation process, not a full-fledged ILP solver.
        It looks for patterns in the facts to propose simple rules.
        Args:
            facts (List[str]): A list of logical facts (e.g., "SHAPE(obj_0,circle)").
        Returns:
            List[BongardRule]: A list of proposed BongardRule objects.
        """
        logger.info(f"ILP: Generating rules from {len(facts)} facts.")
        candidate_rules: List[BongardRule] = []

        # Parse facts to extract attributes and relations
        # Example fact format: "SHAPE(obj_0,circle)" or "LEFT_OF(obj_0,obj_1)"
        parsed_facts = []
        for fact_str in facts:
            try:
                # Basic parsing: split by '(' and ',' and ')'
                parts = fact_str.replace('(', ',').replace(')', '').split(',')
                op = parts[0].strip().upper()
                args = [arg.strip() for arg in parts[1:] if arg.strip()]
                parsed_facts.append({'op': op, 'args': args, 'raw': fact_str})
            except Exception as e:
                logger.warning(f"ILP: Could not parse fact string '{fact_str}': {e}")
                continue

        # Heuristic 1: Look for common attributes across objects
        # Example: "All objects are circles" or "All objects are red"
        attribute_counts = collections.defaultdict(lambda: collections.defaultdict(int)) # attr_type -> value -> count
        object_ids = set()
        for fact in parsed_facts:
            if len(fact['args']) == 2 and fact['args'][0].startswith('obj_'): # Attribute fact
                attr_type = fact['op']
                obj_id = fact['args'][0]
                attr_value = fact['args'][1]
                attribute_counts[attr_type][attr_value] += 1
                object_ids.add(obj_id)
        
        num_objects = len(object_ids)
        if num_objects == 0:
            logger.warning("ILP: No objects found in facts to generate attribute rules.")

        for attr_type, values_count in attribute_counts.items():
            for value, count in values_count.items():
                if count >= num_objects * 0.8 and num_objects > 0: # If most objects have this attribute
                    # Propose a FORALL rule
                    try:
                        # Construct AST for FORALL(O, attr_type(O, value))
                        obj_var_node = ASTNode(cls().primitive_map["object_variable"], value="O")
                        value_node = ASTNode(cls().primitive_map[value], value=value)
                        
                        # Ensure attribute_type exists in primitive_map
                        if attr_type not in cls().primitive_map:
                            logger.warning(f"ILP: Attribute type '{attr_type}' not found in primitive map. Skipping rule.")
                            continue

                        predicate_node = ASTNode(cls().primitive_map[attr_type], 
                                                 children=[obj_var_node, value_node])
                        root_node = ASTNode(cls().primitive_map["FORALL"], 
                                            children=[obj_var_node, predicate_node])
                        
                        program = DSLProgram(root_node)
                        logical_facts = cls().transducer.convert(program)
                        
                        rule_name = f"FORALL_{attr_type}_{value}"
                        rule_desc = f"All objects have {attr_type.lower()} {value.lower()}"
                        candidate_rules.append(BongardRule(rule_name, rule_desc, [root_node.to_dict()], logical_facts))
                        logger.debug(f"ILP: Proposed rule: {rule_name}")
                    except KeyError as e:
                        logger.warning(f"ILP: Missing primitive for rule construction: {e}. Fact: {attr_type}(O,{value})")
                    except Exception as e:
                        logger.error(f"ILP: Error constructing FORALL rule for {attr_type}(O,{value}): {e}")

        # Heuristic 2: Look for common relations between objects (if multiple objects exist)
        # Example: "All objects are left_of another object" (simplified)
        if num_objects > 1:
            relation_counts = collections.defaultdict(int) # relation_type -> count
            for fact in parsed_facts:
                if len(fact['args']) == 2 and fact['args'][0].startswith('obj_') and fact['args'][1].startswith('obj_'): # Relational fact
                    relation_type = fact['op']
                    relation_counts[relation_type] += 1
            
            for rel_type, count in relation_counts.items():
                # If a relation appears frequently, propose an EXISTS rule (simplified)
                if count >= num_objects * (num_objects - 1) * 0.2: # If a significant portion of possible pairs have this relation
                    try:
                        # Propose EXISTS(O1, EXISTS(O2, rel_type(O1, O2))) - very simplified
                        obj_var1_node = ASTNode(cls().primitive_map["object_variable"], value="O1")
                        obj_var2_node = ASTNode(cls().primitive_map["object_variable"], value="O2")

                        if rel_type not in cls().primitive_map:
                            logger.warning(f"ILP: Relation type '{rel_type}' not found in primitive map. Skipping rule.")
                            continue

                        rel_predicate_node = ASTNode(cls().primitive_map[rel_type], 
                                                     children=[obj_var1_node, obj_var2_node])
                        
                        exists_o2_node = ASTNode(cls().primitive_map["EXISTS"], 
                                                 children=[obj_var2_node, rel_predicate_node])
                        
                        root_node = ASTNode(cls().primitive_map["EXISTS"], 
                                            children=[obj_var1_node, exists_o2_node])
                        
                        program = DSLProgram(root_node)
                        logical_facts = cls().transducer.convert(program)

                        rule_name = f"EXISTS_REL_{rel_type}"
                        rule_desc = f"There exists a pair of objects with relation {rel_type.lower()}"
                        candidate_rules.append(BongardRule(rule_name, rule_desc, [root_node.to_dict()], logical_facts))
                        logger.debug(f"ILP: Proposed rule: {rule_name}")
                    except KeyError as e:
                        logger.warning(f"ILP: Missing primitive for rule construction: {e}. Relation: {rel_type}")
                    except Exception as e:
                        logger.error(f"ILP: Error constructing EXISTS relation rule for {rel_type}: {e}")

        # Fallback: if no specific rules are found, propose a very generic rule
        if not candidate_rules:
            logger.info("ILP: No specific rules generated. Proposing a generic rule.")
            # Example: A simple rule like "EXISTS(O, SHAPE(O, circle))"
            try:
                obj_var_node = ASTNode(cls().primitive_map["object_variable"], value="O")
                circle_node = ASTNode(cls().primitive_map["circle"], value="circle")
                shape_predicate_node = ASTNode(cls().primitive_map["SHAPE"], children=[obj_var_node, circle_node])
                root_node = ASTNode(cls().primitive_map["EXISTS"], children=[obj_var_node, shape_predicate_node])
                program = DSLProgram(root_node)
                logical_facts = cls().transducer.convert(program)
                candidate_rules.append(BongardRule(
                    "Generic_Exists_Circle", 
                    "There exists at least one circle.", 
                    [root_node.to_dict()], 
                    logical_facts
                ))
            except Exception as e:
                logger.error(f"ILP: Error constructing generic rule: {e}")

        logger.info(f"ILP: Generated {len(candidate_rules)} candidate rules.")
        return candidate_rules

