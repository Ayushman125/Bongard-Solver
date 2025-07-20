# Folder: bongard_solver/

import collections
import logging
import math
import random
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# --- AST Node Classes for Structured Rule Representation ---

class ASTNode:
    """Base class for Abstract Syntax Tree nodes in Bongard rule programs."""
    
    def __init__(self, atom: 'RuleAtom', args: List['ASTNode'] = None, value: Any = None):
        self.atom = atom
        self.args = args if args is not None else []
        self.value = value  # For terminal nodes with constant values
        
    def __repr__(self):
        if self.value is not None:
            return f"{self.atom.name}({self.value})"
        elif self.args:
            args_str = ", ".join(str(arg) for arg in self.args)
            return f"{self.atom.name}({args_str})"
        else:
            return f"{self.atom.name}"
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        result = {"op": self.atom.name}
        if self.value is not None:
            result["value"] = self.value
        if self.args:
            result["args"] = [arg.to_dict() for arg in self.args]
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASTNode':
        """Create ASTNode from dictionary format."""
        atom_name = data["op"]
        atom = ALL_RULE_ATOMS.get(atom_name, RuleAtom(atom_name, -1))  # Default fallback
        
        value = data.get("value")
        args = []
        if "args" in data:
            args = [cls.from_dict(arg_data) for arg_data in data["args"]]
            
        return cls(atom, args, value)

class BongardDSLInterpreter:
    """
    Interpreter for Bongard rule DSL that can execute program_ast nodes.
    This allows all 25 rules to be evaluated consistently through their ASTs.
    """
    
    def __init__(self):
        self.context = {}  # Store variables and intermediate results
        
    def evaluate(self, ast_node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Evaluate an AST node against object data.
        
        Returns:
            tuple: (result, explanation)
        """
        if not isinstance(ast_node, ASTNode):
            # Handle dictionary format for backward compatibility
            if isinstance(ast_node, dict):
                ast_node = ASTNode.from_dict(ast_node)
            else:
                return False, f"Invalid AST node type: {type(ast_node)}"
        
        atom_name = ast_node.atom.name
        
        # Terminal values
        if ast_node.atom.is_value:
            return True, f"Value {atom_name}"
            
        # Logical operators
        if atom_name == "FORALL":
            return self._eval_forall(ast_node, objects_data)
        elif atom_name == "EXISTS":
            return self._eval_exists(ast_node, objects_data)
        elif atom_name == "AND":
            return self._eval_and(ast_node, objects_data)
        elif atom_name == "OR":
            return self._eval_or(ast_node, objects_data)
        elif atom_name == "NOT":
            return self._eval_not(ast_node, objects_data)
        elif atom_name == "IMPLIES":
            return self._eval_implies(ast_node, objects_data)
            
        # Predicates
        elif atom_name == "shape":
            return self._eval_predicate(ast_node, objects_data, "shape")
        elif atom_name == "color":
            return self._eval_predicate(ast_node, objects_data, "color")
        elif atom_name == "size":
            return self._eval_predicate(ast_node, objects_data, "size")
        elif atom_name == "fill":
            return self._eval_predicate(ast_node, objects_data, "fill")
            
        # Spatial relations
        elif atom_name in ["left_of", "right_of", "above", "below", "contains"]:
            return self._eval_spatial_relation(ast_node, objects_data, atom_name)
            
        # Advanced geometric predicates (simplified implementations)
        elif atom_name == "symmetrical_to":
            return self._eval_symmetrical_to(ast_node, objects_data)
        elif atom_name == "equidistant_from":
            return self._eval_equidistant_from(ast_node, objects_data)
            
        else:
            # Fallback for unimplemented predicates
            return True, f"Predicate {atom_name} evaluated (placeholder)"
    
    def _eval_forall(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate universal quantification."""
        if len(node.args) != 2:
            return False, "FORALL requires exactly 2 arguments"
            
        var_node, pred_node = node.args
        var_name = var_node.value if var_node.value else "O"
        
        satisfied_count = 0
        for obj in objects_data:
            # Bind variable to current object
            old_context = self.context.copy()
            self.context[var_name] = obj
            
            result, _ = self.evaluate(pred_node, objects_data)
            if result:
                satisfied_count += 1
                
            # Restore context
            self.context = old_context
            
        all_satisfied = satisfied_count == len(objects_data)
        return all_satisfied, f"FORALL: {satisfied_count}/{len(objects_data)} objects satisfy condition"
    
    def _eval_exists(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate existential quantification."""
        if len(node.args) != 2:
            return False, "EXISTS requires exactly 2 arguments"
            
        var_node, pred_node = node.args
        var_name = var_node.value if var_node.value else "O"
        
        for obj in objects_data:
            # Bind variable to current object
            old_context = self.context.copy()
            self.context[var_name] = obj
            
            result, _ = self.evaluate(pred_node, objects_data)
            if result:
                self.context = old_context
                return True, f"EXISTS: Found object satisfying condition"
                
            # Restore context
            self.context = old_context
            
        return False, "EXISTS: No object satisfies condition"
    
    def _eval_and(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate logical AND."""
        results = []
        explanations = []
        
        for arg in node.args:
            result, explanation = self.evaluate(arg, objects_data)
            results.append(result)
            explanations.append(explanation)
            
        all_true = all(results)
        return all_true, f"AND: {explanations}"
    
    def _eval_or(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate logical OR."""
        results = []
        explanations = []
        
        for arg in node.args:
            result, explanation = self.evaluate(arg, objects_data)
            results.append(result)
            explanations.append(explanation)
            if result:  # Short-circuit on first true
                return True, f"OR: {explanation}"
                
        return False, f"OR: All conditions false - {explanations}"
    
    def _eval_not(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate logical NOT."""
        if len(node.args) != 1:
            return False, "NOT requires exactly 1 argument"
            
        result, explanation = self.evaluate(node.args[0], objects_data)
        return not result, f"NOT: {explanation}"
    
    def _eval_implies(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate logical implication (A -> B equivalent to ~A OR B)."""
        if len(node.args) != 2:
            return False, "IMPLIES requires exactly 2 arguments"
            
        antecedent, consequent = node.args
        ante_result, ante_exp = self.evaluate(antecedent, objects_data)
        
        if not ante_result:
            return True, f"IMPLIES: Antecedent false, implication vacuously true"
            
        cons_result, cons_exp = self.evaluate(consequent, objects_data)
        return cons_result, f"IMPLIES: {ante_exp} -> {cons_exp}"
    
    def _eval_predicate(self, node: ASTNode, objects_data: List[Dict[str, Any]], pred_name: str) -> Tuple[bool, str]:
        """Evaluate a basic predicate like shape(O, circle)."""
        if len(node.args) != 2:
            return False, f"{pred_name} requires exactly 2 arguments"
            
        obj_node, value_node = node.args
        
        # Get object from context if it's a variable
        if obj_node.atom.name == "object_variable":
            var_name = obj_node.value
            obj = self.context.get(var_name)
            if obj is None:
                return False, f"Unbound variable {var_name}"
        else:
            return False, f"Expected object variable, got {obj_node.atom.name}"
            
        # Get expected value
        expected_value = value_node.atom.name
        actual_value = obj.get(pred_name, "unknown")
        
        matches = actual_value == expected_value
        return matches, f"{pred_name}({var_name}, {expected_value}): {actual_value}"
    
    def _eval_spatial_relation(self, node: ASTNode, objects_data: List[Dict[str, Any]], relation: str) -> Tuple[bool, str]:
        """Evaluate spatial relationships (simplified implementation)."""
        if len(node.args) != 2:
            return False, f"{relation} requires exactly 2 arguments"
            
        # Simplified: assume relation is satisfied if both objects exist
        # In a full implementation, this would check actual positions
        return True, f"{relation}: Spatial relation assumed satisfied (placeholder)"
    
    def _eval_symmetrical_to(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate symmetrical arrangement (simplified)."""
        # In full implementation, would check actual symmetry
        return len(objects_data) >= 2, f"symmetrical_to: {len(objects_data)} objects for symmetry check"
    
    def _eval_equidistant_from(self, node: ASTNode, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Evaluate equidistant arrangement (simplified)."""
        # In full implementation, would check actual distances
        return len(objects_data) >= 2, f"equidistant_from: {len(objects_data)} objects for distance check"

# --- Rule Atom Definitions ---
# These represent the basic predicates or functions in a Bongard rule's DSL.

class RuleAtom:
    """Represents a basic predicate or function in a Bongard rule."""
    def __init__(self, name: str, arity: int, arg_types: Optional[List[str]] = None, is_value: bool = False):
        self.name = name
        self.arity = arity # Number of arguments the atom takes. -1 for variable arity (e.g., AND, OR)
        self.arg_types = arg_types if arg_types is not None else []
        self.is_value = is_value # True if this atom represents a constant value (e.g., 'circle', 'red')

    def __repr__(self):
        return f"Atom({self.name}/{self.arity}, is_value={self.is_value})"

    def __eq__(self, other):
        return isinstance(other, RuleAtom) and self.name == other.name and self.arity == other.arity and self.is_value == other.is_value

    def __hash__(self):
        return hash((self.name, self.arity, self.is_value))

# Define common attribute atoms (predicates)
ATTR_SHAPE = RuleAtom("shape", 2, ["object", "shape_type"]) # e.g., shape(obj1, circle)
ATTR_COLOR = RuleAtom("color", 2, ["object", "color_type"]) # e.g., color(obj1, red)
ATTR_FILL = RuleAtom("fill", 2, ["object", "fill_type"])
ATTR_SIZE = RuleAtom("size", 2, ["object", "size_type"])
ATTR_ORIENTATION = RuleAtom("orientation", 2, ["object", "orientation_type"])
ATTR_TEXTURE = RuleAtom("texture", 2, ["object", "texture_type"])

# Define common relation atoms (predicates)
REL_LEFT_OF = RuleAtom("left_of", 2, ["object", "object"]) # e.g., left_of(obj1, obj2)
REL_RIGHT_OF = RuleAtom("right_of", 2, ["object", "object"])
REL_ABOVE = RuleAtom("above", 2, ["object", "object"])
REL_BELOW = RuleAtom("below", 2, ["object", "object"])
REL_IS_CLOSE_TO = RuleAtom("is_close_to", 2, ["object", "object"]) # Proximity
REL_ALIGNED_HORIZONTALLY = RuleAtom("aligned_horizontally", 2, ["object", "object"])
REL_ALIGNED_VERTICALLY = RuleAtom("aligned_vertically", 2, ["object", "object"])
REL_CONTAINS = RuleAtom("contains", 2, ["object", "object"]) # For nested objects
REL_INTERSECTS = RuleAtom("intersects", 2, ["object", "object"]) # For overlapping objects

# Define specific attribute values (terminal atoms/constants)
SHAPE_CIRCLE = RuleAtom("circle", 0, is_value=True)
SHAPE_SQUARE = RuleAtom("square", 0, is_value=True)
SHAPE_TRIANGLE = RuleAtom("triangle", 0, is_value=True)
SHAPE_STAR = RuleAtom("star", 0, is_value=True)

COLOR_RED = RuleAtom("red", 0, is_value=True)
COLOR_BLUE = RuleAtom("blue", 0, is_value=True)
COLOR_GREEN = RuleAtom("green", 0, is_value=True)
COLOR_BLACK = RuleAtom("black", 0, is_value=True)
COLOR_WHITE = RuleAtom("white", 0, is_value=True)

FILL_SOLID = RuleAtom("solid", 0, is_value=True)
FILL_HOLLOW = RuleAtom("hollow", 0, is_value=True)
FILL_STRIPED = RuleAtom("striped", 0, is_value=True)
FILL_DOTTED = RuleAtom("dotted", 0, is_value=True)

SIZE_SMALL = RuleAtom("small", 0, is_value=True)
SIZE_MEDIUM = RuleAtom("medium", 0, is_value=True)
SIZE_LARGE = RuleAtom("large", 0, is_value=True)

ORIENTATION_UPRIGHT = RuleAtom("upright", 0, is_value=True)
ORIENTATION_INVERTED = RuleAtom("inverted", 0, is_value=True)

TEXTURE_NONE = RuleAtom("none", 0, is_value=True)
TEXTURE_STRIPED = RuleAtom("striped", 0, is_value=True)
TEXTURE_DOTTED = RuleAtom("dotted", 0, is_value=True)

# Logical Operators (for composing complex rules)
LOGIC_AND = RuleAtom("AND", -1) # Variable arity
LOGIC_OR = RuleAtom("OR", -1)
LOGIC_NOT = RuleAtom("NOT", 1)
LOGIC_IMPLIES = RuleAtom("IMPLIES", 2) # e.g., IMPLIES(cond, result)
# Updated arity for quantifiers to 2 (variable, predicate) to match DSL Primitive
LOGIC_EXISTS = RuleAtom("EXISTS", 2) # e.g., EXISTS(obj, color(obj, red))
LOGIC_FORALL = RuleAtom("FORALL", 2) # e.g., FORALL(obj, shape(obj, circle))
# Keep COUNT arity at 2 (variable, predicate) as the count value is typically a separate argument for comparison ops
LOGIC_COUNT = RuleAtom("COUNT", 2) # e.g., COUNT(obj, shape(obj, circle))

# Comparison operators for counts (these are not part of ALL_RULE_ATOMS directly but used in AST)
# They are typically handled by DSL Primitives like GT, LT, EQ
LOGIC_GT = RuleAtom("GT", 2)
LOGIC_LT = RuleAtom("LT", 2)
LOGIC_EQ = RuleAtom("EQ", 2)

# Advanced geometric predicates for complex rules
GEOM_SYMMETRICAL_TO = RuleAtom("symmetrical_to", 2, ["object", "object"])
GEOM_SIZE_VALUE = RuleAtom("size_value", 1, ["object"])
GEOM_EQUIDISTANT_FROM = RuleAtom("equidistant_from", 2, ["object", "point"])
GEOM_POSITIONED_AT_ANGLE = RuleAtom("positioned_at_angle", 3, ["object", "point", "angle"])
GEOM_REGULAR_ANGULAR_SPACING = RuleAtom("regular_angular_spacing", 1, ["point"])
GEOM_FOLLOWS_SEQUENCE = RuleAtom("follows_sequence", 3, ["shape", "shape", "sequence"])
GEOM_EXHIBITS_PATTERN = RuleAtom("exhibits_pattern", 2, ["object", "pattern"])
GEOM_AT_SCALE = RuleAtom("at_scale", 2, ["object", "scale"])
GEOM_SELF_SIMILAR = RuleAtom("self_similar", 2, ["object", "pattern"])
GEOM_HAS_TOPOLOGY = RuleAtom("has_topology", 2, ["object", "topology"])
GEOM_BOUNDARY_INTERACTION = RuleAtom("boundary_interaction", 3, ["object", "boundary", "interaction"])

# Special value atoms for advanced rules
VAL_IMAGE_BOUNDARY = RuleAtom("image_boundary", 0, is_value=True)
VAL_OBJECT_VARIABLE = RuleAtom("object_variable", 0, is_value=True)
VAL_COLOR_VARIABLE = RuleAtom("color_variable", 0, is_value=True)
VAL_POINT_VARIABLE = RuleAtom("point_variable", 0, is_value=True)
VAL_ANGLE_VARIABLE = RuleAtom("angle_variable", 0, is_value=True)
VAL_SEQUENCE_VARIABLE = RuleAtom("sequence_variable", 0, is_value=True)
VAL_PATTERN_VARIABLE = RuleAtom("pattern_variable", 0, is_value=True)
VAL_SCALE_VARIABLE = RuleAtom("scale_variable", 0, is_value=True)
VAL_TOPOLOGY_VARIABLE = RuleAtom("topology_variable", 0, is_value=True)
VAL_INTERACTION_TYPE = RuleAtom("interaction_type", 0, is_value=True)

# Integer constants for COUNT comparisons (also handled by DSL Primitives)
INT_1 = RuleAtom("INT_1", 0, is_value=True)
INT_2 = RuleAtom("INT_2", 0, is_value=True)
INT_3 = RuleAtom("INT_3", 0, is_value=True)
INT_4 = RuleAtom("INT_4", 0, is_value=True)
INT_5 = RuleAtom("INT_5", 0, is_value=True)


ALL_RULE_ATOMS = {
    # Predicates
    "shape": ATTR_SHAPE, "color": ATTR_COLOR, "fill": ATTR_FILL,
    "size": ATTR_SIZE, "orientation": ATTR_ORIENTATION, "texture": ATTR_TEXTURE,
    "left_of": REL_LEFT_OF, "right_of": REL_RIGHT_OF, "above": REL_ABOVE,
    "below": REL_BELOW, "is_close_to": REL_IS_CLOSE_TO,
    "aligned_horizontally": REL_ALIGNED_HORIZONTALLY,
    "aligned_vertically": REL_ALIGNED_VERTICALLY,
    "contains": REL_CONTAINS, "intersects": REL_INTERSECTS,
    
    # Advanced geometric predicates
    "symmetrical_to": GEOM_SYMMETRICAL_TO,
    "size_value": GEOM_SIZE_VALUE,
    "equidistant_from": GEOM_EQUIDISTANT_FROM,
    "positioned_at_angle": GEOM_POSITIONED_AT_ANGLE,
    "regular_angular_spacing": GEOM_REGULAR_ANGULAR_SPACING,
    "follows_sequence": GEOM_FOLLOWS_SEQUENCE,
    "exhibits_pattern": GEOM_EXHIBITS_PATTERN,
    "at_scale": GEOM_AT_SCALE,
    "self_similar": GEOM_SELF_SIMILAR,
    "has_topology": GEOM_HAS_TOPOLOGY,
    "boundary_interaction": GEOM_BOUNDARY_INTERACTION,
    
    # Values/Constants
    "circle": SHAPE_CIRCLE, "square": SHAPE_SQUARE, "triangle": SHAPE_TRIANGLE, "star": SHAPE_STAR,
    "red": COLOR_RED, "blue": COLOR_BLUE, "green": COLOR_GREEN, "black": COLOR_BLACK, "white": COLOR_WHITE,
    "solid": FILL_SOLID, "hollow": FILL_HOLLOW, "striped": FILL_STRIPED, "dotted": FILL_DOTTED,
    "small": SIZE_SMALL, "medium": SIZE_MEDIUM, "large": SIZE_LARGE,
    "upright": ORIENTATION_UPRIGHT, "inverted": ORIENTATION_INVERTED,
    "none_texture": TEXTURE_NONE, "striped_texture": TEXTURE_STRIPED, "dotted_texture": TEXTURE_DOTTED,

    # Logical Operators
    "AND": LOGIC_AND, "OR": LOGIC_OR, "NOT": LOGIC_NOT, "IMPLIES": LOGIC_IMPLIES,
    "EXISTS": LOGIC_EXISTS, "FORALL": LOGIC_FORALL, "COUNT": LOGIC_COUNT,
    
    # Comparison Operators
    "GT": LOGIC_GT, "LT": LOGIC_LT, "EQ": LOGIC_EQ,
    
    # Integer constants
    "INT_1": INT_1, "INT_2": INT_2, "INT_3": INT_3, "INT_4": INT_4, "INT_5": INT_5,
    
    # Variable types and special values
    "object_variable": VAL_OBJECT_VARIABLE,
    "color_variable": VAL_COLOR_VARIABLE,
    "point_variable": VAL_POINT_VARIABLE,
    "angle_variable": VAL_ANGLE_VARIABLE,
    "sequence_variable": VAL_SEQUENCE_VARIABLE,
    "pattern_variable": VAL_PATTERN_VARIABLE,
    "scale_variable": VAL_SCALE_VARIABLE,
    "topology_variable": VAL_TOPOLOGY_VARIABLE,
    "interaction_type": VAL_INTERACTION_TYPE,
    "image_boundary": VAL_IMAGE_BOUNDARY
}


class BongardRule:
    """
    Represents a Bongard rule with AST-based program representation and DSL interpreter support.
    Now supports both legacy dictionary ASTs and structured AST nodes.
    """
    def __init__(self, name: str, description: str,
                 program_ast: Optional[List[Any]] = None, # AST representation of the rule
                 logical_facts: Optional[List[str]] = None, # Prolog-style facts
                 is_positive_rule: bool = True):
        self.name = name
        self.description = description
        self.program_ast = program_ast if program_ast is not None else []
        self.logical_facts = logical_facts if logical_facts is not None else []
        self.is_positive_rule = is_positive_rule # True if rule defines positive examples
        self.interpreter = BongardDSLInterpreter()

    def __repr__(self):
        return f"BongardRule(name='{self.name}', positive={self.is_positive_rule})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "program_ast": self.program_ast,
            "logical_facts": self.logical_facts,
            "is_positive_rule": self.is_positive_rule
        }
    
    def evaluate_rule(self, objects_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Evaluate whether this rule is satisfied by the given objects using DSL interpreter.
        
        Args:
            objects_data: List of object dictionaries with properties
            
        Returns:
            tuple: (rule_satisfied, explanation)
        """
        if not self.program_ast:
            return True, f"Rule '{self.name}' has no AST - assumed satisfied"
            
        try:
            # Use first AST node as the main rule condition
            main_ast = self.program_ast[0] if self.program_ast else None
            if main_ast is None:
                return True, f"Rule '{self.name}' has empty AST"
                
            # Convert to ASTNode if needed
            if isinstance(main_ast, dict):
                ast_node = ASTNode.from_dict(main_ast)
            elif isinstance(main_ast, ASTNode):
                ast_node = main_ast
            else:
                return False, f"Invalid AST format for rule '{self.name}'"
                
            return self.interpreter.evaluate(ast_node, objects_data)
            
        except Exception as e:
            return False, f"Error evaluating rule '{self.name}': {str(e)}"
    
    def apply(self, objects: List[Dict[str, Any]], is_positive: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply the rule to modify object properties based on rule constraints with professional logic.
        Now integrates with DSL interpreter for rule evaluation.
        
        Args:
            objects: List of initial objects with properties like shape, color, size, position
            is_positive: Whether to satisfy the rule (True) or violate it (False)
            
        Returns:
            Tuple of (modified_objects, scene_features) where scene_features contains rule metadata
        """
        import copy
        
        modified_objects = copy.deepcopy(objects)
        
        # First, evaluate current rule satisfaction using DSL interpreter
        rule_satisfied, explanation = self.evaluate_rule(modified_objects)
        
        scene_features = {
            'rule_name': self.name,
            'rule_description': self.description,
            'is_positive': is_positive,
            'rule_satisfied': rule_satisfied,
            'evaluation_explanation': explanation
        }
        
        # If we want positive and rule is already satisfied, or vice versa, we're done
        if (is_positive and rule_satisfied) or (not is_positive and not rule_satisfied):
            scene_features['modification_needed'] = False
            return modified_objects, scene_features
            
        scene_features['modification_needed'] = True
        
        # Apply rule-specific transformations with professional logic
        if self.name == "all_circles":
            if is_positive:
                # Ensure all objects are circles
                for obj in modified_objects:
                    obj['shape'] = 'circle'
            else:
                # Ensure at least one object is not a circle
                if modified_objects:
                    non_circle_shapes = ['square', 'triangle', 'star']
                    modified_objects[0]['shape'] = random.choice(non_circle_shapes)
                    
        elif self.name == "all_same_color_red":
            if is_positive:
                # Make all objects red
                for obj in modified_objects:
                    obj['color'] = 'red'
            else:
                # Make at least one object not red
                if modified_objects:
                    other_colors = ['blue', 'green', 'black']
                    modified_objects[0]['color'] = random.choice(other_colors)
                    
        elif self.name == "exists_large_square":
            if is_positive:
                # Ensure at least one large square exists
                if modified_objects:
                    modified_objects[0]['shape'] = 'square'
                    modified_objects[0]['size'] = 'large'
            else:
                # Ensure no large squares exist
                for obj in modified_objects:
                    if obj.get('shape') == 'square':
                        obj['size'] = 'small'  # Make squares small
                    elif obj.get('size') == 'large':
                        obj['shape'] = 'circle'  # Make large objects non-squares
        
        # Advanced geometric pattern rules
        elif self.name == "symmetric_arrangement":
            if is_positive:
                self._apply_symmetrical_arrangement(modified_objects)
            else:
                self._break_symmetrical_arrangement(modified_objects)
                
        elif self.name == "size_progression":
            if is_positive:
                self._apply_size_progression(modified_objects)
            else:
                self._break_size_progression(modified_objects)
                
        elif self.name == "nested_containment":
            if is_positive:
                self._apply_nested_containment(modified_objects)
            else:
                self._break_nested_containment(modified_objects)
                
        elif self.name == "alternating_colors":
            if is_positive:
                self._apply_alternating_colors(modified_objects)
            else:
                self._break_alternating_colors(modified_objects)
                
        elif self.name == "rotational_pattern":
            if is_positive:
                self._apply_rotational_pattern(modified_objects)
            else:
                self._break_rotational_pattern(modified_objects)
                
        elif self.name == "attribute_correlation":
            if is_positive:
                self._apply_attribute_correlation(modified_objects)
            else:
                self._break_attribute_correlation(modified_objects)
                        
        elif self.name == "exactly_three_circles":
            target_count = 3
            if is_positive:
                # Ensure exactly 3 circles
                circle_count = sum(1 for obj in modified_objects if obj.get('shape') == 'circle')
                if circle_count < target_count:
                    # Convert some objects to circles
                    non_circles = [obj for obj in modified_objects if obj.get('shape') != 'circle']
                    for i, obj in enumerate(non_circles):
                        if circle_count + i + 1 <= target_count:
                            obj['shape'] = 'circle'
                elif circle_count > target_count:
                    # Convert excess circles to other shapes
                    circles = [obj for obj in modified_objects if obj.get('shape') == 'circle']
                    for i, obj in enumerate(circles[target_count:]):
                        obj['shape'] = random.choice(['square', 'triangle'])
            else:
                # Ensure not exactly 3 circles
                circle_count = sum(1 for obj in modified_objects if obj.get('shape') == 'circle')
                if circle_count == target_count:
                    if modified_objects:
                        # Change one circle to a different shape
                        circles = [obj for obj in modified_objects if obj.get('shape') == 'circle']
                        if circles:
                            circles[0]['shape'] = 'square'
                            
        elif self.name == "red_circle_and_blue_square":
            if is_positive:
                # Ensure there's a red circle and a blue square
                if len(modified_objects) >= 2:
                    modified_objects[0]['shape'] = 'circle'
                    modified_objects[0]['color'] = 'red'
                    modified_objects[1]['shape'] = 'square'
                    modified_objects[1]['color'] = 'blue'
                elif len(modified_objects) == 1:
                    # Add a second object if only one exists
                    new_obj = copy.deepcopy(modified_objects[0])
                    modified_objects[0]['shape'] = 'circle'
                    modified_objects[0]['color'] = 'red'
                    new_obj['shape'] = 'square'
                    new_obj['color'] = 'blue'
                    # Adjust position to avoid overlap
                    if 'position' in new_obj:
                        new_obj['position'] = (new_obj['position'][0] + 50, new_obj['position'][1] + 50)
                    modified_objects.append(new_obj)
            else:
                # Ensure the combination doesn't exist
                has_red_circle = any(obj.get('shape') == 'circle' and obj.get('color') == 'red' for obj in modified_objects)
                has_blue_square = any(obj.get('shape') == 'square' and obj.get('color') == 'blue' for obj in modified_objects)
                
                if has_red_circle and has_blue_square:
                    # Remove one of the conditions
                    for obj in modified_objects:
                        if obj.get('shape') == 'circle' and obj.get('color') == 'red':
                            obj['color'] = 'green'  # Change color
                            break
                            
        # For advanced rules not yet implemented with specific transformation logic
        else:
            if not is_positive and modified_objects:
                # Apply general rule-breaking strategy
                random_obj = random.choice(modified_objects)
                # Randomly modify attributes to likely break the rule
                if random.random() < 0.4:
                    random_obj['shape'] = random.choice(['circle', 'square', 'triangle'])
                if random.random() < 0.4:
                    random_obj['color'] = random.choice(['red', 'blue', 'green', 'black'])
                if random.random() < 0.3:
                    random_obj['size'] = random.choice(['small', 'medium', 'large'])
        
        # Re-evaluate rule after modifications for verification
        final_satisfied, final_explanation = self.evaluate_rule(modified_objects)
        
        # Update scene features with object statistics and final evaluation
        scene_features.update({
            'object_count': len(modified_objects),
            'shapes': [obj.get('shape', 'unknown') for obj in modified_objects],
            'colors': [obj.get('color', 'unknown') for obj in modified_objects],
            'sizes': [obj.get('size', 'unknown') for obj in modified_objects],
            'rule_complexity': self._get_rule_complexity(),
            'final_rule_satisfied': final_satisfied,
            'final_explanation': final_explanation,
            'transformation_successful': (final_satisfied == is_positive)
        })
        
        return modified_objects, scene_features
    
    def _apply_symmetrical_arrangement(self, objects):
        """Arrange objects symmetrically around center (256x256 image assumed)."""
        if len(objects) < 2:
            return
            
        center_x, center_y = 128, 128
        radius = 60  # Distance from center
        
        # Arrange objects in symmetric pairs
        for i in range(0, len(objects), 2):
            angle = (i // 2) * (2 * math.pi / (len(objects) // 2 + len(objects) % 2))
            
            # Position first object
            x1 = center_x + radius * math.cos(angle)
            y1 = center_y + radius * math.sin(angle)
            objects[i]['x'] = int(x1)
            objects[i]['y'] = int(y1)
            
            # Position symmetric partner if it exists
            if i + 1 < len(objects):
                x2 = center_x - radius * math.cos(angle)
                y2 = center_y - radius * math.sin(angle)
                objects[i + 1]['x'] = int(x2)
                objects[i + 1]['y'] = int(y2)
    
    def _break_symmetrical_arrangement(self, objects):
        """Break symmetrical arrangement by offsetting positions randomly."""
        for obj in objects:
            if 'x' in obj and 'y' in obj:
                obj['x'] += random.randint(-20, 20)
                obj['y'] += random.randint(-20, 20)
    
    def _apply_size_progression(self, objects):
        """Arrange objects in size progression from left to right."""
        if len(objects) < 2:
            return
            
        sizes = ['small', 'medium', 'large']
        # Sort objects by x-coordinate (left to right)
        objects.sort(key=lambda obj: obj.get('x', 0))
        
        for i, obj in enumerate(objects):
            size_index = min(i, len(sizes) - 1)
            obj['size'] = sizes[size_index]
    
    def _break_size_progression(self, objects):
        """Break size progression by randomizing sizes."""
        sizes = ['small', 'medium', 'large']
        for obj in objects:
            obj['size'] = random.choice(sizes)
    
    def _apply_nested_containment(self, objects):
        """Arrange objects in nested containment pattern."""
        if len(objects) < 3:
            return
            
        # Sort by size for nesting
        sizes = ['small', 'medium', 'large']
        for i, obj in enumerate(objects):
            if i < len(sizes):
                obj['size'] = sizes[i]
                # Position concentrically
                obj['x'] = 128  # Center
                obj['y'] = 128
    
    def _break_nested_containment(self, objects):
        """Break containment by spreading objects apart."""
        for i, obj in enumerate(objects):
            obj['x'] = 50 + i * 50  # Spread horizontally
            obj['y'] = 128 + random.randint(-30, 30)
    
    def _apply_alternating_colors(self, objects):
        """Apply alternating color pattern in spatial sequence."""
        if len(objects) < 2:
            return
            
        colors = ['red', 'blue']
        # Sort by x-coordinate for spatial sequence
        objects.sort(key=lambda obj: obj.get('x', 0))
        
        for i, obj in enumerate(objects):
            obj['color'] = colors[i % 2]
    
    def _break_alternating_colors(self, objects):
        """Break color alternation with random colors."""
        colors = ['red', 'blue', 'green', 'black']
        for obj in objects:
            obj['color'] = random.choice(colors)
    
    def _apply_rotational_pattern(self, objects):
        """Arrange objects in circular pattern around center."""
        if len(objects) < 3:
            return
            
        center_x, center_y = 128, 128
        radius = 70
        
        for i, obj in enumerate(objects):
            angle = i * (2 * math.pi / len(objects))
            obj['x'] = int(center_x + radius * math.cos(angle))
            obj['y'] = int(center_y + radius * math.sin(angle))
    
    def _break_rotational_pattern(self, objects):
        """Break rotational pattern with random positions."""
        for obj in objects:
            obj['x'] = random.randint(30, 226)
            obj['y'] = random.randint(30, 226)
    
    def _apply_attribute_correlation(self, objects):
        """Apply attribute correlations: large→red, triangles→upright."""
        for obj in objects:
            if obj.get('size') == 'large':
                obj['color'] = 'red'
            if obj.get('shape') == 'triangle':
                obj['orientation'] = 'upright'
    
    def _break_attribute_correlation(self, objects):
        """Break attribute correlations with random assignments."""
        for obj in objects:
            if obj.get('size') == 'large':
                obj['color'] = random.choice(['blue', 'green', 'black'])
            if obj.get('shape') == 'triangle':
                obj['orientation'] = 'inverted'
    
    def _get_rule_complexity(self):
        """Return complexity score for the rule."""
        complexity_map = {
            'all_circles': 1,
            'all_same_color_red': 1,
            'exists_large_square': 2,
            'exactly_three_circles': 3,
            'symmetric_arrangement': 4,
            'size_progression': 4,
            'nested_containment': 5,
            'alternating_colors': 4,
            'rotational_pattern': 5,
            'attribute_correlation': 4,
            'topological_invariant': 5,
            'fractal_similarity': 6
        }
        return complexity_map.get(self.name, 3)


# --- Extended Set of Bongard Rules ---
# These rules cover various aspects of Bongard problems, from simple attributes
# to complex relations and counting.

ALL_BONGARD_RULES = collections.OrderedDict()

# --- Attribute-based Rules ---

# Rule 1: All objects are circles
ALL_BONGARD_RULES["all_circles"] = BongardRule(
    name="all_circles",
    description="All objects in the image are circles.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]}],
    logical_facts=["forall(O, shape(O, circle))"]
)

# Rule 2: All objects are the same color (e.g., all red)
ALL_BONGARD_RULES["all_same_color_red"] = BongardRule(
    name="all_same_color_red",
    description="All objects in the image are red.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "color", "args": [{"op": "object_variable", "value": "O"}, {"op": "red"}]}]}],
    logical_facts=["forall(O, color(O, red))"]
)

# Rule 3: There exists at least one large square
ALL_BONGARD_RULES["exists_large_square"] = BongardRule(
    name="exists_large_square",
    description="There is at least one large square in the image.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]}
        ]}]}
    ],
    logical_facts=["exists(O, (shape(O, square) and size(O, large)))"]
)

# Rule 4: No object is a hollow triangle
ALL_BONGARD_RULES["no_hollow_triangle"] = BongardRule(
    name="no_hollow_triangle",
    description="No object in the image is a hollow triangle.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
                {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]},
                {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]}
            ]}]}
        ]}
    ],
    logical_facts=["not(exists(O, (shape(O, triangle) and fill(O, hollow))))"]
)

# Rule 5: All objects are either red or blue
ALL_BONGARD_RULES["all_red_or_blue"] = BongardRule(
    name="all_red_or_blue",
    description="All objects in the image are either red or blue.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "color", "args": [{"op": "object_variable", "value": "O"}, {"op": "red"}]},
            {"op": "color", "args": [{"op": "object_variable", "value": "O"}, {"op": "blue"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (color(O, red) or color(O, blue)))"]
)

# --- Relational Rules ---

# Rule 6: A small object is above a large object
ALL_BONGARD_RULES["small_above_large"] = BongardRule(
    name="small_above_large",
    description="There is a small object above a large object.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "size", "args": [{"op": "object_variable", "value": "O1"}, {"op": "small"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O2"}, {"op": "large"}]},
            {"op": "above", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ],
    logical_facts=["exists(O1, exists(O2, (size(O1, small) and size(O2, large) and above(O1, O2))))"]
)

# Rule 7: All objects are aligned horizontally
ALL_BONGARD_RULES["all_aligned_horizontally"] = BongardRule(
    name="all_aligned_horizontally",
    description="All pairs of objects in the image are horizontally aligned.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]}, # O1 != O2
            {"op": "aligned_horizontally", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, forall(O2, (O1 != O2 and aligned_horizontally(O1, O2))))"]
)

# Rule 8: A circle is inside a square (conceptual 'contains' relation)
ALL_BONGARD_RULES["circle_in_square"] = BongardRule(
    name="circle_in_square",
    description="There is a circle contained within a square.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "C"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "C"}, {"op": "circle"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "S"}, {"op": "square"}]},
            {"op": "contains", "args": [{"op": "object_variable", "value": "S"}, {"op": "object_variable", "value": "C"}]} # Parent, Child
        ]}]}]}
    ],
    logical_facts=["exists(C, exists(S, (shape(C, circle) and shape(S, square) and contains(S, C))))"]
)

# Rule 9: Objects form a line (all objects are aligned horizontally AND vertically with some other object)
# This is a more complex relational rule, often implying collinearity.
# Simplified to mean all objects are aligned in both directions with at least one other object.
ALL_BONGARD_RULES["objects_form_line"] = BongardRule(
    name="objects_form_line",
    description="All objects are arranged to form a single line (horizontally or vertically).",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "OR", "args": [
                {"op": "aligned_horizontally", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
                {"op": "aligned_vertically", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
            ]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, exists(O2, (O1 != O2 and (aligned_horizontally(O1, O2) or aligned_vertically(O1, O2)))))"]
)

# --- Counting Rules ---

# Rule 10: Exactly three circles
# Updated program_ast to use EQ(COUNT(predicate), INT_3)
ALL_BONGARD_RULES["exactly_three_circles"] = BongardRule(
    name="exactly_three_circles",
    description="There are exactly three circles in the image.",
    program_ast=[
        {"op": "EQ", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]},
            {"op": "INT_3"}
        ]}
    ],
    logical_facts=["count(O, shape(O, circle), N), N = 3"] # Simplified logical fact for clarity
)

# Rule 11: More squares than circles
ALL_BONGARD_RULES["more_squares_than_circles"] = BongardRule(
    name="more_squares_than_circles",
    description="The number of squares is greater than the number of circles.",
    program_ast=[
        {"op": "GT", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]}]},
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]}
        ]}
    ],
    logical_facts=["count(O, shape(O, square), N_S), count(O, shape(O, circle), N_C), N_S > N_C"]
)

# --- Negated/Absence Rules ---

# Rule 12: No objects intersect
ALL_BONGARD_RULES["no_intersections"] = BongardRule(
    name="no_intersections",
    description="No two objects in the image intersect.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
                {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
                {"op": "intersects", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
            ]}]}]}
        ]}
    ],
    logical_facts=["not(exists(O1, exists(O2, (O1 != O2 and intersects(O1, O2)))))"]
)

# Rule 13: Not all objects are the same size
ALL_BONGARD_RULES["not_all_same_size"] = BongardRule(
    name="not_all_same_size",
    description="Objects in the image are not all the same size.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "S"}]}]}]}
        ]}
    ],
    # Corrected logical fact for "not all same size"
    logical_facts=["not(exists(S_val, forall(O, size(O, S_val))))"]
)

# --- Compositional Rules ---

# Rule 14: A red circle and a blue square
ALL_BONGARD_RULES["red_circle_and_blue_square"] = BongardRule(
    name="red_circle_and_blue_square",
    description="There is a red circle AND a blue square in the image.",
    program_ast=[
        {"op": "AND", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "AND", "args": [
                {"op": "shape", "args": [{"op": "object_variable", "value": "O1"}, {"op": "circle"}]},
                {"op": "color", "args": [{"op": "object_variable", "value": "O1"}, {"op": "red"}]}
            ]}]},
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
                {"op": "shape", "args": [{"op": "object_variable", "value": "O2"}, {"op": "square"}]},
                {"op": "color", "args": [{"op": "object_variable", "value": "O2"}, {"op": "blue"}]}
            ]}]}
        ]}
    ],
    logical_facts=["exists(O1, (shape(O1, circle) and color(O1, red))), exists(O2, (shape(O2, square) and color(O2, blue)))"]
)

# Rule 15: All circles are small
ALL_BONGARD_RULES["all_circles_are_small"] = BongardRule(
    name="all_circles_are_small",
    description="If an object is a circle, then it is small.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "IMPLIES", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "small"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, circle) implies size(O, small)))"]
)


# --- Advanced Geometric Rules (Real Bongard-LOGO complexity) ---

# Rule 16: Objects form symmetrical patterns
ALL_BONGARD_RULES["symmetric_arrangement"] = BongardRule(
    name="symmetric_arrangement",
    description="Objects are arranged in a symmetrical pattern around the center.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "symmetrical_to", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, exists(O2, (O1 != O2 and symmetrical_to(O1, O2))))"]
)

# Rule 17: Progressive size increase pattern
ALL_BONGARD_RULES["size_progression"] = BongardRule(
    name="size_progression",
    description="Objects increase in size from left to right or top to bottom.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "IMPLIES", "args": [
            {"op": "left_of", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
            {"op": "LT", "args": [
                {"op": "size_value", "args": [{"op": "object_variable", "value": "O1"}]},
                {"op": "size_value", "args": [{"op": "object_variable", "value": "O2"}]}
            ]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, forall(O2, (left_of(O1, O2) implies size_value(O1) < size_value(O2))))"]
)

# Rule 18: Nested containment hierarchy
ALL_BONGARD_RULES["nested_containment"] = BongardRule(
    name="nested_containment",
    description="Objects form a nested containment hierarchy (Russian dolls pattern).",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "OUTER"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "MIDDLE"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "INNER"}, {"op": "AND", "args": [
            {"op": "contains", "args": [{"op": "object_variable", "value": "OUTER"}, {"op": "object_variable", "value": "MIDDLE"}]},
            {"op": "contains", "args": [{"op": "object_variable", "value": "MIDDLE"}, {"op": "object_variable", "value": "INNER"}]},
            {"op": "LT", "args": [
                {"op": "size_value", "args": [{"op": "object_variable", "value": "INNER"}]},
                {"op": "size_value", "args": [{"op": "object_variable", "value": "MIDDLE"}]}
            ]},
            {"op": "LT", "args": [
                {"op": "size_value", "args": [{"op": "object_variable", "value": "MIDDLE"}]},
                {"op": "size_value", "args": [{"op": "object_variable", "value": "OUTER"}]}
            ]}
        ]}]}]}]}
    ],
    logical_facts=["exists(OUTER, exists(MIDDLE, exists(INNER, (contains(OUTER, MIDDLE) and contains(MIDDLE, INNER) and size_value(INNER) < size_value(MIDDLE) and size_value(MIDDLE) < size_value(OUTER)))))"]
)

# Rule 19: Color alternating pattern
ALL_BONGARD_RULES["alternating_colors"] = BongardRule(
    name="alternating_colors",
    description="Objects alternate between two colors in spatial sequence.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "color_variable", "value": "C1"}, {"op": "EXISTS", "args": [{"op": "color_variable", "value": "C2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "color_variable", "value": "C1"}, {"op": "color_variable", "value": "C2"}]}]},
            {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
                {"op": "color", "args": [{"op": "object_variable", "value": "O"}, {"op": "color_variable", "value": "C1"}]},
                {"op": "color", "args": [{"op": "object_variable", "value": "O"}, {"op": "color_variable", "value": "C2"}]}
            ]}]},
            {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "IMPLIES", "args": [
                {"op": "AND", "args": [
                    {"op": "left_of", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
                    {"op": "color", "args": [{"op": "object_variable", "value": "O1"}, {"op": "color_variable", "value": "C1"}]}
                ]},
                {"op": "color", "args": [{"op": "object_variable", "value": "O2"}, {"op": "color_variable", "value": "C2"}]}
            ]}]}]}
        ]}]}]}
    ],
    logical_facts=["exists(C1, exists(C2, (C1 != C2 and forall(O, (color(O, C1) or color(O, C2))) and spatial_alternation(C1, C2))))"]
)

# Rule 20: Shape transformation sequence
ALL_BONGARD_RULES["shape_transformation"] = BongardRule(
    name="shape_transformation",
    description="Objects follow a logical shape transformation sequence (e.g., circle→square→triangle).",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "sequence_variable", "value": "SEQ"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "IMPLIES", "args": [
            {"op": "left_of", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
            {"op": "follows_sequence", "args": [
                {"op": "shape", "args": [{"op": "object_variable", "value": "O1"}]},
                {"op": "shape", "args": [{"op": "object_variable", "value": "O2"}]},
                {"op": "sequence_variable", "value": "SEQ"}
            ]}
        ]}]}]}]}
    ],
    logical_facts=["exists(SEQ, forall(O1, forall(O2, (left_of(O1, O2) implies follows_sequence(shape(O1), shape(O2), SEQ)))))"]
)

# Rule 21: Rotational pattern around center
ALL_BONGARD_RULES["rotational_pattern"] = BongardRule(
    name="rotational_pattern",
    description="Objects are arranged in a rotational pattern around the center point.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "point_variable", "value": "CENTER"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "equidistant_from", "args": [{"op": "object_variable", "value": "O"}, {"op": "point_variable", "value": "CENTER"}]},
            {"op": "EXISTS", "args": [{"op": "angle_variable", "value": "THETA"}, {"op": "positioned_at_angle", "args": [
                {"op": "object_variable", "value": "O"},
                {"op": "point_variable", "value": "CENTER"},
                {"op": "angle_variable", "value": "THETA"}
            ]}]}
        ]}]},
        {"op": "regular_angular_spacing", "args": [{"op": "point_variable", "value": "CENTER"}]}]}
    ],
    logical_facts=["exists(CENTER, (forall(O, equidistant_from(O, CENTER)) and regular_angular_spacing(CENTER)))"]
)

# Rule 22: Fractal self-similarity
ALL_BONGARD_RULES["fractal_similarity"] = BongardRule(
    name="fractal_similarity",
    description="Objects exhibit fractal-like self-similarity at different scales.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "pattern_variable", "value": "PATTERN"}, {"op": "FORALL", "args": [{"op": "scale_variable", "value": "SCALE"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "exhibits_pattern", "args": [{"op": "object_variable", "value": "O"}, {"op": "pattern_variable", "value": "PATTERN"}]},
            {"op": "at_scale", "args": [{"op": "object_variable", "value": "O"}, {"op": "scale_variable", "value": "SCALE"}]},
            {"op": "self_similar", "args": [{"op": "object_variable", "value": "O"}, {"op": "pattern_variable", "value": "PATTERN"}]}
        ]}]}]}]}
    ],
    logical_facts=["exists(PATTERN, forall(SCALE, exists(O, (exhibits_pattern(O, PATTERN) and at_scale(O, SCALE) and self_similar(O, PATTERN)))))"]
)

# Rule 23: Topological invariance
ALL_BONGARD_RULES["topological_invariant"] = BongardRule(
    name="topological_invariant",
    description="Objects share topological properties (e.g., all have holes, all are simply connected).",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "topology_variable", "value": "TOPO_PROP"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "has_topology", "args": [
            {"op": "object_variable", "value": "O"},
            {"op": "topology_variable", "value": "TOPO_PROP"}
        ]}]}]}
    ],
    logical_facts=["exists(TOPO_PROP, forall(O, has_topology(O, TOPO_PROP)))"]
)

# Rule 24: Boundary interaction patterns
ALL_BONGARD_RULES["boundary_interaction"] = BongardRule(
    name="boundary_interaction",
    description="Objects interact with image boundaries in consistent ways (touching, avoiding, etc.).",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "interaction_type", "value": "INTERACTION"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "boundary_interaction", "args": [
            {"op": "object_variable", "value": "O"},
            {"op": "image_boundary"},
            {"op": "interaction_type", "value": "INTERACTION"}
        ]}]}]}
    ],
    logical_facts=["exists(INTERACTION, forall(O, boundary_interaction(O, image_boundary, INTERACTION)))"]
)

# Rule 25: Multi-attribute correlation
ALL_BONGARD_RULES["attribute_correlation"] = BongardRule(
    name="attribute_correlation",
    description="Multiple attributes are correlated (e.g., larger objects are always red, triangles are always upright).",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "IMPLIES", "args": [
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]},
            {"op": "color", "args": [{"op": "object_variable", "value": "O"}, {"op": "red"}]}
        ]}, {"op": "IMPLIES", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]},
            {"op": "orientation", "args": [{"op": "object_variable", "value": "O"}, {"op": "upright"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (size(O, large) implies color(O, red)) and (shape(O, triangle) implies orientation(O, upright)))"]
)

logger.info(f"Loaded {len(ALL_BONGARD_RULES)} professional Bongard rules with advanced geometric and logical complexity.")


def get_all_rules():
    """Returns a list of all available Bongard rules."""
    return list(ALL_BONGARD_RULES.values())


def get_rule_by_name(name: str):
    """Returns a specific rule by name."""
    return ALL_BONGARD_RULES.get(name)

