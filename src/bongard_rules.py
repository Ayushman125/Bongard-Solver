import collections
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# --- Rule Atom Definitions ---
# These represent the basic predicates or functions in a Bongard rule's DSL.
class RuleAtom:
    """Represents a basic predicate or function in a Bongard rule."""
    def __init__(self, name: str, arity: int, arg_types: Optional[List[str]] = None, is_value: bool = False):
        self.name = name
        self.arity = arity  # Number of arguments the atom takes. -1 for variable arity (e.g., AND, OR)
        self.arg_types = arg_types if arg_types is not None else []
        self.is_value = is_value  # True if this atom represents a constant value (e.g., 'circle', 'red')

    def __repr__(self):
        return f"Atom({self.name}/{self.arity}, is_value={self.is_value})"

    def __eq__(self, other):
        return isinstance(other, RuleAtom) and self.name == other.name and self.arity == other.arity and self.is_value == other.is_value

    def __hash__(self):
        return hash((self.name, self.arity, self.is_value))

# Define common attribute atoms (predicates)
ATTR_SHAPE = RuleAtom("shape", 2, ["object", "shape_type"])  # e.g., shape(obj1, circle)
ATTR_FILL = RuleAtom("fill", 2, ["object", "fill_type"])
ATTR_SIZE = RuleAtom("size", 2, ["object", "size_type"])
ATTR_ORIENTATION = RuleAtom("orientation", 2, ["object", "orientation_type"])
ATTR_TEXTURE = RuleAtom("texture", 2, ["object", "texture_type"])

# Define common relation atoms (predicates)
REL_LEFT_OF = RuleAtom("left_of", 2, ["object", "object"])  # e.g., left_of(obj1, obj2)
REL_RIGHT_OF = RuleAtom("right_of", 2, ["object", "object"])
REL_ABOVE = RuleAtom("above", 2, ["object", "object"])
REL_BELOW = RuleAtom("below", 2, ["object", "object"])
REL_IS_CLOSE_TO = RuleAtom("is_close_to", 2, ["object", "object"])  # Proximity
REL_ALIGNED_HORIZONTALLY = RuleAtom("aligned_horizontally", 2, ["object", "object"])
REL_ALIGNED_VERTICALLY = RuleAtom("aligned_vertically", 2, ["object", "object"])
REL_CONTAINS = RuleAtom("contains", 2, ["object", "object"])  # For nested objects
REL_INTERSECTS = RuleAtom("intersects", 2, ["object", "object"])  # For overlapping objects

# Coordinate Predicates (retained for potential future use or more complex rules)
REL_SAME_X = RuleAtom("same_x", 2, ["object","object"])
REL_SAME_Y = RuleAtom("same_y", 2, ["object","object"])


# Define specific attribute values (terminal atoms/constants)
SHAPE_CIRCLE = RuleAtom("circle", 0, is_value=True)
SHAPE_SQUARE = RuleAtom("square", 0, is_value=True)
SHAPE_TRIANGLE = RuleAtom("triangle", 0, is_value=True)
SHAPE_STAR = RuleAtom("star", 0, is_value=True)
SHAPE_PENTAGON = RuleAtom("pentagon", 0, is_value=True)
SHAPE_HEXAGON = RuleAtom("hexagon", 0, is_value=True)

FILL_SOLID = RuleAtom("solid", 0, is_value=True)
FILL_HOLLOW = RuleAtom("hollow", 0, is_value=True)
FILL_STRIPED = RuleAtom("striped", 0, is_value=True)
FILL_DOTTED = RuleAtom("dotted", 0, is_value=True)
FILL_GRADIENT = RuleAtom("gradient", 0, is_value=True)
FILL_CHECKER = RuleAtom("checker", 0, is_value=True)

SIZE_SMALL = RuleAtom("small", 0, is_value=True)
SIZE_MEDIUM = RuleAtom("medium", 0, is_value=True)
SIZE_LARGE = RuleAtom("large", 0, is_value=True)

ORIENTATION_UPRIGHT = RuleAtom("upright", 0, is_value=True)
ORIENTATION_INVERTED = RuleAtom("inverted", 0, is_value=True)
ORIENTATION_ROTATED_45 = RuleAtom("rotated_45", 0, is_value=True)

TEXTURE_NONE = RuleAtom("none", 0, is_value=True)
TEXTURE_STRIPED = RuleAtom("striped", 0, is_value=True)
TEXTURE_DOTTED = RuleAtom("dotted", 0, is_value=True)
TEXTURE_CROSSHATCH = RuleAtom("crosshatch", 0, is_value=True)
TEXTURE_CHECKER = RuleAtom("checker", 0, is_value=True)

# Abstract/Freeform Predicates
ATTR_CONVEX = RuleAtom("convex", 2, ["object", "bool_val"])
ATTR_CONCAVE = RuleAtom("concave", 2, ["object", "bool_val"])
ATTR_HAS_RIGHT_ANGLE = RuleAtom("has_right_angle", 2, ["object", "bool_val"])
ATTR_SYMMETRICAL_VERTICALLY = RuleAtom("symmetrical_vertically", 2, ["object", "bool_val"])
ATTR_NUM_STROKES = RuleAtom("num_strokes", 2, ["object", "int_val"])
ATTR_FAN_PATTERN = RuleAtom("fan_pattern", 2, ["scene", "bool_val"]) # Scene-level attribute
ATTR_IS_CLOSED = RuleAtom("is_closed", 2, ["object", "bool_val"])
ATTR_IS_COMPOSITE = RuleAtom("is_composite", 2, ["object", "bool_val"])
ATTR_IS_SPIRAL = RuleAtom("is_spiral", 2, ["object", "bool_val"])


# Logical Operators (for composing complex rules)
LOGIC_AND = RuleAtom("AND", -1)  # Variable arity
LOGIC_OR = RuleAtom("OR", -1)
LOGIC_NOT = RuleAtom("NOT", 1)
LOGIC_IMPLIES = RuleAtom("IMPLIES", 2)

# Quantifiers
LOGIC_EXISTS = RuleAtom("EXISTS", 2)  # e.g., EXISTS(obj, shape(obj, circle))
LOGIC_FORALL = RuleAtom("FORALL", 2)  # e.g., FORALL(obj, shape(obj, circle))

# Counting
LOGIC_COUNT = RuleAtom("COUNT", 2)  # e.g., COUNT(obj, shape(obj, circle))

# Comparison operators for counts
LOGIC_GT = RuleAtom("GT", 2)
LOGIC_LT = RuleAtom("LT", 2)
LOGIC_EQ = RuleAtom("EQ", 2)

# Integer constants for COUNT comparisons
INT_0 = RuleAtom("INT_0", 0, is_value=True)
INT_1 = RuleAtom("INT_1", 0, is_value=True)
INT_2 = RuleAtom("INT_2", 0, is_value=True)
INT_3 = RuleAtom("INT_3", 0, is_value=True)
INT_4 = RuleAtom("INT_4", 0, is_value=True)
INT_5 = RuleAtom("INT_5", 0, is_value=True)
INT_6 = RuleAtom("INT_6", 0, is_value=True)

# Boolean values for abstract predicates
BOOL_TRUE = RuleAtom("true", 0, is_value=True)
BOOL_FALSE = RuleAtom("false", 0, is_value=True)


ALL_RULE_ATOMS = {
    # Predicates
    "shape": ATTR_SHAPE, "fill": ATTR_FILL,
    "size": ATTR_SIZE, "orientation": ATTR_ORIENTATION, "texture": ATTR_TEXTURE,
    "left_of": REL_LEFT_OF, "right_of": REL_RIGHT_OF, "above": REL_ABOVE,
    "below": REL_BELOW, "is_close_to": REL_IS_CLOSE_TO,
    "aligned_horizontally": REL_ALIGNED_HORIZONTALLY,
    "aligned_vertically": REL_ALIGNED_VERTICALLY,
    "contains": REL_CONTAINS, "intersects": REL_INTERSECTS,
    "same_x": REL_SAME_X, "same_y": REL_SAME_Y,
    "convex": ATTR_CONVEX, "concave": ATTR_CONCAVE, "has_right_angle": ATTR_HAS_RIGHT_ANGLE,
    "symmetrical_vertically": ATTR_SYMMETRICAL_VERTICALLY,
    "num_strokes": ATTR_NUM_STROKES, "fan_pattern": ATTR_FAN_PATTERN,
    "is_closed": ATTR_IS_CLOSED, "is_composite": ATTR_IS_COMPOSITE, "is_spiral": ATTR_IS_SPIRAL,
    
    # Values/Constants
    "circle": SHAPE_CIRCLE, "square": SHAPE_SQUARE, "triangle": SHAPE_TRIANGLE, "star": SHAPE_STAR,
    "pentagon": SHAPE_PENTAGON, "hexagon": SHAPE_HEXAGON,
    "solid": FILL_SOLID, "hollow": FILL_HOLLOW, "striped": FILL_STRIPED, "dotted": FILL_DOTTED, "gradient": FILL_GRADIENT,
    "checker": FILL_CHECKER,
    "small": SIZE_SMALL, "medium": SIZE_MEDIUM, "large": SIZE_LARGE,
    "upright": ORIENTATION_UPRIGHT, "inverted": ORIENTATION_INVERTED, "rotated_45": ORIENTATION_ROTATED_45,
    "none_texture": TEXTURE_NONE, "striped_texture": TEXTURE_STRIPED, "dotted_texture": TEXTURE_DOTTED, "crosshatch_texture": TEXTURE_CROSSHATCH,
    "checker_texture": TEXTURE_CHECKER,
    "true": BOOL_TRUE, "false": BOOL_FALSE,
    
    # Logical Operators
    "AND": LOGIC_AND, "OR": LOGIC_OR, "NOT": LOGIC_NOT, "IMPLIES": LOGIC_IMPLIES,
    "EXISTS": LOGIC_EXISTS, "FORALL": LOGIC_FORALL, "COUNT": LOGIC_COUNT,
    
    # Comparison Operators
    "GT": LOGIC_GT, "LT": LOGIC_LT, "EQ": LOGIC_EQ,
    
    # Integer constants
    "INT_0": INT_0, "INT_1": INT_1, "INT_2": INT_2, "INT_3": INT_3, "INT_4": INT_4, "INT_5": INT_5, "INT_6": INT_6,
    "object_variable": RuleAtom("object_variable", 0, is_value=True),
    "scene": RuleAtom("scene", 0, is_value=True)
}

# --- AST Literal Extraction Helper ---
def extract_literals_from_ast(ast_node: Any, pos_feats: List[Dict[str, Any]], neg_feats: List[Dict[str, Any]], is_negated: bool = False):
    """
    Recursively walks the AST tree to extract positive and negative literals.
    A literal is a concrete feature-value pair (e.g., {"feature": "shape", "value": "circle"}).
    This function is primarily for rules defined with a program_ast.
    For canonical rules, pos_literals and neg_literals are set directly.
    """
    if not isinstance(ast_node, dict):
        return

    op = ast_node.get("op")
    args = ast_node.get("args", [])

    # Handle NOT operator: flips the negation state for its arguments
    if op == "NOT":
        for arg in args:
            extract_literals_from_ast(arg, pos_feats, neg_feats, not is_negated)
        return

    # Handle comparison operators (EQ, GT, LT) for COUNT
    if op in {"EQ", "GT", "LT"} and len(args) == 2:
        count_node = args[0]
        value_node = args[1]
        
        if count_node.get("op") == "COUNT" and len(count_node.get("args", [])) == 2:
            # The predicate for COUNT is the second arg of COUNT
            count_predicate_node = count_node["args"][1]
            # The count value is the value of the second arg of EQ/GT/LT
            count_value = value_node.get("op", value_node) # Get actual value from RuleAtom or direct int
            
            # Recursively extract features from the COUNT's predicate
            temp_pos_feats = []
            temp_neg_feats = []
            extract_literals_from_ast(count_predicate_node, temp_pos_feats, temp_neg_feats, is_negated)
            
            # Apply the count constraint to the extracted features
            for feat_dict in temp_pos_feats:
                feat_dict["count_op"] = op
                # Convert 'INT_3' to 3, handle various formats
                if isinstance(count_value, str) and count_value.startswith('INT_'):
                    feat_dict["count_value"] = int(count_value.replace('INT_', ''))
                elif isinstance(count_value, int):
                    feat_dict["count_value"] = count_value
                else:
                    feat_dict["count_value"] = 1  # Default fallback
                if not is_negated:
                    pos_feats.append(feat_dict)
                else:
                    neg_feats.append(feat_dict)
            for feat_dict in temp_neg_feats:
                feat_dict["count_op"] = op # Should be inverted op for negation
                # Convert 'INT_3' to 3, handle various formats
                if isinstance(count_value, str) and count_value.startswith('INT_'):
                    feat_dict["count_value"] = int(count_value.replace('INT_', ''))
                elif isinstance(count_value, int):
                    feat_dict["count_value"] = count_value
                else:
                    feat_dict["count_value"] = 1  # Default fallback
                if not is_negated: # If the original count was negated
                    neg_feats.append(feat_dict)
                else: # If the original count was positive, but we're in a NOT context
                    pos_feats.append(feat_dict)
            return

    # Extract concrete feature-value pairs
    if op in ["shape", "fill", "size", "orientation", "texture",
              "left_of", "right_of", "above", "below", "is_close_to",
              "aligned_horizontally", "aligned_vertically", "contains", "intersects",
              "same_x", "same_y",
              "convex", "concave", "has_right_angle", "symmetrical_vertically", "num_strokes",
              "fan_pattern", "is_closed", "is_composite", "is_spiral"
              ]:
        
        feature_type = op
        value = None
        
        # For attribute predicates (shape, fill, etc.), the value is the last argument
        if feature_type in ["shape", "fill", "size", "orientation", "texture",
                            "convex", "concave", "has_right_angle", "symmetrical_vertically", "num_strokes",
                            "is_closed", "is_composite", "is_spiral"]:
            if len(args) >= 2:
                val_node = args[-1]
                # Refined value extraction for INT_X, true/false, and other strings
                if isinstance(val_node, dict) and "op" in val_node:
                    op_val = val_node["op"]
                    if isinstance(op_val, str):
                        if op_val.startswith("INT_"):
                            value = int(op_val.replace("INT_", ""))
                        elif op_val.lower() == "true":
                            value = True
                        elif op_val.lower() == "false":
                            value = False
                        else:
                            value = str(op_val).lower() # Default to lowercase string
                    else:
                        value = op_val # If op is not a string (e.g., direct int/bool)
                elif isinstance(val_node, str):
                    if val_node.lower() == "true":
                        value = True
                    elif val_node.lower() == "false":
                        value = False
                    else:
                        value = val_node.lower() # Default to lowercase string
                else:
                    value = val_node # If value is already an int/bool, keep it as is.
        # For relational predicates, the value is the relation type itself
        elif feature_type in ["left_of", "right_of", "above", "below", "is_close_to",
                              "aligned_horizontally", "aligned_vertically", "contains", "intersects",
                              "same_x", "same_y"]:
            value = op # The operation name is the relation value
            feature_type = "relation" # Standardize feature type for relations
        # For scene-level predicates like fan_pattern, the value is the last argument
        elif feature_type == "fan_pattern":
            if len(args) >= 2:
                val_node = args[-1]
                if isinstance(val_node, dict) and "op" in val_node:
                    op_val = val_node["op"]
                    if isinstance(op_val, str):
                        if op_val.lower() == "true":
                            value = True
                        elif op_val.lower() == "false":
                            value = False
                        else:
                            value = str(op_val).lower()
                elif isinstance(val_node, str):
                    if val_node.lower() == "true":
                        value = True
                    elif val_node.lower() == "false":
                        value = False
                    else:
                        value = val_node.lower()
                else:
                    value = val_node
            feature_type = "scene_pattern" # Standardize feature type for scene-level patterns


        if value is not None:
            literal = {"feature": feature_type, "value": value}
            if not is_negated:
                pos_feats.append(literal)
            else:
                neg_feats.append(literal)

    # Recursively process arguments for logical operators and quantifiers
    if op in {"AND", "OR", "EXISTS", "FORALL", "IMPLIES"}:
        for arg in args:
            extract_literals_from_ast(arg, pos_feats, neg_feats, is_negated)
    
    # Special handling for COUNT's predicate argument if not already processed by comparison
    if op == "COUNT" and len(args) == 2:
        # The actual predicate for counting is the second argument
        extract_literals_from_ast(args[1], pos_feats, neg_feats, is_negated)


class BongardRule:
    """
    Represents a Bongard rule, now with support for a DSL program AST,
    logical facts, and explicit positive/negative literals for scene generation.
    """
    def __init__(self, name: str, description: str,
                 program_ast: Optional[List[Any]] = None,  # AST representation of the rule
                 logical_facts: Optional[List[str]] = None,  # Prolog-style facts
                 is_positive_rule: bool = True,
                 pos_literals: Optional[List[Dict[str, Any]]] = None, # Explicit positive literals
                 neg_literals: Optional[List[Dict[str, Any]]] = None): # Explicit negative literals
        self.name = name
        self.description = description
        self.program_ast = program_ast if program_ast is not None else []
        self.logical_facts = logical_facts if logical_facts is not None else []
        self.is_positive_rule = is_positive_rule

        # If explicit literals are provided, use them. Otherwise, generate from AST.
        if pos_literals is not None and neg_literals is not None:
            self._pos_literals = pos_literals
            self._neg_literals = neg_literals
        else:
            self._pos_literals = []
            self._neg_literals = []
            self._generate_literals_from_ast()

    def _generate_literals_from_ast(self):
        """Generates positive and negative literals by walking the program_ast."""
        for node in self.program_ast:
            extract_literals_from_ast(node, self._pos_literals, self._neg_literals)
        
    def __repr__(self):
        return f"BongardRule(name='{self.name}', positive={self.is_positive_rule})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "program_ast": self.program_ast,
            "logical_facts": self.logical_facts,
            "is_positive_rule": self.is_positive_rule,
            "pos_literals": self.pos_literals,
            "neg_literals": self.neg_literals
        }

    @property
    def pos_literals(self) -> List[Dict[str, Any]]:
        """Returns the extracted positive literals."""
        return self._pos_literals

    @property
    def neg_literals(self) -> List[Dict[str, Any]]:
        """Returns the extracted negative literals."""
        return self._neg_literals

# --- Canonical Set of Bongard Rules ---
# These rules are defined directly with their positive/negative literals.
# Their program_ast is intentionally left empty as they are "atomic" canonical rules.
CANONICAL_RULES_STRINGS = [
  # Shapes
  "SHAPE(CIRCLE)", "SHAPE(SQUARE)", "SHAPE(TRIANGLE)", "SHAPE(STAR)",
  # Fills
  "FILL(SOLID)", "FILL(HOLLOW)", "FILL(STRIPED)", "FILL(DOTTED)",
  # Counts
  "COUNT(EQ,1)", "COUNT(EQ,2)", "COUNT(EQ,3)", "COUNT(EQ,4)",
  # Relations
  "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW",
  # Shape properties
  "CONVEX", "CONCAVE",
  # Negation
  "NO_INTERSECTIONS",
]

def load_canonical_rules() -> List[BongardRule]:
    """
    Loads a predefined set of canonical Bongard rules, converting them into
    BongardRule objects with explicit positive and negative literals.
    """
    """
    Loads the canonical Bongard rules as BongardRule objects.
    Handles all edge cases, including negation and correct literal assignment.
    """
    rules = []
    for rule_str in CANONICAL_RULES_STRINGS:
        pos_literals, neg_literals = [], []
        rule_name = rule_str.lower()
        rule_description = f"Canonical rule: {rule_str}"

        if rule_str.startswith("SHAPE("):
            shape_type = rule_str.split("(")[1][:-1].lower()
            pos_literals.append({"feature": "shape", "value": shape_type})
            rule_description = f"All objects are {shape_type}s."
        elif rule_str.startswith("FILL("):
            fill_type = rule_str.split("(")[1][:-1].lower()
            pos_literals.append({"feature": "fill", "value": fill_type})
            rule_description = f"All objects have {fill_type} fill."
        elif rule_str.startswith("COUNT(EQ,"):
            try:
                count_value = int(rule_str.split(',')[1][:-1])
                pos_literals.append({"feature": "count", "value": count_value, "count_op": "EQ"})
                rule_description = f"There are exactly {count_value} objects in the image."
            except ValueError:
                logger.error(f"Invalid count value in rule string: {rule_str}")
                continue
        elif rule_str in {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW"}:
            pos_literals.append({"feature": "relation", "value": rule_str.lower()})
            rule_description = f"There is at least one object {rule_str.lower().replace('_', ' ')} another."
        elif rule_str == "CONVEX":
            pos_literals.append({"feature": "property", "value": "convex"})
            rule_description = "All shapes are convex."
        elif rule_str == "CONCAVE":
            pos_literals.append({"feature": "property", "value": "concave"})
            rule_description = "All shapes are concave."
        elif rule_str == "NO_INTERSECTIONS":
            neg_literals.append({"feature": "relation", "value": "intersects"})
            rule_description = "No two objects intersect."
        else:
            logger.warning(f"Unknown canonical rule string format: {rule_str}")
            continue

        rules.append(
            BongardRule(
                name=rule_name,
                description=rule_description,
                pos_literals=pos_literals,
                neg_literals=neg_literals,
                program_ast=[],
                logical_facts=[]
            )
        )
    return rules

# Override the old list with the canonical set
ALL_BONGARD_RULES = load_canonical_rules()

# Quick test to ensure the correct number of rules are loaded
if len(ALL_BONGARD_RULES) != 19: # 4 shapes + 4 fills + 4 counts + 4 relations + 2 properties + 1 negation = 19
    logger.error(f"Expected 19 canonical rules, but loaded {len(ALL_BONGARD_RULES)}.")
else:
    logger.info(f"Loaded {len(ALL_BONGARD_RULES)} canonical Bongard rules successfully.")

