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

# Coordinate Predicates
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
    "convex": ATTR_CONVEX, "has_right_angle": ATTR_HAS_RIGHT_ANGLE,
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
              "convex", "has_right_angle", "symmetrical_vertically", "num_strokes",
              "fan_pattern", "is_closed", "is_composite", "is_spiral"
              ]:
        
        feature_type = op
        value = None
        
        # For attribute predicates (shape, fill, etc.), the value is the last argument
        if feature_type in ["shape", "fill", "size", "orientation", "texture",
                            "convex", "has_right_angle", "symmetrical_vertically", "num_strokes",
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

# --- Extended Set of Bongard Rules ---
from bongard_generator.rule_loader import BongardRule

CANONICAL_RULES = [
  "SHAPE(CIRCLE)", "SHAPE(SQUARE)", "SHAPE(TRIANGLE)", "SHAPE(STAR)",
  "COLOR(RED)", "COLOR(GREEN)", "COLOR(BLUE)", "COLOR(BLACK)",
  "FILL(SOLID)", "FILL(HOLLOW)", "FILL(STRIPED)", "FILL(DOTTED)",
  "COUNT(EQ,1)", "COUNT(EQ,2)", "COUNT(EQ,3)", "COUNT(EQ,4)",
  "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW",
  "CONVEX", "CONCAVE",
  "NO_INTERSECTIONS",
]

def load_canonical_rules():
    rules = []
    for rule in CANONICAL_RULES:
        pos, neg = [], []
        if rule.startswith("SHAPE"):
            v = rule[6:-1].lower()
            pos = [{"feature":"shape","value":v}]
        elif rule.startswith("COLOR"):
            v = rule[6:-1].lower()
            pos = [{"feature":"color","value":v}]
        elif rule.startswith("FILL"):
            v = rule[5:-1].lower()
            pos = [{"feature":"fill","value":v}]
        elif rule.startswith("COUNT"):
            _, args = rule.split("(")
            _, n = args[:-1].split(",")
            pos = [{"feature":"count","value":int(n)}]
        elif rule in {"LEFT_OF","RIGHT_OF","ABOVE","BELOW"}:
            pos = [{"feature":"relation","value":rule.lower()}]
        elif rule == "CONVEX":
            pos = [{"feature":"property","value":"convex"}]
        elif rule == "CONCAVE":
            pos = [{"feature":"property","value":"concave"}]
        elif rule == "NO_INTERSECTIONS":
            neg = [{"feature":"relation","value":"intersects"}]
        rules.append(
            BongardRule(
                name=rule.lower(),
                description=rule,
                pos_literals=pos,
                neg_literals=neg,
                program_ast=[],
                logical_facts=[]
            )
        )
    return rules

ALL_BONGARD_RULES = load_canonical_rules()
    BongardRule(
        name="shape_triangle_fallback",
        description="SHAPE(TRIANGLE)",
        program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]}]}],
        logical_facts=["forall(O, shape(O, triangle))"]
    )
)

# These rules cover various aspects of Bongard problems, from simple attributes
# to complex relations and counting.
# --- Attribute-based Rules ---

# Rule 1: All objects are circles


# NEW Rule: exists_large_solid (from document)
add_rule_if_new(BongardRule(
    name="exists_large_solid",
    description="There is at least one object that is large and solid.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]},
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]}
        ]}]}
    ],
    logical_facts=["exists(O, (size(O, large) and fill(O, solid)))"]
))

# NEW Rule: all_circle_or_square (from document)
add_rule_if_new(BongardRule(
    name="all_circle_or_square",
    description="All shapes are either circles or squares.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, circle) or shape(O, square)))"]
))

# NEW Rule: no_crosshatch_texture (from document)
add_rule_if_new(BongardRule(
    name="no_crosshatch_texture",
    description="No object has a crosshatch texture.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "texture", "args": [{"op": "object_variable", "value": "O"}, {"op": "crosshatch_texture"}]}]}
        ]}
    ],
    logical_facts=["not(exists(O, texture(O, crosshatch_texture)))"]
))

# NEW Rule: all_medium_rotated_45 (from document)
add_rule_if_new(BongardRule(
    name="all_medium_rotated_45",
    description="All objects are medium-sized and rotated 45 degrees.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "medium"}]},
            {"op": "orientation", "args": [{"op": "object_variable", "value": "O"}, {"op": "rotated_45"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (size(O, medium) and orientation(O, rotated_45)))"]
))

# NEW Rule: every_hollow_or_dotted (from document)
add_rule_if_new(BongardRule(
    name="every_hollow_or_dotted",
    description="Every object is either hollow or dotted.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]},
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "dotted"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (fill(O, hollow) or fill(O, dotted)))"]
))

# NEW Rule: exists_small_striped_triangle (from document)
add_rule_if_new(BongardRule(
    name="exists_small_striped_triangle",
    description="There exists at least one small, striped triangle.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]},
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "striped"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "small"}]}
        ]}]}
    ],
    logical_facts=["exists(O, (shape(O, triangle) and fill(O, striped) and size(O, small)))"]
))

# NEW Rule: all_same_fill_type (from document)
add_rule_if_new(BongardRule(
    name="all_same_fill_type",
    description="All objects have the same fill type.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "F_val"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "F_val"}]}]}]}
    ],
    logical_facts=["exists(F_val, forall(O, fill(O, F_val)))"]
))

# NEW Rule: no_large_hollow_objects (from document)
add_rule_if_new(BongardRule(
    name="no_large_hollow_objects",
    description="No object is both large and hollow.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
                {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]},
                {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]}
            ]}]}
        ]}
    ],
    logical_facts=["not(exists(O, (size(O, large) and fill(O, hollow))))"]
))

# NEW Rule: circle_or_dotted_texture (from document)
add_rule_if_new(BongardRule(
    name="circle_or_dotted_texture",
    description="Every object is either a circle or has a dotted texture.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]},
            {"op": "texture", "args": [{"op": "object_variable", "value": "O"}, {"op": "dotted_texture"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, circle) or texture(O, dotted_texture)))"]
))


# --- Relational Rules ---

# Rule 6: A small object is above a large object
add_rule_if_new(BongardRule(
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
))

# Rule 7: All objects are aligned horizontally
add_rule_if_new(BongardRule(
    name="all_aligned_horizontally",
    description="All pairs of objects in the image are horizontally aligned.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},  # O1 != O2
            {"op": "aligned_horizontally", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, forall(O2, (O1 != O2 and aligned_horizontally(O1, O2))))"]
))

# Rule 8: A circle is inside a square (conceptual 'contains' relation)
add_rule_if_new(BongardRule(
    name="circle_in_square",
    description="There is a circle contained within a square.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "C"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "C"}, {"op": "circle"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "S"}, {"op": "square"}]},
            {"op": "contains", "args": [{"op": "object_variable", "value": "S"}, {"op": "object_variable", "value": "C"}]}  # Parent, Child
        ]}]}]}
    ],
    logical_facts=["exists(C, exists(S, (shape(C, circle) and shape(S, square) and contains(S, C))))"]
))

# Rule 9: Objects form a line (all objects are aligned horizontally AND vertically with some other object)
add_rule_if_new(BongardRule(
    name="objects_form_line",
    description="All objects are arranged to form a single line (horizontally or vertically).",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "IMPLIES", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "OR", "args": [
                {"op": "aligned_horizontally", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
                {"op": "aligned_vertically", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
            ]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, forall(O2, (O1!= O2 implies (aligned_horizontally(O1, O2) or aligned_vertically(O1, O2)))))"]
))

# NEW Rule: all_aligned_vertically (from document)
add_rule_if_new(BongardRule(
    name="all_aligned_vertically",
    description="All objects are aligned vertically.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
        {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
        {"op": "aligned_vertically", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
    ]}]}]}
    ],
    logical_facts=["forall(O1, forall(O2, (O1!= O2 and aligned_vertically(O1, O2))))"]
))

# NEW Rule: square_contains_small_circle (from document)
add_rule_if_new(BongardRule(
    name="square_contains_small_circle",
    description="A square contains a circle, and that circle is small.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "C"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "S"}, {"op": "square"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "C"}, {"op": "circle"}]},
            {"op": "contains", "args": [{"op": "object_variable", "value": "S"}, {"op": "object_variable", "value": "C"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "C"}, {"op": "small"}]}
        ]}]}]}
    ],
    logical_facts=["exists(S, exists(C, (shape(S, square) and shape(C, circle) and contains(S, C) and size(C, small))))"]
))

# NEW Rule: every_circle_left_of_square (from document)
add_rule_if_new(BongardRule(
    name="every_circle_left_of_square",
    description="Every circle is to the left of a square.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "C"}, {"op": "IMPLIES", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "C"}, {"op": "circle"}]},
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S"}, {"op": "AND", "args": [
                {"op": "shape", "args": [{"op": "object_variable", "value": "S"}, {"op": "square"}]},
                {"op": "left_of", "args": [{"op": "object_variable", "value": "C"}, {"op": "object_variable", "value": "S"}]}
            ]}]}
        ]}]}
    ],
    logical_facts=["forall(C, (shape(C, circle) implies exists(S, (shape(S, square) and left_of(C, S)))))"]
))

# NEW Rule: no_same_shape_below (from document)
add_rule_if_new(BongardRule(
    name="no_same_shape_below",
    description="No object is below another object of the same shape.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
                {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
                {"op": "shape", "args": [{"op": "object_variable", "value": "O1"}, {"op": "S_val"}]}, # S_val is a placeholder for a variable that will be bound
                {"op": "shape", "args": [{"op": "object_variable", "value": "O2"}, {"op": "S_val"}]},
                {"op": "below", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
            ]}]}]}
        ]}
    ],
    logical_facts=["not(exists(O1, exists(O2, (O1!= O2 and shape(O1, S_val) and shape(O2, S_val) and below(O1, O2)))))"]
))

# NEW Rule: small_contained_by_large (from document)
add_rule_if_new(BongardRule(
    name="small_contained_by_large",
    description="Every object that is small is contained by a large object.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "IMPLIES", "args": [
            {"op": "size", "args": [{"op": "object_variable", "value": "O1"}, {"op": "small"}]},
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
                {"op": "size", "args": [{"op": "object_variable", "value": "O2"}, {"op": "large"}]},
                {"op": "contains", "args": [{"op": "object_variable", "value": "O2"}, {"op": "object_variable", "value": "O1"}]}
            ]}]}
        ]}]}
    ],
    logical_facts=["forall(O1, (size(O1, small) implies exists(O2, (size(O2, large) and contains(O2, O1)))))"]
))

# NEW Rule: at_least_two_intersect (from document)
add_rule_if_new(BongardRule(
    name="at_least_two_intersect",
    description="There are at least two objects that intersect.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "intersects", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ],
    logical_facts=["exists(O1, exists(O2, (O1!= O2 and intersects(O1, O2))))"]
))

# NEW Rule: no_above_and_left (from document)
add_rule_if_new(BongardRule(
    name="no_above_and_left",
    description="No object is both above and to the left of another object.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
                {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
                {"op": "above", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
                {"op": "left_of", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
            ]}]}]}
        ]}
    ],
    logical_facts=["not(exists(O1, exists(O2, (O1!= O2 and above(O1, O2) and left_of(O1, O2)))))"]
))

# NEW Rule: exactly_one_right_neighbor (from document)
add_rule_if_new(BongardRule(
    name="exactly_one_right_neighbor",
    description="All objects are arranged such that each has exactly one object to its right.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EQ", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O2"}, {"op": "right_of", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "INT_1"}
        ]}]}
    ],
    logical_facts=["forall(O1, (count(O2, right_of(O1, O2), N), N = 1))"]
))

# NEW Rule: every_object_is_close (from document)
add_rule_if_new(BongardRule(
    name="every_object_is_close",
    description="Every object is close to at least one other object.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "is_close_to", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, exists(O2, (O1!= O2 and is_close_to(O1, O2))))"]
))


# --- Counting Rules ---

# Rule 10: Exactly three circles
add_rule_if_new(BongardRule(
    name="exactly_three_circles",
    description="There are exactly three circles in the image.",
    program_ast=[
        {"op": "EQ", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]},
            {"op": "INT_3"}
        ]}
    ],
    logical_facts=["count(O, shape(O, circle), N), N = 3"]
))

# Rule 11: More squares than circles
add_rule_if_new(BongardRule(
    name="more_squares_than_circles",
    description="The number of squares is greater than the number of circles.",
    program_ast=[
        {"op": "GT", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]}]},
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]}
        ]}
    ],
    logical_facts=["count(O, shape(O, square), N_S), count(O, shape(O, circle), N_C), N_S > N_C"]
))

# NEW Rule: exactly_two_large_objects (from document)
add_rule_if_new(BongardRule(
    name="exactly_two_large_objects",
    description="There are exactly two large objects.",
    program_ast=[{"op": "EQ", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]}]},
        {"op": "INT_2"}
    ]}],
    logical_facts=["count(O, size(O, large), N), N = 2"]
))

# NEW Rule: hollow_less_than_solid (from document)
add_rule_if_new(BongardRule(
    name="hollow_less_than_solid",
    description="The number of hollow objects is less than the number of solid objects.",
    program_ast=[{"op": "LT", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]}]},
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]}]}
    ]}],
    logical_facts=["count(O, fill(O, hollow), N_H), count(O, fill(O, solid), N_S), N_H < N_S"]
))

# NEW Rule: at_least_three_triangles (from document)
add_rule_if_new(BongardRule(
    name="at_least_three_triangles",
    description="There are at least three triangles.",
    program_ast=[{"op": "GT", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]}]},
        {"op": "INT_2"}
    ]}],
    logical_facts=["count(O, shape(O, triangle), N), N >= 3"]
))

# NEW Rule: exactly_four_objects (from document)
add_rule_if_new(BongardRule(
    name="exactly_four_objects",
    description="There are exactly four objects.",
    program_ast=[{"op": "EQ", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "O"}]}, # Count all objects
        {"op": "INT_4"}
    ]}],
    logical_facts=["count(O, object(O), N), N = 4"]
))

# NEW Rule: more_upright_than_inverted (from document)
add_rule_if_new(BongardRule(
    name="more_upright_than_inverted",
    description="There are more upright objects than inverted objects.",
    program_ast=[{"op": "GT", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "orientation", "args": [{"op": "object_variable", "value": "O"}, {"op": "upright"}]}]},
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "orientation", "args": [{"op": "object_variable", "value": "O"}, {"op": "inverted"}]}]}
    ]}],
    logical_facts=["count(O, orientation(O, upright), N_U), count(O, orientation(O, inverted), N_I), N_U > N_I"]
))

# NEW Rule: circles_equal_squares (from document)
add_rule_if_new(BongardRule(
    name="circles_equal_squares",
    description="The number of circles equals the number of squares.",
    program_ast=[{"op": "EQ", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]},
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]}]}
    ]}],
    logical_facts=["count(O, shape(O, circle), N_C), count(O, shape(O, square), N_S), N_C = N_S"]
))

# NEW Rule: exactly_one_gradient_fill (from document)
add_rule_if_new(BongardRule(
    name="exactly_one_gradient_fill",
    description="There is exactly one object with a gradient fill.",
    program_ast=[{"op": "EQ", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "gradient"}]}]},
        {"op": "INT_1"}
    ]}],
    logical_facts=["count(O, fill(O, gradient), N), N = 1"]
))

# NEW Rule: no_objects_present (from document)
add_rule_if_new(BongardRule(
    name="no_objects_present",
    description="There are no objects (count is zero).",
    program_ast=[{"op": "EQ", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "O"}]}, # Count all objects
        {"op": "INT_0"}
    ]}],
    logical_facts=["count(O, object(O), N), N = 0"]
))

# NEW Rule: dotted_texture_gte_2 (from document)
add_rule_if_new(BongardRule(
    name="dotted_texture_gte_2",
    description="The number of objects with dotted texture is greater than or equal to 2.",
    program_ast=[{"op": "GT", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "texture", "args": [{"op": "object_variable", "value": "O"}, {"op": "dotted_texture"}]}]},
        {"op": "INT_1"}
    ]}],
    logical_facts=["count(O, texture(O, dotted_texture), N), N >= 2"]
))

# NEW Rule: fewer_than_3_objects (from document)
add_rule_if_new(BongardRule(
    name="fewer_than_3_objects",
    description="There are fewer than 3 objects.",
    program_ast=[{"op": "LT", "args": [
        {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "O"}]}, # Count all objects
        {"op": "INT_3"}
    ]}],
    logical_facts=["count(O, object(O), N), N < 3"]
))


# --- Negated/Absence Rules ---

# Rule 13: No objects intersect
add_rule_if_new(BongardRule(
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
))

# Rule 14: Not all objects are the same size
add_rule_if_new(BongardRule(
    name="not_all_same_size",
    description="Objects in the image are not all the same size.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S_val"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "S_val"}]}]}]}
        ]}
    ],
    logical_facts=["not(exists(S_val, forall(O, size(O, S_val))))"]
))

# NEW Rule: no_triangles (from document)
add_rule_if_new(BongardRule(
    name="no_triangles",
    description="There are no triangles in the image.",
    program_ast=[{"op": "NOT", "args": [{"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]}]}]}],
    logical_facts=["not(exists(O, shape(O, triangle)))"]
))

# NEW Rule: no_small_solid_objects (from document)
add_rule_if_new(BongardRule(
    name="no_small_solid_objects",
    description="No object is both small and solid.",
    program_ast=[{"op": "NOT", "args": [
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "small"}]},
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]}
        ]}]}
    ]}],
    logical_facts=["not(exists(O, (size(O, small) and fill(O, solid))))"]
))

# NEW Rule: not_all_aligned_horizontally (from document)
add_rule_if_new(BongardRule(
    name="not_all_aligned_horizontally",
    description="It is not true that all objects are aligned horizontally.",
    program_ast=[{"op": "NOT", "args": [
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "IMPLIES", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "aligned_horizontally", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
        ]}]}]}
    ]}],
    logical_facts=["not(forall(O1, forall(O2, (O1!= O2 implies aligned_horizontally(O1, O2)))))"]
))

# NEW Rule: no_contained_objects (from document)
add_rule_if_new(BongardRule(
    name="no_contained_objects",
    description="There is no object contained within another object.",
    program_ast=[{"op": "NOT", "args": [
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "contains", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]}]}
    ]}],
    logical_facts=["not(exists(O1, exists(O2, contains(O1, O2))))"]
))

# NEW Rule: square_count_not_two (from document)
add_rule_if_new(BongardRule(
    name="square_count_not_two",
    description="The number of squares is NOT equal to 2.",
    program_ast=[{"op": "NOT", "args": [
        {"op": "EQ", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]}]},
            {"op": "INT_2"}
        ]}
    ]}],
    logical_facts=["count(O, shape(O, square), N), N!= 2"]
))


# --- Compositional Rules ---

# Rule 17: All circles are small
add_rule_if_new(BongardRule(
    name="all_circles_are_small",
    description="If an object is a circle, then it is small.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "IMPLIES", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "small"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, circle) implies size(O, small)))"]
))

# Rule 18: Exactly two objects are striped
add_rule_if_new(BongardRule(
    name="exactly_two_striped",
    description="There are exactly two striped objects in the image.",
    program_ast=[
        {"op": "EQ", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "striped"}]}]},
            {"op": "INT_2"}
        ]}
    ],
    logical_facts=["count(O, fill(O, striped), N), N = 2"]
))

# Rule 19: All objects are either square or triangle
add_rule_if_new(BongardRule(
    name="all_square_or_triangle",
    description="All objects in the image are either squares or triangles.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, square) or shape(O, triangle)))"]
))

# Rule 22: All objects are upright (orientation)
add_rule_if_new(BongardRule(
    name="all_upright",
    description="All objects in the image are in an upright orientation.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "orientation", "args": [{"op": "object_variable", "value": "O"}, {"op": "upright"}]}]}],
    logical_facts=["forall(O, orientation(O, upright))"]
))

# Rule 23: There is at least one object with a dotted texture
add_rule_if_new(BongardRule(
    name="exists_dotted_texture",
    description="There is at least one object with a dotted texture.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "texture", "args": [{"op": "object_variable", "value": "O"}, {"op": "dotted_texture"}]}]}
    ],
    logical_facts=["exists(O, texture(O, dotted_texture))"]
))

# Rule 24: A square is to the left of a circle
add_rule_if_new(BongardRule(
    name="square_left_of_circle",
    description="There is a square to the left of a circle.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "C"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "S"}, {"op": "square"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "C"}, {"op": "circle"}]},
            {"op": "left_of", "args": [{"op": "object_variable", "value": "S"}, {"op": "object_variable", "value": "C"}]}
        ]}]}]}
    ],
    logical_facts=["exists(S, exists(C, (shape(S, square) and shape(C, circle) and left_of(S, C))))"]
))

# Rule 25: All objects are either solid or hollow
add_rule_if_new(BongardRule(
    name="all_solid_or_hollow",
    description="All objects in the image are either solid or hollow.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]},
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (fill(O, solid) or fill(O, hollow)))"]
))

# Rule 26: Exactly one object
add_rule_if_new(BongardRule(
    name="exactly_one_object",
    description="There is exactly one object in the image.",
    program_ast=[
        {"op": "EQ", "args": [
            {"op": "COUNT", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "O"}]}, # Count all objects
            {"op": "INT_1"}
        ]}
    ],
    logical_facts=["count(O, object(O), N), N = 1"]
))

# Rule 27: No large objects
add_rule_if_new(BongardRule(
    name="no_large_objects",
    description="No object in the image is large.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]}]}
        ]}
    ],
    logical_facts=["not(exists(O, size(O, large)))"]
))

# Rule 28: All objects are the same shape
add_rule_if_new(BongardRule(
    name="all_same_shape",
    description="All objects in the image are the same shape.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "S_val"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "object_variable", "value": "S_val"}]}]}]}
    ],
    logical_facts=["exists(S_val, forall(O, shape(O, S_val)))"]
))

# NEW Rule: if_square_then_solid (from document)
add_rule_if_new(BongardRule(
    name="if_square_then_solid",
    description="If an object is a square, then it is solid.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "IMPLIES", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]},
            {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, square) implies fill(O, solid)))"]
))

# NEW Rule: small_large_not_intersect (from document)
add_rule_if_new(BongardRule(
    name="small_large_not_intersect",
    description="There exists a small object AND a large object, but they do not intersect.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
            {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O1"}, {"op": "small"}]},
            {"op": "size", "args": [{"op": "object_variable", "value": "O2"}, {"op": "large"}]},
            {"op": "NOT", "args": [{"op": "intersects", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]}
        ]}]}]}
    ],
    logical_facts=["exists(O1, exists(O2, (O1!= O2 and size(O1, small) and size(O2, large) and not(intersects(O1, O2)))))"]
))

# NEW Rule: triangles_small_hollow (from document)
add_rule_if_new(BongardRule(
    name="triangles_small_hollow",
    description="All objects that are triangles are also small and hollow.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "IMPLIES", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "triangle"}]},
            {"op": "AND", "args": [
                {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "small"}]},
                {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]}
            ]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, triangle) implies (size(O, small) and fill(O, hollow))))"]
))

# NEW Rule: no_circle_and_large (from document)
add_rule_if_new(BongardRule(
    name="no_circle_and_large",
    description="No object is both a circle and large.",
    program_ast=[
        {"op": "NOT", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
                {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]},
                {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]}
            ]}]}
        ]}
    ],
    logical_facts=["not(exists(O, (shape(O, circle) and size(O, large))))"]
))

# NEW Rule: square_or_not_solid (from document)
add_rule_if_new(BongardRule(
    name="square_or_not_solid",
    description="Every object is either a square or it is not solid.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]},
            {"op": "NOT", "args": [{"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]}]}
        ]}]}
    ],
    logical_facts=["forall(O, (shape(O, square) or not(fill(O, solid))))"]
))

# NEW Rule: triangle_not_aligned_horizontally (from document)
add_rule_if_new(BongardRule(
    name="triangle_not_aligned_horizontally",
    description="There exists an object that is a triangle, and it is not aligned horizontally with any other object.",
    program_ast=[
        {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "AND", "args": [
            {"op": "shape", "args": [{"op": "object_variable", "value": "O1"}, {"op": "triangle"}]},
            {"op": "NOT", "args": [{"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "AND", "args": [
                {"op": "NOT", "args": [{"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]},
                {"op": "aligned_horizontally", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
            ]}]}]}
        ]}]}
    ],
    logical_facts=["exists(O1, (shape(O1, triangle) and not(exists(O2, (O1!= O2 and aligned_horizontally(O1, O2))))))"]
))

# NEW Rule: all_small_hollow_or_large_solid (from document)
add_rule_if_new(BongardRule(
    name="all_small_hollow_or_large_solid",
    description="All objects are either small and hollow, or large and solid.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "OR", "args": [
            {"op": "AND", "args": [
                {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "small"}]},
                {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "hollow"}]}
            ]},
            {"op": "AND", "args": [
                {"op": "size", "args": [{"op": "object_variable", "value": "O"}, {"op": "large"}]},
                {"op": "fill", "args": [{"op": "object_variable", "value": "O"}, {"op": "solid"}]}
            ]}
        ]}]}
    ],
    logical_facts=["forall(O, ((size(O, small) and fill(O, hollow)) or (size(O, large) and fill(O, solid))))"]
))

# NEW Rule: not_all_circles (from document)
add_rule_if_new(BongardRule(
    name="not_all_circles",
    description="It is not the case that all objects are circles.",
    program_ast=[{"op": "NOT", "args": [{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]}]}],
    logical_facts=["not(forall(O, shape(O, circle)))"]
))

# NEW Rule: if_square_then_triangle_exists (from document)
add_rule_if_new(BongardRule(
    name="if_square_then_triangle_exists",
    description="If there is a square, then there must also be a triangle.",
    program_ast=[
        {"op": "IMPLIES", "args": [
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O1"}, {"op": "square"}]}]},
            {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "shape", "args": [{"op": "object_variable", "value": "O2"}, {"op": "triangle"}]}]}
        ]}
    ],
    logical_facts=["(exists(O1, shape(O1, square)) implies exists(O2, shape(O2, triangle)))"]
))

# NEW Rule: above_objects_are_circles (from document)
add_rule_if_new(BongardRule(
    name="above_objects_are_circles",
    description="Every object that is above another object is a circle.",
    program_ast=[
        {"op": "FORALL", "args": [{"op": "object_variable", "value": "O1"}, {"op": "FORALL", "args": [{"op": "object_variable", "value": "O2"}, {"op": "IMPLIES", "args": [
            {"op": "above", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]},
            {"op": "shape", "args": [{"op": "object_variable", "value": "O1"}, {"op": "circle"}]}
        ]}]}]}
    ],
    logical_facts=["forall(O1, forall(O2, (above(O1, O2) implies shape(O1, circle))))"]
))


# --- Freeform and Abstract Shape Rules ---

# NEW Rule: all_convex_shapes (from document)
add_rule_if_new(BongardRule(
    name="all_convex_shapes",
    description="All shapes are convex.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "convex", "args": [{"op": "object_variable", "value": "O"}, {"op": "true"}]}]}],
    logical_facts=["forall(O, convex(O, true))"]
))

# NEW Rule: exists_right_angle_shape (from document)
add_rule_if_new(BongardRule(
    name="exists_right_angle_shape",
    description="There exists a shape with a right angle.",
    program_ast=[{"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "has_right_angle", "args": [{"op": "object_variable", "value": "O"}, {"op": "true"}]}]}],
    logical_facts=["exists(O, has_right_angle(O, true))"]
))

# NEW Rule: all_symmetrical_vertically (from document)
add_rule_if_new(BongardRule(
    name="all_symmetrical_vertically",
    description="All objects are symmetrical along their vertical axis.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "symmetrical_vertically", "args": [{"op": "object_variable", "value": "O"}, {"op": "true"}]}]}],
    logical_facts=["forall(O, symmetrical_vertically(O, true))"]
))

# NEW Rule: at_least_one_nested_structure (from document)
add_rule_if_new(BongardRule(
    name="at_least_one_nested_structure",
    description="There exists at least one nested structure (A contains B).",
    program_ast=[{"op": "EXISTS", "args": [{"op": "object_variable", "value": "O1"}, {"op": "EXISTS", "args": [{"op": "object_variable", "value": "O2"}, {"op": "contains", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}]}]}],
    logical_facts=["exists(O1, exists(O2, contains(O1, O2)))"]
))

# NEW Rule: all_two_stroke_shapes (from document)
add_rule_if_new(BongardRule(
    name="all_two_stroke_shapes",
    description="All objects are composed of exactly two strokes/lines.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "EQ", "args": [{"op": "num_strokes", "args": [{"op": "object_variable", "value": "O"}, {"op": "INT_2"}]}, {"op": "true"}]}]}],
    logical_facts=["forall(O, num_strokes(O, 2))"]
))

# NEW Rule: image_has_fan_pattern (from document)
add_rule_if_new(BongardRule(
    name="image_has_fan_pattern",
    description="The image contains a 'fan' pattern (multiple lines originating from a single point).",
    program_ast=[{"op": "fan_pattern", "args": [{"op": "scene", "value": "scene"}, {"op": "true"}]}],
    logical_facts=["fan_pattern(scene, true)"]
))

# NEW Rule: all_open_shapes (from document)
add_rule_if_new(BongardRule(
    name="all_open_shapes",
    description="All objects are \"open\" shapes (e.g., arcs, zigzags, not closed polygons).",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "is_closed", "args": [{"op": "object_variable", "value": "O"}, {"op": "false"}]}]}],
    logical_facts=["forall(O, is_closed(O, false))"]
))

# NEW Rule: every_object_is_composite (from document)
add_rule_if_new(BongardRule(
    name="every_object_is_composite",
    description="Every object is a composite of at least two primitive shapes.",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "is_composite", "args": [{"op": "object_variable", "value": "O"}, {"op": "true"}]}]}],
    logical_facts=["forall(O, is_composite(O, true))"]
))

# NEW Rule: no_perfect_circle_or_square (from document)
add_rule_if_new(BongardRule(
    name="no_perfect_circle_or_square",
    description="No object is a perfect circle or square (implying freeform/irregular shapes).",
    program_ast=[{"op": "FORALL", "args": [{"op": "object_variable", "value": "O"}, {"op": "AND", "args": [
        {"op": "NOT", "args": [{"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "circle"}]}]},
        {"op": "NOT", "args": [{"op": "shape", "args": [{"op": "object_variable", "value": "O"}, {"op": "square"}]}]}
    ]}]}],
    logical_facts=["forall(O, (not(shape(O, circle)) and not(shape(O, square))))"]
))

# NEW Rule: image_contains_spiral (from document)
add_rule_if_new(BongardRule(
    name="image_contains_spiral",
    description="The image contains a spiral.",
    program_ast=[{"op": "EXISTS", "args": [{"op": "object_variable", "value": "O"}, {"op": "is_spiral", "args": [{"op": "object_variable", "value": "O"}, {"op": "true"}]}]}],
    logical_facts=["exists(O, is_spiral(O, true))"]
))

logger.info(f"Loaded {len(ALL_BONGARD_RULES)} extended Bongard rules.")
