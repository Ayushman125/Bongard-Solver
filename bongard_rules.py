# Folder: bongard_solver/

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
    
    # Values/Constants
    "circle": SHAPE_CIRCLE, "square": SHAPE_SQUARE, "triangle": SHAPE_TRIANGLE, "star": SHAPE_STAR,
    "red": COLOR_RED, "blue": COLOR_BLUE, "green": COLOR_GREEN, "black": COLOR_BLACK, "white": COLOR_WHITE,
    "solid": FILL_SOLID, "hollow": FILL_HOLLOW, "striped": FILL_STRIPED, "dotted": FILL_DOTTED,
    "small": SIZE_SMALL, "medium": SIZE_MEDIUM, "large": SIZE_LARGE,
    "upright": ORIENTATION_UPRIGHT, "inverted": ORIENTATION_INVERTED,
    "none_texture": TEXTURE_NONE, "striped_texture": TEXTURE_STRIPED, "dotted_texture": TEXTURE_DOTTED, # Renamed to avoid clash

    # Logical Operators
    "AND": LOGIC_AND, "OR": LOGIC_OR, "NOT": LOGIC_NOT,
    "EXISTS": LOGIC_EXISTS, "FORALL": LOGIC_FORALL, "COUNT": LOGIC_COUNT,
    
    # Comparison Operators (added for completeness, though often handled by DSL Primitives directly)
    "GT": LOGIC_GT, "LT": LOGIC_LT, "EQ": LOGIC_EQ,
    
    # Integer constants (added for completeness, though often handled by DSL Primitives directly)
    "INT_1": INT_1, "INT_2": INT_2, "INT_3": INT_3, "INT_4": INT_4, "INT_5": INT_5,
    "object_variable": RuleAtom("object_variable", 0, is_value=True) # Added for explicit reference
}


class BongardRule:
    """
    Represents a Bongard rule, now with support for a DSL program AST
    and a list of logical facts.
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
        
        ]}
    ],
    logical_facts=["forall(O, (shape(O, circle) implies size(O, small)))"]
)


logger.info(f"Loaded {len(ALL_BONGARD_RULES)} extended Bongard rules.")

