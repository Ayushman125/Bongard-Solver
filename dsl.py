# Folder: bongard_solver/

import collections
import random
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# Import rule atoms from bongard_rules
from bongard_rules import (
    RuleAtom, ATTR_SHAPE, ATTR_COLOR, ATTR_FILL, ATTR_SIZE, ATTR_ORIENTATION, ATTR_TEXTURE,
    REL_LEFT_OF, REL_RIGHT_OF, REL_ABOVE, REL_BELOW, REL_IS_CLOSE_TO,
    REL_ALIGNED_HORIZONTALLY, REL_ALIGNED_VERTICALLY, REL_CONTAINS, REL_INTERSECTS,
    LOGIC_AND, LOGIC_OR, LOGIC_NOT, LOGIC_EXISTS, LOGIC_FORALL, LOGIC_COUNT,
    SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_TRIANGLE, SHAPE_STAR,
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_BLACK, COLOR_WHITE,
    FILL_SOLID, FILL_HOLLOW, FILL_STRIPED, FILL_DOTTED,
    SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE,
    ORIENTATION_UPRIGHT, ORIENTATION_INVERTED,
    TEXTURE_NONE, TEXTURE_STRIPED, TEXTURE_DOTTED
)

logger = logging.getLogger(__name__)

# --- DSL Primitive Definitions ---
# These represent the functions available in our domain-specific language.

class Primitive:
    """Represents a primitive operation in the DSL."""
    def __init__(self, name: str, func: Any, type_signature: str, is_terminal: bool = False):
        self.name = name
        self.func = func # The actual Python function/logic (for AST construction)
        self.type_signature = type_signature # e.g., "(obj, color_type) -> boolean" or "boolean -> boolean"
        self.is_terminal = is_terminal # True if this primitive is a constant/value

    def __repr__(self):
        return f"Primitive({self.name}, type={self.type_signature})"

    def __call__(self, *args):
        # When called, this constructs the dictionary representation for the ASTNode
        if self.is_terminal:
            return {"op": self.name}
        return {"op": self.name, "args": list(args)}

# Value Primitives (Terminals) - correspond to RuleAtom.is_value=True
DSL_VALUES = [
    Primitive("circle", SHAPE_CIRCLE.name, "shape_type", is_terminal=True),
    Primitive("square", SHAPE_SQUARE.name, "shape_type", is_terminal=True),
    Primitive("triangle", SHAPE_TRIANGLE.name, "shape_type", is_terminal=True),
    Primitive("star", SHAPE_STAR.name, "shape_type", is_terminal=True),
    
    Primitive("red", COLOR_RED.name, "color_type", is_terminal=True),
    Primitive("blue", COLOR_BLUE.name, "color_type", is_terminal=True),
    Primitive("green", COLOR_GREEN.name, "color_type", is_terminal=True),
    Primitive("black", COLOR_BLACK.name, "color_type", is_terminal=True),
    Primitive("white", COLOR_WHITE.name, "color_type", is_terminal=True),

    Primitive("solid", FILL_SOLID.name, "fill_type", is_terminal=True),
    Primitive("hollow", FILL_HOLLOW.name, "fill_type", is_terminal=True),
    Primitive("striped", FILL_STRIPED.name, "fill_type", is_terminal=True),
    Primitive("dotted", FILL_DOTTED.name, "fill_type", is_terminal=True),

    Primitive("small", SIZE_SMALL.name, "size_type", is_terminal=True),
    Primitive("medium", SIZE_MEDIUM.name, "size_type", is_terminal=True),
    Primitive("large", SIZE_LARGE.name, "size_type", is_terminal=True),

    Primitive("upright", ORIENTATION_UPRIGHT.name, "orientation_type", is_terminal=True),
    Primitive("inverted", ORIENTATION_INVERTED.name, "orientation_type", is_terminal=True),

    Primitive("none_texture", TEXTURE_NONE.name, "texture_type", is_terminal=True),
    Primitive("striped_texture", TEXTURE_STRIPED.name, "texture_type", is_terminal=True),
    Primitive("dotted_texture", TEXTURE_DOTTED.name, "texture_type", is_terminal=True),
    
    Primitive("object_variable", "O", "object_variable", is_terminal=True) # Placeholder for object variable
]

# Functional Primitives (Operations) - correspond to RuleAtom.is_value=False
# These define the structure of the AST. The 'func' here is a lambda that
# constructs the dictionary representation for the AST.
DSL_FUNCTIONS = [
    # Attribute Predicates: (object_variable, value_type) -> boolean
    Primitive("shape", lambda o, v: {"op": "shape", "args": [o, v]}, "(object_variable, shape_type) -> boolean"),
    Primitive("color", lambda o, v: {"op": "color", "args": [o, v]}, "(object_variable, color_type) -> boolean"),
    Primitive("fill", lambda o, v: {"op": "fill", "args": [o, v]}, "(object_variable, fill_type) -> boolean"),
    Primitive("size", lambda o, v: {"op": "size", "args": [o, v]}, "(object_variable, size_type) -> boolean"),
    Primitive("orientation", lambda o, v: {"op": "orientation", "args": [o, v]}, "(object_variable, orientation_type) -> boolean"),
    Primitive("texture", lambda o, v: {"op": "texture", "args": [o, v]}, "(object_variable, texture_type) -> boolean"),

    # Relational Predicates: (object_variable, object_variable) -> boolean
    Primitive("left_of", lambda o1, o2: {"op": "left_of", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("right_of", lambda o1, o2: {"op": "right_of", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("above", lambda o1, o2: {"op": "above", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("below", lambda o1, o2: {"op": "below", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("is_close_to", lambda o1, o2: {"op": "is_close_to", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("aligned_horizontally", lambda o1, o2: {"op": "aligned_horizontally", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("aligned_vertically", lambda o1, o2: {"op": "aligned_vertically", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("contains", lambda o1, o2: {"op": "contains", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),
    Primitive("intersects", lambda o1, o2: {"op": "intersects", "args": [o1, o2]}, "(object_variable, object_variable) -> boolean"),

    # Logical Operators: (boolean*) -> boolean or (boolean) -> boolean
    Primitive("AND", lambda *args: {"op": "AND", "args": list(args)}, "boolean* -> boolean"),
    Primitive("OR", lambda *args: {"op": "OR", "args": list(args)}, "boolean* -> boolean"),
    Primitive("NOT", lambda arg: {"op": "NOT", "args": [arg]}, "boolean -> boolean"),
    
    # Quantifiers: (object_variable, boolean) -> boolean
    Primitive("EXISTS", lambda var, pred: {"op": "EXISTS", "args": [var, pred]}, "(object_variable, boolean) -> boolean"),
    Primitive("FORALL", lambda var, pred: {"op": "FORALL", "args": [var, pred]}, "(object_variable, boolean) -> boolean"),
    
    # Counting: (object_variable, boolean) -> count_type (count_type is an integer)
    Primitive("COUNT", lambda var, pred: {"op": "COUNT", "args": [var, pred]}, "(object_variable, boolean) -> count_type"),
    
    # Comparison Operators for Counts: (count_type, count_type) -> boolean or (count_type, integer) -> boolean
    Primitive("GT", lambda c1, c2: {"op": "GT", "args": [c1, c2]}, "(count_type, count_type) -> boolean"),
    Primitive("LT", lambda c1, c2: {"op": "LT", "args": [c1, c2]}, "(count_type, count_type) -> boolean"),
    Primitive("EQ", lambda c1, c2: {"op": "EQ", "args": [c1, c2]}, "(count_type, count_type) -> boolean"),
    
    # Implication: (boolean, boolean) -> boolean
    Primitive("IMPLIES", lambda p, q: {"op": "IMPLIES", "args": [p, q]}, "(boolean, boolean) -> boolean"),

    # Integer constants for COUNT comparisons
    Primitive("INT_1", 1, "integer", is_terminal=True),
    Primitive("INT_2", 2, "integer", is_terminal=True),
    Primitive("INT_3", 3, "integer", is_terminal=True),
    Primitive("INT_4", 4, "integer", is_terminal=True),
    Primitive("INT_5", 5, "integer", is_terminal=True),
]

DSL_PRIMITIVES = DSL_VALUES + DSL_FUNCTIONS

# Define a simple grammar for guided generation (tuples of (rule_type, [possible_primitives]))
# This is a simplified representation of a context-free grammar.
GRAMMAR_RULES = {
    "rule": ["FORALL", "EXISTS", "AND", "OR", "NOT", "COUNT_COMPARE", "IMPLIES"],
    "boolean": ["shape", "color", "fill", "size", "orientation", "texture",
                "left_of", "right_of", "above", "below", "is_close_to",
                "aligned_horizontally", "aligned_vertically", "contains", "intersects",
                "AND", "OR", "NOT", "IMPLIES"], # Boolean expressions can be predicates or logical ops
    "object_variable": ["object_variable"],
    "shape_type": ["circle", "square", "triangle", "star"],
    "color_type": ["red", "blue", "green", "black", "white"],
    "fill_type": ["solid", "hollow", "striped", "dotted"],
    "size_type": ["small", "medium", "large"],
    "orientation_type": ["upright", "inverted"],
    "texture_type": ["none_texture", "striped_texture", "dotted_texture"],
    "count_type": ["COUNT"],
    "integer": ["INT_1", "INT_2", "INT_3", "INT_4", "INT_5"],
    "COUNT_COMPARE": ["GT", "LT", "EQ"],
}

class ASTNode:
    """Represents a node in the Abstract Syntax Tree of a DSL program."""
    def __init__(self, primitive: Primitive, children: Optional[List['ASTNode']] = None, value: Any = None):
        self.primitive = primitive
        self.children = children if children is not None else []
        self.value = value # For terminal nodes (values like 'circle', 1, 'O')

    def __repr__(self):
        if self.primitive.is_terminal:
            return f"Node({self.primitive.name}={self.value})"
        return f"Node({self.primitive.name}, children={self.children})"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the AST node to a dictionary representation."""
        node_dict = {"op": self.primitive.name}
        if self.primitive.is_terminal:
            node_dict["value"] = self.value if self.value is not None else self.primitive.func() # For object_variable, value is 'O'
        if self.children:
            node_dict["args"] = [child.to_dict() for child in self.children]
        return node_dict


class DSLProgram:
    """Represents a complete DSL program as an AST."""
    def __init__(self, root_node: ASTNode):
        self.root = root_node

    def __repr__(self):
        return f"Program({self.root})"

    def to_ast_dict(self) -> Dict[str, Any]:
        """Converts the program's AST to a dictionary."""
        return self.root.to_dict()


class DSLProgramGenerator:
    """
    Generates DSL programs (ASTs) guided by a simple grammar.
    This is a stochastic grammar sampler, not a learned DreamCoder.
    """
    def __init__(self, primitives: List[Primitive], grammar_rules: Dict[str, List[str]], max_depth: int = 5):
        self.primitives_map = {p.name: p for p in primitives}
        self.grammar_rules = grammar_rules
        self.max_depth = max_depth
        logger.info(f"DSLProgramGenerator initialized with {len(primitives)} primitives and grammar rules.")

    def generate_program(self) -> DSLProgram:
        """Generates a random DSL program AST starting from a 'rule' non-terminal."""
        root = self._generate_node("rule", current_depth=0)
        return DSLProgram(root)

    def _generate_node(self, expected_type: str, current_depth: int) -> ASTNode:
        """Recursively generates an AST node based on grammar rules."""
        # If max depth reached or it's a terminal type, try to pick a terminal primitive
        if current_depth >= self.max_depth or expected_type not in self.grammar_rules:
            eligible_primitives = [p for p in DSL_VALUES if p.type_signature == expected_type]
            if not eligible_primitives:
                # Fallback: if no exact type match, try to find any terminal (e.g., if 'boolean' is expected but no bool terminals)
                logger.debug(f"No exact terminal for {expected_type} at depth {current_depth}. Trying any terminal.")
                eligible_primitives = [p for p in DSL_VALUES if p.type_signature in ["boolean", "integer"]] # Broaden search
            if not eligible_primitives:
                raise ValueError(f"No terminal primitive found for type {expected_type} at max depth or no rule defined.")
            
            primitive = random.choice(eligible_primitives)
            return ASTNode(primitive, value=primitive.func() if callable(primitive.func) else primitive.func) # Use func() for 'O', else its value

        # Select a production rule from the grammar
        possible_ops = self.grammar_rules.get(expected_type, [])
        if not possible_ops:
            # If no functional ops, try to generate a terminal anyway
            eligible_primitives = [p for p in DSL_VALUES if p.type_signature == expected_type]
            if eligible_primitives:
                primitive = random.choice(eligible_primitives)
                return ASTNode(primitive, value=primitive.func() if callable(primitive.func) else primitive.func)
            raise ValueError(f"No production rules or terminal primitives for type {expected_type}.")

        chosen_op_name = random.choice(possible_ops)
        primitive = self.primitives_map.get(chosen_op_name)

        if not primitive:
            raise ValueError(f"Primitive '{chosen_op_name}' not found in map for type {expected_type}.")

        # Parse argument types from type_signature
        arg_types_str = primitive.type_signature.split('->')[0].strip()
        arg_types = []
        if arg_types_str.startswith('(') and arg_types_str.endswith(')'):
            arg_types = [t.strip() for t in arg_types_str[1:-1].split(',')]
            if arg_types_str.endswith('*'): # Handle variable arity (like AND, OR)
                num_args = random.randint(2, 3) # Randomly pick 2-3 arguments
                arg_types = [arg_types[0]] * num_args
        elif arg_types_str:
            arg_types = [arg_types_str.strip()]

        children = []
        for arg_type in arg_types:
            children.append(self._generate_node(arg_type, current_depth + 1))
        
        return ASTNode(primitive, children=children)


class ASTToFactsTransducer:
    """
    Converts a DSL program AST into a list of Prolog-style logical facts.
    Handles various logical operators and predicates.
    """
    def __init__(self):
        logger.info("ASTToFactsTransducer initialized.")

    def convert(self, program: DSLProgram) -> List[str]:
        """Converts a DSLProgram AST into a list of logical facts."""
        # For a single rule, the root node usually represents the main logical statement.
        # We'll return a single fact string for the rule.
        return [self._traverse_and_convert(program.root)]

    def _traverse_and_convert(self, node: ASTNode) -> str:
        """Recursive helper to traverse AST and build a single fact string."""
        if node.primitive.is_terminal:
            # Terminal nodes represent values or variables
            return str(node.value) if node.value is not None else node.primitive.name

        op_name = node.primitive.name.lower()
        child_fact_strings = [self._traverse_and_convert(child) for child in node.children]

        if op_name == "and":
            return f"({') and ('.join(child_fact_strings)})"
        elif op_name == "or":
            return f"({') or ('.join(child_fact_strings)})"
        elif op_name == "not":
            return f"not({child_fact_strings[0]})"
        elif op_name == "exists":
            var = child_fact_strings[0]
            predicate = child_fact_strings[1]
            return f"exists({var}, {predicate})"
        elif op_name == "forall":
            var = child_fact_strings[0]
            predicate = child_fact_strings[1]
            return f"forall({var}, {predicate})"
        elif op_name == "implies":
            antecedent = child_fact_strings[0]
            consequent = child_fact_strings[1]
            return f"({antecedent} implies {consequent})"
        elif op_name == "count":
            var = child_fact_strings[0]
            predicate = child_fact_strings[1]
            # For ILP, count might be a relation like count(Predicate, N)
            # Or it might be used in a comparison. Here, we return a term.
            return f"count({var}, {predicate})"
        elif op_name in ["gt", "lt", "eq"]:
            # These are comparison operators, typically used with COUNT
            return f"{child_fact_strings[0]} {op_name} {child_fact_strings[1]}"
        else:
            # Attribute or relation predicates (e.g., shape(O, circle))
            return f"{op_name}({', '.join(child_fact_strings)})"
