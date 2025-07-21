"""
Rule Validator for Bongard Problems

This module provides a mechanism to validate if a generated scene
(a list of objects) adheres to a given BongardRule's program_ast.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RuleValidator:
    """
    Validates a scene against the logical AST of a Bongard rule by interpreting the AST.
    """
    def __init__(self, scene_objects: List[Dict[str, Any]]):
        self.objects = scene_objects
        # Ensure all objects have a unique ID for comparisons
        for i, obj in enumerate(self.objects):
            if 'object_id' not in obj:
                obj['object_id'] = f"temp_id_{i}"

    def validate_rule(self, rule_ast: Dict[str, Any]) -> bool:
        """
        Starts the recursive validation of the scene against the rule's AST.
        """
        if not self.objects and "FORALL" not in rule_ast.get("op", ""):
            # An empty scene can't satisfy EXISTS or COUNT, but can satisfy FORALL (vacuously true)
            return "FORALL" in rule_ast.get("op", "")
        return self._interpret(rule_ast, {})

    def _get_object_property(self, obj: Dict[str, Any], prop: str) -> Any:
        """Safely get a property from an object, with fallbacks for categorization."""
        if prop == 'size':
            size_val = obj.get('size', 30)
            if size_val < 30: return 'small'
            if size_val > 60: return 'large'
            return 'medium'
        return obj.get(prop)

    def _interpret(self, node: Dict[str, Any], bindings: Dict[str, Dict[str, Any]]) -> Any:
        """
        Recursively interprets an AST node with a given set of variable bindings.
        Returns boolean for logical ops, integer for COUNT, or value for constants.
        """
        op = node.get("op")
        args = node.get("args", [])

        # --- Logical Quantifiers ---
        if op == "FORALL":
            var_name = args[0].get("value")
            predicate = args[1]
            for obj in self.objects:
                new_bindings = bindings.copy()
                new_bindings[var_name] = obj
                if not self._interpret(predicate, new_bindings):
                    return False
            return True

        if op == "EXISTS":
            var_name = args[0].get("value")
            predicate = args[1]
            for obj in self.objects:
                new_bindings = bindings.copy()
                new_bindings[var_name] = obj
                if self._interpret(predicate, new_bindings):
                    return True
            return False

        # --- Logical Connectives ---
        if op == "NOT":
            return not self._interpret(args[0], bindings)

        if op == "AND":
            return all(self._interpret(arg, bindings) for arg in args)

        if op == "OR":
            return any(self._interpret(arg, bindings) for arg in args)
            
        if op == "IMPLIES":
            antecedent = self._interpret(args[0], bindings)
            if not antecedent:
                return True  # A -> B is true if A is false
            return self._interpret(args[1], bindings)

        # --- Attribute Predicates ---
        if op in ["shape", "color", "fill", "size"]:
            var_name = args[0].get("value")
            target_obj = bindings.get(var_name)
            if not target_obj: return False
            
            expected_value = args[1].get("op")
            actual_value = self._get_object_property(target_obj, op)
            return actual_value == expected_value

        # --- Relational Predicates ---
        if op in ["above", "below", "left_of", "right_of", "contains", "intersects", "aligned_horizontally", "aligned_vertically"]:
            var1_name = args[0].get("value")
            var2_name = args[1].get("value")
            obj1 = bindings.get(var1_name)
            obj2 = bindings.get(var2_name)

            if not obj1 or not obj2 or obj1['object_id'] == obj2['object_id']:
                return False

            if op == "above": return obj1['y'] < obj2['y']
            if op == "below": return obj1['y'] > obj2['y']
            if op == "left_of": return obj1['x'] < obj2['x']
            if op == "right_of": return obj1['x'] > obj2['x']
            
            if op == "aligned_horizontally":
                return abs(obj1['y'] - obj2['y']) < 5 # 5 pixel tolerance
            if op == "aligned_vertically":
                return abs(obj1['x'] - obj2['x']) < 5 # 5 pixel tolerance

            if op == "contains":
                obj1_half, obj2_half = obj1['size'] / 2, obj2['size'] / 2
                return (obj1['x'] - obj1_half <= obj2['x'] - obj2_half and
                        obj1['x'] + obj1_half >= obj2['x'] + obj2_half and
                        obj1['y'] - obj1_half <= obj2['y'] - obj2_half and
                        obj1['y'] + obj1_half >= obj2['y'] + obj2_half)

            if op == "intersects":
                dist_x = abs(obj1['x'] - obj2['x'])
                dist_y = abs(obj1['y'] - obj2['y'])
                min_dist = (obj1['size'] + obj2['size']) / 2
                return dist_x < min_dist and dist_y < min_dist

        # --- Counting and Comparison ---
        if op == "COUNT":
            var_name = args[0].get("value")
            predicate = args[1]
            count = 0
            for obj in self.objects:
                new_bindings = bindings.copy()
                new_bindings[var_name] = obj
                if self._interpret(predicate, new_bindings):
                    count += 1
            return count

        if op in ["EQ", "GT", "LT"]:
            val1 = self._interpret(args[0], bindings)
            val2 = self._interpret(args[1], bindings)
            
            if op == "EQ": return val1 == val2
            if op == "GT": return val1 > val2
            if op == "LT": return val1 < val2

        # --- Value Nodes ---
        if node.get("is_value"):
            return node.get("op")
            
        if op.startswith("INT_"):
            return int(op.split("_")[1])

        # --- Variable Nodes ---
        if op == "object_variable":
            # This node just represents a variable, the actual object is in bindings
            return bindings.get(node.get("value"))

        logger.error(f"Unknown operator in validator: {op}")
        return False
