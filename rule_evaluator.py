# Folder: bongard_solver/
# File: rule_evaluator.py

import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import collections

# Import BongardRule and RuleAtom for type hinting and rule structure
from bongard_rules import BongardRule, ALL_RULE_ATOMS

logger = logging.getLogger(__name__)

# --- Rule Evaluation Functions ---

def _evaluate_predicate(predicate_op: str, args: List[Any], scene_graph: Dict[str, Any], object_bindings: Dict[str, Any]) -> bool:
    """
    Evaluates a single predicate (attribute or relation) against a scene graph
    given current object bindings.
    
    Args:
        predicate_op (str): The name of the predicate (e.g., 'shape', 'left_of').
        args (List[Any]): List of arguments for the predicate (e.g., ['O', 'circle']).
        scene_graph (Dict[str, Any]): The scene graph to evaluate against.
        object_bindings (Dict[str, Any]): Current mapping of object variables (e.g., 'O') to actual object IDs.

    Returns:
        bool: True if the predicate holds, False otherwise.
    """
    # Helper to resolve object variable to actual object ID
    def resolve_obj_id(arg_val):
        if isinstance(arg_val, str) and arg_val.startswith('obj'):
            return int(arg_val.replace('obj', ''))
        elif isinstance(arg_val, str) and arg_val in object_bindings:
            return object_bindings[arg_val]
        return None # Not an object ID or bound variable

    # Resolve arguments
    resolved_args = []
    for arg in args:
        if isinstance(arg, dict) and 'op' in arg and arg['op'] == 'object_variable':
            resolved_args.append(object_bindings.get(arg['value'])) # Get bound ID for 'O'
        elif isinstance(arg, dict) and 'op' in arg and 'value' in arg: # For value nodes like {"op": "circle"}
            resolved_args.append(arg['value'])
        else: # Direct value or already resolved object ID
            resolved_args.append(arg)

    # Handle attribute predicates
    if predicate_op in ["shape", "color", "fill", "size", "orientation", "texture"]:
        obj_id = resolve_obj_id(resolved_args[0])
        target_value = resolved_args[1]
        
        if obj_id is None: return False

        for obj_data in scene_graph.get('objects', []):
            if obj_data['id'] == obj_id:
                return obj_data['attributes'].get(predicate_op) == target_value
        return False

    # Handle relational predicates
    elif predicate_op in ["left_of", "right_of", "above", "below", "is_close_to",
                          "aligned_horizontally", "aligned_vertically", "contains", "intersects"]:
        obj1_id = resolve_obj_id(resolved_args[0])
        obj2_id = resolve_obj_id(resolved_args[1])
        
        if obj1_id is None or obj2_id is None: return False

        for rel_data in scene_graph.get('relations', []):
            if rel_data['type'] == predicate_op and \
               rel_data['subject_id'] == obj1_id and \
               rel_data['object_id'] == obj2_id:
                return True
        return False
    
    # Handle comparison operators (for COUNT)
    elif predicate_op in ["gt", "lt", "eq"]:
        val1 = resolved_args[0]
        val2 = resolved_args[1]
        
        # If values are strings like "count(O, shape(O,circle))", try to parse them
        if isinstance(val1, str) and val1.startswith("count(") and isinstance(val2, (int, str)):
            # This is a simplified evaluation. In a real system, count would be computed.
            # For now, assume if it's a count, it's compared to an integer.
            # This part needs to be refined if actual count computation is integrated.
            logger.warning(f"Direct comparison of COUNT terms not fully implemented. Assuming count value is an integer. {val1} vs {val2}")
            try:
                # Mocking count: if the scene has at least 'val2' objects, assume count is > val2
                # This is a very weak mock, needs real count logic.
                count_val = len(scene_graph.get('objects', [])) # Dummy count
                if predicate_op == "gt": return count_val > int(val2)
                if predicate_op == "lt": return count_val < int(val2)
                if predicate_op == "eq": return count_val == int(val2)
            except ValueError:
                return False # Cannot convert to int
            
        elif isinstance(val1, int) and isinstance(val2, int):
            if predicate_op == "gt": return val1 > val2
            if predicate_op == "lt": return val1 < val2
            if predicate_op == "eq": return val1 == val2
        return False

    elif predicate_op == "eq": # For object variable equality (O1 != O2)
        if resolved_args[0] == resolved_args[1]:
            return True # If args are the same, they are equal
        return False # If args are different, they are not equal

    logger.warning(f"Unknown predicate operation: {predicate_op}")
    return False


def _evaluate_ast_node(node_dict: Dict[str, Any], scene_graph: Dict[str, Any], object_bindings: Dict[str, Any]) -> bool:
    """
    Recursively evaluates an AST node against a scene graph given object bindings.
    """
    op = node_dict['op']
    args = node_dict.get('args', [])

    if op in ["circle", "square", "triangle", "star", "red", "blue", "green", "black", "white",
              "solid", "hollow", "striped", "dotted", "small", "medium", "large",
              "upright", "inverted", "none_texture", "striped_texture", "dotted_texture",
              "object_variable", "INT_1", "INT_2", "INT_3", "INT_4", "INT_5"]:
        # These are terminal values, not evaluable predicates on their own.
        # Their 'value' is used by parent predicates.
        return True # Or handle as error if called directly as a boolean expression

    elif op in ["shape", "color", "fill", "size", "orientation", "texture",
                "left_of", "right_of", "above", "below", "is_close_to",
                "aligned_horizontally", "aligned_vertically", "contains", "intersects",
                "GT", "LT", "EQ"]: # Comparison ops are also predicates
        # Evaluate a predicate
        return _evaluate_predicate(op, args, scene_graph, object_bindings)

    elif op == "AND":
        return all(_evaluate_ast_node(arg, scene_graph, object_bindings) for arg in args)
    elif op == "OR":
        return any(_evaluate_ast_node(arg, scene_graph, object_bindings) for arg in args)
    elif op == "NOT":
        return not _evaluate_ast_node(args[0], scene_graph, object_bindings)
    elif op == "IMPLIES": # A -> B is equivalent to !A or B
        antecedent = _evaluate_ast_node(args[0], scene_graph, object_bindings)
        consequent = _evaluate_ast_node(args[1], scene_graph, object_bindings)
        return (not antecedent) or consequent
    
    elif op == "COUNT":
        # This is a special case: COUNT doesn't return a boolean, but a number.
        # It needs to be handled by comparison operators (GT, LT, EQ) that use its result.
        # For direct evaluation, we'll return a dummy value or raise an error.
        # The AST structure should ensure COUNT is always an argument to a comparison.
        logger.warning("COUNT operator should be an argument to a comparison (GT, LT, EQ), not evaluated directly.")
        # For now, let's return a dummy count, e.g., number of objects matching the predicate
        var_name = args[0]['value'] if isinstance(args[0], dict) and 'value' in args[0] else 'O'
        predicate_node = args[1]
        
        count = 0
        for obj_data in scene_graph.get('objects', []):
            temp_bindings = {var_name: obj_data['id']}
            if _evaluate_ast_node(predicate_node, scene_graph, temp_bindings):
                count += 1
        return count # Return the count value, not a boolean

    elif op == "EXISTS":
        var_name = args[0]['value'] if isinstance(args[0], dict) and 'value' in args[0] else 'O'
        predicate_node = args[1]
        
        for obj_data in scene_graph.get('objects', []):
            # Create new bindings for this object variable
            new_bindings = object_bindings.copy()
            new_bindings[var_name] = obj_data['id']
            if _evaluate_ast_node(predicate_node, scene_graph, new_bindings):
                return True
        return False

    elif op == "FORALL":
        var_name = args[0]['value'] if isinstance(args[0], dict) and 'value' in args[0] else 'O'
        predicate_node = args[1]
        
        if not scene_graph.get('objects'): # If no objects, FORALL is trivially true
            return True

        for obj_data in scene_graph.get('objects', []):
            new_bindings = object_bindings.copy()
            new_bindings[var_name] = obj_data['id']
            if not _evaluate_ast_node(predicate_node, scene_graph, new_bindings):
                return False # Found a counterexample
        return True

    elif op == "ILP_INDUCED":
        # This is a placeholder for rules induced by ILP where the AST is not fully parsed.
        # You would need a more sophisticated evaluation for arbitrary Prolog facts.
        # For now, we'll assume a simple match or a dummy evaluation.
        # The 'args' here would contain the raw logical fact string.
        raw_fact_string = args[0]
        logger.warning(f"Evaluating raw ILP induced fact: {raw_fact_string}. This is a dummy evaluation.")
        # Dummy evaluation: if the fact string contains "circle" and the scene has a circle
        if "circle" in raw_fact_string and any(obj['attributes'].get('shape') == 'circle' for obj in scene_graph.get('objects', [])):
            return True
        return False # Fallback

    logger.error(f"Unsupported AST operation: {op}")
    return False


def evaluate_rule_on_scene_graph(rule: BongardRule, scene_graph: Dict[str, Any]) -> bool:
    """
    Evaluates a given BongardRule (represented by its program_ast) against a single scene graph.
    
    Args:
        rule (BongardRule): The rule to evaluate.
        scene_graph (Dict[str, Any]): The scene graph dictionary representing an image.

    Returns:
        bool: True if the scene graph satisfies the rule, False otherwise.
    """
    if not rule.program_ast:
        logger.warning(f"Rule '{rule.name}' has no program_ast. Cannot evaluate.")
        return False

    # The top-level AST is typically a single rule expression
    try:
        # Assuming the top-level AST is a list with one dictionary representing the root node
        if rule.program_ast and isinstance(rule.program_ast, list) and rule.program_ast:
            root_node_dict = rule.program_ast[0]
            result = _evaluate_ast_node(root_node_dict, scene_graph, {})
            logger.debug(f"Rule '{rule.name}' evaluation on scene graph: {result}")
            return result
        else:
            logger.warning(f"Rule '{rule.name}' program_ast is malformed: {rule.program_ast}")
            return False
    except Exception as e:
        logger.error(f"Error evaluating rule '{rule.name}' on scene graph: {e}. Scene graph: {scene_graph}")
        return False


def evaluate_rule_on_support_set(rule: BongardRule,
                                 support_scene_graphs: List[Dict[str, Any]],
                                 support_labels: List[int]) -> float:
    """
    Evaluates a Bongard rule on a support set and returns a correctness score.
    
    Args:
        rule (BongardRule): The rule to evaluate.
        support_scene_graphs (List[Dict[str, Any]]): List of scene graphs for support images.
        support_labels (List[int]): List of ground truth labels (1 for positive, 0 for negative)
                                    for each support image.

    Returns:
        float: A score indicating how well the rule classifies the support set (e.g., accuracy).
               Returns 0.0 if the rule cannot be evaluated or support set is empty.
    """
    if not support_scene_graphs or len(support_scene_graphs) != len(support_labels):
        logger.warning("Support set is empty or labels mismatch. Cannot evaluate rule.")
        return 0.0

    correct_predictions = 0
    for i, sg in enumerate(support_scene_graphs):
        predicted_satisfies = evaluate_rule_on_scene_graph(rule, sg)
        
        # A positive rule (is_positive_rule=True) should be satisfied by positive examples (label=1)
        # and not satisfied by negative examples (label=0).
        # If is_positive_rule=False, the logic reverses.
        
        is_correct = False
        if rule.is_positive_rule:
            if support_labels[i] == 1 and predicted_satisfies:
                is_correct = True
            elif support_labels[i] == 0 and not predicted_satisfies:
                is_correct = True
        else: # If it's a negative rule (defines negative examples)
            if support_labels[i] == 0 and predicted_satisfies: # Negative rule satisfied by negative example
                is_correct = True
            elif support_labels[i] == 1 and not predicted_satisfies: # Negative rule not satisfied by positive example
                is_correct = True

        if is_correct:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(support_scene_graphs)
    logger.debug(f"Rule '{rule.name}' accuracy on support set: {accuracy:.2f}")
    return accuracy


def find_best_rules(all_rules: List[BongardRule],
                    positive_scene_graphs: List[Dict[str, Any]],
                    negative_scene_graphs: List[Dict[str, Any]],
                    k: int = 1) -> List[Tuple[BongardRule, float]]:
    """
    Evaluates a set of candidate rules against positive and negative scene graphs
    and returns the top-k best-performing rules.
    
    Args:
        all_rules (List[BongardRule]): A list of candidate BongardRule objects.
        positive_scene_graphs (List[Dict[str, Any]]): Scene graphs for positive examples.
        negative_scene_graphs (List[Dict[str, Any]]): Scene graphs for negative examples.
        k (int): Number of top rules to return.

    Returns:
        List[Tuple[BongardRule, float]]: A list of (rule, score) tuples, sorted by score.
    """
    if not all_rules or (not positive_scene_graphs and not negative_scene_graphs):
        return []

    rule_scores = []
    
    # Combine all ground truth scene graphs and their labels for evaluation
    all_gt_sgs = positive_scene_graphs + negative_scene_graphs
    all_gt_labels = [1] * len(positive_scene_graphs) + [0] * len(negative_scene_graphs)

    if not all_gt_sgs:
        logger.warning("No ground truth scene graphs provided for rule evaluation.")
        return []

    for rule in all_rules:
        score = evaluate_rule_on_support_set(rule, all_gt_sgs, all_gt_labels)
        rule_scores.append((rule, score))
    
    # Sort by score in descending order
    rule_scores.sort(key=lambda x: x[1], reverse=True)
    
    return rule_scores[:k]

