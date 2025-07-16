# Folder: bongard_solver/
# File: rule_evaluator.py
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import collections

# Import BongardRule and RuleAtom for type hinting and rule structure
try:
    from bongard_rules import BongardRule, ALL_RULE_ATOMS
except ImportError:
    logging.warning("Could not import bongard_rules. Using dummy BongardRule and RuleAtom.")
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
    ALL_RULE_ATOMS = {} # Empty dummy

logger = logging.getLogger(__name__)

# --- Rule Evaluation Functions ---

def _resolve_ast_arg(arg_node: Any, object_bindings: Dict[str, Any]) -> Any:
    """
    Resolves an argument from an AST node, handling object variables and value nodes.
    Args:
        arg_node (Any): The argument from the AST (can be a dict for node or a direct value).
        object_bindings (Dict[str, Any]): Current mapping of object variables (e.g., 'O') to actual object IDs.
    Returns:
        Any: The resolved value (e.g., an object ID, a string value like 'circle', or an integer).
    """
    if isinstance(arg_node, dict):
        op = arg_node.get('op')
        value = arg_node.get('value')

        if op == 'object_variable':
            # Resolve object variable to its bound ID
            return object_bindings.get(value)
        elif op is not None and 'value' in arg_node: # For terminal values like {"op": "circle", "value": "circle"}
            return value
        elif op is not None and op.startswith('INT_'): # For integer constants like {"op": "INT_3"}
            try:
                return int(op.split('_')[1])
            except ValueError:
                logger.warning(f"Could not parse integer from AST node op: {op}")
                return None
        else:
            # If it's a nested expression (e.g., a COUNT operation), return the node itself for further evaluation
            return arg_node
    else:
        # If it's already a direct value (e.g., an object ID string from a previous resolution)
        return arg_node

def _evaluate_predicate(predicate_op: str, args: List[Any], scene_graph: Dict[str, Any], object_bindings: Dict[str, Any]) -> bool:
    """
    Evaluates a single predicate (attribute, relation, or comparison) against a scene graph
    given current object bindings.
    Args:
        predicate_op (str): The name of the predicate (e.g., 'shape', 'left_of', 'GT').
        args (List[Any]): List of arguments for the predicate, as they appear in the AST.
        scene_graph (Dict[str, Any]): The scene graph to evaluate against.
        object_bindings (Dict[str, Any]): Current mapping of object variables (e.g., 'O') to actual object IDs.
    Returns:
        bool: True if the predicate holds, False otherwise.
    """
    resolved_args = [_resolve_ast_arg(arg, object_bindings) for arg in args]
    
    # Filter out None values that might result from unresolved object variables
    # This is important for relations where both objects must be resolved.
    if any(arg is None for arg in resolved_args) and predicate_op not in ["GT", "LT", "EQ"]:
        # For comparisons, None might indicate a COUNT sub-expression not yet evaluated.
        # For attributes/relations, None means an object variable couldn't be bound.
        return False

    # Handle attribute predicates (e.g., shape(O, circle))
    if predicate_op in ["shape", "color", "fill", "size", "orientation", "texture"]:
        obj_id = resolved_args[0]
        target_value = resolved_args[1]
        
        # Ensure obj_id is an integer (from 'obj_X' string) or directly an int
        if isinstance(obj_id, str) and obj_id.startswith('obj_'):
            try:
                obj_id = int(obj_id.replace('obj_', ''))
            except ValueError:
                logger.warning(f"Invalid object ID format: {obj_id}")
                return False
        elif not isinstance(obj_id, int):
            logger.warning(f"Object ID not an integer or 'obj_X' string: {obj_id}")
            return False

        for obj_data in scene_graph.get('objects', []):
            # Scene graph objects might have integer IDs or string IDs like 'obj_0'
            # Ensure comparison is consistent
            sg_obj_id = obj_data['id']
            if isinstance(sg_obj_id, str) and sg_obj_id.startswith('obj_'):
                try:
                    sg_obj_id = int(sg_obj_id.replace('obj_', ''))
                except ValueError:
                    continue # Skip malformed object ID in scene graph
            
            if sg_obj_id == obj_id:
                return obj_data['attributes'].get(predicate_op) == target_value
        return False

    # Handle relational predicates (e.g., left_of(O1, O2))
    elif predicate_op in ["left_of", "right_of", "above", "below", "is_close_to",
                          "aligned_horizontally", "aligned_vertically", "contains", "intersects"]:
        obj1_id = resolved_args[0]
        obj2_id = resolved_args[1]

        # Ensure obj_ids are integers (from 'obj_X' string) or directly ints
        if isinstance(obj1_id, str) and obj1_id.startswith('obj_'):
            try: obj1_id = int(obj1_id.replace('obj_', ''))
            except ValueError: return False
        if isinstance(obj2_id, str) and obj2_id.startswith('obj_'):
            try: obj2_id = int(obj2_id.replace('obj_', ''))
            except ValueError: return False
        
        if not isinstance(obj1_id, int) or not isinstance(obj2_id, int):
            return False # Both must be valid object IDs

        for rel_data in scene_graph.get('relations', []):
            # Ensure comparison is consistent with scene graph relation IDs (int vs str)
            rel_sub_id = rel_data['subject_id']
            rel_obj_id = rel_data['object_id']
            if isinstance(rel_sub_id, str) and rel_sub_id.startswith('obj_'):
                try: rel_sub_id = int(rel_sub_id.replace('obj_', ''))
                except ValueError: continue
            if isinstance(rel_obj_id, str) and rel_obj_id.startswith('obj_'):
                try: rel_obj_id = int(rel_obj_id.replace('obj_', ''))
                except ValueError: continue

            if rel_data['type'] == predicate_op and \
               rel_sub_id == obj1_id and \
               rel_obj_id == obj2_id:
                return True
        return False
    
    # Handle comparison operators (GT, LT, EQ)
    elif predicate_op in ["GT", "LT", "EQ"]:
        # Arguments to comparison operators can be direct integers or results of COUNT
        val1 = resolved_args[0]
        val2 = resolved_args[1]

        # If arguments are still AST nodes (e.g., a COUNT node), evaluate them recursively
        if isinstance(val1, dict):
            val1 = _evaluate_ast_node(val1, scene_graph, object_bindings)
        if isinstance(val2, dict):
            val2 = _evaluate_ast_node(val2, scene_graph, object_bindings)

        # Ensure both values are numerical for comparison
        if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
            logger.warning(f"Comparison attempted with non-numeric values: {val1} ({type(val1)}) vs {val2} ({type(val2)}). Predicate: {predicate_op}")
            return False

        if predicate_op == "GT": return val1 > val2
        if predicate_op == "LT": return val1 < val2
        if predicate_op == "EQ": return val1 == val2
        return False # Should not reach here

    # Handle object variable equality (e.g., O1 != O2 in FORALL(O1, FORALL(O2, AND(NOT(EQ(O1,O2)), ...))))
    # This is a special case of EQ, handled directly by _evaluate_predicate when op is "EQ"
    # and args are object variables.
    # The AST will represent this as {"op": "EQ", "args": [{"op": "object_variable", "value": "O1"}, {"op": "object_variable", "value": "O2"}]}
    # The _resolve_ast_arg will turn these into their actual bound IDs.
    # So, the above "GT", "LT", "EQ" block will handle this correctly.
    
    logger.warning(f"Unknown predicate operation: {predicate_op}")
    return False

def _evaluate_ast_node(node_dict: Dict[str, Any], scene_graph: Dict[str, Any], object_bindings: Dict[str, Any]) -> Any:
    """
    Recursively evaluates an AST node against a scene graph given object bindings.
    Returns a boolean for logical expressions or a numerical value for COUNT.
    """
    op = node_dict['op']
    args = node_dict.get('args', [])
    
    # Terminal values (e.g., 'circle', 'red', 'INT_3') are resolved by _resolve_ast_arg
    # and passed to their parent predicates. They don't evaluate to a boolean on their own.
    # If this function is called directly on a terminal node, it's likely a malformed AST.
    if op in [p.name for p in ALL_RULE_ATOMS.values() if p.is_value or p.name.startswith('INT_')]:
        logger.warning(f"Attempted to evaluate terminal AST node '{op}' directly as a boolean expression. This is likely an AST construction error.")
        return False # Or raise an error, depending on desired strictness

    # Predicates (attribute, relational, comparison)
    elif op in [p.name for p in ALL_RULE_ATOMS.values() if p.arity != -1 and not p.is_value] + ["GT", "LT", "EQ"]:
        return _evaluate_predicate(op, args, scene_graph, object_bindings)
    
    # Logical Operators
    elif op == "AND":
        return all(_evaluate_ast_node(arg, scene_graph, object_bindings) for arg in args)
    elif op == "OR":
        return any(_evaluate_ast_node(arg, scene_graph, object_bindings) for arg in args)
    elif op == "NOT":
        return not _evaluate_ast_node(args[0], scene_graph, object_bindings)
    elif op == "IMPLIES":  # A -> B is equivalent to !A or B
        antecedent = _evaluate_ast_node(args[0], scene_graph, object_bindings)
        consequent = _evaluate_ast_node(args[1], scene_graph, object_bindings)
        return (not antecedent) or consequent
    
    # Quantifiers
    elif op == "EXISTS":
        var_node = args[0] # This is typically {"op": "object_variable", "value": "O"}
        predicate_node = args[1]
        
        var_name = var_node.get('value', 'O_default') # Get the variable name (e.g., "O", "O1")

        for obj_data in scene_graph.get('objects', []):
            # Create new bindings for this object variable
            new_bindings = object_bindings.copy()
            # Ensure the object ID is correctly extracted from scene_graph's object data
            # Scene graph object IDs might be integers or strings like 'obj_0'
            sg_obj_id = obj_data['id']
            if isinstance(sg_obj_id, str) and sg_obj_id.startswith('obj_'):
                try: sg_obj_id = int(sg_obj_id.replace('obj_', ''))
                except ValueError: continue # Skip malformed IDs
            
            new_bindings[var_name] = sg_obj_id
            
            if _evaluate_ast_node(predicate_node, scene_graph, new_bindings):
                return True
        return False
    elif op == "FORALL":
        var_node = args[0]
        predicate_node = args[1]
        
        var_name = var_node.get('value', 'O_default')

        objects_in_scene = scene_graph.get('objects', [])
        if not objects_in_scene:  # If no objects, FORALL is trivially true
            return True
        
        for obj_data in objects_in_scene:
            new_bindings = object_bindings.copy()
            sg_obj_id = obj_data['id']
            if isinstance(sg_obj_id, str) and sg_obj_id.startswith('obj_'):
                try: sg_obj_id = int(sg_obj_id.replace('obj_', ''))
                except ValueError: continue
            
            new_bindings[var_name] = sg_obj_id
            
            if not _evaluate_ast_node(predicate_node, scene_graph, new_bindings):
                return False  # Found a counterexample
        return True
    
    # Counting
    elif op == "COUNT":
        # COUNT returns a number, not a boolean. It's expected to be an argument to GT/LT/EQ.
        # This function will compute the count.
        var_node = args[0]
        predicate_node = args[1]
        
        var_name = var_node.get('value', 'O_default')

        count = 0
        for obj_data in scene_graph.get('objects', []):
            temp_bindings = object_bindings.copy() # Use a copy to not pollute outer bindings
            sg_obj_id = obj_data['id']
            if isinstance(sg_obj_id, str) and sg_obj_id.startswith('obj_'):
                try: sg_obj_id = int(sg_obj_id.replace('obj_', ''))
                except ValueError: continue
            
            temp_bindings[var_name] = sg_obj_id
            
            if _evaluate_ast_node(predicate_node, scene_graph, temp_bindings):
                count += 1
        return count # Return the actual count

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
    
    try:
        # Assuming the top-level AST is a list with one dictionary representing the root node
        if rule.program_ast and isinstance(rule.program_ast, list) and rule.program_ast:
            root_node_dict = rule.program_ast[0]
            # Start evaluation with empty object bindings
            result = _evaluate_ast_node(root_node_dict, scene_graph, {})
            logger.debug(f"Rule '{rule.name}' evaluation on scene graph: {result}")
            return result
        else:
            logger.warning(f"Rule '{rule.name}' program_ast is malformed: {rule.program_ast}")
            return False
    except Exception as e:
        logger.error(f"Error evaluating rule '{rule.name}' on scene graph: {e}. Scene graph: {scene_graph}", exc_info=True)
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
    for i, sg_wrapper in enumerate(support_scene_graphs):
        # Extract the actual scene graph from the wrapper dictionary
        # This assumes sg_wrapper is like {'image_path': ..., 'scene_graph': {...}}
        sg = sg_wrapper.get('scene_graph', {})
        if not sg:
            logger.warning(f"Scene graph missing for support image {sg_wrapper.get('image_path', 'N/A')}. Skipping.")
            continue

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
        else:  # If it's a negative rule (defines negative examples)
            if support_labels[i] == 0 and predicted_satisfies:  # Negative rule satisfied by negative example
                is_correct = True
            elif support_labels[i] == 1 and not predicted_satisfies:  # Negative rule not satisfied by positive example
                is_correct = True
        
        if is_correct:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(support_scene_graphs) if len(support_scene_graphs) > 0 else 0.0
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
        logger.warning("No rules or scene graphs provided for find_best_rules.")
        return []
    
    rule_scores = []
    
    # Combine all ground truth scene graphs and their labels for evaluation
    # Note: support_scene_graphs in evaluate_rule_on_support_set expects a list of dictionaries
    # like {'image_path': ..., 'scene_graph': {...}}
    all_gt_sgs_wrapped = [{'image_path': f"pos_img_{i}", 'scene_graph': sg} for i, sg in enumerate(positive_scene_graphs)] + \
                         [{'image_path': f"neg_img_{i}", 'scene_graph': sg} for i, sg in enumerate(negative_scene_graphs)]
    all_gt_labels = [1] * len(positive_scene_graphs) + [0] * len(negative_scene_graphs)
    
    if not all_gt_sgs_wrapped:
        logger.warning("No ground truth scene graphs provided for rule evaluation in find_best_rules.")
        return []

    for rule in all_rules:
        score = evaluate_rule_on_support_set(rule, all_gt_sgs_wrapped, all_gt_labels)
        rule_scores.append((rule, score))
    
    # Sort by score in descending order
    rule_scores.sort(key=lambda x: x[1], reverse=True)
    
    return rule_scores[:k]

