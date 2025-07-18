"""CP-SAT constraint-based object placement optimizer"""

import logging
import random
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ortools.sat.python import cp_model
    CP_SAT_AVAILABLE = True
except ImportError:
    logger.warning("OR-Tools not available. CP-SAT optimization will be disabled.")
    CP_SAT_AVAILABLE = False

class PlacementOptimizer:
    """CP-SAT based optimizer for non-overlapping object placement."""
    
    def __init__(self, canvas_size: int, min_obj_size: int = 20, max_obj_size: int = 60):
        self.canvas_size = canvas_size
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.use_cp_sat = CP_SAT_AVAILABLE
        
    def optimize_placement(self, 
                         num_objects: int,
                         fixed_objects: List[Dict[str, Any]] = None,
                         spatial_constraints: List[Dict[str, Any]] = None,
                         max_attempts: int = 100) -> List[Dict[str, Any]]:
        """
        Optimize object placement using CP-SAT or fallback to random sampling.
        
        Args:
            num_objects: Number of objects to place
            fixed_objects: Objects with fixed positions
            spatial_constraints: List of spatial constraints to satisfy
            max_attempts: Maximum optimization attempts
            
        Returns:
            List of object placements with positions and sizes
        """
        if self.use_cp_sat and num_objects > 2:
            return self._cp_sat_placement(num_objects, fixed_objects, spatial_constraints, max_attempts)
        else:
            return self._fallback_placement(num_objects, fixed_objects, spatial_constraints, max_attempts)
    
    def _cp_sat_placement(self, 
                         num_objects: int,
                         fixed_objects: List[Dict[str, Any]] = None,
                         spatial_constraints: List[Dict[str, Any]] = None,
                         max_attempts: int = 100) -> List[Dict[str, Any]]:
        """Use CP-SAT for optimal placement."""
        if not CP_SAT_AVAILABLE:
            return self._fallback_placement(num_objects, fixed_objects, spatial_constraints, max_attempts)
        
        try:
            model = cp_model.CpModel()
            
            # Create variables for each object
            objects = []
            for i in range(num_objects):
                obj = {
                    'id': i,
                    'x': model.NewIntVar(0, self.canvas_size - self.min_obj_size, f'x_{i}'),
                    'y': model.NewIntVar(0, self.canvas_size - self.min_obj_size, f'y_{i}'),
                    'size': model.NewIntVar(self.min_obj_size, self.max_obj_size, f'size_{i}'),
                    'shape': random.choice(['circle', 'triangle', 'square']),
                    'fill': random.choice(['solid', 'outline']),
                    'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
                }
                objects.append(obj)
            
            # Add non-overlap constraints
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    self._add_non_overlap_constraint(model, objects[i], objects[j])
            
            # Add boundary constraints
            for obj in objects:
                model.Add(obj['x'] + obj['size'] <= self.canvas_size)
                model.Add(obj['y'] + obj['size'] <= self.canvas_size)
            
            # Add fixed object constraints
            if fixed_objects:
                for i, fixed_obj in enumerate(fixed_objects):
                    if i < len(objects):
                        model.Add(objects[i]['x'] == fixed_obj['x'])
                        model.Add(objects[i]['y'] == fixed_obj['y'])
                        if 'size' in fixed_obj:
                            model.Add(objects[i]['size'] == fixed_obj['size'])
            
            # Add spatial constraints
            if spatial_constraints:
                for constraint in spatial_constraints:
                    self._add_spatial_constraint(model, objects, constraint)
            
            # Solve the model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10.0  # 10 second timeout
            
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Extract solution
                result_objects = []
                for obj in objects:
                    result_obj = {
                        'x': solver.Value(obj['x']),
                        'y': solver.Value(obj['y']),
                        'size': solver.Value(obj['size']),
                        'shape': obj['shape'],
                        'fill': obj['fill'],
                        'color': obj['color']
                    }
                    result_objects.append(result_obj)
                
                logger.info(f"CP-SAT successfully placed {num_objects} objects")
                return result_objects
            else:
                logger.warning(f"CP-SAT failed to find solution: {status}")
                return self._fallback_placement(num_objects, fixed_objects, spatial_constraints, max_attempts)
                
        except Exception as e:
            logger.error(f"CP-SAT optimization failed: {e}")
            return self._fallback_placement(num_objects, fixed_objects, spatial_constraints, max_attempts)
    
    def _add_non_overlap_constraint(self, model, obj1, obj2):
        """Add non-overlap constraint between two objects."""
        # Objects don't overlap if they are separated in x or y direction
        no_overlap_x1 = model.NewBoolVar(f'no_overlap_x1_{obj1["id"]}_{obj2["id"]}')
        no_overlap_x2 = model.NewBoolVar(f'no_overlap_x2_{obj1["id"]}_{obj2["id"]}')
        no_overlap_y1 = model.NewBoolVar(f'no_overlap_y1_{obj1["id"]}_{obj2["id"]}')
        no_overlap_y2 = model.NewBoolVar(f'no_overlap_y2_{obj1["id"]}_{obj2["id"]}')
        
        # obj1 is to the left of obj2
        model.Add(obj1['x'] + obj1['size'] <= obj2['x']).OnlyEnforceIf(no_overlap_x1)
        # obj2 is to the left of obj1
        model.Add(obj2['x'] + obj2['size'] <= obj1['x']).OnlyEnforceIf(no_overlap_x2)
        # obj1 is above obj2
        model.Add(obj1['y'] + obj1['size'] <= obj2['y']).OnlyEnforceIf(no_overlap_y1)
        # obj2 is above obj1
        model.Add(obj2['y'] + obj2['size'] <= obj1['y']).OnlyEnforceIf(no_overlap_y2)
        
        # At least one separation must be true
        model.AddBoolOr([no_overlap_x1, no_overlap_x2, no_overlap_y1, no_overlap_y2])
    
    def _add_spatial_constraint(self, model, objects, constraint):
        """Add spatial constraint between objects."""
        constraint_type = constraint.get('type', '')
        obj1_idx = constraint.get('obj1', 0)
        obj2_idx = constraint.get('obj2', 1)
        
        if obj1_idx >= len(objects) or obj2_idx >= len(objects):
            return
        
        obj1 = objects[obj1_idx]
        obj2 = objects[obj2_idx]
        
        if constraint_type == 'left_of':
            # obj1 is to the left of obj2
            model.Add(obj1['x'] + obj1['size'] <= obj2['x'])
        elif constraint_type == 'right_of':
            # obj1 is to the right of obj2
            model.Add(obj2['x'] + obj2['size'] <= obj1['x'])
        elif constraint_type == 'above':
            # obj1 is above obj2
            model.Add(obj1['y'] + obj1['size'] <= obj2['y'])
        elif constraint_type == 'below':
            # obj1 is below obj2
            model.Add(obj2['y'] + obj2['size'] <= obj1['y'])
        elif constraint_type == 'near':
            # Objects are close to each other
            min_distance = constraint.get('min_distance', 10)
            max_distance = constraint.get('max_distance', 50)
            
            # Distance variables
            dx = model.NewIntVar(-self.canvas_size, self.canvas_size, f'dx_{obj1_idx}_{obj2_idx}')
            dy = model.NewIntVar(-self.canvas_size, self.canvas_size, f'dy_{obj1_idx}_{obj2_idx}')
            
            # Calculate center-to-center distance
            model.Add(dx == obj1['x'] + obj1['size'] // 2 - obj2['x'] - obj2['size'] // 2)
            model.Add(dy == obj1['y'] + obj1['size'] // 2 - obj2['y'] - obj2['size'] // 2)
            
            # Approximate distance constraint (Manhattan distance)
            abs_dx = model.NewIntVar(0, self.canvas_size, f'abs_dx_{obj1_idx}_{obj2_idx}')
            abs_dy = model.NewIntVar(0, self.canvas_size, f'abs_dy_{obj1_idx}_{obj2_idx}')
            model.AddAbsEquality(abs_dx, dx)
            model.AddAbsEquality(abs_dy, dy)
            
            manhattan_dist = model.NewIntVar(0, 2 * self.canvas_size, f'dist_{obj1_idx}_{obj2_idx}')
            model.Add(manhattan_dist == abs_dx + abs_dy)
            
            model.Add(manhattan_dist >= min_distance)
            model.Add(manhattan_dist <= max_distance)
    
    def _fallback_placement(self, 
                          num_objects: int,
                          fixed_objects: List[Dict[str, Any]] = None,
                          spatial_constraints: List[Dict[str, Any]] = None,
                          max_attempts: int = 100) -> List[Dict[str, Any]]:
        """Fallback random placement with collision avoidance."""
        placed_objects = []
        
        # Add fixed objects first
        if fixed_objects:
            placed_objects.extend(fixed_objects[:num_objects])
        
        # Place remaining objects
        for i in range(len(placed_objects), num_objects):
            best_obj = None
            best_score = -1
            
            for attempt in range(max_attempts):
                # Generate random object
                size = random.randint(self.min_obj_size, self.max_obj_size)
                x = random.randint(0, self.canvas_size - size)
                y = random.randint(0, self.canvas_size - size)
                
                obj = {
                    'x': x,
                    'y': y,
                    'size': size,
                    'shape': random.choice(['circle', 'triangle', 'square']),
                    'fill': random.choice(['solid', 'outline']),
                    'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
                }
                
                # Check for overlaps
                if self._has_overlaps(obj, placed_objects):
                    continue
                
                # Calculate score based on spatial constraints
                score = self._evaluate_spatial_constraints(obj, placed_objects, spatial_constraints, i)
                
                if score > best_score:
                    best_score = score
                    best_obj = obj
                
                # If we found a perfect solution, use it
                if score >= 1.0:
                    break
            
            if best_obj:
                placed_objects.append(best_obj)
            else:
                # Generate a minimal object if we can't find a good placement
                size = self.min_obj_size
                x = random.randint(0, self.canvas_size - size)
                y = random.randint(0, self.canvas_size - size)
                
                obj = {
                    'x': x,
                    'y': y,
                    'size': size,
                    'shape': random.choice(['circle', 'triangle', 'square']),
                    'fill': random.choice(['solid', 'outline']),
                    'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
                }
                placed_objects.append(obj)
        
        logger.info(f"Fallback placement generated {len(placed_objects)} objects")
        return placed_objects
    
    def _has_overlaps(self, obj: Dict[str, Any], existing_objects: List[Dict[str, Any]]) -> bool:
        """Check if object overlaps with existing objects."""
        for existing in existing_objects:
            if self._objects_overlap(obj, existing):
                return True
        return False
    
    def _objects_overlap(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if two objects overlap."""
        # Add small margin to prevent touching
        margin = 5
        
        x1_min, x1_max = obj1['x'] - margin, obj1['x'] + obj1['size'] + margin
        y1_min, y1_max = obj1['y'] - margin, obj1['y'] + obj1['size'] + margin
        
        x2_min, x2_max = obj2['x'], obj2['x'] + obj2['size']
        y2_min, y2_max = obj2['y'], obj2['y'] + obj2['size']
        
        # Check for overlap
        x_overlap = x1_max > x2_min and x1_min < x2_max
        y_overlap = y1_max > y2_min and y1_min < y2_max
        
        return x_overlap and y_overlap
    
    def _evaluate_spatial_constraints(self, 
                                    obj: Dict[str, Any], 
                                    placed_objects: List[Dict[str, Any]],
                                    spatial_constraints: List[Dict[str, Any]],
                                    obj_index: int) -> float:
        """Evaluate how well object satisfies spatial constraints."""
        if not spatial_constraints:
            return 0.5  # Neutral score
        
        total_score = 0.0
        constraint_count = 0
        
        for constraint in spatial_constraints:
            obj1_idx = constraint.get('obj1', 0)
            obj2_idx = constraint.get('obj2', 1)
            
            # Check if this constraint applies to current object
            if obj1_idx == obj_index:
                if obj2_idx < len(placed_objects):
                    other_obj = placed_objects[obj2_idx]
                    score = self._evaluate_single_constraint(obj, other_obj, constraint)
                    total_score += score
                    constraint_count += 1
            elif obj2_idx == obj_index:
                if obj1_idx < len(placed_objects):
                    other_obj = placed_objects[obj1_idx]
                    # Reverse constraint
                    reversed_constraint = self._reverse_constraint(constraint)
                    score = self._evaluate_single_constraint(obj, other_obj, reversed_constraint)
                    total_score += score
                    constraint_count += 1
        
        return total_score / max(constraint_count, 1)
    
    def _evaluate_single_constraint(self, 
                                  obj1: Dict[str, Any], 
                                  obj2: Dict[str, Any], 
                                  constraint: Dict[str, Any]) -> float:
        """Evaluate a single spatial constraint."""
        constraint_type = constraint.get('type', '')
        
        if constraint_type == 'left_of':
            return 1.0 if obj1['x'] + obj1['size'] <= obj2['x'] else 0.0
        elif constraint_type == 'right_of':
            return 1.0 if obj2['x'] + obj2['size'] <= obj1['x'] else 0.0
        elif constraint_type == 'above':
            return 1.0 if obj1['y'] + obj1['size'] <= obj2['y'] else 0.0
        elif constraint_type == 'below':
            return 1.0 if obj2['y'] + obj2['size'] <= obj1['y'] else 0.0
        elif constraint_type == 'near':
            # Calculate distance between centers
            cx1 = obj1['x'] + obj1['size'] // 2
            cy1 = obj1['y'] + obj1['size'] // 2
            cx2 = obj2['x'] + obj2['size'] // 2
            cy2 = obj2['y'] + obj2['size'] // 2
            
            distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            
            min_distance = constraint.get('min_distance', 10)
            max_distance = constraint.get('max_distance', 50)
            
            if min_distance <= distance <= max_distance:
                return 1.0
            else:
                # Partial score based on how close we are
                if distance < min_distance:
                    return max(0.0, distance / min_distance)
                else:
                    return max(0.0, 1.0 - (distance - max_distance) / max_distance)
        
        return 0.5  # Neutral score for unknown constraints
    
    def _reverse_constraint(self, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """Reverse a spatial constraint."""
        constraint_type = constraint.get('type', '')
        reversed_constraint = constraint.copy()
        
        if constraint_type == 'left_of':
            reversed_constraint['type'] = 'right_of'
        elif constraint_type == 'right_of':
            reversed_constraint['type'] = 'left_of'
        elif constraint_type == 'above':
            reversed_constraint['type'] = 'below'
        elif constraint_type == 'below':
            reversed_constraint['type'] = 'above'
        # 'near' constraint is symmetric
        
        return reversed_constraint

class ConstraintGenerator:
    """Generate spatial constraints for CP-SAT optimization."""
    
    def __init__(self):
        self.constraint_types = ['left_of', 'right_of', 'above', 'below', 'near']
    
    def generate_constraints(self, 
                           num_objects: int, 
                           rule_description: str,
                           max_constraints: int = 3) -> List[Dict[str, Any]]:
        """Generate spatial constraints based on rule description."""
        constraints = []
        
        # Parse rule to extract spatial requirements
        if 'LEFT_OF' in rule_description.upper():
            constraints.append({
                'type': 'left_of',
                'obj1': 0,
                'obj2': 1
            })
        elif 'RIGHT_OF' in rule_description.upper():
            constraints.append({
                'type': 'right_of',
                'obj1': 0,
                'obj2': 1
            })
        elif 'ABOVE' in rule_description.upper():
            constraints.append({
                'type': 'above',
                'obj1': 0,
                'obj2': 1
            })
        elif 'BELOW' in rule_description.upper():
            constraints.append({
                'type': 'below',
                'obj1': 0,
                'obj2': 1
            })
        elif 'NEAR' in rule_description.upper():
            constraints.append({
                'type': 'near',
                'obj1': 0,
                'obj2': 1,
                'min_distance': 10,
                'max_distance': 50
            })
        
        # Add random constraints if needed
        while len(constraints) < max_constraints and num_objects > 1:
            obj1 = random.randint(0, num_objects - 1)
            obj2 = random.randint(0, num_objects - 1)
            
            if obj1 != obj2:
                constraint_type = random.choice(self.constraint_types)
                constraint = {
                    'type': constraint_type,
                    'obj1': obj1,
                    'obj2': obj2
                }
                
                if constraint_type == 'near':
                    constraint['min_distance'] = 10
                    constraint['max_distance'] = 50
                
                # Avoid duplicate constraints
                if not any(c['obj1'] == obj1 and c['obj2'] == obj2 and c['type'] == constraint_type 
                          for c in constraints):
                    constraints.append(constraint)
        
        return constraints

if __name__ == "__main__":
    # Test the optimizer
    optimizer = PlacementOptimizer(128)
    
    # Test basic placement
    objects = optimizer.optimize_placement(3)
    print(f"Generated {len(objects)} objects")
    
    # Test with constraints
    constraint_gen = ConstraintGenerator()
    constraints = constraint_gen.generate_constraints(3, "SPATIAL(LEFT_OF)")
    
    objects = optimizer.optimize_placement(3, spatial_constraints=constraints)
    print(f"Generated {len(objects)} objects with constraints")
    
    for i, obj in enumerate(objects):
        print(f"Object {i}: position=({obj['x']}, {obj['y']}), size={obj['size']}")
