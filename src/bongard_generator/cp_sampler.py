"""CP-SAT based scene sampler with constraint validation."""

import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from ortools.sat.python import cp_model
    CP_SAT_AVAILABLE = True
except ImportError:
    logger.warning("OR-Tools not available. CP-SAT sampling will be disabled.")
    CP_SAT_AVAILABLE = False

from .rule_loader import BongardRule
from .spatial_sampler import RelationSampler

@dataclass
class SceneParameters:
    """Parameters for scene generation."""
    canvas_size: int
    min_obj_size: int
    max_obj_size: int
    max_objects: int
    colors: List[str]
    shapes: List[str]
    fills: List[str]

class CPSATSampler:
    """CP-SAT based sampler for constraint-satisfied scene generation."""
    
    def __init__(self, params: SceneParameters):
        self.params = params
        self.relation_sampler = RelationSampler(params.canvas_size)
        self.use_cp_sat = CP_SAT_AVAILABLE
        
    def sample_scene_cp(self, 
                       rule: BongardRule, 
                       num_objects: int, 
                       positive: bool,
                       max_attempts: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Sample a scene using CP-SAT constraints.
        
        Args:
            rule: The Bongard rule to satisfy
            num_objects: Number of objects in the scene
            positive: Whether this is a positive or negative example
            max_attempts: Maximum solver attempts
            
        Returns:
            List of objects or None if sampling failed
        """
        if not self.use_cp_sat:
            return self._fallback_sampling(rule, num_objects, positive)
        
        try:
            model = cp_model.CpModel()
            
            # Create decision variables for each object
            objects = []
            for i in range(num_objects):
                obj_vars = {
                    'x': model.NewIntVar(0, self.params.canvas_size - self.params.min_obj_size, f'x_{i}'),
                    'y': model.NewIntVar(0, self.params.canvas_size - self.params.min_obj_size, f'y_{i}'),
                    'size': model.NewIntVar(self.params.min_obj_size, self.params.max_obj_size, f'size_{i}'),
                    'shape_idx': model.NewIntVar(0, len(self.params.shapes) - 1, f'shape_{i}'),
                    'fill_idx': model.NewIntVar(0, len(self.params.fills) - 1, f'fill_{i}'),
                    'color_idx': model.NewIntVar(0, len(self.params.colors) - 1, f'color_{i}'),
                }
                objects.append(obj_vars)
            
            # Add boundary constraints
            for obj in objects:
                model.Add(obj['x'] + obj['size'] <= self.params.canvas_size)
                model.Add(obj['y'] + obj['size'] <= self.params.canvas_size)
            
            # Add rule-specific constraints
            self._add_rule_constraints(model, objects, rule, positive)
            
            # Add non-overlap constraints for certain relations
            if self._requires_non_overlap(rule):
                self._add_non_overlap_constraints(model, objects)
            
            # Validate model before solving
            validation_errors = model.Validate()
            if validation_errors:
                logger.error(f"Invalid CP-SAT model for rule {rule.description}: {validation_errors}")
                return self._fallback_sampling(rule, num_objects, positive)
            
            # Solve the model
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10.0
            solver.parameters.num_search_workers = 1
            
            # Use solution collector for multiple solutions
            class SolutionCollector(cp_model.CpSolverSolutionCallback):
                def __init__(self, variables, limit=1):
                    cp_model.CpSolverSolutionCallback.__init__(self)
                    self.variables = variables
                    self.solutions = []
                    self.limit = limit
                
                def on_solution_callback(self):
                    if len(self.solutions) < self.limit:
                        solution = {}
                        for name, var in self.variables.items():
                            solution[name] = self.Value(var)
                        self.solutions.append(solution)
            
            # Flatten variables for collector
            flat_vars = {}
            for i, obj in enumerate(objects):
                for key, var in obj.items():
                    flat_vars[f'{key}_{i}'] = var
            
            collector = SolutionCollector(flat_vars, limit=5)
            
            # Search for solutions
            status = solver.SearchForAllSolutions(model, collector)
            
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] and collector.solutions:
                # Convert solution back to object format
                solution = random.choice(collector.solutions)
                return self._solution_to_objects(solution, num_objects)
            else:
                logger.warning(f"No feasible solution found for rule {rule.description}")
                return self._fallback_sampling(rule, num_objects, positive)
                
        except Exception as e:
            logger.error(f"CP-SAT sampling failed: {e}")
            return self._fallback_sampling(rule, num_objects, positive)
    
    def _add_rule_constraints(self, model, objects: List[Dict], rule: BongardRule, positive: bool):
        """Add constraints based on the Bongard rule."""
        features = rule.positive_features if positive else rule.negative_features
        
        for feature_key, feature_value in features.items():
            if feature_key == 'shape':
                shape_idx = self.params.shapes.index(feature_value)
                if positive:
                    # At least one object must have this shape
                    model.AddBoolOr([obj['shape_idx'] == shape_idx for obj in objects])
                else:
                    # No object should have this shape
                    for obj in objects:
                        model.Add(obj['shape_idx'] != shape_idx)
            
            elif feature_key == 'fill':
                fill_idx = self.params.fills.index(feature_value)
                if positive:
                    # At least one object must have this fill
                    model.AddBoolOr([obj['fill_idx'] == fill_idx for obj in objects])
                else:
                    # No object should have this fill
                    for obj in objects:
                        model.Add(obj['fill_idx'] != fill_idx)
            
            elif feature_key == 'count':
                if positive:
                    # Exactly this count is required - handled by num_objects parameter
                    pass
                else:
                    # This count should be avoided - also handled by num_objects parameter
                    pass
            
            elif feature_key == 'relation':
                # Spatial relations are handled by the relation sampler
                # We can add position constraints here if needed
                pass
    
    def _requires_non_overlap(self, rule: BongardRule) -> bool:
        """Check if the rule requires non-overlapping objects."""
        relation = rule.positive_features.get('relation', '')
        return relation not in ['overlap', 'nested', 'inside']
    
    def _add_non_overlap_constraints(self, model, objects: List[Dict]):
        """Add non-overlap constraints between objects."""
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]
                
                # Create boolean variables for each direction
                left = model.NewBoolVar(f'left_{i}_{j}')
                right = model.NewBoolVar(f'right_{i}_{j}')
                above = model.NewBoolVar(f'above_{i}_{j}')
                below = model.NewBoolVar(f'below_{i}_{j}')
                
                # At least one direction must be true (no overlap)
                model.AddBoolOr([left, right, above, below])
                
                # Define the constraints for each direction
                model.Add(obj1['x'] + obj1['size'] <= obj2['x']).OnlyEnforceIf(left)
                model.Add(obj2['x'] + obj2['size'] <= obj1['x']).OnlyEnforceIf(right)
                model.Add(obj1['y'] + obj1['size'] <= obj2['y']).OnlyEnforceIf(above)
                model.Add(obj2['y'] + obj2['size'] <= obj1['y']).OnlyEnforceIf(below)
    
    def _solution_to_objects(self, solution: Dict, num_objects: int) -> List[Dict[str, Any]]:
        """Convert CP-SAT solution to object list."""
        objects = []
        for i in range(num_objects):
            obj = {
                'x': solution[f'x_{i}'],
                'y': solution[f'y_{i}'],
                'size': solution[f'size_{i}'],
                'shape': self.params.shapes[solution[f'shape_{i}']],
                'fill': self.params.fills[solution[f'fill_{i}']],
                'color': self.params.colors[solution[f'color_{i}']],
                'position': (solution[f'x_{i}'], solution[f'y_{i}'])
            }
            objects.append(obj)
        return objects
    
    def _fallback_sampling(self, rule: BongardRule, num_objects: int, positive: bool) -> List[Dict[str, Any]]:
        """Fallback sampling when CP-SAT is not available or fails."""
        logger.info(f"Using fallback sampling for rule {rule.description}")
        
        objects = []
        features = rule.positive_features if positive else rule.negative_features
        
        # Handle spatial relations
        relation = features.get('relation')
        if relation:
            positions = self.relation_sampler.sample(num_objects, relation)
        else:
            positions = self.relation_sampler.sample(num_objects, 'random')
        
        # Generate objects with required properties
        for i in range(num_objects):
            pos = positions[i] if i < len(positions) else positions[0]
            
            obj = {
                'x': int(pos['position'][0]),
                'y': int(pos['position'][1]),
                'size': pos.get('size', random.randint(self.params.min_obj_size, self.params.max_obj_size)),
                'position': pos['position']
            }
            
            # Apply rule constraints
            if positive:
                obj['shape'] = features.get('shape', random.choice(self.params.shapes))
                obj['fill'] = features.get('fill', random.choice(self.params.fills))
            else:
                # For negative examples, avoid the specified features
                available_shapes = [s for s in self.params.shapes 
                                 if s != features.get('shape')]
                available_fills = [f for f in self.params.fills 
                                 if f != features.get('fill')]
                
                obj['shape'] = random.choice(available_shapes if available_shapes else self.params.shapes)
                obj['fill'] = random.choice(available_fills if available_fills else self.params.fills)
            
            obj['color'] = random.choice(self.params.colors)
            objects.append(obj)
        
        return objects

def sample_scene_cp(rule: BongardRule, 
                   num_objects: int, 
                   positive: bool,
                   canvas_size: int,
                   min_obj_size: int = 20,
                   max_obj_size: int = 60,
                   max_attempts: int = 100) -> Optional[List[Dict[str, Any]]]:
    """
    Convenience function for CP-SAT scene sampling.
    
    Args:
        rule: Bongard rule to satisfy
        num_objects: Number of objects
        positive: Whether positive or negative example
        canvas_size: Size of the canvas
        min_obj_size: Minimum object size
        max_obj_size: Maximum object size
        max_attempts: Maximum sampling attempts
        
    Returns:
        List of objects or None if failed
    """
    params = SceneParameters(
        canvas_size=canvas_size,
        min_obj_size=min_obj_size,
        max_obj_size=max_obj_size,
        max_objects=10,
        colors=['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black'],
        shapes=['circle', 'triangle', 'square', 'pentagon', 'star'],
        fills=['solid', 'outline', 'striped', 'gradient']
    )
    
    sampler = CPSATSampler(params)
    return sampler.sample_scene_cp(rule, num_objects, positive, max_attempts)
