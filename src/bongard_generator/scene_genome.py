"""
SceneGenome class for evolutionary scene generation.
Encodes all parameters needed to generate a Bongard scene.
"""

import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import json

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randrange(lo, hi+1)

@dataclass
class SceneGenome:
    """Evolutionary genome for scene generation parameters."""
    rule_desc: str
    label: int  # 1 for positive, 0 for negative
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Evolutionary fitness components
    fitness: float = 0.0
    tester_confidence: float = 0.0
    diversity_score: float = 0.0
    generation: int = 0
    
    # Metadata
    creation_time: float = 0.0
    success_count: int = 0
    
    def __post_init__(self):
        """Initialize default parameters if not provided."""
        if not self.params:
            self.params = self._generate_default_params()
    
    def _generate_default_params(self) -> Dict[str, Any]:
        """Generate default random parameters for scene generation."""
        # Canvas parameters
        canvas_size = 128
        margin = 20
        
        # Object count
        num_objects = safe_randint(1, 4)
        
        # Available attributes
        shapes = ['circle', 'triangle', 'square', 'pentagon', 'hexagon']
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan']
        fills = ['solid', 'hollow', 'striped', 'dotted']
        relations = ['near', 'far', 'overlap', 'inside', 'outside', 'left_of', 'right_of', 'above', 'below']
        
        # Generate object parameters
        objects = []
        for i in range(num_objects):
            obj_size = safe_randint(20, 60)
            obj = {
                'shape': random.choice(shapes),
                'color': random.choice(colors),
                'fill': random.choice(fills),
                'size': obj_size,
                'position': (
                    random.uniform(margin + obj_size//2, canvas_size - margin - obj_size//2),
                    random.uniform(margin + obj_size//2, canvas_size - margin - obj_size//2)
                ),
                'orientation': random.uniform(0, 360),
                'texture': 'flat'
            }
            objects.append(obj)
        
        return {
            'canvas_size': canvas_size,
            'num_objects': num_objects,
            'objects': objects,
            'background_color': 'white',
            'target_relation': random.choice(relations) if 'relation' in self.rule_desc else None,
            'target_shape': random.choice(shapes) if 'shape' in self.rule_desc else None,
            'target_fill': random.choice(fills) if 'fill' in self.rule_desc else None,
            'target_count': num_objects if 'count' in self.rule_desc else None,
            'margin': margin,
            'min_obj_size': 20,
            'max_obj_size': 60
        }
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> 'SceneGenome':
        """Create a mutated copy of this genome."""
        new_params = json.loads(json.dumps(self.params))  # Deep copy
        
        # Mutate object positions
        for obj in new_params['objects']:
            if random.random() < mutation_rate:
                canvas_size = new_params['canvas_size']
                margin = new_params['margin']
                obj_size = obj['size']
                
                # Small position jitter
                dx = random.uniform(-mutation_strength * canvas_size * 0.1, 
                                  mutation_strength * canvas_size * 0.1)
                dy = random.uniform(-mutation_strength * canvas_size * 0.1, 
                                  mutation_strength * canvas_size * 0.1)
                
                new_x = max(margin + obj_size//2, 
                           min(canvas_size - margin - obj_size//2, 
                               obj['position'][0] + dx))
                new_y = max(margin + obj_size//2, 
                           min(canvas_size - margin - obj_size//2, 
                               obj['position'][1] + dy))
                
                obj['position'] = (new_x, new_y)
            
            # Mutate size
            if random.random() < mutation_rate * 0.5:
                size_delta = random.uniform(-5, 5)
                new_size = max(new_params['min_obj_size'], 
                              min(new_params['max_obj_size'], 
                                  obj['size'] + size_delta))
                obj['size'] = int(new_size)
            
            # Mutate orientation
            if random.random() < mutation_rate * 0.3:
                orientation_delta = random.uniform(-30, 30)
                obj['orientation'] = (obj['orientation'] + orientation_delta) % 360
        
        # Mutate object count occasionally
        if random.random() < mutation_rate * 0.2:
            current_count = len(new_params['objects'])
            if random.random() < 0.5 and current_count > 1:
                # Remove an object
                new_params['objects'].pop(random.randint(0, current_count - 1))
                new_params['num_objects'] = len(new_params['objects'])
            elif current_count < 4:
                # Add an object
                new_obj = self._generate_random_object(new_params)
                new_params['objects'].append(new_obj)
                new_params['num_objects'] = len(new_params['objects'])
        
        # Create new genome
        mutated = SceneGenome(
            rule_desc=self.rule_desc,
            label=self.label,
            params=new_params
        )
        mutated.generation = self.generation + 1
        
        return mutated
    
    def _generate_random_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new random object that fits in the canvas."""
        canvas_size = params['canvas_size']
        margin = params['margin']
        
        shapes = ['circle', 'triangle', 'square', 'pentagon', 'hexagon']
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan']
        fills = ['solid', 'hollow', 'striped', 'dotted']
        
        obj_size = safe_randint(params['min_obj_size'], params['max_obj_size'])
        
        return {
            'shape': random.choice(shapes),
            'color': random.choice(colors),
            'fill': random.choice(fills),
            'size': obj_size,
            'position': (
                random.uniform(margin + obj_size//2, canvas_size - margin - obj_size//2),
                random.uniform(margin + obj_size//2, canvas_size - margin - obj_size//2)
            ),
            'orientation': random.uniform(0, 360),
            'texture': 'flat'
        }
    
    def crossover(self, other: 'SceneGenome') -> Tuple['SceneGenome', 'SceneGenome']:
        """Create two offspring by crossing over with another genome."""
        # Ensure compatible genomes
        if self.rule_desc != other.rule_desc or self.label != other.label:
            return self.mutate(), other.mutate()
        
        # Create deep copies
        params1 = json.loads(json.dumps(self.params))
        params2 = json.loads(json.dumps(other.params))
        
        # Crossover object parameters
        min_objects = min(len(params1['objects']), len(params2['objects']))
        crossover_point = random.randint(1, min_objects)
        
        # Swap object lists at crossover point
        new_objects1 = params1['objects'][:crossover_point] + params2['objects'][crossover_point:]
        new_objects2 = params2['objects'][:crossover_point] + params1['objects'][crossover_point:]
        
        params1['objects'] = new_objects1[:4]  # Limit to max 4 objects
        params2['objects'] = new_objects2[:4]
        params1['num_objects'] = len(params1['objects'])
        params2['num_objects'] = len(params2['objects'])
        
        # Mix other parameters
        if random.random() < 0.5:
            params1['background_color'] = params2['background_color']
        else:
            params2['background_color'] = params1['background_color']
        
        # Create offspring
        child1 = SceneGenome(
            rule_desc=self.rule_desc,
            label=self.label,
            params=params1
        )
        child2 = SceneGenome(
            rule_desc=other.rule_desc,
            label=other.label,
            params=params2
        )
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        return child1, child2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization."""
        return {
            'rule_desc': self.rule_desc,
            'label': self.label,
            'params': self.params,
            'fitness': self.fitness,
            'tester_confidence': self.tester_confidence,
            'diversity_score': self.diversity_score,
            'generation': self.generation,
            'creation_time': self.creation_time,
            'success_count': self.success_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneGenome':
        """Create genome from dictionary."""
        genome = cls(
            rule_desc=data['rule_desc'],
            label=data['label'],
            params=data['params']
        )
        genome.fitness = data.get('fitness', 0.0)
        genome.tester_confidence = data.get('tester_confidence', 0.0)
        genome.diversity_score = data.get('diversity_score', 0.0)
        genome.generation = data.get('generation', 0)
        genome.creation_time = data.get('creation_time', 0.0)
        genome.success_count = data.get('success_count', 0)
        return genome
    
    def __str__(self) -> str:
        """String representation of the genome."""
        return (f"SceneGenome(rule='{self.rule_desc}', label={self.label}, "
                f"fitness={self.fitness:.3f}, gen={self.generation})")
    
    def __repr__(self) -> str:
        return self.__str__()

def create_random_genome(rule_desc: str, label: int) -> SceneGenome:
    """Create a completely random genome for the given rule and label."""
    return SceneGenome(rule_desc=rule_desc, label=label)
