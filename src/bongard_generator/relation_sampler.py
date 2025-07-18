# This file has been replaced by spatial_sampler.py
"""Spatial and topological relationship sampler for Bongard problems"""

import logging
import random
import math
from typing import List, Dict, Any, Tuple, Optional

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    # randrange excludes hi, so we +1
    return random.randrange(lo, hi+1)

logger = logging.getLogger(__name__)

class RelationSampler:
    """Sampler for generating objects with specific spatial relationships."""
    
    def __init__(self, canvas_size: int, min_obj_size: int = 20, max_obj_size: int = 60):
        """
        Initialize the relation sampler.
        
        Args:
            canvas_size: Size of the canvas
            min_obj_size: Minimum object size
            max_obj_size: Maximum object size
        """
        self.canvas_size = canvas_size
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        
        # Available spatial relationships
        self.relations = ['left_of', 'right_of', 'above', 'below', 'near', 'overlap', 'nested']
        
    def sample(self, num_objects: int, relation: str = 'random') -> List[Dict[str, Any]]:
        """
        Sample objects with a specific spatial relationship.
        
        Args:
            num_objects: Number of objects to generate
            relation: Spatial relationship ('left_of', 'above', 'near', etc.)
            
        Returns:
            List of object dictionaries with positions and properties
        """
        if num_objects < 1:
            return []
        
        if relation == 'random':
            relation = random.choice(self.relations)
        
        objects = []
        
        # Generate first object
        first_obj = self._generate_random_object()
        objects.append(first_obj)
        
        # Generate remaining objects with specified relationship to previous ones
        for i in range(1, num_objects):
            if relation in ['left_of', 'right_of', 'above', 'below']:
                obj = self._sample_directional_relation(objects, relation)
            elif relation == 'near':
                obj = self._sample_near_relation(objects)
            elif relation == 'overlap':
                obj = self._sample_overlap_relation(objects)
            elif relation == 'nested':
                obj = self._sample_nested_relation(objects)
            else:
                obj = self._generate_random_object()
            
            if obj:
                objects.append(obj)
        
        return objects
    
    def _generate_random_object(self) -> Dict[str, Any]:
        """Generate a random object with random properties."""
        size = safe_randint(self.min_obj_size, self.max_obj_size)
        x = safe_randint(0, self.canvas_size - size)
        y = safe_randint(0, self.canvas_size - size)
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'shape': random.choice(['circle', 'triangle', 'square']),
            'fill': random.choice(['solid', 'outline']),
            'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
        }
    
    def _sample_directional_relation(self, existing_objects: List[Dict[str, Any]], relation: str) -> Optional[Dict[str, Any]]:
        """Sample object with directional relationship to existing objects."""
        if not existing_objects:
            return self._generate_random_object()
        
        # Choose a reference object
        ref_obj = random.choice(existing_objects)
        size = safe_randint(self.min_obj_size, self.max_obj_size)
        
        # Calculate position based on relation
        if relation == 'left_of':
            x = safe_randint(0, max(0, ref_obj['x'] - size - 10))
            y = safe_randint(max(0, ref_obj['y'] - size//2), 
                             min(self.canvas_size - size, ref_obj['y'] + ref_obj['size'] + size//2))
        elif relation == 'right_of':
            x = safe_randint(min(self.canvas_size - size, ref_obj['x'] + ref_obj['size'] + 10), 
                             self.canvas_size - size)
            y = safe_randint(max(0, ref_obj['y'] - size//2),
                             min(self.canvas_size - size, ref_obj['y'] + ref_obj['size'] + size//2))
        elif relation == 'above':
            x = safe_randint(max(0, ref_obj['x'] - size//2),
                             min(self.canvas_size - size, ref_obj['x'] + ref_obj['size'] + size//2))
            y = safe_randint(0, max(0, ref_obj['y'] - size - 10))
        elif relation == 'below':
            x = safe_randint(max(0, ref_obj['x'] - size//2),
                             min(self.canvas_size - size, ref_obj['x'] + ref_obj['size'] + size//2))
            y = safe_randint(min(self.canvas_size - size, ref_obj['y'] + ref_obj['size'] + 10),
                             self.canvas_size - size)
        else:
            return self._generate_random_object()
        
        # Ensure coordinates are within bounds
        x = max(0, min(self.canvas_size - size, x))
        y = max(0, min(self.canvas_size - size, y))
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'shape': random.choice(['circle', 'triangle', 'square']),
            'fill': random.choice(['solid', 'outline']),
            'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
        }
    
    def _sample_near_relation(self, existing_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sample object near existing objects."""
        if not existing_objects:
            return self._generate_random_object()
        
        ref_obj = random.choice(existing_objects)
        size = safe_randint(self.min_obj_size, self.max_obj_size)
        
        # Place object within a certain distance
        max_distance = 50
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(10, max_distance)
        
        center_x = ref_obj['x'] + ref_obj['size'] // 2
        center_y = ref_obj['y'] + ref_obj['size'] // 2
        
        x = int(center_x + distance * math.cos(angle) - size // 2)
        y = int(center_y + distance * math.sin(angle) - size // 2)
        
        # Ensure coordinates are within bounds
        x = max(0, min(self.canvas_size - size, x))
        y = max(0, min(self.canvas_size - size, y))
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'shape': random.choice(['circle', 'triangle', 'square']),
            'fill': random.choice(['solid', 'outline']),
            'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
        }
    
    def _sample_overlap_relation(self, existing_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sample object that overlaps with existing objects."""
        if not existing_objects:
            return self._generate_random_object()
        
        ref_obj = random.choice(existing_objects)
        size = safe_randint(self.min_obj_size, self.max_obj_size)
        
        # Create overlap by placing object partially over reference object
        overlap_x = safe_randint(-size//2, ref_obj['size']//2)
        overlap_y = safe_randint(-size//2, ref_obj['size']//2)
        
        x = ref_obj['x'] + overlap_x
        y = ref_obj['y'] + overlap_y
        
        # Ensure coordinates are within bounds
        x = max(0, min(self.canvas_size - size, x))
        y = max(0, min(self.canvas_size - size, y))
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'shape': random.choice(['circle', 'triangle', 'square']),
            'fill': random.choice(['solid', 'outline']),
            'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
        }
    
    def _sample_nested_relation(self, existing_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Sample object nested inside existing objects."""
        if not existing_objects:
            return self._generate_random_object()
        
        # Find largest object to nest inside
        ref_obj = max(existing_objects, key=lambda obj: obj['size'])
        
        # Make nested object smaller
        max_nested_size = ref_obj['size'] - 10
        if max_nested_size < self.min_obj_size:
            return self._generate_random_object()
        
        size = safe_randint(self.min_obj_size, max_nested_size)
        
        # Place inside reference object
        margin = (ref_obj['size'] - size) // 2
        x = ref_obj['x'] + safe_randint(5, max(5, margin))
        y = ref_obj['y'] + safe_randint(5, max(5, margin))
        
        return {
            'x': x,
            'y': y,
            'size': size,
            'shape': random.choice(['circle', 'triangle', 'square']),
            'fill': random.choice(['solid', 'outline']),
            'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple'])
        }
    
    def validate_relation(self, objects: List[Dict[str, Any]], relation: str) -> bool:
        """
        Validate that objects satisfy the specified spatial relationship.
        
        Args:
            objects: List of objects to validate
            relation: Expected spatial relationship
            
        Returns:
            True if relation is satisfied, False otherwise
        """
        if len(objects) < 2:
            return True
        
        obj1, obj2 = objects[0], objects[1]
        
        if relation == 'left_of':
            return obj1['x'] + obj1['size'] <= obj2['x']
        elif relation == 'right_of':
            return obj2['x'] + obj2['size'] <= obj1['x']
        elif relation == 'above':
            return obj1['y'] + obj1['size'] <= obj2['y']
        elif relation == 'below':
            return obj2['y'] + obj2['size'] <= obj1['y']
        elif relation == 'near':
            # Calculate distance between centers
            cx1 = obj1['x'] + obj1['size'] // 2
            cy1 = obj1['y'] + obj1['size'] // 2
            cx2 = obj2['x'] + obj2['size'] // 2
            cy2 = obj2['y'] + obj2['size'] // 2
            distance = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            return distance <= 50  # Near threshold
        elif relation == 'overlap':
            return self._objects_overlap(obj1, obj2)
        elif relation == 'nested':
            return self._object_nested(obj1, obj2) or self._object_nested(obj2, obj1)
        
        return True
    
    def _objects_overlap(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if two objects overlap."""
        x1_min, x1_max = obj1['x'], obj1['x'] + obj1['size']
        y1_min, y1_max = obj1['y'], obj1['y'] + obj1['size']
        
        x2_min, x2_max = obj2['x'], obj2['x'] + obj2['size']
        y2_min, y2_max = obj2['y'], obj2['y'] + obj2['size']
        
        x_overlap = x1_max > x2_min and x1_min < x2_max
        y_overlap = y1_max > y2_min and y1_min < y2_max
        
        return x_overlap and y_overlap
    
    def _object_nested(self, inner: Dict[str, Any], outer: Dict[str, Any]) -> bool:
        """Check if inner object is nested inside outer object."""
        inner_x_min = inner['x']
        inner_x_max = inner['x'] + inner['size']
        inner_y_min = inner['y']
        inner_y_max = inner['y'] + inner['size']
        
        outer_x_min = outer['x']
        outer_x_max = outer['x'] + outer['size']
        outer_y_min = outer['y']
        outer_y_max = outer['y'] + outer['size']
        
        return (inner_x_min >= outer_x_min and inner_x_max <= outer_x_max and
                inner_y_min >= outer_y_min and inner_y_max <= outer_y_max)

if __name__ == "__main__":
    # Test the relation sampler
    sampler = RelationSampler(128)
    
    # Test different relations
    relations_to_test = ['left_of', 'above', 'near', 'overlap']
    
    for relation in relations_to_test:
        print(f"\nTesting {relation} relation:")
        objects = sampler.sample(2, relation)
        
        for i, obj in enumerate(objects):
            print(f"  Object {i}: pos=({obj['x']}, {obj['y']}), size={obj['size']}")
        
        is_valid = sampler.validate_relation(objects, relation)
        print(f"  Validation: {'✓' if is_valid else '✗'}")
