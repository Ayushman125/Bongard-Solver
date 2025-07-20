"""
Shape-based rule for Bongard problems.
All objects have the same shape vs. different shapes.
"""
import random
from typing import List, Dict, Any, Tuple
from ..rule_loader import AbstractRule

class ShapeRule(AbstractRule):
    """Rule based on shape uniformity."""
    
    @property
    def name(self) -> str:
        return "SHAPE_UNIFORMITY"
    
    @property
    def description(self) -> str:
        return "All objects have the same shape vs. mixed shapes"
    
    def apply(self, objects: List[Dict[str, Any]], is_positive: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not objects:
            return objects, {"rule_applied": self.name, "satisfaction": is_positive}
        
        available_shapes = ["circle", "square", "triangle"]
        
        if is_positive:
            # All objects should have the same shape
            target_shape = objects[0].get("shape", "circle")
            for obj in objects:
                obj["shape"] = target_shape
        else:
            # Objects should have different shapes
            if len(objects) >= 2:
                # Ensure at least two different shapes
                for i, obj in enumerate(objects[:len(available_shapes)]):
                    obj["shape"] = available_shapes[i % len(available_shapes)]
        
        features = {
            "rule_applied": self.name,
            "satisfaction": is_positive,
            "shape_diversity": len(set(obj.get("shape", "circle") for obj in objects)),
            "shapes_used": list(set(obj.get("shape", "circle") for obj in objects))
        }
        
        return objects, features
