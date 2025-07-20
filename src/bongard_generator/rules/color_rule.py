"""
Color-based rule for Bongard problems.
Same color vs. different colors. Only uses grayscale values to maintain black/white output.
"""
import random
from typing import List, Dict, Any, Tuple
from ..rule_loader import AbstractRule

class ColorRule(AbstractRule):
    """Rule based on color uniformity (using grayscale values)."""
    
    @property
    def name(self) -> str:
        return "COLOR_UNIFORMITY"
    
    @property
    def description(self) -> str:
        return "All objects same color vs. mixed colors (grayscale only)"
    
    def apply(self, objects: List[Dict[str, Any]], is_positive: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not objects:
            return objects, {"rule_applied": self.name, "satisfaction": is_positive}
        
        # Use color labels for rule logic, but all shapes will be rendered black
        available_colors = ["color_a", "color_b", "color_c"]  # Logical color labels
        
        if is_positive:
            # All objects should have the same color (logically)
            target_color = objects[0].get("color_label", "color_a")
            for obj in objects:
                obj["color_label"] = target_color
        else:
            # Objects should have different colors (logically)
            if len(objects) >= 2:
                for i, obj in enumerate(objects[:len(available_colors)]):
                    obj["color_label"] = available_colors[i % len(available_colors)]
        
        # All objects get black color for rendering (Bongard problems are black/white)
        for obj in objects:
            obj["color"] = "black"
        
        features = {
            "rule_applied": self.name,
            "satisfaction": is_positive,
            "color_diversity": len(set(obj.get("color_label", "color_a") for obj in objects)),
            "logical_colors_used": list(set(obj.get("color_label", "color_a") for obj in objects))
        }
        
        return objects, features
