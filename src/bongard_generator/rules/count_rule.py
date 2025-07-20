"""
Count-based rule for Bongard problems.
Even vs. odd number of objects.
"""
import random
from typing import List, Dict, Any, Tuple
from ..rule_loader import AbstractRule

class CountRule(AbstractRule):
    """Rule based on object count (even vs odd)."""
    
    @property
    def name(self) -> str:
        return "COUNT_PARITY"
    
    @property
    def description(self) -> str:
        return "Even number of objects vs. odd number of objects"
    
    def apply(self, objects: List[Dict[str, Any]], is_positive: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        current_count = len(objects)
        target_is_even = is_positive
        
        # Modify object list to achieve target parity
        if (current_count % 2 == 0) != target_is_even:
            if target_is_even and current_count % 2 == 1:
                # Need even count, currently odd - add one object
                if objects:
                    new_obj = objects[0].copy()
                    new_obj["position"] = (
                        random.randint(50, 200),
                        random.randint(50, 200)
                    )
                    objects.append(new_obj)
            elif not target_is_even and current_count % 2 == 0 and current_count > 0:
                # Need odd count, currently even - remove one object
                objects.pop()
        
        final_count = len(objects)
        features = {
            "rule_applied": self.name,
            "satisfaction": is_positive,
            "object_count": final_count,
            "is_even_count": final_count % 2 == 0,
            "count_modified": final_count != current_count
        }
        
        return objects, features
