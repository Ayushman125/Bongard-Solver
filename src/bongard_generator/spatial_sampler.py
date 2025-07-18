"""Full RelationSampler with grid-based spatial relationship handling."""

import random
import math
from typing import List, Dict, Any, Tuple

class RelationSampler:
    """Comprehensive spatial and topological relationship sampler."""
    
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.margin = img_size * 0.1  # 10% margin from edges

    def sample(self, n: int, relation: str) -> List[Dict[str, Any]]:
        """Sample n objects with the specified spatial relation."""
        if relation == "near":
            return self._near(n)
        if relation == "overlap":
            return self._overlap(n)
        if relation == "nested":
            return self._nested(n)
        if relation == "left_of":
            return self._left_of(n)
        if relation == "right_of":
            return self._right_of(n)
        if relation == "above":
            return self._above(n)
        if relation == "below":
            return self._below(n)
        if relation == "inside":
            return self._inside(n)
        if relation == "outside":
            return self._outside(n)
        
        # fallback to random placement
        return self._random_placement(n)

    def _left_of(self, n: int) -> List[Dict[str, Any]]:
        """Place objects in a row, left to right with proper spacing."""
        if n == 1:
            return [{"position": (self.img_size * 0.3, self.img_size * 0.5)}]
        
        spacing = (self.img_size - 2 * self.margin) / (n + 1)
        objects = []
        for i in range(n):
            x = self.margin + spacing * (i + 1)
            y = self.img_size * 0.5 + random.uniform(-self.img_size * 0.1, self.img_size * 0.1)
            objects.append({"position": (x, y)})
        return objects
    
    def _right_of(self, n: int) -> List[Dict[str, Any]]:
        """Place objects in a row, right to left."""
        objects = self._left_of(n)
        # Reverse the x coordinates
        for obj in objects:
            obj["position"] = (self.img_size - obj["position"][0], obj["position"][1])
        return objects

    def _above(self, n: int) -> List[Dict[str, Any]]:
        """Place objects in a column, top to bottom with proper spacing."""
        if n == 1:
            return [{"position": (self.img_size * 0.5, self.img_size * 0.3)}]
        
        spacing = (self.img_size - 2 * self.margin) / (n + 1)
        objects = []
        for i in range(n):
            y = self.margin + spacing * (i + 1)
            x = self.img_size * 0.5 + random.uniform(-self.img_size * 0.1, self.img_size * 0.1)
            objects.append({"position": (x, y)})
        return objects
    
    def _below(self, n: int) -> List[Dict[str, Any]]:
        """Place objects in a column, bottom to top."""
        objects = self._above(n)
        # Reverse the y coordinates
        for obj in objects:
            obj["position"] = (obj["position"][0], self.img_size - obj["position"][1])
        return objects

    def _nested(self, n: int) -> List[Dict[str, Any]]:
        """All objects at the same position for nesting."""
        center = (self.img_size // 2, self.img_size // 2)
        return [{"position": center} for _ in range(n)]

    def _overlap(self, n: int) -> List[Dict[str, Any]]:
        """Objects overlap at the center with slight variation."""
        center_x, center_y = self.img_size // 2, self.img_size // 2
        objects = []
        for i in range(n):
            # Small random offset for overlapping but not identical placement
            offset_x = random.uniform(-self.img_size * 0.05, self.img_size * 0.05)
            offset_y = random.uniform(-self.img_size * 0.05, self.img_size * 0.05)
            objects.append({
                "position": (center_x + offset_x, center_y + offset_y)
            })
        return objects

    def _near(self, n: int) -> List[Dict[str, Any]]:
        """Place objects close together in a cluster."""
        center_x, center_y = self.img_size // 2, self.img_size // 2
        cluster_radius = self.img_size * 0.15  # 15% of image size
        
        objects = []
        for i in range(n):
            angle = (2 * math.pi * i) / n if n > 1 else 0
            radius = random.uniform(0, cluster_radius)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Ensure position is within bounds
            x = max(self.margin, min(self.img_size - self.margin, x))
            y = max(self.margin, min(self.img_size - self.margin, y))
            
            objects.append({"position": (x, y)})
        return objects
    
    def _inside(self, n: int) -> List[Dict[str, Any]]:
        """Place smaller objects inside a larger container object."""
        if n < 2:
            return self._random_placement(n)
        
        # First object is the container (larger, centered)
        container = {
            "position": (self.img_size * 0.5, self.img_size * 0.5),
            "size": max(60, self.img_size * 0.3)  # Large container
        }
        
        # Remaining objects are inside the container
        objects = [container]
        container_radius = container["size"] * 0.3
        
        for i in range(1, n):
            angle = (2 * math.pi * i) / (n - 1) if n > 2 else 0
            radius = random.uniform(0, container_radius)
            x = container["position"][0] + radius * math.cos(angle)
            y = container["position"][1] + radius * math.sin(angle)
            
            objects.append({
                "position": (x, y),
                "size": random.randint(15, 30)  # Smaller objects inside
            })
        
        return objects
    
    def _outside(self, n: int) -> List[Dict[str, Any]]:
        """Place objects outside a central region."""
        if n < 1:
            return []
        
        objects = []
        center_x, center_y = self.img_size // 2, self.img_size // 2
        exclusion_radius = self.img_size * 0.2  # Central exclusion zone
        
        for i in range(n):
            # Place in outer ring
            angle = (2 * math.pi * i) / n if n > 1 else 0
            radius = random.uniform(exclusion_radius + self.img_size * 0.1, 
                                  self.img_size * 0.4)
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            
            # Ensure position is within bounds
            x = max(self.margin, min(self.img_size - self.margin, x))
            y = max(self.margin, min(self.img_size - self.margin, y))
            
            objects.append({"position": (x, y)})
        
        return objects
    
    def _random_placement(self, n: int) -> List[Dict[str, Any]]:
        """Fallback random placement."""
        objects = []
        for _ in range(n):
            x = random.uniform(self.margin, self.img_size - self.margin)
            y = random.uniform(self.margin, self.img_size - self.margin)
            objects.append({"position": (x, y)})
        return objects
        return [
            {"position": (base[0] + i * 5, base[1] + i * 5)} for i in range(n)
        ]
