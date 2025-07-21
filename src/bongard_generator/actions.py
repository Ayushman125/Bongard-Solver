"""
Action primitives for generating diverse freeform shapes in Bongard-LOGO style.
These actions enable creating arcs, zigzags, fans, and other complex patterns.
"""

import math
import random
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Action:
    """Base class for all shape-generating actions."""
    
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the action on the given ImageDraw context.
        
        Args:
            draw: PIL ImageDraw context
            center: (x, y) center position
            size: Size parameter for scaling
            **kwargs: Additional drawing parameters
        """
        # Default implementation draws a simple circle
        x, y = center
        radius = size // 4
        color = kwargs.get('color', 'black')
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=color, width=2)

class ArcAction(Action):
    """Draws an arc segment with specified parameters."""
    
    def __init__(self, radius_ratio: float = 0.5, start_deg: float = 0, 
                 end_deg: float = 180, stroke_width: int = 2):
        """Initialize arc parameters.
        
        Args:
            radius_ratio: Radius as fraction of size (0.1-1.0)
            start_deg: Starting angle in degrees
            end_deg: Ending angle in degrees  
            stroke_width: Line thickness
        """
        self.radius_ratio = max(0.1, min(1.0, radius_ratio))
        self.start_deg = start_deg % 360
        self.end_deg = end_deg % 360
        self.stroke_width = stroke_width
        
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the arc."""
        x, y = center
        radius = int(size * self.radius_ratio / 2)
        
        bbox = [x - radius, y - radius, x + radius, y + radius]
        
        try:
            draw.arc(bbox, start=self.start_deg, end=self.end_deg, 
                    fill="black", width=self.stroke_width)
        except Exception as e:
            logger.warning(f"Arc drawing failed: {e}")

class ZigzagAction(Action):
    """Draws a zigzag line with specified parameters."""
    
    def __init__(self, peaks: int = 3, amplitude_ratio: float = 0.3,
                 length_ratio: float = 0.8, stroke_width: int = 2):
        """Initialize zigzag parameters.
        
        Args:
            peaks: Number of peaks in zigzag (2-8)
            amplitude_ratio: Height as fraction of size
            length_ratio: Width as fraction of size
            stroke_width: Line thickness
        """
        self.peaks = max(2, min(8, peaks))
        self.amplitude_ratio = max(0.1, min(0.5, amplitude_ratio))
        self.length_ratio = max(0.3, min(1.0, length_ratio))
        self.stroke_width = stroke_width
        
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the zigzag."""
        x, y = center
        length = int(size * self.length_ratio)
        amplitude = int(size * self.amplitude_ratio)
        
        # Generate zigzag points
        points = []
        step = length / self.peaks
        start_x = x - length // 2
        
        for i in range(self.peaks + 1):
            xi = start_x + i * step
            # Add jitter to make it more organic
            jitter_x = random.uniform(-2, 2)
            jitter_y = random.uniform(-2, 2)
            
            yi = y + ((-1)**i) * amplitude + jitter_y
            points.append((xi + jitter_x, yi))
            
        try:
            if len(points) > 1:
                draw.line(points, fill="black", width=self.stroke_width)
        except Exception as e:
            logger.warning(f"Zigzag drawing failed: {e}")

class FanAction(Action):
    """Draws a fan pattern with radiating lines."""
    
    def __init__(self, spokes: int = 5, spread_deg: float = 90, 
                 radius_ratio: float = 0.6, stroke_width: int = 2):
        """Initialize fan parameters.
        
        Args:
            spokes: Number of radiating lines (3-10)
            spread_deg: Total angle spread in degrees
            radius_ratio: Length as fraction of size
            stroke_width: Line thickness
        """
        self.spokes = max(3, min(10, spokes))
        self.spread_deg = max(30, min(180, spread_deg))
        self.radius_ratio = max(0.2, min(1.0, radius_ratio))
        self.stroke_width = stroke_width
        
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the fan."""
        cx, cy = center
        radius = int(size * self.radius_ratio / 2)
        
        # Calculate angle step
        if self.spokes == 1:
            angles = [0]
        else:
            base_angle = -self.spread_deg / 2
            angle_step = self.spread_deg / (self.spokes - 1)
            angles = [base_angle + i * angle_step for i in range(self.spokes)]
        
        try:
            for angle in angles:
                # Add small random variation to each spoke
                angle_var = angle + random.uniform(-5, 5)
                theta = math.radians(angle_var)
                
                # Add slight length variation
                r_var = radius * random.uniform(0.8, 1.0)
                
                x1 = cx + r_var * math.cos(theta)
                y1 = cy + r_var * math.sin(theta)
                
                draw.line([(cx, cy), (x1, y1)], fill="black", width=self.stroke_width)
        except Exception as e:
            logger.warning(f"Fan drawing failed: {e}")

class SpiralAction(Action):
    """Draws a spiral shape."""
    
    def __init__(self, turns: float = 2.0, radius_ratio: float = 0.5,
                 stroke_width: int = 2):
        """Initialize spiral parameters.
        
        Args:
            turns: Number of full turns
            radius_ratio: Maximum radius as fraction of size
            stroke_width: Line thickness
        """
        self.turns = max(0.5, min(5.0, turns))
        self.radius_ratio = max(0.2, min(1.0, radius_ratio))
        self.stroke_width = stroke_width
        
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the spiral."""
        cx, cy = center
        max_radius = int(size * self.radius_ratio / 2)
        
        points = []
        steps = int(self.turns * 20)  # Resolution
        
        try:
            for i in range(steps):
                t = i / steps
                angle = t * self.turns * 2 * math.pi
                radius = t * max_radius
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill="black", width=self.stroke_width)
        except Exception as e:
            logger.warning(f"Spiral drawing failed: {e}")

class BumpyLineAction(Action):
    """Draws a bumpy/wavy line using sine waves."""
    
    def __init__(self, wavelength: float = 20, amplitude_ratio: float = 0.2,
                 length_ratio: float = 0.8, stroke_width: int = 2):
        """Initialize bumpy line parameters.
        
        Args:
            wavelength: Distance between wave peaks
            amplitude_ratio: Wave height as fraction of size
            length_ratio: Total length as fraction of size
            stroke_width: Line thickness
        """
        self.wavelength = max(5, min(50, wavelength))
        self.amplitude_ratio = max(0.1, min(0.4, amplitude_ratio))
        self.length_ratio = max(0.3, min(1.0, length_ratio))
        self.stroke_width = stroke_width
        
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the bumpy line."""
        x, y = center
        length = int(size * self.length_ratio)
        amplitude = int(size * self.amplitude_ratio)
        
        points = []
        steps = max(20, length // 2)
        start_x = x - length // 2
        
        try:
            for i in range(steps):
                xi = start_x + (i / steps) * length
                # Sine wave with noise
                wave = math.sin(2 * math.pi * xi / self.wavelength)
                noise = random.uniform(-0.3, 0.3)
                yi = y + amplitude * (wave + noise)
                
                points.append((xi, yi))
            
            if len(points) > 1:
                draw.line(points, fill="black", width=self.stroke_width)
        except Exception as e:
            logger.warning(f"Bumpy line drawing failed: {e}")

class CrossAction(Action):
    """Draws intersecting lines forming a cross or X pattern."""
    
    def __init__(self, style: str = "plus", size_ratio: float = 0.7,
                 stroke_width: int = 2):
        """Initialize cross parameters.
        
        Args:
            style: "plus" for +, "x" for X pattern
            size_ratio: Size as fraction of bounding box
            stroke_width: Line thickness
        """
        self.style = style
        self.size_ratio = max(0.3, min(1.0, size_ratio))
        self.stroke_width = stroke_width
        
    def draw(self, draw: ImageDraw.Draw, center: Tuple[int, int], 
             size: int, **kwargs) -> None:
        """Draw the cross."""
        cx, cy = center
        half_size = int(size * self.size_ratio / 2)
        
        try:
            if self.style == "plus":
                # Horizontal line
                draw.line([(cx - half_size, cy), (cx + half_size, cy)], 
                         fill="black", width=self.stroke_width)
                # Vertical line  
                draw.line([(cx, cy - half_size), (cx, cy + half_size)], 
                         fill="black", width=self.stroke_width)
            else:  # X pattern
                # Diagonal lines
                draw.line([(cx - half_size, cy - half_size), 
                          (cx + half_size, cy + half_size)], 
                         fill="black", width=self.stroke_width)
                draw.line([(cx - half_size, cy + half_size), 
                          (cx + half_size, cy - half_size)], 
                         fill="black", width=self.stroke_width)
        except Exception as e:
            logger.warning(f"Cross drawing failed: {e}")

# Factory function for creating random actions
def create_random_action(action_type: str = None) -> Action:
    """Create a random action of the specified type.
    
    Args:
        action_type: Type of action ("arc", "zigzag", "fan", etc.) 
                    or None for random selection
    
    Returns:
        Action instance
    """
    action_types = {
        "arc": lambda: ArcAction(
            radius_ratio=random.uniform(0.3, 0.8),
            start_deg=random.uniform(0, 180),
            end_deg=random.uniform(180, 360),
            stroke_width=random.randint(1, 4)
        ),
        "zigzag": lambda: ZigzagAction(
            peaks=random.randint(3, 6),
            amplitude_ratio=random.uniform(0.2, 0.4),
            length_ratio=random.uniform(0.6, 0.9),
            stroke_width=random.randint(1, 3)
        ),
        "fan": lambda: FanAction(
            spokes=random.randint(4, 8),
            spread_deg=random.uniform(60, 120),
            radius_ratio=random.uniform(0.4, 0.7),
            stroke_width=random.randint(1, 3)
        ),
        "spiral": lambda: SpiralAction(
            turns=random.uniform(1.5, 3.0),
            radius_ratio=random.uniform(0.3, 0.6),
            stroke_width=random.randint(1, 3)
        ),
        "bumpy": lambda: BumpyLineAction(
            wavelength=random.uniform(15, 30),
            amplitude_ratio=random.uniform(0.15, 0.3),
            length_ratio=random.uniform(0.7, 0.9),
            stroke_width=random.randint(1, 3)
        ),
        "cross": lambda: CrossAction(
            style=random.choice(["plus", "x"]),
            size_ratio=random.uniform(0.5, 0.8),
            stroke_width=random.randint(1, 4)
        )
    }
    
    if action_type is None:
        action_type = random.choice(list(action_types.keys()))
    
    if action_type in action_types:
        return action_types[action_type]()
    else:
        # Fallback to arc
        return action_types["arc"]()

def create_action_sequence(num_actions: int = None) -> List[Action]:
    """Create a sequence of actions for complex freeform shapes.
    
    Args:
        num_actions: Number of actions in sequence, or None for random (2-4)
    
    Returns:
        List of Action instances
    """
    if num_actions is None:
        num_actions = random.randint(2, 4)
    
    return [create_random_action() for _ in range(num_actions)]
