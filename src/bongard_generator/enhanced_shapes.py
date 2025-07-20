"""
Enhanced shape generators for nested, concave, and complex polygons.
Implements all missing Bongard-LOGO shape variations.
"""

import math
import random
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


def regular_polygon(cx: float, cy: float, size: float, sides: int) -> List[Tuple[float, float]]:
    """Generate vertices for a regular polygon."""
    radius = size / 2
    vertices = []
    angle_step = 2 * math.pi / sides
    
    for i in range(sides):
        angle = i * angle_step - math.pi / 2  # Start from top
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y))
    
    return vertices


def star_polygon(cx: float, cy: float, size: float, points: int, inner_ratio: float = 0.5) -> List[Tuple[float, float]]:
    """Generate vertices for a star polygon (concave)."""
    outer_radius = size / 2
    inner_radius = outer_radius * inner_ratio
    vertices = []
    angle_step = math.pi / points  # Half step for alternating inner/outer
    
    for i in range(points * 2):
        angle = i * angle_step - math.pi / 2
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y))
    
    return vertices


def create_nested_shape(draw: ImageDraw.Draw, cx: int, cy: int, size: int, 
                       fill_color: Any = 255, outline_color: Any = 0, stroke_width: int = 2):
    """Create nested shape (shape within shape)."""
    outer_radius = size // 2
    inner_radius = size // 4
    
    # Draw outer circle
    draw.ellipse([cx - outer_radius, cy - outer_radius, 
                  cx + outer_radius, cy + outer_radius], 
                 fill=fill_color, outline=outline_color, width=stroke_width)
    
    # Draw inner circle (creates hole)
    draw.ellipse([cx - inner_radius, cy - inner_radius, 
                  cx + inner_radius, cy + inner_radius], 
                 fill=0, outline=outline_color, width=stroke_width)


def create_concave_shape(draw: ImageDraw.Draw, cx: int, cy: int, size: int, 
                        fill_color: Any = 255, outline_color: Any = 0, stroke_width: int = 2):
    """Create concave shape (star with inward points)."""
    vertices = star_polygon(cx, cy, size, 5, 0.4)  # 5-point star, deep concavity
    draw.polygon(vertices, fill=fill_color, outline=outline_color, width=stroke_width)


def create_convex_shape(draw: ImageDraw.Draw, cx: int, cy: int, size: int, 
                       fill_color: Any = 255, outline_color: Any = 0, stroke_width: int = 2):
    """Create convex shape (regular polygon)."""
    sides = random.choice([5, 6, 7, 8])  # Pentagon to octagon
    vertices = regular_polygon(cx, cy, size, sides)
    draw.polygon(vertices, fill=fill_color, outline=outline_color, width=stroke_width)


def draw_dashed_line(draw: ImageDraw.Draw, start: Tuple[int, int], end: Tuple[int, int], 
                    dash_len: int = 5, gap_len: int = 3, color: Any = 0, width: int = 2):
    """Draw a dashed line between two points."""
    x1, y1 = start
    x2, y2 = end
    
    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    
    if length == 0:
        return
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Draw dashed line
    current_pos = 0
    dash_on = True
    
    while current_pos < length:
        segment_length = dash_len if dash_on else gap_len
        end_pos = min(current_pos + segment_length, length)
        
        if dash_on:
            seg_x1 = x1 + dx_norm * current_pos
            seg_y1 = y1 + dy_norm * current_pos
            seg_x2 = x1 + dx_norm * end_pos
            seg_y2 = y1 + dy_norm * end_pos
            
            draw.line([(seg_x1, seg_y1), (seg_x2, seg_y2)], fill=color, width=width)
        
        current_pos = end_pos
        dash_on = not dash_on


def draw_dotted_line(draw: ImageDraw.Draw, start: Tuple[int, int], end: Tuple[int, int], 
                    dot_spacing: int = 8, dot_size: int = 2, color: Any = 0):
    """Draw a dotted line between two points."""
    x1, y1 = start
    x2, y2 = end
    
    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    
    if length == 0:
        return
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Draw dots along the line
    num_dots = int(length / dot_spacing) + 1
    for i in range(num_dots):
        pos = i * dot_spacing
        if pos > length:
            break
            
        dot_x = x1 + dx_norm * pos
        dot_y = y1 + dy_norm * pos
        
        # Draw dot as small circle
        draw.ellipse([dot_x - dot_size, dot_y - dot_size, 
                      dot_x + dot_size, dot_y + dot_size], fill=color)


def draw_dashed_polygon(draw: ImageDraw.Draw, vertices: List[Tuple[float, float]], 
                       dash_len: int = 5, gap_len: int = 3, color: Any = 0, width: int = 2):
    """Draw a polygon with dashed outline."""
    if len(vertices) < 3:
        return
    
    # Draw dashed lines between consecutive vertices
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)]
        draw_dashed_line(draw, start, end, dash_len, gap_len, color, width)


def draw_dotted_polygon(draw: ImageDraw.Draw, vertices: List[Tuple[float, float]], 
                       dot_spacing: int = 8, dot_size: int = 2, color: Any = 0):
    """Draw a polygon with dotted outline."""
    if len(vertices) < 3:
        return
    
    # Draw dotted lines between consecutive vertices
    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)]
        draw_dotted_line(draw, start, end, dot_spacing, dot_size, color)


def create_complex_shape(draw: ImageDraw.Draw, cx: int, cy: int, size: int, 
                        shape_type: str, stroke_style: str = 'solid', 
                        fill_color: Any = 255, outline_color: Any = 0, stroke_width: int = 2):
    """Create complex shapes with various stroke styles."""
    
    if shape_type == 'nested':
        create_nested_shape(draw, cx, cy, size, fill_color, outline_color, stroke_width)
    
    elif shape_type == 'concave':
        if stroke_style == 'dashed':
            vertices = star_polygon(cx, cy, size, 5, 0.4)
            # Fill first
            draw.polygon(vertices, fill=fill_color)
            # Then dashed outline
            draw_dashed_polygon(draw, vertices, color=outline_color, width=stroke_width)
        elif stroke_style == 'dotted':
            vertices = star_polygon(cx, cy, size, 5, 0.4)
            draw.polygon(vertices, fill=fill_color)
            draw_dotted_polygon(draw, vertices, color=outline_color)
        else:
            create_concave_shape(draw, cx, cy, size, fill_color, outline_color, stroke_width)
    
    elif shape_type == 'convex':
        sides = random.choice([5, 6, 7, 8])
        vertices = regular_polygon(cx, cy, size, sides)
        
        if stroke_style == 'dashed':
            draw.polygon(vertices, fill=fill_color)
            draw_dashed_polygon(draw, vertices, color=outline_color, width=stroke_width)
        elif stroke_style == 'dotted':
            draw.polygon(vertices, fill=fill_color)
            draw_dotted_polygon(draw, vertices, color=outline_color)
        else:
            draw.polygon(vertices, fill=fill_color, outline=outline_color, width=stroke_width)
    
    elif shape_type == 'irregular':
        # Create irregular polygon
        vertices = []
        num_points = random.randint(6, 10)
        angle_step = 2 * math.pi / num_points
        
        for i in range(num_points):
            angle = i * angle_step
            # Add randomness to radius
            radius_variation = random.uniform(0.7, 1.3)
            radius = (size / 2) * radius_variation
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append((x, y))
        
        if stroke_style == 'dashed':
            draw.polygon(vertices, fill=fill_color)
            draw_dashed_polygon(draw, vertices, color=outline_color, width=stroke_width)
        elif stroke_style == 'dotted':
            draw.polygon(vertices, fill=fill_color)
            draw_dotted_polygon(draw, vertices, color=outline_color)
        else:
            draw.polygon(vertices, fill=fill_color, outline=outline_color, width=stroke_width)


def apply_background_texture(img: Image.Image, noise_level: float = 0.1, 
                           noise_opacity: float = 0.1) -> Image.Image:
    """Apply procedural texture to background."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Create noise texture
    noise = Image.effect_noise(img.size, int(noise_level * 255)).convert('L')
    
    # Convert noise to RGB and blend
    noise_rgb = noise.convert('RGB')
    
    # Apply opacity
    opacity = int(noise_opacity * 255)
    
    # Create alpha mask
    alpha = Image.new('L', img.size, opacity)
    
    # Blend with original image
    textured = Image.composite(noise_rgb, img, alpha)
    
    return textured


def create_checker_fill(size: Tuple[int, int], checker_size: int = 8, 
                       color1: Tuple[int, int, int] = (255, 255, 255),
                       color2: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Create a checkerboard pattern fill."""
    width, height = size
    img = Image.new('RGB', size, color1)
    draw = ImageDraw.Draw(img)
    
    for y in range(0, height, checker_size):
        for x in range(0, width, checker_size):
            # Determine checker color based on grid position
            checker_x = x // checker_size
            checker_y = y // checker_size
            
            if (checker_x + checker_y) % 2 == 1:
                draw.rectangle([x, y, x + checker_size, y + checker_size], fill=color2)
    
    return img


def create_gradient_fill(size: Tuple[int, int], direction: str = 'horizontal',
                        color1: Tuple[int, int, int] = (255, 255, 255),
                        color2: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Create a gradient fill pattern."""
    width, height = size
    img = Image.new('RGB', size)
    
    if direction == 'horizontal':
        for x in range(width):
            ratio = x / width
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            
            for y in range(height):
                img.putpixel((x, y), (r, g, b))
    
    elif direction == 'vertical':
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            
            for x in range(width):
                img.putpixel((x, y), (r, g, b))
    
    elif direction == 'radial':
        cx, cy = width // 2, height // 2
        max_radius = math.sqrt(cx*cx + cy*cy)
        
        for y in range(height):
            for x in range(width):
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                ratio = min(distance / max_radius, 1.0)
                
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                
                img.putpixel((x, y), (r, g, b))
    
    return img


# Integration functions for the main dataset system
def get_extended_shape_types() -> List[str]:
    """Get all available shape types including complex ones."""
    return [
        # Basic shapes
        'circle', 'rectangle', 'triangle', 'ellipse', 'polygon',
        # Action-based shapes
        'arc', 'zigzag', 'fan', 'spiral', 'bumpy_line', 'cross',
        # Complex shapes
        'nested', 'concave', 'convex', 'irregular'
    ]


def get_stroke_styles() -> List[str]:
    """Get all available stroke styles."""
    return ['solid', 'dashed', 'dotted']


def get_fill_patterns() -> List[str]:
    """Get all available fill patterns."""
    return ['solid', 'striped', 'dotted', 'gradient', 'checker', 'noise']
