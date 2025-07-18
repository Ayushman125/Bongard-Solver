"""Mask utilities for advanced shape drawing and effects."""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, List, Optional, Dict, Any
import math
import random

def create_shape_mask(shape: str, size: int, hr_factor: int = 2) -> Image.Image:
    """
    Create a high-resolution mask for a given shape.
    
    Args:
        shape: Shape type ('circle', 'triangle', 'square', 'pentagon', 'star')
        size: Target size of the mask
        hr_factor: High-resolution factor for better quality
        
    Returns:
        PIL Image mask in L mode
    """
    hr_size = size * hr_factor
    mask = Image.new("L", (hr_size, hr_size), 0)
    draw = ImageDraw.Draw(mask)
    
    center = hr_size // 2
    radius = hr_size // 2 - 5  # Small margin
    
    if shape == 'circle':
        draw.ellipse([center - radius, center - radius, 
                     center + radius, center + radius], fill=255)
    
    elif shape == 'square':
        draw.rectangle([center - radius, center - radius,
                       center + radius, center + radius], fill=255)
    
    elif shape == 'triangle':
        points = [
            (center, center - radius),  # Top
            (center - radius * 0.866, center + radius * 0.5),  # Bottom left
            (center + radius * 0.866, center + radius * 0.5),  # Bottom right
        ]
        draw.polygon(points, fill=255)
    
    elif shape == 'pentagon':
        points = []
        for i in range(5):
            angle = (i * 2 * math.pi) / 5 - math.pi / 2  # Start from top
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=255)
    
    elif shape == 'star':
        points = []
        for i in range(10):  # 5 outer + 5 inner points
            angle = (i * math.pi) / 5 - math.pi / 2  # Start from top
            if i % 2 == 0:  # Outer points
                r = radius
            else:  # Inner points
                r = radius * 0.4
            x = center + r * math.cos(angle)
            y = center + r * math.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=255)
    
    else:
        # Default to circle for unknown shapes
        draw.ellipse([center - radius, center - radius, 
                     center + radius, center + radius], fill=255)
    
    # Resize back to target size with anti-aliasing
    return mask.resize((size, size), Image.Resampling.LANCZOS)

def apply_fill_to_mask(mask: Image.Image, 
                      fill_type: str, 
                      color: Tuple[int, int, int, int],
                      **kwargs) -> Image.Image:
    """
    Apply different fill types to a mask.
    
    Args:
        mask: L mode mask image
        fill_type: Type of fill ('solid', 'outline', 'striped', 'gradient', 'noise')
        color: RGBA color tuple
        **kwargs: Additional parameters for specific fill types
        
    Returns:
        RGBA image with the applied fill
    """
    size = mask.size
    
    if fill_type == 'solid':
        return _apply_solid_fill(mask, color)
    
    elif fill_type == 'outline':
        return _apply_outline_fill(mask, color, kwargs.get('thickness', 3))
    
    elif fill_type == 'striped':
        return _apply_striped_fill(mask, color, 
                                 kwargs.get('stripe_width', 4),
                                 kwargs.get('stripe_angle', 45))
    
    elif fill_type == 'gradient':
        return _apply_gradient_fill(mask, color, 
                                  kwargs.get('direction', 'vertical'))
    
    elif fill_type == 'noise':
        return _apply_noise_fill(mask, color, 
                               kwargs.get('noise_scale', 0.3))
    
    else:
        # Default to solid fill
        return _apply_solid_fill(mask, color)

def _apply_solid_fill(mask: Image.Image, color: Tuple[int, int, int, int]) -> Image.Image:
    """Apply solid color fill."""
    result = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    solid = Image.new('RGBA', mask.size, color)
    solid.putalpha(mask)
    return solid

def _apply_outline_fill(mask: Image.Image, color: Tuple[int, int, int, int], thickness: int) -> Image.Image:
    """Apply outline fill by creating an outline mask."""
    from PIL import ImageFilter
    
    # Create outline by subtracting eroded mask from original
    eroded = mask.filter(ImageFilter.MinFilter(thickness * 2 + 1))
    outline_mask = Image.new('L', mask.size)
    
    for i in range(mask.width):
        for j in range(mask.height):
            orig = mask.getpixel((i, j))
            eroded_val = eroded.getpixel((i, j))
            outline_mask.putpixel((i, j), max(0, orig - eroded_val))
    
    result = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    outline_color = Image.new('RGBA', mask.size, color)
    outline_color.putalpha(outline_mask)
    return outline_color

def _apply_striped_fill(mask: Image.Image, 
                       color: Tuple[int, int, int, int], 
                       stripe_width: int, 
                       angle: float) -> Image.Image:
    """Apply striped fill pattern."""
    stripe_mask = Image.new('L', mask.size, 0)
    draw = ImageDraw.Draw(stripe_mask)
    
    # Calculate stripe lines based on angle
    width, height = mask.size
    angle_rad = math.radians(angle)
    
    # Create stripes
    if abs(angle_rad) < math.pi / 4:  # More horizontal
        step = stripe_width * 2
        for y in range(0, height + width, step):
            x1 = 0
            y1 = y
            x2 = width
            y2 = y - width * math.tan(angle_rad)
            draw.line([(x1, y1), (x2, y2)], fill=255, width=stripe_width)
    else:  # More vertical
        step = stripe_width * 2
        for x in range(0, width + height, step):
            x1 = x
            y1 = 0
            x2 = x - height / math.tan(angle_rad)
            y2 = height
            draw.line([(x1, y1), (x2, y2)], fill=255, width=stripe_width)
    
    # Combine with original mask
    combined_mask = Image.new('L', mask.size)
    for i in range(mask.width):
        for j in range(mask.height):
            orig = mask.getpixel((i, j))
            stripe = stripe_mask.getpixel((i, j))
            combined_mask.putpixel((i, j), min(orig, stripe))
    
    result = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    striped_color = Image.new('RGBA', mask.size, color)
    striped_color.putalpha(combined_mask)
    return striped_color

def _apply_gradient_fill(mask: Image.Image, 
                        color: Tuple[int, int, int, int], 
                        direction: str) -> Image.Image:
    """Apply gradient fill."""
    from .draw_utils import make_gradient_mask
    
    gradient_mask = make_gradient_mask(mask.width, vertical=(direction == 'vertical'))
    
    # Combine gradient with shape mask
    combined_mask = Image.new('L', mask.size)
    for i in range(mask.width):
        for j in range(mask.height):
            shape_alpha = mask.getpixel((i, j))
            grad_alpha = gradient_mask.getpixel((i, j))
            # Multiply alpha values
            combined_alpha = int((shape_alpha * grad_alpha) / 255)
            combined_mask.putpixel((i, j), combined_alpha)
    
    result = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    gradient_color = Image.new('RGBA', mask.size, color)
    gradient_color.putalpha(combined_mask)
    return gradient_color

def _apply_noise_fill(mask: Image.Image, 
                     color: Tuple[int, int, int, int], 
                     noise_scale: float) -> Image.Image:
    """Apply noise texture fill."""
    # Generate noise
    noise_array = np.random.random(mask.size[::-1])  # PIL uses (width, height), numpy uses (height, width)
    noise_array = (noise_array * 255).astype(np.uint8)
    noise_mask = Image.fromarray(noise_array, mode='L')
    
    # Scale noise
    noise_mask = noise_mask.point(lambda x: int(x * noise_scale + 255 * (1 - noise_scale)))
    
    # Combine with shape mask
    combined_mask = Image.new('L', mask.size)
    for i in range(mask.width):
        for j in range(mask.height):
            shape_alpha = mask.getpixel((i, j))
            noise_alpha = noise_mask.getpixel((i, j))
            combined_alpha = int((shape_alpha * noise_alpha) / 255)
            combined_mask.putpixel((i, j), combined_alpha)
    
    result = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    noise_color = Image.new('RGBA', mask.size, color)
    noise_color.putalpha(combined_mask)
    return noise_color

def create_composite_scene(objects: List[Dict[str, Any]], 
                          canvas_size: int, 
                          background_color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    """
    Create a composite scene from a list of objects.
    
    Args:
        objects: List of object dictionaries with position, shape, fill, color, etc.
        canvas_size: Size of the output canvas
        background_color: Background color
        
    Returns:
        Composite RGBA image
    """
    canvas = Image.new('RGBA', (canvas_size, canvas_size), background_color)
    
    # Sort objects by size (largest first) for better layering
    sorted_objects = sorted(objects, key=lambda obj: obj.get('size', 30), reverse=True)
    
    for obj in sorted_objects:
        # Get object properties
        x, y = obj['position']
        size = obj.get('size', 30)
        shape = obj.get('shape', 'circle')
        fill_type = obj.get('fill', 'solid')
        color_name = obj.get('color', 'blue')
        
        # Convert color name to RGBA
        color_rgba = _get_color_rgba(color_name)
        
        # Create shape mask
        shape_mask = create_shape_mask(shape, size)
        
        # Apply fill
        filled_shape = apply_fill_to_mask(shape_mask, fill_type, color_rgba)
        
        # Calculate paste position (center the shape on the given position)
        paste_x = int(x - size // 2)
        paste_y = int(y - size // 2)
        
        # Ensure the shape fits within canvas bounds
        paste_x = max(0, min(paste_x, canvas_size - size))
        paste_y = max(0, min(paste_y, canvas_size - size))
        
        # Paste onto canvas
        try:
            canvas.paste(filled_shape, (paste_x, paste_y), filled_shape)
        except Exception as e:
            # If pasting fails, skip this object
            print(f"Warning: Failed to paste object {obj}: {e}")
            continue
    
    return canvas

def _get_color_rgba(color_name: str) -> Tuple[int, int, int, int]:
    """Convert color name to RGBA tuple."""
    color_map = {
        'red': (255, 0, 0, 255),
        'blue': (0, 0, 255, 255),
        'green': (0, 255, 0, 255),
        'yellow': (255, 255, 0, 255),
        'purple': (128, 0, 128, 255),
        'orange': (255, 165, 0, 255),
        'black': (0, 0, 0, 255),
        'white': (255, 255, 255, 255),
        'gray': (128, 128, 128, 255),
        'cyan': (0, 255, 255, 255),
        'magenta': (255, 0, 255, 255),
    }
    return color_map.get(color_name.lower(), (128, 128, 128, 255))  # Default to gray
