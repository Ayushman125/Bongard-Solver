"""
Professional Shape Renderer for Bongard Problems
This module provides comprehensive shape rendering with support for:
- Basic primitives: circle, square, triangle, pentagon, star
- Fill types: solid, hollow, striped, dotted  
- Stroke styles: solid, dashed, dotted
- Transformations: rotation, jitter, flipping
"""

import math
import random
from PIL import Image, ImageDraw
from typing import Dict, Any, Tuple, List

def _jitter_pt(pt: Tuple[float, float], jitter_px: float) -> Tuple[float, float]:
    """Apply random jitter to a point."""
    x, y = pt
    return (x + random.uniform(-jitter_px, jitter_px),
            y + random.uniform(-jitter_px, jitter_px))

def _rotate_pts(pts: List[Tuple[float, float]], center: Tuple[float, float], angle_deg: float) -> List[Tuple[float, float]]:
    """Rotate points around a center by angle_deg degrees."""
    cx, cy = center
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        rx = cx + dx * cos_t - dy * sin_t
        ry = cy + dx * sin_t + dy * cos_t
        out.append((rx, ry))
    return out

def _draw_dashed_line(draw: ImageDraw.Draw, p0: Tuple[float, float], p1: Tuple[float, float], 
                     width: int, dash_len: float, gap_len: float):
    """Draw a dashed line between two points."""
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist == 0:
        return
    
    dx, dy = (x1 - x0) / dist, (y1 - y0) / dist
    t = 0
    on = True
    
    while t < dist:
        end = min(t + (dash_len if on else gap_len), dist)
        sx, sy = x0 + dx * t, y0 + dy * t
        ex, ey = x0 + dx * end, y0 + dy * end
        if on:
            draw.line([(sx, sy), (ex, ey)], fill='black', width=width)
        t += (dash_len if on else gap_len)
        on = not on

def draw_shape(draw: ImageDraw.Draw, obj: Dict[str, Any], cfg: Any):
    """
    Renders a single shape according to obj['shape'], obj['fill'], etc.
    
    Args:
        draw: PIL ImageDraw object
        obj: Object dictionary with shape properties
        cfg: Configuration object with rendering settings
    """
    # Extract object properties with defaults
    x = obj.get('center_x', obj.get('x', 128))
    y = obj.get('center_y', obj.get('y', 128))
    s = obj.get('size', 30)
    
    # Convert string sizes to numeric
    if isinstance(s, str):
        size_map = {'small': 20, 'medium': 35, 'large': 50}
        s = size_map.get(s, 30)
    
    shape = obj.get('shape', 'circle')
    fill_type = obj.get('fill', 'solid')
    stroke = obj.get('stroke_width', getattr(cfg, 'stroke_width', 2))
    stroke_style = obj.get('stroke_style', 'solid')
    rotation = obj.get('rotation', 0)
    
    # Configuration parameters with defaults
    jitter = getattr(cfg, 'jitter_px', 2.0)
    hatch_gap = getattr(cfg, 'hatch_gap', 4)
    dash_len = getattr(cfg, 'dash_len', 5)
    dash_gap = getattr(cfg, 'dash_gap', 3)
    canvas_size = getattr(cfg, 'img_size', 256)
    
    # Apply jitter if enabled
    if getattr(cfg, 'enable_jitter', True):
        x += random.uniform(-jitter, jitter)
        y += random.uniform(-jitter, jitter)
    
    # Draw based on shape type
    if shape == 'circle':
        bbox = [x - s/2, y - s/2, x + s/2, y + s/2]
        
        if fill_type == 'solid':
            draw.ellipse(bbox, fill='black', outline=None)
        elif fill_type == 'hollow':
            draw.ellipse(bbox, outline='black', width=stroke, fill=None)
        elif fill_type == 'striped':
            # Create mask for circle area
            mask = Image.new('L', (canvas_size, canvas_size), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.ellipse(bbox, fill=255)
            
            # Draw horizontal stripes within circle
            for yy in range(int(y - s/2), int(y + s/2), hatch_gap):
                for xx in range(int(x - s/2), int(x + s/2)):
                    if xx < canvas_size and yy < canvas_size and mask.getpixel((xx, yy)) > 0:
                        draw.point((xx, yy), fill='black')
            
            # Add outline
            draw.ellipse(bbox, outline='black', width=stroke)
        elif fill_type == 'dotted':
            # Fill with dots
            for xx in range(int(x - s/2), int(x + s/2), hatch_gap):
                for yy in range(int(y - s/2), int(y + s/2), hatch_gap):
                    if (xx - x)**2 + (yy - y)**2 <= (s/2)**2:  # Inside circle
                        draw.ellipse((xx-1, yy-1, xx+1, yy+1), fill='black')
            draw.ellipse(bbox, outline='black', width=stroke)

    elif shape in ('square', 'triangle', 'pentagon', 'star'):
        # Generate base points
        if shape == 'square':
            base_pts = [(x - s/2, y - s/2), (x + s/2, y - s/2), 
                       (x + s/2, y + s/2), (x - s/2, y + s/2)]
        elif shape == 'triangle':
            base_pts = []
            for i in range(3):
                theta = math.radians(-90 + i * 120)  # Point up
                base_pts.append((x + s/2 * math.cos(theta),
                               y + s/2 * math.sin(theta)))
        elif shape == 'pentagon':
            base_pts = []
            for i in range(5):
                theta = math.radians(-90 + i * 72)  # Point up
                base_pts.append((x + s/2 * math.cos(theta),
                               y + s/2 * math.sin(theta)))
        else:  # star
            base_pts = []
            for i in range(10):
                r = s/2 if i % 2 == 0 else s/4
                theta = math.radians(-90 + i * 36)  # Point up
                base_pts.append((x + r * math.cos(theta),
                               y + r * math.sin(theta)))
        
        # Apply jitter and rotation
        if getattr(cfg, 'enable_jitter', True):
            pts = [_jitter_pt(p, jitter) for p in base_pts]
        else:
            pts = base_pts
            
        if getattr(cfg, 'enable_rotation', True) and rotation != 0:
            pts = _rotate_pts(pts, (x, y), rotation)
        
        # Fill the shape
        if fill_type == 'solid':
            draw.polygon(pts, fill='black')
        elif fill_type == 'hollow':
            draw.polygon(pts, outline='black', width=stroke, fill=None)
        elif fill_type == 'striped':
            # Create mask then fill with stripes
            mask = Image.new('L', (canvas_size, canvas_size), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.polygon(pts, fill=255)
            
            for yy in range(0, canvas_size, hatch_gap):
                for xx in range(canvas_size):
                    if mask.getpixel((xx, yy)) > 0:
                        draw.point((xx, yy), fill='black')
        elif fill_type == 'dotted':
            # Create mask then fill with dots
            mask = Image.new('L', (canvas_size, canvas_size), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.polygon(pts, fill=255)
            
            for xx in range(0, canvas_size, hatch_gap):
                for yy in range(0, canvas_size, hatch_gap):
                    if mask.getpixel((xx, yy)) > 0:
                        draw.ellipse((xx-1, yy-1, xx+1, yy+1), fill='black')
        
        # Draw outline with stroke style
        if stroke_style == 'solid':
            draw.polygon(pts, outline='black', width=stroke)
        elif stroke_style == 'dashed':
            # Draw dashed outline
            for i in range(len(pts)):
                p0, p1 = pts[i], pts[(i + 1) % len(pts)]
                _draw_dashed_line(draw, p0, p1, stroke, dash_len, dash_gap)
        elif stroke_style == 'dotted':
            # Draw dotted outline
            for px, py in pts:
                draw.ellipse((px - stroke, py - stroke, px + stroke, py + stroke), 
                           outline='black', width=1)

    else:
        # Unknown shape: fallback to a small circle
        fallback_size = min(s, 10)
        bbox = [x - fallback_size/2, y - fallback_size/2, 
               x + fallback_size/2, y + fallback_size/2]
        draw.ellipse(bbox, fill='black')

def render_scene(objects: List[Dict[str, Any]], cfg: Any) -> Image.Image:
    """
    Render a complete scene with multiple objects.
    
    Args:
        objects: List of object dictionaries
        cfg: Configuration object
        
    Returns:
        PIL Image of the rendered scene
    """
    canvas_size = getattr(cfg, 'img_size', 256)
    bg_color = getattr(cfg, 'bg_color', (255, 255, 255))
    
    img = Image.new('RGB', (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Sort objects by size (largest first) for proper layering
    def get_size(obj):
        size = obj.get('size', 30)
        if isinstance(size, str):
            size_map = {'small': 20, 'medium': 35, 'large': 50}
            return size_map.get(size, 30)
        return size if isinstance(size, (int, float)) else 30
    
    sorted_objects = sorted(objects, key=get_size, reverse=True)
    
    # Render each object
    for obj in sorted_objects:
        try:
            draw_shape(draw, obj, cfg)
        except Exception as e:
            # Fallback: draw a simple circle
            x = obj.get('center_x', obj.get('x', 128))
            y = obj.get('center_y', obj.get('y', 128))
            draw.ellipse((x-5, y-5, x+5, y+5), fill='black')
    
    return img
