"""Advanced drawing utilities with gradient fills and precise masking."""

import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import os
import string

from ...config import CONFIG, ATTRIBUTE_COLOR_MAP

def make_gradient_mask(size: int, vertical: bool = True) -> Image.Image:
    """Create a proper gradient mask for advanced fill effects."""
    mask = Image.new("L", (size, size))
    for i in range(size):
        v = int(255 * (i / (size - 1)))
        if vertical:
            mask.paste(v, (0, i, size, i + 1))
        else:
            mask.paste(v, (i, 0, i + 1, size))
    return mask

def make_linear_gradient(size: int, direction: str = "vertical") -> Image.Image:
    """Create a linear gradient mask for advanced fill effects."""
    return make_gradient_mask(size, vertical=(direction == "vertical"))

def generate_perlin_noise(w: int, h: int, scale: int = 10) -> np.ndarray:
    """Generate Perlin noise for texture effects."""
    # Simple noise generation for texture
    noise = np.random.random((h//scale + 1, w//scale + 1))
    noise = np.repeat(np.repeat(noise, scale, axis=0), scale, axis=1)
    return noise[:h, :w]

def find_coeffs(pa: List[Tuple[float, float]], pb: List[Tuple[float, float]]) -> Optional[List[float]]:
    """Find perspective transformation coefficients."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    
    try:
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8).tolist()
    except np.linalg.LinAlgError:
        return None

def make_gradient_mask(size: int, vertical: bool = True) -> Image.Image:
    """Create a simple vertical or horizontal gradient mask."""
    mask = Image.new("L", (size, size))
    for i in range(size):
        v = int(255 * (i / (size - 1)))
        if vertical:
            mask.paste(v, (0, i, size, i + 1))
        else:
            mask.paste(v, (i, 0, i + 1, size))
    return mask

class AdvancedDrawingUtils:
    """Advanced drawing utilities with gradient and noise effects."""
    
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.jitter_dash_options = [None, (5, 5), (10, 5), (15, 10)]
    
    def draw_precise_mask(
        self,
        draw_fn: Callable[[ImageDraw.ImageDraw, int], None],
        fill_type: str = "solid",
        vertical_gradient: bool = True
    ) -> Image.Image:
        """Draw a mask with optional gradient fill."""
        mask_hr = Image.new("L", (self.img_size, self.img_size), 0)
        draw_hr = ImageDraw.Draw(mask_hr)
        draw_fn(draw_hr, self.img_size)
        if fill_type == "gradient":
            grad = make_gradient_mask(self.img_size, vertical=vertical_gradient)
            mask_hr = Image.composite(grad, mask_hr, mask_hr)
        return mask_hr
    
    def draw_dashed_line(
        self,
        draw: ImageDraw.ImageDraw,
        xy: Tuple[float, float, float, float],
        dash: Tuple[int, int],
        fill: Union[int, Tuple[int, int, int, int]] = 255,
        width: int = 1
    ):
        """Draw a dashed line."""
        x1, y1, x2, y2 = xy
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
            
        dash_on, dash_off = dash
        dash_total = dash_on + dash_off
        
        segments = int(length / dash_total)
        for i in range(segments):
            t1 = (i * dash_total) / length
            t2 = (i * dash_total + dash_on) / length
            
            seg_x1 = x1 + t1 * dx
            seg_y1 = y1 + t1 * dy
            seg_x2 = x1 + t2 * dx
            seg_y2 = y1 + t2 * dy
            
            draw.line((seg_x1, seg_y1, seg_x2, seg_y2), fill=fill, width=width)
    
    def draw_dashed_arc(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: List[float],
        start: float,
        end: float,
        dash: Tuple[int, int],
        fill: Union[int, Tuple[int, int, int, int]] = 255,
        width: int = 1,
        steps: int = 200
    ):
        """Draw a dashed arc."""
        dash_on, dash_off = dash
        arc_length = end - start
        dash_total = dash_on + dash_off
        
        segments = int((arc_length * steps) / dash_total)
        
        for i in range(segments):
            t1 = start + (i * dash_total * arc_length) / (segments * dash_total)
            t2 = start + ((i * dash_total + dash_on) * arc_length) / (segments * dash_total)
            
            if t2 > end:
                t2 = end
                
            draw.arc(bbox, t1, t2, fill=fill, width=width)
    
    def apply_gradient_fill(
        self, 
        obj_canvas: Image.Image, 
        bbox: List[float], 
        fill_color_rgba: Tuple[int, int, int, int],
        direction: str = "vertical"
    ) -> Image.Image:
        """Apply gradient fill to an object canvas with proper masking."""
        
        # Create high-resolution mask for better gradient quality
        hr_size = max(obj_canvas.width * 2, 128)
        mask_hr = Image.new("L", (hr_size, hr_size), 0)
        draw_hr = ImageDraw.Draw(mask_hr)
        
        # Draw shape on high-res mask (simplified - you'd call your shape drawing function here)
        draw_hr.ellipse([0, 0, hr_size, hr_size], fill=255)  # Example for circle
        
        # Apply gradient
        if direction == "gradient":
            grad = make_gradient_mask(hr_size, vertical=True)
            mask_hr = Image.composite(grad, mask_hr, mask_hr)
        
        # Resize back to original size
        mask_final = mask_hr.resize(obj_canvas.size, Image.Resampling.LANCZOS)
        
        # Create solid color fill
        solid_color_fill = Image.new('RGBA', obj_canvas.size, fill_color_rgba)
        
        # Apply mask to create gradient effect
        solid_color_fill.putalpha(mask_final)
        
        # Composite onto object canvas
        return Image.alpha_composite(obj_canvas, solid_color_fill)
    
    def apply_noise_fill(
        self, 
        obj_canvas: Image.Image, 
        fill_color_rgba: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Apply noise fill to an object canvas."""
        noise = generate_perlin_noise(obj_canvas.width, obj_canvas.height)
        noise_img = Image.fromarray((noise*255).astype('uint8')).convert('L')
        
        # Create RGBA noise image
        noise_img_with_alpha = Image.new('RGBA', noise_img.size, fill_color_rgba)
        noise_img_with_alpha.putalpha(noise_img)
        
        # Composite onto object canvas
        return Image.alpha_composite(obj_canvas, noise_img_with_alpha)
    
    def apply_perspective_transform(
        self, 
        obj_canvas: Image.Image, 
        transform_probability: float = 0.2
    ) -> Image.Image:
        """Apply perspective transformation with given probability."""
        if random.random() < transform_probability:
            target_pts = [
                (0, 0), 
                (obj_canvas.width, 0), 
                (obj_canvas.width, obj_canvas.height), 
                (0, obj_canvas.height)
            ]
            source_pts = [
                (random.uniform(0, 20), random.uniform(0, 20)),
                (obj_canvas.width - random.uniform(0, 20), random.uniform(0, 20)),
                (obj_canvas.width - random.uniform(0, 20), obj_canvas.height - random.uniform(0, 20)),
                (random.uniform(0, 20), obj_canvas.height - random.uniform(0, 20))
            ]
            
            coeffs = find_coeffs(source_pts, target_pts)
            if coeffs:
                return obj_canvas.transform(
                    obj_canvas.size, 
                    Image.PERSPECTIVE, 
                    coeffs, 
                    Image.BILINEAR
                )
        
        return obj_canvas
