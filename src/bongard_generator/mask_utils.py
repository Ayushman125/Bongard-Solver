"""
Utilities for creating and rendering scenes with various objects,
shapes, and textures, driven by a central configuration.
This module orchestrates the rendering pipeline, delegating shape-specific
drawing to the PrototypeAction class and the new shape renderer.
"""

import logging
import random
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw
from .shape_renderer import draw_shape, render_scene

from .config import GeneratorConfig
from .prototype_action import PrototypeAction, create_prototype_config
from .rule_loader import AbstractRule

logger = logging.getLogger(__name__)

def create_composite_scene(
    config: GeneratorConfig,
    rule: AbstractRule,
    is_positive: bool,
    prototype_action: PrototypeAction
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Creates a single Bongard scene by generating objects, applying a rule,
    and rendering them using the PrototypeAction system.

    Args:
        config: The central generator configuration.
        rule: The abstract rule to apply to the scene.
        is_positive: Whether the scene should satisfy or violate the rule.
        prototype_action: The action handler for drawing shapes.

    Returns:
        A tuple containing the rendered PIL Image (RGB) and scene features.
    """
    canvas = Image.new('RGB', (config.img_size, config.img_size), config.bg_color)
    
    # 1. Generate initial object properties
    num_objects = random.randint(config.min_shapes, config.max_shapes)
    objects = _generate_initial_objects(num_objects, config, prototype_action)

    # 2. Apply the rule to modify object properties
    modified_objects, scene_features = rule.apply(objects, is_positive)
    
    # 3. Render the final objects onto the canvas using PrototypeAction
    _render_objects(canvas, modified_objects, config, prototype_action)

    # 4. Apply advanced background textures if enabled
    _apply_advanced_textures(canvas, config)

    scene_features['object_count'] = len(modified_objects)
    return canvas, scene_features

def _generate_initial_objects(num_objects: int, config: GeneratorConfig, prototype_action: PrototypeAction) -> List[Dict[str, Any]]:
    """Generates a list of basic, rule-agnostic object properties."""
    objects = []
    if not prototype_action.shapes:
        logger.error("No shapes available from PrototypeAction. Cannot generate objects.")
        return []

    for _ in range(num_objects):
        # Shape selection is now implicitly handled by prototype_action.draw()
        obj = {
            "position": (random.randint(0, config.img_size), random.randint(0, config.img_size)),
            "size": random.randint(30, config.img_size // 3),
            "color": "black",  # All shapes are black for Bongard problems
            "rotation": random.uniform(0, 360) if config.enable_rotation else 0,
            "fill_type": config.fill_type,
        }
        objects.append(obj)
    return objects

def _render_objects(canvas: Image.Image, objects: List[Dict[str, Any]], config: GeneratorConfig, prototype_action: PrototypeAction):
    """Renders all objects onto the canvas using professional geometric primitives."""
    import math
    
    # Robust sorting that handles string/int conversion issues  
    def get_size(obj):
        size = obj.get("size", 0)
        if isinstance(size, str):
            # Convert string sizes to numeric
            if size == "small":
                return 20
            elif size == "medium": 
                return 40
            elif size == "large":
                return 60
            try:
                return int(size)
            except ValueError:
                return 30
        return size if isinstance(size, (int, float)) else 30
    
    sorted_objects = sorted(objects, key=get_size, reverse=True)

    for obj in sorted_objects:
        # Get object properties with fallbacks
        pos = obj.get("position", (config.img_size//2, config.img_size//2))
        if isinstance(pos, tuple) and len(pos) == 2:
            center_x, center_y = pos
        else:
            center_x = obj.get("center_x", config.img_size//2)  
            center_y = obj.get("center_y", config.img_size//2)
        
        size = get_size(obj)
        shape = obj.get("shape", "circle")  # Use rule-specified shape
        color = obj.get("color", "black")
        rotation = obj.get("rotation", 0)
        fill_type = obj.get("fill_type", "solid")
        stroke_style = obj.get("stroke_style", "solid")
        stroke_width = obj.get("stroke_width", 2)
        
        # Use the professional shape renderer
        try:
            draw_shape(canvas, shape, (center_x, center_y), size, color, 
                      fill_type=fill_type, stroke_style=stroke_style, 
                      stroke_width=stroke_width, rotation=rotation)
        except Exception as e:
            logging.warning(f"Failed to draw shape {shape} with shape renderer: {e}, falling back to basic shapes")
            # Fallback to basic drawing if shape renderer fails
            draw = ImageDraw.Draw(canvas)
            _draw_geometric_shape(draw, center_x, center_y, size, shape, _get_color(color), rotation)

def _get_color(color_spec):
    """Convert color specification to RGB tuple."""
    if isinstance(color_spec, tuple) and len(color_spec) == 3:
        return color_spec
    elif isinstance(color_spec, str):
        color_map = {
            "black": (0, 0, 0),
            "red": (255, 0, 0), 
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255)
        }
        return color_map.get(color_spec, (0, 0, 0))
    return (0, 0, 0)

def _draw_geometric_shape(draw, center_x, center_y, size, shape, color, rotation):
    """Draw a specific geometric shape."""
    half_size = size // 2
    
    if shape == "circle":
        # Draw circle
        bbox = [center_x - half_size, center_y - half_size, 
                center_x + half_size, center_y + half_size]
        draw.ellipse(bbox, fill=color, outline=color)
        
    elif shape == "square" or shape == "rectangle":
        # Draw square/rectangle
        bbox = [center_x - half_size, center_y - half_size,
                center_x + half_size, center_y + half_size] 
        draw.rectangle(bbox, fill=color, outline=color)
        
    elif shape == "triangle":
        # Draw triangle (pointing up)
        points = [
            (center_x, center_y - half_size),           # Top point
            (center_x - half_size, center_y + half_size), # Bottom left
            (center_x + half_size, center_y + half_size)  # Bottom right
        ]
        draw.polygon(points, fill=color, outline=color)
        
    else:
        # Default to circle for unknown shapes
        bbox = [center_x - half_size, center_y - half_size,
                center_x + half_size, center_y + half_size]
        draw.ellipse(bbox, fill=color, outline=color)

def _apply_advanced_textures(canvas: Image.Image, config: GeneratorConfig):
    """Applies advanced procedural textures to the background."""
    if config.bg_texture == "noise":
        _add_noise_background(canvas, config)
    elif config.bg_texture == "checker":
        _add_checkerboard_background(canvas, config)

def _add_noise_background(img: Image.Image, cfg: GeneratorConfig):
    """Generates and blends a noise texture onto the image background."""
    if cfg.noise_level == 0:
        return
    noise_arr = np.random.randint(0, 256, (*img.size, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr, 'RGB')
    img.paste(Image.blend(img, noise_img, alpha=cfg.noise_opacity))

def _add_checkerboard_background(img: Image.Image, cfg: GeneratorConfig):
    """Draws a checkerboard pattern onto the image background."""
    if cfg.checker_size == 0:
        return
    w, h = img.size
    tile = cfg.checker_size
    draw = ImageDraw.Draw(img, 'RGBA')
    fill_color = (230, 230, 230, int(255 * cfg.noise_opacity)) 
    for x in range(0, w, tile):
        for y in range(0, h, tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle([x, y, x + tile, y + tile], fill=fill_color, outline=None)

