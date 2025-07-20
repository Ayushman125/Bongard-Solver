"""
Utilities for creating and rendering scenes with various objects,
shapes, and textures, driven by a central configuration.
This module orchestrates the rendering pipeline, delegating shape-specific
drawing to the PrototypeAction class.
"""

import logging
import random
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

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
    """Renders all objects onto the canvas using the PrototypeAction system."""
    sorted_objects = sorted(objects, key=lambda o: o.get("size", 0), reverse=True)

    for obj in sorted_objects:
        # Create a specific drawing configuration for this object
        draw_config = create_prototype_config(
            jitter_px=config.jitter_strength * config.img_size,
            enable_rotation=config.enable_rotation,
            fill_color=(0, 0, 0),  # Always black
            fill_pattern=obj.get("fill_type", "solid"),
        )
        
        # Delegate drawing entirely to the prototype action instance
        prototype_action.draw(
            img=canvas,
            center=obj["position"],
            size=obj["size"],
            config=draw_config
        )

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

