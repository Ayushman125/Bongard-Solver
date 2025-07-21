# --- NVlabs-compatible BongardLogoRenderer ---
from PIL import Image
import numpy as np

class BongardLogoRenderer:
    def __init__(self, canvas_size=128, bg_color='white'):
        self.canvas_size = canvas_size
        self.bg_color = bg_color

    def render_with_masks(self, program):
        """
        Render a Bongard program (list of shape dicts) into an image and per-shape masks.
        Returns: (image, masks, shape_types)
        """
        # Render the full image
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), self.bg_color)
        masks = []
        shape_types = []
        for obj in program:
            # Draw shape on main image
            draw_shape(img, obj, cfg=None)
            # Create a mask for this shape only
            mask = Image.new('L', (self.canvas_size, self.canvas_size), 0)
            draw_shape(mask, obj, cfg=None)
            # Binarize mask
            mask = mask.point(lambda p: 255 if p > 0 else 0, mode='1')
            masks.append(np.array(mask))
            shape_types.append(obj.get('shape', 'unknown'))
        return img, masks, shape_types
# src/bongard_generator/shape_renderer.py
import math
import random
from PIL import Image, ImageDraw, ImageColor
from pathlib import Path

DEFAULT_SIZE_MAP = {
    "small": 32,
    "medium": 64,
    "large": 96
}

def _draw_primitive(img_layer, shape_type, color, fill_style, stroke_width):
    """Draws a primitive shape onto its own layer."""
    draw = ImageDraw.Draw(img_layer)
    width, height = img_layer.size
    
    # Define bounding box for the shape, with a small margin
    box = [stroke_width, stroke_width, width - stroke_width, height - stroke_width]
    
    fill_color = None
    outline_color = None

    # --- Comprehensive Fill Style Implementation ---
    if fill_style == 'solid':
        fill_color = color
        outline_color = 'black' if color != 'black' else 'darkgrey'
    elif fill_style == 'hollow':
        fill_color = None
        outline_color = color
    elif fill_style == 'striped':
        fill_color = None  # Draw stripes manually
        outline_color = color
        for i in range(0, width, 8):
            draw.line([(i, 0), (i, height)], fill=color, width=2)
    elif fill_style == 'dotted':
        fill_color = None  # Draw dots manually
        outline_color = color
        for x in range(0, width, 10):
            for y in range(0, height, 10):
                draw.ellipse([x, y, x + 2, y + 2], fill=color)
    elif fill_style == 'gradient':
        # Create a linear gradient from the given color to white
        c1 = ImageColor.getrgb(color)
        c2 = (255, 255, 255) # White
        for i in range(height):
            r = int(c1[0] + (c2[0] - c1[0]) * (i / height))
            g = int(c1[1] + (c2[1] - c1[1]) * (i / height))
            b = int(c1[2] + (c2[2] - c1[2]) * (i / height))
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        fill_color = None # Gradient is already drawn
        outline_color = 'black'
    elif fill_style == 'texture':
        try:
            # Load a random texture and tile it
            texture_path = random.choice(list(Path('data/textures').glob('**/*.png')))
            texture = Image.open(texture_path).convert('RGBA')
            for x in range(0, width, texture.width):
                for y in range(0, height, texture.height):
                    img_layer.paste(texture, (x, y), texture)
        except (FileNotFoundError, IndexError):
            # Fallback to solid if textures are not found
            fill_color = color
        outline_color = 'black'
    else: # Default fallback
        fill_color = color
        outline_color = 'black'

    # --- Shape Drawing Logic ---
    if shape_type == 'circle':
        draw.ellipse(box, fill=fill_color, outline=outline_color, width=stroke_width)
    elif shape_type == 'square':
        draw.rectangle(box, fill=fill_color, outline=outline_color, width=stroke_width)
    elif shape_type == 'triangle':
        points = [(width / 2, stroke_width), (stroke_width, height - stroke_width), (width - stroke_width, height - stroke_width)]
        draw.polygon(points, fill=fill_color, outline=outline_color, width=stroke_width)
    elif shape_type == 'pentagon':
        points = _create_polygon_points(5, width / 2, height / 2, min(width, height) / 2 - stroke_width)
        draw.polygon(points, fill=fill_color, outline=outline_color, width=stroke_width)
    elif shape_type == 'hexagon':
        points = _create_polygon_points(6, width / 2, height / 2, min(width, height) / 2 - stroke_width)
        draw.polygon(points, fill=fill_color, outline=outline_color, width=stroke_width)
    elif shape_type == 'star':
        points = _create_polygon_points(5, width / 2, height / 2, min(width, height) / 2 - stroke_width, 2)
        draw.polygon(points, fill=fill_color, outline=outline_color, width=stroke_width)
    elif shape_type == 'arc':
        draw.arc(box, start=45, end=225, fill=color, width=stroke_width)
    elif shape_type == 'line':
        draw.line([(stroke_width, height/2), (width-stroke_width, height/2)], fill=color, width=stroke_width)
    elif shape_type == 'zigzag':
        points = []
        for i in range(6):
            x = (width / 5) * i
            y = height / 2 + (20 * (-1)**i)
            points.append((x, y))
        draw.line(points, fill=color, width=stroke_width, joint='curve')
    elif shape_type == 'path':
        # Example of a complex path
        points = [(width*0.1, height*0.1), (width*0.3, height*0.5), (width*0.7, height*0.2), (width*0.9, height*0.9)]
        draw.line(points, fill=color, width=stroke_width, joint='curve')
    elif shape_type == 'prototype':
        # Prototypes are handled by their own drawing logic, but we can draw a placeholder
        draw.rectangle(box, fill=None, outline='purple', width=stroke_width+1)
        draw.text((10, 10), "Proto", fill='purple')
    else:
        # Fallback for unknown shapes
        draw.rectangle(box, fill='lightgrey', outline='red', width=1)
        draw.text((10, 10), "?", fill='red')

def draw_shape(canvas, obj, cfg):
    """Draws a complex shape, possibly rotated, onto the main canvas."""
    try:
        # Get size in pixels. It can be an int or a list/tuple [width, height].
        size = obj.get('size', 32)
        if isinstance(size, (list, tuple)):
            width, height = int(size[0]), int(size[1])
        else:
            width = height = int(size)

        if width <= 0 or height <= 0:
            # logger.warning(f"Invalid size for shape: {obj}")
            return

        # Create a transparent layer for the shape
        img_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Determine fill style and color
        fill_style = obj.get('fill', 'solid')
        stroke_width = obj.get('stroke_width', 2)
        color = obj.get('color', 'black')

        # Draw the primitive shape onto its layer
        _draw_primitive(img_layer, obj['shape'], color, fill_style, stroke_width)

        # Handle rotation
        angle = obj.get('rotation', 0)
        if angle != 0:
            # Rotate the layer, expanding the canvas to fit the new dimensions
            final_layer = img_layer.rotate(angle, expand=True, resample=Image.BICUBIC)
        else:
            final_layer = img_layer

        # Get position in pixels. This is the center of the shape.
        x = int(obj.get('x', 64))
        y = int(obj.get('y', 64))
        
        # Calculate the top-left corner for pasting, accounting for the new layer size
        paste_x = x - final_layer.width // 2
        paste_y = y - final_layer.height // 2
        box = (paste_x, paste_y)

        # Composite the final shape layer onto the main canvas
        # The mask ensures that transparent pixels are not copied
        mask = final_layer.getchannel('A')
        canvas.paste(final_layer, box, mask)

    except Exception as e:
        # Use logger if available, otherwise print
        error_msg = f"Error drawing shape {obj.get('shape', 'unknown')}: {e}"
        try:
            from logging import getLogger
            getLogger(__name__).error(error_msg, exc_info=True)
        except ImportError:
            print(f"ERROR: {error_msg}")

def _create_polygon_points(sides, center_x, center_y, radius, skip=1):
    """Helper to create points for regular polygons and stars."""
    points = []
    for i in range(sides):
        angle = (i * skip) * (360 / sides)
        rad_angle = math.radians(angle - 90)
        x = center_x + radius * math.cos(rad_angle)
        y = center_y + radius * math.sin(rad_angle)
        points.append((x, y))
    return points