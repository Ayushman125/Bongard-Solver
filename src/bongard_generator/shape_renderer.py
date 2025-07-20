# src/bongard_generator/shape_renderer.py
import math
from PIL import Image, ImageDraw

def draw_shape(draw, obj, cfg):
    """Draw a single shape on the canvas."""
    shape = obj.get('shape', 'circle')
    x, y = obj.get('x', 64), obj.get('y', 64)
    s = int(obj.get('size', 32))  # Ensure size is an integer
    c = obj.get('color', 'black')
    fill = obj.get('fill', None)
    rotation = obj.get('rotation', 0)
    
    # Get stroke width from config, ensuring it's an integer
    stroke_width = int(getattr(cfg, 'stroke_width', 1))

    if rotation != 0:
        # To rotate, we need to work on a separate layer and paste it back
        # Use a larger layer to avoid clipping after rotation
        layer_size = int(s * 1.5)
        img_layer = Image.new('RGBA', (layer_size, layer_size))
        draw_layer = ImageDraw.Draw(img_layer)
        
        # Draw the shape on the layer at its center
        _draw_primitive(draw_layer, shape, (layer_size // 2, layer_size // 2), s, c, fill, stroke_width)
        
        # Rotate the layer
        rotated_layer = img_layer.rotate(rotation, expand=True, resample=Image.BICUBIC)
        
        # Paste the rotated layer onto the main image
        paste_x = x - rotated_layer.width // 2
        paste_y = y - rotated_layer.height // 2
        # The existing draw object is for the main canvas, which might not support alpha paste directly.
        # A common approach is to create a temporary RGBA canvas to paste onto, then paste that.
        # However, for simplicity here, we'll assume the main canvas can handle it.
        # This might need adjustment depending on the main canvas mode.
        # For now, let's try pasting the mask directly.
        mask = rotated_layer.split()[3]  # Get the alpha channel as a mask
        draw.bitmap((paste_x, paste_y), mask, fill=c)

    else:
        _draw_primitive(draw, shape, (x, y), s, c, fill, stroke_width)


def _draw_primitive(draw, shape, center, size, color, fill, stroke_width):
    """Draws the geometric primitive without rotation."""
    x, y = center
    s = size # s is already an int here

    # Determine fill color
    fill_color = color if fill else None

    if shape == 'circle':
        bbox = [x - s / 2, y - s / 2, x + s / 2, y + s / 2]
        draw.ellipse(bbox, fill=fill_color, outline=color, width=stroke_width)
    elif shape == 'square':
        bbox = [x - s / 2, y - s / 2, x + s / 2, y + s / 2]
        draw.rectangle(bbox, fill=fill_color, outline=color, width=stroke_width)
    elif shape == 'triangle':
        points = [
            (x, y - s / 2),
            (x - s / 2, y + s / 2),
            (x + s / 2, y + s / 2)
        ]
        draw.polygon(points, fill=fill_color, outline=color, width=stroke_width)
    elif shape == 'star':
        points = []
        for i in range(10):
            angle = i * math.pi / 5
            r = (s / 2) if (i % 2 == 0) else (s / 4)
            points.append((x + r * math.sin(angle), y - r * math.cos(angle)))
        draw.polygon(points, fill=fill_color, outline=color, width=stroke_width)
    elif shape == 'hexagon':
        points = []
        for i in range(6):
            angle = i * math.pi / 3
            points.append((x + s / 2 * math.cos(angle), y + s / 2 * math.sin(angle)))
        draw.polygon(points, fill=fill_color, outline=color, width=stroke_width)
