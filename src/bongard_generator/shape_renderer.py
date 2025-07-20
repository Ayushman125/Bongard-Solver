# src/bongard_generator/shape_renderer.py
import math
from PIL import Image, ImageDraw

DEFAULT_SIZE_MAP = {
    "small": 32,
    "medium": 64,
    "large": 96
}

def draw_shape(draw, obj, cfg):
    """Draw a single shape on the canvas."""
    shape = obj.get('shape', 'circle')
    x, y = obj.get('x', 64), obj.get('y', 64)
    raw_size = obj.get('size', 'medium')
    # Use config override if present
    size_map = getattr(cfg, 'size_mapping', DEFAULT_SIZE_MAP)
    if isinstance(raw_size, str):
        try:
            s = int(size_map[raw_size])
        except (KeyError, ValueError):
            raise ValueError(f"Unknown size label: {raw_size}")
    elif isinstance(raw_size, (int, float)):
        s = int(raw_size)
    else:
        raise TypeError(f"Unsupported size type: {type(raw_size)}")
    assert isinstance(s, int), f"Size must be int, got {type(s)}"
    c = obj.get('color', 'black')
    fill = obj.get('fill', None)
    rotation = obj.get('rotation', 0)
    stroke_width = int(getattr(cfg, 'stroke_width', 1))

    if rotation != 0:
        layer_size = int(s * 1.5)
        img_layer = Image.new('RGBA', (layer_size, layer_size))
        draw_layer = ImageDraw.Draw(img_layer)
        _draw_primitive(draw_layer, shape, (layer_size // 2, layer_size // 2), s, c, fill, stroke_width)
        rotated_layer = img_layer.rotate(rotation, expand=True, resample=Image.BICUBIC)
        paste_x = x - rotated_layer.width // 2
        paste_y = y - rotated_layer.height // 2
        mask = rotated_layer.split()[3]
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
