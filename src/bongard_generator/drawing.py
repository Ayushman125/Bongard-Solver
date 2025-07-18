"""Drawing utilities for shape generation"""

import os
import random
import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import string

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    # randrange excludes hi, so we +1
    return random.randrange(lo, hi+1)

from .utils import generate_perlin_noise, find_coeffs, make_linear_gradient, draw_dashed_line, draw_dashed_arc
from .config_loader import get_config

logger = logging.getLogger(__name__)

# Get config safely
try:
    CONFIG = get_config()
    ATTRIBUTE_COLOR_MAP = {
        'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
        'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'gray': (128, 128, 128)
    }
except:
    CONFIG = {'data': {'synthetic_data_config': {}}}
    ATTRIBUTE_COLOR_MAP = {
        'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
        'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'gray': (128, 128, 128)
    }

logger = logging.getLogger(__name__)

class TurtleCanvas:
    """Turtle graphics canvas for drawing programs."""
    
    def __init__(self, size: int, color: Tuple[int, int, int] = (0, 0, 0), width: int = 2, stamps: List[Image.Image] = None):
        # Use 'RGBA' mode for stamps to support transparency
        self.img: Image.Image = Image.new('RGBA', (size, size), (255, 255, 255, 0))  # Transparent background
        self.draw: ImageDraw.ImageDraw = ImageDraw.Draw(self.img)
        self.pos: Tuple[float, float] = (size//2, size//2)
        self.ang: float = 0
        self.color: Tuple[int, int, int, int] = (*color, 255)  # Add alpha channel
        self.width: int = width
        self.stamps = stamps or []  # Default to empty list if no stamps provided
        self.size = size

    def forward(self, dist: float, pen: bool = True) -> None:
        """Move forward, optionally drawing a line."""
        rad: float = math.radians(self.ang)
        nxt: Tuple[float, float] = (self.pos[0] + dist * math.cos(rad), self.pos[1] + dist * math.sin(rad))
        if pen: 
            self.draw.line([self.pos, nxt], fill=self.color, width=self.width)
        self.pos = nxt

    def turn(self, delta: float) -> None:
        """Turn by delta degrees."""
        self.ang = (self.ang + delta) % 360
    
    def set_background(self, color: str = 'white') -> None:
        """Set background color."""
        if color == 'white':
            bg_color = (255, 255, 255, 255)
        elif color == 'lightgray':
            bg_color = (240, 240, 240, 255)
        elif color == 'lightblue':
            bg_color = (230, 240, 255, 255)
        else:
            bg_color = (255, 255, 255, 255)
        
        # Create background
        bg_img = Image.new('RGBA', (self.size, self.size), bg_color)
        self.img = Image.alpha_composite(bg_img, self.img)
    
    def draw_shape(self, shape: str, x: int, y: int, size: int, color: str, filled: bool = True) -> None:
        """Draw a shape at specified position."""
        # Convert color name to RGB
        color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'black': (0, 0, 0),
            'white': (255, 255, 255)
        }
        rgb_color = color_map.get(color, (0, 0, 0))
        
        # Calculate bounding box
        left = x - size // 2
        top = y - size // 2
        right = x + size // 2
        bottom = y + size // 2
        
        if shape == 'circle':
            if filled:
                self.draw.ellipse([left, top, right, bottom], fill=rgb_color)
            else:
                self.draw.ellipse([left, top, right, bottom], outline=rgb_color, width=2)
        elif shape == 'square':
            if filled:
                self.draw.rectangle([left, top, right, bottom], fill=rgb_color)
            else:
                self.draw.rectangle([left, top, right, bottom], outline=rgb_color, width=2)
        elif shape == 'triangle':
            # Draw triangle
            points = [
                (x, top),  # Top point
                (left, bottom),  # Bottom left
                (right, bottom)  # Bottom right
            ]
            if filled:
                self.draw.polygon(points, fill=rgb_color)
            else:
                self.draw.polygon(points, outline=rgb_color, width=2)
    
    def get_image(self) -> Image.Image:
        """Get the final image."""
        return self.img.convert('RGB')  # Convert to RGB for final output
    
    def stamp_at_current_pos(self, stamp_idx: int, size: int = 20) -> None:
        """Stamp an image at the current position."""
        if self.stamps and 0 <= stamp_idx < len(self.stamps):
            stamp_img = self.stamps[stamp_idx].resize((size, size), Image.LANCZOS)
            # Calculate top-left corner for pasting
            paste_x = int(self.pos[0] - size / 2)
            paste_y = int(self.pos[1] - size / 2)
            # Ensure stamp_img is RGBA for alpha_composite
            if stamp_img.mode != 'RGBA':
                stamp_img = stamp_img.convert('RGBA')
            
            # Create a blank image to composite onto, ensuring it's RGBA
            temp_img = Image.new('RGBA', self.img.size, (255, 255, 255, 0))
            temp_img.paste(stamp_img, (paste_x, paste_y), stamp_img)  # Use stamp's own alpha
            self.img = Image.alpha_composite(self.img, temp_img)  # Composite with existing drawing

class ShapeDrawer:
    """Handles drawing of various shapes with advanced rendering techniques."""
    
    def __init__(self, config=None):
        self.config = config
        self.img_size = getattr(config, 'img_size', 128) if config else 128
        self.font: Optional[ImageFont.FreeTypeFont] = None
        self._load_font()
    
    def _load_font(self) -> None:
        """Load font for text rendering."""
        try:
            font_path = CONFIG['data']['synthetic_data_config'].get('font_path')
            if font_path and os.path.exists(font_path):
                try:
                    self.font = ImageFont.truetype(font_path, 24)
                except Exception as e:
                    logger.warning(f"Failed to load font from {font_path}: {e}")
                    self.font = ImageFont.load_default()
            else:
                self.font = ImageFont.load_default()
        except:
            # Fallback if CONFIG is not available
            self.font = ImageFont.load_default()

    def draw_precise_mask(
        self,
        draw_fn: Callable[[ImageDraw.ImageDraw, int], None],
        size: int,
        supersample: int = 4
    ) -> Image.Image:
        """Draw a precise mask with supersampling."""
        high_res = size * supersample
        mask_hr = Image.new("L", (high_res, high_res), 0)
        draw_hr = ImageDraw.Draw(mask_hr)
        draw_fn(draw_hr, high_res)
        mask_lr = mask_hr.resize((size, size), resample=Image.LANCZOS)
        return mask_lr.point(lambda p: 255 if p > 128 else 0, mode="1")

    def _draw_polygon_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, pts_hr: List[Tuple[float, float]], 
                           stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a polygon shape."""
        if dash_pattern_hr:
            verts = pts_hr + [pts_hr[0]]
            for (x0, y0), (x1, y1) in zip(verts, verts[1:]):
                draw_dashed_line(draw_hr, (x0, y0, x1, y1), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.polygon(pts_hr, outline=fill_color, fill=fill_color, width=stroke_width_hr)

    def _draw_rectangle_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, bbox_hr: List[float], 
                             stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a rectangle shape."""
        if dash_pattern_hr:
            x0, y0, x1, y1 = bbox_hr
            verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            for (sx, sy), (ex, ey) in zip(verts, verts[1:]):
                draw_dashed_line(draw_hr, (sx, sy, ex, ey), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.rectangle(bbox_hr, outline=fill_color, fill=fill_color, width=stroke_width_hr)

    def _draw_ellipse_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, bbox_hr: List[float], 
                           stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw an ellipse shape."""
        if dash_pattern_hr:
            draw_dashed_arc(draw_hr, bbox_hr, 0, 360, dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.ellipse(bbox_hr, outline=fill_color, fill=fill_color, width=stroke_width_hr)

    def _draw_arc_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, bbox_hr: List[float], 
                       stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw an arc shape."""
        start: float = random.uniform(0, 360)
        extent: float = random.uniform(60, 300)
        if dash_pattern_hr:
            draw_dashed_arc(draw_hr, bbox_hr, start, start + extent, dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.arc(bbox_hr, start=start, end=start + extent, fill=fill_color, width=stroke_width_hr)

    def _draw_zigzag_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, bbox_hr: List[float], 
                          stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a zigzag shape."""
        pts_hr: List[Tuple[float, float]] = []
        segments: int = safe_randint(3, 7)
        for k in range(segments + 1):
            px: float = bbox_hr[0] + (bbox_hr[2] - bbox_hr[0]) * k / segments
            py: float = bbox_hr[1] + (bbox_hr[3] - bbox_hr[1]) * random.choice([0, 1])
            pts_hr.append((px, py))
        
        if dash_pattern_hr:
            for p0, p1 in zip(pts_hr, pts_hr[1:]):
                draw_dashed_line(draw_hr, (p0[0], p0[1], p1[0], p1[1]), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.line(pts_hr, fill=fill_color, width=stroke_width_hr)

    def _draw_fan_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, center_offset_hr: int, half_size_hr: int, 
                       stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a fan shape."""
        center_hr: Tuple[int, int] = (center_offset_hr, center_offset_hr)
        spokes: int = safe_randint(5, 12)
        for k in range(spokes):
            ang: float = math.radians(k * 360 / spokes)
            ex: float = center_hr[0] + half_size_hr * math.cos(ang)
            ey: float = center_hr[1] + half_size_hr * math.sin(ang)
            if dash_pattern_hr:
                draw_dashed_line(draw_hr, (center_hr[0], center_hr[1], ex, ey), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
            else:
                draw_hr.line([center_hr, (ex, ey)], fill=fill_color, width=stroke_width_hr)

    def _draw_grid_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, bbox_hr: List[float], 
                        stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a grid shape."""
        rows: int = safe_randint(2, 5)
        cols: int = safe_randint(2, 5)
        cell_w: float = (bbox_hr[2] - bbox_hr[0]) / cols
        cell_h: float = (bbox_hr[3] - bbox_hr[1]) / rows
        for i in range(rows):
            for j in range(cols):
                x0: float = bbox_hr[0] + j * cell_w
                y0: float = bbox_hr[1] + i * cell_h
                x1, y1 = x0 + cell_w * 0.8, y0 + cell_h * 0.8
                if dash_pattern_hr:
                    # Draw dashed rectangle
                    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
                    for (sx, sy), (ex, ey) in zip(verts, verts[1:]):
                        draw_dashed_line(draw_hr, (sx, sy, ex, ey), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
                else:
                    draw_hr.rectangle([x0, y0, x1, y1], outline=fill_color, width=stroke_width_hr)

    def _draw_text_char_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, size_hr: int, center_offset_hr: int, 
                             fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a text character shape."""
        font_size_hr: int = size_hr
        font: ImageFont.FreeTypeFont = self.font or ImageFont.load_default()
        char: str = random.choice(string.ascii_uppercase + string.digits)
        try:
            left, top, right, bottom = draw_hr.textbbox((0, 0), char, font=font)
            tw, th = right - left, bottom - top
        except AttributeError:
            tw, th = draw_hr.textsize(char, font=font)
        
        draw_hr.text((center_offset_hr - tw/2, center_offset_hr - th/2), char, font=font, fill=fill_color)

    def _draw_icon_arrow_shape(self, draw_hr: ImageDraw.ImageDraw, high_res: int, center_offset_hr: int, half_size_hr: int, 
                              stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw an arrow icon shape."""
        pts_hr: List[Tuple[float, float]] = [
            (center_offset_hr - half_size_hr, center_offset_hr),
            (center_offset_hr + half_size_hr, center_offset_hr),
            (center_offset_hr + half_size_hr - 10 * high_res / self.img_size, center_offset_hr - 10 * high_res / self.img_size),
            (center_offset_hr + half_size_hr, center_offset_hr),
            (center_offset_hr + half_size_hr - 10 * high_res / self.img_size, center_offset_hr + 10 * high_res / self.img_size)
        ]
        if dash_pattern_hr:
            for p0, p1 in zip(pts_hr, pts_hr[1:]):
                draw_dashed_line(draw_hr, (p0[0], p0[1], p1[0], p1[1]), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.line(pts_hr, fill=fill_color, width=stroke_width_hr)

    def _draw_bump_curve(self, draw_hr: ImageDraw.ImageDraw, high_res: int, bbox_hr: List[float], 
                        stroke_width_hr: int, dash_pattern_hr: Optional[Tuple[int, ...]], fill_color: Union[int, Tuple[int, int, int, int]]) -> None:
        """Draw a bump curve shape."""
        pts = []
        for i in range(20):
            t = i / 19
            x = bbox_hr[0] + t * (bbox_hr[2] - bbox_hr[0])
            y = bbox_hr[1] + (bbox_hr[3] - bbox_hr[1]) * 0.5 + math.sin(2 * math.pi * t * safe_randint(1, 3)) * high_res * 0.1
            pts.append((x, y))
        
        if dash_pattern_hr:
            for p0, p1 in zip(pts, pts[1:]):
                draw_dashed_line(draw_hr, (p0[0], p0[1], p1[0], p1[1]), dash=dash_pattern_hr, fill=fill_color, width=stroke_width_hr)
        else:
            draw_hr.line(pts, fill=fill_color, width=stroke_width_hr)

def apply_occluder(img: Image.Image, canvas_width: int, canvas_height: int, 
                  occluder_prob: float, min_occluder_size: int, max_occluder_size: int = 20) -> Image.Image:
    """Apply random rectangular or elliptical occluders to the image."""
    if random.random() >= occluder_prob:
        return img
    
    W, H = canvas_width, canvas_height
    
    occluder_mask = Image.new('L', (W, H), color=0)
    occluder_draw = ImageDraw.Draw(occluder_mask)
    num_occluders = safe_randint(0, 3)
    
    for _ in range(num_occluders):
        shape: str = random.choice(['rect', 'ellipse'])
        x0: float = random.uniform(0, W)
        y0: float = random.uniform(0, H)
        w: float = random.uniform(min_occluder_size, max_occluder_size)
        h: float = random.uniform(min_occluder_size, max_occluder_size)
        x1, y1 = x0 + w, y0 + h
        if shape == 'rect':
            occluder_draw.rectangle([x0, y0, x1, y1], fill=255)
        else:
            occluder_draw.ellipse([x0, y0, x1, y1], fill=255)
    
    inverted_occluder_mask = Image.eval(occluder_mask, lambda x: 255 - x)
    white_background = Image.new('L', (W, H), color=255)
    img = Image.composite(img, white_background, inverted_occluder_mask)
    return img

def randomize_canvas(canvas_width: int, canvas_height: int, grayscale: bool = True, bg_textures: List[Image.Image] = None) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    """Randomize the canvas by optionally applying a background texture or a random color."""
    W, H = canvas_width, canvas_height
    if grayscale:
        img = Image.new('L', (W, H), color=255)  
        draw = ImageDraw.Draw(img)
    else:
        if random.random() < 0.2:
            arr = np.random.randint(0, 255, (H, W), dtype=np.uint8)
            img = Image.fromarray(arr).convert('RGB')
        elif bg_textures and random.random() < 0.3:
            bg = random.choice(bg_textures).resize((W, H))
            img = Image.merge('RGB', (bg, bg, bg))
        else:
            color = tuple(safe_randint(50, 200) for _ in range(3))
            img = Image.new('RGB', (W, H), color)
        draw = ImageDraw.Draw(img)
    return img, draw


class BongardRenderer:
    """Complete renderer for Bongard scene generation."""
    
    def __init__(self, canvas_size: int = 128, high_res_factor: int = 4):
        self.canvas_size = canvas_size
        self.high_res_factor = high_res_factor
        self.high_res_size = canvas_size * high_res_factor
        
        # Initialize shape drawer
        self.shape_drawer = ShapeDrawer()
        
        logger.info(f"Initialized BongardRenderer with canvas_size={canvas_size}")
    
    def render_scene(self, objects: List[Dict[str, Any]], 
                    canvas_size: Optional[int] = None,
                    background_color: str = 'white',
                    output_format: str = 'pil') -> Any:
        """
        Render a complete Bongard scene.
        
        Args:
            objects: List of object dictionaries with shape, position, size, etc.
            canvas_size: Override default canvas size
            background_color: Background color ('white' or 'black')
            output_format: 'pil' for PIL Image, 'numpy' for numpy array
            
        Returns:
            Rendered scene as PIL Image or numpy array
        """
        size = canvas_size or self.canvas_size
        
        # Create high-resolution canvas for anti-aliasing
        hr_size = size * self.high_res_factor
        
        # Initialize canvas
        if background_color == 'white':
            img = Image.new('L', (hr_size, hr_size), color=255)
        else:
            img = Image.new('L', (hr_size, hr_size), color=0)
        
        draw = ImageDraw.Draw(img)
        
        # Render each object
        for obj in objects:
            self._render_object(draw, obj, hr_size)
        
        # Downscale for final output
        img = img.resize((size, size), Image.LANCZOS)
        
        # Convert to requested format
        if output_format == 'numpy':
            return np.array(img)
        elif output_format == 'pil':
            return img
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _render_object(self, draw: ImageDraw.ImageDraw, obj: Dict[str, Any], canvas_size: int):
        """Render a single object on the canvas."""
        # Extract object properties
        shape = obj.get('shape', 'circle')
        position = obj.get('position', (canvas_size//2, canvas_size//2))
        size = obj.get('size', obj.get('width_pixels', 30))
        fill = obj.get('fill', 'solid')
        color = obj.get('color', 'black')
        orientation = obj.get('orientation', 0)
        
        # Scale position and size for high-resolution
        x, y = position
        x_hr = int(x * self.high_res_factor)
        y_hr = int(y * self.high_res_factor)
        size_hr = int(size * self.high_res_factor)
        
        # Convert color to grayscale value
        if color == 'black' or fill == 'solid':
            fill_color = 0  # Black
        else:
            fill_color = 0  # Default to black for now
        
        # Calculate bounding box
        half_size = size_hr // 2
        bbox = [x_hr - half_size, y_hr - half_size, x_hr + half_size, y_hr + half_size]
        
        # Render based on shape
        if shape == 'circle':
            if fill == 'solid':
                draw.ellipse(bbox, fill=fill_color)
            else:
                draw.ellipse(bbox, outline=fill_color, width=max(2, size_hr//10))
        
        elif shape == 'square' or shape == 'rectangle':
            if fill == 'solid':
                draw.rectangle(bbox, fill=fill_color)
            else:
                draw.rectangle(bbox, outline=fill_color, width=max(2, size_hr//10))
        
        elif shape == 'triangle':
            # Create triangle points
            points = [
                (x_hr, y_hr - half_size),  # Top
                (x_hr - half_size, y_hr + half_size),  # Bottom left
                (x_hr + half_size, y_hr + half_size)   # Bottom right
            ]
            
            if fill == 'solid':
                draw.polygon(points, fill=fill_color)
            else:
                draw.polygon(points, outline=fill_color, width=max(2, size_hr//10))
        
        elif shape == 'pentagon':
            # Create pentagon points
            points = []
            for i in range(5):
                angle = (2 * math.pi * i / 5) - (math.pi / 2)  # Start from top
                px = x_hr + half_size * math.cos(angle)
                py = y_hr + half_size * math.sin(angle)
                points.append((px, py))
            
            if fill == 'solid':
                draw.polygon(points, fill=fill_color)
            else:
                draw.polygon(points, outline=fill_color, width=max(2, size_hr//10))
        
        elif shape == 'hexagon':
            # Create hexagon points
            points = []
            for i in range(6):
                angle = (2 * math.pi * i / 6) - (math.pi / 2)  # Start from top
                px = x_hr + half_size * math.cos(angle)
                py = y_hr + half_size * math.sin(angle)
                points.append((px, py))
            
            if fill == 'solid':
                draw.polygon(points, fill=fill_color)
            else:
                draw.polygon(points, outline=fill_color, width=max(2, size_hr//10))
        
        else:
            # Default to circle for unknown shapes
            if fill == 'solid':
                draw.ellipse(bbox, fill=fill_color)
            else:
                draw.ellipse(bbox, outline=fill_color, width=max(2, size_hr//10))
    
    def render_scene_from_genome(self, genome, scene_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Render scene specifically from a SceneGenome."""
        canvas_size = genome.params.get('canvas_size', self.canvas_size)
        return self.render_scene(
            objects=scene_objects,
            canvas_size=canvas_size,
            background_color='white',
            output_format='numpy'
        )
