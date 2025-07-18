"""Utility functions for Bongard generator"""

import os
import json
import random
import math
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Global RNG for reproducibility
GLOBAL_SEED: int = 42
rng = np.random.default_rng(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

def set_global_seed(seed: int) -> None:
    """Set the global seed for reproducibility."""
    global GLOBAL_SEED, rng
    GLOBAL_SEED = seed
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

def iou(box_a: List[float], box_b: List[float]) -> float:
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    xa0, ya0, xa1, ya1 = box_a
    xb0, yb0, xb1, yb1 = box_b
    x0, y0 = max(xa0, xb0), max(ya0, yb0)
    x1, y1 = min(xa1, xb1), min(ya1, yb1)
    inter = max(0, x1-x0) * max(0, y1-y0)
    area_a = (xa1-xa0)*(ya1-ya0)
    area_b = (xb1-xb0)*(yb1-yb0)
    return inter / (area_a + area_b - inter + 1e-8)

def generate_perlin_noise(w: int, h: int, scale: int = 10) -> np.ndarray:
    """Generate simple Perlin-like noise."""
    noise_map: np.ndarray = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            noise_map[y, x] = random.uniform(0, 1)
    return noise_map

def find_coeffs(pa: List[Tuple[float, float]], pb: List[Tuple[float, float]]) -> Optional[List[float]]:
    """Find coefficients for perspective transform."""
    matrix: List[List[float]] = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0,0,0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0,0,0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    try:
        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return res.tolist()
    except np.linalg.LinAlgError:
        logger.warning("LinAlgError in find_coeffs, returning None for perspective transform.")
        return None

def make_linear_gradient(size: int, direction: str = "vertical") -> Image.Image:
    """Create a linear gradient mask."""
    mask = Image.new("L", (size, size))
    for i in range(size):
        val = int(255 * (i / (size - 1)))
        if direction == "vertical":
            for x in range(size):
                mask.putpixel((x, i), val)
        else:
            for y in range(size):
                mask.putpixel((i, y), val)
    return mask

def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float, float, float],
    dash: Tuple[int, int],
    fill: any = 255,
    width: int = 1
) -> None:
    """Draw a dashed line."""
    x0, y0, x1, y1 = xy
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist == 0: 
        return
    ux, uy = dx / dist, dy / dist
    dash_len, gap_len = dash
    pos = 0.0
    while pos < dist:
        start = pos
        end = min(pos + dash_len, dist)
        sx, sy = x0 + ux * start, y0 + uy * start
        ex, ey = x0 + ux * end,   y0 + uy * end
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += dash_len + gap_len

def draw_dashed_arc(
    draw: ImageDraw.ImageDraw,
    bbox: List[float],
    start: float,
    end: float,
    dash: Tuple[int,int],
    fill: any = 255,
    width: int = 1,
    steps: int = 200
) -> None:
    """Draw a dashed arc."""
    center_x = (bbox[0]+bbox[2])/2
    center_y = (bbox[1]+bbox[3])/2
    radius_x = (bbox[2]-bbox[0])/2
    radius_y = (bbox[3]-bbox[1])/2
    angles = np.linspace(start, end, steps)
    points = [
        (
            center_x + radius_x * math.cos(math.radians(a)),
            center_y + radius_y * math.sin(math.radians(a))
        )
        for a in angles
    ]
    for p0, p1 in zip(points, points[1:]):
        draw_dashed_line(draw, (p0[0], p0[1], p1[0], p1[1]), dash=dash, fill=fill, width=width)

def ensure_directories_exist() -> None:
    """Ensure necessary directories exist."""
    import os
    from .config_loader import CONFIG, DATA_ROOT_PATH
    
    directories = [
        DATA_ROOT_PATH,
        CONFIG['data']['synthetic_data_config'].get('background_texture_path', './data/textures'),
        CONFIG['data']['synthetic_data_config'].get('stamp_path', './data/stamps'),
        CONFIG['data']['synthetic_data_config'].get('program_library_path', './data/programs')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def create_dummy_files_if_missing() -> None:
    """Create dummy files for testing if real ones are not present."""
    import os
    import json
    from .config_loader import CONFIG
    
    # Create dummy textures
    texture_path = CONFIG['data']['synthetic_data_config'].get('background_texture_path', './data/textures')
    if not os.listdir(texture_path):
        logger.warning("No background textures found. Creating dummy textures.")
        Image.new('RGB', (128, 128), color=(100, 100, 100)).save(os.path.join(texture_path, 'texture1.png'))
        Image.new('RGB', (128, 128), color=(150, 150, 150)).save(os.path.join(texture_path, 'texture2.png'))
    
    # Create dummy stamps
    stamp_path = CONFIG['data']['synthetic_data_config'].get('stamp_path', './data/stamps')
    if not os.listdir(stamp_path):
        logger.warning("No stamps found. Creating dummy stamps.")
        Image.new('L', (32, 32), color=0).save(os.path.join(stamp_path, 'stamp1.png'))
        Image.new('L', (32, 32), color=255).save(os.path.join(stamp_path, 'stamp2.png'))
        Image.new('L', (32, 32), color=128).save(os.path.join(stamp_path, 'stamp3.png'))
    
    # Create dummy program library
    program_lib_path = CONFIG['data']['synthetic_data_config'].get('program_library_path', './data/programs')
    if not os.path.exists(os.path.join(program_lib_path, 'simple_program.json')):
        logger.warning("No program library found. Creating a dummy program file.")
        dummy_program_data = [
            {"type": "line", "dist": 50},
            {"type": "turn", "ang": 90},
            {"type": "line", "dist": 50}
        ]
        with open(os.path.join(program_lib_path, 'simple_program.json'), 'w') as f:
            json.dump(dummy_program_data, f)

def unnormalize(tensor, mean: List[float], std: List[float]):
    """Unnormalize tensor for display."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def set_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    set_global_seed(seed)

def add_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add noise to an image."""
    if noise_level <= 0:
        return image
    
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def create_gradient_background(size: int, colors: List[Tuple[int, int, int]] = None) -> Image.Image:
    """Create a gradient background."""
    if colors is None:
        colors = [(255, 255, 255), (240, 240, 240)]
    
    # Create a simple linear gradient
    img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)
    
    color1, color2 = colors[0], colors[-1]
    
    for y in range(size):
        ratio = y / size
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    
    return img
