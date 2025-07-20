"""
PrototypeAction system for integrating human-designed shapes from NVlabs repository.
Handles loading, transforming, and rendering of pre-designed silhouettes.
"""

import os
import random
import math
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageOps
from pathlib import Path
import numpy as np


class PrototypeAction:
    """
    Stamps one silhouette from human-designed shapes collection.
    Supports loading from various sources and applying transformations.
    """
    
    def __init__(self, shapes_dir: Optional[str] = None, fallback_shapes: bool = True):
        """Initialize prototype action with shapes directory.
        
        Args:
            shapes_dir: Path to directory containing PNG silhouettes
            fallback_shapes: If True, create procedural shapes when no directory found
        """
        self.shapes = []
        self.fallback_shapes = fallback_shapes
        
        # Try to load from shapes directory
        if shapes_dir and os.path.exists(shapes_dir):
            self._load_shapes_from_directory(shapes_dir)
        
        # If no shapes loaded and fallback enabled, create procedural shapes
        if not self.shapes and self.fallback_shapes:
            self._create_fallback_shapes()
        
        if not self.shapes:
            print(f"Warning: No prototype shapes loaded from {shapes_dir}")
    
    def _load_shapes_from_directory(self, shapes_dir: str):
        """Load PNG files from shapes directory."""
        shapes_path = Path(shapes_dir)
        
        for root, _, files in os.walk(shapes_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    try:
                        # Test if image can be loaded
                        test_img = Image.open(full_path)
                        test_img.close()
                        self.shapes.append(full_path)
                    except Exception as e:
                        print(f"Warning: Could not load {full_path}: {e}")
        
        print(f"Loaded {len(self.shapes)} prototype shapes from {shapes_dir}")
    
    def _create_fallback_shapes(self):
        """Create procedural fallback shapes when no directory is available."""
        # Generate simple procedural shapes as PIL Image objects
        fallback_templates = [
            self._create_arrow_shape,
            self._create_fish_shape,
            self._create_house_shape,
            self._create_flower_shape,
            self._create_butterfly_shape,
            self._create_tree_shape,
            self._create_car_shape,
            self._create_lamp_shape
        ]
        
        # Create template images
        for i, template_func in enumerate(fallback_templates):
            try:
                template_img = template_func()
                # Store as tuple (image, name) for fallback shapes
                self.shapes.append((template_img, f"fallback_{i}"))
            except Exception as e:
                print(f"Warning: Could not create fallback shape {i}: {e}")
        
        print(f"Created {len(self.shapes)} fallback prototype shapes")
    
    def sample(self) -> Image.Image:
        """Sample a random shape from the collection."""
        if not self.shapes:
            # Return simple circle as ultimate fallback
            img = Image.new('L', (64, 64), 0)
            draw = ImageDraw.Draw(img)
            draw.ellipse([8, 8, 56, 56], fill=255)
            return img
        
        shape_item = random.choice(self.shapes)
        
        # Handle file path (real shapes)
        if isinstance(shape_item, str):
            try:
                img = Image.open(shape_item).convert('L')
                return img
            except Exception as e:
                print(f"Warning: Could not load shape {shape_item}: {e}")
                return self._create_simple_fallback()
        
        # Handle tuple (fallback shapes)
        elif isinstance(shape_item, tuple):
            img, name = shape_item
            return img.copy()
        
        return self._create_simple_fallback()
    
    def _create_simple_fallback(self) -> Image.Image:
        """Create a simple fallback shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        
        # Random simple shape
        shape_type = random.choice(['circle', 'rectangle', 'triangle'])
        
        if shape_type == 'circle':
            draw.ellipse([8, 8, 56, 56], fill=255)
        elif shape_type == 'rectangle':
            draw.rectangle([12, 12, 52, 52], fill=255)
        elif shape_type == 'triangle':
            draw.polygon([(32, 8), (8, 56), (56, 56)], fill=255)
        
        return img
    
    def draw(self, img: Image.Image, center: Tuple[int, int], size: int, 
             config: Optional[Dict] = None) -> Image.Image:
        """
        Draw prototype shape onto the canvas.
        
        Args:
            img: Target RGB canvas
            center: (x, y) center position
            size: Target max dimension
            config: Configuration for transformations and effects
            
        Returns:
            Modified image
        """
        if config is None:
            config = {}
        
        # Sample and prepare prototype shape
        mask = self.sample()
        
        # Apply transformations
        mask = self._apply_transformations(mask, size, config)
        
        # Apply positioning and effects
        img = self._composite_shape(img, mask, center, config)
        
        return img
    
    def _apply_transformations(self, mask: Image.Image, target_size: int, 
                             config: Dict) -> Image.Image:
        """Apply size, rotation, flip transformations."""
        
        # 1) Resize to fit target_size while preserving aspect ratio
        w, h = mask.size
        if max(w, h) == 0:
            return mask
        
        # Ensure target_size is numeric
        if isinstance(target_size, str):
            try:
                target_size = int(target_size)
            except ValueError:
                target_size = 32  # fallback
        
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            mask = mask.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        
        # 2) Random flip
        if config.get('enable_flip', True):
            if random.random() < 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:  # Less frequent vertical flip
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # 3) Random rotation
        if config.get('enable_rotation', True):
            angle = random.uniform(0, 360)
            mask = mask.rotate(angle, expand=True, fillcolor=0)
        
        return mask
    
    def _composite_shape(self, img: Image.Image, mask: Image.Image, 
                        center: Tuple[int, int], config: Dict) -> Image.Image:
        """Composite the shape onto the target image."""
        
        x0, y0 = center
        
        # 4) Jitter position
        jitter_range = config.get('jitter_px', 5)
        if jitter_range > 0:
            jitter_x = x0 + random.uniform(-jitter_range, jitter_range)
            jitter_y = y0 + random.uniform(-jitter_range, jitter_range)
        else:
            jitter_x, jitter_y = x0, y0
        
        # 5) Convert mask to binary
        threshold = config.get('threshold', 128)
        bin_mask = mask.point(lambda p: 255 if p > threshold else 0, mode='1')
        
        # 6) Create shape with desired color/fill
        shape_color = config.get('fill_color', (0, 0, 0))  # Default black
        fill_pattern = config.get('fill_pattern', 'solid')
        
        # Create filled shape
        shape_img = self._create_filled_shape(mask, shape_color, fill_pattern, config)
        
        # 7) Calculate paste position
        paste_x = int(jitter_x - shape_img.width // 2)
        paste_y = int(jitter_y - shape_img.height // 2)
        
        # Ensure paste position is within bounds
        paste_x = max(0, min(paste_x, img.width - shape_img.width))
        paste_y = max(0, min(paste_y, img.height - shape_img.height))
        
        # 8) Composite onto target image
        try:
            img.paste(shape_img, (paste_x, paste_y), mask=bin_mask)
        except Exception as e:
            print(f"Warning: Could not composite shape: {e}")
        
        # 9) Optional stroke/outline
        if config.get('stroke_width', 0) > 0:
            img = self._add_stroke(img, bin_mask, (paste_x, paste_y), config)
        
        return img
    
    def _create_filled_shape(self, mask: Image.Image, color: Tuple[int, int, int], 
                            pattern: str, config: Dict) -> Image.Image:
        """Create filled shape with specified pattern."""
        
        shape_img = Image.new('RGB', mask.size, (255, 255, 255))  # White background
        
        if pattern == 'solid':
            shape_img = Image.new('RGB', mask.size, color)
        
        elif pattern == 'striped':
            stripe_width = config.get('stripe_width', 4)
            stripe_color2 = config.get('stripe_color2', (255, 255, 255))
            
            draw = ImageDraw.Draw(shape_img)
            for y in range(0, mask.height, stripe_width * 2):
                draw.rectangle([0, y, mask.width, y + stripe_width], fill=color)
                draw.rectangle([0, y + stripe_width, mask.width, y + stripe_width * 2], fill=stripe_color2)
        
        elif pattern == 'dotted':
            dot_spacing = config.get('dot_spacing', 8)
            dot_size = config.get('dot_size', 2)
            
            shape_img = Image.new('RGB', mask.size, (255, 255, 255))
            draw = ImageDraw.Draw(shape_img)
            
            for y in range(0, mask.height, dot_spacing):
                for x in range(0, mask.width, dot_spacing):
                    draw.ellipse([x - dot_size, y - dot_size, 
                                 x + dot_size, y + dot_size], fill=color)
        
        elif pattern == 'gradient':
            # Simple horizontal gradient
            for x in range(mask.width):
                ratio = x / mask.width if mask.width > 0 else 0
                r = int(color[0] * (1 - ratio) + 255 * ratio)
                g = int(color[1] * (1 - ratio) + 255 * ratio)  
                b = int(color[2] * (1 - ratio) + 255 * ratio)
                
                for y in range(mask.height):
                    shape_img.putpixel((x, y), (r, g, b))
        
        else:  # Default to solid
            shape_img = Image.new('RGB', mask.size, color)
        
        return shape_img
    
    def _add_stroke(self, img: Image.Image, mask: Image.Image, 
                   offset: Tuple[int, int], config: Dict) -> Image.Image:
        """Add stroke/outline to the shape."""
        
        stroke_width = config.get('stroke_width', 2)
        stroke_color = config.get('stroke_color', (0, 0, 0))
        stroke_style = config.get('stroke_style', 'solid')
        
        # Create stroke by dilating and subtracting original mask
        try:
            # Dilate mask to create outline
            dilate_size = stroke_width * 2
            kernel = Image.new('L', (dilate_size, dilate_size), 255)
            outline_mask = mask.filter(ImageFilter.MaxFilter(dilate_size))
            
            # Subtract original to get just the outline
            if mask.size == outline_mask.size:
                stroke_mask = ImageChops.subtract(outline_mask, mask)
                
                # Create stroke image
                stroke_img = Image.new('RGB', stroke_mask.size, stroke_color)
                
                # Composite stroke
                paste_x, paste_y = offset
                img.paste(stroke_img, (paste_x, paste_y), mask=stroke_mask)
        
        except Exception as e:
            print(f"Warning: Could not add stroke: {e}")
        
        return img
    
    # Fallback shape creation methods
    def _create_arrow_shape(self) -> Image.Image:
        """Create arrow shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Arrow pointing right
        points = [(8, 32), (40, 20), (40, 28), (56, 28), (56, 36), (40, 36), (40, 44)]
        draw.polygon(points, fill=255)
        return img
    
    def _create_fish_shape(self) -> Image.Image:
        """Create fish shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Simple fish body (ellipse) + tail
        draw.ellipse([8, 20, 40, 44], fill=255)  # Body
        draw.polygon([(40, 24), (56, 18), (56, 46), (40, 40)], fill=255)  # Tail
        draw.ellipse([32, 28, 36, 32], fill=0)  # Eye
        return img
    
    def _create_house_shape(self) -> Image.Image:
        """Create house shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # House body
        draw.rectangle([16, 32, 48, 56], fill=255)
        # Roof
        draw.polygon([(10, 32), (32, 12), (54, 32)], fill=255)
        # Door
        draw.rectangle([28, 44, 36, 56], fill=0)
        return img
    
    def _create_flower_shape(self) -> Image.Image:
        """Create flower shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Center
        draw.ellipse([28, 28, 36, 36], fill=255)
        # Petals
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x = 32 + 18 * math.cos(rad)
            y = 32 + 18 * math.sin(rad)
            draw.ellipse([x-4, y-4, x+4, y+4], fill=255)
        return img
    
    def _create_butterfly_shape(self) -> Image.Image:
        """Create butterfly shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Body
        draw.ellipse([30, 16, 34, 48], fill=255)
        # Wings
        draw.ellipse([8, 16, 30, 32], fill=255)   # Top left
        draw.ellipse([34, 16, 56, 32], fill=255)  # Top right
        draw.ellipse([12, 32, 30, 44], fill=255)  # Bottom left
        draw.ellipse([34, 32, 52, 44], fill=255)  # Bottom right
        return img
    
    def _create_tree_shape(self) -> Image.Image:
        """Create tree shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Trunk
        draw.rectangle([28, 36, 36, 56], fill=255)
        # Foliage (circle)
        draw.ellipse([16, 8, 48, 40], fill=255)
        return img
    
    def _create_car_shape(self) -> Image.Image:
        """Create car shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Car body
        draw.rectangle([8, 32, 56, 48], fill=255)
        # Car roof
        draw.rectangle([20, 20, 44, 32], fill=255)
        # Wheels
        draw.ellipse([12, 44, 20, 52], fill=255)
        draw.ellipse([44, 44, 52, 52], fill=255)
        return img
    
    def _create_lamp_shape(self) -> Image.Image:
        """Create lamp shape."""
        img = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(img)
        # Base
        draw.rectangle([24, 48, 40, 56], fill=255)
        # Pole
        draw.rectangle([30, 24, 34, 48], fill=255)
        # Shade
        draw.polygon([(20, 24), (44, 24), (40, 12), (24, 12)], fill=255)
        return img


# Integration functions
def create_prototype_config(jitter_px: int = 5, enable_rotation: bool = True, 
                          enable_flip: bool = True, fill_color: Tuple[int, int, int] = (0, 0, 0),
                          fill_pattern: str = 'solid', stroke_width: int = 0) -> Dict[str, Any]:
    """Create configuration for prototype action."""
    return {
        'jitter_px': jitter_px,
        'enable_rotation': enable_rotation,
        'enable_flip': enable_flip,
        'fill_color': fill_color,
        'fill_pattern': fill_pattern,
        'stroke_width': stroke_width,
        'stroke_color': (0, 0, 0),
        'stroke_style': 'solid',
        'threshold': 128,
        'stripe_width': 4,
        'dot_spacing': 8,
        'dot_size': 2
    }


# Factory function
def create_prototype_action(shapes_dir: Optional[str] = None, 
                           fallback_shapes: bool = True) -> PrototypeAction:
    """Factory function to create prototype action."""
    return PrototypeAction(shapes_dir, fallback_shapes)
