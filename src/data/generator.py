# Folder: bongard_solver/src/data/
# File: generator.py
import random
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2  # For potential image processing, e.g., converting PIL to OpenCV format
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import json
import math  # For geometric calculations
import glob  # Added for background textures loader

# Import global configuration and Bongard rules from the project root
try:
    from config import CONFIG, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, \
                         ATTRIBUTE_FILL_MAP, ATTRIBUTE_SIZE_MAP, \
                         ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP, \
                         RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS, DATA_ROOT_PATH
    from src.bongard_rules import ALL_BONGARD_RULES  # Assuming bongard_rules.py is in src/
    from src.utils.augment import augment_image  # Import augmentation function
    HAS_CONFIG = True
except ImportError as e:
    logging.error(f"Failed to import from config or src.bongard_rules/augment: {e}. Using dummy values.")
    CONFIG = {
        'data': {
            'image_size': [128, 128],
            'num_channels': 3,
            'synthetic_data_config': {
                'min_objects_per_image': 1,
                'max_objects_per_image': 3,
                'object_size_range': [20, 80],
                'padding': 10,
                'font_path': None,
                'max_support_images_per_problem': 5,
                'min_dist_objects': 0.1,  # Normalized minimum distance
                'relation_density': 0.5,
                'background_texture_path': './data/textures',  # Dummy path
                'num_positive_examples': 6,
                'num_negative_examples': 6,
                'occluder_prob': 0.3,
                'blur_prob': 0.2,
                'min_occluder_size': 5,
                'max_occluder_size': 20,
                'jitter_width_range': [1, 3],
                'jitter_dash_options': [None, (4,2), (2,2,2)],
                'use_background_textures': True # Added for dummy config to enable texture loading
            }
        },
        'training': {'augmentation_config': {}}
    }
    ATTRIBUTE_SHAPE_MAP = {'circle':0, 'square':1, 'triangle':2, 'pentagon':3, 'star':4, 'text_character':5}
    ATTRIBUTE_COLOR_MAP = {'red':0, 'blue':1, 'green':2, 'yellow':3, 'black':4, 'white':5}
    ATTRIBUTE_FILL_MAP = {'solid':0, 'outlined':1, 'striped':2, 'dotted':3}
    ATTRIBUTE_SIZE_MAP = {'small':0, 'medium':1, 'large':2}
    ATTRIBUTE_ORIENTATION_MAP = {'upright':0, 'rotated_45':1, 'rotated_90':2, 'rotated_135':3}
    ATTRIBUTE_TEXTURE_MAP = {'flat':0, 'rough':1, 'smooth':2}
    RELATION_MAP = {'above':0, 'left_of':1, 'inside':2, 'overlapping':3, 'touching':4, 'unrelated':5}
    ALL_BONGARD_RULES = []  # Empty list for dummy rules
    def augment_image(img): return img  # Dummy augment function
    IMAGENET_MEAN = [0.5, 0.5, 0.5]  # For RGB images
    IMAGENET_STD = [0.5, 0.5, 0.5]  # For RGB images
    NUM_CHANNELS = 3  # Default to 3 channels
    DATA_ROOT_PATH = "./data"
    HAS_CONFIG = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LogoGenerator(Dataset):
    """
    Generates synthetic Bongard-like problems with various attributes and relations.
    Supports domain randomization techniques like background textures, jitter, occluders, and blur.
    """
    def __init__(self, image_size, bg_textures_dir=None):
        """
        Args:
            image_size: target height & width for generated images.
            bg_textures_dir: optional directory of background textures.
        """
        self.size = image_size
        self.bg_textures_dir = bg_textures_dir
        self.bg_textures = []
        if bg_textures_dir:
            from PIL import Image
            import glob
            paths = glob.glob(f"{bg_textures_dir}/*.*")
            self.bg_textures = [Image.open(p).convert('L') for p in paths]

    def _randomize_canvas(self) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        """
        Randomizes the canvas by optionally applying a background texture or a random color.
        Returns a PIL Image and its ImageDraw object.
        """
        # Use a random background texture if available and enabled by config
        if self.bg_textures and self.cfg.get('use_background_textures', False) and random.random() < 0.2:
            bg = random.choice(self.bg_textures).copy()  # Use a copy to avoid modifying original texture
            return bg, ImageDraw.Draw(bg)
        
        # Otherwise, use a random solid color background
        # Ensure it's not too close to black or white to provide contrast for objects
        r, g, b = random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)
        img = Image.new('RGB', (self.canvas_width, self.canvas_height), (r, g, b))
        return img, ImageDraw.Draw(img)

    def _jitter_params(self) -> Dict[str, Any]:
        """Returns randomized drawing parameters for line width and dashing."""
        return {
            'width': random.choice(self.cfg.get('jitter_width_range', [1,2,3])),
            'dash': random.choice(self.cfg.get('jitter_dash_options', [None, (4,2), (2,2,2)]))
        }

    def _apply_occluder(self, img: Image.Image) -> Image.Image:
        """Applies random rectangular or elliptical occluders to the image."""
        draw = ImageDraw.Draw(img)
        num_occluders = random.randint(0, self.cfg.get('max_occluders', 3))
        min_size = self.cfg.get('min_occluder_size', 5)
        max_size = self.cfg.get('max_occluder_size', 20)
        for _ in range(num_occluders):
            shape = random.choice(['rect', 'ellipse'])
            x0, y0 = random.uniform(0, self.canvas_width), random.uniform(0, self.canvas_height)
            x1, y1 = x0 + random.uniform(min_size, max_size), y0 + random.uniform(min_size, max_size)
            
            # Randomly choose occluder color (e.g., white or a random background-like color)
            fill_color = random.choice([(255, 255, 255), (200, 200, 200), (150, 150, 150)])
            
            if shape == 'rect':
                draw.rectangle([x0, y0, x1, y1], fill=fill_color)
            else:
                draw.ellipse([x0, y0, x1, y1], fill=fill_color)
        return img

    def draw_shape(self, draw: ImageDraw.ImageDraw, shape: str, size: int, pos: Tuple[int, int], fill_color_rgb: Tuple[int, int, int], outline_color_rgb: Tuple[int, int, int] = (0,0,0), fill_type: str = 'solid'):
        """
        Draws a shape with randomized parameters.
        Args:
            draw (ImageDraw.ImageDraw): The drawing context.
            shape (str): Type of shape (e.g., 'triangle', 'square', 'circle').
            size (int): Size of the shape in pixels.
            pos (Tuple[int, int]): Center (x, y) position of the shape.
            fill_color_rgb (Tuple[int, int, int]): RGB tuple for fill color.
            outline_color_rgb (Tuple[int, int, int]): RGB tuple for outline color.
            fill_type (str): 'solid', 'outlined', 'striped', 'dotted'.
        """
        x, y = pos
        params = self._jitter_params()
        line_width = params['width']
        dash_pattern = params['dash'] # Not directly used by PIL's basic drawing functions for polygons/rectangles

        # Create a temporary blank image for drawing the object, then rotate and paste
        # Make the temporary canvas large enough to accommodate rotation without clipping
        temp_canvas_size = int(size * 2.5)  # Larger buffer for rotated shapes
        obj_canvas = Image.new('RGBA', (temp_canvas_size, temp_canvas_size), (0, 0, 0, 0))  # Transparent background
        obj_draw = ImageDraw.Draw(obj_canvas)
        
        # Calculate bbox relative to obj_canvas center
        center_offset = temp_canvas_size // 2
        half_size = size // 2
        bbox_rel = [center_offset - half_size, center_offset - half_size, center_offset + half_size, center_offset + half_size]
        current_fill_color = fill_color_rgb if fill_type == 'solid' else None

        if shape == 'triangle':
            h = size * (3**0.5 / 2)
            pts = [(center_offset, center_offset - h / 2),
                   (center_offset - size / 2, center_offset + h / 2),
                   (center_offset + size / 2, center_offset + h / 2)]
            obj_draw.polygon(pts, outline=outline_color_rgb, fill=current_fill_color, width=line_width)
        elif shape == 'square':
            obj_draw.rectangle(bbox_rel, outline=outline_color_rgb, fill=current_fill_color, width=line_width)
        elif shape == 'circle':
            obj_draw.ellipse(bbox_rel, outline=outline_color_rgb, fill=current_fill_color, width=line_width)
        elif shape == 'pentagon':
            points = []
            for i in range(5):
                angle = math.radians(i * 72 - 90)  # Start from top
                px = center_offset + half_size * math.cos(angle)
                py = center_offset + half_size * math.sin(angle)
                points.append((px, py))
            obj_draw.polygon(points, outline=outline_color_rgb, fill=current_fill_color, width=line_width)
        elif shape == 'star':
            points = []
            outer_radius = half_size
            inner_radius = outer_radius * 0.4
            for i in range(5):
                angle = math.radians(i * 72 - 90)
                px_outer = center_offset + outer_radius * math.cos(angle)
                py_outer = center_offset + outer_radius * math.sin(angle)
                points.append((px_outer, py_outer))
                angle_inner = math.radians(i * 72 - 90 + 36)
                px_inner = center_offset + inner_radius * math.cos(angle_inner)
                py_inner = center_offset + inner_radius * math.sin(angle_inner)
                points.append((px_inner, py_inner))
            obj_draw.polygon(points, outline=outline_color_rgb, fill=current_fill_color, width=line_width)
        elif shape == 'text_character' and self.font:
            char = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            current_font_size = int(size * 0.7)
            font_to_use = ImageFont.truetype(self.cfg['font_path'], size=current_font_size) if self.cfg['font_path'] else ImageFont.load_default()
            # textbbox is preferred over textsize for newer PIL versions
            try:
                # PIL 9.2.0+
                left, top, right, bottom = obj_draw.textbbox((0, 0), char, font=font_to_use)
                text_width, text_height = right - left, bottom - top
            except AttributeError:
                # Older PIL versions
                text_width, text_height = obj_draw.textsize(char, font=font_to_use)

            text_x = center_offset - text_width // 2
            text_y = center_offset - text_height // 2
            obj_draw.text((text_x, text_y), char, font=font_to_use, fill=fill_color_rgb)
        else:
            logger.warning(f"Unsupported shape type for drawing: {shape}. Drawing a square as fallback.")
            obj_draw.rectangle(bbox_rel, outline=outline_color_rgb, fill=current_fill_color, width=line_width)

        # Apply fill pattern if not 'solid' or 'outlined'
        if fill_type == 'striped':
            for y_stripe in range(0, temp_canvas_size, 5):  # 5 pixel wide stripes
                obj_draw.line([(0, y_stripe), (temp_canvas_size, y_stripe)], fill=fill_color_rgb, width=2)
        elif fill_type == 'dotted':
            for x_dot in range(0, temp_canvas_size, 10):
                for y_dot in range(0, temp_canvas_size, 10):
                    obj_draw.ellipse([x_dot-1, y_dot-1, x_dot+1, y_dot+1], fill=fill_color_rgb)

        # Paste the object onto the main image
        paste_x = x - obj_canvas.width // 2
        paste_y = y - obj_canvas.height // 2
        draw.paste(obj_canvas, (paste_x, paste_y), obj_canvas)  # Use RGBA mask for transparency

    def _get_random_color(self) -> Tuple[int, int, int]:
        """Returns a random RGB color from the defined map."""
        return random.choice(list(self.color_rgb_map.values()))

    def _get_random_position(self, obj_size: int, existing_bboxes: List[List[int]]) -> Optional[Tuple[int, int]]:
        """
        Returns a random (x, y) position for an object of given size,
        ensuring it doesn't overlap significantly with existing objects.
        Returns None if no suitable position is found after several attempts.
        """
        attempts = 0
        max_attempts = 100
        min_dist_pixels = int(self.min_dist_objects_normalized * max(self.canvas_width, self.canvas_height))
        
        while attempts < max_attempts:
            x = random.randint(self.padding + obj_size // 2, self.canvas_width - self.padding - obj_size // 2)
            y = random.randint(self.padding + obj_size // 2, self.canvas_height - self.padding - obj_size // 2)
            
            new_bbox = [x - obj_size // 2, y - obj_size // 2, x + obj_size // 2, y + obj_size // 2]
            
            overlap = False
            for existing_bbox in existing_bboxes:
                center_x_new, center_y_new = x, y
                center_x_exist = (existing_bbox[0] + existing_bbox[2]) / 2
                center_y_exist = (existing_bbox[1] + existing_bbox[3]) / 2
                
                dist = np.sqrt((center_x_new - center_x_exist)**2 + (center_y_new - center_y_exist)**2)
                
                size_exist = max(existing_bbox[2] - existing_bbox[0], existing_bbox[3] - existing_bbox[1])
                min_allowed_dist = (obj_size + size_exist) / 2 + min_dist_pixels
                
                if dist < min_allowed_dist:
                    overlap = True
                    break
            
            if not overlap:
                return x, y
            attempts += 1
        
        logger.warning(f"Could not find non-overlapping position after {max_attempts} attempts. Returning None.")
        return None

    def _generate_scene_graph(self, objects_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a simplified scene graph from object data, including relations.
        """
        scene_graph = {'objects': [], 'relations': []}
        
        # Add objects to scene graph
        for i, obj in enumerate(objects_data):
            # Convert pixel position to normalized position (0-1)
            norm_x = obj['position'][0] / self.canvas_width
            norm_y = obj['position'][1] / self.canvas_height
            
            # Convert pixel size to normalized size
            norm_size = obj['size_pixels'] / max(self.canvas_width, self.canvas_height)
            # Bounding box (x_min, y_min, x_max, y_max)
            bbox = [
                obj['position'][0] - obj['size_pixels'] // 2,
                obj['position'][1] - obj['size_pixels'] // 2,
                obj['position'][0] + obj['size_pixels'] // 2,
                obj['position'][1] + obj['size_pixels'] // 2
            ]
            scene_graph['objects'].append({
                'id': f"obj_{i}",
                'bbox': bbox,  # Pixel coordinates
                'attributes': {
                    'shape': obj['shape'],
                    'color': obj['color'],
                    'size': obj['size'],  # Symbolic size
                    'fill': obj['fill'],
                    'orientation': obj['orientation'],
                    'texture': obj['texture'],
                    'position_x_norm': norm_x,
                    'position_y_norm': norm_y,
                    'size_norm': norm_size
                },
                'confidence': 1.0  # Ground truth has 1.0 confidence
            })
        
        # Add relations based on spatial properties and random chance
        for i in range(len(objects_data)):
            for j in range(len(objects_data)):
                if i == j:
                    continue  # No self-relations
                
                obj1 = objects_data[i]
                obj2 = objects_data[j]
                
                # Calculate relative positions
                x1, y1 = obj1['position']
                s1 = obj1['size_pixels']
                x2, y2 = obj2['position']
                s2 = obj2['size_pixels']

                # Simple spatial relations
                if x1 < x2 and random.random() < self.relation_density:
                    scene_graph['relations'].append({
                        'subject_id': f"obj_{i}",
                        'object_id': f"obj_{j}",
                        'type': 'left_of',
                        'confidence': 1.0
                    })
                if y1 < y2 and random.random() < self.relation_density:
                    scene_graph['relations'].append({
                        'subject_id': f"obj_{i}",
                        'object_id': f"obj_{j}",
                        'type': 'above',
                        'confidence': 1.0
                    })
                
                # Check for 'inside' (simplified: center of obj2 is within obj1's bbox)
                if (x2 > (x1 - s1/2) and x2 < (x1 + s1/2) and
                    y2 > (y1 - s1/2) and y2 < (y1 + s1/2) and
                    s2 < s1 and random.random() < self.relation_density):
                    scene_graph['relations'].append({
                        'subject_id': f"obj_{j}",  # obj2 is inside obj1
                        'object_id': f"obj_{i}",
                        'type': 'inside',
                        'confidence': 1.0
                    })
                
                # Check for 'overlapping' (simplified: bboxes intersect significantly)
                bbox1 = [x1 - s1/2, y1 - s1/2, x1 + s1/2, y1 + s1/2]
                bbox2 = [x2 - s2/2, y2 - s2/2, x2 + s2/2, y2 + s2/2]
                
                x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
                y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
                intersection_area = x_overlap * y_overlap
                
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                
                union_area = area1 + area2 - intersection_area
                
                if union_area > 0 and (intersection_area / union_area) > 0.1 and random.random() < self.relation_density:  # IoU > 0.1
                    scene_graph['relations'].append({
                        'subject_id': f"obj_{i}",
                        'object_id': f"obj_{j}",
                        'type': 'overlapping',
                        'confidence': 1.0
                    })
        
        return scene_graph

    def _generate_image_and_sg(self, num_objects: int, target_rule: Optional[Any] = None, is_positive: bool = True) -> Tuple[np.ndarray, Dict[str, Any], float]:
        """
        Generates a single image with objects and its corresponding scene graph.
        Attempts to make the image satisfy/not satisfy the target_rule based on is_positive.
        Returns the image (numpy array), scene graph, and a confidence score for the image.
        The confidence score reflects the amount of randomization applied.
        """
        img, draw = self._randomize_canvas()  # Use randomized canvas
        
        objects_data = []
        existing_bboxes = []  # To keep track of drawn object bounding boxes for overlap check

        # Calculate base confidence for the image (starts high, reduced by randomization)
        image_confidence = 1.0
        if self.bg_textures and self.cfg.get('use_background_textures', False) and random.random() < 0.2:  # If background texture was used
            image_confidence *= 0.9  # Slightly reduce confidence for more complex backgrounds

        for i in range(num_objects):
            obj_size_pixels = random.randint(int(self.min_obj_size_pixels), int(self.max_obj_size_pixels))
            pos = self._get_random_position(obj_size_pixels, existing_bboxes)
            if pos is None:
                logger.warning(f"Could not place object {i}. Skipping this object.")
                image_confidence *= 0.9  # Reduce confidence if object placement failed
                continue  # Skip if no suitable position found

            # Randomly select attributes
            shape = random.choice(self.all_shapes)
            color_name = random.choice(self.all_colors)
            color_rgb = self.color_rgb_map.get(color_name, (128, 128, 128))
            fill = random.choice(self.all_fills)
            size_symbolic = random.choice(self.all_sizes)  # Symbolic size
            orientation = random.choice(self.all_orientations)
            texture = random.choice(self.all_textures)

            # Enforce rule compliance/violation (simplified logic)
            if target_rule:
                if is_positive:  # Try to make it satisfy the rule
                    if hasattr(target_rule, 'positive_features') and target_rule.positive_features:
                        for feat, val in target_rule.positive_features.items():
                            if feat == 'shape' and val in self.all_shapes: shape = val
                            elif feat == 'color' and val in self.all_colors: color_name = val; color_rgb = self.color_rgb_map.get(color_name, (128, 128, 128))
                            elif feat == 'fill' and val in self.all_fills: fill = val
                            elif feat == 'size' and val in self.all_sizes: size_symbolic = val
                            elif feat == 'orientation' and val in self.all_orientations: orientation = val
                            elif feat == 'texture' and val in self.all_textures: texture = val
                else:  # Try to make it violate the rule
                    if hasattr(target_rule, 'negative_features') and target_rule.negative_features:
                        for feat, val in target_rule.negative_features.items():
                            if feat == 'shape' and val in self.all_shapes: shape = val
                            elif feat == 'color' and val in self.all_colors: color_name = val; color_rgb = self.color_rgb_map.get(color_name, (128, 128, 128))
                            elif feat == 'fill' and val in self.all_fills: fill = val
                            elif feat == 'size' and val in self.all_sizes: size_symbolic = val
                            elif feat == 'orientation' and val in self.all_orientations: orientation = val
                            elif feat == 'texture' and val in self.all_textures: texture = val
                    else:  # If no specific negative features, just randomize
                        pass

            objects_data.append({
                'shape': shape, 'color': color_name, 'fill': fill, 'size': size_symbolic,
                'orientation': orientation, 'texture': texture,
                'position': pos, 'size_pixels': obj_size_pixels, 'color_rgb': color_rgb
            })
            
            # Draw the object with jittered parameters
            self.draw_shape(draw, shape, obj_size_pixels, pos, color_rgb, fill_type=fill)
            
            # Update existing_bboxes
            existing_bboxes.append([
                pos[0] - obj_size_pixels // 2,
                pos[1] - obj_size_pixels // 2,
                pos[0] + obj_size_pixels // 2,
                pos[1] + obj_size_pixels // 2
            ])

        # Apply occluders
        if random.random() < self.cfg.get('occluder_prob', 0.3):
            img = self._apply_occluder(img)
            image_confidence *= 0.9  # Reduce confidence for occlusions

        # Apply optional blur
        if random.random() < self.cfg.get('blur_prob', 0.2):
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            image_confidence *= 0.95  # Slightly reduce confidence for blur
        
        # Apply augmentations after drawing and other randomization
        augmented_img_np = augment_image(np.array(img))  # augment_image expects numpy array
        
        # Generate scene graph from the generated objects
        scene_graph = self._generate_scene_graph(objects_data)
        
        # Clamp confidence between 0 and 1
        image_confidence = max(0.0, min(1.0, image_confidence))
        return augmented_img_np, scene_graph, image_confidence

    def sample_rule(self):
        """Return a random rule object or dummy if not available."""
        if not ALL_BONGARD_RULES:
            # Dummy rule
            rule_obj = type('Rule', (object,), {
                'description': "SHAPE(TRIANGLE)",
                'positive_features': {'shape': 'triangle'},
                'negative_features': {'shape': 'square'},
                'is_positive_rule': True
            })()
        else:
            rule_obj = random.choice(ALL_BONGARD_RULES)
        return rule_obj

    def sample(self, rule_feat, rule_val):
        """Draw positives, negatives, and return gt_rule string."""
        gt_rule = f"{rule_feat.upper()}({rule_val.upper()})"
        # For demonstration, use make_problem to generate examples
        positives, negatives = [], []
        rule_obj = self.sample_rule() # This line was originally `self .sample_rule()`
        for _ in range(self.cfg.get('num_positive_examples', 6)):
            # make_problem returns a large tuple, we only need the image for this sample function
            img_np, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.make_problem(random.randint(0, 10000))
            positives.append(img_np)
        for _ in range(self.cfg.get('num_negative_examples', 6)):
            # make_problem returns a large tuple, we only need the image for this sample function
            _, img_np, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.make_problem(random.randint(0, 10000))
            negatives.append(img_np)
        return positives, negatives, gt_rule

    def make_problem(self, problem_id: int) -> Tuple[Any, ...]:
        """
        Generates a complete Bongard problem (positive and negative examples)
        and its ground-truth DSL rule.
        """
        # Randomly select a rule from ALL_BONGARD_RULES
        if not ALL_BONGARD_RULES:
            logger.warning("No Bongard rules defined in src.bongard_rules. Using a dummy rule.")
            # Fallback to a simple dummy rule
            rule_obj = type('Rule', (object,), {
                'description': "SHAPE(TRIANGLE)",
                'positive_features': {'shape': 'triangle'},
                'negative_features': {'shape': 'square'},
                'is_positive_rule': True  # Needed for SymbolicConsistencyLoss
            })()
        else:
            rule_obj = random.choice(ALL_BONGARD_RULES)
        
        gt_rule_description = rule_obj.description
        
        # Generate positive examples
        positive_imgs_np = []
        positive_sgs = []
        positive_confs = []
        for _ in range(self.cfg.get('num_positive_examples', 6)):  # Default 6 positive examples
            num_objects = random.randint(self.cfg['min_objects_per_image'], self.cfg['max_objects_per_image'])
            img_np, sg, conf = self._generate_image_and_sg(num_objects, target_rule=rule_obj, is_positive=True)
            positive_imgs_np.append(img_np)
            positive_sgs.append(sg)
            positive_confs.append(conf)
        
        # Generate negative examples
        negative_imgs_np = []
        negative_sgs = []
        negative_confs = []
        for _ in range(self.cfg.get('num_negative_examples', 6)):  # Default 6 negative examples
            num_objects = random.randint(self.cfg['min_objects_per_image'], self.cfg['max_objects_per_image'])
            img_np, sg, conf = self._generate_image_and_sg(num_objects, target_rule=rule_obj, is_positive=False)  # Ensure it violates the rule
            negative_imgs_np.append(img_np)
            negative_sgs.append(sg)
            negative_confs.append(conf)
        
        # Select query images (e.g., first positive and first negative)
        query_img1_np = positive_imgs_np[0]
        query_img2_np = negative_imgs_np[0]
        query_conf1 = positive_confs[0]
        query_conf2 = negative_confs[0]
        
        # The label for a Bongard problem is typically 1 (meaning the rule holds for the positive set, not the negative)
        problem_label = 1
        
        gt_json_view1 = json.dumps(positive_sgs[0]).encode('utf-8')
        gt_json_view2 = json.dumps(negative_sgs[0]).encode('utf-8')
        
        difficulty = random.random()  # Dummy difficulty
        affine1 = np.eye(3).tolist()  # Dummy affine
        affine2 = np.eye(3).tolist()  # Dummy affine
        original_index = problem_id  # Use problem_id as original_index
        
        # Support images: remaining positive and negative examples
        # Flatten the list of support images and their corresponding labels/SGs/confidences
        support_imgs_flat_np = positive_imgs_np[1:] + negative_imgs_np[1:]
        support_labels_flat = [1] * (len(positive_imgs_np) - 1) + [0] * (len(negative_imgs_np) - 1)
        support_sgs_flat_bytes = [json.dumps(sg).encode('utf-8') for sg in positive_sgs[1:] + negative_sgs[1:]]
        support_confs_flat = positive_confs[1:] + negative_confs[1:]

        # Pad support images/labels/SGs/confidences to max_support_images_per_problem
        max_support = self.cfg['max_support_images_per_problem']
        num_actual_support = len(support_imgs_flat_np)
        
        while len(support_imgs_flat_np) < max_support:
            # Create a blank image for padding
            blank_img = np.zeros((self.canvas_height, self.canvas_width, NUM_CHANNELS), dtype=np.uint8)
            support_imgs_flat_np.append(blank_img)
            support_labels_flat.append(-1)  # Use -1 for padding labels
            support_sgs_flat_bytes.append(json.dumps({'objects': [], 'relations': []}).encode('utf-8'))
            support_confs_flat.append(0.0)  # Padding confidence is 0

        # Convert to tensors (these will be converted to tensors in __getitem__ of LogoDataset)
        # For make_problem's direct return, keep them as Python types for now, or convert if needed by caller.
        # The original snippet had them as tensors here, so we will keep them as tensors for consistency.
        num_support_per_problem_tensor = torch.tensor(num_actual_support, dtype=torch.long)
        tree_indices_tensor = torch.tensor(original_index, dtype=torch.long)
        is_weights_tensor = torch.tensor(1.0, dtype=torch.float)  # This will be updated by replay buffer

        # Dummy detected bboxes and masks for query images and support images.
        dummy_query_bboxes_view1 = [[]]
        dummy_query_masks_view1 = [[]]
        dummy_query_bboxes_view2 = [[]]
        dummy_query_masks_view2 = [[]]
        dummy_support_bboxes_flat = [[] for _ in range(max_support)]
        dummy_support_masks_flat = [[] for _ in range(max_support)]

        # Emit ground-truth DSL rule
        # The gt_rule_description is already captured from rule_obj.description
        
        return (query_img1_np, query_img2_np, problem_label,
                gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index,
                support_imgs_flat_np, support_labels_flat, support_sgs_flat_bytes,
                num_support_per_problem_tensor, tree_indices_tensor, is_weights_tensor,
                dummy_query_bboxes_view1, dummy_query_masks_view1,
                dummy_query_bboxes_view2, dummy_query_masks_view2,
                dummy_support_bboxes_flat, dummy_support_masks_flat,
                query_conf1, query_conf2, support_confs_flat, gt_rule_description)  # Added confidences and gt_rule

class LogoDataset(Dataset):
    """
    A PyTorch Dataset for generated Bongard-LOGO problems.
    """
    def __init__(self, cfg: Dict[str, Any], num_samples: int, transform: Optional[Any] = None):
        self.cfg = cfg
        self.num_samples = num_samples
        self.transform = transform
        
        # Pass the background_texture_path from config to LogoGenerator
        bg_textures_dir = cfg['data']['synthetic_data_config'].get('background_texture_path', './data/textures')
        self.generator = LogoGenerator(cfg, bg_textures_dir=bg_textures_dir)  # Pass the directory
        logger.info(f"LogoDataset initialized to generate {num_samples} problems.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: Union[int, Tuple[int, float]]) -> Tuple[Any, ...]:
        if isinstance(idx, tuple):
            original_idx, is_weight = idx
        else:
            original_idx = idx
            is_weight = 1.0
        
        problem_data = self.generator.make_problem(original_idx)
        
        # Convert problem_data (tuple) to list to allow modification
        problem_list = list(problem_data)
        
        # Apply transform to images if provided
        if self.transform is not None:
            # query_img1_np (index 0) and query_img2_np (index 1)
            # transform expects PIL Image, so convert from numpy array
            problem_list[0] = self.transform(Image.fromarray(problem_list[0]))
            problem_list[1] = self.transform(Image.fromarray(problem_list[1]))
            
            # padded_support_imgs_np (index 9) - list of numpy arrays
            transformed_support_imgs = []
            for img_np in problem_list[9]:
                transformed_support_imgs.append(self.transform(Image.fromarray(img_np)))
            problem_list[9] = torch.stack(transformed_support_imgs)  # Stack into a single tensor
        else:
            # If no transform, convert numpy arrays to tensors for consistency with Dataloader
            # Ensure images are converted to float and normalized to [0,1] before permuting
            problem_list[0] = torch.from_numpy(problem_list[0]).permute(2,0,1).float() / 255.0  # HWC to CHW, normalize
            problem_list[1] = torch.from_numpy(problem_list[1]).permute(2,0,1).float() / 255.0
            transformed_support_imgs = []
            for img_np in problem_list[9]:
                transformed_support_imgs.append(torch.from_numpy(img_np).permute(2,0,1).float() / 255.0)
            problem_list[9] = torch.stack(transformed_support_imgs)

        # Update the is_weights with the sampler's IS weight
        problem_list[14] = torch.tensor(is_weight, dtype=torch.float) 
        problem_list[8] = original_idx  # Ensure original_index is correct (used by collate_fn for tree_indices)
        problem_list[13] = torch.tensor(original_idx, dtype=torch.long)  # tree_indices maps to original_index in replay buffer
        
        # Convert confidence values to tensors
        problem_list[21] = torch.tensor(problem_list[21], dtype=torch.float)  # query_conf1
        problem_list[22] = torch.tensor(problem_list[22], dtype=torch.float)  # query_conf2
        problem_list[23] = torch.tensor(problem_list[23], dtype=torch.float)  # support_confs_flat
        return tuple(problem_list)

if __name__ == '__main__':
    # Example usage of LogoGenerator
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Dummy CONFIG if not imported
    if not HAS_CONFIG:
        logger.warning("Using dummy CONFIG for LogoGenerator example.")
        # Ensure CONFIG has necessary structure for LogoGenerator
        CONFIG = {
            'data': {
                'image_size': [128, 128],
                'num_channels': 3,
                'synthetic_data_config': {
                    'min_objects_per_image': 1,
                    'max_objects_per_image': 3,
                    'object_size_range': [20, 80],
                    'padding': 10,
                    'font_path': None,
                    'max_support_images_per_problem': 5,
                    'min_dist_objects': 0.1,
                    'relation_density': 0.5,
                    'background_texture_path': './data/textures',  # Dummy path
                    'num_positive_examples': 6,
                    'num_negative_examples': 6,
                    'occluder_prob': 0.3,
                    'blur_prob': 0.2,
                    'min_occluder_size': 5,
                    'max_occluder_size': 20,
                    'jitter_width_range': [1, 3],
                    'jitter_dash_options': [None, (4,2), (2,2,2)],
                    'use_background_textures': True  # Enable for dummy test
                }
            },
            'training': {'augmentation_config': {}}
        }
        # Populate dummy ALL_BONGARD_RULES for testing
        class DummyBongardRule:
            def __init__(self, description, positive_features, negative_features, is_positive_rule=True):
                self.description = description
                self.positive_features = positive_features
                self.negative_features = negative_features
                self.is_positive_rule = is_positive_rule

        ALL_BONGARD_RULES = [
            DummyBongardRule("SHAPE(TRIANGLE)", {'shape': 'triangle'}, {'shape': 'square'}),
            DummyBongardRule("COLOR(RED)", {'color': 'red'}, {'color': 'blue'}),
            DummyBongardRule("FILL(SOLID)", {'fill': 'solid'}, {'fill': 'outlined'})
        ]
        # Create dummy texture directory and files for testing
        os.makedirs('./data/textures', exist_ok=True)
        Image.new('RGB', (128, 128), color = (100, 100, 100)).save('./data/textures/texture1.png')
        Image.new('RGB', (128, 128), color = (150, 150, 150)).save('./data/textures/texture2.png')

    generator = LogoGenerator(CONFIG, bg_textures_dir=CONFIG['data']['synthetic_data_config']['background_texture_path'])
    
    logger.info("Generating a sample Bongard problem...")
    problem_data = generator.make_problem(problem_id=0)
    
    query_img1_np, query_img2_np, problem_label, gt_json_view1, gt_json_view2, \
    difficulty, affine1, affine2, original_index, \
    support_imgs_flat_np, support_labels_flat, support_sgs_flat_bytes, \
    num_support_per_problem_tensor, tree_indices_tensor, is_weights_tensor, \
    dummy_query_bboxes_view1, dummy_query_masks_view1, \
    dummy_query_bboxes_view2, dummy_query_masks_view2, \
    dummy_support_bboxes_flat, dummy_support_masks_flat, \
    query_conf1, query_conf2, support_confs_flat, gt_rule_description = problem_data

    logger.info(f"Problem ID: {original_index}, Label: {problem_label}, Difficulty: {difficulty:.2f}")
    logger.info(f"GT Rule Description: {gt_rule_description}")
    logger.info(f"Query Image 1 shape: {query_img1_np.shape}, Confidence: {query_conf1:.4f}")
    logger.info(f"Query Image 2 shape: {query_img2_np.shape}, Confidence: {query_conf2:.4f}")
    logger.info(f"GT Scene Graph 1 (bytes): {gt_json_view1[:100]}...")
    logger.info(f"GT Scene Graph 2 (bytes): {gt_json_view2[:100]}...")
    logger.info(f"Number of support images: {num_support_per_problem_tensor.item()}")
    logger.info(f"Support image confidences (first 5): {support_confs_flat[:5]}")
    
    # Display images (optional, requires matplotlib/cv2)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(query_img1_np)
        axes[0].set_title(f"Query Image 1 (Positive)")
        axes[0].axis('off')
        axes[1].imshow(query_img2_np)
        axes[1].set_title(f"Query Image 2 (Negative)")
        axes[1].axis('off')
        plt.suptitle(f"Bongard Problem {original_index} (Label: {problem_label})")
        plt.show()

        # Display some support images
        if num_support_per_problem_tensor.item() > 0:
            num_display = min(4, num_support_per_problem_tensor.item())
            fig_sup, axes_sup = plt.subplots(1, num_display, figsize=(num_display * 3, 3))
            for i in range(num_display):
                ax = axes_sup[i] if num_display > 1 else axes_sup
                ax.imshow(support_imgs_flat_np[i])
                ax.set_title(f"Support {i+1} (Label: {support_labels_flat[i]})")
                ax.axis('off')
            plt.suptitle("Support Images")
            plt.show()
    except ImportError:
        logger.warning("Matplotlib not found. Skipping image display.")

    # Test LogoDataset
    logger.info("Testing LogoDataset...")
    dataset = LogoDataset(CONFIG, num_samples=2)
    for i in range(len(dataset)):
        problem = dataset[i]
        logger.info(f"Dataset item {i}: Query1 shape {problem[0].shape}, Query2 shape {problem[1].shape}")
        logger.info(f"Dataset item {i}: Support images shape {problem[9].shape}")
        logger.info(f"Dataset item {i}: Query1 confidence {problem[21].item():.4f}, Query2 confidence {problem[22].item():.4f}")
        logger.info(f"Dataset item {i}: Support confidences (first 5) {problem[23][:5].tolist()}")
        logger.info(f"Dataset item {i}: GT Rule: {problem[24]}")  # Accessing the new gt_rule_description
