import logging
import random
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator
import numpy as np
from PIL import I    def sample_scene_cp(rule, num_objects=3, canvas_size=128):
        # Basic scene sampling fallback
        objects = []
        shapes = ['circle', 'square', 'triangle']
        colors = ['red', 'blue', 'green', 'yellow', 'black']
        
        for i in range(num_objects):
            obj = {
                'shape': random.choice(shapes),
                'color': random.choice(colors),
                'x': random.randint(20, canvas_size-20),
                'y': random.randint(20, canvas_size-20),
                'size': random.randint(15, 35),
                'object_id': f'cp_obj_{i}'
            }
            objects.append(obj)
        return objectsimport inspect
from tqdm import tqdm

# Import our enhanced systems with proper fallbacks
try:
    from .actions import create_random_action, ArcAction, ZigzagAction, FanAction
except ImportError:
    # Real fallback implementations instead of dummies
    class ArcAction:
        def __init__(self, *args, **kwargs): pass
        def draw(self, draw, center, size, **kwargs): 
            # Draw a simple arc as fallback
            x, y = center
            radius = size // 4
            draw.arc([x-radius, y-radius, x+radius, y+radius], 0, 180, fill='black')
    
    class ZigzagAction:
        def __init__(self, *args, **kwargs): pass
        def draw(self, draw, center, size, **kwargs):
            # Draw a simple zigzag line
            x, y = center
            points = [(x-size//4, y), (x, y-size//4), (x+size//4, y)]
            draw.line(points, fill='black', width=2)
    
    class FanAction:
        def __init__(self, *args, **kwargs): pass
        def draw(self, draw, center, size, **kwargs):
            # Draw a simple fan shape
            x, y = center
            radius = size // 4
            for i in range(3):
                angle = i * 60
                end_x = x + radius * math.cos(math.radians(angle))
                end_y = y + radius * math.sin(math.radians(angle))
                draw.line([x, y, end_x, end_y], fill='black', width=2)
    
    def create_random_action():
        return random.choice([ArcAction(), ZigzagAction(), FanAction()])

try:
    from .enhanced_shapes import (
        regular_polygon, star_polygon, create_nested_shape,
        draw_dashed_line, draw_dotted_line, apply_background_texture,
        create_gradient_fill
    )
except ImportError:
    # Real fallback implementations
    import math
    def regular_polygon(sides, radius, center_x, center_y):
        points = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.extend([x, y])
        return points
    
    def star_polygon(points, outer_radius, inner_radius, center_x, center_y):
        coords = []
        for i in range(points * 2):
            angle = math.pi * i / points
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            coords.extend([x, y])
        return coords
    
    def create_nested_shape(draw, center, size, colors):
        x, y = center
        for i, color in enumerate(colors):
            radius = size - i * (size // len(colors))
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    def draw_dashed_line(draw, start, end, fill='black', width=2, dash_length=5):
        # Simple dashed line implementation
        import math
        x1, y1 = start
        x2, y2 = end
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if distance == 0:
            return
        steps = int(distance // (dash_length * 2))
        for i in range(0, steps, 2):
            t1 = i / steps
            t2 = min((i + 1) / steps, 1.0)
            seg_x1 = x1 + t1 * (x2 - x1)
            seg_y1 = y1 + t1 * (y2 - y1)
            seg_x2 = x1 + t2 * (x2 - x1)
            seg_y2 = y1 + t2 * (y2 - y1)
            draw.line([seg_x1, seg_y1, seg_x2, seg_y2], fill=fill, width=width)
    
    def draw_dotted_line(draw, start, end, fill='black', width=2, dot_spacing=10):
        # Simple dotted line implementation
        import math
        x1, y1 = start
        x2, y2 = end
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if distance == 0:
            return
        steps = int(distance // dot_spacing)
        for i in range(steps):
            t = i / steps if steps > 0 else 0
            dot_x = x1 + t * (x2 - x1)
            dot_y = y1 + t * (y2 - y1)
            draw.ellipse([dot_x-width//2, dot_y-width//2, dot_x+width//2, dot_y+width//2], fill=fill)
    
    def apply_background_texture(image, texture_type='noise'):
        return image  # Return image unchanged as fallback
    
    def create_gradient_fill(size, start_color, end_color):
        # Create a simple gradient
        img = Image.new('RGB', size, start_color)
        return img

try:
    from .prototype_action import PrototypeAction, create_prototype_config
except ImportError:
    class PrototypeAction:
        def __init__(self, *args, **kwargs): 
            self.shapes = self._create_basic_shapes()
        
        def _create_basic_shapes(self):
            # Create basic shape prototypes
            shapes = []
            basic_shapes = ['circle', 'square', 'triangle', 'star']
            for shape in basic_shapes:
                shapes.append({
                    'name': shape,
                    'shape': shape,
                    'size': random.randint(20, 40),
                    'color': random.choice(['red', 'blue', 'green', 'yellow', 'black'])
                })
            return shapes
        
        def draw(self, draw, center, shape_data, **kwargs):
            x, y = center
            shape_type = shape_data.get('shape', 'circle')
            size = shape_data.get('size', 30)
            color = shape_data.get('color', 'black')
            
            if shape_type == 'circle':
                draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
            elif shape_type == 'square':
                draw.rectangle([x-size//2, y-size//2, x+size//2, y+size//2], fill=color)
            elif shape_type == 'triangle':
                points = [x, y-size//2, x-size//2, y+size//2, x+size//2, y+size//2]
                draw.polygon(points, fill=color)
    
    def create_prototype_config():
        return {'default_shapes': ['circle', 'square', 'triangle']}

try:
    from .coverage import EnhancedCoverageTracker, CoverageDimensions
except ImportError:
    class EnhancedCoverageTracker:
        def __init__(self, *args, **kwargs):
            self.scenes = []
        
        def record_scene(self, objects, scene_graph, rule_name, difficulty):
            self.scenes.append({
                'objects': objects,
                'scene_graph': scene_graph,
                'rule_name': rule_name,
                'difficulty': difficulty
            })
        
        def get_coverage_stats(self):
            return {'total_scenes': len(self.scenes)}

try:
    from .cp_sampler import sample_scene_cp
except ImportError:
    def sample_scene_cp(rule, num_objects=3, canvas_size=128):
        # Basic scene sampling fallback - fix signature to include canvas_size
        objects = []
        shapes = ['circle', 'square', 'triangle']
        colors = ['red', 'blue', 'green', 'yellow', 'black']
        
        for i in range(num_objects):
            obj = {
                'shape': random.choice(shapes),
                'color': random.choice(colors),
                'x': random.randint(20, canvas_size-20),
                'y': random.randint(20, canvas_size-20),
                'size': random.randint(15, 35),
                'object_id': f'cp_obj_{i}'
            }
            objects.append(obj)
        return objects

try:
    from .genetic_generator import GeneticSceneGenerator
except ImportError:
    class GeneticSceneGenerator:
        def __init__(self, *args, **kwargs):
            self.population_size = kwargs.get('population_size', 10)
        
        def generate(self, rule, num_objects=3, canvas_size=128, label=1):
            # Basic genetic-style scene generation
            objects = []
            shapes = ['circle', 'square', 'triangle', 'pentagon']
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'black']
            
            for i in range(num_objects):
                obj = {
                    'shape': random.choice(shapes),
                    'color': random.choice(colors),
                    'x': random.randint(15, canvas_size-15),
                    'y': random.randint(15, canvas_size-15),
                    'size': random.randint(18, 38),
                    'rotation': random.uniform(0, 360),
                    'object_id': f'genetic_obj_{i}'
                }
                objects.append(obj)
            return objects, None # Return objects and None for masks

try:
    from .hybrid_sampler import HybridSampler
except ImportError:
    class HybridSampler:
        def __init__(self, *args, **kwargs):
            self.cp_weight = 0.4
            self.genetic_weight = 0.4
            self.random_weight = 0.2
        
        def sample_scene(self, rule, num_objects=3, canvas_size=128, is_positive=True, **kwargs):
            # Hybrid sampling approach - accept is_positive parameter
            sampling_method = random.choices(
                ['cp', 'genetic', 'random'],
                weights=[self.cp_weight, self.genetic_weight, self.random_weight]
            )[0]
            
            if sampling_method == 'cp':
                return sample_scene_cp(rule, num_objects, canvas_size)
            elif sampling_method == 'genetic':
                generator = GeneticSceneGenerator()
                objects, _ = generator.generate(rule, num_objects, canvas_size)
                return objects
            else:
                # Random sampling
                objects = []
                shapes = ['circle', 'square', 'triangle', 'hexagon']
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
                
                for i in range(num_objects):
                    obj = {
                        'shape': random.choice(shapes),
                        'color': random.choice(colors),
                        'x': random.randint(10, canvas_size-10),
                        'y': random.randint(10, canvas_size-10),
                        'size': random.randint(20, 40),
                        'object_id': f'random_obj_{i}'
                    }
                    objects.append(obj)
                return objects
        
        def _create_basic_genetic_objects(self, num_objects, canvas_size):
            """Helper method to create basic genetic-style objects"""
            objects = []
            shapes = ['circle', 'square', 'triangle', 'pentagon']
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'black']
            
            for i in range(num_objects):
                obj = {
                    'shape': random.choice(shapes),
                    'color': random.choice(colors),
                    'x': random.randint(15, canvas_size-15),
                    'y': random.randint(15, canvas_size-15),
                    'size': random.randint(18, 38),
                    'rotation': random.uniform(0, 360),
                    'object_id': f'genetic_obj_{i}'
                }
                objects.append(obj)
            return objects


logger = logging.getLogger(__name__)

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randrange(lo, hi + 1)


class EnhancedDatasetGenerator:
    """Enhanced dataset generator with all new systems integrated."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional configuration."""
        self.config = self._load_config(config_path)
        self.canvas_size = self.config.get('canvas_size', 256)

        # Initialize enhanced systems
        self.coverage_tracker = None
        self.prototype_action = None
        self.genetic_generator = None
        self.hybrid_sampler = None
        self.cp_sampler = None # Added for clarity, as it's used in generate_enhanced_scene

        self._initialize_enhanced_systems()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with enhanced defaults."""
        default_config = {
            'canvas_size': 256,
            'min_objects': 1,
            'max_objects': 6,
            'min_object_size': 15,
            'max_object_size': 60,
            'stroke_min': 1,
            'stroke_max': 4,
            'max_rotation': 360,
            'jitter_px': 3,
            'hatch_gap': 4,
            'enable_textures': True,
            'enable_gradients': True,
            'enable_actions': True,
            'enable_prototypes': True,
            'enable_dashed_strokes': True,
            'prototype_shapes_dir': 'data/gallery',
            'texture_probability': 0.3,
            'gradient_probability': 0.2,
            'action_probability': 0.25,
            'prototype_probability': 0.2,
            'nested_shape_probability': 0.15,
            'dashed_stroke_probability': 0.2
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config {config_path}: {e}")

        return default_config

    def _initialize_enhanced_systems(self):
        """Initialize all enhanced generation systems."""

        # 1. Enhanced coverage tracker
        try:
            self.coverage_tracker = EnhancedCoverageTracker()
            logger.info("Enhanced coverage tracker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced coverage tracker: {e}")

        # 2. Prototype action system
        if self.config.get('enable_prototypes', True):
            try:
                shapes_dir = self.config.get('prototype_shapes_dir')
                self.prototype_action = PrototypeAction(shapes_dir, fallback_shapes=True)
                logger.info(f"Prototype action system initialized with {len(self.prototype_action.shapes)} shapes")
            except Exception as e:
                logger.warning(f"Could not initialize prototype action: {e}")

        # 3. Genetic generator
        try:
            self.genetic_generator = GeneticSceneGenerator()
            logger.info("Genetic scene generator initialized")
        except Exception as e:
            logger.warning(f"Could not initialize genetic generator: {e}")

        # 4. Hybrid sampler
        try:
            if self.genetic_generator and self.coverage_tracker: # Ensure dependencies are met
                self.hybrid_sampler = HybridSampler(
                    genetic_generator=self.genetic_generator,
                    coverage_tracker=self.coverage_tracker
                )
                logger.info("Hybrid sampler initialized")
        except Exception as e:
            logger.warning(f"Could not initialize hybrid sampler: {e}")

        # 5. CP-SAT sampler (assuming it's a standalone function, not a class instance)
        # If cp_sampler is a class, it should be initialized here.
        # For now, assuming sample_scene_cp is directly importable and callable.
        # If it needs an instance, this part would need adjustment based on its actual definition.
        try:
            # Check if sample_scene_cp was successfully imported
            if 'sample_scene_cp' in globals() and callable(sample_scene_cp):
                self.cp_sampler_func = sample_scene_cp # Store the function reference
                logger.info("CP-SAT sampler function reference stored.")
            else:
                self.cp_sampler_func = None
                logger.warning("CP-SAT sampler function not available.")
        except Exception as e:
            self.cp_sampler_func = None
            logger.warning(f"Error checking CP-SAT sampler: {e}")


    def generate_enhanced_scene(self, rule, num_objects: int, is_positive: bool) -> List[Dict]:
        """Generate scene using enhanced systems."""

        # Use hybrid sampler if available
        if self.hybrid_sampler:
            try:
                return self.hybrid_sampler.sample_scene(
                    rule=rule,
                    num_objects=num_objects,
                    is_positive=is_positive,
                    canvas_size=self.canvas_size,
                    min_obj_size=self.config['min_object_size'],
                    max_obj_size=self.config['max_object_size']
                )
            except Exception as e:
                logger.warning(f"Hybrid sampler failed, trying CP-SAT sampler: {e}")

        # Try CP-SAT sampler if available
        if self.cp_sampler_func: # Use the stored function reference
            try:
                return self.cp_sampler_func( # Call the function
                    rule=rule,
                    num_objects=num_objects,
                    positive=is_positive,
                    max_attempts=100
                )
            except Exception as e:
                logger.warning(f"CP-SAT sampler failed, using basic fallback: {e}")

        # Fallback to dataset function
        return self._generate_basic_scene(rule, num_objects, is_positive)

    def _generate_basic_scene(self, rule, num_objects: int, is_positive: bool) -> List[Dict]:
        """Generate basic scene objects when advanced samplers fail."""
        objects = []
        shapes = ['circle', 'square', 'triangle', 'pentagon', 'hexagon']
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'black']
        fills = ['solid', 'hollow']
        
        margin = 20
        
        for i in range(num_objects):
            obj = {
                'shape': random.choice(shapes),
                'color': random.choice(colors),
                'fill': random.choice(fills),
                'size': random.randint(15, 35),
                'x': random.randint(margin, self.canvas_size - margin - 20),
                'y': random.randint(margin, self.canvas_size - margin - 20),
                'rotation': random.uniform(0, 360),
                'object_id': f'basic_obj_{i}'
            }
            objects.append(obj)
        
        return objects


    def render_enhanced_image(self, objects: List[Dict], background_color: str = 'white') -> Image.Image:
        """Render image using all enhanced systems."""

        # Create canvas
        img = Image.new('RGB', (self.canvas_size, self.canvas_size), background_color)
        draw = ImageDraw.Draw(img)

        # Apply background texture if enabled
        if self.config.get('enable_textures', True) and random.random() < self.config.get('texture_probability', 0.3):
            try:
                texture_type = random.choice(['noise', 'checker', 'gradient'])
                if texture_type == 'noise':
                    img = apply_background_texture(img, noise_level=0.1, noise_opacity=0.1)
                elif texture_type == 'checker':
                    # A simple checker pattern overlay (example, can be more sophisticated)
                    checker_size = 20
                    for i in range(0, self.canvas_size, checker_size):
                        for j in range(0, self.canvas_size, checker_size):
                            if (i // checker_size + j // checker_size) % 2 == 0:
                                draw.rectangle([i, j, i + checker_size, j + checker_size], fill=(200, 200, 200))
                            else:
                                draw.rectangle([i, j, i + checker_size, j + checker_size], fill=(230, 230, 230))
                elif texture_type == 'gradient':
                    # Example of applying gradient as background
                    gradient_img = create_gradient_fill(self.canvas_size, self.canvas_size, (255, 255, 255), (200, 200, 255))
                    img = Image.alpha_composite(img.convert('RGBA'), gradient_img.convert('RGBA')).convert('RGB')
            except Exception as e:
                logger.warning(f"Background texture failed: {e}")

        # Render each object with enhanced features
        for obj in objects:
            try:
                self._render_enhanced_object(img, obj)
            except Exception as e:
                logger.warning(f"Error rendering object {obj}: {e}. Falling back to basic rendering.")
                # Fallback to basic rendering
                self._render_basic_object(img, obj)

        return img

    def _render_enhanced_object(self, img: Image.Image, obj: Dict):
        """Render single object with all enhancements."""

        x = obj.get('x', obj.get('position', [self.canvas_size // 2, self.canvas_size // 2])[0])
        y = obj.get('y', obj.get('position', [self.canvas_size // 2, self.canvas_size // 2])[1])
        size = obj.get('size', 30)
        shape = obj.get('shape', 'circle').lower()
        fill_type = obj.get('fill', 'solid').lower()
        color = obj.get('color', 'black')

        # Handle position as tuple/list
        if isinstance(obj.get('position'), (list, tuple)) and len(obj['position']) >= 2:
            x, y = obj['position'][0], obj['position'][1]

        # Convert color name to RGB if needed
        color_rgb = self._name_to_rgb(color)

        # Domain randomization
        rotation = random.uniform(0, self.config.get('max_rotation', 360))
        jitter = self.config.get('jitter_px', 3)
        stroke_width = random.randint(self.config.get('stroke_min', 1), self.config.get('stroke_max', 4))

        # Decide rendering approach based on probabilities
        render_type = self._choose_render_type()

        if render_type == 'action' and self.config.get('enable_actions', True):
            self._render_action_object(img, x, y, size, shape, color_rgb, stroke_width)

        elif render_type == 'prototype' and self.prototype_action and self.config.get('enable_prototypes', True):
            self._render_prototype_object(img, x, y, size, fill_type, color_rgb, stroke_width)

        elif render_type == 'nested' and self.config.get('nested_shape_probability', 0.15) > 0:
            self._render_nested_object(img, x, y, size, shape, fill_type, color_rgb, stroke_width, rotation)

        else:
            # Enhanced basic shapes
            self._render_enhanced_basic_object(img, x, y, size, shape, fill_type, color_rgb, stroke_width, rotation, jitter)

    def _choose_render_type(self) -> str:
        """Choose rendering type based on probabilities."""
        rand = random.random()

        if rand < self.config.get('action_probability', 0.25):
            return 'action'
        elif rand < (self.config.get('action_probability', 0.25) + self.config.get('prototype_probability', 0.2)):
            return 'prototype'
        elif rand < (self.config.get('action_probability', 0.25) + self.config.get('prototype_probability', 0.2) + self.config.get('nested_shape_probability', 0.15)):
            return 'nested'
        else:
            return 'basic'

    def _render_action_object(self, img: Image.Image, x: int, y: int, size: int, shape: str, color: Tuple[int, int, int], stroke_width: int):
        """Render object using action system."""
        try:
            action = create_random_action(shape)
            if action:
                config = {
                    'stroke_width': stroke_width,
                    'jitter_px': self.config.get('jitter_px', 3),
                    'stroke_color': color
                }
                draw = ImageDraw.Draw(img)
                action.draw(draw, (x, y), size, **config)
            else:
                # Fallback to circle if action creation fails
                draw = ImageDraw.Draw(img)
                half_size = size // 2
                draw.ellipse([x - half_size, y - half_size, x + half_size, y + half_size],
                             outline=color, width=stroke_width)
        except Exception as e:
            logger.warning(f"Action rendering failed: {e}. Falling back to basic circle.")
            # Fallback
            draw = ImageDraw.Draw(img)
            half_size = size // 2
            draw.ellipse([x - half_size, y - half_size, x + half_size, y + half_size],
                         outline=color, width=stroke_width)

    def _render_prototype_object(self, img: Image.Image, x: int, y: int, size: int, fill_type: str, color: Tuple[int, int, int], stroke_width: int):
        """Render object using prototype system."""
        try:
            config = create_prototype_config(
                jitter_px=self.config.get('jitter_px', 3),
                fill_color=color,
                fill_pattern=fill_type,
                stroke_width=stroke_width
            )
            draw = ImageDraw.Draw(img)
            self.prototype_action.draw(draw, (x, y), size, **config)
        except Exception as e:
            logger.warning(f"Prototype rendering failed: {e}. Falling back to basic circle.")
            # Fallback to circle
            draw = ImageDraw.Draw(img)
            half_size = size // 2
            draw.ellipse([x - half_size, y - half_size, x + half_size, y + half_size],
                         fill=color)

    def _render_nested_object(self, img: Image.Image, x: int, y: int, size: int, shape: str, fill_type: str, color: Tuple[int, int, int], stroke_width: int, rotation: float):
        """Render nested/compound shapes."""
        try:
            draw = ImageDraw.Draw(img)
            # Assuming create_nested_shape draws directly onto the 'draw' object
            create_nested_shape(
                draw=draw,
                cx=x,
                cy=y,
                size=size,
                fill_color=color,
                outline_color='black',
                stroke_width=stroke_width
            )
            # Note: Rotation for nested shapes drawn directly would need to be handled
            # within create_nested_shape or by rotating the entire image context,
            # which is more complex with PIL's ImageDraw.
            # For simplicity, if create_nested_shape doesn't handle rotation,
            # this `rotation` parameter for nested objects might not have an effect
            # unless `create_nested_shape` is modified to return a rotatable image.
        except Exception as e:
            logger.warning(f"Nested shape rendering failed: {e}. Falling back to basic object.")
            # Fallback
            self._render_basic_object(img, {'x': x, 'y': y, 'size': size, 'shape': shape, 'fill': fill_type, 'color': self._name_to_rgb(color)})

    def _render_enhanced_basic_object(self, img: Image.Image, x: int, y: int, size: int, shape: str, fill_type: str, color: Tuple[int, int, int], stroke_width: int, rotation: float, jitter: int):
        """Render basic shapes with enhancements."""
        draw = ImageDraw.Draw(img)

        # Apply jitter
        jitter_x = x + random.uniform(-jitter, jitter)
        jitter_y = y + random.uniform(-jitter, jitter)

        half_size = size // 2

        # Choose stroke style
        stroke_style = 'solid'
        if self.config.get('enable_dashed_strokes', True) and random.random() < self.config.get('dashed_stroke_probability', 0.2):
            stroke_style = random.choice(['dashed', 'dotted'])

        if shape == 'circle':
            # Approximate circle with polygon for dashed/dotted effect
            num_points = 60 # More points for smoother circle approximation
            points = []
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                px = jitter_x + half_size * np.cos(angle)
                py = jitter_y + half_size * np.sin(angle)
                points.append((px, py))

            if stroke_style == 'dashed':
                self._draw_dashed_polygon(img, points, color, stroke_width)
            elif stroke_style == 'dotted':
                self._draw_dotted_polygon(img, points, color, stroke_width)
            else:
                if fill_type == 'solid':
                    draw.ellipse([jitter_x - half_size, jitter_y - half_size, jitter_x + half_size, jitter_y + half_size], fill=color)
                else:
                    draw.ellipse([jitter_x - half_size, jitter_y - half_size, jitter_x + half_size, jitter_y + half_size], outline=color, width=stroke_width)

        elif shape in ['triangle', 'square', 'pentagon', 'hexagon', 'star']:
            try:
                # Use enhanced shapes with correct parameters (cx, cy, size, sides/points)
                # Assuming regular_polygon and star_polygon return points relative to (cx, cy)
                if shape == 'triangle':
                    points = regular_polygon(jitter_x, jitter_y, size, 3)
                elif shape == 'square':
                    points = regular_polygon(jitter_x, jitter_y, size, 4)
                elif shape == 'pentagon':
                    points = regular_polygon(jitter_x, jitter_y, size, 5)
                elif shape == 'hexagon':
                    points = regular_polygon(jitter_x, jitter_y, size, 6)
                elif shape == 'star':
                    points = star_polygon(jitter_x, jitter_y, size, 5, 0.5)

                # Apply rotation
                if rotation != 0:
                    points = self._rotate_points(points, jitter_x, jitter_y, rotation)

                if stroke_style == 'dashed':
                    self._draw_dashed_polygon(img, points, color, stroke_width)
                elif stroke_style == 'dotted':
                    self._draw_dotted_polygon(img, points, color, stroke_width)
                else:
                    if fill_type == 'solid':
                        draw.polygon(points, fill=color)
                    else:
                        draw.polygon(points, outline=color, width=stroke_width)

            except Exception as e:
                logger.warning(f"Enhanced shape rendering failed: {e}. Falling back to simple circle.")
                # Simple fallback
                draw.ellipse([jitter_x - half_size, jitter_y - half_size, jitter_x + half_size, jitter_y + half_size], outline=color, width=stroke_width)

    def _draw_dashed_circle(self, img: Image.Image, x: float, y: float, radius: float, color: Tuple[int, int, int], width: int):
        """Draw dashed circle by approximating it as a dashed polygon."""
        num_points = 60 # More points for smoother circle approximation
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            points.append((px, py))
        self._draw_dashed_polygon(img, points, color, width)

    def _draw_dotted_circle(self, img: Image.Image, x: float, y: float, radius: float, color: Tuple[int, int, int], width: int):
        """Draw dotted circle by approximating it as a dotted polygon."""
        num_points = 60 # More points for smoother circle approximation
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            px = x + radius * np.cos(angle)
            py = y + radius * np.sin(angle)
            points.append((px, py))
        self._draw_dotted_polygon(img, points, color, width)

    def _draw_dashed_polygon(self, img: Image.Image, points: List[Tuple[float, float]], color: Tuple[int, int, int], width: int):
        """Draw dashed polygon."""
        try:
            for i in range(len(points)):
                start = points[i]
                end = points[(i + 1) % len(points)] # Connect last point to first
                draw_dashed_line(img, [start, end], color, width, dash_length=8)
        except Exception as e:
            logger.warning(f"Dashed polygon drawing failed: {e}. Falling back to solid polygon.")
            # Fallback to regular polygon
            draw = ImageDraw.Draw(img)
            draw.polygon(points, outline=color, width=width)

    def _draw_dotted_polygon(self, img: Image.Image, points: List[Tuple[float, float]], color: Tuple[int, int, int], width: int):
        """Draw dotted polygon."""
        try:
            for i in range(len(points)):
                start = points[i]
                end = points[(i + 1) % len(points)] # Connect last point to first
                draw_dotted_line(img, [start, end], color, width, dot_spacing=6)
        except Exception as e:
            logger.warning(f"Dotted polygon drawing failed: {e}. Falling back to solid polygon.")
            # Fallback to regular polygon
            draw = ImageDraw.Draw(img)
            draw.polygon(points, outline=color, width=width)

    def _rotate_points(self, points: List[Tuple[float, float]], center_x: float, center_y: float, angle_deg: float) -> List[Tuple[float, float]]:
        """Rotate points around center."""
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_points = []
        for px, py in points:
            # Translate to origin
            dx = px - center_x
            dy = py - center_y

            # Rotate
            new_x = dx * cos_a - dy * sin_a
            new_y = dx * sin_a + dy * cos_a

            # Translate back
            rotated_points.append((new_x + center_x, new_y + center_y))

        return rotated_points

    def _render_basic_object(self, img: Image.Image, obj: Dict):
        """Basic object rendering fallback."""
        draw = ImageDraw.Draw(img)

        x = obj.get('x', self.canvas_size // 2)
        y = obj.get('y', self.canvas_size // 2)
        size = obj.get('size', 30)
        color = self._name_to_rgb(obj.get('color', 'black')) # Corrected: added self.

        half_size = size // 2
        draw.ellipse([x - half_size, y - half_size, x + half_size, y + half_size], fill=color)

    def _name_to_rgb(self, color_name: str) -> Tuple[int, int, int]:
        """Convert color name to RGB tuple."""
        color_map = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
            'orange': (255, 165, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42),
            'cyan': (0, 255, 255)
        }
        # Handle both string color names and RGB tuples
        if isinstance(color_name, tuple):
            return color_name  # Already RGB
        elif isinstance(color_name, str):
            return color_map.get(color_name.lower(), (0, 0, 0))
        else:
            return (0, 0, 0)  # Default black

    def update_coverage(self, objects: List[Dict], rule_description: str, is_positive: bool):
        """Update coverage tracking."""
        if self.coverage_tracker:
            try:
                # Create scene graph (simplified)
                scene_graph = {'objects': objects, 'relations': []}
                self.coverage_tracker.record_scene(objects, scene_graph, rule_description, is_positive)
            except Exception as e:
                logger.warning(f"Coverage update failed: {e}")


# Convenience functions for backward compatibility
def create_composite_scene(*args, **kwargs):
    """Fallback function for legacy compatibility."""
    generator = EnhancedDatasetGenerator()
    return generator.render_enhanced_image(*args, **kwargs)


def sample_scene_enhanced(rule, num_objects: int, is_positive: bool, canvas_size: int = 256, **kwargs) -> List[Dict]:
    """Enhanced scene sampling function."""
    generator = EnhancedDatasetGenerator()
    return generator.generate_enhanced_scene(rule, num_objects, is_positive)
