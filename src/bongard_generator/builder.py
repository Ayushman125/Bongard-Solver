# src/bongard_generator/builder.py (master generator)
import logging
from pathlib import Path
import random
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from .cp_sampler       import CPSATSampler as CPSampler
from .genetic_generator import GeneticSceneGenerator as GeneticSampler
from .prototype_action import PrototypeAction
from .dataset         import create_composite_scene
from .coverage        import CoverageTracker
from .scene_graph import build_scene_graph
from .gnn_model  import SceneGNN

logger = logging.getLogger(__name__)

class BongardGenerator:
    """Professional Bongard problem scene generator with multiple sampling strategies."""
    
    def __init__(self, cfg):
        """
        Initialize the BongardGenerator with robust configuration handling.
        Ensures all required attributes are present and correctly typed.
        """
        # Accept either a full config or just the generator config
        # Always work with self.cfg as the unified config object
        self.cfg = cfg

        # Unify generator config: if passed only generator section, treat as main config
        # If there's a nested generator config, merge its keys
        generator_keys = [
            'min_objects', 'max_objects', 'canvas_size', 'output_dir',
            'cp_quota', 'ga_quota', 'prototype_quota', 'use_gnn', 'gnn_thresh',
            'stroke_min', 'num_backgrounds', 'jitter_px', 'noise_strength', 'checker_opacity'
        ]
        if hasattr(cfg, 'generator'):
            gen_cfg = cfg.generator
            for key in generator_keys:
                if hasattr(gen_cfg, key) and not hasattr(self.cfg, key):
                    setattr(self.cfg, key, getattr(gen_cfg, key))
        # Set defaults for any missing keys
        defaults = {
            'min_objects': 1,
            'max_objects': 5,
            'canvas_size': 128,
            'output_dir': 'generated_scenes',
            'cp_quota': 0.3,
            'ga_quota': 0.4,
            'prototype_quota': 0.3,
            'use_gnn': False,
            'gnn_thresh': 0.5,
            'stroke_min': 1,
            'num_backgrounds': 10,
            'jitter_px': 0.5,
            'noise_strength': 0.1,
            'checker_opacity': 0.3
        }
        for key, value in defaults.items():
            if not hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        # Ensure all numeric config values are correct types
        self._ensure_numeric_config_types(self.cfg)

        # Output directory
        self.output_dir = Path(getattr(self.cfg, 'output_dir', 'generated_scenes'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Coverage tracker
        coverage_cfg = getattr(cfg, 'coverage', None)
        if coverage_cfg is None and isinstance(cfg, dict):
            coverage_cfg = cfg.get('coverage', None)
        if coverage_cfg is None:
            raise ValueError("Config must provide a 'coverage' dictionary with 'coverage_goals'.")
        from types import SimpleNamespace
        if isinstance(coverage_cfg, dict):
            coverage_cfg = SimpleNamespace(**coverage_cfg)
        self.coverage = CoverageTracker(coverage_cfg)

        # Initialize samplers based on availability
        self._init_samplers(self.cfg)

        # Initialize GNN if enabled
        if getattr(self.cfg, 'use_gnn', False):
            try:
                self._gnn = self._lazy_load_gnn(self.cfg)
            except FileNotFoundError:
                logger.warning(f"GNN checkpoint not found at {getattr(self.cfg, 'gnn_ckpt', 'N/A')}. GNN filtering disabled.")
                self.cfg.use_gnn = False
                self._gnn = None
        
        logger.info(f"BongardGenerator initialized with samplers: {self._get_available_samplers()}")
    
    def _ensure_numeric_config_types(self, cfg):
        """Ensure all numeric configuration values are proper types to prevent string/int division errors."""
        try:
            # Primary numeric fields that must be integers
            integer_fields = ['canvas_size', 'stroke_min', 'min_objects', 'max_objects', 'num_backgrounds']
            for field in integer_fields:
                if hasattr(cfg, field):
                    try:
                        setattr(cfg, field, int(getattr(cfg, field)))
                    except (ValueError, TypeError):
                        # Set safe defaults
                        defaults = {'canvas_size': 128, 'stroke_min': 1, 'min_objects': 1, 'max_objects': 5, 'num_backgrounds': 10}
                        setattr(cfg, field, defaults.get(field, 1))
                        
            # Numeric fields that should be floats
            float_fields = ['jitter_px', 'noise_strength', 'checker_opacity']
            for field in float_fields:
                if hasattr(cfg, field):
                    try:
                        setattr(cfg, field, float(getattr(cfg, field)))
                    except (ValueError, TypeError):
                        # Set safe defaults
                        defaults = {'jitter_px': 0.5, 'noise_strength': 0.1, 'checker_opacity': 0.3}
                        setattr(cfg, field, defaults.get(field, 0.5))
                        
            # Handle nested config objects
            nested_configs = ['generator', 'dataset', 'coverage', 'cp', 'genetic', 'ga']
            for nested_name in nested_configs:
                if hasattr(cfg, nested_name):
                    nested_cfg = getattr(cfg, nested_name)
                    if nested_cfg is not None:
                        self._ensure_numeric_config_types(nested_cfg)
                        
        except Exception as e:
            logger.warning(f"Error during config type conversion: {e}")
    
    def _init_samplers(self, cfg):
        """Initialize available sampling strategies."""
        self.samplers = {}
        
        # Try to initialize CP-SAT sampler
        try:
            if hasattr(cfg, 'cp') or hasattr(cfg, 'constraint_solver'):
                cp_cfg = getattr(cfg, 'cp', getattr(cfg, 'constraint_solver', None))
                if cp_cfg:
                    # Construct the SceneParameters object required by CPSATSampler
                    from .cp_sampler import SceneParameters
                    scene_params = SceneParameters(
                        canvas_size=getattr(cfg, 'canvas_size', 128),
                        min_obj_size=getattr(cp_cfg, 'min_obj_size', 10),
                        max_obj_size=getattr(cp_cfg, 'max_obj_size', 50),
                        max_objects=getattr(cfg, 'max_objects', 5),
                        colors=["black", "white"],  # Force black and white
                        shapes=getattr(cp_cfg, 'shapes', ["circle", "square", "triangle"]),
                        fills=getattr(cp_cfg, 'fills', ["solid", "hollow"])
                    )
                    self.samplers['cp_sat'] = CPSampler(scene_params)
                else:
                    self.samplers['cp_sat'] = None
        except Exception as e:
            logger.warning(f"CP-SAT sampler unavailable: {e}")
            self.samplers['cp_sat'] = None
        
        # Try to initialize Genetic Algorithm sampler  
        try:
            if hasattr(cfg, 'genetic') or hasattr(cfg, 'ga'):
                ga_cfg = getattr(cfg, 'genetic', getattr(cfg, 'ga', None))
                if ga_cfg:
                    self.samplers['genetic'] = GeneticSampler(ga_cfg)
                else:
                    # Fallback to default genetic config from genetic_config.py
                    self.samplers['genetic'] = GeneticSampler()
            else:
                self.samplers['genetic'] = None
        except Exception as e:
            logger.warning(f"Genetic sampler unavailable: {e}")
            self.samplers['genetic'] = None
        
        # Always enable the random sampler as a fallback
        self.samplers['random'] = True  # Flag to enable random generation
        
        # Try to initialize Prototype Action system
        try:
            prototypes_dir = getattr(cfg, 'prototypes_dir', 'data/prototypes')
            action_programs_path = getattr(cfg, 'action_programs_path', None)
            self.samplers['prototype'] = PrototypeAction(prototypes_dir, action_programs_path)
        except Exception as e:
            logger.warning(f"Prototype sampler unavailable: {e}")
            self.samplers['prototype'] = None
    
    def _get_available_samplers(self):
        """Get list of available sampler names."""
        return [name for name, sampler in self.samplers.items() if sampler is not None]

    def _lazy_load_gnn(self, cfg):
        # lazy-load once
        # These SHAPES, COLORS, FILLS should be imported from a central place
        SHAPES = ["circle","square","triangle","pentagon","star","arc","zigzag","prototype"]
        COLORS = ["black","white","red","blue","green"]
        FILLS  = ["solid","hollow","striped","dotted"]
        # Always use in_feats=16 for checkpoint compatibility
        gnn = SceneGNN(in_feats=16).to(cfg.device)
        gnn.load_state_dict(torch.load(cfg.generator.gnn_ckpt))
        gnn.eval()
        return gnn

    def generate_for_rule(self, rule, N, is_positive=True):
        """
        Professional implementation of scene generation for a given rule.
        Uses multiple sampling strategies: CP-SAT, Genetic Algorithm, and Prototype injection.
        """
        scenes = []
        
        # Generate base scenes using different strategies
        base_scenes = self._generate_base_scenes(N, rule)
        
        # Apply the rule to each base scene
        for i, (base_objects, generation_method) in enumerate(base_scenes):
            try:
                # Apply the rule to modify objects according to the rule logic
                if not base_objects:
                    logger.warning(f"Skipping rule application for empty base scene {i}")
                    continue
                modified_objects, rule_features = rule.apply(base_objects, is_positive)
                
                # Add metadata for tracking
                scene_metadata = {
                    'generation_method': generation_method,
                    'rule_applied': rule.name,
                    'is_positive': is_positive,
                    'scene_id': i,
                    **rule_features
                }
                
                scenes.append((modified_objects, scene_metadata))
                
            except Exception as e:
                logger.error(f"Failed to apply rule {rule.name} to scene {i}: {e}")
                continue
        
        # Filter and render scenes
        return self._filter_and_render_scenes(scenes, rule)
    
    def _check_overlap(self, new_obj, existing_objects):
        """Check if a new object overlaps with any existing objects, adding a small buffer."""
        for old_obj in existing_objects:
            # Simple bounding box collision detection
            dist_x = abs(new_obj['x'] - old_obj['x'])
            dist_y = abs(new_obj['y'] - old_obj['y'])
            
            # Approximate size check (sum of half-sizes) with a 5px buffer
            buffer = 5
            min_dist_x = (new_obj['size'] + old_obj['size']) / 2 + buffer
            min_dist_y = (new_obj['size'] + old_obj['size']) / 2 + buffer
            
            if dist_x < min_dist_x and dist_y < min_dist_y:
                return True  # Overlap detected
        return False

    def _place_object_without_overlap(self, generated_objects, object_id, shapes, color_palette, fills):
        """
        Tries to place a new object without overlap, reducing its size on failure.
        Returns the new object or None if placement fails.
        """
        max_initial_attempts = 250  # Increased attempts
        size_reduction_factor = 0.95 # Slower size reduction
        min_size_ratio = 0.25 # Allow smaller objects if needed

        canvas_size = self.cfg.canvas_size
        margin = 15
        # More reasonable size range, e.g., 15% to 40% of canvas size
        initial_size = random.randint(
            int(canvas_size * 0.15), int(canvas_size * 0.40)
        )
        size = initial_size

        for attempt in range(max_initial_attempts):
            # Ensure x, y are valid for the current size
            half_size = int(size) // 2
            if canvas_size - margin - half_size <= margin + half_size:
                # Size is too large for the canvas with margins, cannot place.
                logger.debug(f"Object {object_id} with size {size} is too large for canvas {canvas_size} with margin {margin}.")
                # Reduce size and retry
                size = int(size * size_reduction_factor)
                continue

            obj = {
                'shape': random.choice(shapes),
                'color': random.choice(color_palette),
                'fill': random.choice(fills),
                'size': int(size),
                'x': random.randint(margin + half_size, canvas_size - margin - half_size),
                'y': random.randint(margin + half_size, canvas_size - margin - half_size),
                'object_id': object_id,
                'rotation': random.uniform(0, 360),
            }

            if not self._check_overlap(obj, generated_objects):
                return obj # Placement successful

            # After a certain number of failures, reduce size
            if attempt > 0 and attempt % 10 == 0: # Reduce size more often
                new_size = int(size * size_reduction_factor)
                if new_size < int(initial_size * min_size_ratio):
                    logger.warning(f"Could not place object {object_id}, size reduced below minimum threshold.")
                    return None
                size = new_size
                logger.debug(f"Reducing size for object {object_id} to {size}")

        logger.warning(f"Could not place object {object_id} without overlap after {max_initial_attempts} attempts.")
        return None

    def _generate_base_scenes(self, N, rule):
        """Generate base scenes using different sampling strategies with anti-overlap."""
        base_scenes = []
        
        # Get sampling quotas from config
        cp_quota = getattr(self.cfg, 'cp_quota', 0.2)
        ga_quota = getattr(self.cfg, 'ga_quota', 0.2)
        random_quota = getattr(self.cfg, 'random_quota', 0.5)
        prototype_quota = 1.0 - cp_quota - ga_quota - random_quota

        sampler_choices = []
        sampler_choices.extend(['cp_sat'] * int(N * cp_quota))
        sampler_choices.extend(['genetic'] * int(N * ga_quota))
        sampler_choices.extend(['random'] * int(N * random_quota))
        sampler_choices.extend(['prototype'] * int(N * prototype_quota))
        random.shuffle(sampler_choices)

        for i in range(N):
            generation_method = sampler_choices[i] if i < len(sampler_choices) else 'random'
            
            num_objects = random.randint(self.cfg.min_objects, self.cfg.max_objects)
            objects = None

            # This block handles object generation from various samplers
            if generation_method == 'cp_sat' and self.samplers.get('cp_sat'):
                logger.debug(f"Using CP-SAT sampler for scene {i}")
                objects = self.samplers['cp_sat'].sample_scene_cp(rule, num_objects, positive=True)
            
            elif generation_method == 'genetic' and self.samplers.get('genetic'):
                logger.debug(f"Using Genetic sampler for scene {i}")
                generated_data = self.samplers['genetic'].generate(rule, label=1)
                if generated_data and isinstance(generated_data, tuple) and len(generated_data) > 0:
                    objects = generated_data[0]
            
            # The random and prototype samplers now include anti-overlap logic
            elif generation_method in ['random', 'prototype'] or objects is None:
                generation_method = 'random' if objects is None else generation_method
                logger.debug(f"Using {generation_method} sampler with anti-overlap for scene {i}")

                # Force black and white color palette
                color_palette = ["black", "white"]
                
                shapes = ["circle", "square", "triangle", "pentagon", "star", "hexagon"]
                fills = ["solid", "hollow", "striped", "dotted"]
                
                generated_objects = []
                for j in range(num_objects):
                    obj = self._place_object_without_overlap(
                        generated_objects,
                        f"obj_{i}_{j}",
                        shapes,
                        color_palette,
                        fills
                    )
                    if obj:
                        generated_objects.append(obj)
                
                objects = generated_objects

            # Final fallback if all else fails
            if objects is None:
                    logger.error(f"All samplers failed for scene {i}. Generating a minimal random scene.")
                    # Fallback: force black and white palette
                    objects = [{
                        'shape': 'circle', 'color': 'black', 'fill': 'solid', 'size': 30,
                        'x': self.cfg.canvas_size // 2, 'y': self.cfg.canvas_size // 2,
                        'object_id': 'fallback_obj', 'rotation': 0
                    }]

            base_scenes.append((objects, generation_method))
        
        return base_scenes
    
    def _filter_and_render_scenes(self, scenes, rule):
        """Filter scenes using GNN and render final images."""
        from tqdm import tqdm
        filtered_scenes = []
        gnn_filtered_count = 0
        for scene_index, (objects, metadata) in enumerate(tqdm(scenes, desc=f"Filtering/Rendering scenes for {rule.name}")):
            try:
                # GNN filtering if enabled
                if getattr(self.cfg, 'use_gnn', False) and self._gnn:
                    scene_graph = build_scene_graph(objects, self.cfg)
                    scene_graph = scene_graph.to(self.cfg.device)
                    with torch.no_grad():
                        quality_score = self._gnn(scene_graph).item()
                    # Skip low-quality scenes
                    gnn_threshold = getattr(self.cfg, 'gnn_thresh', 0.5)
                    if quality_score < gnn_threshold:
                        gnn_filtered_count += 1
                        continue
                    metadata['gnn_quality_score'] = quality_score
                # Render the scene to an image
                if hasattr(self.cfg, 'canvas_size') and isinstance(self.cfg.canvas_size, str):
                    self.cfg.canvas_size = int(self.cfg.canvas_size)
                
                # If the generation method was prototype, use the prototype sampler to draw
                if metadata['generation_method'] == 'prototype' and self.samplers.get('prototype'):
                    scene_image = Image.new('RGBA', (self.cfg.canvas_size, self.cfg.canvas_size), (255, 255, 255, 255))
                    prototype_sampler = self.samplers['prototype']
                    for obj in objects:
                        # Ensure obj['size'] is always an int for drawing
                        size = obj.get('size', 32)
                        if isinstance(size, str):
                            try:
                                size = int(size)
                            except Exception:
                                size = 32  # fallback default
                        
                        # Create a mock shape_data dictionary for the draw method
                        shape_data = {
                            'shape': obj.get('shape', 'circle'),
                            'size': size,
                            'color': obj.get('color', 'black'),
                            'fill': obj.get('fill', 'solid')
                        }
                        prototype_sampler.draw(ImageDraw.Draw(scene_image), (obj['x'], obj['y']), shape_data)
                else:
                    scene_image = create_composite_scene(objects, self.cfg)

                # --- DEBUG LOGGING ---
                filename = f"{rule.name}_{metadata['generation_method']}_{scene_index}.png"
                output_path = self.output_dir / filename
                logger.debug(f"SAVING to {output_path} (type: {type(output_path)})")
                scene_image.save(output_path)
                # --- END DEBUG ---
                # Record coverage information
                if hasattr(self, 'coverage'):
                    self.coverage.record(objects)
                metadata['object_count'] = len(objects)
                metadata['render_success'] = True
                filtered_scenes.append((scene_image, objects, metadata))
            except Exception as e:
                logger.error(f"Failed to filter/render scene: {e}", exc_info=True)
                continue
        self.gnn_filter_count = gnn_filtered_count
        return filtered_scenes
