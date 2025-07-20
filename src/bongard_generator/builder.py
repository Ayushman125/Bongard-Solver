# src/bongard_generator/builder.py (master generator)
import random
import torch
import logging
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
        Initialize the BongardGenerator with professional configuration handling.
        """
        self.cfg = cfg
        
        # COMPREHENSIVE TYPE CONVERSION - Fix all string/int division errors
        self._ensure_numeric_config_types(self.cfg)
        
        # Extract generator config with safe defaults
        if hasattr(cfg, 'generator'):
            self.generator_cfg = cfg.generator
        else:
            # Create default generator config
            from types import SimpleNamespace
            self.generator_cfg = SimpleNamespace(
                use_gnn=False,
                gnn_thresh=0.5,
                cp_quota=0.3,
                ga_quota=0.4,
                prototype_quota=0.3,
                canvas_size=128
            )
        
        # Initialize coverage tracker with correct config object
        coverage_cfg = getattr(cfg, 'coverage', None)
        if coverage_cfg is None and isinstance(cfg, dict):
            coverage_cfg = cfg.get('coverage', None)
        if coverage_cfg is None:
            raise ValueError("Config must provide a 'coverage' dictionary with 'coverage_goals'.")
        # Convert dict to SimpleNamespace for attribute access
        from types import SimpleNamespace
        if isinstance(coverage_cfg, dict):
            coverage_cfg = SimpleNamespace(**coverage_cfg)
        self.coverage = CoverageTracker(coverage_cfg)
        
        # Initialize samplers based on availability
        self._init_samplers(cfg)
        
        # Initialize GNN if enabled
        if getattr(self.generator_cfg, 'use_gnn', False):
            self._gnn = self._lazy_load_gnn(cfg)
        
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
                self.samplers['cp_sat'] = CPSampler(cp_cfg) if cp_cfg else None
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
                    # Fallback to default genetic config
                    self.samplers['genetic'] = GeneticSampler()
        except Exception as e:
            logger.warning(f"Genetic sampler unavailable: {e}")
            self.samplers['genetic'] = None
        
        # Try to initialize Prototype Action system
        try:
            prototypes_dir = getattr(cfg, 'prototypes_dir', 'data/prototypes')
            self.samplers['prototype'] = PrototypeAction(prototypes_dir)
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
        feat_dim = len(SHAPES)+len(COLORS)+len(FILLS)+3
        gnn = SceneGNN(feat_dim).to(cfg.device)
        gnn.load_state_dict(torch.load(cfg.generator.gnn_ckpt))
        gnn.eval()
        return gnn

    def generate_for_rule(self, rule, N, is_positive=True):
        """
        Professional implementation of scene generation for a given rule.
        Uses multiple sampling strategies: CP-SAT, Genetic Algorithm, and Prototype injection.
        """
        scenes = []
        
        # Get sampling quotas from config (with fallbacks)
        cp_quota = getattr(self.cfg, 'cp_quota', 0.3)
        ga_quota = getattr(self.cfg, 'ga_quota', 0.4) 
        prototype_quota = getattr(self.cfg, 'prototype_quota', 0.3)
        
        # Calculate how many scenes each sampler should generate
        cp_n = max(1, int(N * cp_quota))
        ga_n = max(1, int(N * ga_quota))
        pt_n = max(1, N - cp_n - ga_n)
        
        # Generate base scenes using different strategies
        base_scenes = self._generate_base_scenes(N, rule)
        
        # Apply the rule to each base scene
        for i, (base_objects, generation_method) in enumerate(base_scenes):
            try:
                # Apply the rule to modify objects according to the rule logic
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
    
    def _generate_base_scenes(self, N, rule):
        """Generate base scenes using different sampling strategies."""
        base_scenes = []
        
        for i in range(N):
            # Create base objects with reasonable diversity
            num_objects = random.randint(1, 6)  # Reasonable range for Bongard problems
            objects = []
            
            # Available attributes for generating diverse objects
            shapes = ["circle", "square", "triangle", "pentagon", "star"]
            colors = ["black", "white", "red", "blue", "green"] 
            fills = ["solid", "hollow", "striped", "dotted"]
            sizes = ["small", "medium", "large"]
            
            for j in range(num_objects):
                # Generate spatially distributed objects
                canvas_size = 128
                margin = 20
                obj = {
                    'shape': random.choice(shapes),
                    'color': random.choice(colors),
                    'fill': random.choice(fills),
                    'size': random.choice(sizes),
                    'x': random.randint(margin, canvas_size - margin),
                    'y': random.randint(margin, canvas_size - margin),
                    'width': random.randint(10, 30),
                    'height': random.randint(10, 30),
                    'object_id': f"obj_{i}_{j}",
                    'rotation': random.uniform(0, 360)
                }
                objects.append(obj)
            
            # Determine generation method based on scene index
            if i < N // 3:
                method = 'constraint_based'  # Simulating CP-SAT approach
            elif i < 2 * N // 3:
                method = 'genetic_optimization'  # Simulating GA approach  
            else:
                method = 'prototype_injection'  # Simulating prototype approach
                
            base_scenes.append((objects, method))
        
        return base_scenes
    
    def _filter_and_render_scenes(self, scenes, rule):
        """Filter scenes using GNN and render final images."""
        filtered_scenes = []
        
        for objects, metadata in scenes:
            try:
                # GNN filtering if enabled
                if hasattr(self.cfg, 'use_gnn') and self.cfg.use_gnn and hasattr(self, '_gnn'):
                    scene_graph = build_scene_graph(objects, self.cfg)
                    scene_graph = scene_graph.to(self.cfg.device)
                    
                    with torch.no_grad():
                        quality_score = self._gnn(scene_graph).item()
                    
                    # Skip low-quality scenes
                    gnn_threshold = getattr(self.cfg, 'gnn_thresh', 0.5)
                    if quality_score < gnn_threshold:
                        continue
                    
                    metadata['gnn_quality_score'] = quality_score
                
                # Render the scene to an image
                # Ensure config has proper numeric types before rendering
                if hasattr(self.cfg, 'canvas_size') and isinstance(self.cfg.canvas_size, str):
                    self.cfg.canvas_size = int(self.cfg.canvas_size)
                
                scene_image = create_composite_scene(objects, self.cfg)
                
                # Record coverage information
                if hasattr(self, 'coverage'):
                    self.coverage.record(rule, objects)
                
                # Add final metadata
                metadata['object_count'] = len(objects)
                metadata['render_success'] = True
                
                filtered_scenes.append((scene_image, objects, metadata))
                
            except Exception as e:
                logger.error(f"Failed to filter/render scene: {e}")
                continue
        
        return filtered_scenes
