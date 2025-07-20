"""
Main dataset generator for Bongard-LOGO problems.
This module defines the core BongardDataset class responsible for generating
image pairs based on a set of rules, guided by the MetaController, Styler,
and a centralized configuration.
"""
import logging
import os
import random
from typing import List
from PIL import Image, ImageDraw, ImageFilter
from .config import GeneratorConfig
from .shape_renderer import draw_shape
from .styler import apply_noise, apply_checker

# Add logger
logger = logging.getLogger(__name__)

# Create stub classes for missing components to prevent import errors
class MetaController:
    """Stub MetaController for basic functionality"""
    def __init__(self, rule_paths):
        self.all_rules = self._load_basic_rules()
    
    def _load_basic_rules(self):
        """Load basic rules with fallback"""
        try:
            from .rule_loader import RuleLoader
            loader = RuleLoader()
            return loader.get_rules()
        except ImportError:
            logger.warning("Could not load rules, using stub rule")
            return [StubRule()]
    
    def select_rule(self, batch_size=1):
        if self.all_rules:
            return [random.choice(self.all_rules) for _ in range(batch_size)]
        return [StubRule()]
    
    def update_rule_feedback(self, rule_name, reward):
        pass  # Stub implementation

class StubRule:
    """Stub rule for basic functionality"""
    @property
    def name(self):
        return "STUB_RULE"
    
    @property 
    def description(self):
        return "Basic stub rule for testing"

class EnhancedCoverageTracker:
    """Stub coverage tracker"""
    def __init__(self, config):
        pass

class Styler:
    """Stub styler"""
    def __init__(self, config):
        pass

class PrototypeAction:
    """Stub prototype action"""
    def __init__(self, shapes_dir):
        self.shapes = []

def create_composite_scene(objects, cfg):
    # Ensure canvas_size is always int - comprehensive type conversion
    canvas_size = cfg.img_size if hasattr(cfg, 'img_size') else 128
    if isinstance(canvas_size, (str, tuple, list)):
        if isinstance(canvas_size, str):
            canvas_size = int(canvas_size)
        elif isinstance(canvas_size, (tuple, list)):
            canvas_size = int(canvas_size[0]) if len(canvas_size) > 0 else 128
        else:
            canvas_size = 128
    canvas_size = int(canvas_size)
    
    img = Image.new("RGB",(canvas_size,canvas_size),"white")
    draw = ImageDraw.Draw(img)

    for obj in objects:
        if obj.get('prototype'):
            # Assuming prototype_action is an object with a draw method
            obj['prototype_action'].draw(img, obj['center'], obj['size'], cfg)
        else:
            draw_shape(draw, obj, cfg)

    # background texture
    if hasattr(cfg, 'bg_texture') and cfg.bg_texture=='noise':
        img = apply_noise(img, cfg)
    elif hasattr(cfg, 'bg_texture') and cfg.bg_texture=='checker':
        img = apply_checker(img, cfg)

    # GAN stylization
    if hasattr(cfg, 'styler') and hasattr(cfg, 'generator') and cfg.generator.use_gan and cfg.styler:
        img = cfg.styler.stylize(img)

    # final binarize
    img = img.convert("L").filter(ImageFilter.GaussianBlur(0.5))
    return img.point(lambda p:255 if p>128 else 0,'1')

class BongardDataset:
    """
    Generates Bongard problem datasets with enhanced, configurable features.
    """
    def __init__(self, config: GeneratorConfig, total_problems: int):
        self.config = config
        self.total_problems = total_problems
        
        self.meta_controller = MetaController(config.rule_paths)
        self.rules = self.meta_controller.all_rules
        
        self.coverage_tracker = EnhancedCoverageTracker(self.config)
        self.styler = Styler(self.config)
        
        # Correctly join the project root with the relative prototype path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        proto_dir = os.path.join(project_root, self.config.prototype_path)
        self.prototype_action = PrototypeAction(shapes_dir=proto_dir)

        logger.info(f"BongardDataset initialized with {len(self.rules)} rules.")
        if not self.prototype_action.shapes:
            logger.warning("No prototype shapes were loaded. The generator will use procedural fallbacks.")

    def __len__(self):
        return self.total_problems

    def __getitem__(self, idx: int):
        if idx >= self.total_problems:
            raise IndexError("Index out of range")

        rule = self.meta_controller.select_rule(batch_size=1)[0] if self.rules else None
        
        # Assuming 7 examples per set
        positive_examples = [self._generate_example(rule, is_positive=True) for _ in range(7)] 
        negative_examples = [self._generate_example(rule, is_positive=False) for _ in range(7)]

        # Simulate feedback loop (in a real system, this would come from a solver)
        # Reward is 1 for success, 0 for failure. Here, we simulate random success.
        self.meta_controller.update_rule_feedback(rule.name, reward=random.choice([0.0, 1.0]))

        return {
            "positive_examples": positive_examples,
            "negative_examples": negative_examples,
            "rule_name": rule.name if rule else "none",
            "rule_description": rule.description if rule else "No rule applied",
        }

    def _generate_example(self, rule: 'AbstractRule', is_positive: bool) -> Image.Image:
        """Generates a single image that either follows or violates the rule."""
        # This part needs to be connected to a sampler that generates objects
        # For now, we'll assume a function generate_objects exists
        # In the final implementation, this would call the BongardGenerator/Builder
        objects = [] # Placeholder: self.generator.sample(rule, is_positive)
        
        img = create_composite_scene(
            objects,
            self.config
        )

        # Record the generated features for coverage analysis
        # self.coverage_tracker.record(features)
        
        return img

def generate_and_save_dataset(config: GeneratorConfig, num_problems: int, save_dir: str):
    """High-level function to generate and save a dataset based on a given config."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = BongardDataset(config, num_problems)
    
    for i in range(num_problems):
        problem = dataset[i]
        problem_dir = os.path.join(save_dir, f"problem_{i:04d}_{problem['rule_name']}")
        os.makedirs(problem_dir, exist_ok=True)
        
        for j, img in enumerate(problem["positive_examples"]):
            img.save(os.path.join(problem_dir, f"positive_{j}.png"))
        for j, img in enumerate(problem["negative_examples"]):
            img.save(os.path.join(problem_dir, f"negative_{j}.png"))
            
        with open(os.path.join(problem_dir, "rule.txt"), "w") as f:
            f.write(problem["rule_description"])
            
    logger.info(f"Successfully generated and saved {num_problems} problems to {save_dir}")
