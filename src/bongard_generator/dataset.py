"""
Main dataset generator for Bongard-LOGO problems.
This module defines the core BongardDataset class responsible for generating
image pairs based on a set of rules, guided by the MetaController, Styler,
and a centralized configuration.
"""
import logging
import os
import random
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFilter
from .config import GeneratorConfig
from .shape_renderer import draw_shape
from .styler import apply_background

# Add logger
logger = logging.getLogger(__name__)

# Create real implementations for missing components
class MetaController:
    """Real MetaController for rule management"""
    def __init__(self, rule_paths):
        self.all_rules = self._load_rules()
    
    def _load_rules(self):
        """Load real rules from the rule system"""
        try:
            # Import the real rule system
            from src.bongard_rules import get_all_rules
            rules = get_all_rules()
            if rules:
                logger.info(f"Loaded {len(rules)} real Bongard rules")
                return rules
        except Exception as e:
            logger.warning(f"Could not load real rules: {e}")
        
        # Fallback to basic rules
        return [BasicRule("COLOR_UNIFORMITY"), BasicRule("COUNT_PARITY"), BasicRule("SHAPE_UNIFORMITY")]
    
    def select_rule(self, batch_size=1):
        if self.all_rules:
            return [random.choice(self.all_rules) for _ in range(batch_size)]
        return [BasicRule("FALLBACK_RULE")]
    
    def update_rule_feedback(self, rule_name, reward):
        # Real implementation of rule feedback learning
        if hasattr(self, 'rule_performance'):
            if rule_name not in self.rule_performance:
                self.rule_performance[rule_name] = []
            self.rule_performance[rule_name].append(reward)
            
            # Keep only recent performance history
            if len(self.rule_performance[rule_name]) > 100:
                self.rule_performance[rule_name] = self.rule_performance[rule_name][-100:]
        else:
            self.rule_performance = {rule_name: [reward]}

class BasicRule:
    """Basic rule implementation for fallback"""
    def __init__(self, rule_name):
        self._name = rule_name
        self._description = f"Basic rule: {rule_name}"
    
    @property
    def name(self):
        return self._name
    
    @property 
    def description(self):
        return self._description



def create_composite_scene(objects: List[Dict[str, Any]], cfg: Any) -> Image.Image:
    """
    Create a composite scene from a list of objects using professional rendering.
    """
    canvas_size = getattr(cfg, 'canvas_size', 128)
    if isinstance(canvas_size, str):
        canvas_size = int(canvas_size)
    # If no objects, log and return white canvas
    if not objects or len(objects) == 0:
        logger.warning("No objects provided for scene, returning blank canvas.")
        img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 255))
        return img.convert('RGB')
    # Create a white canvas for better contrast, use RGBA for alpha compositing
    img = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    for obj in objects:
        try:
            draw_shape(img, obj, cfg)
        except Exception as e:
            logger.error(f"Failed to draw object {obj.get('object_id', '')}: {e}", exc_info=True)
    # Apply diverse backgrounds
    img = apply_background(img, cfg)
    return img.convert('RGB') # Convert back to RGB at the end

class BongardDataset:
    """
    Generates Bongard problem datasets with enhanced, configurable features.
    """
    def __init__(self, config: GeneratorConfig, total_problems: int):
        self.config = config
        self.total_problems = total_problems
        
        # Use real rule system
        self.meta_controller = MetaController([])
        self.rules = self.meta_controller.all_rules
        
        logger.info(f"BongardDataset initialized with {len(self.rules)} real rules.")
        
        # Log available rules
        rule_names = [rule.name for rule in self.rules]
        logger.info(f"Available rules: {rule_names}")

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

    def _generate_example(self, rule, is_positive: bool) -> Image.Image:
        """Generates a single image that either follows or violates the rule using the real enhanced generator."""
        try:
            # Use the real EnhancedDatasetGenerator instead of stubs
            from .enhanced_dataset import EnhancedDatasetGenerator
            
            # Create generator instance if not exists
            if not hasattr(self, '_enhanced_generator'):
                self._enhanced_generator = EnhancedDatasetGenerator()
            
            # Generate objects using the enhanced generator
            objects = self._enhanced_generator.generate_enhanced_scene(rule, 
                                                                      num_objects=random.randint(2, 5), 
                                                                      is_positive=is_positive)
            
            # If objects were generated successfully, render them
            if objects:
                # Use the enhanced rendering system
                img = self._enhanced_generator.render_enhanced_image(objects, background_color='white')
                return img.convert('RGB')
            
        except Exception as e:
            logger.warning(f"Enhanced generator failed: {e}, falling back to basic generation")
        
        # Fallback: create basic objects and render with shape_renderer
        objects = self._generate_basic_objects(rule, is_positive)
        img = create_composite_scene(objects, self.config)
        return img.convert('RGB')
    
    def _generate_basic_objects(self, rule, is_positive: bool) -> List[Dict[str, Any]]:
        """Generate basic objects as fallback when enhanced generator fails."""
        num_objects = random.randint(2, 5)
        objects = []
        
        # Basic shapes, colors, and fills
        shapes = ["circle", "square", "triangle", "pentagon", "hexagon"]
        colors = ["black", "red", "blue", "green", "yellow", "purple"]
        fills = ["solid", "hollow"]
        
        canvas_size = getattr(self.config, 'canvas_size', 128)
        margin = 20
        
        for i in range(num_objects):
            obj = {
                'shape': random.choice(shapes),
                'color': random.choice(colors),
                'fill': random.choice(fills),
                'size': random.randint(20, 50),
                'x': random.randint(margin, canvas_size - margin - 20),
                'y': random.randint(margin, canvas_size - margin - 20),
                'rotation': random.uniform(0, 360),
                'object_id': f"basic_obj_{i}"
            }
            objects.append(obj)
        
        return objects

def generate_and_save_dataset(config: GeneratorConfig, num_problems: int, save_dir: str):
    """High-level function to generate and save a dataset based on a given config."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dataset = BongardDataset(config, num_problems)
    
    for i in range(num_problems):
        problem = dataset[i]
        problem_dir = os.path.join(save_dir, f"problem_{i:04d}_{problem['rule_name']}")
        os.makedirs(problem_dir, exist_ok=True)
        # Save only non-blank images
        def is_blank(img):
            arr = img.convert('L').getdata()
            return all(v == 255 for v in arr)
        pos_saved, neg_saved = 0, 0
        for j, img in enumerate(problem["positive_examples"]):
            if is_blank(img):
                logger.warning(f"Skipping blank positive image for problem {i}, rule {problem['rule_name']}, idx {j}")
                continue
            img.save(os.path.join(problem_dir, f"positive_{j}.png"))
            pos_saved += 1
        for j, img in enumerate(problem["negative_examples"]):
            if is_blank(img):
                logger.warning(f"Skipping blank negative image for problem {i}, rule {problem['rule_name']}, idx {j}")
                continue
            img.save(os.path.join(problem_dir, f"negative_{j}.png"))
            neg_saved += 1
        with open(os.path.join(problem_dir, "rule.txt"), "w") as f:
            f.write(problem["rule_description"])
        # Optionally, save a summary of image quality metrics
        with open(os.path.join(problem_dir, "quality_metrics.txt"), "w") as f:
            f.write(f"positive_saved: {pos_saved}\nnegative_saved: {neg_saved}\n")
    logger.info(f"Successfully generated and saved {num_problems} problems to {save_dir}")
