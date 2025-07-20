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
from PIL import Image

from .config import GeneratorConfig
from .coverage import EnhancedCoverageTracker
from .mask_utils import create_composite_scene
from .meta_controller import MetaController
from .prototype_action import PrototypeAction
from .styler import Styler

logger = logging.getLogger(__name__)

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

    def _generate_example(self, rule: AbstractRule, is_positive: bool) -> Image.Image:
        """Generates a single image that either follows or violates the rule."""
        img, features = create_composite_scene(
            self.config,
            rule,
            is_positive,
            self.prototype_action  # Pass the action object itself
        )

        # Record the generated features for coverage analysis
        self.coverage_tracker.record(features)

        # Apply GAN-based stylization if enabled
        if self.config.use_gan_stylization:
            img = self.styler.apply(img)
        
        # CRITICAL: Final binarization to ensure pure black and white output
        bw_img = img.convert("L").point(lambda p: 255 if p > 128 else 0, mode='L')
        
        return bw_img

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
