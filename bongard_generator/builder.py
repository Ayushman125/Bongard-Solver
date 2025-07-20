"""
BongardGenerator Builder
This module provides the master BongardGenerator class that unifies all sampling and generation techniques.
"""

from bongard_rules import BongardRule
from config import CONFIG
from bongard_generator.cp_sampler import CPSampler
from bongard_generator.genetic_sampler import GeneticSampler
# Import the actual professional PrototypeAction implementation
from src.bongard_generator.prototype_action import PrototypeAction
from src.bongard_generator.config import GeneratorConfig
from bongard_generator.style import Styler, apply_background
from bongard_generator.coverage import CoverageTracker
from src.bongard_generator.mask_utils import create_composite_scene

class BongardGenerator:
    """
    Master generator that combines CP-SAT, Genetic, and Prototype samplers.
    """
    def __init__(self, cfg):
        # Get generator config from the config.yaml section
        self.cfg = cfg.get('generator', {
            'cp_quota': 0.5,
            'ga_quota': 0.3,
            'pt_quota': 0.2,
            'use_gan': False,
            'bg_texture': 'none',
            'canvas_size': 256
        })
        
        # Configure shapes directory for PrototypeAction
        shapes_dir = cfg.get('shapes_dir', 'src/bongard_generator/shapes')
        
        self.cp = CPSampler(cfg.get('cp_sat', {}))
        self.ga = GeneticSampler(cfg.get('ga', cfg.get('genetic', {})))
        self.proto = PrototypeAction(shapes_dir=shapes_dir, fallback_shapes=True)
        self.styler = Styler(cfg.get('gan_ckpt', None)) if self.cfg.get('use_gan', False) else None
        self.coverage = CoverageTracker(cfg.get('coverage_target', 100))

    def generate_for_rule(self, rule: BongardRule, n_scenes: int):
        """
        Generates a specified number of scenes for a given rule using a hybrid approach.
        """
        scenes = []
        # Compute counts for each sampler based on quotas
        cp_n = int(n_scenes * self.cfg.get('cp_quota', 0.5))
        ga_n = int(n_scenes * self.cfg.get('ga_quota', 0.3))
        pt_n = n_scenes - cp_n - ga_n

        # 1) CP-SAT seeding
        for _ in range(cp_n):
            objs = self.cp.sample_scene(rule)
            if objs:
                scenes.append((objs, "cp_sat"))

        # 2) Genetic refinement
        for _ in range(ga_n):
            objs = self.ga.sample_scene(rule)
            if objs:
                scenes.append((objs, "genetic"))

        # 3) Prototype stamping
        for _ in range(pt_n):
            objs = self._generate_prototype_objects(rule, num_shapes=2)
            if objs:
                scenes.append((objs, "prototype"))

        # 4) Render + post-process
        images = []
        for objs, tag in scenes:
            # Ensure objs is a list of dictionaries
            if not isinstance(objs, list) or not all(isinstance(o, dict) for o in objs):
                # print(f"Skipping invalid object format from {tag}: {objs}")
                continue

            # Create generator config object
            canvas_size = self.cfg.get('canvas_size', 256)
            generator_config = GeneratorConfig()
            generator_config.img_size = canvas_size
            generator_config.shape_color = (0, 0, 0)
            generator_config.bg_color = (255, 255, 255)
            generator_config.allow_overlap = self.cfg.get('allow_overlap', False)
            
            img, scene_features = create_composite_scene(
                config=generator_config,
                rule=rule, 
                is_positive=True,  # Default to positive examples
                prototype_action=self.proto
            )
            
            if self.cfg.get('bg_texture', 'none') != "none":
                # Convert dict to object for background function
                class ConfigObj:
                    def __init__(self, cfg_dict):
                        for key, value in cfg_dict.items():
                            setattr(self, key, value)
                config_obj = ConfigObj(self.cfg)
                img = apply_background(img, config_obj)
            
            if self.styler:
                img = self.styler.stylize(img)
            
            images.append((img, rule, tag))
            self.coverage.record(rule, objs)

        return images

    def _generate_prototype_objects(self, rule: BongardRule, num_shapes: int = 2):
        """Generate objects using prototype approach for consistency with other samplers."""
        import random
        
        objs = []
        canvas_size = self.cfg.get('canvas_size', 256)
        
        for i in range(num_shapes):
            # Create object dictionary matching CP-SAT/genetic format
            obj = {
                'type': 'prototype',
                'center_x': random.randint(50, canvas_size - 50),
                'center_y': random.randint(50, canvas_size - 50), 
                'size': random.randint(20, 60),
                'color': (0, 0, 0),  # Black shapes
                'rotation': random.randint(0, 360) if random.random() < 0.3 else 0,
                'proto_sampler': self.proto,  # Store reference to use in rendering
                'shape_id': i
            }
            objs.append(obj)
        
        return objs

    def get_coverage_report(self):
        """
        Returns the coverage report.
        """
        return self.coverage.report()
