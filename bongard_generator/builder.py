"""
BongardGenerator Builder
This module provides the master BongardGenerator class that unifies all sampling and generation techniques.
"""

from BongordSolver.bongard_rules import BongardRule
from BongordSolver.config import GeneratorConfig, load_config
from BongordSolver.cp_sampler import CPSampler
from BongordSolver.genetic_sampler import GeneticSampler
from BongordSolver.prototype_action import PrototypeAction
from BongordSolver.styler import Styler, apply_background
from BongordSolver.coverage import CoverageTracker
from BongordSolver.utils.drawing import create_composite_scene

class BongardGenerator:
    """
    Master generator that combines CP-SAT, Genetic, and Prototype samplers.
    """
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg.generator
        self.cp = CPSampler(cfg.cp_sat)
        self.ga = GeneticSampler(cfg.ga)
        self.proto = PrototypeAction(cfg.prototypes_dir)
        self.styler = Styler(cfg.gan_ckpt) if self.cfg.use_gan else None
        self.coverage = CoverageTracker(cfg.coverage_target)

    def generate_for_rule(self, rule: BongardRule, n_scenes: int):
        """
        Generates a specified number of scenes for a given rule using a hybrid approach.
        """
        scenes = []
        # Compute counts for each sampler based on quotas
        cp_n = int(n_scenes * self.cfg.cp_quota)
        ga_n = int(n_scenes * self.cfg.ga_quota)
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
            objs = self.proto.inject_random(rule)
            if objs:
                scenes.append((objs, "prototype"))

        # 4) Render + post-process
        images = []
        for objs, tag in scenes:
            # Ensure objs is a list of dictionaries
            if not isinstance(objs, list) or not all(isinstance(o, dict) for o in objs):
                # print(f"Skipping invalid object format from {tag}: {objs}")
                continue

            img = create_composite_scene(objs, self.cfg.canvas_size, self.cfg)
            
            if self.cfg.bg_texture != "none":
                img = apply_background(img, self.cfg)
            
            if self.styler:
                img = self.styler.stylize(img)
            
            images.append((img, rule, tag))
            self.coverage.record(rule, objs)

        return images

    def get_coverage_report(self):
        """
        Returns the coverage report.
        """
        return self.coverage.report()
