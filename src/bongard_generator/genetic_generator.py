import random
import numpy as np
from config import CONFIG
import torch
from src.bongard_generator.tester_cnn import create_tester_model
from PIL import Image

class GeneticSceneGenerator:
    """
    Genetic algorithm-based Bongard scene generator using config values.
    """
    def __init__(self, config=None):
        # Use config from main CONFIG dict if not provided
        if config is None:
            from genetic_config import GENETIC_CONFIG
            config = GENETIC_CONFIG
        self.config = config
        self.population_size = int(config.get('population_size', 50))
        self.num_generations = int(config.get('num_generations', 30))
        self.mutation_rate = float(config.get('mutation_rate', 0.2))
        self.crossover_rate = float(config.get('crossover_rate', 0.5))
        self.diversity_weight = float(config.get('diversity_weight', 0.5))
        self.tester_weight = float(config.get('tester_weight', 0.5)) if 'tester_weight' in config else 0.5
        self.coverage_weight = float(config.get('coverage_weight', 1.0))
        self.elitism = int(config.get('elitism', 2))
        self.max_attempts = int(config.get('max_attempts', 100))
        self.cache_enabled = bool(config.get('cache_enabled', True))
        self.seed = int(config.get('seed', 42))
        self.tester_checkpoint = config.get('tester_checkpoint', None)
        self.num_rules = int(config.get('num_rules', 23))
        self.device = config.get('device', 'cpu')
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.tester = create_tester_model(self.tester_checkpoint, num_rules=self.num_rules)
        # ...initialize other components as needed...

    def evaluate_fitness(self, genome):
        # Assume genome has 'refined_image' (PIL Image) and 'rule_idx'
        img_refined = genome.refined_image  # PIL image, B/W
        rule_idx = genome.rule_idx
        # 1) CNN semantic confidence
        conf = self.tester.predict_confidence(np.array(img_refined), rule_idx)
        # 2) Diversity score (placeholder, implement min_cosine as needed)
        div = 1.0  # TODO: Replace with actual diversity calculation
        # 3) Final fitness
        alpha = self.config.get('alpha', 0.5)
        beta = self.config.get('beta', 0.5)
        genome.fitness = alpha * conf + beta * div
        genome.tester_confidence = conf
        genome.diversity_score = div
        return genome.fitness

    def generate(self, rule_obj, label):
        """
        Generate a Bongard scene using genetic algorithm for a given rule and label.
        Returns (objects, masks) or falls back to a default scene if failed.
        """
        # --- Genetic algorithm logic would go here ---
        # For demonstration, try to generate a scene; fallback if failed.
        scene = self._run_genetic_algorithm(rule_obj, label)
        if scene is not None and self._is_valid_scene(scene):
            # Scene graph export: return objects metadata as part of scene
            objects, masks = scene
            return objects, masks  # objects is a list of dicts
        # Fallback: return a random valid scene
        fallback_scene = self._fallback_scene(rule_obj, label)
        return fallback_scene

    def _run_genetic_algorithm(self, rule_obj, label):
        # Placeholder for actual genetic algorithm logic
        # Return None to simulate failure, or a tuple (objects, masks) if successful
        return None  # Simulate failure for now

    def _is_valid_scene(self, scene):
        # Placeholder for scene validation logic
        return scene is not None and isinstance(scene, tuple) and len(scene) == 2

    def _fallback_scene(self, rule_obj, label):
        # Generate a simple default scene (objects, masks)
        # Ensure all values are JSON-serializable
        objects = [{
            'shape': 'circle',
            'fill': 'solid',
            'position': [32, 32],  # Use list for JSON compatibility
            'size': 16,
            'color': 'black',
            'label': int(label),
            'rule': str(getattr(rule_obj, 'description', ''))
        }]
        masks = np.zeros((1, 64, 64), dtype=np.uint8).tolist()  # Convert to list for JSON
        return (objects, masks)
