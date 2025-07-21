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
        # Prefer config.data.genetic if available
        gd = getattr(config, 'data', None)
        if gd and hasattr(gd, 'genetic'):
            gd = gd.genetic
        else:
            gd = config
        self.population_size = getattr(config, 'population_size', getattr(gd, 'population_size', 50))
        self.generations     = getattr(config, 'generations', getattr(gd, 'generations', 20))
        self.alpha           = getattr(config, 'alpha', getattr(gd, 'alpha', 0.7))
        self.beta            = getattr(config, 'beta', getattr(gd, 'beta', 0.3))
        self.tester_checkpoint = getattr(config, 'tester_checkpoint', getattr(gd, 'tester_checkpoint', None))
        self.num_rules = getattr(config, 'num_rules', getattr(gd, 'num_rules', 23))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mutation_rate = getattr(config, 'mutation_rate', getattr(gd, 'mutation_rate', 0.2))
        self.crossover_rate = getattr(config, 'crossover_rate', getattr(gd, 'crossover_rate', 0.5))
        self.diversity_weight = getattr(config, 'diversity_weight', getattr(gd, 'diversity_weight', 0.5))
        self.tester_weight = getattr(config, 'tester_weight', getattr(gd, 'tester_weight', 0.5))
        self.coverage_weight = getattr(config, 'coverage_weight', getattr(gd, 'coverage_weight', 1.0))
        self.elitism = getattr(config, 'elitism', getattr(gd, 'elitism', 2))
        self.max_attempts = getattr(config, 'max_attempts', getattr(gd, 'max_attempts', 100))
        self.cache_enabled = getattr(config, 'cache_enabled', getattr(gd, 'cache_enabled', True))
        self.seed = getattr(config, 'seed', getattr(gd, 'seed', 42))
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.tester = create_tester_model(self.tester_checkpoint, num_rules=self.num_rules)
        self.tester.to(self.device)
        # ...initialize other components as needed...

    def evaluate_fitness(self, genome):
        # Assume genome has 'refined_image' (PIL Image) and 'rule_idx'
        img_refined = genome.refined_image  # PIL image, B/W
        rule_idx = genome.rule_idx
        # Convert image to tensor and move to correct device
        if isinstance(img_refined, Image.Image):
            image = np.array(img_refined.convert('L'))
        else:
            image = np.array(img_refined)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        # Sanity check: print device info
        print("Input device:", tensor.device)
        print("Model device:", next(self.tester.parameters()).device)
        # 1) CNN semantic confidence
        with torch.no_grad():
            logits = self.tester.forward(tensor)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            conf = probabilities[0, rule_idx].item()
        # 2) Diversity score (placeholder, implement min_cosine as needed)
        div = 1.0  # TODO: Replace with actual diversity calculation
        # 3) Final fitness
        alpha = getattr(self.config, 'alpha', 0.5)
        beta = getattr(self.config, 'beta', 0.5)
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
            'color': 'black', # Force black color
            'label': int(label),
            'rule': str(getattr(rule_obj, 'description', ''))
        }]
        masks = np.zeros((1, 64, 64), dtype=np.uint8).tolist()  # Convert to list for JSON
        return (objects, masks)
