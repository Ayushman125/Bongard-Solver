"""Main Bongard problem sampler orchestrating all components"""


import logging
import random
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from PIL import Image
from .dataset import BongardDataset

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    # randrange excludes hi, so we +1
    return random.randrange(lo, hi+1)

from .config_loader import get_config, get_sampler_config, SamplerConfig
from src.bongard_rules import BongardRule
from .rule_loader import get_all_rules, get_rule_by_description
from .utils import set_seed, add_noise, create_gradient_background
from .drawing import TurtleCanvas, ShapeDrawer
from .spatial_sampler import RelationSampler
from .coverage import CoverageTracker, AdversarialSampler
from .constraints import PlacementOptimizer, ConstraintGenerator
from .validation import ValidationSuite

logger = logging.getLogger(__name__)

class BongardSampler:
    def flush_caches_and_reseed(self, seed=None):
        """Flush caches and reseed RNGs to ensure generator changes take effect."""
        import importlib
        import sys
        # Reload relevant modules (if needed)
        for mod_name in [
            'src.bongard_rules',
            'bongard_generator.rule_loader',
            'bongard_generator.sampler',
            'bongard_generator.dataset',
        ]:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
        # Reseed RNGs
        if seed is None:
            seed = safe_randint(0, 1_000_000)
        random.seed(seed)
        np.random.seed(seed)
        # Clear any custom caches (if present)
        if hasattr(self, 'coverage_tracker') and hasattr(self.coverage_tracker, 'clear_cache'):
            self.coverage_tracker.clear_cache()
        if hasattr(self, 'adversarial_sampler') and hasattr(self.adversarial_sampler, 'clear_cache'):
            self.adversarial_sampler.clear_cache()
        # If sampler has a cache attribute, clear it
        if hasattr(self, 'cache'):
            self.cache.clear()
        logger.info(f"Caches flushed and RNG reseeded with seed {seed}")
    """Main sampler class orchestrating all components for Bongard problem generation."""
    
    def __init__(self, config: Optional[SamplerConfig] = None):
        """
        Initialize the Bongard sampler.
        Args:
            config: Optional sampler configuration. If None, loads from config files.
        """
        self.config = config or get_sampler_config()
        self.rules = get_all_rules()
        self.generator_mode = getattr(self.config, 'generator_mode', 'default')
        # Initialize components
        self.shape_drawer = ShapeDrawer(self.config)
        self.relation_sampler = RelationSampler(
            self.config.img_size, 
            self.config.min_obj_size, 
            self.config.max_obj_size
        )
        self.coverage_tracker = CoverageTracker()
        self.adversarial_sampler = AdversarialSampler()
        self.placement_optimizer = PlacementOptimizer(
            self.config.img_size,
            self.config.min_obj_size,
            self.config.max_obj_size
        )
        self.constraint_generator = ConstraintGenerator()
        self.validator = ValidationSuite()
        # Wire up genetic generator
        try:
            from bongard_generator.genetic_generator import GeneticSceneGenerator
            self.genetic = GeneticSceneGenerator(self.config)
        except ImportError:
            self.genetic = None
        # Initialize random seed
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            set_seed(self.config.seed)
        logger.info(f"BongardSampler initialized with {len(self.rules)} rules, generator_mode={self.generator_mode}")
    
    def sample_problem(self, 
                      rule_description: Optional[str] = None,
                      num_pos_scenes: int = 7,
                      num_neg_scenes: int = 7,
                      use_adversarial: bool = False) -> Dict[str, Any]:
        """
        Sample a complete Bongard problem.
        
        Args:
            rule_description: Optional specific rule to use
            num_pos_scenes: Number of positive example scenes
            num_neg_scenes: Number of negative example scenes
            use_adversarial: Whether to use adversarial sampling
            
        Returns:
            Dictionary containing the complete Bongard problem
        """
        # Select rule
        if rule_description:
            rule = get_rule_by_description(rule_description)
            if not rule:
                logger.warning(f"Rule '{rule_description}' not found, using random rule")
                rule = random.choice(self.rules)
        else:
            rule = random.choice(self.rules)
        
        logger.info(f"Generating Bongard problem for rule: {rule.description}")
        
        # Generate positive scenes
        positive_scenes = []
        for i in range(num_pos_scenes):
            scene = self.sample_scene(rule, is_positive=True, use_adversarial=use_adversarial)
            if scene:
                positive_scenes.append(scene)
        
        # Generate negative scenes
        negative_scenes = []
        for i in range(num_neg_scenes):
            scene = self.sample_scene(rule, is_positive=False, use_adversarial=use_adversarial)
            if scene:
                negative_scenes.append(scene)
        
        # Create problem dictionary
        problem = {
            'rule': {
                'description': rule.description,
                'positive_features': rule.pos_literals,
                'negative_features': rule.neg_literals
            },
            'positive_scenes': positive_scenes,
            'negative_scenes': negative_scenes,
            'metadata': {
                'num_positive': len(positive_scenes),
                'num_negative': len(negative_scenes),
                'config': self.config.__dict__,
                'adversarial': use_adversarial
            }
        }
        
        # Record in coverage tracker
        all_scenes = positive_scenes + negative_scenes
        for scene in all_scenes:
            self.coverage_tracker.record_scene(
                scene['objects'],
                scene.get('scene_graph', {}),
                rule.description,
                1 if scene in positive_scenes else 0
            )
        
        logger.info(f"Generated Bongard problem with {len(positive_scenes)} positive and {len(negative_scenes)} negative scenes")
        return problem
    
    def sample_scene(self, 
                    rule: BongardRule, 
                    is_positive: bool = True,
                    use_adversarial: bool = False,
                    max_attempts: int = 100) -> Optional[Dict[str, Any]]:
        """
        Sample a single scene conforming to or violating the rule.
        If generator_mode is 'genetic', use GeneticSceneGenerator.
        """
        if getattr(self, 'generator_mode', 'genetic') == 'genetic' and self.genetic is not None:
            # Use genetic generator for scene creation
            rule_obj = rule
            label = 1 if is_positive else 0
            result = self.genetic.generate(rule_obj, label)
            if result:
                objs, masks = result
                # Use SyntheticBongardDataset for final rendering
                from bongard_generator.dataset import BongardDataset
                # Only pass the current rule for mini-dataset
                print(f">> sampling rule key: {getattr(rule, 'name', None)}")
                ds = BongardDataset(
                    target_quota=1,  # Generate exactly 1 example for this rule
                    rule_list=[rule.name]
                )
                selected_rule = next((r for r in ds.rules if getattr(r, 'name', None) == rule.name), None)
                if not selected_rule:
                    logger.warning(f"Rule {rule.name} not found in BongardDataset rules!")
                    return None
                # You may need to call a method like ds.generate_scene(selected_rule, is_positive)
                # For now, just return the rule for demonstration
                sample = selected_rule
                # Use the dataset's public API to get the rendered scene
                sample = ds[0]  # Only one example in this tiny dataset
                img = sample['image']
                final_masks = sample.get('masks', [])
                # CNN scoring
                rule_idx = getattr(rule_obj, 'rule_idx', 0) if hasattr(rule_obj, 'rule_idx') else 0
                cnn_score = None
                if hasattr(self.genetic, 'evaluate_fitness'):
                    # Create a mock genome for scoring
                    class MockGenome:
                        pass
                    genome = MockGenome()
                    genome.refined_image = img
                    genome.rule_idx = rule_idx
                    cnn_score = self.genetic.evaluate_fitness(genome)
                return {
                    'objects': objs,
                    'scene_graph': {},
                    'image': img,
                    'masks': final_masks,
                    'rule_satisfaction': is_positive,
                    'cnn_score': cnn_score,
                    'metadata': {
                        'generator': 'genetic',
                        'rule_description': rule.description
                    }
                }
            else:
                logger.error("Genetic generator failed to produce a scene")
                return None
        ds = BongardDataset(
            target_quota=1,
            rule_list=[rule.name]
        )
        # Find the rule in ds.rules matching rule.name
        selected_rule = next((r for r in ds.rules if getattr(r, 'name', None) == rule.name), None)
        if not selected_rule:
            logger.warning(f"Rule {rule.name} not found in BongardDataset rules!")
            return None
        # Generate a single scene for the selected rule
        # You may need to call a method like ds.generate_scene(selected_rule, is_positive)
        # For now, just return the rule for demonstration
        return selected_rule
    
    def _determine_scene_params(self, 
                              rule: BongardRule, 
                              is_positive: bool,
                              use_adversarial: bool) -> Dict[str, Any]:
        """Determine scene generation parameters with improved diversity."""
        
        # Use weighted random selection for better object count diversity
        possible_counts = [1, 2, 3, 4, 5, 6]
        weights = [0.05, 0.2, 0.3, 0.25, 0.15, 0.05]  # Favor 2-4 objects but allow variety
        num_objects = random.choices(possible_counts, weights=weights)[0]
        
        params = {
            'num_objects': num_objects,
            'background_type': random.choice(['solid', 'gradient', 'texture']),
            'lighting': random.choice(['normal', 'bright', 'dim']),
            'noise_level': random.uniform(0.0, 0.1),
            'use_cp_sat': random.choice([True, False]),
        }
        
        # Extract rule-specific parameters
        if 'count' in rule.positive_features:
            target_count = rule.positive_features['count']
            if is_positive:
                params['num_objects'] = target_count
            else:
                # For negative examples, use different count with better diversity
                other_counts = [i for i in possible_counts if i != target_count]
                if other_counts:
                    # Use weights for the other counts as well
                    other_weights = [weights[i-1] for i in other_counts]
                    total_weight = sum(other_weights)
                    normalized_weights = [w/total_weight for w in other_weights]
                    params['num_objects'] = random.choices(other_counts, weights=normalized_weights)[0]
                else:
                    params['num_objects'] = random.choice([2, 3, 4])  # Fallback
        
        # Apply adversarial modifications
        if use_adversarial:
            adversarial_params = self.adversarial_sampler.sample_adversarial_params()
            params.update(adversarial_params)
        
        return params
    
    def _generate_objects(self, 
                        scene_params: Dict[str, Any], 
                        rule: BongardRule,
                        is_positive: bool) -> List[Dict[str, Any]]:
        """Generate objects for the scene."""
        num_objects = scene_params['num_objects']
        
        # Generate spatial constraints based on rule
        spatial_constraints = []
        if any(rel in rule.description.upper() for rel in ['LEFT_OF', 'RIGHT_OF', 'ABOVE', 'BELOW', 'NEAR']):
            spatial_constraints = self.constraint_generator.generate_constraints(
                num_objects, rule.description
            )
        
        # Use CP-SAT for optimal placement if enabled
        if scene_params.get('use_cp_sat', False):
            objects = self.placement_optimizer.optimize_placement(
                num_objects,
                spatial_constraints=spatial_constraints
            )
        else:
            # Use relation sampler for specific spatial relationships
            if spatial_constraints:
                relation_type = spatial_constraints[0]['type']
                objects = self.relation_sampler.sample(num_objects, relation_type)
            else:
                objects = self.relation_sampler.sample(num_objects, 'random')
        
        # Apply rule-specific features
        objects = self._apply_rule_features(objects, rule, is_positive)
        
        return objects
    
    def _apply_rule_features(self, 
                           objects: List[Dict[str, Any]], 
                           rule: BongardRule,
                           is_positive: bool) -> List[Dict[str, Any]]:
        """Apply rule-specific features to objects."""
        features = rule.pos_literals if is_positive else rule.neg_literals
        
        for obj in objects:
            # Apply shape constraints
            if 'shape' in features:
                target_shape = features['shape']
                if is_positive:
                    obj['shape'] = target_shape
                else:
                    # For negative examples, use different shapes
                    other_shapes = [s for s in ['circle', 'triangle', 'square'] if s != target_shape]
                    obj['shape'] = random.choice(other_shapes)
            
            # Apply color constraints
            if 'color' in features:
                target_color = features['color']
                if is_positive:
                    obj['color'] = target_color
                else:
                    other_colors = [c for c in ['red', 'blue', 'green', 'yellow', 'purple'] if c != target_color]
                    obj['color'] = random.choice(other_colors)
            
            # Apply fill constraints
            if 'fill' in features:
                target_fill = features['fill']
                if is_positive:
                    obj['fill'] = target_fill
                else:
                    obj['fill'] = 'outline' if target_fill == 'solid' else 'solid'
            
            # Apply size constraints
            if 'size' in features:
                target_size = features['size']
                if is_positive:
                    if target_size == 'large':
                        obj['size'] = safe_randint(40, 60)
                    elif target_size == 'small':
                        obj['size'] = safe_randint(15, 30)
                    else:
                        obj['size'] = safe_randint(25, 45)
                else:
                    # For negative examples, use opposite size
                    if target_size == 'large':
                        obj['size'] = safe_randint(15, 30)
                    elif target_size == 'small':
                        obj['size'] = safe_randint(40, 60)
        
        return objects
    
    def _create_scene_graph(self, 
                          objects: List[Dict[str, Any]], 
                          scene_params: Dict[str, Any]) -> Dict[str, Any]:
        """Create scene graph with object relationships."""
        scene_graph = {
            'objects': len(objects),
            'relations': []
        }
        
        # Compute spatial relationships
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1, obj2 = objects[i], objects[j]
                
                # Check for various spatial relationships
                if self.relation_sampler.validate_relation([obj1, obj2], 'left_of'):
                    scene_graph['relations'].append({
                        'type': 'left_of',
                        'object1': i,
                        'object2': j
                    })
                elif self.relation_sampler.validate_relation([obj1, obj2], 'above'):
                    scene_graph['relations'].append({
                        'type': 'above',
                        'object1': i,
                        'object2': j
                    })
                elif self.relation_sampler.validate_relation([obj1, obj2], 'near'):
                    scene_graph['relations'].append({
                        'type': 'near',
                        'object1': i,
                        'object2': j
                    })
        
        return scene_graph
    
    def _render_scene(self, 
                    objects: List[Dict[str, Any]], 
                    scene_graph: Dict[str, Any]) -> Optional[np.ndarray]:
        """Render the scene to an image, binarize, and add binary noise."""
        try:
            # Create canvas
            canvas = TurtleCanvas(self.config.img_size)

            # Add background
            bg_color = 'white'  # Always white for Bongard-LOGO style
            canvas.set_background(bg_color)

            # Draw objects
            for obj in objects:
                canvas.draw_shape(
                    obj['shape'],
                    obj['x'] + obj['size'] // 2,  # Center coordinates
                    obj['y'] + obj['size'] // 2,
                    obj['size'],
                    'black',  # Always black for Bongard-LOGO style
                    obj['fill'] == 'solid'
                )

            # Convert to PIL Image in 'L' mode
            pil_img = canvas.get_image().convert('L')

            # Add binary noise before binarization
            def binary_noise(img, prob=0.01):
                arr = np.array(img, dtype=np.uint8)
                mask = np.random.rand(*arr.shape) < prob
                arr[mask] = 255 - arr[mask]
                return Image.fromarray(arr, mode='L')
            pil_img = binary_noise(pil_img, prob=0.01)

            # Binarize to pure black-and-white
            # Fix binarization: shapes should be black (0), background white (255)
            pil_img = pil_img.point(lambda p: 0 if p < 128 else 255, 'L')

            # Convert back to numpy array (0 or 255)
            image = np.array(pil_img, dtype=np.uint8)

            return image

        except Exception as e:
            logger.error(f"Scene rendering failed: {e}")
            return None
    
    def _validate_scene(self, 
                       objects: List[Dict[str, Any]], 
                       scene_graph: Dict[str, Any],
                       rule: BongardRule,
                       is_positive: bool) -> bool:
        """Validate that scene satisfies or violates the rule appropriately."""
        try:
            # Count-based validation
            if 'count' in rule.positive_features:
                target_count = rule.positive_features['count']
                actual_count = len(objects)
                
                if is_positive:
                    return actual_count == target_count
                else:
                    return actual_count != target_count
            
            # Shape-based validation
            if 'shape' in rule.positive_features:
                target_shape = rule.positive_features['shape']
                has_target_shape = any(obj['shape'] == target_shape for obj in objects)
                
                if is_positive:
                    return has_target_shape
                else:
                    return not has_target_shape
            
            # Color-based validation
            if 'color' in rule.positive_features:
                target_color = rule.positive_features['color']
                has_target_color = any(obj['color'] == target_color for obj in objects)
                
                if is_positive:
                    return has_target_color
                else:
                    return not has_target_color
            
            # Spatial relationship validation
            if 'relation' in rule.positive_features:
                target_relation = rule.positive_features['relation']
                has_relation = any(rel['type'] == target_relation for rel in scene_graph['relations'])
                
                if is_positive:
                    return has_relation
                else:
                    return not has_relation
            
            # If no specific validation criteria, accept the scene
            return True
            
        except Exception as e:
            logger.error(f"Scene validation failed: {e}")
            return False
    
    def generate_dataset(self, 
                        num_problems: int = 100,
                        output_dir: Optional[str] = None,
                        use_adversarial: bool = False,
                        stratify_rules: bool = True) -> Dict[str, Any]:
        """
        Generate a complete dataset of Bongard problems.
        
        Args:
            num_problems: Number of problems to generate
            output_dir: Optional directory to save the dataset
            use_adversarial: Whether to use adversarial sampling
            stratify_rules: Whether to ensure balanced rule coverage
            
        Returns:
            Dataset dictionary
        """
        dataset = {
            'problems': [],
            'metadata': {
                'num_problems': num_problems,
                'config': self.config.__dict__,
                'adversarial': use_adversarial,
                'stratified': stratify_rules
            }
        }
        
        # Determine rule distribution
        if stratify_rules:
            problems_per_rule = max(1, num_problems // len(self.rules))
            rule_schedule = []
            for rule in self.rules:
                rule_schedule.extend([rule.description] * problems_per_rule)
            
            # Fill remaining with random rules
            while len(rule_schedule) < num_problems:
                rule_schedule.append(random.choice(self.rules).description)
            
            random.shuffle(rule_schedule)
        else:
            rule_schedule = [None] * num_problems
        
        # Generate problems
        for i in range(num_problems):
            rule_desc = rule_schedule[i] if stratify_rules else None
            
            problem = self.sample_problem(
                rule_description=rule_desc,
                use_adversarial=use_adversarial
            )
            
            if problem:
                problem['id'] = i
                dataset['problems'].append(problem)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated {i + 1}/{num_problems} problems")
        
        # Add coverage statistics
        coverage_stats = self.coverage_tracker.get_coverage_stats()
        dataset['metadata']['coverage'] = coverage_stats
        
        # Save dataset if output directory specified
        if output_dir:
            self._save_dataset(dataset, output_dir)
        
        logger.info(f"Dataset generation complete: {len(dataset['problems'])} problems")
        return dataset
    
    def _save_dataset(self, dataset: Dict[str, Any], output_dir: str) -> None:
        """Save dataset to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metadata = self._make_json_serializable(dataset['metadata'])
            json.dump(serializable_metadata, f, indent=2)
        
        # Save problems (without images to reduce size)
        problems_path = output_path / "problems.json"
        with open(problems_path, 'w') as f:
            problems_data = []
            for problem in dataset['problems']:
                problem_copy = problem.copy()
                # Remove image data to reduce file size
                for scene in problem_copy.get('positive_scenes', []):
                    if 'image' in scene:
                        del scene['image']
                for scene in problem_copy.get('negative_scenes', []):
                    if 'image' in scene:
                        del scene['image']
                problems_data.append(problem_copy)
            
            json.dump(problems_data, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
    
    def _generate_fallback_scene(self, 
                               rule: BongardRule, 
                               is_positive: bool = True,
                               use_adversarial: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate a simple fallback scene when regular generation fails.
        Uses relaxed constraints and randomized parameters for diversity.
        """
        try:
            logger.info(f"Attempting fallback generation for {'positive' if is_positive else 'negative'} scene")
            
            # Use simplified, more diverse scene parameters for fallback
            fallback_params = self._generate_fallback_params(rule, is_positive)
            
            # Generate objects with relaxed constraints
            objects = self._generate_fallback_objects(fallback_params, rule, is_positive)
            if not objects:
                logger.warning("Fallback object generation failed; forcing at least one random object")
                # Guarantee at least one object
                shapes = ['circle', 'triangle', 'square']
                colors = ['red', 'blue', 'green', 'yellow', 'purple']
                fills = ['solid', 'outline']
                size = safe_randint(20, 50)
                margin = 30
                x = safe_randint(margin, self.config.img_size - margin - 50)
                y = safe_randint(margin, self.config.img_size - margin - 50)
                objects = [{
                    'x': x,
                    'y': y,
                    'size': size,
                    'shape': random.choice(shapes),
                    'color': random.choice(colors),
                    'fill': random.choice(fills),
                    'id': 0
                }]
            
            # Create minimal scene graph
            scene_graph = {
                'objects': len(objects),
                'relations': []  # Keep minimal for fallback
            }
            
            # Force render with simplified settings
            image = self._render_fallback_scene(objects)
            if image is None:
                logger.warning("Fallback scene rendering failed")
                return None
            
            # Relaxed validation - accept more cases
            if self._validate_fallback_scene(objects, rule, is_positive):
                scene = {
                    'objects': objects,
                    'scene_graph': scene_graph,
                    'image': image,
                    'rule_satisfaction': is_positive,
                    'metadata': {
                        'fallback_generation': True,
                        'rule_description': rule.description,
                        'adversarial': use_adversarial,
                        'scene_params': fallback_params
                    }
                }
                
                logger.info(f"Successfully generated fallback {'positive' if is_positive else 'negative'} scene")
                return scene
            else:
                logger.warning("Fallback scene validation failed")
                return None
                
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return None
    
    def _generate_fallback_params(self, rule: BongardRule, is_positive: bool) -> Dict[str, Any]:
        """Generate relaxed parameters for fallback generation with more diversity."""
        
        # Randomize object count more aggressively to avoid "always 2 objects" issue
        possible_counts = [1, 2, 3, 4, 5, 6]
        weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]  # Favor 3-4 objects but allow variety
        num_objects = random.choices(possible_counts, weights=weights)[0]
        
        # Extract rule-specific count constraints
        if 'count' in rule.positive_features:
            target_count = rule.positive_features['count']
            if is_positive:
                num_objects = target_count
            else:
                # For negative examples, explicitly avoid target count
                other_counts = [c for c in possible_counts if c != target_count]
                if other_counts:
                    num_objects = random.choice(other_counts)
        
        return {
            'num_objects': num_objects,
            'background_type': 'solid',  # Keep simple for fallback
            'lighting': 'normal',
            'noise_level': 0.0,  # No noise for fallback
            'use_cp_sat': False,  # Disable CP-SAT for faster fallback
            'fallback_mode': True
        }
    
    def _generate_fallback_objects(self, 
                                 params: Dict[str, Any], 
                                 rule: BongardRule,
                                 is_positive: bool) -> List[Dict[str, Any]]:
        """Generate objects for fallback scene with simplified constraints."""
        num_objects = params['num_objects']
        objects = []
        
        # Define possible attributes for diversity
        shapes = ['circle', 'triangle', 'square']
        colors = ['red', 'blue', 'green', 'yellow', 'purple']
        fills = ['solid', 'outline']
        
        # Apply rule constraints but with fallback logic
        target_shape = rule.positive_features.get('shape', None)
        target_color = rule.positive_features.get('color', None) 
        target_fill = rule.positive_features.get('fill', None)
        
        for i in range(num_objects):
            # Random base position with better spacing
            margin = 30
            x = safe_randint(margin, self.config.img_size - margin - 50)
            y = safe_randint(margin, self.config.img_size - margin - 50)
            
            # Randomized size
            size = safe_randint(20, 50)
            
            # Apply rule features with fallback diversity
            if is_positive and target_shape:
                shape = target_shape
            elif not is_positive and target_shape:
                # For negative, use different shapes but ensure some diversity
                other_shapes = [s for s in shapes if s != target_shape]
                shape = random.choice(other_shapes) if other_shapes else random.choice(shapes)
            else:
                shape = random.choice(shapes)
            
            if is_positive and target_color:
                color = target_color
            elif not is_positive and target_color:
                other_colors = [c for c in colors if c != target_color]
                color = random.choice(other_colors) if other_colors else random.choice(colors)
            else:
                color = random.choice(colors)
                
            if is_positive and target_fill:
                fill = target_fill
            elif not is_positive and target_fill:
                fill = 'outline' if target_fill == 'solid' else 'solid'
            else:
                fill = random.choice(fills)
            
            obj = {
                'x': x,
                'y': y,
                'size': size,
                'shape': shape,
                'color': color,
                'fill': fill,
                'id': i
            }
            objects.append(obj)
        
        return objects
    
    def _render_fallback_scene(self, objects: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Render fallback scene with simplified settings."""
        try:
            # Create canvas
            canvas = TurtleCanvas(self.config.img_size)
            
            # Simple white background
            canvas.set_background('white')
            
            # Draw objects with black color for Bongard-LOGO style
            for obj in objects:
                canvas.draw_shape(
                    obj['shape'],
                    obj['x'] + obj['size'] // 2,
                    obj['y'] + obj['size'] // 2,
                    obj['size'],
                    'black',  # Always black for Bongard-LOGO
                    obj['fill'] == 'solid'
                )
            
            # Convert to grayscale and binarize
            pil_img = canvas.get_image().convert('L')
            # Fix binarization: shapes should be black (0), background white (255)
            pil_img = pil_img.point(lambda p: 0 if p < 128 else 255, 'L')
            image = np.array(pil_img, dtype=np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Fallback scene rendering failed: {e}")
            return None
    
    def _validate_fallback_scene(self, 
                                objects: List[Dict[str, Any]], 
                                rule: BongardRule,
                                is_positive: bool) -> bool:
        """Relaxed validation for fallback scenes."""
        try:
            # Count-based validation
            if 'count' in rule.positive_features:
                target_count = rule.positive_features['count']
                actual_count = len(objects)
                
                if is_positive:
                    return actual_count == target_count
                else:
                    return actual_count != target_count
            
            # For other rule types, use relaxed validation
            # Shape-based validation 
            if 'shape' in rule.positive_features:
                target_shape = rule.positive_features['shape']
                shape_counts = {}
                for obj in objects:
                    shape_counts[obj['shape']] = shape_counts.get(obj['shape'], 0) + 1
                
                if is_positive:
                    # At least one object should have target shape
                    return target_shape in shape_counts
                else:
                    # Either no target shape or other shapes present
                    return target_shape not in shape_counts or len(shape_counts) > 1
            
            # Color-based validation
            if 'color' in rule.positive_features:
                target_color = rule.positive_features['color'] 
                color_counts = {}
                for obj in objects:
                    color_counts[obj['color']] = color_counts.get(obj['color'], 0) + 1
                
                if is_positive:
                    return target_color in color_counts
                else:
                    return target_color not in color_counts or len(color_counts) > 1
            
            # If no specific rule features, accept the scene
            return True
            
        except Exception as e:
            logger.warning(f"Fallback validation error: {e}")
            return True  # Be permissive in fallback mode

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-safe types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def run_validation(self) -> bool:
        """Run validation suite on the sampler."""
        logger.info("Running validation suite...")
        results = self.validator.run_all_validations()
        self.validator.print_validation_report()
        return all(results.values())
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Get coverage report from the tracker."""
        return self.coverage_tracker.get_coverage_stats()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize sampler
    sampler = BongardSampler()

    # Flush caches and reseed before every run
    sampler.flush_caches_and_reseed()

    # Run validation
    if sampler.run_validation():
        print("✓ Validation passed")
    else:
        print("✗ Validation failed")
        exit(1)

    # Generate a single problem
    sampler.flush_caches_and_reseed()
    problem = sampler.sample_problem(rule_description="SHAPE(CIRCLE)")
    if problem:
        print(f"Generated problem with rule: {problem['rule']['description']}")
        print(f"Positive scenes: {len(problem['positive_scenes'])}")
        print(f"Negative scenes: {len(problem['negative_scenes'])}")

    # Generate small dataset
    sampler.flush_caches_and_reseed()
    dataset = sampler.generate_dataset(num_problems=5, use_adversarial=True)
    print(f"Generated dataset with {len(dataset['problems'])} problems")

    # Print coverage report
    coverage = sampler.get_coverage_report()
    print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
