"""Main Bongard problem sampler orchestrating all components"""

import logging
import random
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from .config_loader import get_config, get_sampler_config, SamplerConfig
from .rule_loader import get_all_rules, get_rule_by_description, BongardRule
from .utils import set_seed, add_noise, create_gradient_background
from .drawing import TurtleCanvas, ShapeDrawer
from .spatial_sampler import RelationSampler
from .coverage import CoverageTracker, AdversarialSampler
from .constraints import PlacementOptimizer, ConstraintGenerator
from .validation import ValidationSuite

logger = logging.getLogger(__name__)

class BongardSampler:
    """Main sampler class orchestrating all components for Bongard problem generation."""
    
    def __init__(self, config: Optional[SamplerConfig] = None):
        """
        Initialize the Bongard sampler.
        
        Args:
            config: Optional sampler configuration. If None, loads from config files.
        """
        self.config = config or get_sampler_config()
        self.rules = get_all_rules()
        
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
        
        # Validation suite
        self.validator = ValidationSuite()
        
        # Initialize random seed
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            set_seed(self.config.seed)
        
        logger.info(f"BongardSampler initialized with {len(self.rules)} rules")
    
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
                'positive_features': rule.positive_features,
                'negative_features': rule.negative_features
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
                    max_attempts: int = 50) -> Optional[Dict[str, Any]]:
        """
        Sample a single scene conforming to or violating the rule.
        
        Args:
            rule: The Bongard rule to follow/violate
            is_positive: Whether scene should satisfy the rule
            use_adversarial: Whether to use adversarial sampling
            max_attempts: Maximum generation attempts
            
        Returns:
            Scene dictionary or None if generation failed
        """
        for attempt in range(max_attempts):
            try:
                # Determine scene parameters
                scene_params = self._determine_scene_params(rule, is_positive, use_adversarial)
                
                # Generate objects
                objects = self._generate_objects(scene_params, rule, is_positive)
                if not objects:
                    continue
                
                # Create scene graph (relationships between objects)
                scene_graph = self._create_scene_graph(objects, scene_params)
                
                # Render the scene
                image = self._render_scene(objects, scene_graph)
                if image is None:
                    continue
                
                # Validate scene against rule
                if self._validate_scene(objects, scene_graph, rule, is_positive):
                    scene = {
                        'objects': objects,
                        'scene_graph': scene_graph,
                        'image': image,
                        'rule_satisfaction': is_positive,
                        'metadata': {
                            'attempt': attempt + 1,
                            'rule_description': rule.description,
                            'adversarial': use_adversarial,
                            'scene_params': scene_params
                        }
                    }
                    
                    logger.debug(f"Generated {'positive' if is_positive else 'negative'} scene in {attempt + 1} attempts")
                    return scene
                
            except Exception as e:
                logger.debug(f"Scene generation attempt {attempt + 1} failed: {e}")
                continue
        
        logger.warning(f"Failed to generate {'positive' if is_positive else 'negative'} scene after {max_attempts} attempts")
        return None
    
    def _determine_scene_params(self, 
                              rule: BongardRule, 
                              is_positive: bool,
                              use_adversarial: bool) -> Dict[str, Any]:
        """Determine scene generation parameters."""
        params = {
            'num_objects': random.randint(2, 5),
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
                # For negative examples, use different count
                params['num_objects'] = random.choice([i for i in range(1, 6) if i != target_count])
        
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
        features = rule.positive_features if is_positive else rule.negative_features
        
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
                        obj['size'] = random.randint(40, 60)
                    elif target_size == 'small':
                        obj['size'] = random.randint(15, 30)
                    else:
                        obj['size'] = random.randint(25, 45)
                else:
                    # For negative examples, use opposite size
                    if target_size == 'large':
                        obj['size'] = random.randint(15, 30)
                    elif target_size == 'small':
                        obj['size'] = random.randint(40, 60)
        
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
        """Render the scene to an image."""
        try:
            # Create canvas
            canvas = TurtleCanvas(self.config.img_size)
            
            # Add background
            bg_color = random.choice(['white', 'lightgray', 'lightblue'])
            canvas.set_background(bg_color)
            
            # Draw objects
            for obj in objects:
                canvas.draw_shape(
                    obj['shape'],
                    obj['x'] + obj['size'] // 2,  # Center coordinates
                    obj['y'] + obj['size'] // 2,
                    obj['size'],
                    obj['color'],
                    obj['fill'] == 'solid'
                )
            
            # Convert to numpy array
            image = np.array(canvas.get_image())
            
            # Add noise if specified
            noise_level = getattr(self.config, 'noise_level', 0.0)
            if noise_level > 0:
                image = add_noise(image, noise_level)
            
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
    
    # Run validation
    if sampler.run_validation():
        print("✓ Validation passed")
    else:
        print("✗ Validation failed")
        exit(1)
    
    # Generate a single problem
    problem = sampler.sample_problem(rule_description="SHAPE(CIRCLE)")
    if problem:
        print(f"Generated problem with rule: {problem['rule']['description']}")
        print(f"Positive scenes: {len(problem['positive_scenes'])}")
        print(f"Negative scenes: {len(problem['negative_scenes'])}")
    
    # Generate small dataset
    dataset = sampler.generate_dataset(num_problems=5, use_adversarial=True)
    print(f"Generated dataset with {len(dataset['problems'])} problems")
    
    # Print coverage report
    coverage = sampler.get_coverage_report()
    print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
