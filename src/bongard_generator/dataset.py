"""Dataset generation and management for Bongard problems."""

import logging
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator
import numpy as np
from PIL import Image

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    # randrange excludes hi, so we +1
    return random.randrange(lo, hi+1)

# Assuming these modules are in the same package or correctly imported in a real environment
# from .cp_sampler import sample_scene_cp, SceneParameters
# from .rule_loader import get_all_rules, get_rule_by_description, BongardRule
# from .coverage import CoverageTracker
# from .mask_utils import create_composite_scene
# from .config_loader import get_config

# Dummy imports for demonstration purposes if original modules are not available
# In a real scenario, ensure these are correctly imported from your project structure.
class SceneParameters:
    def __init__(self, canvas_size, min_obj_size, max_obj_size, max_objects, colors, shapes, fills):
        self.canvas_size = canvas_size
        self.min_obj_size = min_obj_size
        self.max_obj_size = max_obj_size
        self.max_objects = max_objects
        self.colors = colors
        self.shapes = shapes
        self.fills = fills

class BongardRule:
    def __init__(self, description, positive_features, negative_features=None):
        self.description = description
        self.positive_features = positive_features
        self.negative_features = negative_features if negative_features is not None else {}

def get_all_rules():
    # Dummy rules for demonstration
    return [
        BongardRule("SHAPE(circle)", {"shape": "circle"}),
        BongardRule("FILL(solid)", {"fill": "solid"}),
        BongardRule("COUNT(2)", {"count": 2}),
        BongardRule("RELATION(overlap)", {"relation": "overlap"}),
    ]

def get_rule_by_description(description):
    for rule in get_all_rules():
        if rule.description == description:
            return rule
    return None

def sample_scene_cp(rule, num_objects, is_positive, canvas_size, min_obj_size, max_obj_size, max_attempts):
    # Dummy implementation for scene sampling
    # In a real scenario, this would use a CP-SAT solver
    objects = []
    for _ in range(num_objects):
        obj = {
            'x': safe_randint(min_obj_size, canvas_size - min_obj_size),
            'y': safe_randint(min_obj_size, canvas_size - min_obj_size),
            'size': safe_randint(min_obj_size, max_obj_size),
            'shape': random.choice(['circle', 'triangle', 'square', 'pentagon', 'star']),
            'fill': random.choice(['solid', 'outline', 'striped', 'gradient']),
            'color': random.choice(['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black'])
        }
        # Apply rule features if positive, or avoid them if negative
        if is_positive:
            if 'shape' in rule.positive_features:
                obj['shape'] = rule.positive_features['shape']
            if 'fill' in rule.positive_features:
                obj['fill'] = rule.positive_features['fill']
            # For count and relation, the logic is more complex and would be handled by the CP-SAT solver
        else: # is_negative
            if 'shape' in rule.positive_features: # Negative rule means avoiding positive features
                # Ensure the generated shape is NOT the positive feature shape
                available_shapes = [s for s in ['circle', 'triangle', 'square', 'pentagon', 'star'] if s != rule.positive_features['shape']]
                if available_shapes:
                    obj['shape'] = random.choice(available_shapes)
            if 'fill' in rule.positive_features:
                available_fills = [f for f in ['solid', 'outline', 'striped', 'gradient'] if f != rule.positive_features['fill']]
                if available_fills:
                    obj['fill'] = random.choice(available_fills)
        
        objects.append(obj)
    
    # Add 'position' key for consistency with the main class
    for obj in objects:
        obj['position'] = (obj['x'], obj['y'])
    
    return objects


class CoverageTracker:
    def __init__(self):
        self.ALL_CELLS = [
            ("circle", "solid", 1, None), ("circle", "outline", 1, None),
            ("square", "solid", 1, None), ("square", "outline", 1, None),
            (None, None, 2, None), (None, None, 3, None),
            (None, None, None, "overlap"), (None, None, None, "near")
        ] # Example cells
        self.coverage = {cell: 0 for cell in self.ALL_CELLS}

    def record_scene(self, objects, scene_graph, rule_description, is_positive):
        # Dummy recording logic for demonstration
        # In a real scenario, this would analyze the scene and update coverage based on features
        # For simplicity, we'll just increment counts for some features based on the rule
        rule = get_rule_by_description(rule_description)
        if rule:
            features = rule.positive_features if is_positive else rule.negative_features
            
            # Simple heuristic to update coverage based on rule features
            if 'shape' in features:
                for cell in self.ALL_CELLS:
                    if cell[0] == features['shape']:
                        self.coverage[cell] += 1
            if 'fill' in features:
                for cell in self.ALL_CELLS:
                    if cell[1] == features['fill']:
                        self.coverage[cell] += 1
            if 'count' in features:
                for cell in self.ALL_CELLS:
                    if cell[2] == features['count']:
                        self.coverage[cell] += 1
            if 'relation' in features:
                for cell in self.ALL_CELLS:
                    if cell[3] == features['relation']:
                        self.coverage[cell] += 1


    def get_under_covered_cells(self, target_quota):
        return [cell for cell, count in self.coverage.items() if count < target_quota]

    def get_coverage_heatmap_data(self):
        return {
            "total_cells": len(self.ALL_CELLS),
            "covered_cells": sum(1 for count in self.coverage.values() if count > 0),
            "coverage_counts": self.coverage
        }

def create_composite_scene(objects, canvas_size):
    # Dummy image creation for demonstration
    # In a real scenario, this would render shapes onto a canvas
    # Always start with white background, grayscale
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    # Simple drawing logic (replace with actual rendering)
    # ...existing code...
    # After all drawing and occluders/noise, binarize
    img_bw = img.point(lambda p: 255 if p > 128 else 0, mode='1')
    img_bw_L = img_bw.convert('L')
    return img_bw_L

def get_config():
    # Dummy config for demonstration
    return {}

logger = logging.getLogger(__name__)

class BongardDataset:
    """Main dataset class for generating and managing Bongard problem datasets."""
    
    def __init__(self, 
                 output_dir: str = "synthetic_images",
                 canvas_size: int = 128,
                 min_obj_size: int = 20,
                 max_obj_size: int = 60,
                 target_quota: int = 50):
        """
        Initialize the Bongard dataset generator.
        
        Args:
            output_dir: Directory to save generated images
            canvas_size: Size of generated images
            min_obj_size: Minimum object size
            max_obj_size: Maximum object size
            target_quota: Target number of examples per coverage cell
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.canvas_size = canvas_size
        self.target_quota = target_quota
        
        # Initialize scene parameters
        self.scene_params = SceneParameters(
            canvas_size=canvas_size,
            min_obj_size=min_obj_size,
            max_obj_size=max_obj_size,
            max_objects=6,
            colors=['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black'],
            shapes=['circle', 'triangle', 'square', 'pentagon', 'star'],
            fills=['solid', 'outline', 'striped', 'gradient']
        )
        
        # Load rules and initialize tracking
        self.rules = get_all_rules()
        self.coverage_tracker = CoverageTracker()
        
        # Dataset statistics
        self.generated_count = 0
        self.failed_count = 0
        
        logger.info(f"Initialized dataset generator with {len(self.rules)} rules")
    
    def generate_dataset(self, 
                         total_examples: int = 10000,
                         positive_ratio: float = 0.5,
                         max_objects_range: Tuple[int, int] = (1, 4),
                         save_images: bool = True,
                         save_metadata: bool = True) -> Dict[str, Any]:
        """
        Generate a complete Bongard dataset.
        
        Args:
            total_examples: Total number of examples to generate
            positive_ratio: Ratio of positive to negative examples
            max_objects_range: Range of object counts per scene
            save_images: Whether to save generated images
            save_metadata: Whether to save metadata JSON
            
        Returns:
            Dataset statistics and metadata
        """
        logger.info(f"Starting dataset generation: {total_examples} examples")
        
        examples = []
        attempts = 0
        max_attempts = total_examples * 3  # Give up after 3x attempts
        
        # Real-time coverage counter for (shape, fill, count, relation)
        coverage_counter = self.coverage_tracker.coverage
        while len(examples) < total_examples and attempts < max_attempts:
            attempts += 1
            # Halt generation once every cell hits target quota N
            if all(coverage_counter[cell] >= self.target_quota for cell in self.coverage_tracker.ALL_CELLS):
                logger.info("All coverage cells have met target quota. Halting generation.")
                break
            
            # Select rule and parameters
            rule = self._select_rule_for_generation()
            is_positive = random.random() < positive_ratio
            num_objects = safe_randint(*max_objects_range)
            
            # Inject adversarial corner cases (5% of scenes)
            adversarial = random.random() < 0.05
            if adversarial:
                # For overlap: set positions to be just at the threshold
                if 'relation' in rule.positive_features and rule.positive_features['relation'] in ['overlap', 'near', 'nested']:
                    # Use custom positions for boundary overlap
                    base_pos = self.canvas_size // 2
                    jitter = int(self.scene_params.max_obj_size * 0.49)  # 49% overlap
                    positions = [
                        {'position': (base_pos, base_pos)},
                        {'position': (base_pos + jitter, base_pos)}
                    ]
                    # Generate objects with these positions
                    objects = []
                    for i in range(num_objects):
                        obj = {
                            'x': positions[i % len(positions)]['position'][0],
                            'y': positions[i % len(positions)]['position'][1],
                            'size': self.scene_params.max_obj_size,
                            'shape': rule.positive_features.get('shape', random.choice(self.scene_params.shapes)),
                            'fill': rule.positive_features.get('fill', random.choice(self.scene_params.fills)),
                            'color': random.choice(self.scene_params.colors),
                            'position': positions[i % len(positions)]['position']
                        }
                        objects.append(obj)
                    scene = {
                        'objects': objects,
                        'scene_graph': self._generate_scene_graph(objects),
                        'rule': rule.description,
                        'positive': is_positive
                    }
                else: # If not a relation rule, generate a normal scene but mark as adversarial
                    try:
                        scene = self._generate_single_scene(rule, num_objects, is_positive)
                        if scene is None:
                            self.failed_count += 1
                            continue
                    except Exception as e:
                        logger.error(f"Failed to generate adversarial example {attempts}: {e}")
                        self.failed_count += 1
                        continue
            else: # Not adversarial
                try:
                    scene = self._generate_single_scene(rule, num_objects, is_positive)
                    if scene is None:
                        self.failed_count += 1
                        continue
                except Exception as e:
                    logger.error(f"Failed to generate example {attempts}: {e}")
                    self.failed_count += 1
                    continue

            # Active tester feedback: auto-reject low-confidence scenes
            if not self._tester_confidence(scene):
                self.failed_count += 1
                continue
            
            example = {
                'id': len(examples),
                'rule': rule.description,
                'positive': is_positive,
                'objects': scene['objects'],
                'scene_graph': scene.get('scene_graph', {}),
                'image_path': None
            }
            if save_images:
                image_path = self._generate_and_save_image(example, scene['objects'])
                example['image_path'] = str(image_path)
            self.coverage_tracker.record_scene(
                scene['objects'], 
                scene.get('scene_graph', {}),
                rule.description,
                1 if is_positive else 0
            )
            examples.append(example)
            self.generated_count += 1
            if len(examples) % 100 == 0:
                self._log_progress(len(examples), total_examples)
        
        # Save metadata
        if save_metadata:
            self._save_dataset_metadata(examples)
        
        # Generate final statistics
        stats = self._generate_final_statistics(examples)
        
        logger.info(f"Dataset generation complete: {len(examples)} examples generated")
        return stats
    
    def _tester_confidence(self, scene: Dict[str, Any], threshold: float = 0.9) -> bool:
        """Lightweight pretrained tester network stub. Returns True if rule confidence >= threshold."""
        # For demonstration, use a simple heuristic: if all objects match the rule, confidence is 1.0
        # In practice, replace with a real model
        rule = scene.get('rule', '')
        objects = scene.get('objects', [])
        # Example: if rule is shape, check all objects
        if 'SHAPE(' in rule:
            shape = rule.split('(')[1].split(')')[0].lower()
            confidence = sum(1 for obj in objects if obj.get('shape') == shape) / max(1, len(objects))
            return confidence >= threshold
        if 'FILL(' in rule:
            fill = rule.split('(')[1].split(')')[0].lower()
            confidence = sum(1 for obj in objects if obj.get('fill') == fill) / max(1, len(objects))
            return confidence >= threshold
        # Default: pass
        return True
    
    def _select_rule_for_generation(self) -> BongardRule:
        """Select a rule for generation based on coverage needs."""
        under_covered_cells = self.coverage_tracker.get_under_covered_cells(self.target_quota)
        
        if under_covered_cells:
            # Prioritize rules for under-covered cells
            cell = random.choice(under_covered_cells)
            shape, fill, count, relation = cell
            
            # Try to find a rule that matches this cell
            for rule in self.rules:
                features = rule.positive_features
                if (features.get('shape') == shape or
                    features.get('fill') == fill or
                    features.get('count') == count or
                    features.get('relation') == relation):
                    return rule
        
        # Fallback to random rule selection
        return random.choice(self.rules)
    
    def _generate_single_scene(self, 
                               rule: BongardRule, 
                               num_objects: int, 
                               is_positive: bool,
                               max_attempts: int = 10) -> Optional[Dict[str, Any]]:
        """Generate a single scene following the given rule."""
        
        for attempt in range(max_attempts):
            try:
                # Use CP-SAT sampler to generate objects
                objects = sample_scene_cp(
                    rule, 
                    num_objects, 
                    is_positive,
                    self.canvas_size,
                    self.scene_params.min_obj_size,
                    self.scene_params.max_obj_size,
                    max_attempts=5
                )
                # Enforce model.Validate() if possible (handled inside sample_scene_cp)
                if objects is None:
                    continue
                # Always honor the requested rule
                if not self._validate_scene(objects, rule, is_positive):
                    continue
                # Generate scene graph
                scene_graph = self._generate_scene_graph(objects)
                return {
                    'objects': objects,
                    'scene_graph': scene_graph,
                    'rule': rule.description,
                    'positive': is_positive
                }
            except Exception as e:
                logger.debug(f"Scene generation attempt {attempt + 1} failed: {e}")
                continue
        logger.warning(f"Failed to generate scene for rule {rule.description} after {max_attempts} attempts")
        return None
    
    def _validate_scene(self, objects: List[Dict[str, Any]], rule: BongardRule, is_positive: bool) -> bool:
        """Validate that the scene follows the rule correctly."""
        try:
            features = rule.positive_features if is_positive else rule.negative_features
            
            # Check shape constraints
            if 'shape' in features:
                shapes = [obj.get('shape') for obj in objects]
                target_shape = features['shape']
                
                if is_positive:
                    if target_shape not in shapes:
                        return False
                else:
                    if target_shape in shapes:
                        return False
            
            # Check fill constraints
            if 'fill' in features:
                fills = [obj.get('fill') for obj in objects]
                target_fill = features['fill']
                
                if is_positive:
                    if target_fill not in fills:
                        return False
                else:
                    if target_fill in fills:
                        return False
            
            # Check count constraints
            if 'count' in features:
                target_count = features['count']
                actual_count = len(objects)
                
                if is_positive:
                    if actual_count != target_count:
                        return False
                else:
                    if actual_count == target_count:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Scene validation failed: {e}")
            return False
    
    def _generate_scene_graph(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate scene graph with spatial relationships."""
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i + 1:], i + 1):
                relation = self._compute_spatial_relation(obj1, obj2)
                if relation:
                    relations.append({
                        'object1': i,
                        'object2': j,
                        'type': relation
                    })
        
        return {
            'objects': len(objects),
            'relations': relations
        }
    
    def _compute_spatial_relation(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> Optional[str]:
        """Compute spatial relationship between two objects."""
        try:
            pos1 = obj1['position']
            pos2 = obj2['position']
            size1 = obj1.get('size', 30)
            size2 = obj2.get('size', 30)
            
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            distance = (dx**2 + dy**2)**0.5
            
            # Check for overlap/nesting
            if distance < (size1 + size2) / 4:
                return 'overlap'
            elif distance < (size1 + size2) / 2:
                return 'near'
            
            # Check directional relationships
            if abs(dx) > abs(dy):
                return 'right_of' if dx > 0 else 'left_of'
            else:
                return 'below' if dy > 0 else 'above'
                
        except Exception:
            return None
    
    def _generate_and_save_image(self, example: Dict[str, Any], objects: List[Dict[str, Any]]) -> Path:
        """Generate and save the visual representation of the scene."""
        try:
            # Create composite image (already binarized)
            image = create_composite_scene(objects, self.canvas_size)
            # Generate filename
            rule_name = example['rule'].replace('(', '_').replace(')', '').replace(' ', '_')
            label = 'pos' if example['positive'] else 'neg'
            filename = f"{example['id']:06d}_{rule_name}_{label}.png"
            # Save image
            image_path = self.output_dir / filename
            image.save(image_path)
            return image_path.relative_to(self.output_dir)
        except Exception as e:
            logger.error(f"Failed to save image for example {example['id']}: {e}")
            return Path("error.png")
    
    def _save_dataset_metadata(self, examples: List[Dict[str, Any]]):
        """Save dataset metadata to JSON."""
        metadata = {
            'dataset_info': {
                'total_examples': len(examples),
                'canvas_size': self.canvas_size,
                'scene_parameters': {
                    'min_obj_size': self.scene_params.min_obj_size,
                    'max_obj_size': self.scene_params.max_obj_size,
                    'available_shapes': self.scene_params.shapes,
                    'available_fills': self.scene_params.fills,
                    'available_colors': self.scene_params.colors
                },
                'generation_stats': {
                    'generated_count': self.generated_count,
                    'failed_count': self.failed_count,
                    'success_rate': self.generated_count / (self.generated_count + self.failed_count) if (self.generated_count + self.failed_count) > 0 else 0
                }
            },
            'coverage_stats': self.coverage_tracker.get_coverage_heatmap_data(),
            'examples': examples
        }
        
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved dataset metadata to {metadata_path}")
    
    def _generate_final_statistics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics."""
        stats = {
            'total_examples': len(examples),
            'positive_examples': sum(1 for ex in examples if ex['positive']),
            'negative_examples': sum(1 for ex in examples if not ex['positive']),
            'generation_success_rate': self.generated_count / (self.generated_count + self.failed_count) if (self.generated_count + self.failed_count) > 0 else 0,
            'coverage_stats': self.coverage_tracker.get_coverage_heatmap_data(),
            'rule_distribution': self._compute_rule_distribution(examples),
            'object_count_distribution': self._compute_object_count_distribution(examples)
        }
        
        return stats
    
    def _compute_rule_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute distribution of examples per rule."""
        distribution = {}
        for example in examples:
            rule = example['rule']
            distribution[rule] = distribution.get(rule, 0) + 1
        return distribution
    
    def _compute_object_count_distribution(self, examples: List[Dict[str, Any]]) -> Dict[int, int]:
        """Compute distribution of object counts."""
        distribution = {}
        for example in examples:
            count = len(example['objects'])
            distribution[count] = distribution.get(count, 0) + 1
        return distribution
    
    def _log_progress(self, current: int, total: int):
        """Log generation progress."""
        percentage = (current / total) * 100
        coverage_stats = self.coverage_tracker.get_coverage_heatmap_data()
        coverage_pct = (coverage_stats['covered_cells'] / coverage_stats['total_cells']) * 100
        
        logger.info(f"Progress: {current}/{total} ({percentage:.1f}%) - "
                    f"Coverage: {coverage_pct:.1f}% - "
                    f"Success rate: {self.generated_count/(self.generated_count + self.failed_count):.2f}")

class SyntheticBongardDataset:
    def __init__(self, rules, img_size=128, grayscale=True, flush_cache=False):
        self.dataset = BongardDataset(canvas_size=img_size)
        self.examples = []
        
        # Clear cache and reseed between runs
        if flush_cache:
            if hasattr(self.dataset, 'sampler') and hasattr(self.dataset.sampler, 'cache'):
                self.dataset.sampler.cache.clear()
        
        # Reseed RNG for diversity
        import random, numpy as np
        random.seed(None)
        np.random.seed(None)
        
        for rule_desc, count in rules:
            rule = self.dataset._select_rule_for_generation()
            for i in range(count):
                scene = self.dataset._generate_single_scene(rule, num_objects=2, is_positive=True)
                if scene:
                    img = create_composite_scene(scene['objects'], img_size)
                    self.examples.append({'image': img, 'rule': rule_desc, 'label': 1, 'scene_graph': scene['scene_graph']})
                
                # Force at least one object if scene generation failed
                if not scene:
                    logger.warning("Scene generation failed; forcing at least one random object")
                    obj = {
                        'x': safe_randint(20, img_size - 40),
                        'y': safe_randint(20, img_size - 40), 
                        'size': safe_randint(20, 40),
                        'shape': random.choice(['circle', 'triangle', 'square']),
                        'fill': random.choice(['solid', 'outline']),
                        'color': random.choice(['red', 'blue', 'green'])
                    }
                    img = create_composite_scene([obj], img_size)
                    self.examples.append({'image': img, 'rule': rule_desc, 'label': 1, 'scene_graph': {'objects': 1, 'relations': []}})
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def generate_bongard_dataset(output_dir: str = "synthetic_images",
                             total_examples: int = 10000,
                             canvas_size: int = 128,
                             target_quota: int = 50) -> Dict[str, Any]:
    """
    Convenience function to generate a complete Bongard dataset.
    
    Args:
        output_dir: Output directory for images and metadata
        total_examples: Number of examples to generate
        canvas_size: Size of generated images
        target_quota: Target examples per coverage cell
        
    Returns:
        Dataset generation statistics
    """
    dataset = BongardDataset(
        output_dir=output_dir,
        canvas_size=canvas_size,
        target_quota=target_quota
    )
    
    return dataset.generate_dataset(total_examples=total_examples)

