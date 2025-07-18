"""Fallback samplers for when CP-SAT fails or for specific rule types."""

import random
import logging
from typing import Dict, Any, List, Tuple

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    # randrange excludes hi, so we +1
    return random.randrange(lo, hi+1)

from ..bongard_rules import BongardRule
from .config_loader import SamplerConfig
from .spatial_sampler import RelationSampler
from ...config import ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, ATTRIBUTE_FILL_MAP

logger = logging.getLogger(__name__)

class FallbackSamplers:
    """Collection of fallback sampling strategies."""
    
    def __init__(self, cfg: SamplerConfig):
        self.cfg = cfg
        self.relation_sampler = RelationSampler(
            cfg.img_size, cfg.min_obj_size, cfg.max_obj_size
        )
        
        # Available attributes
        self.all_shapes = list(ATTRIBUTE_SHAPE_MAP.keys())
        self.all_colors = list(ATTRIBUTE_COLOR_MAP.keys())
        self.all_fills = list(ATTRIBUTE_FILL_MAP.keys())
    
    def sample(
        self,
        rule_obj: BongardRule,
        label: int,
        n_shapes: int
    ) -> List[List[Dict[str, Any]]]:
        """Main fallback sampling dispatch."""
        scenes = []
        
        # Determine which fallback strategy to use
        if 'relation' in rule_obj.positive_features:
            scenes = self._relation_sampler(rule_obj, label, n_shapes)
        elif 'shape' in rule_obj.positive_features:
            scenes = self._shape_sampler(rule_obj, label, n_shapes)
        elif 'fill' in rule_obj.positive_features:
            scenes = self._fill_sampler(rule_obj, label, n_shapes)
        elif 'count' in rule_obj.positive_features:
            scenes = self._count_sampler(rule_obj, label, n_shapes)
        else:
            # Generic fallback
            scenes = self._generic_sampler(rule_obj, label, n_shapes)
        
        logger.info(f"Fallback sampler generated {len(scenes)} scenes for rule '{rule_obj.description}', label {label}")
        return scenes
    
    def _relation_sampler(
        self,
        rule_obj: BongardRule,
        label: int,
        n_shapes: int
    ) -> List[List[Dict[str, Any]]]:
        """Sample scenes for relation rules."""
        scenes = []
        relation = rule_obj.positive_features.get('relation', 'none')
        
        for _ in range(self.cfg.batch_size):
            if label == 1:  # Positive example
                # Use relation sampler to create appropriate spatial configuration
                objs = self.relation_sampler.sample(n_shapes, relation)
            else:  # Negative example
                # Use random positioning to avoid the relation
                objs = [self._random_object() for _ in range(n_shapes)]
            
            # Add missing attributes
            for obj in objs:
                self._complete_object_attributes(obj)
            
            scenes.append(objs)
        
        return scenes
    
    def _shape_sampler(
        self,
        rule_obj: BongardRule,
        label: int,
        n_shapes: int
    ) -> List[List[Dict[str, Any]]]:
        """Sample scenes for shape rules."""
        scenes = []
        target_shape = rule_obj.positive_features.get('shape', 'triangle')
        
        for _ in range(self.cfg.batch_size):
            objs = []
            
            for i in range(n_shapes):
                obj = self._random_object()
                
                if label == 1:  # Positive example
                    obj['shape'] = target_shape
                else:  # Negative example
                    # At least one object should have different shape
                    if i == 0:  # First object is different
                        available_shapes = [s for s in self.all_shapes if s != target_shape]
                        obj['shape'] = random.choice(available_shapes)
                    else:
                        # Other objects can be target shape or different
                        obj['shape'] = random.choice(self.all_shapes)
                
                objs.append(obj)
            
            scenes.append(objs)
        
        return scenes
    
    def _fill_sampler(
        self,
        rule_obj: BongardRule,
        label: int,
        n_shapes: int
    ) -> List[List[Dict[str, Any]]]:
        """Sample scenes for fill rules."""
        scenes = []
        target_fill = rule_obj.positive_features.get('fill', 'solid')
        
        for _ in range(self.cfg.batch_size):
            objs = []
            
            for i in range(n_shapes):
                obj = self._random_object()
                
                if label == 1:  # Positive example
                    obj['fill'] = target_fill
                else:  # Negative example
                    if i == 0:  # First object is different
                        available_fills = [f for f in self.all_fills if f != target_fill]
                        obj['fill'] = random.choice(available_fills)
                    else:
                        obj['fill'] = random.choice(self.all_fills)
                
                objs.append(obj)
            
            scenes.append(objs)
        
        return scenes
    
    def _count_sampler(
        self,
        rule_obj: BongardRule,
        label: int,
        n_shapes: int
    ) -> List[List[Dict[str, Any]]]:
        """Sample scenes for count rules."""
        scenes = []
        target_count = rule_obj.positive_features.get('count', n_shapes)
        
        for _ in range(self.cfg.batch_size):
            if label == 1:  # Positive example
                actual_count = target_count
            else:  # Negative example
                # Different count
                possible_counts = [c for c in range(1, self.cfg.max_objs + 1) if c != target_count]
                actual_count = random.choice(possible_counts) if possible_counts else 1
            
            # Ensure count is within bounds
            actual_count = max(1, min(actual_count, self.cfg.max_objs))
            
            objs = [self._random_object() for _ in range(actual_count)]
            scenes.append(objs)
        
        return scenes
    
    def _generic_sampler(
        self,
        rule_obj: BongardRule,
        label: int,
        n_shapes: int
    ) -> List[List[Dict[str, Any]]]:
        """Generic fallback sampler for unknown rule types."""
        scenes = []
        
        for _ in range(self.cfg.batch_size):
            objs = [self._random_object() for _ in range(n_shapes)]
            scenes.append(objs)
        
        return scenes
    
    def _random_object(self) -> Dict[str, Any]:
        """Generate a random object with basic attributes."""
        size = safe_randint(self.cfg.min_obj_size, self.cfg.max_obj_size)
        margin = size // 2
        
        return {
            'shape': random.choice(self.all_shapes),
            'color': random.choice(self.all_colors),
            'fill': random.choice(self.all_fills),
            'orientation': 'upright',
            'texture': 'flat',
            'position': (
                random.uniform(margin, self.cfg.img_size - margin),
                random.uniform(margin, self.cfg.img_size - margin)
            ),
            'width_pixels': size,
            'height_pixels': size
        }
    
    def _complete_object_attributes(self, obj: Dict[str, Any]):
        """Complete missing attributes in an object."""
        # Ensure all required attributes are present
        if 'shape' not in obj:
            obj['shape'] = random.choice(self.all_shapes)
        if 'color' not in obj:
            obj['color'] = random.choice(self.all_colors)
        if 'fill' not in obj:
            obj['fill'] = random.choice(self.all_fills)
        if 'orientation' not in obj:
            obj['orientation'] = 'upright'
        if 'texture' not in obj:
            obj['texture'] = 'flat'
        # Ensure size attributes
        if 'width_pixels' not in obj or 'height_pixels' not in obj:
            size = safe_randint(self.cfg.min_obj_size, self.cfg.max_obj_size)
            obj['width_pixels'] = size
            obj['height_pixels'] = size
        
        # Ensure position
        if 'position' not in obj:
            margin = max(obj.get('width_pixels', 20), obj.get('height_pixels', 20)) // 2
            obj['position'] = (
                random.uniform(margin, self.cfg.img_size - margin),
                random.uniform(margin, self.cfg.img_size - margin)
            )
