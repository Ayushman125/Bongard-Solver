"""Configuration loader for Bongard generator"""

import os
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SamplerConfig:
    """Configuration for the Bongard generator sampler."""
    img_size: int = 128
    min_obj_size: int = 20
    max_obj_size: int = 80
    max_objs: int = 4
    pad: int = 2
    diversity_archive: bool = True
    cp_time_limit: float = 5.0
    cp_workers: int = 8
    fallback: str = "cp_sat"
    batch_size: int = 100
    debug_mode: bool = False
    generator_mode: str = "cp_sat"
    default_fill_type: str = "solid"
    cache_stratified_cells: bool = True
    use_cp_sat: bool = True
    use_advanced_drawing: bool = True
    enable_caching: bool = True
    max_generation_attempts: int = 10
    # Additional parameters for backward compatibility
    total: int = 1000
    hybrid_split: dict = None
    
    def __post_init__(self):
        """Initialize default values after construction."""
        if self.hybrid_split is None:
            self.hybrid_split = {'cp': 0.7, 'ga': 0.3}

# Try to import from the main config, fall back to defaults
try:
    from config import CONFIG, ATTRIBUTE_SHAPE_MAP, ATTRIBUTE_COLOR_MAP, \
                        ATTRIBUTE_FILL_MAP, ATTRIBUTE_SIZE_MAP, \
                        ATTRIBUTE_ORIENTATION_MAP, ATTRIBUTE_TEXTURE_MAP, \
                        RELATION_MAP, IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS, DATA_ROOT_PATH
    
    logger.info("Successfully loaded configuration from main config.py")
    
except ImportError:
    logger.warning("Could not import from main config.py, using fallback configuration")
    
    # Fallback configuration
    CONFIG = {
        'data': {
            'image_size': [128, 128],
            'synthetic_data_config': {
                'max_objects_per_image': 4,
                'object_size_range': [20, 80],
                'padding': 10,
                'min_dist_objects': 0.1,
                'relation_density': 0.5,
                'occluder_prob': 0.3,
                'blur_prob': 0.2,
                'min_occluder_size': 5,
                'jitter_width_range': [1, 3],
                'jitter_dash_options': [None, (4,2), (2,2,2)],
                'use_background_textures': True,
                'background_texture_path': './data/textures',
                'stamp_path': './data/stamps',
                'program_library_path': './data/programs',
                'font_path': None,
                'cp_time_limit': 60.0,
                'sampler_batch_size': 100,
                'generator_mode': 'cp_sat',
                'default_fill_type': 'solid'
            }
        }
    }
    
    ATTRIBUTE_SHAPE_MAP = {
        'triangle': 0, 'square': 1, 'circle': 2, 'pentagon': 3, 'star': 4,
        'arc': 5, 'zigzag': 6, 'fan': 7, 'composite': 8, 'freeform': 9,
        'grid': 10, 'text_char': 11, 'icon_arrow': 12, 'bump': 13,
        'stamp': 14, 'program': 15
    }
    
    ATTRIBUTE_COLOR_MAP = {
        'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
        'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
        'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'gray': (128, 128, 128)
    }
    
    ATTRIBUTE_FILL_MAP = {
        'solid': 0, 'outline': 1, 'striped': 2, 'dotted': 3, 'gradient': 4, 'noise': 5
    }
    
    ATTRIBUTE_SIZE_MAP = {'small': 0, 'medium': 1, 'large': 2}
    
    ATTRIBUTE_ORIENTATION_MAP = {'upright': 0, 'rotated': 1}
    
    ATTRIBUTE_TEXTURE_MAP = {'flat': 0, 'textured': 1}
    
    RELATION_MAP = {
        'none': 0, 'left_of': 1, 'above': 2, 'near': 3, 'overlap': 4, 'nested': 5
    }
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    NUM_CHANNELS = 3
    DATA_ROOT_PATH = './data'

def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return CONFIG

def get_sampler_config(**kwargs) -> SamplerConfig:
    """Create a SamplerConfig with optional overrides."""
    # Extract relevant config from CONFIG
    synth_cfg = CONFIG['data'].get('synthetic_data_config', {})
    
    defaults = {
        'img_size': CONFIG['data']['image_size'][0],
        'min_obj_size': synth_cfg.get('object_size_range', [20, 80])[0],
        'max_obj_size': synth_cfg.get('object_size_range', [20, 80])[1],
        'max_objs': synth_cfg.get('max_objects_per_image', 4),
        'cp_time_limit': synth_cfg.get('cp_time_limit', 60.0),
        'batch_size': synth_cfg.get('sampler_batch_size', 100),
        'generator_mode': synth_cfg.get('generator_mode', 'cp_sat'),
        'default_fill_type': synth_cfg.get('default_fill_type', 'solid'),
        'total': kwargs.get('total', 1000),
        'hybrid_split': kwargs.get('hybrid_split', {'cp': 0.7, 'ga': 0.3})
    }
    
    # Override with any passed kwargs
    defaults.update(kwargs)
    
    # Create SamplerConfig
    config = SamplerConfig(**defaults)
    
    # Add backward compatibility for dict-style access
    # Some legacy code expects config['data']['hybrid_split']
    if hasattr(config, '__getitem__'):
        # Already supports dict access
        pass
    else:
        # Add dict-style access support
        class BackwardCompatibleConfig(SamplerConfig):
            def __getitem__(self, key):
                if key == 'data':
                    return {
                        'hybrid_split': self.hybrid_split,
                        'total': self.total,
                        'image_size': [self.img_size, self.img_size],
                        'synthetic_data_config': {
                            'img_size': self.img_size,
                            'min_obj_size': self.min_obj_size,
                            'max_obj_size': self.max_obj_size,
                            'max_objs': self.max_objs,
                            'cp_time_limit': self.cp_time_limit,
                            'batch_size': self.batch_size,
                            'generator_mode': self.generator_mode,
                            'default_fill_type': self.default_fill_type
                        }
                    }
                return getattr(self, key)
            
            def __setitem__(self, key, value):
                setattr(self, key, value)
        
        # Transfer all attributes from SamplerConfig to BackwardCompatibleConfig
        config = BackwardCompatibleConfig(**defaults)
    
    return config
