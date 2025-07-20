"""
Modular Bongard problem generator package.

This package provides a comprehensive framework for generating Bongard problems
with configurable rules, constraints, and validation capabilities.

Key improvements in this version:
- Proper gradient fill implementation with mask-based rendering
- Full spatial relationship sampler with grid-based positioning
- CP-SAT constraint validation before solving
- Comprehensive coverage tracking with quota management
- Modular architecture with separate mask utilities
- Extensive test suite for validation and smoke testing
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

__version__ = "1.0.0"

# Import core components with error handling
try:
    from .config_loader import SamplerConfig, get_config, get_sampler_config
    from .rule_loader import BongardRule, get_all_rules, get_rule_by_description
    from .sampler import BongardSampler
    from .coverage import CoverageTracker, AdversarialSampler
    from .validation import ValidationSuite
    from .constraints import PlacementOptimizer, ConstraintGenerator
    from .spatial_sampler import RelationSampler
    from .drawing import TurtleCanvas, ShapeDrawer
    
    # All imports successful
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    # Handle import errors gracefully
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    IMPORTS_SUCCESSFUL = False
    
    # Create minimal fallback classes
    class SamplerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class BongardRule:
        def __init__(self, description="", positive_features=None, negative_features=None):
            self.description = description
            self.positive_features = positive_features or {}
            self.negative_features = negative_features or {}
    
    class BongardSampler:
        def __init__(self, config=None):
            self.config = config or SamplerConfig()
        
        def sample_problem(self, **kwargs):
            return {"error": "BongardSampler not available due to import errors"}
        
        def run_validation(self):
            return False
    
    def get_config():
        return {}
    
    def get_sampler_config(**kwargs):
        return SamplerConfig(**kwargs)
    
    def get_all_rules():
        return []
    
    def get_rule_by_description(desc):
        return None
    
    def validate_installation():
        return False

# Legacy compatibility class
class BongardDataset:
    """
    Legacy dataset class providing backward compatibility.
    
    This class wraps the new BongardSampler for compatibility with existing code.
    New code should use BongardSampler directly.
    """
    
    def __init__(self, config=None):
        """Initialize with optional configuration."""
        if config is None:
            config = get_sampler_config()
        elif isinstance(config, dict):
            # Convert dict config to SamplerConfig
            config = SamplerConfig(**config)
        
        self.config = config
        self.sampler = BongardSampler(config)
        self.problems = []
        
        # Run validation on initialization
        if not self.sampler.run_validation():
            logging.warning("Validation failed during initialization")
    
    def generate_problem(self, rule=None, **kwargs):
        """
        Generate a single Bongard problem.
        
        Args:
            rule: Rule description string (optional)
            **kwargs: Additional arguments passed to sampler
            
        Returns:
            Dictionary containing the Bongard problem
        """
        problem = self.sampler.sample_problem(rule_description=rule, **kwargs)
        if problem:
            self.problems.append(problem)
        return problem
    
    def generate_dataset(self, num_problems=10, **kwargs):
        """
        Generate a dataset of Bongard problems.
        
        Args:
            num_problems: Number of problems to generate
            **kwargs: Additional arguments passed to sampler
            
        Returns:
            Dataset dictionary
        """
        dataset = self.sampler.generate_dataset(num_problems=num_problems, **kwargs)
        self.problems.extend(dataset.get('problems', []))
        return dataset
    
    def get_coverage_report(self):
        """Get coverage statistics."""
        return self.sampler.get_coverage_report()
    
    def run_validation(self):
        """Run validation suite."""
        return self.sampler.run_validation()

# Convenience functions
def create_sampler(config=None, **config_kwargs):
    """
    Create a new BongardSampler instance.
    
    Args:
        config: Optional SamplerConfig instance
        **config_kwargs: Configuration parameters to override
        
    Returns:
        BongardSampler instance
    """
    if config is None:
        config = get_sampler_config()
    
    # Override config with any provided kwargs
    if config_kwargs:
        config_dict = config.__dict__.copy()
        config_dict.update(config_kwargs)
        config = SamplerConfig(**config_dict)
    
    return BongardSampler(config)

def generate_single_problem(rule=None, config=None, **kwargs):
    """
    Generate a single Bongard problem quickly.
    
    Args:
        rule: Optional rule description
        config: Optional sampler configuration
        **kwargs: Additional generation parameters
        
    Returns:
        Bongard problem dictionary
    """
    sampler = create_sampler(config)
    return sampler.sample_problem(rule_description=rule, **kwargs)

def generate_dataset(num_problems=10, config=None, **kwargs):
    """
    Generate a dataset of Bongard problems quickly.
    
    Args:
        num_problems: Number of problems to generate
        config: Optional sampler configuration  
        **kwargs: Additional generation parameters
        
    Returns:
        Dataset dictionary
    """
    sampler = create_sampler(config)
    return sampler.generate_dataset(num_problems=num_problems, **kwargs)

def validate_installation():
    """
    Validate that the package is properly installed and configured.
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        validator = ValidationSuite()
        results = validator.run_all_validations()
        validator.print_validation_report()
        return all(results.values())
    except Exception as e:
        logging.error(f"Installation validation failed: {e}")
        return False

# Main exports
__all__ = [
    # Core classes
    'BongardSampler',
    'SamplerConfig', 
    'BongardRule',
    'BongardDataset',  # Legacy compatibility
    
    # Component classes
    'CoverageTracker',
    'AdversarialSampler',
    'ValidationSuite',
    'PlacementOptimizer',
    'ConstraintGenerator',
    'RelationSampler',
    'TurtleCanvas',
    'ShapeDrawer',
    
    # Convenience functions
    'create_sampler',
    'generate_single_problem',
    'generate_dataset',
    'validate_installation',
    
    # Configuration functions
    'get_config',
    'get_sampler_config',
    'get_all_rules',
    'get_rule_by_description',
    
    # Version
    '__version__'
]

# Package-level logger
logger = logging.getLogger(__name__)
logger.info(f"Bongard Generator v{__version__} loaded successfully")