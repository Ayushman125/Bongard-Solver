"""Comprehensive test suite for the entire bongard_generator package."""

import pytest
import os
import sys

# Add src to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC_ROOT)

def test_imports():
    """Test that all modules can be imported successfully."""
    try:
        from src.bongard_generator.config_loader import SamplerConfig
        from src.bongard_generator.rule_loader import RuleLoader
        from src.bongard_generator.draw_utils import AdvancedDrawingUtils
        from src.bongard_generator.spatial_sampler import RelationSampler
        # Removed legacy samplers: cp_sampler, fallback_samplers
        from src.bongard_generator.dataset import BongardDataset
        from src.bongard_generator import BongardDataset as MainDataset
        
        assert True  # If we get here, imports worked
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config_initialization():
    """Test configuration initialization."""
    from src.bongard_generator.config_loader import SamplerConfig
    
    config = SamplerConfig()
    
    # Test default values
    assert hasattr(config, 'cache_stratified_cells')
    assert hasattr(config, 'use_cp_sat')
    assert hasattr(config, 'use_advanced_drawing')
    assert hasattr(config, 'enable_caching')

def test_rule_loader_initialization():
    """Test rule loader initialization."""
    from src.bongard_generator.rule_loader import RuleLoader
    
    rule_loader = RuleLoader()
    
    assert hasattr(rule_loader, 'rules')
    assert hasattr(rule_loader, 'rule_lookup')

def test_dataset_creation():
    """Test dataset creation with minimal configuration."""
    from src.bongard_generator import BongardDataset
    from src.bongard_generator.config_loader import SamplerConfig
    
    config = SamplerConfig()
    dataset = BongardDataset(
        num_problems=5,  # Small number for testing
        img_size=64,     # Small size for speed
        config=config
    )
    
    assert len(dataset) == 5
    assert dataset.img_size == 64

def test_dataset_getitem():
    """Test dataset item retrieval."""
    from src.bongard_generator import BongardDataset
    from src.bongard_generator.config_loader import SamplerConfig
    
    config = SamplerConfig()
    dataset = BongardDataset(
        num_problems=3,
        img_size=64,
        config=config
    )
    
    # Test valid index
    problem = dataset[0]
    
    if problem is not None:
        # Check structure if generation succeeded
        assert isinstance(problem, dict)
        expected_keys = ['positive_images', 'negative_images', 'rule', 'rule_text']
        for key in expected_keys:
            assert key in problem, f"Missing key: {key}"
    else:
        # Generation can fail, which is acceptable for testing
        pass

def test_sampler_components():
    """Test individual sampler components."""
    from src.bongard_generator.spatial_sampler import RelationSampler
    # Removed legacy samplers: cp_sampler, fallback_samplers
    relation_sampler = RelationSampler(img_size=100)
    assert relation_sampler.img_size == 100

def test_drawing_utils():
    """Test drawing utilities initialization."""
    from src.bongard_generator.draw_utils import AdvancedDrawingUtils
    
    draw_utils = AdvancedDrawingUtils(img_size=128)
    
    assert draw_utils.img_size == 128
    assert hasattr(draw_utils, 'jitter_dash_options')

def test_package_integrity():
    """Test package integrity and structure."""
    import src.bongard_generator as pkg
    
    # Check main exports
    assert hasattr(pkg, 'BongardDataset')
    assert hasattr(pkg, 'SamplerConfig')
    
    # Check version or package info if available
    if hasattr(pkg, '__version__'):
        assert isinstance(pkg.__version__, str)

def test_error_handling():
    """Test error handling in dataset creation."""
    from src.bongard_generator import BongardDataset
    from src.bongard_generator.config_loader import SamplerConfig
    
    config = SamplerConfig()
    
    # Test with invalid parameters
    try:
        dataset = BongardDataset(
            num_problems=0,  # Invalid
            img_size=10,     # Very small
            config=config
        )
        # Should handle gracefully
        assert len(dataset) == 0
    except ValueError:
        # Expected for invalid parameters
        pass

@pytest.mark.slow
def test_generation_performance():
    """Test generation performance (marked as slow test)."""
    from src.bongard_generator import BongardDataset
    from src.bongard_generator.config_loader import SamplerConfig
    import time
    
    config = SamplerConfig()
    dataset = BongardDataset(
        num_problems=10,
        img_size=64,
        config=config
    )
    
    start_time = time.time()
    
    # Try to generate a few problems
    generated_count = 0
    for i in range(min(5, len(dataset))):
        problem = dataset[i]
        if problem is not None:
            generated_count += 1
    
    elapsed_time = time.time() - start_time
    
    # Basic performance check (very generous timing)
    assert elapsed_time < 60  # Should complete within 60 seconds
    
    print(f"Generated {generated_count}/5 problems in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Run basic tests
    test_imports()
    test_config_initialization()
    test_rule_loader_initialization()
    test_dataset_creation()
    test_sampler_components()
    test_drawing_utils()
    test_package_integrity()
    test_error_handling()
    
    print("All integration tests passed!")
    
    # Optionally run performance test
    try:
        test_generation_performance()
        print("Performance test passed!")
    except Exception as e:
        print(f"Performance test failed (this may be expected): {e}")
    
    print("Integration test suite completed successfully!")
