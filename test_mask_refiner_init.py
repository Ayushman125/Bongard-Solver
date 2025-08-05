#!/usr/bin/env python3
"""
Test MaskRefiner initialization fix.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.bongard_augmentor.hybrid import HybridAugmentationPipeline

def test_mask_refiner_init():
    """Test that MaskRefiner initializes correctly with valid parameters."""
    
    try:
        # Test with empty config
        config = {}
        pipeline = HybridAugmentationPipeline(config)
        print("✅ HybridAugmentationPipeline initialized successfully with empty config")
        
        # Test with some refinement config
        config_with_refinement = {
            'refinement': {
                'contour_approx_factor': 0.01,
                'min_component_size': 30,
                'closing_kernel_size': 3,
                'opening_kernel_size': 2,
                'passes': 1,
                'invalid_param': 'should_be_filtered'  # This should be filtered out
            }
        }
        pipeline2 = HybridAugmentationPipeline(config_with_refinement)
        print("✅ HybridAugmentationPipeline initialized successfully with refinement config")
        
        # Test mask generation
        test_commands = ['line_normal_1.000-0.500']
        mask = pipeline.mask_generator.generate_mask_from_actions(test_commands)
        print(f"✅ Generated mask with {mask.sum()} pixels")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mask_refiner_init()
    if success:
        print("\n🎯 All tests passed! MaskRefiner initialization is fixed.")
    else:
        print("\n💥 Tests failed! Need further fixes.")
