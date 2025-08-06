#!/usr/bin/env python3
"""
Quick test of the cleaned ActionMaskGenerator
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from src.bongard_augmentor.hybrid import ActionMaskGenerator
import numpy as np

def test_clean_hybrid():
    print("=== Testing Cleaned Hybrid.py ===")
    
    # Create ActionMaskGenerator
    try:
        generator = ActionMaskGenerator(canvas_size=128)
        print("✅ ActionMaskGenerator created successfully")
    except Exception as e:
        print(f"❌ Failed to create ActionMaskGenerator: {e}")
        return False
    
    # Test mask generation with sample commands
    test_commands = ["line_triangle_1.000-0.500", "line_normal_0.600-0.750"]
    
    try:
        mask = generator.generate_mask(test_commands, "test_problem")
        print(f"✅ Mask generated: shape={mask.shape if mask is not None else 'None'}")
        
        if mask is not None:
            nonzero = np.count_nonzero(mask)
            print(f"   Non-zero pixels: {nonzero}")
            return True
        else:
            print("❌ Mask is None")
            return False
            
    except Exception as e:
        print(f"❌ Failed to generate mask: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clean_hybrid()
    if success:
        print("\n🎉 Clean hybrid.py test passed!")
    else:
        print("\n❌ Clean hybrid.py test failed!")
