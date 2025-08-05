#!/usr/bin/env python3
"""
Debug MaskRefiner constructor parameters.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.bongard_augmentor.refiners import MaskRefiner
import inspect

def debug_mask_refiner():
    """Debug what parameters MaskRefiner actually accepts."""
    
    print("=== MaskRefiner Class Debug ===")
    
    # Get the constructor signature
    sig = inspect.signature(MaskRefiner.__init__)
    print(f"Constructor signature: {sig}")
    
    # List all parameters
    print("\nAccepted parameters:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}: {param.default if param.default != param.empty else 'required'}")
    
    # Try to create with minimal parameters
    try:
        refiner = MaskRefiner()
        print(f"\n✅ Created MaskRefiner with default parameters")
    except Exception as e:
        print(f"\n❌ Failed to create MaskRefiner with defaults: {e}")
    
    # Try with specific parameters
    try:
        refiner2 = MaskRefiner(contour_approx_factor=0.01)
        print(f"✅ Created MaskRefiner with contour_approx_factor=0.01")
    except Exception as e:
        print(f"❌ Failed with contour_approx_factor: {e}")
        
    # Try with passes parameter
    try:
        refiner3 = MaskRefiner(passes=1)
        print(f"✅ Created MaskRefiner with passes=1")
    except Exception as e:
        print(f"❌ Failed with passes parameter: {e}")

if __name__ == "__main__":
    debug_mask_refiner()
