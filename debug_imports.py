#!/usr/bin/env python3
"""Simple test to check imports and basic functionality."""

print("Starting debug script...")

try:
    import numpy as np
    print("✅ NumPy imported")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")
    exit(1)

try:
    import cv2
    print("✅ OpenCV imported")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")
    exit(1)

try:
    from src.bongard_augmentor.hybrid import ActionMaskGenerator
    print("✅ ActionMaskGenerator imported")
except Exception as e:
    print(f"❌ ActionMaskGenerator import failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

try:
    from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
    print("✅ ComprehensiveNVLabsParser imported")
except Exception as e:
    print(f"❌ ComprehensiveNVLabsParser import failed: {e}")
    import traceback
    traceback.print_exc()

print("All imports successful!")

# Test basic functionality
try:
    generator = ActionMaskGenerator(image_size=(512, 512))
    print("✅ ActionMaskGenerator initialized")
    
    # Test simple mask generation
    simple_vertices = [(100.0, 100.0), (400.0, 400.0)]
    mask = generator._manual_render_fallback(simple_vertices)
    print(f"✅ Manual render test: sum={np.sum(mask)}, nonzero={np.count_nonzero(mask)}")
    
except Exception as e:
    print(f"❌ Basic functionality test failed: {e}")
    import traceback
    traceback.print_exc()

print("Debug script completed!")
