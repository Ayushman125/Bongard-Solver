#!/usr/bin/env python3
"""Debug the specific case where solid fill images become white."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from bongard_generator.dataset import SyntheticBongardDataset
from bongard_generator.mask_utils import create_composite_scene
from PIL import Image

def debug_solid_fill_issue():
    """Debug why solid fill images sometimes become all white."""
    print("🔍 Debugging Solid Fill Issue")
    print("=" * 50)
    
    # Test the exact objects that failed
    failed_objects = [
        {'x': 81, 'y': 42, 'size': 22, 'shape': 'square', 'fill': 'solid', 'color': 'yellow', 'position': (81, 42)}, 
        {'x': 77, 'y': 22, 'size': 25, 'shape': 'star', 'fill': 'solid', 'color': 'green', 'position': (77, 22)}
    ]
    
    print("📋 Testing failed objects:")
    for i, obj in enumerate(failed_objects):
        print(f"   Object {i}: {obj}")
    
    # Test direct image creation
    print("\n🎨 Testing create_composite_scene directly:")
    try:
        test_image = create_composite_scene(failed_objects, canvas_size=128)
        test_array = np.array(test_image)
        
        print(f"   Created image - Shape: {test_array.shape}")
        print(f"   Created image - Unique values: {np.unique(test_array)}")
        print(f"   Created image - Mean: {test_array.mean():.1f}")
        print(f"   Created image - Type: {test_array.dtype}")
        
        # Save the direct test
        test_image.save("debug_solid_fill_direct.png")
        print("   💾 Saved debug_solid_fill_direct.png")
        
        # Test conversion to binary
        if test_image.mode != 'L':
            gray_image = test_image.convert('L')
            print(f"   Grayscale - Unique values: {np.unique(np.array(gray_image))}")
            
            # Apply threshold (same as dataset does)
            binary_array = np.array(gray_image)
            binary_array = (binary_array > 128).astype(np.uint8) * 255
            print(f"   Binary (>128) - Unique values: {np.unique(binary_array)}")
            
            # Save binary version
            Image.fromarray(binary_array).save("debug_solid_fill_binary.png")
            print("   💾 Saved debug_solid_fill_binary.png")
            
    except Exception as e:
        print(f"   ❌ Direct creation failed: {e}")
        import traceback
        traceback.print_exc()

def test_various_solid_objects():
    """Test different solid fill combinations to find the pattern."""
    print("\n🧪 Testing Various Solid Fill Combinations:")
    print("=" * 50)
    
    test_cases = [
        # Different shapes with solid fill
        [{'x': 40, 'y': 40, 'size': 30, 'shape': 'circle', 'fill': 'solid', 'color': 'red'}],
        [{'x': 40, 'y': 40, 'size': 30, 'shape': 'square', 'fill': 'solid', 'color': 'blue'}],
        [{'x': 40, 'y': 40, 'size': 30, 'shape': 'star', 'fill': 'solid', 'color': 'green'}],
        [{'x': 40, 'y': 40, 'size': 30, 'shape': 'triangle', 'fill': 'solid', 'color': 'yellow'}],
        # The problematic combination
        [{'x': 81, 'y': 42, 'size': 22, 'shape': 'square', 'fill': 'solid', 'color': 'yellow'}, 
         {'x': 77, 'y': 22, 'size': 25, 'shape': 'star', 'fill': 'solid', 'color': 'green'}]
    ]
    
    for i, objects in enumerate(test_cases):
        print(f"\n🧪 Test Case {i+1}: {len(objects)} object(s)")
        for obj in objects:
            print(f"    {obj['shape']} ({obj['color']}, {obj['fill']})")
        
        try:
            img = create_composite_scene(objects, canvas_size=128)
            arr = np.array(img)
            
            # Convert to binary as dataset does
            if img.mode != 'L':
                gray = img.convert('L')
                binary = (np.array(gray) > 128).astype(np.uint8) * 255
            else:
                binary = (arr > 128).astype(np.uint8) * 255
            
            unique_vals = np.unique(binary)
            print(f"    Result: {unique_vals} (Mean: {binary.mean():.1f})")
            
            if len(unique_vals) <= 1:
                print(f"    ❌ BLANK IMAGE detected!")
            else:
                print(f"    ✅ Good image")
                
        except Exception as e:
            print(f"    💥 ERROR: {e}")

if __name__ == "__main__":
    debug_solid_fill_issue()
    test_various_solid_objects()
