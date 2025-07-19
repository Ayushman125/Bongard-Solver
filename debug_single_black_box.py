#!/usr/bin/env python3
"""Debug the specific case where image 0 becomes a black box."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from bongard_generator.dataset import SyntheticBongardDataset

def debug_single_image_generation():
    """Debug why some images are still black boxes."""
    print("🔍 Debugging Single Image Generation")
    print("=" * 50)
    
    # Create the same dataset
    rules = [('All objects are circles.', 1)]  # Just 1 example to debug
    dataset = SyntheticBongardDataset(rules, img_size=128)
    
    if len(dataset) == 0:
        print("❌ No examples generated!")
        return
    
    example = dataset[0]
    img = np.array(example['image'])
    scene_graph = example.get('scene_graph', {})
    objects = scene_graph.get('objects', [])
    
    print(f"📊 Image Analysis:")
    print(f"   Shape: {img.shape}")
    print(f"   Unique values: {np.unique(img)}")
    print(f"   Min/Max: {img.min()}/{img.max()}")
    print(f"   Mean: {img.mean():.1f}")
    print(f"   Type: {img.dtype}")
    
    print(f"\n📋 Scene Graph:")
    print(f"   Objects count: {len(objects)}")
    for i, obj in enumerate(objects):
        print(f"   Object {i}: {obj}")
    
    # Let's manually test the create_composite_scene function
    print(f"\n🎨 Testing create_composite_scene directly:")
    try:
        from bongard_generator.mask_utils import create_composite_scene
        
        # Test with the same objects
        test_image = create_composite_scene(objects, canvas_size=128)
        test_array = np.array(test_image)
        
        print(f"   Direct creation - Shape: {test_array.shape}")
        print(f"   Direct creation - Unique values: {np.unique(test_array)}")
        print(f"   Direct creation - Mean: {test_array.mean():.1f}")
        
        # Save for inspection
        test_image.save("debug_direct_image.png")
        print(f"   💾 Saved debug_direct_image.png")
        
    except Exception as e:
        print(f"   ❌ Direct creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_image_generation()
