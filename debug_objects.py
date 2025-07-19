#!/usr/bin/env python3
"""Debug object positioning and drawing."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bongard_generator.dataset import create_composite_scene
import numpy as np
import matplotlib.pyplot as plt

def test_object_drawing():
    """Test if objects are being drawn correctly with known positions."""
    print("Testing object drawing with known positions...")
    
    # Create test objects with very clear, centered positions
    test_objects = [
        {
            'x': 32, 'y': 32, 'size': 20,
            'shape': 'circle', 'fill': 'solid', 'color': 'black',
            'position': (32, 32)
        },
        {
            'x': 96, 'y': 32, 'size': 20,
            'shape': 'square', 'fill': 'solid', 'color': 'red',
            'position': (96, 32)
        },
        {
            'x': 32, 'y': 96, 'size': 20,
            'shape': 'triangle', 'fill': 'outline', 'color': 'blue',
            'position': (32, 96)
        },
        {
            'x': 96, 'y': 96, 'size': 20,
            'shape': 'pentagon', 'fill': 'solid', 'color': 'green',
            'position': (96, 96)
        }
    ]
    
    print("Test objects:")
    for i, obj in enumerate(test_objects):
        print(f"  {i}: {obj}")
    
    # Generate image
    img = create_composite_scene(test_objects, 128)
    
    # Convert to numpy for analysis
    img_array = np.array(img)
    
    print(f"\nImage shape: {img_array.shape}")
    print(f"Image dtype: {img_array.dtype}")
    print(f"Unique values: {np.unique(img_array)}")
    print(f"Min: {img_array.min()}, Max: {img_array.max()}")
    print(f"Mean: {img_array.mean():.2f}")
    
    # Count non-white pixels (should be > 0 if shapes are drawn)
    non_white_pixels = np.sum(img_array < 255)
    total_pixels = img_array.size
    print(f"Non-white pixels: {non_white_pixels}/{total_pixels} ({non_white_pixels/total_pixels*100:.1f}%)")
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_array, cmap='gray')
    plt.title('Generated Image - Debug Test')
    plt.colorbar()
    
    # Add grid to see positioning
    for i in range(0, 128, 32):
        plt.axhline(y=i, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.3)
    
    # Mark expected object positions
    positions = [(32, 32), (96, 32), (32, 96), (96, 96)]
    colors = ['yellow', 'cyan', 'magenta', 'orange']
    for (x, y), color in zip(positions, colors):
        plt.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    plt.show()
    
    if non_white_pixels > 0:
        print("✅ SUCCESS: Objects are being drawn!")
        return True
    else:
        print("❌ FAILED: No objects drawn - all pixels are white!")
        return False

if __name__ == "__main__":
    success = test_object_drawing()
    if not success:
        print("\nThe issue is likely in the object generation or positioning logic.")
