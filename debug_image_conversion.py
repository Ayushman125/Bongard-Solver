#!/usr/bin/env python
"""Debug image generation and conversion issues."""
import os
import sys

# Add project root to path 
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = SCRIPT_DIR  # We're at root level
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)     # for core_models/
sys.path.insert(0, SRC_ROOT)      # for src/data, src/perception, src/utils

from PIL import Image, ImageDraw
import numpy as np

def test_image_conversion():
    """Test the create_composite_scene function to see why images might appear as black boxes."""
    
    # Test the same function that's used in dataset.py
    canvas_size = 128
    
    # Create RGB canvas for better color handling
    img = Image.new('RGB', (canvas_size, canvas_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Test object similar to what the generator produces
    obj = {
        'x': 64, 'y': 64, 'size': 40,
        'shape': 'circle', 'fill': 'solid', 'color': 'red',
        'position': (64, 64)
    }
    
    # Extract object properties with defaults
    x = obj.get('x', obj.get('position', [canvas_size // 2, canvas_size // 2])[0])
    y = obj.get('y', obj.get('position', [canvas_size // 2, canvas_size // 2])[1])
    size = obj.get('size', 30)
    shape = obj.get('shape', 'circle')
    fill_type = obj.get('fill', 'solid')
    color = obj.get('color', 'black')
    
    print(f"Drawing: {shape} at ({x},{y}) size={size} fill={fill_type} color={color}")
    
    # Convert color name to RGB if needed
    color_map = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0)
    }
    fill_color = color_map.get(color, (0, 0, 0))  # Default to black
    
    # Calculate bounding box
    half_size = size // 2
    bbox = [x - half_size, y - half_size, x + half_size, y + half_size]
    
    print(f"Fill color: {fill_color}, bbox: {bbox}")
    
    # Draw circle
    if fill_type == 'solid':
        draw.ellipse(bbox, fill=fill_color)
    else:
        draw.ellipse(bbox, outline=fill_color, width=3)
    
    # Save original RGB version
    img.save("debug_original_rgb.png")
    print("✓ Saved debug_original_rgb.png")
    
    # Test conversion to grayscale
    img_gray = img.convert('L')
    img_gray.save("debug_grayscale.png")
    print("✓ Saved debug_grayscale.png")
    
    # Test binarization (this is the key step that might be causing issues)
    img_bw = img_gray.point(lambda p: 0 if p < 128 else 255, mode='L')
    img_bw.save("debug_binarized.png")
    print("✓ Saved debug_binarized.png")
    
    # Check the pixel values
    gray_array = np.array(img_gray)
    bw_array = np.array(img_bw)
    
    print(f"Grayscale image stats: min={gray_array.min()}, max={gray_array.max()}, mean={gray_array.mean():.1f}")
    print(f"Binary image stats: min={bw_array.min()}, max={bw_array.max()}, unique values={np.unique(bw_array)}")
    
    # Check if the shape is actually drawn
    center_value = gray_array[64, 64]  # Should be the red circle
    corner_value = gray_array[10, 10]  # Should be white background
    
    print(f"Center pixel (red circle): grayscale={center_value}, binary={bw_array[64, 64]}")
    print(f"Corner pixel (white bg): grayscale={corner_value}, binary={bw_array[10, 10]}")
    
    # Test multiple objects like the real generator does
    print("\n=== Testing multiple objects ===")
    img2 = Image.new('RGB', (canvas_size, canvas_size), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    
    # Draw multiple shapes like the real generator
    objects = [
        {'x': 37, 'y': 33, 'size': 54, 'shape': 'circle', 'fill': 'solid', 'color': 'red', 'position': (37, 33)},
        {'x': 90, 'y': 90, 'size': 30, 'shape': 'square', 'fill': 'solid', 'color': 'blue', 'position': (90, 90)}
    ]
    
    for obj in objects:
        x = obj['x']
        y = obj['y'] 
        size = obj['size']
        shape = obj['shape']
        fill_type = obj['fill']
        color = obj['color']
        
        fill_color = color_map.get(color, (0, 0, 0))
        half_size = size // 2
        bbox = [x - half_size, y - half_size, x + half_size, y + half_size]
        
        if shape == 'circle':
            if fill_type == 'solid':
                draw2.ellipse(bbox, fill=fill_color)
            else:
                draw2.ellipse(bbox, outline=fill_color, width=3)
        elif shape == 'square':
            if fill_type == 'solid':
                draw2.rectangle(bbox, fill=fill_color)
            else:
                draw2.rectangle(bbox, outline=fill_color, width=3)
    
    # Save multi-object versions
    img2.save("debug_multi_original.png")
    img2_gray = img2.convert('L')
    img2_gray.save("debug_multi_grayscale.png")  
    img2_bw = img2_gray.point(lambda p: 0 if p < 128 else 255, mode='L')
    img2_bw.save("debug_multi_binarized.png")
    print("✓ Saved multi-object test images")
    
    # Final array inspection
    multi_array = np.array(img2_bw)
    print(f"Multi-object binary image: min={multi_array.min()}, max={multi_array.max()}")
    print(f"Pixel at (37,33): {multi_array[33, 37]} (should be red circle)")
    print(f"Pixel at (90,90): {multi_array[90, 90]} (should be blue square)")

if __name__ == "__main__":
    print("🔍 Debugging Image Generation and Conversion")
    print("=" * 50)
    test_image_conversion()
    print("=" * 50)
    print("✅ Debug complete. Check generated images for quality issues.")
