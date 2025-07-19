#!/usr/bin/env python3
"""Debug the image generation issue with black boxes."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def test_color_conversion():
    """Test how colors convert to grayscale and binarize."""
    print("Testing color to grayscale conversion...")
    
    # Create a test image
    img = Image.new('RGB', (128, 128), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)
    
    # Test colors
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'green': (0, 255, 0),
        'black': (0, 0, 0),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128)
    }
    
    # Draw colored rectangles
    y = 10
    for name, color in colors.items():
        draw.rectangle([10, y, 50, y+15], fill=color)
        y += 20
        
        # Check grayscale conversion
        gray_val = int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        print(f"{name:8} RGB{color} -> Gray={gray_val} -> Binary={'BLACK' if gray_val < 128 else 'WHITE'}")
    
    # Convert to grayscale
    img_gray = img.convert('L')
    
    # Test current binarization logic
    img_bw_current = img_gray.point(lambda p: 0 if p < 128 else 255, mode='L')
    
    # Test improved binarization logic (threshold at 250 to catch near-white background)
    img_bw_improved = img_gray.point(lambda p: 0 if p < 250 else 255, mode='L')
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0,0].imshow(img)
    axes[0,0].set_title('Original RGB')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img_gray, cmap='gray')
    axes[0,1].set_title('Grayscale')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(img_bw_current, cmap='gray')
    axes[1,0].set_title('Binary (threshold=128)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(img_bw_improved, cmap='gray')
    axes[1,1].set_title('Binary (threshold=250)')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Check unique values
    print(f"\nUnique values in grayscale: {np.unique(np.array(img_gray))}")
    print(f"Unique values in binary (128): {np.unique(np.array(img_bw_current))}")
    print(f"Unique values in binary (250): {np.unique(np.array(img_bw_improved))}")

if __name__ == "__main__":
    test_color_conversion()
