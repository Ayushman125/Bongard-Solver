#!/usr/bin/env python3
"""
Test script to verify mask refinement with cropping fixes.
"""

import numpy as np
import cv2
from src.bongard_augmentor.hybrid import ActionMaskGenerator

def test_mask_refinement():
    """Test mask refinement to ensure no cropping occurs."""
    print("=== TESTING MASK REFINEMENT WITH CROPPING FIXES ===")
    
    # Initialize mask generator
    mask_gen = ActionMaskGenerator()
    
    # Test semantic commands that should produce full-size masks
    test_commands = [
        "line_triangle_1.000-0.500",
        "arc_circle_0.500_0.625-0.500",
        "line_square_0.800-0.750"
    ]
    
    for i, command in enumerate(test_commands):
        print(f"\n{i+1}. Testing refinement for: {command}")
        
        # Generate the mask
        mask = mask_gen.generate_mask_from_actions([command])
        
        # Check mask properties
        mask_shape = mask.shape
        pixel_count = np.sum(mask > 0)
        non_zero_coords = np.where(mask > 0)
        
        if len(non_zero_coords[0]) > 0:
            min_y, max_y = non_zero_coords[0].min(), non_zero_coords[0].max()
            min_x, max_x = non_zero_coords[1].min(), non_zero_coords[1].max()
            bbox_width = max_x - min_x + 1
            bbox_height = max_y - min_y + 1
        else:
            min_y = max_y = min_x = max_x = 0
            bbox_width = bbox_height = 0
        
        print(f"   📏 Mask shape: {mask_shape}")
        print(f"   🎨 Pixel count: {pixel_count}")
        print(f"   📦 Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
        print(f"   📐 Bbox size: {bbox_width}x{bbox_height}")
        
        # Verify mask is full size (should be 512x512)
        expected_size = (512, 512)
        if mask_shape == expected_size:
            print(f"   ✅ Mask maintains full canvas size: {expected_size}")
        else:
            print(f"   ❌ Mask cropped! Expected {expected_size}, got {mask_shape}")
        
        # Check if mask has content (not empty)
        if pixel_count > 0:
            print(f"   ✅ Mask has content: {pixel_count} pixels")
        else:
            print(f"   ❌ Mask is empty!")
        
        # Save test mask for visual inspection
        output_path = f"test_refinement_mask_{i}.png"
        cv2.imwrite(output_path, mask)
        print(f"   💾 Saved mask: {output_path}")

if __name__ == "__main__":
    test_mask_refinement()
