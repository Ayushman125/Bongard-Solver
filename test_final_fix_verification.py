#!/usr/bin/env python3
"""Test the fixed mask generation pipeline with real action commands."""

import numpy as np
import cv2
import logging
from src.bongard_augmentor.hybrid import ActionMaskGenerator

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_with_original_commands():
    """Test with the exact action commands from your original issue."""
    
    print("=== TESTING ORIGINAL PROBLEMATIC COMMANDS ===")
    
    # These are the exact commands from your issue
    action_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121', 
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121'
    ]
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print(f"Action commands: {action_commands}")
    
    # Test the full pipeline
    mask = generator.generate_mask_from_actions(action_commands, "bd_sector30_0000_pos_0")
    
    # Analyze the results
    mask_sum = np.sum(mask)
    nonzero_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    avg_pixel_value = mask_sum / total_pixels
    
    print(f"\n=== RESULTS ===")
    print(f"✅ Mask generated successfully!")
    print(f"   Total pixels: {total_pixels}")
    print(f"   Nonzero pixels: {nonzero_pixels} ({100*nonzero_pixels/total_pixels:.2f}%)")
    print(f"   Mask sum: {mask_sum}")
    print(f"   Average pixel value: {avg_pixel_value:.2f}")
    
    # Verify it's a proper binary mask
    unique_values = np.unique(mask)
    print(f"   Unique pixel values: {unique_values}")
    
    # Check if this is the problematic white mask
    if mask_sum > total_pixels * 200:
        print("❌ WARNING: Mask appears to be mostly white (coordinate issue)")
    elif nonzero_pixels == 0:
        print("❌ WARNING: Mask is completely empty")
    elif len(unique_values) <= 2 and 0 in unique_values and np.max(unique_values) == 255:
        print("✅ EXCELLENT: Proper binary mask with expected values")
    else:
        print(f"⚠️  Mask has unusual values: {unique_values}")
    
    # Save the result
    cv2.imwrite("mask_generation_FIXED.png", mask)
    print(f"   Saved result as 'mask_generation_FIXED.png'")
    
    # Compare with your original issue
    print(f"\n=== COMPARISON WITH ORIGINAL ISSUE ===")
    print(f"   Original issue: sum=65,925,915 (all white)")
    print(f"   Fixed version:  sum={mask_sum} ({'✅ FIXED' if mask_sum < total_pixels * 100 else '❌ Still broken'})")
    
    return mask

def create_comparison_visualization():
    """Create a visual comparison to show the fix."""
    
    print(f"\n=== CREATING COMPARISON VISUALIZATION ===")
    
    # Create the problematic "all white" mask that was the original issue
    white_mask = np.ones((512, 512), dtype=np.uint8) * 255
    
    # Load our fixed mask
    try:
        fixed_mask = cv2.imread("mask_generation_FIXED.png", cv2.IMREAD_GRAYSCALE)
        if fixed_mask is None:
            print("❌ Could not load fixed mask for comparison")
            return
    except:
        print("❌ Could not load fixed mask for comparison")
        return
    
    # Create side-by-side comparison
    comparison = np.hstack([white_mask, fixed_mask])
    
    # Add labels
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    
    # Add text labels
    cv2.putText(comparison_bgr, "BEFORE (White Mask Bug)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(comparison_bgr, "AFTER (Fixed)", (522, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imwrite("coordinate_fix_comparison.png", comparison_bgr)
    print("✅ Comparison saved as 'coordinate_fix_comparison.png'")

if __name__ == "__main__":
    mask = test_with_original_commands()
    create_comparison_visualization()
