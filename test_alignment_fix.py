#!/usr/bin/env python3
"""Test the coordinate system fix for alignment issues."""

import numpy as np
import cv2
import logging
from src.bongard_augmentor.hybrid import ActionMaskGenerator

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_coordinate_alignment_fix():
    """Test the coordinate alignment fix with the problematic commands from logs."""
    
    print("=== TESTING COORDINATE ALIGNMENT FIX ===")
    
    # Use the exact commands from the problematic logs
    action_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121', 
        'line_normal_0.224-0.600',
        'line_normal_1.000-0.074',
        'line_normal_0.224-0.926',
        'line_normal_0.224-0.352'
    ]
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print(f"Testing with problematic commands: {action_commands}")
    
    # Test the full pipeline
    try:
        mask = generator.generate_mask_from_actions(action_commands, "bd_asymmetric_unbala_x_0000")
        
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
        
        # Check if this resolves the white mask issue
        if mask_sum > total_pixels * 200:
            print("❌ WARNING: Mask is still mostly white")
        elif nonzero_pixels == 0:
            print("❌ WARNING: Mask is completely empty")
        elif 1000 < nonzero_pixels < 50000:  # Reasonable range for line art
            print("✅ EXCELLENT: Mask has reasonable content!")
        else:
            print(f"⚠️  Mask coverage might be unusual")
        
        # Save the result for visual inspection
        cv2.imwrite("mask_alignment_FIXED.png", mask)
        print(f"   💾 Saved result as 'mask_alignment_FIXED.png'")
        
        # Test with another command set that was causing issues
        print(f"\n=== TESTING SECOND PROBLEMATIC SET ===")
        test_commands_2 = [
            'line_normal_0.500-0.500',
            'line_normal_0.707-0.875', 
            'line_normal_0.500-0.875',
            'line_normal_0.300-0.167'
        ]
        
        mask2 = generator.generate_mask_from_actions(test_commands_2, "test_2")
        mask2_sum = np.sum(mask2)
        mask2_nonzero = np.count_nonzero(mask2)
        
        print(f"Second test: sum={mask2_sum}, nonzero={mask2_nonzero}")
        
        if mask2_sum > total_pixels * 200:
            print("❌ Second test still produces white masks")
        else:
            print("✅ Second test looks reasonable")
            
        cv2.imwrite("mask_alignment_test2.png", mask2)
        
        return mask, mask2
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_coordinate_approaches():
    """Compare different coordinate interpretation approaches."""
    
    print(f"\n=== COMPARING COORDINATE APPROACHES ===")
    
    # Test vertices from logs that were causing issues
    test_vertices = [
        (0.0, 0.0), 
        (1440.0, 0.0), 
        (805.55, -603.33), 
        (747.10, -920.55),
        (345.35, 462.27),
        (286.91, 145.05),
        (-2.32, 2.23)
    ]
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print(f"Test vertices: {test_vertices[:3]}...")
    
    # Test the alternative rendering approach
    mask_alt = generator._render_vertices_alternative(test_vertices)
    alt_sum = np.sum(mask_alt)
    alt_nonzero = np.count_nonzero(mask_alt)
    
    print(f"Alternative approach: sum={alt_sum}, nonzero={alt_nonzero}")
    
    # Test manual fallback
    mask_manual = generator._manual_render_fallback(test_vertices)
    manual_sum = np.sum(mask_manual)
    manual_nonzero = np.count_nonzero(mask_manual)
    
    print(f"Manual fallback: sum={manual_sum}, nonzero={manual_nonzero}")
    
    # Save comparison
    cv2.imwrite("compare_alternative.png", mask_alt)
    cv2.imwrite("compare_manual.png", mask_manual)
    
    print("Comparison masks saved")

if __name__ == "__main__":
    test_coordinate_alignment_fix()
    compare_coordinate_approaches()
