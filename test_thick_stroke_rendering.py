#!/usr/bin/env python3
"""Test the improved thick stroke rendering to match real dataset appearance."""

import numpy as np
import cv2
import logging
from src.bongard_augmentor.hybrid import ActionMaskGenerator

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_thick_stroke_rendering():
    """Test the improved thick stroke rendering with real action commands."""
    
    print("=== TESTING THICK STROKE RENDERING ===")
    
    # Use the exact action commands from the logs
    action_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121', 
        'line_normal_0.224-0.600',
        'line_normal_1.000-0.074',
        'line_normal_0.224-0.926',
        'line_normal_0.224-0.352'
    ]
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print(f"Action commands: {action_commands}")
    
    # Test the full pipeline with thick stroke rendering
    mask = generator.generate_mask_from_actions(action_commands, "test_thick_strokes")
    
    # Analyze the results
    mask_sum = np.sum(mask)
    nonzero_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    coverage_percent = 100 * nonzero_pixels / total_pixels
    
    print(f"\n=== THICK STROKE RESULTS ===")
    print(f"✅ Mask generated successfully!")
    print(f"   Total pixels: {total_pixels}")
    print(f"   Nonzero pixels: {nonzero_pixels} ({coverage_percent:.1f}%)")
    print(f"   Mask sum: {mask_sum}")
    print(f"   Average pixel value: {mask_sum/total_pixels:.1f}")
    
    # Compare with target coverage (real images have ~99% coverage)
    if coverage_percent > 20:
        print(f"✅ EXCELLENT: Coverage {coverage_percent:.1f}% much closer to real dataset!")
    elif coverage_percent > 5:
        print(f"✅ GOOD: Coverage {coverage_percent:.1f}% improved significantly")
    else:
        print(f"❌ Still low coverage: {coverage_percent:.1f}%")
    
    # Save the thick stroke result
    cv2.imwrite("test_thick_stroke_mask.png", mask)
    print(f"   Saved thick stroke mask: test_thick_stroke_mask.png")
    
    # Compare with real image if available
    try:
        real_img = cv2.imread("data/bd_asymmetric_unbala_x_0000_pos_0_real.png", cv2.IMREAD_GRAYSCALE)
        if real_img is not None:
            real_coverage = 100 * np.count_nonzero(real_img) / real_img.size
            print(f"   Real image coverage: {real_coverage:.1f}%")
            print(f"   Coverage ratio: {coverage_percent/real_coverage:.3f} (target: ~1.0)")
            
            # Create comparison
            comparison = create_comparison_image(real_img, mask)
            cv2.imwrite("thick_stroke_vs_real_comparison.png", comparison)
            print(f"   Saved comparison: thick_stroke_vs_real_comparison.png")
    except Exception as e:
        print(f"   Could not load real image for comparison: {e}")
    
    return mask

def create_comparison_image(real_img, generated_mask):
    """Create a side-by-side comparison of real image and generated mask."""
    
    # Resize to match if needed
    h, w = real_img.shape
    mask_resized = cv2.resize(generated_mask, (w, h))
    
    # Create side-by-side comparison
    comparison = np.hstack([real_img, mask_resized])
    
    # Convert to BGR for text overlay
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    
    # Add labels
    cv2.putText(comparison_bgr, "REAL IMAGE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comparison_bgr, "THICK STROKE MASK", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return comparison_bgr

def test_stroke_thickness_variations():
    """Test different stroke thickness values to find optimal match."""
    
    print("\n=== TESTING STROKE THICKNESS VARIATIONS ===")
    
    action_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121'
    ]
    
    # Test different thickness multipliers
    thickness_multipliers = [1, 2, 3, 4, 5]
    
    for multiplier in thickness_multipliers:
        # Temporarily modify the thickness calculation
        generator = ActionMaskGenerator(image_size=(512, 512))
        
        # Override the base thickness calculation
        original_method = generator._manual_render_fallback
        
        def thick_render_test(vertices):
            mask = np.zeros(generator.image_size, dtype=np.uint8)
            if len(vertices) < 2:
                return mask
            
            base_thickness = max(8, int(generator.image_size[0] * 0.02)) * multiplier
            scale_factor = 4
            high_res_size = (generator.image_size[0] * scale_factor, generator.image_size[1] * scale_factor)
            high_res_mask = np.zeros(high_res_size, dtype=np.uint8)
            
            points = []
            for vertex in vertices:
                if len(vertex) >= 2:
                    pixel_x = vertex[0] * scale_factor
                    pixel_y = vertex[1] * scale_factor
                    pixel_x = max(0, min(high_res_size[1] - 1, int(round(pixel_x))))
                    pixel_y = max(0, min(high_res_size[0] - 1, int(round(pixel_y))))
                    points.append([pixel_x, pixel_y])
            
            if len(points) >= 2:
                points_array = np.array(points, dtype=np.int32)
                thick_stroke = base_thickness * scale_factor
                cv2.polylines(high_res_mask, [points_array], False, 255, 
                             thickness=thick_stroke, lineType=cv2.LINE_AA)
            
            mask = cv2.resize(high_res_mask, (generator.image_size[1], generator.image_size[0]), interpolation=cv2.INTER_AREA)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            return mask
        
        generator._manual_render_fallback = thick_render_test
        
        mask = generator.generate_mask_from_actions(action_commands, f"thickness_test_{multiplier}")
        coverage = 100 * np.count_nonzero(mask) / mask.size
        
        print(f"Thickness x{multiplier}: {coverage:.1f}% coverage")
        cv2.imwrite(f"test_thickness_{multiplier}x.png", mask)

if __name__ == "__main__":
    test_thick_stroke_rendering()
    test_stroke_thickness_variations()
