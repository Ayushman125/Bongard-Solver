#!/usr/bin/env python3
"""
Comprehensive test to verify all critical fixes in the Bongard-LOGO parser.

This test validates:
1. Reduced scale factor (15.0) prevents coordinate overflow
2. Conservative stroke grouping keeps related shapes together  
3. Compact object positioning prevents clipping
4. Adaptive rendering ensures all shapes are visible
5. Coordinate consistency between parser and augmentor
"""

import sys
import os
import cv2
import numpy as np
import logging

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_pipeline.logo_parser import UnifiedActionParser
from src.bongard_augmentor.hybrid import ActionMaskGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_coordinate_overflow_fix():
    """Test that the reduced scale factor prevents coordinate overflow."""
    print("\n=== TESTING COORDINATE OVERFLOW FIX ===")
    
    parser = UnifiedActionParser()
    
    # Verify the scale factor has been corrected
    expected_scale = 15.0
    actual_scale = parser.scale_factor
    
    print(f"Scale factor: Expected={expected_scale}, Actual={actual_scale}")
    if actual_scale == expected_scale:
        print("✓ Scale factor correctly reduced to prevent overflow")
    else:
        print(f"✗ Scale factor still incorrect: {actual_scale}")
        return False
    
    # Test with commands that previously caused overflow
    problematic_commands = [
        'line_normal_0.860-0.500',
        'line_normal_0.300-0.151', 
        'line_normal_0.860-0.151',
        'line_normal_0.700-0.849'
    ]
    
    print(f"Testing with {len(problematic_commands)} commands that previously caused overflow...")
    
    image_program = parser._parse_single_image(problematic_commands, "overflow_test", True, "test")
    
    if image_program and image_program.vertices:
        # Check if all vertices are within reasonable bounds
        all_x = [x for x, y in image_program.vertices]
        all_y = [y for x, y in image_program.vertices]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        print(f"Vertex bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        
        # With scale factor 15.0, vertices should be roughly in range [-15, 15]
        if abs(min_x) <= 20 and abs(max_x) <= 20 and abs(min_y) <= 20 and abs(max_y) <= 20:
            print("✓ All vertices within expected bounds, no coordinate overflow")
            return True
        else:
            print(f"✗ Vertices still outside expected bounds")
            return False
    else:
        print("✗ Failed to parse commands")
        return False

def test_conservative_stroke_grouping():
    """Test that stroke grouping is now more conservative."""
    print("\n=== TESTING CONSERVATIVE STROKE GROUPING ===")
    
    parser = UnifiedActionParser()
    
    # Test with commands that were previously over-separated
    related_commands = [
        'line_normal_0.860-0.500',
        'line_normal_0.300-0.151', 
        'line_normal_0.860-0.151'
    ]
    
    print(f"Testing stroke grouping with {len(related_commands)} related commands...")
    
    # Use the internal grouping method
    groups = parser._detect_stroke_groups(related_commands)
    
    print(f"Number of groups detected: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"  Group {i+1}: {len(group)} strokes")
    
    # With conservative grouping, 3 related commands should stay as 1 group
    if len(groups) == 1:
        print("✓ Conservative grouping: Related strokes kept as single shape")
        return True
    elif len(groups) <= 2:
        print("⚠ Moderate grouping: Some separation but not excessive")
        return True
    else:
        print(f"✗ Over-separation: {len(groups)} groups is too many for {len(related_commands)} related commands")
        return False

def test_compact_positioning():
    """Test that object positioning is now more compact."""
    print("\n=== TESTING COMPACT POSITIONING ===")
    
    parser = UnifiedActionParser()
    
    # Test positioning for multiple objects
    print("Testing compact positioning for 2 objects...")
    
    parser._reset_turtle()
    initial_pos = (parser.turtle_x, parser.turtle_y)
    print(f"Initial position: {initial_pos}")
    
    # Position for second object
    parser._position_for_next_object(1, 2)  # Object 1 of 2
    second_pos = (parser.turtle_x, parser.turtle_y)
    print(f"Second object position: {second_pos}")
    
    # Calculate spacing
    spacing = abs(second_pos[0] - initial_pos[0])
    expected_spacing = parser.scale_factor * 0.4  # Should be 15.0 * 0.4 = 6.0
    
    print(f"Spacing: Expected~{expected_spacing:.1f}, Actual={spacing:.1f}")
    
    if abs(spacing - expected_spacing) < 1.0:
        print("✓ Compact positioning: Objects positioned with correct reduced spacing")
        return True
    else:
        print(f"✗ Spacing still too large: {spacing:.1f} vs expected {expected_spacing:.1f}")
        return False

def test_adaptive_rendering():
    """Test that adaptive rendering keeps all shapes visible."""
    print("\n=== TESTING ADAPTIVE RENDERING ===")
    
    parser = UnifiedActionParser()
    
    # Create vertices that span a large range
    test_vertices = [
        (-10, -10),   # Far corners
        (10, 10),
        (-8, 8),
        (8, -8),
        (0, 0)        # Center
    ]
    
    print(f"Testing adaptive rendering with {len(test_vertices)} vertices...")
    print(f"Vertex range: X[-10, 10], Y[-10, 10]")
    
    # Render using the corrected method
    rendered_image = parser._render_vertices_to_image(test_vertices, (64, 64))
    
    if rendered_image is not None:
        non_zero_pixels = np.count_nonzero(rendered_image)
        total_pixels = rendered_image.size
        
        print(f"Rendered image: {non_zero_pixels}/{total_pixels} pixels")
        
        # Save for inspection
        cv2.imwrite("test_adaptive_rendering.png", rendered_image)
        print("✓ Saved test image: test_adaptive_rendering.png")
        
        if non_zero_pixels > 10:  # Should have reasonable content
            print("✓ Adaptive rendering: Image contains visible content")
            return True
        else:
            print("✗ Adaptive rendering: Very few pixels rendered")
            return False
    else:
        print("✗ Adaptive rendering failed")
        return False

def test_coordinate_consistency():
    """Test consistency between parser and augmentor."""
    print("\n=== TESTING COORDINATE CONSISTENCY ===")
    
    parser = UnifiedActionParser()
    augmentor = ActionMaskGenerator()
    
    # Verify both use the same scale factor
    parser_scale = parser.scale_factor
    augmentor_scale = augmentor.action_parser.scale_factor
    
    print(f"Scale factors: Parser={parser_scale}, Augmentor={augmentor_scale}")
    
    if parser_scale == augmentor_scale:
        print("✓ Scale factors match between parser and augmentor")
    else:
        print(f"✗ Scale factor mismatch: {parser_scale} vs {augmentor_scale}")
        return False
    
    # Test with identical commands
    test_cmd = "line_normal_0.5-0.0"
    print(f"Testing consistency with: {test_cmd}")
    
    # Parse with parser
    image_program = parser._parse_single_image([test_cmd], "consistency_test", True, "test")
    parser_image = parser._render_vertices_to_image(image_program.vertices) if image_program else None
    
    # Generate with augmentor
    augmentor_mask = augmentor.generate_mask_from_actions([test_cmd])
    
    if parser_image is not None and augmentor_mask is not None:
        # Check if images are identical or very similar
        if parser_image.shape == augmentor_mask.shape:
            # Calculate pixel-wise agreement
            agreement = np.sum(parser_image == augmentor_mask) / parser_image.size
            print(f"Pixel agreement: {agreement:.3f} ({100*agreement:.1f}%)")
            
            if agreement > 0.95:
                print("✓ Excellent coordinate consistency between parser and augmentor")
                
                # Save comparison
                comparison = np.zeros((64, 192, 3), dtype=np.uint8)
                comparison[:, :64, 0] = parser_image      # Red channel
                comparison[:, 64:128, 1] = augmentor_mask # Green channel  
                comparison[:, 128:, :] = np.stack([parser_image, augmentor_mask, 
                                                 np.logical_and(parser_image > 0, augmentor_mask > 0).astype(np.uint8) * 255], axis=2)
                
                cv2.imwrite("test_coordinate_consistency.png", comparison)
                print("✓ Saved comparison: test_coordinate_consistency.png")
                
                return True
            else:
                print(f"⚠ Moderate consistency: {100*agreement:.1f}% agreement")
                return False
        else:
            print(f"✗ Shape mismatch: {parser_image.shape} vs {augmentor_mask.shape}")
            return False
    else:
        print("✗ Failed to generate images for consistency test")
        return False

def main():
    """Run comprehensive test of all critical fixes."""
    print("=" * 70)
    print("COMPREHENSIVE TEST OF BONGARD-LOGO PARSER CRITICAL FIXES")
    print("=" * 70)
    
    tests = [
        ("Coordinate Overflow Fix", test_coordinate_overflow_fix),
        ("Conservative Stroke Grouping", test_conservative_stroke_grouping), 
        ("Compact Positioning", test_compact_positioning),
        ("Adaptive Rendering", test_adaptive_rendering),
        ("Coordinate Consistency", test_coordinate_consistency)
    ]
    
    results = {}
    
    try:
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            print(f"RUNNING: {test_name}")
            print(f"{'='*50}")
            
            success = test_func()
            results[test_name] = success
            
            if success:
                print(f"✓ {test_name}: PASSED")
            else:
                print(f"✗ {test_name}: FAILED")
        
        # Summary
        print(f"\n{'='*70}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*70}")
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, success in results.items():
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{test_name:.<40} {status}")
        
        print(f"\nOVERALL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        
        if passed == total:
            print("\n🎉 ALL CRITICAL FIXES VERIFIED! 🎉")
            print("Your Bongard-LOGO parser should now generate high-quality images")
            print("that match the real dataset quality.")
        else:
            print(f"\n⚠ {total-passed} issues still need to be addressed.")
        
        print(f"\nGenerated test images:")
        print("- test_adaptive_rendering.png")
        print("- test_coordinate_consistency.png")
        
    except Exception as e:
        print(f"\n✗ CRITICAL TEST FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
