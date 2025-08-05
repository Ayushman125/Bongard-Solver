#!/usr/bin/env python3
"""
Test script to verify that different line and arc shape types are handled correctly.
This tests the distinction between arc_normal vs line_normal, arc_triangle vs line_triangle, etc.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import cv2
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser

def test_shape_type_differences():
    """Test that different shape types create visually distinct patterns."""
    print("🧪 TESTING SHAPE TYPE DIFFERENCES")
    print("=" * 60)
    
    parser = ComprehensiveNVLabsParser(canvas_size=512)
    
    # Test different line types
    line_types = ["normal", "triangle", "square", "zigzag", "circle"]
    arc_types = ["normal", "triangle", "square", "zigzag", "circle"]
    
    print("\n📏 TESTING LINE TYPES:")
    print("-" * 30)
    
    for line_type in line_types:
        test_command = f"line_{line_type}_1.000-0.500"
        print(f"\nTesting: {test_command}")
        
        try:
            image = parser.process_action_commands_to_image([test_command], f"line_{line_type}_test")
            if image is not None:
                nonzero_pixels = np.count_nonzero(image == 0)  # Black pixels
                print(f"  ✅ Success: {image.shape} image, {nonzero_pixels} drawn pixels")
                
                # Save test image
                cv2.imwrite(f"test_line_{line_type}.png", image)
                print(f"  💾 Saved: test_line_{line_type}.png")
            else:
                print(f"  ❌ Failed to generate image")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n🌀 TESTING ARC TYPES:")
    print("-" * 30)
    
    for arc_type in arc_types:
        test_command = f"arc_{arc_type}_0.500_30.0-0.750"
        print(f"\nTesting: {test_command}")
        
        try:
            image = parser.process_action_commands_to_image([test_command], f"arc_{arc_type}_test")
            if image is not None:
                nonzero_pixels = np.count_nonzero(image == 0)  # Black pixels
                print(f"  ✅ Success: {image.shape} image, {nonzero_pixels} drawn pixels")
                
                # Save test image
                cv2.imwrite(f"test_arc_{arc_type}.png", image)
                print(f"  💾 Saved: test_arc_{arc_type}.png")
            else:
                print(f"  ❌ Failed to generate image")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_real_data_format():
    """Test with real data formats to ensure compatibility."""
    print("\n🎯 TESTING REAL DATA FORMATS:")
    print("-" * 40)
    
    parser = ComprehensiveNVLabsParser(canvas_size=512)
    
    # Real format examples from Bongard-LOGO dataset
    real_commands = [
        # Basic formats
        "line_1.000-0.500",           # 2-part line
        "line_normal_0.600-0.750",    # 3-part line with explicit type
        "arc_0.5_30.0-0.750",         # 3-part arc (radius_angle)
        "arc_circle_0.5_45.0-0.500",  # 4-part arc with explicit type
        
        # More complex patterns
        "line_triangle_1.000-0.500",
        "line_zigzag_0.424-0.875",
        "arc_triangle_0.600_60.0-0.250",
        "arc_square_0.400_90.0-0.500",
    ]
    
    for i, command in enumerate(real_commands):
        print(f"\nTesting real format {i+1}: {command}")
        
        try:
            image = parser.process_action_commands_to_image([command], f"real_test_{i+1}")
            if image is not None:
                nonzero_pixels = np.count_nonzero(image == 0)
                print(f"  ✅ Success: {nonzero_pixels} drawn pixels")
                
                # Save test image
                cv2.imwrite(f"test_real_format_{i+1}.png", image)
                print(f"  💾 Saved: test_real_format_{i+1}.png")
            else:
                print(f"  ❌ Failed to generate image")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_comparison_between_types():
    """Create side-by-side comparison of different shape types."""
    print("\n🔍 TESTING TYPE COMPARISONS:")
    print("-" * 40)
    
    parser = ComprehensiveNVLabsParser(canvas_size=256)  # Smaller for comparison
    
    # Compare line vs arc with same shape type
    comparisons = [
        ("line_normal_1.000-0.500", "arc_normal_0.500_45.0-0.500"),
        ("line_triangle_1.000-0.500", "arc_triangle_0.500_45.0-0.500"),
        ("line_circle_1.000-0.500", "arc_circle_0.500_45.0-0.500"),
        ("line_zigzag_1.000-0.500", "arc_zigzag_0.500_45.0-0.500"),
    ]
    
    for i, (line_cmd, arc_cmd) in enumerate(comparisons):
        print(f"\nComparison {i+1}:")
        print(f"  Line: {line_cmd}")
        print(f"  Arc:  {arc_cmd}")
        
        try:
            # Generate both images
            line_image = parser.process_action_commands_to_image([line_cmd], f"comp_line_{i+1}")
            arc_image = parser.process_action_commands_to_image([arc_cmd], f"comp_arc_{i+1}")
            
            if line_image is not None and arc_image is not None:
                # Create side-by-side comparison
                comparison = np.hstack([line_image, arc_image])
                cv2.imwrite(f"comparison_{i+1}_line_vs_arc.png", comparison)
                
                line_pixels = np.count_nonzero(line_image == 0)
                arc_pixels = np.count_nonzero(arc_image == 0)
                
                print(f"  ✅ Success: Line={line_pixels} pixels, Arc={arc_pixels} pixels")
                print(f"  💾 Saved: comparison_{i+1}_line_vs_arc.png")
            else:
                print(f"  ❌ Failed to generate one or both images")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    print("🚀 SHAPE TYPE DIFFERENTIATION TEST")
    print("=" * 70)
    print("Testing that arc and line shape types are distinct...")
    
    test_shape_type_differences()
    test_real_data_format()
    test_comparison_between_types()
    
    print(f"\n🎉 TESTING COMPLETE!")
    print("=" * 70)
    print("Check the generated PNG files to verify visual differences between:")
    print("• line_normal vs arc_normal")
    print("• line_triangle vs arc_triangle") 
    print("• line_circle vs arc_circle")
    print("• line_zigzag vs arc_zigzag")
    print("• line_square vs arc_square")

if __name__ == "__main__":
    main()
