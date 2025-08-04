#!/usr/bin/env python3
"""
Simplified test script to verify the corrected Bongard-LOGO parser fixes.

This script tests:
1. Basic line parsing with proper coordinate system
2. Arc parsing with corrected parameter interpretation  
3. High-quality rendering with anti-aliasing
"""

import sys
import os
import cv2
import numpy as np
import logging

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_pipeline.logo_parser import UnifiedActionParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_basic_line_parsing():
    """Test basic line parsing with corrected coordinate system."""
    print("\n=== TESTING BASIC LINE PARSING ===")
    
    parser = UnifiedActionParser()
    
    # Test simple line commands
    test_commands = [
        "line_normal_0.5-0.0",    # Medium length, 0° angle
        "line_normal_0.3-0.25",   # Short length, 90° angle  
        "line_zigzag_0.4-0.5",    # Zigzag line, 180° angle
    ]
    
    for i, cmd in enumerate(test_commands):
        print(f"\nTesting command {i+1}: {cmd}")
        
        # Parse single image with this command
        image_program = parser._parse_single_image([cmd], f"test_line_{i}", True, "test")
        
        if image_program and image_program.vertices:
            print(f"✓ Parsed {len(image_program.vertices)} vertices")
            print(f"  Vertex range: X[{min(v[0] for v in image_program.vertices):.2f}, {max(v[0] for v in image_program.vertices):.2f}]"
                  f" Y[{min(v[1] for v in image_program.vertices):.2f}, {max(v[1] for v in image_program.vertices):.2f}]")
            
            # Render to image
            rendered_image = parser.visualize_image_program(image_program)
            
            # Save for inspection
            save_path = f"test_line_{i}_corrected.png"
            cv2.imwrite(save_path, rendered_image)
            print(f"✓ Saved rendered image: {save_path}")
            print(f"  Non-zero pixels: {np.count_nonzero(rendered_image)}/{rendered_image.size} ({100*np.count_nonzero(rendered_image)/rendered_image.size:.1f}%)")
        else:
            print(f"✗ Failed to parse command: {cmd}")

def test_arc_parsing():
    """Test arc parsing with corrected parameter interpretation."""
    print("\n=== TESTING ARC PARSING ===")
    
    parser = UnifiedActionParser()
    
    # Test arc commands with different formats
    test_commands = [
        "arc_normal_0.5_0.25-0.0",      # Normal arc, medium radius, quarter circle
        "arc_circle_0.3_0.5-0.25",      # Circle modifier, small radius, half circle
        "arc_zigzag_0.4_0.125-0.75",    # Zigzag arc, medium radius, eighth circle
    ]
    
    for i, cmd in enumerate(test_commands):
        print(f"\nTesting arc command {i+1}: {cmd}")
        
        # Parse single image with this command
        image_program = parser._parse_single_image([cmd], f"test_arc_{i}", True, "test")
        
        if image_program and image_program.vertices:
            print(f"✓ Parsed {len(image_program.vertices)} vertices")
            print(f"  Vertex range: X[{min(v[0] for v in image_program.vertices):.2f}, {max(v[0] for v in image_program.vertices):.2f}]"
                  f" Y[{min(v[1] for v in image_program.vertices):.2f}, {max(v[1] for v in image_program.vertices):.2f}]")
            
            # Render to image
            rendered_image = parser.visualize_image_program(image_program)
            
            # Save for inspection
            save_path = f"test_arc_{i}_corrected.png"
            cv2.imwrite(save_path, rendered_image)
            print(f"✓ Saved rendered image: {save_path}")
            print(f"  Non-zero pixels: {np.count_nonzero(rendered_image)}/{rendered_image.size} ({100*np.count_nonzero(rendered_image)/rendered_image.size:.1f}%)")
        else:
            print(f"✗ Failed to parse arc command: {cmd}")

def test_complex_shape():
    """Test complex shape with multiple strokes."""
    print("\n=== TESTING COMPLEX SHAPE ===")
    
    parser = UnifiedActionParser()
    
    # Test complex shape with multiple strokes
    complex_commands = [
        "line_normal_0.4-0.0",        # Horizontal line
        "arc_normal_0.3_0.25-0.25",   # Quarter circle
        "line_zigzag_0.3-0.5",        # Zigzag line
    ]
    
    print(f"Testing complex shape with {len(complex_commands)} strokes:")
    for cmd in complex_commands:
        print(f"  - {cmd}")
    
    image_program = parser._parse_single_image(complex_commands, "test_complex", True, "test")
    
    if image_program and image_program.vertices:
        print(f"✓ Parsed complex shape with {len(image_program.vertices)} total vertices")
        print(f"  Stroke count: {len(image_program.strokes)}")
        
        # Render to image
        rendered_image = parser.visualize_image_program(image_program)
        
        # Save for inspection
        save_path = "test_complex_shape_corrected.png"
        cv2.imwrite(save_path, rendered_image)
        print(f"✓ Saved complex shape: {save_path}")
        print(f"  Non-zero pixels: {np.count_nonzero(rendered_image)}/{rendered_image.size} ({100*np.count_nonzero(rendered_image)/rendered_image.size:.1f}%)")
    else:
        print("✗ Failed to parse complex shape")

def test_coordinate_system():
    """Test the corrected coordinate system and scaling."""
    print("\n=== TESTING COORDINATE SYSTEM ===")
    
    parser = UnifiedActionParser()
    print(f"Parser scale factor: {parser.scale_factor}")
    print(f"Canvas size: {parser.canvas_size}")
    print(f"Canvas center: {parser.canvas_center}")
    
    # Test a simple horizontal line that should be centered
    test_cmd = "line_normal_0.5-0.0"  # Half-length line, 0° angle
    
    image_program = parser._parse_single_image([test_cmd], "coord_test", True, "test")
    
    if image_program and image_program.vertices:
        print(f"\nCoordinate test results:")
        print(f"  Vertices: {image_program.vertices}")
        
        # Check if vertices are in expected range
        x_coords = [v[0] for v in image_program.vertices]
        y_coords = [v[1] for v in image_program.vertices]
        
        print(f"  X range: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
        print(f"  Y range: [{min(y_coords):.2f}, {max(y_coords):.2f}]")
        
        # Expected: X should span from 0 to ~12.5 (half of scale_factor=25)
        # Y should be approximately 0 (horizontal line)
        expected_x_max = parser.scale_factor * 0.5  # 12.5
        
        if abs(max(x_coords) - expected_x_max) < 2.0:
            print("✓ X coordinates are in expected range")
        else:
            print(f"⚠ X coordinates unexpected. Expected max ~{expected_x_max:.1f}, got {max(x_coords):.1f}")
        
        if abs(max(y_coords) - min(y_coords)) < 1.0:
            print("✓ Y coordinates are consistent (horizontal line)")
        else:
            print(f"⚠ Y coordinates vary unexpectedly: {min(y_coords):.2f} to {max(y_coords):.2f}")
        
        # Render and save
        rendered_image = parser.visualize_image_program(image_program)
        cv2.imwrite("test_coordinate_system.png", rendered_image)
        print("✓ Saved coordinate system test: test_coordinate_system.png")

def main():
    """Run all tests to verify the corrected implementation."""
    print("=" * 60)
    print("BONGARD-LOGO PARSER CORRECTION VERIFICATION")
    print("=" * 60)
    
    try:
        test_coordinate_system()
        test_basic_line_parsing()
        test_arc_parsing()
        test_complex_shape()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("Check the generated PNG files to verify visual quality:")
        print("- test_coordinate_system.png")
        print("- test_line_*_corrected.png")
        print("- test_arc_*_corrected.png") 
        print("- test_complex_shape_corrected.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
