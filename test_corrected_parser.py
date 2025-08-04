#!/usr/bin/env python3
"""
Test script to verify the corrected Bongard-LOGO parser and augmentation fixes.

This script tests:
1. Basic line parsing with proper coordinate system
2. Arc parsing with corrected parameter interpretation
3. High-quality rendering with anti-aliasing
4. Augmentation pipeline with corrected coordinate transformation
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

def test_augmentation_pipeline():
    """Test the corrected augmentation pipeline."""
    print("\n=== TESTING AUGMENTATION PIPELINE ===")
    
    mask_generator = ActionMaskGenerator()
    
    # Test simple action commands
    test_commands = [
        "line_normal_0.5-0.0",
        "arc_circle_0.4_0.5-0.0"
    ]
    
    for i, cmd in enumerate(test_commands):
        print(f"\nTesting augmentation with command: {cmd}")
        
        # Generate mask using corrected pipeline
        mask = mask_generator.generate_mask_from_actions([cmd])
        
        if mask is not None and np.any(mask > 0):
            print(f"✓ Generated mask with {np.count_nonzero(mask)}/{mask.size} non-zero pixels")
            
            # Save for inspection
            save_path = f"test_augmentation_{i}_corrected.png"
            cv2.imwrite(save_path, mask)
            print(f"✓ Saved augmentation mask: {save_path}")
        else:
            print(f"✗ Failed to generate mask for command: {cmd}")

def test_coordinate_consistency():
    """Test that parser and augmentor use consistent coordinates."""
    print("\n=== TESTING COORDINATE CONSISTENCY ===")
    
    parser = UnifiedActionParser()
    mask_generator = ActionMaskGenerator()
    
    test_cmd = "line_normal_0.5-0.0"
    
    # Parse with parser
    image_program = parser._parse_single_image([test_cmd], "consistency_test", True, "test")
    parser_image = parser.visualize_image_program(image_program) if image_program else None
    
    # Generate with augmentor
    augmentor_mask = mask_generator.generate_mask_from_actions([test_cmd])
    
    if parser_image is not None and augmentor_mask is not None:
        # Calculate similarity
        parser_binary = (parser_image > 127).astype(np.uint8) * 255
        augmentor_binary = (augmentor_mask > 127).astype(np.uint8) * 255
        
        # Calculate overlap
        overlap = np.logical_and(parser_binary > 0, augmentor_binary > 0)
        union = np.logical_or(parser_binary > 0, augmentor_binary > 0)
        
        iou = np.sum(overlap) / np.sum(union) if np.sum(union) > 0 else 0
        
        print(f"Parser image: {np.count_nonzero(parser_binary)} pixels")
        print(f"Augmentor mask: {np.count_nonzero(augmentor_binary)} pixels")
        print(f"IoU (Intersection over Union): {iou:.3f}")
        
        if iou > 0.8:
            print("✓ Good coordinate consistency between parser and augmentor")
        else:
            print("⚠ Coordinate consistency could be improved")
            
        # Save comparison
        comparison = np.zeros((64, 64, 3), dtype=np.uint8)
        comparison[:, :, 0] = parser_binary  # Red channel
        comparison[:, :, 1] = augmentor_binary  # Green channel
        # Overlap appears as yellow
        
        cv2.imwrite("test_coordinate_consistency.png", comparison)
        print("✓ Saved coordinate consistency comparison: test_coordinate_consistency.png")
        print("  (Red=Parser, Green=Augmentor, Yellow=Overlap)")
    else:
        print("✗ Failed to generate images for consistency test")

def main():
    """Run all tests to verify the corrected implementation."""
    print("=" * 60)
    print("BONGARD-LOGO PARSER CORRECTION VERIFICATION")
    print("=" * 60)
    
    try:
        test_basic_line_parsing()
        test_arc_parsing()
        test_complex_shape()
        test_augmentation_pipeline()
        test_coordinate_consistency()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("Check the generated PNG files to verify visual quality:")
        print("- test_line_*_corrected.png")
        print("- test_arc_*_corrected.png") 
        print("- test_complex_shape_corrected.png")
        print("- test_augmentation_*_corrected.png")
        print("- test_coordinate_consistency.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
