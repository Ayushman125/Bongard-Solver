#!/usr/bin/env python3
"""Test the coordinate transformation fix."""

import numpy as np
import cv2
import logging
from src.bongard_augmentor.hybrid import ActionMaskGenerator

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_coordinate_fix():
    """Test the coordinate transformation fix with the problematic action commands."""
    
    print("=== TESTING COORDINATE TRANSFORMATION FIX ===")
    
    # Use the exact action commands from your log that were causing white masks
    action_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121', 
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121',
        'line_normal_0.608-0.121'
    ]
    
    # Initialize mask generator
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print(f"Testing with commands: {action_commands}")
    
    # Test the full pipeline
    try:
        mask = generator.generate_mask_from_actions(action_commands, "test_problem_0000")
        
        mask_sum = np.sum(mask)
        nonzero_pixels = np.count_nonzero(mask)
        total_pixels = mask.size
        
        print(f"✅ Mask generated successfully!")
        print(f"   Sum: {mask_sum}")
        print(f"   Nonzero pixels: {nonzero_pixels}/{total_pixels} ({100*nonzero_pixels/total_pixels:.1f}%)")
        print(f"   Average pixel value: {mask_sum/total_pixels:.1f}")
        
        # Check if it's still all white
        if mask_sum > total_pixels * 200:
            print("❌ Mask is still mostly white - coordinate issue persists")
        elif nonzero_pixels == 0:
            print("❌ Mask is completely black - no rendering occurred")
        else:
            print("✅ Mask appears to have proper content!")
            
        # Save the result
        cv2.imwrite("debug_fixed_mask.png", mask)
        print("   Saved as debug_fixed_mask.png")
        
        # Create a visualization with the actual coordinates
        print("\n--- COORDINATE ANALYSIS ---")
        parsed_data = generator.action_parser.parse_action_commands(action_commands, "test_problem_0000")
        print(f"Parsed vertices: {parsed_data.vertices}")
        
        if parsed_data.vertices:
            verts = np.array(parsed_data.vertices)
            print(f"Coordinate ranges: X[{np.min(verts[:,0]):.1f}, {np.max(verts[:,0]):.1f}], Y[{np.min(verts[:,1]):.1f}, {np.max(verts[:,1]):.1f}]")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_simple_case():
    """Test with simple coordinates to verify basic functionality."""
    
    print("\n=== TESTING SIMPLE CASE ===")
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    # Simple line vertices that should definitely work
    simple_vertices = [(100.0, 100.0), (400.0, 400.0)]
    
    mask = generator._render_vertices_to_mask(simple_vertices)
    
    mask_sum = np.sum(mask)
    nonzero_pixels = np.count_nonzero(mask)
    
    print(f"Simple test: sum={mask_sum}, nonzero={nonzero_pixels}")
    
    if nonzero_pixels > 0:
        print("✅ Simple case works correctly")
        cv2.imwrite("debug_simple_fixed.png", mask)
    else:
        print("❌ Even simple case fails")

if __name__ == "__main__":
    test_coordinate_fix()
    test_simple_case()
