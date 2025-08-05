#!/usr/bin/env python3
"""Test script to verify all 10 semantic command types are handled correctly."""

from src.bongard_augmentor.hybrid import ActionMaskGenerator
import numpy as np
import cv2

def test_all_semantic_commands():
    """Test all 10 distinct semantic command types."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    # Define all 10 command types with sample parameters
    test_commands = [
        # Line commands (5 types)
        "line_normal_1.000-0.500",
        "line_triangle_1.000-0.500", 
        "line_square_1.000-0.500",
        "line_circle_1.000-0.500",
        "line_zigzag_1.000-0.500",
        
        # Arc commands (5 types)
        "arc_normal_0.500_0.625-0.500",
        "arc_triangle_0.500_0.625-0.500",
        "arc_square_0.500_0.625-0.500", 
        "arc_circle_0.500_0.625-0.500",
        "arc_zigzag_0.500_0.625-0.500"
    ]
    
    print("=== TESTING ALL 10 SEMANTIC COMMAND TYPES ===")
    
    for i, cmd in enumerate(test_commands):
        print(f"\n{i+1}. Testing: {cmd}")
        
        try:
            # Test single command
            mask = generator._render_semantic_commands_to_mask([cmd])
            pixels = np.sum(mask > 0)
            print(f"   ✅ Rendered successfully: {pixels} pixels")
            
            # Test command parsing
            parsed = generator._parse_semantic_command(cmd)
            if parsed:
                stroke_type, shape_type, params = parsed
                print(f"   📋 Parsed: stroke='{stroke_type}', shape='{shape_type}'")
                scale = params.get('scale', 'N/A')
                final_param = params.get('final_param', 'N/A')
                
                if stroke_type == 'line':
                    print(f"       Line Parameters (2): scale={scale}, final_param={final_param}")
                elif stroke_type == 'arc':
                    arc_param = params.get('arc_param', 'N/A')
                    print(f"       Arc Parameters (3): scale={scale}, arc_param={arc_param}, final_param={final_param}")
            else:
                print(f"   ❌ Failed to parse command")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Test combination commands (like real Bongard programs)
    print(f"\n=== TESTING COMBINATION COMMANDS ===")
    combo_commands = [
        "line_triangle_1.000-0.500",
        "line_square_1.000-0.833", 
        "arc_circle_0.500_0.625-0.500"
    ]
    
    try:
        combo_mask = generator._render_semantic_commands_to_mask(combo_commands)
        combo_pixels = np.sum(combo_mask > 0)
        print(f"✅ Combination rendering: {combo_pixels} pixels from {len(combo_commands)} commands")
    except Exception as e:
        print(f"❌ Combination error: {e}")

if __name__ == "__main__":
    test_all_semantic_commands()
