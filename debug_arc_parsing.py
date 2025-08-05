#!/usr/bin/env python3
"""
Debug parameter parsing for arc commands.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.bongard_augmentor.hybrid import ActionMaskGenerator

def debug_arc_parsing():
    """Debug arc parameter parsing step by step."""
    
    mask_gen = ActionMaskGenerator(image_size=(512, 512))
    
    # Test arc command that should have 3 parameters
    test_cmd = "arc_normal_0.500_0.542-0.750"
    print(f"Testing command: {test_cmd}")
    
    # Split by underscore first
    parts = test_cmd.strip().split('_')
    print(f"Split by underscore: {parts}")
    
    if len(parts) >= 3:
        stroke_type = parts[0]  # 'arc'
        shape_type = parts[1]   # 'normal'
        param_str = parts[2]    # '0.500_0.542-0.750'
        print(f"stroke_type: {stroke_type}")
        print(f"shape_type: {shape_type}")
        print(f"param_str: {param_str}")
        
        # Now test parameter parsing
        print(f"\nParsing param_str: '{param_str}'")
        
        # Check if dash is present
        if '-' in param_str:
            main_part, final_param = param_str.rsplit('-', 1)
            print(f"main_part: '{main_part}', final_param: '{final_param}'")
            
            # Parse main part
            main_values = main_part.split('_')
            print(f"main_values: {main_values}")
            
            if len(main_values) >= 1:
                scale = float(main_values[0])
                print(f"scale: {scale}")
            if len(main_values) >= 2:
                arc_param = float(main_values[1])
                print(f"arc_param: {arc_param}")
            
            final_param_val = float(final_param)
            print(f"final_param: {final_param_val}")
    
    # Now test the actual parsing function
    print(f"\n=== ACTUAL PARSING FUNCTION ===")
    result = mask_gen._parse_semantic_command(test_cmd)
    if result:
        stroke_type, shape_type, params = result
        print(f"Result: {stroke_type}_{shape_type}: {params}")
    else:
        print("PARSING FAILED")

if __name__ == "__main__":
    debug_arc_parsing()
