#!/usr/bin/env python3
"""Debug script to investigate empty mask generation issue."""

from src.bongard_augmentor.hybrid import ActionMaskGenerator
import numpy as np

def debug_empty_mask_issue():
    """Debug why masks are empty despite successful action program processing."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print("=== DEBUGGING EMPTY MASK ISSUE ===")
    
    # Test simple action commands
    action_commands = ['move_to 0 0', 'line_to 50 0', 'line_to 50 50', 'line_to 0 50', 'line_to 0 0']
    print(f"Action commands: {action_commands}")
    
    # Test 1: Try NVLabs parser with problem_id
    print("\n1. NVLabs parser test with problem_id:")
    try:
        shape = generator.action_parser.parse_action_commands(action_commands, "test_problem_0000")
        print(f"   Parser result: {shape}")
        if shape:
            print(f"   Shape type: {type(shape)}")
            if hasattr(shape, 'vertices'):
                print(f"   Vertices count: {len(shape.vertices)}")
                print(f"   First vertices: {shape.vertices[:3] if shape.vertices else 'None'}")
        else:
            print("   Parser returned None")
    except Exception as e:
        print(f"   Parser error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Try different command format (no spaces in move_to/line_to)
    print("\n2. Test different command format:")
    alt_commands = ['move_to', '0', '0', 'line_to', '50', '0', 'line_to', '50', '50', 'line_to', '0', '50', 'line_to', '0', '0']
    try:
        shape2 = generator.action_parser.parse_action_commands(alt_commands, "test_problem_0000")
        print(f"   Parser result: {shape2}")
        if shape2 and hasattr(shape2, 'vertices'):
            print(f"   Vertices count: {len(shape2.vertices)}")
    except Exception as e:
        print(f"   Alt format error: {e}")
    
    # Test 3: Test fallback rendering
    print("\n3. Fallback rendering test:")
    try:
        fallback_mask = generator._render_simple_commands_to_mask(action_commands)
        print(f"   Fallback mask pixels: {np.sum(fallback_mask > 0)}")
    except Exception as e:
        print(f"   Fallback error: {e}")
    
    # Test 4: Full render_single_object_to_mask with problem_id
    print("\n4. Full rendering pipeline with problem_id:")
    try:
        full_mask = generator._render_single_object_to_mask(action_commands, "test_problem_0000")
        print(f"   Full mask pixels: {np.sum(full_mask > 0)}")
        print(f"   Mask shape: {full_mask.shape}")
        print(f"   Mask dtype: {full_mask.dtype}")
        print(f"   Mask range: {full_mask.min()} to {full_mask.max()}")
    except Exception as e:
        print(f"   Full rendering error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_empty_mask_issue()
