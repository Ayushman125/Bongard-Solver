#!/usr/bin/env python3
"""Test real Bongard action programs to verify semantic rendering works with actual data."""

from src.bongard_augmentor.hybrid import ActionMaskGenerator
import numpy as np
import json

def test_real_bongard_programs():
    """Test semantic rendering with real Bongard action programs."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    # Load some real action programs
    try:
        with open('bd_action_programs.json', 'r') as f:
            bd_data = json.load(f)
    except FileNotFoundError:
        print("❌ bd_action_programs.json not found")
        return
    
    print("=== TESTING REAL BONGARD ACTION PROGRAMS ===")
    
    # Test a few real programs
    test_problems = list(bd_data.keys())[:3]
    
    for problem_id in test_problems:
        print(f"\n🔍 Testing problem: {problem_id}")
        
        problem_data = bd_data[problem_id]
        if not isinstance(problem_data, list) or len(problem_data) < 2:
            print("   ❌ Invalid problem data format")
            continue
            
        positive_examples, negative_examples = problem_data
        
        # Test first positive example
        if positive_examples and len(positive_examples) > 0:
            first_example = positive_examples[0]
            if isinstance(first_example, list) and len(first_example) > 0:
                first_image = first_example[0]
                if isinstance(first_image, list):
                    print(f"   📝 Commands: {first_image[:3]}...")  # Show first 3 commands
                    
                    try:
                        # Test rendering with real commands
                        mask = generator._render_semantic_commands_to_mask(first_image)
                        pixels = np.sum(mask > 0)
                        print(f"   ✅ Rendered successfully: {pixels} pixels")
                        
                        # Test full pipeline
                        full_mask = generator.generate_mask_from_actions(first_image, problem_id)
                        full_pixels = np.sum(full_mask > 0)
                        print(f"   🔧 Full pipeline: {full_pixels} pixels")
                        
                        # Analyze command types in this example
                        line_commands = [cmd for cmd in first_image if str(cmd).startswith('line_')]
                        arc_commands = [cmd for cmd in first_image if str(cmd).startswith('arc_')]
                        print(f"   📊 Analysis: {len(line_commands)} line commands, {len(arc_commands)} arc commands")
                        
                        if line_commands:
                            print(f"      Line types: {set([str(cmd).split('_')[1] for cmd in line_commands if '_' in str(cmd)])}")
                        if arc_commands:
                            print(f"      Arc types: {set([str(cmd).split('_')[1] for cmd in arc_commands if '_' in str(cmd)])}")
                            
                    except Exception as e:
                        print(f"   ❌ Rendering failed: {e}")
                        import traceback
                        traceback.print_exc()

def test_parameter_variations():
    """Test various parameter combinations found in real data."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print("\n=== TESTING PARAMETER VARIATIONS ===")
    
    # Real parameter patterns found in the data
    test_cases = [
        # Line commands with various scales and final_params
        "line_triangle_1.000-0.500",
        "line_square_1.000-0.833", 
        "line_circle_1.000-0.917",
        "line_normal_0.700-0.500",
        "line_zigzag_0.707-0.875",
        
        # Arc commands with various parameters
        "arc_circle_0.500_0.625-0.500",
        "arc_square_0.500_0.625-0.500", 
        "arc_triangle_0.500_0.625-0.750",
        "arc_zigzag_0.500_0.625-0.500",
        "arc_normal_0.866_0.750-0.500"
    ]
    
    for cmd in test_cases:
        print(f"\n📝 Testing: {cmd}")
        try:
            # Parse parameters
            parsed = generator._parse_semantic_command(cmd)
            if parsed:
                stroke_type, shape_type, params = parsed
                print(f"   ✅ Parsed: {stroke_type}_{shape_type}")
                print(f"   📊 Parameters: {params}")
                
                # Render
                mask = generator._render_semantic_commands_to_mask([cmd])
                pixels = np.sum(mask > 0)
                print(f"   🎨 Rendered: {pixels} pixels")
            else:
                print(f"   ❌ Failed to parse")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_real_bongard_programs()
    test_parameter_variations()
