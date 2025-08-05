#!/usr/bin/env python3
"""
Comprehensive debugging script to analyze action command parsing issues.
Specifically investigating:
1. Nested list structures in action commands
2. Coordinate range and normalization issues
3. Differences between real, parsed, and generated images
"""

import sys
import json
import logging
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analyze_action_commands_structure(commands):
    """Analyze the structure and format of action commands."""
    print(f"\n=== ACTION COMMANDS ANALYSIS ===")
    print(f"Type: {type(commands)}")
    print(f"Length: {len(commands) if hasattr(commands, '__len__') else 'N/A'}")
    
    if isinstance(commands, list):
        print(f"List contents preview: {commands[:3]}...")
        
        for i, cmd in enumerate(commands[:5]):  # Show first 5 commands
            print(f"  Command {i}: type={type(cmd)}, value={cmd}")
            
            # Check for nested lists
            if isinstance(cmd, list):
                print(f"    ⚠️  NESTED LIST DETECTED! Length: {len(cmd)}")
                for j, sub_cmd in enumerate(cmd[:3]):
                    print(f"      Sub-command {j}: type={type(sub_cmd)}, value={sub_cmd}")
                    
        # Count nested vs flat commands
        nested_count = sum(1 for cmd in commands if isinstance(cmd, list))
        flat_count = len(commands) - nested_count
        print(f"  Structure: {flat_count} flat commands, {nested_count} nested lists")
        
        if nested_count > 0:
            print(f"  ⚠️  WARNING: Nested lists detected! This may cause parsing issues.")
            
    return commands

def test_nvlabs_parser():
    """Test the NVLabs parser with various command formats."""
    print(f"\n=== NVLABS PARSER TEST ===")
    
    try:
        from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
        parser = ComprehensiveNVLabsParser()
        
        # Test cases with different formats
        test_cases = [
            {
                "name": "Simple flat commands",
                "commands": ["line_triangle_1.000-0.500", "line_normal_0.600-0.750"]
            },
            {
                "name": "Nested list commands (problematic)",
                "commands": [["line_triangle_1.000-0.500"], ["line_normal_0.600-0.750"]]
            },
            {
                "name": "Mixed format",
                "commands": ["line_triangle_1.000-0.500", ["line_normal_0.600-0.750"]]
            }
        ]
        
        for test_case in test_cases:
            print(f"\n--- Testing: {test_case['name']} ---")
            commands = test_case['commands']
            analyze_action_commands_structure(commands)
            
            try:
                # Test with parser
                parsed_data = parser.parse_action_commands(commands, "test_problem")
                if parsed_data and hasattr(parsed_data, 'vertices'):
                    vertices = parsed_data.vertices
                    print(f"  ✅ Parsing successful! Vertices: {len(vertices)} points")
                    if vertices:
                        verts_array = np.array(vertices)
                        print(f"     Vertex range: min=({np.min(verts_array[:,0]):.2f}, {np.min(verts_array[:,1]):.2f}), max=({np.max(verts_array[:,0]):.2f}, {np.max(verts_array[:,1]):.2f})")
                else:
                    print(f"  ❌ Parsing failed: No vertices returned")
                    
            except Exception as e:
                print(f"  ❌ Parsing error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize NVLabs parser: {e}")

def test_hybrid_pipeline():
    """Test the hybrid pipeline with action commands."""
    print(f"\n=== HYBRID PIPELINE TEST ===")
    
    try:
        from src.bongard_augmentor.hybrid import HybridAugmentationPipeline
        
        # Initialize pipeline
        config = {
            'canvas_size': 512,
            'vertex_sampling_density': 0.1,
            'coordinate_noise_std': 0.01,
            'enable_post_processing': False
        }
        
        pipeline = HybridAugmentationPipeline(config)
        
        # Test with different command formats
        test_commands = [
            ["line_triangle_1.000-0.500", "line_normal_0.600-0.750"],
            [["line_triangle_1.000-0.500"], ["line_normal_0.600-0.750"]],  # Nested - problematic
        ]
        
        for i, commands in enumerate(test_commands):
            print(f"\n--- Test Case {i+1} ---")
            analyze_action_commands_structure(commands)
            
            try:
                mask = pipeline.process_action_commands(commands, f"test_{i}")
                if mask is not None:
                    print(f"  ✅ Pipeline successful! Mask shape: {mask.shape}, nonzero pixels: {np.count_nonzero(mask)}")
                    
                    # Save for visual inspection
                    cv2.imwrite(f"debug_pipeline_mask_{i}.png", mask)
                    print(f"  💾 Saved debug_pipeline_mask_{i}.png")
                else:
                    print(f"  ❌ Pipeline failed: No mask returned")
                    
            except Exception as e:
                print(f"  ❌ Pipeline error: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Failed to initialize hybrid pipeline: {e}")
        import traceback
        traceback.print_exc()

def analyze_coordinate_normalization():
    """Analyze coordinate normalization issues."""
    print(f"\n=== COORDINATE NORMALIZATION ANALYSIS ===")
    
    # Simulate extreme coordinates like in logs
    extreme_coords = [
        (-1011, -500),
        (908, 750),
        (-800, 600),
        (500, -900)
    ]
    
    print(f"Extreme coordinates: {extreme_coords}")
    coords_array = np.array(extreme_coords)
    
    min_x, min_y = np.min(coords_array, axis=0)
    max_x, max_y = np.max(coords_array, axis=0)
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    print(f"Original range: x=({min_x:.2f} to {max_x:.2f}, range={range_x:.2f}), y=({min_y:.2f} to {max_y:.2f}, range={range_y:.2f})")
    
    # Test normalization
    image_size = (512, 512)  # (height, width)
    margin = 10
    target_w = image_size[1] - 2 * margin  # 492
    target_h = image_size[0] - 2 * margin  # 492
    
    scale_x = target_w / range_x if range_x > 0 else 1.0
    scale_y = target_h / range_y if range_y > 0 else 1.0
    scale = min(scale_x, scale_y)
    
    print(f"Scaling: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}, final_scale={scale:.4f}")
    
    # Apply normalization
    coords_norm = coords_array.copy().astype(float)
    coords_norm[:,0] = (coords_array[:,0] - min_x) * scale + margin
    coords_norm[:,1] = (coords_array[:,1] - min_y) * scale + margin
    
    # Center
    canvas_center_x = image_size[1] / 2
    canvas_center_y = image_size[0] / 2
    shape_center_x = np.mean(coords_norm[:,0])
    shape_center_y = np.mean(coords_norm[:,1])
    offset_x = canvas_center_x - shape_center_x
    offset_y = canvas_center_y - shape_center_y
    
    coords_norm[:,0] += offset_x
    coords_norm[:,1] += offset_y
    
    # Clip
    coords_norm[:,0] = np.clip(coords_norm[:,0], margin, image_size[1] - margin)
    coords_norm[:,1] = np.clip(coords_norm[:,1], margin, image_size[0] - margin)
    
    print(f"Normalized coordinates: {coords_norm}")
    print(f"Final range: x=({np.min(coords_norm[:,0]):.2f} to {np.max(coords_norm[:,0]):.2f}), y=({np.min(coords_norm[:,1]):.2f} to {np.max(coords_norm[:,1]):.2f})")
    
    # Check if scale is too small (detail loss)
    if scale < 0.5:
        print(f"  ⚠️  WARNING: Very small scale factor ({scale:.4f}) may cause detail loss!")
    if scale < 0.1:
        print(f"  🚨 CRITICAL: Extremely small scale factor may cause severe distortion!")

def flatten_nested_commands(commands):
    """Fix nested command lists by flattening them."""
    print(f"\n=== COMMAND FLATTENING TEST ===")
    
    if not isinstance(commands, list):
        return commands
        
    flattened = []
    had_nested = False
    
    for cmd in commands:
        if isinstance(cmd, list):
            had_nested = True
            # Flatten nested lists
            for sub_cmd in cmd:
                flattened.append(sub_cmd)
        else:
            flattened.append(cmd)
            
    if had_nested:
        print(f"  Original: {commands}")
        print(f"  Flattened: {flattened}")
        print(f"  ✅ Fixed {sum(1 for cmd in commands if isinstance(cmd, list))} nested lists")
    else:
        print(f"  ✅ No nested lists detected")
        
    return flattened

def main():
    print("🔍 ACTION COMMAND PARSING DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Run all diagnostic tests
    test_nvlabs_parser()
    test_hybrid_pipeline()
    analyze_coordinate_normalization()
    
    # Test command flattening fix
    nested_commands = [["line_triangle_1.000-0.500"], ["line_normal_0.600-0.750"]]
    fixed_commands = flatten_nested_commands(nested_commands)
    
    print(f"\n=== FINAL RECOMMENDATIONS ===")
    print("1. 🔧 Fix nested list parsing by flattening command structures")
    print("2. ⚙️  Improve coordinate normalization to prevent extreme scaling")
    print("3. 🖼️  Add better bounds checking for vertex coordinates")
    print("4. 📊 Implement detailed logging for coordinate transformation pipeline")

if __name__ == "__main__":
    main()
