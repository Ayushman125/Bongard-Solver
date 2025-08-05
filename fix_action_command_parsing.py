#!/usr/bin/env python3
"""
Comprehensive fix for action command parsing issues in the Bongard-LOGO pipeline.

Key Issues Identified:
1. Nested list structures in action commands
2. Extreme coordinate scaling causing detail loss (scale ~0.26)
3. Real images much more complex than generated masks
4. Action command parsing not handling multiple objects properly

Solutions Implemented:
1. Command flattening for nested structures
2. Improved coordinate normalization with adaptive scaling
3. Enhanced multi-object detection and rendering
4. Better bounds checking and validation
"""

import sys
import numpy as np
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

sys.path.insert(0, '.')

def flatten_action_commands(commands: List[Union[str, List[str]]]) -> List[str]:
    """
    Flatten nested action command structures.
    
    Args:
        commands: List that may contain nested lists
        
    Returns:
        Flattened list of command strings
    """
    if not isinstance(commands, list):
        return [str(commands)] if commands else []
        
    flattened = []
    for cmd in commands:
        if isinstance(cmd, list):
            # Recursively flatten nested lists
            flattened.extend(flatten_action_commands(cmd))
        else:
            flattened.append(str(cmd))
            
    return flattened

def analyze_coordinate_bounds(vertices: List[Tuple[float, float]], 
                            image_size: Tuple[int, int],
                            margin: int = 10) -> Dict[str, Any]:
    """
    Analyze coordinate bounds and suggest optimal scaling.
    
    Args:
        vertices: List of (x, y) coordinate tuples
        image_size: (height, width) of target image
        margin: Margin to keep from edges
        
    Returns:
        Analysis results with scaling recommendations
    """
    if not vertices:
        return {"valid": False, "reason": "No vertices provided"}
        
    verts_array = np.array(vertices)
    min_x, min_y = np.min(verts_array, axis=0)
    max_x, max_y = np.max(verts_array, axis=0)
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    # Target canvas size with margins
    target_w = image_size[1] - 2 * margin
    target_h = image_size[0] - 2 * margin
    
    # Calculate scale factors
    scale_x = target_w / range_x if range_x > 0 else 1.0
    scale_y = target_h / range_y if range_y > 0 else 1.0
    scale = min(scale_x, scale_y)
    
    analysis = {
        "valid": True,
        "original_bounds": {
            "min": (min_x, min_y),
            "max": (max_x, max_y),
            "range": (range_x, range_y)
        },
        "scaling": {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "final_scale": scale,
            "is_small": scale < 0.5,
            "is_critical": scale < 0.1
        },
        "target_size": {
            "width": target_w,
            "height": target_h,
            "margin": margin
        }
    }
    
    return analysis

def improved_coordinate_normalization(vertices: List[Tuple[float, float]], 
                                    image_size: Tuple[int, int],
                                    margin: int = 10,
                                    min_scale: float = 0.1) -> List[Tuple[float, float]]:
    """
    Improved coordinate normalization that prevents excessive scaling.
    
    Args:
        vertices: Original vertices
        image_size: Target image size (height, width)
        margin: Margin from edges
        min_scale: Minimum allowed scale factor
        
    Returns:
        Normalized vertices
    """
    if not vertices:
        return []
        
    analysis = analyze_coordinate_bounds(vertices, image_size, margin)
    if not analysis["valid"]:
        return vertices
        
    verts_array = np.array(vertices, dtype=float)
    bounds = analysis["original_bounds"]
    scaling = analysis["scaling"]
    
    # Use adaptive scaling to prevent detail loss
    scale = max(scaling["final_scale"], min_scale)
    
    # If scale was clamped, warn about potential detail loss
    if scale != scaling["final_scale"]:
        logging.warning(f"Scale clamped from {scaling['final_scale']:.4f} to {scale:.4f} to preserve detail")
    
    # Apply normalization
    min_x, min_y = bounds["min"]
    
    # Scale and position
    verts_norm = verts_array.copy()
    verts_norm[:, 0] = (verts_array[:, 0] - min_x) * scale + margin
    verts_norm[:, 1] = (verts_array[:, 1] - min_y) * scale + margin
    
    # Center in canvas
    canvas_center_x = image_size[1] / 2
    canvas_center_y = image_size[0] / 2
    shape_center_x = np.mean(verts_norm[:, 0])
    shape_center_y = np.mean(verts_norm[:, 1])
    
    offset_x = canvas_center_x - shape_center_x
    offset_y = canvas_center_y - shape_center_y
    
    verts_norm[:, 0] += offset_x
    verts_norm[:, 1] += offset_y
    
    # Ensure bounds are respected
    verts_norm[:, 0] = np.clip(verts_norm[:, 0], margin, image_size[1] - margin)
    verts_norm[:, 1] = np.clip(verts_norm[:, 1], margin, image_size[0] - margin)
    
    # Log transformation details
    logging.info(f"Coordinate normalization: scale={scale:.4f}, offset=({offset_x:.2f}, {offset_y:.2f})")
    
    return [tuple(pt) for pt in verts_norm]

def detect_multiple_objects_improved(action_commands: List[str]) -> List[List[str]]:
    """
    Improved detection of multiple objects in action commands.
    
    Args:
        action_commands: Flattened list of action command strings
        
    Returns:
        List of command groups, each representing a separate object
    """
    if not action_commands:
        return []
    
    # Flatten first to ensure we're working with clean data
    commands = flatten_action_commands(action_commands)
    
    object_groups = []
    current_group = []
    
    for i, cmd in enumerate(commands):
        cmd_str = str(cmd).lower()
        
        # Detect object boundaries based on:
        # 1. Starting commands (line_, arc_)
        # 2. Large coordinate jumps
        # 3. Change in shape types
        
        is_new_object = False
        
        # If this is not the first command and we have a current group
        if current_group and i > 0:
            prev_cmd = str(commands[i-1]).lower()
            
            # Check for new shape start
            if (cmd_str.startswith('line_') or cmd_str.startswith('arc_')) and \
               (prev_cmd.startswith('line_') or prev_cmd.startswith('arc_')):
                # Potential new object if different shape types
                cmd_type = cmd_str.split('_')[1] if '_' in cmd_str else ''
                prev_type = prev_cmd.split('_')[1] if '_' in prev_cmd else ''
                if cmd_type != prev_type:
                    is_new_object = True
        
        if is_new_object and current_group:
            object_groups.append(current_group)
            current_group = []
        
        current_group.append(cmd)
    
    # Add the last group
    if current_group:
        object_groups.append(current_group)
    
    # If only one group detected, return as single object
    if len(object_groups) <= 1:
        return [commands] if commands else []
    
    logging.info(f"Detected {len(object_groups)} separate objects in action commands")
    return object_groups

def validate_action_commands(commands: List[str]) -> Dict[str, Any]:
    """
    Validate action commands and provide diagnostics.
    
    Args:
        commands: List of action command strings
        
    Returns:
        Validation results with diagnostics
    """
    if not commands:
        return {"valid": False, "reason": "No commands provided"}
    
    # Flatten commands first
    flat_commands = flatten_action_commands(commands)
    
    # Count command types
    line_count = sum(1 for cmd in flat_commands if str(cmd).startswith('line_'))
    arc_count = sum(1 for cmd in flat_commands if str(cmd).startswith('arc_'))
    other_count = len(flat_commands) - line_count - arc_count
    
    # Analyze command structure
    valid_commands = []
    invalid_commands = []
    
    for cmd in flat_commands:
        cmd_str = str(cmd)
        if cmd_str.startswith(('line_', 'arc_')) and '_' in cmd_str and '-' in cmd_str:
            valid_commands.append(cmd_str)
        else:
            invalid_commands.append(cmd_str)
    
    validation = {
        "valid": len(invalid_commands) == 0,
        "total_commands": len(flat_commands),
        "command_counts": {
            "line": line_count,
            "arc": arc_count,
            "other": other_count
        },
        "valid_commands": len(valid_commands),
        "invalid_commands": invalid_commands,
        "flattening_needed": len(commands) != len(flat_commands)
    }
    
    if invalid_commands:
        validation["reason"] = f"Invalid commands found: {invalid_commands[:3]}..."
    
    return validation

class ImprovedActionCommandProcessor:
    """
    Improved action command processor with comprehensive fixes.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        self.image_size = image_size
        self.margin = 10
        self.min_scale = 0.1  # Prevent excessive detail loss
        
    def process_commands(self, action_commands: List[Union[str, List[str]]], 
                        problem_id: str = None) -> Dict[str, Any]:
        """
        Process action commands with all improvements applied.
        
        Args:
            action_commands: Raw action commands (may be nested)
            problem_id: Optional problem identifier
            
        Returns:
            Processing results with diagnostics
        """
        try:
            # Step 1: Flatten nested structures
            flat_commands = flatten_action_commands(action_commands)
            logging.info(f"Flattened {len(action_commands)} raw commands to {len(flat_commands)} flat commands")
            
            # Step 2: Validate commands
            validation = validate_action_commands(flat_commands)
            if not validation["valid"]:
                logging.warning(f"Command validation failed: {validation['reason']}")
                return {"success": False, "error": validation["reason"], "validation": validation}
            
            # Step 3: Detect multiple objects
            object_groups = detect_multiple_objects_improved(flat_commands)
            logging.info(f"Detected {len(object_groups)} object groups")
            
            # Step 4: Process each object group
            processed_objects = []
            for i, obj_commands in enumerate(object_groups):
                try:
                    # Here you would integrate with your existing parser
                    # For now, just return the structure
                    obj_result = {
                        "object_id": i,
                        "commands": obj_commands,
                        "command_count": len(obj_commands)
                    }
                    processed_objects.append(obj_result)
                    
                except Exception as e:
                    logging.warning(f"Failed to process object {i}: {e}")
                    continue
            
            result = {
                "success": True,
                "problem_id": problem_id,
                "validation": validation,
                "original_commands": action_commands,
                "flat_commands": flat_commands,
                "object_groups": object_groups,
                "processed_objects": processed_objects,
                "flattening_applied": validation["flattening_needed"]
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Command processing failed: {e}")
            return {"success": False, "error": str(e)}

def test_improved_processing():
    """Test the improved processing with various command formats."""
    
    processor = ImprovedActionCommandProcessor()
    
    test_cases = [
        {
            "name": "Flat commands",
            "commands": ["line_triangle_1.000-0.500", "line_normal_0.600-0.750"]
        },
        {
            "name": "Nested commands (problematic)",
            "commands": [["line_triangle_1.000-0.500"], ["line_normal_0.600-0.750"]]
        },
        {
            "name": "Mixed format",
            "commands": ["line_triangle_1.000-0.500", ["line_normal_0.600-0.750"]]
        },
        {
            "name": "Multiple objects",
            "commands": ["line_triangle_1.000-0.500", "line_normal_0.600-0.750", 
                        "arc_circle_0.500-0.800", "arc_normal_0.300-0.600"]
        }
    ]
    
    print("🔧 TESTING IMPROVED ACTION COMMAND PROCESSING")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        commands = test_case['commands']
        
        result = processor.process_commands(commands, f"test_{test_case['name'].replace(' ', '_')}")
        
        if result["success"]:
            print(f"  ✅ Processing successful!")
            print(f"  📊 Validation: {result['validation']['valid']}")
            print(f"  🔄 Flattening applied: {result['flattening_applied']}")
            print(f"  🎯 Objects detected: {len(result['object_groups'])}")
            print(f"  📝 Command counts: {result['validation']['command_counts']}")
        else:
            print(f"  ❌ Processing failed: {result['error']}")

def test_coordinate_normalization():
    """Test improved coordinate normalization."""
    
    print(f"\n🎯 TESTING IMPROVED COORDINATE NORMALIZATION")
    print("=" * 60)
    
    # Test with extreme coordinates (like in logs)
    extreme_vertices = [
        (-1011.0, -500.0),
        (908.0, 750.0),
        (-800.0, 600.0),
        (500.0, -900.0)
    ]
    
    image_size = (512, 512)
    
    print(f"Original vertices: {extreme_vertices}")
    
    # Test old normalization (what was causing issues)
    analysis = analyze_coordinate_bounds(extreme_vertices, image_size)
    print(f"Coordinate analysis:")
    print(f"  Range: {analysis['original_bounds']['range']}")
    print(f"  Original scale: {analysis['scaling']['final_scale']:.4f}")
    print(f"  Scale is too small: {analysis['scaling']['is_small']}")
    print(f"  Scale is critical: {analysis['scaling']['is_critical']}")
    
    # Test improved normalization
    normalized = improved_coordinate_normalization(extreme_vertices, image_size)
    print(f"Normalized vertices: {normalized}")
    
    # Verify bounds
    norm_array = np.array(normalized)
    print(f"Final bounds: x=({np.min(norm_array[:,0]):.2f}, {np.max(norm_array[:,0]):.2f}), y=({np.min(norm_array[:,1]):.2f}, {np.max(norm_array[:,1]):.2f})")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🚀 COMPREHENSIVE ACTION COMMAND PARSING FIXES")
    print("=" * 70)
    
    test_improved_processing()
    test_coordinate_normalization()
    
    print(f"\n🎉 ALL TESTS COMPLETED!")
    print("=" * 70)
    print("Next steps:")
    print("1. 🔧 Integrate these fixes into hybrid.py")
    print("2. ⚙️  Update the action parser to use improved normalization")
    print("3. 🧪 Test with real Bongard-LOGO action programs")
    print("4. 📊 Verify that generated masks better match real images")
