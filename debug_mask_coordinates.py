#!/usr/bin/env python3
"""Debug coordinate transformation issues in mask generation."""

import numpy as np
import cv2
import logging
from src.bongard_augmentor.hybrid import ActionMaskGenerator
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def debug_coordinate_transformation():
    """Debug the coordinate transformation that's causing white masks."""
    
    print("=== DEBUGGING COORDINATE TRANSFORMATION ===")
    
    # Use the exact action commands from your log
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
    parser = generator.action_parser
    
    print(f"Input commands: {action_commands}")
    
    # Step 1: Parse the commands to get vertices
    try:
        parsed_data = parser.parse_action_commands(action_commands, "test_problem_0000")
        print(f"✅ Parsing successful")
        print(f"Vertices count: {len(parsed_data.vertices)}")
        print(f"Raw vertices: {parsed_data.vertices}")
        
        # Step 2: Analyze coordinate ranges
        if parsed_data.vertices:
            verts = np.array(parsed_data.vertices)
            min_x, min_y = np.min(verts, axis=0)
            max_x, max_y = np.max(verts, axis=0)
            print(f"Coordinate ranges: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
            
            # Step 3: Simulate the normalization process from the hybrid.py code
            print("\n--- NORMALIZATION PROCESS ---")
            verts_norm = verts.copy()
            range_x = max_x - min_x
            range_y = max_y - min_y
            print(f"Ranges: X={range_x:.2f}, Y={range_y:.2f}")
            
            if range_x > 0 and range_y > 0:
                margin = 10
                target_w = 512 - 2 * margin  # 492
                target_h = 512 - 2 * margin  # 492
                scale_x = target_w / range_x
                scale_y = target_h / range_y
                scale = min(scale_x, scale_y)
                print(f"Calculated scale: min({scale_x:.4f}, {scale_y:.4f}) = {scale:.4f}")
                
                min_scale = 0.15
                if scale < min_scale:
                    print(f"⚠️  Scale {scale:.4f} clamped to {min_scale:.4f}")
                    scale = min_scale
                
                # Apply scaling
                verts_norm[:,0] = (verts[:,0] - min_x) * scale + margin
                verts_norm[:,1] = (verts[:,1] - min_y) * scale + margin
                print(f"After initial scaling: {verts_norm}")
                
                # Apply centering
                canvas_center_x = 512 / 2  # 256
                canvas_center_y = 512 / 2  # 256
                shape_center_x = np.mean(verts_norm[:,0])
                shape_center_y = np.mean(verts_norm[:,1])
                offset_x = canvas_center_x - shape_center_x
                offset_y = canvas_center_y - shape_center_y
                print(f"Centering offsets: ({offset_x:.2f}, {offset_y:.2f})")
                
                verts_norm[:,0] += offset_x
                verts_norm[:,1] += offset_y
                print(f"After centering: {verts_norm}")
                
                # Apply clipping
                verts_norm[:,0] = np.clip(verts_norm[:,0], margin, 512 - margin)
                verts_norm[:,1] = np.clip(verts_norm[:,1], margin, 512 - margin)
                print(f"After clipping: {verts_norm}")
        
        # Step 4: Test mask generation with original vertices
        print("\n--- TESTING MASK GENERATION ---")
        
        # Test 1: Original parsed vertices before normalization
        print("Test 1: Original vertices")
        mask1 = generator._render_vertices_to_mask(parsed_data.vertices)
        print(f"Mask sum: {np.sum(mask1)}, nonzero pixels: {np.count_nonzero(mask1)}")
        
        # Test 2: Use the parser's own rendering method directly
        print("Test 2: Parser's _render_vertices_to_image method")
        try:
            mask2 = parser._render_vertices_to_image(parsed_data.vertices, (512, 512))
            print(f"Mask sum: {np.sum(mask2)}, nonzero pixels: {np.count_nonzero(mask2)}")
        except Exception as e:
            print(f"Parser rendering failed: {e}")
        
        # Test 3: Manual fallback rendering
        print("Test 3: Manual fallback rendering")
        mask3 = generator._manual_render_fallback(parsed_data.vertices)
        print(f"Mask sum: {np.sum(mask3)}, nonzero pixels: {np.count_nonzero(mask3)}")
        
        # Test 4: Try different coordinate interpretation
        print("Test 4: Alternative coordinate interpretation")
        # Maybe the coordinates are already in pixel space?
        simple_mask = np.zeros((512, 512), dtype=np.uint8)
        if len(parsed_data.vertices) >= 2:
            points = []
            for v in parsed_data.vertices:
                x, y = int(v[0]), int(v[1])
                if 0 <= x < 512 and 0 <= y < 512:
                    points.append([x, y])
            
            if len(points) >= 2:
                points_array = np.array(points, dtype=np.int32)
                cv2.polylines(simple_mask, [points_array], False, 255, thickness=2)
                print(f"Direct pixel interpretation - Mask sum: {np.sum(simple_mask)}, nonzero: {np.count_nonzero(simple_mask)}")
        
        # Save debug images
        cv2.imwrite("debug_mask_test1.png", mask1)
        cv2.imwrite("debug_mask_test2.png", mask2 if 'mask2' in locals() else np.zeros((512,512), dtype=np.uint8))
        cv2.imwrite("debug_mask_test3.png", mask3)
        cv2.imwrite("debug_mask_simple.png", simple_mask)
        print("Debug masks saved as debug_mask_test*.png")
        
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()

def test_simple_vertices():
    """Test with simple, known vertices to isolate the issue."""
    
    print("\n=== TESTING SIMPLE VERTICES ===")
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    # Test with simple line vertices
    simple_vertices = [(100.0, 100.0), (400.0, 400.0)]
    print(f"Testing simple vertices: {simple_vertices}")
    
    mask = generator._render_vertices_to_mask(simple_vertices)
    print(f"Result: sum={np.sum(mask)}, nonzero={np.count_nonzero(mask)}")
    
    # Try manual rendering
    manual_mask = generator._manual_render_fallback(simple_vertices)
    print(f"Manual: sum={np.sum(manual_mask)}, nonzero={np.count_nonzero(manual_mask)}")
    
    cv2.imwrite("debug_simple_mask.png", mask)
    cv2.imwrite("debug_simple_manual.png", manual_mask)

if __name__ == "__main__":
    debug_coordinate_transformation()
    test_simple_vertices()
