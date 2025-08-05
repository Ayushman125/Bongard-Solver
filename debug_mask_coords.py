#!/usr/bin/env python3
"""Debug script to check mask dimensions and coordinate bounds."""

from src.bongard_augmentor.hybrid import ActionMaskGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

def debug_mask_coordinates():
    """Debug mask generation to find coordinate issues."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print("=== DEBUGGING MASK COORDINATE ISSUES ===")
    
    # Test with various commands that might go out of bounds
    test_commands = [
        # Test normal cases
        ["line_triangle_1.000-0.500"],
        ["arc_circle_0.500_0.625-0.500"],
        
        # Test extreme parameters that might cause cropping
        ["line_triangle_1.000-0.900"],  # High final_param
        ["line_square_1.000-0.100"],    # Low final_param
        ["arc_circle_0.900_0.900-0.900"],  # All high params
        ["line_normal_2.000-0.500"],    # High scale (might go out of bounds)
        
        # Test realistic Bongard combinations
        ["line_triangle_1.000-0.500", "line_square_1.000-0.833", "line_circle_1.000-0.917"]
    ]
    
    for i, commands in enumerate(test_commands):
        print(f"\n{i+1}. Testing: {commands}")
        
        try:
            # Generate mask using semantic rendering
            mask = generator._render_semantic_commands_to_mask(commands)
            
            # Check mask properties
            pixels = np.sum(mask > 0)
            mask_height, mask_width = mask.shape
            
            # Find bounding box of non-zero pixels
            if pixels > 0:
                coords = np.where(mask > 0)
                min_y, max_y = coords[0].min(), coords[0].max()
                min_x, max_x = coords[1].min(), coords[1].max()
                
                print(f"   📊 Mask: {mask_width}x{mask_height}, {pixels} pixels")
                print(f"   📍 Bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
                
                # Check if content touches edges (might be cropped)
                edge_touch = {
                    'left': min_x <= 2,
                    'right': max_x >= mask_width - 3,
                    'top': min_y <= 2, 
                    'bottom': max_y >= mask_height - 3
                }
                
                touching_edges = [k for k, v in edge_touch.items() if v]
                if touching_edges:
                    print(f"   ⚠️  EDGE TOUCH: {touching_edges} (possible cropping!)")
                else:
                    print(f"   ✅ Content within bounds")
                    
                # Check center-relative coordinates
                center_x, center_y = mask_width // 2, mask_height // 2
                relative_bounds = {
                    'x_range': (min_x - center_x, max_x - center_x),
                    'y_range': (min_y - center_y, max_y - center_y)
                }
                print(f"   🎯 Center-relative: x={relative_bounds['x_range']}, y={relative_bounds['y_range']}")
                
            else:
                print(f"   ❌ EMPTY MASK!")
                
            # Test full pipeline
            full_mask = generator.generate_mask_from_actions(commands, f"debug_test_{i}")
            full_pixels = np.sum(full_mask > 0)
            print(f"   🔧 Full pipeline: {full_pixels} pixels")
            
            # Save debug visualization
            if pixels > 0:
                debug_img = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
                debug_img[mask > 0] = [255, 255, 255]  # White for mask
                
                # Mark center
                cv2.circle(debug_img, (center_x, center_y), 5, (0, 255, 0), -1)  # Green center
                
                # Mark bounds
                cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)  # Blue bbox
                
                cv2.imwrite(f"debug_mask_{i}.png", debug_img)
                print(f"   💾 Saved debug_mask_{i}.png")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

def test_coordinate_ranges():
    """Test what coordinate ranges are being used in the drawing functions."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print(f"\n=== COORDINATE RANGE ANALYSIS ===")
    print(f"Canvas size: {generator.image_size}")
    print(f"Canvas center: ({generator.image_size[1]//2}, {generator.image_size[0]//2})")
    
    # Test base_size calculation
    for scale in [0.5, 1.0, 1.5, 2.0]:
        base_size = int(min(generator.image_size) * 0.15 * scale)
        print(f"Scale {scale}: base_size = {base_size}")
        
        # Check if this could go out of bounds
        center = (generator.image_size[1]//2, generator.image_size[0]//2)
        max_reach = center[0] + base_size
        min_reach = center[0] - base_size
        
        out_of_bounds = max_reach >= generator.image_size[1] or min_reach < 0
        print(f"   Reach: [{min_reach}, {max_reach}] {'❌ OUT OF BOUNDS' if out_of_bounds else '✅ OK'}")

if __name__ == "__main__":
    try:
        debug_mask_coordinates()
        test_coordinate_ranges()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
