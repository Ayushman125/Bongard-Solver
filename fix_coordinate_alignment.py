#!/usr/bin/env python3
"""
Comprehensive fix for coordinate system alignment between real Bongard-LOGO images and generated masks.

Based on official NVLabs BongardImagePainter analysis:
- Canvas size: 800x800
- Coordinate range: (-360, 360) for both X and Y  
- Final image size: 512x512
- Background: white for final images
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_pipeline.logo_parser import UnifiedActionParser
from bongard_augmentor.hybrid import ActionMaskGenerator

class NVLabsAlignedCoordinateSystem:
    """
    Coordinate system that exactly matches official NVLabs BongardImagePainter.
    """
    
    def __init__(self):
        # Official NVLabs parameters from bongard_painter.py
        self.canvas_size = 800  # Official canvas width/height
        self.coord_range = (-360, 360)  # Official coordinate range
        self.final_image_size = (512, 512)  # Official final image size
        self.coord_range_size = self.coord_range[1] - self.coord_range[0]  # 720
        
    def nvlabs_to_pixel_coordinates(self, vertices: List[Tuple[float, float]], 
                                  target_size: Tuple[int, int] = None) -> List[Tuple[int, int]]:
        """
        Convert NVLabs coordinates (-360,360) to pixel coordinates exactly like official implementation.
        
        Official NVLabs process:
        1. Turtle graphics on 800x800 canvas with (-360,360) range
        2. Save as PostScript
        3. Resize to 512x512 PNG
        """
        if target_size is None:
            target_size = self.final_image_size
            
        pixel_vertices = []
        
        for x, y in vertices:
            # Map from (-360, 360) to (0, canvas_size) like official turtle graphics
            canvas_x = (x - self.coord_range[0]) / self.coord_range_size * self.canvas_size
            canvas_y = (y - self.coord_range[0]) / self.coord_range_size * self.canvas_size
            
            # Convert from 800x800 canvas to final image size (exactly like NVLabs resize)
            pixel_x = int(canvas_x * target_size[1] / self.canvas_size)
            pixel_y = int(canvas_y * target_size[0] / self.canvas_size)
            
            # Flip Y axis (turtle graphics has (0,0) at center, image has (0,0) at top-left)
            pixel_y = target_size[0] - 1 - pixel_y
            
            # Clamp to image bounds
            pixel_x = max(0, min(target_size[1] - 1, pixel_x))
            pixel_y = max(0, min(target_size[0] - 1, pixel_y))
            
            pixel_vertices.append((pixel_x, pixel_y))
            
        return pixel_vertices

class FixedActionMaskGenerator(ActionMaskGenerator):
    """
    Action mask generator with corrected coordinate system alignment.
    """
    
    def __init__(self, image_size=(512, 512), use_nvlabs=True):
        super().__init__(image_size, use_nvlabs)
        self.coord_system = NVLabsAlignedCoordinateSystem()
        
    def _render_vertices_to_mask_fixed(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Render vertices using the corrected coordinate system that matches real Bongard-LOGO images.
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if len(vertices) < 2:
            return mask
            
        # Convert to pixel coordinates using official NVLabs coordinate system
        pixel_vertices = self.coord_system.nvlabs_to_pixel_coordinates(vertices, self.image_size)
        
        if len(pixel_vertices) < 2:
            return mask
            
        # Create high-resolution mask for anti-aliasing
        scale_factor = 4
        high_res_size = (self.image_size[0] * scale_factor, self.image_size[1] * scale_factor)
        high_res_mask = np.zeros(high_res_size, dtype=np.uint8)
        
        # Scale pixel coordinates to high-res
        high_res_vertices = [(x * scale_factor, y * scale_factor) for x, y in pixel_vertices]
        points_array = np.array(high_res_vertices, dtype=np.int32)
        
        # Use thick strokes to match real dataset appearance
        # Real Bongard-LOGO images have thick, prominent strokes
        stroke_thickness = max(24, int(high_res_size[0] * 0.05))  # ~5% of image height
        
        # Enhanced shape detection and rendering
        if len(points_array) >= 3:
            # Check if this forms a closed shape
            first_point = points_array[0]
            last_point = points_array[-1]
            distance_to_start = np.linalg.norm(last_point - first_point)
            
            if distance_to_start < stroke_thickness * 2:  # Closed shape threshold
                # Render as filled polygon with thick outline
                cv2.fillPoly(high_res_mask, [points_array], 255)
                cv2.polylines(high_res_mask, [points_array], True, 255, 
                             thickness=stroke_thickness, lineType=cv2.LINE_AA)
            else:
                # Render as thick open path
                cv2.polylines(high_res_mask, [points_array], False, 255, 
                             thickness=stroke_thickness, lineType=cv2.LINE_AA)
                
                # Add thick end caps
                cap_radius = stroke_thickness // 2
                cv2.circle(high_res_mask, tuple(points_array[0]), cap_radius, 255, -1)
                cv2.circle(high_res_mask, tuple(points_array[-1]), cap_radius, 255, -1)
        else:
            # Simple line with extra thick stroke
            cv2.polylines(high_res_mask, [points_array], False, 255, 
                         thickness=stroke_thickness * 2, lineType=cv2.LINE_AA)
        
        # Downsample to target resolution with anti-aliasing
        final_mask = cv2.resize(high_res_mask, 
                               (self.image_size[1], self.image_size[0]), 
                               interpolation=cv2.INTER_AREA)
        
        return final_mask

def test_coordinate_alignment():
    """
    Test the coordinate alignment fix with a real Bongard-LOGO example.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Test with a real action command
    test_commands = [
        "line_normal_1.000-0.500",
        "line_normal_0.500-1.000"
    ]
    
    print("Testing coordinate alignment fix...")
    
    # Initialize parsers
    parser = UnifiedActionParser()
    fixed_generator = FixedActionMaskGenerator()
    original_generator = ActionMaskGenerator()
    
    # Parse action commands to get vertices
    try:
        # Use the comprehensive parser directly to get vertices
        shape = parser.comprehensive_parser.parse_action_commands(test_commands, "test_image")
        if not shape or not shape.vertices:
            print("Failed to parse test commands")
            return
            
        vertices = shape.vertices
        print(f"Parsed {len(vertices)} vertices from test commands")
        
        # Generate masks with both systems
        fixed_mask = fixed_generator._render_vertices_to_mask_fixed(vertices)
        original_mask = original_generator._render_vertices_to_mask(vertices)
        
        # Calculate mask statistics
        fixed_coverage = np.mean(fixed_mask > 0) * 100
        original_coverage = np.mean(original_mask > 0) * 100
        fixed_sum = np.sum(fixed_mask)
        original_sum = np.sum(original_mask)
        
        print(f"Fixed mask: coverage={fixed_coverage:.2f}%, sum={fixed_sum}")
        print(f"Original mask: coverage={original_coverage:.2f}%, sum={original_sum}")
        
        # Save comparison images
        cv2.imwrite('fixed_mask_test.png', fixed_mask)
        cv2.imwrite('original_mask_test.png', original_mask)
        
        # Create side-by-side comparison
        comparison = np.hstack([original_mask, fixed_mask])
        cv2.imwrite('coordinate_alignment_comparison.png', comparison)
        
        print("Coordinate alignment test completed!")
        print("Generated files:")
        print("- fixed_mask_test.png")
        print("- original_mask_test.png") 
        print("- coordinate_alignment_comparison.png")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coordinate_alignment()
