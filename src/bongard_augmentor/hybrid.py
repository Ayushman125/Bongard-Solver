# ------------------------------------------------------------------
# Action-Based Augmentation Pipeline
# Processes Action Programs instead of Real Images  
# ------------------------------------------------------------------

import os
import sys
import numpy as np
import cv2
import time
import logging
import json
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy import ndimage as ndi
from skimage import morphology as ski_morphology
from skimage.metrics import structural_similarity as ssim

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import data loading and parsing components (same as logo_to_shape.py)
from src.data_pipeline.data_loader import load_action_programs
from src.data_pipeline.logo_parser import UnifiedActionParser, ComprehensiveNVLabsParser, NVLABS_AVAILABLE

# Import the NVLabs parser directly - it now has direct NVLabs integration
NVLabsActionParser = ComprehensiveNVLabsParser

if NVLABS_AVAILABLE:
    logging.info("NVLabs classes available - using direct NVLabs integration")
else:
    logging.warning("NVLabs classes not available - using fallback implementation")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoaderWrapper:
    """Wrapper for data loading functions to match the expected interface."""
    
    def load_action_programs(self, input_dir, problems_list=None, n_select=50):
        """Load action programs with optional filtering."""
        # Load all action programs
        action_data = load_action_programs(input_dir)
        
        # Filter by problems list if provided
        if problems_list and os.path.exists(problems_list):
            with open(problems_list, 'r') as f:
                selected_problems = [line.strip() for line in f if line.strip()]
            action_data = {k: v for k, v in action_data.items() if k in selected_problems}
        
        # Limit number if n_select is specified
        if n_select and len(action_data) > n_select:
            action_data = dict(list(action_data.items())[:n_select])
        
        # Restructure data to match expected format
        result = {}
        for problem_id, problem_data in action_data.items():
            if isinstance(problem_data, list) and len(problem_data) == 2:
                positive_examples, negative_examples = problem_data
                
                # Determine category from problem_id
                if problem_id.startswith('bd_'):
                    category = 'bd'
                elif problem_id.startswith('ff_'):
                    category = 'ff'
                elif problem_id.startswith('hd_'):
                    category = 'hd'
                else:
                    category = 'unknown'
                
                result[problem_id] = {
                    'category': category,
                    'positive_examples': positive_examples,
                    'negative_examples': negative_examples
                }
        
    def run_pipeline(self):
        pass

class ActionMaskGenerator:
    """CORRECTED: Generates high-quality masks using the fixed coordinate system."""
    
    def __init__(self, image_size=(512, 512), use_nvlabs=True):
        # FIXED: Use consistent 512x512 canvas size to match Bongard-LOGO dataset
        self.image_size = image_size
        
        # Use NVLabs-integrated parser for maximum compatibility
        if use_nvlabs and NVLABS_AVAILABLE:
            # CRITICAL FIX: Use consistent 512x512 canvas size and coordinate range
            self.action_parser = NVLabsActionParser(
                canvas_size=512,  # Always use 512 to match real dataset
                coordinate_range=(-360, 360)  # Standard NVLabs coordinate range
            )
            self.parser_type = "NVLabs-Direct"
            logging.info("ActionMaskGenerator: Using NVLabs direct integration parser")
        else:
            self.action_parser = UnifiedActionParser()
            self.parser_type = "Custom"
            logging.info("ActionMaskGenerator: Using custom UnifiedActionParser fallback")
        
        # Verify the parser settings
        # (No stray imports or code fragments)
    
    def _render_vertices_to_mask(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        CRITICAL FIX: Generate masks using coordinate system that matches official NVLabs BongardImagePainter.
        
        Official NVLabs process:
        1. Turtle graphics on 800x800 canvas with (-360,360) coordinate range
        2. Save as PostScript, resize to 512x512 PNG
        3. White background, black drawings
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if len(vertices) < 2:
            logging.warning("[MASKGEN_FIXED] Less than 2 vertices provided for rendering")
            return mask
            
        logging.info(f"[MASKGEN_FIXED] Rendering {len(vertices)} vertices using NVLabs-aligned coordinate system")
        logging.info(f"[MASKGEN_FIXED] Vertex sample: {vertices[:3]}{'...' if len(vertices) > 3 else ''}")
        
        # Official NVLabs coordinate system parameters
        canvas_size = 800  # Official canvas width/height
        coord_range = (-360, 360)  # Official coordinate range
        coord_range_size = coord_range[1] - coord_range[0]  # 720
        
        # Convert NVLabs coordinates (-360,360) to pixel coordinates like official implementation
        pixel_vertices = []
        for x, y in vertices:
            # Map from (-360, 360) to (0, canvas_size) like official turtle graphics
            canvas_x = (x - coord_range[0]) / coord_range_size * canvas_size
            canvas_y = (y - coord_range[0]) / coord_range_size * canvas_size
            
            # Convert from 800x800 canvas to final image size (exactly like NVLabs resize)
            pixel_x = int(canvas_x * self.image_size[1] / canvas_size)
            pixel_y = int(canvas_y * self.image_size[0] / canvas_size)
            
            # Flip Y axis (turtle graphics has (0,0) at center, image has (0,0) at top-left)
            pixel_y = self.image_size[0] - 1 - pixel_y
            
            # Clamp to image bounds
            pixel_x = max(0, min(self.image_size[1] - 1, pixel_x))
            pixel_y = max(0, min(self.image_size[0] - 1, pixel_y))
            
            pixel_vertices.append((pixel_x, pixel_y))
            
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
        
        nonzero_pixels = np.count_nonzero(final_mask)
        mask_sum = np.sum(final_mask)
        logging.info(f"[MASKGEN_FIXED] NVLabs-aligned mask: {nonzero_pixels}/{final_mask.size} pixels ({100*nonzero_pixels/final_mask.size:.1f}%), sum={mask_sum}")
        
        return final_mask
    
    def _manual_render_fallback(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Fallback manual rendering method with THICK STROKE rendering to match real dataset.
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if len(vertices) < 2:
            return mask
        
        # CRITICAL FIX: Use much thicker strokes to match real Bongard-LOGO images
        # Real images show thick, filled shapes, not thin lines
        base_thickness = max(20, int(self.image_size[0] * 0.06))  # Increased to ~6% of image height
        
        # Create high-resolution canvas for anti-aliasing
        scale_factor = 4  # 4x supersampling for anti-aliasing
        high_res_size = (self.image_size[0] * scale_factor, self.image_size[1] * scale_factor)
        high_res_mask = np.zeros(high_res_size, dtype=np.uint8)
        
        # Convert vertices to pixel coordinates (already normalized by the calling function)
        points = []
        for vertex in vertices:
            if len(vertex) >= 2 and isinstance(vertex[0], (int, float)) and isinstance(vertex[1], (int, float)):
                # Scale up for high-res rendering
                pixel_x = vertex[0] * scale_factor
                pixel_y = vertex[1] * scale_factor
                
                # Clamp to valid range
                pixel_x = max(0, min(high_res_size[1] - 1, int(round(pixel_x))))
                pixel_y = max(0, min(high_res_size[0] - 1, int(round(pixel_y))))
                points.append([pixel_x, pixel_y])
        
        logging.debug(f"Thick stroke rendering: {len(vertices)} vertices -> {len(points)} valid points")
        
        # CRITICAL FIX: Render with thick strokes to match real dataset appearance
        if len(points) >= 2:
            points_array = np.array(points, dtype=np.int32)
            thick_stroke = base_thickness * scale_factor
            
            # Enhanced shape analysis for better filling
            if len(points) >= 3:
                # Calculate if this resembles a closed shape
                first_point = np.array(points[0])
                last_point = np.array(points[-1])
                distance = np.linalg.norm(first_point - last_point)
                closure_threshold = thick_stroke * 3
                
                # Also check if points form a roughly enclosed area
                points_np = np.array(points)
                area = cv2.contourArea(points_array)
                perimeter = cv2.arcLength(points_array, False)
                
                is_closed = distance < closure_threshold
                is_substantial_area = area > (thick_stroke * 2) ** 2
                
                if is_closed and is_substantial_area:
                    # Fill the shape for closed, substantial shapes
                    cv2.fillPoly(high_res_mask, [points_array], 255)
                    logging.debug(f"Rendered filled shape: area={area:.0f}, perimeter={perimeter:.0f}")
                else:
                    # Use extra thick strokes for complex open paths
                    extra_thick = int(thick_stroke * 1.5)
                    cv2.polylines(high_res_mask, [points_array], False, 255, 
                                 thickness=extra_thick, lineType=cv2.LINE_AA)
                    
                    # Add thick rounded caps for better visual impact
                    cap_radius = extra_thick // 2
                    for point in points:
                        cv2.circle(high_res_mask, tuple(point), cap_radius, 255, -1)
                    
                    logging.debug(f"Rendered thick open path: {len(points)} points, thickness={extra_thick//scale_factor}")
            else:
                # Simple line - use very thick stroke
                ultra_thick = int(thick_stroke * 2)
                cv2.polylines(high_res_mask, [points_array], False, 255, 
                             thickness=ultra_thick, lineType=cv2.LINE_AA)
                
                # Add large end caps
                cap_radius = ultra_thick // 2
                for point in points:
                    cv2.circle(high_res_mask, tuple(point), cap_radius, 255, -1)
        
        # Downsample with anti-aliasing to target resolution
        mask = cv2.resize(high_res_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        
        # Apply threshold to create clean binary image
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _render_vertices_alternative(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Alternative vertex rendering approach that properly handles NVLabs coordinate system.
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if len(vertices) < 2:
            return mask
        
        try:
            # Convert from NVLabs coordinate system to pixel coordinates
            # NVLabs uses (-360, 360) range, we need to map to (0, 512)
            points = []
            coordinate_range = 720.0  # (-360, 360) = 720 total range
            
            for vertex in vertices:
                if len(vertex) >= 2 and isinstance(vertex[0], (int, float)) and isinstance(vertex[1], (int, float)):
                    # Map from NVLabs coordinates to pixel coordinates
                    # (-360, 360) -> (0, 512)
                    x_norm = (vertex[0] + 360.0) / coordinate_range  # Normalize to [0, 1]
                    y_norm = (vertex[1] + 360.0) / coordinate_range  # Normalize to [0, 1]
                    
                    # Scale to image size
                    x = int(x_norm * self.image_size[1])
                    y = int(y_norm * self.image_size[0])
                    
                    # Clamp to valid pixel range
                    x = max(0, min(self.image_size[1] - 1, x))
                    y = max(0, min(self.image_size[0] - 1, y))
                    points.append([x, y])
            
            if len(points) >= 2:
                points_array = np.array(points, dtype=np.int32)
                
                # Use thicker lines to match real dataset appearance
                thickness = 3
                cv2.polylines(mask, [points_array], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
                
                # Also draw points as small circles for better connectivity
                for point in points:
                    cv2.circle(mask, tuple(point), 2, 255, -1)
                
                logging.info(f"[MASKGEN] Alternative rendering: {len(points)} points rendered with NVLabs coordinate mapping")
                logging.info(f"[MASKGEN] Sample transformed points: {points[:3]}{'...' if len(points) > 3 else ''}")
            
        except Exception as e:
            logging.warning(f"Alternative rendering failed: {e}")
        
        return mask
    
    def _render_semantic_commands_to_mask(self, action_commands: List[str]) -> np.ndarray:
        """Professional rendering for Bongard semantic action commands.
        
        Handles commands like:
        - line_triangle_1.000-0.500
        - arc_zigzag_0.500_0.625-0.500
        - line_square_1.000-0.833
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        canvas_center = (self.image_size[1] // 2, self.image_size[0] // 2)
        
        try:
            for cmd in action_commands:
                # Convert to string if needed
                cmd_str = str(cmd) if not isinstance(cmd, str) else cmd
                
                # Parse semantic command format
                parsed = self._parse_semantic_command(cmd_str)
                if not parsed:
                    continue
                    
                stroke_type, shape_type, params = parsed
                
                # Render based on stroke and shape type
                if stroke_type == 'line':
                    self._render_line_shape(mask, shape_type, params, canvas_center)
                elif stroke_type == 'arc':
                    self._render_arc_shape(mask, shape_type, params, canvas_center)
                    
        except Exception as e:
            logging.warning(f"Failed to render semantic commands: {e}")
        
        return mask
    
    def _parse_semantic_command(self, cmd: str) -> Optional[Tuple[str, str, Dict]]:
        """Parse semantic command into stroke_type, shape_type, and parameters.
        
        Examples:
        - 'line_triangle_1.000-0.500' -> ('line', 'triangle', {'scale': 1.0, 'param': 0.5})
        - 'arc_zigzag_0.500_0.625-0.500' -> ('arc', 'zigzag', {'scale': 0.5, 'arc_param': 0.625, 'param': 0.5})
        """
        try:
            # Split by underscore to get components
            parts = cmd.strip().split('_')
            if len(parts) < 3:
                return None
                
            stroke_type = parts[0]  # 'line' or 'arc'
            shape_type = parts[1]   # 'normal', 'triangle', 'square', 'circle', 'zigzag'
            
            # FIXED: Handle cases where parameters contain underscores
            # Rejoin all parts after the shape_type to get the full parameter string
            param_str = '_'.join(parts[2:])  # '1.000-0.500' or '0.500_0.625-0.500'
            
            # Parse parameters
            params = self._parse_command_parameters(param_str)
            
            return stroke_type, shape_type, params
            
        except Exception as e:
            logging.debug(f"Failed to parse semantic command '{cmd}': {e}")
            return None
    
    def _parse_command_parameters(self, param_str: str) -> Dict:
        """Parse parameter string into structured parameters.
        
        Handles all Bongard command parameter formats:
        - '1.000-0.500' -> {'scale': 1.0, 'final_param': 0.5}
        - '0.500_0.625-0.500' -> {'scale': 0.5, 'arc_param': 0.625, 'final_param': 0.5}
        - '1.000-0.917' -> {'scale': 1.0, 'final_param': 0.917}
        - '0.518-0.792' -> {'scale': 0.518, 'final_param': 0.792}
        
        Parameters meaning:
        - scale: Size/scale factor (first number)
        - arc_param: Arc curvature parameter (middle number, arc commands only)
        - final_param: Position/orientation parameter (number after dash)
        """
        params = {}
        
        try:
            # Split by dash to separate main params from final param
            if '-' in param_str:
                main_part, final_param = param_str.rsplit('-', 1)
                params['final_param'] = float(final_param)
            else:
                main_part = param_str
                params['final_param'] = 0.5  # Default
            
            # Parse main part (could have underscores for multiple values)
            main_values = main_part.split('_')
            
            if len(main_values) >= 1:
                params['scale'] = float(main_values[0])
            else:
                params['scale'] = 1.0  # Default scale
                
            if len(main_values) >= 2:
                # Second parameter is arc parameter for arc commands
                params['arc_param'] = float(main_values[1])
            else:
                params['arc_param'] = 0.625  # Default arc parameter
                
            # Additional parameters if present (for complex commands)
            if len(main_values) >= 3:
                params['extra_param'] = float(main_values[2])
                
        except ValueError as e:
            # Default parameters if parsing fails
            logging.debug(f"Parameter parsing failed for '{param_str}': {e}")
            params = {'scale': 1.0, 'arc_param': 0.625, 'final_param': 0.5, 'extra_param': 0.0}
            
        return params
    
    def _render_line_shape(self, mask: np.ndarray, shape_type: str, params: Dict, center: Tuple[int, int]):
        """Render line-based shapes with proper geometry and parameter interpretation."""
        scale = params.get('scale', 1.0)
        final_param = params.get('final_param', 0.5)
        
        # IMPROVED: Better base size calculation that respects scale parameter range
        # Scale parameter can vary from ~0.5 to 1.0 in real data
        max_safe_size = min(self.image_size) // 3  # Safe maximum to stay within bounds
        base_scale_factor = 0.12 + (scale * 0.08)  # Scale from 0.12 to 0.20 based on scale param
        base_size = min(int(min(self.image_size) * base_scale_factor), max_safe_size)
        
        if shape_type == 'normal':
            self._draw_normal_line(mask, center, base_size, final_param)
        elif shape_type == 'triangle':
            self._draw_triangle_line(mask, center, base_size, final_param)
        elif shape_type == 'square':
            self._draw_square_line(mask, center, base_size, final_param)
        elif shape_type == 'circle':
            self._draw_circle_line(mask, center, base_size, final_param)
        elif shape_type == 'zigzag':
            self._draw_zigzag_line(mask, center, base_size, final_param)
    
    def _render_arc_shape(self, mask: np.ndarray, shape_type: str, params: Dict, center: Tuple[int, int]):
        """Render arc-based shapes with proper curvature and parameter interpretation."""
        scale = params.get('scale', 1.0)
        # Use actual parsed arc_param, no hardcoded defaults
        arc_param = params.get('arc_param', 0.625)  # Only as last resort fallback
        final_param = params.get('final_param', 0.5)
        
        # IMPROVED: Better base size calculation that respects scale parameter range
        # Arc scale parameter can vary from ~0.35 to 1.0 in real data
        max_safe_size = min(self.image_size) // 3  # Safe maximum to stay within bounds
        base_scale_factor = 0.10 + (scale * 0.10)  # Scale from 0.10 to 0.20 based on scale param
        base_size = min(int(min(self.image_size) * base_scale_factor), max_safe_size)
        
        if shape_type == 'normal':
            self._draw_normal_arc(mask, center, base_size, arc_param, final_param)
        elif shape_type == 'triangle':
            self._draw_triangle_arc(mask, center, base_size, arc_param, final_param)
        elif shape_type == 'square':
            self._draw_square_arc(mask, center, base_size, arc_param, final_param)
        elif shape_type == 'circle':
            self._draw_circle_arc(mask, center, base_size, arc_param, final_param)
        elif shape_type == 'zigzag':
            self._draw_zigzag_arc(mask, center, base_size, arc_param, final_param)
    
    def _draw_normal_line(self, mask: np.ndarray, center: Tuple[int, int], size: int, param: float):
        """Draw a normal straight line."""
        angle = param * 2 * np.pi
        end_x = int(center[0] + size * np.cos(angle))
        end_y = int(center[1] + size * np.sin(angle))
        cv2.line(mask, center, (end_x, end_y), 255, 3)
    
    def _draw_triangle_line(self, mask: np.ndarray, center: Tuple[int, int], size: int, param: float):
        """Draw a triangular line pattern."""
        # Create triangular path
        height = int(size * param)
        points = np.array([
            [center[0] - size//2, center[1]],
            [center[0], center[1] - height],
            [center[0] + size//2, center[1]]
        ], dtype=np.int32)
        cv2.polylines(mask, [points], False, 255, 3)
    
    def _draw_square_line(self, mask: np.ndarray, center: Tuple[int, int], size: int, param: float):
        """Draw a square line pattern."""
        half_size = int(size * param / 2)
        points = np.array([
            [center[0] - half_size, center[1] - half_size],
            [center[0] + half_size, center[1] - half_size],
            [center[0] + half_size, center[1] + half_size],
            [center[0] - half_size, center[1] + half_size]
        ], dtype=np.int32)
        cv2.polylines(mask, [points], True, 255, 3)
    
    def _draw_circle_line(self, mask: np.ndarray, center: Tuple[int, int], size: int, param: float):
        """Draw a circular line pattern."""
        radius = int(size * param)
        cv2.circle(mask, center, radius, 255, 3)
    
    def _draw_zigzag_line(self, mask: np.ndarray, center: Tuple[int, int], size: int, param: float):
        """Draw a zigzag line pattern."""
        # Create zigzag pattern - points based on param
        num_points = max(3, int(5 * param + 2))  # Scale points with param
        points = []
        for i in range(num_points):
            x = center[0] + int((i - num_points//2) * size / num_points)
            y = center[1] + int(size * param * ((-1)**i) * 0.5)
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [points], False, 255, 3)
    
    def _draw_normal_arc(self, mask: np.ndarray, center: Tuple[int, int], size: int, arc_param: float, param: float):
        """Draw a normal arc."""
        radius = int(size * param)
        start_angle = int(0)
        end_angle = int(arc_param * 360)
        cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, 3)
    
    def _draw_triangle_arc(self, mask: np.ndarray, center: Tuple[int, int], size: int, arc_param: float, param: float):
        """Draw a triangular arc pattern."""
        # Combine arc with triangular elements
        radius = int(size * param)
        start_angle = int(0)
        end_angle = int(arc_param * 360)
        cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, 3)
        
        # Add triangular markers at arc endpoints - size based on parameters
        triangle_size = max(3, int(size * param * 0.1))  # Scale with param
        for angle in [start_angle, end_angle]:
            angle_rad = np.radians(angle)
            arc_x = int(center[0] + radius * np.cos(angle_rad))
            arc_y = int(center[1] + radius * np.sin(angle_rad))
            triangle_points = np.array([
                [arc_x, arc_y - triangle_size],
                [arc_x - triangle_size, arc_y + triangle_size],
                [arc_x + triangle_size, arc_y + triangle_size]
            ], dtype=np.int32)
            cv2.fillPoly(mask, [triangle_points], 255)
    
    def _draw_square_arc(self, mask: np.ndarray, center: Tuple[int, int], size: int, arc_param: float, param: float):
        """Draw a square arc pattern."""
        radius = int(size * param)
        start_angle = int(0)
        end_angle = int(arc_param * 360)
        cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, 3)
        
        # Add square markers at arc endpoints - size based on parameters
        square_size = max(2, int(size * param * 0.08))  # Scale with param
        for angle in [start_angle, end_angle]:
            angle_rad = np.radians(angle)
            arc_x = int(center[0] + radius * np.cos(angle_rad))
            arc_y = int(center[1] + radius * np.sin(angle_rad))
            cv2.rectangle(mask,
                         (arc_x - square_size, arc_y - square_size),
                         (arc_x + square_size, arc_y + square_size),
                         255, -1)
    
    def _draw_circle_arc(self, mask: np.ndarray, center: Tuple[int, int], size: int, arc_param: float, param: float):
        """Draw a circular arc pattern."""
        radius = int(size * param)
        start_angle = int(0)
        end_angle = int(arc_param * 360)
        cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, 3)
        
        # Add circular markers at arc endpoints - size based on parameters
        circle_radius = max(2, int(size * param * 0.06))  # Scale with param
        for angle in [start_angle, end_angle]:
            angle_rad = np.radians(angle)
            arc_x = int(center[0] + radius * np.cos(angle_rad))
            arc_y = int(center[1] + radius * np.sin(angle_rad))
            cv2.circle(mask, (arc_x, arc_y), circle_radius, 255, -1)
    
    def _draw_zigzag_arc(self, mask: np.ndarray, center: Tuple[int, int], size: int, arc_param: float, param: float):
        """Draw a zigzag arc pattern."""
        radius = int(size * param)
        start_angle = 0
        end_angle = arc_param * 360
        
        # Create zigzag along arc path - segments based on arc_param
        num_segments = max(4, int(8 * arc_param))  # Scale segments with arc_param
        points = []
        for i in range(num_segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_segments
            angle_rad = np.radians(angle)
            
            # Alternate between inner and outer radius for zigzag effect
            zigzag_offset = max(3, int(size * param * 0.1))  # Scale with param
            current_radius = radius + (zigzag_offset if i % 2 == 0 else -zigzag_offset)
            x = int(center[0] + current_radius * np.cos(angle_rad))
            y = int(center[1] + current_radius * np.sin(angle_rad))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [points], False, 255, 3)
    
    def generate_mask_from_actions(self, action_commands: List[str], problem_id: str = None) -> np.ndarray:
        """
        Generate a binary mask from a list of action commands.
        Enhanced to detect and handle multiple objects and stroke holes properly.
        """
        try:
            # CRITICAL FIX: First flatten any nested command structures
            original_count = len(action_commands) if hasattr(action_commands, '__len__') else 0
            flat_commands = self._flatten_action_commands(action_commands)
            
            if len(flat_commands) != original_count:
                logging.info(f"[MASKGEN] Flattened {original_count} raw commands to {len(flat_commands)} flat commands")
                action_commands = flat_commands
            
            # CRITICAL FIX: Detect multiple objects and holes in action commands
            object_groups = self._detect_multiple_objects(action_commands)
            
            if len(object_groups) > 1:
                logging.debug(f"Detected {len(object_groups)} separate objects in action commands")
                return self._render_multiple_objects_to_mask(object_groups, problem_id)
            else:
                # Single object - use standard parsing
                return self._render_single_object_to_mask(action_commands, problem_id)
                
        except Exception as e:
            logging.warning(f"generate_mask_from_actions: Failed to parse/render action commands: {e}")
            return self._generate_fallback_mask()

    def _generate_fallback_mask(self) -> np.ndarray:
        """Generate a simple fallback mask when action parsing fails."""
        mask = np.zeros(self.image_size, dtype=np.uint8)
        # Create a simple shape in the center
        center_x, center_y = self.image_size[1] // 2, self.image_size[0] // 2
        cv2.circle(mask, (center_x, center_y), min(self.image_size) // 4, 255, -1)
        return mask

    def _flatten_action_commands(self, commands) -> List[str]:
        """Flatten nested action command structures."""
        if not isinstance(commands, list):
            return [str(commands)] if commands else []
            
        flattened = []
        for cmd in commands:
            if isinstance(cmd, list):
                # Recursively flatten nested lists
                flattened.extend(self._flatten_action_commands(cmd))
            else:
                flattened.append(str(cmd))
                
        return flattened

    def _detect_multiple_objects(self, action_commands: List[str]) -> List[List[str]]:
        """
        Detect multiple objects and holes in action commands.
        Enhanced logic with improved nested command handling.
        """
        if not action_commands:
            return []
        
        # CRITICAL FIX: First flatten any nested command structures
        flat_commands = self._flatten_action_commands(action_commands)
        
        if len(flat_commands) != len(action_commands):
            logging.info(f"[MULTIOBJ] Flattened {len(action_commands)} raw commands to {len(flat_commands)} flat commands")
        
        # Look for patterns that indicate multiple objects:
        # 1. Multiple 'start_' commands (new object starts)
        # 2. Large gaps in coordinate space
        # 3. Different shape types
        
        object_groups = []
        current_group = []
        last_coords = None
        
        for cmd in flat_commands:
            cmd_str = str(cmd)
            cmd_lower = cmd_str.lower()
            
            # Extract coordinates from command
            coords = self._extract_coordinates(cmd_str)
            
            # Check for new object indicators
            is_new_object = (
                cmd_lower.startswith('start_') or
                cmd_lower.startswith('move_to') or
                (coords and last_coords and self._has_large_gap(coords, last_coords)) or
                (current_group and self._is_different_shape_type(cmd_str, str(current_group[0]) if current_group else ""))
            )
            
            if is_new_object and current_group:
                object_groups.append(current_group)
                current_group = []
            
            current_group.append(cmd)
            if coords:
                last_coords = coords
        
        # Add the last group
        if current_group:
            object_groups.append(current_group)
        
        # If no clear separation found, treat as single object
        if len(object_groups) <= 1:
            return [flat_commands] if flat_commands else []
            
        return object_groups

    def _extract_coordinates(self, command: str) -> Tuple[float, float]:
        """Extract x, y coordinates from action command."""
        try:
            # Handle both string and list inputs
            if isinstance(command, list):
                command_str = str(command)
            else:
                command_str = str(command)
            
            # Look for patterns like "Circle(x, y, r)" or "Rectangle(x, y, w, h)"
            import re
            coord_pattern = r'[(-]?[\d.]+[)]?'
            matches = re.findall(coord_pattern, command_str)
            if len(matches) >= 2:
                x = float(matches[0].strip('()'))
                y = float(matches[1].strip('()'))
                return (x, y)
        except (ValueError, IndexError):
            pass
        return None

    def _has_large_gap(self, coords1: Tuple[float, float], coords2: Tuple[float, float], threshold: float = 100.0) -> bool:
        """Check if there's a large spatial gap between two coordinate pairs."""
        if not coords1 or not coords2:
            return False
        distance = ((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)**0.5
        return distance > threshold

    def _is_different_shape_type(self, cmd1: str, cmd2: str) -> bool:
        """Check if two commands represent different shape types."""
        # Ensure both inputs are strings
        cmd1_str = str(cmd1)
        cmd2_str = str(cmd2)
        
        cmd1_type = cmd1_str.split('(')[0].lower() if '(' in cmd1_str else cmd1_str.lower()
        cmd2_type = cmd2_str.split('(')[0].lower() if '(' in cmd2_str else cmd2_str.lower()
        
        shape_types = ['circle', 'rectangle', 'line', 'arc', 'polygon']
        cmd1_shape = next((s for s in shape_types if s in cmd1_type), 'unknown')
        cmd2_shape = next((s for s in shape_types if s in cmd2_type), 'unknown')
        
        return cmd1_shape != cmd2_shape and cmd1_shape != 'unknown' and cmd2_shape != 'unknown'

    def _render_multiple_objects_to_mask(self, object_groups: List[List[str]], problem_id: str = None) -> np.ndarray:
        """
        Render multiple object groups to a single mask with proper layering.
        """
        final_mask = np.zeros(self.image_size, dtype=np.uint8)
        
        for i, obj_commands in enumerate(object_groups):
            try:
                # Render each object group separately
                obj_mask = self._render_single_object_to_mask(obj_commands, problem_id)
                
                # Check if this is a hole pattern (should subtract from existing mask)
                if self._is_hole_pattern(obj_commands):
                    # Subtract hole from existing mask
                    final_mask = np.where(obj_mask > 0, 0, final_mask)
                else:
                    # Add object to mask
                    final_mask = np.maximum(final_mask, obj_mask)
                    
            except Exception as e:
                logging.warning(f"Failed to render object group {i}: {e}")
                continue
                
        return final_mask

    def _render_single_object_to_mask(self, action_commands: List[str], problem_id: str = None) -> np.ndarray:
        """
        Render a single object (group of commands) to a mask.
        Supports both NVLabs parser and semantic command rendering.
        """
        try:
            # Try NVLabs parser first for compatibility
            if hasattr(self.action_parser, 'parse_action_commands'):
                try:
                    # Always provide a problem_id to avoid missing argument error
                    if problem_id:
                        parsed_data = self.action_parser.parse_action_commands(action_commands, problem_id)
                    else:
                        parsed_data = self.action_parser.parse_action_commands(action_commands, "test_problem_0000")
                    # Log the real action commands and parsed output
                    logging.info(f"[MASKGEN] Input action_commands: {action_commands}")
                    if parsed_data and hasattr(parsed_data, 'vertices'):
                        logging.info(f"[MASKGEN] Parsed vertices: {parsed_data.vertices}")
                        # Log min/max values of parsed vertices for out-of-bounds diagnosis
                        try:
                            verts = np.array(parsed_data.vertices)
                            if verts.size > 0:
                                min_x, min_y = np.min(verts, axis=0)
                                max_x, max_y = np.max(verts, axis=0)
                                logging.info(f"[MASKGEN] Vertices min: ({min_x:.2f}, {min_y:.2f}), max: ({max_x:.2f}, {max_y:.2f})")
                                # CRITICAL FIX: Use the approach that actually works
                                # Complex normalization creates all-white masks, so try simpler approach first
                                logging.info(f"[MASKGEN] Using simplified coordinate handling to avoid white mask issues")
                                
                                # Let the renderer handle coordinates as-is and rely on alternative rendering
                                # if needed - this prevents the destructive normalization
                                pass
                        except Exception as e:
                            logging.debug(f"Failed to log min/max vertices: {e}")
                    if parsed_data and hasattr(parsed_data, 'vertices') and parsed_data.vertices:
                        return self._render_vertices_to_mask(parsed_data.vertices)
                except Exception as e:
                    logging.debug(f"NVLabs parser failed: {e}")
            # Use semantic command rendering for Bongard action programs
            logging.info(f"[MASKGEN] Semantic command input: {action_commands}")
            mask = self._render_semantic_commands_to_mask(action_commands)
            logging.info(f"[MASKGEN] Semantic mask nonzero pixels: {np.count_nonzero(mask)}")
            return mask
        except Exception as e:
            logging.warning(f"Single object rendering failed: {e}")
            return self._generate_fallback_mask()

    def _is_hole_pattern(self, commands: List[str]) -> bool:
        """
        Detect if a set of commands represents a hole (negative space).
        """
        # Look for indicators that suggest this is a hole:
        # 1. Commands with 'hole', 'cut', 'subtract' keywords
        # 2. Smaller objects inside larger ones
        # 3. Circular patterns that might be holes
        
        # Handle mixed list/string formats
        command_texts = []
        for cmd in commands:
            if isinstance(cmd, list):
                command_texts.append(str(cmd))
            else:
                command_texts.append(str(cmd))
        
        command_text = ' '.join(command_texts).lower()
        hole_keywords = ['hole', 'cut', 'subtract', 'inner', 'void', 'negative']
        
        return any(keyword in command_text for keyword in hole_keywords)


# Use MaskRefiner for all mask refinement and QA
from src.bongard_augmentor.refiners import MaskRefiner

# --- SkeletonProcessor Class ---
class SkeletonProcessor:
    """
    Advanced skeleton processing for line art and geometric shapes.
    Handles branch detection, pruning, and skeleton-based analysis.
    """
    def __init__(self, min_branch_length: int = 10):
        """
        Initializes the SkeletonProcessor.

        Args:
            min_branch_length (int): Minimum length of a branch to be considered
                                      significant; shorter branches will be pruned.
        """
        self.min_branch_length = min_branch_length

    def _get_branch_points(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detects branch points in a skeletonized image.
        A branch point is a pixel with more than two neighbors in a 8-connected window.

        Args:
            skeleton (np.ndarray): A binary skeleton image (2D array, values 0 or 255).

        Returns:
            List[Tuple[int, int]]: A list of (row, column) coordinates of branch points.
        """
        branch_points = []
        # Create a kernel for counting 8-connected neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1], # Center is 0 to exclude self from count
                           [1, 1, 1]], dtype=np.uint8)

        # Convolve the skeleton (normalized to 1s and 0s)
        # BORDER_CONSTANT with value 0 ensures pixels outside are treated as non-existent
        convolved_neighbors = cv2.filter2D((skeleton > 0).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # Branch points are skeleton pixels with 3 or more neighbors
        # (convolved_neighbors >= 3) and are part of the skeleton (skeleton > 0)
        branch_point_coords = np.argwhere((skeleton > 0) & (convolved_neighbors >= 3))

        return [tuple(pt) for pt in branch_point_coords]

    def process_skeleton(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Extract and process skeleton from a binary mask.
        
        Args:
            mask (np.ndarray): Binary mask to skeletonize
            
        Returns:
            Tuple[np.ndarray, Dict]: Processed skeleton and metadata
        """
        try:
            # Ensure binary mask
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Extract skeleton
            skeleton = ski_morphology.skeletonize(binary_mask)
            skeleton = (skeleton * 255).astype(np.uint8)
            
            # Get branch points
            branch_points = self._get_branch_points(skeleton)
            
            # Basic processing metadata
            metadata = {
                'branch_points': len(branch_points),
                'skeleton_pixels': np.sum(skeleton > 0)
            }
            
            return skeleton, metadata
            
        except Exception as e:
            logging.warning(f"Skeleton processing failed: {e}")
            return np.zeros_like(mask), {}

    def skeleton_aware_refinement(self, mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        """
        Refines a mask using the corresponding skeleton structure.
        
        Args:
            mask (np.ndarray): The original binary mask (uint8).
            skeleton (np.ndarray): The processed skeleton of the mask (uint8).

        Returns:
            np.ndarray: The refined binary mask (uint8).
        """
        try:
            # Combine the original mask with the skeleton to fill any gaps along the skeleton path
            refined = cv2.bitwise_or(mask, skeleton)

            # Apply a small dilation to further connect close components, if necessary
            # The kernel size should be small to avoid excessive thickening
            dilation_kernel = np.ones((3, 3), np.uint8)
            refined = cv2.dilate(refined, dilation_kernel, iterations=1)

            # Fill any remaining holes within the now more connected regions
            filled = ndi.binary_fill_holes(refined > 0).astype(np.uint8) * 255
            logging.info("Skeleton-aware refinement applied: mask combined with skeleton, dilated, and holes filled.")
            return filled
        except Exception as e:
            logging.error(f"Skeleton-aware refinement failed: {e}. Returning original mask.")
            return mask


        cv2.circle(mask, center, radius, 255, -1)
        return mask
    
class HybridAugmentationPipeline:
    def visualize_mask_and_actions(self, mask: np.ndarray, action_commands: List[str], image_id: str = None, show: bool = True, save_path: str = None):
        """
        Visualize the generated mask alongside the action commands for diagnostics.
        Displays the mask and overlays action commands as text.
        Optionally saves the visualization to a file.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(mask, cmap='gray')
        ax.set_title(f"Mask Visualization: {image_id if image_id else ''}")
        ax.axis('off')
        # Overlay action commands as text
        if action_commands:
            txt = '\n'.join([str(cmd) for cmd in action_commands])
            ax.text(0.01, 0.99, txt, fontsize=7, color='yellow', va='bottom', ha='left', transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, pad=2))
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
    """
    Action-based augmentation pipeline that processes action programs
    instead of real images, similar to logo_to_shape.py approach.
    """

    def __init__(self, config: Dict = None):
        """
        Initializes the pipeline with action-based processing.
        """
        self.config = config or {}
        
        # Initialize data loader (same pattern as logo_to_shape.py)
        self.data_loader = DataLoaderWrapper()
        
        # Initialize action-based mask generator
        # CRITICAL FIX: Use consistent 512x512 size and coordinate range everywhere
        image_size = (512, 512)  # Force 512x512 to match real dataset
        self.mask_generator = ActionMaskGenerator(image_size=image_size)
        
        # Initialize MaskRefiner for post-processing with valid parameters only
        ref_cfg = self.config.get('refinement', {}) or {}
        valid_refiner_args = ['contour_approx_factor', 'min_component_size', 'closing_kernel_size', 'opening_kernel_size', 'sam_model_types', 'sam_cache_dir']
        filtered_ref_cfg = {k: v for k, v in ref_cfg.items() if k in valid_refiner_args}
        
        # Note: Canvas size preservation needs to be handled in the rendering methods instead
        
        self.mask_refiner = MaskRefiner(**filtered_ref_cfg)
        
        # Initialize SkeletonProcessor for optional refinement
        self.skeleton_processor = SkeletonProcessor(**(self.config.get('skeleton', {}) or {}))

    def process_action_commands(self, action_commands: List[str], image_id: str = None) -> np.ndarray:
        """
        Process action commands to generate a binary mask.
        This replaces the complex classical CV processing with action-based generation.
        """
        try:
            # Generate mask from action commands
            mask = self.mask_generator.generate_mask_from_actions(action_commands)
            
            # Optional post-processing for quality improvement
            if self.config.get('enable_post_processing', False):
                mask = self._post_process_mask(mask)
            
            # --- Overlay diagnostics integration ---
            # Try to find real and parsed images for this image_id and run overlay diagnostics
            import os, cv2, numpy as np
            if image_id:
                mask_path = f"data/{image_id}_mask.png"
                real_path = f"data/{image_id}_real.png"
                parsed_path = f"data/{image_id}_parsed.png"
                if os.path.exists(mask_path) and os.path.exists(real_path) and os.path.exists(parsed_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
                    parsed_img = cv2.imread(parsed_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None and real_img is not None and parsed_img is not None:
                        # Save overlay: mask vs real
                        overlay_mask_real = cv2.addWeighted(mask_img, 0.5, real_img, 0.5, 0)
                        overlay_path_mask_real = f"data/{image_id}_mask_vs_real_overlay.png"
                        cv2.imwrite(overlay_path_mask_real, overlay_mask_real)
                        logging.info(f"[DIAGNOSTICS] Saved mask vs real overlay for {image_id} at {overlay_path_mask_real}")

                        # Save overlay: mask vs parsed
                        overlay_mask_parsed = cv2.addWeighted(mask_img, 0.5, parsed_img, 0.5, 0)
                        overlay_path_mask_parsed = f"data/{image_id}_mask_vs_parsed_overlay.png"
                        cv2.imwrite(overlay_path_mask_parsed, overlay_mask_parsed)
                        logging.info(f"[DIAGNOSTICS] Saved mask vs parsed overlay for {image_id} at {overlay_path_mask_parsed}")

                        # Save overlay: real vs parsed
                        overlay_real_parsed = cv2.addWeighted(real_img, 0.5, parsed_img, 0.5, 0)
                        overlay_path_real_parsed = f"data/{image_id}_real_vs_parsed_overlay.png"
                        cv2.imwrite(overlay_path_real_parsed, overlay_real_parsed)
                        logging.info(f"[DIAGNOSTICS] Saved real vs parsed overlay for {image_id} at {overlay_path_real_parsed}")

                        # Log pixel stats
                        mask_sum = int(np.sum(mask_img > 0))
                        real_sum = int(np.sum(real_img > 0))
                        parsed_sum = int(np.sum(parsed_img > 0))
                        logging.info(f"[DIAGNOSTICS] Pixel stats for {image_id}: mask={mask_sum}, real={real_sum}, parsed={parsed_sum}")
                    else:
                        logging.warning(f"[DIAGNOSTICS] Could not load one or more images for overlay diagnostics: {image_id}")
                else:
                    logging.warning(f"[DIAGNOSTICS] Overlay image files not found for {image_id}")
            return mask
        except Exception as e:
            logging.error(f"Failed to process action commands for {image_id}: {e}")
            # Return fallback mask
            return self.mask_generator._generate_fallback_mask()

    def _render_parsed_image_from_vertices(self, vertices: List[Tuple[float, float]], image_id: str = None, parsed_program=None) -> Optional[np.ndarray]:
        """
        Render a high-quality parsed image from vertices to closely match real dataset images.
        Enhanced to handle multiple distinct objects and complex stroke patterns.
        """
        try:
            if not vertices or len(vertices) < 2:
                logging.warning(f"Insufficient vertices to render image for {image_id}: {len(vertices) if vertices else 0} vertices")
                return None
            
            # Use higher resolution for better quality, then downsample
            base_size = self.mask_generator.image_size
            scale_factor = 8  # 8x supersampling for high-quality anti-aliasing
            high_res_size = (base_size[0] * scale_factor, base_size[1] * scale_factor)
            
            # Create high-resolution canvas
            parsed_image = np.zeros((*high_res_size, 3), dtype=np.uint8)
            
            # Transform vertices to high-resolution pixel space
            center_x = high_res_size[1] // 2
            center_y = high_res_size[0] // 2
            
            # Enhanced rendering strategy: Handle object boundaries if available
            if parsed_program and hasattr(parsed_program, 'object_boundaries'):
                logging.debug(f"RENDERER: Found {len(parsed_program.object_boundaries)} distinct objects for {image_id}")
                self._render_multiple_objects(parsed_image, parsed_program.object_boundaries, scale_factor, center_x, center_y)
            else:
                # Fallback: Render as single connected path
                logging.debug(f"RENDERER: Rendering as single path for {image_id}")
                self._render_single_path(parsed_image, vertices, scale_factor, center_x, center_y)
            
            # Convert to grayscale
            parsed_image_gray = cv2.cvtColor(parsed_image, cv2.COLOR_RGB2GRAY)
            
            # Downsample with high-quality anti-aliasing
            final_image = cv2.resize(parsed_image_gray, base_size, interpolation=cv2.INTER_AREA)
            
            # Enhanced post-processing for realism
            final_image = self._apply_realistic_post_processing(final_image)
            
            logging.debug(f"RENDERER: Enhanced rendering for {image_id} completed, final size: {final_image.shape}")
            
            return final_image
            
        except Exception as e:
            logging.error(f"Failed to render parsed image from vertices for {image_id}: {e}")
            return None
    
    def _render_multiple_objects(self, canvas: np.ndarray, object_boundaries: List[Dict], 
                                scale_factor: int, center_x: int, center_y: int):
        """Render multiple distinct objects with proper separation and styling"""
        
        # Color scheme for different objects
        object_colors = [
            (255, 255, 255),  # White for primary object
            (200, 200, 200),  # Light gray for secondary
            (180, 180, 180),  # Medium gray for tertiary
            (160, 160, 160)   # Darker gray for additional objects
        ]
        
        for obj_idx, obj_info in enumerate(object_boundaries):
            obj_vertices = obj_info['vertices']
            if not obj_vertices or len(obj_vertices) < 2:
                continue
            
            # Transform object vertices to high-resolution pixel space
            obj_points = []
            for vertex in obj_vertices:
                if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                    pixel_x = center_x + int(vertex[0] * scale_factor)
                    pixel_y = center_y - int(vertex[1] * scale_factor)  # Flip Y axis
                    
                    # Clamp to bounds
                    pixel_x = max(0, min(canvas.shape[1] - 1, pixel_x))
                    pixel_y = max(0, min(canvas.shape[0] - 1, pixel_y))
                    obj_points.append([pixel_x, pixel_y])
            
            if len(obj_points) < 2:
                continue
            
            # Analyze object characteristics
            obj_strokes = obj_info.get('strokes', [])
            shape_modifiers = obj_info.get('shape_modifiers', [])
            stroke_types = obj_info.get('stroke_types', [])
            
            # Choose rendering style based on object characteristics
            color = object_colors[min(obj_idx, len(object_colors) - 1)]
            self._render_object_with_style(canvas, obj_points, obj_strokes, shape_modifiers, 
                                         stroke_types, color, scale_factor)
            
            logging.debug(f"RENDERER: Rendered object {obj_idx+1} with {len(obj_points)} points, "
                         f"modifiers: {set(shape_modifiers)}")
    
    def _render_object_with_style(self, canvas: np.ndarray, points: List[List[int]], 
                                 strokes: List, shape_modifiers: List[str], 
                                 stroke_types: List[str], color: Tuple[int, int, int], 
                                 scale_factor: int):
        """Render a single object with appropriate styling based on its characteristics"""
        
        if len(points) < 2:
            return
        
        points_array = np.array(points, dtype=np.int32)
        
        # Determine rendering strategy based on shape modifiers and stroke types
        has_complex_shapes = any(mod in ['square', 'triangle', 'circle', 'zigzag'] for mod in shape_modifiers)
        has_arcs = 'arc' in stroke_types
        is_closed_shape = self._is_closed_shape(points)
        
        line_thickness = max(3, 4 * scale_factor)
        
        if is_closed_shape and len(points) >= 3:
            # Render as filled shape with outline
            fill_color = tuple(int(c * 0.7) for c in color)  # Darker fill
            cv2.fillPoly(canvas, [points_array], fill_color)
            cv2.polylines(canvas, [points_array], True, color, line_thickness, cv2.LINE_AA)
            
            # Add internal details for complex shapes
            if has_complex_shapes:
                self._add_internal_details(canvas, points_array, shape_modifiers, color, scale_factor)
        else:
            # Render as stroke/line art
            for i in range(len(points) - 1):
                cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), 
                        color, line_thickness, cv2.LINE_AA)
            
            # Add shape-specific decorations
            if has_complex_shapes:
                self._add_stroke_decorations(canvas, points, shape_modifiers, color, scale_factor)
            
            # Add endpoint markers
            endpoint_radius = max(4, 3 * scale_factor)
            cv2.circle(canvas, tuple(points[0]), endpoint_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(points[-1]), endpoint_radius, color, -1, cv2.LINE_AA)
    
    def _add_internal_details(self, canvas: np.ndarray, points: np.ndarray, 
                             shape_modifiers: List[str], color: Tuple[int, int, int], 
                             scale_factor: int):
        """Add internal details to filled shapes based on their modifiers"""
        
        detail_color = tuple(min(255, int(c * 1.3)) for c in color)  # Lighter detail color
        detail_thickness = max(1, scale_factor // 2)
        
        if 'square' in shape_modifiers:
            # Add grid pattern
            bbox = cv2.boundingRect(points)
            for i in range(1, 4):
                x = bbox[0] + i * bbox[2] // 4
                cv2.line(canvas, (x, bbox[1]), (x, bbox[1] + bbox[3]), 
                        detail_color, detail_thickness, cv2.LINE_AA)
                y = bbox[1] + i * bbox[3] // 4
                cv2.line(canvas, (bbox[0], y), (bbox[0] + bbox[2], y), 
                        detail_color, detail_thickness, cv2.LINE_AA)
        
        elif 'triangle' in shape_modifiers:
            # Add triangular internal lines
            center = np.mean(points, axis=0).astype(int)
            for point in points[::2]:  # Every other point
                cv2.line(canvas, tuple(center), tuple(point), 
                        detail_color, detail_thickness, cv2.LINE_AA)
        
        elif 'circle' in shape_modifiers:
            # Add concentric circles
            center = np.mean(points, axis=0).astype(int)
            radius = int(np.mean([np.linalg.norm(p - center) for p in points]) * 0.7)
            for r in range(radius // 3, radius, radius // 3):
                cv2.circle(canvas, tuple(center), r, detail_color, detail_thickness, cv2.LINE_AA)
    
    def _add_stroke_decorations(self, canvas: np.ndarray, points: List[List[int]], 
                               shape_modifiers: List[str], color: Tuple[int, int, int], 
                               scale_factor: int):
        """Add decorative elements to stroke-based objects"""
        
        decoration_radius = max(2, 2 * scale_factor)
        
        if 'zigzag' in shape_modifiers:
            # Add zigzag markers at vertices
            for point in points[::2]:  # Every other point
                cv2.circle(canvas, tuple(point), decoration_radius, color, -1, cv2.LINE_AA)
        
        elif 'square' in shape_modifiers:
            # Add square markers
            for point in points[::2]:
                square_size = decoration_radius
                square_points = np.array([
                    [point[0] - square_size, point[1] - square_size],
                    [point[0] + square_size, point[1] - square_size],
                    [point[0] + square_size, point[1] + square_size],
                    [point[0] - square_size, point[1] + square_size]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [square_points], color)
            # Add square markers
            for point in points[::3]:  # Every third point
                cv2.rectangle(canvas, 
                            (point[0] - decoration_radius, point[1] - decoration_radius),
                            (point[0] + decoration_radius, point[1] + decoration_radius),
                            color, -1)
        
        elif 'triangle' in shape_modifiers:
            # Add triangle markers
            for point in points[::3]:
                triangle_points = np.array([
                    [point[0], point[1] - decoration_radius],
                    [point[0] - decoration_radius, point[1] + decoration_radius],
                    [point[0] + decoration_radius, point[1] + decoration_radius]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [triangle_points], color)
                triangle_points = np.array([
                    [point[0], point[1] - decoration_radius],
                    [point[0] - decoration_radius, point[1] + decoration_radius],
                    [point[0] + decoration_radius, point[1] + decoration_radius]
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [triangle_points], color)
    
    def _is_closed_shape(self, points: List[List[int]]) -> bool:
        """Determine if a set of points forms a closed shape"""
        if len(points) < 3:
            return False
        
        start_point = points[0]
        end_point = points[-1]
        distance = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)
        
        # Consider closed if start and end points are close
        return distance < 20  # Threshold for "close enough"
    
    def _render_single_path(self, canvas: np.ndarray, vertices: List[Tuple[float, float]], 
                           scale_factor: int, center_x: int, center_y: int):
        """Fallback rendering for single connected path"""
        
        points = []
        for vertex in vertices:
            if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
                pixel_x = center_x + int(vertex[0] * scale_factor)
                pixel_y = center_y - int(vertex[1] * scale_factor)  # Flip Y axis
                
                # Clamp to bounds
                pixel_x = max(0, min(canvas.shape[1] - 1, pixel_x))
                pixel_y = max(0, min(canvas.shape[0] - 1, pixel_y))
                points.append([pixel_x, pixel_y])
        
        if len(points) < 2:
            return
        
        points_array = np.array(points, dtype=np.int32)
        line_thickness = max(3, 4 * scale_factor)
        color = (255, 255, 255)
        
        # Render as polyline
        cv2.polylines(canvas, [points_array], False, color, line_thickness, cv2.LINE_AA)
        
        # Add endpoint markers
        endpoint_radius = max(4, 3 * scale_factor)
        cv2.circle(canvas, tuple(points[0]), endpoint_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), endpoint_radius, color, -1, cv2.LINE_AA)
    
    def _apply_realistic_post_processing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply realistic post-processing to match real dataset characteristics.
        Includes Gaussian blur, contrast enhancement, and removal of isolated pixels.
        """
        import cv2
        processed = cv2.GaussianBlur(image, (3, 3), 0.8)
        # Enhance contrast for non-background pixels
        threshold_mask = processed > 10
        if np.any(threshold_mask):
            # FIX: Apply histogram equalization to the entire image instead of indexing assignment
            # This avoids the NumPy boolean indexing assignment error
            processed_uint8 = processed.astype(np.uint8)
            enhanced_full = cv2.equalizeHist(processed_uint8)
            # Only apply enhancement where threshold_mask is True
            processed = np.where(threshold_mask, enhanced_full, processed).astype(processed.dtype)
        # Remove isolated pixels (morphological opening)
        if np.any(processed > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        return processed

    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Optional post-processing to clean up the generated mask.
        Uses MaskRefiner for robust QA and fallback logic.
        FIXED: Preserves original canvas size to prevent cropping.
        """
        try:
            # Store original mask shape to prevent cropping
            original_shape = mask.shape
            from src.bongard_augmentor.refiners import MaskRefiner
            refiner = MaskRefiner()
            # Validate mask quality and apply fallback if needed
            validated_mask, quality, metrics = refiner.validate_mask_quality_with_confidence(mask, mask, prediction_scores=[1.0])
            if quality < 0.5:
                processed_mask = refiner.ensemble_fallback_stack(mask, mask)
            else:
                processed_mask = validated_mask
            # Always restore to original canvas size (center and pad/crop)
            ph, pw = processed_mask.shape
            oh, ow = original_shape
            logging.info(f"[POSTPROCESS] Restoring mask: original_shape=({oh},{ow}), processed_shape=({ph},{pw})")
            restored_mask = np.zeros(original_shape, dtype=mask.dtype)
            crop_h = min(ph, oh)
            crop_w = min(pw, ow)
            start_h = max(0, (oh - crop_h) // 2)
            start_w = max(0, (ow - crop_w) // 2)
            src_start_h = max(0, (ph - crop_h) // 2)
            src_start_w = max(0, (pw - crop_w) // 2)
            logging.info(f"[POSTPROCESS] Paste region: dst=({start_h}:{start_h+crop_h}, {start_w}:{start_w+crop_w}), src=({src_start_h}:{src_start_h+crop_h}, {src_start_w}:{src_start_w+crop_w})")
            restored_mask[start_h:start_h+crop_h, start_w:start_w+crop_w] = processed_mask[src_start_h:src_start_h+crop_h, src_start_w:src_start_w+crop_w]
            if processed_mask.shape != original_shape:
                logging.info(f"[POSTPROCESS] Mask restored to original canvas size for output.")
            return restored_mask
        except Exception as e:
            logging.warning(f"Post-processing failed: {e}")
            return mask

    def load_derived_labels_from_actions(self, input_dir: str, problems_list: str = None, n_select: int = 50) -> list:
        """
        Load action programs and convert them to the same format as derived_labels.json
        to maintain compatibility with the existing pipeline.
        ENFORCES filtering by problems_list and n_select to prevent processing all entries.
        """
        try:
            from src.data_pipeline.data_loader import load_action_programs
            action_data = load_action_programs(input_dir)
            
            # CRITICAL: Apply filtering by problems_list first
            filtered_problems = set()
            if problems_list and os.path.exists(problems_list):
                logging.info(f"[FILTER] Loading problems list from: {problems_list}")
                with open(problems_list, 'r') as f:
                    filtered_problems = set(line.strip() for line in f if line.strip())
                logging.info(f"[FILTER] Found {len(filtered_problems)} problems in filter list")
            else:
                logging.warning(f"[FILTER] No problems list provided or file doesn't exist: {problems_list}")
                # If no filter provided, take first n_select problems
                filtered_problems = set(list(action_data.keys())[:n_select])
                logging.info(f"[FILTER] Using first {len(filtered_problems)} problems as fallback")
            
            # Filter and restructure as needed
            result = []
            processed_count = 0
            for problem_id, problem_data in action_data.items():
                # CRITICAL: Skip problems not in filter list
                if problem_id not in filtered_problems:
                    continue
                    
                # CRITICAL: Enforce n_select limit
                if processed_count >= n_select:
                    logging.info(f"[FILTER] Reached n_select limit of {n_select}, stopping")
                    break
                    
                # Handle both dict and list formats
                if isinstance(problem_data, dict):
                    pos_examples = problem_data.get('positive_examples', [])
                    neg_examples = problem_data.get('negative_examples', [])
                elif isinstance(problem_data, list) and len(problem_data) == 2:
                    pos_examples, neg_examples = problem_data
                else:
                    pos_examples, neg_examples = [], []
                    
                for i, action_commands in enumerate(pos_examples):
                    result.append({
                        'image_id': f'{problem_id}_pos_{i}',
                        'action_commands': action_commands,
                        'problem_id': problem_id,
                        'is_positive': True
                    })
                for i, action_commands in enumerate(neg_examples):
                    result.append({
                        'image_id': f'{problem_id}_neg_{i}',
                        'action_commands': action_commands,
                        'problem_id': problem_id,
                        'is_positive': False
                    })
                processed_count += 1
                
            logging.info(f"[FILTER] Processed {processed_count} problems, generated {len(result)} entries")
            return result
        except Exception as e:
            logging.error(f"Failed to load action programs: {e}")
            return []

    def run_pipeline(self):
        """
        Main entry point for action-based augmentation pipeline.
        
        Can work in two modes:
        1. Process existing derived_labels.json (from logo_to_shape.py output)
        2. Generate derived_labels from action programs directly
        
        Generates masks from action programs and saves results for scene graph builder compatibility.
        """
        # Get config values
        input_path = self.config.get('data', {}).get('input_path')
        output_path = self.config.get('data', {}).get('output_path')
        batch_size = self.config.get('processing', {}).get('batch_size', 8)
        inspection_dir = self.config.get('inspection_dir', None)
        
        # Check if we should load from action programs or derived_labels.json
        action_programs_dir = self.config.get('data', {}).get('action_programs_dir')
        problems_list = self.config.get('data', {}).get('problems_list')
        n_select = self.config.get('data', {}).get('n_select', 50)

        if not output_path:
            logging.critical("Output path not specified in config. Please check 'data.output_path'.")
            raise ValueError("Output path not specified.")

        # Load data - either from derived_labels.json or action programs
        if input_path and os.path.exists(input_path):
            # Mode 1: Process existing derived_labels.json
            logging.info(f"Loading derived labels from: {input_path}")
            with open(input_path, 'r') as f:
                derived_labels = json.load(f)
        elif action_programs_dir and os.path.exists(action_programs_dir):
            # Mode 2: Generate derived_labels from action programs
            logging.info(f"Loading action programs from: {action_programs_dir}")
            derived_labels = self.load_derived_labels_from_actions(
                action_programs_dir, problems_list, n_select
            )
        else:
            logging.critical("No valid input source found. Provide either 'data.input_path' (derived_labels.json) or 'data.action_programs_dir'.")
            raise ValueError("No valid input source found.")

        if not derived_labels:
            logging.warning("No data loaded. Exiting.")
            return

        # Output directory for masks
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Processing {len(derived_labels)} entries. Output masks will be saved to: {output_dir}")

        # Process each entry to generate masks from action commands
        output_records = []
        for entry in tqdm(derived_labels, desc="Generating masks from action programs", mininterval=0.5):
            try:
                record = dict(entry)
                action_commands = record.get('action_commands', [])
                image_id = record.get('image_id', 'unknown')
                problem_id = record.get('problem_id', None)
                # Log action commands and their hash for uniqueness check
                import hashlib
                cmds_str = str(action_commands)
                cmds_hash = hashlib.md5(cmds_str.encode('utf-8')).hexdigest()
                logging.info(f"[PIPELINE] Action commands for {image_id}: {action_commands}")
                logging.info(f"[PIPELINE] Action commands hash for {image_id}: {cmds_hash}")
                # Generate mask from action commands
                logging.info(f"[PIPELINE] Generating mask for {image_id} (problem_id={problem_id})")
                mask = self.mask_generator.generate_mask_from_actions(action_commands, problem_id=problem_id)
                logging.info(f"[PIPELINE] Mask shape before post-processing: {mask.shape}, dtype={mask.dtype}, sum={np.sum(mask)}")
                mask = self._post_process_mask(mask)
                logging.info(f"[PIPELINE] Mask shape after post-processing: {mask.shape}, dtype={mask.dtype}, sum={np.sum(mask)}")
                parsed_image = self.refine_and_render_from_mask(mask, image_id)
                base_name = image_id.replace('/', '_')
                output_dir = os.path.dirname(self.config.get('data', {}).get('output_path'))
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                parsed_path = os.path.join(output_dir, f"{base_name}_parsed.png")
                cv2.imwrite(mask_path, mask)
                logging.info(f"[PIPELINE] Saved mask for {image_id} at {mask_path}")
                if parsed_image is not None:
                    cv2.imwrite(parsed_path, parsed_image)
                    logging.info(f"[PIPELINE] Saved parsed image for {image_id} at {parsed_path}")
                # Save real image for direct visual comparison
                # Recursively search for real image in bd, ff, hd categories and save beside mask
                import glob
                from PIL import Image
                real_img_found = False
                categories = ['bd', 'ff', 'hd']
                # Extract idx from image_id for both pos and neg
                idx = None
                if '_pos_' in image_id:
                    parts = image_id.split('_pos_')
                    if len(parts) == 2:
                        idx = parts[1]
                elif '_neg_' in image_id:
                    parts = image_id.split('_neg_')
                    if len(parts) == 2:
                        idx = parts[1]
                if idx is not None:
                    for cat in categories:
                        # Try positive and negative folders
                        for label in ['1', '0']:
                            search_pattern = os.path.join('data', 'raw', 'ShapeBongard_V2', cat, 'images', problem_id, label, f"{idx}.*")
                            matches = glob.glob(search_pattern)
                            if matches:
                                real_image_path = matches[0]
                                try:
                                    pil_img = Image.open(real_image_path).convert('L')
                                    real_img = np.array(pil_img)
                                    real_img_save_path = os.path.join(output_dir, f"{base_name}_real.png")
                                    cv2.imwrite(real_img_save_path, real_img)
                                    record['real_image_path'] = real_img_save_path
                                    logging.info(f"Saved real image for {image_id} at {real_img_save_path}")
                                    real_img_found = True
                                    break
                                except Exception as e:
                                    logging.warning(f"Failed to open/save real image for {image_id} at {real_image_path}: {e}")
                        if real_img_found:
                            break
                if not real_img_found:
                    logging.warning(f"Real image not found for {image_id} in any category (bd, ff, hd) with idx {idx}")
                record['mask_path'] = mask_path
                record['parsed_image_path'] = parsed_path if parsed_image is not None else None
                output_records.append(record)
            except Exception as e:
            
                logging.error(f"Failed to process entry {entry.get('image_id', 'unknown')}: {e}")
                continue
        # Save results as pickle file (same format as before)
        logging.info(f"Saving {len(output_records)} augmented records to: {output_path}")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(output_records, f)
            logging.info("Augmentation pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Failed to save output file {output_path}: {e}")
            raise

    def _save_inspection_images(self, mask: np.ndarray, action_commands: List[str], image_id: str, inspection_dir: str):
        """Save inspection images for debugging, validation, and quality comparison with real image."""
        try:
            import matplotlib.pyplot as plt
            from src.data_pipeline.data_loader import robust_image_open, remap_path
            os.makedirs(inspection_dir, exist_ok=True)
            base_name = image_id.replace('/', '_')

            # Save the generated mask
            mask_save_path = os.path.join(inspection_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_save_path, mask)

            # Save action commands as text file for reference
            actions_save_path = os.path.join(inspection_dir, f"{base_name}_actions.txt")
            with open(actions_save_path, 'w') as f:
                f.write(f"Image ID: {image_id}\n")
                f.write("Action Commands:\n")
                for i, cmd in enumerate(action_commands):
                    f.write(f"{i+1}: {cmd}\n")

            # Try to fetch the real image using image_path logic
            real_image_path = None
            # Try to infer path from image_id (assumes format: problemid_pos_0 or problemid_neg_0)
            if '_pos_' in image_id:
                parts = image_id.split('_pos_')
                problem_id = parts[0]
                idx = parts[1]
                real_image_path = f"data/raw/ShapeBongard_V2/bd/images/{problem_id}/1/{idx}.png"
            elif '_neg_' in image_id:
                parts = image_id.split('_neg_')
                problem_id = parts[0]
                idx = parts[1]
                real_image_path = f"data/raw/ShapeBongard_V2/bd/images/{problem_id}/0/{idx}.png"

            real_img = None
            if real_image_path and os.path.exists(remap_path(real_image_path)):
                try:
                    pil_img = robust_image_open(real_image_path).convert('L')
                    real_img = np.array(pil_img)
                    real_img_save_path = os.path.join(inspection_dir, f"{base_name}_real.png")
                    cv2.imwrite(real_img_save_path, real_img)
                except Exception as e:
                    logging.warning(f"Failed to load/save real image for {image_id}: {e}")

            # Quality comparison: SSIM and IoU
            if real_img is not None:
                # Resize mask to match real image if needed
                if mask.shape != real_img.shape:
                    mask_resized = cv2.resize(mask, real_img.shape[::-1], interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask
                # Binarize both
                mask_bin = (mask_resized > 127).astype(np.uint8)
                real_bin = (real_img > 127).astype(np.uint8)
                # SSIM
                ssim_score = ssim(mask_bin, real_bin, data_range=1)
                # IoU
                intersection = np.logical_and(mask_bin, real_bin).sum()
                union = np.logical_or(mask_bin, real_bin).sum()
                iou_score = intersection / union if union > 0 else 0.0
                logging.info(f"QUALITY CHECK {image_id}: SSIM={ssim_score:.3f}, IoU={iou_score:.3f}")

                # Save comparison visualization
                comparison_img = np.zeros((mask_bin.shape[0], mask_bin.shape[1], 3), dtype=np.uint8)
                comparison_img[..., 0] = real_bin * 255  # Red: real
                comparison_img[..., 1] = mask_bin * 255  # Green: mask
                comparison_img[..., 2] = ((real_bin & mask_bin) * 255)  # Yellow: overlap
                comp_save_path = os.path.join(inspection_dir, f"{base_name}_comparison.png")
                cv2.imwrite(comp_save_path, comparison_img)

                # Also save a matplotlib figure with metrics
                fig, axs = plt.subplots(1, 3, figsize=(10, 4))
                axs[0].imshow(real_img, cmap='gray'); axs[0].set_title('Real Image'); axs[0].axis('off')
                axs[1].imshow(mask, cmap='gray'); axs[1].set_title('Generated Mask'); axs[1].axis('off')
                axs[2].imshow(comparison_img); axs[2].set_title(f'Comparison\nSSIM={ssim_score:.3f}, IoU={iou_score:.3f}')
                axs[2].axis('off')
                plt.tight_layout()
                fig_save_path = os.path.join(inspection_dir, f"{base_name}_qualitycheck.png")
                plt.savefig(fig_save_path)
                plt.close(fig)
            else:
                logging.warning(f"No real image found for {image_id}, skipping quality check.")
        except Exception as e:
            logging.warning(f"Failed to save inspection images for {image_id}: {e}")

    def overlay_mask_and_real_image(self, mask_img: np.ndarray, real_img: np.ndarray, alpha: float = 0.5, show_bbox: bool = True) -> np.ndarray:
        """
        Overlay mask and real image for diagnostics.
        Args:
            mask_img: Mask image as numpy array (grayscale or RGB)
            real_img: Real image as numpy array (grayscale or RGB)
            alpha: Blending factor for mask (0.0-1.0)
            show_bbox: If True, draw bounding boxes for mask and real image
        Returns:
            Blended RGB image for visualization
        """
        import cv2
        import numpy as np
        # Ensure both images are 3-channel RGB
        def to_rgb(img):
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return img.copy()
        mask_rgb = to_rgb(mask_img)
        real_rgb = to_rgb(real_img)
        # Resize if needed
        if mask_rgb.shape != real_rgb.shape:
            real_rgb = cv2.resize(real_rgb, (mask_rgb.shape[1], mask_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Blend images
        overlay = cv2.addWeighted(mask_rgb, alpha, real_rgb, 1 - alpha, 0)
        # Optionally draw bounding boxes
        if show_bbox:
            def get_bbox(img):
                # Find non-white pixels
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                coords = np.column_stack(np.where(gray < 250))
                if coords.size == 0:
                    return None
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                return (x0, y0, x1, y1)
            mask_bbox = get_bbox(mask_rgb)
            real_bbox = get_bbox(real_rgb)
            # Draw mask bbox in red
            if mask_bbox:
                cv2.rectangle(overlay, (mask_bbox[0], mask_bbox[1]), (mask_bbox[2], mask_bbox[3]), (255,0,0), 2)
            # Draw real bbox in green
            if real_bbox:
                cv2.rectangle(overlay, (real_bbox[0], real_bbox[1]), (real_bbox[2], real_bbox[3]), (0,255,0), 2)
        return overlay

    def refine_and_render_from_mask(self, mask: np.ndarray, image_id: str) -> Optional[np.ndarray]:
        """
        Refines a binary mask and renders it into a high-quality grayscale image,
        matching the style of the real dataset.
        """
        try:
            if np.sum(mask) == 0:
                logging.warning(f"Cannot refine an empty mask for {image_id}")
                return None

            # Apply realistic post-processing to the mask to create the parsed image
            rendered_image = self._apply_realistic_post_processing(mask)

            logging.info(f"Successfully refined and rendered mask for {image_id}")
            return rendered_image

        except Exception as e:
            logging.error(f"Failed to refine and render mask for {image_id}: {e}")
            return None


def diagnostic_overlay_analysis(sample_prefix: str = "data/bd_asymmetric_unbala_x_0000_pos_0"):
    """
    Loads mask, real, and parsed images for a sample and overlays them for diagnostics.
    Saves overlay results and prints basic stats.
    """
    import cv2
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    mask_path = f"{sample_prefix}_mask.png"
    real_path = f"{sample_prefix}_real.png"
    parsed_path = f"{sample_prefix}_parsed.png"
    # Load images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    parsed = cv2.imread(parsed_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or real is None or parsed is None:
        print(f"Failed to load one or more images: {mask_path}, {real_path}, {parsed_path}")
        return
    # Overlay mask and real
    overlay_mask_real = ActionMaskGenerator().overlay_mask_and_real_image(mask, real, alpha=0.5, show_bbox=True)
    overlay_mask_parsed = ActionMaskGenerator().overlay_mask_and_real_image(mask, parsed, alpha=0.5, show_bbox=True)
    overlay_real_parsed = ActionMaskGenerator().overlay_mask_and_real_image(real, parsed, alpha=0.5, show_bbox=True)
    # Save overlays
    cv2.imwrite(f"{sample_prefix}_overlay_mask_real.png", overlay_mask_real)
    cv2.imwrite(f"{sample_prefix}_overlay_mask_parsed.png", overlay_mask_parsed)
    cv2.imwrite(f"{sample_prefix}_overlay_real_parsed.png", overlay_real_parsed)
    # Show overlays and stats
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(overlay_mask_real, cv2.COLOR_BGR2RGB)); plt.title("Mask vs Real"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(cv2.cvtColor(overlay_mask_parsed, cv2.COLOR_BGR2RGB)); plt.title("Mask vs Parsed"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(cv2.cvtColor(overlay_real_parsed, cv2.COLOR_BGR2RGB)); plt.title("Real vs Parsed"); plt.axis('off')
    plt.tight_layout(); plt.show()
    # Print pixel sums for quick comparison
    print(f"Mask sum: {np.sum(mask)}, Real sum: {np.sum(real)}, Parsed sum: {np.sum(parsed)}")


# Legacy compatibility - create an alias for the old class name
HybridAugmentor = HybridAugmentationPipeline
