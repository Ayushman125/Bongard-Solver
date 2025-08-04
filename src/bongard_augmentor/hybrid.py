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
from src.data_pipeline.logo_parser import UnifiedActionParser

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
        
        return result

class ActionMaskGenerator:
    """Generates synthetic masks from action programs with corrected coordinate handling."""
    
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        # CRITICAL FIX: Use the corrected UnifiedActionParser with proper coordinate system
        self.action_parser = UnifiedActionParser()
        logging.debug(f"ActionMaskGenerator: Initialized with image_size={image_size}")
        logging.debug(f"ActionMaskGenerator: Parser scale_factor={self.action_parser.scale_factor}")
    
    def generate_mask_from_actions(self, action_commands: List[str]) -> np.ndarray:
        """Generate a binary mask from action commands with corrected coordinate handling."""
        try:
            logging.debug(f"ActionMaskGenerator: Processing {len(action_commands)} action commands")
            logging.debug(f"ActionMaskGenerator: Commands preview: {action_commands[:3] if action_commands else 'Empty'}")
            
            # Parse the action commands using the corrected UnifiedActionParser
            parsed_program = self.action_parser._parse_single_image(
                action_commands, 
                image_id="temp", 
                is_positive=True, 
                problem_id="temp"
            )
            
            # Create a blank mask
            mask = np.zeros(self.image_size, dtype=np.uint8)
            
            if parsed_program and parsed_program.vertices:
                logging.debug(f"ActionMaskGenerator: Got {len(parsed_program.vertices)} vertices from parser")
                
                # Analyze vertex distribution
                vertices = parsed_program.vertices
                if vertices:
                    x_coords = [v[0] for v in vertices if len(v) >= 2]
                    y_coords = [v[1] for v in vertices if len(v) >= 2]
                    if x_coords and y_coords:
                        logging.debug(f"ActionMaskGenerator: Vertex ranges - X: [{min(x_coords):.1f}, {max(x_coords):.1f}], "
                                     f"Y: [{min(y_coords):.1f}, {max(y_coords):.1f}]")
                
                # CRITICAL FIX: Use the high-quality rendering from the corrected parser
                mask = self._render_vertices_to_mask(parsed_program.vertices)
            else:
                logging.warning("ActionMaskGenerator: No vertices in parsed program, using fallback")
                if parsed_program:
                    logging.debug(f"ActionMaskGenerator: Parsed program attributes: {dir(parsed_program)}")
                # Fallback: try to parse individual commands manually
                mask = self._render_simple_commands_to_mask(action_commands)
            
            logging.debug(f"ActionMaskGenerator: Generated mask with {np.sum(mask > 0)} non-zero pixels")
            return mask
            
        except Exception as e:
            logging.warning(f"Failed to generate mask from actions: {e}")
            import traceback
            logging.debug(f"Full traceback: {traceback.format_exc()}")
            # Return a simple fallback mask
            return self._generate_fallback_mask()
    
    def _render_vertices_to_mask(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Render parsed vertices to a binary mask with CORRECTED coordinate transformation.
        Uses the same high-quality rendering as the fixed logo_parser.
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        try:
            if len(vertices) < 2:
                return mask
            
            # CRITICAL FIX: Use the same rendering method as the corrected parser
            # This ensures consistency between parsing and augmentation
            mask = self.action_parser._render_vertices_to_image(vertices, self.image_size)
            
            logging.debug(f"Final mask: {np.count_nonzero(mask)}/{mask.size} pixels ({100*np.count_nonzero(mask)/mask.size:.1f}%)")
            
        except Exception as e:
            logging.warning(f"Failed to render vertices to mask: {e}")
            import traceback
            logging.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Fallback to manual rendering if parser method fails
            mask = self._manual_render_fallback(vertices)
        
        return mask
    
    def _manual_render_fallback(self, vertices: List[Tuple[float, float]]) -> np.ndarray:
        """
        Fallback manual rendering method with corrected coordinate transformation.
        """
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        if len(vertices) < 2:
            return mask
        
        # Create high-resolution canvas for anti-aliasing
        scale_factor = 4  # 4x supersampling for anti-aliasing
        high_res_size = (self.image_size[0] * scale_factor, self.image_size[1] * scale_factor)
        high_res_mask = np.zeros(high_res_size, dtype=np.uint8)
        
        # CRITICAL FIX: Use the same coordinate transformation as the parser
        center_x = self.image_size[1] // 2  # 32 for 64x64 image
        center_y = self.image_size[0] // 2  # 32 for 64x64 image
        
        points = []
        for vertex in vertices:
            if len(vertex) >= 2 and isinstance(vertex[0], (int, float)) and isinstance(vertex[1], (int, float)):
                # CORRECTED: Transform from world coordinates to canvas coordinates
                # Turtle coordinates are centered at (0,0), canvas coordinates start at top-left
                pixel_x = (vertex[0] + center_x) * scale_factor  # Center and scale
                pixel_y = (vertex[1] + center_y) * scale_factor  # Center and scale (no Y flip needed)
                
                # Clamp to valid range
                pixel_x = max(0, min(high_res_size[1] - 1, int(round(pixel_x))))
                pixel_y = max(0, min(high_res_size[0] - 1, int(round(pixel_y))))
                points.append([pixel_x, pixel_y])
        
        logging.debug(f"Fallback coordinate transformation: {len(vertices)} vertices -> {len(points)} valid points")
        
        # Render at high resolution with proper thickness
        if len(points) >= 2:
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(high_res_mask, [points_array], False, 255, thickness=2*scale_factor, lineType=cv2.LINE_AA)
            
            # Draw vertices as small circles for better connectivity
            for point in points:
                cv2.circle(high_res_mask, tuple(point), scale_factor, 255, -1)
        
        # Downsample with anti-aliasing to target resolution
        mask = cv2.resize(high_res_mask, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Apply threshold to create clean binary image
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _render_simple_commands_to_mask(self, action_commands: List[str]) -> np.ndarray:
        """Simple fallback rendering for action commands."""
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        try:
            # Simple turtle-like interpretation
            x, y = self.image_size[1] // 2, self.image_size[0] // 2  # Start at center
            
            for cmd in action_commands:
                if isinstance(cmd, str):
                    parts = cmd.strip().split()
                    if len(parts) >= 3:
                        command = parts[0].lower()
                        try:
                            param1 = float(parts[1])
                            param2 = float(parts[2])
                            
                            if command in ['move_to', 'moveto']:
                                x = int((param1 + 1) * self.image_size[1] / 2)
                                y = int((param2 + 1) * self.image_size[0] / 2)
                            elif command in ['line_to', 'lineto']:
                                new_x = int((param1 + 1) * self.image_size[1] / 2)
                                new_y = int((param2 + 1) * self.image_size[0] / 2)
                                cv2.line(mask, (x, y), (new_x, new_y), 255, 2)
                                x, y = new_x, new_y
                            elif command == 'circle':
                                center_x = int((param1 + 1) * self.image_size[1] / 2)
                                center_y = int((param2 + 1) * self.image_size[0] / 2)
                                if len(parts) >= 4:
                                    radius = int(float(parts[3]) * min(self.image_size) / 4)
                                else:
                                    radius = 5
                                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                        except (ValueError, IndexError):
                            continue
                            
        except Exception as e:
            logging.warning(f"Failed to render simple commands: {e}")
        
        return mask
    
    def _generate_fallback_mask(self) -> np.ndarray:
        """Generate a simple fallback mask when action parsing fails."""
        mask = np.zeros(self.image_size, dtype=np.uint8)
        # Create a simple shape in the center
        center_x, center_y = self.image_size[1] // 2, self.image_size[0] // 2
        cv2.circle(mask, (center_x, center_y), min(self.image_size) // 4, 255, -1)
        return mask


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
        image_size = self.config.get('image_size', (64, 64))
        self.mask_generator = ActionMaskGenerator(image_size=image_size)
        
        # Initialize MaskRefiner for post-processing
        ref_cfg = self.config.get('refinement', {}) or {}
        valid_refiner_args = ['contour_approx_factor', 'min_component_size', 'closing_kernel_size', 'opening_kernel_size']
        filtered_ref_cfg = {k: v for k, v in ref_cfg.items() if k in valid_refiner_args}
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
        """Apply realistic post-processing to match real dataset characteristics"""
        
        # Apply Gaussian blur for smoothing (mimics real dataset)
        processed = cv2.GaussianBlur(image, (3, 3), 0.8)
        
        # Simple contrast enhancement that preserves black background
        # Only enhance pixels above threshold (drawn pixels)
        threshold_mask = processed > 10
        if np.any(threshold_mask):
            # Apply enhancement only where there are drawn pixels
            enhanced = processed.copy()
            enhanced[threshold_mask] = np.clip(processed[threshold_mask] * 1.1 + 10, 50, 255).astype(np.uint8)
            processed = enhanced
        
        # Final cleanup: remove isolated pixels
        if np.any(processed > 0):
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return processed

    def _post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Optional post-processing to clean up the generated mask."""
        try:
            # Apply minimal morphological cleaning
            kernel = np.ones((3, 3), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill small holes
            mask_filled = ndi.binary_fill_holes(mask_clean > 0).astype(np.uint8) * 255
            
            return mask_filled
        except Exception as e:
            logging.warning(f"Post-processing failed: {e}")
            return mask

    def load_derived_labels_from_actions(self, input_dir: str, problems_list: str = None, n_select: int = 50) -> List[Dict]:
        """
        Load action programs and convert them to the same format as derived_labels.json
        to maintain compatibility with the existing pipeline.
        """
        try:
            # Load action programs using the same approach as logo_to_shape.py
            problems_data = self.data_loader.load_action_programs(
                input_dir, 
                problems_list=problems_list,
                n_select=n_select
            )
            
            # Convert to derived_labels format
            derived_labels = []
            
            for problem_id, problem_data in problems_data.items():
                category = problem_data.get('category', 'unknown')
                positive_examples = problem_data.get('positive_examples', [])
                negative_examples = problem_data.get('negative_examples', [])
                
                # Process positive examples
                for i, action_commands in enumerate(positive_examples):
                    image_id = f"{problem_id}_pos_{i}"
                    image_path = f"images/{problem_id}/category_1/{i}.png"
                    
                    entry = {
                        'image_id': image_id,
                        'image_path': image_path,
                        'problem_id': problem_id,
                        'category': category,
                        'is_positive': True,
                        'action_commands': action_commands,
                        # Add minimal required fields for compatibility
                        'geometry': {},
                        'features': {}
                    }
                    derived_labels.append(entry)
                
                # Process negative examples
                for i, action_commands in enumerate(negative_examples):
                    image_id = f"{problem_id}_neg_{i}"
                    image_path = f"images/{problem_id}/category_0/{i}.png"
                    
                    entry = {
                        'image_id': image_id,
                        'image_path': image_path,
                        'problem_id': problem_id,
                        'category': category,
                        'is_positive': False,
                        'action_commands': action_commands,
                        # Add minimal required fields for compatibility
                        'geometry': {},
                        'features': {}
                    }
                    derived_labels.append(entry)
            
            logging.info(f"Loaded {len(derived_labels)} entries from action programs")
            return derived_labels
            
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
                # Defensive copy to avoid mutating input
                record = dict(entry)
                # Get action commands - either from the entry or use empty list as fallback
                action_commands = record.get('action_program', record.get('action_commands', []))
                image_id = record.get('image_id', 'unknown')
                if not action_commands:
                    logging.warning(f"No action commands found for {image_id}. Skipping.")
                    continue
                # Generate mask from action commands (not from real image)
                mask = self.process_action_commands(action_commands, image_id)
                # Generate parsed image from vertices and strokes (since UnifiedActionParser doesn't provide images)
                try:
                    logging.info(f"Attempting to generate parsed image from action commands for {image_id}")
                    logging.debug(f"Input action commands for {image_id}: {action_commands}")
                    parsed_program = self.mask_generator.action_parser._parse_single_image(
                        action_commands,
                        image_id=image_id,
                        is_positive=record.get('is_positive', True),
                        problem_id=record.get('problem_id', 'unknown')
                    )
                    
                    parsed_image = None
                    if parsed_program is None:
                        logging.warning(f"parsed_program is None for {image_id}")
                    elif parsed_program.vertices and len(parsed_program.vertices) > 0:
                        logging.info(f"Generating parsed image from {len(parsed_program.vertices)} vertices for {image_id}")
                        logging.debug(f"Raw vertices for {image_id}: {parsed_program.vertices}")
                        
                        # Add detailed stroke analysis
                        if hasattr(parsed_program, 'strokes'):
                            logging.debug(f"Parsed strokes for {image_id}: {len(parsed_program.strokes) if parsed_program.strokes else 0} strokes")
                            for i, stroke in enumerate(parsed_program.strokes or []):
                                if hasattr(stroke, 'vertices'):
                                    logging.debug(f"  Stroke {i}: {len(stroke.vertices)} vertices: {stroke.vertices}")
                                elif hasattr(stroke, 'points'):
                                    logging.debug(f"  Stroke {i}: {len(stroke.points)} points: {stroke.points}")
                        
                        parsed_image = self._render_parsed_image_from_vertices(parsed_program.vertices, image_id, parsed_program)
                        if parsed_image is not None:
                            logging.info(f"Successfully generated parsed image for {image_id}, shape: {parsed_image.shape}")
                        else:
                            logging.warning(f"Failed to generate parsed image from vertices for {image_id}")
                    else:
                        logging.warning(f"No vertices available to generate parsed image for {image_id}")
                        if hasattr(parsed_program, 'strokes'):
                            logging.debug(f"Strokes available: {parsed_program.strokes}")
                        if hasattr(parsed_program, 'action_sequence'):
                            logging.debug(f"Action sequence: {parsed_program.action_sequence}")
                except Exception as e:
                    logging.error(f"Failed to generate parsed image for {image_id}: {e}")
                    import traceback
                    logging.debug(f"Full traceback: {traceback.format_exc()}")
                    parsed_image = None
                # Save mask to file
                base_name = image_id.replace('/', '_')
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                try:
                    cv2.imwrite(mask_path, mask)
                except Exception as e:
                    logging.error(f"Failed to save mask to {mask_path}: {e}")
                    continue
                record['mask_path'] = mask_path
                # Save parsed image to file if available
                if parsed_image is not None:
                    parsed_image_path = os.path.join(output_dir, f"{base_name}_parsed.png")
                    try:
                        logging.info(f"Attempting to save parsed image for {image_id} to {parsed_image_path}")
                        cv2.imwrite(parsed_image_path, parsed_image)
                        record['parsed_image_path'] = parsed_image_path
                        logging.info(f"Successfully saved parsed image for {image_id} to {parsed_image_path}")
                    except Exception as e:
                        logging.error(f"Failed to save parsed image for {image_id}: {e}")
                else:
                    logging.info(f"No parsed image to save for {image_id} (parsed_image is None)")
                # Ensure compatibility with scene graph builder
                if 'geometry' in record and 'features' in record:
                    obj = {
                        'geometry': record['geometry'],
                        'features': record['features']
                    }
                    record['objects'] = [obj]
                if inspection_dir:
                    self._save_inspection_images(mask, action_commands, image_id, inspection_dir)
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
        """Save inspection images for debugging and validation."""
        try:
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
                    
        except Exception as e:
            logging.warning(f"Failed to save inspection images for {image_id}: {e}")


# Legacy compatibility - create an alias for the old class name
HybridAugmentor = HybridAugmentationPipeline
