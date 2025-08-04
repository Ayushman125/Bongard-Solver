"""
Unified Action Program Parser for Bongard-LOGO Dataset

This module correctly parses action program JSON files where:
- Each problem contains [positive_examples, negative_examples] 
- Each example is a list of stroke commands that combine to form one complete image
- Each stroke command is either 'line_' or 'arc_' with shape modifiers

Key insight: Multiple stroke commands combine to form a single image, not individual objects.
"""

import json
import logging
import math
import numpy as np
import cv2  # Added for high-quality rendering
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class StrokeType(Enum):
    """Primary stroke types in Bongard-LOGO"""
    LINE = "line"
    ARC = "arc"
    UNKNOWN = "unknown"

class ShapeModifier(Enum):
    """Shape modifiers for strokes (discovered from dataset analysis)"""
    NORMAL = "normal"      # 24,107 occurrences - straight lines or arcs
    CIRCLE = "circle"      # 6,256 occurrences - circular shapes/arcs  
    SQUARE = "square"      # 6,519 occurrences - square-based shapes
    TRIANGLE = "triangle"  # 5,837 occurrences - triangular shapes
    ZIGZAG = "zigzag"      # 6,729 occurrences - zigzag patterns
    UNKNOWN = "unknown"

@dataclass
class StrokeCommand:
    """Represents a single stroke command in an action program"""
    stroke_type: StrokeType
    shape_modifier: ShapeModifier
    parameters: Dict[str, float]
    raw_command: str
    
    def __post_init__(self):
        """Validate the stroke command after initialization"""
        if self.stroke_type == StrokeType.UNKNOWN:
            self.is_valid = False
        else:
            self.is_valid = True

@dataclass 
class ImageProgram:
    """Represents a complete image composed of multiple stroke commands"""
    strokes: List[StrokeCommand]
    image_id: str
    is_positive: bool
    problem_id: str
    vertices: List[Tuple[float, float]] = None
    geometry: Dict[str, Any] = None
    
    def __post_init__(self):
        """Calculate geometry after initialization"""
        if self.vertices is None:
            self.vertices = []
        if self.geometry is None:
            self.geometry = {}

class UnifiedActionParser:
    """
    Unified parser for Bongard-LOGO action program JSON files.
    
    Correctly handles the structure where:
    - JSON contains: {"problem_id": [positive_examples, negative_examples]}
    - Each example is a list of stroke commands forming one complete image
    - Each stroke command modifies the turtle state to draw part of the image
    """
    
    def __init__(self):
        # CRITICAL FIX: Start turtle at canvas center for proper positioning
        self.canvas_size = 64
        self.canvas_center = self.canvas_size // 2  # 32 for 64x64
        self.turtle_x = 0.0  # Will be centered in canvas coordinates during rendering
        self.turtle_y = 0.0  # Will be centered in canvas coordinates during rendering
        self.turtle_heading = 0.0  # degrees
        
        # CRITICAL FIX: Corrected scale factor for proper 64x64 image rendering
        # Bongard-LOGO uses normalized [0,1] coordinates, scale to ~15 for proper bounds
        self.scale_factor = 15.0  # FIXED: Reduced from 25.0 to prevent coordinate overflow
        
        # Enhanced precision settings
        self.high_precision_mode = True
        self.connectivity_tolerance = 0.1  # For detecting stroke connections
        self.adaptive_segmentation = True  # Use adaptive vertex density
        
        logging.debug(f"PARSER INIT: CORRECTED UnifiedActionParser initialized with scale_factor={self.scale_factor}")
        logging.debug(f"PARSER INIT: Canvas center: {self.canvas_center}, High precision mode: {self.high_precision_mode}")
        logging.debug(f"PARSER INIT: Turtle starts at world origin (0,0), will be centered during rendering")
        
    def parse_action_file(self, json_file_path: str) -> Dict[str, List[ImageProgram]]:
        """
        Parse an action program JSON file.
        
        Returns:
            Dict mapping problem_id to list of ImageProgram objects
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            action_data = json.load(f)
        
        parsed_problems = {}
        
        for problem_id, problem_data in action_data.items():
            if not isinstance(problem_data, list) or len(problem_data) != 2:
                print(f"Warning: Unexpected structure for {problem_id}")
                continue
                
            positive_examples, negative_examples = problem_data
            
            # Parse positive examples (7 images)
            positive_programs = []
            for i, stroke_commands in enumerate(positive_examples):
                image_id = f"{problem_id}_pos_{i}"
                image_program = self._parse_single_image(stroke_commands, image_id, True, problem_id)
                if image_program:
                    positive_programs.append(image_program)
            
            # Parse negative examples (7 images)  
            negative_programs = []
            for i, stroke_commands in enumerate(negative_examples):
                image_id = f"{problem_id}_neg_{i}"
                image_program = self._parse_single_image(stroke_commands, image_id, False, problem_id)
                if image_program:
                    negative_programs.append(image_program)
            
            parsed_problems[problem_id] = positive_programs + negative_programs
            
        return parsed_problems
    
    def _parse_single_image(self, stroke_commands: List[str], image_id: str, 
                           is_positive: bool, problem_id: str) -> Optional[ImageProgram]:
        """
        Parse stroke commands for a single complete image.
        Enhanced to properly handle multiple separate objects and complex stroke patterns.
        
        Args:
            stroke_commands: List of stroke commands that combine to form one image
                           Note: May have extra nesting level in actual data
            image_id: Unique identifier for this image
            is_positive: Whether this is a positive or negative example
            problem_id: The problem this image belongs to
            
        Returns:
            ImageProgram object with parsed strokes and geometry
        """

        # Unwrap all levels of nesting until we reach a list of strings
        actual_commands = stroke_commands
        while (
            isinstance(actual_commands, list)
            and len(actual_commands) > 0
            and isinstance(actual_commands[0], list)
        ):
            actual_commands = actual_commands[0]

        # Check if we have a list of strings
        if not actual_commands or not all(isinstance(cmd, str) for cmd in actual_commands):
            print(f"Warning: Unexpected image data structure for image_id={image_id}, problem_id={problem_id}. Data: {actual_commands}")
            return None
        
        logging.debug(f"PARSER: Processing {len(actual_commands)} commands for {image_id}")
        logging.debug(f"PARSER: Commands for {image_id}: {actual_commands}")
        
        # Enhanced parsing strategy: Detect object boundaries and stroke groupings
        parsed_strokes = []
        all_vertices = []
        object_boundaries = []  # Track where each distinct object starts/ends
        
        # Analyze stroke patterns to detect object separation
        stroke_groups = self._detect_stroke_groups(actual_commands)
        logging.debug(f"PARSER: Detected {len(stroke_groups)} stroke groups for {image_id}")
        
        for group_idx, stroke_group in enumerate(stroke_groups):
            # Reset turtle state for each distinct object/group
            group_start_vertex_idx = len(all_vertices)
            self._reset_turtle()
            
            # Apply intelligent positioning for multiple objects
            if group_idx > 0:
                self._position_for_next_object(group_idx, len(stroke_groups))
            
            logging.debug(f"PARSER: Processing group {group_idx+1}/{len(stroke_groups)} with {len(stroke_group)} strokes")
            logging.debug(f"PARSER: Group {group_idx+1} turtle start: ({self.turtle_x:.3f}, {self.turtle_y:.3f})")
            
            group_strokes = []
            group_vertices = []
            
            for i, stroke_cmd in enumerate(stroke_group):
                logging.debug(f"PARSER: Processing stroke {i+1}/{len(stroke_group)} in group {group_idx+1}: {stroke_cmd}")
                stroke = self._parse_stroke_command(stroke_cmd)
                if stroke and stroke.is_valid:
                    group_strokes.append(stroke)
                    parsed_strokes.append(stroke)
                    logging.debug(f"PARSER: Valid stroke parsed: {stroke}")
                    
                    # Generate vertices for this stroke and update turtle position
                    stroke_vertices = self._execute_stroke(stroke)
                    logging.debug(f"PARSER: Generated {len(stroke_vertices)} vertices from stroke: {stroke_vertices}")
                    group_vertices.extend(stroke_vertices)
                    all_vertices.extend(stroke_vertices)
                    logging.debug(f"PARSER: Updated turtle position to: ({self.turtle_x}, {self.turtle_y}), heading: {self.turtle_heading}")
                else:
                    logging.warning(f"PARSER: Invalid stroke for {image_id}, command: {stroke_cmd}")
            
            # Track object boundary
            group_end_vertex_idx = len(all_vertices)
            if group_vertices:
                object_boundaries.append({
                    'group_idx': group_idx,
                    'strokes': group_strokes,
                    'vertex_start': group_start_vertex_idx,
                    'vertex_end': group_end_vertex_idx,
                    'vertices': group_vertices,
                    'stroke_types': [s.stroke_type.value for s in group_strokes],
                    'shape_modifiers': [s.shape_modifier.value for s in group_strokes]
                })
                logging.debug(f"PARSER: Group {group_idx+1} completed with {len(group_vertices)} vertices")

        logging.debug(f"PARSER: Total vertices for {image_id}: {len(all_vertices)} across {len(object_boundaries)} objects")
        
        if not parsed_strokes:
            logging.warning(f"PARSER: No valid strokes parsed for {image_id}")
            return None
            
        # Calculate enhanced geometry for the complete image with object information
        geometry = self._calculate_enhanced_image_geometry(all_vertices, parsed_strokes, object_boundaries)
        
        image_program = ImageProgram(
            strokes=parsed_strokes,
            image_id=image_id,
            is_positive=is_positive,
            problem_id=problem_id,
            vertices=all_vertices,
            geometry=geometry
        )
        
        # Add object boundary information to the image program
        image_program.object_boundaries = object_boundaries
        
        return image_program
    
    def _calculate_modifier_similarity(self, mod1: Optional[str], mod2: Optional[str]) -> float:
        """
        Calculate similarity distance between two shape modifiers.
        Returns 0.0 for identical, 1.0 for completely different.
        """
        if mod1 == mod2:
            return 0.0
        
        if mod1 is None or mod2 is None:
            return 0.5  # Moderate difference if one is None
        
        # Define modifier families for similarity calculation
        geometric_modifiers = {'zigzag', 'triangle', 'square', 'circle'}
        curve_modifiers = {'curve', 'arc', 'spiral'}
        line_modifiers = {'line', 'straight', 'solid'}
        
        # Check if both are in same family
        mod1_families = []
        mod2_families = []
        
        if mod1 in geometric_modifiers:
            mod1_families.append('geometric')
        if mod1 in curve_modifiers:
            mod1_families.append('curve')
        if mod1 in line_modifiers:
            mod1_families.append('line')
            
        if mod2 in geometric_modifiers:
            mod2_families.append('geometric')
        if mod2 in curve_modifiers:
            mod2_families.append('curve')
        if mod2 in line_modifiers:
            mod2_families.append('line')
        
        # Calculate similarity based on family overlap
        if set(mod1_families) & set(mod2_families):
            return 0.3  # Same family, moderate similarity
        else:
            return 1.0  # Different families, very different
    
    def _parse_stroke_command(self, command: str) -> Optional[StrokeCommand]:
        """
        Parse a single stroke command string with corrected parameter interpretation.
        Handles all expected formats robustly and logs parsing failures.
        """
        if not isinstance(command, str):
            logging.warning(f"[PARSE ERROR] Command not a string: {command}")
            return None

        parts = command.split('_')
        if len(parts) < 3:
            logging.warning(f"[PARSE ERROR] Not enough parts: {command}")
            return None

        stroke_type_str = parts[0]
        if stroke_type_str == 'line':
            stroke_type = StrokeType.LINE
        elif stroke_type_str == 'arc':
            stroke_type = StrokeType.ARC
        else:
            logging.warning(f"[PARSE ERROR] Unknown stroke type: {command}")
            return None

        shape_modifier_str = parts[1]
        try:
            shape_modifier = ShapeModifier(shape_modifier_str)
        except ValueError:
            logging.warning(f"[PARSE ERROR] Unknown shape modifier: {command}")
            shape_modifier = ShapeModifier.UNKNOWN

        parameters = {}

        if stroke_type == StrokeType.LINE:
            # Format: line_shape_param1-param2
            # param1 = length (normalized [0,1]), param2 = angle (normalized [0,1])
            if len(parts) >= 3 and '-' in parts[2]:
                try:
                    param1, param2 = parts[2].split('-', 1)
                    parameters = {
                        'length': float(param1),  # Normalized length [0,1]
                        'angle': float(param2)    # Normalized angle [0,1] -> [0,2π] radians
                    }
                except ValueError:
                    logging.warning(f"[PARSE ERROR] Line param parse failed: {command}")
                    return None
            else:
                logging.warning(f"[PARSE ERROR] Line format unexpected: {command}")
                return None

        elif stroke_type == StrokeType.ARC:
            # CORRECTED Format: arc_shape_radius_span_angle-end_angle
            # All parameters are normalized [0,1]
            try:
                # Join parts 2 onwards to handle multiple underscores correctly
                param_section = '_'.join(parts[2:])
                
                # CRITICAL FIX: Split by underscore first to separate radius from span-end part
                underscore_split = param_section.split('_')
                if len(underscore_split) < 2:
                    logging.warning(f"[PARSE ERROR] Arc missing underscore separation: {command}")
                    return None
                    
                radius = float(underscore_split[0])  # Normalized radius [0,1]
                
                # CRITICAL FIX: Handle remaining parts that may contain underscores and dashes
                span_end_part = '_'.join(underscore_split[1:])  # Join remaining parts
                
                # Now split by dash to get span and end angles
                if '-' not in span_end_part:
                    logging.warning(f"[PARSE ERROR] Arc missing dash separation: {command}")
                    return None
                    
                span_str, end_str = span_end_part.split('-', 1)
                parameters = {
                    'radius': radius,                    # Normalized radius [0,1]
                    'span_angle': float(span_str),       # Normalized span [0,1] -> [0,2π] radians
                    'end_angle': float(end_str)          # Normalized end [0,1] -> [0,2π] radians
                }
                
                logging.debug(f"[PARSE SUCCESS] Arc parsed: radius={radius:.3f}, span={float(span_str):.3f}, end={float(end_str):.3f}")
                
            except Exception as e:
                logging.warning(f"[PARSE ERROR] Arc param parse failed: {command} ({e})")
                return None

        return StrokeCommand(
            stroke_type=stroke_type,
            shape_modifier=shape_modifier,
            parameters=parameters,
            raw_command=command
        )
    
    def _execute_stroke(self, stroke: StrokeCommand) -> List[Tuple[float, float]]:
        """
        Execute a stroke command to generate high-quality vertices and update turtle state.
        Enhanced version with improved precision and connectivity between strokes.
        
        Returns:
            List of (x, y) vertices generated by this stroke
        """
        start_pos = (self.turtle_x, self.turtle_y)
        logging.debug(f"STROKE EXEC: Enhanced stroke {stroke.stroke_type} from position {start_pos}")
        logging.debug(f"STROKE EXEC: Stroke parameters: {stroke.parameters}")
        
        # Always include precise start position for connectivity
        vertices = [start_pos]
        
        if stroke.stroke_type == StrokeType.LINE:
            line_vertices = self._execute_line_stroke(stroke)
            vertices.extend(line_vertices)
            logging.debug(f"STROKE EXEC: Enhanced line generated {len(line_vertices)} additional vertices")
        elif stroke.stroke_type == StrokeType.ARC:
            arc_vertices = self._execute_arc_stroke(stroke)
            vertices.extend(arc_vertices)
            logging.debug(f"STROKE EXEC: Enhanced arc generated {len(arc_vertices)} additional vertices")
        
        # Enhanced connectivity check
        if len(vertices) > 1:
            start_to_end_distance = math.sqrt(
                (vertices[-1][0] - vertices[0][0])**2 + 
                (vertices[-1][1] - vertices[0][1])**2
            )
            logging.debug(f"STROKE EXEC: Stroke span distance: {start_to_end_distance:.3f}")
        
        logging.debug(f"STROKE EXEC: Total {len(vertices)} vertices from enhanced stroke")
        logging.debug(f"STROKE EXEC: Turtle moved from {start_pos} to ({self.turtle_x:.6f}, {self.turtle_y:.6f})")
        
        return vertices
    
    def _execute_line_stroke(self, stroke: StrokeCommand) -> List[Tuple[float, float]]:
        """
        Execute a line stroke command with corrected coordinate system and parameter interpretation.
        Generates high-quality geometric patterns with accurate angles and smooth connections.
        """
        # Extract normalized parameters [0,1]
        length_norm = stroke.parameters.get('length', 1.0)
        angle_norm = stroke.parameters.get('angle', 0.0)
        
        # Convert to actual values
        length = length_norm * self.scale_factor  # Scale to canvas size
        angle_rad = angle_norm * 2 * math.pi      # Convert [0,1] to [0,2π] radians
        
        logging.debug(f"LINE EXEC: Corrected processing - length_norm: {length_norm:.3f}, angle_norm: {angle_norm:.3f}")
        logging.debug(f"LINE EXEC: Converted - length: {length:.3f}, angle_rad: {angle_rad:.3f} ({math.degrees(angle_rad):.1f}°)")
        
        # Calculate movement vector
        dx = length * math.cos(angle_rad)
        dy = length * math.sin(angle_rad)
        
        end_x = self.turtle_x + dx
        end_y = self.turtle_y + dy
        
        logging.debug(f"LINE EXEC: Movement from ({self.turtle_x:.3f}, {self.turtle_y:.3f}) to ({end_x:.3f}, {end_y:.3f})")
        
        # Generate enhanced vertices with high-quality shape modifiers
        vertices = self._generate_enhanced_line_vertices(
            (self.turtle_x, self.turtle_y), 
            (end_x, end_y), 
            stroke.shape_modifier
        )
        
        logging.debug(f"LINE EXEC: Generated {len(vertices)} enhanced vertices")
        
        # Update turtle position and heading
        self.turtle_x = end_x
        self.turtle_y = end_y
        self.turtle_heading = math.degrees(angle_rad)  # Store heading in degrees
        
        return vertices
    
    def _execute_arc_stroke(self, stroke: StrokeCommand) -> List[Tuple[float, float]]:
        """
        Execute an arc stroke command with corrected parameter interpretation and coordinate system.
        Generates high-density, smooth curves with precise shape modifier implementations.
        """
        # Extract normalized parameters [0,1]
        radius_norm = stroke.parameters.get('radius', 1.0)
        span_angle_norm = stroke.parameters.get('span_angle', 0.25)
        end_angle_norm = stroke.parameters.get('end_angle', 0.0)
        
        # Convert to actual values
        radius = radius_norm * self.scale_factor
        span_angle_rad = span_angle_norm * 2 * math.pi  # Convert [0,1] to [0,2π]
        end_angle_rad = end_angle_norm * 2 * math.pi
        
        logging.debug(f"ARC EXEC: Corrected params - radius_norm: {radius_norm:.3f}, span_norm: {span_angle_norm:.3f}, end_norm: {end_angle_norm:.3f}")
        logging.debug(f"ARC EXEC: Converted - radius: {radius:.3f}, span_rad: {span_angle_rad:.3f}, end_rad: {end_angle_rad:.3f}")
        
        # Calculate arc center relative to turtle position
        # The turtle is at the start point of the arc
        center_x = self.turtle_x
        center_y = self.turtle_y
        
        # Calculate number of segments for smooth arc (adaptive based on span)
        span_degrees = abs(math.degrees(span_angle_rad))
        num_segments = max(8, int(span_degrees / 5))  # At least 8 segments, more for larger arcs
        
        logging.debug(f"ARC EXEC: Arc center at ({center_x:.3f}, {center_y:.3f}), {num_segments} segments for {span_degrees:.1f}° span")
        
        vertices = []
        
        # Generate arc vertices
        for i in range(1, num_segments + 1):
            t = i / num_segments
            # Current angle along the arc
            current_angle = end_angle_rad + t * span_angle_rad
            
            # Calculate position on arc
            x = center_x + radius * math.cos(current_angle)
            y = center_y + radius * math.sin(current_angle)
            
            # Apply shape modifier effects
            if stroke.shape_modifier == ShapeModifier.ZIGZAG:
                # Zigzag pattern along the arc
                zigzag_freq = 8.0  # Number of zigzag cycles
                zigzag_amplitude = radius * 0.08
                zigzag_phase = math.sin(t * zigzag_freq * 2 * math.pi)
                
                # Apply zigzag perpendicular to arc direction
                perp_angle = current_angle + math.pi/2
                x += zigzag_amplitude * zigzag_phase * math.cos(perp_angle)
                y += zigzag_amplitude * zigzag_phase * math.sin(perp_angle)
                
            elif stroke.shape_modifier == ShapeModifier.SQUARE:
                # Square corners at regular intervals
                square_freq = 4.0
                corner_phase = (t * square_freq) % 1.0
                square_offset = radius * 0.1
                
                if corner_phase < 0.1 or corner_phase > 0.9:
                    direction = 1 if int(t * square_freq) % 2 == 0 else -1
                    offset_angle = current_angle + math.pi/2
                    x += direction * square_offset * math.cos(offset_angle)
                    y += direction * square_offset * math.sin(offset_angle)
                    
            elif stroke.shape_modifier == ShapeModifier.TRIANGLE:
                # Sharp triangular protrusions
                triangle_freq = 6.0
                peak_phase = (t * triangle_freq) % 1.0
                
                if 0.3 < peak_phase < 0.7:  # Peak region
                    peak_intensity = math.sin((peak_phase - 0.3) / 0.4 * math.pi)
                    triangle_offset = radius * 0.12 * peak_intensity
                    offset_angle = current_angle + math.pi/2
                    x += triangle_offset * math.cos(offset_angle)
                    y += triangle_offset * math.sin(offset_angle)
                    
            elif stroke.shape_modifier == ShapeModifier.CIRCLE:
                # Circular beading effect along the arc
                bead_freq = 10.0
                bead_phase = t * bead_freq * 2 * math.pi
                bead_amplitude = radius * 0.06
                bead_offset = bead_amplitude * (0.5 + 0.5 * math.cos(bead_phase))
                
                # Apply beading radially outward
                offset_angle = current_angle + math.pi/2
                x += bead_offset * math.cos(offset_angle)
                y += bead_offset * math.sin(offset_angle)
            
            vertices.append((x, y))
        
        # Update turtle position to the end of the arc
        if vertices:
            self.turtle_x, self.turtle_y = vertices[-1]
            self.turtle_heading = math.degrees(end_angle_rad + span_angle_rad)
            
            # Normalize heading to 0-360 range
            while self.turtle_heading >= 360:
                self.turtle_heading -= 360
            while self.turtle_heading < 0:
                self.turtle_heading += 360

        logging.debug(f"ARC EXEC: Generated {len(vertices)} vertices, final turtle: ({self.turtle_x:.3f}, {self.turtle_y:.3f}), heading: {self.turtle_heading:.1f}°")

        return vertices
    
    def _generate_enhanced_line_vertices(self, start: Tuple[float, float], end: Tuple[float, float], 
                                       shape_modifier: ShapeModifier) -> List[Tuple[float, float]]:
        """
        Generate enhanced, high-precision vertices for a line stroke based on its shape modifier.
        Produces detailed geometric patterns that closely match real dataset characteristics.
        """
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate base line properties
        line_length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        if line_length < 0.001:  # Degenerate line
            return [(end_x, end_y)]
        
        # Normalized direction vector
        dx_norm = (end_x - start_x) / line_length
        dy_norm = (end_y - start_y) / line_length
        
        # Perpendicular vector for offsets
        perp_x = -dy_norm
        perp_y = dx_norm
        
        if shape_modifier == ShapeModifier.NORMAL:
            # High-precision straight line with interpolated points for smooth rendering
            num_points = max(3, int(line_length / 2.0))  # Adaptive density
            vertices = []
            for i in range(1, num_points + 1):
                t = i / num_points
                x = start_x + t * (end_x - start_x)
                y = start_y + t * (end_y - start_y)
                vertices.append((x, y))
            return vertices
            
        elif shape_modifier == ShapeModifier.ZIGZAG:
            # Enhanced zigzag with smooth transitions
            num_segments = max(6, int(line_length / 3.0))  # More segments for smoother zigzag
            vertices = []
            amplitude = line_length * 0.12  # Slightly larger amplitude for visibility
            
            for i in range(1, num_segments + 1):
                t = i / num_segments
                base_x = start_x + t * (end_x - start_x)
                base_y = start_y + t * (end_y - start_y)
                
                # Sinusoidal zigzag for smoother appearance
                zigzag_freq = 4.0  # Number of zigzag cycles
                zigzag_phase = math.sin(t * zigzag_freq * 2 * math.pi)
                offset_magnitude = amplitude * zigzag_phase
                
                x = base_x + offset_magnitude * perp_x
                y = base_y + offset_magnitude * perp_y
                vertices.append((x, y))
                
            return vertices
            
        elif shape_modifier == ShapeModifier.TRIANGLE:
            # Enhanced triangular pattern with sharp peaks
            num_triangles = max(3, int(line_length / 8.0))  # Adaptive number of triangles
            vertices = []
            triangle_height = line_length * 0.2  # Height of triangular peaks
            
            for i in range(num_triangles):
                # Base points of triangle
                t1 = (i * 2) / (num_triangles * 2)
                t2 = ((i * 2) + 1) / (num_triangles * 2)  # Peak
                t3 = ((i * 2) + 2) / (num_triangles * 2)
                
                # First base point
                if i == 0 or t1 > 0:  # Avoid duplicate start point
                    x1 = start_x + t1 * (end_x - start_x)
                    y1 = start_y + t1 * (end_y - start_y)
                    vertices.append((x1, y1))
                
                # Peak point
                x_peak = start_x + t2 * (end_x - start_x)
                y_peak = start_y + t2 * (end_y - start_y)
                x_peak += triangle_height * perp_x
                y_peak += triangle_height * perp_y
                vertices.append((x_peak, y_peak))
                
                # Second base point
                x3 = start_x + t3 * (end_x - start_x)
                y3 = start_y + t3 * (end_y - start_y)
                vertices.append((x3, y3))
                
            return vertices
            
        elif shape_modifier == ShapeModifier.SQUARE:
            # Enhanced square pattern with sharp corners
            num_squares = max(2, int(line_length / 10.0))  # Adaptive number of squares
            vertices = []
            square_size = line_length * 0.15  # Size of square protrusions
            
            for i in range(num_squares):
                # Square corner positions along the line
                t_start = i / num_squares
                t_mid = (i + 0.5) / num_squares
                t_end = (i + 1) / num_squares
                
                # Start corner
                x_start = start_x + t_start * (end_x - start_x)
                y_start = start_y + t_start * (end_y - start_y)
                vertices.append((x_start, y_start))
                
                # Square corner (perpendicular offset)
                x_corner = start_x + t_mid * (end_x - start_x)
                y_corner = start_y + t_mid * (end_y - start_y)
                direction = 1 if i % 2 == 0 else -1  # Alternate sides
                x_corner += direction * square_size * perp_x
                y_corner += direction * square_size * perp_y
                vertices.append((x_corner, y_corner))
                
                # End corner
                x_end = start_x + t_end * (end_x - start_x)
                y_end = start_y + t_end * (end_y - start_y)
                vertices.append((x_end, y_end))
                
            return vertices
            
        elif shape_modifier == ShapeModifier.CIRCLE:
            # Enhanced circular beading pattern
            num_circles = max(4, int(line_length / 6.0))  # Adaptive number of circles
            vertices = []
            circle_radius = line_length * 0.08  # Radius of circular beads
            
            for i in range(1, num_circles * 8 + 1):  # 8 points per circle for smoothness
                t = i / (num_circles * 8)
                if t > 1.0:
                    break
                    
                # Base position along line
                base_x = start_x + t * (end_x - start_x)
                base_y = start_y + t * (end_y - start_y)
                
                # Circular motion around the line
                circle_phase = t * num_circles * 2 * math.pi
                offset_magnitude = circle_radius * math.cos(circle_phase)
                
                x = base_x + offset_magnitude * perp_x
                y = base_y + offset_magnitude * perp_y
                vertices.append((x, y))
                
            return vertices
        
        # Default: high-precision straight line
        return [(end_x, end_y)]
    
    def _calculate_enhanced_image_geometry(self, vertices: List[Tuple[float, float]], 
                                          strokes: List[StrokeCommand], 
                                          object_boundaries: List[Dict]) -> Dict[str, Any]:
        """Calculate enhanced geometric properties including object-level analysis"""
        if not vertices:
            return {}
            
        # Basic bounding box
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        
        bbox = {
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys)
        }
        
        # Centroid
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        
        # Stroke type distribution
        stroke_types = {}
        shape_modifiers = {}
        
        for stroke in strokes:
            stroke_type = stroke.stroke_type.value
            shape_mod = stroke.shape_modifier.value
            
            stroke_types[stroke_type] = stroke_types.get(stroke_type, 0) + 1
            shape_modifiers[shape_mod] = shape_modifiers.get(shape_mod, 0) + 1
        
        # Enhanced object-level analysis
        object_analysis = []
        for obj_info in object_boundaries:
            obj_vertices = obj_info['vertices']
            if obj_vertices:
                obj_xs = [v[0] for v in obj_vertices]
                obj_ys = [v[1] for v in obj_vertices]
                
                object_analysis.append({
                    'object_id': obj_info['group_idx'],
                    'vertex_count': len(obj_vertices),
                    'stroke_count': len(obj_info['strokes']),
                    'stroke_types': obj_info['stroke_types'],
                    'shape_modifiers': obj_info['shape_modifiers'],
                    'bbox': {
                        'min_x': min(obj_xs),
                        'max_x': max(obj_xs),
                        'min_y': min(obj_ys),
                        'max_y': max(obj_ys)
                    },
                    'centroid': (sum(obj_xs) / len(obj_xs), sum(obj_ys) / len(obj_ys)),
                    'complexity': len(set(obj_info['shape_modifiers']))  # Number of unique shape types
                })
        
        return {
            'bbox': bbox,
            'centroid': centroid,
            'num_strokes': len(strokes),
            'num_vertices': len(vertices),
            'num_objects': len(object_boundaries),
            'stroke_type_distribution': stroke_types,
            'shape_modifier_distribution': shape_modifiers,
            'width': bbox['max_x'] - bbox['min_x'],
            'height': bbox['max_y'] - bbox['min_y'],
            'objects': object_analysis,
            'spatial_complexity': self._calculate_spatial_complexity(object_boundaries),
            'pattern_diversity': len(set(shape_modifiers.keys()))
        }
    
    def _calculate_spatial_complexity(self, object_boundaries: List[Dict]) -> float:
        """Calculate spatial complexity based on object distribution and interactions"""
        if len(object_boundaries) <= 1:
            return 0.0
        
        # Calculate pairwise distances between object centroids
        centroids = []
        for obj_info in object_boundaries:
            if obj_info['vertices']:
                obj_xs = [v[0] for v in obj_info['vertices']]
                obj_ys = [v[1] for v in obj_info['vertices']]
                centroid = (sum(obj_xs) / len(obj_xs), sum(obj_ys) / len(obj_ys))
                centroids.append(centroid)
        
        if len(centroids) < 2:
            return 0.0
        
        # Calculate average inter-object distance
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = math.sqrt((centroids[i][0] - centroids[j][0])**2 + 
                               (centroids[i][1] - centroids[j][1])**2)
                total_distance += dist
                pair_count += 1
        
        avg_distance = total_distance / pair_count if pair_count > 0 else 0.0
        
        # Normalize by scale factor
        normalized_complexity = avg_distance / self.scale_factor
        
        return min(1.0, normalized_complexity)  # Cap at 1.0
    
    def _reset_turtle(self):
        """Reset turtle graphics state for a new image with enhanced precision"""
        self.turtle_x = 0.0
        self.turtle_y = 0.0
        self.turtle_heading = 0.0
        logging.debug(f"TURTLE RESET: Enhanced precision - Position: ({self.turtle_x:.6f}, {self.turtle_y:.6f}), heading: {self.turtle_heading:.3f}°")
        logging.debug(f"TURTLE RESET: Scale factor: {self.scale_factor}, High precision: {self.high_precision_mode}")
    
    def _detect_stroke_groups(self, stroke_commands: List[str]) -> List[List[str]]:
        """
        CORRECTED: Conservative stroke grouping to prevent over-separation.
        Most Bongard-LOGO images should be single connected shapes, not scattered objects.
        """
        if len(stroke_commands) <= 1:
            return [stroke_commands]
        
        # CRITICAL FIX: Use conservative grouping to keep related strokes together
        # Only split into multiple groups when there are very clear separations
        
        # For small numbers of strokes, keep them as a single connected shape
        if len(stroke_commands) <= 3:
            logging.debug(f"STROKE GROUP: Keeping {len(stroke_commands)} strokes as single connected shape")
            return [stroke_commands]
        
        # Check if all strokes have similar parameters (likely one object)
        stroke_types = set()
        shape_modifiers = set()
        lengths = []
        
        for cmd in stroke_commands:
            stroke = self._parse_stroke_command(cmd)
            if stroke:
                stroke_types.add(stroke.stroke_type)
                shape_modifiers.add(stroke.shape_modifier)
                if 'length' in stroke.parameters:
                    lengths.append(stroke.parameters['length'])
        
        # If parameters are similar, keep as single object
        if len(stroke_types) <= 1 and len(shape_modifiers) <= 2:
            logging.debug(f"STROKE GROUP: Similar parameters detected, keeping as single object")
            return [stroke_commands]
        
        # Conservative splitting only for very different stroke patterns
        groups = []
        current_group = []
        
        prev_stroke = None
        for i, cmd in enumerate(stroke_commands):
            stroke = self._parse_stroke_command(cmd)
            if not stroke:
                current_group.append(cmd)
                continue
                
            should_start_new_group = False
            
            if prev_stroke and len(current_group) > 0:
                # Only split on very dramatic changes (increased thresholds)
                if stroke.stroke_type == StrokeType.LINE and prev_stroke.stroke_type == StrokeType.LINE:
                    curr_length = stroke.parameters.get('length', 1.0)
                    prev_length = prev_stroke.parameters.get('length', 1.0)
                    
                    # MUCH higher threshold for length changes
                    length_ratio = abs(curr_length - prev_length) / max(prev_length, 0.1)
                    if length_ratio > 1.5:  # Increased from 0.6 to 1.5
                        should_start_new_group = True
                        logging.debug(f"STROKE GROUP: Major length change detected ({length_ratio:.3f})")
                
                # Only split on extreme modifier differences
                if stroke.shape_modifier != prev_stroke.shape_modifier:
                    modifier_distance = self._calculate_modifier_similarity(stroke.shape_modifier, prev_stroke.shape_modifier)
                    if modifier_distance > 0.9 and len(current_group) >= 4:  # Very high threshold
                        should_start_new_group = True
                        logging.debug(f"STROKE GROUP: Extreme modifier change detected ({modifier_distance:.3f})")
            
            # Much larger group size limit
            if len(current_group) >= 8:  # Increased from 5 to 8
                should_start_new_group = True
                logging.debug(f"STROKE GROUP: Large group size limit reached")
            
            if should_start_new_group and current_group:
                groups.append(current_group)
                current_group = [cmd]
            else:
                current_group.append(cmd)
            
            prev_stroke = stroke
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        # If no clear separation found, use intelligent fallback strategies
        if len(groups) == 1 and len(stroke_commands) > 4:
            groups = self._apply_fallback_grouping(stroke_commands)
        
        logging.debug(f"STROKE GROUPS: Detected {len(groups)} groups from {len(stroke_commands)} commands")
        for i, group in enumerate(groups):
            logging.debug(f"  Group {i+1}: {len(group)} strokes - {[cmd.split('_')[1] if '_' in cmd else cmd for cmd in group]}")
        
        return groups
    
    def _apply_fallback_grouping(self, stroke_commands: List[str]) -> List[List[str]]:
        """Apply fallback grouping strategies when primary detection fails"""
        
        # Strategy 1: Split based on stroke length patterns
        groups = []
        current_group = []
        
        lengths = []
        for cmd in stroke_commands:
            try:
                if '-' in cmd:
                    length_str = cmd.split('-')[0].split('_')[-1]
                    length = float(length_str)
                    lengths.append(length)
                else:
                    lengths.append(1.0)
            except (ValueError, IndexError):
                lengths.append(1.0)
        
        # Find significant length transitions
        for i, cmd in enumerate(stroke_commands):
            if i > 0 and len(current_group) >= 2:
                # Check for return to similar length (indicating new object start)
                if i < len(lengths) - 1:
                    curr_length = lengths[i]
                    first_length = lengths[0] if current_group else 1.0
                    
                    if abs(curr_length - first_length) < 0.2 and len(current_group) >= 3:
                        groups.append(current_group)
                        current_group = [cmd]
                        continue
            
            current_group.append(cmd)
        
        if current_group:
            groups.append(current_group)
        
        # Strategy 2: If still one group, split roughly in half (common case)
        if len(groups) == 1 and len(stroke_commands) >= 6:
            mid_point = len(stroke_commands) // 2
            # Adjust split point to avoid breaking obvious patterns
            for offset in range(-1, 2):
                split_idx = mid_point + offset
                if 0 < split_idx < len(stroke_commands):
                    groups = [stroke_commands[:split_idx], stroke_commands[split_idx:]]
                    break
        
        logging.debug(f"FALLBACK GROUPING: Applied fallback, got {len(groups)} groups")
        return groups
    
    def _calculate_modifier_similarity(self, mod1: ShapeModifier, mod2: ShapeModifier) -> float:
        """Calculate dissimilarity between shape modifiers (0=same, 1=completely different)"""
        if mod1 == mod2:
            return 0.0
        
        # Define modifier similarity matrix
        similarity_map = {
            (ShapeModifier.NORMAL, ShapeModifier.CIRCLE): 0.3,
            (ShapeModifier.NORMAL, ShapeModifier.SQUARE): 0.5,
            (ShapeModifier.NORMAL, ShapeModifier.TRIANGLE): 0.5,
            (ShapeModifier.NORMAL, ShapeModifier.ZIGZAG): 0.7,
            (ShapeModifier.CIRCLE, ShapeModifier.SQUARE): 0.6,
            (ShapeModifier.CIRCLE, ShapeModifier.TRIANGLE): 0.6,
            (ShapeModifier.CIRCLE, ShapeModifier.ZIGZAG): 0.8,
            (ShapeModifier.SQUARE, ShapeModifier.TRIANGLE): 0.4,
            (ShapeModifier.SQUARE, ShapeModifier.ZIGZAG): 0.8,
            (ShapeModifier.TRIANGLE, ShapeModifier.ZIGZAG): 0.8,
        }
        
        key = (mod1, mod2) if mod1.value < mod2.value else (mod2, mod1)
        return similarity_map.get(key, 1.0)
    
    def _position_for_next_object(self, object_index: int, total_objects: int):
        """
        CORRECTED: Compact positioning to keep objects within 64x64 canvas bounds.
        Reduced spacing by 70% to prevent coordinate overflow.
        """
        # CRITICAL FIX: Much smaller spacing to prevent coordinates going outside canvas
        spacing = self.scale_factor * 0.4  # Reduced from 1.2 to 0.4 (70% reduction)
        
        if total_objects == 2:
            # Closer side-by-side layout
            if object_index == 1:
                self.turtle_x = spacing
                self.turtle_y = 0.0
        elif total_objects == 3:
            # Compact triangle layout
            if object_index == 1:
                self.turtle_x = spacing * 0.5  # Further reduced
                self.turtle_y = spacing * 0.3
            elif object_index == 2:
                self.turtle_x = -spacing * 0.5
                self.turtle_y = spacing * 0.3
        elif total_objects >= 4:
            # Very compact grid layout
            cols = 2
            rows = (total_objects + cols - 1) // cols
            col = object_index % cols
            row = object_index // cols
            self.turtle_x = (col - cols/2 + 0.5) * spacing * 0.6  # Additional reduction
            self.turtle_y = (row - rows/2 + 0.5) * spacing * 0.6
        
        self.turtle_heading = 0.0  # Reset heading for each object
        
        logging.debug(f"POSITIONING: Compact layout - Object {object_index+1}/{total_objects} positioned at ({self.turtle_x:.3f}, {self.turtle_y:.3f})")

    def get_image_summary(self, image_program: ImageProgram) -> Dict[str, Any]:
        """Get a summary of an image program for analysis"""
        return {
            'image_id': image_program.image_id,
            'problem_id': image_program.problem_id,
            'is_positive': image_program.is_positive,
            'num_strokes': len(image_program.strokes),
            'stroke_types': [s.stroke_type.value for s in image_program.strokes],
            'shape_modifiers': [s.shape_modifier.value for s in image_program.strokes],
            'geometry': image_program.geometry,
            'vertices_count': len(image_program.vertices) if image_program.vertices else 0
        }


    def visualize_image_program(self, image_program: ImageProgram, save_path: str = None) -> np.ndarray:
        """
        Render the parsed image program using high-quality anti-aliased rendering.
        Returns a 64x64 binary image that matches the real Bongard-LOGO dataset quality.
        """
        if not image_program or not image_program.vertices:
            print("No vertices to visualize")
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        # CRITICAL FIX: High-quality anti-aliased rendering
        return self._render_vertices_to_image(image_program.vertices)
    
    def _render_vertices_to_image(self, vertices: List[Tuple[float, float]], 
                                  size: Tuple[int, int] = None) -> np.ndarray:
        """
        CORRECTED: High-quality rendering with adaptive coordinate bounds checking.
        Prevents coordinate overflow and ensures all shapes are visible within canvas.
        """
        import cv2
        
        if size is None:
            size = (self.canvas_size, self.canvas_size)
        
        if not vertices:
            return np.zeros(size, dtype=np.uint8)
        
        # CRITICAL FIX: Adaptive coordinate scaling to ensure all vertices fit
        all_x = [x for x, y in vertices]
        all_y = [y for x, y in vertices]
        
        # Calculate bounds of all vertices
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Calculate required scale to fit within canvas with margins
        margin = 8  # Leave 8-pixel margin on all sides
        canvas_width = size[0] - 2 * margin
        canvas_height = size[1] - 2 * margin
        
        # Calculate adaptive scale factors
        if max_x - min_x > 0:
            scale_x = canvas_width / (max_x - min_x)
        else:
            scale_x = 1.0
            
        if max_y - min_y > 0:
            scale_y = canvas_height / (max_y - min_y)
        else:
            scale_y = 1.0
        
        # Use the smaller scale to maintain aspect ratio
        adaptive_scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
        
        logging.debug(f"RENDER: Bounds X[{min_x:.2f}, {max_x:.2f}] Y[{min_y:.2f}, {max_y:.2f}], adaptive_scale={adaptive_scale:.3f}")
        
        # Create 4x resolution canvas for anti-aliasing
        aa_scale = 4
        high_res_size = (size[0] * aa_scale, size[1] * aa_scale)
        canvas = np.zeros(high_res_size, dtype=np.uint8)
        
        # Transform vertices to canvas coordinates with adaptive scaling
        canvas_vertices = []
        center_x, center_y = high_res_size[1] // 2, high_res_size[0] // 2
        
        for x, y in vertices:
            # Apply adaptive scaling and centering
            canvas_x = center_x + (x - (min_x + max_x) / 2) * adaptive_scale * aa_scale
            canvas_y = center_y + (y - (min_y + max_y) / 2) * adaptive_scale * aa_scale
            
            # CRITICAL FIX: Clamp coordinates to canvas bounds
            canvas_x = max(0, min(high_res_size[1] - 1, canvas_x))
            canvas_y = max(0, min(high_res_size[0] - 1, canvas_y))
            
            canvas_vertices.append((int(canvas_x), int(canvas_y)))
        
        # Draw high-resolution path with proper thickness
        if len(canvas_vertices) > 1:
            cv2.polylines(canvas, [np.array(canvas_vertices, dtype=np.int32)], 
                         False, 255, thickness=2*aa_scale, lineType=cv2.LINE_AA)
        
        # Draw vertices as small circles for better connectivity
        for x, y in canvas_vertices:
            cv2.circle(canvas, (x, y), aa_scale, 255, -1)
        
        # Downsample with anti-aliasing to target resolution
        result = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
        
        # Apply threshold to create clean binary image
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
        
        logging.debug(f"RENDER: Generated {size} image with {np.count_nonzero(result)} non-zero pixels")
        
        return result

        # Show stroke details in text
        stroke_details = []
        for i, stroke in enumerate(image_program.strokes):
            details = f"{i+1}. {stroke.stroke_type.value}_{stroke.shape_modifier.value}"
            stroke_details.append(details)

        # Add text box with stroke details
        text_content = "\n".join(stroke_details[:10])  # Show first 10 strokes
        if len(image_program.strokes) > 10:
            text_content += f"\n... and {len(image_program.strokes) - 10} more"

        ax.text(0.02, 0.98, text_content, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()
    """
    Analyze the structure of action program data to validate our understanding.
    
    Args:
        sample_data: The JSON snippet you provided
    """
    print("=== BONGARD-LOGO ACTION DATA STRUCTURE ANALYSIS ===")
    print()
    
    # Parse the sample JSON (just the beginning snippet you provided)
    sample = '''{"bd_equilateral_traingle_0000": [[[["line_triangle_1.000-0.500", "line_square_1.000-0.833", "line_circle_1.000-0.833"]], [["line_zigzag_1.000-0.500", "line_normal_1.000-0.833", "line_triangle_1.000-0.833"]], [["line_normal_1.000-0.500", "line_normal_1.000-0.833", "line_normal_1.000-0.833"]], [["line_zigzag_1.000-0.500", "line_normal_1.000-0.833", "line_circle_1.000-0.833"]], [["line_normal_1.000-0.500", "line_circle_1.000-0.833", "line_triangle_1.000-0.833"]], [["line_zigzag_1.000-0.500", "line_triangle_1.000-0.833", "line_triangle_1.000-0.833"]], [["line_square_1.000-0.500", "line_square_1.000-0.833", "line_triangle_1.000-0.833"]]], [[["line_circle_1.000-0.500", "line_triangle_1.000-0.917", "line_circle_0.518-0.792"]], [["line_zigzag_0.700-0.500", "line_normal_0.700-0.667"]], [["line_circle_0.700-0.500", "line_normal_0.700-0.667"]], [["line_normal_1.000-0.500", "line_circle_1.000-0.917", "line_triangle_0.518-0.792"]], [["line_normal_1.000-0.500", "line_zigzag_0.707-0.875", "line_zigzag_0.707-0.750"]], [["line_circle_1.000-0.500", "line_square_0.500-0.833", "line_circle_0.866-0.750"]], [["line_normal_0.700-0.500", "line_normal_0.700-0.583"]]]]}'''
    
    try:
        data = json.loads(sample)
        problem_id = list(data.keys())[0]
        problem_data = data[problem_id]
        
        print(f"Problem ID: {problem_id}")
        print(f"Problem data type: {type(problem_data)}")
        print(f"Number of example categories: {len(problem_data)}")
        print()
        
        for category_idx, category in enumerate(problem_data):
            category_name = "POSITIVE" if category_idx == 0 else "NEGATIVE" 
            print(f"{category_name} Examples:")
            print(f"  Number of images: {len(category)}")
            
            for img_idx, image_data in enumerate(category[:3]):  # Show first 3 images
                # The data has an extra level of nesting - each image_data is wrapped in another list
                if isinstance(image_data, list) and len(image_data) > 0:
                    actual_strokes = image_data[0]  # Unwrap the extra level
                else:
                    actual_strokes = image_data
                    
                print(f"  Image {img_idx + 1}:")
                print(f"    Raw structure: {type(image_data)} with {len(image_data)} elements")
                print(f"    Actual strokes: {actual_strokes}")
                print(f"    Number of strokes: {len(actual_strokes) if isinstance(actual_strokes, list) else 'N/A'}")
                
                # Analyze stroke types and shapes
                stroke_types = {}
                shape_types = {}
                if isinstance(actual_strokes, list):
                    for stroke in actual_strokes:
                        if isinstance(stroke, str) and '_' in stroke:
                            parts = stroke.split('_')
                            if len(parts) >= 2:
                                stroke_type = parts[0]
                                shape_type = parts[1]
                                stroke_types[stroke_type] = stroke_types.get(stroke_type, 0) + 1
                                shape_types[shape_type] = shape_types.get(shape_type, 0) + 1
                
                print(f"    Stroke types: {stroke_types}")
                print(f"    Shape types: {shape_types}")
                print()
            
            if len(category) > 3:
                print(f"  ... and {len(category) - 3} more images")
            print()
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")


def test_parser_with_sample_data():
    """Test the parser with sample arc and line commands"""
    parser = UnifiedActionParser()
    
    # Test commands from your JSON data
    test_commands = [
        ["line_normal_0.500-0.500", "line_normal_0.500-0.083", "arc_normal_0.500_0.542-0.167"],
        ["line_triangle_1.000-0.500", "arc_circle_0.500_0.750-0.750"],
        ["line_zigzag_0.500-0.500", "arc_triangle_0.500_0.625-0.750"]
    ]
    
    print("=== TESTING IMPROVED LOGO PARSER ===\\n")
    
    for i, commands in enumerate(test_commands):
        print(f"Test {i+1}: {commands}")
        
        # Parse the image
        image_program = parser._parse_single_image(
            commands, f"test_img_{i+1}", True, "test_problem"
        )
        
        if image_program:
            print(f"  ✅ Successfully parsed {len(image_program.strokes)} strokes")
            print(f"  ✅ Generated {len(image_program.vertices)} vertices")
            
            # Show first few vertices
            for j, vertex in enumerate(image_program.vertices[:5]):
                print(f"    Vertex {j+1}: ({vertex[0]:.3f}, {vertex[1]:.3f})")
            if len(image_program.vertices) > 5:
                print(f"    ... and {len(image_program.vertices) - 5} more vertices")
            
            # Visualize (try to import matplotlib, skip if not available)
            try:
                save_path = f"test_parsed_output_{i+1}.png"
                parser.visualize_image_program(image_program, save_path)
            except ImportError:
                print("    (matplotlib not available for visualization)")
        else:
            print(f"  ❌ Failed to parse commands")
        
        print()


def test_arc_parsing():
    """Test arc parsing specifically"""
    parser = UnifiedActionParser()
    
    # Test arc commands from your JSON data
    test_arcs = [
        "arc_normal_0.500_0.542-0.167",
        "arc_circle_0.500_0.639-0.750", 
        "arc_square_0.500_0.625-0.750",
        "arc_triangle_0.500_0.583-0.750",
        "arc_zigzag_0.500_0.608-0.750"
    ]
    
    print("=== ARC PARSING TEST ===")
    for arc_cmd in test_arcs:
        print(f"\nTesting: {arc_cmd}")
        stroke = parser._parse_stroke_command(arc_cmd)
        if stroke:
            print(f"  ✓ Parsed successfully")
            print(f"    Type: {stroke.stroke_type}")
            print(f"    Shape: {stroke.shape_modifier}")
            print(f"    Parameters: {stroke.parameters}")
            
            # Test execution
            parser._reset_turtle()
            vertices = parser._execute_arc_stroke(stroke)
            print(f"    Generated {len(vertices)} vertices")
            if vertices:
                print(f"    First vertex: {vertices[0]}")
                print(f"    Last vertex: {vertices[-1]}")
        else:
            print(f"  ✗ Failed to parse!")
    
    # Create a simple test image with one arc
    test_program = ["arc_normal_0.500_0.542-0.167"]
    image_program = parser._parse_single_image(test_program, "test_arc", True, "test")
    
    if image_program:
        print(f"\n=== ARC IMAGE TEST ===")
        print(f"Strokes parsed: {len(image_program.strokes)}")
        print(f"Total vertices: {len(image_program.vertices)}")
        
        # Visualize the arc
        parser.visualize_image_program(image_program, "test_arc_visualization.png")
        print("Saved test_arc_visualization.png")
    
    return parser


if __name__ == "__main__":
    # Test the improved parser
    test_parser_with_sample_data()
    
    # Test arc parsing specifically
    print("\n" + "="*50)
    test_arc_parsing()
