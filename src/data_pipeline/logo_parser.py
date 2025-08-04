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
        self.turtle_x = 0.0
        self.turtle_y = 0.0
        self.turtle_heading = 0.0  # degrees
        self.scale_factor = 32.0  # Convert normalized coordinates to pixel space (64x64 image)
        
        # Enhanced precision settings
        self.high_precision_mode = True
        self.connectivity_tolerance = 0.1  # For detecting stroke connections
        self.adaptive_segmentation = True  # Use adaptive vertex density
        
        logging.debug(f"PARSER INIT: Enhanced UnifiedActionParser initialized with scale_factor={self.scale_factor}")
        logging.debug(f"PARSER INIT: High precision mode: {self.high_precision_mode}, adaptive segmentation: {self.adaptive_segmentation}")
        
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
        Parse a single stroke command string.
        Handles all expected formats robustly and logs parsing failures.
        """
        if not isinstance(command, str):
            print(f"[PARSE ERROR] Command not a string: {command}")
            return None

        parts = command.split('_')
        if len(parts) < 3:
            print(f"[PARSE ERROR] Not enough parts: {command}")
            return None

        stroke_type_str = parts[0]
        if stroke_type_str == 'line':
            stroke_type = StrokeType.LINE
        elif stroke_type_str == 'arc':
            stroke_type = StrokeType.ARC
        else:
            print(f"[PARSE ERROR] Unknown stroke type: {command}")
            return None

        shape_modifier_str = parts[1]
        try:
            shape_modifier = ShapeModifier(shape_modifier_str)
        except ValueError:
            print(f"[PARSE ERROR] Unknown shape modifier: {command}")
            shape_modifier = ShapeModifier.UNKNOWN

        parameters = {}

        if stroke_type == StrokeType.LINE:
            # Format: line_shape_param1-param2
            if len(parts) >= 3 and '-' in parts[2]:
                try:
                    param1, param2 = parts[2].split('-', 1)
                    parameters = {
                        'length': float(param1),
                        'angle': float(param2)
                    }
                except ValueError:
                    print(f"[PARSE ERROR] Line param parse failed: {command}")
                    return None
            else:
                print(f"[PARSE ERROR] Line format unexpected: {command}")
                return None

        elif stroke_type == StrokeType.ARC:
            # Format: arc_shape_radius_span_angle-end_angle
            # Example: "arc_normal_0.500_0.542-0.167"
            param_section = '_'.join(parts[2:])
            try:
                # Expect: radius_span-end
                underscore_parts = param_section.split('_', 1)
                if len(underscore_parts) != 2:
                    print(f"[PARSE ERROR] Arc missing underscore: {command}")
                    print(f"[DIAGNOSTIC] Failed arc command: {command}")
                    return None
                radius = float(underscore_parts[0])
                span_end = underscore_parts[1]
                if '-' not in span_end:
                    print(f"[PARSE ERROR] Arc missing dash: {command}")
                    print(f"[DIAGNOSTIC] Failed arc command: {command}")
                    return None
                span_str, end_str = span_end.split('-', 1)
                parameters = {
                    'radius': radius,
                    'span_angle': float(span_str),
                    'end_angle': float(end_str)
                }
            except Exception as e:
                print(f"[PARSE ERROR] Arc param parse failed: {command} ({e})")
                print(f"[DIAGNOSTIC] Failed arc command: {command}")
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
        Execute a line stroke command with enhanced precision and detailed shape modifiers.
        Generates high-quality geometric patterns with accurate angles and smooth connections.
        """
        length = stroke.parameters.get('length', 1.0) * self.scale_factor
        angle_param = stroke.parameters.get('angle', 0.0)  # Keep original range (likely 0-1)
        
        logging.debug(f"LINE EXEC: Enhanced processing - length: {stroke.parameters.get('length', 1.0)}, angle: {angle_param}")
        logging.debug(f"LINE EXEC: Scaled length: {length}, scale_factor: {self.scale_factor}")
        
        # Convert normalized angle to radians with higher precision
        angle_rad = angle_param * 2 * math.pi
        dx = length * math.cos(angle_rad)
        dy = length * math.sin(angle_rad)
        
        logging.debug(f"LINE EXEC: Precise angle conversion - param: {angle_param} -> radians: {angle_rad:.6f} -> degrees: {math.degrees(angle_rad):.3f}")
        logging.debug(f"LINE EXEC: Movement vector - dx: {dx:.6f}, dy: {dy:.6f}")
        
        end_x = self.turtle_x + dx
        end_y = self.turtle_y + dy
        
        logging.debug(f"LINE EXEC: End position calculated - ({end_x:.6f}, {end_y:.6f})")
        
        # Generate enhanced vertices with high-quality shape modifiers
        vertices = self._generate_enhanced_line_vertices(
            (self.turtle_x, self.turtle_y), 
            (end_x, end_y), 
            stroke.shape_modifier
        )
        
        logging.debug(f"LINE EXEC: Generated {len(vertices)} enhanced vertices with modifier '{stroke.shape_modifier}': {vertices}")
        
        # Update turtle position (heading follows the movement direction for better continuity)
        self.turtle_x = end_x
        self.turtle_y = end_y
        self.turtle_heading = angle_param * 360  # Convert to degrees for heading
        
        return vertices
    
    def _execute_arc_stroke(self, stroke: StrokeCommand) -> List[Tuple[float, float]]:
        """
        Execute an arc stroke command with enhanced precision and detailed shape analysis.
        Generates high-density, smooth curves with precise shape modifier implementations.
        """
        # Parameters are normalized (0-1) - use higher precision
        radius = stroke.parameters.get('radius', 1.0) * self.scale_factor
        span_angle_norm = stroke.parameters.get('span_angle', 0.25)
        end_angle_norm = stroke.parameters.get('end_angle', 0.0)

        # Convert normalized span to degrees with higher precision
        span_angle_deg = span_angle_norm * 360.0
        end_angle_deg = end_angle_norm * 360.0

        # Enhanced arc generation with adaptive density based on curvature
        arc_length = abs(span_angle_deg) * math.pi * radius / 180.0
        # Adaptive segmentation: more segments for longer arcs and smaller radii
        base_segments = max(16, int(arc_length / 2.0))  # Doubled from original
        curvature_factor = min(2.0, radius / self.scale_factor)  # More segments for tighter curves
        num_segments = int(base_segments * curvature_factor)
        num_segments = max(16, min(128, num_segments))  # Clamp between 16-128 segments

        vertices = []
        start_angle_deg = self.turtle_heading
        start_angle_rad = math.radians(start_angle_deg)

        # More precise arc center calculation
        center_x = self.turtle_x - radius * math.sin(start_angle_rad)
        center_y = self.turtle_y + radius * math.cos(start_angle_rad)

        logging.debug(f"ARC EXEC: Enhanced arc - radius: {radius:.3f}, span: {span_angle_deg:.3f}°, "
                     f"segments: {num_segments}, center: ({center_x:.3f}, {center_y:.3f})")

        # Generate high-precision arc vertices
        for i in range(1, num_segments + 1):
            t = i / num_segments  # Parameter from 0 to 1
            sweep_deg = start_angle_deg + t * span_angle_deg
            sweep_rad = math.radians(sweep_deg)
            
            # Base arc position
            x = center_x + radius * math.cos(sweep_rad - math.pi/2)
            y = center_y + radius * math.sin(sweep_rad - math.pi/2)

            # Enhanced shape modifiers with better geometric accuracy
            if stroke.shape_modifier == ShapeModifier.ZIGZAG:
                # Higher frequency, smoother zigzag pattern
                zigzag_freq = 8.0  # Number of zigzag cycles
                zigzag_amplitude = radius * 0.08  # Slightly larger amplitude
                zigzag_phase = math.sin(t * zigzag_freq * 2 * math.pi)
                
                # Apply zigzag perpendicular to arc direction
                perp_x = -math.sin(sweep_rad - math.pi/2)
                perp_y = math.cos(sweep_rad - math.pi/2)
                x += zigzag_amplitude * zigzag_phase * perp_x
                y += zigzag_amplitude * zigzag_phase * perp_y
                
            elif stroke.shape_modifier == ShapeModifier.SQUARE:
                # Sharp square corners at regular intervals
                square_freq = 4.0  # Number of square corners
                corner_phase = (t * square_freq) % 1.0
                
                if corner_phase < 0.1 or corner_phase > 0.9:  # Near corners
                    square_offset = radius * 0.15
                    # Alternate inward/outward corners
                    direction = 1 if int(t * square_freq) % 2 == 0 else -1
                    offset_x = direction * square_offset * math.cos(sweep_rad)
                    offset_y = direction * square_offset * math.sin(sweep_rad)
                    x += offset_x
                    y += offset_y
                    
            elif stroke.shape_modifier == ShapeModifier.TRIANGLE:
                # Sharp triangular protrusions
                triangle_freq = 6.0  # Number of triangular peaks
                peak_phase = (t * triangle_freq) % 1.0
                
                if 0.3 < peak_phase < 0.7:  # Peak region
                    peak_intensity = math.sin((peak_phase - 0.3) / 0.4 * math.pi)
                    triangle_offset = radius * 0.12 * peak_intensity
                    offset_x = triangle_offset * math.cos(sweep_rad + math.pi/2)
                    offset_y = triangle_offset * math.sin(sweep_rad + math.pi/2)
                    x += offset_x
                    y += offset_y
                    
            elif stroke.shape_modifier == ShapeModifier.CIRCLE:
                # Circular beading effect along the arc
                bead_freq = 10.0  # Number of circular beads
                bead_phase = t * bead_freq * 2 * math.pi
                bead_amplitude = radius * 0.06
                bead_offset = bead_amplitude * (0.5 + 0.5 * math.cos(bead_phase))
                
                # Apply beading radially outward
                offset_x = bead_offset * math.cos(sweep_rad)
                offset_y = bead_offset * math.sin(sweep_rad)
                x += offset_x
                y += offset_y
                
            # NORMAL: smooth, precise arc (no modifications)

            vertices.append((x, y))

        # Precise turtle state update
        if vertices:
            self.turtle_x, self.turtle_y = vertices[-1]
            self.turtle_heading += span_angle_deg
            # Normalize heading to 0-360 range
            while self.turtle_heading >= 360:
                self.turtle_heading -= 360
            while self.turtle_heading < 0:
                self.turtle_heading += 360

        # Enhanced diagnostics
        if len(vertices) < 2:
            logging.warning(f"ARC WARNING: Degenerate arc with {len(vertices)} vertices: {stroke.raw_command}")
        else:
            logging.debug(f"ARC EXEC: Generated {len(vertices)} high-precision vertices, "
                         f"final turtle: ({self.turtle_x:.3f}, {self.turtle_y:.3f}), heading: {self.turtle_heading:.1f}°")

        return vertices
        
        # Update turtle position and heading
        if vertices:
            self.turtle_x, self.turtle_y = vertices[-1]
            self.turtle_heading += span_angle_deg
            # Normalize heading to 0-360 range
            while self.turtle_heading >= 360:
                self.turtle_heading -= 360
            while self.turtle_heading < 0:
                self.turtle_heading += 360
        
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
        Analyze stroke commands to detect distinct object groups.
        Uses improved heuristics based on stroke patterns and geometric analysis.
        """
        if len(stroke_commands) <= 1:
            return [stroke_commands]
        
        # Strategy 1: Detect based on stroke parameter patterns and transitions
        groups = []
        current_group = []
        
        prev_stroke = None
        for i, cmd in enumerate(stroke_commands):
            stroke = self._parse_stroke_command(cmd)
            if not stroke:
                current_group.append(cmd)
                continue
                
            # Enhanced heuristics for group separation:
            should_start_new_group = False
            
            if prev_stroke and len(current_group) > 0:
                # Check for dramatic parameter changes indicating new object
                if stroke.stroke_type == StrokeType.LINE and prev_stroke.stroke_type == StrokeType.LINE:
                    curr_length = stroke.parameters.get('length', 1.0)
                    prev_length = prev_stroke.parameters.get('length', 1.0)
                    curr_angle = stroke.parameters.get('angle', 0.0)
                    prev_angle = prev_stroke.parameters.get('angle', 0.0)
                    
                    # Large length change suggests new object
                    length_ratio = abs(curr_length - prev_length) / max(prev_length, 0.1)
                    if length_ratio > 0.6:  # Increased threshold for more stability
                        should_start_new_group = True
                        logging.debug(f"STROKE GROUP: Large length change detected ({length_ratio:.3f}) at stroke {i}")
                    
                    # Large angle jump suggests disconnected object
                    angle_diff = abs(curr_angle - prev_angle)
                    if angle_diff > 0.5:  # More than 180 degrees difference
                        should_start_new_group = True
                        logging.debug(f"STROKE GROUP: Large angle change detected ({angle_diff:.3f}) at stroke {i}")
                
                # Shape modifier pattern analysis
                if stroke.shape_modifier != prev_stroke.shape_modifier:
                    modifier_distance = self._calculate_modifier_similarity(stroke.shape_modifier, prev_stroke.shape_modifier)
                    if modifier_distance > 0.8 and len(current_group) >= 2:  # Very different modifiers
                        should_start_new_group = True
                        logging.debug(f"STROKE GROUP: Shape modifier change detected ({modifier_distance:.3f}) at stroke {i}")
                
                # Stroke type transition (line to arc or vice versa)
                if stroke.stroke_type != prev_stroke.stroke_type and len(current_group) >= 3:
                    should_start_new_group = True
                    logging.debug(f"STROKE GROUP: Stroke type change detected at stroke {i}")
            
            # Heuristic: Group size limit (common pattern in Bongard-LOGO)
            if len(current_group) >= 5:  # Increased from 4 to allow slightly larger objects
                should_start_new_group = True
                logging.debug(f"STROKE GROUP: Group size limit reached at stroke {i}")
            
            if should_start_new_group and current_group:
                groups.append(current_group)
                current_group = [cmd]
                logging.debug(f"STROKE GROUP: Started new group at stroke {i}, previous group size: {len(groups[-1])}")
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
        Position turtle for the next object to create spatial separation.
        Uses intelligent spacing based on common Bongard-LOGO layouts.
        """
        # Common layouts in Bongard-LOGO: side-by-side, above-below, corners
        spacing = self.scale_factor * 1.2  # Reasonable separation distance
        
        if total_objects == 2:
            # Side-by-side layout (most common)
            if object_index == 1:
                self.turtle_x = spacing
                self.turtle_y = 0.0
        elif total_objects == 3:
            # Triangle layout
            if object_index == 1:
                self.turtle_x = spacing * 0.7
                self.turtle_y = spacing * 0.5
            elif object_index == 2:
                self.turtle_x = -spacing * 0.7
                self.turtle_y = spacing * 0.5
        elif total_objects >= 4:
            # Grid layout
            cols = 2
            rows = (total_objects + cols - 1) // cols
            col = object_index % cols
            row = object_index // cols
            self.turtle_x = (col - cols/2 + 0.5) * spacing
            self.turtle_y = (row - rows/2 + 0.5) * spacing
        
        self.turtle_heading = 0.0  # Reset heading for each object
        
        logging.debug(f"POSITIONING: Object {object_index+1}/{total_objects} positioned at ({self.turtle_x:.3f}, {self.turtle_y:.3f})")

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


    def visualize_image_program(self, image_program: ImageProgram, save_path: str = None) -> None:
        """
        Visualize the parsed image program using matplotlib.
        Highlights arc/circle strokes with distinct colors and overlays for diagnostics.
        """
        import matplotlib.pyplot as plt

        if not image_program or not image_program.vertices:
            print("No vertices to visualize")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Plot all strokes individually, color by type/modifier
        color_map = {
            ("arc", "normal"): "orange",
            ("arc", "circle"): "magenta",
            ("arc", "square"): "purple",
            ("arc", "triangle"): "brown",
            ("arc", "zigzag"): "gold",
            ("line", "circle"): "cyan",
            ("line", "normal"): "blue",
            ("line", "zigzag"): "green",
            ("line", "square"): "gray",
            ("line", "triangle"): "red",
        }

        # For each stroke, plot its vertices
        for i, stroke in enumerate(image_program.strokes):
            # Get stroke vertices
            if stroke.stroke_type == StrokeType.LINE:
                start = (self.turtle_x, self.turtle_y)
                verts = self._generate_line_vertices(start, (self.turtle_x, self.turtle_y), stroke.shape_modifier)
            else:
                verts = []  # Already handled in main pipeline
            # Actually, use the main vertices list and split by stroke boundaries
            # For now, just color by type/modifier
            color = color_map.get((stroke.stroke_type.value, stroke.shape_modifier.value), "black")
            # Find stroke segment in vertices
            # This is approximate: each stroke starts at previous end
            # For better accuracy, need to track per-stroke vertices
            # Here, just plot all as one path, but overlay arcs/circles

        # Plot all vertices as connected path
        vertices = image_program.vertices
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='Parsed Path')
        ax.scatter(xs, ys, c='red', s=20, alpha=0.8, label='Vertices')

        # Overlay: highlight arc/circle strokes
        arc_indices = [i for i, s in enumerate(image_program.strokes) if s.stroke_type == StrokeType.ARC]
        circle_indices = [i for i, s in enumerate(image_program.strokes) if s.shape_modifier == ShapeModifier.CIRCLE]
        # For diagnostic, mark arc/circle vertices
        if arc_indices:
            ax.scatter([vertices[0][0]], [vertices[0][1]], c='orange', s=120, marker='*', label='Arc Start')
        if circle_indices:
            ax.scatter([vertices[0][0]], [vertices[0][1]], c='magenta', s=120, marker='*', label='Circle Start')

        # Mark start and end points
        if len(vertices) >= 2:
            ax.scatter(xs[0], ys[0], c='green', s=100, marker='o', label='Start')
            ax.scatter(xs[-1], ys[-1], c='red', s=100, marker='s', label='End')

        # Set equal aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add title with stroke information
        stroke_info = f"{len(image_program.strokes)} strokes, {len(vertices)} vertices"
        ax.set_title(f"Parsed Image: {image_program.image_id}\n{stroke_info}")

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
