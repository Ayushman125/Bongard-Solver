#!/usr/bin/env python3
"""
Comprehensive NVLabs Bongard-LOGO Parser - WORKING VERSION
This parser uses the NVLabs coordinate system and geometry without hardcoding.
Now using DIRECT NVLabs classes for exact compatibility.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import logging
import json
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Add the NVLabs Bongard-LOGO directory to Python path
nvlabs_path = Path(__file__).parent.parent.parent / "Bongard-LOGO"
sys.path.insert(0, str(nvlabs_path))

# Import the original NVLabs classes
try:
    from bongard.bongard import LineAction as NVLabsLineAction, ArcAction as NVLabsArcAction, OneStrokeShape as NVLabsOneStrokeShape, BasicAction
    print("✅ Successfully imported NVLabs classes")
    NVLABS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NVLabs classes not available, using fallback: {e}")
    NVLABS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Compatibility layer classes (needed for existing imports)
class StrokeType(Enum):
    LINE = "line"
    ARC = "arc"

class ShapeModifier(Enum):
    NORMAL = "normal"
    TRIANGLE = "triangle"
    SQUARE = "square"
    CIRCLE = "circle"
    ZIGZAG = "zigzag"

@dataclass
class StrokeCommand:
    """Compatibility class for existing code."""
    stroke_type: StrokeType
    shape_modifier: ShapeModifier
    length: float
    turn_angle: float
    command_string: str

@dataclass 
class ImageProgram:
    """Compatibility class for existing code."""
    problem_id: str
    image_id: str
    is_positive: bool
    stroke_commands: List[StrokeCommand]
    vertices: Optional[List[Tuple[float, float]]] = None

# NVLabs-compatible action classes
if NVLABS_AVAILABLE:
    # Use the original NVLabs classes directly but extend OneStrokeShape with vertices property
    LineAction = NVLabsLineAction
    ArcAction = NVLabsArcAction
    
    class OneStrokeShape(NVLabsOneStrokeShape):
        """Extended NVLabs OneStrokeShape with vertices property for compatibility."""
        
        def __init__(self, basic_actions, start_coordinates=None, start_orientation=None, scaling_factors=None):
            super().__init__(basic_actions, start_coordinates, start_orientation, scaling_factors)
            self._vertices = None
            
        @property
        def vertices(self):
            """
            Dynamic vertices generation for compatibility with existing codebase.
            This mimics what BongardPainter would generate from basic_actions.
            """
            if self._vertices is None:
                self._vertices = self._calculate_vertices()
            # Ensure we return a proper Python list, not numpy array
            if isinstance(self._vertices, np.ndarray):
                return self._vertices.tolist()
            return self._vertices
        
        def _calculate_vertices(self):
            """Calculate vertices from actions using NVLabs coordinate system with turtle graphics simulation."""
            vertices = []
            current_x, current_y = 0.0, 0.0  # Start at center
            current_angle = 0.0  # Start facing right (0 degrees)
            
            vertices.append((current_x, current_y))
            
            try:
                for action in self.basic_actions:
                    # Turn first (like turtle graphics)
                    if action.turn_direction == "L":
                        current_angle += action.turn_angle
                    elif action.turn_direction == "R":
                        current_angle -= action.turn_angle
                    
                    if isinstance(action, LineAction):
                        # Move forward by line_length in current direction
                        distance = action.line_length
                        angle_rad = math.radians(current_angle)
                        
                        new_x = current_x + distance * math.cos(angle_rad)
                        new_y = current_y + distance * math.sin(angle_rad)
                        
                        vertices.append((new_x, new_y))
                        current_x, current_y = new_x, new_y
                            
                    elif isinstance(action, ArcAction):
                        # Handle arc drawing (approximate with line segments)
                        radius = action.arc_radius
                        arc_angle = action.arc_angle
                        
                        # Approximate arc with line segments
                        num_segments = max(4, int(abs(arc_angle) / 10))
                        angle_step = arc_angle / num_segments
                        
                        for i in range(num_segments):
                            segment_angle = current_angle + (i + 1) * angle_step
                            angle_rad = math.radians(segment_angle)
                            
                            # For arcs, the turtle follows a curved path
                            step_distance = radius * math.radians(abs(angle_step))
                            new_x = current_x + step_distance * math.cos(angle_rad)
                            new_y = current_y + step_distance * math.sin(angle_rad)
                            
                            vertices.append((new_x, new_y))
                            current_x, current_y = new_x, new_y
                        
                        # Update angle after arc
                        current_angle += arc_angle
                            
                return vertices
                
            except Exception as e:
                logger.error(f"Failed to calculate vertices: {e}")
                return []
else:
    # Fallback implementation when NVLabs classes are not available
    class LineAction:
        def __init__(self, line_length, line_type, turn_direction, turn_angle):
            self.line_length = line_length
            self.line_type = line_type
            self.turn_direction = turn_direction
            self.turn_angle = turn_angle
            self.name = "line"
            
        @classmethod
        def import_from_action_string(cls, action_string, line_length_normalization_factor=None):
            """Parse line action string using NVLabs format with 2 or 3 parameters."""
            try:
                movement, turn_angle = action_string.split("-")
                turn_angle = float(turn_angle)
                
                # Line format can be:
                # 2 parts: line_length (e.g., line_1.0) - default "normal" type  
                # 3 parts: line_type_length (e.g., line_triangle_1.000, line_zigzag_0.600, line_circle_0.5)
                parts = movement.split("_")
                if len(parts) == 2:
                    action_name, line_length = parts
                    line_type = "normal"  # Default type for 2-part format
                elif len(parts) == 3:
                    action_name, line_type, line_length = parts
                    # Validate line type - common types: normal, triangle, square, circle, zigzag
                    valid_line_types = {"normal", "triangle", "square", "circle", "zigzag"}
                    if line_type not in valid_line_types:
                        logger.warning(f"Unknown line type '{line_type}' in {movement}, treating as custom type")
                else:
                    logger.error(f"Invalid line format: {movement}, expected 2 or 3 parts, got {len(parts)}: {parts}")
                    return None
                
                line_length = float(line_length)
                
                if line_length_normalization_factor is not None:
                    denormalized_line_length = line_length * line_length_normalization_factor
                else:
                    denormalized_line_length = line_length
                    
                # Denormalize turn angle using NVLabs logic
                # [0,1] -> [-180,180] where 0.5 = 0 degrees
                if 0 <= turn_angle <= 1:
                    if turn_angle >= 0.5:
                        turn_direction = "L"
                        denormalized_turn_angle = (turn_angle - 0.5) * 360
                    else:
                        turn_direction = "R"
                        denormalized_turn_angle = (0.5 - turn_angle) * 360
                else:
                    turn_direction = "L"
                    denormalized_turn_angle = 0
                    
                return cls(denormalized_line_length, line_type, turn_direction, denormalized_turn_angle)
            except Exception as e:
                logger.error(f"Failed to parse line action string '{action_string}': {e}")
                return None
                
        def __str__(self):
            return f"LineAction({self.line_type}, {self.line_length}, {self.turn_direction}{self.turn_angle})"

    class ArcAction:
        def __init__(self, arc_radius, arc_angle, arc_type, turn_direction, turn_angle):
            self.arc_radius = arc_radius
            self.arc_angle = arc_angle
            self.arc_type = arc_type
            self.turn_direction = turn_direction
            self.turn_angle = turn_angle
            self.name = "arc"
            
        @classmethod
        def import_from_action_string(cls, action_string, arc_radius_normalizaton_factor=None):
            """Parse arc action string using NVLabs format with 3 or 4 parameters."""
            try:
                movement, turn_angle = action_string.split("-")
                turn_angle = float(turn_angle)
                
                # Arc format can be:
                # 3 parts: arc_radius_angle (e.g., arc_0.5_30.0) - default "normal" type
                # 4 parts: arc_type_radius_angle (e.g., arc_triangle_0.5_30.0, arc_circle_0.5_30.0)
                parts = movement.split("_")
                if len(parts) == 3:
                    action_name, arc_radius, arc_angle = parts
                    arc_type = "normal"  # Default type for 3-part format
                elif len(parts) == 4:
                    # Full format with explicit arc type
                    action_name, arc_type, arc_radius, arc_angle = parts
                else:
                    logger.error(f"Invalid arc format: {movement}, expected 3 or 4 parts, got {len(parts)}: {parts}")
                    return None
                
                arc_radius = float(arc_radius)
                arc_angle = float(arc_angle)
                
                # Validate arc type - common types: normal, triangle, square, circle, zigzag
                valid_arc_types = {"normal", "triangle", "square", "circle", "zigzag"}
                if arc_type not in valid_arc_types:
                    logger.warning(f"Unknown arc type '{arc_type}' in {movement}, treating as custom type")
                
                if arc_radius_normalizaton_factor is not None:
                    denormalized_arc_radius = arc_radius * arc_radius_normalizaton_factor
                else:
                    denormalized_arc_radius = arc_radius
                    
                # Arc angle normalization for NVLabs compatibility
                # Real data has angles in degrees, NVLabs expects [0,1] normalized
                if arc_angle > 1.0:
                    # Convert from degrees to normalized [0,1] range
                    # 360 degrees = 1.0, so divide by 360
                    normalized_arc_angle = arc_angle / 360.0  # For NVLabs compatibility
                    denormalized_arc_angle = arc_angle  # Keep original degrees for our calculation
                else:
                    # Already normalized
                    normalized_arc_angle = arc_angle
                    denormalized_arc_angle = arc_angle * 360  # Convert to degrees for our calculation
                
                # Denormalize turn angle
                if 0 <= turn_angle <= 1:
                    if turn_angle >= 0.5:
                        turn_direction = "L"
                        denormalized_turn_angle = (turn_angle - 0.5) * 360
                    else:
                        turn_direction = "R"
                        denormalized_turn_angle = (0.5 - turn_angle) * 360
                else:
                    turn_direction = "L"
                    denormalized_turn_angle = 0
                    
                return cls(denormalized_arc_radius, denormalized_arc_angle, arc_type, 
                          turn_direction, denormalized_turn_angle)
            except Exception as e:
                logger.error(f"Failed to parse arc action string '{action_string}': {e}")
                return None
                
        def __str__(self):
            return f"ArcAction({self.arc_type}, {self.arc_radius}, {self.arc_angle}, {self.turn_direction}{self.turn_angle})"

    class OneStrokeShape:
        def __init__(self, basic_actions, start_coordinates=None, start_orientation=None, scaling_factors=None):
            # NVLabs-compatible attributes
            self.basic_actions = basic_actions  # This is what NVLabs uses
            self.shape_actions = basic_actions  # Keep for compatibility
            self.start_coordinates = start_coordinates
            self.start_orientation = start_orientation
            self.scaling_factors = scaling_factors
            
            # Generate vertices dynamically for compatibility with existing code
            self._vertices = None
            
        @property
        def vertices(self):
            """
            Dynamic vertices generation for compatibility with existing codebase.
            This mimics what BongardPainter would generate from basic_actions.
            """
            if self._vertices is None:
                self._vertices = self._calculate_vertices()
            # Ensure we return a proper Python list, not numpy array
            if isinstance(self._vertices, np.ndarray):
                return self._vertices.tolist()
            return self._vertices
        
        def _calculate_vertices(self):
            """Calculate vertices from actions using NVLabs coordinate system."""
            vertices = []
            current_x, current_y = 0.0, 0.0  # Start at center
            current_angle = 0.0  # Start facing right (0 degrees)
            
            vertices.append((current_x, current_y))
            
            try:
                for action in self.basic_actions:
                    if isinstance(action, LineAction):
                        # Different line types create different geometric patterns
                        vertices_to_add = self._calculate_line_vertices(action, current_x, current_y, current_angle)
                        vertices.extend(vertices_to_add)
                        
                        # Update position to the last vertex
                        if vertices_to_add:
                            current_x, current_y = vertices_to_add[-1]
                        
                        # Turn by the specified angle
                        if action.turn_direction == "L":
                            current_angle += action.turn_angle
                        elif action.turn_direction == "R":
                            current_angle -= action.turn_angle
                            
                    elif isinstance(action, ArcAction):
                        # Different arc types create different curved patterns
                        vertices_to_add = self._calculate_arc_vertices(action, current_x, current_y, current_angle)
                        vertices.extend(vertices_to_add)
                        
                        # Update position to the last vertex
                        if vertices_to_add:
                            current_x, current_y = vertices_to_add[-1]
                        
                        # Update angle after arc
                        current_angle += action.arc_angle
                        
                        # Turn by the specified angle
                        if action.turn_direction == "L":
                            current_angle += action.turn_angle
                        elif action.turn_direction == "R":
                            current_angle -= action.turn_angle
                            
                return vertices
                
            except Exception as e:
                logger.error(f"Failed to calculate vertices: {e}")
                return []

        def _calculate_line_vertices(self, action, start_x, start_y, current_angle):
            """Calculate vertices for different line types."""
            vertices = []
            distance = action.line_length
            angle_rad = np.radians(current_angle)
            
            if action.line_type == "normal":
                # Simple straight line
                new_x = start_x + distance * np.cos(angle_rad)
                new_y = start_y + distance * np.sin(angle_rad)
                vertices.append((new_x, new_y))
                
            elif action.line_type == "triangle":
                # Create triangular zigzag pattern
                segments = 3
                segment_length = distance / segments
                for i in range(segments):
                    segment_angle = current_angle + (30 if i % 2 == 0 else -30)  # Alternate angles
                    seg_angle_rad = np.radians(segment_angle)
                    new_x = start_x + segment_length * np.cos(seg_angle_rad)
                    new_y = start_y + segment_length * np.sin(seg_angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.line_type == "square":
                # Create square wave pattern
                segments = 4
                segment_length = distance / segments
                for i in range(segments):
                    segment_angle = current_angle + (90 if i % 2 == 0 else -90)  # Right angles
                    seg_angle_rad = np.radians(segment_angle)
                    new_x = start_x + segment_length * np.cos(seg_angle_rad)
                    new_y = start_y + segment_length * np.sin(seg_angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.line_type == "zigzag":
                # Create zigzag pattern
                segments = 6
                segment_length = distance / segments
                for i in range(segments):
                    segment_angle = current_angle + (45 if i % 2 == 0 else -45)  # Zigzag angles
                    seg_angle_rad = np.radians(segment_angle)
                    new_x = start_x + segment_length * np.cos(seg_angle_rad)
                    new_y = start_y + segment_length * np.sin(seg_angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.line_type == "circle":
                # Approximate circular arc as line segments
                num_segments = 8
                angle_step = 360 / num_segments  # Full circle
                radius = distance / (2 * np.pi)  # Convert length to radius
                for i in range(num_segments):
                    segment_angle = current_angle + i * angle_step
                    seg_angle_rad = np.radians(segment_angle)
                    new_x = start_x + radius * np.cos(seg_angle_rad)
                    new_y = start_y + radius * np.sin(seg_angle_rad)
                    vertices.append((new_x, new_y))
            else:
                # Default to normal line for unknown types
                new_x = start_x + distance * np.cos(angle_rad)
                new_y = start_y + distance * np.sin(angle_rad)
                vertices.append((new_x, new_y))
                
            return vertices

        def _calculate_arc_vertices(self, action, start_x, start_y, current_angle):
            """Calculate vertices for different arc types."""
            vertices = []
            radius = action.arc_radius
            arc_angle = action.arc_angle
            
            if action.arc_type == "normal":
                # Simple arc
                num_segments = max(4, int(abs(arc_angle) / 10))
                angle_step = arc_angle / num_segments
                
                for i in range(num_segments):
                    segment_angle = current_angle + (i + 1) * angle_step
                    angle_rad = np.radians(segment_angle)
                    step_distance = radius * np.radians(abs(angle_step))
                    new_x = start_x + step_distance * np.cos(angle_rad)
                    new_y = start_y + step_distance * np.sin(angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.arc_type == "triangle":
                # Triangular arc pattern - more angular
                num_segments = 3
                angle_step = arc_angle / num_segments
                for i in range(num_segments):
                    segment_angle = current_angle + (i + 1) * angle_step
                    angle_rad = np.radians(segment_angle)
                    # Create angular pattern
                    step_distance = radius * 1.2  # Slightly longer steps for triangular effect
                    new_x = start_x + step_distance * np.cos(angle_rad)
                    new_y = start_y + step_distance * np.sin(angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.arc_type == "square":
                # Square arc pattern - right-angled segments
                num_segments = 4
                angle_step = arc_angle / num_segments
                for i in range(num_segments):
                    segment_angle = current_angle + (i + 1) * angle_step
                    # Snap to 90-degree increments for square effect
                    segment_angle = round(segment_angle / 90) * 90
                    angle_rad = np.radians(segment_angle)
                    step_distance = radius * np.radians(abs(angle_step))
                    new_x = start_x + step_distance * np.cos(angle_rad)
                    new_y = start_y + step_distance * np.sin(angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.arc_type == "zigzag":
                # Zigzag arc pattern
                num_segments = max(6, int(abs(arc_angle) / 15))
                angle_step = arc_angle / num_segments
                for i in range(num_segments):
                    # Add zigzag variation to arc
                    zigzag_offset = 15 if i % 2 == 0 else -15
                    segment_angle = current_angle + (i + 1) * angle_step + zigzag_offset
                    angle_rad = np.radians(segment_angle)
                    step_distance = radius * np.radians(abs(angle_step))
                    new_x = start_x + step_distance * np.cos(angle_rad)
                    new_y = start_y + step_distance * np.sin(angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            elif action.arc_type == "circle":
                # Circular arc - very smooth
                num_segments = max(8, int(abs(arc_angle) / 5))  # More segments for smoothness
                angle_step = arc_angle / num_segments
                center_x = start_x
                center_y = start_y
                
                for i in range(num_segments):
                    segment_angle = current_angle + (i + 1) * angle_step
                    angle_rad = np.radians(segment_angle)
                    new_x = center_x + radius * np.cos(angle_rad)
                    new_y = center_y + radius * np.sin(angle_rad)
                    vertices.append((new_x, new_y))
            else:
                # Default to normal arc for unknown types
                num_segments = max(4, int(abs(arc_angle) / 10))
                angle_step = arc_angle / num_segments
                
                for i in range(num_segments):
                    segment_angle = current_angle + (i + 1) * angle_step
                    angle_rad = np.radians(segment_angle)
                    step_distance = radius * np.radians(abs(angle_step))
                    new_x = start_x + step_distance * np.cos(angle_rad)
                    new_y = start_y + step_distance * np.sin(angle_rad)
                    vertices.append((new_x, new_y))
                    start_x, start_y = new_x, new_y
                    
            return vertices
class ComprehensiveNVLabsParser:
    """
    Comprehensive parser using NVLabs coordinate system and geometry.
    Uses their official coordinate system (-360, 360) and scaling.
    """
    
    def __init__(self, 
                 canvas_size: int = 512,
                 coordinate_range: Tuple[int, int] = (-360, 360),
                 line_width: int = 2):
        """
        Initialize with NVLabs official settings.
        
        Args:
            canvas_size: Canvas size (512x512 to match real dataset)
            coordinate_range: NVLabs coordinate range (-360, 360)
            line_width: Line width for rendering
        """
        self.canvas_size = canvas_size
        self.coordinate_range = coordinate_range
        self.line_width = line_width
        
        # Calculate scaling factor using NVLabs formula
        # Their base_scaling_factor = 180 for 64x64, so for 512x512:
        self.base_scaling_factor = 180 * (canvas_size / 64)
        
        logger.info(f"ComprehensiveNVLabsParser initialized:")
        logger.info(f"  Canvas size: {canvas_size}x{canvas_size}")
        logger.info(f"  Coordinate range: {coordinate_range}")
        logger.info(f"  Base scaling factor: {self.base_scaling_factor}")
        
    def parse_action_commands(self, commands: List[str], problem_id: str) -> Optional[OneStrokeShape]:
        logger.info(f"[LOGOPARSER] Input action_commands for problem_id={problem_id}: {commands}")
        """
        Parse action commands using NVLabs system.
        
        Args:
            commands: List of action command strings
            problem_id: Identifier for the problem
            
        Returns:
            OneStrokeShape object or None if parsing fails
        """
        try:
            logger.debug(f"Parsing {len(commands)} commands for {problem_id}")
            
            # Parse each command using NVLabs parsers
            actions = []
            for cmd in commands:
                try:
                    # Handle nested list case (commands might be wrapped in an extra list)
                    if isinstance(cmd, list):
                        # If it's a nested list, flatten it
                        for sub_cmd in cmd:
                            if isinstance(sub_cmd, str):
                                action = self._parse_single_command(sub_cmd)
                                if action:
                                    actions.append(action)
                        continue
                    elif isinstance(cmd, str):
                        action = self._parse_single_command(cmd)
                        if action:
                            actions.append(action)
                        
                except Exception as e:
                    logger.error(f"Failed to parse command '{cmd}': {e}")
                    continue
                    
            if not actions:
                logger.error(f"No valid actions parsed from {commands}")
                return None
            # Create OneStrokeShape using NVLabs or fallback implementation
            if NVLABS_AVAILABLE:
                shape = OneStrokeShape(
                    basic_actions=actions,
                    start_coordinates=None,
                    start_orientation=None,
                    scaling_factors=None
                )
            else:
                shape = OneStrokeShape(
                    basic_actions=actions,
                    start_coordinates=None,
                    start_orientation=None,
                    scaling_factors=None
                )
            # Log vertices before normalization
            if hasattr(shape, 'vertices'):
                verts = np.array(shape.vertices)
                logger.info(f"[LOGOPARSER] Parsed vertices (raw): {verts.tolist()}")
                if verts.size > 0:
                    min_x, min_y = np.min(verts, axis=0)
                    max_x, max_y = np.max(verts, axis=0)
                    center_x = np.mean(verts[:,0])
                    center_y = np.mean(verts[:,1])
                    logger.info(f"[LOGOPARSER] Vertices bounding box: min=({min_x:.2f},{min_y:.2f}), max=({max_x:.2f},{max_y:.2f}), center=({center_x:.2f},{center_y:.2f})")
            logger.info(f"Successfully created OneStrokeShape with {len(actions)} actions using {'NVLabs' if NVLABS_AVAILABLE else 'fallback'} implementation")
            return shape
            
        except Exception as e:
            logger.error(f"Failed to parse actions for {problem_id}: {e}")
            return None
    
    def _parse_single_command(self, cmd: str):
        """Parse a single command using NVLabs or fallback parsers."""
        if cmd.startswith("line_"):
            # Ensure line commands are in 3-part format for NVLabs compatibility
            normalized_cmd = self._normalize_line_command_for_nvlabs(cmd)
            return LineAction.import_from_action_string(
                normalized_cmd, 
                line_length_normalization_factor=self.base_scaling_factor
            )
        elif cmd.startswith("arc_"):
            # For NVLabs compatibility, we need to normalize the arc angle if it's in degrees
            normalized_cmd = self._normalize_arc_command_for_nvlabs(cmd)
            return ArcAction.import_from_action_string(
                normalized_cmd,
                arc_radius_normalizaton_factor=self.base_scaling_factor
            )
        else:
            logger.warning(f"Unknown command format: {cmd}")
            return None
    
    def _normalize_line_command_for_nvlabs(self, cmd: str) -> str:
        """Normalize line command to NVLabs format (ensure 3-part format)."""
        try:
            if "-" not in cmd:
                return cmd
                
            movement, turn_angle = cmd.split("-")
            parts = movement.split("_")
            
            if len(parts) == 2:
                # Convert 2-part format to 3-part by adding "normal" type
                action_name, line_length = parts
                normalized_movement = f"{action_name}_normal_{line_length}"
                return f"{normalized_movement}-{turn_angle}"
            elif len(parts) == 3:
                # Already in correct format
                return cmd
            else:
                logger.error(f"Invalid line format: {movement}, expected 2 or 3 parts, got {len(parts)}: {parts}")
                return cmd
                
        except Exception as e:
            logger.error(f"Failed to normalize line command {cmd}: {e}")
            return cmd

    def _normalize_arc_command_for_nvlabs(self, cmd: str) -> str:
        """Normalize arc command to NVLabs format (ensure 4-part format with normalized angle)."""
        try:
            if "-" not in cmd:
                return cmd
                
            movement, turn_angle = cmd.split("-")
            parts = movement.split("_")
            
            if len(parts) == 3:
                # arc_radius_angle format - convert to 4-part by adding "normal" type
                action_name, arc_radius, arc_angle = parts
                arc_angle_float = float(arc_angle)
                
                # Convert degrees to normalized [0,1] if needed
                if arc_angle_float > 1.0:
                    normalized_angle = arc_angle_float / 360.0
                else:
                    normalized_angle = arc_angle_float
                    
                # Convert to 4-part format with "normal" type
                normalized_cmd = f"{action_name}_normal_{arc_radius}_{normalized_angle}-{turn_angle}"
                return normalized_cmd
                    
            elif len(parts) == 4:
                # arc_type_radius_angle format  
                action_name, arc_type, arc_radius, arc_angle = parts
                arc_angle_float = float(arc_angle)
                
                # Convert degrees to normalized [0,1] if needed
                if arc_angle_float > 1.0:
                    normalized_angle = arc_angle_float / 360.0
                    normalized_cmd = f"{action_name}_{arc_type}_{arc_radius}_{normalized_angle}-{turn_angle}"
                    return normalized_cmd
                    
            return cmd  # Return original if no normalization needed
            
        except Exception as e:
            logger.warning(f"Failed to normalize arc command '{cmd}': {e}")
            return cmd
    
    def render_shape_to_image(self, shape: OneStrokeShape, output_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        logger.info(f"[LOGOPARSER] Rendering shape to image. Output size: {output_size if output_size else self.canvas_size}")
        """
        Render shape to image using coordinate calculation.
        
        Args:
            shape: OneStrokeShape to render
            output_size: Output image size (defaults to canvas_size)
            
        Returns:
            Rendered image as numpy array or None if rendering fails
        """
        if output_size is None:
            output_size = (self.canvas_size, self.canvas_size)
            
        try:
            # Calculate vertices using NVLabs coordinate system
            vertices = self._calculate_vertices_from_shape(shape)
            if not vertices:
                return None
            # Log vertices after normalization to image coordinates
            verts = np.array(vertices)
            if verts.size > 0:
                min_x, min_y = np.min(verts, axis=0)
                max_x, max_y = np.max(verts, axis=0)
                center_x = np.mean(verts[:,0])
                center_y = np.mean(verts[:,1])
                logger.info(f"[LOGOPARSER] Render vertices (raw): {verts.tolist()}")
                logger.info(f"[LOGOPARSER] Render vertices bounding box: min=({min_x:.2f},{min_y:.2f}), max=({max_x:.2f},{max_y:.2f}), center=({center_x:.2f},{center_y:.2f})")
            # Create image
            img = np.zeros(output_size, dtype=np.uint8)
            img.fill(255)  # White background
            # Convert NVLabs coordinates to image coordinates
            img_vertices = []
            for x, y in vertices:
                # Map from (-360, 360) to (0, canvas_size)
                # CRITICAL FIX: Correct Y coordinate mapping - use Y coordinate range, not X range for Y
                coord_range_size = self.coordinate_range[1] - self.coordinate_range[0]
                img_x = int((x - self.coordinate_range[0]) / coord_range_size * output_size[1])
                img_y = int((y - self.coordinate_range[0]) / coord_range_size * output_size[0])
                # Clamp to image bounds
                img_x = max(0, min(output_size[1] - 1, img_x))
                img_y = max(0, min(output_size[0] - 1, img_y))
                img_vertices.append((img_x, img_y))
            # Log image vertices (after mapping)
            img_verts_np = np.array(img_vertices)
            if img_verts_np.size > 0:
                min_x, min_y = np.min(img_verts_np, axis=0)
                max_x, max_y = np.max(img_verts_np, axis=0)
                center_x = np.mean(img_verts_np[:,0])
                center_y = np.mean(img_verts_np[:,1])
                logger.info(f"[LOGOPARSER] Image vertices: {img_vertices}")
                logger.info(f"[LOGOPARSER] Image vertices bounding box: min=({min_x},{min_y}), max=({max_x},{max_y}), center=({center_x},{center_y})")
            # Draw lines between consecutive vertices
            for i in range(len(img_vertices) - 1):
                cv2.line(img, img_vertices[i], img_vertices[i + 1], 0, self.line_width)
            logger.debug(f"Rendered {len(vertices)} vertices -> {output_size} image with {np.count_nonzero(img == 0)} black pixels")
            logger.info(f"[LOGOPARSER] Output image shape: {img.shape}, nonzero pixels: {np.count_nonzero(img == 0)}")
            return img
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return None
    
    def _calculate_vertices_from_shape(self, shape: OneStrokeShape) -> List[Tuple[float, float]]:
        """
        Calculate vertices from shape using NVLabs coordinate system.
        Uses turtle graphics simulation for exact compatibility.
        """
        vertices = []
        current_x, current_y = 0.0, 0.0  # Start at center
        current_angle = 0.0  # Start facing right (0 degrees)
        
        vertices.append((current_x, current_y))
        
        try:
            for action in shape.basic_actions:
                # Turn first (like turtle graphics)
                if action.turn_direction == "L":
                    current_angle += action.turn_angle
                elif action.turn_direction == "R":
                    current_angle -= action.turn_angle
                
                if isinstance(action, LineAction):
                    # Use shape-specific line calculation
                    vertices_to_add = self._calculate_line_vertices_for_parser(action, current_x, current_y, current_angle)
                    vertices.extend(vertices_to_add)
                    
                    # Update position to the last vertex
                    if vertices_to_add:
                        current_x, current_y = vertices_to_add[-1]
                        
                elif isinstance(action, ArcAction):
                    # Use shape-specific arc calculation
                    vertices_to_add = self._calculate_arc_vertices_for_parser(action, current_x, current_y, current_angle)
                    vertices.extend(vertices_to_add)
                    
                    # Update position to the last vertex
                    if vertices_to_add:
                        current_x, current_y = vertices_to_add[-1]
                    
                    # Update angle after arc
                    current_angle += action.arc_angle
                        
            return vertices
            
        except Exception as e:
            logger.error(f"Failed to calculate vertices: {e}")
            return []

    def _calculate_line_vertices_for_parser(self, action, start_x, start_y, current_angle):
        """Calculate vertices for different line types - parser version."""
        vertices = []
        distance = action.line_length
        angle_rad = math.radians(current_angle)
        
        if action.line_type == "normal":
            # Simple straight line
            new_x = start_x + distance * math.cos(angle_rad)
            new_y = start_y + distance * math.sin(angle_rad)
            vertices.append((new_x, new_y))
            
        elif action.line_type == "triangle":
            # Create triangular pattern
            segments = 3
            segment_length = distance / segments
            for i in range(segments):
                segment_angle = current_angle + (30 if i % 2 == 0 else -30)  # Alternate angles
                seg_angle_rad = math.radians(segment_angle)
                new_x = start_x + segment_length * math.cos(seg_angle_rad)
                new_y = start_y + segment_length * math.sin(seg_angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.line_type == "square":
            # Create square wave pattern
            segments = 4
            segment_length = distance / segments
            for i in range(segments):
                segment_angle = current_angle + (90 if i % 2 == 0 else -90)  # Right angles
                seg_angle_rad = math.radians(segment_angle)
                new_x = start_x + segment_length * math.cos(seg_angle_rad)
                new_y = start_y + segment_length * math.sin(seg_angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.line_type == "zigzag":
            # Create zigzag pattern
            segments = 6
            segment_length = distance / segments
            for i in range(segments):
                segment_angle = current_angle + (45 if i % 2 == 0 else -45)  # Zigzag angles
                seg_angle_rad = math.radians(segment_angle)
                new_x = start_x + segment_length * math.cos(seg_angle_rad)
                new_y = start_y + segment_length * math.sin(seg_angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.line_type == "circle":
            # Approximate circular arc as line segments
            num_segments = 8
            angle_step = 360 / num_segments  # Full circle
            radius = distance / (2 * math.pi)  # Convert length to radius
            for i in range(num_segments):
                segment_angle = current_angle + i * angle_step
                seg_angle_rad = math.radians(segment_angle)
                new_x = start_x + radius * math.cos(seg_angle_rad)
                new_y = start_y + radius * math.sin(seg_angle_rad)
                vertices.append((new_x, new_y))
        else:
            # Default to normal line for unknown types
            new_x = start_x + distance * math.cos(angle_rad)
            new_y = start_y + distance * math.sin(angle_rad)
            vertices.append((new_x, new_y))
            
        return vertices

    def _calculate_arc_vertices_for_parser(self, action, start_x, start_y, current_angle):
        """Calculate vertices for different arc types - parser version."""
        vertices = []
        radius = action.arc_radius
        arc_angle = action.arc_angle
        
        if action.arc_type == "normal":
            # Simple arc
            num_segments = max(4, int(abs(arc_angle) / 10))
            angle_step = arc_angle / num_segments
            
            for i in range(num_segments):
                segment_angle = current_angle + (i + 1) * angle_step
                angle_rad = math.radians(segment_angle)
                step_distance = radius * math.radians(abs(angle_step))
                new_x = start_x + step_distance * math.cos(angle_rad)
                new_y = start_y + step_distance * math.sin(angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.arc_type == "triangle":
            # Triangular arc pattern - more angular
            num_segments = 3
            angle_step = arc_angle / num_segments
            for i in range(num_segments):
                segment_angle = current_angle + (i + 1) * angle_step
                angle_rad = math.radians(segment_angle)
                step_distance = radius * 1.2  # Slightly longer for triangular effect
                new_x = start_x + step_distance * math.cos(angle_rad)
                new_y = start_y + step_distance * math.sin(angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.arc_type == "square":
            # Square arc pattern - right-angled segments
            num_segments = 4
            angle_step = arc_angle / num_segments
            for i in range(num_segments):
                segment_angle = current_angle + (i + 1) * angle_step
                # Snap to 90-degree increments for square effect
                segment_angle = round(segment_angle / 90) * 90
                angle_rad = math.radians(segment_angle)
                step_distance = radius * math.radians(abs(angle_step))
                new_x = start_x + step_distance * math.cos(angle_rad)
                new_y = start_y + step_distance * math.sin(angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.arc_type == "zigzag":
            # Zigzag arc pattern
            num_segments = max(6, int(abs(arc_angle) / 15))
            angle_step = arc_angle / num_segments
            for i in range(num_segments):
                # Add zigzag variation to arc
                zigzag_offset = 15 if i % 2 == 0 else -15
                segment_angle = current_angle + (i + 1) * angle_step + zigzag_offset
                angle_rad = math.radians(segment_angle)
                step_distance = radius * math.radians(abs(angle_step))
                new_x = start_x + step_distance * math.cos(angle_rad)
                new_y = start_y + step_distance * math.sin(angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        elif action.arc_type == "circle":
            # Circular arc - very smooth
            num_segments = max(8, int(abs(arc_angle) / 5))  # More segments for smoothness
            angle_step = arc_angle / num_segments
            center_x = start_x
            center_y = start_y
            
            for i in range(num_segments):
                segment_angle = current_angle + (i + 1) * angle_step
                angle_rad = math.radians(segment_angle)
                new_x = center_x + radius * math.cos(angle_rad)
                new_y = center_y + radius * math.sin(angle_rad)
                vertices.append((new_x, new_y))
        else:
            # Default to normal arc for unknown types
            num_segments = max(4, int(abs(arc_angle) / 10))
            angle_step = arc_angle / num_segments
            
            for i in range(num_segments):
                segment_angle = current_angle + (i + 1) * angle_step
                angle_rad = math.radians(segment_angle)
                step_distance = radius * math.radians(abs(angle_step))
                new_x = start_x + step_distance * math.cos(angle_rad)
                new_y = start_y + step_distance * math.sin(angle_rad)
                vertices.append((new_x, new_y))
                start_x, start_y = new_x, new_y
                
        return vertices
    
    def process_action_commands_to_image(self, commands: List[str], problem_id: str, 
                                       output_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Complete pipeline: parse commands and render to image.
        """
        # Parse commands to shape
        shape = self.parse_action_commands(commands, problem_id)
        if shape is None:
            return None
            
        # Render to image
        image = self.render_shape_to_image(shape, output_size)
        return image


    def _render_vertices_to_image(self, vertices: List[Tuple[float, float]], 
                                image_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """
        Render vertices to image for compatibility with ActionMaskGenerator.
        This method is called by the hybrid system.
        """
        if image_size is None:
            image_size = (self.canvas_size, self.canvas_size)
            
        try:
            if not vertices or len(vertices) < 2:
                logger.warning("No vertices or insufficient vertices to render")
                return None
                
            # Create image
            img = np.zeros(image_size, dtype=np.uint8)
            img.fill(255)  # White background
            
            # Convert NVLabs coordinates to image coordinates
            img_vertices = []
            for x, y in vertices:
                # Map from (-360, 360) to (0, image_size)
                img_x = int((x - self.coordinate_range[0]) / (self.coordinate_range[1] - self.coordinate_range[0]) * image_size[1])
                img_y = int((y - self.coordinate_range[0]) / (self.coordinate_range[1] - self.coordinate_range[0]) * image_size[0])
                
                # Clamp to image bounds
                img_x = max(0, min(image_size[1] - 1, img_x))
                img_y = max(0, min(image_size[0] - 1, img_y))
                
                img_vertices.append((img_x, img_y))
            
            # Draw lines between consecutive vertices
            for i in range(len(img_vertices) - 1):
                cv2.line(img, img_vertices[i], img_vertices[i + 1], 0, self.line_width)
                
            logger.debug(f"Rendered {len(vertices)} vertices -> {image_size} image with {np.count_nonzero(img == 0)} black pixels")
            return img
            
        except Exception as e:
            logger.error(f"_render_vertices_to_image failed: {e}")
            return None
    
    def _parse_single_image_nvlabs(self, stroke_commands: List[str], image_id: str, 
                                  image_height: int = 512, image_width: int = 512,
                                  canvas_size: int = 512, is_positive: bool = True,
                                  problem_id: str = None) -> Optional[np.ndarray]:
        """
        NVLabs-compatible single image parsing method.
        This is the method called by ActionMaskGenerator.
        """
        logger.debug(f"_parse_single_image_nvlabs called for {image_id}")
        
        # Update canvas size if provided
        if canvas_size != self.canvas_size:
            self.canvas_size = canvas_size
            self.base_scaling_factor = 180 * (canvas_size / 64)
        
        # Parse the action commands
        shape = self.parse_action_commands(stroke_commands, image_id)
        if shape is None:
            logger.warning(f"Failed to parse action commands for {image_id}")
            return None
        
        # Render to image
        rendered_image = self.render_shape_to_image(shape, (image_height, image_width))
        if rendered_image is None:
            logger.warning(f"Failed to render image for {image_id}")
            return None
            
        logger.debug(f"Successfully parsed and rendered {image_id}: {rendered_image.shape}")
        return rendered_image

# Compatibility classes for existing imports
class UnifiedActionParser:
    """
    Compatibility wrapper around ComprehensiveNVLabsParser.
    Maintains the same interface as the old parser for existing code.
    """
    
    def __init__(self, canvas_size=512, coordinate_range=(-360, 360)):
        """Initialize with NVLabs parser."""
        self.comprehensive_parser = ComprehensiveNVLabsParser(
            canvas_size=canvas_size,
            coordinate_range=coordinate_range
        )
        self.canvas_size = canvas_size
        
    def parse_action_programs_json(self, file_path: str) -> Dict[str, Any]:
        """Parse action programs from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            return {}
    
    def parse_single_image(self, commands: List[str], problem_id: str, 
                          image_id: str, is_positive: bool) -> Optional[ImageProgram]:
        """Parse a single image program for compatibility."""
        try:
            # Convert commands to stroke commands for compatibility
            stroke_commands = []
            for cmd in commands:
                if cmd.startswith("line_"):
                    parts = cmd.split("_")
                    if len(parts) >= 3:
                        try:
                            shape_modifier = ShapeModifier(parts[1])
                        except ValueError:
                            shape_modifier = ShapeModifier.NORMAL
                        length_turn = parts[2].split("-")
                        if len(length_turn) == 2:
                            length = float(length_turn[0])
                            turn_angle = float(length_turn[1])
                            stroke_commands.append(StrokeCommand(
                                stroke_type=StrokeType.LINE,
                                shape_modifier=shape_modifier,
                                length=length,
                                turn_angle=turn_angle,
                                command_string=cmd
                            ))
                elif cmd.startswith("arc_"):
                    stroke_commands.append(StrokeCommand(
                        stroke_type=StrokeType.ARC,
                        shape_modifier=ShapeModifier.NORMAL,
                        length=1.0,
                        turn_angle=0.0,
                        command_string=cmd
                    ))
            
            return ImageProgram(
                problem_id=problem_id,
                image_id=image_id,
                is_positive=is_positive,
                stroke_commands=stroke_commands
            )
            
        except Exception as e:
            logger.error(f"Failed to parse single image: {e}")
            return None
    
    def render_image_program(self, image_program: ImageProgram, 
                           output_size: Tuple[int, int] = None) -> Optional[np.ndarray]:
        """Render image program using comprehensive parser."""
        commands = [stroke.command_string for stroke in image_program.stroke_commands]
        return self.comprehensive_parser.process_action_commands_to_image(
            commands, image_program.problem_id, output_size
        )
    
    def _parse_single_image(self, stroke_commands: List[str], image_id: str, 
                          image_height: int = 512, image_width: int = 512,
                          canvas_size: int = 512, is_positive: bool = True,
                          problem_id: str = None) -> Optional[np.ndarray]:
        """
        Compatibility method for ActionMaskGenerator.
        """
        return self.comprehensive_parser._parse_single_image_nvlabs(
            stroke_commands, image_id, image_height, image_width, 
            canvas_size, is_positive, problem_id
        )


class BongardLogoParser:
    """
    Compatibility wrapper for BongardLogoParser.
    """
    
    def __init__(self, canvas_size=512):
        """Initialize with NVLabs parser."""
        self.comprehensive_parser = ComprehensiveNVLabsParser(canvas_size=canvas_size)
        self.canvas_size = canvas_size
    
    def parse_problem(self, problem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Bongard problem."""
        return problem_data
    
    def render_commands(self, commands: List[str], problem_id: str = "default") -> Optional[np.ndarray]:
        """Render commands to image."""
        return self.comprehensive_parser.process_action_commands_to_image(
            commands, problem_id, (self.canvas_size, self.canvas_size)
        )


# For backwards compatibility, create aliases
NVLabsActionParser = ComprehensiveNVLabsParser


def test_comprehensive_parser():
    """Test the comprehensive parser with sample commands."""
    print("=== TESTING COMPREHENSIVE NVLABS PARSER ===")
        
    # Initialize parser with 512x512 to match real dataset
    parser = ComprehensiveNVLabsParser(
        canvas_size=512,
        coordinate_range=(-360, 360)
    )
    
    # Test with sample commands
    test_commands = [
        "line_triangle_1.000-0.500",
        "line_normal_0.600-0.750",
        "line_zigzag_0.424-0.875"
    ]
    
    print(f"Testing with commands: {test_commands}")
    
    # Process commands to image
    image = parser.process_action_commands_to_image(
        test_commands, 
        "comprehensive_test",
        output_size=(512, 512)
    )
    
    if image is not None:
        print(f"✅ Successfully generated {image.shape} image")
        print(f"   Non-zero pixels: {np.count_nonzero(image)}")
        
        # Save test image
        cv2.imwrite('comprehensive_parser_test.png', image)
        print("✅ Saved: comprehensive_parser_test.png")
        
        return True
    else:
        print("❌ Failed to generate image")
        return False


if __name__ == "__main__":
    test_comprehensive_parser()
