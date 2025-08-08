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

# Import the original NVLabs classes DIRECTLY - no redefinitions needed!
try:
    from bongard import LineAction, ArcAction, OneStrokeShape, BongardImage
    from bongard.bongard import BasicAction  # Import BasicAction directly from bongard.py
    from bongard.bongard_painter import BongardImagePainter, BongardProblemPainter, BongardShapePainter
    print("✅ Successfully imported ALL NVLabs classes directly from Bongard-LOGO repo")
    NVLABS_AVAILABLE = True
    
    # We can use these directly without any wrapper classes!
    NVLabsLineAction = LineAction
    NVLabsArcAction = ArcAction
    NVLabsOneStrokeShape = OneStrokeShape
    
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
                        # --- FIX: Ensure line_length is properly scaled ---
                        distance = action.line_length
                        if hasattr(action, 'line_length_normalization_factor') and action.line_length_normalization_factor:
                            distance *= action.line_length_normalization_factor
                        angle_rad = math.radians(current_angle)
                        new_x = current_x + distance * math.cos(angle_rad)
                        new_y = current_y + distance * math.sin(angle_rad)
                        vertices.append((new_x, new_y))
                        current_x, current_y = new_x, new_y
                    elif isinstance(action, ArcAction):
                        # Handle arc drawing (approximate with line segments)
                        radius = action.arc_radius
                        if hasattr(action, 'arc_radius_normalizaton_factor') and action.arc_radius_normalizaton_factor:
                            radius *= action.arc_radius_normalizaton_factor
                        arc_angle = action.arc_angle
                        num_segments = max(4, int(abs(arc_angle) / 10))
                        angle_step = arc_angle / num_segments
                        for i in range(num_segments):
                            segment_angle = current_angle + (i + 1) * angle_step
                            angle_rad = math.radians(segment_angle)
                            step_distance = radius * math.radians(abs(angle_step))
                            new_x = current_x + step_distance * math.cos(angle_rad)
                            new_y = current_y + step_distance * math.sin(angle_rad)
                            vertices.append((new_x, new_y))
                            current_x, current_y = new_x, new_y
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
            """Parse line action string using NVLabs format."""
            try:
                movement, turn_angle = action_string.split("-")
                turn_angle = float(turn_angle)
                action_name, line_type, line_length = movement.split("_")
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
            """Parse arc action string using NVLabs format."""
            try:
                movement, turn_angle = action_string.split("-")
                turn_angle = float(turn_angle)
                action_name, arc_type, arc_radius, arc_angle = movement.split("_")
                arc_radius = float(arc_radius)
                arc_angle = float(arc_angle)
                
                if arc_radius_normalizaton_factor is not None:
                    denormalized_arc_radius = arc_radius * arc_radius_normalizaton_factor
                else:
                    denormalized_arc_radius = arc_radius
                    
                denormalized_arc_angle = arc_angle * 360  # Assuming normalized
                
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
                        # Move forward by line_length in current direction
                        distance = action.line_length
                        angle_rad = np.radians(current_angle)
                        
                        new_x = current_x + distance * np.cos(angle_rad)
                        new_y = current_y + distance * np.sin(angle_rad)
                        
                        vertices.append((new_x, new_y))
                        current_x, current_y = new_x, new_y
                        
                        # Turn by the specified angle
                        if action.turn_direction == "L":
                            current_angle += action.turn_angle
                        elif action.turn_direction == "R":
                            current_angle -= action.turn_angle
                            
                    elif isinstance(action, ArcAction):
                        # Handle arc drawing (simplified as line segments)
                        radius = action.arc_radius
                        arc_angle = action.arc_angle
                        
                        # Approximate arc with line segments
                        num_segments = max(4, int(abs(arc_angle) / 10))
                        angle_step = arc_angle / num_segments
                        
                        for i in range(num_segments):
                            segment_angle = current_angle + (i + 1) * angle_step
                            angle_rad = np.radians(segment_angle)
                            
                            new_x = current_x + radius * np.cos(angle_rad)
                            new_y = current_y + radius * np.sin(angle_rad)
                            
                            vertices.append((new_x, new_y))
                            current_x, current_y = new_x, new_y
                        
                        # Update angle after arc
                        current_angle += arc_angle
                        
                        # Turn by the specified angle
                        if action.turn_direction == "L":
                            current_angle += action.turn_angle
                        elif action.turn_direction == "R":
                            current_angle -= action.turn_angle
                            
                return vertices
                
            except Exception as e:
                logger.error(f"Failed to calculate vertices: {e}")
                return []


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
            for idx, cmd in enumerate(commands):
                try:
                    # Handle nested list case (commands might be wrapped in an extra list)
                    if isinstance(cmd, list):
                        # If it's a nested list, flatten it
                        for sub_cmd in cmd:
                            if isinstance(sub_cmd, str):
                                action = self._parse_single_command(sub_cmd)
                                if action:
                                    action.raw_command = sub_cmd  # PATCH: Set raw_command for each action
                                    actions.append(action)
                        continue
                    elif isinstance(cmd, str):
                        action = self._parse_single_command(cmd)
                        if action:
                            action.raw_command = cmd  # PATCH: Set raw_command for each action
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
            # --- Patch: Log degenerate shapes ---
            num_vertices = len(getattr(shape, 'vertices', [])) if hasattr(shape, 'vertices') else 0
            if num_vertices < 3:
                logger.warning(f"[LOGOPARSER] Degenerate shape detected for problem_id={problem_id}: num_vertices={num_vertices}, actions={actions}")
            else:
                logger.info(f"[LOGOPARSER] Shape for problem_id={problem_id} has {num_vertices} vertices.")
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
            return LineAction.import_from_action_string(
                cmd, 
                line_length_normalization_factor=self.base_scaling_factor
            )
        elif cmd.startswith("arc_"):
            return ArcAction.import_from_action_string(
                cmd,
                arc_radius_normalizaton_factor=self.base_scaling_factor
            )
        else:
            logger.warning(f"Unknown command format: {cmd}")
            return None
    
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
                img_x = int((x - self.coordinate_range[0]) / (self.coordinate_range[1] - self.coordinate_range[0]) * output_size[1])
                img_y = int((y - self.coordinate_range[0]) / (self.coordinate_range[1] - self.coordinate_range[0]) * output_size[0])
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
