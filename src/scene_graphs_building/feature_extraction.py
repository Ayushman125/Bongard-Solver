"""
Feature Extraction Module for Bongard-LOGO Dataset

This module computes physics attributes for shapes with proper differentiation 
between arc and line stroke types. Updated to only handle the 5 discovered 
Bongard-LOGO shape types with their proper geometric calculations.

Shape type frequencies from dataset analysis:
- normal: 24,107 occurrences (48.7%)
- circle: 6,256 occurrences (12.6%) 
- square: 6,519 occurrences (13.2%)
- triangle: 5,837 occurrences (11.8%)
- zigzag: 6,729 occurrences (13.6%)
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def compute_physics_attributes(vertices_or_node_data, stroke_type: str = None) -> Dict[str, Any]:
    """
    Compute physics attributes for a shape based on vertices and stroke type.
    
    Can be called in two ways:
    1. compute_physics_attributes(vertices, stroke_type) - new interface
    2. compute_physics_attributes(node_data) - backward compatible interface
    
    Key insight: Arc vs line stroke types have different geometric properties:
    - Line strokes: Sharp corners, discrete vertices, angular calculations
    - Arc strokes: Smooth curves, interpolated paths, curvature calculations
    
    Args:
        vertices_or_node_data: numpy array of shape vertices OR node data dict
        stroke_type: 'line' or 'arc' indicating the stroke type (optional for node_data)
        
    Returns:
        Dictionary of physics attributes
    """
    # Handle backward compatibility
    if isinstance(vertices_or_node_data, dict):
        # Called with node_data (old interface)
        node_data = vertices_or_node_data
        vertices = node_data.get('vertices', [])
        
        # Determine stroke type from node_data
        if stroke_type is None:
            stroke_type = node_data.get('stroke_type', 'unknown')
            if stroke_type == 'unknown':
                # Try to infer from action_command
                action_command = node_data.get('action_command', '')
                if action_command.startswith('line_'):
                    stroke_type = 'line'
                elif action_command.startswith('arc_'):
                    stroke_type = 'arc'
                else:
                    stroke_type = 'line'  # default
        
        # Convert vertices to numpy array
        if isinstance(vertices, list):
            vertices = np.array(vertices) if vertices else np.array([])
        
        # Update node_data with computed attributes
        attributes = _compute_attributes_internal(vertices, stroke_type)
        node_data.update(attributes)
        return node_data
    else:
        # Called with vertices (new interface)
        vertices = vertices_or_node_data
        if stroke_type is None:
            stroke_type = 'line'  # default
        return _compute_attributes_internal(vertices, stroke_type)

def _compute_attributes_internal(vertices: np.ndarray, stroke_type: str) -> Dict[str, Any]:
    """
    Compute physics attributes for a shape based on vertices and stroke type.
    
    Key insight: Arc vs line stroke types have different geometric properties:
    - Line strokes: Sharp corners, discrete vertices, angular calculations
    - Arc strokes: Smooth curves, interpolated paths, curvature calculations
    
    Args:
        vertices: numpy array of shape vertices
        stroke_type: 'line' or 'arc' indicating the stroke type
        
    Returns:
        Dictionary of physics attributes
    """
    if vertices is None or len(vertices) == 0:
        return _get_default_attributes()
    
    # Ensure vertices is a 2D array
    if len(vertices.shape) == 1:
        vertices = vertices.reshape(-1, 2)
    
    try:
        # Basic geometric properties
        centroid = np.mean(vertices, axis=0)
        
        # Different calculations based on stroke type
        if stroke_type == 'arc':
            return _compute_arc_attributes(vertices, centroid)
        else:  # line stroke type
            return _compute_line_attributes(vertices, centroid)
            
    except Exception as e:
        logger.warning(f"Error computing physics attributes: {e}")
        return _get_default_attributes()

def _compute_arc_attributes(vertices: np.ndarray, centroid: np.ndarray) -> Dict[str, Any]:
    """Compute attributes for arc-based shapes (smooth curves)"""
    try:
        # For arc strokes, focus on curvature and smoothness
        distances = np.linalg.norm(vertices - centroid, axis=1)
        radius = np.mean(distances)
        radius_variance = np.var(distances) if len(distances) > 1 else 0.0
        
        # Compute curvature approximation
        if len(vertices) >= 3:
            # Use change in direction between consecutive segments
            vectors = np.diff(vertices, axis=0)
            if len(vectors) >= 2:
                angles = []
                for i in range(len(vectors) - 1):
                    v1, v2 = vectors[i], vectors[i + 1]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angles.append(np.arccos(cos_angle))
                curvature = np.mean(angles) if angles else 0.0
            else:
                curvature = 0.0
        else:
            curvature = 0.0
        
        # Arc-specific features
        circumference = 2 * np.pi * radius
        area = np.pi * radius**2
        
        return {
            'area': float(area),
            'perimeter': float(circumference),
            'centroid_x': float(centroid[0]),
            'centroid_y': float(centroid[1]),
            'radius': float(radius),
            'radius_variance': float(radius_variance),
            'curvature': float(curvature),
            'curvature_score': float(curvature),  # Ensure consistent naming
            'max_curvature': float(curvature),    # For arcs, max = average curvature
            'orientation_variance': float(radius_variance),  # Use radius variance as orientation measure
            'is_arc_stroke': True,
            'is_line_stroke': False,
            'stroke_type': 'arc',
            'vertices_count': len(vertices),
            'regularity': float(1.0 - radius_variance / (radius + 1e-6))
        }
        
    except Exception as e:
        logger.warning(f"Error in arc attribute computation: {e}")
        return _get_default_attributes()

def _compute_line_attributes(vertices: np.ndarray, centroid: np.ndarray) -> Dict[str, Any]:
    """Compute attributes for line-based shapes (angular, discrete)"""
    try:
        # For line strokes, focus on angles and discrete segments
        perimeter = 0.0
        angles = []
        
        if len(vertices) >= 3:
            # Compute perimeter
            for i in range(len(vertices)):
                next_i = (i + 1) % len(vertices)
                perimeter += np.linalg.norm(vertices[next_i] - vertices[i])
            
            # Compute internal angles
            for i in range(len(vertices)):
                prev_i = (i - 1) % len(vertices)
                next_i = (i + 1) % len(vertices)
                
                v1 = vertices[i] - vertices[prev_i]
                v2 = vertices[next_i] - vertices[i]
                
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angles.append(np.arccos(cos_angle))
        
        # Compute area using shoelace formula
        area = 0.0
        if len(vertices) >= 3:
            for i in range(len(vertices)):
                j = (i + 1) % len(vertices)
                area += vertices[i][0] * vertices[j][1]
                area -= vertices[j][0] * vertices[i][1]
            area = abs(area) / 2.0
        
        # Line-specific features
        angle_variance = np.var(angles) if angles else 0.0
        mean_angle = np.mean(angles) if angles else 0.0
        
        # Measure regularity based on angle consistency
        if len(angles) > 0:
            expected_angle = (len(vertices) - 2) * np.pi / len(vertices) if len(vertices) >= 3 else np.pi
            angle_deviation = np.mean([abs(a - expected_angle) for a in angles])
            regularity = max(0.0, 1.0 - angle_deviation / np.pi)
        else:
            regularity = 0.5
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'centroid_x': float(centroid[0]),
            'centroid_y': float(centroid[1]),
            'mean_angle': float(mean_angle),
            'angle_variance': float(angle_variance),
            'is_arc_stroke': False,
            'is_line_stroke': True,
            'stroke_type': 'line',
            'vertices_count': len(vertices),
            'regularity': float(regularity),
            'angles': [float(a) for a in angles]
        }
        
    except Exception as e:
        logger.warning(f"Error in line attribute computation: {e}")
        return _get_default_attributes()

def _get_default_attributes() -> Dict[str, Any]:
    """Return default attributes when computation fails"""
    return {
        'area': 0.0,
        'perimeter': 0.0,
        'centroid_x': 0.0,
        'centroid_y': 0.0,
        'is_arc_stroke': False,
        'is_line_stroke': False,
        'stroke_type': 'unknown',
        'vertices_count': 0,
        'regularity': 0.0
    }

def extract_line_segments(action_program):
    """
    Extract line segments from action program for polygon decomposition.
    This is a simplified implementation for profiling purposes.
    """
    if not action_program:
        return []
    
    segments = []
    current_pos = np.array([0.0, 0.0])
    
    try:
        for action in action_program:
            if isinstance(action, dict):
                action_type = action.get('type', action.get('command_type', ''))
                
                if action_type == 'start':
                    current_pos = np.array([action.get('x', 0.0), action.get('y', 0.0)])
                elif action_type == 'line':
                    # Simple line segment
                    end_pos = current_pos + np.array([action.get('x', 0.0), action.get('y', 0.0)])
                    segments.append([current_pos.tolist(), end_pos.tolist()])
                    current_pos = end_pos
                elif action_type == 'arc':
                    # Approximate arc as line segment for simplicity
                    angle = action.get('angle', 0.0)
                    radius = action.get('radius', 1.0)
                    end_pos = current_pos + np.array([radius * np.cos(angle), radius * np.sin(angle)])
                    segments.append([current_pos.tolist(), end_pos.tolist()])
                    current_pos = end_pos
    except Exception as e:
        logger.warning(f"Error extracting line segments: {e}")
        return []
    
    return segments
