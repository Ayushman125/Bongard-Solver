"""
LOGO Physics Computation Module

This module provides comprehensive physics attribute computation specifically designed 
for LOGO mode integration in the BongardSolver pipeline. It replaces fallback logic 
with semantic LOGO-derived computation for accurate scene graph generation.

Key Features:
- LOGO action program parsing and vertex extraction
- Comprehensive physics attribute computation (area, perimeter, asymmetry, etc.)
- Enhanced geometric analysis (curvature, apex detection, orientation)  
- Support for the 5 discovered Bongard-LOGO shape types: normal, circle, square, triangle, zigzag
- Seamless integration with advanced scene graph techniques

Shape Type Discovery Results:
- normal: 24,107 occurrences (48.7%) - straight lines, most common
- circle: 6,256 occurrences (12.6%) - circular shapes/arcs
- square: 6,519 occurrences (13.2%) - square-based shapes
- triangle: 5,837 occurrences (11.8%) - triangular shapes
- zigzag: 6,729 occurrences (13.6%) - zigzag patterns
Total: 49,448 action commands analyzed
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# Discovered Bongard-LOGO shape types with properties
BONGARD_SHAPE_TYPES = {
    'normal': {
        'frequency': 24107,
        'percentage': 48.7,
        'properties': {'is_regular': False, 'is_curved': False, 'complexity': 1}
    },
    'circle': {
        'frequency': 6256, 
        'percentage': 12.6,
        'properties': {'is_regular': True, 'is_curved': True, 'complexity': 2}
    },
    'square': {
        'frequency': 6519,
        'percentage': 13.2, 
        'properties': {'is_regular': True, 'is_curved': False, 'complexity': 2}
    },
    'triangle': {
        'frequency': 5837,
        'percentage': 11.8,
        'properties': {'is_regular': True, 'is_curved': False, 'complexity': 2}
    },
    'zigzag': {
        'frequency': 6729,
        'percentage': 13.6,
        'properties': {'is_regular': False, 'is_curved': True, 'complexity': 3}
    }
}

class LOGOPhysicsComputation:
    """
    Comprehensive physics computation module for LOGO mode integration with 5 shape type support
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.shape_type_stats = BONGARD_SHAPE_TYPES.copy()
    
    def get_shape_type_properties(self, shape_type: str) -> Dict[str, Any]:
        """Get properties for one of the 5 discovered shape types"""
        return self.shape_type_stats.get(shape_type, {
            'frequency': 0, 
            'percentage': 0.0,
            'properties': {'is_regular': False, 'is_curved': False, 'complexity': 1}
        })
    
    def process_problem_records(self, problem_id: str, problem_records: List[Dict[str, Any]], use_enhanced_features: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all problem records using LOGO physics computation
        """
        all_objects_by_image = defaultdict(list)

        for idx, rec in enumerate(problem_records):
            parent_shape_id = f"{problem_id}_{idx}"
            # Use flattened_actions for proper command processing
            action_program = rec.get('flattened_actions', rec.get('action_program', []))
            
            # Common attributes for all objects in this record
            common_attrs = {
                'label': rec.get('label', ''), 
                'shape_label': rec.get('shape_label', ''),
                'category': rec.get('category', ''), 
                'programmatic_label': rec.get('programmatic_label', ''),
                'image_path': rec.get('image_path'), 
                'original_record_idx': idx,
            }
            
            # Debug logging for categorization
            self.logger.info(f"Processing record {idx} for {problem_id}: category='{rec.get('category')}', label='{rec.get('label')}', image_path='{rec.get('image_path')}'")
            
            if not action_program:
                continue
                
            # LOGO MODE: Use enhanced vertex extraction with physics computation
            objects_for_image = self._extract_enhanced_vertices(
                action_program, 
                parent_shape_id,
                problem_id,
                idx,
                common_attrs
            )
            
            all_objects_by_image[parent_shape_id].extend(objects_for_image)
        
        return dict(all_objects_by_image)

    def _extract_enhanced_vertices(self, action_program: List[str], parent_shape_id: str, 
                                 problem_id: str, record_idx: int, 
                                 common_attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract vertices with enhanced LOGO physics computation
        """
        objects = []
        turtle_pos = [0.0, 0.0]
        turtle_heading = 0.0
        
        # LOGO simulation to generate strokes with enhanced physics
        last_stroke_obj = None
        for stroke_idx, cmd in enumerate(action_program):
            parsed_cmd = self._parse_action_command(cmd)
            if not parsed_cmd: 
                continue
                
            cmd_type = parsed_cmd.get('type')
            
            verts = []
            length = 0.0
            orientation = 0.0

            if cmd_type == 'start':
                turtle_pos = [parsed_cmd['x'], parsed_cmd['y']]
                continue

            start_pos_for_stroke = list(turtle_pos)

            if cmd_type == 'line':
                # Scale coordinates properly - these are normalized values that need scaling
                dx, dy = parsed_cmd['x'], parsed_cmd['y']
                
                # Apply proper scaling factor for LOGO coordinate system
                LOGO_SCALE_FACTOR = 100.0  # Scale up from normalized coordinates
                dx_scaled = dx * LOGO_SCALE_FACTOR
                dy_scaled = dy * LOGO_SCALE_FACTOR
                
                new_pos = [turtle_pos[0] + dx_scaled, turtle_pos[1] + dy_scaled]
                verts = [start_pos_for_stroke, list(new_pos)]
                length = np.linalg.norm(np.array(new_pos) - np.array(turtle_pos))
                orientation = np.degrees(np.arctan2(dy_scaled, dx_scaled))
                turtle_pos = new_pos
                
            elif cmd_type == 'arc':
                # Scale arc parameters properly
                radius = parsed_cmd.get('radius', 1.0) * 50.0  # Scale radius
                angle = parsed_cmd.get('angle', 0.0) * 360.0  # Scale angle to degrees
                num_points = max(6, int(abs(angle) // 10))
                verts = [start_pos_for_stroke]
                start_angle_rad = np.radians(turtle_heading)
                center_of_rotation = [
                    turtle_pos[0] - radius * np.sin(start_angle_rad),
                    turtle_pos[1] + radius * np.cos(start_angle_rad)
                ]
                for i in range(1, num_points + 1):
                    theta_rad = start_angle_rad + np.radians((angle / num_points) * i)
                    x = center_of_rotation[0] + radius * np.sin(theta_rad)
                    y = center_of_rotation[1] - radius * np.cos(theta_rad)
                    verts.append([x, y])
                length = abs(np.radians(angle) * radius)
                orientation = np.degrees(np.arctan2(verts[-1][1] - verts[0][1], verts[-1][0] - verts[0][0]))
                turtle_pos = verts[-1]
                turtle_heading += angle
                
            elif cmd_type == 'turn':
                turtle_heading += parsed_cmd.get('angle', 0.0)
                continue
            else:
                continue
            
            if not verts: 
                continue

            # Apply stroke regularization
            regularized_verts = self._regularize_stroke_vertices(verts, tolerance=3.0)
            
            obj_id = f"{problem_id}_{record_idx}_{stroke_idx}"
            
            # Extract shape type from parsed command for better type assignment
            shape_type = parsed_cmd.get('shape', None)
            
            # CRITICAL FIX: Pass cmd_type to preserve arc vs line distinction
            cmd_type = parsed_cmd.get('type', None)
            
            # LOGO MODE: Calculate enhanced geometric features with LOGO physics
            base_obj = {
                'object_id': obj_id, 
                'parent_shape_id': parent_shape_id, 
                'action_index': stroke_idx,
                'vertices': regularized_verts, 
                'original_vertices': verts,
                'object_type': self._assign_object_type(regularized_verts, shape_type, cmd_type), 
                'shape_type': shape_type,  # Store the Bongard-LOGO shape type
                'action_command': cmd,
                'endpoints': [regularized_verts[0], regularized_verts[-1]], 
                'length': length, 
                'orientation': orientation,
                'source': 'action_program', 
                'is_closed': len(verts) > 2 and self._is_closed_shape(verts),
                'is_valid': True,  # Mark as valid if we have meaningful geometry
                **common_attrs
            }
            
            # Enhance with LOGO physics computation
            enhanced_features = self.compute_enhanced_physics(base_obj)
            
            # Add relationships
            relationships = []
            if last_stroke_obj:
                # Check for adjacency based on endpoint proximity
                if np.allclose(last_stroke_obj['endpoints'][-1], verts[0], atol=1e-5):
                    relationships.append(f"adjacent_to_{last_stroke_obj['object_id']}")
            
            enhanced_features['relationships'] = relationships
            
            # Debug logging for object categorization
            self.logger.info(f"Created object {enhanced_features.get('object_id')} with category='{enhanced_features.get('category')}', label='{enhanced_features.get('label')}', image_path='{enhanced_features.get('image_path')}'")
            
            objects.append(enhanced_features)
            last_stroke_obj = enhanced_features
        
        return objects

    def compute_enhanced_physics(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute comprehensive physics attributes for a LOGO object with enhanced features
        """
        enhanced_obj = obj_data.copy()
        
        # Get vertices for computation
        vertices = obj_data.get('vertices', [])
        if not vertices or len(vertices) < 2:
            # Set default values for degenerate cases
            enhanced_obj.update({
                'area': 0.0,
                'perimeter': 0.0,
                'aspect_ratio': 1.0,
                'orientation': 0.0,
                'curvature_score': 0.0,
                'horizontal_asymmetry': 0.0,
                'vertical_asymmetry': 0.0,
                'apex_x_position': 0.5,
                'is_highly_curved': False,
                'bounding_box': [0, 0, 0, 0],
                'centroid': [0, 0],
                'compactness': 0.0
            })
            return enhanced_obj
        
        # Convert to numpy for easier computation
        verts_array = np.array(vertices)
        
        # 1. Basic geometric properties
        bbox = self._compute_bounding_box(verts_array)
        centroid = self._compute_centroid(verts_array)
        area = self._compute_area(verts_array, obj_data.get('is_closed', False))
        perimeter = self._compute_perimeter(verts_array)
        
        # 2. Shape analysis
        aspect_ratio = self._compute_aspect_ratio(bbox)
        orientation = self._compute_orientation(verts_array)
        
        # 3. LOGO-specific curvature analysis
        curvature_score = self._compute_curvature_score(verts_array)
        is_highly_curved = curvature_score > 0.3
        
        # 4. Advanced asymmetry analysis
        h_asymmetry, v_asymmetry = self._compute_asymmetry_measures(verts_array, centroid)
        
        # 5. Apex detection for triangular shapes
        apex_x_pos = self._detect_apex_position(verts_array)
        
        # 6. Compactness measure
        compactness = self._compute_compactness(area, perimeter) if perimeter > 0 else 0.0
        
        # Update object with all computed features
        enhanced_obj.update({
            'area': float(area),
            'perimeter': float(perimeter),
            'aspect_ratio': float(aspect_ratio),
            'orientation': float(orientation),
            'curvature_score': float(curvature_score),
            'horizontal_asymmetry': float(h_asymmetry),
            'vertical_asymmetry': float(v_asymmetry),
            'apex_x_position': float(apex_x_pos),
            'is_highly_curved': bool(is_highly_curved),
            'bounding_box': bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox),
            'centroid': centroid.tolist() if hasattr(centroid, 'tolist') else list(centroid),
            'compactness': float(compactness),
            'is_valid': True,  # Mark as valid since we computed meaningful features
            'geometry_reason': 'logo_physics'
        })
        
        return enhanced_obj

    def _parse_action_command(self, cmd: Any) -> Optional[Dict[str, Any]]:
        """Parse LOGO action command string for all 5 discovered shape types"""
        if isinstance(cmd, dict):
            return cmd
        if not isinstance(cmd, str):
            return None
        
        parts = cmd.split('_')
        
        # Handle different command formats based on discovered patterns
        if len(parts) < 2:
            return None
        
        command_type = parts[0]  # 'line', 'arc', 'start', 'turn'
        
        if command_type in ['line', 'arc']:
            # Handle both 3-part and 4-part formats:
            # 3-part: line_<shape>_<params> or arc_<shape>_<params> 
            # 4-part: line_<shape>_<size>_<thickness-y> or arc_<shape>_<radius>_<angle-y>
            if len(parts) >= 4:
                # 4-part format: arc_normal_0.500_0.542-0.750
                shape_type = parts[1]
                size_or_radius = parts[2] 
                thickness_or_angle_part = parts[3]
                
                # Validate shape type
                if shape_type not in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
                    return None
                
                # Parse the final parameter with dash
                if '-' in thickness_or_angle_part:
                    try:
                        first_param, second_param = thickness_or_angle_part.split('-', 1)
                        
                        if command_type == 'line':
                            return {
                                'type': command_type,
                                'shape': shape_type, 
                                'size': float(size_or_radius),
                                'thickness': float(first_param),
                                'x': float(size_or_radius),
                                'y': float(first_param)
                            }
                        elif command_type == 'arc':
                            return {
                                'type': command_type,
                                'shape': shape_type,
                                'radius': float(size_or_radius), 
                                'angle': float(first_param),
                                'size': float(size_or_radius),  # For compatibility
                                'thickness': float(first_param)  # For compatibility
                            }
                    except (ValueError, IndexError):
                        return None
                        
            elif len(parts) >= 3:
                # 3-part format: line_normal_0.5-45.0 or arc_normal_1.0-45.0
                shape_type = parts[1]  # One of: normal, circle, square, triangle, zigzag
                params = parts[2]
                
                # Validate shape type is one of the 5 discovered types
                if shape_type not in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
                    return None
                
                # Parse parameters (format: size-thickness)
                if '-' in params:
                    try:
                        size, thickness = params.split('-', 1)
                        return {
                            'type': command_type,
                            'shape': shape_type, 
                            'size': float(size),
                            'thickness': float(thickness),
                            'x': float(size),  # Use size as x-component for compatibility
                            'y': float(thickness)  # Use thickness as y-component
                        }
                    except (ValueError, IndexError):
                        return None
            
        elif command_type == "start":
            # Format: start_x_y
            if len(parts) >= 3:
                try:
                    return {'type': 'start', 'x': float(parts[1]), 'y': float(parts[2])}
                except (ValueError, IndexError):
                    return None
                    
        elif command_type == "turn":
            # Format: turn_angle
            if len(parts) >= 2:
                try:
                    return {'type': 'turn', 'angle': float(parts[1])}
                except (ValueError, IndexError):
                    return None
        
        return None

    def _regularize_stroke_vertices(self, verts: List[List[float]], tolerance: float = 3.0) -> List[List[float]]:
        """Apply stroke regularization to reduce over-segmentation"""
        if len(verts) <= 2:
            return verts
        
        regularized = [verts[0]]
        for i in range(1, len(verts)):
            # Keep vertex if it's far enough from the last kept vertex
            if np.linalg.norm(np.array(verts[i]) - np.array(regularized[-1])) > tolerance:
                regularized.append(verts[i])
        
        # Always keep the last vertex
        if not np.allclose(regularized[-1], verts[-1]):
            regularized.append(verts[-1])
            
        return regularized

    def _assign_object_type(self, verts: List[List[float]], shape_type: str = None, cmd_type: str = None) -> str:
        """Assign object type based on vertex analysis and Bongard-LOGO shape types"""
        if len(verts) < 2:
            return 'point'
        elif len(verts) == 2:
            return 'line'
        
        # CRITICAL FIX: Check cmd_type first to preserve arc vs line distinction
        if cmd_type == 'arc':
            return 'arc'  # All arc commands should get object_type = 'arc'
        elif cmd_type == 'line':
            # For line commands, use shape type mapping
            if shape_type in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
                shape_to_object = {
                    'normal': 'line',
                    'circle': 'polygon' if self._is_closed_shape(verts) else 'curve',
                    'square': 'polygon',
                    'triangle': 'polygon',
                    'zigzag': 'curve'
                }
                return shape_to_object.get(shape_type, 'line')
            return 'line'
        
        # Use shape type from action command if available (one of 5 discovered types)
        if shape_type in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
            # Map shape types to object types
            shape_to_object = {
                'normal': 'line',  # Normal lines are straight lines
                'circle': 'polygon' if self._is_closed_shape(verts) else 'curve',  # Circles are curved
                'square': 'polygon',  # Squares are closed polygons
                'triangle': 'polygon',  # Triangles are closed polygons  
                'zigzag': 'curve'  # Zigzag patterns are curves
            }
            return shape_to_object.get(shape_type, 'unknown')
        
        # ACTION PROGRAMS ONLY: No fallback geometric analysis
        # All object types must be derived from action commands
        return 'unknown'
    
    def _is_closed_shape(self, verts: List[List[float]]) -> bool:
        """Check if shape is closed based on endpoint proximity"""
        if len(verts) < 3:
            return False
        return np.allclose(verts[0], verts[-1], atol=1e-5)

    def _compute_bounding_box(self, verts_array: np.ndarray) -> np.ndarray:
        """Compute bounding box [min_x, min_y, max_x, max_y]"""
        min_x, min_y = np.min(verts_array, axis=0)
        max_x, max_y = np.max(verts_array, axis=0)
        return np.array([min_x, min_y, max_x, max_y])

    def _compute_centroid(self, verts_array: np.ndarray) -> np.ndarray:
        """Compute geometric centroid"""
        return np.mean(verts_array, axis=0)

    def _compute_area(self, verts_array: np.ndarray, is_closed: bool) -> float:
        """Compute area using shoelace formula for closed shapes"""
        if not is_closed or len(verts_array) < 3:
            return 0.0
        
        # Shoelace formula
        x = verts_array[:, 0]
        y = verts_array[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, len(x) - 1)))

    def _compute_perimeter(self, verts_array: np.ndarray) -> float:
        """Compute perimeter as sum of edge lengths"""
        if len(verts_array) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(verts_array) - 1):
            perimeter += np.linalg.norm(verts_array[i+1] - verts_array[i])
        
        return perimeter

    def _compute_aspect_ratio(self, bbox: np.ndarray) -> float:
        """Compute aspect ratio from bounding box"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if height == 0:
            return float('inf') if width > 0 else 1.0
        
        return width / height

    def _compute_orientation(self, verts_array: np.ndarray) -> float:
        """Compute dominant orientation using PCA"""
        if len(verts_array) < 2:
            return 0.0
        
        # Center the points
        centered = verts_array - np.mean(verts_array, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Get principal component (largest eigenvalue)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Return angle in degrees
        return np.degrees(np.arctan2(principal_component[1], principal_component[0]))

    def _compute_curvature_score(self, verts_array: np.ndarray) -> float:
        """Compute curvature score based on direction changes"""
        if len(verts_array) < 3:
            return 0.0
        
        total_curvature = 0.0
        valid_points = 0
        
        for i in range(1, len(verts_array) - 1):
            v1 = verts_array[i] - verts_array[i-1]
            v2 = verts_array[i+1] - verts_array[i]
            
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                curvature = np.arccos(cos_angle)
                total_curvature += curvature
                valid_points += 1
        
        return total_curvature / valid_points if valid_points > 0 else 0.0

    def _compute_asymmetry_measures(self, verts_array: np.ndarray, centroid: np.ndarray) -> Tuple[float, float]:
        """Compute horizontal and vertical asymmetry measures"""
        if len(verts_array) < 3:
            return 0.0, 0.0
        
        # Horizontal asymmetry
        left_points = verts_array[verts_array[:, 0] < centroid[0]]
        right_points = verts_array[verts_array[:, 0] >= centroid[0]]
        
        h_asymmetry = 0.0
        if len(left_points) > 0 and len(right_points) > 0:
            left_extent = centroid[0] - np.min(left_points[:, 0])
            right_extent = np.max(right_points[:, 0]) - centroid[0]
            if max(left_extent, right_extent) > 1e-10:
                h_asymmetry = abs(left_extent - right_extent) / max(left_extent, right_extent)
        
        # Vertical asymmetry
        bottom_points = verts_array[verts_array[:, 1] < centroid[1]]
        top_points = verts_array[verts_array[:, 1] >= centroid[1]]
        
        v_asymmetry = 0.0
        if len(bottom_points) > 0 and len(top_points) > 0:
            bottom_extent = centroid[1] - np.min(bottom_points[:, 1])
            top_extent = np.max(top_points[:, 1]) - centroid[1]
            if max(bottom_extent, top_extent) > 1e-10:
                v_asymmetry = abs(bottom_extent - top_extent) / max(bottom_extent, top_extent)
        
        return h_asymmetry, v_asymmetry

    def _detect_apex_position(self, verts_array: np.ndarray) -> float:
        """Detect apex position for triangular-like shapes"""
        if len(verts_array) < 3:
            return 0.5
        
        # Find the vertex that's farthest from the base line
        max_distance = 0.0
        apex_idx = 0
        
        # Use first and last points as base line
        base_start = verts_array[0]
        base_end = verts_array[-1]
        base_vector = base_end - base_start
        base_length = np.linalg.norm(base_vector)
        
        if base_length < 1e-10:
            return 0.5
        
        base_unit = base_vector / base_length
        
        for i, vertex in enumerate(verts_array):
            # Distance from point to line
            to_vertex = vertex - base_start
            projection_length = np.dot(to_vertex, base_unit)
            closest_on_line = base_start + projection_length * base_unit
            distance = np.linalg.norm(vertex - closest_on_line)
            
            if distance > max_distance:
                max_distance = distance
                apex_idx = i
        
        # Return normalized x-position of apex
        apex_vertex = verts_array[apex_idx]
        bbox = self._compute_bounding_box(verts_array)
        width = bbox[2] - bbox[0]
        
        if width < 1e-10:
            return 0.5
        
        return (apex_vertex[0] - bbox[0]) / width

    def _compute_compactness(self, area: float, perimeter: float) -> float:
        """Compute compactness measure (isoperimetric ratio)"""
        if perimeter <= 0:
            return 0.0
        
        # Isoperimetric ratio: 4π * area / perimeter²
        # Perfect circle has compactness = 1
        return (4 * np.pi * area) / (perimeter ** 2)
