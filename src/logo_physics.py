"""
LOGO Physics Computation Module

This module provides comprehensive physics attribute computation specifically designed 
for LOGO mode integration in the BongardSolver pipeline. It replaces fallback logic 
with semantic LOGO-derived computation for accurate scene graph generation.

Key Features:
- LOGO action program parsing and vertex extraction
- Comprehensive physics attribute computation (area, perimeter, asymmetry, etc.)
- Enhanced geometric analysis (curvature, apex detection, orientation)
- Seamless integration with advanced scene graph techniques
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

class LOGOPhysicsComputation:
    """
    Comprehensive physics computation module for LOGO mode integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_problem_records(self, problem_id: str, problem_records: List[Dict[str, Any]], use_enhanced_features: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all problem records using LOGO physics computation
        """
        all_objects_by_image = defaultdict(list)

        for idx, rec in enumerate(problem_records):
            parent_shape_id = f"{problem_id}_{idx}"
            action_program = rec.get('action_program', [])
            
            # Common attributes for all objects in this record
            common_attrs = {
                'label': rec.get('label', ''), 
                'shape_label': rec.get('shape_label', ''),
                'category': rec.get('category', ''), 
                'programmatic_label': rec.get('programmatic_label', ''),
                'image_path': rec.get('image_path'), 
                'original_record_idx': idx,
            }
            
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
                dx, dy = parsed_cmd['x'], parsed_cmd['y']
                new_pos = [turtle_pos[0] + dx, turtle_pos[1] + dy]
                verts = [start_pos_for_stroke, list(new_pos)]
                length = np.linalg.norm(np.array(new_pos) - np.array(turtle_pos))
                orientation = np.degrees(np.arctan2(dy, dx))
                turtle_pos = new_pos
                
            elif cmd_type == 'arc':
                radius = parsed_cmd.get('radius', 1.0)
                angle = parsed_cmd.get('angle', 0.0)
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
            
            # LOGO MODE: Calculate enhanced geometric features with LOGO physics
            base_obj = {
                'object_id': obj_id, 
                'parent_shape_id': parent_shape_id, 
                'action_index': stroke_idx,
                'vertices': regularized_verts, 
                'original_vertices': verts,
                'object_type': self._assign_object_type(regularized_verts), 
                'action_command': cmd,
                'endpoints': [regularized_verts[0], regularized_verts[-1]], 
                'length': length, 
                'orientation': orientation,
                'source': 'action_program', 
                'is_closed': len(verts) > 2 and np.allclose(verts[0], verts[-1], atol=1e-5),
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
            'compactness': float(compactness)
        })
        
        return enhanced_obj

    def _parse_action_command(self, cmd: Any) -> Optional[Dict[str, Any]]:
        """Parse LOGO action command string"""
        if isinstance(cmd, dict):
            return cmd
        if not isinstance(cmd, str):
            return None
        
        parts = cmd.split('_', 2)
        
        # Handle formats like "line_normal_0.2-0.3" (3 parts) and "start_0.1-0.2" (2 parts)
        if len(parts) == 3:
            shape, mode, rest = parts
        elif len(parts) == 2:
            shape, rest = parts
            mode = None # No mode specified
        else:
            return None

        if shape == "line" and '-' in rest:
            try:
                a, b = rest.split('-', 1)
                return {'type': 'line', 'mode': mode, 'x': float(a), 'y': float(b)}
            except (ValueError, IndexError):
                return None
        elif shape == "start" and '-' in rest:
            try:
                a, b = rest.split('-', 1)
                return {'type': 'start', 'x': float(a), 'y': float(b)}
            except (ValueError, IndexError):
                return None
        elif shape == "arc" and '-' in rest:
            try:
                radius, angle = rest.split('-', 1)
                return {'type': 'arc', 'mode': mode, 'radius': float(radius), 'angle': float(angle)}
            except (ValueError, IndexError):
                return None
        elif shape == "turn" and '-' in rest:
            try:
                angle = rest
                return {'type': 'turn', 'mode': mode, 'angle': float(angle)}
            except (ValueError, IndexError):
                return None
        
        # Extend for other commands as needed
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

    def _assign_object_type(self, verts: List[List[float]]) -> str:
        """Assign object type based on vertex analysis"""
        if len(verts) < 2:
            return 'point'
        elif len(verts) == 2:
            return 'line'
        elif len(verts) > 2:
            # Check if closed
            if np.allclose(verts[0], verts[-1], atol=1e-5):
                return 'polygon'
            else:
                # Check curvature to distinguish between line and curve
                if len(verts) > 3:
                    # Simple curvature check based on direction changes
                    angles = []
                    for i in range(1, len(verts) - 1):
                        v1 = np.array(verts[i]) - np.array(verts[i-1])
                        v2 = np.array(verts[i+1]) - np.array(verts[i])
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)
                    
                    if angles and np.mean(angles) > 0.1:  # More than ~6 degrees average deviation
                        return 'curve'
                
                return 'open_line'
        
        return 'unknown'

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
