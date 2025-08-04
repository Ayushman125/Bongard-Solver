import logging
import os
import math
import numpy as np
import traceback
from typing import List, Dict, Any
from .data_loading import remap_path
import networkx as nx
from shapely.geometry import Polygon, LineString
from collections import defaultdict

# Import the new advanced predicates
from .advanced_predicates import ADVANCED_PREDICATE_REGISTRY

# Import semantic parsing capabilities
from .semantic_action_parser import (
    SemanticActionParser, 
    BongardPredicateEngine, 
    enhance_node_with_semantic_features
)

def _detect_and_add_composite_shapes(graph, parent_shape_id):
    """Detect quarter circles and other composite shapes from connected line segments"""
    from typing import List, Tuple
    
    # Get all line nodes from the graph
    line_nodes = []
    for node_id, data in graph.nodes(data=True):
        if data.get('object_type') == 'line' and data.get('vertices'):
            line_nodes.append((node_id, data))
    
    if len(line_nodes) < 3:  # Need at least 3 lines for meaningful shapes
        return graph
    
    # ACTION PROGRAMS ONLY: No artificial shape detection needed
    # All shape information comes from action commands
    return graph

def _find_connected_line_groups(line_nodes):
    """Find groups of connected line segments"""
    groups = []
    used_nodes = set()
    
    for i, (node_id, data) in enumerate(line_nodes):
        if node_id in used_nodes:
            continue
            
        # Start a new group with this line
        current_group = [(node_id, data)]
        used_nodes.add(node_id)
        
        # Find all lines connected to this group
        group_expanded = True
        while group_expanded:
            group_expanded = False
            for j, (other_node_id, other_data) in enumerate(line_nodes):
                if other_node_id in used_nodes:
                    continue
                    
                # Check if this line connects to any line in the current group
                if _lines_are_connected(current_group, (other_node_id, other_data)):
                    current_group.append((other_node_id, other_data))
                    used_nodes.add(other_node_id)
                    group_expanded = True
        
        if len(current_group) >= 2:  # Only keep groups with multiple lines
            groups.append(current_group)
    
    return groups

def _lines_are_connected(group, candidate_line):
    """Check if a candidate line connects to any line in the group"""
    candidate_id, candidate_data = candidate_line
    candidate_vertices = candidate_data.get('vertices', [])
    
    if len(candidate_vertices) < 2:
        return False
    
    candidate_start = tuple(candidate_vertices[0])
    candidate_end = tuple(candidate_vertices[-1])
    
    tolerance = 0.01  # Small tolerance for floating point comparison
    
    for group_id, group_data in group:
        group_vertices = group_data.get('vertices', [])
        if len(group_vertices) < 2:
            continue
            
        group_start = tuple(group_vertices[0])
        group_end = tuple(group_vertices[-1])
        
        # Check all possible connection points
        connections = [
            (candidate_start, group_start),
            (candidate_start, group_end),
            (candidate_end, group_start),
            (candidate_end, group_end)
        ]
        
        for (p1, p2) in connections:
            distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if distance < tolerance:
                return True
    
    return False

def _extract_path_from_connected_lines(connected_lines):
    """Extract ordered path points from connected line segments"""
    if not connected_lines:
        return []
    
    # Start with the first line
    path_points = []
    used_lines = set()
    
    # Find the starting line (one that has an unconnected endpoint)
    start_line = None
    for line_id, line_data in connected_lines:
        vertices = line_data.get('vertices', [])
        if len(vertices) >= 2:
            start_line = (line_id, line_data)
            break
    
    if not start_line:
        return []
    
    current_line_id, current_line_data = start_line
    current_vertices = current_line_data.get('vertices', [])
    
    # Add vertices from first line
    path_points.extend(current_vertices)
    used_lines.add(current_line_id)
    
    # Follow the connected path
    while len(used_lines) < len(connected_lines):
        current_end = tuple(path_points[-1])
        
        # Find next connected line
        next_line = None
        for line_id, line_data in connected_lines:
            if line_id in used_lines:
                continue
                
            vertices = line_data.get('vertices', [])
            if len(vertices) < 2:
                continue
                
            start_point = tuple(vertices[0])
            end_point = tuple(vertices[-1])
            
            tolerance = 0.01
            
            # Check if this line connects to current end
            if math.sqrt((current_end[0] - start_point[0])**2 + (current_end[1] - start_point[1])**2) < tolerance:
                next_line = (line_id, line_data, False)  # False = use normal order
                break
            elif math.sqrt((current_end[0] - end_point[0])**2 + (current_end[1] - end_point[1])**2) < tolerance:
                next_line = (line_id, line_data, True)  # True = use reverse order
                break
        
        if not next_line:
            break  # No more connected lines found
            
        next_line_id, next_line_data, reverse_order = next_line
        next_vertices = next_line_data.get('vertices', [])
        
        if reverse_order:
            # Add vertices in reverse order, skipping the connection point
            path_points.extend(reversed(next_vertices[:-1]))
        else:
            # Add vertices in normal order, skipping the connection point
            path_points.extend(next_vertices[1:])
        
        used_lines.add(next_line_id)
    
    return path_points

# REMOVED: All artificial shape detection functions
# Action programs provide all shape information directly
    return {'is_arc': False, 'confidence': 0.0}  # Placeholder

def _angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    # Normalize vectors
    len1 = math.sqrt(v1[0]**2 + v1[1]**2)
    len2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
        
    norm_v1 = (v1[0] / len1, v1[1] / len1)
    norm_v2 = (v2[0] / len2, v2[1] / len2)
    
    # Dot product
    dot_product = norm_v1[0] * norm_v2[0] + norm_v1[1] * norm_v2[1]
    
    # Clamp to avoid numerical errors
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Angle in radians, then convert to degrees
    angle_rad = math.acos(dot_product)
    return math.degrees(angle_rad)

def _calculate_bounding_box(vertices):
    """Calculate bounding box from vertices"""
    if not vertices:
        return [0, 0, 0, 0]
    
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def _calculate_centroid(vertices):
    """Calculate centroid from vertices"""
    if not vertices:
        return [0, 0]
    
    x_sum = sum(v[0] for v in vertices)
    y_sum = sum(v[1] for v in vertices)
    
    return [x_sum / len(vertices), y_sum / len(vertices)]

def _calculate_predicate_importance(graph, problem_id):
    """Calculate importance scores for predicates based on their discriminative power"""
    predicate_counts = {}
    total_edges = 0
    
    # Count predicate frequencies
    for u, v, data in graph.edges(data=True):
        predicate = data.get('predicate', 'unknown')
        predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        total_edges += 1
    
    # Calculate importance scores (inverse frequency for rarity bonus)
    importance_scores = {}
    for predicate, count in predicate_counts.items():
        # Semantic predicates get higher base importance
        base_importance = 1.0
        if predicate in ['has_apex_at_left', 'has_asymmetric_base', 'has_tilted_orientation', 
                        'has_length_ratio_imbalance', 'exhibits_mirror_asymmetry', 
                        'forms_open_vs_closed_distinction', 'has_geometric_complexity_difference']:
            base_importance = 3.0
        elif predicate in ['same_shape_class', 'forms_symmetry', 'has_compactness_difference']:
            base_importance = 2.0
        
        # Rarity bonus (less frequent predicates are more discriminative)
        frequency = count / total_edges if total_edges > 0 else 0
        rarity_bonus = 1.0 / (frequency + 0.1)  # Avoid division by zero
        
        importance_scores[predicate] = base_importance * rarity_bonus
    
    return importance_scores

def _filter_low_importance_edges(graph, importance_threshold=1.5):
    """Remove edges with low discriminative importance to reduce graph complexity"""
    importance_scores = _calculate_predicate_importance(graph, "analysis")
    
    edges_to_remove = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        predicate = data.get('predicate', 'unknown')
        importance = importance_scores.get(predicate, 1.0)
        
        # Keep high-importance edges and always keep structural edges
        if importance < importance_threshold and data.get('source') != 'program':
            edges_to_remove.append((u, v, key))
    
    # Remove low-importance edges
    for edge in edges_to_remove:
        if graph.has_edge(edge[0], edge[1], edge[2]):
            graph.remove_edge(edge[0], edge[1], edge[2])
    
    return graph

def _add_abstract_conceptual_nodes(graph, parent_shape_id):
    """Add higher-level conceptual nodes that represent abstract concepts"""
    composite_shapes = [n for n, data in graph.nodes(data=True) 
                       if data.get('source') == 'geometric_grouping']
    
    if not composite_shapes:
        return graph
    
    # Analyze overall shape properties to create conceptual nodes
    shape_properties = {}
    for shape_id in composite_shapes:
        data = graph.nodes[shape_id]
        
        # Collect semantic properties
        properties = []
        
        # Check asymmetry
        verts = data.get('vertices', [])
        if len(verts) >= 3:
            # Simple asymmetry check
            centroid = data.get('centroid', [0, 0])
            x_coords = [v[0] for v in verts]
            left_extent = centroid[0] - min(x_coords)
            right_extent = max(x_coords) - centroid[0]
            
            if max(left_extent, right_extent) > 0:
                asymmetry_ratio = abs(left_extent - right_extent) / max(left_extent, right_extent)
                if asymmetry_ratio > 0.2:
                    properties.append('asymmetric')
                else:
                    properties.append('symmetric')
        
        # Check orientation
        aspect_ratio = data.get('aspect_ratio', 1.0)
        if aspect_ratio > 1.5:
            properties.append('horizontal_dominant')
        elif aspect_ratio < 0.67:
            properties.append('vertical_dominant')
        else:
            properties.append('balanced_proportions')
        
        # Check complexity
        stroke_count = data.get('stroke_count', 1)
        if stroke_count > 6:
            properties.append('complex')
        elif stroke_count <= 3:
            properties.append('simple')
        else:
            properties.append('moderate_complexity')
        
        # Check closure
        if data.get('is_closed', False):
            properties.append('closed_shape')
        else:
            properties.append('open_shape')
        
        shape_properties[shape_id] = properties
    
    # Create conceptual nodes for common properties
    property_groups = {}
    for shape_id, props in shape_properties.items():
        for prop in props:
            if prop not in property_groups:
                property_groups[prop] = []
            property_groups[prop].append(shape_id)
    
    # Add conceptual nodes for properties that apply to multiple shapes
    for prop, shapes in property_groups.items():
        if len(shapes) > 1:  # Only create concepts that apply to multiple shapes
            concept_id = f"{parent_shape_id}_concept_{prop}"
            graph.add_node(concept_id, 
                          object_id=concept_id,
                          object_type='conceptual',
                          source='semantic_abstraction',
                          concept_type=prop,
                          applies_to=shapes,
                          abstraction_level='high')
            
            # Connect conceptual node to shapes it describes
            for shape_id in shapes:
                graph.add_edge(concept_id, shape_id, 
                             predicate='describes', 
                             source='semantic_abstraction')
    
    return graph




def are_points_collinear(verts, tol=1e-6):
    arr = np.asarray(verts)
    if len(arr) < 3:
        return True
    v0 = arr[0]
    v1 = arr[1]
    direction = v1 - v0
    norm = np.linalg.norm(direction)
    if norm < tol:
        return True
    direction = direction / norm
    for v in arr[2:]:
        rel = v - v0
        proj = np.dot(rel, direction)
        perp = rel - proj * direction
        if np.linalg.norm(perp) > tol:
            return False
    return True

def assign_object_type(verts, action_command=None):
    """ACTION PROGRAMS ONLY: Assign object type based on action command only"""
    
    # ACTION PROGRAMS ONLY: Only use action command for type information
    if action_command and isinstance(action_command, str):
        if action_command.startswith('arc_'):
            return "arc"
        elif action_command.startswith('line_'):
            return "line"
    
    # ACTION PROGRAMS ONLY: No fallback geometry analysis
    # If no action command, return unknown - all types must come from action programs
    return "unknown"

def parse_action_command(cmd):
    """Parse LOGO action command string for all 5 discovered shape types, handling both old and new formats"""
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
        # Handle both formats:
        # New format: line_<shape>_<size>_<thickness>-<y> or arc_<shape>_<radius>_<angle>-<y>
        # Old format: line_<shape>_<params> or arc_<shape>_<params>
        if len(parts) >= 4:
            # New format: arc_normal_0.500_0.542-0.750
            shape_type = parts[1]  # One of: normal, circle, square, triangle, zigzag
            size_or_radius = parts[2]
            thickness_or_angle_part = parts[3]
            
            # Validate shape type is one of the 5 discovered types
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
                            'x': float(size_or_radius),  # Use size as x-component
                            'y': float(first_param),     # Use thickness as y-component
                            'mode': shape_type
                        }
                    elif command_type == 'arc':
                        return {
                            'type': command_type,
                            'shape': shape_type,
                            'radius': float(size_or_radius), 
                            'angle': float(first_param),
                            'mode': shape_type
                        }
                except (ValueError, IndexError):
                    return None
        elif len(parts) == 3:
            # Old format: line_normal_0.2-0.3 or arc_normal_1.0-45.0
            shape_type = parts[1]
            params = parts[2]
            
            if shape_type not in ['normal', 'circle', 'square', 'triangle', 'zigzag']:
                return None
                
            if '-' in params:
                try:
                    a, b = params.split('-', 1)
                    if command_type == 'line':
                        return {'type':'line', 'mode':shape_type, 'shape':shape_type, 'x':float(a), 'y':float(b)}
                    elif command_type == 'arc':
                        return {'type':'arc', 'mode':shape_type, 'shape':shape_type, 'radius':float(a), 'angle':float(b)}
                except Exception:
                    return None
                    
    elif command_type == "start":
        # Format: start_x_y or start_x-y
        if len(parts) >= 3:
            try:
                return {'type': 'start', 'x': float(parts[1]), 'y': float(parts[2])}
            except (ValueError, IndexError):
                pass
        elif len(parts) == 2 and '-' in parts[1]:
            try:
                a, b = parts[1].split('-',1)
                return {'type':'start', 'x':float(a), 'y':float(b)}
            except Exception:
                return None
                
    elif command_type == "turn":
        # Format: turn_angle
        if len(parts) >= 2:
            try:
                return {'type': 'turn', 'angle': float(parts[1])}
            except (ValueError, IndexError):
                return None
    
    return None


def _calculate_stroke_geometry(verts):
    from shapely.geometry import Polygon, LineString
    bounding_box = None
    centroid = None
    area = None
    perimeter = None
    aspect_ratio = None
    compactness = None
    
    if len(verts) < 2:
        return bounding_box, centroid, area, perimeter, aspect_ratio, compactness

    try:
        # Use shapely for robust geometry calculation
        is_closed = len(verts) >= 3 and np.allclose(verts[0], verts[-1], atol=1e-5)
        if is_closed:
            shape = Polygon(verts)
            area = shape.area
            perimeter = shape.length
        else:
            shape = LineString(verts)
            area = 0.0
            perimeter = shape.length

        bounding_box = shape.bounds
        centroid = list(shape.centroid.coords)[0]
        
        minx, miny, maxx, maxy = bounding_box
        width = maxx - minx
        height = maxy - miny
        
        aspect_ratio = width / height if height > 1e-6 else 0.0
        
        if perimeter > 1e-6:
            compactness = (4 * np.pi * area) / (perimeter ** 2) if is_closed else 0.0
        else:
            compactness = 0.0
            
    except Exception as e:
        logging.warning(f"Could not compute geometry for stroke with {len(verts)} verts: {e}")

    return bounding_box, centroid, area, perimeter, aspect_ratio, compactness

def _regularize_stroke_vertices(vertices, tolerance=2.0):
    """
    Regularize stroke vertices using Ramer-Douglas-Peucker algorithm
    to reduce over-segmentation of near-straight curves
    """
    if len(vertices) <= 2:
        return vertices
    
    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment"""
        if np.allclose(line_start, line_end):
            return np.linalg.norm(np.array(point) - np.array(line_start))
        
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-10:
            return np.linalg.norm(point_vec)
        
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = np.array(line_start) + t * line_vec
        return np.linalg.norm(np.array(point) - projection)
    
    def douglas_peucker(points, tolerance):
        """Ramer-Douglas-Peucker algorithm implementation"""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line start-end
        max_dist = 0
        max_index = 0
        for i in range(1, len(points) - 1):
            dist = perpendicular_distance(points[i], points[0], points[-1])
            if dist > max_dist:
                max_dist = dist
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            # Recursive call for both parts
            left_part = douglas_peucker(points[:max_index + 1], tolerance)
            right_part = douglas_peucker(points[max_index:], tolerance)
            
            # Combine results (remove duplicate point at junction)
            return left_part[:-1] + right_part
        else:
            # If all points are within tolerance, return just endpoints
            return [points[0], points[-1]]
    
    try:
        simplified = douglas_peucker(vertices, tolerance)
        return simplified
    except Exception as e:
        logging.warning(f"Failed to regularize stroke vertices: {e}")
        return vertices

def _calculate_enhanced_geometry_features(verts, existing_attrs=None):
    """Calculate enhanced geometric features for better semantic analysis"""
    if existing_attrs is None:
        existing_attrs = {}
    
    features = existing_attrs.copy()
    
    if len(verts) < 2:
        return features
    
    try:
        # Basic features
        bb, cent, area, perim, ar, comp = _calculate_stroke_geometry(verts)
        features.update({
            'bounding_box': bb,
            'centroid': cent,
            'area': area,
            'perimeter': perim,
            'aspect_ratio': ar,
            'compactness': comp
        })
        
        # Enhanced asymmetry analysis
        if len(verts) >= 3 and cent:
            x_coords = [v[0] for v in verts]
            y_coords = [v[1] for v in verts]
            
            # Calculate extent asymmetry
            left_extent = cent[0] - min(x_coords)
            right_extent = max(x_coords) - cent[0]
            top_extent = max(y_coords) - cent[1]
            bottom_extent = cent[1] - min(y_coords)
            
            horizontal_asymmetry = 0.0
            vertical_asymmetry = 0.0
            
            if max(left_extent, right_extent) > 0:
                horizontal_asymmetry = abs(left_extent - right_extent) / max(left_extent, right_extent)
            
            if max(top_extent, bottom_extent) > 0:
                vertical_asymmetry = abs(top_extent - bottom_extent) / max(top_extent, bottom_extent)
            
            features.update({
                'horizontal_asymmetry': horizontal_asymmetry,
                'vertical_asymmetry': vertical_asymmetry,
                'left_extent': left_extent,
                'right_extent': right_extent,
                'top_extent': top_extent,
                'bottom_extent': bottom_extent
            })
            
            # Find apex and base characteristics
            max_y_idx = np.argmax(y_coords)
            min_y_idx = np.argmin(y_coords)
            
            apex_x = x_coords[max_y_idx] if abs(y_coords[max_y_idx] - cent[1]) > abs(y_coords[min_y_idx] - cent[1]) else x_coords[min_y_idx]
            
            features.update({
                'apex_x_position': apex_x,
                'apex_relative_to_center': 'left' if apex_x < cent[0] else 'right',
                'has_prominent_apex': max(top_extent, bottom_extent) > 1.5 * min(top_extent, bottom_extent)
            })
        
        # Enhanced orientation analysis
        if len(verts) >= 2:
            # Calculate dominant direction using PCA-like approach
            points = np.array(verts)
            if len(points) > 1:
                # Center the points
                centered = points - np.mean(points, axis=0)
                
                # Calculate covariance matrix
                cov_matrix = np.cov(centered.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                
                # Principal direction
                principal_direction = eigenvecs[:, -1]  # Eigenvector with largest eigenvalue
                angle = np.degrees(np.arctan2(principal_direction[1], principal_direction[0]))
                
                # Normalize to [0, 180) for principal axis
                if angle < 0:
                    angle += 180
                
                features.update({
                    'principal_orientation': angle,
                    'orientation_variance': eigenvals[-1] / (eigenvals[0] + 1e-10),  # Ratio of principal to secondary axis
                })
        
        # Enhanced curvature analysis
        if len(verts) >= 3:
            # Calculate curvature score based on deviation from straight line
            points = np.array(verts)
            curvature_scores = []
            
            for i in range(1, len(points) - 1):
                # Calculate angle between three consecutive points
                v1 = points[i] - points[i-1]
                v2 = points[i+1] - points[i]
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 1e-6 and v2_norm > 1e-6:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
                    angle = np.arccos(cos_angle)
                    curvature_scores.append(abs(angle - np.pi))  # Deviation from straight line
            
            avg_curvature = np.mean(curvature_scores) if curvature_scores else 0.0
            max_curvature = np.max(curvature_scores) if curvature_scores else 0.0
            
            features.update({
                'curvature_score': avg_curvature,
                'max_curvature': max_curvature,
                'is_highly_curved': avg_curvature > 0.5,  # Threshold for significant curvature
            })
        
        # Geometric complexity measures
        features.update({
            'vertex_count': len(verts),
            'geometric_complexity': len(verts) * (1 + features.get('horizontal_asymmetry', 0) + features.get('vertical_asymmetry', 0))
        })
        
        # Ensure essential fields are not empty
        if 'stroke_count' not in features or features['stroke_count'] is None:
            features['stroke_count'] = 1  # Default for single shapes
        
        if 'is_valid' not in features:
            features['is_valid'] = True  # Default to valid unless explicitly marked otherwise
            
        # Ensure numeric fields have valid defaults
        numeric_defaults = {
            'curvature_score': 0.0,
            'max_curvature': 0.0,
            'horizontal_asymmetry': 0.0,
            'vertical_asymmetry': 0.0,
            'left_extent': 0.0,
            'right_extent': 0.0,
            'top_extent': 0.0,
            'bottom_extent': 0.0,
            'principal_orientation': 0.0,
            'orientation_variance': 1.0,
            'aspect_ratio': 1.0,
            'compactness': 0.0,
            'area': 0.0,
            'perimeter': 0.0
        }
        
        for field, default_val in numeric_defaults.items():
            if field not in features or features[field] is None or (isinstance(features[field], (int, float)) and np.isnan(features[field])):
                features[field] = default_val
        
    except Exception as e:
        logging.warning(f"Could not compute enhanced geometry features: {e}")
    
    return features

def _group_strokes_into_shapes(stroke_objects: List[Dict[str, Any]], parent_shape_id: str) -> List[Dict[str, Any]]:
    """
    Groups connected primitive strokes into composite shapes using a graph approach.
    This is a more robust implementation based on endpoint connectivity.
    """
    if not stroke_objects:
        return []

    # Build a graph to find connected components of primitives
    connectivity_graph = nx.Graph()
    # Add nodes with index to handle multiple strokes
    for i, node in enumerate(stroke_objects):
        connectivity_graph.add_node(i)

    # Add edges for adjacent primitives by checking endpoint proximity
    for i in range(len(stroke_objects)):
        for j in range(i + 1, len(stroke_objects)):
            node_a = stroke_objects[i]
            node_b = stroke_objects[j]
            
            ep_a = node_a.get('endpoints')
            ep_b = node_b.get('endpoints')

            if ep_a and ep_b:
                # Check all 4 combinations of start/end points for a connection
                if any(np.allclose(p1, p2, atol=1e-5) for p1 in ep_a for p2 in ep_b if p1 is not None and p2 is not None):
                    connectivity_graph.add_edge(i, j)

    new_shapes = []
    # Each connected component in the graph represents a single, continuous shape
    for i, component_indices in enumerate(nx.connected_components(connectivity_graph)):
        component_primitives = [stroke_objects[idx] for idx in component_indices]
        
        # Combine vertices in the order they were drawn
        sorted_primitives = sorted(component_primitives, key=lambda x: x.get('action_index', 0))
        
        all_verts = []
        if sorted_primitives:
            all_verts.extend(sorted_primitives[0]['vertices'])
            for k in range(1, len(sorted_primitives)):
                prev_v_last = all_verts[-1]
                curr_v = sorted_primitives[k]['vertices']
                # Smartly merge vertex lists to avoid duplicates at connection points
                if np.allclose(prev_v_last, curr_v[0]):
                    all_verts.extend(curr_v[1:])
                else:
                    # This case is for non-sequential connections, simple extend is a fallback
                    all_verts.extend(curr_v)

        if not all_verts:
            continue

        shape_id = f"{parent_shape_id}_shape_{i}"
        is_closed = len(all_verts) > 2 and np.allclose(all_verts[0], all_verts[-1], atol=1e-5)
        # ACTION PROGRAMS ONLY: Object type comes from action commands only
        # Get object_type from the first constituent stroke that has it
        object_type = None
        for obj in component_primitives:
            if obj.get('object_type'):
                object_type = obj['object_type']
                break
        
        # Fallback if no object_type found
        if not object_type:
            is_closed = len(all_verts) > 2 and np.allclose(all_verts[0], all_verts[-1], atol=1e-5)
            object_type = 'polygon' if is_closed else 'line'  # Use 'line' instead of 'open_curve'
        
        # CRITICAL FIX: Determine shape_type from constituent strokes for composite shapes
        # Priority: use the most common discovered Bongard-LOGO shape type from constituent strokes
        discovered_types = ['normal', 'circle', 'square', 'triangle', 'zigzag']
        constituent_shape_types = [obj.get('shape_type') for obj in component_primitives if obj.get('shape_type') in discovered_types]
        
        composite_shape_type = None
        if constituent_shape_types:
            # Use the most common shape type from constituent strokes
            from collections import Counter
            shape_type_counts = Counter(constituent_shape_types)
            composite_shape_type = shape_type_counts.most_common(1)[0][0]
        else:
            # ACTION PROGRAMS ONLY: No fallback to object type inference
            # All shape types must come from action programs only
            composite_shape_type = None
        
        # ACTION PROGRAMS ONLY: No default shape type assignment
        # If no shape type found in action programs, leave as None
        
        # Calculate enhanced geometric properties for the new composite shape
        base_obj = sorted_primitives[0]
        
        # Determine composite stroke type from constituent strokes
        constituent_stroke_types = [obj.get('stroke_type') for obj in component_primitives if obj.get('stroke_type')]
        if constituent_stroke_types:
            from collections import Counter
            stroke_type_counts = Counter(constituent_stroke_types)
            composite_stroke_type = stroke_type_counts.most_common(1)[0][0]
        else:
            composite_stroke_type = 'unknown'
        
        enhanced_features = _calculate_enhanced_geometry_features(all_verts, {
            'object_id': shape_id, 
            'parent_shape_id': parent_shape_id,
            'object_type': object_type, 
            'shape_type': composite_shape_type,  # CRITICAL: Preserve shape_type for composite shapes
            'stroke_type': composite_stroke_type,  # CRITICAL: Preserve stroke_type for composite shapes
            'source': 'geometric_grouping',
            'vertices': all_verts, 
            'is_closed': is_closed, 
            'stroke_count': len(component_primitives),
            'label': base_obj.get('label'), 
            'shape_label': base_obj.get('shape_label'),
            'category': base_obj.get('category'), 
            'original_record_idx': base_obj.get('original_record_idx'),
            'image_path': base_obj.get('image_path'), 
            'relationships': [], 
            'is_valid': True,
        })
        
        shape_obj = enhanced_features
        new_shapes.append(shape_obj)

        # Update original strokes to link them to their new parent shape
        for prim in component_primitives:
            prim['part_of'] = shape_id
            # The composite shape 'has' the primitive as a part
            shape_obj['relationships'].append(f"has_part_{prim['object_id']}")
            
    return new_shapes


def _ensure_complete_node_data(graph):
    """Ensure all nodes have complete data with proper defaults before CSV export"""
    required_fields = {
        'stroke_count': 1,
        'is_valid': True,
        'curvature_score': 0.0,
        'max_curvature': 0.0,
        'horizontal_asymmetry': 0.0,
        'vertical_asymmetry': 0.0,
        'left_extent': 0.0,
        'right_extent': 0.0,
        'top_extent': 0.0,
        'bottom_extent': 0.0,
        'principal_orientation': 0.0,
        'orientation_variance': 1.0,
        'aspect_ratio': 1.0,
        'compactness': 0.0,
        'area': 0.0,
        'perimeter': 0.0,
        'vertex_count': 0,
        'geometric_complexity': 0.0,
        'has_prominent_apex': False,
        'is_highly_curved': False,
        'apex_relative_to_center': 'center',
        'apex_x_position': 0.0,
        'bbox_aspect_ratio': 1.0,
        'vl_embed': [0.0] * 512,
        'vl_embed_norm': 0.0,
        'vl_embed_nonzero_ratio': 0.0,
        'has_vl_features': False,
        'data_completeness_score': 0.0,
        'missing_field_count': 0,
        'object_type': 'unknown',
        'source': 'unknown',
        'label': '',
        'shape_label': '',
        'category': '',
        'programmatic_label': '',
        'image_path': '',
        'original_record_idx': 0,
        'relationships': []
    }
    
    for node_id, node_data in graph.nodes(data=True):
        for field, default_value in required_fields.items():
            if field not in node_data or node_data[field] is None:
                node_data[field] = default_value
            elif isinstance(node_data[field], (int, float)) and np.isnan(node_data[field]):
                node_data[field] = default_value


async def _process_single_problem(problem_id: str, problem_records: List[Dict[str, Any]], feature_cache, *, args=None):
    """
    Processes a single Bongard problem to generate a scene graph for EACH image.
    Returns a dictionary mapping image_id to its scene graph.
    """
    
    # Get ConceptNet KB instance if available
    kb = None
    try:
        # Try to get the ConceptNet instance from the global state
        if 'ConceptNetClient' in globals():
            from scripts.build_scene_graphs import ConceptNetClient
            kb = ConceptNetClient.get()
            if kb:
                logging.info(f"[_process_single_problem] ConceptNet KB available: {type(kb)}")
            else:
                logging.info("[_process_single_problem] ConceptNet KB not available")
        else:
            # Fallback: try to create a new instance
            from src.commonsense_kb_api import ConceptNetAPI
            kb = ConceptNetAPI()
            logging.info(f"[_process_single_problem] Created new ConceptNet KB instance: {type(kb)}")
    except Exception as e:
        logging.warning(f"[_process_single_problem] Failed to initialize ConceptNet KB: {e}")
        kb = None
    # LOGO MODE: Use enhanced physics computation for vertex processing
    try:
        from src.logo_physics import LOGOPhysicsComputation
        
        logo_physics = LOGOPhysicsComputation()
        
        # Process all records using LOGO physics computation
        all_objects_by_image = logo_physics.process_problem_records(
            problem_id, 
            problem_records,
            use_enhanced_features=True
        )
        
        logging.info(f"LOGO physics computation processed {len(all_objects_by_image)} images with enhanced attributes")
        
    except Exception as e:
        logging.error(f"LOGO physics computation failed: {e}")
        # ACTION PROGRAMS ONLY: Fallback to manual processing if LOGO physics fails
        logging.info("Falling back to manual action program processing...")
        
        for idx, rec in enumerate(problem_records):
            parent_shape_id = f"{problem_id}_{idx}"
            # Use flattened_actions instead of nested action_program
            action_program = rec.get('flattened_actions', rec.get('action_program', []))
            # Common attributes for all objects in this record
            common_attrs = {
                'label': rec.get('label', ''), 'shape_label': rec.get('shape_label', ''),
                'category': rec.get('category', ''), 'programmatic_label': rec.get('programmatic_label', ''),
                'image_path': rec.get('image_path'), 'original_record_idx': idx,
            }
            if not action_program:
                continue
            turtle_pos = [0.0, 0.0]
            turtle_heading = 0.0
            
            # LOGO simulation to generate strokes
            last_stroke_obj = None
            for stroke_idx, cmd in enumerate(action_program):
                parsed_cmd = parse_action_command(cmd)
                if not parsed_cmd: continue
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
                
                if not verts: continue

                # Apply stroke regularization to reduce over-segmentation
                regularized_verts = _regularize_stroke_vertices(verts, tolerance=3.0)
                
                obj_id = f"{problem_id}_{idx}_{stroke_idx}"
                
                # Calculate enhanced geometric features with regularized vertices
                enhanced_features = _calculate_enhanced_geometry_features(regularized_verts, {
                    'object_id': obj_id, 
                    'parent_shape_id': parent_shape_id, 
                    'action_index': stroke_idx,
                    'vertices': regularized_verts, 
                    'original_vertices': verts,  # Keep original for debugging
                    'object_type': assign_object_type(regularized_verts, cmd), 
                    'action_command': cmd,
                    'endpoints': [regularized_verts[0], regularized_verts[-1]], 
                    'length': length, 
                    'orientation': orientation,
                    'source': 'action_program', 
                    'is_closed': len(verts) > 2 and np.allclose(verts[0], verts[-1], atol=1e-5),
                    # CRITICAL: Assign stroke_type based on action_command for proper arc vs line differentiation
                    'stroke_type': 'arc' if cmd and cmd.startswith('arc_') else 'line' if cmd and cmd.startswith('line_') else 'unknown',
                    # CRITICAL: Assign shape_type from parsed command for proper normal/triangle/square/circle/zigzag differentiation
                    'shape_type': parsed_cmd.get('shape', 'unknown') if parsed_cmd else 'unknown',
                    **common_attrs
                })
                
                relationships = []
                if last_stroke_obj:
                    # Check for adjacency based on endpoint proximity
                    if np.allclose(last_stroke_obj['endpoints'][-1], verts[0], atol=1e-5):
                        relationships.append(f"adjacent_to_{last_stroke_obj['object_id']}")
                
                enhanced_features['relationships'] = relationships
                obj = enhanced_features
                # SEMANTIC ENHANCEMENT: Add semantic parsing to existing object
                obj = enhance_node_with_semantic_features(obj, action_program)
                # Ensure shape_type is set to discovered Bongard-LOGO type if detected
                # Priority: shape_type from semantic enhancement > object_type
                discovered_types = ['normal', 'circle', 'square', 'triangle', 'zigzag']
                shape_type = obj.get('shape_type')
                if shape_type not in discovered_types:
                    # Try to infer from action_command if possible
                    cmd_str = str(obj.get('action_command', ''))
                    for t in discovered_types:
                        if t in cmd_str:
                            obj['shape_type'] = t
                            break
                # If still not set, fallback to object_type if it's a discovered type
                if obj.get('shape_type') not in discovered_types and obj.get('object_type') in discovered_types:
                    obj['shape_type'] = obj['object_type']
                all_objects_by_image[parent_shape_id].append(obj)
                last_stroke_obj = obj

    # --- Graph Construction (per image) ---
    final_graphs = {}
    all_objects_for_return = []
    
    # ACTION PROGRAMS ONLY: No default predicate assignment
    # Predicates must be derived from action program analysis
    chosen_predicate = None  # Will be set from action program analysis only
    predicate_params = None
    
    # Separate positive and negative examples for contrastive analysis
    positive_objects = []
    negative_objects = []
    for parent_shape_id, objects_in_image in all_objects_by_image.items():
        for obj in objects_in_image:
            all_objects_for_return.append(obj)
            
            # Fix categorization: Use label field instead of category field
            # The category field contains 'bd' (dataset category), but we need positive/negative classification
            label = str(obj.get('label', '')).lower()
            image_path = obj.get('image_path', '')
            
            # Determine if this is positive or negative based on label or image path
            is_positive = False
            is_negative = False
            
            if label in ['positive', 'category_1']:
                is_positive = True
            elif label in ['negative', 'category_0']:
                is_negative = True
            elif 'category_1' in image_path or 'positive' in image_path:
                is_positive = True
            elif 'category_0' in image_path or 'negative' in image_path:
                is_negative = True
            
            if is_positive:
                positive_objects.append(obj)
                logging.info(f"Categorized as POSITIVE: {obj.get('object_id')} (label={label}, path={image_path})")
            elif is_negative:
                negative_objects.append(obj)
                logging.info(f"Categorized as NEGATIVE: {obj.get('object_id')} (label={label}, path={image_path})")
            else:
                logging.info(f"NOT CATEGORIZED: {obj.get('object_id')} (label={label}, category={obj.get('category')}, path={image_path})")
    
    # Perform contrastive predicate induction if we have both positive and negative examples
    if positive_objects and negative_objects:
        try:
            from src.scene_graphs_building.predicate_induction import induce_predicate_for_problem
            
            logging.info(f"Performing contrastive predicate induction for {problem_id}: {len(positive_objects)} positive, {len(negative_objects)} negative objects")
            
            # Use the enhanced predicate induction with contrastive analysis
            chosen_predicate, predicate_params = induce_predicate_for_problem(
                objects=None,  # Not used in contrastive mode
                positive_objects=positive_objects,
                negative_objects=negative_objects
            )
            
            logging.info(f"Induced predicate for {problem_id}: {chosen_predicate}")
            if predicate_params:
                logging.info(f"Predicate parameters: {predicate_params}")
                
        except Exception as e:
            logging.warning(f"Predicate induction failed for {problem_id}: {e}")
            logging.debug(traceback.format_exc())
    else:
        logging.info(f"No contrastive analysis for {problem_id}: {len(positive_objects)} pos, {len(negative_objects)} neg objects")
        # Debug: Show a sample of objects to understand categorization issue
        if all_objects_for_return:
            sample_objects = all_objects_for_return[:3]  # Show first 3 objects
            for obj in sample_objects:
                logging.info(f"Sample object: {obj.get('object_id')} - category='{obj.get('category')}', label='{obj.get('label')}', image_path='{obj.get('image_path')}'")

    for parent_shape_id, objects_in_image in all_objects_by_image.items():
        # --- Hierarchical Grouping Step ---
        # Correctly group strokes into higher-level shapes
        newly_created_shapes = _group_strokes_into_shapes(objects_in_image, parent_shape_id)
        
        # Combine primitives and new composite shapes
        all_nodes_for_graph = objects_in_image + newly_created_shapes
        G = nx.MultiDiGraph()
        for obj in all_nodes_for_graph:
            # Ensure composite shapes get correct shape_type if possible
            discovered_types = ['normal', 'circle', 'square', 'triangle', 'zigzag']
            shape_type = obj.get('shape_type')
            if shape_type not in discovered_types:
                # Try to infer from shape_label or object_type
                label = str(obj.get('shape_label', ''))
                for t in discovered_types:
                    if t in label:
                        obj['shape_type'] = t
                        break
            if obj.get('shape_type') not in discovered_types and obj.get('object_type') in discovered_types:
                obj['shape_type'] = obj['object_type']
            G.add_node(obj['object_id'], **obj)
        
        # === ENHANCED SHAPE DETECTION ===
        # Detect quarter circles and other composite shapes from connected line segments
        G = _detect_and_add_composite_shapes(G, parent_shape_id)
        
        all_objects_for_return.extend(all_nodes_for_graph)

        # Add edges based on relationships defined during creation
        for u, data in G.nodes(data=True):
            # Connect primitives with 'adjacent_endpoints'
            if 'adjacent_to_' in ''.join(data.get('relationships', [])):
                for rel in data.get('relationships', []):
                    if rel.startswith('adjacent_to_'):
                        v_id = rel.split('adjacent_to_')[1]
                        if G.has_node(v_id):
                            G.add_edge(u, v_id, predicate='adjacent_endpoints', source='program')
            
            # Connect composite shapes to their parts with 'part_of'
            if 'has_part_' in ''.join(data.get('relationships', [])):
                 for rel in data.get('relationships', []):
                    if rel.startswith('has_part_'):
                        v_id = rel.split('has_part_')[1]
                        if G.has_node(v_id):
                            # Edge direction: composite -> part
                            G.add_edge(u, v_id, predicate='part_of', source='geometric_grouping')

        # Add advanced predicate edges between ALL relevant node pairs
        all_nodes = list(G.nodes(data=True))
        
        # Initialize semantic predicate engine
        semantic_engine = BongardPredicateEngine()
        
        from itertools import combinations
        for (id_a, data_a), (id_b, data_b) in combinations(all_nodes, 2):
            # Apply advanced predicates to ALL node pairs (not just higher-level shapes)
            # This ensures comprehensive spatial relationships are captured
            try:
                for pred_name, pred_func in ADVANCED_PREDICATE_REGISTRY.items():
                    # Check A -> B
                    if pred_func(data_a, data_b):
                        G.add_edge(id_a, id_b, predicate=pred_name, source='advanced_geometry')
                    # Check B -> A for non-symmetric predicates
                    if pred_name in ['contains', 'is_above'] and pred_func(data_b, data_a):
                        G.add_edge(id_b, id_a, predicate=pred_name, source='advanced_geometry')
                        
                # SEMANTIC ENHANCEMENT: Add Bongard-relevant predicates
                semantic_predicates_a = semantic_engine.evaluate_predicates(data_a)
                semantic_predicates_b = semantic_engine.evaluate_predicates(data_b)
                semantic_predicates_ab = semantic_engine.evaluate_predicates(data_a, data_b)
                
                # Add unary semantic predicates as node attributes
                for pred in semantic_predicates_a:
                    G.nodes[id_a][f'semantic_{pred}'] = True
                for pred in semantic_predicates_b:
                    G.nodes[id_b][f'semantic_{pred}'] = True
                    
                # Add binary semantic predicates as edges
                for pred in semantic_predicates_ab:
                    G.add_edge(id_a, id_b, predicate=pred, source='semantic_bongard')
                    
            except Exception as e:
                logging.warning(f"Failed to apply advanced predicates between {id_a} and {id_b}: {e}")

        # === ENHANCED PROCESSING: Add semantic abstraction and feature selection ===
        
        # === ADVANCED TECHNIQUES INTEGRATION ===
        
        # 1. Enhanced Vision-Language (VL) Features with Missing Calculations
        if getattr(args, 'use_vl', False):
            try:
                from src.scene_graphs_building.vl_features import CLIPEmbedder
                from src.scene_graphs_building.data_loading import robust_image_open
                
                logging.info("Loading CLIP model for enhanced VL features...")
                clip_embedder = CLIPEmbedder()
                
                # Validate CLIP model is loaded properly
                if not hasattr(clip_embedder, 'model') or clip_embedder.model is None:
                    logging.error("CLIP model failed to load, skipping VL enhancements")
                    raise Exception("CLIP model not available")
                
                logging.info(f"CLIP model loaded successfully on device: {clip_embedder.device}")
                
                # Extract enhanced features for each node
                for node_id, node_data in G.nodes(data=True):
                    # ACTION PROGRAMS: Construct image path from problem structure
                    # Since action programs don't include image paths, construct them from the node data
                    image_path = None
                    
                    # Get the label from node data to determine if it's positive (category_1) or negative (category_0)
                    label = node_data.get('label', 'positive')
                    category_folder = 'category_1' if label in ['positive', 'category_1'] else 'category_0'
                    
                    # Extract base problem name from problem_id
                    # problem_id format: bd_asymmetric_unbala_x_0000
                    # Extract category (bd, ff, hd) from problem_id
                    category = problem_id.split('_')[0] if '_' in problem_id else 'bd'
                    
                    # Get original record index to determine which image (0.png, 1.png, etc.)
                    original_record_idx = node_data.get('original_record_idx', 0)
                    
                    # CRITICAL FIX: Cap the record index to avoid accessing non-existent images
                    # Bongard problems typically have 7 images per category (0-6), but may have more records
                    max_image_index = 6  # Images are numbered 0-6 (7 total images)
                    capped_record_idx = min(original_record_idx, max_image_index)
                    
                    if original_record_idx > max_image_index:
                        logging.debug(f"Capping record index {original_record_idx} to {capped_record_idx} for node {node_id}")
                    
                    # Construct the image path
                    image_path = f"data/raw/ShapeBongard_V2/{category}/images/{problem_id}\\{category_folder}\\{capped_record_idx}.png"
                    
                    # Verify the constructed path exists (after remapping)
                    from src.scene_graphs_building.data_loading import remap_path
                    try:
                        remapped_path = remap_path(image_path)
                        if os.path.exists(remapped_path):
                            logging.debug(f"Constructed valid image path for node {node_id}: {image_path} -> {remapped_path}")
                        else:
                            logging.warning(f"Constructed image path does not exist for node {node_id}: {remapped_path}")
                            image_path = None
                    except Exception as e:
                        logging.warning(f"Failed to validate image path for node {node_id}: {e}")
                        image_path = None
                    
                    if image_path:
                        try:
                            logging.debug(f"Computing VL embedding for node {node_id} with image {image_path}")
                            # Enhanced VL embedding with context
                            vl_data = clip_embedder.compute_enhanced_vl_embedding(
                                image_path, node_data, f"bongard puzzle {problem_id}"
                            )
                            
                            # Add enhanced VL features to node
                            G.nodes[node_id].update(vl_data)
                            G.nodes[node_id]['has_vl_features'] = True
                            
                            # Compute curvature metrics for curves and lines
                            vertices = node_data.get('vertices', [])
                            if vertices and len(vertices) >= 3:
                                curvature_data = clip_embedder.compute_curvature_metrics(vertices)
                                
                                # Add curvature features with enhanced_ prefix to distinguish
                                for key, value in curvature_data.items():
                                    G.nodes[node_id][f'enhanced_{key}'] = value
                                
                                # Update the original curvature_score if it was missing/zero
                                if G.nodes[node_id].get('curvature_score', 0.0) == 0.0:
                                    G.nodes[node_id]['curvature_score'] = curvature_data['curvature_mean']
                            
                            # Compute additional geometric features
                            if vertices:
                                vertices_array = np.array(vertices)
                                
                                # Bounding box calculations
                                if len(vertices_array) > 0:
                                    x_coords = vertices_array[:, 0]
                                    y_coords = vertices_array[:, 1]
                                    bbox_width = np.max(x_coords) - np.min(x_coords)
                                    bbox_height = np.max(y_coords) - np.min(y_coords)
                                    G.nodes[node_id]['bbox_aspect_ratio'] = bbox_width / max(bbox_height, 1e-6)
                                
                                # Convex hull area for complexity measure
                                if len(vertices_array) >= 3:
                                    try:
                                        from scipy.spatial import ConvexHull
                                        hull = ConvexHull(vertices_array)
                                        G.nodes[node_id]['convex_hull_area'] = hull.volume  # volume is area in 2D
                                    except:
                                        G.nodes[node_id]['convex_hull_area'] = 0.0
                                
                                # Enhanced symmetry calculations
                                centroid = node_data.get('centroid', [0, 0])
                                if centroid and len(centroid) == 2:
                                    # Horizontal symmetry score
                                    left_points = vertices_array[vertices_array[:, 0] < centroid[0]]
                                    right_points = vertices_array[vertices_array[:, 0] > centroid[0]]
                                    h_symmetry = 1.0 / (1.0 + abs(len(left_points) - len(right_points)))
                                    G.nodes[node_id]['horizontal_symmetry_score'] = h_symmetry
                                    
                                    # Vertical symmetry score
                                    top_points = vertices_array[vertices_array[:, 1] < centroid[1]]
                                    bottom_points = vertices_array[vertices_array[:, 1] > centroid[1]]
                                    v_symmetry = 1.0 / (1.0 + abs(len(top_points) - len(bottom_points)))
                                    G.nodes[node_id]['vertical_symmetry_score'] = v_symmetry
                                
                                # Turning angle variation and inflection point count
                                if len(vertices_array) >= 4:
                                    turning_angles = []
                                    for i in range(1, len(vertices_array) - 1):
                                        v1 = vertices_array[i] - vertices_array[i-1]
                                        v2 = vertices_array[i+1] - vertices_array[i]
                                        
                                        norm_v1 = np.linalg.norm(v1)
                                        norm_v2 = np.linalg.norm(v2)
                                        
                                        if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                                            cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                                            angle = np.arccos(cos_angle)
                                            turning_angles.append(angle)
                                    
                                    if turning_angles:
                                        G.nodes[node_id]['turning_angle_variation'] = float(np.std(turning_angles))
                                        
                                        # Count inflection points (significant direction changes)
                                        inflection_threshold = np.pi / 4  # 45 degrees
                                        significant_turns = [a for a in turning_angles if a > inflection_threshold]
                                        G.nodes[node_id]['inflection_points'] = len(significant_turns)
                            
                            # Data completeness scoring
                            required_fields = ['area', 'perimeter', 'centroid', 'vertices', 'object_type']
                            missing_fields = [f for f in required_fields if not node_data.get(f)]
                            completeness_score = (len(required_fields) - len(missing_fields)) / len(required_fields)
                            G.nodes[node_id]['data_completeness_score'] = completeness_score
                            G.nodes[node_id]['missing_field_count'] = len(missing_fields)
                            
                        except Exception as e:
                            logging.warning(f"Failed to extract enhanced features for node {node_id}: {e}")
                            # Try to compute VL embedding with CLIP
                            try:
                                if clip_embedder and clip_embedder.model is not None:
                                    # Use object image path and bounding box for precise embedding
                                    img_path = node_data.get('image_path')
                                    bbox = node_data.get('bounding_box')
                                    if img_path and bbox:
                                        vl_embed = clip_embedder.embed_image(img_path, bounding_box=bbox)
                                        if vl_embed is not None and not np.allclose(vl_embed, 0):
                                            G.nodes[node_id]['vl_embed'] = vl_embed.tolist()
                                            G.nodes[node_id]['vl_embed_norm'] = float(np.linalg.norm(vl_embed))
                                            G.nodes[node_id]['vl_embed_nonzero_ratio'] = float(np.count_nonzero(vl_embed) / len(vl_embed))
                                        else:
                                            G.nodes[node_id]['vl_embed'] = [0.0] * 512
                                            G.nodes[node_id]['vl_embed_norm'] = 0.0
                                            G.nodes[node_id]['vl_embed_nonzero_ratio'] = 0.0
                                    else:
                                        G.nodes[node_id]['vl_embed'] = [0.0] * 512
                                        G.nodes[node_id]['vl_embed_norm'] = 0.0
                                        G.nodes[node_id]['vl_embed_nonzero_ratio'] = 0.0
                                else:
                                    G.nodes[node_id]['vl_embed'] = [0.0] * 512
                                    G.nodes[node_id]['vl_embed_norm'] = 0.0
                                    G.nodes[node_id]['vl_embed_nonzero_ratio'] = 0.0
                            except Exception as e:
                                logging.warning(f"VL embedding failed for node {node_id}: {e}")
                                G.nodes[node_id]['vl_embed'] = [0.0] * 512
                                G.nodes[node_id]['vl_embed_norm'] = 0.0
                                G.nodes[node_id]['vl_embed_nonzero_ratio'] = 0.0
                            G.nodes[node_id]['has_vl_features'] = False
                            G.nodes[node_id]['data_completeness_score'] = 0.0
                            G.nodes[node_id]['missing_field_count'] = len(['area', 'perimeter', 'centroid', 'vertices', 'object_type'])
                
                # Add VL-based edges with enhanced similarity (limited to most meaningful)
                vl_edges = clip_embedder.contrastive_edges(
                    [G.nodes[n] for n in G.nodes()], 
                    threshold=0.25,  # Increased threshold for more selective edges
                    use_roi=getattr(args, 'use_roi', False),
                    logo_mode=True
                )
                
                # Limit VL edges to prevent explosion - only add top similarity pairs
                if len(vl_edges) > 10:  # Limit to max 10 VL edges per problem
                    vl_edges = sorted(vl_edges, key=lambda x: x.get('similarity', 0.0), reverse=True)[:10]
                
                for edge_data in vl_edges:
                    if 'source' in edge_data and 'target' in edge_data:
                        # Only add if similarity is above threshold and no duplicate edge exists
                        if (edge_data.get('similarity', 0.0) > 0.25 and 
                            not G.has_edge(edge_data['source'], edge_data['target'])):
                            G.add_edge(edge_data['source'], edge_data['target'], 
                                     predicate='visual_similarity', source='vision_language',
                                     similarity_score=edge_data.get('similarity', 0.0))
                
                logging.info(f"Added {len([e for e in vl_edges if e.get('similarity', 0.0) > 0.25])} VL-based edges to graph for {parent_shape_id}")
                
            except Exception as e:
                logging.warning(f"Failed to apply VL features: {e}")
        
        # 2. Enhanced Motif Mining with Missing Feature Calculations
        if getattr(args, 'use_motifs', False):
            try:
                from src.scene_graphs_building.motif_miner import MotifMiner
                
                motif_miner = MotifMiner()
                node_objects = [G.nodes[n] for n in G.nodes()]
                
                # Discover motifs and create super-nodes
                motif_dict, motif_nodes = motif_miner.cluster_motifs(node_objects, method='graph+type')
                
                # Add enhanced motif features using CLIPEmbedder if available
                if getattr(args, 'use_vl', False) and 'clip_embedder' in locals():
                    for motif_node in motif_nodes:
                        try:
                            # Get motif members
                            member_ids = motif_node.get('member_nodes', [])
                            motif_members = [G.nodes[mid] for mid in member_ids if G.has_node(mid)]
                            
                            # Get motif relationships
                            motif_relationships = []
                            for u in member_ids:
                                for v in member_ids:
                                    if u != v and G.has_edge(u, v):
                                        for edge_data in G[u][v].values():
                                            motif_relationships.append(edge_data)
                            
                            # Compute comprehensive motif features
                            motif_features = clip_embedder.compute_motif_features(
                                motif_members, motif_relationships
                            )
                            
                            # Add motif features to node
                            motif_node.update(motif_features)
                            
                            # Compute motif-level VL embedding if image data available
                            image_paths = []
                            member_bboxes = []
                            for member in motif_members:
                                # Try to find image path from problem records
                                for rec in problem_records:
                                    if rec.get('image_path') and parent_shape_id in rec.get('image_path', ''):
                                        image_paths.append(rec['image_path'])
                                        bbox = member.get('bounding_box')
                                        member_bboxes.append(bbox if bbox else [0, 0, 100, 100])
                                        break
                            
                            if image_paths:
                                try:
                                    motif_vl_embedding = clip_embedder.compute_motif_embedding(
                                        image_paths[:1], member_bboxes[:1]  # Use first image as representative
                                    )
                                    motif_node['motif_vl_embed'] = motif_vl_embedding
                                    motif_node['has_motif_vl_features'] = True
                                except Exception as e:
                                    logging.warning(f"Failed to compute motif VL embedding: {e}")
                                    motif_node['motif_vl_embed'] = [0.0] * 512
                                    motif_node['has_motif_vl_features'] = False
                            
                            # Calculate motif complexity metrics
                            motif_node['motif_complexity_score'] = (
                                motif_features['motif_type_diversity'] * 0.3 +
                                motif_features['motif_relationship_diversity'] * 0.3 +
                                motif_features['motif_internal_connectivity'] * 0.2 +
                                (1.0 - motif_features['motif_size_uniformity']) * 0.2  # diversity bonus
                            )
                            
                        except Exception as e:
                            logging.warning(f"Failed to compute enhanced motif features: {e}")
                            # Add default empty motif features
                            empty_features = clip_embedder._empty_motif_features() if 'clip_embedder' in locals() else {}
                            motif_node.update(empty_features)
                
                # Add motif super-nodes to graph
                for motif_node in motif_nodes:
                    motif_id = motif_node['object_id']
                    G.add_node(motif_id, **motif_node)
                    
                    # Add part_of_motif edges
                    for member_id in motif_node.get('member_nodes', []):
                        if G.has_node(member_id):
                            G.add_edge(member_id, motif_id, predicate='part_of_motif', source='motif_mining')
                
                # Discover structural patterns specific to Bongard problems
                bridge_patterns = motif_miner.find_bridge_patterns(G)
                apex_patterns = motif_miner.find_apex_patterns(G)
                symmetry_patterns = motif_miner.find_symmetry_patterns(G)
                
                # Add pattern-based edges with enhanced metadata (limited to high-confidence patterns)
                pattern_count = 0
                max_pattern_edges = 15  # Limit total pattern edges per problem
                all_patterns = bridge_patterns + apex_patterns + symmetry_patterns
                
                # Sort patterns by confidence and only use top ones
                sorted_patterns = sorted(all_patterns, key=lambda x: x.get('confidence', 0.0), reverse=True)[:5]
                
                for pattern in sorted_patterns:
                    if pattern_count >= max_pattern_edges:
                        break
                        
                    pattern_type = pattern.get('type', 'unknown')
                    pattern_confidence = pattern.get('confidence', 0.0)
                    
                    # Only add patterns with high confidence
                    if pattern_confidence < 0.7:
                        continue
                    
                    pattern_nodes = pattern.get('nodes', [])
                    # Limit edges per pattern to prevent explosion
                    edges_added_this_pattern = 0
                    max_edges_per_pattern = min(3, len(pattern_nodes))
                    
                    for i, node_a in enumerate(pattern_nodes):
                        if edges_added_this_pattern >= max_edges_per_pattern:
                            break
                        for node_b in pattern_nodes[i+1:]:
                            if (edges_added_this_pattern >= max_edges_per_pattern or
                                pattern_count >= max_pattern_edges):
                                break
                            if G.has_node(node_a) and G.has_node(node_b) and not G.has_edge(node_a, node_b):
                                G.add_edge(node_a, node_b, 
                                         predicate=f'forms_{pattern_type}_pattern', 
                                         source='motif_discovery',
                                         pattern_confidence=pattern_confidence,
                                         pattern_id=f'pattern_{pattern_count}')
                                pattern_count += 1
                                edges_added_this_pattern += 1
                
                logging.info(f"Enhanced motif mining added {len(motif_nodes)} super-nodes with comprehensive features and {pattern_count} high-confidence pattern relationships")
                
            except Exception as e:
                logging.warning(f"Failed to apply motif mining: {e}")
        
        # 3. GNN-based Reasoning and Graph Enhancement
        if getattr(args, 'use_gnn', False):
            try:
                from src.reasoner.gnn_reasoner import GNNReasoner
                import torch
                
                # LOGO MODE: Enhanced GNN reasoning with LOGO-derived features
                if G.number_of_nodes() > 0:
                    # Initialize GNN reasoner with proper input dimensions
                    # Calculate expected input dimension based on LOGO features
                    sample_node = next(iter(G.nodes(data=True)))[1]
                    
                    # LOGO MODE: Ensure all nodes have required features for GNN
                    required_gnn_features = ['area', 'aspect_ratio', 'orientation', 'curvature_score', 'perimeter', 'centroid', 
                                           'horizontal_asymmetry', 'vertical_asymmetry', 'apex_x_position', 'is_highly_curved']
                    
                    for node_id, node_data in G.nodes(data=True):
                        # Fill missing features with defaults
                        for feat in required_gnn_features:
                            if feat not in node_data or node_data[feat] is None:
                                if feat == 'is_highly_curved':
                                    node_data[feat] = False
                                elif feat in ['area', 'aspect_ratio', 'orientation', 'curvature_score', 'perimeter', 'centroid',
                                            'horizontal_asymmetry', 'vertical_asymmetry', 'apex_x_position']:
                                    node_data[feat] = 0.0
                        
                        # Ensure VL embedding is available or set to zeros
                        if node_data.get('vl_embed') is None:
                            node_data['vl_embed'] = np.zeros(512)
                    
                    # Initialize GNN reasoner
                    expected_dim = 5 + len(required_gnn_features) + len(required_gnn_features) + 4 + 10  # type_onehot + features + validity + action + vl
                    gnn_reasoner = GNNReasoner(in_dim=expected_dim)
                    
                    # Convert NetworkX graph to PyTorch Geometric format and apply GNN reasoning
                    try:
                        # Use the proper GNN predict method which handles the full conversion
                        enhanced_graph = gnn_reasoner.predict(G)
                        
                        # Extract GNN embeddings and scores from enhanced graph
                        if enhanced_graph.graph.get('gnn_status') == 'success':
                            logging.info(f"GNN processing successfully enhanced {len(enhanced_graph.nodes())} nodes with learned representations")
                        else:
                            logging.warning(f"GNN processing failed with status: {enhanced_graph.graph.get('gnn_status', 'unknown')}")
                        
                        # Copy enhanced graph back to G
                        G = enhanced_graph
                    
                    except Exception as e:
                        logging.warning(f"GNN tensor conversion failed: {e}")
                        # Set default GNN features for all nodes
                        for node_id in G.nodes():
                            G.nodes[node_id]['gnn_embedding'] = [0.0] * 64
                            G.nodes[node_id]['gnn_score'] = 0.0
                            G.nodes[node_id]['has_gnn_features'] = False
                
                else:
                    logging.warning("Empty graph - skipping GNN processing")
                
            except Exception as e:
                logging.warning(f"Failed to apply GNN reasoning: {e}")
                # Ensure all nodes have default GNN features to prevent downstream errors
                for node_id in G.nodes():
                    G.nodes[node_id]['gnn_embedding'] = [0.0] * 64
                    G.nodes[node_id]['gnn_score'] = 0.0
                    G.nodes[node_id]['has_gnn_features'] = False
        
        # 4. Add ConceptNet commonsense edges 
        if kb is not None and G.number_of_nodes() > 0:
            try:
                from src.scene_graphs_building.graph_building import add_commonsense_edges
                from src.scene_graphs_building.config import COMMONSENSE_LABEL_MAP
                
                # Ensure all nodes have proper shape_label and kb_concept for ConceptNet
                logging.info(f"Preparing {G.number_of_nodes()} nodes for ConceptNet integration")
                for node_id, node_data in G.nodes(data=True):
                    # ENHANCED FIX: Support both basic Bongard-LOGO types AND enhanced shape types
                    object_type = node_data.get('object_type')
                    shape_type = node_data.get('shape_type')
                    shape_label = node_data.get('shape_label')
                    
                    logging.info(f"[process_single_problem kb_concept] Node {node_id}: object_type='{object_type}', shape_type='{shape_type}', shape_label='{shape_label}'")
                    
                    # Enhanced mapping to support both basic and composite shape types
                    kb_concept = None
                    normalized_label = None
                    
                    # CRITICAL FIX: Stroke-specific KB concept mapping
                    # Combine object_type and shape_type for proper ConceptNet differentiation
                    if object_type and shape_type:
                        # Line objects with different shape types
                        if object_type == 'line':
                            if shape_type == 'normal':
                                kb_concept = 'line'
                                normalized_label = 'line'
                            elif shape_type == 'circle':
                                kb_concept = 'circle'
                                normalized_label = 'circle'
                            elif shape_type in ['square', 'triangle', 'zigzag']:
                                kb_concept = shape_type
                                normalized_label = shape_type
                            else:
                                kb_concept = 'line'  # Default for unknown line types
                                normalized_label = 'line'
                            logging.info(f"[process_single_problem kb_concept] Node {node_id}: Line object with shape_type '{shape_type}' -> kb_concept '{kb_concept}'")
                        
                        # Arc objects with different shape types  
                        elif object_type == 'arc':
                            if shape_type == 'normal':
                                kb_concept = 'arc'
                                normalized_label = 'arc'
                            elif shape_type == 'circle':
                                kb_concept = 'circle'
                                normalized_label = 'circle'
                            elif shape_type in ['square', 'triangle', 'zigzag']:
                                kb_concept = shape_type
                                normalized_label = shape_type
                            else:
                                kb_concept = 'arc'  # Default for unknown arc types
                                normalized_label = 'arc'
                            logging.info(f"[process_single_problem kb_concept] Node {node_id}: Arc object with shape_type '{shape_type}' -> kb_concept '{kb_concept}'")
                        
                        # Pure shape types without object_type context
                        elif shape_type in ['circle', 'square', 'triangle', 'zigzag']:
                            kb_concept = shape_type
                            normalized_label = shape_type
                            logging.info(f"[process_single_problem kb_concept] Node {node_id}: Pure shape_type '{shape_type}' -> kb_concept '{kb_concept}'")
                        
                        # Object type without specific shape_type
                        elif object_type in ['circle', 'square', 'triangle', 'zigzag']:
                            kb_concept = object_type
                            normalized_label = object_type
                            logging.info(f"[process_single_problem kb_concept] Node {node_id}: Object_type '{object_type}' -> kb_concept '{kb_concept}'")
                        
                        else:
                            # Fall back to object_type if available
                            kb_concept = object_type
                            normalized_label = object_type
                            logging.info(f"[process_single_problem kb_concept] Node {node_id}: Fallback object_type '{object_type}' -> kb_concept '{kb_concept}'")
                    # Legacy fallback mappings for other object types
                    elif object_type and object_type in ['segment', 'ray']:
                        kb_concept = 'line'  # Map line segments to line concept
                        normalized_label = 'line'
                        logging.info(f"[process_single_problem kb_concept] Node {node_id}: Mapping legacy object_type '{object_type}' to 'line'")
                        
                    elif object_type and object_type in ['polygon', 'quadrilateral']:
                        kb_concept = 'polygon'  # Map polygons to polygon concept
                        normalized_label = 'polygon'
                        logging.info(f"[process_single_problem kb_concept] Node {node_id}: Mapping object_type '{object_type}' to 'polygon'")
                        
                    elif object_type and object_type in ['motif', 'cluster', 'group']:
                        kb_concept = 'pattern'  # Map motifs to pattern concept
                        normalized_label = 'pattern'
                        logging.info(f"[process_single_problem kb_concept] Node {node_id}: Mapping object_type '{object_type}' to 'pattern'")
                        
                    else:
                        # Final fallback: mark as unmapped but continue processing
                        kb_concept = None
                        normalized_label = None
                        logging.info(f"[process_single_problem kb_concept] Node {node_id}: No mapping available (object_type='{object_type}', shape_type='{shape_type}')")
                    
                    # Update node data only if we have a valid kb_concept
                    if kb_concept is not None:
                        node_data['shape_label'] = normalized_label
                        node_data['kb_concept'] = kb_concept
                    else:
                        # Mark node as unmapped for ConceptNet but preserve other data
                        node_data['shape_label'] = 'unmapped'
                        node_data['kb_concept'] = None
                
                logging.info(f"Adding ConceptNet commonsense edges to graph with {G.number_of_nodes()} nodes")
                add_commonsense_edges(G, top_k=2, kb=kb)  # Reduced from 3 to 2 for fewer edges
                
                # Count ConceptNet edges added

                kb_edges = [data for u, v, data in G.edges(data=True) if data.get('source') == 'kb']
                logging.info(f"ConceptNet integration completed for {parent_shape_id}: {len(kb_edges)} KB edges added")
            except Exception as e:
                logging.warning(f"Failed to add ConceptNet edges: {e}")
                logging.debug(traceback.format_exc())
        else:
            if kb is None:
                logging.info("ConceptNet KB not available, skipping commonsense edges")
            else:
                logging.info("Empty graph, skipping ConceptNet integration")
        
        # 5. Add abstract conceptual nodes for high-level patterns
        G = _add_abstract_conceptual_nodes(G, parent_shape_id)
        
        # 6. Apply predicate importance filtering to reduce noise
        G = _filter_low_importance_edges(G, importance_threshold=1.2)
        
        # 7. Calculate and store predicate importance scores for analysis
        importance_scores = _calculate_predicate_importance(G, problem_id)
        
        # === SEMANTIC SCENE-LEVEL ANALYSIS ===
        # Add comprehensive semantic features to the graph
        semantic_parser = SemanticActionParser()
        
        # Collect all action programs from this image
        all_action_programs = []
        for rec in problem_records:
            if rec.get('action_program') and parent_shape_id in str(rec.get('image_path', '')):
                all_action_programs.extend(rec['action_program'])
        
        # Extract scene-level semantic features
        if all_action_programs:
            scene_semantic_info = semantic_parser.extract_semantic_intent(all_action_programs)
            
            # Add scene-level semantic metadata
            G.graph.update({
                'semantic_shapes': scene_semantic_info['shapes'],
                'semantic_properties': scene_semantic_info['properties'],
                'semantic_features': scene_semantic_info['semantic_features'],
                'scene_complexity': scene_semantic_info['complexity'],
                'bongard_relevant_features': {
                    'total_triangles': sum(1 for s in scene_semantic_info['shapes'] if s['type'] == 'triangle'),
                    'total_squares': sum(1 for s in scene_semantic_info['shapes'] if s['type'] == 'square'),
                    'total_circles': sum(1 for s in scene_semantic_info['shapes'] if s['type'] == 'circle'),
                    'total_lines': sum(1 for s in scene_semantic_info['shapes'] if s['type'] == 'line'),
                    'shape_diversity': len(set(s['type'] for s in scene_semantic_info['shapes'])),
                    'has_semantic_shapes': len(scene_semantic_info['shapes']) > 0,
                    'semantic_complexity_score': scene_semantic_info['complexity']
                }
            })
            
            logging.info(f"Added scene-level semantic features: {len(scene_semantic_info['shapes'])} shapes, "
                        f"complexity: {scene_semantic_info['complexity']}")
        
        # Calculate edge reduction metrics
        original_edge_count = G.number_of_edges()
        semantic_edge_count = len([1 for u, v, d in G.edges(data=True) if d.get('source') == 'semantic_bongard'])
        
        # Add metadata to graph for analysis
        G.graph['predicate_importance'] = importance_scores
        G.graph['problem_id'] = problem_id
        G.graph['processing_mode'] = 'enhanced_semantic'
        G.graph['edge_statistics'] = {
            'total_edges': original_edge_count,
            'semantic_edges': semantic_edge_count,
            'geometric_edges': len([1 for u, v, d in G.edges(data=True) if d.get('source') == 'advanced_geometry']),
            'edge_density': original_edge_count / max(1, G.number_of_nodes()),
            'semantic_edge_ratio': semantic_edge_count / max(1, original_edge_count)
        }

        final_graphs[parent_shape_id] = G

    # --- Save Outputs ---
    # For simplicity, we save a combined graph for visualization, but return the structured dict
    logging.info(f"[LOGO Visualization] Starting visualization process for {problem_id}. Final graphs count: {len(final_graphs)}")
    if final_graphs:
        # Combine all graphs from the problem into one for a single visualization file
        combined_graph = nx.compose_all(list(final_graphs.values()))
        logging.info(f"[LOGO Visualization] Combined graph created with {combined_graph.number_of_nodes()} nodes and {combined_graph.number_of_edges()} edges")
        try:
            from scripts.scene_graph_visualization import save_scene_graph_visualization, save_scene_graph_csv
            feedback_vis_dir = os.path.join('feedback', 'visualizations_logo')
            os.makedirs(feedback_vis_dir, exist_ok=True)
            logging.info(f"[LOGO Visualization] Created visualization directory: {feedback_vis_dir}")
            
            # Clean up and ensure complete node data before export
            _ensure_complete_node_data(combined_graph)
            
            # Save CSVs and visualizations
            logging.info(f"[LOGO Visualization] Calling save_scene_graph_csv for {problem_id}")
            save_scene_graph_csv(combined_graph, feedback_vis_dir, problem_id)
            logging.info(f"[LOGO Visualization] CSV export completed for {problem_id}")
            
            # ACTION PROGRAMS: Find or construct a valid image path for visualization
            image_path_vis = next((rec.get('image_path') for rec in problem_records if rec.get('image_path')), None)
            logging.info(f"[LOGO Visualization] Found image path from records: {image_path_vis}")
            
            # If no image path in records (action-program mode), construct one from the first available image
            if not image_path_vis:
                # Try to construct a representative image path for visualization
                # Use the first positive example image (category_1/0.png)
                category = problem_id.split('_')[0] if '_' in problem_id else 'bd'
                constructed_path = f"data/raw/ShapeBongard_V2/{category}/images/{problem_id}\\category_1\\0.png"
                logging.info(f"[LOGO Visualization] Constructed path: {constructed_path}")
                
                try:
                    from src.scene_graphs_building.data_loading import remap_path
                    remapped_constructed_path = remap_path(constructed_path)
                    logging.info(f"[LOGO Visualization] Remapped constructed path: {remapped_constructed_path}")
                    if os.path.exists(remapped_constructed_path):
                        image_path_vis = constructed_path
                        logging.info(f"Using constructed image path for visualization: {image_path_vis}")
                    else:
                        logging.warning(f"Constructed image path does not exist: {remapped_constructed_path}")
                except Exception as e:
                    logging.warning(f"Failed to construct image path for visualization: {e}")
            
            # Generate visualization if we have a valid image path
            logging.info(f"[LOGO Visualization] Image path for visualization: {image_path_vis}")
            if image_path_vis:
                try:
                    # Pass abstract_view=True to generate the high-level graph visualization
                    logging.info(f"[LOGO Visualization] Calling save_scene_graph_visualization with image path")
                    save_scene_graph_visualization(combined_graph, remap_path(image_path_vis), feedback_vis_dir, problem_id, abstract_view=True)
                    logging.info(f"Successfully saved visualization for {problem_id}")
                except Exception as viz_error:
                    logging.warning(f"Visualization generation failed for {problem_id}: {viz_error}")
                    import traceback
                    logging.warning(f"Visualization error traceback: {traceback.format_exc()}")
            else:
                # Generate graph-only visualization without image overlay
                try:
                    logging.info(f"[LOGO Visualization] Calling save_scene_graph_visualization without image path")
                    save_scene_graph_visualization(combined_graph, None, feedback_vis_dir, problem_id, abstract_view=True)
                    logging.info(f"Successfully saved graph-only visualization for {problem_id}")
                except Exception as viz_error:
                    logging.warning(f"Graph-only visualization generation failed for {problem_id}: {viz_error}")
                    import traceback
                    logging.warning(f"Graph-only visualization error traceback: {traceback.format_exc()}")
        except Exception as e:
            logging.warning(f"[LOGO Visualization] Failed to save outputs for {problem_id}: {e}\n{traceback.format_exc()}")
    else:
        logging.warning(f"[LOGO Visualization] No final graphs found for {problem_id} - skipping visualization")

    return {'scene_graphs': final_graphs, 'objects': all_objects_for_return, 'mode': 'logo', 'rules': None}

def _calculate_stroke_geometry_old(verts):
    pass