"""
LOGO MODE: Physics attributes computation integrated with LOGO action program data.
This module replaces the physics computation logic to use LOGO-derived geometry.
"""

import numpy as np
import logging
from shapely.geometry import Polygon, LineString, Point


def compute_logo_physics_attributes(node_data):
    """
    LOGO MODE: Computes physics attributes directly from LOGO action program data.
    Completely rewritten to use LOGO-derived geometry and action sequences.
    """
    # LOGO MODE: Extract LOGO-specific data
    vertices = node_data.get('vertices', [])
    object_type = node_data.get('object_type', 'unknown')
    is_closed = node_data.get('is_closed', False)
    action_program = node_data.get('action_program', [])
    action_command = node_data.get('action_command', '')
    
    # Initialize all required features
    required_features = [
        'centroid', 'cx', 'cy', 'area', 'perimeter', 'bbox', 'aspect_ratio', 'orientation',
        'compactness', 'inertia', 'convexity', 'num_segments', 'num_junctions', 'curvature_score',
        'skeleton_length', 'symmetry_axis', 'horizontal_asymmetry', 'vertical_asymmetry',
        'left_extent', 'right_extent', 'top_extent', 'bottom_extent', 'apex_x_position',
        'apex_relative_to_center', 'has_prominent_apex', 'principal_orientation',
        'orientation_variance', 'max_curvature', 'is_highly_curved', 'vertex_count',
        'geometric_complexity'
    ]
    
    required_valid_flags = [f"{feat}_valid" for feat in required_features]
    feature_valid = {}
    
    # LOGO MODE: Process vertices if available
    if vertices and len(vertices) >= 2:
        try:
            # Convert to numpy array for processing
            arr = np.array(vertices)
            
            # Basic bounding box and extents
            x_coords = arr[:, 0]
            y_coords = arr[:, 1]
            
            bbox = [float(np.min(x_coords)), float(np.min(y_coords)), 
                   float(np.max(x_coords)), float(np.max(y_coords))]
            node_data['bbox'] = bbox
            feature_valid['bbox_valid'] = True
            
            # Extents for asymmetry analysis
            node_data['left_extent'] = float(np.min(x_coords))
            node_data['right_extent'] = float(np.max(x_coords))
            node_data['top_extent'] = float(np.max(y_coords))
            node_data['bottom_extent'] = float(np.min(y_coords))
            feature_valid['left_extent_valid'] = True
            feature_valid['right_extent_valid'] = True
            feature_valid['top_extent_valid'] = True
            feature_valid['bottom_extent_valid'] = True
            
            # Centroid computation
            centroid = np.mean(arr, axis=0)
            node_data['centroid'] = centroid.tolist()
            node_data['cx'] = float(centroid[0])
            node_data['cy'] = float(centroid[1])
            feature_valid['centroid_valid'] = True
            feature_valid['cx_valid'] = True
            feature_valid['cy_valid'] = True
            
            # Vertex count and geometric complexity
            node_data['vertex_count'] = len(vertices)
            node_data['geometric_complexity'] = len(vertices) + (2 if is_closed else 0)
            feature_valid['vertex_count_valid'] = True
            feature_valid['geometric_complexity_valid'] = True
            
            # LOGO MODE: Object-type specific geometry
            if object_type == 'polygon' and is_closed and len(vertices) >= 3:
                # Create polygon geometry
                try:
                    poly = Polygon(vertices)
                    if poly.is_valid and poly.area > 1e-6:
                        # Area and perimeter
                        node_data['area'] = float(poly.area)
                        node_data['perimeter'] = float(poly.length)
                        feature_valid['area_valid'] = True
                        feature_valid['perimeter_valid'] = True
                        
                        # Compactness (isoperimetric ratio)
                        if poly.length > 0:
                            compactness = 4 * np.pi * poly.area / (poly.length ** 2)
                            node_data['compactness'] = float(compactness)
                            feature_valid['compactness_valid'] = True
                        
                        # Convexity
                        try:
                            convex_hull = poly.convex_hull
                            convexity = poly.area / convex_hull.area if convex_hull.area > 0 else 0.0
                            node_data['convexity'] = float(convexity)
                            feature_valid['convexity_valid'] = True
                        except:
                            node_data['convexity'] = 0.0
                            feature_valid['convexity_valid'] = False
                        
                        # Moment of inertia (simplified)
                        try:
                            coords = np.array(poly.exterior.coords[:-1])
                            cx, cy = centroid
                            inertia = np.sum((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2) / len(coords)
                            node_data['inertia'] = float(inertia)
                            feature_valid['inertia_valid'] = True
                        except:
                            node_data['inertia'] = 0.0
                            feature_valid['inertia_valid'] = False
                            
                    else:
                        # Invalid polygon fallback
                        node_data['area'] = 0.0
                        node_data['perimeter'] = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
                        node_data['compactness'] = 0.0
                        node_data['convexity'] = 0.0
                        node_data['inertia'] = 0.0
                        feature_valid['area_valid'] = False
                        feature_valid['perimeter_valid'] = True
                        feature_valid['compactness_valid'] = False
                        feature_valid['convexity_valid'] = False
                        feature_valid['inertia_valid'] = False
                        
                except Exception as e:
                    # Fallback values for failed polygon processing
                    node_data['area'] = 0.0
                    node_data['perimeter'] = 0.0
                    node_data['compactness'] = 0.0
                    node_data['convexity'] = 0.0
                    node_data['inertia'] = 0.0
                    for feat in ['area', 'perimeter', 'compactness', 'convexity', 'inertia']:
                        feature_valid[f'{feat}_valid'] = False
            
            elif object_type in ['line', 'arc']:
                # Line/arc geometry
                node_data['area'] = 0.0
                node_data['perimeter'] = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
                node_data['skeleton_length'] = node_data['perimeter']
                feature_valid['area_valid'] = True  # 0 is valid for lines
                feature_valid['perimeter_valid'] = True
                feature_valid['skeleton_length_valid'] = True
                
                # Line orientation
                if len(vertices) >= 2:
                    start, end = arr[0], arr[-1]
                    orientation = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
                    node_data['orientation'] = float(orientation % 360)
                    node_data['principal_orientation'] = node_data['orientation']
                    feature_valid['orientation_valid'] = True
                    feature_valid['principal_orientation_valid'] = True
                
                # Curvature analysis for LOGO arcs
                if object_type == 'arc' or 'arc_' in action_command:
                    try:
                        # Simple curvature estimation
                        if len(vertices) >= 3:
                            curvatures = []
                            for i in range(1, len(vertices) - 1):
                                p1, p2, p3 = arr[i-1], arr[i], arr[i+1]
                                v1 = p2 - p1
                                v2 = p3 - p2
                                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                                if norm1 > 1e-6 and norm2 > 1e-6:
                                    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                                    angle = np.arccos(cos_angle)
                                    curvature = angle / (norm1 + norm2) * 2
                                    curvatures.append(curvature)
                            
                            if curvatures:
                                node_data['curvature_score'] = float(np.mean(curvatures))
                                node_data['max_curvature'] = float(np.max(curvatures))
                                node_data['is_highly_curved'] = bool(node_data['max_curvature'] > 0.5)
                                feature_valid['curvature_score_valid'] = True
                                feature_valid['max_curvature_valid'] = True
                                feature_valid['is_highly_curved_valid'] = True
                            else:
                                node_data['curvature_score'] = 0.0
                                node_data['max_curvature'] = 0.0
                                node_data['is_highly_curved'] = False
                                feature_valid['curvature_score_valid'] = False
                                feature_valid['max_curvature_valid'] = False
                                feature_valid['is_highly_curved_valid'] = False
                        else:
                            node_data['curvature_score'] = 0.0
                            node_data['max_curvature'] = 0.0
                            node_data['is_highly_curved'] = False
                            feature_valid['curvature_score_valid'] = False
                            feature_valid['max_curvature_valid'] = False
                            feature_valid['is_highly_curved_valid'] = False
                    except:
                        node_data['curvature_score'] = 0.0
                        node_data['max_curvature'] = 0.0
                        node_data['is_highly_curved'] = False
                        feature_valid['curvature_score_valid'] = False
                        feature_valid['max_curvature_valid'] = False
                        feature_valid['is_highly_curved_valid'] = False
                else:
                    # Straight line
                    node_data['curvature_score'] = 0.0
                    node_data['max_curvature'] = 0.0
                    node_data['is_highly_curved'] = False
                    feature_valid['curvature_score_valid'] = True
                    feature_valid['max_curvature_valid'] = True
                    feature_valid['is_highly_curved_valid'] = True
            
            # Common geometric properties
            # Aspect ratio
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if height > 1e-6:
                node_data['aspect_ratio'] = float(width / height)
                feature_valid['aspect_ratio_valid'] = True
            else:
                node_data['aspect_ratio'] = 1.0
                feature_valid['aspect_ratio_valid'] = False
            
            # LOGO MODE: Asymmetry analysis
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Horizontal asymmetry
            left_mass = np.sum(x_coords < center_x)
            right_mass = np.sum(x_coords > center_x)
            total_mass = len(x_coords)
            if total_mass > 0:
                h_asymmetry = abs(left_mass - right_mass) / total_mass
                node_data['horizontal_asymmetry'] = float(h_asymmetry)
                feature_valid['horizontal_asymmetry_valid'] = True
            else:
                node_data['horizontal_asymmetry'] = 0.0
                feature_valid['horizontal_asymmetry_valid'] = False
            
            # Vertical asymmetry
            top_mass = np.sum(y_coords > center_y)
            bottom_mass = np.sum(y_coords < center_y)
            if total_mass > 0:
                v_asymmetry = abs(top_mass - bottom_mass) / total_mass
                node_data['vertical_asymmetry'] = float(v_asymmetry)
                feature_valid['vertical_asymmetry_valid'] = True
            else:
                node_data['vertical_asymmetry'] = 0.0
                feature_valid['vertical_asymmetry_valid'] = False
            
            # Apex detection
            if len(vertices) >= 3:
                # Find topmost point as potential apex
                top_idx = np.argmax(y_coords)
                apex_x = x_coords[top_idx]
                node_data['apex_x_position'] = float(apex_x)
                node_data['apex_relative_to_center'] = 'left' if apex_x < center_x else 'center' if abs(apex_x - center_x) < width * 0.1 else 'right'
                node_data['has_prominent_apex'] = bool(y_coords[top_idx] > center_y + height * 0.3)
                feature_valid['apex_x_position_valid'] = True
                feature_valid['apex_relative_to_center_valid'] = True
                feature_valid['has_prominent_apex_valid'] = True
            else:
                node_data['apex_x_position'] = center_x
                node_data['apex_relative_to_center'] = 'center'
                node_data['has_prominent_apex'] = False
                feature_valid['apex_x_position_valid'] = False
                feature_valid['apex_relative_to_center_valid'] = False
                feature_valid['has_prominent_apex_valid'] = False
            
            # LOGO ACTION: Extract programmatic attributes from action command
            if action_command:
                # Parse LOGO command for symmetry and predicate info
                if 'line_' in action_command:
                    node_data['stroke_type'] = 'line'
                    # Extract line parameters if available
                    parts = action_command.split('_')
                    if len(parts) >= 3:
                        try:
                            mode = parts[1]  # e.g., 'normal', 'long', 'short'
                            coords = parts[2].split('-')
                            if len(coords) == 2:
                                length_param = float(coords[0])
                                angle_param = float(coords[1])
                                node_data['symmetry_axis'] = float(angle_param * 360)  # Convert to degrees
                                feature_valid['symmetry_axis_valid'] = True
                            else:
                                node_data['symmetry_axis'] = node_data.get('orientation', 0.0)
                                feature_valid['symmetry_axis_valid'] = False
                        except:
                            node_data['symmetry_axis'] = node_data.get('orientation', 0.0)
                            feature_valid['symmetry_axis_valid'] = False
                    else:
                        node_data['symmetry_axis'] = node_data.get('orientation', 0.0)
                        feature_valid['symmetry_axis_valid'] = False
                        
                elif 'arc_' in action_command:
                    node_data['stroke_type'] = 'arc'
                    # Extract arc parameters if available
                    parts = action_command.split('_')
                    if len(parts) >= 4:
                        try:
                            mode = parts[1]  # e.g., 'normal', 'wide', 'tight'
                            radius = float(parts[2])
                            arc_span = float(parts[3])
                            # Use arc span as a measure of symmetry
                            node_data['symmetry_axis'] = float(arc_span * 180)  # Convert to degrees
                            feature_valid['symmetry_axis_valid'] = True
                        except:
                            node_data['symmetry_axis'] = 0.0
                            feature_valid['symmetry_axis_valid'] = False
                    else:
                        node_data['symmetry_axis'] = 0.0
                        feature_valid['symmetry_axis_valid'] = False
                else:
                    node_data['stroke_type'] = 'composite'
                    node_data['symmetry_axis'] = 0.0
                    feature_valid['symmetry_axis_valid'] = False
            else:
                node_data['stroke_type'] = 'unknown'
                node_data['symmetry_axis'] = 0.0
                feature_valid['symmetry_axis_valid'] = False
            
            # Orientation variance (for complex shapes)
            if len(vertices) > 2:
                orientations = []
                for i in range(len(vertices) - 1):
                    dx = vertices[i+1][0] - vertices[i][0]
                    dy = vertices[i+1][1] - vertices[i][1]
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        orientation = np.degrees(np.arctan2(dy, dx))
                        orientations.append(orientation)
                
                if orientations:
                    node_data['orientation_variance'] = float(np.var(orientations))
                    feature_valid['orientation_variance_valid'] = True
                else:
                    node_data['orientation_variance'] = 0.0
                    feature_valid['orientation_variance_valid'] = False
            else:
                node_data['orientation_variance'] = 0.0
                feature_valid['orientation_variance_valid'] = False
            
            # Segment and junction counting
            node_data['num_segments'] = max(1, len(vertices) - 1)
            node_data['num_junctions'] = len(vertices) if is_closed else max(0, len(vertices) - 2)
            feature_valid['num_segments_valid'] = True
            feature_valid['num_junctions_valid'] = True
            
            # Set remaining unset features to defaults
            for feat in required_features:
                if feat not in node_data:
                    node_data[feat] = 0.0 if feat not in ['apex_relative_to_center', 'stroke_type'] else 'unknown'
                if f'{feat}_valid' not in feature_valid:
                    feature_valid[f'{feat}_valid'] = False
                    
            # Mark as having valid geometry
            node_data['geometry_valid'] = True
            node_data['fallback_geometry'] = False
            
            logging.info(f"compute_logo_physics_attributes: Node {node_data.get('object_id', 'unknown')} processed with LOGO geometry - type: {object_type}, vertices: {len(vertices)}")
            
        except Exception as e:
            logging.warning(f"compute_logo_physics_attributes: Error processing LOGO geometry for node {node_data.get('object_id', 'unknown')}: {e}")
            # Set all features to default values
            for feat in required_features:
                if feat == 'apex_relative_to_center':
                    node_data[feat] = 'center'
                elif feat == 'stroke_type':
                    node_data[feat] = 'unknown'
                else:
                    node_data[feat] = 0.0
            for flag in required_valid_flags:
                feature_valid[flag] = False
            node_data['geometry_valid'] = False
            node_data['fallback_geometry'] = True
    
    else:
        # No valid vertices - set all to defaults
        logging.warning(f"Node {node_data.get('object_id', 'unknown')} missing or insufficient vertices for LOGO geometry processing.")
        for feat in required_features:
            if feat == 'apex_relative_to_center':
                node_data[feat] = 'center'
            elif feat == 'stroke_type':
                node_data[feat] = 'unknown'
            else:
                node_data[feat] = 0.0
        for flag in required_valid_flags:
            feature_valid[flag] = False
        node_data['geometry_valid'] = False
        node_data['fallback_geometry'] = True
    
    # Store feature validity
    node_data['feature_valid'] = feature_valid
    
    # Compute centroid from vertices if needed (for degenerate cases)
    if not node_data.get('centroid') or len(node_data.get('centroid', [])) != 2:
        # Use proper geometric centroid calculation from vertices
        vertices = node_data.get('vertices', [])
        if vertices and len(vertices) >= 2:
            try:
                verts_array = np.array(vertices)
                computed_centroid = np.mean(verts_array, axis=0).tolist()
                node_data['centroid'] = computed_centroid
                node_data['cx'] = computed_centroid[0]
                node_data['cy'] = computed_centroid[1]
                logging.info(f"compute_logo_physics_attributes: Node {node_data.get('object_id', 'unknown')} computed centroid {computed_centroid} from vertices.")
            except Exception as e:
                logging.warning(f"compute_logo_physics_attributes: Node {node_data.get('object_id', 'unknown')} failed to compute centroid from vertices: {e}. Using default.")
                node_data['centroid'] = [0.0, 0.0]
                node_data['cx'] = 0.0
                node_data['cy'] = 0.0
        else:
            logging.warning(f"compute_logo_physics_attributes: Node {node_data.get('object_id', 'unknown')} has no valid vertices for centroid computation. Using default.")
            node_data['centroid'] = [0.0, 0.0]
            node_data['cx'] = 0.0
            node_data['cy'] = 0.0
    
    return node_data
