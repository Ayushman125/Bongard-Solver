import logging
import os
import numpy as np
import traceback
from typing import List, Dict, Any
from src.scene_graphs_building.data_loading import remap_path
import networkx as nx
from shapely.geometry import Polygon, LineString
from collections import defaultdict

# Import the new advanced predicates
from src.scene_graphs_building.advanced_predicates import ADVANCED_PREDICATE_REGISTRY

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

def assign_object_type(verts):
    arr = np.asarray(verts)
    if len(arr) == 1 or np.allclose(np.std(arr, axis=0), 0, atol=1e-8):
        return "point"
    elif len(arr) == 2 or are_points_collinear(arr):
        return "line"
    elif len(arr) >= 3 and not are_points_collinear(arr):
        return "polygon"
    else:
        # SOTA: treat degenerate/ambiguous as 'curve' if possible
        return "curve" if len(arr) > 1 else "unknown"

def parse_action_command(cmd):
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
            a, b = rest.split('-',1)
            return {'type':'line', 'mode':mode, 'x':float(a), 'y':float(b)}
        except Exception:
            return None
    elif shape == "start" and '-' in rest:
        try:
            a, b = rest.split('-',1)
            return {'type':'start', 'x':float(a), 'y':float(b)}
        except Exception:
            return None
    elif shape == "arc" and '-' in rest:
        try:
            radius, angle = rest.split('-',1)
            return {'type':'arc', 'mode':mode, 'radius':float(radius), 'angle':float(angle)}
        except Exception:
            return None
    elif shape == "turn" and '-' in rest:
        try:
            angle = rest
            return {'type':'turn', 'mode':mode, 'angle':float(angle)}
        except Exception:
            return None
    # Extend for other commands as needed
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
        
        # Geometric complexity measures
        features.update({
            'vertex_count': len(verts),
            'geometric_complexity': len(verts) * (1 + features.get('horizontal_asymmetry', 0) + features.get('vertical_asymmetry', 0))
        })
        
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
        object_type = 'polygon' if is_closed else 'open_curve'
        
        # Calculate enhanced geometric properties for the new composite shape
        base_obj = sorted_primitives[0]
        enhanced_features = _calculate_enhanced_geometry_features(all_verts, {
            'object_id': shape_id, 
            'parent_shape_id': parent_shape_id,
            'object_type': object_type, 
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




async def _process_single_problem(problem_id: str, problem_records: List[Dict[str, Any]], feature_cache, *, args=None):
    """
    Processes a single Bongard problem to generate a scene graph for EACH image.
    Returns a dictionary mapping image_id to its scene graph.
    """
    all_objects_by_image = defaultdict(list)

    for idx, rec in enumerate(problem_records):
        parent_shape_id = f"{problem_id}_{idx}"
        action_program = rec.get('action_program', [])
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

            obj_id = f"{problem_id}_{idx}_{stroke_idx}"
            
            # Calculate enhanced geometric features
            enhanced_features = _calculate_enhanced_geometry_features(verts, {
                'object_id': obj_id, 
                'parent_shape_id': parent_shape_id, 
                'action_index': stroke_idx,
                'vertices': verts, 
                'object_type': assign_object_type(verts), 
                'action_command': cmd,
                'endpoints': [verts[0], verts[-1]], 
                'length': length, 
                'orientation': orientation,
                'source': 'action_program', 
                'is_closed': len(verts) > 2 and np.allclose(verts[0], verts[-1], atol=1e-5),
                **common_attrs
            })
            
            relationships = []
            if last_stroke_obj:
                # Check for adjacency based on endpoint proximity
                if np.allclose(last_stroke_obj['endpoints'][-1], verts[0], atol=1e-5):
                    relationships.append(f"adjacent_to_{last_stroke_obj['object_id']}")
            
            enhanced_features['relationships'] = relationships
            obj = enhanced_features
            all_objects_by_image[parent_shape_id].append(obj)
            last_stroke_obj = obj

    # --- Graph Construction (per image) ---
    final_graphs = {}
    all_objects_for_return = []

    for parent_shape_id, objects_in_image in all_objects_by_image.items():
        # --- Hierarchical Grouping Step ---
        # Correctly group strokes into higher-level shapes
        newly_created_shapes = _group_strokes_into_shapes(objects_in_image, parent_shape_id)
        
        # Combine primitives and new composite shapes
        all_nodes_for_graph = objects_in_image + newly_created_shapes
        
        G = nx.MultiDiGraph()
        for obj in all_nodes_for_graph:
            G.add_node(obj['object_id'], **obj)
        
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
            except Exception as e:
                logging.warning(f"Failed to apply advanced predicates between {id_a} and {id_b}: {e}")

        # === ENHANCED PROCESSING: Add semantic abstraction and feature selection ===
        
        # 1. Add abstract conceptual nodes for high-level patterns
        G = _add_abstract_conceptual_nodes(G, parent_shape_id)
        
        # 2. Apply predicate importance filtering to reduce noise
        G = _filter_low_importance_edges(G, importance_threshold=1.2)
        
        # 3. Calculate and store predicate importance scores for analysis
        importance_scores = _calculate_predicate_importance(G, problem_id)
        
        # Add metadata to graph for analysis
        G.graph['predicate_importance'] = importance_scores
        G.graph['problem_id'] = problem_id
        G.graph['processing_mode'] = 'enhanced_semantic'

        final_graphs[parent_shape_id] = G

    # --- Save Outputs ---
    # For simplicity, we save a combined graph for visualization, but return the structured dict
    if final_graphs:
        # Combine all graphs from the problem into one for a single visualization file
        combined_graph = nx.compose_all(list(final_graphs.values()))
        try:
            from scripts.scene_graph_visualization import save_scene_graph_visualization, save_scene_graph_csv
            feedback_vis_dir = os.path.join('feedback', 'visualizations_logo')
            os.makedirs(feedback_vis_dir, exist_ok=True)
            
            # Save CSVs and visualizations
            save_scene_graph_csv(combined_graph, feedback_vis_dir, problem_id)
            image_path_vis = next((rec.get('image_path') for rec in problem_records if rec.get('image_path')), None)
            if image_path_vis:
                # Pass abstract_view=True to generate the high-level graph visualization
                save_scene_graph_visualization(combined_graph, remap_path(image_path_vis), feedback_vis_dir, problem_id, abstract_view=True)
        except Exception as e:
            logging.warning(f"[LOGO Visualization] Failed to save outputs for {problem_id}: {e}\n{traceback.format_exc()}")

    return {'scene_graphs': final_graphs, 'objects': all_objects_for_return, 'mode': 'logo', 'rules': None}

def _calculate_stroke_geometry_old(verts):
    pass