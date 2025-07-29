import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key   = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        # features: Tensor of shape (N, feature_dim)
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)
        attn = self.softmax(Q @ K.T / (K.shape[-1]**0.5))
        context = attn @ V
        return context

def extract_batch_features(batch_flat_commands, shape_creation_dependencies, D=32):
    # Dummy batch feature extraction for demo
    import numpy as np
    batch_features = []
    for flat_commands in batch_flat_commands:
        # Use LOGO parser to get shape objects, then extract features
        objects, _ = create_shape_from_stroke_pipeline(flat_commands, **shape_creation_dependencies)
        if objects and 'coords' in objects[0]:
            feat = image_processing_features(objects[0]['coords'])
        else:
            feat = np.zeros(D)
        batch_features.append(feat)
    return np.stack(batch_features)
import torch
import torch.nn as nn
import logging
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import polygonize, linemerge
import copy

# Import PhysicsInference for physics-based features
from src.data_pipeline.physics_infer import PhysicsInference



# Import internal modularized functions
from derive_label.confidence_scoring import calculate_confidence
from derive_label.geometric_detectors import (
    ransac_line, ransac_circle, ellipse_fit, hough_lines_detector,
    convexity_defects_detector, fourier_descriptors, cluster_points_detector,
    is_roughly_circular_detector, is_rectangle_detector, is_point_cloud_detector,
    compute_symmetry_axis, compute_orientation, compute_symmetry_score as compute_geom_symmetry_score,
    detect_vertices, min_bounding_rectangle, min_bounding_circle # NEW: Added min_bounding functions
)
from derive_label.image_features import (
    image_processing_features, compute_euler_characteristic, persistent_homology_features
    # Removed count_holes_shapely, contour_hierarchy_holes, canny_edge_density - now inside image_processing_features
)
from derive_label.semantic_refinement import (
    detect_connected_line_segments, is_quadrilateral_like, is_smooth_curve,
    is_composite_shape, generate_semantic_label, infer_shape_type_ensemble,
    analyze_detector_consensus
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ContextEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, hidden_dim)
        self.key   = nn.Linear(feature_dim, hidden_dim)
        self.value = nn.Linear(feature_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features):
        # features: Tensor of shape (N, feature_dim)
        Q = self.query(features)
        K = self.key(features)
        V = self.value(features)
        attn = self.softmax(Q @ K.T / (K.shape[-1]**0.5))
        context = attn @ V
        return context

# Example usage in pipeline:
# features = extract_image_features(batch_images)   # (N, D)
# context_feats = ContextEncoder(D, H)(torch.Tensor(features))

def flatten_action_program(action_program):
    """Recursively flattens nested lists in an action program."""
    flat_list = []
    for item in action_program:
        if isinstance(item, list):
            flat_list.extend(flatten_action_program(item))
        else:
            flat_list.append(item)
    return flat_list

def create_shape_from_stroke_pipeline(flat_commands, problem_id=None, **kwargs):
    """
    Main pipeline to create shape objects from LOGO commands, apply various detectors,
    and enrich with properties, including physics-based inferences and cutting-edge features.
    """
    bongard_logo_parser = kwargs.get('BongardLogoParser')
    
    if not bongard_logo_parser:
        logging.error("BongardLogoParser not provided to create_shape_from_stroke_pipeline.")
        return [], "Missing BongardLogoParser"

    try:
        # Use the correct method for BongardLogoParser
        strokes_data = bongard_logo_parser.parse_action_program(flat_commands, scale=100.0)
    except Exception as e:
        logging.error(f"Error parsing action program to strokes: {e}")
        return [], f"Parsing error: {e}"

    objects = []
    object_id_counter = 0

    for stroke_idx, stroke_obj in enumerate(strokes_data):
        coords = stroke_obj.get('coords')
        if coords is None or (isinstance(coords, (list, np.ndarray)) and len(coords) == 0):
            logging.warning(f"Skipping stroke {stroke_idx}: No coordinates found.")
            continue

        object_id = f"obj_{object_id_counter}"
        vertices_np = np.array(coords)
        if vertices_np is None or vertices_np.size == 0 or vertices_np.shape[0] < 1:
            logging.warning(f"Skipping object {object_id}: Empty vertices array.")
            continue

        # Initialize object dictionary
        obj = {
            'id': object_id,
            'original_stroke_idx': stroke_idx,
            'vertices': vertices_np.tolist(), # Store as list for JSON serialization
            'draw_order': stroke_idx, # Assuming draw_order is stroke index
            'properties': {},
            'possible_labels': [], # List to store candidate labels from various detectors
            'flags': [] # New: for storing specific issues with this object
        }

        # --- Basic Geometric Properties (Calculated early for wider use) ---
        try:
            poly_geom = None
            is_closed_geom = False
            if vertices_np.shape[0] >= 3:
                # Attempt to form a Shapely Polygon to check closure and area
                # Use PhysicsInference's polygon_from_vertices for robust handling
                try:
                    poly_geom = PhysicsInference.polygon_from_vertices(vertices_np.tolist())
                    is_closed_geom = poly_geom.is_valid and not poly_geom.is_empty and poly_geom.area > 0
                except Exception:
                    # Fallback if Shapely Polygon construction fails, assume not closed or invalid
                    is_closed_geom = False
            else:
                is_closed_geom = bool(np.linalg.norm(vertices_np[0] - vertices_np[-1]) < 1e-1) if vertices_np.shape[0] > 1 else False # Simple check for two points

            obj['properties']['is_closed_geometric'] = is_closed_geom

            if poly_geom and poly_geom.is_valid:
                obj['properties']['area_geometric'] = float(poly_geom.area)
                obj['properties']['perimeter_geometric'] = float(poly_geom.length) # Shapely perimeter
                obj['properties']['centroid_geometric'] = PhysicsInference.centroid(poly_geom)
            else:
                obj['properties']['area_geometric'] = 0.0
                obj['properties']['perimeter_geometric'] = 0.0
                obj['properties']['centroid_geometric'] = np.mean(vertices_np, axis=0).tolist() if vertices_np.shape[0] > 0 else [0.0, 0.0]

            obj['properties']['num_points_in_stroke'] = len(vertices_np)
            
        except Exception as e:
            logging.error(f"Error computing initial geometric properties for object {object_id}: {e}")
            obj['flags'].append('initial_geom_prop_error')

        # --- Apply Cutting-Edge Geometric Detectors ---
        
        # RANSAC Line
        is_line, line_conf, line_props = ransac_line(vertices_np)
        if is_line:
            obj['possible_labels'].append({'label': 'line_segment', 'confidence': line_conf, 'type': 'geometric_fit', 'props': line_props})
        if line_props: obj['properties'].update({'line_fit': line_props})

        # RANSAC Circle
        is_circle, circle_conf, circle_props = ransac_circle(vertices_np)
        if is_circle:
            obj['possible_labels'].append({'label': 'circle', 'confidence': circle_conf, 'type': 'geometric_fit', 'props': circle_props})
        if circle_props: obj['properties'].update({'circle_fit': circle_props})

        # Ellipse Fit
        is_ellipse, ellipse_conf, ellipse_props = ellipse_fit(vertices_np)
        if is_ellipse:
            obj['possible_labels'].append({'label': 'ellipse', 'confidence': ellipse_conf, 'type': 'geometric_fit', 'props': ellipse_props})
        if ellipse_props: obj['properties'].update({'ellipse_fit': ellipse_props})
        
        # Hough Lines
        is_hough_line, hough_line_conf, hough_line_props = hough_lines_detector(vertices_np)
        if is_hough_line:
            obj['possible_labels'].append({'label': 'hough_line', 'confidence': hough_line_conf, 'type': 'image_geometric_detector', 'props': hough_line_props})
        if hough_line_props: obj['properties'].update({'hough_line_detection': hough_line_props})


        # Minimum Bounding Geometries (NEW in geometric_detectors)
        is_mbr, mbr_conf, mbr_props = min_bounding_rectangle(vertices_np)
        if is_mbr:
            obj['possible_labels'].append({'label': 'min_bounding_rectangle_fit', 'confidence': mbr_conf, 'type': 'geometric_bounding', 'props': mbr_props})
        if mbr_props: obj['properties'].update({'min_bounding_rectangle': mbr_props})

        is_mbc, mbc_conf, mbc_props = min_bounding_circle(vertices_np)
        if is_mbc:
            obj['possible_labels'].append({'label': 'min_bounding_circle_fit', 'confidence': mbc_conf, 'type': 'geometric_bounding', 'props': mbc_props})
        if mbc_props: obj['properties'].update({'min_bounding_circle': mbc_props})


        # Geometric Properties & General Shape Types
        is_roughly_circular, r_circ_conf = is_roughly_circular_detector(vertices_np)
        if is_roughly_circular:
            obj['possible_labels'].append({'label': 'roughly_circular_geom', 'confidence': r_circ_conf, 'type': 'geometric_general'})
        
        is_rectangle_geom, rect_geom_conf, rect_geom_props = is_rectangle_detector(vertices_np) # Now returns props too
        if is_rectangle_geom:
            obj['possible_labels'].append({'label': 'rectangle_geom', 'confidence': rect_geom_conf, 'type': 'geometric_general', 'props': rect_geom_props})
        if rect_geom_props: obj['properties'].update({'rectangle_geom_props': rect_geom_props})

        is_point_cloud, pc_conf, pc_props = is_point_cloud_detector(vertices_np)
        if is_point_cloud:
            obj['possible_labels'].append({'label': 'point_cloud_geom', 'confidence': pc_conf, 'type': 'geometric_general', 'props': pc_props})
        if pc_props: obj['properties'].update({'point_cloud_props': pc_props})
        
        num_defects, defect_label, defects_conf, defects_extra_props = convexity_defects_detector(vertices_np) # Returns extra props
        obj['possible_labels'].append({'label': defect_label, 'confidence': defects_conf, 'type': 'morphological_geom', 'props': {'num_convexity_defects': num_defects}})
        if defects_extra_props: obj['properties'].update({'convexity_defects_analysis': defects_extra_props})


        num_vertices, vertices_conf, vertices_coords = detect_vertices(vertices_np) # Now returns coords
        obj['properties']['num_detected_vertices'] = num_vertices
        obj['properties']['detected_vertices_coords'] = vertices_coords
        if num_vertices is not None and num_vertices > 0:
             obj['possible_labels'].append({'label': f'polygon_vertices_{num_vertices}', 'confidence': vertices_conf, 'type': 'geometric_structure'})

        # Fourier Descriptors
        fourier_desc, fd_conf = fourier_descriptors(vertices_np)
        obj['properties']['fourier_descriptors'] = fourier_desc
        if fd_conf > 0.6: # If descriptors are meaningful
             obj['possible_labels'].append({'label': 'fourier_described', 'confidence': fd_conf, 'type': 'shape_descriptor'})

        # Symmetry & Orientation
        sym_score, sym_type, sym_conf = compute_geom_symmetry_score(vertices_np)
        obj['properties']['geometric_symmetry_score'] = sym_score
        obj['properties']['geometric_symmetry_type'] = sym_type
        if sym_type != 'none' and sym_conf > 0.6:
            obj['possible_labels'].append({'label': f'geometric_symmetric_{sym_type}', 'confidence': sym_conf, 'type': 'geometric_property'})

        orientation_deg, orient_conf = compute_orientation(vertices_np)
        obj['properties']['geometric_orientation_deg'] = orientation_deg
        if orient_conf > 0.5:
            obj['possible_labels'].append({'label': 'oriented_shape', 'confidence': orient_conf, 'type': 'geometric_property'})


        # --- Cutting-Edge Image Processing Features ---
        img_features_props = image_processing_features(vertices_np) # All features consolidated here
        if img_features_props:
            obj['properties'].update({'image_features': img_features_props}) # Nest under 'image_features'
            # Add labels based on strong image features
            if img_features_props.get('n_holes', 0) > 0:
                # Use confidence for holes based on how clearly they are detected (e.g., from Betti[1] if available)
                # For now, a fixed high confidence if detected.
                obj['possible_labels'].append({'label': 'has_holes_imgproc', 'confidence': 0.9, 'type': 'image_processing'})
            
            # Form factor based labels
            form_factor = img_features_props.get('form_factor', 0.0)
            if form_factor > 0.8: # Close to circular/compact
                obj['possible_labels'].append({'label': 'compact_shape_imgproc', 'confidence': calculate_confidence(form_factor, 1.0, 0.8), 'type': 'image_processing'})
            elif form_factor < 0.2: # Very elongated/jagged
                obj['possible_labels'].append({'label': 'elongated_jagged_shape_imgproc', 'confidence': calculate_confidence(form_factor, 0.2, 0.0, is_higher_better=False), 'type': 'image_processing'})

            # Hu Moments can also be used for specific shape recognition, but usually in ensemble
            # or for pattern matching. Not direct labels, but robust features.

        # --- Topological Features (Persistent Homology) ---
        betti_numbers, betti_conf = persistent_homology_features(vertices_np)
        obj['properties']['betti_numbers'] = betti_numbers
        if betti_conf > 0.6: # If topology is confidently detected
            if betti_numbers[0] > 1: # Multiple connected components
                obj['possible_labels'].append({'label': f'multiple_components_{betti_numbers[0]}', 'confidence': betti_conf, 'type': 'topological'})
            if betti_numbers[1] > 0: # Has holes
                obj['possible_labels'].append({'label': f'has_betti_holes_{betti_numbers[1]}', 'confidence': betti_conf, 'type': 'topological'})


        # --- Semantic Refinement Features (from semantic_refinement.py) ---
        connected_segments_list = detect_connected_line_segments(vertices_np)
        obj['properties']['semantic_connected_segments'] = len(connected_segments_list)
        obj['properties']['is_quadrilateral_like_semantic'] = is_quadrilateral_like(vertices_np)
        obj['properties']['is_smooth_curve_semantic'] = is_smooth_curve(vertices_np)
        obj['properties']['is_composite_shape_semantic'] = is_composite_shape(vertices_np)


        # --- Physics Inference Properties & Candidate Labels ---
        try:
            poly_for_physics = None
            if vertices_np.shape[0] >= 3:
                try:
                    poly_for_physics = PhysicsInference.polygon_from_vertices(vertices_np.tolist())
                    if not poly_for_physics.is_valid:
                        logging.warning(f"Physics Inference: Polygon for object {object_id} invalid, attempting repair with buffer(0).")
                        poly_for_physics = poly_for_physics.buffer(0) # Attempt self-correction for invalid polygons
                        if not poly_for_physics.is_valid or poly_for_physics.is_empty:
                            logging.warning(f"Physics Inference: Polygon for object {object_id} still invalid/empty after buffer. Some physics features might fail.")
                            obj['flags'].append('invalid_polygon_for_physics')
                            obj['possible_labels'].append({'label': 'invalid_physics_shape', 'confidence': 0.1, 'type': 'degenerate_physics'})
                except Exception as e:
                    logging.warning(f"Physics Inference: Polygon creation failed for {object_id}: {e}. Skipping some physics features.")
                    poly_for_physics = None # Ensure it's None if creation fails
                    obj['flags'].append('polygon_creation_failed_for_physics')
            else: # For single points or lines, polygon_from_vertices won't work
                poly_for_physics = None

            if poly_for_physics: # Only if a valid polygon was formed for physics
                # Individual shape physics properties
                is_conv, conf_conv = PhysicsInference.is_convex(poly_for_physics)
                obj['properties']['is_convex_physics'] = is_conv
                if is_conv and conf_conv > 0.8:
                    obj['possible_labels'].append({'label': 'physically_convex', 'confidence': conf_conv, 'type': 'physics_geometric'})
                elif not is_conv and conf_conv > 0.8: # Confidently non-convex
                    obj['possible_labels'].append({'label': 'physically_concave', 'confidence': conf_conv, 'type': 'physics_geometric'})

                # Symmetry and straight/arc segments from PhysicsInference
                # (even if geometric_detectors has similar, PhysicsInference's are physics-contextual)
                phy_sym_score, phy_sym_type, phy_sym_conf = PhysicsInference.symmetry_score(vertices_np)
                obj['properties']['physics_symmetry_score'] = phy_sym_score
                obj['properties']['physics_symmetry_type'] = phy_sym_type
                if phy_sym_type != 'none' and phy_sym_conf > 0.7:
                    obj['possible_labels'].append({'label': f'physics_symmetric_{phy_sym_type}', 'confidence': phy_sym_conf, 'type': 'physics_geometric'})
                
                num_arcs_phy, conf_arcs_phy = PhysicsInference.count_arcs(vertices_np)
                obj['properties']['num_arcs_physics'] = num_arcs_phy
                if num_arcs_phy > 0 and conf_arcs_phy > 0.7:
                    obj['possible_labels'].append({'label': f'physics_has_{num_arcs_phy}_arcs', 'confidence': conf_arcs_phy, 'type': 'physics_geometric'})

                num_straight_seg_phy, conf_straight_seg_phy = PhysicsInference.count_straight_segments(vertices_np)
                obj['properties']['num_straight_segments_physics'] = num_straight_seg_phy
                if num_straight_seg_phy > 0 and conf_straight_seg_phy > 0.7:
                    obj['possible_labels'].append({'label': f'physics_has_{num_straight_seg_phy}_straight_segments', 'confidence': conf_straight_seg_phy, 'type': 'physics_geometric'})

            else: # No valid polygon for physics
                 obj['properties']['is_convex_physics'] = False
                 obj['properties']['physics_symmetry_score'] = 0.0
                 obj['properties']['physics_symmetry_type'] = 'none'
                 obj['properties']['num_arcs_physics'] = 0
                 obj['properties']['num_straight_segments_physics'] = 0

        except Exception as e:
            logging.error(f"Error during individual physics inference for object {object_id}: {e}")
            obj['flags'].append('physics_inference_error_individual')
            obj['possible_labels'].append({'label': 'physics_error_feature_extraction', 'confidence': 0.1, 'type': 'degenerate_physics'})

        objects.append(obj)
        object_id_counter += 1

    # --- Ensemble labeling for initial primary label ---
    for obj in objects:
        final_label, final_confidence, ensemble_props = infer_shape_type_ensemble(
            np.array(obj['vertices']), problem_id, obj['possible_labels'],
            calculate_confidence, is_rectangle_detector, # from kwargs
            is_roughly_circular_detector, detect_connected_line_segments, # from kwargs
            is_quadrilateral_like, is_smooth_curve, is_composite_shape, # from kwargs
            convexity_defects_detector, is_point_cloud_detector # from kwargs
        )
        obj['label'] = final_label
        obj['confidence'] = final_confidence
        obj['properties'].update(ensemble_props) # Update properties with final ensemble selected props

        # Based on final label, assign a high-level shape_type (optional, for categorization)
        if 'line' in final_label: obj['shape_type'] = 'line'
        elif 'circle' in final_label or 'ellipse' in final_label or 'circular' in final_label: obj['shape_type'] = 'circular'
        elif 'polygon' in final_label or 'triangle' in final_label or 'quadrilateral' in final_label: obj['shape_type'] = 'polygon'
        elif 'point_cloud' in final_label: obj['shape_type'] = 'point_cloud'
        else: obj['shape_type'] = 'other'


    # --- Scene-Level Physics Inferences (Requires all objects to be processed) ---
    # This block requires access to *all* objects' initial properties to calculate relationships.
    if objects:
        try:
            # Determine a global ground_y_coord based on the lowest point of any object in the scene
            all_verts_in_scene = []
            for s_obj in objects:
                # Ensure we use the original vertices, not just the centroid
                verts = PhysicsInference.safe_extract_vertices(s_obj['vertices'])
                if verts:
                    all_verts_in_scene.extend(verts)
            
            global_ground_y_coord = None
            if all_verts_in_scene:
                global_ground_y_coord = np.min(np.array(all_verts_in_scene)[:, 1])
            
            # Apply scene-dependent physics features to each object
            for obj in objects:
                # Create a snapshot of all objects' IDs and vertices for PhysicsInference
                # This prevents direct modification issues during complex scene physics calls
                all_objects_snapshot = [
                    {'id': o['id'], 'vertices': o['vertices']} for o in objects
                ]
                
                # Relational physics features
                is_occ, conf_occ = PhysicsInference.is_occluded_physically(obj['id'], all_objects_snapshot)
                obj['properties']['is_occluded_physics'] = is_occ
                if is_occ and conf_occ > 0.7:
                    obj['possible_labels'].append({'label': 'occluded_object', 'confidence': conf_occ, 'type': 'physics_relation'})
                    obj['flags'].append('physically_occluded')

                # Stability/Support features
                is_touching, conf_touching = PhysicsInference.is_touching_ground(obj, global_ground_y_coord)
                obj['properties']['is_touching_ground_physics'] = is_touching
                if is_touching and conf_touching > 0.7:
                    obj['possible_labels'].append({'label': 'touching_ground', 'confidence': conf_touching, 'type': 'physics_relation'})
                    obj['flags'].append('physically_touching_ground')

                is_floating, conf_floating = PhysicsInference.is_floating(obj, global_ground_y_coord)
                obj['properties']['is_floating_physics'] = is_floating
                if is_floating and conf_floating > 0.7:
                    obj['possible_labels'].append({'label': 'floating_object', 'confidence': conf_floating, 'type': 'physics_relation'})
                    obj['flags'].append('physically_floating')

                is_stable_obj, conf_stable_obj = PhysicsInference.is_stable(obj['id'], all_objects_snapshot, global_ground_y_coord)
                obj['properties']['is_stable_physics'] = is_stable_obj
                if is_stable_obj and conf_stable_obj > 0.7:
                    obj['possible_labels'].append({'label': 'stable_object', 'confidence': conf_stable_obj, 'type': 'physics_relation'})
                    obj['flags'].append('physically_stable')

                num_contacts, conf_contacts = PhysicsInference.count_contact_points(obj['id'], all_objects_snapshot)
                obj['properties']['num_contact_points_physics'] = num_contacts
                if num_contacts > 0 and conf_contacts > 0.7:
                    obj['possible_labels'].append({'label': f'has_{num_contacts}_contacts_physics', 'confidence': conf_contacts, 'type': 'physics_relation'})
                    obj['flags'].append('physically_in_contact')

        except Exception as e:
            logging.error(f"Error during scene-level physics inference: {e}")
            # Add a general flag for physics errors affecting multiple objects
            for obj in objects:
                obj['flags'].append('physics_inference_error_scene_level')


    return objects, None # No fallback reasons if objects were created successfully

