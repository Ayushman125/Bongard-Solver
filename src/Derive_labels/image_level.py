from typing import List, Dict, Any, Optional

def ensure_all_strings(lst):
    """Recursively convert all items in a (possibly nested) list to strings."""
    if isinstance(lst, list):
        return [ensure_all_strings(x) for x in lst]
    if hasattr(lst, 'raw_command'):
        return str(lst.raw_command)
    return str(lst)
import logging
logger = logging.getLogger(__name__)
from src.Derive_labels.shape_utils import ensure_vertex_list, json_safe, _calculate_perimeter, calculate_complexity, _calculate_eccentricity, _check_horizontal_symmetry, _check_vertical_symmetry, _check_rotational_symmetry, _calculate_irregularity
from src.Derive_labels.features import extract_multiscale_features
import math
import os
import time

from src.Derive_labels.shape_utils import extract_position_and_rotation
from src.Derive_labels.file_io import FileIO
from src.physics_inference import PhysicsInference

from bongard.bongard import BongardImage
from src.Derive_labels.stroke_types import _extract_modifier_from_stroke
from src.Derive_labels.features import _detect_alternation
from src.Derive_labels.spatial_topological_features import compute_spatial_topological_features
from src.Derive_labels.utils import robust_flatten_and_stringify
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
from src.Derive_labels.context_features import BongardFeatureExtractor
from src.Derive_labels.compositional_features import hierarchical_clustering_heights, composition_tree_depth, composition_tree_branching_factor, subgraph_isomorphism_frequencies, composition_regularity_score, multi_level_symmetry_chains, layered_edge_complexity, overlapping_substructure_ratios, nested_convex_hull_levels
from src.Derive_labels.shape_utils import _calculate_compactness

# If FLAGGING_THRESHOLDS is not defined, set safe defaults:
FLAGGING_THRESHOLDS = {'min_aspect_ratio': 0.2, 'max_aspect_ratio': 5.0}

def process_single_image(action_commands: List[str], image_id: str, 
                       is_positive: bool, problem_id: str, category: str,
                       image_path: str, processing_stats=None, flag_case=None,
                       calculate_vertices_from_action=None,
                       calculate_composition_features=None,
                       calculate_physics_features=None) -> Optional[Dict[str, Any]]:
    """
    Refactored from monolithic logo_to_shape. All 'self.' references replaced by explicit arguments.
    - processing_stats: dict to track stats (optional)
    - flag_case: function to flag cases (optional)
    - calculate_vertices_from_action: function to calculate vertices from action (optional)
    - calculate_composition_features: function for composition features (optional)
    - calculate_physics_features: function for physics features (optional)
    """
    def ensure_float_tuples(vertices):
        """Convert a list of vertices to tuples of floats, ignoring strings."""
        result = []
        for v in vertices:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    result.append((float(v[0]), float(v[1])))
                except Exception:
                    continue
        return result
    logger.info(f"[INPUT] Processing image_id={image_id}, problem_id={problem_id}, is_positive={is_positive}")
    logger.info(f"[INPUT] action_commands type: {type(action_commands)}, value: {action_commands}")
    # Improved shape splitting logic
    logger.info(f"[SHAPE SPLIT] Raw action_commands: {action_commands}")
    shapes_commands = []
    if isinstance(action_commands, list) and action_commands and all(isinstance(cmd, list) for cmd in action_commands):
        # List of lists: each sublist is a shape/object
        for idx, sublist in enumerate(action_commands):
            logger.info(f"[SHAPE SPLIT] Sublist {idx}: {sublist}")
            valid_sublist = [cmd for cmd in sublist if isinstance(cmd, str) and cmd.strip()]
            if not valid_sublist:
                logger.warning(f"[SHAPE SPLIT] Sublist {idx} is empty or invalid: {sublist}")
            else:
                shapes_commands.append(valid_sublist)
        logger.info(f"[SHAPE SPLIT] Parsed shapes_commands (multi-object): {shapes_commands}")
    elif isinstance(action_commands, list) and all(isinstance(cmd, str) for cmd in action_commands):
        # Flat list: single shape
        valid_flat = [cmd for cmd in action_commands if isinstance(cmd, str) and cmd.strip()]
        if not valid_flat:
            logger.warning(f"[SHAPE SPLIT] Flat action_commands is empty or invalid: {action_commands}")
        else:
            shapes_commands.append(valid_flat)
        logger.info(f"[SHAPE SPLIT] Parsed shapes_commands (single-object): {shapes_commands}")
    else:
        logger.error(f"[SHAPE SPLIT] action_commands format not recognized: {action_commands}")
        return None

    # Parse each shape separately
    parser = ComprehensiveNVLabsParser()
    one_stroke_shapes = []
    original_action_commands = []
    for idx, shape_cmds in enumerate(shapes_commands):
        logger.info(f"[PARSER] Parsing shape {idx} commands: {shape_cmds}")
        parsed_shape = parser.parse_action_commands(shape_cmds, problem_id)
        # Normalize basic_actions to strings immediately after parsing
        if hasattr(parsed_shape, 'basic_actions'):
            norm_actions = []
            for a in parsed_shape.basic_actions:
                if type(a).__name__ in ['LineAction', 'ArcAction']:
                    norm_actions.append(str(getattr(a, 'raw_command', a)))
                else:
                    norm_actions.append(str(a))
            parsed_shape.basic_actions = norm_actions
        # Defensive: always initialize attributes, geometry, posrot_labels
        if not hasattr(parsed_shape, 'attributes') or parsed_shape.attributes is None:
            parsed_shape.attributes = {}
        if not hasattr(parsed_shape, 'geometry') or parsed_shape.geometry is None:
            parsed_shape.geometry = {}
        if not hasattr(parsed_shape, 'posrot_labels') or parsed_shape.posrot_labels is None:
            parsed_shape.posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
        if isinstance(parsed_shape, list):
            one_stroke_shapes.extend(parsed_shape)
            original_action_commands.extend(shape_cmds)
        else:
            one_stroke_shapes.append(parsed_shape)
            original_action_commands.extend(shape_cmds)
        # Safe logging for parsed_shape
        try:
            if hasattr(parsed_shape, 'basic_actions'):
                action_types = [type(a).__name__ for a in getattr(parsed_shape, 'basic_actions', [])]
                logger.info(f"[PARSER] Parsed shape {idx}: type={type(parsed_shape).__name__}, num_actions={len(getattr(parsed_shape, 'basic_actions', []))}, action_types={action_types}, attributes={getattr(parsed_shape, 'attributes', None)}")
            else:
                logger.info(f"[PARSER] Parsed shape {idx}: {type(parsed_shape).__name__}")
        except Exception as log_exc:
            logger.warning(f"[PARSER] Could not log parsed shape {idx}: {log_exc}")
    # --- PATCH: Assign aggregated features to each shape's attributes property ---
    # Initialize geometry and posrot_labels before loop
    geometry = {}
    posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
    feature_extractor = BongardFeatureExtractor()
    for idx, shape in enumerate(one_stroke_shapes):
        image_dict = {}
        image_dict['vertices'] = []
        image_dict['degenerate_case'] = False
        # Geometry extraction
        try:
            logger.info(f"[FEATURE EXTRACTION][geometry] INPUT: shape {idx} vertices: {getattr(shape, 'vertices', None)}")
            if hasattr(shape, 'vertices') and isinstance(shape.vertices, list):
                safe_vertices = [tuple(v) for v in shape.vertices if v is not None and isinstance(v, (list, tuple)) and len(v) == 2]
                image_dict['vertices'] = safe_vertices
                if len(safe_vertices) < 3:
                    image_dict['degenerate_case'] = True
                    logger.warning(f"[ATTR DEBUG] Shape {idx} has degenerate vertices: {safe_vertices}")
            else:
                image_dict['degenerate_case'] = True
                logger.warning(f"[ATTR DEBUG] Shape {idx} has no vertices.")
        except Exception as e:
            logger.error(f"[FEATURE EXTRACTION][geometry] Exception: {e}")
            image_dict['vertices'] = []
            image_dict['degenerate_case'] = True
        image_dict['strokes'] = []
        # Stroke-level feature extraction
        if hasattr(shape, 'basic_actions'):
            for i, a in enumerate(shape.basic_actions):
                command_str = str(a)
                try:
                    logger.info(f"[FEATURE EXTRACTION][stroke] INPUT: shape {idx} stroke {i} command: {command_str}")
                    stroke_vertices = calculate_vertices_from_action(a, i, bongard_image=shape)
                    from src.Derive_labels.shape_utils import calculate_geometry_consistent, compute_open_stroke_geometry
                    from src.Derive_labels.stroke_types import _calculate_stroke_specific_features
                    stroke_features = _calculate_stroke_specific_features(a, i, bongard_image=shape, parent_shape_vertices=stroke_vertices)
                    analytic_verts = stroke_features.get('analytic_vertices', stroke_vertices)
                    stroke_dict = {}
                    # Robust geometry assignment
                    if analytic_verts and len(analytic_verts) == 2:
                        stroke_geometry = compute_open_stroke_geometry(analytic_verts)
                        stroke_geometry['compactness'] = 0.0
                        stroke_geometry['convexity_ratio'] = 0.0
                        stroke_geometry['geom_complexity'] = min(len(analytic_verts)/10, 1)
                        stroke_geometry['degenerate_case'] = True
                        stroke_geometry['visual_complexity'] = min(max(stroke_geometry['perimeter']/max(stroke_geometry['perimeter'],1)*(1+stroke_features.get('robust_curvature',0)),0),1)
                    elif analytic_verts and len(analytic_verts) >= 3:
                        stroke_geometry = calculate_geometry_consistent(analytic_verts)
                        # Arc-specific calculations
                        if 'arc' in command_str:
                            r = stroke_geometry.get('width', 1.0) / 2
                            theta = math.radians(90)
                            area = 0.5 * r * r * (theta - math.sin(theta))
                            perim = r * theta
                            stroke_geometry['compactness'] = min(max(4 * math.pi * area / (perim ** 2), 0), 1)
                            stroke_geometry['convexity_ratio'] = min(max(area / perim, 0), 1)
                        else:
                            stroke_geometry['compactness'] = 0.0
                            stroke_geometry['convexity_ratio'] = 0.0
                        stroke_geometry['geom_complexity'] = min(len(analytic_verts) / 10, 1)
                        stroke_geometry['degenerate_case'] = False
                        stroke_geometry['visual_complexity'] = min(max(stroke_geometry.get('perimeter', 0.0) / max(stroke_geometry.get('perimeter', 1.0), 1) * (1 + stroke_features.get('robust_curvature', 0)), 0), 1)
                    else:
                        stroke_geometry = {
                            'width': 0.0,
                            'height': 0.0,
                            'area': 0.0,
                            'perimeter': 0.0,
                            'centroid': [0.0, 0.0],
                            'bounds': [0, 0, 0, 0],
                            'compactness': 0.0,
                            'convexity_ratio': 0.0,
                            'geom_complexity': 0.0,
                            'degenerate_case': True,
                            'visual_complexity': 0.0
                        }
                    # Robust post-processing for NaN/inf
                    for k, v in stroke_geometry.items():
                        if isinstance(v, float) and (not math.isfinite(v) or abs(v) > 1e6):
                            stroke_geometry[k] = 0.0
                            logger.warning(f"[STROKE FEATURE] Non-finite value for {k}: {v} (set to 0.0)")
                    stroke_dict.update(stroke_geometry)
                    stroke_dict['analytic_vertices'] = analytic_verts
                    stroke_dict['open_stroke_perimeter'] = stroke_features.get('open_stroke_perimeter', stroke_geometry.get('perimeter', 0.0))
                    if hasattr(a, 'arc_radius') and hasattr(a, 'arc_angle'):
                        radius = a.arc_radius
                        span = a.arc_angle
                        stroke_dict['arc_length'] = span/360*2*math.pi*radius
                        stroke_dict['arc_curvature'] = 1/max(radius,1e-6)
                    if 'line' in command_str:
                        angle = stroke_features.get('line_angle',0)
                        if abs(angle) < 5:
                            stroke_dict['shape_modifier'] = 'horizontal'
                        elif abs(angle-90) < 5 or abs(angle+90) < 5:
                            stroke_dict['shape_modifier'] = 'vertical'
                        else:
                            stroke_dict['shape_modifier'] = 'normal'
                    stroke_dict['command'] = command_str
                    stroke_dict['vertices'] = analytic_verts
                    stroke_dict['geometry'] = stroke_geometry
                    stroke_dict['label'] = category if category else 'unknown'
                    stroke_dict['class_label'] = problem_id if problem_id else 'unknown'
                    stroke_dict['is_positive'] = is_positive
                    logger.debug(f"[PATCH][is_positive] image_id={image_id}, stroke_idx={i}, assigned is_positive={is_positive}")
                    if not is_positive:
                        logger.info(f"[PATCH][NEGATIVE STROKE] image_id={image_id}, stroke_idx={i}, stroke_dict: {stroke_dict}")
                    if 'support_set_context' in stroke_dict:
                        del stroke_dict['support_set_context']
                    if 'discriminative_features' in stroke_dict and 'stats' in stroke_dict['discriminative_features']:
                        logger.info(f"[logo_to_shape] Attached discriminative_features['stats'] to stroke: {stroke_dict['discriminative_features']['stats']}")
                    image_dict['strokes'].append(stroke_dict)
                except Exception as e:
                    logger.error(f"[FEATURE EXTRACTION][stroke] Exception for shape {idx} stroke {i}: {e}")
                    image_dict['strokes'].append({'command': command_str, 'error': str(e), 'geometry': {}, 'vertices': [], 'is_positive': is_positive})
            logger.info(f"[ATTR DEBUG] Shape {idx} strokes (dicts): {image_dict['strokes']}")
        else:
            logger.warning(f"[ATTR DEBUG] Shape {idx} has no basic_actions.")
        analytic_attrs = {}
        if 'strokes' in image_dict:
            for stroke in image_dict['strokes']:
                # Try to extract analytic attributes from stroke geometry or stroke itself
                geom = stroke.get('geometry', {})
                mod = geom.get('shape_modifier') or stroke.get('shape_modifier')
                # Add all analytic attributes, not just horizontal/vertical
                if mod:
                    analytic_attrs[mod] = analytic_attrs.get(mod, 0) + 1
                # If stroke has 'analytic_attributes' field, merge those too
                if 'analytic_attributes' in stroke:
                    for attr, val in stroke['analytic_attributes'].items():
                        analytic_attrs[attr] = analytic_attrs.get(attr, 0) + val
        # Merge with shape.attributes
        shape_attrs = shape.attributes if hasattr(shape, 'attributes') and isinstance(shape.attributes, dict) else {}
        merged_attrs = dict(shape_attrs)
        for k, v in analytic_attrs.items():
            merged_attrs[k] = v
        image_dict['attributes'] = merged_attrs
        # --- PATCH: Always enforce and log shape-level geometry ---
        try:
            from src.Derive_labels.shape_utils import calculate_geometry_consistent, normalize_vertices
            shape_vertices = image_dict['vertices']
            # Normalize per shape before geometry computation
            norm_shape_vertices = normalize_vertices(shape_vertices) if shape_vertices and len(shape_vertices) >= 2 else shape_vertices
            if norm_shape_vertices and len(norm_shape_vertices) >= 3:
                shape_geometry = calculate_geometry_consistent(norm_shape_vertices)
                if not isinstance(shape_geometry, dict):
                    shape_geometry = {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0]}
                logger.info(f"[PATCH][SHAPE GEOMETRY] idx={idx}, vertices={norm_shape_vertices}, geometry={shape_geometry}")
                # --- PATCH: Add physics features to shape_geometry ---
                shape_geometry['robust_curvature'] = PhysicsInference.robust_curvature(norm_shape_vertices)
                shape_geometry['robust_angular_variance'] = PhysicsInference.robust_angular_variance(norm_shape_vertices)
                raw_vc = shape_geometry.get('perimeter',1.0)/max(shape_geometry.get('perimeter',1.0),1)*(1+shape_geometry.get('robust_curvature',0))
                shape_geometry['visual_complexity'] = min(max(raw_vc,0),1)
                shape_geometry['line_curvature_score'] = PhysicsInference.line_curvature_score(norm_shape_vertices)
                shape_geometry['arc_curvature_score'] = None
                if 'arc' in str(shape_geometry):
                    radius = shape_geometry.get('width', 1.0)
                    span_angle = 90
                    delta_theta = math.radians(span_angle)
                    shape_geometry['arc_curvature_score'] = PhysicsInference.arc_curvature_score(radius, delta_theta)
                logger.info(f"[PATCH][SHAPE PHYSICS] idx={idx}, physics features: {{'robust_curvature': {shape_geometry['robust_curvature']}, 'robust_angular_variance': {shape_geometry['robust_angular_variance']}, 'visual_complexity': {shape_geometry['visual_complexity']}, 'line_curvature_score': {shape_geometry['line_curvature_score']}, 'arc_curvature_score': {shape_geometry['arc_curvature_score']}}}")
                    # logger.info(f"[PATCH][SHAPE GEOMETRY] idx={idx}, vertices={norm_shape_vertices}, geometry={shape_geometry}")
                    # --- PATCH: Add physics features to shape_geometry ---
                    # shape_geometry['robust_curvature'] = PhysicsInference.robust_curvature(norm_shape_vertices)
                    # shape_geometry['robust_angular_variance'] = PhysicsInference.robust_angular_variance(norm_shape_vertices)
                    # raw_vc = shape_geometry.get('perimeter',1.0)/max(shape_geometry.get('perimeter',1.0),1)*(1+shape_geometry.get('robust_curvature',0))
                    # shape_geometry['visual_complexity'] = min(max(raw_vc,0),1)
                    # shape_geometry['line_curvature_score'] = PhysicsInference.line_curvature_score(norm_shape_vertices)
                    # shape_geometry['arc_curvature_score'] = None
                    # if 'arc' in str(shape_geometry):
                    #     radius = shape_geometry.get('width', 1.0)
                    #     span_angle = 90
                    #     delta_theta = math.radians(span_angle)
                    #     shape_geometry['arc_curvature_score'] = PhysicsInference.arc_curvature_score(radius, delta_theta)
                    # logger.info(f"[PATCH][SHAPE PHYSICS] idx={idx}, physics features: {{'robust_curvature': {shape_geometry['robust_curvature']}, 'robust_angular_variance': {shape_geometry['robust_angular_variance']}, 'visual_complexity': {shape_geometry['visual_complexity']}, 'line_curvature_score': {shape_geometry['line_curvature_score']}, 'arc_curvature_score': {shape_geometry['arc_curvature_score']}}}")
            else:
                shape_geometry = {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0]}
                logger.warning(f"[PATCH][SHAPE GEOMETRY] idx={idx} has insufficient vertices for geometry. Using default geometry: {shape_geometry}")
            image_dict['geometry'] = shape_geometry
            shape.geometry = shape_geometry
        except Exception as geo_exc:
            logger.error(f"[PATCH][SHAPE GEOMETRY] idx={idx} failed to calculate geometry: {geo_exc}")
            shape_geometry = {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0]}
            image_dict['geometry'] = shape_geometry
            shape.geometry = shape_geometry
        # Defensive: always ensure attributes and geometry are dicts
        if not hasattr(shape, 'attributes') or shape.attributes is None:
            shape.attributes = {}
        if not hasattr(shape, 'geometry') or shape.geometry is None:
            shape.geometry = shape_geometry
        # Bail out early if degenerate (no vertices or strokes)
        if not image_dict['vertices'] or not image_dict['strokes']:
            logger.warning(f"[ATTR DEBUG] Shape {idx} is degenerate (missing vertices or strokes). Skipping feature extraction and record construction.")
            return None
        # --- Now run diagnostics and feature extraction ---
        # logger.info(f"[DIAG] Shape {idx} image_dict for feature extraction: {image_dict}")
        try:
            # logger.info(f"[DIAG] feature_extractor type: {type(feature_extractor)}, module: {getattr(feature_extractor, '__module__', type(feature_extractor).__module__)}")
            # logger.info(f"[DIAG] About to extract features for image_dict={image_dict}")
            if not image_dict.get('vertices') or not isinstance(image_dict['vertices'], list) or not all(isinstance(v, (list, tuple)) and len(v) == 2 for v in image_dict['vertices']):
                logger.error(f"[DIAG] Problem with vertices in image_dict: {image_dict.get('vertices')}")
            if 'attributes' not in image_dict or not isinstance(image_dict.get('attributes'), dict):
                image_dict['attributes'] = {}
            if 'geometry' not in image_dict or not isinstance(image_dict.get('geometry'), dict):
                image_dict['geometry'] = shape_geometry
            # logger.info(f"[DIAG] Calling extract_image_features() on image_dict={image_dict}")
            result = feature_extractor.extract_image_features(image_dict)
            # logger.info(f"[DIAG] extract_image_features returned type={type(result)}, value={result}")
            # If features are empty/default for valid shapes, log a warning
            if not result or all(v in (None, [], {}, '') for v in result.values()):
                logger.warning(f"[DIAG] Features are empty/default for Shape {idx}. Input: {image_dict}")
            if result is None:
                # logger.error(f"[DIAG] extract_image_features returned None for image_dict={image_dict}. Setting attributes to empty dict.")
                shape.attributes = {}
            elif not isinstance(result, dict):
                # logger.error(f"[DIAG] extract_image_features returned non-dict type {type(result)} for image_dict={image_dict}. Setting attributes to empty dict.")
                shape.attributes = {}
            else:
                shape.attributes = result
            logger.info(f"[DIAG] Shape {idx} extracted attributes: {shape.attributes}")
        except Exception as e:
            logger.warning(f"[DIAG] Failed to extract features for shape {idx}: {e}")
            shape.attributes = {}


    # --- Validation: Check shape/stroke count match ---
    expected_strokes = sum(len(cmds) for cmds in shapes_commands)
    actual_shapes = len(one_stroke_shapes)
    logger.info(f"[VALIDATION] expected_strokes={expected_strokes}, actual_shapes={actual_shapes}")
    # Accept if both are 1 (single-object image)
    if len(shapes_commands) > 1 and len(one_stroke_shapes) != len(shapes_commands):
        logger.error(f"[DATA ISSUE] image_id={image_id}, problem_id={problem_id}: Number of action commands ({expected_strokes}) does not match number of parsed shapes ({actual_shapes}).\n    Action commands: {shapes_commands}\n    Parsed shapes: {one_stroke_shapes}")
        flag_case(
            category="data_issue",
            problem_id=problem_id,
            message=f"Number of action commands ({expected_strokes}) does not match number of parsed shapes ({actual_shapes}).",
            tags=["action_shape_mismatch"]
        )
        return None
    else:
        logger.info(f"[VALIDATION] image_id={image_id}, problem_id={problem_id}: Number of action commands matches number of parsed shapes ({expected_strokes})")
    # --- Only construct BongardImage and run downstream logic if shape count matches ---
    try:
        # Defensive: guarantee geometry, posrot_labels, and attributes are always dicts before output
        for shape in one_stroke_shapes:
            if not hasattr(shape, 'attributes') or shape.attributes is None:
                shape.attributes = {}
            if not hasattr(shape, 'geometry') or shape.geometry is None:
                shape.geometry = {}
            if not hasattr(shape, 'posrot_labels') or shape.posrot_labels is None:
                shape.posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
        geometry = geometry if isinstance(geometry, dict) else {'width': 0.0, 'height': 0.0, 'centroid': [0.0, 0.0]}
        posrot_labels = posrot_labels if isinstance(posrot_labels, dict) else {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
        # Restore BongardImage construction before any reference
        bongard_image = BongardImage(one_stroke_shapes)

        # --- Always include raw vertices from BongardImage ---
        vertices_raw = []
        for shape in getattr(bongard_image, 'one_stroke_shapes', []):
            if hasattr(shape, 'vertices') and isinstance(shape.vertices, list):
                # Defensive: ensure vertices are tuples of floats
                safe_vertices = ensure_float_tuples(shape.vertices)
                vertices_raw.extend(safe_vertices)
        # --- Restore all_actions as a list of strings from all shapes' basic_actions ---
        all_actions = []
        for shape in getattr(bongard_image, 'one_stroke_shapes', []):
            for a in getattr(shape, 'basic_actions', []):
                if hasattr(a, 'raw_command'):
                    all_actions.append(str(a.raw_command))
                else:
                    all_actions.append(str(a))
        all_actions = ensure_all_strings(all_actions)

        # --- Use standardize_coordinates for normalization ---
        from src.Derive_labels.shape_utils import standardize_coordinates, calculate_geometry_consistent
        normalized_vertices = standardize_coordinates(vertices_raw)
        # Ensure normalized_vertices are tuples of floats
        normalized_vertices = ensure_float_tuples(normalized_vertices)

        # --- Use calculate_geometry_consistent for geometry ---
        try:
            geometry = calculate_geometry_consistent(normalized_vertices)
            if not isinstance(geometry, dict):
                geometry = {}
        except Exception as geo_exc:
            logger.error(f"[FALLBACK LOGIC] image_id={image_id}, problem_id={problem_id}: Geometry calculation failed, fallback logic triggered. Error: {geo_exc}\nVertices: {normalized_vertices}")
            geometry = {}

        # For downstream compatibility, use normalized_vertices for features
        norm_vertices_for_features = normalized_vertices

        # --- Derive position and rotation labels from normalized vertices ---
        posrot_labels = extract_position_and_rotation(norm_vertices_for_features)
        # If degenerate, use safe defaults
        if not posrot_labels or not isinstance(posrot_labels, dict):
            posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}

        # --- Contextual/relational feature extraction ---
        # Prepare shapes as dicts with 'vertices' for relational extractor
        shape_list = []
        for shape in getattr(bongard_image, 'one_stroke_shapes', []):
            v = getattr(shape, 'vertices', None)
            shape_list.append({'vertices': v if v is not None else []})
        from src.Derive_labels.features import extract_relational_features
        logger.debug(f"[process_single_image] Passing shape_list to extract_relational_features: {shape_list}")
        context_relationships = extract_relational_features(shape_list) if shape_list else {}
        # Multi-scale features (using normalized vertices)
        multiscale_features = extract_multiscale_features(norm_vertices_for_features) if norm_vertices_for_features else {}

    # --- PATCH: Aggregate positive/negative shapes across all images at the problem level ---
    # Contextual features are now extracted at the problem level in main(), not per-image.
        # --- Calculate image features using robust polygon ---
        image_features = _calculate_image_features(norm_vertices_for_features, image_dict['strokes'], geometry if isinstance(geometry, dict) else {})
        # --- Aggregate stroke features to form image-level feature vector ---
        def aggregate_stroke_features(strokes):
            import numpy as np
            # Example: mean of area, perimeter, width, height, visual_complexity
            areas = [s.get('geometry', {}).get('area', 0.0) for s in strokes if isinstance(s, dict)]
            perimeters = [s.get('geometry', {}).get('perimeter', 0.0) for s in strokes if isinstance(s, dict)]
            widths = [s.get('geometry', {}).get('width', 0.0) for s in strokes if isinstance(s, dict)]
            heights = [s.get('geometry', {}).get('height', 0.0) for s in strokes if isinstance(s, dict)]
            visual_complexities = [s.get('geometry', {}).get('visual_complexity', 0.0) for s in strokes if isinstance(s, dict)]
            return {
                'area_mean': float(np.mean(areas)) if areas else 0.0,
                'perimeter_mean': float(np.mean(perimeters)) if perimeters else 0.0,
                'width_mean': float(np.mean(widths)) if widths else 0.0,
                'height_mean': float(np.mean(heights)) if heights else 0.0,
                'visual_complexity_mean': float(np.mean(visual_complexities)) if visual_complexities else 0.0,
            }
        image_level_features = aggregate_stroke_features(image_dict['strokes'])
        # Guarantee image_level_features is a dict
        if not isinstance(image_level_features, dict):
            logger.warning(f"image_level_features was not a dict: {image_level_features!r}; forcing empty dict")
            image_level_features = {}
        # Attach image-level features to output record
        centroid = geometry.get('centroid')
        width = geometry.get('width')
        height = geometry.get('height')
        if width is None:
            logger.error("Missing geometry['width']")
            width = 0.0
        if height is None:
            logger.error("Missing geometry['height']")
            height = 0.0
        composition_features = calculate_composition_features(all_actions)  # all_actions is now a list of strings
        physics_features = calculate_physics_features(norm_vertices_for_features, centroid=centroid, strokes=all_actions)

        # --- Relational/Topological/Sequential Features ---
        # Calculate relationships between shapes (not strokes)
        from src.Derive_labels.relational_features import calculate_shape_relationships
        robust_relational_features = calculate_shape_relationships(
            [{'vertices': shape.vertices} for shape in getattr(bongard_image, 'one_stroke_shapes', []) if hasattr(shape, 'vertices') and isinstance(shape.vertices, list) and len(shape.vertices) >= 3]
        )
        expected_rel_keys = ['adjacency', 'intersections', 'containment', 'overlap']
        if not robust_relational_features or not isinstance(robust_relational_features, dict):
            robust_relational_features = {k: 0 for k in expected_rel_keys}
        else:
            for k in expected_rel_keys:
                robust_relational_features.setdefault(k, 0)
        intersections = robust_relational_features.get('intersections')
        adjacency = robust_relational_features.get('adjacency')
        containment = robust_relational_features.get('containment')
        overlap = robust_relational_features.get('overlap')

        # Sequential pattern features (n-gram, alternation, regularity, dominant modifiers)
        from src.Derive_labels.features import extract_regularity_features, extract_dominant_shape_modifiers
        # Use corrected modifiers for n-gram and sequential features
        corrected_modifiers = []
        for s in all_actions:
            mod = _extract_modifier_from_stroke(s)
            # Use analytic attributes for modifier
            if 'horizontal' in str(s):
                corrected_modifiers.append('horizontal')
            elif 'vertical' in str(s):
                corrected_modifiers.append('vertical')
            elif 'arc' in str(s):
                corrected_modifiers.append('arc')
            else:
                corrected_modifiers.append(mod if mod in ['horizontal','vertical','arc','normal'] else 'normal')
            from src.Derive_labels.features import _extract_ngram_features
            ngram_features = _extract_ngram_features(corrected_modifiers)
        alternation = _detect_alternation(corrected_modifiers)
        regularity = extract_regularity_features(corrected_modifiers)
        dominant_modifiers = extract_dominant_shape_modifiers({'modifiers': corrected_modifiers})

        # Canonical summary and support set context
        from src.Derive_labels.file_io import FileIO
        from src.Derive_labels.relational_features import extract_support_set_context
        canonical_summary = {}
        support_set_context = {}
        try:
            # Example: load canonical summary from TSV using FileIO
            canonical_summary = FileIO.load_canonical_features(image_id)
        except Exception as e:
            logger.warning(f"[PATCH] Failed to load canonical summary for image_id={image_id}: {e}")
        # Remove per-image support-set context extraction. Only compute support-set context at the problem level in main().

        # Topological features (connectivity/type detection)
        from src.Derive_labels.features import extract_topological_features
        topo_shape_list = [{'vertices': shape.vertices} for shape in getattr(bongard_image, 'one_stroke_shapes', []) if hasattr(shape, 'vertices') and isinstance(shape.vertices, list)]
        logger.debug(f"[process_single_image] Passing topo_shape_list to extract_topological_features: {topo_shape_list}")
        graph_features = extract_topological_features(topo_shape_list)
        advanced_spatial_topological = compute_spatial_topological_features(image_dict)

        # --- PATCH: Use robust group-based stroke feature extraction ---
        from src.Derive_labels.stroke_types import extract_stroke_features_from_shapes, _calculate_stroke_type_differentiated_features
        stroke_features = extract_stroke_features_from_shapes(bongard_image, problem_id=problem_id)
        line_features = [sf['features'] for sf in stroke_features if 'line' in sf['stroke_command']]
        arc_features = [sf['features'] for sf in stroke_features if 'arc' in sf['stroke_command']]
        differentiated_features = _calculate_stroke_type_differentiated_features(
            {'line_features': line_features, 'arc_features': arc_features}, all_actions)

        action_program = []
        for shape in getattr(bongard_image, 'one_stroke_shapes', []):
            # Always build all_actions as strings before any join/logging/serialization
            all_actions = []
            for action in getattr(shape, 'basic_actions', []):
                if hasattr(action, 'raw_command') and isinstance(action.raw_command, str):
                    all_actions.append(action.raw_command)
                else:
                    all_actions.append(str(action))
            # Use robust flatten and stringify before joining
            actions_flat_str = robust_flatten_and_stringify(all_actions)
            try:
                joined = ",".join(actions_flat_str)
                logger.debug(f"[OUTPUT PATCH][action_program] Sublist joined: {joined}")
            except Exception as e:
                logger.error(f"FINAL JOIN EXCEPTION: {e}")
                raise
            action_program.append(actions_flat_str)
            logger.debug(f"[OUTPUT PATCH][action_program] Final action_program before serialization: {action_program}")
        # Defensive: ensure ngram features are stringified if joined/logged/serialized
        safe_ngram = ensure_all_strings(ngram_features) if isinstance(ngram_features, list) else ngram_features
        if isinstance(safe_ngram, list):
            print("[DEBUG FINAL NGRAM JOIN] types", [type(x) for x in safe_ngram])
            print("[DEBUG FINAL NGRAM JOIN] vals", safe_ngram)
            safe_ngram_strs = ensure_all_strings(safe_ngram)
            print("[DEBUG FINAL NGRAM after flatten]", safe_ngram_strs)
            joined_ngram = safe_join(safe_ngram_strs)
            print("[DEBUG FINAL NGRAM joined]", joined_ngram)
        else:
            print("[DEBUG FINAL NGRAM JOIN]", safe_ngram)
        # Defensive: ensure stroke_features lists are stringified
        for stroke in stroke_features:
            for k, v in stroke.items():
                if isinstance(v, list):
                    print(f"[DEBUG FINAL STROKE JOIN] {k} types", [type(x) for x in v])
                    print(f"[DEBUG FINAL STROKE JOIN] {k} vals", v)
                    try:
                        safe_v = ensure_all_strings(v)
                        print(f"[DEBUG FINAL STROKE after flatten] {k}", safe_v)
                        joined_v = safe_join(v)
                        print(f"[DEBUG FINAL STROKE joined] {k}", joined_v)
                    except Exception as e:
                        print(f"FINAL STROKE JOIN EXCEPTION {k}:", e)
                        raise
                    stroke[k] = safe_v
        # Defensive: ensure sequential features are stringified
        sequential_features = {
            'ngram': ensure_all_strings(ngram_features),
            'alternation': ensure_all_strings(alternation) if isinstance(alternation, list) else alternation,
            'regularity': ensure_all_strings(regularity) if isinstance(regularity, list) else regularity
        }

        from src.Derive_labels.shape_utils import json_safe
        # Defensive: ensure all action lists are json_safe before output
        # PATCH: Robust stringification and diagnostics before serialization
        def robust_stringify_list(lst):
            logger.debug(f"[PATCH][robust_stringify_list] Before: types={[type(x) for x in lst]}, values={lst}")
            result = [getattr(x, 'raw_command', str(x)) if type(x).__name__ in ['LineAction', 'ArcAction'] else str(x) if not isinstance(x, str) else x for x in lst]
            logger.debug(f"[PATCH][robust_stringify_list] After: types={[type(x) for x in result]}, values={result}")
            return result

        safe_action_program = [robust_stringify_list(sublist) if isinstance(sublist, list) else robust_stringify_list([sublist]) for sublist in action_program]
        logger.debug(f"[PATCH][process_single_image] Final safe_action_program before serialization: {safe_action_program}")
        safe_stroke_features = [
            {k: robust_stringify_list(v) if isinstance(v, list) else v for k, v in stroke.items()} for stroke in stroke_features
        ]
        logger.debug(f"[PATCH][process_single_image] Final safe_stroke_features before serialization: {safe_stroke_features}")
        # Robust stringification for vertices_raw and norm_vertices_for_features
        safe_vertices_raw = [str(v) if not isinstance(v, str) else v for v in vertices_raw] if isinstance(vertices_raw, list) else vertices_raw
        safe_vertices = [str(v) if not isinstance(v, str) else v for v in norm_vertices_for_features] if isinstance(norm_vertices_for_features, list) else norm_vertices_for_features
        safe_relational_features = json_safe(robust_relational_features)
        safe_context_relational_features = json_safe({
            'intersections': intersections,
            'adjacency': adjacency,
            'containment': containment,
            'overlap': overlap,
            'context_adjacency_matrix': context_relationships.get('adjacency_matrix'),
            'context_containment': context_relationships.get('containment'),
            'context_intersection_pattern': context_relationships.get('intersection_pattern'),
            'multiscale_features': multiscale_features
        })
        safe_sequential_features = json_safe({
            'ngram': ensure_all_strings(ngram_features),
            'alternation': ensure_all_strings(alternation) if isinstance(alternation, list) else alternation,
            'regularity': ensure_all_strings(regularity) if isinstance(regularity, list) else regularity
        })
        logger.debug(f"[OUTPUT PATCH] Types in stroke_features: {[type(x) for x in stroke_features]}")
        logger.debug(f"[OUTPUT PATCH] Types in action_program: {[type(x) for x in action_program]}")
        # Diagnostic logging and robust stringification for actions
        # FINAL GUARD: Ensure posrot_labels and geometry are always dicts with required keys
        if not isinstance(posrot_labels, dict):
            posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
        else:
            posrot_labels.setdefault('centroid', [0.0, 0.0])
            posrot_labels.setdefault('orientation_degrees', 0.0)
        if not isinstance(geometry, dict):
            geometry = {}
        logger.debug(f"[FINAL GUARD] posrot_labels={posrot_labels}, geometry={geometry}")
        logger.debug(f"[PATCH][actions field] Types: {[type(a) for a in all_actions]}, Values: {all_actions}")
        actions_joined = ",".join(robust_flatten_and_stringify(all_actions))
        logger.debug(f"[PATCH][actions field] Joined: {actions_joined}")

        # Defensive checks and logs for all dict/list variables used in output record construction
        # --- PATCH: Explicitly initialize all output record variables to safe defaults ---
        safe_stroke_features = stroke_features if stroke_features is not None else []
        safe_vertices_raw = vertices_raw if vertices_raw is not None else []
        safe_vertices = norm_vertices_for_features if norm_vertices_for_features is not None else []
        safe_action_program = shapes_commands if shapes_commands is not None else []
        safe_relational_features = robust_relational_features if robust_relational_features is not None else {}
        safe_context_relational_features = context_relationships if context_relationships is not None else {}
        safe_sequential_features = ngram_features if ngram_features is not None else {}
        differentiated_features = {'line_features': line_features, 'arc_features': arc_features} if 'line_features' in locals() and 'arc_features' in locals() else {}

        # Log types and values for all output record variables
        logger.info(f"[PATCH INIT] safe_stroke_features type: {type(safe_stroke_features)}, value: {safe_stroke_features}")
        logger.info(f"[PATCH INIT] safe_vertices_raw type: {type(safe_vertices_raw)}, value: {safe_vertices_raw}")
        logger.info(f"[PATCH INIT] safe_vertices type: {type(safe_vertices)}, value: {safe_vertices}")
        logger.info(f"[PATCH INIT] safe_action_program type: {type(safe_action_program)}, value: {safe_action_program}")
        logger.info(f"[PATCH INIT] safe_relational_features type: {type(safe_relational_features)}, value: {safe_relational_features}")
        logger.info(f"[PATCH INIT] safe_context_relational_features type: {type(safe_context_relational_features)}, value: {safe_context_relational_features}")
        logger.info(f"[PATCH INIT] safe_sequential_features type: {type(safe_sequential_features)}, value: {safe_sequential_features}")
        logger.info(f"[PATCH INIT] differentiated_features type: {type(differentiated_features)}, value: {differentiated_features}")

        # If any are None, set to safe default and log error
        if safe_stroke_features is None:
            logger.error("[PATCH ERROR] safe_stroke_features is None, setting to []")
            safe_stroke_features = []
        if safe_vertices_raw is None:
            logger.error("[PATCH ERROR] safe_vertices_raw is None, setting to []")
            safe_vertices_raw = []
        if safe_vertices is None:
            logger.error("[PATCH ERROR] safe_vertices is None, setting to []")
            safe_vertices = []
        if safe_action_program is None:
            logger.error("[PATCH ERROR] safe_action_program is None, setting to []")
            safe_action_program = []
        if safe_relational_features is None:
            logger.error("[PATCH ERROR] safe_relational_features is None, setting to {}")
            safe_relational_features = {}
        if safe_context_relational_features is None:
            logger.error("[PATCH ERROR] safe_context_relational_features is None, setting to {}")
            safe_context_relational_features = {}
        if safe_sequential_features is None:
            logger.error("[PATCH ERROR] safe_sequential_features is None, setting to {}")
            safe_sequential_features = {}
        if differentiated_features is None:
            logger.error("[PATCH ERROR] differentiated_features is None, setting to {}")
            differentiated_features = {}
        if not isinstance(posrot_labels, dict):
            logger.error(f"[DEFENSIVE ERROR] posrot_labels is None or not dict before serialization! Value: {posrot_labels}")
            posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
        if not isinstance(geometry, dict):
            logger.error(f"[DEFENSIVE ERROR] geometry is None or not dict before serialization! Value: {geometry}")
            geometry = {}
        if stroke_features is None:
            logger.error(f"[DEFENSIVE ERROR] stroke_features is None before serialization!")
            stroke_features = []
        if all_actions is None:
            logger.error(f"[DEFENSIVE ERROR] all_actions is None before serialization!")
            all_actions = []
        if vertices_raw is None:
            logger.error(f"[DEFENSIVE ERROR] vertices_raw is None before serialization!")
            vertices_raw = []
        if norm_vertices_for_features is None:
            logger.error(f"[DEFENSIVE ERROR] norm_vertices_for_features is None before serialization!")
            norm_vertices_for_features = []
        for shape in getattr(bongard_image, 'one_stroke_shapes', []):
            if not hasattr(shape, 'attributes') or shape.attributes is None:
                logger.error(f"[DEFENSIVE ERROR] shape.attributes is None before serialization! For shape: {shape}")
                shape.attributes = {}
        image_features = image_features if image_features is not None else []
        physics_features = physics_features if physics_features is not None else []
        composition_features = composition_features if composition_features is not None else []
        graph_features = graph_features if graph_features is not None else []
        logger.info(f"[PATCH FINAL] Mapping output record fields...")
        logger.info(f"[PATCH FINAL] image_features (raw): {image_features}")
        logger.info(f"[PATCH FINAL] relational_features (raw): {safe_relational_features}")
        logger.info(f"[PATCH FINAL] image_canonical_summary (raw): {canonical_summary}")
        logger.info(f"[PATCH FINAL] support_set_context (raw): {support_set_context}")
        # --- PATCH: Collect analytic attributes for all strokes ---
        analytic_attributes = []
        for shape in one_stroke_shapes:
            if hasattr(shape, 'basic_actions'):
                for a in shape.basic_actions:
                    from src.Derive_labels.stroke_types import _calculate_stroke_specific_features
                    stroke_features = _calculate_stroke_specific_features(a, 0, bongard_image=shape)
                    analytic_attr = stroke_features.get('analytic_attribute', None)
                    if analytic_attr:
                        analytic_attributes.append(analytic_attr)
        # Compute analytic n-gram features
        from src.Derive_labels.features import _extract_ngram_features
        analytic_ngram = _extract_ngram_features(analytic_attributes)
        # Ensure advanced_compositional is always defined before output record construction
        advanced_compositional = {}
        complete_record = {
            'image_id': image_id,
            'problem_id': problem_id,
            'category': category,
            'label': 'positive' if is_positive else 'negative',
            'image_path': image_path,
            'strokes': robust_flatten_and_stringify(safe_stroke_features),
            'num_strokes': len(safe_stroke_features),
            'raw_vertices': robust_flatten_and_stringify(safe_vertices_raw),
            'vertices': robust_flatten_and_stringify(safe_vertices),
            'num_vertices': len(safe_vertices) if safe_vertices else 0,
            'position_label': robust_flatten_and_stringify(posrot_labels.get('centroid')),
            'rotation_label_degrees': robust_flatten_and_stringify(posrot_labels.get('orientation_degrees')),
            'image_features': image_features if isinstance(image_features, dict) else image_features,
            'image_level_features': image_level_features,
            'physics_features': physics_features if isinstance(physics_features, dict) else physics_features,
            'composition_features': composition_features if isinstance(composition_features, dict) else composition_features,
            'stroke_type_features': differentiated_features if isinstance(differentiated_features, dict) else differentiated_features,
            'image_canonical_summary': canonical_summary if isinstance(canonical_summary, dict) else canonical_summary,
            'support_set_context': support_set_context if isinstance(support_set_context, dict) else support_set_context,
            'dominant_shape_modifiers': dominant_modifiers,
            'processing_metadata': {
                'processing_timestamp': time.time(),
                'feature_count': len(image_features) + len(physics_features) + len(composition_features) if isinstance(image_features, dict) and isinstance(physics_features, dict) and isinstance(composition_features, dict) else 0
            },
            'actions': [",".join(robust_flatten_and_stringify(sublist)) for sublist in safe_action_program],
            'action_program': [",".join(robust_flatten_and_stringify(sublist)) for sublist in safe_action_program],
            'geometry': geometry if isinstance(geometry, dict) else geometry,
            'relational_features': safe_relational_features if isinstance(safe_relational_features, dict) else {},
            'context_relational_features': safe_context_relational_features if isinstance(safe_context_relational_features, dict) else {},
            'sequential_features': {
                'ngram': ngram_features if isinstance(ngram_features, dict) else ngram_features,
                'analytic_attributes': analytic_attributes,
                'analytic_ngram': analytic_ngram,
                'alternation': alternation,
                'regularity': regularity
            },
            'topological_features': graph_features if isinstance(graph_features, dict) else {},
            'advanced_spatial_topological': advanced_spatial_topological if isinstance(advanced_spatial_topological, dict) else {},
            'advanced_compositional_features': advanced_compositional,
            'multiscale': multiscale_features if isinstance(multiscale_features, dict) else {},
        }
        # Ensure required fields are always present, even if empty
        for key in ['multiscale', 'relational_features', 'topological_features']:
            if key not in complete_record or complete_record[key] is None:
                complete_record[key] = {}
        logger.info(f"[PATCH FINAL] Output record: {complete_record}")
            # Granular logging before every subscript operation
        try:
            logger.debug(f"[GRANULAR LOG] posrot_labels type: {type(posrot_labels)}, value: {posrot_labels}")
            _ = posrot_labels['centroid']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] posrot_labels['centroid'] failed: {e}")
        try:
            _ = posrot_labels['orientation_degrees']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] posrot_labels['orientation_degrees'] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] geometry type: {type(geometry)}, value: {geometry}")
            _ = geometry['centroid']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] geometry['centroid'] failed: {e}")
        try:
            _ = geometry['width']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] geometry['width'] failed: {e}")
        try:
            _ = geometry['height']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] geometry['height'] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] safe_stroke_features type: {type(safe_stroke_features)}, value: {safe_stroke_features}")
            _ = safe_stroke_features[0]
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] safe_stroke_features[0] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] safe_action_program type: {type(safe_action_program)}, value: {safe_action_program}")
            _ = safe_action_program[0]
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] safe_action_program[0] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] safe_vertices_raw type: {type(safe_vertices_raw)}, value: {safe_vertices_raw}")
            _ = safe_vertices_raw[0]
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] safe_vertices_raw[0] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] safe_vertices type: {type(safe_vertices)}, value: {safe_vertices}")
            _ = safe_vertices[0]
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] safe_vertices[0] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] image_features type: {type(image_features)}, value: {image_features}")
            if isinstance(image_features, dict):
                if 'bounding_box' not in image_features or image_features['bounding_box'] is None:
                    logger.warning(f"[PATCH][image_features] 'bounding_box' missing or None for image_id={image_id}, problem_id={problem_id}. Setting to safe default.")
                    image_features['bounding_box'] = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                _ = image_features['bounding_box']
            else:
                logger.error(f"[PATCH][image_features] image_features is not a dict for image_id={image_id}, problem_id={problem_id}. Value: {image_features}")
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] image_features['bounding_box'] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] physics_features type: {type(physics_features)}, value: {physics_features}")
            _ = physics_features['moment_of_inertia']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] physics_features['moment_of_inertia'] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] composition_features type: {type(composition_features)}, value: {composition_features}")
            _ = composition_features['some_key'] if isinstance(composition_features, dict) and 'some_key' in composition_features else None
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] composition_features['some_key'] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] differentiated_features type: {type(differentiated_features)}, value: {differentiated_features}")
            _ = differentiated_features['line_features']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] differentiated_features['line_features'] failed: {e}")
        try:
            logger.debug(f"[GRANULAR LOG] graph_features type: {type(graph_features)}, value: {graph_features}")
            _ = graph_features['type']
        except Exception as e:
            logger.error(f"[GRANULAR ERROR] graph_features['type'] failed: {e}")

        processing_stats['successful'] += 1
        return json_safe(complete_record)
    except Exception as e:
        import traceback
        error_msg = f"Error processing image {image_id}: {e}"
        stack_trace = traceback.format_exc()
        logger.error(f"{error_msg}\n{stack_trace}")
        flag_case('unknown', problem_id, error_msg, ['image_processing_error'])
        # Save to flagged_issues.txt
        try:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
            os.makedirs(output_dir, exist_ok=True)
            flag_path = os.path.join(output_dir, 'flagged_issues.txt')
            with open(flag_path, 'a', encoding='utf-8') as f:
                f.write(f"[ERROR] image_id={image_id}, problem_id={problem_id}, error={e}\n{stack_trace}\n")
        except Exception as file_exc:
            logger.error(f"[process_single_image] Could not write to flagged_issues.txt: {file_exc}")
        return None

def _calculate_image_features(vertices: List[tuple], strokes: List, geometry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove self from signature. All logic remains unchanged.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"[_calculate_image_features] INPUTS: vertices={vertices}, strokes={strokes}, geometry={geometry}")
    vertices = ensure_vertex_list(vertices)
    default_features = {
        'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
        'centroid': [0.0, 0.0],
        'width': 0.0,
        'height': 0.0,
        'area_raw': 0.0,
        'area_normalized': 0.0,
        'perimeter_raw': 0.0,
        'perimeter_normalized': 0.0,
        'aspect_ratio': 1.0,
        'convexity_ratio': 0.0,
        'is_convex': False,
        'compactness': 0.0,
        'polsby_popper_compactness': 0.0,
        'eccentricity': 0.0,
        'symmetry_score': 0.0,
        'horizontal_symmetry': 0.0,
        'vertical_symmetry': 0.0,
        'rotational_symmetry': 0.0,
        'has_quadrangle': False,
        'geometric_complexity': 0.0,
        'visual_complexity': 0.0,
        'curvature_score': 0.0,
        'angular_variance': 0.0,
        'moment_of_inertia': 0.0,
        'irregularity_score': 0.0,
        'standardized_complexity': 0.0
    }
    if not vertices:
        logger.warning("[_calculate_image_features] No vertices provided, returning default features.")
        logger.info(f"[_calculate_image_features] FALLBACK: Returning default features due to missing vertices. INPUTS: vertices={vertices}, strokes={strokes}, geometry={geometry}")
        return json_safe(default_features)


    # --- Real Data Extraction for Advanced Features ---
    # Build feature_matrix, labels, shape_types, class_labels, graphs, etc. from strokes and geometry
    import numpy as np
    labels, shape_types, class_labels = [], [], []
    feature_matrix = []
    graphs = []
    tree, root = {}, None
    symmetries = {}
    edges = []
    substructures = []
    parts = []
    position_func = None
    groups = []

    # Only extract per-stroke features for stroke-level analytics, not for context metrics
    for stroke in strokes:
        if isinstance(stroke, dict):
            fvec = [stroke.get('geometry', {}).get('area', 0.0),
                    stroke.get('geometry', {}).get('perimeter', 0.0),
                    stroke.get('geometry', {}).get('width', 0.0),
                    stroke.get('geometry', {}).get('height', 0.0),
                    stroke.get('geometry', {}).get('visual_complexity', 0.0),
                    stroke.get('geometry', {}).get('geom_complexity', 0.0),
                    stroke.get('geometry', {}).get('symmetry_score', 0.0)]
            feature_matrix.append(fvec)
            if 'label' in stroke:
                labels.append(stroke['label'])
            if 'geometry' in stroke and 'shape_modifier' in stroke['geometry']:
                shape_types.append(stroke['geometry']['shape_modifier'])
            if 'class_label' in stroke:
                class_labels.append(stroke['class_label'])
            if 'geometry' in stroke and 'vertices' in stroke['geometry']:
                try:
                    from networkx import Graph
                    g = Graph()
                    verts = stroke['geometry']['vertices']
                    for i in range(len(verts)-1):
                        g.add_edge(i, i+1)
                    graphs.append(g)
                except Exception:
                    pass
            if 'geometry' in stroke and 'analytic_vertices' in stroke['geometry']:
                substructures.append(stroke['geometry']['analytic_vertices'])
                edges.extend([(stroke['geometry']['analytic_vertices'][i], stroke['geometry']['analytic_vertices'][i+1])
                             for i in range(len(stroke['geometry']['analytic_vertices'])-1)])
            if 'geometry' in stroke and 'centroid' in stroke['geometry']:
                parts.append(stroke['geometry']['centroid'])

    feature_matrix = np.array(feature_matrix) if feature_matrix else np.array([])

    if len(parts) > 1:
        tree = {i: [i+1] for i in range(len(parts)-1)}
        root = 0
    if parts:
        groups = [parts]
    try:
        from shapely.geometry import Polygon
        poly = None
        try:
            logger.debug(f"[POLYGON CREATION] Attempting to create Polygon with vertices: {vertices}")
            from src.physics_inference import PhysicsInference
            poly = PhysicsInference.polygon_from_vertices(vertices)
            if poly is not None:
                logger.debug(f"[POLYGON REPAIR] Final polygon: is_valid={poly.is_valid}, is_empty={poly.is_empty}, area={poly.area}")
            else:
                logger.error(f"[POLYGON REPAIR] Could not repair polygon for vertices: {vertices}")
        except Exception as e:
            logger.error(f"[POLYGON REPAIR] Exception: {e}. Geometry cannot be recovered for vertices: {vertices}")
            poly = None

        # --- Defensive conversion of geometry values to float ---
        def safe_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                logger.warning(f"[_calculate_image_features] Value '{val}' could not be converted to float. Using default {default}.")
                return default

        num_strokes = len(strokes)
        max_strokes = 50
        perimeter_raw = safe_float(_calculate_perimeter(vertices))
        area_raw = safe_float(PhysicsInference.area(poly) if poly else 0.0)
        hull_perimeter = perimeter_raw
        hull_area = area_raw
        if poly is not None and hasattr(poly, 'convex_hull'):
            try:
                hull_perimeter = safe_float(poly.convex_hull.length)
                hull_area = safe_float(poly.convex_hull.area)
            except Exception as e:
                logger.warning(f"[_calculate_image_features] Error getting convex hull: {e}")
        # Normalized perimeter and area (relative to convex hull)
        perimeter_norm = min(max(safe_float(perimeter_raw) / safe_float(hull_perimeter), 0.0), 1.0) if hull_perimeter else 0.0
        area_norm = min(max(safe_float(area_raw) / safe_float(hull_area), 0.0), 1.0) if hull_area else 0.0

        # Use robust, analytic, normalized formulas for all features:
        try:
            curvature_score = safe_float(PhysicsInference.robust_curvature(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in robust_curvature: {e}")
            curvature_score = 0.0
        try:
            angular_variance = safe_float(PhysicsInference.robust_angular_variance(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in robust_angular_variance: {e}")
            angular_variance = 0.0
        try:
            moment_of_inertia = safe_float(PhysicsInference.moment_of_inertia(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in moment_of_inertia: {e}")
            moment_of_inertia = 0.0
        try:
            visual_complexity = safe_float(PhysicsInference.visual_complexity(num_strokes, max_strokes, perimeter_raw, hull_perimeter, curvature_score))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in visual_complexity: {e}")
            visual_complexity = 0.0
        visual_complexity_norm = min(max(visual_complexity, 0.0), 1.0) if visual_complexity is not None else 0.0

        # --- Standardized complexity metric ---
        try:
            logger.info(f"[complexity] Calling calculate_complexity with vertices: {vertices}")
            complexity = safe_float(calculate_complexity(vertices))
            logger.info(f"[complexity] Output for vertices count {len(vertices)}: {complexity}")
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in calculate_complexity: {e}")
            complexity = 0.0

        from src.Derive_labels.stroke_types import _compute_bounding_box
        try:
            bbox = _compute_bounding_box(vertices)
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in bounding box: {e}")
            bbox = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        width = safe_float(geometry.get('width', 0.0))
        height = safe_float(geometry.get('height', 0.0))
        area_val = safe_float(geometry.get('area', 0.0))
        # Defensive: ensure width and height are present
        if 'width' not in geometry or 'height' not in geometry:
            logger.warning(f"[_calculate_image_features] Geometry missing width/height: {geometry}. Setting to 0.0.")
            width = 0.0
            height = 0.0
            geometry['width'] = width
            geometry['height'] = height
        try:
            # Compute area and perimeter from vertices using Shapely
            from shapely.geometry import Polygon
            poly_tmp = Polygon(vertices)
            area = poly_tmp.area
            perimeter = poly_tmp.length
            polsby_popper = safe_float(PhysicsInference.polsby_popper_compactness(area, perimeter))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in polsby_popper_compactness: {e}")
            polsby_popper = 0.0
        logger.info(f"[_calculate_image_features] polsby_popper: {polsby_popper}")
        try:
            eccentricity = safe_float(_calculate_eccentricity(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in eccentricity: {e}")
            eccentricity = 0.0
        try:
            symmetry_score = safe_float(PhysicsInference.symmetry_score(vertices)) if perimeter_raw > 0 and area_raw > 0 else 0.0
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in symmetry_score: {e}")
            symmetry_score = 0.0
        try:
            horizontal_symmetry = safe_float(_check_horizontal_symmetry(vertices, poly))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in horizontal_symmetry: {e}")
            horizontal_symmetry = 0.0
        try:
            vertical_symmetry = safe_float(_check_vertical_symmetry(vertices, poly))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in vertical_symmetry: {e}")
            vertical_symmetry = 0.0
        try:
            rotational_symmetry = safe_float(_check_rotational_symmetry(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in rotational_symmetry: {e}")
            rotational_symmetry = 0.0
        try:
            has_quadrangle = True if poly and hasattr(poly, 'exterior') and hasattr(poly.exterior, 'coords') and len(poly.exterior.coords)-1 == 4 else PhysicsInference.has_quadrangle(vertices)
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in has_quadrangle: {e}")
            has_quadrangle = False
        try:
            geometric_complexity = safe_float(PhysicsInference.geometric_complexity(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in geometric_complexity: {e}")
            geometric_complexity = 0.0
        try:
            irregularity_score = safe_float(_calculate_irregularity(vertices))
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in irregularity_score: {e}")
            irregularity_score = 0.0

        aspect_ratio = max(FLAGGING_THRESHOLDS['min_aspect_ratio'], min(safe_float(width) / safe_float(height, 1.0), FLAGGING_THRESHOLDS['max_aspect_ratio'])) if height else 1.0
        try:
            convexity_ratio = (max(0.0, min(1.0, safe_float(poly.area) / safe_float(poly.convex_hull.area))) if poly and safe_float(poly.area) != 0 and safe_float(poly.convex_hull.area) != 0 else 0.0)
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in convexity_ratio: {e}")
            convexity_ratio = 0.0
        try:
            is_convex = PhysicsInference.is_convex(poly) if poly else False
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in is_convex: {e}")
            is_convex = False
        try:
            compactness = _calculate_compactness(area_raw, perimeter_raw)
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error in compactness: {e}")
            compactness = 0.0

        features = {
            'bounding_box': bbox,
            'centroid': geometry.get('centroid', [0.0, 0.0]),
            'width': width,
            'height': height,
            'area_raw': area_raw,
            'area_normalized': area_norm,
            'perimeter_raw': perimeter_raw,
            'perimeter_normalized': perimeter_norm,
            'aspect_ratio': aspect_ratio,
            'convexity_ratio': convexity_ratio,
            'is_convex': is_convex,
            'compactness': compactness,
            'polsby_popper_compactness': polsby_popper,
            'eccentricity': eccentricity,
            'symmetry_score': symmetry_score,
            'horizontal_symmetry': horizontal_symmetry,
            'vertical_symmetry': vertical_symmetry,
            'rotational_symmetry': rotational_symmetry,
            'has_quadrangle': has_quadrangle,
            'geometric_complexity': geometric_complexity,
            'visual_complexity': visual_complexity_norm,  # normalized [0,1]
            'curvature_score': curvature_score,  # analytic, normalized
            'angular_variance': angular_variance,  # analytic, normalized
            'moment_of_inertia': moment_of_inertia,  # analytic, normalized
            'irregularity_score': irregularity_score,
            'standardized_complexity': complexity
        }
        # --- Compositional & Hierarchical Features Integration ---
        features['clustering_heights'] = hierarchical_clustering_heights(feature_matrix).tolist() if feature_matrix.size else []
        features['tree_depth'] = composition_tree_depth(tree, root) if tree and root else 0
        features['tree_branching_factor'] = composition_tree_branching_factor(tree, root) if tree and root else 0.0
        features['subgraph_isomorphism_count'] = subgraph_isomorphism_frequencies(graphs) if graphs else 0
        features['composition_regularity'] = composition_regularity_score(parts, position_func) if parts and position_func else 0.0
        features['symmetry_chains'] = multi_level_symmetry_chains(symmetries) if symmetries else 0
        features['layered_edge_complexity'] = layered_edge_complexity(edges) if edges else 0.0
        features['overlap_ratio'] = overlapping_substructure_ratios(substructures, lambda x, y: 0) if substructures else 0.0
        features['nested_convex_hull_levels'] = nested_convex_hull_levels(groups) if groups else 0
        logger.info(f"[_calculate_image_features] OUTPUT: {features}")
        return json_safe(features)
    except Exception as e:
        logger.error(f"[_calculate_image_features] Exception: {e}", exc_info=True)
        logger.warning("[_calculate_image_features] Returning default features due to error.")
        return json_safe(default_features)