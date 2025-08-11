import logging
import importlib.util
import sys
sys.stdout.reconfigure(encoding='utf-8')
def fully_stringify(obj):
    if isinstance(obj, dict):
        return {fully_stringify(k): fully_stringify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fully_stringify(x) for x in obj]
    elif hasattr(obj, 'raw_command'):
        return str(obj.raw_command)
    elif type(obj).__name__ in ['LineAction', 'ArcAction']:
        return str(getattr(obj, 'raw_command', str(obj)))
    else:
        return str(obj)

# Robust flatten and stringify for any nested actions list
def robust_flatten_and_stringify(lst):
    result = []
    if isinstance(lst, list):
        for x in lst:
            if isinstance(x, list):
                result.extend(robust_flatten_and_stringify(x))
            elif hasattr(x, 'raw_command'):
                result.append(str(x.raw_command))
            elif type(x).__name__ in ['LineAction', 'ArcAction']:
                result.append(str(getattr(x, 'raw_command', str(x))))
            else:
                result.append(str(x))
    else:
        result.append(str(lst))
    return result
#!/usr/bin/env python3
"""
Logo to Shape Conversion Script - Complete End-to-End Pipeline

This script integrates the data loader, logo parser, and physics inference
to create comprehensive derived labels for Bongard-LOGO images.

Handles complex images composed of multiple strokes and calculates:
- Individual stroke features
- Composite image features  
- Physics and geometry attributes
- Semantic and structural properties
"""
def ensure_all_strings(lst):
    """Recursively convert all items in a (possibly nested) list to strings."""
    if isinstance(lst, list):
        return [ensure_all_strings(x) for x in lst]
    if hasattr(lst, 'raw_command'):
        return str(lst.raw_command)
    return str(lst)

def safe_join(lst, sep=','):
    """Join a list into a string, robustly converting all items to strings first using fully_stringify."""
    def fully_stringify(obj):
        if isinstance(obj, dict):
            return {fully_stringify(k): fully_stringify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fully_stringify(x) for x in obj]
        elif hasattr(obj, 'raw_command'):
            return str(obj.raw_command)
        elif type(obj).__name__ in ['LineAction', 'ArcAction']:
            return str(getattr(obj, 'raw_command', str(obj)))
        else:
            return str(obj)
    if isinstance(lst, list):
        safe_items = [fully_stringify(x) for x in lst]
        logger.debug(f"[SAFE_JOIN DEBUG] safe_items: {safe_items}")
        safe_items = [str(x) for x in safe_items]
        return sep.join(safe_items)
    return str(lst)


import argparse
import csv
import sys
import os
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.data_pipeline.data_loader import load_action_programs
from src.bongard_augmentor.hybrid import HybridAugmentor

from bongard.bongard import BongardImage
from src.physics_inference import PhysicsInference

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Flagging thresholds and constants
FLAGGING_THRESHOLDS = {
    'min_vertices': 3,
    'max_vertices': 1000, 
    'min_area': 1e-6,
    'max_area': 1e6,
    'min_aspect_ratio': 1e-3,
    'max_aspect_ratio': 1000,
    'max_stroke_count': 50,
    'geometry_nan_tolerance': 0,
    'symmetry_score_max': 2.0,  # RMSE for [0,1] normalized points
    'suspicious_parameter_threshold': 1e6
}



from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser, OneStrokeShape
from src.Derive_labels.stroke_types import extract_action_type_prefixes
from src.Derive_labels.shape_utils import normalize_vertices, calculate_geometry, extract_position_and_rotation, ensure_vertex_list, _calculate_perimeter, _calculate_curvature_score, _calculate_edge_length_variance, json_safe, _calculate_irregularity, calculate_complexity
from src.Derive_labels.stroke_types import _extract_modifier_from_stroke, _calculate_stroke_specific_features, _calculate_stroke_type_differentiated_features
from src.Derive_labels.features import _actions_to_geometries, _extract_ngram_features, _extract_graph_features
from src.Derive_labels.features import _detect_alternation
from src.Derive_labels.shape_utils import _calculate_homogeneity, _calculate_angular_variance,safe_divide, _calculate_compactness, _calculate_eccentricity, _check_horizontal_symmetry, _check_vertical_symmetry, _check_rotational_symmetry
from src.Derive_labels.file_io import FileIO
from src.Derive_labels.features import extract_multiscale_features
from src.Derive_labels.context_features import BongardFeatureExtractor
from src.Derive_labels.spatial_topological_features import compute_spatial_topological_features
from src.Derive_labels.contextual_features import (
    positive_negative_contrast_score,
    support_set_mutual_information,
    label_consistency_ratio,
    concept_drift_score,
    support_set_shape_cooccurrence,
    category_consistency_score,
    class_prototype_distance,
    feature_importance_ranking,
    cross_set_symmetry_difference
)
from src.Derive_labels.compositional_hierarchical import (
    hierarchical_clustering_heights,
    composition_tree_depth,
    composition_tree_branching_factor,
    subgraph_isomorphism_frequencies,
    recursive_shape_patterns,
    multi_level_symmetry_chains,
    layered_edge_complexity,
    overlapping_substructure_ratios,
    composition_regularity_score,
    nested_convex_hull_levels
)




class ComprehensiveBongardProcessor:
    def __init__(self):
        # ...existing code...
        self.context_extractor = BongardFeatureExtractor()

    def _calculate_vertices_from_action(self, action, stroke_index, bongard_image=None):
        """Always use analytic vertices from the action parser for all strokes."""
        try:
            if hasattr(action, 'vertices_from_command'):
                verts = action.vertices_from_command()
                if verts:
                    return verts
            # Fallback to previous extraction if analytic not available
            from src.Derive_labels.stroke_types import _extract_stroke_vertices, _compute_bounding_box
            verts = _extract_stroke_vertices(action, stroke_index, None, bongard_image=bongard_image)
            if verts:
                bbox = _compute_bounding_box(verts)
                logger.info(f"[_calculate_vertices_from_action] Bounding box: {bbox}")
            return verts
        except Exception as e:
            logger.debug(f"Failed to calculate vertices from action: {e}")
        return []


    def _calculate_pattern_regularity_from_modifiers(self, modifier_sequence: list) -> float:
        """
        Pattern regularity using PhysicsInference.pattern_regularity. Returns NaN if sequence too short.
        """
        # Use corrected modifiers (horizontal/vertical/arc) for n-gram and regularity
        return PhysicsInference.pattern_regularity(modifier_sequence)
    
    
    
    """
    Enhanced comprehensive processor for Bongard-LOGO data that handles:
    - Multi-stroke image composition with stroke-type specific calculations
    - Differentiated geometry analysis for line vs arc strokes
    - Shape-modifier aware feature extraction
    - Comprehensive flagging logic for suspicious entries
    - Physics and geometry computation with validation
    """

    def __init__(self):
        # No longer use HybridAugmentor for parsing; use BongardImage.import_from_action_string_list
        logger.info("[INFO] BongardImage.import_from_action_string_list will be used for action program parsing.")
        self.flagged_cases = []
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'flagged': 0,
            'stroke_type_counts': {'line': 0, 'arc': 0, 'unknown': 0},
            'shape_modifier_counts': {}
        }
        

    def process_single_image(self, action_commands: List[str], image_id: str, 
                           is_positive: bool, problem_id: str, category: str,
                           image_path: str) -> Optional[Dict[str, Any]]:
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
        # --- Patch: Assign aggregated features to each shape's attributes property ---
        # Initialize geometry and posrot_labels before loop
        geometry = {}
        posrot_labels = {'centroid': [0.0, 0.0], 'orientation_degrees': 0.0}
        feature_extractor = BongardFeatureExtractor()
        for idx, shape in enumerate(one_stroke_shapes):
            # --- Correct assignment and population of image_dict ---
            image_dict = {}
            # Vertices
            image_dict['vertices'] = []
            if hasattr(shape, 'vertices') and isinstance(shape.vertices, list):
                safe_vertices = [tuple(v) for v in shape.vertices if v is not None and isinstance(v, (list, tuple)) and len(v) == 2]
                image_dict['vertices'] = safe_vertices
            else:
                logger.warning(f"[ATTR DEBUG] Shape {idx} has no vertices.")
            # Strokes as dicts with vertices
            image_dict['strokes'] = []
            if hasattr(shape, 'basic_actions'):
                for i, a in enumerate(shape.basic_actions):
                    command_str = str(a)
                    stroke_vertices = self._calculate_vertices_from_action(a, i, bongard_image=shape)
                    from src.Derive_labels.shape_utils import calculate_geometry_consistent, compute_open_stroke_geometry
                    from src.Derive_labels.stroke_types import _calculate_stroke_specific_features
                    stroke_features = _calculate_stroke_specific_features(a, i, bongard_image=shape, parent_shape_vertices=stroke_vertices)
                    analytic_verts = stroke_features.get('analytic_vertices', stroke_vertices)
                    if analytic_verts and len(analytic_verts) == 2:
                        stroke_geometry = compute_open_stroke_geometry(analytic_verts)
                        stroke_geometry['compactness'] = 0.0
                        stroke_geometry['convexity_ratio'] = 0.0
                        stroke_geometry['geom_complexity'] = min(len(analytic_verts)/10, 1)
                        stroke_geometry['degenerate_case'] = True
                        stroke_geometry['visual_complexity'] = min(max(stroke_geometry['perimeter']/max(stroke_geometry['perimeter'],1)*(1+stroke_features.get('robust_curvature',0)),0),1)
                    elif analytic_verts and len(analytic_verts) >= 3:
                        stroke_geometry = calculate_geometry_consistent(analytic_verts)
                        if 'arc' in command_str:
                            r = stroke_geometry.get('width',1.0)/2
                            theta = math.radians(90)
                            area = 0.5*r*r*(theta-math.sin(theta))
                            perim = r*theta
                            stroke_geometry['compactness'] = min(max(4*math.pi*area/(perim**2),0),1)
                            stroke_geometry['convexity_ratio'] = min(max(area/perim,0),1)
                        stroke_geometry['geom_complexity'] = min(len(analytic_verts)/10, 1)
                        stroke_geometry['degenerate_case'] = False
                        stroke_geometry['visual_complexity'] = min(max(stroke_geometry['perimeter']/max(stroke_geometry['perimeter'],1)*(1+stroke_features.get('robust_curvature',0)),0),1)
                    else:
                        stroke_geometry = {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0]}
                        stroke_geometry['compactness'] = 0.0
                        stroke_geometry['convexity_ratio'] = 0.0
                        stroke_geometry['geom_complexity'] = 0.0
                        stroke_geometry['degenerate_case'] = True
                        stroke_geometry['visual_complexity'] = 0.0
                    stroke_geometry['analytic_vertices'] = analytic_verts
                    stroke_geometry['open_stroke_perimeter'] = stroke_features.get('open_stroke_perimeter', stroke_geometry.get('perimeter', 0.0))
                    if hasattr(a, 'arc_radius') and hasattr(a, 'arc_angle'):
                        radius = a.arc_radius
                        span = a.arc_angle
                        stroke_geometry['arc_length'] = span/360*2*math.pi*radius
                        stroke_geometry['arc_curvature'] = 1/max(radius,1e-6)
                    if 'line' in command_str:
                        angle = stroke_features.get('line_angle',0)
                        if abs(angle) < 5:
                            stroke_geometry['shape_modifier'] = 'horizontal'
                        elif abs(angle-90) < 5 or abs(angle+90) < 5:
                            stroke_geometry['shape_modifier'] = 'vertical'
                        else:
                            stroke_geometry['shape_modifier'] = 'normal'
                    stroke_dict = {'command': command_str, 'vertices': analytic_verts, 'geometry': stroke_geometry}
                    # PATCH: Propagate support_set_context and discriminative_features (including stats)
                    stroke_dict['support_set_context'] = stroke_features.get('support_set_context', {})
                    stroke_dict['discriminative_features'] = stroke_features.get('discriminative_features', {})
                    # PATCH: Log when stats are attached to each stroke dict
                    if 'support_set_context' in stroke_dict and 'stats' in stroke_dict['support_set_context']:
                        logger.info(f"[logo_to_shape] Attached support_set_context['stats'] to stroke: {stroke_dict['support_set_context']['stats']}")
                    # Always tag every stroke dict with is_positive, label, and class_label
                    stroke_dict['label'] = category if category else 'unknown'
                    stroke_dict['class_label'] = problem_id if problem_id else 'unknown'
                    stroke_dict['is_positive'] = is_positive
                    logger.debug(f"[PATCH][is_positive] image_id={image_id}, stroke_idx={i}, assigned is_positive={is_positive}")
                    if not is_positive:
                        logger.info(f"[PATCH][NEGATIVE STROKE] image_id={image_id}, stroke_idx={i}, stroke_dict: {stroke_dict}")
                    if 'discriminative_features' in stroke_dict and 'stats' in stroke_dict['discriminative_features']:
                        logger.info(f"[logo_to_shape] Attached discriminative_features['stats'] to stroke: {stroke_dict['discriminative_features']['stats']}")
                    image_dict['strokes'].append(stroke_dict)
                logger.info(f"[ATTR DEBUG] Shape {idx} strokes (dicts): {image_dict['strokes']}")
            else:
                logger.warning(f"[ATTR DEBUG] Shape {idx} has no basic_actions.")
            # Attributes
            # Aggregate analytic attributes from strokes if available
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
            self._flag_case(
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
            image_features = self._calculate_image_features(norm_vertices_for_features, image_dict['strokes'], geometry if isinstance(geometry, dict) else {})
            # Patch: Extract contextual features for output record
            advanced_contextual = {k: image_features[k] for k in [
                'contrast_score', 'mutual_information', 'label_consistency', 'shape_cooccurrence',
                'category_consistency', 'class_prototype_distance', 'feature_importance_ranking',
                'cross_set_symmetry_difference', 'concept_drift_score'
            ] if k in image_features}
            centroid = geometry.get('centroid')
            width = geometry.get('width')
            height = geometry.get('height')
            if width is None:
                logger.error("Missing geometry['width']")
                width = 0.0
            if height is None:
                logger.error("Missing geometry['height']")
                height = 0.0
            composition_features = self._calculate_composition_features(all_actions)  # all_actions is now a list of strings
            physics_features = self._calculate_physics_features(norm_vertices_for_features, centroid=centroid, strokes=all_actions)

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
            try:
                # Use the function from relational_features.py
                # You may need to pass positive_images and negative_images if available
                # Here, we pass image_id for compatibility, but update as needed for your pipeline
                support_set_context = extract_support_set_context([image_id], [])
            except Exception as e:
                logger.warning(f"[PATCH] Failed to extract support set context for image_id={image_id}: {e}")

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
                'advanced_contextual_features': advanced_contextual,
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

            self.processing_stats['successful'] += 1
            return json_safe(complete_record)
        except Exception as e:
            import traceback
            error_msg = f"Error processing image {image_id}: {e}"
            stack_trace = traceback.format_exc()
            logger.error(f"{error_msg}\n{stack_trace}")
            self._flag_case('unknown', problem_id, error_msg, ['image_processing_error'])
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
    def _flag_case(self, category, problem_id, message, tags=None):
        """Log and store flagged cases for inspection."""
        logger.warning(f"[FLAG CASE] category={category}, problem_id={problem_id}, message={message}, tags={tags}")
        case = {
            'category': category,
            'problem_id': problem_id,
            'message': message,
            'tags': tags if tags else []
        }
        self.flagged_cases.append(case)


    
    def _calculate_image_features(self, vertices: List[tuple], strokes: List, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive image-level features with robust polygon recovery and improved metrics. Now includes standardized complexity. Outputs both raw and normalized values for area and perimeter."""
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
        pos_features, neg_features = [], []
        labels, shape_types, class_labels = [], [], []
        feature_matrix = []
        pos_symmetry, neg_symmetry = [], []
        graphs = []
        tree, root = {}, None
        symmetries = {}
        edges = []
        substructures = []
        parts = []
        position_func = None
        groups = []

        # Extract from strokes (dicts with geometry, label, type, etc.)
        for stroke in strokes:
            if isinstance(stroke, dict):
                # Feature vector: area, perimeter, width, height, visual_complexity, geom_complexity, symmetry_score
                fvec = [stroke.get('geometry', {}).get('area', 0.0),
                        stroke.get('geometry', {}).get('perimeter', 0.0),
                        stroke.get('geometry', {}).get('width', 0.0),
                        stroke.get('geometry', {}).get('height', 0.0),
                        stroke.get('geometry', {}).get('visual_complexity', 0.0),
                        stroke.get('geometry', {}).get('geom_complexity', 0.0),
                        stroke.get('geometry', {}).get('symmetry_score', 0.0)]
                feature_matrix.append(fvec)
                # Label, type, class_label
                if 'label' in stroke:
                    labels.append(stroke['label'])
                if 'geometry' in stroke and 'shape_modifier' in stroke['geometry']:
                    shape_types.append(stroke['geometry']['shape_modifier'])
                if 'class_label' in stroke:
                    class_labels.append(stroke['class_label'])
                # Symmetry scores for pos/neg
                if stroke.get('is_positive', None) is True:
                    pos_symmetry.append(stroke.get('geometry', {}).get('symmetry_score', 0.0))
                    pos_features.append(stroke.get('geometry', {}).get('area', 0.0))
                elif stroke.get('is_positive', None) is False:
                    neg_symmetry.append(stroke.get('geometry', {}).get('symmetry_score', 0.0))
                    neg_features.append(stroke.get('geometry', {}).get('area', 0.0))
                # Build graph from stroke relationships if available
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
                # Add to substructures, edges, etc.
                if 'geometry' in stroke and 'analytic_vertices' in stroke['geometry']:
                    substructures.append(stroke['geometry']['analytic_vertices'])
                    edges.extend([(stroke['geometry']['analytic_vertices'][i], stroke['geometry']['analytic_vertices'][i+1])
                                 for i in range(len(stroke['geometry']['analytic_vertices'])-1)])
                if 'geometry' in stroke and 'centroid' in stroke['geometry']:
                    parts.append(stroke['geometry']['centroid'])

        feature_matrix = np.array(feature_matrix) if feature_matrix else np.array([])

        # Build tree/root from shape relationships if available
        if len(parts) > 1:
            tree = {i: [i+1] for i in range(len(parts)-1)}
            root = 0
        # Symmetries: collect symmetry scores
        if pos_symmetry or neg_symmetry:
            symmetries = {'pos': pos_symmetry, 'neg': neg_symmetry}
        # Groups: cluster centroids if available
        if parts:
            groups = [parts]

        logger.info(f"[PATCH][ADVANCED INPUTS] pos_features: {pos_features}")
        logger.info(f"[PATCH][ADVANCED INPUTS] neg_features: {neg_features}")
        logger.info(f"[PATCH][ADVANCED INPUTS] labels: {labels}")
        logger.info(f"[PATCH][ADVANCED INPUTS] shape_types: {shape_types}")
        logger.info(f"[PATCH][ADVANCED INPUTS] class_labels: {class_labels}")
        logger.info(f"[PATCH][ADVANCED INPUTS] feature_matrix: {feature_matrix}")
        logger.info(f"[PATCH][ADVANCED INPUTS] pos_symmetry: {pos_symmetry}")
        logger.info(f"[PATCH][ADVANCED INPUTS] neg_symmetry: {neg_symmetry}")

        # ...existing code...
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
            # --- Contextual & Support-Set Features Integration ---
            # Now passing real, non-empty data to advanced feature functions
            # Defensive contextual feature calculation and logging
            if not pos_features or not neg_features:
                logger.warning("[CONTEXTUAL] pos_features or neg_features are empty, contrast_score will be 0.0.")
            features['contrast_score'] = positive_negative_contrast_score(pos_features, neg_features) if pos_features and neg_features else 0.0
            logger.info(f"[CONTEXTUAL] contrast_score: {features['contrast_score']}")

            if feature_matrix.size:
                features['mutual_information'] = support_set_mutual_information(feature_matrix.flatten())
            else:
                logger.warning("[CONTEXTUAL] feature_matrix is empty, mutual_information will be 0.0.")
                features['mutual_information'] = 0.0
            logger.info(f"[CONTEXTUAL] mutual_information: {features['mutual_information']}")

            if labels:
                features['label_consistency'] = label_consistency_ratio(labels)
            else:
                logger.warning("[CONTEXTUAL] labels are empty, label_consistency will be 0.0.")
                features['label_consistency'] = 0.0
            logger.info(f"[CONTEXTUAL] label_consistency: {features['label_consistency']}")

            if shape_types:
                features['shape_cooccurrence'] = support_set_shape_cooccurrence(shape_types).tolist()
                features['category_consistency'] = category_consistency_score(shape_types)
            else:
                logger.warning("[CONTEXTUAL] shape_types are empty, shape_cooccurrence and category_consistency will be default.")
                features['shape_cooccurrence'] = []
                features['category_consistency'] = 0.0
            logger.info(f"[CONTEXTUAL] shape_cooccurrence: {features['shape_cooccurrence']}")
            logger.info(f"[CONTEXTUAL] category_consistency: {features['category_consistency']}")

            if feature_matrix.size and class_labels:
                features['class_prototype_distance'] = class_prototype_distance(feature_matrix, class_labels)
            else:
                logger.warning("[CONTEXTUAL] feature_matrix or class_labels are empty, class_prototype_distance will be default.")
                features['class_prototype_distance'] = {}
            logger.info(f"[CONTEXTUAL] class_prototype_distance: {features['class_prototype_distance']}")

            if feature_matrix.size:
                features['feature_importance_ranking'] = feature_importance_ranking(feature_matrix)
            else:
                logger.warning("[CONTEXTUAL] feature_matrix is empty, feature_importance_ranking will be default.")
                features['feature_importance_ranking'] = []
            logger.info(f"[CONTEXTUAL] feature_importance_ranking: {features['feature_importance_ranking']}")

            if pos_symmetry and neg_symmetry:
                features['cross_set_symmetry_difference'] = cross_set_symmetry_difference(pos_symmetry, neg_symmetry)
            else:
                logger.warning("[CONTEXTUAL] pos_symmetry or neg_symmetry are empty, cross_set_symmetry_difference will be 0.0.")
                features['cross_set_symmetry_difference'] = 0.0
            logger.info(f"[CONTEXTUAL] cross_set_symmetry_difference: {features['cross_set_symmetry_difference']}")

            if feature_matrix.size:
                features['concept_drift_score'] = concept_drift_score(feature_matrix.flatten())
            else:
                logger.warning("[CONTEXTUAL] feature_matrix is empty, concept_drift_score will be 0.0.")
                features['concept_drift_score'] = 0.0
            logger.info(f"[CONTEXTUAL] concept_drift_score: {features['concept_drift_score']}")
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
    
    def _calculate_physics_features(self, vertices: List[tuple], centroid=None, strokes=None) -> Dict[str, Any]:
        """Calculate physics-based features using PhysicsInference. Accepts centroid override and strokes for correct counting. Uses correct center_of_mass and stroke counts."""
        logger = logging.getLogger(__name__)
        # --- PATCH: Robust input validation and logging ---
        logger.info(f"[_calculate_physics_features] INPUT vertices: {vertices}")
        logger.info(f"[_calculate_physics_features] INPUT centroid: {centroid}")
        logger.info(f"[_calculate_physics_features] INPUT strokes: {strokes}")
        expected_keys = [
            'moment_of_inertia', 'center_of_mass', 'polsby_popper_compactness',
            'num_straight_segments', 'num_arcs', 'has_quadrangle', 'has_obtuse_angle',
            'curvature_score', 'angular_variance', 'edge_length_variance'
        ]
        defaults = {
            'moment_of_inertia': 0.0,
            'center_of_mass': [0.0, 0.0],
            'polsby_popper_compactness': 0.0,
            'num_straight_segments': 0,
            'num_arcs': 0,
            'has_quadrangle': False,
            'has_obtuse_angle': False,
            'curvature_score': 0.0,
            'angular_variance': 0.0,
            'edge_length_variance': 0.0
        }
        # Validate vertices
        if not vertices or not isinstance(vertices, (list, tuple)) or len(vertices) < 3:
            logger.warning(f"[_calculate_physics_features] Invalid or insufficient vertices: {vertices}")
            return defaults.copy()
        # Validate each vertex
        for v in vertices:
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(coord, (int, float)) for coord in v)):
                logger.warning(f"[_calculate_physics_features] Malformed vertex: {v}")
                return defaults.copy()
        try:
            poly = None
            try:
                poly = PhysicsInference.polygon_from_vertices(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in polygon_from_vertices: {e}")
            # Use centroid from geometry if provided, else fallback to centroid of vertices
            if centroid is not None:
                center_of_mass = centroid
            elif poly is not None:
                try:
                    center_of_mass = PhysicsInference.centroid(poly)
                except Exception as e:
                    logger.error(f"[_calculate_physics_features] Error in centroid calculation: {e}")
                    center_of_mass = [0.0, 0.0]
            else:
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                center_of_mass = [sum(xs)/len(xs), sum(ys)/len(ys)] if xs and ys else [0.0, 0.0]
            # Count actual LineAction and ArcAction objects if strokes provided
            num_straight_segments = 0
            num_arcs = 0
            if strokes is not None:
                try:
                    import importlib.util
                    import sys
                    spec = importlib.util.spec_from_file_location("bongard_module", "Bongard-LOGO/bongard/bongard.py")
                    bongard_module = importlib.util.module_from_spec(spec)
                    sys.modules["bongard_module"] = bongard_module
                    spec.loader.exec_module(bongard_module)
                    LineAction = bongard_module.LineAction
                    ArcAction = bongard_module.ArcAction
                    for s in strokes:
                        if isinstance(s, LineAction):
                            num_straight_segments += 1
                        elif isinstance(s, ArcAction):
                            num_arcs += 1
                except Exception as e:
                    logger.error(f"[_calculate_physics_features] Error in stroke counting: {e}")
            else:
                # fallback to geometry-based
                try:
                    num_straight_segments = PhysicsInference.count_straight_segments(vertices)
                except Exception as e:
                    logger.error(f"[_calculate_physics_features] Error in count_straight_segments: {e}")
                    num_straight_segments = 0
                try:
                    num_arcs = PhysicsInference.count_arcs(vertices)
                except Exception as e:
                    logger.error(f"[_calculate_physics_features] Error in count_arcs: {e}")
                    num_arcs = 0

            # Calculate area and perimeter for polsby_popper_compactness
            try:
                area = None
                perimeter = None
                # Try to get area and perimeter from geometry calculation if available
                if poly is not None:
                    area = poly.area
                    perimeter = poly.length
                else:
                    # Fallback: estimate from vertices
                    try:
                        from shapely.geometry import Polygon
                        poly_tmp = Polygon(vertices)
                        area = poly_tmp.area
                        perimeter = poly_tmp.length
                    except Exception as e:
                        logger.error(f"[_calculate_physics_features] Error estimating area/perimeter: {e}")
                        area = 0.0
                        perimeter = 0.0
                polsby_popper = PhysicsInference.polsby_popper_compactness(area, perimeter)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in polsby_popper_compactness: {e}")
                polsby_popper = 0.0
            # brinkhoff_complexity removed from physics features
            logger.info(f"[_calculate_physics_features] polsby_popper: {polsby_popper}")
            features = {}
            try:
                features['moment_of_inertia'] = PhysicsInference.moment_of_inertia(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in moment_of_inertia: {e}")
                features['moment_of_inertia'] = 0.0
            features['center_of_mass'] = center_of_mass
            features['polsby_popper_compactness'] = polsby_popper
            features['num_straight_segments'] = num_straight_segments
            features['num_arcs'] = num_arcs
            try:
                features['has_quadrangle'] = PhysicsInference.has_quadrangle(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in has_quadrangle: {e}")
                features['has_quadrangle'] = False
            try:
                features['has_obtuse_angle'] = PhysicsInference.has_obtuse(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in has_obtuse: {e}")
                features['has_obtuse_angle'] = False
            try:
                features['curvature_score'] = _calculate_curvature_score(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in curvature_score: {e}")
                features['curvature_score'] = 0.0
            try:
                features['angular_variance'] = _calculate_angular_variance(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in angular_variance: {e}")
                features['angular_variance'] = 0.0
            try:
                features['edge_length_variance'] = _calculate_edge_length_variance(vertices)
            except Exception as e:
                logger.error(f"[_calculate_physics_features] Error in edge_length_variance: {e}")
                features['edge_length_variance'] = 0.0
            # Defensive: ensure all expected keys are present
            for k in expected_keys:
                if k not in features:
                    logger.warning(f"[_calculate_physics_features] Missing key '{k}', setting default value.")
                    features[k] = defaults[k]
            logger.info(f"[_calculate_physics_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logger.error(f"[_calculate_physics_features] Error: {e}, returning all defaults.")
            return defaults.copy()

    
    def _calculate_composition_features(self, action_commands: List[str]) -> Dict[str, Any]:
        """
        Calculate features about stroke composition and relationships.
        Expects a list of action command strings (not objects).
        Defensive: always convert to strings before any operation.
        """
        logger = logging.getLogger(__name__)
        strokes = ensure_all_strings(action_commands)
        logger.debug(f"[_calculate_composition_features] INPUTS: strokes count={len(strokes) if strokes else 0}")
        if not strokes:
            logger.debug("[_calculate_composition_features] No strokes, returning empty dict")
            return {}
        try:
            stroke_types = {}
            shape_modifiers = {}
            modifier_sequence = []
            for stroke in strokes:
                # Defensive: parse type and modifier from string
                if isinstance(stroke, str):
                    parts = stroke.split('_')
                    stroke_type = parts[0] if parts else 'unknown'
                    modifier = parts[1] if len(parts) > 1 else 'normal'
                else:
                    stroke_type = type(stroke).__name__.replace('Action', '').lower()
                    modifier = _extract_modifier_from_stroke(stroke)
                stroke_types[stroke_type] = stroke_types.get(stroke_type, 0) + 1
                shape_modifiers[modifier] = shape_modifiers.get(modifier, 0) + 1
                modifier_sequence.append(modifier)
            # Use json.dumps for serialization of distributions
            features = {
                'stroke_type_distribution': stroke_types,
                'shape_modifier_distribution': shape_modifiers,
                'stroke_diversity': len(stroke_types),
                'shape_diversity': len(shape_modifiers),
                'dominant_stroke_type': max(stroke_types.items(), key=lambda x: x[1])[0] if stroke_types else 'unknown',
                'dominant_shape_modifier': max(shape_modifiers.items(), key=lambda x: x[1])[0] if shape_modifiers else 'unknown'
            }
            features.update({
                'composition_complexity': len(strokes) + len(shape_modifiers),
                'homogeneity_score': _calculate_homogeneity(shape_modifiers),
                'pattern_regularity': self._calculate_pattern_regularity_from_modifiers(modifier_sequence)
            })
            logger.debug(f"[_calculate_composition_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logger.warning(f"[_calculate_composition_features] Error: {e}")
            return {}





def main():
    def fully_stringify(obj):
        if isinstance(obj, dict):
            return {fully_stringify(k): fully_stringify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fully_stringify(x) for x in obj]
        elif hasattr(obj, 'raw_command'):
            return str(obj.raw_command)
        elif type(obj).__name__ in ['LineAction', 'ArcAction']:
            return str(getattr(obj, 'raw_command', str(obj)))
        else:
            return str(obj)
    parser = argparse.ArgumentParser(description='Generate comprehensive derived labels for Bongard-LOGO dataset')
    parser.add_argument('--input-dir', required=True, help='Input directory containing Bongard-LOGO data')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--problems-list', default=None, help='Optional file containing list of problems to process')
    parser.add_argument('--n-select', type=int, default=50, help='Number of problems to select if no problems-list')
    args = parser.parse_args()

    # Initialize processor
    processor = ComprehensiveBongardProcessor()

    # Load data using the same logic as hybrid.py
    try:
        problems_data = load_action_programs(args.input_dir)

        # Filter by problems list if provided
        if args.problems_list and os.path.exists(args.problems_list):
            with open(args.problems_list, 'r') as f:
                selected_problems = [line.strip() for line in f if line.strip()]
            problems_data = {k: v for k, v in problems_data.items() if k in selected_problems}

        # Limit number if n_select is specified
        if args.n_select and len(problems_data) > args.n_select:
            problems_data = dict(list(problems_data.items())[:args.n_select])

        logger.info(f"Loaded {len(problems_data)} problems from {args.input_dir}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # --- DEBUG: Extract and log all unique action type prefixes in the dataset ---
    action_type_prefixes = extract_action_type_prefixes(problems_data)
    logger.info(f"[DEBUG] Unique action type prefixes in dataset: {sorted(action_type_prefixes)}")
    # --- END DEBUG ---

    # Process all images

    all_results = []
    total_images = 0
    successful_images = 0
    problem_summaries = []

    # Defensive validation and logging for problem_data structure
    logger.info("[DEFENSIVE CHECK] Validating problem_data structure for all problems...")
    malformed_problems = []
    for pid, pdata in problems_data.items():
        if not (isinstance(pdata, list) and len(pdata) == 2):
            logger.warning(f"[DEFENSIVE CHECK] Problem {pid} has malformed data: type={type(pdata)}, value={pdata}")
            malformed_problems.append(pid)
    if malformed_problems:
        logger.error(f"[DEFENSIVE CHECK] Found malformed problems: {malformed_problems}")
    else:
        logger.info("[DEFENSIVE CHECK] All problems have valid [positive_examples, negative_examples] structure.")

    for problem_id, problem_data in problems_data.items():
        try:
            # Determine category from problem_id
            if problem_id.startswith('bd_'):
                category = 'bd'
            elif problem_id.startswith('ff_'):
                category = 'ff'
            elif problem_id.startswith('hd_'):
                category = 'hd'
            else:
                category = 'unknown'

            # Always initialize these variables for every problem
            pos_results, neg_results = [], []
            problem_unique_shape_functions = set()
            problem_shape_function_counts = {}
            problem_modifiers = set()
            num_images_in_problem = 0

            if isinstance(problem_data, list) and len(problem_data) == 2:
                positive_examples, negative_examples = problem_data
                # Process all images, collect both sets
                logger.info(f"[DEBUG][logo_to_shape] Problem {problem_id}: Processing {len(positive_examples)} positive, {len(negative_examples)} negative images.")
                for i, img in enumerate(positive_examples):
                    logger.debug(f"[DEBUG][logo_to_shape] Problem {problem_id}: Positive image {i}: {img}")
                for i, img in enumerate(negative_examples):
                    logger.debug(f"[DEBUG][logo_to_shape] Problem {problem_id}: Negative image {i}: {img}")
                for i, action_commands in enumerate(positive_examples):
                    total_images += 1
                    num_images_in_problem += 1
                    image_id = f"{problem_id}_pos_{i}"
                    image_path = f"images/{problem_id}/category_1/{i}.png"
                    # Tag image as positive before feature extraction
                    result = processor.process_single_image(
                        action_commands, image_id, True, problem_id, category, image_path
                    )
                    if result:
                        result['is_positive'] = True
                        pos_results.append(result)
                        all_results.append(result)
                        successful_images += 1
                        summary = result.get('image_canonical_summary', {})
                        for fn in summary.get('unique_shape_functions', []):
                            problem_unique_shape_functions.add(fn)
                        for fn, count in summary.get('shape_function_counts', {}).items():
                            problem_shape_function_counts[fn] = problem_shape_function_counts.get(fn, 0) + count
                        for mod in summary.get('modifiers', []):
                            problem_modifiers.add(mod)
                for i, action_commands in enumerate(negative_examples):
                    total_images += 1
                    num_images_in_problem += 1
                    image_id = f"{problem_id}_neg_{i}"
                    image_path = f"images/{problem_id}/category_0/{i}.png"
                    # Tag image as negative before feature extraction
                    logger.debug(f"[DEBUG][logo_to_shape] Problem {problem_id}: Extracting features for negative image {i} (id={image_id})")
                    result = processor.process_single_image(
                        action_commands, image_id, False, problem_id, category, image_path
                    )
                    if result:
                        result['is_positive'] = False
                        neg_results.append(result)
                        all_results.append(result)
                        successful_images += 1
                        summary = result.get('image_canonical_summary', {})
                        for fn in summary.get('unique_shape_functions', []):
                            problem_unique_shape_functions.add(fn)
                        for fn, count in summary.get('shape_function_counts', {}).items():
                            problem_shape_function_counts[fn] = problem_shape_function_counts.get(fn, 0) + count
                        for mod in summary.get('modifiers', []):
                            problem_modifiers.add(mod)
            else:
                logger.warning(f"Problem {problem_id} has unexpected data format, skipping.")
                continue

            # --- Verification: Ensure 7-7 split ---
            assert len(pos_results) == 7 and len(neg_results) == 7, \
                f"Expected 7 positives/7 negatives, got {len(pos_results)}/{len(neg_results)} for problem {problem_id}"

            # --- Compute and add support-set context features ---
            logger.info(f"[DEBUG][CONTEXTUAL] pos_results count: {len(pos_results)}")
            logger.info(f"[DEBUG][CONTEXTUAL] neg_results count: {len(neg_results)}")
            if pos_results:
                logger.info(f"[DEBUG][CONTEXTUAL] Sample pos_results[0]: {json.dumps(pos_results[0], indent=2, ensure_ascii=False)}")
            if neg_results:
                logger.info(f"[DEBUG][CONTEXTUAL] Sample neg_results[0]: {json.dumps(neg_results[0], indent=2, ensure_ascii=False)}")
            if not all(isinstance(r, dict) for r in pos_results):
                logger.error(f"[ERROR][CONTEXTUAL] pos_results contains non-dict entries!")
            if not all(isinstance(r, dict) for r in neg_results):
                logger.error(f"[ERROR][CONTEXTUAL] neg_results contains non-dict entries!")
            # Compute statistical support set context using context_features.py
            from src.Derive_labels.context_features import BongardFeatureExtractor
            logger.info(f"[SUPPORT-SET CONTEXT] Calling BongardFeatureExtractor.extract_support_set_context for problem {problem_id}")
            bfe = BongardFeatureExtractor()
            support_set_context = bfe.extract_support_set_context(pos_results, neg_results)

            # --- Compute and add support-set context features ---
            # Defensive logging for contextual feature extraction
            logger.info(f"[DEBUG][CONTEXTUAL] pos_results count: {len(pos_results)}")
            logger.info(f"[DEBUG][CONTEXTUAL] neg_results count: {len(neg_results)}")
            if pos_results:
                logger.info(f"[DEBUG][CONTEXTUAL] Sample pos_results[0]: {json.dumps(pos_results[0], indent=2, ensure_ascii=False)}")
            if neg_results:
                logger.info(f"[DEBUG][CONTEXTUAL] Sample neg_results[0]: {json.dumps(neg_results[0], indent=2, ensure_ascii=False)}")
            # Validate that results are image-level dicts
            if not all(isinstance(r, dict) for r in pos_results):
                logger.error(f"[ERROR][CONTEXTUAL] pos_results contains non-dict entries!")
            if not all(isinstance(r, dict) for r in neg_results):
                logger.error(f"[ERROR][CONTEXTUAL] neg_results contains non-dict entries!")
            # Compute statistical support set context using context_features.py
            from src.Derive_labels.context_features import BongardFeatureExtractor
            logger.info(f"[SUPPORT-SET CONTEXT] Calling BongardFeatureExtractor.extract_support_set_context for problem {problem_id}")
            bfe = BongardFeatureExtractor()
            support_set_context = bfe.extract_support_set_context(pos_results, neg_results)
            logger.info(f"[SUPPORT-SET CONTEXT] OUTPUT for problem {problem_id}: {json.dumps(support_set_context, indent=2, ensure_ascii=False)}")
            # Add support_set_context and discriminative features to each image result
            for r in pos_results + neg_results:
                    # Always attach image-level support_set_context and discriminative_features
                    r['support_set_context_image'] = support_set_context if isinstance(support_set_context, dict) else {}
                    if 'discriminative' in support_set_context:
                        r['discriminative_features_image'] = support_set_context['discriminative']
                    # If support set is missing, attach a meaningful reason
                    if not support_set_context or not support_set_context.get('positive_stats', {}).get('valid', False):
                        r['support_set_context_image']['valid'] = False
                        r['support_set_context_image']['reason'] = support_set_context.get('positive_stats', {}).get('reason', 'missing_support_set')
                    if not support_set_context or not support_set_context.get('discriminative', {}).get('valid', False):
                        r['discriminative_features_image'] = {
                            'valid': False,
                            'reason': support_set_context.get('discriminative', {}).get('reason', 'missing_discriminative_set'),
                            'stats': {}
                        }
                # --- Advanced contextual features aggregation and attachment ---
                # Import contextual feature functions


                # Example: aggregate a few key features (adjust keys as needed)
            def safe_get(results, key, default=0.0):
                return [r.get(key, default) for r in results]

            # You may need to adjust these keys to match your result dicts
            pos_contrast_feats = safe_get(pos_results, 'contrast_score')
            neg_contrast_feats = safe_get(neg_results, 'contrast_score')
            pos_mi_feats = safe_get(pos_results, 'mutual_information')
            neg_mi_feats = safe_get(neg_results, 'mutual_information')
            pos_labels = safe_get(pos_results, 'class_label', '')
            neg_labels = safe_get(neg_results, 'class_label', '')
            pos_shape_types = safe_get(pos_results, 'shape_type', '')
            neg_shape_types = safe_get(neg_results, 'shape_type', '')
            pos_symmetry = safe_get(pos_results, 'symmetry_score')
            neg_symmetry = safe_get(neg_results, 'symmetry_score')

            # Feature matrix for ranking (example: use 'feature_vector' key if present)
            pos_feature_matrix = [r.get('feature_vector', []) for r in pos_results if 'feature_vector' in r]
            neg_feature_matrix = [r.get('feature_vector', []) for r in neg_results if 'feature_vector' in r]
            all_feature_matrix = pos_feature_matrix + neg_feature_matrix

            # Compute advanced contextual features
            contextual_features = {
                'contrast_score': positive_negative_contrast_score(pos_contrast_feats, neg_contrast_feats),
                'mutual_information': support_set_mutual_information(pos_contrast_feats + neg_contrast_feats),
                'label_consistency': label_consistency_ratio(pos_labels + neg_labels),
                'concept_drift_score': concept_drift_score(pos_contrast_feats + neg_contrast_feats),
                'shape_cooccurrence': support_set_shape_cooccurrence(pos_shape_types + neg_shape_types).tolist(),
                'category_consistency': category_consistency_score(pos_shape_types + neg_shape_types),
                'class_prototype_distance': class_prototype_distance(
                    np.array(all_feature_matrix) if all_feature_matrix else np.zeros((0,)),
                    pos_labels + neg_labels
                ),
                'feature_importance_ranking': feature_importance_ranking(
                    np.array(all_feature_matrix) if all_feature_matrix else np.zeros((0,))
                ),
                'cross_set_symmetry_difference': cross_set_symmetry_difference(pos_symmetry, neg_symmetry)
            }

            # Attach contextual features to each image result
            for r in pos_results + neg_results:
                r['contextual_features_problem_level'] = contextual_features

            # Save problem-level canonical summary
            problem_summary = {
                'problem_id': problem_id,
                'unique_shape_functions': sorted(list(problem_unique_shape_functions)),
                'shape_function_counts': problem_shape_function_counts,
                'modifiers': sorted(list(problem_modifiers)),
                'num_images': num_images_in_problem
            }
            logger.info(f"[PROBLEM SUMMARY] Problem: {problem_id}\n{json.dumps(problem_summary, indent=2, ensure_ascii=False)}")
            problem_summaries.append(problem_summary)

        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {e}")
            processor._flag_case('unknown', problem_id, f'Problem processing failed: {e}', ['problem_processing_error'])


    # Save results
    try:
        from src.Derive_labels.shape_utils import json_safe
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure all_results is json_safe
        safe_results = json_safe(all_results)
        # Defensive output patch: ensure all action lists are flattened and stringified before output

        # Before saving all_results

        def robust_action_list_to_str(lst):
            # Converts a list of actions (possibly nested) to a list of strings using raw_command
            if isinstance(lst, list):
                # Defensive: ensure all items are strings
                return [robust_action_list_to_str(x) for x in lst] if lst and isinstance(lst[0], list) else [getattr(x, 'raw_command', str(x)) if type(x).__name__ in ['LineAction', 'ArcAction'] else str(x) for x in lst]
            elif type(lst).__name__ in ['LineAction', 'ArcAction']:
                return str(getattr(lst, 'raw_command', str(lst)))
            else:
                return str(lst)



        # Helper: recursively sanitize None values in dicts/lists
        def sanitize_none(obj, path="root"):
            if isinstance(obj, dict):
                return {k: sanitize_none(v, f"{path}.{k}") for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_none(x, f"{path}[{i}]") for i, x in enumerate(obj)]
            elif obj is None:
                logger.warning(f"[SERIALIZE PATCH] None value found at {path}, replacing with safe default.")
                return 0
            else:
                return obj

        # Defensive patch: ensure all results, flagged cases, stats, and summaries are robustly stringified and sanitized before serialization
        safe_results = ensure_all_strings(all_results)
        safe_results = sanitize_none(safe_results)
        logger.info(f"[SERIALIZE DEBUG][main] Final processed_results before writing: {safe_results}")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(safe_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[SERIALIZE DEBUG][main] Exception during output serialization: {e}")
            raise

        logger.info(f"Successfully processed {successful_images}/{total_images} images")
        logger.info(f"Saved {len(all_results)} records to {args.output}")

        # Save flagged cases
        if processor.flagged_cases:
            flagged_path = os.path.join(output_dir, 'flagged_cases.json')
            safe_flagged = ensure_all_strings(processor.flagged_cases)
            safe_flagged = sanitize_none(safe_flagged)
            logger.info(f"[SERIALIZE DEBUG][main][flagged_cases] Final processed_flagged before writing: {safe_flagged}")
            try:
                with open(flagged_path, 'w', encoding='utf-8') as f:
                    json.dump(safe_flagged, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"[SERIALIZE DEBUG][main][flagged_cases] Exception during flagged case serialization: {e}")
                raise

        # Save processing statistics
        processor.processing_stats['processing_summary'] = {
            'success_rate': processor.processing_stats['successful'] / max(processor.processing_stats['total_processed'], 1),
            'flag_rate': processor.processing_stats['flagged'] / max(processor.processing_stats['total_processed'], 1),
            'total_features_calculated': len(all_results) * 4 if all_results else 0  # 4 feature sets per record
        }

        stats_path = os.path.join(output_dir, 'processing_statistics.json')
        safe_stats = ensure_all_strings(processor.processing_stats)
        safe_stats = sanitize_none(safe_stats)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(safe_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processing statistics to {stats_path}")

        # Save problem-level canonical summaries
        problem_summary_path = os.path.join(output_dir, 'problem_summaries.json')
        safe_summaries = ensure_all_strings(problem_summaries)
        safe_summaries = sanitize_none(safe_summaries)
        with open(problem_summary_path, 'w', encoding='utf-8') as f:
            json.dump(safe_summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(problem_summaries)} problem-level canonical summaries to {problem_summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1


if __name__ == '__main__':
    exit(main())