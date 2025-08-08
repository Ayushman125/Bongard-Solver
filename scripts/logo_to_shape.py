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
from src.Derive_labels.features import ensure_str_list
# Fix BongardImage import if needed
try:
    from bongard.bongard import BongardImage
except ImportError:
    try:
        from Bongard_LOGO.bongard.bongard import BongardImage
    except ImportError:
        from Bongard_LOGO.bongard.bongard import BongardImage
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




class ComprehensiveBongardProcessor:
    def __init__(self):
        # ...existing code...
        self.context_extractor = BongardFeatureExtractor()

    def _calculate_vertices_from_action(self, action, stroke_index, bongard_image=None):
        from src.Derive_labels.stroke_types import _extract_stroke_vertices, _compute_bounding_box
        """Calculate line segment vertices for a single action, using bongard_image context."""
        try:
            verts = _extract_stroke_vertices(action, stroke_index, None, bongard_image=bongard_image)
            bbox = None
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
        return PhysicsInference.pattern_regularity(modifier_sequence)
        diversity_penalty = (len(unique_mods) - 1) / max(n-1, 1)
        pattern_score = max(repetition_score, alternation_score)
        diversity_factor = 1.0 - diversity_penalty
        pattern_regularity = pattern_score * diversity_factor
        pattern_regularity = max(0.0, min(1.0, pattern_regularity))
        logger.debug(f"Pattern regularity: repetition_score={repetition_score}, alternation_score={alternation_score}, diversity_penalty={diversity_penalty}, result={pattern_regularity}")
        return pattern_regularity
    
    
    
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
        for idx, shape_cmds in enumerate(shapes_commands):
            logger.info(f"[PARSER] Parsing shape {idx} commands: {shape_cmds}")
            parsed_shape = parser.parse_action_commands(shape_cmds, problem_id)
            if isinstance(parsed_shape, list):
                one_stroke_shapes.extend(parsed_shape)
            else:
                one_stroke_shapes.append(parsed_shape)
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
        feature_extractor = BongardFeatureExtractor()

        for idx, shape in enumerate(one_stroke_shapes):
            image_dict = {}
            # Log shape object before feature extraction
            logger.info(f"[ATTR DEBUG] Shape {idx} raw object: {shape}")
            if hasattr(shape, 'vertices'):
                logger.info(f"[ATTR DEBUG] Shape {idx} vertices: {shape.vertices}")
                image_dict['vertices'] = shape.vertices
            else:
                logger.warning(f"[ATTR DEBUG] Shape {idx} has no vertices.")
            if hasattr(shape, 'basic_actions'):
                # Defensive: ensure all basic_actions are strings for logging
                safe_basic_actions = ensure_str_list(shape.basic_actions)
                logger.info(f"[ATTR DEBUG] Shape {idx} basic_actions: {safe_basic_actions}")
                image_dict['strokes'] = [vars(a) for a in shape.basic_actions]
            else:
                logger.warning(f"[ATTR DEBUG] Shape {idx} has no basic_actions.")
            # Check for degenerate shapes
            if not image_dict.get('vertices') or not image_dict.get('strokes'):
                logger.warning(f"[ATTR DEBUG] Shape {idx} is degenerate (missing vertices or strokes). Skipping feature extraction.")
                shape.attributes = {}
                continue
            # Log input to feature extraction
            logger.info(f"[ATTR DEBUG] Shape {idx} image_dict for feature extraction: {image_dict}")
            try:
                shape.attributes = feature_extractor.extract_image_features(image_dict)
                logger.info(f"[ATTR DEBUG] Shape {idx} extracted attributes: {shape.attributes}")
            except Exception as e:
                logger.warning(f"[ATTR ASSIGN] Failed to extract features for shape {idx}: {e}")
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
            # Determine if input is multi-object (list of lists) or single-object (flat list)
            is_multi_object = isinstance(action_commands, list) and action_commands and all(isinstance(cmd, list) for cmd in action_commands)
            actual_shapes = len(one_stroke_shapes)
            if is_multi_object:
                expected_shapes = len(shapes_commands)
                # Accept if number of parsed shapes matches number of sublists
                if actual_shapes != expected_shapes or actual_shapes == 0:
                    logger.error(f"[DATA ISSUE] image_id={image_id}, problem_id={problem_id}: Number of sublists ({expected_shapes}) does not match number of parsed shapes ({actual_shapes}).\nAction commands: {shapes_commands}\nParsed shapes: {one_stroke_shapes}")
                    self._flag_case('data_issue', problem_id, f"Number of sublists ({expected_shapes}) does not match number of parsed shapes ({actual_shapes})", ['data_issue'])
                    return None
            else:
                # Single-object: accept if exactly one shape is parsed
                if actual_shapes != 1:
                    logger.error(f"[DATA ISSUE] image_id={image_id}, problem_id={problem_id}: Expected 1 parsed shape for single-object, got {actual_shapes}.\nAction commands: {shapes_commands}\nParsed shapes: {one_stroke_shapes}")
                    self._flag_case('data_issue', problem_id, f"Expected 1 parsed shape for single-object, got {actual_shapes}", ['data_issue'])
                    return None
            # Now construct BongardImage and run attribute mapping
            bongard_image = BongardImage(one_stroke_shapes)
            logger.info(f"[ATTR MAP] Mapping attributes for multi-object image: {image_id}")
            for idx, shape in enumerate(getattr(bongard_image, 'one_stroke_shapes', [])):
                attrs = getattr(shape, 'attributes', None)
                # Defensive: ensure attributes dict is json-safe
                safe_attrs = json_safe(attrs)
                logger.info(f"[ATTR MAP] Shape {idx}: attributes={safe_attrs}")
                if attrs is None:
                    logger.warning(f"[ATTR MAP] Shape {idx} has no attributes.")
                elif isinstance(attrs, dict):
                    for k, v in safe_attrs.items():
                        logger.info(f"[ATTR MAP] Shape {idx} attribute: {k}={v}")
                else:
                    logger.info(f"[ATTR MAP] Shape {idx} attributes (non-dict): {safe_attrs}")
            bongard_image = BongardImage(one_stroke_shapes)
            # --- Attribute mapping for multi-object images ---
            logger.info(f"[ATTR MAP] Mapping attributes for multi-object image: {image_id}")
            for idx, shape in enumerate(getattr(bongard_image, 'one_stroke_shapes', [])):
                attrs = getattr(shape, 'attributes', None)
                logger.info(f"[ATTR MAP] Shape {idx}: attributes={attrs}")
                if attrs is None:
                    logger.warning(f"[ATTR MAP] Shape {idx} has no attributes.")
                elif isinstance(attrs, dict):
                    for k, v in attrs.items():
                        logger.info(f"[ATTR MAP] Shape {idx} attribute: {k}={v}")
                else:
                    logger.info(f"[ATTR MAP] Shape {idx} attributes (non-dict): {attrs}")

            # --- Always include raw vertices from BongardImage ---
            vertices_raw = []
            for shape in getattr(bongard_image, 'one_stroke_shapes', []):
                if hasattr(shape, 'vertices'):
                    # Defensive: ensure vertices are json-safe
                    safe_vertices = json_safe(shape.vertices)
                    vertices_raw.extend(safe_vertices)

            # --- Use standardize_coordinates for normalization ---
            from src.Derive_labels.shape_utils import standardize_coordinates, calculate_geometry_consistent
            logger.info(f"[standardize_coordinates] INPUT: {vertices_raw}")
            normalized_vertices = standardize_coordinates(vertices_raw)
            logger.info(f"[standardize_coordinates] OUTPUT: {normalized_vertices}")

            # --- Use calculate_geometry_consistent for geometry ---
            logger.info(f"[calculate_geometry_consistent] INPUT: {normalized_vertices}")
            try:
                geometry = calculate_geometry_consistent(normalized_vertices)
                logger.info(f"[calculate_geometry_consistent] OUTPUT: {geometry}")
            except Exception as geo_exc:
                logger.error(f"[FALLBACK LOGIC] image_id={image_id}, problem_id={problem_id}: Geometry calculation failed, fallback logic triggered. Error: {geo_exc}\nVertices: {normalized_vertices}")
                geometry = {}

            # For downstream compatibility, use normalized_vertices for features
            norm_vertices_for_features = normalized_vertices

            # --- Derive position and rotation labels from normalized vertices ---
            posrot_labels = extract_position_and_rotation(norm_vertices_for_features)

            # --- Contextual/relational feature extraction ---
            # Prepare strokes as dicts with 'vertices' for context extractor
            context_strokes = []
            for shape in getattr(bongard_image, 'one_stroke_shapes', []):
                v = getattr(shape, 'vertices', None)
                context_strokes.append({'vertices': v if v is not None else []})
            from src.Derive_labels.features import extract_relational_features
            context_relationships = extract_relational_features(context_strokes) if context_strokes else {}
            logger.info(f"[process_single_image] context_relationships: {context_relationships}")
            # Multi-scale features (using normalized vertices)
            multiscale_features = extract_multiscale_features(norm_vertices_for_features) if norm_vertices_for_features else {}
            logger.info(f"[process_single_image] multiscale_features: {multiscale_features}")

            # --- Calculate image features using robust polygon ---
            all_actions = []
            for shape in getattr(bongard_image, 'one_stroke_shapes', []):
                # Defensive: ensure all actions are strings for downstream use
                safe_actions = ensure_str_list(getattr(shape, 'basic_actions', []))
                all_actions.extend(safe_actions)
            image_features = self._calculate_image_features(norm_vertices_for_features, all_actions, geometry)
            centroid = geometry.get('centroid')
            composition_features = self._calculate_composition_features(all_actions)
            physics_features = self._calculate_physics_features(norm_vertices_for_features, centroid=centroid, strokes=all_actions)

            # --- Relational/Topological/Sequential Features ---
            # Convert actions to shapely geometries for robust relational features
            from src.Derive_labels.features import _actions_to_geometries
            stroke_geometries = _actions_to_geometries(bongard_image)
            logger.debug(f"Number of stroke geometries: {len(stroke_geometries)}")
            for idx, g in enumerate(stroke_geometries):
                logger.debug(f"Geometry {idx}: type={g.geom_type}, is_valid={g.is_valid}")
            from src.Derive_labels.features import extract_relational_features as extract_relational_features_geom
            robust_relational_features = extract_relational_features_geom(stroke_geometries) if stroke_geometries else {}
            intersections = context_relationships.get('intersections')
            adjacency = context_relationships.get('adjacency')
            containment = context_relationships.get('containment')
            overlap = context_relationships.get('overlap')

            # Sequential pattern features (n-gram, alternation, regularity)
            modifier_sequence = [_extract_modifier_from_stroke(s) for s in all_actions]
            ngram_features = _extract_ngram_features(modifier_sequence)
            alternation = _detect_alternation(modifier_sequence)
            regularity = self._calculate_pattern_regularity_from_modifiers(modifier_sequence)

            # Topological features (chain/star/cycle detection, connectivity)
            context_adj = context_relationships.get('adjacency_matrix')
            if context_adj is not None:
                graph_features = _extract_graph_features({'adjacency_matrix': context_adj})
                logger.info(f"[logo_to_shape] Graph topology INPUT adjacency_matrix: {context_adj}")
                logger.info(f"[logo_to_shape] Graph topology OUTPUT: {graph_features}")
            else:
                graph_features = {'type': 'none', 'connectivity': 0}
                logger.warning(f"[logo_to_shape] No adjacency matrix for graph topology detection.")

            # --- Aggregate line and arc features for stroke_type_features ---
            line_features = []
            arc_features = []
            stroke_features = []
            shape_attr_map = FileIO.get_shape_attribute_map()
            shape_def_map = FileIO.get_shape_def_map()
            unique_shape_functions = set()
            shape_function_counts = {}
            unique_modifiers = set()
            original_action_commands = [cmd for sublist in shapes_commands for cmd in sublist]

            # --- NEW: Extract canonical problem name for TSV lookup (robust normalization) ---
            canonical_name = problem_id
            if '_' in canonical_name:
                if canonical_name.startswith('bd_') or canonical_name.startswith('ff_') or canonical_name.startswith('hd_'):
                    canonical_name = canonical_name.split('_', 1)[1]
            import re
            canonical_base = re.sub(r'(_\d+)?$', '', canonical_name)
            canonical_key = canonical_base.lower().replace('-', '_').replace(' ', '_').rstrip('_')
            canonical_shape_attributes = shape_attr_map.get(canonical_key)
            canonical_shape_def = shape_def_map.get(canonical_key)

            for i, shape in enumerate(getattr(bongard_image, 'one_stroke_shapes', [])):
                # Defensive: ensure all actions are strings for logging/serialization
                safe_actions = ensure_str_list(getattr(shape, 'basic_actions', []))
                for j, action in enumerate(safe_actions):
                    stroke_type_val = type(action).__name__.replace('Action', '').lower() if not isinstance(action, str) else 'unknown'
                    raw_command = getattr(action, 'raw_command', None) if not isinstance(action, str) else action
                    function_name = getattr(action, 'function_name', None) if not isinstance(action, str) else None
                    if not raw_command and original_action_commands and j < len(original_action_commands):
                        if isinstance(original_action_commands[j], str):
                            raw_command = original_action_commands[j]
                    shape_modifier_val = None
                    parameters = {}
                    if raw_command and isinstance(raw_command, str):
                        parts = raw_command.split('_')
                        if len(parts) >= 2:
                            shape_modifier_val = parts[1]
                        param_str = '_'.join(parts[2:]) if len(parts) > 2 else ''
                        if param_str:
                            main_params = param_str.split('-')
                            for idx, p in enumerate(main_params):
                                try:
                                    parameters[f'param{idx+1}'] = float(p)
                                except Exception:
                                    parameters[f'param{idx+1}'] = p
                    if not shape_modifier_val and function_name and isinstance(function_name, str):
                        fn_parts = function_name.split('_')
                        if len(fn_parts) >= 2:
                            shape_modifier_val = fn_parts[1]
                    if not function_name and raw_command and isinstance(raw_command, str):
                        parts = raw_command.split('_')
                        if len(parts) >= 2:
                            function_name = f"{parts[0]}_{parts[1]}"
                    if not shape_modifier_val and not isinstance(action, str) and hasattr(action, 'shape_modifier'):
                        smod = getattr(action, 'shape_modifier')
                        if hasattr(smod, 'value'):
                            shape_modifier_val = smod.value
                        elif isinstance(smod, str):
                            shape_modifier_val = smod
                    if not shape_modifier_val:
                        shape_modifier_val = 'normal'
                    is_valid = getattr(action, 'is_valid', True) if not isinstance(action, str) else True
                    stroke_specific_features = _calculate_stroke_specific_features(
                        action, j, stroke_type_val, shape_modifier_val, parameters, bongard_image=bongard_image)
                    if stroke_type_val == 'line':
                        line_features.append(stroke_specific_features)
                    elif stroke_type_val == 'arc':
                        arc_features.append(stroke_specific_features)
                    stroke_data = {
                        'stroke_id': f"{image_id}_stroke_{i}_{j}",
                        'stroke_type': stroke_type_val,
                        'shape_modifier': shape_modifier_val,
                        'parameters': parameters,
                        'raw_command': raw_command,
                        'function_name': function_name,
                        'is_valid': is_valid,
                        'specific_features': stroke_specific_features,
                    }
                    if canonical_shape_attributes:
                        stroke_data['canonical_shape_attributes'] = canonical_shape_attributes
                    if canonical_shape_def:
                        stroke_data['canonical_shape_def'] = canonical_shape_def
                    stroke_features.append(stroke_data)
                    if shape_modifier_val:
                        unique_modifiers.add(shape_modifier_val)
                    if function_name:
                        unique_shape_functions.add(function_name)
                        shape_function_counts[function_name] = shape_function_counts.get(function_name, 0) + 1

            differentiated_features = _calculate_stroke_type_differentiated_features(
                {'line_features': line_features, 'arc_features': arc_features}, all_actions)

            action_program = []
            for shape in getattr(bongard_image, 'one_stroke_shapes', []):
                # Always convert basic_actions to strings before any join/logging/serialization
                safe_actions = ensure_str_list(getattr(shape, 'basic_actions', []))
                for j, a in enumerate(safe_actions):
                    rc = getattr(a, 'raw_command', None) if not isinstance(a, str) else a
                    if isinstance(rc, str):
                        action_program.append(rc)
                    else:
                        action_str = str(a)
                        logger.warning(f"[SERIALIZE] raw_command not string for action {j}, using str(a): {action_str}")
                        action_program.append(action_str)
                # Defensive: log joined basic_actions
                logger.debug(f"[BASIC_ACTIONS_JOIN] {','.join(safe_actions)}")
            # Final check: ensure all items are strings using ensure_str_list
            action_program = ensure_str_list(action_program)
            logger.debug(f"[ACTION_PROGRAM_JOIN] {','.join(action_program)}")
            # Defensive: ensure ngram features are stringified if joined/logged/serialized
            safe_ngram = ensure_str_list(ngram_features) if isinstance(ngram_features, list) else ngram_features
            logger.debug(f"[NGRAM_JOIN] {','.join(safe_ngram) if isinstance(safe_ngram, list) else safe_ngram}")
            # Defensive: ensure stroke_features lists are stringified
            for stroke in stroke_features:
                for k, v in stroke.items():
                    if isinstance(v, list):
                        stroke[k] = ensure_str_list(v)
            # Defensive: ensure sequential features are stringified
            sequential_features = {
                'ngram': ensure_str_list(ngram_features),
                'alternation': ensure_str_list(alternation) if isinstance(alternation, list) else alternation,
                'regularity': ensure_str_list(regularity) if isinstance(regularity, list) else regularity
            }

            complete_record = {
                'image_id': image_id,
                'problem_id': problem_id,
                'category': category,
                'label': 'positive' if is_positive else 'negative',
                'image_path': image_path,
                'strokes': stroke_features,
                'num_strokes': len(stroke_features),
                'raw_vertices': vertices_raw,
                'vertices': norm_vertices_for_features,
                'num_vertices': len(norm_vertices_for_features) if norm_vertices_for_features else 0,
                'position_label': posrot_labels.get('centroid'),
                'rotation_label_degrees': posrot_labels.get('orientation_degrees'),
                'image_features': image_features,
                'physics_features': physics_features,
                'composition_features': composition_features,
                'stroke_type_features': differentiated_features,
                'image_canonical_summary': image_canonical_summary,
                'processing_metadata': {
                    'processing_timestamp': time.time(),
                    'feature_count': len(image_features) + len(physics_features) + len(composition_features)
                },
                'action_program': action_program,
                'geometry': geometry,
                'relational_features': robust_relational_features,
                'context_relational_features': {
                    'intersections': intersections,
                    'adjacency': adjacency,
                    'containment': containment,
                    'overlap': overlap,
                    'context_adjacency_matrix': context_relationships.get('adjacency_matrix'),
                    'context_containment': context_relationships.get('containment'),
                    'context_intersection_pattern': context_relationships.get('intersection_pattern'),
                    'multiscale_features': multiscale_features
                },
                'sequential_features': {
                    'ngram': ngram_features,
                    'alternation': alternation,
                    'regularity': regularity
                },
                'topological_features': graph_features
            }
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
        logger.debug(f"[_calculate_image_features] INPUTS: vertices count={len(vertices) if vertices else 0}, strokes count={len(strokes) if strokes else 0}, geometry keys={list(geometry.keys()) if geometry else []}")
        vertices = ensure_vertex_list(vertices)
        if not vertices:
            logger.debug("[_calculate_image_features] No vertices, returning empty dict")
            return {}
        try:
            from shapely.geometry import Polygon
            poly = None
            try:
                poly = Polygon(vertices)
                if not poly.is_valid or poly.area == 0:
                    logger.error(f"[FALLBACK LOGIC] Polygon invalid or zero area for vertices: {vertices}. Applying buffer(0) fallback.")
                    poly = poly.buffer(0)
            except Exception as e:
                logger.error(f"[FALLBACK LOGIC] Error in Polygon(vertices): {e}. Attempting convex hull fallback for vertices: {vertices}")
                poly = None
            if poly is None or not poly.is_valid or poly.is_empty:
                # fallback: convex hull
                try:
                    logger.error(f"[FALLBACK LOGIC] Polygon still invalid or empty after buffer(0). Falling back to convex hull for vertices: {vertices}")
                    poly = Polygon(vertices).convex_hull
                except Exception as e:
                    logger.error(f"[FALLBACK LOGIC] Error in convex hull: {e}. Geometry cannot be recovered for vertices: {vertices}")
                    poly = None

            # --- Robust, normalized feature extraction ---
            num_strokes = len(strokes)
            max_strokes = 50
            perimeter_raw = _calculate_perimeter(vertices)
            area_raw = PhysicsInference.area(poly) if poly else 0.0
            hull_perimeter = perimeter_raw
            hull_area = area_raw
            if poly is not None and hasattr(poly, 'convex_hull'):
                try:
                    hull_perimeter = poly.convex_hull.length
                    hull_area = poly.convex_hull.area
                except Exception:
                    pass
            # Normalized perimeter and area (relative to convex hull)
            perimeter_norm = min(max(perimeter_raw / hull_perimeter, 0.0), 1.0) if hull_perimeter else 0.0
            area_norm = min(max(area_raw / hull_area, 0.0), 1.0) if hull_area else 0.0

            # Use robust, analytic, normalized formulas for all features:
            curvature_score = PhysicsInference.robust_curvature(vertices)
            angular_variance = PhysicsInference.robust_angular_variance(vertices)
            moment_of_inertia = PhysicsInference.moment_of_inertia(vertices)
            visual_complexity = PhysicsInference.visual_complexity(num_strokes, max_strokes, perimeter_raw, hull_perimeter, curvature_score)
            visual_complexity_norm = min(max(visual_complexity, 0.0), 1.0) if visual_complexity is not None else 0.0

            # --- Standardized complexity metric ---
            logger.info(f"[complexity] Calling calculate_complexity with vertices: {vertices}")
            complexity = calculate_complexity(vertices)
            logger.info(f"[complexity] Output for vertices count {len(vertices)}: {complexity}")

            from src.Derive_labels.stroke_types import _compute_bounding_box
            bbox = _compute_bounding_box(vertices)
            logger.info(f"[_calculate_image_features] Bounding box: {bbox}")
            features = {
                'bounding_box': bbox,
                'centroid': geometry.get('centroid', [0.0, 0.0]),
                'width': geometry.get('width', 0.0),
                'height': geometry.get('height', 0.0),
                'area_raw': area_raw,
                'area_normalized': area_norm,
                'perimeter_raw': perimeter_raw,
                'perimeter_normalized': perimeter_norm,
                'aspect_ratio': max(FLAGGING_THRESHOLDS['min_aspect_ratio'], min(safe_divide(geometry.get('width', 1.0), geometry.get('height', 1.0), 1.0), FLAGGING_THRESHOLDS['max_aspect_ratio'])),
                'convexity_ratio': (max(0.0, min(1.0, safe_divide(poly.area, poly.convex_hull.area))) if poly and poly.area != 0 and poly.convex_hull.area != 0 else 0.0),
                'is_convex': PhysicsInference.is_convex(poly) if poly else False,
                'compactness': _calculate_compactness(area_raw, perimeter_raw),
                'eccentricity': _calculate_eccentricity(vertices),
                'symmetry_score': (PhysicsInference.symmetry_score(vertices) if perimeter_raw > 0 and area_raw > 0 else None),
                'horizontal_symmetry': _check_horizontal_symmetry(vertices, poly),
                'vertical_symmetry': _check_vertical_symmetry(vertices, poly),
                'rotational_symmetry': _check_rotational_symmetry(vertices),
                'has_quadrangle': (True if poly and hasattr(poly, 'exterior') and hasattr(poly.exterior, 'coords') and len(poly.exterior.coords)-1 == 4 else PhysicsInference.has_quadrangle(vertices)),
                'geometric_complexity': PhysicsInference.geometric_complexity(vertices),
                'visual_complexity': visual_complexity_norm,  # normalized [0,1]
                'curvature_score': curvature_score,  # analytic, normalized
                'angular_variance': angular_variance,  # analytic, normalized
                'moment_of_inertia': moment_of_inertia,  # analytic, normalized
                'irregularity_score': _calculate_irregularity(vertices),
                'standardized_complexity': complexity
            }
            logger.debug(f"[_calculate_image_features] OUTPUT: {features}")
            return json_safe(features)
        except Exception as e:
            logger.warning(f"[_calculate_image_features] Error: {e}")
            return {}
    
    def _calculate_physics_features(self, vertices: List[tuple], centroid=None, strokes=None) -> Dict[str, Any]:
        """Calculate physics-based features using PhysicsInference. Accepts centroid override and strokes for correct counting. Uses correct center_of_mass and stroke counts."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_calculate_physics_features] INPUTS: vertices count={len(vertices) if vertices else 0}, centroid={centroid}, strokes count={len(strokes) if strokes else 0 if strokes is not None else 'None'}")
        if not vertices:
            logger.debug("[_calculate_physics_features] No vertices, returning empty dict")
            return {}
        try:
            poly = None
            try:
                poly = PhysicsInference.polygon_from_vertices(vertices)
            except Exception as e:
                logger.debug(f"[_calculate_physics_features] Error in polygon_from_vertices: {e}")
            # Use centroid from geometry if provided, else fallback to centroid of vertices
            if centroid is not None:
                center_of_mass = centroid
            elif poly is not None:
                center_of_mass = PhysicsInference.centroid(poly)
            else:
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                center_of_mass = [sum(xs)/len(xs), sum(ys)/len(ys)] if xs and ys else [0.0, 0.0]
            # Count actual LineAction and ArcAction objects if strokes provided
            num_straight_segments = 0
            num_arcs = 0
            if strokes is not None:
                try:
                    from bongard.bongard import LineAction, ArcAction
                    for s in strokes:
                        if isinstance(s, LineAction):
                            num_straight_segments += 1
                        elif isinstance(s, ArcAction):
                            num_arcs += 1
                except Exception as e:
                    logger.debug(f"[_calculate_physics_features] Error in stroke counting: {e}")
            else:
                # fallback to geometry-based
                num_straight_segments = PhysicsInference.count_straight_segments(vertices)
                num_arcs = PhysicsInference.count_arcs(vertices)
            features = {
                # Core physics properties
                'moment_of_inertia': PhysicsInference.moment_of_inertia(vertices),
                'center_of_mass': center_of_mass,
                # Shape metrics
                'num_straight_segments': num_straight_segments,
                'num_arcs': num_arcs,
                'has_quadrangle': PhysicsInference.has_quadrangle(vertices),
                'has_obtuse_angle': PhysicsInference.has_obtuse(vertices),
                # Advanced metrics
                'curvature_score': _calculate_curvature_score(vertices),
                'angular_variance': _calculate_angular_variance(vertices),
                'edge_length_variance':_calculate_edge_length_variance(vertices)
            }
            logger.debug(f"[_calculate_physics_features] OUTPUT: {features}")
            return features
        except Exception as e:
            logger.warning(f"[_calculate_physics_features] Error: {e}")
            return {}

    
    def _calculate_composition_features(self, strokes: List) -> Dict[str, Any]:
        """Calculate features about stroke composition and relationships. FIXED: Use actual modifiers from strokes."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_calculate_composition_features] INPUTS: strokes count={len(strokes) if strokes else 0}")
        if not strokes:
            logger.debug("[_calculate_composition_features] No strokes, returning empty dict")
            return {}
        try:
            stroke_types = {}
            shape_modifiers = {}
            modifier_sequence = []
            for stroke in strokes:
                stroke_type = type(stroke).__name__.replace('Action', '').lower()
                modifier = _extract_modifier_from_stroke(stroke)
                stroke_types[stroke_type] = stroke_types.get(stroke_type, 0) + 1
                shape_modifiers[modifier] = shape_modifiers.get(modifier, 0) + 1
                modifier_sequence.append(modifier)
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

            if isinstance(problem_data, list) and len(problem_data) == 2:
                positive_examples, negative_examples = problem_data
            else:
                logger.warning(f"Problem {problem_id} has unexpected data format, skipping.")
                continue

            # For problem-level aggregation
            problem_unique_shape_functions = set()
            problem_shape_function_counts = {}
            problem_modifiers = set()
            num_images_in_problem = 0

            # Process positive examples
            pos_results = []
            for i, action_commands in enumerate(positive_examples):
                total_images += 1
                num_images_in_problem += 1
                image_id = f"{problem_id}_pos_{i}"
                image_path = f"images/{problem_id}/category_1/{i}.png"

                result = processor.process_single_image(
                    action_commands, image_id, True, problem_id, category, image_path
                )

                if result:
                    logger.info(f"[LABEL OUTPUT] Image: {image_id} | Problem: {problem_id}\n{json.dumps(result, indent=2, ensure_ascii=False)}")
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

            # Process negative examples
            neg_results = []
            for i, action_commands in enumerate(negative_examples):
                total_images += 1
                num_images_in_problem += 1
                image_id = f"{problem_id}_neg_{i}"
                image_path = f"images/{problem_id}/category_0/{i}.png"

                result = processor.process_single_image(
                    action_commands, image_id, False, problem_id, category, image_path
                )

                if result:
                    logger.info(f"[LABEL OUTPUT] Image: {image_id} | Problem: {problem_id}\n{json.dumps(result, indent=2, ensure_ascii=False)}")
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

            # --- Compute and add support-set context features ---
            from src.Derive_labels.relational_features import extract_support_set_context as extract_relational_support_set_context
            logger.info(f"[SUPPORT-SET CONTEXT] Calling extract_relational_support_set_context for problem {problem_id}")
            support_set_context = extract_relational_support_set_context(pos_results, neg_results)
            logger.info(f"[SUPPORT-SET CONTEXT] OUTPUT for problem {problem_id}: {json.dumps(support_set_context, indent=2, ensure_ascii=False)}")
            # Add support_set_context to each image result
            for r in pos_results + neg_results:
                r['support_set_context'] = support_set_context

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
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(safe_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully processed {successful_images}/{total_images} images")
        logger.info(f"Saved {len(all_results)} records to {args.output}")

        # Save flagged cases
        if processor.flagged_cases:
            flagged_path = os.path.join(output_dir, 'flagged_cases.json')
            safe_flagged = json_safe(processor.flagged_cases)
            with open(flagged_path, 'w', encoding='utf-8') as f:
                json.dump(safe_flagged, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(processor.flagged_cases)} flagged cases to {flagged_path}")

        # Save processing statistics
        processor.processing_stats['processing_summary'] = {
            'success_rate': processor.processing_stats['successful'] / max(processor.processing_stats['total_processed'], 1),
            'flag_rate': processor.processing_stats['flagged'] / max(processor.processing_stats['total_processed'], 1),
            'total_features_calculated': len(all_results) * 4 if all_results else 0  # 4 feature sets per record
        }

        stats_path = os.path.join(output_dir, 'processing_statistics.json')
        safe_stats = json_safe(processor.processing_stats)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(safe_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processing statistics to {stats_path}")

        # Save problem-level canonical summaries
        problem_summary_path = os.path.join(output_dir, 'problem_summaries.json')
        safe_summaries = json_safe(problem_summaries)
        with open(problem_summary_path, 'w', encoding='utf-8') as f:
            json.dump(safe_summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(problem_summaries)} problem-level canonical summaries to {problem_summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1


if __name__ == '__main__':
    exit(main())