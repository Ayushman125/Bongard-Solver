
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
from src.Derive_labels.features import extract_multiscale_features
from src.Derive_labels.context_features import BongardFeatureExtractor

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



class ComprehensiveBongardProcessor:
    def __init__(self):
        # ...existing code...
        self.context_extractor = BongardFeatureExtractor()

    def _calculate_vertices_from_action(self, action, stroke_index):
        import numpy as np
        """Calculate line segment vertices for a single action."""
        try:
            if hasattr(action, 'parameters'):
                params = action.parameters
                length = params.get('param1', 0.1)
                angle = params.get('param2', 0.5)
                # Convert normalized values to actual coordinates
                start_x = stroke_index * 0.1
                start_y = 0.5
                end_x = start_x + length * np.cos(angle * 2 * np.pi)
                end_y = start_y + length * np.sin(angle * 2 * np.pi)
                return [(start_x, start_y), (end_x, end_y)]
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to calculate vertices from action: {e}")
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
        logger.debug(f"Processing image_id={image_id}, problem_id={problem_id}, is_positive={is_positive}")
        logger.debug(f"action_commands type: {type(action_commands)}, value: {action_commands}")
        # --- Contextual/relational feature extraction ---
        # Prepare strokes as dicts with 'vertices' for context extractor
        context_strokes = []
        for a in getattr(shape, 'basic_actions', []) if 'shape' in locals() else []:
            v = getattr(a, 'vertices', None)
            if v is None and hasattr(a, 'get_world_coordinates'):
                v = a.get_world_coordinates()
            context_strokes.append({'vertices': v if v is not None else []})
        # Extract spatial relationships (adjacency, containment, intersection)
        context_relationships = self.context_extractor.extract_spatial_relationships(context_strokes) if context_strokes else {}
        logger.info(f"[process_single_image] context_relationships: {context_relationships}")
        # Multi-scale features (using normalized vertices)
        multiscale_features = extract_multiscale_features(norm_vertices_for_features) if 'norm_vertices_for_features' in locals() else {}
        logger.info(f"[process_single_image] multiscale_features: {multiscale_features}")
        import traceback
        try:
            # Flatten action_commands if it is a nested list (e.g., [[...]]), as in hybrid.py
            if isinstance(action_commands, list) and len(action_commands) == 1 and isinstance(action_commands[0], list):
                action_commands = action_commands[0]

            parser = ComprehensiveNVLabsParser()
            parsed_actions = parser.parse_action_commands(action_commands, problem_id)
            if not parsed_actions:
                logger.error(f"[process_single_image] Failed to parse action_commands: {action_commands}")
                # Save to flagged_issues.txt
                try:
                    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
                    os.makedirs(output_dir, exist_ok=True)
                    flag_path = os.path.join(output_dir, 'flagged_issues.txt')
                    with open(flag_path, 'a', encoding='utf-8') as f:
                        f.write(f"[FAILED PARSE] image_id={image_id}, problem_id={problem_id}, action_commands={action_commands}\n")
                except Exception as file_exc:
                    logger.error(f"[process_single_image] Could not write to flagged_issues.txt: {file_exc}")
                return None

            # If parser returns a list of actions, wrap in OneStrokeShape; if already OneStrokeShape, use as is
            if hasattr(parsed_actions, 'basic_actions'):
                shape = parsed_actions
            else:
                shape = OneStrokeShape(basic_actions=parsed_actions)

            # --- Always include raw vertices ---
            vertices_raw = getattr(shape, 'vertices', [])

            # --- Normalize vertices (aspect ratio preserved, centered) ---
            normalized_vertices = normalize_vertices(vertices_raw)

            # --- Robust polygon recovery on normalized vertices ---
            from shapely.geometry import Polygon
            from shapely.ops import polygonize
            poly = None
            norm_vertices_for_features = normalized_vertices
            try:
                poly = Polygon(normalized_vertices)
                if not poly.is_valid or poly.area == 0:
                    # Try buffer(0) fix
                    poly = poly.buffer(0)
                if not poly.is_valid or poly.area == 0:
                    # Try polygonize
                    polys = list(polygonize([normalized_vertices]))
                    if polys:
                        poly = polys[0]
                if (not poly.is_valid or poly.area == 0) and len(normalized_vertices) >= 3:
                    # Fallback: convex hull
                    poly = Polygon(normalized_vertices).convex_hull
                if poly.is_valid and poly.area > 0:
                    norm_vertices_for_features = list(poly.exterior.coords)
                else:
                    poly = None
            except Exception:
                poly = None
                norm_vertices_for_features = normalized_vertices

            # --- Calculate geometry from normalized (possibly recovered) vertices ---
            geometry = calculate_geometry(norm_vertices_for_features)

            # --- Derive position and rotation labels from normalized vertices ---
            posrot_labels = extract_position_and_rotation(norm_vertices_for_features)

            # --- Calculate image features using robust polygon ---
            image_features = self._calculate_image_features(norm_vertices_for_features, getattr(shape, 'basic_actions', []), geometry)
            # Use actual centroid for center_of_mass
            centroid = geometry.get('centroid')
            # Count actual LineAction and ArcAction objects for stroke counting
            composition_features = self._calculate_composition_features(getattr(shape, 'basic_actions', []))
            physics_features = self._calculate_physics_features(norm_vertices_for_features, centroid=centroid, strokes=getattr(shape, 'basic_actions', []))

            # --- Relational/Topological/Sequential Features ---
            # Convert actions to shapely geometries for relational features
            stroke_geometries = _actions_to_geometries(shape)
            logger.debug(f"Number of stroke geometries: {len(stroke_geometries)}")
            for idx, g in enumerate(stroke_geometries):
                logger.debug(f"Geometry {idx}: type={g.geom_type}, is_valid={g.is_valid}")
            # Intersections, adjacency, containment, overlap (relational) -- use buffered polygons for overlap/containment
            buffer_amt = 0.001  # Smaller buffer for robust relational features (documented: use exact intersections for lines)
            try:
                # For lines, use exact intersections; for polygons/arcs, use small buffer
                buffered_geoms = []
                for g in stroke_geometries:
                    if hasattr(g, 'geom_type') and g.geom_type == 'LineString':
                        buffered_geoms.append(g)  # No buffer for lines
                    elif hasattr(g, 'buffer'):
                        buffered_geoms.append(g.buffer(buffer_amt))
                    else:
                        buffered_geoms.append(g)
            except Exception as e:
                logger.warning(f"Buffering error in relational features: {e}")
                buffered_geoms = stroke_geometries
            try:
                intersections = PhysicsInference.find_stroke_intersections(stroke_geometries)
            except Exception as e:
                logger.warning(f"Intersections error: {e}")
                intersections = None
            try:
                adjacency = PhysicsInference.strokes_touching(stroke_geometries)
            except Exception as e:
                logger.warning(f"Adjacency error: {e}")
                adjacency = None
            try:
                containment = PhysicsInference.stroke_contains_stroke(buffered_geoms)
            except Exception as e:
                logger.warning(f"Containment error: {e}")
                containment = None
            try:
                overlap = PhysicsInference.stroke_overlap_area(buffered_geoms)
            except Exception as e:
                logger.warning(f"Overlap error: {e}")
                overlap = None

            # Sequential pattern features (n-gram, alternation, regularity)
            modifier_sequence = [_extract_modifier_from_stroke(s) for s in getattr(shape, 'basic_actions', [])]
            ngram_features = _extract_ngram_features(modifier_sequence)
            alternation = _detect_alternation(modifier_sequence)
            regularity = self._calculate_pattern_regularity_from_modifiers(modifier_sequence)

            # Topological features (chain/star/cycle detection, connectivity)
            # Use context adjacency matrix for graph topology detection
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
            original_action_commands = action_commands if isinstance(action_commands, list) else []

            # --- NEW: Extract canonical problem name for TSV lookup (robust normalization) ---
            canonical_name = problem_id
            if '_' in canonical_name:
                if canonical_name.startswith('bd_') or canonical_name.startswith('ff_') or canonical_name.startswith('hd_'):
                    canonical_name = canonical_name.split('_', 1)[1]
            # Remove numeric suffix (e.g. _0000)
            import re
            canonical_base = re.sub(r'(_\d+)?$', '', canonical_name)
            # Normalize: lowercase, replace hyphens/spaces, remove trailing underscores
            canonical_key = canonical_base.lower().replace('-', '_').replace(' ', '_').rstrip('_')
            canonical_shape_attributes = shape_attr_map.get(canonical_key)
            canonical_shape_def = shape_def_map.get(canonical_key)

            for i, action in enumerate(getattr(shape, 'basic_actions', [])):
                stroke_type_val = type(action).__name__.replace('Action', '').lower()
                raw_command = getattr(action, 'raw_command', None)
                function_name = getattr(action, 'function_name', None)
                # Fallback: if raw_command is None, try to reconstruct from original_action_commands
                if not raw_command and original_action_commands and i < len(original_action_commands):
                    if isinstance(original_action_commands[i], str):
                        raw_command = original_action_commands[i]
                shape_modifier_val = None
                parameters = {}
                # 1. Try raw_command (preferred)
                if raw_command and isinstance(raw_command, str):
                    parts = raw_command.split('_')
                    if len(parts) >= 2:
                        shape_modifier_val = parts[1]
                    # Extract parameters from the rest of the string
                    param_str = '_'.join(parts[2:]) if len(parts) > 2 else ''
                    if param_str:
                        main_params = param_str.split('-')
                        for idx, p in enumerate(main_params):
                            try:
                                parameters[f'param{idx+1}'] = float(p)
                            except Exception:
                                parameters[f'param{idx+1}'] = p
                # 2. Try function_name if not found
                if not shape_modifier_val and function_name and isinstance(function_name, str):
                    fn_parts = function_name.split('_')
                    if len(fn_parts) >= 2:
                        shape_modifier_val = fn_parts[1]
                if not function_name and raw_command and isinstance(raw_command, str):
                    # fallback: use first two parts as function_name
                    parts = raw_command.split('_')
                    if len(parts) >= 2:
                        function_name = f"{parts[0]}_{parts[1]}"
                if not shape_modifier_val and hasattr(action, 'shape_modifier'):
                    smod = getattr(action, 'shape_modifier')
                    if hasattr(smod, 'value'):
                        shape_modifier_val = smod.value
                    elif isinstance(smod, str):
                        shape_modifier_val = smod
                if not shape_modifier_val:
                    shape_modifier_val = 'normal'
                is_valid = getattr(action, 'is_valid', True)
                # --- Use canonical_name for all strokes ---
                # No per-stroke lookup; all strokes get the same canonical attributes/def
                stroke_specific_features = _calculate_stroke_specific_features(
                    action, i, stroke_type_val, shape_modifier_val, parameters)
                if stroke_type_val == 'line':
                    line_features.append(stroke_specific_features)
                elif stroke_type_val == 'arc':
                    arc_features.append(stroke_specific_features)
                stroke_data = {
                    'stroke_id': f"{image_id}_stroke_{i}",
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

            # --- Calculate stroke_type_features with correct aggregation ---
            differentiated_features = _calculate_stroke_type_differentiated_features(
                {'line_features': line_features, 'arc_features': arc_features}, getattr(shape, 'basic_actions', []))

            # --- Robust action_program: always use best available command string ---
            action_program = []
            for i, a in enumerate(getattr(shape, 'basic_actions', [])):
                rc = getattr(a, 'raw_command', None)
                if not rc and original_action_commands and i < len(original_action_commands):
                    if isinstance(original_action_commands[i], str):
                        rc = original_action_commands[i]
                if not rc:
                    rc = str(a)
                action_program.append(rc)

            image_canonical_summary = {
                'unique_shape_functions': sorted(list(unique_shape_functions)),
                'shape_function_counts': shape_function_counts,
                'modifiers': sorted(list(unique_modifiers)),
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
                # --- New relational/topological/sequential features ---
                'relational_features': {
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
            error_msg = f"Error processing image {image_id}: {e}"
            stack_trace = traceback.format_exc()
            logger.error(f"{error_msg}\n{stack_trace}")
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


    
    def _calculate_image_features(self, vertices: List[tuple], strokes: List, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive image-level features with robust polygon recovery and improved metrics. Now includes standardized complexity."""
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
                    logger.debug("[_calculate_image_features] Polygon invalid or zero area, applying buffer(0)")
                    poly = poly.buffer(0)
            except Exception as e:
                logger.debug(f"[_calculate_image_features] Error in Polygon(vertices): {e}")
                poly = None
            if poly is None or not poly.is_valid or poly.is_empty:
                # fallback: convex hull
                try:
                    logger.debug("[_calculate_image_features] Falling back to convex hull")
                    poly = Polygon(vertices).convex_hull
                except Exception as e:
                    logger.debug(f"[_calculate_image_features] Error in convex hull: {e}")
                    poly = None

            # --- Robust, normalized feature extraction ---
            num_strokes = len(strokes)
            max_strokes = 50
            perimeter = _calculate_perimeter(vertices)
            hull_perimeter = perimeter
            if poly is not None and hasattr(poly, 'convex_hull'):
                try:
                    hull_perimeter = poly.convex_hull.length
                except Exception:
                    pass
            # Use robust, analytic, normalized formulas for all features:
            curvature_score = PhysicsInference.robust_curvature(vertices)
            angular_variance = PhysicsInference.robust_angular_variance(vertices)
            moment_of_inertia = PhysicsInference.moment_of_inertia(vertices)
            visual_complexity = PhysicsInference.visual_complexity(num_strokes, max_strokes, perimeter, hull_perimeter, curvature_score)
            # Normalize perimeter: max for [0,1] box is 4 (square), use 4 for safety
            perimeter_norm = min(max(perimeter / 4.0, 0.0), 1.0) if perimeter is not None else 0.0
            # Normalize visual_complexity: robust fallback, already [0,1] if implemented as such
            visual_complexity_norm = min(max(visual_complexity, 0.0), 1.0) if visual_complexity is not None else 0.0

            # --- Standardized complexity metric ---
            logger.info(f"[complexity] Calling calculate_complexity with vertices: {vertices}")
            complexity = calculate_complexity(vertices)
            logger.info(f"[complexity] Output for vertices count {len(vertices)}: {complexity}")

            features = {
                'bounding_box': geometry.get('bbox', {}),
                'centroid': geometry.get('centroid', [0.0, 0.0]),
                'width': geometry.get('width', 0.0),
                'height': geometry.get('height', 0.0),
                'area': PhysicsInference.area(poly) if poly else 0.0,
                'perimeter': perimeter_norm,  # normalized [0,1]
                'aspect_ratio': max(FLAGGING_THRESHOLDS['min_aspect_ratio'], min(safe_divide(geometry.get('width', 1.0), geometry.get('height', 1.0), 1.0), FLAGGING_THRESHOLDS['max_aspect_ratio'])),
                'convexity_ratio': (max(0.0, min(1.0, safe_divide(poly.area, poly.convex_hull.area))) if poly and poly.area != 0 and poly.convex_hull.area != 0 else 0.0),
                'is_convex': PhysicsInference.is_convex(poly) if poly else False,
                'compactness': _calculate_compactness(PhysicsInference.area(poly) if poly else 0.0, perimeter),
                'eccentricity': _calculate_eccentricity(vertices),
                'symmetry_score': (PhysicsInference.symmetry_score(vertices) if perimeter > 0 and (PhysicsInference.area(poly) if poly else 0.0) > 0 else None),
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
            for i, action_commands in enumerate(positive_examples):
                total_images += 1
                num_images_in_problem += 1
                image_id = f"{problem_id}_pos_{i}"
                image_path = f"images/{problem_id}/category_1/{i}.png"

                result = processor.process_single_image(
                    action_commands, image_id, True, problem_id, category, image_path
                )

                if result:
                    # Log the full label/data for this image
                    logger.info(f"[LABEL OUTPUT] Image: {image_id} | Problem: {problem_id}\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                    all_results.append(result)
                    successful_images += 1
                    # Aggregate image-level canonical summary into problem-level
                    summary = result.get('image_canonical_summary', {})
                    for fn in summary.get('unique_shape_functions', []):
                        problem_unique_shape_functions.add(fn)
                    for fn, count in summary.get('shape_function_counts', {}).items():
                        problem_shape_function_counts[fn] = problem_shape_function_counts.get(fn, 0) + count
                    for mod in summary.get('modifiers', []):
                        problem_modifiers.add(mod)

            # Process negative examples
            for i, action_commands in enumerate(negative_examples):
                total_images += 1
                num_images_in_problem += 1
                image_id = f"{problem_id}_neg_{i}"
                image_path = f"images/{problem_id}/category_0/{i}.png"

                result = processor.process_single_image(
                    action_commands, image_id, False, problem_id, category, image_path
                )

                if result:
                    # Log the full label/data for this image
                    logger.info(f"[LABEL OUTPUT] Image: {image_id} | Problem: {problem_id}\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                    all_results.append(result)
                    successful_images += 1
                    # Aggregate image-level canonical summary into problem-level
                    summary = result.get('image_canonical_summary', {})
                    for fn in summary.get('unique_shape_functions', []):
                        problem_unique_shape_functions.add(fn)
                    for fn, count in summary.get('shape_function_counts', {}).items():
                        problem_shape_function_counts[fn] = problem_shape_function_counts.get(fn, 0) + count
                    for mod in summary.get('modifiers', []):
                        problem_modifiers.add(mod)

            # Save problem-level canonical summary
            problem_summary = {
                'problem_id': problem_id,
                'unique_shape_functions': sorted(list(problem_unique_shape_functions)),
                'shape_function_counts': problem_shape_function_counts,
                'modifiers': sorted(list(problem_modifiers)),
                'num_images': num_images_in_problem
            }
            # Log the full problem-level summary
            logger.info(f"[PROBLEM SUMMARY] Problem: {problem_id}\n{json.dumps(problem_summary, indent=2, ensure_ascii=False)}")
            problem_summaries.append(problem_summary)

        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {e}")
            processor._flag_case('unknown', problem_id, f'Problem processing failed: {e}', ['problem_processing_error'])


    # Save results
    try:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully processed {successful_images}/{total_images} images")
        logger.info(f"Saved {len(all_results)} records to {args.output}")

        # Save flagged cases
        if processor.flagged_cases:
            flagged_path = os.path.join(output_dir, 'flagged_cases.json')
            with open(flagged_path, 'w', encoding='utf-8') as f:
                json.dump(processor.flagged_cases, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(processor.flagged_cases)} flagged cases to {flagged_path}")

        # Save processing statistics
        processor.processing_stats['processing_summary'] = {
            'success_rate': processor.processing_stats['successful'] / max(processor.processing_stats['total_processed'], 1),
            'flag_rate': processor.processing_stats['flagged'] / max(processor.processing_stats['total_processed'], 1),
            'total_features_calculated': len(all_results) * 4 if all_results else 0  # 4 feature sets per record
        }

        stats_path = os.path.join(output_dir, 'processing_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(processor.processing_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processing statistics to {stats_path}")

        # Save problem-level canonical summaries
        problem_summary_path = os.path.join(output_dir, 'problem_summaries.json')
        with open(problem_summary_path, 'w', encoding='utf-8') as f:
            json.dump(problem_summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(problem_summaries)} problem-level canonical summaries to {problem_summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1


if __name__ == '__main__':
    exit(main())