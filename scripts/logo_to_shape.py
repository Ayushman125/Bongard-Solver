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
    'symmetry_score_max': 1000,
    'suspicious_parameter_threshold': 1e6
}



class ComprehensiveBongardProcessor:
    def _calculate_pattern_regularity_from_modifiers(self, modifier_sequence: List[str]) -> float:
        """Calculate regularity of a sequence of shape modifiers (pattern regularity)."""
        if not modifier_sequence or len(modifier_sequence) < 2:
            return 1.0  # Perfect regularity if only one or zero modifiers

        # 1. Repetition score: fraction of consecutive pairs that are the same
        repetition_count = 0
        for i in range(len(modifier_sequence) - 1):
            if modifier_sequence[i] == modifier_sequence[i + 1]:
                repetition_count += 1
        repetition_score = repetition_count / (len(modifier_sequence) - 1)

        # 2. Alternation score: fraction of pairs that alternate (A B A B ...)
        alternation_count = 0
        for i in range(len(modifier_sequence) - 2):
            if modifier_sequence[i] == modifier_sequence[i + 2] and modifier_sequence[i] != modifier_sequence[i + 1]:
                alternation_count += 1
        alternation_score = alternation_count / (len(modifier_sequence) - 2) if len(modifier_sequence) > 2 else 0.0

        # 3. Diversity penalty: more unique modifiers = less regularity
        unique_mods = set(modifier_sequence)
        diversity_penalty = (len(unique_mods) - 1) / max(len(modifier_sequence) - 1, 1)

        # 4. Final score: weighted sum (repetition + alternation) * (1 - diversity_penalty)
        base_score = max(repetition_score, alternation_score)
        pattern_regularity = base_score * (1.0 - diversity_penalty)
        # Clamp to [0,1]
        return max(0.0, min(1.0, pattern_regularity))
    def normalize_vertices(self, vertices_raw):
        """
        Normalize coordinates to [0,1] in both axes, preserving aspect ratio and centering shape if needed.
        Returns normalized vertices.
        """
        if not vertices_raw:
            return []
        xs, ys = zip(*vertices_raw)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        # Avoid division by zero
        if width == 0:
            width = 1e-8
        if height == 0:
            height = 1e-8
        # Aspect ratio preserving: fit to [0,1] in both axes, centered
        scale = max(width, height)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        # Center to (0.5,0.5) after scaling
        norm_vertices = []
        for x, y in vertices_raw:
            nx = (x - cx) / scale + 0.5
            ny = (y - cy) / scale + 0.5
            norm_vertices.append((nx, ny))
        return norm_vertices

    def calculate_geometry(self, vertices):
        """Calculate geometry properties from normalized vertices."""
        if not vertices:
            return {}
        xs, ys = zip(*vertices)
        bbox = {'min_x': min(xs), 'max_x': max(xs), 'min_y': min(ys), 'max_y': max(ys)}
        centroid = [sum(xs)/len(xs), sum(ys)/len(ys)]
        width = bbox['max_x'] - bbox['min_x']
        height = bbox['max_y'] - bbox['min_y']
        return {
            'bbox': bbox,
            'centroid': centroid,
            'width': width,
            'height': height
        }
    # Load TSVs once for all instances
    _shape_attributes = None
    _shape_defs = None

    @staticmethod
    def _load_tsv(path):
        if not os.path.exists(path):
            return []
        with open(path, newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='\t'))

    @classmethod
    def get_shape_attributes(cls):
        if cls._shape_attributes is None:
            tsv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Bongard-LOGO', 'data', 'human_designed_shapes_attributes.tsv'))
            cls._shape_attributes = cls._load_tsv(tsv_path)
        return cls._shape_attributes

    @classmethod
    def get_shape_defs(cls):
        if cls._shape_defs is None:
            tsv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Bongard-LOGO', 'data', 'human_designed_shapes.tsv'))
            cls._shape_defs = cls._load_tsv(tsv_path)
        return cls._shape_defs

    @classmethod
    def get_shape_attribute_map(cls):
        # Map from shape function name to attribute dict
        return {row['shape function name']: row for row in cls.get_shape_attributes() if row.get('shape function name')}

    @classmethod
    def get_shape_def_map(cls):
        return {row['shape function name']: row for row in cls.get_shape_defs() if row.get('shape function name')}
    """
    Enhanced comprehensive processor for Bongard-LOGO data that handles:
    - Multi-stroke image composition with stroke-type specific calculations
    - Differentiated geometry analysis for line vs arc strokes
    - Shape-modifier aware feature extraction
    - Comprehensive flagging logic for suspicious entries
    - Physics and geometry computation with validation
    """
    
    def ensure_vertex_list(self, vertices):
        """Convert Polygon or similar geometry object to list of tuples."""
        if hasattr(vertices, 'exterior') and hasattr(vertices.exterior, 'coords'):
            return list(vertices.exterior.coords)
        elif hasattr(vertices, 'coords'):
            return list(vertices.coords)
        return vertices
    
    def safe_divide(self, a, b, default=0.0):
        """Safe division avoiding zero/NaN."""
        if abs(b) < 1e-10:
            return default
        return a / b
    
    def json_safe(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self.json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.json_safe(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj
    
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
        """
        Parse action strings, then use Bongard-LOGO repo for all geometry, feature, and canonicalization calculations.
        Implements robust normalization, polygon recovery, and always includes raw_vertices.
        """
        from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
        from bongard.bongard import OneStrokeShape, LineAction, ArcAction
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Bongard-LOGO', 'Bongard-LOGO')))

        self.processing_stats['total_processed'] += 1
        logger.debug(f"Processing image_id={image_id}, problem_id={problem_id}, is_positive={is_positive}")
        logger.debug(f"action_commands type: {type(action_commands)}, value: {action_commands}")
        try:
            # Flatten action_commands if it is a nested list (e.g., [[...]]), as in hybrid.py
            if isinstance(action_commands, list) and len(action_commands) == 1 and isinstance(action_commands[0], list):
                action_commands = action_commands[0]

            parser = ComprehensiveNVLabsParser()
            parsed_actions = parser.parse_action_commands(action_commands, problem_id)
            if not parsed_actions:
                logger.error(f"[process_single_image] Failed to parse action_commands: {action_commands}")
                return None

            # If parser returns a list of actions, wrap in OneStrokeShape; if already OneStrokeShape, use as is
            if hasattr(parsed_actions, 'basic_actions'):
                shape = parsed_actions
            else:
                shape = OneStrokeShape(basic_actions=parsed_actions)

            # --- Always include raw vertices ---
            vertices_raw = getattr(shape, 'vertices', [])

            # --- Normalize vertices (aspect ratio preserved, centered) ---
            normalized_vertices = self.normalize_vertices(vertices_raw)

            # --- Robust polygon recovery on normalized vertices ---
            from shapely.geometry import Polygon
            from shapely.ops import polygonize
            poly = Polygon(normalized_vertices)
            if not poly.is_valid or poly.area == 0:
                try:
                    poly = list(polygonize([normalized_vertices]))[0]
                except Exception:
                    poly = Polygon(normalized_vertices)
            # Use the possibly recovered polygon for geometry/area/complexity
            norm_vertices_for_features = list(poly.exterior.coords) if hasattr(poly, 'exterior') else normalized_vertices

            # --- Calculate geometry from normalized (possibly recovered) vertices ---
            geometry = self.calculate_geometry(norm_vertices_for_features)

            # --- Calculate image features using robust polygon ---
            image_features = self._calculate_image_features(norm_vertices_for_features, getattr(shape, 'basic_actions', []), geometry)
            physics_features = self._calculate_physics_features(norm_vertices_for_features)
            composition_features = self._calculate_composition_features(getattr(shape, 'basic_actions', []))

            # --- Aggregate line and arc features for stroke_type_features ---
            line_features = []
            arc_features = []
            stroke_features = []
            shape_attr_map = self.get_shape_attribute_map()
            shape_def_map = self.get_shape_def_map()
            unique_shape_functions = set()
            shape_function_counts = {}
            unique_modifiers = set()
            original_action_commands = action_commands if isinstance(action_commands, list) else []
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
                    # Extract function_name for TSV lookup (base name only)
                    if len(parts) >= 1:
                        base_function_name = parts[0]
                else:
                    base_function_name = None
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
                # Use base_function_name for TSV lookup
                lookup_name = base_function_name if base_function_name else (function_name.split('_')[0] if function_name else None)
                canonical_shape_attributes = shape_attr_map.get(lookup_name) if lookup_name else None
                canonical_shape_def = shape_def_map.get(lookup_name) if lookup_name else None
                # Debug logging for missing TSV keys
                if lookup_name and (canonical_shape_attributes is None or canonical_shape_def is None):
                    logger.debug(f"[TSV LOOKUP] No TSV entry for function_name '{lookup_name}' (raw_command: {raw_command}, function_name: {function_name})")
                stroke_specific_features = self._calculate_stroke_specific_features(
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
            differentiated_features = self._calculate_stroke_type_differentiated_features(
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
                'geometry': geometry
            }
            self.processing_stats['successful'] += 1
            return self.json_safe(complete_record)
        except Exception as e:
            error_msg = f"Error processing image {image_id}: {e}"
            logger.error(error_msg)
            return None
    
    def _flag_case(self, image_id: str, problem_id: str, reason: str, flags: List[str]):
        """Add a case to the flagged cases list"""
        self.flagged_cases.append({
            'image_id': image_id,
            'problem_id': problem_id,
            'reason': reason,
            'flags': flags,
            'timestamp': time.time()
        })
        logger.warning(f"Flagged case: {image_id} - {reason}")
    
    def _validate_stroke_parameters(self, stroke) -> List[str]:
        """Validate stroke parameters for suspicious values"""
        flags = []
        
        for param_name, value in stroke.parameters.items():
            if not isinstance(value, (int, float)):
                flags.append(f"invalid_parameter_type_{param_name}")
                continue
                
            if math.isnan(value) or math.isinf(value):
                flags.append(f"invalid_parameter_value_{param_name}")
                continue
                
            if abs(value) > FLAGGING_THRESHOLDS['suspicious_parameter_threshold']:
                flags.append(f"suspicious_parameter_{param_name}")
        
        # Stroke-type specific validation
        if stroke.stroke_type.value == 'line':
            length = stroke.parameters.get('length', 0)
            if length <= 0 or length > 10:
                flags.append("suspicious_line_length")
        elif stroke.stroke_type.value == 'arc':
            radius = stroke.parameters.get('radius', 0)
            if radius <= 0 or radius > 10:
                flags.append("suspicious_arc_radius")
            span_angle = stroke.parameters.get('span_angle', 0)
            if abs(span_angle) > 720:  # More than 2 full rotations
                flags.append("suspicious_arc_span")
        
        return flags
    
    def _validate_vertices(self, vertices: List[tuple]) -> List[str]:
        """Validate vertex data"""
        flags = []
        
        if len(vertices) < FLAGGING_THRESHOLDS['min_vertices']:
            flags.append("insufficient_vertices")
        
        if len(vertices) > FLAGGING_THRESHOLDS['max_vertices']:
            flags.append("excessive_vertices")
        
        # Check for NaN or infinite coordinates
        for i, (x, y) in enumerate(vertices):
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                flags.append("invalid_vertex_type")
                break
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                flags.append("invalid_vertex_coordinates")
                break
        
        # Check for duplicate consecutive vertices
        duplicate_count = 0
        for i in range(len(vertices) - 1):
            if vertices[i] == vertices[i + 1]:
                duplicate_count += 1
        
        if duplicate_count > len(vertices) * 0.5:
            flags.append("excessive_duplicate_vertices")
        
        return flags
    
    def _validate_image_features(self, features: Dict[str, Any]) -> List[str]:
        """Validate computed image features"""
        flags = []
        
        area = features.get('area', 0)
        if area < FLAGGING_THRESHOLDS['min_area']:
            flags.append("suspicious_area_too_small")
        elif area > FLAGGING_THRESHOLDS['max_area']:
            flags.append("suspicious_area_too_large")
        
        aspect_ratio = features.get('aspect_ratio', 1)
        if aspect_ratio < FLAGGING_THRESHOLDS['min_aspect_ratio']:
            flags.append("suspicious_aspect_ratio_too_small")
        elif aspect_ratio > FLAGGING_THRESHOLDS['max_aspect_ratio']:
            flags.append("suspicious_aspect_ratio_too_large")
        
        # Check for NaN values in critical features
        critical_features = ['area', 'perimeter', 'aspect_ratio', 'compactness']
        for feature_name in critical_features:
            value = features.get(feature_name)
            if value is not None and (math.isnan(value) or math.isinf(value)):
                flags.append(f"invalid_feature_{feature_name}")
        
        return flags
    
    def _validate_physics_features(self, features: Dict[str, Any]) -> List[str]:
        """Validate physics computation results"""
        flags = []
        
        symmetry_score = features.get('symmetry_score', 0)
        if symmetry_score > FLAGGING_THRESHOLDS['symmetry_score_max']:
            flags.append("suspicious_symmetry_score")
        
        # Check moment of inertia
        moi = features.get('moment_of_inertia', 0)
        if moi < 0:
            flags.append("negative_moment_of_inertia")
        
        return flags
    
    def _calculate_stroke_specific_features(self, stroke, stroke_index: int, stroke_type_val=None, shape_modifier_val=None, parameters=None) -> Dict[str, Any]:
        """Calculate features specific to stroke type and shape modifier"""
        features = {'stroke_index': stroke_index}
        stype = stroke_type_val or type(stroke).__name__.replace('Action', '').lower()
        smod = shape_modifier_val or 'normal'
        params = parameters or {}
        if stype == 'line':
            features.update(self._calculate_line_specific_features_from_params(params))
        elif stype == 'arc':
            features.update(self._calculate_arc_specific_features_from_params(params))
        features.update(self._calculate_shape_modifier_features_from_val(smod))
        return features

    def _calculate_line_specific_features_from_params(self, params: dict) -> Dict[str, Any]:
        length = params.get('param1', 0)
        angle = params.get('param2', 0)
        return {
            'line_length': length,
            'line_angle': angle,
            'line_length_normalized': self.safe_divide(min(length, 2.0), 2.0),
            'line_angle_normalized': (angle % 1.0),
            'line_direction': 'horizontal' if abs(angle - 0.5) < 0.1 else 'vertical' if abs(angle) < 0.1 or abs(angle - 1.0) < 0.1 else 'diagonal',
            'line_is_short': length < 0.3,
            'line_is_long': length > 1.5
        }

    def _calculate_arc_specific_features_from_params(self, params: dict) -> Dict[str, Any]:
        radius = params.get('param1', 0)
        span_angle = params.get('param2', 0)
        end_angle = params.get('param3', 0)
        arc_length = abs(span_angle) * radius * self.safe_divide(math.pi, 180) if radius > 0 else 0
        is_major_arc = abs(span_angle) > 180
        is_full_circle = abs(span_angle) >= 350
        return {
            'arc_radius': radius,
            'arc_span_angle': span_angle,
            'arc_end_angle': end_angle,
            'arc_length': arc_length,
            'arc_curvature': self.safe_divide(1.0, max(radius, 1e-6)),
            'arc_is_major': is_major_arc,
            'arc_is_full_circle': is_full_circle,
            'arc_direction': 'clockwise' if span_angle < 0 else 'counterclockwise',
            'arc_radius_normalized': self.safe_divide(min(radius, 2.0), 2.0),
            'arc_span_normalized': self.safe_divide(abs(span_angle), 360.0),
            'arc_is_small': radius < 0.3,
            'arc_is_large': radius > 1.5,
            'arc_is_tight': abs(span_angle) > 270,
            'arc_is_gentle': abs(span_angle) < 90
        }
    

    def _calculate_shape_modifier_features_from_val(self, modifier: str) -> Dict[str, Any]:
        """Calculate features based on shape modifier string value (not from action object)"""
        base_features = {
            'shape_modifier': modifier,
            'is_normal': modifier == 'normal',
            'is_geometric': modifier in ['circle', 'square', 'triangle'],
            'is_pattern': modifier == 'zigzag'
        }
        # Shape-specific features
        if modifier == 'triangle':
            base_features['geometric_complexity'] = 3
            base_features['has_sharp_angles'] = True
        elif modifier == 'square':
            base_features['geometric_complexity'] = 4
            base_features['has_right_angles'] = True
        elif modifier == 'circle':
            base_features['geometric_complexity'] = 10  # Use a large but finite value for circles
            base_features['has_curved_edges'] = True
        elif modifier == 'zigzag':
            base_features['pattern_complexity'] = 'high'
            base_features['has_repetitive_pattern'] = True
        else:  # normal or unknown
            base_features['geometric_complexity'] = 1
            base_features['is_simple'] = True
        return base_features
    
    def _calculate_stroke_type_differentiated_features(self, stroke_type_features: Dict, strokes: List) -> Dict[str, Any]:
        """Calculate features that differentiate between stroke types"""
        line_features = stroke_type_features['line_features']
        arc_features = stroke_type_features['arc_features']
        
        # Basic counts
        num_lines = len(line_features)
        num_arcs = len(arc_features)
        total_strokes = num_lines + num_arcs
        
        features = {
            'stroke_composition': {
                'num_lines': num_lines,
                'num_arcs': num_arcs,
                'line_ratio': self.safe_divide(num_lines, max(total_strokes, 1)),
                'arc_ratio': self.safe_divide(num_arcs, max(total_strokes, 1)),
                'stroke_diversity': 1 if num_lines > 0 and num_arcs > 0 else 0
            }
        }
        
        # Line-specific aggregate features
        if line_features:
            line_lengths = [f['line_length'] for f in line_features]
            line_angles = [f['line_angle'] for f in line_features]
            
            features['line_aggregate'] = {
                'total_line_length': sum(line_lengths),
                'avg_line_length': self.safe_divide(sum(line_lengths), len(line_lengths)),
                'line_length_variance': self._calculate_variance(line_lengths),
                'line_angle_variance': self._calculate_variance(line_angles),
                'has_short_lines': any(f['line_is_short'] for f in line_features),
                'has_long_lines': any(f['line_is_long'] for f in line_features),
                'dominant_direction': self._calculate_dominant_direction(line_features)
            }
        
        # Arc-specific aggregate features  
        if arc_features:
            arc_radii = [f['arc_radius'] for f in arc_features]
            arc_spans = [f['arc_span_angle'] for f in arc_features]
            arc_lengths = [f['arc_length'] for f in arc_features]
            
            features['arc_aggregate'] = {
                'total_arc_length': sum(arc_lengths),
                'avg_arc_radius': self.safe_divide(sum(arc_radii), len(arc_radii)),
                'avg_arc_span': self.safe_divide(sum(arc_spans), len(arc_spans)),
                'arc_radius_variance': self._calculate_variance(arc_radii),
                'arc_span_variance': self._calculate_variance(arc_spans),
                'total_curvature': sum(f['arc_curvature'] for f in arc_features),
                'has_full_circles': any(f['arc_is_full_circle'] for f in arc_features),
                'has_major_arcs': any(f['arc_is_major'] for f in arc_features),
                'curvature_complexity': len([f for f in arc_features if f['arc_curvature'] > 1.0])
            }
        
        return features
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = self.safe_divide(sum(values), len(values))
        return self.safe_divide(sum((x - mean) ** 2 for x in values), len(values))
    
    def _calculate_dominant_direction(self, line_features: List[Dict]) -> str:
        """Calculate the dominant direction of line strokes"""
        if not line_features:
            return 'none'
        
        directions = [f['line_direction'] for f in line_features]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        return max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else 'none'
    
    def _calculate_image_features(self, vertices: List[tuple], strokes: List, 
                                geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive image-level features with robust polygon recovery."""
        vertices = self.ensure_vertex_list(vertices)
        if not vertices:
            return {}
        try:
            from shapely.geometry import Polygon
            from shapely.ops import polygonize
            try:
                poly = Polygon(vertices)
                if not poly.is_valid or poly.area == 0:
                    # Try to recover with polygonize
                    poly = list(polygonize([vertices]))[0]
            except Exception:
                poly = Polygon(vertices)
            features = {
                'bounding_box': geometry.get('bbox', {}),
                'centroid': geometry.get('centroid', [0.0, 0.0]),
                'width': geometry.get('width', 0.0),
                'height': geometry.get('height', 0.0),
                'area': PhysicsInference.area(poly),
                'perimeter': self._calculate_perimeter(vertices),
                'aspect_ratio': self.safe_divide(geometry.get('width', 1.0), geometry.get('height', 1.0), 1.0)
            }
            features.update({
                'is_convex': PhysicsInference.is_convex(poly),
                'convexity_ratio': self._calculate_convexity_ratio(poly),
                'compactness': self._calculate_compactness(features['area'], features['perimeter']),
                'eccentricity': self._calculate_eccentricity(vertices)
            })
            features.update({
                'symmetry_score': PhysicsInference.symmetry_score(vertices),
                'horizontal_symmetry': self._check_horizontal_symmetry(vertices),
                'vertical_symmetry': self._check_vertical_symmetry(vertices),
                'rotational_symmetry': self._check_rotational_symmetry(vertices)
            })
            features.update({
                'geometric_complexity': self._calculate_geometric_complexity(vertices),
                'visual_complexity': len(strokes) + self.safe_divide(len(vertices), 10.0),
                'irregularity_score': self._calculate_irregularity(vertices)
            })
            return self.json_safe(features)
        except Exception as e:
            logger.warning(f"Error calculating image features: {e}")
            return {}
    
    def _calculate_physics_features(self, vertices: List[tuple]) -> Dict[str, Any]:
        """Calculate physics-based features using PhysicsInference."""
        if not vertices:
            return {}
            
        try:
            poly = PhysicsInference.polygon_from_vertices(vertices)
            
            features = {
                # Core physics properties
                'moment_of_inertia': PhysicsInference.moment_of_inertia(vertices),
                'center_of_mass': PhysicsInference.centroid(poly),
                
                # Shape metrics
                'num_straight_segments': PhysicsInference.count_straight_segments(vertices),
                'num_arcs': PhysicsInference.count_arcs(vertices),
                'has_quadrangle': PhysicsInference.has_quadrangle(vertices),
                'has_obtuse_angle': PhysicsInference.has_obtuse(vertices),
                
                # Advanced metrics
                'curvature_score': self._calculate_curvature_score(vertices),
                'angular_variance': self._calculate_angular_variance(vertices),
                'edge_length_variance': self._calculate_edge_length_variance(vertices)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating physics features: {e}")
            return {}
    
    def _calculate_composition_features(self, strokes: List) -> Dict[str, Any]:
        """Calculate features about stroke composition and relationships. FIXED: Use actual modifiers from strokes."""
        if not strokes:
            return {}
        try:
            stroke_types = {}
            shape_modifiers = {}
            modifier_sequence = []
            for stroke in strokes:
                stroke_type = type(stroke).__name__.replace('Action', '').lower()
                modifier = self._extract_modifier_from_stroke(stroke)
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
                'composition_complexity': len(strokes) * len(shape_modifiers),
                'homogeneity_score': self._calculate_homogeneity(shape_modifiers),
                'pattern_regularity': self._calculate_pattern_regularity_from_modifiers(modifier_sequence)
            })
            return features
        except Exception as e:
            logger.warning(f"Error calculating composition features: {e}")
            return {}

    def _extract_modifier_from_stroke(self, stroke) -> str:
        """Extract the actual shape modifier from a stroke object, robustly."""
        # Priority: attribute > raw_command > function_name > fallback
        if hasattr(stroke, 'shape_modifier'):
            smod = getattr(stroke, 'shape_modifier')
            if hasattr(smod, 'value'):
                if smod.value:
                    return str(smod.value)
            elif isinstance(smod, str) and smod:
                return smod
        raw_command = getattr(stroke, 'raw_command', None)
        if raw_command and isinstance(raw_command, str):
            parts = raw_command.split('_')
            if len(parts) >= 2 and parts[1]:
                return parts[1]
        function_name = getattr(stroke, 'function_name', None)
        if function_name and isinstance(function_name, str):
            fn_parts = function_name.split('_')
            if len(fn_parts) >= 2 and fn_parts[1]:
                return fn_parts[1]
        return 'normal'

    # Helper methods for feature calculation
    def _calculate_perimeter(self, vertices: List[tuple]) -> float:
        """Calculate perimeter of the shape."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            perimeter += ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        
        return self.json_safe(perimeter)
    
    def _calculate_convexity_ratio(self, poly) -> float:
        """Calculate ratio of polygon area to convex hull area."""
        try:
            if poly.area == 0:
                return 0.0
            return self.safe_divide(poly.area, poly.convex_hull.area)
        except:
            return 0.0
    
    def _calculate_compactness(self, area: float, perimeter: float) -> float:
        """Calculate compactness (isoperimetric ratio)."""
        if perimeter == 0:
            return 0.0
        return self.json_safe(self.safe_divide(4 * 3.14159 * area, perimeter * perimeter))
    
    def _calculate_eccentricity(self, vertices: List[tuple]) -> float:
        """Calculate eccentricity based on principal component analysis."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 3:
            return 0.0
        
        try:
            import numpy as np
            points = np.array(vertices)
            
            # Center the points
            centered = points - np.mean(points, axis=0)
            
            # Calculate covariance matrix and eigenvalues
            cov_matrix = np.cov(centered.T)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
            
            if eigenvals[1] == 0:
                return 1.0
            
            return self.json_safe(1.0 - self.safe_divide(eigenvals[1], eigenvals[0]))
        except:
            return 0.0
    
    def _check_horizontal_symmetry(self, vertices: List[tuple]) -> float:
        """Check horizontal reflection symmetry."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 2:
            return 0.0
        
        try:
            import numpy as np
            points = np.array(vertices)
            centroid = np.mean(points, axis=0)
            
            # Reflect points horizontally about centroid
            reflected = points.copy()
            reflected[:, 0] = 2 * centroid[0] - reflected[:, 0]
            
            # Calculate similarity (inverse of RMSE)
            rmse = np.sqrt(np.mean((points - reflected)**2))
            return max(0.0, 1.0 - self.safe_divide(rmse, 100.0))  # Normalize
        except:
            return 0.0
    
    def _check_vertical_symmetry(self, vertices: List[tuple]) -> float:
        """Check vertical reflection symmetry."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 2:
            return 0.0
        
        try:
            import numpy as np
            points = np.array(vertices)
            centroid = np.mean(points, axis=0)
            
            # Reflect points vertically about centroid
            reflected = points.copy()
            reflected[:, 1] = 2 * centroid[1] - reflected[:, 1]
            
            # Calculate similarity (inverse of RMSE)
            rmse = np.sqrt(np.mean((points - reflected)**2))
            return max(0.0, 1.0 - self.safe_divide(rmse, 100.0))  # Normalize
        except:
            return 0.0
    
    def _check_rotational_symmetry(self, vertices: List[tuple]) -> int:
        """Check rotational symmetry order."""
        # Simplified check for common rotational symmetries
        h_sym = self._check_horizontal_symmetry(vertices)
        v_sym = self._check_vertical_symmetry(vertices)
        
        if h_sym > 0.8 and v_sym > 0.8:
            return 4  # 4-fold symmetry
        elif h_sym > 0.8 or v_sym > 0.8:
            return 2  # 2-fold symmetry
        else:
            return 1  # No rotational symmetry
    
    def _calculate_geometric_complexity(self, vertices: List[tuple]) -> float:
        """Calculate geometric complexity score."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 3:
            return 1.0
        
        # Base complexity from vertex count
        complexity = self.safe_divide(len(vertices), 10.0)
        
        # Add complexity from angular variance
        angular_var = self._calculate_angular_variance(vertices)
        complexity += self.safe_divide(angular_var, 100.0)
        
        # Add complexity from edge length variance
        edge_var = self._calculate_edge_length_variance(vertices)
        complexity += self.safe_divide(edge_var, 100.0)
        
        return self.json_safe(complexity)
    
    def _calculate_irregularity(self, vertices: List[tuple]) -> float:
        """Calculate irregularity score based on deviations from regular polygon."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 3:
            return 0.0
        
        try:
            import numpy as np
            
            # Calculate angles between consecutive edges
            angles = []
            n = len(vertices)
            
            for i in range(n):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % n])
                p3 = np.array(vertices[(i + 2) % n])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angles.append(angle)
            
            if not angles:
                return 0.0
            
            # Expected angle for regular polygon
            expected_angle = self.safe_divide((n - 2) * np.pi, n)
            
            # Calculate variance from expected angle
            variance = np.var([abs(angle - expected_angle) for angle in angles])
            return self.json_safe(min(1.0, variance))
            
        except:
            return 0.0
    
    def _calculate_curvature_score(self, vertices: List[tuple]) -> float:
        """Calculate overall curvature score."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        return self.safe_divide(PhysicsInference.count_arcs(vertices), max(len(vertices), 1))
    
    def _calculate_angular_variance(self, vertices: List[tuple]) -> float:
        """Calculate variance in angles between edges."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 3:
            return 0.0
        
        try:
            import numpy as np
            angles = []
            n = len(vertices)
            
            for i in range(n):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % n])
                p3 = np.array(vertices[(i + 2) % n])
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angles.append(np.degrees(angle))
            
            return self.json_safe(np.var(angles) if angles else 0.0)
        except:
            return 0.0
    
    def _calculate_edge_length_variance(self, vertices: List[tuple]) -> float:
        """Calculate variance in edge lengths."""
        # Fix: Ensure vertices is a list of tuples, not a Polygon object
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 2:
            return 0.0
        
        try:
            import numpy as np
            lengths = []
            
            for i in range(len(vertices)):
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % len(vertices)])
                length = np.linalg.norm(p2 - p1)
                lengths.append(length)
            
            return self.json_safe(np.var(lengths) if lengths else 0.0)
        except:
            return 0.0
    
    def _calculate_homogeneity(self, shape_modifiers: Dict[str, int]) -> float:
        """Calculate homogeneity score of shape modifiers."""
        if not shape_modifiers:
            return 0.0
        
        total = sum(shape_modifiers.values())
        if total == 0:
            return 0.0
        
        # Calculate entropy-based homogeneity
        entropy = 0.0
        for count in shape_modifiers.values():
            if count > 0:
                prob = self.safe_divide(count, total)
                entropy -= prob * math.log2(prob) if prob > 0 else 0
        
        # Convert entropy to homogeneity (higher entropy = lower homogeneity)
        max_entropy = math.log2(len(shape_modifiers)) if len(shape_modifiers) > 1 else 1
        return 1.0 - self.safe_divide(entropy, max_entropy) if max_entropy > 0 else 1.0
    
    def _calculate_pattern_regularity(self, strokes: List) -> float:
        """Calculate regularity of stroke patterns."""
        if len(strokes) < 2:
            return 1.0
        
        # Check for repeated patterns
        stroke_sequence = [s.shape_modifier.value for s in strokes]
        
        # Simple pattern detection
        regularities = []
        
        # Check for alternating patterns
        if len(stroke_sequence) >= 4:
            alternating_score = 0
            for i in range(len(stroke_sequence) - 1):
                if i % 2 == 0:
                    if stroke_sequence[i] == stroke_sequence[i + 2] if i + 2 < len(stroke_sequence) else True:
                        alternating_score += 1
            regularities.append(alternating_score / (len(stroke_sequence) // 2))
        
        # Check for repetition
        if len(stroke_sequence) >= 2:
            repetition_score = 0
            for i in range(len(stroke_sequence) - 1):
                if stroke_sequence[i] == stroke_sequence[i + 1]:
                    repetition_score += 1
            regularities.append(repetition_score / (len(stroke_sequence) - 1))
        
        return max(regularities) if regularities else 0.0
    
    def extract_action_type_prefixes(self, problems_data):
        """
        Use BongardImage.import_from_action_string_list to robustly extract all unique action type prefixes from the dataset.
        This mirrors hybrid.py's handling and avoids information loss from naive string splitting.
        """
        prefixes = set()
        for problem_data in problems_data.values():
            if not (isinstance(problem_data, list) and len(problem_data) == 2):
                continue
            for example_list in problem_data:
                for action_commands in example_list:
                    # Flatten if needed
                    if isinstance(action_commands, list) and len(action_commands) == 1 and isinstance(action_commands[0], list):
                        action_commands = action_commands[0]
                    try:
                        bongard_image = BongardImage.import_from_action_string_list(action_commands)
                        for shape in getattr(bongard_image, 'one_stroke_shapes', []):
                            stroke_type = getattr(shape, 'stroke_type', None)
                            if hasattr(stroke_type, 'value'):
                                prefix = stroke_type.value
                            elif stroke_type is not None:
                                prefix = str(stroke_type)
                            else:
                                prefix = shape.__class__.__name__
                            prefixes.add(prefix)
                    except Exception as e:
                        logger.warning(f"[extract_action_type_prefixes] Failed to robustly parse action_commands: {action_commands} | Error: {e}")
                        continue
        return prefixes

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
    processor = ComprehensiveBongardProcessor()
    action_type_prefixes = processor.extract_action_type_prefixes(problems_data)
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