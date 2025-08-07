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
    'symmetry_score_max': 2.0,  # RMSE for [0,1] normalized points
    'suspicious_parameter_threshold': 1e6
}



from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser, OneStrokeShape

class ComprehensiveBongardProcessor:
    def _check_horizontal_symmetry(self, vertices, poly=None):
        """Check horizontal symmetry using PhysicsInference or geometric comparison."""
        try:
            
            # Prefer PhysicsInference if available
            if hasattr(PhysicsInference, 'horizontal_symmetry'):
                return PhysicsInference.horizontal_symmetry(vertices)
            # Fallback: compare top and bottom halves
            if poly is not None and hasattr(poly, 'centroid'):
                centroid_y = poly.centroid.y
            else:
                centroid_y = sum(v[1] for v in vertices) / len(vertices)
            reflected = [(x, 2*centroid_y - y) for x, y in vertices]
            # Compare original and reflected (simple mean distance)
            import numpy as np
            orig = np.array(vertices)
            refl = np.array(reflected)
            if orig.shape == refl.shape:
                return float(np.mean(np.linalg.norm(orig - refl, axis=1)))
            return 0.0
        except Exception as e:
            logging.getLogger(__name__).warning(f"Horizontal symmetry error: {e}")
            return 0.0

    def _check_vertical_symmetry(self, vertices, poly=None):
        """Check vertical symmetry using PhysicsInference or geometric comparison."""
        try:
            if hasattr(PhysicsInference, 'vertical_symmetry'):
                return PhysicsInference.vertical_symmetry(vertices)
            if poly is not None and hasattr(poly, 'centroid'):
                centroid_x = poly.centroid.x
            else:
                centroid_x = sum(v[0] for v in vertices) / len(vertices)
            reflected = [(2*centroid_x - x, y) for x, y in vertices]
            import numpy as np
            orig = np.array(vertices)
            refl = np.array(reflected)
            if orig.shape == refl.shape:
                return float(np.mean(np.linalg.norm(orig - refl, axis=1)))
            return 0.0
        except Exception as e:
            logging.getLogger(__name__).warning(f"Vertical symmetry error: {e}")
            return 0.0

    def _calculate_edge_length_variance(self, vertices):
        """Calculate variance of edge lengths for a polygon."""
        try:
            if not vertices or len(vertices) < 2:
                return float('nan')
            import numpy as np
            n = len(vertices)
            lengths = [np.linalg.norm(np.array(vertices[(i+1)%n]) - np.array(vertices[i])) for i in range(n)]
            if len(lengths) < 2:
                return float('nan')
            return float(np.var(lengths))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Edge length variance error: {e}")
            return float('nan')


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
    
    def _actions_to_geometries(self, shape, arc_points=24):
        """
        Convert all basic_actions in a shape to shapely geometries (LineString), using the true world-space vertices from shape.vertices.
        Each stroke is a segment between consecutive vertices. Fallback to synthetic only if vertices are missing.
        """
        from shapely.geometry import LineString
        import logging
        verts = getattr(shape, 'vertices', None)
        geoms = []
        if verts and isinstance(verts, (list, tuple)) and len(verts) >= 2:
            for i in range(len(verts) - 1):
                try:
                    seg = LineString([verts[i], verts[i+1]])
                    if seg.is_valid and not seg.is_empty:
                        geoms.append(seg)
                    else:
                        logging.debug(f"Stroke {i}: invalid or empty LineString from vertices {verts[i]}, {verts[i+1]}")
                except Exception as e:
                    logging.debug(f"Stroke {i}: failed to create LineString: {e}")
        else:
            # Fallback: try to synthesize as before (should rarely happen)
            actions = getattr(shape, 'basic_actions', [])
            for i, action in enumerate(actions):
                v = getattr(action, 'vertices', None)
                if v and isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        seg = LineString(v)
                        if seg.is_valid and not seg.is_empty:
                            geoms.append(seg)
                    except Exception as e:
                        logging.debug(f"Fallback: failed to create LineString for stroke {i}: {e}")
        logging.debug(f"Number of stroke geometries: {len(geoms)}")
        return geoms
    
    def extract_position_and_rotation(self, vertices):
        """Given a list of (x, y) normalized vertices, return centroid and orientation angle (degrees)."""
        import numpy as np
        try:
            pts = np.array(vertices)
            if pts.shape[0] < 2:
                return {'centroid': [float(pts[0,0]), float(pts[0,1])] if pts.shape[0] == 1 else [0.5, 0.5], 'orientation_degrees': 0.0}
            centroid = pts.mean(axis=0)
            pts_centered = pts - centroid
            # PCA: first principal axis
            cov = np.cov(pts_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, np.argmax(eigvals)]
            angle = np.degrees(np.arctan2(axis[1], axis[0]))
            return {
                'centroid': [float(centroid[0]), float(centroid[1])],
                'orientation_degrees': float(angle)
            }
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"extract_position_and_rotation failed: {e}")
            return {'centroid': [0.5, 0.5], 'orientation_degrees': 0.0}

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

    def _extract_stroke_vertices(self, stroke, stroke_index, all_vertices):
        """Extract vertices for individual stroke from overall shape vertices."""
        try:
            # Method 1: Direct vertices from stroke
            if hasattr(stroke, 'vertices') and stroke.vertices:
                return stroke.vertices
            # Method 2: Calculate from stroke parameters
            if hasattr(stroke, 'raw_command'):
                return self._vertices_from_command(stroke.raw_command, stroke_index)
            # Method 3: Segment from overall vertices
            if all_vertices and len(all_vertices) > stroke_index + 1:
                return [all_vertices[stroke_index], all_vertices[stroke_index + 1]]
            return []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to extract vertices for stroke {stroke_index}: {e}")
            return []

    def _vertices_from_command(self, command, stroke_index):
        import numpy as np
        """Generate vertices from action command string."""
        try:
            parts = command.split('_')
            if len(parts) >= 3:
                params = parts[2].split('-')
                if len(params) >= 2:
                    length = float(params[0])
                    angle = float(params[1])
                    start = (stroke_index * 0.2, 0.5)
                    end_x = start[0] + length * np.cos(angle * 2 * np.pi)
                    end_y = start[1] + length * np.sin(angle * 2 * np.pi)
                    return [start, (end_x, end_y)]
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Failed to parse command {command}: {e}")
        return []
    def _calculate_pattern_regularity_from_modifiers(self, modifier_sequence: List[str]) -> float:
        """Calculate regularity of a sequence of shape modifiers (pattern regularity) with improved formula."""
        import logging
        logger = logging.getLogger(__name__)
        n = len(modifier_sequence)
        if not modifier_sequence or n < 3:
            logger.debug("Pattern regularity: sequence too short, returning NaN")
            return float('nan')
        # Repetition: fraction of consecutive repeats
        repetition_score = sum(1 for i in range(n-1) if modifier_sequence[i] == modifier_sequence[i+1]) / (n-1)
        # Alternation: fraction of strict alternations (A,B,A,B,...)
        alternation_score = 0.0
        if n >= 4:
            alt = [modifier_sequence[i] for i in range(2)]
            is_alt = all(modifier_sequence[i] == alt[i%2] for i in range(n)) and alt[0] != alt[1]
            if is_alt:
                alternation_score = 1.0
        # Diversity penalty: more unique = less regular
        unique_mods = set(modifier_sequence)
        diversity_penalty = (len(unique_mods) - 1) / max(n-1, 1)
        pattern_score = max(repetition_score, alternation_score)
        diversity_factor = 1.0 - diversity_penalty
        pattern_regularity = pattern_score * diversity_factor
        pattern_regularity = max(0.0, min(1.0, pattern_regularity))
        logger.debug(f"Pattern regularity: repetition_score={repetition_score}, alternation_score={alternation_score}, diversity_penalty={diversity_penalty}, result={pattern_regularity}")
        return pattern_regularity
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
        logger.debug(f"Processing image_id={image_id}, problem_id={problem_id}, is_positive={is_positive}")
        logger.debug(f"action_commands type: {type(action_commands)}, value: {action_commands}")
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
            normalized_vertices = self.normalize_vertices(vertices_raw)

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
            geometry = self.calculate_geometry(norm_vertices_for_features)

            # --- Derive position and rotation labels from normalized vertices ---
            posrot_labels = self.extract_position_and_rotation(norm_vertices_for_features)

            # --- Calculate image features using robust polygon ---
            image_features = self._calculate_image_features(norm_vertices_for_features, getattr(shape, 'basic_actions', []), geometry)
            # Use actual centroid for center_of_mass
            centroid = geometry.get('centroid')
            # Count actual LineAction and ArcAction objects for stroke counting
            composition_features = self._calculate_composition_features(getattr(shape, 'basic_actions', []))
            physics_features = self._calculate_physics_features(norm_vertices_for_features, centroid=centroid, strokes=getattr(shape, 'basic_actions', []))

            # --- Relational/Topological/Sequential Features ---
            # Convert actions to shapely geometries for relational features
            stroke_geometries = self._actions_to_geometries(shape)
            logger.debug(f"Number of stroke geometries: {len(stroke_geometries)}")
            for idx, g in enumerate(stroke_geometries):
                logger.debug(f"Geometry {idx}: type={g.geom_type}, is_valid={g.is_valid}")
            # Intersections, adjacency, containment, overlap (relational) -- use buffered polygons for overlap/containment
            buffer_amt = 0.01  # Small buffer for robust relational features
            buffered_geoms = [g.buffer(buffer_amt) if hasattr(g, 'buffer') else g for g in stroke_geometries]
            intersections = PhysicsInference.find_stroke_intersections(stroke_geometries)
            adjacency = PhysicsInference.strokes_touching(stroke_geometries)
            containment = PhysicsInference.stroke_contains_stroke(buffered_geoms)
            overlap = PhysicsInference.stroke_overlap_area(buffered_geoms)

            # Sequential pattern features (n-gram, alternation, regularity)
            modifier_sequence = [self._extract_modifier_from_stroke(s) for s in getattr(shape, 'basic_actions', [])]
            ngram_features = self._extract_ngram_features(modifier_sequence)
            alternation = self._detect_alternation(modifier_sequence)
            regularity = self._calculate_pattern_regularity_from_modifiers(modifier_sequence)

            # Topological features (chain/star/cycle detection, connectivity)
            graph_features = self._extract_graph_features(getattr(shape, 'basic_actions', []))

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
                    'overlap': overlap
                },
                'sequential_features': {
                    'ngram': ngram_features,
                    'alternation': alternation,
                    'regularity': regularity
                },
                'topological_features': graph_features
            }
            self.processing_stats['successful'] += 1
            return self.json_safe(complete_record)
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

    def _extract_ngram_features(self, sequence, n=2):
        """Extract n-gram counts from a sequence, with string keys for JSON compatibility."""
        from collections import Counter
        ngrams = zip(*[sequence[i:] for i in range(n)])
        # Convert tuple n-grams to string keys (e.g., 'A|B')
        ngram_list = ['|'.join(map(str, ng)) for ng in ngrams]
        return dict(Counter(ngram_list))

    def _detect_alternation(self, sequence):
        """Detect if sequence alternates between two values."""
        if len(sequence) < 2:
            return False
        a, b = sequence[0], sequence[1]
        for i, val in enumerate(sequence):
            if val != (a if i % 2 == 0 else b):
                return False
        return True

    def _extract_graph_features(self, strokes):
        """Detect chain/star/cycle topology and connectivity from stroke relationships."""
        # Placeholder: count strokes, check for cycles (if all touch), star (one central), chain (ends=2)
        n = len(strokes)
        if n == 0:
            return {'type': 'none', 'connectivity': 0}
        # For now, just return counts; real implementation would use adjacency/intersection
        return {'num_strokes': n}
    
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
        # Robustly convert parameters to float, fallback to 0 if conversion fails
        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        radius = to_float(params.get('param1', 0))
        span_angle = to_float(params.get('param2', 0))
        end_angle = to_float(params.get('param3', 0))

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
        """Calculate variance of a list of values. For length 2, return squared diff. For length 1, return NaN and log."""
        import numpy as np
        logger = logging.getLogger(__name__)
        n = len(values)
        if n < 1:
            logger.warning("Variance: empty list, returning NaN")
            return float('nan')
        if n == 1:
            logger.warning("Variance: only one value, returning NaN")
            return float('nan')
        if n == 2:
            diff = values[1] - values[0]
            return diff * diff / 2.0
        mean = self.safe_divide(sum(values), n)
        return self.safe_divide(sum((x - mean) ** 2 for x in values), n)
    
    def _calculate_dominant_direction(self, line_features: List[Dict]) -> str:
        """Calculate the dominant direction of line strokes"""
        if not line_features:
            return 'none'
        
        directions = [f['line_direction'] for f in line_features]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        return max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else 'none'
    
    def _calculate_image_features(self, vertices: List[tuple], strokes: List, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive image-level features with robust polygon recovery and improved metrics."""
        import numpy as np
        logger = logging.getLogger(__name__)
        vertices = self.ensure_vertex_list(vertices)
        if not vertices:
            logger.debug("Image features: no vertices, returning empty dict")
            return {}
        try:
            from shapely.geometry import Polygon
            poly = None
            try:
                poly = Polygon(vertices)
                if not poly.is_valid or poly.area == 0:
                    logger.debug("Image features: polygon invalid or zero area, applying buffer(0)")
                    poly = poly.buffer(0)
            except Exception as e:
                logger.debug(f"Image features: error in Polygon(vertices): {e}")
                poly = None
            if poly is None or not poly.is_valid or poly.is_empty:
                # fallback: convex hull
                try:
                    logger.debug("Image features: falling back to convex hull")
                    poly = Polygon(vertices).convex_hull
                except Exception as e:
                    logger.debug(f"Image features: error in convex hull: {e}")
                    poly = None
            # Robust bounding box: fallback to min/max if geometry['bbox'] missing or invalid
            bbox = geometry.get('bbox', None)
            if not bbox or not isinstance(bbox, dict) or any(k not in bbox for k in ['xmin','ymin','xmax','ymax']):
                xs = [v[0] for v in vertices]
                ys = [v[1] for v in vertices]
                bbox = {'xmin': float(np.min(xs)), 'ymin': float(np.min(ys)), 'xmax': float(np.max(xs)), 'ymax': float(np.max(ys))}
            width = bbox['xmax'] - bbox['xmin']
            height = bbox['ymax'] - bbox['ymin']
            width = max(width, 1e-6)
            height = max(height, 1e-6)
            features = {
                'bounding_box': bbox,
                'centroid': geometry.get('centroid', [0.0, 0.0]),
                'width': width,
                'height': height,
                'area': PhysicsInference.area(poly) if poly else 0.0,
                'perimeter': self._calculate_perimeter(vertices),
            }
            # Aspect ratio: always positive, robust to zero division, flag outliers
            raw_ar = self.safe_divide(width, height, 1.0)
            ar = max(FLAGGING_THRESHOLDS['min_aspect_ratio'], min(raw_ar, FLAGGING_THRESHOLDS['max_aspect_ratio']))
            features['aspect_ratio'] = ar

            # Convexity ratio: robust, clamp to [0,1], handle degenerate
            if poly is None or poly.area == 0 or poly.convex_hull.area == 0:
                features['convexity_ratio'] = 0.0
            else:
                ratio = self.safe_divide(poly.area, poly.convex_hull.area)
                features['convexity_ratio'] = max(0.0, min(1.0, ratio))

            features['is_convex'] = PhysicsInference.is_convex(poly) if poly else False
            # Compactness: robust to zero area/perimeter
            if features['perimeter'] == 0 or features['area'] == 0:
                features['compactness'] = 0.0
            else:
                features['compactness'] = self._calculate_compactness(features['area'], features['perimeter'])
            # Eccentricity: robust helper
            try:
                features['eccentricity'] = float(self._calculate_eccentricity(vertices))
            except Exception as e:
                logger.debug(f"Eccentricity calculation failed: {e}")
                features['eccentricity'] = 0.0

            # Symmetry and compactness: None if area or perimeter is zero
            if features['perimeter'] == 0 or features['area'] == 0:
                features['symmetry_score'] = None
            else:
                features['symmetry_score'] = PhysicsInference.symmetry_score(vertices)

            features['horizontal_symmetry'] = self._check_horizontal_symmetry(vertices, poly)
            features['vertical_symmetry'] = self._check_vertical_symmetry(vertices, poly)
            features['rotational_symmetry'] = self._check_rotational_symmetry(vertices)

            # has_quadrangle: robust check for 4-vertex polygons
            if poly and hasattr(poly, 'exterior') and hasattr(poly.exterior, 'coords') and len(poly.exterior.coords)-1 == 4:
                features['has_quadrangle'] = True
            else:
                # fallback: check convex hull
                try:
                    hull = poly.convex_hull if poly else None
                    if hull and hasattr(hull, 'exterior') and hasattr(hull.exterior, 'coords'):
                        features['has_quadrangle'] = (len(hull.exterior.coords)-1 == 4)
                    else:
                        features['has_quadrangle'] = False
                except Exception:
                    features['has_quadrangle'] = False

            features['geometric_complexity'] = PhysicsInference.geometric_complexity(vertices)
            # Improved visual complexity: alpha*(V-3)/(Vmax-3) + (1-alpha)*(S-1)/(Smax-1)
            alpha, V_max, S_max = 0.5, 30, 10
            V, S = len(vertices), len(strokes)
            vcomp = alpha * self.safe_divide(V-3, V_max-3) + (1-alpha) * self.safe_divide(S-1, S_max-1)
            features['visual_complexity'] = max(0.0, vcomp)
            features['irregularity_score'] = self._calculate_irregularity(vertices)
            logger.debug(f"Image features: area={features['area']}, perimeter={features['perimeter']}, is_convex={features['is_convex']}")
            return self.json_safe(features)
        except Exception as e:
            logger.warning(f"Error calculating image features: {e}")
            return {}
    
    # Duplicate definitions removed for clarity and to avoid context mismatch.
    def validate_features(self, features: dict) -> dict:
        """Validate key features and flag issues. Returns dict of issues found."""
        import numpy as np
        issues = {}
        # Area
        area = features.get('image_features', {}).get('area', None)
        if area is not None and (area <= 0 or not np.isfinite(area)):
            issues['area'] = area
        # Center of mass
        com = features.get('physics_features', {}).get('center_of_mass', None)
        if com is not None and (not isinstance(com, (list, tuple)) or len(com) != 2 or not all(np.isfinite(c) for c in com)):
            issues['center_of_mass'] = com
        # Stroke counts
        nline = features.get('physics_features', {}).get('num_straight_segments', None)
        narc = features.get('physics_features', {}).get('num_arcs', None)
        if nline is not None and nline < 0:
            issues['num_straight_segments'] = nline
        if narc is not None and narc < 0:
            issues['num_arcs'] = narc
        # Angular variance
        angvar = features.get('physics_features', {}).get('angular_variance', None)
        if angvar is not None and (angvar < 0 or angvar > 180):
            issues['angular_variance'] = angvar
        # Pattern regularity
        preg = features.get('composition_features', {}).get('pattern_regularity', None)
        if preg is not None and (preg < 0 or preg > 1):
            issues['pattern_regularity'] = preg
        return issues
    
    def _calculate_composition_features(self, strokes: List) -> Dict[str, Any]:
        """Calculate features about stroke composition and relationships. FIXED: Use actual modifiers from strokes."""
        if not strokes:
            return {}
        try:
            stroke_types = {}
            shape_modifiers = {}
            modifier_sequence = []
            # Sort strokes by index for deterministic pattern regularity
            strokes_sorted = sorted(strokes, key=lambda s: getattr(s, 'stroke_index', 0))
            for stroke in strokes_sorted:
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
                'composition_complexity': len(strokes) + len(shape_modifiers),
                'homogeneity_score': self._calculate_homogeneity(shape_modifiers),
                'pattern_regularity': self._calculate_pattern_regularity_from_modifiers(modifier_sequence)
            })
            return features
        except Exception as e:
            logger.warning(f"Error calculating composition features: {e}")
            return {}

    def _extract_modifier_from_stroke(self, stroke) -> str:
        """Extract the actual shape modifier from a stroke object, robustly, with debug logging."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f'Stroke type: {type(stroke)}, attributes: {dir(stroke)}')
        logger.debug(f'Raw command: {getattr(stroke, "raw_command", None)}')
        logger.debug(f'Shape modifier: {getattr(stroke, "shape_modifier", None)}')
        # Priority: attribute > raw_command > function_name > fallback
        if hasattr(stroke, 'shape_modifier'):
            smod = getattr(stroke, 'shape_modifier')
            if hasattr(smod, 'value'):
                if smod.value:
                    logger.debug(f"Extracted modifier from .shape_modifier.value: {smod.value}")
                    return str(smod.value)
            elif isinstance(smod, str) and smod:
                logger.debug(f"Extracted modifier from .shape_modifier: {smod}")
                return smod
        raw_command = getattr(stroke, 'raw_command', None)
        if raw_command and isinstance(raw_command, str):
            parts = raw_command.split('_')
            if len(parts) >= 2 and parts[1]:
                logger.debug(f"Extracted modifier from raw_command: {parts[1]}")
                return parts[1]
        function_name = getattr(stroke, 'function_name', None)
        if function_name and isinstance(function_name, str):
            fn_parts = function_name.split('_')
            if len(fn_parts) >= 2 and fn_parts[1]:
                logger.debug(f"Extracted modifier from function_name: {fn_parts[1]}")
                return fn_parts[1]
        logger.debug("Falling back to 'normal' modifier")
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
        """Calculate compactness (isoperimetric ratio) with correct formula and debug logging."""
        import logging
        logger = logging.getLogger(__name__)
        import numpy as np
        if perimeter == 0:
            logger.debug("Compactness: perimeter is zero, returning 0.0")
            return 0.0
        compactness = self.safe_divide(4 * np.pi * area, perimeter * perimeter)
        logger.debug(f"Compactness: area={area}, perimeter={perimeter}, compactness={compactness}")
        return self.json_safe(compactness)
    
    def _calculate_eccentricity(self, vertices: List[tuple]) -> float:
        """Calculate eccentricity as 1 - (min_eigenvalue / max_eigenvalue) from PCA."""
        import numpy as np
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 3:
            return 0.0
        try:
            points = np.array(vertices)
            centered = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered.T)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(np.abs(eigenvals))[::-1]
            if eigenvals[0] == 0:
                return 0.0
            return self.json_safe(1.0 - self.safe_divide(eigenvals[-1], eigenvals[0]))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Eccentricity error: {e}")
            return 0.0

    def _calculate_angular_variance(self, vertices: List[tuple]) -> float:
        """Calculate raw variance of interior angles (degrees) for polygon. Return NaN if <2 angles."""
        import numpy as np
        logger = logging.getLogger(__name__)
        vertices = self.ensure_vertex_list(vertices)
        n = len(vertices)
        if n < 3:
            logger.debug("Angular variance: not enough vertices, returning NaN")
            return float('nan')
        try:
            angles = []
            for i in range(n):
                p0 = np.array(vertices[i - 1])
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % n])
                v1 = p0 - p1
                v2 = p2 - p1
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                    angle = np.arccos(dot)
                    angle_deg = np.degrees(angle)
                    angles.append(angle_deg)
            if len(angles) < 2:
                logger.warning("Angular variance: insufficient angles, returning NaN")
                return float('nan')
            var = np.var(angles)
            logger.debug(f"Angular variance: angles={angles}, variance={var}")
            return self.json_safe(var)
        except Exception as e:
            logger.warning(f"Angular variance: error {e}")
            return float('nan')
    
    def _check_rotational_symmetry(self, vertices: List[tuple]) -> int:
        """Check rotational symmetry order using k-fold RMSE (k=2,3,4)."""
        return PhysicsInference.rotational_symmetry(vertices)

    def _calculate_irregularity(self, vertices: List[tuple]) -> float:
        """Calculate normalized mean absolute deviation from regular n-gon angle (0=regular, 1=irregular)."""
        import numpy as np
        vertices = self.ensure_vertex_list(vertices)
        n = len(vertices)
        if n < 3:
            return 0.0
        try:
            angles = []
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
            expected_angle = self.safe_divide((n - 2) * np.pi, n)
            mad = np.mean([abs(angle - expected_angle) for angle in angles])
            # Normalize: 0 = regular, 1 = max deviation (pi)
            norm_mad = min(1.0, mad / np.pi)
            return self.json_safe(norm_mad)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Irregularity error: {e}")
            return 0.0
    def _calculate_curvature_score(self, vertices: list) -> float:
        """Curvature score using angular variance from PhysicsInference."""
        return PhysicsInference.angular_variance(vertices)

    def _calculate_homogeneity(self, modifier_distribution: dict) -> float:
        """Calculate a simple homogeneity score: 1.0 if all modifiers are the same, lower otherwise (Gini impurity)."""
        total = sum(modifier_distribution.values())
        if total == 0:
            return 1.0
        probs = [v / total for v in modifier_distribution.values()]
        gini = 1.0 - sum(p ** 2 for p in probs)
        return 1.0 - gini  # 1.0 = homogeneous, 0 = maximally diverse
    
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
        # Robustly convert parameters to float, fallback to 0 if conversion fails
        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        radius = to_float(params.get('param1', 0))
        span_angle = to_float(params.get('param2', 0))
        end_angle = to_float(params.get('param3', 0))

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
        """Calculate variance of a list of values. For length 2, return squared diff. For length 1, return NaN and log."""
        import numpy as np
        logger = logging.getLogger(__name__)
        n = len(values)
        if n < 1:
            logger.warning("Variance: empty list, returning NaN")
            return float('nan')
        if n == 1:
            logger.warning("Variance: only one value, returning NaN")
            return float('nan')
        if n == 2:
            diff = values[1] - values[0]
            return diff * diff / 2.0
        mean = self.safe_divide(sum(values), n)
        return self.safe_divide(sum((x - mean) ** 2 for x in values), n)
    
    def _calculate_dominant_direction(self, line_features: List[Dict]) -> str:
        """Calculate the dominant direction of line strokes"""
        if not line_features:
            return 'none'
        
        directions = [f['line_direction'] for f in line_features]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        return max(direction_counts.items(), key=lambda x: x[1])[0] if direction_counts else 'none'
    
    def _calculate_image_features(self, vertices: List[tuple], strokes: List, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive image-level features with robust polygon recovery and improved metrics."""
        import numpy as np
        logger = logging.getLogger(__name__)
        vertices = self.ensure_vertex_list(vertices)
        if not vertices:
            logger.debug("Image features: no vertices, returning empty dict")

            return {}
        try:
            from shapely.geometry import Polygon
            poly = None
            try:
                poly = Polygon(vertices)
                if not poly.is_valid or poly.area == 0:
                    logger.debug("Image features: polygon invalid or zero area, applying buffer(0)")
                    poly = poly.buffer(0)
            except Exception as e:
                logger.debug(f"Image features: error in Polygon(vertices): {e}")
                poly = None
            if poly is None or not poly.is_valid or poly.is_empty:
                # fallback: convex hull
                try:
                    logger.debug("Image features: falling back to convex hull")
                    poly = Polygon(vertices).convex_hull
                except Exception as e:
                    logger.debug(f"Image features: error in convex hull: {e}")
                    poly = None
            features = {
                'bounding_box': geometry.get('bbox', {}),
                'centroid': geometry.get('centroid', [0.0, 0.0]),
                'width': geometry.get('width', 0.0),
                'height': geometry.get('height', 0.0),
                'area': PhysicsInference.area(poly) if poly else 0.0,
                'perimeter': self._calculate_perimeter(vertices),
            }
            # Aspect ratio: clip and flag outliers
            raw_ar = self.safe_divide(geometry.get('width', 1.0), geometry.get('height', 1.0), 1.0)
            ar = max(FLAGGING_THRESHOLDS['min_aspect_ratio'], min(raw_ar, FLAGGING_THRESHOLDS['max_aspect_ratio']))
            features['aspect_ratio'] = ar

            # Convexity ratio: robust, clamp to [0,1], handle degenerate
            if poly is None or poly.area == 0 or poly.convex_hull.area == 0:
                features['convexity_ratio'] = 0.0
            else:
                ratio = self.safe_divide(poly.area, poly.convex_hull.area)
                features['convexity_ratio'] = max(0.0, min(1.0, ratio))

            features['is_convex'] = PhysicsInference.is_convex(poly) if poly else False
            features['compactness'] = self._calculate_compactness(features['area'], features['perimeter'])
            features['eccentricity'] = self._calculate_eccentricity(vertices)

            # Symmetry and compactness: None if area or perimeter is zero
            if features['perimeter'] == 0 or features['area'] == 0:
                features['compactness'] = None
                features['symmetry_score'] = None
            else:
                features['symmetry_score'] = PhysicsInference.symmetry_score(vertices)

            features['horizontal_symmetry'] = self._check_horizontal_symmetry(vertices, poly)
            features['vertical_symmetry'] = self._check_vertical_symmetry(vertices, poly)
            features['rotational_symmetry'] = self._check_rotational_symmetry(vertices)

            # has_quadrangle: robust check for 4-vertex polygons
            if poly and hasattr(poly, 'exterior') and hasattr(poly.exterior, 'coords') and len(poly.exterior.coords)-1 == 4:
                features['has_quadrangle'] = True
            else:
                features['has_quadrangle'] = PhysicsInference.has_quadrangle(vertices)
            features['geometric_complexity'] = PhysicsInference.geometric_complexity(vertices)
            # Improved visual complexity: alpha*(V-3)/(Vmax-3) + (1-alpha)*(S-1)/(Smax-1)
            alpha, V_max, S_max = 0.5, 30, 10
            V, S = len(vertices), len(strokes)
            vcomp = alpha * self.safe_divide(V-3, V_max-3) + (1-alpha) * self.safe_divide(S-1, S_max-1)
            features['visual_complexity'] = max(0.0, vcomp)
            features['irregularity_score'] = self._calculate_irregularity(vertices)
            logger.debug(f"Image features: area={features['area']}, perimeter={features['perimeter']}, is_convex={features['is_convex']}")
            return self.json_safe(features)
        except Exception as e:
            logger.warning(f"Error calculating image features: {e}")
            return {}
    
    def _calculate_physics_features(self, vertices: List[tuple], centroid=None, strokes=None) -> Dict[str, Any]:
        """Calculate physics-based features using PhysicsInference. Accepts centroid override and strokes for correct counting. Uses correct center_of_mass and stroke counts."""
        import logging
        logger = logging.getLogger(__name__)
        if not vertices:
            logger.debug("Physics features: no vertices, returning empty dict")
            return {}
        try:
            poly = None
            try:
                poly = PhysicsInference.polygon_from_vertices(vertices)
            except Exception as e:
                logger.debug(f"Physics features: error in polygon_from_vertices: {e}")
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
                    logger.debug(f"Physics features: error in stroke counting: {e}")
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
                'curvature_score': self._calculate_curvature_score(vertices),
                'angular_variance': self._calculate_angular_variance(vertices),
                'edge_length_variance': self._calculate_edge_length_variance(vertices)
            }
            logger.debug(f"Physics features: center_of_mass={center_of_mass}, num_straight_segments={num_straight_segments}, num_arcs={num_arcs}")
            return features
        except Exception as e:
            logger.warning(f"Error calculating physics features: {e}")
            return {}
    def validate_features(self, features: dict) -> dict:
        """Validate key features and flag issues. Returns dict of issues found."""
        import numpy as np
        issues = {}
        # Area
        area = features.get('image_features', {}).get('area', None)
        if area is not None and (area <= 0 or not np.isfinite(area)):
            issues['area'] = area
        # Center of mass
        com = features.get('physics_features', {}).get('center_of_mass', None)
        if com is not None and (not isinstance(com, (list, tuple)) or len(com) != 2 or not all(np.isfinite(c) for c in com)):
            issues['center_of_mass'] = com
        # Stroke counts
        nline = features.get('physics_features', {}).get('num_straight_segments', None)
        narc = features.get('physics_features', {}).get('num_arcs', None)
        if nline is not None and nline < 0:
            issues['num_straight_segments'] = nline
        if narc is not None and narc < 0:
            issues['num_arcs'] = narc
        # Angular variance
        angvar = features.get('physics_features', {}).get('angular_variance', None)
        if angvar is not None and (angvar < 0 or angvar > 180):
            issues['angular_variance'] = angvar
        # Pattern regularity
        preg = features.get('composition_features', {}).get('pattern_regularity', None)
        if preg is not None and (preg < 0 or preg > 1):
            issues['pattern_regularity'] = preg
        return issues
    
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
                'composition_complexity': len(strokes) + len(shape_modifiers),
                'homogeneity_score': self._calculate_homogeneity(shape_modifiers),
                'pattern_regularity': self._calculate_pattern_regularity_from_modifiers(modifier_sequence)
            })
            return features
        except Exception as e:
            logger.warning(f"Error calculating composition features: {e}")
            return {}

    def _extract_modifier_from_stroke(self, stroke) -> str:
        """Extract the actual shape modifier from a stroke object, robustly, with debug logging."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f'Stroke type: {type(stroke)}, attributes: {dir(stroke)}')
        logger.debug(f'Raw command: {getattr(stroke, "raw_command", None)}')
        logger.debug(f'Shape modifier: {getattr(stroke, "shape_modifier", None)}')
        # Priority: attribute > raw_command > function_name > fallback
        if hasattr(stroke, 'shape_modifier'):
            smod = getattr(stroke, 'shape_modifier')
            if hasattr(smod, 'value'):
                if smod.value:
                    logger.debug(f"Extracted modifier from .shape_modifier.value: {smod.value}")
                    return str(smod.value)
            elif isinstance(smod, str) and smod:
                logger.debug(f"Extracted modifier from .shape_modifier: {smod}")
                return smod
        raw_command = getattr(stroke, 'raw_command', None)
        if raw_command and isinstance(raw_command, str):
            parts = raw_command.split('_')
            if len(parts) >= 2 and parts[1]:
                logger.debug(f"Extracted modifier from raw_command: {parts[1]}")
                return parts[1]
        function_name = getattr(stroke, 'function_name', None)
        if function_name and isinstance(function_name, str):
            fn_parts = function_name.split('_')
            if len(fn_parts) >= 2 and fn_parts[1]:
                logger.debug(f"Extracted modifier from function_name: {fn_parts[1]}")
                return fn_parts[1]
        logger.debug("Falling back to 'normal' modifier")
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
        """Calculate compactness (isoperimetric ratio) with correct formula and debug logging."""
        import logging
        logger = logging.getLogger(__name__)
        import numpy as np
        if perimeter == 0:
            logger.debug("Compactness: perimeter is zero, returning 0.0")
            return 0.0
        compactness = self.safe_divide(4 * np.pi * area, perimeter * perimeter)
        logger.debug(f"Compactness: area={area}, perimeter={perimeter}, compactness={compactness}")
        return self.json_safe(compactness)
    
    def _calculate_eccentricity(self, vertices: List[tuple]) -> float:
        """Calculate eccentricity as 1 - (min_eigenvalue / max_eigenvalue) from PCA."""
        import numpy as np
        vertices = self.ensure_vertex_list(vertices)
        if len(vertices) < 3:
            return 0.0
        try:
            points = np.array(vertices)
            centered = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered.T)
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(np.abs(eigenvals))[::-1]
            if eigenvals[0] == 0:
                return 0.0
            return self.json_safe(1.0 - self.safe_divide(eigenvals[-1], eigenvals[0]))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Eccentricity error: {e}")
            return 0.0

    def _calculate_angular_variance(self, vertices: List[tuple]) -> float:
        """Calculate raw variance of interior angles (degrees) for polygon. Return NaN if <2 angles."""
        import numpy as np
        logger = logging.getLogger(__name__)
        vertices = self.ensure_vertex_list(vertices)
        n = len(vertices)
        if n < 3:
            logger.debug("Angular variance: not enough vertices, returning NaN")
            return float('nan')
        try:
            angles = []
            for i in range(n):
                p0 = np.array(vertices[i - 1])
                p1 = np.array(vertices[i])
                p2 = np.array(vertices[(i + 1) % n])
                v1 = p0 - p1
                v2 = p2 - p1
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    dot = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                    angle = np.arccos(dot)
                    angle_deg = np.degrees(angle)
                    angles.append(angle_deg)
            if len(angles) < 2:
                logger.warning("Angular variance: insufficient angles, returning NaN")
                return float('nan')
            var = np.var(angles)
            logger.debug(f"Angular variance: angles={angles}, variance={var}")
            return self.json_safe(var)
        except Exception as e:
            logger.warning(f"Angular variance: error {e}")
            return float('nan')
    
    def _check_rotational_symmetry(self, vertices: List[tuple]) -> int:
        """Check rotational symmetry order using k-fold RMSE (k=2,3,4)."""
        return PhysicsInference.rotational_symmetry(vertices)

    def _calculate_irregularity(self, vertices: List[tuple]) -> float:
        """Calculate normalized mean absolute deviation from regular n-gon angle (0=regular, 1=irregular)."""
        import numpy as np
        vertices = self.ensure_vertex_list(vertices)
        n = len(vertices)
        if n < 3:
            return 0.0
        try:
            angles = []
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
            expected_angle = self.safe_divide((n - 2) * np.pi, n)
            mad = np.mean([abs(angle - expected_angle) for angle in angles])
            # Normalize: 0 = regular, 1 = max deviation (pi)
            norm_mad = min(1.0, mad / np.pi)
            return self.json_safe(norm_mad)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Irregularity error: {e}")
            return 0.0

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