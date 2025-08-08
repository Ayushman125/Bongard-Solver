import math
import logging
from typing import Dict, List, Any, Optional
from src.physics_inference import PhysicsInference
from bongard.bongard import BongardImage
from src.Derive_labels.shape_utils import safe_divide, _calculate_dominant_direction, _calculate_variance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_action_type_prefixes(problems_data):
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

def _extract_stroke_vertices(stroke, stroke_index, all_vertices):
        """Extract vertices for individual stroke from overall shape vertices."""
        try:
            # Method 1: Direct vertices from stroke
            if hasattr(stroke, 'vertices') and stroke.vertices:
                return stroke.vertices
            # Method 2: Calculate from stroke parameters
            if hasattr(stroke, 'raw_command'):
                return _vertices_from_command(stroke.raw_command, stroke_index)
            # Method 3: Segment from overall vertices
            if all_vertices and len(all_vertices) > stroke_index + 1:
                return [all_vertices[stroke_index], all_vertices[stroke_index + 1]]
            return []
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to extract vertices for stroke {stroke_index}: {e}")
            return []

def _vertices_from_command(command, stroke_index):
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

def _calculate_stroke_specific_features(stroke, stroke_index: int, stroke_type_val=None, shape_modifier_val=None, parameters=None) -> Dict[str, Any]:
        """Calculate features specific to stroke type and shape modifier, using robust geometric/physics formulas."""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_calculate_stroke_specific_features] INPUTS: stroke_index={stroke_index}, stroke_type_val={stroke_type_val}, shape_modifier_val={shape_modifier_val}, parameters={parameters}")
        features = {'stroke_index': stroke_index}
        stype = stroke_type_val or type(stroke).__name__.replace('Action', '').lower()
        smod = shape_modifier_val or 'normal'
        params = parameters or {}
        verts = _extract_stroke_vertices(stroke, stroke_index, None)
        logger.debug(f"[_calculate_stroke_specific_features] verts: {verts}")
        # Angular variance
        features['angular_variance'] = PhysicsInference.robust_angular_variance(verts)
        # Curvature
        features['curvature_score'] = PhysicsInference.robust_curvature(verts)
        # Moment of inertia
        features['moment_of_inertia'] = PhysicsInference.robust_moment_of_inertia(verts, stype, params)
        # For line strokes, add line_length, line_angle, etc.
        if stype == 'line':
            features.update(_calculate_line_specific_features_from_params(params))
        elif stype == 'arc':
            features.update(_calculate_arc_specific_features_from_params(params))
        features.update(_calculate_shape_modifier_features_from_val(smod))
        logger.debug(f"[_calculate_stroke_specific_features] OUTPUT: {features}")
        return features

def _calculate_line_specific_features_from_params(params: dict) -> Dict[str, Any]:
        length = params.get('param1', 0)
        angle = params.get('param2', 0)
        diag = math.sqrt(2)
        length_norm = safe_divide(length, diag)
        # Angle in degrees in [-180, 180]
        angle_deg = ((angle % 1.0) * 360.0)
        angle_deg = ((angle_deg + 180) % 360) - 180
        # ±10° for horizontal/vertical, else diagonal
        if abs(angle_deg) <= 10:
            direction = 'horizontal'
        elif abs(angle_deg - 90) <= 10 or abs(angle_deg + 90) <= 10:
            direction = 'vertical'
        else:
            direction = 'diagonal'
        return {
            'line_length': length,
            'line_angle': angle,
            'line_length_normalized': length_norm,
            'line_angle_normalized': (angle % 1.0),
            'line_direction': direction,
            'line_is_short': PhysicsInference.is_short_line(length, diag),
            'line_is_long': PhysicsInference.is_long_line(length, diag)
        }

def _calculate_arc_specific_features_from_params(params: dict) -> Dict[str, Any]:
        # Robustly convert parameters to float, fallback to 0 if conversion fails
        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        radius = to_float(params.get('param1', 0))
        span_angle = to_float(params.get('param2', 0))
        end_angle = to_float(params.get('param3', 0))

        arc_length = abs(span_angle) * radius * safe_divide(math.pi, 180) if radius > 0 else 0
        is_major_arc = abs(span_angle) > 180
        is_full_circle = abs(span_angle) >= 350
        return {
            'arc_radius': radius,
            'arc_span_angle': span_angle,
            'arc_end_angle': end_angle,
            'arc_length': arc_length,
        'arc_curvature': safe_divide(1.0, max(radius, 1e-6)),
            'arc_is_major': is_major_arc,
            'arc_is_full_circle': is_full_circle,
            'arc_direction': 'clockwise' if span_angle < 0 else 'counterclockwise',
        'arc_radius_normalized': safe_divide(min(radius, 2.0), 2.0),
        'arc_span_normalized': safe_divide(abs(span_angle), 360.0),
            'arc_is_small': radius < 0.3,
            'arc_is_large': radius > 1.5,
            'arc_is_tight': abs(span_angle) > 270,
            'arc_is_gentle': abs(span_angle) < 90
        }
def _calculate_shape_modifier_features_from_val(modifier: str) -> Dict[str, Any]:
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

def _calculate_stroke_type_differentiated_features(stroke_type_features: Dict, strokes: List) -> Dict[str, Any]:
        """Calculate features that differentiate between stroke types"""
        logger = logging.getLogger(__name__)
        logger.debug(f"[_calculate_stroke_type_differentiated_features] INPUTS: stroke_type_features keys: {list(stroke_type_features.keys())}, strokes count: {len(strokes)}")
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
            'line_ratio': safe_divide(num_lines, max(total_strokes, 1)),
            'arc_ratio': safe_divide(num_arcs, max(total_strokes, 1)),
                'stroke_diversity': 1 if num_lines > 0 and num_arcs > 0 else 0
            }
        }
        
        # Line-specific aggregate features
        if line_features:
            line_lengths = [f['line_length'] for f in line_features]
            line_angles = [f['line_angle'] for f in line_features]
            
            features['line_aggregate'] = {
                'total_line_length': sum(line_lengths),
            'avg_line_length': safe_divide(sum(line_lengths), len(line_lengths)),
            'line_length_variance': _calculate_variance(line_lengths),
            'line_angle_variance': _calculate_variance(line_angles),
                'has_short_lines': any(f['line_is_short'] for f in line_features),
                'has_long_lines': any(f['line_is_long'] for f in line_features),
            'dominant_direction': _calculate_dominant_direction(line_features)
            }
        
        # Arc-specific aggregate features  
        if arc_features:
            arc_radii = [f['arc_radius'] for f in arc_features]
            arc_spans = [f['arc_span_angle'] for f in arc_features]
            arc_lengths = [f['arc_length'] for f in arc_features]
            
            features['arc_aggregate'] = {
                'total_arc_length': sum(arc_lengths),
            'avg_arc_radius': safe_divide(sum(arc_radii), len(arc_radii)),
            'avg_arc_span': safe_divide(sum(arc_spans), len(arc_spans)),
            'arc_radius_variance': _calculate_variance(arc_radii),
            'arc_span_variance': _calculate_variance(arc_spans),
                'total_curvature': sum(f['arc_curvature'] for f in arc_features),
                'has_full_circles': any(f['arc_is_full_circle'] for f in arc_features),
                'has_major_arcs': any(f['arc_is_major'] for f in arc_features),
                'curvature_complexity': len([f for f in arc_features if f['arc_curvature'] > 1.0])
            }
        
        logger.debug(f"[_calculate_stroke_type_differentiated_features] OUTPUT: {features}")
        return features

def _extract_modifier_from_stroke(stroke) -> str:
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


