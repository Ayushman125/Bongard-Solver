import math
import logging
from typing import Dict, List, Any, Optional
from src.physics_inference import PhysicsInference
from bongard.bongard import BongardImage
from src.Derive_labels.shape_utils import safe_divide, _calculate_dominant_direction, _calculate_variance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_action_type_prefixes(problems_data):
    logger.info(f"[extract_action_type_prefixes] INPUT: problems_data keys={list(problems_data.keys())}")
    """
    Use BongardImage.import_from_action_string_list to robustly extract all unique action type prefixes from the dataset.
    This mirrors hybrid.py's handling and avoids information loss from naive string splitting.
    """
    prefixes = set()
    for problem_data in problems_data.values():
        if not (isinstance(problem_data, list) and len(problem_data) == 2):
            logger.info(f"[extract_action_type_prefixes] Skipping non-standard problem_data: {problem_data}")
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
    logger.info(f"[extract_action_type_prefixes] OUTPUT: {prefixes}")
    return prefixes

def _extract_stroke_vertices(stroke, stroke_index, all_vertices, bongard_image=None):
    """Robustly extract the full set of vertices for a stroke, not just endpoints."""
    import numpy as np
    logger = logging.getLogger(__name__)
    logger.info(f"[_extract_stroke_vertices] INPUT: stroke_index={stroke_index}, stroke={stroke}, all_vertices={all_vertices}")
    # --- FIX: Use NVLabs API correctly ---
    logger.info(f"[_extract_stroke_vertices] INPUT: stroke_index={stroke_index}, stroke={stroke}, all_vertices={all_vertices}")
    # Try to use BongardImage.one_stroke_shapes[stroke_index].vertices if available
    if bongard_image and hasattr(bongard_image, 'one_stroke_shapes'):
        try:
            for shape_idx, shape in enumerate(bongard_image.one_stroke_shapes):
                if hasattr(shape, 'basic_actions') and stroke_index < len(bongard_image.one_stroke_shapes):
                    if shape_idx == stroke_index:
                        verts = getattr(shape, 'vertices', None)
                        if verts and len(verts) > 2:
                            logger.info(f"[_extract_stroke_vertices] Using NVLabs vertices: {verts}")
                            logger.info(f"[_extract_stroke_vertices] OUTPUT: {verts}")
                            return verts
        except Exception as e:
            logger.warning(f"[_extract_stroke_vertices] NVLabs vertex extraction failed: {e}")
    # Fallbacks as before
    # 1. Use stroke.vertices if available and has >2 points
    if hasattr(stroke, 'vertices') and stroke.vertices and len(stroke.vertices) > 2:
        logger.info(f"[_extract_stroke_vertices] Using stroke.vertices: {stroke.vertices}")
        logger.info(f"[_extract_stroke_vertices] OUTPUT: {stroke.vertices}")
        return stroke.vertices
    # 2. Robust arc interpolation if arc parameters are available
    stype = getattr(stroke, 'stroke_type', None)
    if hasattr(stype, 'value'):
        stype = stype.value
    if stype is None and hasattr(stroke, 'function_name'):
        stype = str(stroke.function_name).split('_')[0]
    if stype and 'arc' in str(stype).lower():
        params = getattr(stroke, 'parameters', None) or {}
        cx = params.get('center_x') or params.get('cx') or params.get('param4')
        cy = params.get('center_y') or params.get('cy') or params.get('param5')
        radius = params.get('radius') or params.get('param1')
        start_angle = params.get('start_angle') or params.get('param2')
        end_angle = params.get('end_angle') or params.get('param3')
        # Fallback: try to parse from raw_command
        if None in [cx, cy, radius, start_angle, end_angle]:
            raw_command = getattr(stroke, 'raw_command', None)
            if raw_command and isinstance(raw_command, str):
                parts = raw_command.split('_')
                if len(parts) >= 4:
                    try:
                        radius = float(parts[2])
                        span = float(parts[3])
                        # If span is normalized, convert to degrees
                        if 0 <= span <= 1:
                            span_angle = span * 360
                        else:
                            span_angle = span
                        start_angle = 0
                        end_angle = span_angle
                        cx, cy = 0.5, 0.5
                    except Exception:
                        pass
        try:
            radius = float(radius)
            cx = float(cx)
            cy = float(cy)
            start_angle = float(start_angle)
            end_angle = float(end_angle)
            num_points = 24
            theta1 = np.deg2rad(start_angle)
            theta2 = np.deg2rad(end_angle)
            if theta2 < theta1:
                theta2 += 2 * np.pi
            thetas = np.linspace(theta1, theta2, num_points)
            arc_points = [(cx + radius * np.cos(t), cy + radius * np.sin(t)) for t in thetas]
            logger.info(f"[_extract_stroke_vertices] Interpolated arc points: {arc_points}")
            logger.info(f"[_extract_stroke_vertices] OUTPUT: {arc_points}")
            return arc_points
        except Exception as e:
            logger.warning(f"[_extract_stroke_vertices] Arc interpolation failed: {e}")
    # 3. For polylines, use all available points
    if hasattr(stroke, 'polyline_points') and stroke.polyline_points and len(stroke.polyline_points) > 2:
        logger.info(f"[_extract_stroke_vertices] Using stroke.polyline_points: {stroke.polyline_points}")
        logger.info(f"[_extract_stroke_vertices] OUTPUT: {stroke.polyline_points}")
        return stroke.polyline_points
    # 4. Fallback: extract from shape_vertices using stroke boundaries (legacy)
    if all_vertices and len(all_vertices) > stroke_index + 2:
        logger.info(f"[_extract_stroke_vertices] Using all_vertices slice: {all_vertices[stroke_index:stroke_index+2]}")
        logger.info(f"[_extract_stroke_vertices] OUTPUT: {all_vertices[stroke_index:stroke_index+2]}")
        return all_vertices[stroke_index:stroke_index+2]
    # 5. Last resort: use world coordinates if available
    if hasattr(stroke, 'get_world_coordinates') and callable(stroke.get_world_coordinates):
        verts = stroke.get_world_coordinates()
        if verts and len(verts) > 2:
            logger.info(f"[_extract_stroke_vertices] Using get_world_coordinates: {verts}")
            logger.info(f"[_extract_stroke_vertices] OUTPUT: {verts}")
            return verts
    # 6. Fallback: try to synthesize from command string
    raw_command = getattr(stroke, 'raw_command', None)
    if raw_command:
        verts = _vertices_from_command(raw_command, stroke_index)
        if verts and len(verts) > 1:
            logger.info(f"[_extract_stroke_vertices] Synthesized from command: {verts}")
            logger.info(f"[_extract_stroke_vertices] OUTPUT: {verts}")
            return verts
    logger.warning(f"[_extract_stroke_vertices] Could not robustly extract vertices for stroke {stroke_index} (type={stype}). Returning empty list.")
    logger.info(f"[_extract_stroke_vertices] OUTPUT: []")
    return []

def _vertices_from_command(command, stroke_index):
        logger.info(f"[_vertices_from_command] INPUT: command={command}, stroke_index={stroke_index}")
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
                    logger.info(f"[_vertices_from_command] OUTPUT: {[start, (end_x, end_y)]}")
                    return [start, (end_x, end_y)]
        except Exception as e:
            logging.getLogger(__name__).debug(f"Failed to parse command {command}: {e}")
        logger.info(f"[_vertices_from_command] Fallback: returning []")
        return []

def _calculate_stroke_specific_features(stroke, stroke_index: int, stroke_type_val=None, shape_modifier_val=None, parameters=None, bongard_image=None) -> Dict[str, Any]:
    logger.info(f"[_calculate_stroke_specific_features] Called with stroke={stroke}, stroke_index={stroke_index}, stroke_type_val={stroke_type_val}, shape_modifier_val={shape_modifier_val}, parameters={parameters}")
    """Calculate features specific to stroke type and shape modifier, using robust geometric/physics formulas."""
    # Use module-level logger only (fix UnboundLocalError)
    logger.info(f"[_calculate_stroke_specific_features] INPUTS: stroke_index={stroke_index}, stroke_type_val={stroke_type_val}, shape_modifier_val={shape_modifier_val}, parameters={parameters}")
    features = {'stroke_index': stroke_index}
    stype = stroke_type_val or type(stroke).__name__.replace('Action', '').lower()
    smod = shape_modifier_val or 'normal'
    params = parameters or {}
    # Always try to use full NVLabs vertices if available
    verts = _extract_stroke_vertices(stroke, stroke_index, None, bongard_image=bongard_image)
    logger.info(f"[_calculate_stroke_specific_features] verts: {verts}")
    # If not enough points for arcs, try to interpolate
    stype_lower = stype.lower() if stype else ''
    if verts and len(verts) <= 2 and 'arc' in stype_lower:
        logger.info(f"[_calculate_stroke_specific_features] Only {len(verts)} points for arc, attempting interpolation.")
        # Try to interpolate arc points using parameters or command string
        import numpy as np
        params = parameters or getattr(stroke, 'parameters', {}) or {}
        cx = params.get('center_x') or params.get('cx') or params.get('param4')
        cy = params.get('center_y') or params.get('cy') or params.get('param5')
        radius = params.get('radius') or params.get('param1')
        start_angle = params.get('start_angle') or params.get('param2')
        end_angle = params.get('end_angle') or params.get('param3')
        # Fallback: try to parse from raw_command
        if None in [cx, cy, radius, start_angle, end_angle]:
            raw_command = getattr(stroke, 'raw_command', None)
            if raw_command and isinstance(raw_command, str):
                parts = raw_command.split('_')
                if len(parts) >= 4:
                    try:
                        radius = float(parts[2])
                        start_angle = float(parts[3])
                        end_angle = float(parts[4]) if len(parts) > 4 else start_angle + 90
                        cx, cy = 0.5, 0.5  # fallback center
                        logger.info(f"[_calculate_stroke_specific_features] Fallback: arc params from raw_command: radius={radius}, start_angle={start_angle}, end_angle={end_angle}, cx={cx}, cy={cy}")
                    except Exception:
                        pass
        try:
            radius = float(radius)
            cx = float(cx)
            cy = float(cy)
            start_angle = float(start_angle)
            end_angle = float(end_angle)
            num_points = 8  # minimum for geometric calculation
            theta1 = np.deg2rad(start_angle)
            theta2 = np.deg2rad(end_angle)
            if theta2 < theta1:
                theta2 += 2 * np.pi
            thetas = np.linspace(theta1, theta2, num_points)
            verts = [(cx + radius * np.cos(t), cy + radius * np.sin(t)) for t in thetas]
            logger.info(f"[_calculate_stroke_specific_features] Interpolated arc points for geometric features: {verts}")
        except Exception as e:
            logger.warning(f"[_calculate_stroke_specific_features] Arc interpolation failed: {e}")
    # Only compute geometric features if there are enough points
    if verts and len(verts) >= 3:
        features['angular_variance'] = PhysicsInference.robust_angular_variance(verts)
        features['curvature_score'] = PhysicsInference.robust_curvature(verts)
        features['moment_of_inertia'] = PhysicsInference.robust_moment_of_inertia(verts, stype, params)
        logger.info(f"[_calculate_stroke_specific_features] OUTPUT: angular_variance={features['angular_variance']}, curvature_score={features['curvature_score']}, moment_of_inertia={features['moment_of_inertia']}")
    else:
        features['angular_variance'] = None
        features['curvature_score'] = None
        features['moment_of_inertia'] = None
        logger.warning(f"[_calculate_stroke_specific_features] Insufficient vertices for geometric features (stroke_index={stroke_index}). verts: {verts}")
        logger.info(f"[_calculate_stroke_specific_features] OUTPUT: angular_variance=None, curvature_score=None, moment_of_inertia=None")
    # For line strokes, add line_length, line_angle, etc.
    if stype == 'line':
        features.update(_calculate_line_specific_features_from_params(params))
    elif stype == 'arc':
        features.update(_calculate_arc_specific_features_from_params(params, stroke=stroke))
    features.update(_calculate_shape_modifier_features_from_val(smod))
    logger.info(f"[_calculate_stroke_specific_features] FINAL OUTPUT features: {features}")
    return features

def _calculate_line_specific_features_from_params(params: dict) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)
        logger.info(f"[_calculate_line_specific_features_from_params] INPUT: params={params}")
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
            logger.info(f"[_calculate_line_specific_features_from_params] OUTPUT: length={length}, angle={angle}, direction={direction}")
        return {
            'line_length': length,
            'line_angle': angle,
            'line_length_normalized': length_norm,
            'line_angle_normalized': (angle % 1.0),
            'line_direction': direction,
            'line_is_short': PhysicsInference.is_short_line(length, diag),
            'line_is_long': PhysicsInference.is_long_line(length, diag)
        }

def _calculate_arc_specific_features_from_params(params: dict, stroke=None) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"[_calculate_arc_specific_features_from_params] INPUT: params={params}")
    # Prefer stroke.arc_radius and stroke.arc_angle if available
    radius = getattr(stroke, 'arc_radius', None) if stroke is not None else None
    span = getattr(stroke, 'arc_angle', None) if stroke is not None else None
    if radius is None:
        radius = float(params.get('param1', 0))
    if span is None:
        span = float(params.get('param2', 0))
    try:
        end_angle = float(params.get('param3', span))
    except Exception:
        end_angle = span
    span_angle = span
    arc_length = abs(span_angle) * radius * safe_divide(math.pi, 180) if radius > 0 else 0
    is_major_arc = abs(span_angle) > 180
    is_full_circle = abs(span_angle) >= 350
    logger.info(f"[_calculate_arc_specific_features_from_params] OUTPUT: radius={radius}, span_angle={span_angle}, arc_length={arc_length}, is_major_arc={is_major_arc}, is_full_circle={is_full_circle}")
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
    logger = logging.getLogger(__name__)
    logger.info(f"[_calculate_shape_modifier_features_from_val] INPUT: modifier={modifier}")
    """Calculate features based on shape modifier string value (not from action object), using real geometry if available."""
    from src.Derive_labels.shape_utils import calculate_geometry
    base_features = {
        'shape_modifier': modifier,
        'is_normal': modifier == 'normal',
        'is_geometric': modifier in ['circle', 'square', 'triangle'],
        'is_pattern': modifier == 'zigzag'
    }
    # Compute geometric complexity from geometry if possible
    import inspect
    frame = inspect.currentframe().f_back
    vertices = frame.f_locals.get('verts', None) or frame.f_locals.get('vertices', None)
    geom = calculate_geometry(vertices) if vertices else None
    if geom:
        num_vertices = len(vertices) if vertices else 0
        convexity = geom.get('convexity_ratio', 1.0)
        compactness = None
        area = geom.get('area', None)
        perimeter = geom.get('perimeter', None)
        if area and perimeter:
            from src.Derive_labels.shape_utils import _calculate_compactness
            compactness = _calculate_compactness(area, perimeter)
        complexity = num_vertices * (1.0 / convexity if convexity > 0 else 1.0)
        if compactness:
            complexity *= (1.0 / compactness if compactness > 0 else 1.0)
        base_features['geometric_complexity'] = round(complexity, 3)
        base_features['num_vertices'] = num_vertices
        base_features['convexity_ratio'] = convexity
        if compactness:
            base_features['compactness'] = compactness
        logger.info(f"[_calculate_shape_modifier_features_from_val] Calculated geometry-based features: {base_features}")
    # Add shape-specific tags
    if modifier == 'triangle':
        base_features['has_sharp_angles'] = True
    elif modifier == 'square':
        base_features['has_right_angles'] = True
    elif modifier == 'circle':
        base_features['has_curved_edges'] = True
    elif modifier == 'zigzag':
        base_features['pattern_complexity'] = 'high'
        base_features['has_repetitive_pattern'] = True
    else:  # normal or unknown
        base_features['is_simple'] = True
    logger.info(f"[_calculate_shape_modifier_features_from_val] OUTPUT: {base_features}")
    return base_features

def _calculate_stroke_type_differentiated_features(stroke_type_features: Dict, strokes: List) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"[_calculate_stroke_type_differentiated_features] INPUT: stroke_type_features keys={list(stroke_type_features.keys())}, strokes count={len(strokes)}")
    """Calculate features that differentiate between stroke types"""
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

    logger.info(f"[_calculate_stroke_type_differentiated_features] OUTPUT: {features}")
    logger.debug(f"[_calculate_stroke_type_differentiated_features] OUTPUT: {features}")
    return features

def _extract_modifier_from_stroke(stroke) -> str:
    logger = logging.getLogger(__name__)
    logger.info(f"[_extract_modifier_from_stroke] INPUT: type={type(stroke)}, attributes={dir(stroke)}")
    """Extract the actual shape modifier from a stroke object, using geometric and semantic analysis with debug logging."""
    from src.Derive_labels.shape_utils import calculate_geometry, _calculate_compactness
    logger = logging.getLogger(__name__)
    logger.debug(f"[_extract_modifier_from_stroke] INPUT: type={type(stroke)}, attributes={dir(stroke)}")
    # Try geometric analysis first
    verts = None
    if hasattr(stroke, 'vertices') and stroke.vertices and len(stroke.vertices) > 2:
        verts = stroke.vertices
    elif hasattr(stroke, 'polyline_points') and stroke.polyline_points and len(stroke.polyline_points) > 2:
        verts = stroke.polyline_points
    elif hasattr(stroke, 'get_world_coordinates') and callable(stroke.get_world_coordinates):
        verts = stroke.get_world_coordinates()
    # If we have enough vertices, analyze geometry
    if verts and len(verts) > 2:
        geom = calculate_geometry(verts)
        num_vertices = len(verts)
        convexity = geom.get('convexity_ratio', 1.0)
        compactness = None
        area = geom.get('area', None)
        perimeter = geom.get('perimeter', None)
        if area and perimeter:
            compactness = _calculate_compactness(area, perimeter)
        # Triangle: 3 vertices, high convexity, compactness ~0.6
        if num_vertices == 3 and convexity > 0.95:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: triangle")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: triangle (geometry)")
            return 'triangle'
        # Square: 4 vertices, right angles, high convexity, compactness ~0.785
        if num_vertices == 4 and convexity > 0.95 and compactness and 0.7 < compactness < 0.85:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: square")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: square (geometry)")
            return 'square'
        # Circle: many vertices, high compactness, high convexity
        if num_vertices > 6 and compactness and compactness > 0.85 and convexity > 0.95:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: circle")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: circle (geometry)")
            return 'circle'
        # Zigzag: high edge variance, low compactness
        edge_var = geom.get('edge_length_variance', 0.0)
        if edge_var and edge_var > 0.1 and compactness and compactness < 0.5:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: zigzag")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: zigzag (geometry)")
            return 'zigzag'
    # Fallback: string-based extraction
    if hasattr(stroke, 'shape_modifier'):
        smod = getattr(stroke, 'shape_modifier')
        if hasattr(smod, 'value'):
            if smod.value:
                logger.debug(f"[_extract_modifier_from_stroke] Extracted modifier from .shape_modifier.value: {smod.value}")
                logger.info(f"[_extract_modifier_from_stroke] OUTPUT: {smod.value} (shape_modifier.value)")
                return str(smod.value)
        elif isinstance(smod, str) and smod:
            logger.debug(f"[_extract_modifier_from_stroke] Extracted modifier from .shape_modifier: {smod}")
            logger.info(f"[_extract_modifier_from_stroke] OUTPUT: {smod} (shape_modifier)")
            return smod
    raw_command = getattr(stroke, 'raw_command', None)
    if raw_command and isinstance(raw_command, str):
        parts = raw_command.split('_')
        if len(parts) >= 2 and parts[1]:
            logger.debug(f"[_extract_modifier_from_stroke] Extracted modifier from raw_command: {parts[1]}")
            logger.info(f"[_extract_modifier_from_stroke] OUTPUT: {parts[1]} (raw_command)")
            return parts[1]
    function_name = getattr(stroke, 'function_name', None)
    if function_name and isinstance(function_name, str):
        fn_parts = function_name.split('_')
        if len(fn_parts) >= 2 and fn_parts[1]:
            logger.debug(f"[_extract_modifier_from_stroke] Extracted modifier from function_name: {fn_parts[1]}")
            logger.info(f"[_extract_modifier_from_stroke] OUTPUT: {fn_parts[1]} (function_name)")
            return fn_parts[1]
    logger.debug("[_extract_modifier_from_stroke] Fallback to 'normal' modifier")
    logger.info("[_extract_modifier_from_stroke] OUTPUT: normal (fallback)")
    return 'normal'


