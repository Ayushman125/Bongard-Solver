"""
IMPORTANT PIPELINE NOTE:
All feature extraction and stroke vertex extraction in this module should use the painter-generated BongardImage context.
Always pass the bongard_image argument to _extract_stroke_vertices and _calculate_stroke_specific_features.
Only use fallback/synthesized vertices if bongard_image.one_stroke_shapes[stroke_index].vertices is missing or too short, and log a warning if you do.
"""

import math
from src.Derive_labels.shape_utils import calculate_geometry_consistent, _calculate_compactness, open_stroke_convexity, validate_features
import logging
from typing import Dict, List, Any, Optional
import numpy as np
    # Import at top of function to avoid indentation error
from src.physics_inference import PhysicsInference
from bongard.bongard import BongardImage
from src.Derive_labels.shape_utils import safe_divide, _calculate_dominant_direction, _calculate_variance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_shape_segments(basic_actions, shape_vertices):
    """Partition shape_vertices into contiguous segments for each stroke."""
    # Estimate segment lengths by uniform partitioning (fallback if no metadata)
    n = len(basic_actions)
    m = len(shape_vertices)
    if n < 1 or m < 2:
        return [(0, m-1)] * n
    # If closed polygon, ignore last vertex for partitioning
    closed = (shape_vertices[0] == shape_vertices[-1])
    m_eff = m-1 if closed else m
    seg_len = m_eff // n
    segments = []
    idx = 0
    for i in range(n):
        start = idx
        end = idx + seg_len
        if i == n-1:
            end = m_eff
        segments.append((start, end))
        idx = end
    return segments

def extract_stroke_features_from_shapes(bongard_image, problem_id=None):
    """
    For each shape in bongard_image.one_stroke_shapes, iterate over its actions/strokes.
    Extract features for each stroke within its shape context, logging group membership and mismatches.
    Returns a list of dicts: [{shape_index, stroke_index, stroke_command, features, group_info, ...}, ...]
    """
    logging.info(f"[extract_stroke_features_from_shapes] INPUT bongard_image type: {type(bongard_image)} | problem_id={problem_id}")
    logging.info(f"[extract_stroke_features_from_shapes] INPUT bongard_image.one_stroke_shapes: {getattr(bongard_image, 'one_stroke_shapes', None)}")
    results = []
    if not hasattr(bongard_image, 'one_stroke_shapes') or not bongard_image.one_stroke_shapes:
        logging.warning(f"[extract_stroke_features_from_shapes] No shapes found in BongardImage for problem_id={problem_id}")
        logging.info(f"[extract_stroke_features_from_shapes] OUTPUT: {results}")
        return results
    shapes = bongard_image.one_stroke_shapes
    total_shapes = len(shapes)
    total_strokes = sum(len(getattr(shape, 'actions', getattr(shape, 'strokes', []))) for shape in shapes)
    logging.info(f"[extract_stroke_features_from_shapes] problem_id={problem_id} | num_shapes={total_shapes} | total_strokes={total_strokes}")
    logging.info(f"[extract_stroke_features_from_shapes] PATCH: shapes={shapes}")

    # --- Aggregate positive and negative strokes ---
    positive_strokes = []
    negative_strokes = []
    stroke_results = []

    for shape_idx, shape in enumerate(shapes):
        actions = getattr(shape, 'actions', None)
        if actions is None:
            actions = getattr(shape, 'basic_actions', None)
            logging.warning(f"[extract_stroke_features_from_shapes] Shape {shape_idx} missing 'actions', using 'basic_actions': {actions}")
        else:
            logging.info(f"[extract_stroke_features_from_shapes] Shape {shape_idx} actions: {actions}")
        logging.info(f"[extract_stroke_features_from_shapes] Shape {shape_idx}: {shape}")
        start_coords = getattr(shape, 'start_coordinates', None)
        if start_coords is not None:
            logging.info(f"[extract_stroke_features_from_shapes] Shape {shape_idx} start_coordinates: {start_coords}")
        geometry = getattr(shape, 'geometry', None)
        if geometry is not None:
            logging.info(f"[extract_stroke_features_from_shapes] Shape {shape_idx} geometry: {geometry}")
            if 'width' not in geometry or 'height' not in geometry:
                logging.error(f"[extract_stroke_features_from_shapes] Shape {shape_idx} geometry missing width/height: {geometry}")
        if not actions:
            logging.error(f"[extract_stroke_features_from_shapes] Shape {shape_idx} has no actions! Shape: {shape}")
            continue
        logging.info(f"[extract_stroke_features_from_shapes] Shape {shape_idx}: num_actions={len(actions)} | actions={actions}")
        shape_vertices = getattr(shape, 'vertices', None)
        from src.Derive_labels.shape_utils import ensure_vertex_list
        shape_vertices = ensure_vertex_list(shape_vertices)
        if shape_vertices and len(shape_vertices) >= 3:
            deduped = [shape_vertices[0]]
            for pt in shape_vertices[1:]:
                if pt != deduped[-1]:
                    deduped.append(pt)
            shape_vertices = deduped
            if shape_vertices[0] != shape_vertices[-1]:
                shape_vertices.append(shape_vertices[0])
            logging.info(f"[extract_stroke_features_from_shapes] PATCH: Validated/corrected shape_vertices for shape {shape_idx}: {shape_vertices}")
        else:
            logging.warning(f"[extract_stroke_features_from_shapes] PATCH: Insufficient vertices for shape {shape_idx}: {shape_vertices}")
        segments = compute_shape_segments(actions, shape_vertices)
        stroke_lengths = []
        def sanitize_feature_value(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except Exception:
                return default

        is_positive = getattr(shape, 'is_positive', None)
        for stroke_idx, stroke in enumerate(actions):
            logging.info(f"[extract_stroke_features_from_shapes] shape_idx={shape_idx} | stroke_idx={stroke_idx} | stroke={stroke}")
            try:
                seg_start, seg_end = segments[stroke_idx]
                stroke_vertices = shape_vertices[seg_start:seg_end+1]
                logging.info(f"[extract_stroke_features_from_shapes] PATCH: Input stroke_vertices to _calculate_stroke_specific_features: {stroke_vertices}")
                features = _calculate_stroke_specific_features(stroke, stroke_idx, bongard_image=bongard_image, parent_shape_vertices=stroke_vertices, shape_obj=shape)
                for key in ['area', 'compactness', 'convexity_ratio', 'stroke_length', 'avg_stroke_length', 'geom_complexity', 'arc_curvature_score', 'robust_curvature', 'robust_angular_variance', 'visual_complexity']:
                    if key in features:
                        features[key] = sanitize_feature_value(features[key])
                import numpy as np
                arr = np.array(stroke_vertices)
                if len(arr) >= 2:
                    length = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))
                else:
                    length = 0.0
                stroke_lengths.append(sanitize_feature_value(length))
                features['stroke_length'] = sanitize_feature_value(length)
                logging.info(f"[extract_stroke_features_from_shapes] PATCH: Output features for shape {shape_idx}, stroke {stroke_idx}: {features}")
            except Exception as e:
                logging.error(f"[extract_stroke_features_from_shapes] Error extracting features for shape {shape_idx}, stroke {stroke_idx}: {e}")
                features = {'error': str(e)}
            result = {
                'problem_id': problem_id,
                'shape_index': shape_idx,
                'stroke_index': stroke_idx,
                'stroke_command': getattr(stroke, 'raw_command', str(stroke)),
                'features': features,
                'group_info': {
                    'num_shapes': total_shapes,
                    'num_actions_in_shape': len(actions),
                    'shape_type': getattr(shape, '__class__', type(shape)).__name__,
                }
            }
            stroke_results.append(result)
            # Aggregate by positive/negative
            if is_positive:
                positive_strokes.append(features)
            else:
                negative_strokes.append(features)
        avg_stroke_length = float(np.mean([sanitize_feature_value(l) for l in stroke_lengths])) if stroke_lengths else 0.0
        for r in stroke_results[-len(actions):]:
            r['features']['avg_stroke_length'] = avg_stroke_length

    # --- Compute statistics for both sets ---
    from src.Derive_labels.context_features import BongardFeatureExtractor
    extractor = BongardFeatureExtractor()
    pos_stats = extractor.compute_feature_statistics(positive_strokes, label=f"positive_{problem_id}") if positive_strokes else {'valid': False, 'reason': 'no_positive_strokes', 'stats': {}}
    neg_stats = extractor.compute_feature_statistics(negative_strokes, label=f"negative_{problem_id}") if negative_strokes else {'valid': False, 'reason': 'no_negative_strokes', 'stats': {}}

    # --- Compute discriminative features ---
    discriminative_stats = {}
    for key in pos_stats.get('stats', {}):
        if key in neg_stats.get('stats', {}):
            discriminative_stats[key] = {
                'pos_mean': pos_stats['stats'][key].get('mean', 0.0),
                'neg_mean': neg_stats['stats'][key].get('mean', 0.0),
                'mean_diff': pos_stats['stats'][key].get('mean', 0.0) - neg_stats['stats'][key].get('mean', 0.0)
            }
    discriminative_features = {
        'valid': True if positive_strokes and negative_strokes else False,
        'reason': 'computed_from_pos_neg_sets' if positive_strokes and negative_strokes else 'missing_set',
        'stats': discriminative_stats
    }

    # Attach support/discriminative context to each stroke result
    for r in stroke_results:
        r['features']['support_set_context'] = pos_stats
        r['features']['discriminative_features'] = discriminative_features
        results.append(r)

    #logging.info(f"[extract_stroke_features_from_shapes] OUTPUT: {results}")
    if total_strokes > total_shapes:
        logging.info(f"[extract_stroke_features_from_shapes] INFO: Number of strokes ({total_strokes}) and shapes ({total_shapes}) for problem_id={problem_id} -- grouping is correct, no mismatch.")
    return results


def extract_action_type_prefixes(problems_data):
    logger.info(f"[extract_action_type_prefixes] INPUT: problems_data keys={list(problems_data.keys())}")
    """
    Use BongardImage.import_from_action_string_list to robustly extract all unique action type prefixes from the dataset.
    This mirrors hybrid.py's handling and avoids information loss from naive string splitting.
    """
    prefixes = set()
    def robust_flatten(cmds):
        # Recursively flatten nested lists
        if isinstance(cmds, list):
            result = []
            for item in cmds:
                if isinstance(item, list):
                    result.extend(robust_flatten(item))
                else:
                    result.append(item)
            return result
        else:
            return [cmds]

    for problem_data in problems_data.values():
        if not (isinstance(problem_data, list) and len(problem_data) == 2):
            logger.info(f"[extract_action_type_prefixes] Skipping non-standard problem_data: {problem_data}")
            continue
        for example_list in problem_data:
            for action_commands in example_list:
                # Log the structure and type of action_commands before flattening
                logger.info(f"[DEBUG] action_commands type: {type(action_commands)}, value: {action_commands}")
                # Robustly flatten action_commands before parsing
                flat_commands = robust_flatten(action_commands)
                logger.info(f"[DEBUG] flat_commands type: {type(flat_commands)}, value: {flat_commands}")
                try:
                    bongard_image = BongardImage.import_from_action_string_list(flat_commands)
                    for shape in getattr(bongard_image, 'one_stroke_shapes', []):
                        stroke_type = getattr(shape, 'stroke_type', None)
                        if hasattr(stroke_type, 'value'):
                            prefix = stroke_type.value
                        elif stroke_type is not None:
                            prefix = getattr(stroke_type, 'raw_command', str(stroke_type))
                        else:
                            prefix = shape.__class__.__name__
                        prefixes.add(prefix)
                except Exception as e:
                    logger.warning(f"[extract_action_type_prefixes] Failed to robustly parse action_commands: {flat_commands} | Error: {e}")
                    continue
    logger.info(f"[extract_action_type_prefixes] OUTPUT: {prefixes}")
    return prefixes


import numpy as np
def interpolate_vertices(verts, target_count=3):
    """Interpolate vertices to ensure at least target_count points."""
    if len(verts) >= target_count:
        return verts
    interp_verts = []
    for i in range(len(verts) - 1):
        interp_verts.append(verts[i])
        for j in range(1, target_count - len(verts) + 1):
            frac = j / (target_count - len(verts) + 1)
            x = verts[i][0] + frac * (verts[i+1][0] - verts[i][0])
            y = verts[i][1] + frac * (verts[i+1][1] - verts[i][1])
            interp_verts.append((x, y))
    interp_verts.append(verts[-1])
    return interp_verts



def _extract_stroke_type_from_command(stroke) -> str:
    """Robustly extract stroke type from command string or object."""
    if isinstance(stroke, str):
        parts = stroke.split('_')
        if len(parts) >= 1:
            if parts[0] in ['line', 'arc']:
                return parts[0]
    raw_command = getattr(stroke, 'raw_command', None)
    if raw_command and isinstance(raw_command, str):
        parts = raw_command.split('_')
        if len(parts) >= 1:
            if parts[0] in ['line', 'arc']:
                return parts[0]
    function_name = getattr(stroke, 'function_name', None)
    if function_name and isinstance(function_name, str):
        fn_parts = function_name.split('_')
        if len(fn_parts) >= 1:
            if fn_parts[0] in ['line', 'arc']:
                return fn_parts[0]
    return 'unknown'

def _extract_stroke_vertices(stroke, stroke_index, all_vertices, bongard_image=None, parent_shape_vertices=None):
    logger = logging.getLogger(__name__)
    # logger.info(f"[_extract_stroke_vertices] INPUT: stroke_index={stroke_index}, stroke={stroke}, all_vertices={all_vertices}, bongard_image={bongard_image}, parent_shape_vertices={parent_shape_vertices}")  # PATCH: Suppressed verbose stroke vertices log
    # Use analytic vertices from BongardImage parser if available
    from src.Derive_labels.shape_utils import compute_open_stroke_geometry, valid_verts
    verts = []
    if bongard_image and hasattr(bongard_image, 'one_stroke_shapes'):
        shape = bongard_image.one_stroke_shapes[stroke_index] if stroke_index < len(bongard_image.one_stroke_shapes) else None
        if shape and hasattr(shape, 'vertices') and valid_verts(shape.vertices):
            verts = shape.vertices
    if not verts and parent_shape_vertices and len(parent_shape_vertices) > 1:
        verts = parent_shape_vertices
    if not verts and hasattr(stroke, 'vertices') and valid_verts(stroke.vertices):
        verts = stroke.vertices
    if not verts and hasattr(stroke, 'polyline_points') and valid_verts(stroke.polyline_points):
        verts = stroke.polyline_points
    if not verts:
        command = getattr(stroke, 'raw_command', stroke if isinstance(stroke, str) else None)
        if command:
            verts = _vertices_from_command(command, stroke_index)
    if not verts:
        verts = []
    geometry = compute_open_stroke_geometry(verts)
    # logger.info(f"[_extract_stroke_vertices] OUTPUT: stroke_index={stroke_index}, verts={verts}, geometry={geometry}")
    return verts

def _vertices_from_command(command, stroke_index):
    logger.info(f"[_vertices_from_command] INPUT: command={command}, stroke_index={stroke_index}")
    import re
    import math
    import numpy as np
    try:
        pattern = r'^(arc|line)_(\w+)_([\d.]+)(?:_([\d.]+))?-([\d.]+)$'
        match = re.match(pattern, command)
        if match:
            action_type = match.group(1)
            modifier = match.group(2)
            param1 = float(match.group(3))
            param2 = float(match.group(4)) if match.group(4) else None
            param3 = float(match.group(5))
            logger.info(f"[_vertices_from_command] Parsed: action_type={action_type}, modifier={modifier}, param1={param1}, param2={param2}, param3={param3}")
            # Circle modifier: sample 20 points
            if modifier == 'circle':
                center = (0.5, 0.5)
                radius = param1 if param1 else 0.3
                num_points = 20
                vertices = []
                for i in range(num_points):
                    theta = 2 * math.pi * i / num_points
                    x = center[0] + radius * math.cos(theta)
                    y = center[1] + radius * math.sin(theta)
                    vertices.append((x, y))
                logger.info(f"[_vertices_from_command] Circle vertices: {vertices}")
                return vertices
            # Geometric templates for modifiers
            if modifier in ['triangle', 'square']:
                center = (0.5, 0.5)
                size = param1 if param1 else 0.3
                angle_offset = param2 * 360.0 if param2 else 0.0
                if modifier == 'triangle':
                    vertices = []
                    for i in range(3):
                        theta = math.radians(angle_offset + i * 120)
                        x = center[0] + size * math.cos(theta)
                        y = center[1] + size * math.sin(theta)
                        vertices.append((x, y))
                    # Close polygon for triangle
                    vertices.append(vertices[0])
                    logger.info(f"[_vertices_from_command] Triangle vertices: {vertices}")
                    return vertices
                elif modifier == 'square':
                    vertices = []
                    for i in range(4):
                        theta = math.radians(angle_offset + i * 90)
                        x = center[0] + size * math.cos(theta)
                        y = center[1] + size * math.sin(theta)
                        vertices.append((x, y))
                    # Close polygon for square
                    vertices.append(vertices[0])
                    logger.info(f"[_vertices_from_command] Square vertices: {vertices}")
                    return vertices
            # Turtle simulation for line
            if action_type == 'line':
                length = param1
                angle = param3 * 360.0  # normalized to degrees
                x0, y0 = 0.5, 0.5  # center start
                x1 = x0 + length * math.cos(math.radians(angle))
                y1 = y0 + length * math.sin(math.radians(angle))
                num_samples = max(8, min(20, int(length * 20)))
                xs = np.linspace(x0, x1, num_samples)
                ys = np.linspace(y0, y1, num_samples)
                vertices = list(zip(xs, ys))
                logger.info(f"[_vertices_from_command] OUTPUT: vertices={vertices}")
                return vertices
            elif action_type == 'arc':
                radius = param1
                span = param2 if param2 is not None else 90.0
                start_angle = param3 * 360.0  # normalized to degrees
                num_points = max(20, int(abs(span) // 5))
                cx, cy = 0.5, 0.5  # center
                vertices = []
                for i in range(num_points + 1):
                    theta = math.radians(start_angle + (span * i / num_points))
                    x = cx + radius * math.cos(theta)
                    y = cy + radius * math.sin(theta)
                    vertices.append((x, y))
                logger.info(f"[_vertices_from_command] OUTPUT: vertices={vertices}")
                return vertices
            # Zigzag modifier: generate a polyline with alternating y
            if modifier == 'zigzag':
                length = param1 if param1 else 0.3
                angle = param3 * 360.0 if param3 else 0.0
                num_zigs = 8
                x0, y0 = 0.5, 0.5
                vertices = [(x0, y0)]
                for i in range(1, num_zigs + 1):
                    frac = i / num_zigs
                    x = x0 + length * frac * math.cos(math.radians(angle))
                    y = y0 + length * frac * math.sin(math.radians(angle)) + ((-1) ** i) * 0.05
                    vertices.append((x, y))
                logger.info(f"[_vertices_from_command] Zigzag vertices: {vertices}")
                return vertices
        # Fallback: try to extract all floats and simulate as polyline
        floats = [float(x) for x in re.findall(r'[\d.]+', command)]
        if len(floats) >= 4:
            vertices = [(floats[i], floats[i+1]) for i in range(0, len(floats)-1, 2)]
            logger.info(f"[_vertices_from_command] Fallback float extraction: vertices={vertices}")
            return vertices
    except Exception as e:
        logger.error(f"[_vertices_from_command] Failed to parse command {command}: {e}")
    logger.info(f"[_vertices_from_command] Fallback: returning []")
    return []


def _calculate_stroke_specific_features(stroke, stroke_index: int, stroke_type_val=None, shape_modifier_val=None, parameters=None, bongard_image=None, parent_shape_vertices=None, shape_obj=None) -> Dict[str, Any]:
    # logger.info(f"[_calculate_stroke_specific_features] Called with stroke={stroke}, stroke_index={stroke_index}, stroke_type_val={stroke_type_val}, shape_modifier_val={shape_modifier_val}, parameters={parameters}")  # PATCH: Suppressed verbose log
    """Calculate features specific to stroke type and shape modifier, using robust geometric/physics formulas."""
    features = {'stroke_index': stroke_index}
    stype = stroke_type_val or _extract_stroke_type_from_command(stroke)
    smod = shape_modifier_val or _extract_modifier_from_stroke(stroke)
    if not smod:
        smod = 'normal'
    params = parameters or {}
    verts = _extract_stroke_vertices(stroke, stroke_index, None, bongard_image=bongard_image, parent_shape_vertices=parent_shape_vertices)
    # If analytic parameters available for arc, propagate them
    if 'arc' in stype.lower() and hasattr(stroke, 'parameters') and stroke.parameters:
        params = stroke.parameters
    stype_lower = stype.lower() if stype else ''
    # --- PATCH: Propagate analytic vertices and open stroke perimeter ---
    analytic_vertices = verts if verts else []
    features['analytic_vertices'] = analytic_vertices
    features['open_stroke_perimeter'] = float(np.sum(np.linalg.norm(np.array(analytic_vertices)[1:] - np.array(analytic_vertices)[:-1], axis=1))) if len(analytic_vertices) >= 2 else 0.0
    def is_collinear(vs):
        if not vs or len(vs) < 2:
            return False
        arr = np.array(vs)
        if len(arr) < 3:
            return True
        v0 = arr[0]
        v1 = arr[-1]
        direction = v1 - v0
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return True
        direction = direction / norm
        for pt in arr[1:-1]:
            vec = pt - v0
            proj = np.dot(vec, direction)
            perp = vec - proj * direction
            if np.linalg.norm(perp) > 1e-6:
                return False
        return True

    # --- Geometry calculation ---
    geometry = None
    # PATCH: If verts collapse to two points, set degenerate_case and compute minimal features
    if verts and (is_collinear(verts) or len(verts) == 2):
        arr = np.array(verts)
        length = float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1))) if len(arr) >= 2 else 0.0
        min_x, min_y = np.min(arr[:,0]), np.min(arr[:,1])
        max_x, max_y = np.max(arr[:,0]), np.max(arr[:,1])
        width = abs(max_x - min_x)
        height = abs(max_y - min_y)
        geometry = {
            'width': width,
            'height': height,
            'area': 0.0,
            'perimeter': length,
            'centroid': list(np.mean(arr, axis=0)) if arr.size else [0.0, 0.0],
            'bounds': [min_x, min_y, max_x, max_y],
            'num_vertices': len(verts),
            'degenerate_case': True
        }
        features['compactness'] = 0.0
        features['convexity_ratio'] = 0.0
        features['geom_complexity'] = 0.0
        features['arc_curvature_score'] = 0.0
        features['degenerate_case'] = True
        features['stroke_length'] = length
        features['avg_stroke_length'] = length
        shape_perimeter = parent_shape_vertices and len(parent_shape_vertices) > 1 and sum(
            math.sqrt((parent_shape_vertices[i+1][0] - parent_shape_vertices[i][0])**2 + (parent_shape_vertices[i+1][1] - parent_shape_vertices[i][1])**2)
            for i in range(len(parent_shape_vertices)-1)
        ) or 0.0
        raw = length/max(shape_perimeter,1e-6)*(1+0.0) if length and shape_perimeter else 0.0
    # Clamp visual_complexity to [0, 1] to avoid out-of-range values
        features['visual_complexity'] = max(0.0, min(raw, 1.0))
        features['robust_curvature'] = 0.0
        features['robust_angular_variance'] = 0.0
        features['line_curvature_score'] = 0.0
        # PATCH: Add clarifying comment
        # Degenerate strokes (collapsed to two points) are flagged and processed with minimal analytic features.
        # --- PATCH: Fill stats dicts for support_set_context and discriminative_features with explicit diagnostics ---
        from src.Derive_labels.context_features import BongardFeatureExtractor
        extractor = BongardFeatureExtractor()
        # Initialize actions_list to None to avoid UnboundLocalError
        actions_list = None
        if shape_obj is not None:
            actions_list = getattr(shape_obj, 'actions', None) or getattr(shape_obj, 'basic_actions', None)
        logger.debug(f"[PATCH][DIAG] actions_list type: {type(actions_list)}, value: {actions_list}")
        problem_id_val = getattr(shape_obj, 'problem_id', None) if shape_obj is not None else None
        is_positive_val = getattr(shape_obj, 'is_positive', None) if shape_obj is not None else None
        parsed_strokes = []
        if actions_list:
            # If actions_list contains strings, try to parse them into action objects
            if all(isinstance(a, str) for a in actions_list):
                try:
                    bongard_image_obj = BongardImage.import_from_action_string_list(actions_list)
                    actions_list = getattr(bongard_image_obj, 'actions', actions_list)
                    logger.debug(f"[PATCH][DIAG] Parsed actions_list into BongardImage actions: {actions_list}")
                except Exception as e:
                    logger.error(f"[PATCH][DIAG] Failed to parse actions_list strings: {e}")
            for i, s in enumerate(actions_list):
                try:
                    logger.debug(f"[PATCH][DIAG] Processing action {i}: type={type(s)}, value={s}")
                    v = _extract_stroke_vertices(s, i, None, bongard_image=bongard_image, parent_shape_vertices=parent_shape_vertices)
                    f = _calculate_stroke_specific_features(s, i, bongard_image=bongard_image, parent_shape_vertices=v, shape_obj=None)
                    # Only add dicts with geometry
                    if isinstance(f, dict) and 'geometry' in f:
                        stroke_dict = {
                            'stroke_index': i,
                            'command': getattr(s, 'raw_command', str(s)),
                            'geometry': f['geometry']
                        }
                        parsed_strokes.append(stroke_dict)
                except Exception as e:
                    logger.error(f"[PATCH][DIAG] Error extracting features for action {i}: {e}")
            # Use collect_stroke_features for robust feature collection
            shape_strokes = extractor.collect_stroke_features(parsed_strokes, problem_id_val, is_positive_val)
            # logger.info(f"[PATCH] shape_strokes before stats: count={len(shape_strokes)}")
            # Compute stats with proper error handling
            if shape_strokes:
                stats_result = extractor.compute_feature_statistics(shape_strokes, label=f"shape_{problem_id_val}")
                support_set_context = {
                    'valid': stats_result.get('valid', False),
                    'reason': stats_result.get('reason', 'computed_from_strokes'),
                    'stats': stats_result.get('stats', {})
                }
            else:
                logger.warning("[PATCH] No valid stroke features collected; diagnostics: actions_list type: {} value: {}".format(type(actions_list), actions_list))
                support_set_context = {
                    'valid': False,
                    'reason': 'no_stroke_features_collected',
                    'stats': {}
                }
            features['support_set_context'] = support_set_context
            features['discriminative_features'] = {'valid': False, 'reason': 'missing_negative_set', 'stats': support_set_context['stats']}
            # logger.info(f"[PATCH] support_set_context: {features['support_set_context']}")
            # logger.info(f"[PATCH] discriminative_features: {features['discriminative_features']}")
            # PATCH: Log context-aware statistics if present
            if 'support_set_context' in features and 'stats' in features['support_set_context']:
                pass
            if 'discriminative_features' in features and 'stats' in features['discriminative_features']:
                pass
        elif verts and len(verts) >= 3:
            geometry = calculate_geometry_consistent(verts)
        try:
            compactness = _calculate_compactness(geometry.get('area', 0.0), geometry.get('perimeter', 0.0))
            features['compactness'] = min(max(compactness, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"[_calculate_stroke_specific_features] Error calculating compactness: {e}")
            features['compactness'] = 0.0
        try:
            max_n = 20
            n = geometry.get('num_vertices', len(verts)) if 'num_vertices' in geometry else len(verts)
            features['geom_complexity'] = min(n/max_n,1)
        except Exception as e:
            logger.warning(f"[_calculate_stroke_specific_features] Error calculating geom_complexity: {e}")
            features['geom_complexity'] = 0.0
        try:
            convexity = open_stroke_convexity(verts)
            features['convexity_ratio'] = min(max(convexity, 0.0), 1.0)
        except Exception as e:
            logger.warning(f"[_calculate_stroke_specific_features] Error calculating convexity_ratio: {e}")
            features['convexity_ratio'] = 0.0
        features['degenerate_case'] = False
        safe_verts = verts if len(verts) >= 3 else [verts[0], verts[0], verts[0]]
    # try:
    #     features['robust_curvature'] = PhysicsInference.robust_curvature(safe_verts)
    # except Exception as e:
    #     logger.warning(f"Curvature calculation failed for stroke {stroke_index}: {e}")
    #     features['robust_curvature'] = 0.0
        try:
            features['robust_angular_variance'] = PhysicsInference.robust_angular_variance(safe_verts)
        except Exception as e:
            logger.warning(f"Angular variance calculation failed for stroke {stroke_index}: {e}")
            features['robust_angular_variance'] = 0.0
        shape_perimeter = parent_shape_vertices and len(parent_shape_vertices) > 1 and sum(
            math.sqrt((parent_shape_vertices[i+1][0] - parent_shape_vertices[i][0])**2 + (parent_shape_vertices[i+1][1] - parent_shape_vertices[i][1])**2)
            for i in range(len(parent_shape_vertices)-1)
        ) or 1.0
        stroke_length = geometry.get('perimeter', 0.0)
        features['stroke_length'] = stroke_length
        features['avg_stroke_length'] = stroke_length
        base = stroke_length / max(shape_perimeter, 1e-6)
        weight = 1 + (features.get('robust_curvature') or 0)
        raw_vc = base * weight
    # Clamp visual_complexity to [0, 1] to avoid out-of-range values
        features['visual_complexity'] = max(0.0, min(raw_vc, 1.0))
        # PATCH: Ensure aggregation features are always populated
        features['support_set_context'] = features.get('support_set_context', {'valid': True, 'stats': {}})
        features['discriminative_features'] = features.get('discriminative_features', {'valid': True, 'stats': {}})
        logger.debug(f"[PATCH] support_set_context: {features['support_set_context']}")
        logger.debug(f"[PATCH] discriminative_features: {features['discriminative_features']}")
        if geometry.get('area', 0) == 0:
            features['degenerate_case'] = True
            features['compactness'] = 0.0
            features['convexity_ratio'] = 1e-6
        if 'line' in stype_lower:
            features['line_curvature_score'] = 0.0
        elif 'arc' in stype_lower:
            radius = params.get('radius', geometry.get('width', 1.0))
            span_angle = params.get('span_angle', 90)
            delta_theta = math.radians(span_angle)
            features['arc_curvature_score'] = PhysicsInference.arc_curvature_score(radius, delta_theta)
    elif 'arc' in stype_lower and params.get('radius', None) and params.get('span_angle', None):
        radius = params['radius']
        span_angle = params['span_angle']
        arc_length = abs(span_angle) / 360 * 2 * math.pi * radius
        geometry = {
            'width': radius,
            'height': radius,
            'area': 0.0,
            'perimeter': arc_length,
            'centroid': [0.0, 0.0],
            'bounds': [0, 0, 0, 0],
            'num_vertices': 2
        }
        features['robust_curvature'] = 1.0 / max(radius, 1e-6)
        features['robust_angular_variance'] = 0.0
        features['arc_curvature_score'] = 1.0 / max(radius, 1e-6)
        features['stroke_length'] = arc_length
        features['avg_stroke_length'] = arc_length
        shape_perimeter = parent_shape_vertices and len(parent_shape_vertices) > 1 and sum(
            math.sqrt((parent_shape_vertices[i+1][0] - parent_shape_vertices[i][0])**2 + (parent_shape_vertices[i+1][1] - parent_shape_vertices[i][1])**2)
            for i in range(len(parent_shape_vertices)-1)
        ) or 1.0
        base = arc_length / max(shape_perimeter, 1e-6)
        weight = 1 + (features.get('robust_curvature') or 0)
        raw_vc = base * weight
    # Clamp visual_complexity to [0, 1] to avoid out-of-range values
        features['visual_complexity'] = max(0.0, min(raw_vc, 1.0))
        features['compactness'] = 0.0
        try:
            convexity = open_stroke_convexity(verts)
            features['convexity_ratio'] = min(max(convexity, 1e-6), 1.0)
        except Exception as e:
            features['convexity_ratio'] = 1e-6
        features['geom_complexity'] = 0.0
        features['degenerate_case'] = geometry['area'] == 0
    elif verts and len(verts) == 1:
        geometry = {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': list(verts[0]), 'bounds': [verts[0][0], verts[0][1], verts[0][0], verts[0][1]], 'num_vertices': 1}
        features['compactness'] = 1e-6
        features['convexity_ratio'] = 1e-6
        features['geom_complexity'] = 0.0
        features['arc_curvature_score'] = 0.0
        features['degenerate_case'] = True
        features['stroke_length'] = 0.0
        features['avg_stroke_length'] = 0.0
        features['visual_complexity'] = 0.0
        features['robust_curvature'] = 0.0
        features['robust_angular_variance'] = 0.0
        features['line_curvature_score'] = 0.0
        features['degenerate_case'] = True
        features['stroke_length'] = 0.0
        features['avg_stroke_length'] = 0.0
        features['visual_complexity'] = 0.0
        features['robust_curvature'] = 0.0
        features['robust_angular_variance'] = 0.0
        features['line_curvature_score'] = 0.0
    else:
        geometry = {'width': 0.0, 'height': 0.0, 'area': 0.0, 'perimeter': 0.0, 'centroid': [0.0, 0.0], 'bounds': [0, 0, 0, 0], 'num_vertices': 0}
        features['compactness'] = 1e-6
        features['convexity_ratio'] = 1e-6
        features['geom_complexity'] = 0.0
        features['arc_curvature_score'] = 0.0
        features['degenerate_case'] = True
        features['stroke_length'] = 0.0
        features['avg_stroke_length'] = 0.0
        features['visual_complexity'] = 0.0
        features['robust_curvature'] = 0.0
        features['robust_angular_variance'] = 0.0
        features['line_curvature_score'] = 0.0

    # Validate features
    validation_issues = validate_features({**geometry, **features})
    if validation_issues:
        logger.warning(f"Feature validation issues for stroke {stroke_index}: {validation_issues}")
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            logger.warning(f"[_calculate_stroke_specific_features] Value '{val}' could not be converted to float. Using default {default}.")
            return default
    width = safe_float(geometry.get('width', 0.0))
    height = safe_float(geometry.get('height', 0.0))
    geometry['width'] = width
    geometry['height'] = height
    features['geometry'] = geometry
    # PATCH: Always compute and attach individual stroke stats
    try:
        from src.Derive_labels.context_features import BongardFeatureExtractor
        extractor = BongardFeatureExtractor()
        stroke_features_for_stats = [{
            'stroke_index': stroke_index,
            'command': str(stroke),
            'area': float(geometry.get('area', 0.0)),
            'perimeter': float(geometry.get('perimeter', 0.0)),
            'width': float(geometry.get('width', 0.0)),
            'height': float(geometry.get('height', 0.0)),
            'visual_complexity': float(features.get('visual_complexity', 0.0)),
            'geom_complexity': float(features.get('geom_complexity', 0.0))
        }]
        stats_result = extractor.compute_feature_statistics(stroke_features_for_stats, label=f"stroke_{stroke_index}")
        features['support_set_context'] = {
            'valid': stats_result.get('valid', False),
            'reason': stats_result.get('reason', 'individual_stroke'),
            'stats': stats_result.get('stats', {})
        }
        features['discriminative_features'] = {
            'valid': stats_result.get('valid', False),
            'reason': 'single_stroke',
            'stats': stats_result.get('stats', {})
        }
    except Exception as e:
        features['support_set_context'] = {'valid': False, 'reason': f'error: {e}', 'stats': {}}
        features['discriminative_features'] = {'valid': False, 'reason': f'error: {e}', 'stats': {}}
    if not verts or (isinstance(verts, list) and len(verts) == 1 and verts[0] == (0.0, 0.0)):
        logger.error(f"[_calculate_stroke_specific_features] Fallback or missing vertices for stroke_index={stroke_index}, stroke={stroke}, type={stype}, modifier={smod}")
    # logger.info(f"[_calculate_stroke_specific_features] verts: {verts}")  # PATCH: Suppressed verbose log
    # Calculate basic geometric features from vertices
    if verts and len(verts) > 1:
        total_length = 0
        for i in range(len(verts) - 1):
            dx = verts[i+1][0] - verts[i][0]
            dy = verts[i+1][1] - verts[i][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        if len(verts) > 1:
            dx = verts[1][0] - verts[0][0]
            dy = verts[1][1] - verts[0][1]
            angle = math.atan2(dy, dx) * 180 / math.pi
        else:
            angle = 0
        if 'line' in stype_lower:
            features.update({
                'line_length': total_length,
                'line_angle': angle,
                'line_is_short': total_length < 1.0,
                'line_is_long': total_length > 2.0,
                'line_is_horizontal': abs(angle) < 5 or abs(angle) > 175,
                'line_is_vertical': 85 < abs(angle) < 95,
                'line_direction': _categorize_direction(angle)
            })
            # Modifier logic
            if abs(angle) < 5 or abs(angle) > 175:
                features['shape_modifier'] = 'horizontal'
            elif 85 < abs(angle) < 95:
                features['shape_modifier'] = 'vertical'
            else:
                features['shape_modifier'] = 'normal'
        elif 'arc' in stype_lower or smod == 'circle':
            radius = params.get('radius', 0.5)
            span_angle = params.get('span_angle', 90)
            arc_length = abs(span_angle) * radius * safe_divide(math.pi, 180) if radius > 0 else total_length
            features.update({
                'arc_radius': radius,
                'arc_span_angle': span_angle,
                'arc_length': arc_length,
                'arc_curvature': safe_divide(1.0, max(radius, 1e-6)),
                'arc_is_major': abs(span_angle) > 180,
                'arc_is_full_circle': abs(span_angle) >= 350,
                'arc_direction': 'clockwise' if span_angle < 0 else 'counterclockwise',
                'arc_is_small': radius < 0.3,
                'arc_is_large': radius > 1.5
            })
            features['shape_modifier'] = 'arc'
    else:
        if 'line' in stype_lower:
            features.update({
                'line_length': 0.5,
                'line_angle': 0,
                'line_is_short': False,
                'line_is_long': False,
                'line_is_horizontal': False,
                'line_is_vertical': False,
                'line_direction': 'horizontal'
            })
        elif 'arc' in stype_lower:
            features.update({
                'arc_radius': 0.5,
                'arc_span_angle': 90,
                'arc_length': 0.5,
                'arc_curvature': 2.0,
                'arc_is_major': False,
                'arc_is_full_circle': False,
                'arc_direction': 'counterclockwise',
                'arc_is_small': False,
                'arc_is_large': False
            })
    features.update(_calculate_shape_modifier_features_from_val(smod))
    # Set analytic_attribute to shape_modifier for downstream analytics
    features['analytic_attribute'] = features.get('shape_modifier', stype)
    return features

def _categorize_direction(angle):
    """Categorize line direction based on angle."""
    angle = angle % 360
    if angle >= 337.5 or angle < 22.5:
        return 'horizontal'
    elif 22.5 <= angle < 67.5:
        return 'diagonal-up'
    elif 67.5 <= angle < 112.5:
        return 'vertical'
    elif 112.5 <= angle < 157.5:
        return 'diagonal-down'
    elif 157.5 <= angle < 202.5:
        return 'horizontal'
    elif 202.5 <= angle < 247.5:
        return 'diagonal-up'
    elif 247.5 <= angle < 292.5:
        return 'vertical'
    elif 292.5 <= angle < 337.5:
        return 'diagonal-down'
    return 'unknown'

def _compute_bounding_box(vertices):
    if not vertices or len(vertices) < 1:
        return (0, 0, 0, 0)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x, min_y, max_x, max_y)
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
    num_vertices = 0
    convexity = 1e-6
    compactness = 1e-6
    if geom:
        num_vertices = len(vertices) if vertices else 0
        convexity = geom.get('convexity_ratio', 1.0)
        compactness = None
        area = geom.get('area', None)
        perimeter = geom.get('perimeter', None)
        if area and perimeter:
            from src.Derive_labels.shape_utils import _calculate_compactness
            compactness = _calculate_compactness(area, perimeter)
        # Clamp/sanitize convexity and compactness
        if convexity is None or not isinstance(convexity, (float, int)) or convexity != convexity:
            convexity = 1e-6
        else:
            convexity = max(convexity, 1e-6)
        if compactness is None or not isinstance(compactness, (float, int)) or compactness != compactness:
            compactness = 1e-6
        else:
            compactness = max(compactness, 1e-6)
    max_expected_vertices = 50
    complexity = min(num_vertices / max_expected_vertices, 1.0)
    base_features['geometric_complexity'] = complexity
    base_features['num_vertices'] = num_vertices
    base_features['convexity_ratio'] = convexity
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
    raw_line_feats = stroke_type_features.get('line_features', [])
    raw_arc_feats  = stroke_type_features.get('arc_features', [])

    # Only keep dicts with required keys
    line_features = [f for f in raw_line_feats if isinstance(f, dict) and 'line_length' in f and 'line_direction' in f]
    arc_features  = [f for f in raw_arc_feats  if isinstance(f, dict) and 'arc_radius' in f]

    # Count stroke types directly
    num_lines = sum(_extract_stroke_type_from_command(s) == 'line' for s in strokes)
    num_arcs = sum(_extract_stroke_type_from_command(s) == 'arc' for s in strokes)
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

    if line_features:
        line_lengths = [f.get('line_length', 0) for f in line_features]
        line_angles = [f.get('line_angle', 0) for f in line_features]
        features['line_aggregate'] = {
            'total_line_length': sum(line_lengths),
            'avg_line_length': safe_divide(sum(line_lengths), len(line_lengths)),
            'line_length_variance': _calculate_variance(line_lengths) if line_lengths else 0,
            'line_angle_variance': _calculate_variance(line_angles) if line_angles else 0,
            'has_short_lines': any(f.get('line_is_short', False) for f in line_features),
            'has_long_lines': any(f.get('line_is_long', False) for f in line_features),
            'dominant_direction': _calculate_dominant_direction(line_features) if line_features else 'none'
        }

    if arc_features:
        arc_radii = [f.get('arc_radius', 0) for f in arc_features]
        arc_spans = [f.get('arc_span_angle', 0) for f in arc_features]
        arc_lengths = [f.get('arc_length', 0) for f in arc_features]
        features['arc_aggregate'] = {
            'total_arc_length': sum(arc_lengths),
            'avg_arc_radius': safe_divide(sum(arc_radii), len(arc_radii)),
            'avg_arc_span': safe_divide(sum(arc_spans), len(arc_spans)),
            'arc_radius_variance': _calculate_variance(arc_radii) if arc_radii else 0,
            'arc_span_variance': _calculate_variance(arc_spans) if arc_spans else 0,
            'total_curvature': sum(f.get('arc_curvature', 0) for f in arc_features),
            'has_full_circles': any(f.get('arc_is_full_circle', False) for f in arc_features),
            'has_major_arcs': any(f.get('arc_is_major', False) for f in arc_features),
            'curvature_complexity': len([f for f in arc_features if f.get('arc_curvature', 0) > 1.0])
        }

    logger.info(f"[_calculate_stroke_type_differentiated_features] OUTPUT: {features}")
    return features

def _extract_modifier_from_stroke(stroke) -> str:

    logger = logging.getLogger(__name__)
    # logger.info(f"[_extract_modifier_from_stroke] INPUT: type={type(stroke)}, attributes={dir(stroke) if not isinstance(stroke, str) else 'str'}")  # PATCH: Suppressed verbose log
    """
    Extract the actual shape modifier from a stroke object or string, using geometric and semantic analysis with debug logging.
    """
    from src.Derive_labels.shape_utils import calculate_geometry, _calculate_compactness
    # If stroke is a string, parse modifier directly
    if isinstance(stroke, str):
        # Expect format like 'line_circle_1.000-0.500'
        parts = stroke.split('_')
        if len(parts) >= 2 and parts[1]:
            # logger.debug(f"[_extract_modifier_from_stroke] Extracted modifier from string: {parts[1]}")  # PATCH: Suppressed verbose log
            # logger.info(f"[_extract_modifier_from_stroke] OUTPUT: {parts[1]} (string)")  # PATCH: Suppressed verbose log
            return parts[1]
        # If line or arc, default to 'normal'
        if len(parts) >= 1 and parts[0] in ['line', 'arc']:
            return 'normal'
        else:
            logger.warning(f"[_extract_modifier_from_stroke] Could not extract modifier from string: {stroke}")
            return None
    # Try geometric analysis for object strokes
    verts = None
    if hasattr(stroke, 'vertices') and stroke.vertices and len(stroke.vertices) > 2:
        verts = stroke.vertices
    elif hasattr(stroke, 'polyline_points') and stroke.polyline_points and len(stroke.polyline_points) > 2:
        verts = stroke.polyline_points
    elif hasattr(stroke, 'get_world_coordinates') and callable(stroke.get_world_coordinates):
        verts = stroke.get_world_coordinates()
    if verts and len(verts) > 2:
        geom = calculate_geometry(verts)
        num_vertices = len(verts)
        convexity = geom.get('convexity_ratio', 1.0)
        compactness = None
        area = geom.get('area', None)
        perimeter = geom.get('perimeter', None)
        if area and perimeter:
            compactness = _calculate_compactness(area, perimeter)
        if num_vertices == 3 and convexity > 0.95:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: triangle")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: triangle (geometry)")
            return 'triangle'
        if num_vertices == 4 and convexity > 0.95 and compactness and 0.7 < compactness < 0.85:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: square")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: square (geometry)")
            return 'square'
        if num_vertices > 6 and compactness and compactness > 0.85 and convexity > 0.95:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: circle")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: circle (geometry)")
            return 'circle'
        edge_var = geom.get('edge_length_variance', 0.0)
        if edge_var and edge_var > 0.1 and compactness and compactness < 0.5:
            logger.debug("[_extract_modifier_from_stroke] Geometric modifier: zigzag")
            logger.info("[_extract_modifier_from_stroke] OUTPUT: zigzag (geometry)")
            return 'zigzag'
    # Fallback: attribute-based extraction
    if hasattr(stroke, 'shape_modifier'):
        smod = getattr(stroke, 'shape_modifier')
        if hasattr(smod, 'value') and smod.value:
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
    logger.warning("[_extract_modifier_from_stroke] Could not extract modifier from stroke object. Returning None.")
    return None


