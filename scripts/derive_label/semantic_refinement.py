from derive_label.confidence_scoring import consensus_vote, DETECTOR_RELIABILITY

def analyze_detector_consensus(obj, scene_context=None):
    """
    Aggregates candidate labels and confidences from all detectors for an object,
    applies confidence-weighted voting and context-aware arbitration.
    Modifies obj in place to add 'detector_consensus' property.
    """
    candidate_labels = obj.get('possible_labels', {})
    detector_confidences = obj.get('detector_confidences', {})
    context = {}
    if scene_context and scene_context.get('grid_pattern', {}).get('is_grid'):
        context['is_grid'] = scene_context['grid_pattern']['is_grid']
    final_label, label_ranking, label_sources = consensus_vote(
        candidate_labels, detector_confidences, DETECTOR_RELIABILITY, context)
    obj['detector_consensus'] = {
        'final_label': final_label,
        'label_ranking': label_ranking,
        'label_sources': label_sources,
        'agreement': final_label != 'AMBIGUOUS',
        'disagreement_reason': 'Ambiguous consensus' if final_label == 'AMBIGUOUS' else None
    }
import numpy as np
import logging
from shapely.geometry import Polygon
from math import hypot

# Define a hierarchy for specific labels (higher value means more specific/preferred)
label_specificity_score = {
    'point': 1, 'short_line': 2,
    'line': 10, 'line_segment': 11, 'colinear_line': 12, 'line_hough': 13,
    'polyline': 20, 'piecewise_polyline': 21, 'curved_path': 22, 'principal_axis_curve': 23,
    'open_shape': 30, 'roughly_circular': 31,
    'self_intersecting_path': 40, 'self_intersecting_decomposed_polygon': 41,
    'triangle': 50, 'quadrilateral': 51, 'rectangle': 52, 'pentagon': 53, 'hexagon': 54, 'polygon': 55,
    'min_area_rect': 56, 'irregular_quadrangle': 57, 'six_straight_line_figure': 58, 'multi_line_composite': 59, 'multi_line_figure': 60,
    'convex_polygon': 65, 'concave_polygon': 66, 'polygon_with_defects_error': 67,
    'circle': 70, 'ellipse': 71,
    'alpha_shape': 80, 'polygon_imgproc': 81, 'polygon_with_holes': 82, 'polygon_with_holes_imgproc': 83, 'multi_contour_shape_imgproc': 84, 'branched_shape_imgproc': 85, 'noisy_or_fragmented_imgproc': 86, 'complex_edge_shape': 87, 'multi_component_or_holey': 88, 'branched_shape': 89,
    'degenerate_point_cloud': 90, 'multi_segment_line': 91, 'degenerate_hull_polygon': 92,
    'unknown': 0, 'error': -1, 'missing_deps_polygon': -2,

    # New physics-based labels (assign appropriate specificity)
    'physically_convex': 60, # Slightly higher than general polygons
    'physically_concave': 61,
    'symmetric_vertical_reflection': 62,
    'symmetric_horizontal_reflection': 63,
    'symmetric_rotational_180': 64,
    'has_1_arcs': 32, 'has_2_arcs': 33, 'has_3_arcs': 34, # Arc counts
    'has_1_straight_segments': 14, 'has_2_straight_segments': 15, # Straight segment counts
    'occluded_object': 95, # High specificity for occlusion
    'touching_ground': 96,
    'floating_object': 97,
    'stable_object': 98, # High specificity for stability
    'has_1_contacts': 99, 'has_2_contacts': 100, 'has_3_contacts': 101, # Contact counts
}

# Weights for different detector types (heuristic, can be tuned)
type_weights = {
    'geometric': 1.2,
    'geometric_fit': 1.1,
    'morphological': 1.0,
    'image_processing': 0.9,
    'semantic': 0.8,
    'physics_geometric': 1.15, # New category for physics-derived shape properties
    'physics_relation': 1.05,  # New category for relational physics properties (occlusion, support)
    'degenerate_geometric': 0.7,
    'degenerate_structural': 0.6,
    'degenerate_composite': 0.5,
    'degenerate_scattered': 0.4,
    'alpha_shape': 0.7,
    'geometric_general': 0.3,
    'fallback': 0.1,
    'invalid_physics_shape': 0.05 # Low weight for shapes that broke physics processing
}

def detect_connected_line_segments(vertices_np, epsilon=5.0, min_len=10):
    """Detects straight line segments in a polyline using RDP."""
    def rdp(points_list, rdp_epsilon):
        if len(points_list) < 3:
            return points_list
        
        dists = []
        start_pt = points_list[0]
        end_pt = points_list[-1]
        
        line_vec = np.array(end_pt) - np.array(start_pt)
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return [start_pt, end_pt]
            
        max_dist = 0
        index = 0
        for i in range(1, len(points_list) - 1):
            point_vec = np.array(points_list[i]) - np.array(start_pt)
            line_vec_3d = np.pad(line_vec, (0, 1), 'constant') if line_vec.shape[-1] == 2 else line_vec
            point_vec_3d = np.pad(point_vec, (0, 1), 'constant') if point_vec.shape[-1] == 2 else point_vec
            cross_prod = np.abs(np.cross(line_vec_3d, point_vec_3d))
            dist = cross_prod / line_len
            # Ensure dist is a scalar for comparison
            if isinstance(dist, np.ndarray):
                dist = float(np.max(dist))
            if dist > max_dist:
                max_dist = dist
                index = i
        
        if max_dist > rdp_epsilon:
            rec_results1 = rdp(points_list[:index + 1], rdp_epsilon)
            rec_results2 = rdp(points_list[index:], rdp_epsilon)
            result = rec_results1[:-1] + rec_results2
        else:
            result = [start_pt, end_pt]
        return result
    
    if hasattr(vertices_np, 'shape'):
        n_points = vertices_np.shape[0]
    else:
        n_points = len(vertices_np)
    if n_points < 2:
        logging.debug(f"[SemanticRefinement] Not enough points for line segments: {vertices_np}")
        return []
    elif n_points == 2:
        diff = np.array(vertices_np[1]) - np.array(vertices_np[0])
        if np.linalg.norm(diff) > min_len:
            return [(np.array(vertices_np[0]).tolist(), np.array(vertices_np[1]).tolist())]
        else:
            logging.debug(f"[SemanticRefinement] Two points but too short for line segment: {vertices_np}")
            return []
    
    simplified_points_list = rdp(np.array(vertices_np).tolist(), epsilon)
    simplified_points = np.array(simplified_points_list)

    segments = []
    for i in range(len(simplified_points) - 1):
        p1 = np.array(simplified_points[i])
        p2 = np.array(simplified_points[i + 1])
        if np.linalg.norm(p2 - p1) > min_len:
            segments.append((p1.tolist(), p2.tolist()))
    return segments

def is_quadrilateral_like(vertices_np):
    segments = detect_connected_line_segments(vertices_np, epsilon=5.0, min_len=5)
    if segments is not None and len(segments) == 4:
        first_point = np.array(segments[0][0])
        last_point = np.array(segments[-1][1])
        perimeter = sum(np.linalg.norm(np.array(s[1]) - np.array(s[0])) for s in segments)
        # Explicitly check array shapes and use np.linalg.norm safely
        if (
            isinstance(first_point, np.ndarray) and isinstance(last_point, np.ndarray)
            and first_point.shape == last_point.shape
            and np.linalg.norm(first_point - last_point) < 20 and perimeter > 0
        ):
            return True
    return False

def is_smooth_curve(vertices_np, angle_change_threshold=0.2, curvature_std_threshold=0.1):
    if len(vertices_np) < 5:
        return False
    diffs = np.diff(vertices_np, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    valid_diffs = diffs[segment_lengths > 1e-6]
    
    if len(valid_diffs) < 2: return False

    angles = np.arctan2(valid_diffs[:, 1], valid_diffs[:, 0])
    angle_diffs = np.diff(angles)
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi

    mean_angle_change = np.mean(np.abs(angle_diffs))
    std_angle_change = np.std(angle_diffs)

    return mean_angle_change < angle_change_threshold and std_angle_change < curvature_std_threshold

def is_composite_shape(vertices_np, min_vertices_for_composite=20):
    if len(vertices_np) < min_vertices_for_composite:
        return False
    return not is_smooth_curve(vertices_np, angle_change_threshold=0.5, curvature_std_threshold=0.3)

def generate_semantic_label(vertices_np, primary_label, problem_id=None, candidate_labels=None,
                            is_roughly_circular_detector=None, convexity_defects_detector=None):
    semantic_label = primary_label

    line_segments = detect_connected_line_segments(vertices_np, epsilon=5.0, min_len=10)
    
    if len(line_segments) >= 6:
        if problem_id and 'six_straight_lines' in problem_id:
            semantic_label = 'six_straight_line_figure'
        elif 'line' in semantic_label or 'polyline' in semantic_label:
            semantic_label = 'multi_line_composite'
    elif len(line_segments) >= 4:
        if (problem_id and 'quadrangle' in problem_id) or is_quadrilateral_like(vertices_np):
            semantic_label = 'irregular_quadrangle'
        elif 'line' in semantic_label or 'polyline' in semantic_label:
            semantic_label = 'multi_line_figure'
    elif is_smooth_curve(vertices_np):
        if 'curve' not in semantic_label:
            semantic_label = 'curved_path'
    elif is_composite_shape(vertices_np):
        if 'composite' not in semantic_label and 'multi' not in semantic_label:
            semantic_label = 'composite_shape'
    
    if problem_id:
        if 'circle' in problem_id and 'circle' not in semantic_label and 'ellipse' not in semantic_label:
            if is_roughly_circular_detector:
                is_circular, _ = is_roughly_circular_detector(vertices_np)
                if is_circular:
                    semantic_label = 'circular_object'
        
        if 'star' in problem_id and 'star' not in semantic_label and convexity_defects_detector:
            num_defects, defect_label, _, _ = convexity_defects_detector(vertices_np)
            if num_defects > 4 and 'concave' in defect_label:
                 semantic_label = 'star_shape'

    return semantic_label

def infer_shape_type_ensemble(vertices_np, problem_id, candidate_labels, calculate_confidence,
                            is_rectangle_detector, is_roughly_circular_detector,
                            detect_connected_line_segments, is_quadrilateral_like, is_smooth_curve,
                            is_composite_shape, convexity_defects_detector, is_point_cloud_detector):
    best_label_info = {'label': 'unknown', 'confidence': 0.0, 'type': 'fallback', 'props': {}}

    best_overall_score = -1.0
    for candidate in candidate_labels:
        label_score = label_specificity_score.get(candidate['label'], 0)
        type_weight = type_weights.get(candidate.get('type', 'fallback'), 0.1)
        
        current_overall_score = candidate['confidence'] * 1000 + label_score * 10 + type_weight
        
        if current_overall_score > best_overall_score:
            best_overall_score = current_overall_score
            best_label_info = candidate
        elif current_overall_score == best_overall_score:
            if candidate['confidence'] > best_label_info['confidence']:
                best_label_info = candidate

    # Apply semantic refinement after initial best label selection
    refined_label = generate_semantic_label(
        vertices_np, best_label_info['label'], problem_id, candidate_labels,
        is_roughly_circular_detector=is_roughly_circular_detector,
        convexity_defects_detector=convexity_defects_detector
    )

    if refined_label != best_label_info['label']:
        best_label_info['label'] = refined_label
        best_label_info['type'] = 'semantic'
        best_label_info['confidence'] = max(best_label_info['confidence'], 0.7)

    if best_label_info['confidence'] < 0.2:
        best_label_info['label'] = 'unknown'
        best_label_info['confidence'] = 0.1
        best_label_info['props'] = {} # Changed from 'properties' to 'props' to match candidate structure

    # Return props from the best candidate as the ensemble's properties
    # Note: `props` from candidate is already being added to main `properties` in logo_processing.py
    # This return simply makes it explicit which properties were chosen for the best label.
    return best_label_info['label'], best_label_info['confidence'], best_label_info.get('props', {})

def analyze_detector_consensus(possible_labels, final_label):
    """
    Analyzes the consensus among different detectors for a given object's final label.
    Returns a dictionary indicating agreement status and reasons.
    """
    if not possible_labels:
        return {'agreement': False, 'disagreement_reason': 'No candidate labels from detectors'}

    # Collect labels proposed by high-confidence detectors (e.g., confidence > 0.7)
    strong_candidate_labels = [
        cand['label'] for cand in possible_labels if cand['confidence'] > 0.7
    ]

    # Check if the final label is among the strong candidates
    if final_label not in strong_candidate_labels:
        # If final label is not supported by strong candidates, it's a disagreement
        return {
            'agreement': False,
            'disagreement_reason': f"Final label '{final_label}' not strongly supported by high-confidence detectors. Strong candidates: {strong_candidate_labels}"
        }

    # Check if strong candidates are diverse
    unique_strong_candidates = set(strong_candidate_labels)

    if len(unique_strong_candidates) > 1:
        # If multiple *different* strong candidates exist, it's a disagreement
        return {
            'agreement': False,
            'disagreement_reason': f"Multiple strong candidate labels found: {list(unique_strong_candidates)}"
        }
    
    # If final label is strongly supported and no other strong candidates contradict it
    return {
        'agreement': True,
        'disagreement_reason': None # No disagreement
    }

