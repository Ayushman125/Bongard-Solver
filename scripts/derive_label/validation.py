import logging
import numpy as np
from math import hypot # Import hypot for distance calculations

def validate_label_quality(
    generated_labels_for_image: dict,
    problem_id: str,
    min_confidence_for_rule: float = 0.5,
    min_object_count_for_degenerate_check: int = 1,
    min_line_segment_length: float = 10.0 # Example: to distinguish true lines from noisy points
) -> list[dict]:
    """
    Performs comprehensive validation of generated labels for an image, leveraging
    semantic clues from the problem ID, consistency checks with scene-level analysis,
    and physics-based inferences.

    This function identifies potential inaccuracies, inconsistencies, or ambiguities,
    returning detailed issues rather than a simple pass/fail.

    Args:
        generated_labels_for_image (dict): A dictionary containing processed image data,
                                           including 'objects' (with 'label', 'confidence', 'properties'),
                                           'overall_confidence', 'confidence_tier', and 'scene_level_analysis'.
        problem_id (str): The ID of the current problem, which often contains semantic clues.
        min_confidence_for_rule (float): Minimum confidence an object label must have
                                         to be considered when applying problem-specific rules.
        min_object_count_for_degenerate_check (int): Minimum number of objects for
                                                      the 'all degenerate/unknown' check to apply.
        min_line_segment_length (float): Minimum pixel length for a detected line segment
                                         to be considered a 'significant' line.

    Returns:
        list[dict]: A list of dictionaries, each representing a detected validation issue.
                    Each dictionary contains:
                    - 'severity' (str): 'CRITICAL', 'WARNING', 'INFO'. CRITICAL issues indicate
                                        a likely incorrect label.
                    - 'rule_name' (str): Name of the validation rule that triggered.
                    - 'message' (str): A detailed explanation of the issue.
                    - 'object_id' (str, optional): The ID of the specific object if relevant.
                    - 'found_label' (str, optional): The label found for the object.
                    - 'expected_keywords' (list[str], optional): Keywords expected in the problem.
    """
    validation_issues = []

    if not problem_id:
        validation_issues.append({
            'severity': 'INFO',
            'rule_name': 'NoProblemID',
            'message': "No problem_id provided; problem-specific validation rules skipped."
        })
        return validation_issues
    
    # Pre-process for easier access
    objects = generated_labels_for_image.get('objects', [])
    scene_context = generated_labels_for_image.get('scene_level_analysis', {})
    
    # Helper for searching keywords in problem_id
    problem_id_lower = problem_id.lower()

    def _add_issue(severity, rule_name, message, obj_id=None, found_label=None, expected_keywords=None):
        issue = {
            'severity': severity,
            'rule_name': rule_name,
            'message': message,
        }
        if obj_id is not None:
            issue['object_id'] = obj_id
        if found_label is not None:
            issue['found_label'] = found_label
        if expected_keywords is not None:
            issue['expected_keywords'] = expected_keywords
        validation_issues.append(issue)

    # --- Problem-ID Driven Validation Rules ---

    # Rule 1: "six_straight_lines" Problem
    if 'six_straight_lines' in problem_id_lower:
        found_significant_line_like_count = 0
        for obj in objects:
            label = obj.get('label', '').lower()
            confidence = obj.get('confidence', 0.0)
            is_line_segment_significant = False
            
            # Check for actual line length if it's a line segment or similar
            if 'line_segments' in obj.get('properties', {}):
                total_length = sum(
                    hypot(p2[0]-p1[0], p2[1]-p1[1]) for s in obj['properties']['line_segments'] for p1, p2 in [s]
                )
                if total_length >= min_line_segment_length:
                    is_line_segment_significant = True

            if confidence >= min_confidence_for_rule and (
                'line' in label or 'polyline' in label or 'multi_line' in label or
                'straight_line_figure' in label or (is_line_segment_significant and 'line_segment' in label) or
                'has_straight_segments' in label # From physics inference
            ):
                found_significant_line_like_count += 1
        
        if found_significant_line_like_count < 6:
            _add_issue('CRITICAL', 'SixStraightLinesCountMismatch',
                       f"Expected at least 6 significant line-like objects for 'six_straight_lines' problem, but found only {found_significant_line_like_count}.",
                       expected_keywords=['line', 'polyline', 'multi_line', 'straight_line_figure', 'has_straight_segments'])

    # Rule 2: "quadrangle" or "rectangle" Problems
    if 'quadrangle' in problem_id_lower or 'rectangle' in problem_id_lower:
        found_quad_like = False
        for obj in objects:
            label = obj.get('label', '').lower()
            confidence = obj.get('confidence', 0.0)
            if confidence >= min_confidence_for_rule and (
                'quadrilateral' in label or 'rectangle' in label or 'quadrangle' in label or 'min_area_rect' in label
            ):
                # Optionally, add a check for number of vertices = 4 here as a stronger rule for quad-like shapes
                if obj.get('properties', {}).get('num_vertices') == 4:
                    found_quad_like = True
                    break
        
        if not found_quad_like:
            _add_issue('CRITICAL', 'QuadrangleMismatch',
                       f"Expected quadrilateral or rectangle labels for '{problem_id}' problem, but none found with sufficient confidence and 4 vertices.",
                       expected_keywords=['quadrilateral', 'rectangle', 'quadrangle'])

    # Rule 3: "circle" Problem
    if 'circle' in problem_id_lower:
        found_circle_like = False
        for obj in objects:
            label = obj.get('label', '').lower()
            confidence = obj.get('confidence', 0.0)
            if confidence >= min_confidence_for_rule and (
                'circle' in label or 'circular' in label or 'ellipse' in label or 'roughly_circular' in label
            ):
                # Optionally, check circularity property for stricter validation
                if obj.get('properties', {}).get('circularity', 0) > 0.7:
                    found_circle_like = True
                    break

        if not found_circle_like:
            _add_issue('CRITICAL', 'CircleMismatch',
                       f"Expected circular/elliptical labels for '{problem_id}' problem, but none found with sufficient confidence and circularity.",
                       expected_keywords=['circle', 'circular', 'ellipse'])

    # Rule 4: "polygon" general (distinguish from degenerate/simple shapes)
    if 'polygon' in problem_id_lower and 'polygons' not in problem_id_lower: # Avoid "many polygons"
        found_generic_polygon = False
        for obj in objects:
            label = obj.get('label', '').lower()
            confidence = obj.get('confidence', 0.0)
            if confidence >= min_confidence_for_rule and (
                'polygon' in label or 'triangle' in label or 'quadrilateral' in label or
                'pentagon' in label or 'hexagon' in label or 'convex_polygon' in label or
                'concave_polygon' in label or 'alpha_shape' in label # Include alpha_shape as a general polygon
            ) and not ( # Exclude very simple or degenerate shapes for 'polygon' problem
                'point' in label or 'line' in label or 'polyline' in label or 'degenerate' in label or
                'short_line' in label
            ):
                # Optionally, check num_vertices > 2 for stricter polygon definition
                if obj.get('properties', {}).get('num_vertices', 0) > 2:
                    found_generic_polygon = True
                    break
        if not found_generic_polygon:
            _add_issue('CRITICAL', 'GenericPolygonMismatch',
                       f"Expected a meaningful polygon-like label for '{problem_id}' problem, but suitable ones not found with sufficient confidence.",
                       expected_keywords=['polygon', 'triangle', 'quadrilateral', 'pentagon', 'hexagon', 'convex_polygon', 'concave_polygon'])

    # Rule 5: "star" Problem
    if 'star' in problem_id_lower:
        found_star_like = False
        for obj in objects:
            label = obj.get('label', '').lower()
            confidence = obj.get('confidence', 0.0)
            if confidence >= min_confidence_for_rule and 'star_shape' in label:
                # Optionally, verify number of convexity defects
                if obj.get('properties', {}).get('num_convexity_defects', 0) >= 5: # Stars typically have >= 5 points/defects
                    found_star_like = True
                    break
        if not found_star_like:
            _add_issue('CRITICAL', 'StarShapeMismatch',
                       f"Expected 'star_shape' label for '{problem_id}' problem, but not found with sufficient confidence and characteristic defects.",
                       expected_keywords=['star_shape'])
            
    # Rule 6: Pattern Validation (e.g., grid, periodicity, tiling)
    if 'grid' in problem_id_lower and scene_context.get('grid_pattern', {}).get('confidence', 0) < 0.7:
        _add_issue('WARNING', 'GridPatternLowConfidence',
                   f"Problem ID suggests a 'grid' but scene grid pattern detection confidence is low ({scene_context['grid_pattern']['confidence']:.2f}).")
    
    if 'periodic' in problem_id_lower and scene_context.get('periodicity', {}).get('confidence', 0) < 0.7:
         _add_issue('WARNING', 'PeriodicityLowConfidence',
                   f"Problem ID suggests 'periodic' arrangement but scene periodicity detection confidence is low ({scene_context['periodicity']['confidence']:.2f}).")

    if 'tiling' in problem_id_lower and scene_context.get('tiling', {}).get('confidence', 0) < 0.7:
        _add_issue('WARNING', 'TilingLowConfidence',
                   f"Problem ID suggests 'tiling' but scene tiling detection confidence is low ({scene_context['tiling']['confidence']:.2f}).")
    
    # Rule 7: Detector Disagreement Flags (from semantic_refinement.py's analyze_detector_consensus)
    for obj in objects:
        if 'detector_consensus' in obj.get('properties', {}) and not obj['properties']['detector_consensus']['agreement']:
            _add_issue('WARNING', 'DetectorDisagreement',
                       f"Object {obj['id']} label '{obj.get('label')}' shows disagreement among individual detectors: {obj['properties']['detector_consensus']['disagreement_reason']}.",
                       obj_id=obj['id'], found_label=obj.get('label'))

    # Rule 8: Structural/Semantic Consistency Flags (from post_processing.py)
    for obj in objects:
        for flag in obj.get('flags', []):
            if flag == 'overlapping_shapes_flag':
                _add_issue('INFO', 'OverlappingShapes',
                           f"Object {obj['id']} is part of an overlapping configuration.",
                           obj_id=obj['id'])
            elif flag == 'occluded_shape_flag':
                _add_issue('INFO', 'OccludedShape',
                           f"Object {obj['id']} is flagged as potentially occluded.",
                           obj_id=obj['id'])
            elif flag == 'small_area_polygon_check':
                _add_issue('WARNING', 'SmallAreaPolygon',
                           f"Object {obj['id']} labeled as a polygon-like shape has a very small area ({obj.get('properties', {}).get('area', 'N/A')}). Might be degenerate.",
                           obj_id=obj['id'], found_label=obj.get('label'))
            elif flag == 'potential_composite_shape_mismatch':
                _add_issue('WARNING', 'CompositeMismatch',
                           f"Object {obj['id']} is labeled simply ('{obj.get('label')}') but detected as a potential composite shape.",
                           obj_id=obj['id'], found_label=obj.get('label'))
            elif flag == 'potential_smooth_curve_mismatch':
                 _add_issue('WARNING', 'SmoothCurveMismatch',
                           f"Object {obj['id']} is labeled with sharp corners ('{obj.get('label')}') but detected as a potential smooth curve.",
                           obj_id=obj['id'], found_label=obj.get('label'))
            elif flag == 'invalid_polygon_for_physics':
                 _add_issue('CRITICAL', 'PhysicsPolygonInvalid',
                           f"Object {obj['id']}'s polygon was invalid for physics inference. Its properties may be unreliable.",
                           obj_id=obj['id'], found_label=obj.get('label'))
            elif flag == 'physics_inference_error_individual':
                 _add_issue('CRITICAL', 'PhysicsInferenceErrorIndividual',
                           f"An error occurred during individual physics inference for object {obj['id']}.",
                           obj_id=obj['id'], found_label=obj.get('label'))
            elif flag == 'physics_inference_error_scene_level':
                 _add_issue('CRITICAL', 'PhysicsInferenceErrorScene',
                           f"An error occurred during scene-level physics inference, potentially affecting object {obj['id']} and others.",
                           obj_id=obj['id'], found_label=obj.get('label'))
            # Add more flags as needed from post_processing

    # --- New Physics-based Validation Rules ---

    # Rule 9: Check for "balanced" or "stable" problems
    if 'balanced' in problem_id_lower or 'stable' in problem_id_lower:
        found_stable_or_balanced = False
        # Check overall scene balance if it was calculated
        if scene_context.get('is_balanced_physics') is not None:
            is_balanced_scene = scene_context['is_balanced_physics'].get('is_balanced', False)
            conf_balanced_scene = scene_context['is_balanced_physics'].get('confidence', 0.0)
            if is_balanced_scene and conf_balanced_scene > 0.7:
                found_stable_or_balanced = True
        
        # Also check if individual objects are explicitly stable and the problem implies singular stability
        if not found_stable_or_balanced and 's_only' not in problem_id_lower: # Example heuristic: if problem doesn't imply only one object
            for obj in objects:
                if obj.get('properties', {}).get('is_stable_physics') and \
                   obj.get('properties', {}).get('is_stable_physics_confidence', 0.0) > 0.7:
                    found_stable_or_balanced = True
                    break
        
        if not found_stable_or_balanced:
            _add_issue('CRITICAL', 'StabilityMismatch',
                       f"Problem ID '{problem_id}' suggests stability/balance, but no strong evidence found in physics inference (scene or objects).",
                       expected_keywords=['balanced', 'stable'])

    # Rule 10: Check for "occluded" or "visible" problems
    if 'occluded' in problem_id_lower:
        found_occluded = False
        for obj in objects:
            if obj.get('properties', {}).get('is_occluded_physics') and \
               obj.get('properties', {}).get('is_occluded_physics_confidence', 0.0) > 0.7:
                found_occluded = True
                break
        if not found_occluded:
            _add_issue('CRITICAL', 'OcclusionMismatch',
                       f"Problem ID '{problem_id}' suggests occlusion, but no strongly occluded object found via physics inference.",
                       expected_keywords=['occluded'])
    
    if 'visible' in problem_id_lower and 'occluded' not in problem_id_lower: # If problem explicitly says "visible" and not "occluded"
        found_highly_occluded = False
        for obj in objects:
            if obj.get('properties', {}).get('is_occluded_physics') and \
               obj.get('properties', {}).get('is_occluded_physics_confidence', 0.0) > 0.5: # Lower threshold to catch potential issues
                found_highly_occluded = True
                break
        if found_highly_occluded:
            _add_issue('WARNING', 'UnexpectedOcclusion',
                       f"Problem ID '{problem_id}' suggests visibility, but some objects are still flagged as physically occluded.",
                       expected_keywords=['visible', '!occluded'])

    # Rule 11: Check for "floating" or "ground" problems
    if 'floating' in problem_id_lower:
        found_floating = False
        for obj in objects:
            if obj.get('properties', {}).get('is_floating_physics') and \
               obj.get('properties', {}).get('is_floating_physics_confidence', 0.0) > 0.7:
                found_floating = True
                break
        if not found_floating:
            _add_issue('CRITICAL', 'FloatingMismatch',
                       f"Problem ID '{problem_id}' suggests floating objects, but none strongly detected.",
                       expected_keywords=['floating'])

    if 'ground' in problem_id_lower or 'touching' in problem_id_lower:
        found_touching_ground = False
        for obj in objects:
            if obj.get('properties', {}).get('is_touching_ground_physics') and \
               obj.get('properties', {}).get('is_touching_ground_physics_confidence', 0.0) > 0.7:
                found_touching_ground = True
                break
        if not found_touching_ground:
            _add_issue('CRITICAL', 'TouchingGroundMismatch',
                       f"Problem ID '{problem_id}' suggests objects touching ground, but none strongly detected.",
                       expected_keywords=['ground', 'touching'])

    # Rule 12: General Convexity/Concavity Check (if problem implies a certain type of shape)
    if 'convex' in problem_id_lower and 'polygon' in problem_id_lower:
        found_non_convex_or_concave = False
        for obj in objects:
            # If an object is confidently labeled as concave, or physically concave, and the problem implies convex
            if obj.get('confidence', 0.0) > 0.7 and (
                'concave_polygon' in obj.get('label', '').lower() or
                (obj.get('properties', {}).get('is_convex_physics') is False and obj.get('properties', {}).get('is_convex_physics_confidence', 0.0) > 0.7)
            ):
                found_non_convex_or_concave = True
                break
        if found_non_convex_or_concave:
            _add_issue('CRITICAL', 'ConvexityMismatch',
                       f"Problem ID '{problem_id}' implies convex shapes, but a confidently concave object was found.",
                       obj_id=obj.get('id'), found_label=obj.get('label'), expected_keywords=['convex'])

    # Rule 13: General Symmetry Check
    if 'symmetric' in problem_id_lower or 'symmetry' in problem_id_lower:
        found_symmetric_object = False
        for obj in objects:
            if obj.get('properties', {}).get('symmetry_type_physics') != 'none' and \
               obj.get('properties', {}).get('symmetry_score_physics_confidence', 0.0) > 0.7:
                found_symmetric_object = True
                break
        if not found_symmetric_object:
            _add_issue('WARNING', 'SymmetryMismatch',
                       f"Problem ID '{problem_id}' suggests symmetry, but no strongly symmetric object was found.",
                       expected_keywords=['symmetric', 'symmetry'])


    # --- General Image-Level Validation Checks ---

    # Rule 14: Check for presence of only 'unknown' or 'degenerate' labels
    num_objects = len(objects)
    if num_objects > 0 and num_objects >= min_object_count_for_degenerate_check:
        all_degenerate_or_unknown = True
        for obj in objects:
            label = obj.get('label', '').lower()
            if 'degenerate' not in label and 'unknown' not in label and obj.get('confidence', 0.0) >= min_confidence_for_rule:
                all_degenerate_or_unknown = False
                break
        if all_degenerate_or_unknown:
            _add_issue('CRITICAL', 'AllObjectsDegenerateOrUnknown',
                       f"Image contains {num_objects} objects, but all are labeled as 'unknown' or 'degenerate' (or have very low confidence). This suggests a processing failure.")

    # Rule 15: Check if overall image confidence is very low, regardless of specific flags
    if generated_labels_for_image.get('overall_confidence', 0.0) < 0.3: # Arbitrary low threshold
        _add_issue('WARNING', 'OverallLowConfidence',
                   f"The average confidence of all derived labels for this image is very low ({generated_labels_for_image.get('overall_confidence', 0.0):.2f}).")


    # If no issues, add an INFO message for clarity
    if not validation_issues:
        validation_issues.append({
            'severity': 'INFO',
            'rule_name': 'NoIssuesDetected',
            'message': "No critical, warning, or informational validation issues detected. Labels appear consistent."
        })

    return validation_issues

