import logging
import numpy as np
from shapely.geometry import Polygon

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
}

def post_process_scene_labels(objects, problem_id, scene_context=None):
    """
    Applies post-processing rules to refine labels based on scene-level consistency
    and adds flags for structural/semantic inconsistencies.
    Modifies objects in place.
    `scene_context` is a dictionary containing aggregated scene-level insights.
    """
    logging.debug(f"[PostProcessing] Starting post-processing for problem_id: {problem_id}")

    # Create a lookup for objects by ID
    objects_by_id = {obj['id']: obj for obj in objects}

    # Rule 1: Part-Whole Consistency
    for obj in objects:
        # Fix ambiguous array truth value
        if 'part_of' in obj and obj.get('part_of') is not None:
            logging.debug(f"[PostProcessing] part_of check passed for obj {obj.get('id')}")
            parent_id = obj['part_of']
            parent_obj = objects_by_id.get(parent_id)
            if parent_obj:
                # Example: If child is 'triangle' and parent is 'polygon', could be 'polygon_with_subparts'
                if obj['label'] in ['triangle', 'quadrilateral', 'circle'] and \
                   parent_obj['label'] in ['polygon', 'alpha_shape', 'unknown']:
                   
                    new_parent_label = f"{parent_obj['label']}_containing_{obj['label']}_part"
                    if label_specificity_score.get(new_parent_label, 0) > label_specificity_score.get(parent_obj['label'], 0):
                        parent_obj['label'] = new_parent_label
                        parent_obj['properties']['contains_parts'] = True
                        parent_obj['confidence'] = max(parent_obj['confidence'], 0.6) # Boost confidence
                        logging.debug(f"[PostProcess] Refined parent {parent_id} to {new_parent_label} due to part-whole relation.")

    # Rule 2: Ambiguity/Occlusion Resolution
    for obj in objects:
        # Fix ambiguous array truth value
        if 'ambiguity' in obj and obj.get('ambiguity') is not None and isinstance(obj['ambiguity'], (list, tuple)) and len(obj['ambiguity']) > 0:
            logging.debug(f"[PostProcessing] ambiguity check passed for obj {obj.get('id')}")
            for amb_info in obj['ambiguity']:
                relation = amb_info.get('relation')
                if relation == 'overlap_similar_size':
                    other_shape_id = [s_id for s_id in amb_info['shapes'] if s_id != obj['id']][0]
                    other_obj = objects_by_id.get(other_shape_id)
                    if other_obj:
                        common_label = 'overlapping_shapes'
                        obj['label'] = common_label
                        other_obj['label'] = common_label
                        obj['confidence'] = min(obj['confidence'], amb_info['iou'])
                        other_obj['confidence'] = min(other_obj['confidence'], amb_info['iou'])
                        obj['properties']['overlapping_with'] = other_shape_id
                        other_obj['properties']['overlapping_with'] = obj['id']
                        logging.debug(f"[PostProcess] Labeled {obj['id']} and {other_shape_id} as {common_label} due to ambiguity.")
                        obj.setdefault('flags', []).append('overlapping_shapes_flag')
                        other_obj.setdefault('flags', []).append('overlapping_shapes_flag')

                elif relation == 'occlusion_by_container' and amb_info.get('occluded') == obj['id']:
                    occluder_id = amb_info['occluder']
                    occluder_obj = objects_by_id.get(occluder_id)
                    if occluder_obj:
                        obj['label'] = f"occluded_by_{occluder_obj['label']}"
                        obj['confidence'] = obj['confidence'] * 0.8
                        obj['properties']['occluded_by'] = occluder_id
                        logging.debug(f"[PostProcess] Labeled {obj['id']} as {obj['label']} due to occlusion.")
                        obj.setdefault('flags', []).append('occluded_shape_flag')

    # Rule 3: Global Scene Patterns (Grid, Periodicity, Tiling)
    if scene_context:
        is_grid = scene_context.get('grid_pattern', {}).get('is_grid', False)
        if is_grid and scene_context.get('grid_pattern', {}).get('confidence', 0) > 0.7:
            for obj in objects:
                obj['properties']['part_of_grid_pattern'] = True
                obj['confidence'] = max(obj['confidence'], 0.7)
                obj.setdefault('flags', []).append('part_of_grid_pattern')
            logging.debug(f"[PostProcess] Marked objects as part of grid pattern.")
        
        if scene_context.get('periodicity', {}).get('score', 0) > 0.8:
            for obj in objects: obj['properties']['is_periodic_element'] = True
            for obj in objects: obj.setdefault('flags', []).append('is_periodic_element')
        
        if scene_context.get('tiling', {}).get('is_tiling', False):
            for obj in objects: obj['properties']['participates_in_tiling'] = True
            for obj in objects: obj.setdefault('flags', []).append('participates_in_tiling')

    # Rule 4: Uniformity Enforcement
    if objects is not None and isinstance(objects, (list, tuple)) and len(objects) > 1:
        # Heuristic: If problem ID contains a plural and objects are many and simple, assume uniformity
        assume_uniformity = False
        if problem_id and ("circles" in problem_id or "triangles" in problem_id or "squares" in problem_id or "polygons" in problem_id):
            if len(objects) > 2 and all(('num_points' in o and isinstance(o['num_points'], (int, float)) and o['num_points'] < 10) for o in objects):
                assume_uniformity = True

        if assume_uniformity:
            all_labels_in_scene = [obj['label'] for obj in objects]
            if all_labels_in_scene:
                most_common_label = max(set(all_labels_in_scene), key=all_labels_in_scene.count)
                
                for obj in objects:
                    if obj['label'] != most_common_label and obj['confidence'] < 0.8:
                        logging.debug(f"[PostProcess] Coercing label of {obj['id']} from {obj['label']} to {most_common_label} due to scene uniformity.")
                        obj['label'] = most_common_label
                        obj['confidence'] = obj['confidence'] * 0.9
                        obj['properties']['label_coerced_by_uniformity'] = True
                        obj.setdefault('flags', []).append('label_coerced_by_uniformity')

    # Rule 5: Structural & Semantic Consistency Checks (within each object)
    # Adding these directly as flags based on current properties
    for obj in objects:
        obj.setdefault('flags', []) # Ensure flags list exists

        # Check for rectangle-like properties if not labeled a rectangle
        if obj['label'] not in ['rectangle', 'min_area_rect'] and obj.get('properties', {}).get('num_vertices', 0) == 4:
            # Re-check rectangle properties more strictly here if needed, or rely on original detector
            # For now, just a flag if it's a quad but not detected as rect
            if obj['label'] == 'quadrilateral':
                obj['flags'].append('potential_rectangle_check')

        # Check for circularity/ellipse properties if not labeled as such
        if obj['label'] not in ['circle', 'ellipse', 'roughly_circular'] and \
           obj.get('properties', {}).get('circularity', 0) > 0.7 and obj.get('properties', {}).get('aspect_ratio', 0) > 0.8:
            obj['flags'].append('potential_circular_check')
        
        # Flag if area is very small for a detected polygon (might be degenerate)
        if obj['label'] in ['polygon', 'triangle', 'quadrilateral'] and obj.get('properties', {}).get('area', 0) < 50: # Heuristic threshold
            obj['flags'].append('small_area_polygon_check')

        # Flag if `is_composite_shape` heuristic applies but label is simple
        if obj.get('properties', {}).get('is_composite_shape', False) and \
           obj['label'] in ['polygon', 'line', 'circle', 'unknown']:
           obj['flags'].append('potential_composite_shape_mismatch')
        
        # Flag if `is_smooth_curve` applies but label indicates sharp corners
        if obj.get('properties', {}).get('is_smooth_curve', False) and \
           obj['label'] in ['triangle', 'quadrilateral', 'polygon', 'polyline']:
           obj['flags'].append('potential_smooth_curve_mismatch')


    logging.debug("[PostProcessing] Finished post-processing.")
    return objects # Return the modified objects list
