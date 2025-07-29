import argparse
import sys
import os
import ijson
import json
import logging
import numpy as np
import shutil # For copying files for human review

# --- Add missing imports for helper functions/classes ---
from derive_label.geometric_detectors import detect_vertices
from shapely.geometry import Polygon
from derive_label.pattern_analysis import symmetry_score_from_vertices, boundary_autocorr_symmetry

# Ensure src is importable (needed for BongardLoader, BongardLogoParser, PhysicsInference)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modularized functions
from derive_label.confidence_scoring import calculate_confidence
from derive_label.geometric_detectors import (
    ransac_line, ransac_circle, ellipse_fit, hough_lines_detector,
    convexity_defects_detector, fourier_descriptors, cluster_points_detector,
    is_roughly_circular_detector, is_rectangle_detector, is_point_cloud_detector,
    compute_symmetry_axis, compute_orientation, compute_symmetry_score
)
from derive_label.image_features import (
    image_processing_features, compute_euler_characteristic,
    persistent_homology_features,

)
# Note: create_shape_from_stroke_pipeline is expected to use a BongardLogoParser instance.
# We are ensuring that the BongardLogoParser instance it receives has the correct parse_action_program method.
from derive_label.logo_processing import flatten_action_program, create_shape_from_stroke_pipeline
from derive_label.pattern_analysis import (
    detect_grid_pattern, detect_periodicity, detect_tiling,
    detect_rotation_scale_reflection, triplet_relations
)
from derive_label.semantic_refinement import (
    detect_connected_line_segments, is_quadrilateral_like, is_smooth_curve,
    is_composite_shape, generate_semantic_label, infer_shape_type_ensemble,
    analyze_detector_consensus
)
from derive_label.post_processing import post_process_scene_labels
from derive_label.validation import validate_label_quality

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# --- Start of Corrected BongardLogoParser based on old code's implied behavior ---
# This class provides the parsing logic for LOGO action programs,
# which the previous BongardLogoParser seemed to be missing or have an incorrect signature for.
class _CorrectedBongardLogoParser:
    def __init__(self):
        self.pen_down = False
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0

    def parse_action_program(self, action_list, scale=120):
        import re
        vertices = []
        self.pen_down = False
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0

        for cmd_str in action_list:
            parts = cmd_str.strip().split()
            command = parts[0].upper()

            try:
                if command == 'PD':
                    self.pen_down = True
                    if not vertices:
                        vertices.append([self.x, self.y])
                elif command == 'PU':
                    self.pen_down = False
                elif command == 'FD':
                    distance = float(parts[1]) * scale / 100.0
                    rad_angle = np.deg2rad(self.angle)
                    new_x = self.x + distance * np.cos(rad_angle)
                    new_y = self.y + distance * np.sin(rad_angle)
                    if self.pen_down:
                        vertices.append([new_x, new_y])
                    self.x = new_x
                    self.y = new_y
                elif command == 'RT':
                    angle_change = float(parts[1])
                    self.angle -= angle_change
                elif command == 'LT':
                    angle_change = float(parts[1])
                    self.angle += angle_change
                elif command == 'HOME':
                    self.x = 0.0
                    self.y = 0.0
                    self.angle = 0.0
                    if self.pen_down:
                        vertices.append([self.x, self.y])
                else:
                    # Generic: extract last two numbers as x, y (handles arc_*, line_*, etc.)
                    # Accepts both _ and - as separators
                    # Examples: arc_zigzag_0.500_0.625-0.500, line_normal_0.217-0.750
                    match = re.search(r'([-+]?[0-9]*\.?[0-9]+)[_-]([-+]?[0-9]*\.?[0-9]+)$', command)
                    if match:
                        x = float(match.group(1)) * scale
                        y = float(match.group(2)) * scale
                        vertices.append([x, y])
                    else:
                        logging.warning(f"Unknown LOGO command: {cmd_str}")
            except (ValueError, IndexError) as e:
                logging.error(f"Error parsing LOGO command '{cmd_str}': {e}")

        # Remove consecutive duplicates
        if vertices:
            unique_vertices = [vertices[0]]
            for i in range(1, len(vertices)):
                if not np.allclose(vertices[i], vertices[i-1]):
                    unique_vertices.append(vertices[i])
            # Convert all coords to native Python float for JSON safety
            py_vertices = [[float(x), float(y)] for x, y in unique_vertices]
            return [{'coords': py_vertices}]
        return []
# --- End of Corrected BongardLogoParser ---

def main():

    # --- Argument Parsing and Variable Initialization ---
    parser = argparse.ArgumentParser(description='Process ShapeBongard_V2 data and extract shape attributes.')
    parser.add_argument('--input-dir', required=True, help='Input directory containing ShapeBongard_V2 data')
    parser.add_argument('--output', required=True, help='Output JSON file to save extracted shape data')
    parser.add_argument('--problems-list', required=False, help='Optional file with list of problem IDs to process')
    parser.add_argument('--human-review-dir', required=False, default='human_review_batches', help='Directory to save images for human review')

    args = parser.parse_args()

    # Ensure output directories exist
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    human_review_base_dir = args.human_review_dir
    if not os.path.exists(human_review_base_dir):
        os.makedirs(human_review_base_dir)

    categories = ['bd', 'ff', 'hd']
    base_dir = args.input_dir
    output_path = args.output
    problems_list_path = args.problems_list

    # Read problem IDs from problems-list file
    required_ids = None
    if problems_list_path:
        with open(problems_list_path, 'r') as f:
            required_ids = set(line.strip() for line in f if line.strip())

    all_results = []
    flagged_cases = [] # This will store cases for potential human review
    all_labels = set()
    all_shape_types = set()
    all_categories = set()

    # Pass the _CorrectedBongardLogoParser to the shape creation pipeline dependencies
    # This ensures that create_shape_from_stroke_pipeline uses our corrected parser.
    shape_creation_dependencies = {
        'BongardLogoParser': _CorrectedBongardLogoParser(), # Use the locally defined corrected parser
        'calculate_confidence': calculate_confidence,
        'ransac_line': ransac_line,
        'ransac_circle': ransac_circle,
        'ellipse_fit': ellipse_fit,
        'hough_lines_detector': hough_lines_detector,
        'convexity_defects_detector': convexity_defects_detector,
        'fourier_descriptors': fourier_descriptors,
        'cluster_points_detector': cluster_points_detector,
        'is_roughly_circular_detector': is_roughly_circular_detector,
        'is_rectangle_detector': is_rectangle_detector,
        'is_point_cloud_detector': is_point_cloud_detector,
        'image_processing_features': image_processing_features,
        'compute_euler_characteristic': compute_euler_characteristic,
        'persistent_homology_features': persistent_homology_features,
        'compute_symmetry_axis': compute_symmetry_axis,
        'compute_orientation': compute_orientation,
        'compute_symmetry_score': compute_symmetry_score,
        'detect_vertices': detect_vertices,
        'detect_connected_line_segments': detect_connected_line_segments,
        'generate_semantic_label': generate_semantic_label,
        'infer_shape_type_ensemble': infer_shape_type_ensemble,
        'is_quadrilateral_like': is_quadrilateral_like,
        'is_smooth_curve': is_smooth_curve,
        'is_composite_shape': is_composite_shape
    }

    for cat in categories:
        json_path = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/{cat}_action_programs.json")
        if not os.path.exists(json_path):
            logging.warning(f"Missing JSON file: {json_path}")
            continue

        with open(json_path, 'r') as f:
            for problem_id, pos_neg_lists in ijson.kvitems(f, ''):
                if required_ids and problem_id not in required_ids:
                    continue
                for label_type, group in zip(['category_1', 'category_0'], pos_neg_lists):
                    norm_label = 'positive' if label_type == 'category_1' else 'negative'
                    for idx, action_program in enumerate(group):
                        # Remap label_type to match actual folder names (category_0 -> 0, category_1 -> 1)
                        mapped_label_type = label_type.replace('category_0', '0').replace('category_1', '1')
                        img_dir = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/images/{problem_id}", mapped_label_type)
                        img_path = os.path.join(img_dir, f"{idx}.png")
                        # Normalize path separators for consistency (especially for logging and downstream use)
                        img_path = os.path.normpath(img_path)
                        
                        flat_commands = [cmd for cmd in flatten_action_program(action_program) if isinstance(cmd, str)]
                        
                        current_image_flag_reasons = [] # Reasons for high-level flagging this specific image
                        validation_issues_list = [] # List to store detailed validation issues
                        
                        try:
                            # Extract multiple shapes from the action program
                            objects, fallback_reasons = create_shape_from_stroke_pipeline(
                                flat_commands, problem_id=problem_id, **shape_creation_dependencies
                            )
                            if not objects:
                                current_image_flag_reasons.append(f"No objects extracted from LOGO commands: {fallback_reasons}")
                                logging.warning(f"No objects extracted for problem_id={problem_id}, image={img_path}, fallback_reasons={fallback_reasons}")
                                # No need to continue processing if no objects, add to flagged_cases later.
                                image_record = { # Still create an image record for error logging
                                    'problem_id': problem_id,
                                    'category': cat,
                                    'label': norm_label,
                                    'image_path': img_path,
                                    'objects': [],
                                    'action_program': flat_commands,
                                    'num_objects': 0,
                                    'object_labels': [],
                                    'object_shape_types': [],
                                    'object_categories': [],
                                    'scene_level_analysis': {},
                                    'overall_confidence': 0.0,
                                    'confidence_tier': "Tier 4: No Objects Extracted",
                                    'flagged_reasons': current_image_flag_reasons,
                                    'validation_issues': [{'severity': 'CRITICAL', 'rule_name': 'NoObjectsExtracted', 'message': fallback_reasons}]
                                }
                                all_results.append(image_record)
                                # Flag this case for review
                                flagged_cases.append({
                                    'problem_id': problem_id,
                                    'image_path': img_path,
                                    'review_image_path': os.path.join(human_review_base_dir, f"{problem_id}_{norm_label}", os.path.basename(img_path)) if os.path.exists(img_path) else "SOURCE_MISSING",
                                    'derived_labels_summary': ["NO_OBJECTS"],
                                    'overall_confidence': 0.0,
                                    'confidence_tier': "Tier 4: No Objects Extracted",
                                    'flagging_reasons': current_image_flag_reasons,
                                    'action_program': flat_commands,
                                    'validation_issues': image_record['validation_issues']
                                })
                                continue


                            # --- Scene-level Pattern Analysis (grid, periodicity, tiling, triplet relations) ---
                            # Extract centers and polygons for pattern analysis
                            centers = [np.mean(obj['coords'], axis=0) if 'coords' in obj and obj['coords'] else None for obj in objects]
                            centers = [c for c in centers if c is not None]
                            polygons = []
                            for obj in objects:
                                if 'coords' in obj and obj['coords'] and len(obj['coords']) >= 3:
                                    try:
                                        polygons.append(Polygon(obj['coords']))
                                    except Exception:
                                        pass
                            scene_bbox = None
                            if polygons:
                                union_poly = polygons[0]
                                for p in polygons[1:]:
                                    union_poly = union_poly.union(p)
                                if not union_poly.is_empty:
                                    minx, miny, maxx, maxy = union_poly.bounds
                                    scene_bbox = (minx, miny, maxx, maxy)

                            # Pattern analysis
                            grid_is_grid, grid_info, grid_conf = detect_grid_pattern(centers) if centers else (False, None, 0.0)
                            periodic_is_periodic, periodic_periods, periodic_conf = detect_periodicity(centers) if centers else (False, [], 0.0)
                            tiling_is_tiling, tiling_fill, tiling_overlap, tiling_conf = detect_tiling(polygons, scene_bbox) if polygons else (False, 0.0, 0.0, 0.0)
                            triplets, triplet_conf = triplet_relations(centers) if centers else ([], 0.0)

                            scene_level_analysis = {
                                'grid_pattern': {'is_grid': grid_is_grid, 'info': grid_info, 'confidence': grid_conf},
                                'periodicity': {'is_periodic': periodic_is_periodic, 'periods': periodic_periods, 'score': periodic_conf},
                                'tiling': {'is_tiling': tiling_is_tiling, 'fill_ratio': tiling_fill, 'overlap_ratio': tiling_overlap, 'confidence': tiling_conf},
                                'triplet_relations': {'triplets': triplets, 'confidence': triplet_conf},
                            }

                            # --- MCTS Scene-level Labeling (optional, advanced) ---
                            try:
                                from derive_label.mcts_labeling import label_scene_with_mcts
                                mcts_labeled_objects = label_scene_with_mcts(objects, n_sim=100)
                                # Attach MCTS labels to each object
                                for obj, mcts_obj in zip(objects, mcts_labeled_objects):
                                    obj['mcts_label'] = mcts_obj.get('mcts_label', None)
                                scene_level_analysis['mcts_labels'] = [obj.get('mcts_label', None) for obj in objects]
                            except Exception as mcts_e:
                                scene_level_analysis['mcts_labels'] = f"MCTS labeling failed: {mcts_e}"

                            # --- Post-processing with scene context ---
                            post_processed_objects = post_process_scene_labels(objects, problem_id, scene_level_analysis)

                            # --- Cross-Detector Consensus Scoring & Confidence Aggregation ---
                            overall_image_confidence = 0.0
                            has_detector_disagreement = False
                            for obj in post_processed_objects:
                                # Analyze consensus for each object
                                consensus_info = analyze_detector_consensus(obj['possible_labels'], obj['label'])
                                obj['properties']['detector_consensus'] = consensus_info
                                # --- Robust symmetry detection from LOGO vertices ---
                                coords = obj.get('coords') or obj.get('properties', {}).get('coords')
                                if coords is not None and isinstance(coords, list) and len(coords) >= 3:
                                    score, axis = symmetry_score_from_vertices(coords)
                                    obj['properties']['robust_symmetry_score'] = score
                                    obj['properties']['robust_symmetry_axis_deg'] = axis
                                    # --- Unsupervised boundary autocorrelation symmetry ---
                                    obj['properties']['boundary_autocorr_symmetry'] = boundary_autocorr_symmetry(coords)
                                else:
                                    obj['properties']['robust_symmetry_score'] = None
                                    obj['properties']['robust_symmetry_axis_deg'] = None
                                    obj['properties']['boundary_autocorr_symmetry'] = None
                                if not consensus_info['agreement']:
                                    has_detector_disagreement = True
                                    current_image_flag_reasons.append(f"Object {obj['id']} has detector disagreement: {consensus_info['disagreement_reason']}")
                                overall_image_confidence += obj['confidence']

                            if len(post_processed_objects) > 0:
                                overall_image_confidence /= len(post_processed_objects) # Average confidence
                            else:
                                overall_image_confidence = 0.0 # No objects, no confidence

                            # Create the full image_record to pass to validation
                            image_record = {
                                'problem_id': problem_id,
                                'category': cat,
                                'label': norm_label,
                                'image_path': img_path,
                                'objects': post_processed_objects, # Use post-processed objects here
                                'action_program': flat_commands,
                                'num_objects': len(post_processed_objects),
                                'object_labels': list(sorted(list(set(obj.get('label', '') for obj in post_processed_objects)))),
                                'object_shape_types': list(sorted(list(set(obj.get('shape_type', '') for obj in post_processed_objects)))),
                                'object_categories': list(sorted(list(set(obj.get('category', '') for obj in post_processed_objects)))),
                                'scene_level_analysis': scene_level_analysis,
                                'overall_confidence': overall_image_confidence,
                                'flagged_reasons': current_image_flag_reasons, # Keep these high-level flags
                            }

                            # --- Professional Validation Call ---
                            validation_issues_list = validate_label_quality(image_record, problem_id)
                            has_critical_validation_issue = any(issue['severity'] == 'CRITICAL' for issue in validation_issues_list)
                            confidence_tier = "Tier 4: Low-Confidence/Ambiguous"
                            if overall_image_confidence > 0.98 and not has_detector_disagreement and not has_critical_validation_issue and not current_image_flag_reasons:
                                confidence_tier = "Tier 1: Gold Labels"
                            elif overall_image_confidence > 0.90 and not has_detector_disagreement and not has_critical_validation_issue:
                                confidence_tier = "Tier 2: High-Confidence"
                            elif overall_image_confidence > 0.50 and not has_critical_validation_issue:
                                confidence_tier = "Tier 3: Uncertain"
                            if has_detector_disagreement:
                                confidence_tier = "Tier 4: Low-Confidence/Ambiguous (Detector Disagreement)"
                            if has_critical_validation_issue:
                                confidence_tier = "Tier 4: Low-Confidence/Ambiguous (Validation Failed)"
                            if current_image_flag_reasons:
                                confidence_tier = "Tier 4: Low-Confidence/Ambiguous (Flagged Issues)"

                            image_record['confidence_tier'] = confidence_tier
                            image_record['validation_issues'] = validation_issues_list # Store all detailed validation issues

                            logging.info(f"Processed image: {img_path} | problem_id={problem_id} | num_objects={len(post_processed_objects)} | labels={image_record['object_labels']} | confidence_tier={confidence_tier} | Validation Issues: {len(validation_issues_list) - (1 if any(issue['rule_name'] == 'NoIssuesDetected' for issue in validation_issues_list) else 0)}")
                            all_results.append(image_record)

                            # --- Human Review Data Export Logic ---
                            needs_review = (
                                confidence_tier in [
                                    "Tier 3: Uncertain",
                                    "Tier 4: Low-Confidence/Ambiguous",
                                    "Tier 4: Low-Confidence/Ambiguous (Detector Disagreement)",
                                    "Tier 4: Low-Confidence/Ambiguous (Flagged Issues)",
                                    "Tier 4: Low-Confidence/Ambiguous (Validation Failed)",
                                    "Tier 4: Processing Error"
                                ] or
                                has_critical_validation_issue or
                                any(issue['severity'] == 'WARNING' for issue in validation_issues_list)
                            )

                            if needs_review:
                                review_sub_dir = os.path.join(human_review_base_dir, f"{problem_id}_{norm_label}")
                                os.makedirs(review_sub_dir, exist_ok=True)
                                dest_img_path = os.path.join(review_sub_dir, os.path.basename(img_path))
                                dest_label_path = os.path.splitext(dest_img_path)[0] + '.json'
                                try:
                                    if os.path.exists(img_path):
                                        shutil.copy(img_path, dest_img_path)
                                        # Save label/metadata JSON alongside the image
                                        with open(dest_label_path, 'w') as label_out:
                                            json.dump(make_json_safe({
                                                'problem_id': problem_id,
                                                'generated_labels': image_record['object_labels'],
                                                'num_objects': image_record['num_objects'],
                                                'confidence_tier': confidence_tier,
                                                'flagging_reasons': current_image_flag_reasons,
                                                'validation_issues': validation_issues_list,
                                                'action_program': flat_commands,
                                                'object_properties': [obj.get('properties', {}) for obj in post_processed_objects],
                                            }), label_out, indent=2)
                                        flagged_cases.append({
                                            'problem_id': problem_id,
                                            'real_image_path': dest_img_path, # Path to the copied image in review dir
                                            'generated_labels': image_record['object_labels'],
                                            'num_objects': image_record['num_objects'],
                                            'confidence_tier': confidence_tier,
                                            'flagging_reasons': current_image_flag_reasons,
                                            'validation_issues': validation_issues_list,
                                            'action_program': flat_commands
                                        })
                                    else:
                                        logging.error(f"Source image not found for review: {img_path}")
                                        flagged_cases.append({
                                            'problem_id': problem_id,
                                            'real_image_path': "SOURCE_MISSING",
                                            'generated_labels': image_record['object_labels'],
                                            'num_objects': image_record['num_objects'],
                                            'confidence_tier': confidence_tier,
                                            'flagging_reasons': current_image_flag_reasons + ["Image source file missing."],
                                            'validation_issues': validation_issues_list + [{'severity': 'CRITICAL', 'rule_name': 'ImageSourceMissing', 'message': "Source image file not found for review export."}],
                                            'action_program': flat_commands
                                        })
                                except Exception as copy_e:
                                    logging.error(f"Failed to copy image {img_path} for review: {copy_e}")
                                    flagged_cases.append({
                                        'problem_id': problem_id,
                                        'real_image_path': "COPY_FAILED",
                                        'generated_labels': image_record['object_labels'],
                                        'num_objects': image_record['num_objects'],
                                        'confidence_tier': confidence_tier,
                                        'flagging_reasons': current_image_flag_reasons + [f"Image copy failed: {copy_e}"],
                                        'validation_issues': validation_issues_list + [{'severity': 'CRITICAL', 'rule_name': 'ImageCopyFailed', 'message': f"Failed to copy image for review: {copy_e}"}],
                                        'action_program': flat_commands
                                    })

                        except Exception as e:
                            import traceback
                            logging.error(f"Unhandled exception for problem_id={problem_id}, image={img_path}: {e}\n" + traceback.format_exc())
                            # Log local variable context for deep debugging
                            logging.error(f"Local context: problem_id={problem_id}, img_path={img_path}, flat_commands={flat_commands}, cat={cat}, label_type={label_type}, norm_label={norm_label}")
                            error_details = {'problem_id': problem_id, 'image_path': img_path, 'error': str(e), 'action_program': flat_commands}
                            flagged_cases.append({
                                'problem_id': problem_id,
                                'image_path': img_path,
                                'review_image_path': "N/A - Processing Error",
                                'derived_labels_summary': ["PROCESSING_ERROR"],
                                'overall_confidence': 0.0,
                                'confidence_tier': "Tier 4: Processing Error",
                                'flagging_reasons': current_image_flag_reasons + [f"Unhandled processing error: {e}"],
                                'validation_issues': [{'severity': 'CRITICAL', 'rule_name': 'UnhandledProcessingError', 'message': str(e)}],
                                'action_program': flat_commands
                            })


    # --- Final Output: Save Results and Print Summary (MOVED TO END) ---
    # Save all results to the main output JSON
    try:
        with open(output_path, 'w') as out_f:
            json.dump(make_json_safe(all_results), out_f, indent=2)
        logging.info(f"Saved {len(all_results)} cases to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save main output JSON: {e}")

    # Save flagged cases for review to a separate file
    review_json_path = os.path.splitext(output_path)[0] + '_flagged_for_review.json'
    try:
        with open(review_json_path, 'w') as review_f:
            json.dump(make_json_safe(flagged_cases), review_f, indent=2)
        logging.info(f"Saved {len(flagged_cases)} flagged cases for review to {review_json_path}")
    except Exception as e:
        logging.error(f"Failed to save flagged review JSON: {e}")

    # Print summary
    print("\n--- Summary ---")
    print(f"Total cases processed: {len(all_results)}")
    print(f"Cases flagged for review: {len(flagged_cases)}")
    print(f"Output saved to: {output_path}")
    print(f"Flagged review cases saved to: {review_json_path}")


# --- Utility: Make JSON Safe ---
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float32)): # Fixed a typo here, was np.float32, np.float64
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

if __name__ == '__main__':
    main()
