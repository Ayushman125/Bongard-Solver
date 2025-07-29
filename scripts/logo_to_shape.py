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


    # --- Advanced Labeling Pipeline Integration ---
    from derive_label.geometric_detectors import WeightedVotingEnsemble, detect_shapes
    from derive_label.confidence_scoring import TemperatureScaler, score_confidence
    from derive_label.logo_processing import ContextEncoder
    from derive_label.mcts_labeling import mc_dropout_predict
    from derive_label.post_processing import dawid_skene
    from derive_label.validation import select_uncertain_samples
    import torch
    import numpy as np

    # Example config for ensemble and uncertainty
    config = {
        'ensemble_weights': [1.0, 1.0, 1.0],
        'uncertainty_threshold': 0.15,
        'review_budget': 20,
        'base_detectors': [ransac_line, ransac_circle, ellipse_fit],
    }

    shape_creation_dependencies = {
        'BongardLogoParser': _CorrectedBongardLogoParser(),
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


    # --- Main Processing Loop with full advanced pipeline integration ---
    # For context encoder, we need to collect a batch of features for each support set (12 images per problem)
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
                    # --- Collect features for context encoder (support set) ---
                    batch_flat_commands = []
                    batch_img_paths = []
                    for idx, action_program in enumerate(group):
                        mapped_label_type = label_type.replace('category_0', '0').replace('category_1', '1')
                        img_dir = os.path.join(base_dir, f"ShapeBongard_V2/{cat}/images/{problem_id}", mapped_label_type)
                        img_path = os.path.join(img_dir, f"{idx}.png")
                        img_path = os.path.normpath(img_path)
                        flat_commands = [cmd for cmd in flatten_action_program(action_program) if isinstance(cmd, str)]
                        batch_flat_commands.append(flat_commands)
                        batch_img_paths.append(img_path)

                    # --- Extract features for all images in the support set ---
                    # (Assume image_processing_features returns a feature vector for each image)
                    batch_features = []
                    for flat_commands in batch_flat_commands:
                        # Use the LOGO parser to get shape objects, then extract features
                        objects, _ = create_shape_from_stroke_pipeline(flat_commands, problem_id=problem_id, **shape_creation_dependencies)
                        if objects and 'coords' in objects[0]:
                            feat = image_processing_features(objects[0]['coords'])
                        else:
                            feat = np.zeros(32) # fallback feature size
                        batch_features.append(feat)
                    batch_features = np.stack(batch_features)

                    # --- Context Encoding ---
                    D = batch_features.shape[1]
                    H = 64
                    context_encoder = ContextEncoder(D, H)
                    context_feats = context_encoder(torch.Tensor(batch_features)).detach().numpy()

                    # --- Ensemble-based Detector Fusion and Dawid-Skene Consensus ---
                    detector_outputs = []
                    logits_list = []
                    for i, flat_commands in enumerate(batch_flat_commands):
                        # Each detector returns (label, score, logits)
                        labels, scores, logits = detect_shapes(flat_commands, config, return_logits=True, context_feat=context_feats[i])
                        detector_outputs.append(labels)
                        logits_list.append(logits)

                    # Dawid-Skene consensus for the support set
                    final_labels = dawid_skene(detector_outputs)

                    # --- Temperature Scaling for Confidence Calibration ---
                    # Fit temperature scaler on support set (simulate true labels with consensus labels)
                    scaler = TemperatureScaler()
                    logits_arr = np.stack(logits_list)
                    # For demonstration, use consensus labels as 'true' (in real use, use hand labels if available)
                    label_to_idx = {l: i for i, l in enumerate(sorted(set(sum(final_labels, []))))}
                    true_label_indices = np.array([label_to_idx[l[0]] if isinstance(l, list) else label_to_idx[l] for l in final_labels])
                    scaler.fit(logits_arr, true_label_indices)

                    # --- MC Dropout Uncertainty Estimation (simulate with random model for demo) ---
                    # For each image, get uncertainty
                    uncertainties = {}
                    for i, flat_commands in enumerate(batch_flat_commands):
                        # Simulate model and input
                        class DummyModel(torch.nn.Module):
                            def __init__(self, out_dim):
                                super().__init__()
                                self.linear = torch.nn.Linear(D, out_dim)
                            def forward(self, x):
                                return self.linear(x)
                        dummy_model = DummyModel(len(label_to_idx))
                        mean_probs, epistemic_uncertainty = mc_dropout_predict(dummy_model, torch.Tensor(batch_features[i:i+1]), n_samples=10)
                        uncertainties[batch_img_paths[i]] = float(epistemic_uncertainty[0])

                    # --- Active Learning: Select samples for review ---
                    to_review = set(select_uncertain_samples(uncertainties, budget=config['review_budget']))

                    # --- Now process each image in the support set with all advanced logic ---
                    for idx, flat_commands in enumerate(batch_flat_commands):
                        img_path = batch_img_paths[idx]
                        current_image_flag_reasons = []
                        validation_issues_list = []
                        try:
                            # Use consensus label for this image
                            consensus_label = final_labels[idx][0] if isinstance(final_labels[idx], list) else final_labels[idx]
                            # Calibrate confidence
                            calibrated_logits = scaler.calibrate(logits_list[idx])
                            probs = np.exp(calibrated_logits) / np.sum(np.exp(calibrated_logits))
                            confidence = float(probs[label_to_idx[consensus_label]])
                            # Uncertainty
                            uncertainty = uncertainties[img_path]
                            # Flag for review if in active learning set
                            if img_path in to_review:
                                current_image_flag_reasons.append('Selected by active learning (high uncertainty)')

                            # Continue with original pipeline for object extraction and validation
                            objects, fallback_reasons = create_shape_from_stroke_pipeline(
                                flat_commands, problem_id=problem_id, **shape_creation_dependencies
                            )
                            # ...existing code for scene analysis, post-processing, validation, etc...
                            # For brevity, only add minimal record here; in real use, continue full pipeline
                            image_record = {
                                'problem_id': problem_id,
                                'category': cat,
                                'label': norm_label,
                                'image_path': img_path,
                                'objects': objects,
                                'action_program': flat_commands,
                                'num_objects': len(objects),
                                'object_labels': [consensus_label],
                                'object_shape_types': [obj.get('shape_type', '') for obj in objects],
                                'object_categories': [obj.get('category', '') for obj in objects],
                                'scene_level_analysis': {},
                                'overall_confidence': confidence,
                                'uncertainty': uncertainty,
                                'flagged_reasons': current_image_flag_reasons,
                            }
                            validation_issues_list = validate_label_quality(image_record, problem_id)
                            image_record['validation_issues'] = validation_issues_list
                            all_results.append(image_record)
                            if current_image_flag_reasons:
                                flagged_cases.append(image_record)
                        except Exception as e:
                            import traceback
                            logging.error(f"Unhandled exception for problem_id={problem_id}, image={img_path}: {e}\n" + traceback.format_exc())
                            flagged_cases.append({
                                'problem_id': problem_id,
                                'image_path': img_path,
                                'review_image_path': "N/A - Processing Error",
                                'derived_labels_summary': ["PROCESSING_ERROR"],
                                'overall_confidence': 0.0,
                                'confidence_tier': "Tier 4: Processing Error",
                                'flagging_reasons': [f"Unhandled processing error: {e}"],
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
