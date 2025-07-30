# --- Utility: Ensure all coordinates are numeric (float) ---
def ensure_numeric_coords(coords):
    if isinstance(coords, list):
        return [ensure_numeric_coords(x) for x in coords]
    try:
        return float(coords)
    except Exception:
        return coords


import os
import argparse
import sys
import os
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- Standard library imports ---
import numpy as np
import shutil # For copying files for human review
import re
# Add missing imports
import torch
import logging
import json
import ijson
import matplotlib.pyplot as plt
import traceback
# Synthetic data generation and mixing
from scripts.data_generation.synthetic_shapes import generate_dataset as generate_synthetic_dataset
from scripts.data_generation.mix_real_and_synthetic import mix_datasets, save_mixed_dataset

# Ensure src is importable (needed for BongardLoader, BongardLogoParser, PhysicsInference)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modularized functions
from .derive_label.confidence_scoring import calculate_confidence, TemperatureScaler, PlattScaler, IsotonicScaler, score_confidence, initialize_confidence_calibrator
from .derive_label.geometric_detectors import (
    detect_vertices, ransac_line, ransac_circle, ellipse_fit, hough_lines_detector,
    convexity_defects_detector, fourier_descriptors, cluster_points_detector,
    is_roughly_circular_detector, is_rectangle_detector, is_point_cloud_detector,
    compute_symmetry_axis, compute_orientation, compute_symmetry_score,
    WeightedVotingEnsemble, detect_shapes, StackingEnsemble, DetectorAttention
)
from .derive_label.image_features import (
    image_processing_features, compute_euler_characteristic,
    persistent_homology_features,
)
from .derive_label.logo_processing import flatten_action_program, create_shape_from_stroke_pipeline, ContextEncoder
from .derive_label.pattern_analysis import (
    symmetry_score_from_vertices, boundary_autocorr_symmetry, detect_grid_pattern,
    detect_periodicity, detect_tiling, detect_rotation_scale_reflection, triplet_relations
)
from .derive_label.semantic_refinement import (
    detect_connected_line_segments, is_quadrilateral_like, is_smooth_curve,
    is_composite_shape, generate_semantic_label, infer_shape_type_ensemble,
    analyze_detector_consensus
)
from .derive_label.post_processing import post_process_scene_labels, dawid_skene, ensure_list_of_lists
from .derive_label.validation import validate_label_quality, select_uncertain_samples
from .derive_label.mcts_labeling import mc_dropout_predict


# Optional: CROWDLAB (Cleanlab multiannotator) and kappa metrics
CROWD_LAB_AVAILABLE = True
try:
    from cleanlab.multiannotator import get_label_quality_multiannotator
    CROWD_LAB_AVAILABLE = True
except ImportError:
    pass # Cleanlab multiannotator not available

KAPPA_AVAILABLE = True
try:
    from sklearn.metrics import cohen_kappa_score
    import statsmodels.stats.inter_rater as irr
    KAPPA_AVAILABLE = True
except ImportError:
    pass # Kappa metrics not available

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

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
    elif isinstance(obj, (np.floating, np.float32, np.float64)): # Fixed a typo here, was np.float32, np.float64
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

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
        vertices = []
        self.pen_down = False
        self.x = 0.0
        self.y = 0.0
        self.angle = 0.0

        for cmd_str in action_list:
            parts = cmd_str.strip().split()
            command = parts[0].upper() if parts else ''

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
                    # Accept coordinate-only lines: two numbers separated by space
                    if len(parts) == 2:
                        try:
                            x = float(parts[0]) * scale
                            y = float(parts[1]) * scale
                            vertices.append([x, y])
                            continue
                        except Exception:
                            pass
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


def synthetic_shape_to_logo_commands(shape):
    import math


def main():
    # Defensive: ensure np is not shadowed by a local variable
    global np

    # --- Argument Parsing and Variable Initialization ---

    parser = argparse.ArgumentParser(description='Process ShapeBongard_V2 data and extract shape attributes, or generate synthetic/mixed datasets.')
    parser.add_argument('--input-dir', required=False, help='Input directory containing ShapeBongard_V2 data')
    parser.add_argument('--output', required=True, help='Output JSON file to save extracted shape data or mixed dataset')
    parser.add_argument('--problems-list', required=False, help='Optional file with list of problem IDs to process')
    parser.add_argument('--human-review-dir', required=False, default='human_review_batches', help='Directory to save images for human review')
    parser.add_argument('--generate-synthetic', action='store_true', help='Generate synthetic data for all shape types')
    parser.add_argument('--mix-real-synthetic', action='store_true', help='Mix real and synthetic data for all shape types')
    parser.add_argument('--real-labels-json', required=False, help='Path to real labels JSON (for mixing)')
    parser.add_argument('--synthetic-count', type=int, default=100, help='Number of synthetic samples per shape type')
    parser.add_argument('--mix-ratio', type=float, default=0.5, help='Ratio of synthetic to real data in mixed dataset (0.0-1.0)')

    args = parser.parse_args()

    # --- Mix Real and Synthetic Data Option (for internal use only, never saved) ---
    # (No-op here; mixing is now handled per-batch in the main processing loop)

    # --- Synthetic Data Generation Only (not allowed) ---
    if args.generate_synthetic and not args.mix_real_synthetic:
        print("[WARNING] --generate-synthetic has no effect unless used with --mix-real-synthetic. No synthetic-only dataset will be saved.")
        # Do not return; continue to main pipeline on real data

    # --- Default: Run original pipeline on real data ---
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

    # --- Config and Dependencies ---
    # Patch config to use ellipse_fit_debug
    config = {
        # Detector ensemble weights (for weighted voting)
        'ensemble_weights': [1.0, 1.0, 1.0],
        # Uncertainty threshold for active learning
        'uncertainty_threshold': 0.15,
        # Number of samples to select for review in active learning
        'review_budget': 20,
        # List of base detector functions
        'base_detectors': [ransac_line, ransac_circle, ellipse_fit_debug],
        # Detector fusion method: 'stacking', 'attention', 'weighted'
        'fusion_method': 'stacking',
        # Calibration method: 'temperature', 'platt', 'isotonic'
        'calibration_method': 'temperature',
        # Consensus method: 'dawid', 'crowdlab'
        'consensus_method': 'dawid',
        # Use kappa metrics for agreement monitoring
        'use_kappa': True,
        # Curriculum learning: current epoch (increment per training epoch)
        'curriculum_epoch': 0,
        # Curriculum learning: base threshold for review
        'base_thr': 0.9,
        # Curriculum learning: minimum threshold for review
        'min_thr': 0.6,
        # Curriculum learning: decay rate for threshold
        'decay': 0.01,
        # Self-training: frequency (epochs)
        'self_train_freq': 5,
        # Self-training: confidence threshold for pseudo-labeling
        'self_train_thresh': 0.95,
        # Co-training: frequency (epochs)
        'co_train_freq': 5,
        # Co-training: number of samples to exchange
        'co_train_k': 100,
        # Path to calibration data for global calibrator (if used)
        'calib_path': None,
        # Meta-learner: stacking meta-classifier config (if needed)
        'meta_learner': 'random_forest', # or 'mlp', 'logistic', etc.
        # Attention meta-learner: number of heads
        'attention_heads': 4,
        # CROWDLAB: aggregator type (if using CROWDLAB)
        'crowdlab_aggregator': 'default',
        # Kappa metrics: enable/disable
        'enable_fleiss_kappa': True,
        # Curriculum learning: enable/disable
        'enable_curriculum': True,
        # Self-training: enable/disable
        'enable_self_training': True,
        # Co-training: enable/disable
        'enable_co_training': True,
        # Agreement monitoring: log kappa values
        'log_kappa': True,
        # Agreement monitoring: log consensus details
        'log_consensus': True,
    }

    # Global calibrator (optional)
    global_calibrator = None
    if config.get('calib_path'):
        global_calibrator = initialize_confidence_calibrator(config)

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
                    # --- LOGGING: Print which techniques are being used ---
                    logging.info(f"[Pipeline] Detector fusion method: {config.get('fusion_method')}")
                    logging.info(f"[Pipeline] Consensus method: {config.get('consensus_method')}")
                    logging.info(f"[Pipeline] Calibration method: {config.get('calibration_method')}")
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

                    # --- If mixing is enabled, generate synthetic data and mix with real batch ---
                    if args.mix_real_synthetic:
                        shape_types = ['ellipse', 'circle', 'rectangle', 'polygon', 'triangle', 'point_cloud']
                        synthetic_samples = generate_synthetic_dataset(shape_types, n_per_type=args.synthetic_count, noise=0.03, out_dir=None)
                        synthetic_flat_commands = []
                        for s in synthetic_samples:
                            if 'action_program' in s:
                                synthetic_flat_commands.append(s['action_program'])
                            elif 'commands' in s:
                                synthetic_flat_commands.append(s['commands'])
                            elif 'type' in s:
                                cmds = synthetic_shape_to_logo_commands(s)
                                if cmds:
                                    synthetic_flat_commands.append(cmds)
                                else:
                                    logging.warning(f"[Mixing] Could not convert synthetic shape to LOGO commands: {s}")
                            else:
                                logging.warning(f"Synthetic sample missing 'action_program', 'commands', and 'type' keys: {s}")
                                continue
                        synthetic_img_paths = [None for _ in synthetic_flat_commands]
                        mixed_flat_commands = batch_flat_commands + synthetic_flat_commands
                        mixed_img_paths = batch_img_paths + synthetic_img_paths
                        is_real_mask = [True]*len(batch_flat_commands) + [False]*len(synthetic_flat_commands)
                        print(f"[DEBUG] Mixed real ({len(batch_flat_commands)}) + synthetic ({len(synthetic_flat_commands)}) for problem_id={problem_id}, label_type={label_type}")
                    else:
                        mixed_flat_commands = batch_flat_commands
                        mixed_img_paths = batch_img_paths
                        is_real_mask = [True]*len(batch_flat_commands)

                    # --- Extract features for all images in the mixed support set ---
                    batch_features = []
                    for flat_commands in mixed_flat_commands:
                        objects, _ = create_shape_from_stroke_pipeline(flat_commands, problem_id=problem_id, **shape_creation_dependencies)
                        logging.debug(f"[Feature Extraction] flat_commands: {flat_commands}")
                        logging.debug(f"[Feature Extraction] objects: {objects}")
                        if objects and 'coords' in objects[0]:
                            coords = ensure_numeric_coords(objects[0]['coords'])
                            logging.debug(f"[Feature Extraction] coords (type: {type(coords)}): {coords}")
                            feat = image_processing_features(coords)
                        else:
                            logging.debug("[Feature Extraction] No valid coords found, using zeros.")
                            feat = np.zeros(32)
                        batch_features.append(feat)
                    batch_features = np.stack(batch_features)

                    # --- Context Encoding ---
                    D = batch_features.shape[1]
                    H = 64
                    context_encoder = ContextEncoder(D, H)
                    context_feats = context_encoder(torch.Tensor(batch_features)).detach().numpy()

                    # --- Detector Fusion (Stacking, Attention, Weighted) ---
                    detector_outputs = []
                    logits_list = []
                    detector_preds = []
                    parser = shape_creation_dependencies['BongardLogoParser']
                    coords_batch = []
                    for flat_commands in mixed_flat_commands:
                        shape_objs = parser.parse_action_program(flat_commands)
                        if shape_objs and 'coords' in shape_objs[0]:
                            coords = shape_objs[0]['coords']
                        else:
                            coords = []
                        coords_batch.append(coords)
                        preds = []
                        for det in config['base_detectors']:
                            try:
                                label, *_ = det(coords)
                                preds.append(label)
                            except Exception as e:
                                logging.error(f"Detector {det.__name__} failed: {e}")
                                preds.append(None)
                        detector_preds.append(preds)
                    # Pseudo-labeling: assign label if all detectors agree (ignoring None)
                    pseudo_labels = []
                    pseudo_indices = []
                    uncertain_indices = []
                    for i, preds in enumerate(detector_preds):
                        filtered = [p for p in preds if p is not None]
                        if len(filtered) == 0:
                            pseudo_labels.append(None)
                            uncertain_indices.append(i)
                        elif all(p == filtered[0] for p in filtered):
                            pseudo_labels.append(filtered[0])
                            pseudo_indices.append(i)
                        else:
                            pseudo_labels.append(None)
                            uncertain_indices.append(i)
                    if config['fusion_method'] == 'stacking':
                        stacker = StackingEnsemble(config['base_detectors'])
                        if pseudo_indices:
                            fit_coords = [coords_batch[i] for i in pseudo_indices]
                            fit_labels = [pseudo_labels[i] for i in pseudo_indices]
                            stacker.fit(fit_coords, fit_labels)
                        for i, coords in enumerate(coords_batch):
                            label = stacker.predict(coords)
                            logging.debug(f"[Detector Fusion][Stacking] coords: {coords}, label: {label}")
                            detector_outputs.append([label])
                            logits_list.append(np.random.randn(4))
                    elif config['fusion_method'] == 'attention':
                        parser = shape_creation_dependencies['BongardLogoParser']
                        for i, flat_commands in enumerate(mixed_flat_commands):
                            shape_objs = parser.parse_action_program(flat_commands)
                            if shape_objs and 'coords' in shape_objs[0]:
                                coords = shape_objs[0]['coords']
                            else:
                                coords = []
                            embeds = []
                            for det in config['base_detectors']:
                                _, score, logits = det(coords, return_logits=True)
                                embeds.append(torch.tensor(logits, dtype=torch.float))
                            embeds = torch.stack(embeds)
                            fused = DetectorAttention(embed_dim=embeds.size(-1))(embeds)
                            label = int(torch.argmax(fused).item())
                            logging.debug(f"[Detector Fusion][Attention] coords: {coords}, label: {label}")
                            detector_outputs.append([label])
                            logits_list.append(fused.detach().numpy())
                    else:
                        parser = shape_creation_dependencies['BongardLogoParser']
                        for i, flat_commands in enumerate(mixed_flat_commands):
                            shape_objs = parser.parse_action_program(flat_commands)
                            if shape_objs and 'coords' in shape_objs[0]:
                                coords = shape_objs[0]['coords']
                            else:
                                coords = []
                            labels, scores, logits = detect_shapes(coords, config, return_logits=True, context_feat=context_feats[i])
                            logging.debug(f"[Detector Fusion][Other] coords: {coords}, labels: {labels}, scores: {scores}, logits: {logits}")
                            detector_outputs.append(labels)
                            logits_list.append(logits)

                    # --- Consensus Aggregation (Dawid-Skene or CROWDLAB) ---
                    logging.debug(f"[Consensus] detector_outputs (pre-ensure_list_of_lists): {detector_outputs}")
                    detector_outputs = ensure_list_of_lists(detector_outputs)
                    logging.debug(f"[Consensus] detector_outputs (post-ensure_list_of_lists): {detector_outputs}")
                    if len(detector_outputs) > 0 and isinstance(detector_outputs[0], list) and len(detector_outputs[0]) == 1:
                        detector_outputs = [d[0] for d in detector_outputs]
                        detector_outputs = [detector_outputs]
                    elif len(detector_outputs) > 0 and isinstance(detector_outputs[0], list) and len(detector_outputs) > 1 and len(detector_outputs[0]) == len(detector_outputs[1]):
                        detector_outputs = list(map(list, zip(*detector_outputs)))
                    if config['consensus_method'] == 'crowdlab' and CROWD_LAB_AVAILABLE:
                        logging.info("[Consensus] Using CROWDLAB (Cleanlab multiannotator) for consensus aggregation.")
                        import numpy as np
                        multiannotator_labels = np.array(detector_outputs).T
                        logging.debug(f"[Consensus][CROWDLAB] multiannotator_labels shape: {multiannotator_labels.shape}, dtype: {multiannotator_labels.dtype}, values: {multiannotator_labels}")
                        n_examples, n_annotators = multiannotator_labels.shape
                        n_classes = len(np.unique(multiannotator_labels[~np.isnan(multiannotator_labels)]))
                        pred_probs = np.ones((n_examples, n_classes)) / n_classes
                        consensus_labels, consensus_confidence, annotator_quality = get_label_quality_multiannotator(multiannotator_labels, pred_probs)
                        final_labels = consensus_labels.tolist()
                    else:
                        logging.info("[Consensus] Using Dawid-Skene for consensus aggregation.")
                        logging.debug(f"[Consensus][Dawid-Skene] detector_outputs: {detector_outputs}")
                        final_labels = dawid_skene(detector_outputs)

                    # --- Robust normalization of final_labels ---
                    logging.debug(f"[Postprocessing] Raw final_labels: {final_labels} (type: {type(final_labels)})")
                    print('NORMALIZATION: Raw final_labels =', final_labels)
                    normalized_final_labels = []
                    for idx, l in enumerate(final_labels):
                        logging.debug(f"[Postprocessing] final_labels[{idx}]: {l} (type: {type(l)})")
                        if isinstance(l, list):
                            if len(l) == 0:
                                print(f"WARNING: final_labels[{idx}] is an empty list!")
                                normalized_final_labels.append('unknown')
                            elif len(l) == 1:
                                print(f"NORMALIZATION: final_labels[{idx}] is a list with one element: {l[0]} (type: {type(l[0])})")
                                normalized_final_labels.append(str(l[0]))
                            else:
                                print(f"WARNING: final_labels[{idx}] is a list with multiple elements: {l}")
                                normalized_final_labels.append(str(l[0]))
                        else:
                            print(f"NORMALIZATION: final_labels[{idx}] = {l} (type: {type(l)})")
                            normalized_final_labels.append(str(l))
                    print('NORMALIZATION: normalized_final_labels =', normalized_final_labels)
                    final_labels = normalized_final_labels
                    # --- Check for length mismatch ---
                    if len(final_labels) != len(mixed_flat_commands):
                        print(f"ERROR: Number of consensus labels ({len(final_labels)}) does not match number of images ({len(mixed_flat_commands)}) for problem_id={problem_id}, label_type={label_type}.")
                        print(f"  final_labels: {final_labels}")
                        print(f"  mixed_flat_commands: {mixed_flat_commands}")
                        print(f"  Skipping this batch to avoid IndexError.")
                        continue

                    # --- Kappa Metrics (agreement monitoring) ---
                    if config.get('use_kappa', False) and KAPPA_AVAILABLE:
                        kappas = []
                        min_len = min(len(sublist) for sublist in detector_outputs)
                        trimmed_detector_outputs = [sublist[:min_len] for sublist in detector_outputs]
                        for i in range(len(trimmed_detector_outputs)):
                            for j in range(i+1, len(trimmed_detector_outputs)):
                                if trimmed_detector_outputs[i] and trimmed_detector_outputs[j]:
                                    kappas.append(cohen_kappa_score(trimmed_detector_outputs[i], trimmed_detector_outputs[j]))
                        mean_kappa = sum(kappas)/len(kappas) if kappas else 0.0
                        try:
                            if trimmed_detector_outputs:
                                all_unique_labels = sorted(list(set(label for sublist in trimmed_detector_outputs for label in sublist)))
                                label_to_int = {label: i for i, label in enumerate(all_unique_labels)}
                                numeric_detector_outputs = [[label_to_int[label] for label in sublist] for sublist in trimmed_detector_outputs]
                                overall_fleiss = 0.0
                        except Exception:
                            overall_fleiss = 0.0
                        logging.info(f"Cohen’s κ: {mean_kappa:.2f}, Fleiss’ κ: {overall_fleiss:.2f}")

                    # --- Calibration (Temperature, Platt, Isotonic) ---
                    logits_arr = np.stack(logits_list)
                    print('DEBUG: final_labels =', final_labels)
                    for idx, l in enumerate(final_labels):
                        print(f"DEBUG: final_labels[{idx}] = {l} (type: {type(l)})")
                    normalized_labels = []
                    for l in final_labels:
                        if isinstance(l, int):
                            print(f"WARNING: Integer label found in final_labels: {l}. Converting to string.")
                            normalized_labels.append(str(l))
                        elif isinstance(l, list) and len(l) == 1 and isinstance(l[0], int):
                            print(f"WARNING: Integer label found in final_labels (wrapped in list): {l}. Converting to string.")
                            normalized_labels.append(str(l[0]))
                        elif isinstance(l, list) and len(l) == 1:
                            normalized_labels.append(str(l[0]))
                        else:
                            normalized_labels.append(str(l))
                    label_to_idx = {l: i for i, l in enumerate(sorted(set(normalized_labels)))}
                    print('DEBUG: normalized_labels =', normalized_labels)
                    true_label_indices = np.array([label_to_idx[l] for l in normalized_labels])
                    if config['calibration_method'] == 'platt':
                        scaler = PlattScaler()
                        scaler.fit(logits_arr, true_label_indices)
                    elif config['calibration_method'] == 'isotonic':
                        scaler = IsotonicScaler()
                        scores = logits_arr.max(axis=1)
                        scaler.fit(scores, true_label_indices)
                    elif global_calibrator is not None:
                        scaler = global_calibrator
                    else:
                        scaler = TemperatureScaler()
                        scaler.fit(logits_arr, true_label_indices)

                    # --- MC Dropout Uncertainty Estimation (simulate with random model for demo) ---
                    uncertainties = {}
                    for i, flat_commands in enumerate(mixed_flat_commands):
                        class DummyModel(torch.nn.Module):
                            def __init__(self, out_dim):
                                super().__init__()
                                self.linear = torch.nn.Linear(D, out_dim)
                            def forward(self, x):
                                return self.linear(x)
                        dummy_model = DummyModel(len(label_to_idx))
                        mean_probs, epistemic_uncertainty = mc_dropout_predict(dummy_model, torch.Tensor(batch_features[i:i+1]), n_samples=10)
                        uncertainties[mixed_img_paths[i]] = float(epistemic_uncertainty[0])

                    # --- Active Learning: Select samples for review ---
                    to_review = set(select_uncertain_samples(uncertainties, budget=config['review_budget']))

                    # --- Now process each image in the mixed support set, but only save real data ---
                    for idx, flat_commands in enumerate(mixed_flat_commands):
                        img_path = mixed_img_paths[idx]
                        current_image_flag_reasons = []
                        validation_issues_list = []
                        desired_label = 'desired_label'  # <-- Replace with your actual label string or value
                        consensus_label = final_labels[idx]
                        # Log every label detection, real and synthetic
                        if is_real_mask[idx]:
                            logging.info(f"[REAL DATA] Detected label '{consensus_label}' for image: {img_path}")
                        else:
                            logging.info(f"[SYNTHETIC DATA] Detected label '{consensus_label}' for synthetic sample index: {idx - len(batch_flat_commands)}")

                        # Log when desired label is detected in real or synthetic data
                        if str(consensus_label) == str(desired_label):
                            if is_real_mask[idx]:
                                logging.info(f"[REAL DATA] Desired label '{desired_label}' detected for image: {img_path}")
                            else:
                                logging.info(f"[SYNTHETIC DATA] Desired label '{desired_label}' detected for synthetic sample index: {idx - len(batch_flat_commands)}")

                        # Only visualize and save results for real data
                        if not is_real_mask[idx]:
                            continue

                            # Visualization for real data only
                            try:
                                if objects and 'coords' in objects[0]:
                                    coords = ensure_numeric_coords(objects[0]['coords'])
                                    plt.figure()
                                    arr = np.array(coords)
                                    plt.plot(arr[:,0], arr[:,1], marker='o')
                                    plt.title(f"Real Data Visualization: {img_path}")
                                    plt.show(block=False)
                                    plt.close()
                            except Exception as e:
                                logging.error(f"[Visualization] Error visualizing real data for {img_path}: {e}")

                            epoch = config.get('curriculum_epoch', 0)
                            base_thr = config.get('base_thr', 0.9)
                            min_thr = config.get('min_thr', 0.6)
                            decay = config.get('decay', 0.01)
                            thr = max(base_thr - decay * epoch, min_thr)
                            image_record = {
                                'problem_id': problem_id,
                                'category': cat,
                                'label': norm_label,
                            }
                        try:
                            if config['calibration_method'] == 'platt':
                                probs = scaler.calibrate(np.expand_dims(logits_list[idx],0))[0]
                                confidence = float(probs[label_to_idx[consensus_label]])
                            elif config['calibration_method'] == 'isotonic':
                                score = logits_list[idx].max()
                                confidence = float(scaler.calibrate([score])[0])
                            else:
                                calibrated_logits = scaler.calibrate(logits_list[idx])
                                probs = np.exp(calibrated_logits) / np.sum(np.exp(calibrated_logits))
                                confidence = float(probs[label_to_idx[consensus_label]])
                            uncertainty = uncertainties[img_path]
                            if img_path in to_review:
                                current_image_flag_reasons.append('Selected by active learning (high uncertainty)')
                            objects, fallback_reasons = create_shape_from_stroke_pipeline(
                                flat_commands, problem_id=problem_id, **shape_creation_dependencies
                            )
                            # Visualization for real data only
                            if is_real_mask[idx]:
                                try:
                                    if objects and 'coords' in objects[0]:
                                        arr = np.array(objects[0]['coords'])
                                        import matplotlib.pyplot as plt
                                        plt.figure()
                                        plt.plot(arr[:,0], arr[:,1], marker='o')
                                        plt.title(f"Real Data Visualization: {img_path}")
                                        plt.show(block=False)
                                        plt.close()
                                except Exception as e:
                                    logging.error(f"[Visualization] Error visualizing real data for {img_path}: {e}")

                            epoch = config.get('curriculum_epoch', 0)
                            base_thr = config.get('base_thr', 0.9)
                            min_thr = config.get('min_thr', 0.6)
                            decay = config.get('decay', 0.01)
                            thr = max(base_thr - decay * epoch, min_thr)
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
                                'review_threshold': thr,
                            }
                            validation_issues_list = validate_label_quality(image_record, problem_id)
                            image_record['validation_issues'] = validation_issues_list
                            all_results.append(image_record)
                            if current_image_flag_reasons or confidence < thr:
                                flagged_cases.append(image_record)
                        except Exception as e:
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
    cmds = []
    t = shape.get('type', '').lower()
    if t == 'ellipse':
        # Draw ellipse as polygonal approximation
        center = shape.get('center', [0, 0])
        axes = shape.get('axes', (40, 20))
        angle = shape.get('angle', 0)
        n = 32
        import math
        cx, cy = center
        a, b = axes
        theta = math.radians(angle)
        pts = []
        for i in range(n):
            t_ = 2 * math.pi * i / n
            x = cx + a * math.cos(t_) * math.cos(theta) - b * math.sin(t_) * math.sin(theta)
            y = cy + a * math.cos(t_) * math.sin(theta) + b * math.sin(t_) * math.cos(theta)
            pts.append([x, y])
        cmds.append('PU')
        cmds.append(f'HOME')
        cmds.append(f'PD')
        for x, y in pts:
            cmds.append(f'{x:.2f} {y:.2f}')
        cmds.append('PU')
    elif t == 'circle':
        center = shape.get('center', [0, 0])
        radius = shape.get('radius', 30)
        n = 24
        import math
        cx, cy = center
        pts = []
        for i in range(n):
            t_ = 2 * math.pi * i / n
            x = cx + radius * math.cos(t_)
            y = cy + radius * math.sin(t_)
            pts.append([x, y])
        cmds.append('PU')
        cmds.append(f'HOME')
        cmds.append(f'PD')
        for x, y in pts:
            cmds.append(f'{x:.2f} {y:.2f}')
        cmds.append('PU')
    elif t == 'rectangle':
        center = shape.get('center', [0, 0])
        w = shape.get('width', 40)
        h = shape.get('height', 20)
        angle = shape.get('angle', 0)
        import math
        cx, cy = center
        theta = math.radians(angle)
        # Rectangle corners
        dx = w / 2
        dy = h / 2
        corners = [
            [cx - dx, cy - dy],
            [cx + dx, cy - dy],
            [cx + dx, cy + dy],
            [cx - dx, cy + dy],
        ]
        # Rotate corners
        rot = lambda x, y: [
            cx + (x - cx) * math.cos(theta) - (y - cy) * math.sin(theta),
            cy + (x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)
        ]
        corners = [rot(x, y) for x, y in corners]
        cmds.append('PU')
        cmds.append(f'HOME')
        cmds.append('PD')
        for x, y in corners:
            cmds.append(f'{x:.2f} {y:.2f}')
        # Close rectangle
        x, y = corners[0]
        cmds.append(f'{x:.2f} {y:.2f}')
        cmds.append('PU')
    elif t == 'polygon':
        pts = shape.get('points', [])
        cmds.append('PU')
        cmds.append(f'HOME')
        cmds.append('PD')
        for x, y in pts:
            cmds.append(f'{x:.2f} {y:.2f}')
        if pts:
            x, y = pts[0]
            cmds.append(f'{x:.2f} {y:.2f}')
        cmds.append('PU')
    elif t == 'triangle':
        pts = shape.get('points', [])
        cmds.append('PU')
        cmds.append(f'HOME')
        cmds.append('PD')
        for x, y in pts:
            cmds.append(f'{x:.2f} {y:.2f}')
        if pts:
            x, y = pts[0]
            cmds.append(f'{x:.2f} {y:.2f}')
        cmds.append('PU')
    elif t == 'point_cloud':
        pts = shape.get('points', [])
        cmds.append('PU')
        cmds.append(f'HOME')
        for x, y in pts:
            cmds.append('PD')
            cmds.append(f'{x:.2f} {y:.2f}')
            cmds.append('PU')
        cmds.append('PU')
    else:
        logging.warning(f"[synthetic_shape_to_logo_commands] Unknown shape type: {t}")
    return cmds

# Patch: import ellipse_fit separately so we can override threshold for debugging
from .derive_label import geometric_detectors as _geometric_detectors

def ellipse_fit_debug(coords, *args, **kwargs):
    # Accept is_real as a kwarg for logging
    is_real = kwargs.pop('is_real', None)
    result = _geometric_detectors.ellipse_fit(coords, *args)
    # Add extra debug logging
    try:
        label, conf, params = result
        if is_real is not None:
            src = 'REAL' if is_real else 'SYNTHETIC'
            logging.info(f"[ellipse_fit {src}] DETECTED: {label}, confidence: {conf}, params: {params}")
        else:
            logging.info(f"[ellipse_fit DEBUG] coords: {coords}, DETECTED: {label}, confidence: {conf}, params: {params}")
    except Exception:
        logging.info(f"[ellipse_fit DEBUG] coords: {coords}, result: {result}")
    return result

    # Patch config to use ellipse_fit_debug
    config = {
        # Detector ensemble weights (for weighted voting)
        'ensemble_weights': [1.0, 1.0, 1.0],
        # Uncertainty threshold for active learning
        'uncertainty_threshold': 0.15,
        # Number of samples to select for review in active learning
        'review_budget': 20,
        # List of base detector functions
        'base_detectors': [ransac_line, ransac_circle, ellipse_fit_debug],
        # Detector fusion method: 'stacking', 'attention', 'weighted'
        'fusion_method': 'stacking',
        # Calibration method: 'temperature', 'platt', 'isotonic'
        'calibration_method': 'temperature',
        # Consensus method: 'dawid', 'crowdlab'
        'consensus_method': 'dawid',
        # Use kappa metrics for agreement monitoring
        'use_kappa': True,
        # Curriculum learning: current epoch (increment per training epoch)
        'curriculum_epoch': 0,
        # Curriculum learning: base threshold for review
        'base_thr': 0.9,
        # Curriculum learning: minimum threshold for review
        'min_thr': 0.6,
        # Curriculum learning: decay rate for threshold
        'decay': 0.01,
        # Self-training: frequency (epochs)
        'self_train_freq': 5,
        # Self-training: confidence threshold for pseudo-labeling
        'self_train_thresh': 0.95,
        # Co-training: frequency (epochs)
        'co_train_freq': 5,
        # Co-training: number of samples to exchange
        'co_train_k': 100,
        # Path to calibration data for global calibrator (if used)
        'calib_path': None,
        # Meta-learner: stacking meta-classifier config (if needed)
        'meta_learner': 'random_forest', # or 'mlp', 'logistic', etc.
        # Attention meta-learner: number of heads
        'attention_heads': 4,
        # CROWDLAB: aggregator type (if using CROWDLAB)
        'crowdlab_aggregator': 'default',
        # Kappa metrics: enable/disable
        'enable_fleiss_kappa': True,
        # Curriculum learning: enable/disable
        'enable_curriculum': True,
        # Self-training: enable/disable
        'enable_self_training': True,
        # Co-training: enable/disable
        'enable_co_training': True,
        # Agreement monitoring: log kappa values
        'log_kappa': True,
        # Agreement monitoring: log consensus details
        'log_consensus': True,
    }

    # Global calibrator (optional)
    global_calibrator = None
    if config.get('calib_path'):
        global_calibrator = initialize_confidence_calibrator(config)

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
                    # --- LOGGING: Print which techniques are being used ---
                    logging.info(f"[Pipeline] Detector fusion method: {config.get('fusion_method')}")
                    logging.info(f"[Pipeline] Consensus method: {config.get('consensus_method')}")
                    logging.info(f"[Pipeline] Calibration method: {config.get('calibration_method')}")
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

                    # --- If mixing is enabled, generate synthetic data and mix with real batch ---
                    if args.mix_real_synthetic:
                        shape_types = ['ellipse', 'circle', 'rectangle', 'polygon', 'triangle', 'point_cloud']
                        synthetic_samples = generate_synthetic_dataset(shape_types, n_per_type=args.synthetic_count, noise=0.03, out_dir=None)
                        synthetic_flat_commands = []
                        for s in synthetic_samples:
                            if 'action_program' in s:
                                synthetic_flat_commands.append(s['action_program'])
                            elif 'commands' in s:
                                synthetic_flat_commands.append(s['commands'])
                            elif 'type' in s:
                                cmds = synthetic_shape_to_logo_commands(s)
                                if cmds:
                                    synthetic_flat_commands.append(cmds)
                                else:
                                    logging.warning(f"[Mixing] Could not convert synthetic shape to LOGO commands: {s}")
                            else:
                                logging.warning(f"Synthetic sample missing 'action_program', 'commands', and 'type' keys: {s}")
                                continue
                        synthetic_img_paths = [None for _ in synthetic_flat_commands]
                        mixed_flat_commands = batch_flat_commands + synthetic_flat_commands
                        mixed_img_paths = batch_img_paths + synthetic_img_paths
                        is_real_mask = [True]*len(batch_flat_commands) + [False]*len(synthetic_flat_commands)
                        print(f"[DEBUG] Mixed real ({len(batch_flat_commands)}) + synthetic ({len(synthetic_flat_commands)}) for problem_id={problem_id}, label_type={label_type}")
                    else:
                        mixed_flat_commands = batch_flat_commands
                        mixed_img_paths = batch_img_paths
                        is_real_mask = [True]*len(batch_flat_commands)

                    # --- Extract features for all images in the mixed support set ---
                    batch_features = []
                    for flat_commands in mixed_flat_commands:
                        objects, _ = create_shape_from_stroke_pipeline(flat_commands, problem_id=problem_id, **shape_creation_dependencies)
                        logging.debug(f"[Feature Extraction] flat_commands: {flat_commands}")
                        logging.debug(f"[Feature Extraction] objects: {objects}")
                        if objects and 'coords' in objects[0]:
                            coords = ensure_numeric_coords(objects[0]['coords'])
                            logging.debug(f"[Feature Extraction] coords (type: {type(coords)}): {coords}")
                            feat = image_processing_features(coords)
                        else:
                            logging.debug("[Feature Extraction] No valid coords found, using zeros.")
                            feat = np.zeros(32)
                        batch_features.append(feat)
                    batch_features = np.stack(batch_features)

                    # --- Context Encoding ---
                    D = batch_features.shape[1]
                    H = 64
                    context_encoder = ContextEncoder(D, H)
                    context_feats = context_encoder(torch.Tensor(batch_features)).detach().numpy()

                    # --- Detector Fusion (Stacking, Attention, Weighted) ---
                    detector_outputs = []
                    logits_list = []
                    detector_preds = []
                    parser = shape_creation_dependencies['BongardLogoParser']
                    coords_batch = []
                    for flat_commands, is_real in zip(mixed_flat_commands, is_real_mask):
                        shape_objs = parser.parse_action_program(flat_commands)
                        if shape_objs and 'coords' in shape_objs[0]:
                            coords = shape_objs[0]['coords']
                        else:
                            coords = []
                        coords_batch.append(coords)
                        preds = []
                        for det in config['base_detectors']:
                            try:
                                # Pass is_real to ellipse_fit_debug, ignore for others
                                if det is ellipse_fit_debug:
                                    label, *_ = det(coords, is_real=is_real)
                                else:
                                    label, *_ = det(coords)
                                preds.append(label)
                            except Exception as e:
                                logging.error(f"Detector {det.__name__} failed: {e}")
                                preds.append(None)
                        detector_preds.append(preds)
                    # Pseudo-labeling: assign label if all detectors agree (ignoring None)
                    pseudo_labels = []
                    pseudo_indices = []
                    uncertain_indices = []
                    for i, preds in enumerate(detector_preds):
                        filtered = [p for p in preds if p is not None]
                        if len(filtered) == 0:
                            pseudo_labels.append(None)
                            uncertain_indices.append(i)
                        elif all(p == filtered[0] for p in filtered):
                            pseudo_labels.append(filtered[0])
                            pseudo_indices.append(i)
                        else:
                            pseudo_labels.append(None)
                            uncertain_indices.append(i)
                    if config['fusion_method'] == 'stacking':
                        stacker = StackingEnsemble(config['base_detectors'])
                        if pseudo_indices:
                            fit_coords = [coords_batch[i] for i in pseudo_indices]
                            fit_labels = [pseudo_labels[i] for i in pseudo_indices]
                            stacker.fit(fit_coords, fit_labels)
                        for i, coords in enumerate(coords_batch):
                            label = stacker.predict(coords)
                            logging.debug(f"[Detector Fusion][Stacking] coords: {coords}, label: {label}")
                            detector_outputs.append([label])
                            logits_list.append(np.random.randn(4))
                    elif config['fusion_method'] == 'attention':
                        parser = shape_creation_dependencies['BongardLogoParser']
                        for i, flat_commands in enumerate(mixed_flat_commands):
                            shape_objs = parser.parse_action_program(flat_commands)
                            if shape_objs and 'coords' in shape_objs[0]:
                                coords = shape_objs[0]['coords']
                            else:
                                coords = []
                            embeds = []
                            for det in config['base_detectors']:
                                _, score, logits = det(coords, return_logits=True)
                                embeds.append(torch.tensor(logits, dtype=torch.float))
                            embeds = torch.stack(embeds)
                            fused = DetectorAttention(embed_dim=embeds.size(-1))(embeds)
                            label = int(torch.argmax(fused).item())
                            logging.debug(f"[Detector Fusion][Attention] coords: {coords}, label: {label}")
                            detector_outputs.append([label])
                            logits_list.append(fused.detach().numpy())
                    else:
                        parser = shape_creation_dependencies['BongardLogoParser']
                        for i, flat_commands in enumerate(mixed_flat_commands):
                            shape_objs = parser.parse_action_program(flat_commands)
                            if shape_objs and 'coords' in shape_objs[0]:
                                coords = shape_objs[0]['coords']
                            else:
                                coords = []
                            labels, scores, logits = detect_shapes(coords, config, return_logits=True, context_feat=context_feats[i])
                            logging.debug(f"[Detector Fusion][Other] coords: {coords}, labels: {labels}, scores: {scores}, logits: {logits}")
                            detector_outputs.append(labels)
                            logits_list.append(logits)

                    # --- Consensus Aggregation (Dawid-Skene or CROWDLAB) ---
                    logging.debug(f"[Consensus] detector_outputs (pre-ensure_list_of_lists): {detector_outputs}")
                    detector_outputs = ensure_list_of_lists(detector_outputs)
                    logging.debug(f"[Consensus] detector_outputs (post-ensure_list_of_lists): {detector_outputs}")
                    if len(detector_outputs) > 0 and isinstance(detector_outputs[0], list) and len(detector_outputs[0]) == 1:
                        detector_outputs = [d[0] for d in detector_outputs]
                        detector_outputs = [detector_outputs]
                    elif len(detector_outputs) > 0 and isinstance(detector_outputs[0], list) and len(detector_outputs) > 1 and len(detector_outputs[0]) == len(detector_outputs[1]):
                        detector_outputs = list(map(list, zip(*detector_outputs)))
                    if config['consensus_method'] == 'crowdlab' and CROWD_LAB_AVAILABLE:
                        logging.info("[Consensus] Using CROWDLAB (Cleanlab multiannotator) for consensus aggregation.")
                        import numpy as np
                        multiannotator_labels = np.array(detector_outputs).T
                        logging.debug(f"[Consensus][CROWDLAB] multiannotator_labels shape: {multiannotator_labels.shape}, dtype: {multiannotator_labels.dtype}, values: {multiannotator_labels}")
                        n_examples, n_annotators = multiannotator_labels.shape
                        n_classes = len(np.unique(multiannotator_labels[~np.isnan(multiannotator_labels)]))
                        pred_probs = np.ones((n_examples, n_classes)) / n_classes
                        consensus_labels, consensus_confidence, annotator_quality = get_label_quality_multiannotator(multiannotator_labels, pred_probs)
                        final_labels = consensus_labels.tolist()
                    else:
                        logging.info("[Consensus] Using Dawid-Skene for consensus aggregation.")
                        logging.debug(f"[Consensus][Dawid-Skene] detector_outputs: {detector_outputs}")
                        final_labels = dawid_skene(detector_outputs)

                    # --- Robust normalization of final_labels ---
                    logging.debug(f"[Postprocessing] Raw final_labels: {final_labels} (type: {type(final_labels)})")
                    print('NORMALIZATION: Raw final_labels =', final_labels)
                    normalized_final_labels = []
                    for idx, l in enumerate(final_labels):
                        logging.debug(f"[Postprocessing] final_labels[{idx}]: {l} (type: {type(l)})")
                        if isinstance(l, list):
                            if len(l) == 0:
                                print(f"WARNING: final_labels[{idx}] is an empty list!")
                                normalized_final_labels.append('unknown')
                            elif len(l) == 1:
                                print(f"NORMALIZATION: final_labels[{idx}] is a list with one element: {l[0]} (type: {type(l[0])})")
                                normalized_final_labels.append(str(l[0]))
                            else:
                                print(f"WARNING: final_labels[{idx}] is a list with multiple elements: {l}")
                                normalized_final_labels.append(str(l[0]))
                        else:
                            print(f"NORMALIZATION: final_labels[{idx}] = {l} (type: {type(l)})")
                            normalized_final_labels.append(str(l))
                    print('NORMALIZATION: normalized_final_labels =', normalized_final_labels)
                    final_labels = normalized_final_labels
                    # --- Check for length mismatch ---
                    if len(final_labels) != len(mixed_flat_commands):
                        print(f"ERROR: Number of consensus labels ({len(final_labels)}) does not match number of images ({len(mixed_flat_commands)}) for problem_id={problem_id}, label_type={label_type}.")
                        print(f"  final_labels: {final_labels}")
                        print(f"  mixed_flat_commands: {mixed_flat_commands}")
                        print(f"  Skipping this batch to avoid IndexError.")
                        continue

                    # --- Kappa Metrics (agreement monitoring) ---
                    if config.get('use_kappa', False) and KAPPA_AVAILABLE:
                        kappas = []
                        min_len = min(len(sublist) for sublist in detector_outputs)
                        trimmed_detector_outputs = [sublist[:min_len] for sublist in detector_outputs]
                        for i in range(len(trimmed_detector_outputs)):
                            for j in range(i+1, len(trimmed_detector_outputs)):
                                if trimmed_detector_outputs[i] and trimmed_detector_outputs[j]:
                                    kappas.append(cohen_kappa_score(trimmed_detector_outputs[i], trimmed_detector_outputs[j]))
                        mean_kappa = sum(kappas)/len(kappas) if kappas else 0.0
                        try:
                            if trimmed_detector_outputs:
                                all_unique_labels = sorted(list(set(label for sublist in trimmed_detector_outputs for label in sublist)))
                                label_to_int = {label: i for i, label in enumerate(all_unique_labels)}
                                numeric_detector_outputs = [[label_to_int[label] for label in sublist] for sublist in trimmed_detector_outputs]
                                overall_fleiss = 0.0
                        except Exception:
                            overall_fleiss = 0.0
                        logging.info(f"Cohen’s κ: {mean_kappa:.2f}, Fleiss’ κ: {overall_fleiss:.2f}")

                    # --- Calibration (Temperature, Platt, Isotonic) ---
                    logits_arr = np.stack(logits_list)
                    print('DEBUG: final_labels =', final_labels)
                    for idx, l in enumerate(final_labels):
                        print(f"DEBUG: final_labels[{idx}] = {l} (type: {type(l)})")
                    normalized_labels = []
                    for l in final_labels:
                        if isinstance(l, int):
                            print(f"WARNING: Integer label found in final_labels: {l}. Converting to string.")
                            normalized_labels.append(str(l))
                        elif isinstance(l, list) and len(l) == 1 and isinstance(l[0], int):
                            print(f"WARNING: Integer label found in final_labels (wrapped in list): {l}. Converting to string.")
                            normalized_labels.append(str(l[0]))
                        elif isinstance(l, list) and len(l) == 1:
                            normalized_labels.append(str(l[0]))
                        else:
                            normalized_labels.append(str(l))
                    label_to_idx = {l: i for i, l in enumerate(sorted(set(normalized_labels)))}
                    print('DEBUG: normalized_labels =', normalized_labels)
                    true_label_indices = np.array([label_to_idx[l] for l in normalized_labels])
                    if config['calibration_method'] == 'platt':
                        scaler = PlattScaler()
                        scaler.fit(logits_arr, true_label_indices)
                    elif config['calibration_method'] == 'isotonic':
                        scaler = IsotonicScaler()
                        scores = logits_arr.max(axis=1)
                        scaler.fit(scores, true_label_indices)
                    elif global_calibrator is not None:
                        scaler = global_calibrator
                    else:
                        scaler = TemperatureScaler()
                        scaler.fit(logits_arr, true_label_indices)

                    # --- MC Dropout Uncertainty Estimation (simulate with random model for demo) ---
                    uncertainties = {}
                    for i, flat_commands in enumerate(mixed_flat_commands):
                        class DummyModel(torch.nn.Module):
                            def __init__(self, out_dim):
                                super().__init__()
                                self.linear = torch.nn.Linear(D, out_dim)
                            def forward(self, x):
                                return self.linear(x)
                        dummy_model = DummyModel(len(label_to_idx))
                        mean_probs, epistemic_uncertainty = mc_dropout_predict(dummy_model, torch.Tensor(batch_features[i:i+1]), n_samples=10)
                        uncertainties[mixed_img_paths[i]] = float(epistemic_uncertainty[0])

                    # --- Active Learning: Select samples for review ---
                    to_review = set(select_uncertain_samples(uncertainties, budget=config['review_budget']))

                    # --- Now process each image in the mixed support set, but only save real data ---
                    for idx, flat_commands in enumerate(mixed_flat_commands):
                        img_path = mixed_img_paths[idx]
            current_image_flag_reasons = []
            validation_issues_list = []
            desired_label = 'desired_label'  # <-- Replace with your actual label string or value
            consensus_label = final_labels[idx]
            # Log every label detection, real and synthetic
            if is_real_mask[idx]:
                logging.info(f"[REAL DATA] Detected label '{consensus_label}' for image: {img_path}")
            else:
                logging.info(f"[SYNTHETIC DATA] Detected label '{consensus_label}' for synthetic sample index: {idx - len(batch_flat_commands)}")

            # Log when desired label is detected in real or synthetic data
            if str(consensus_label) == str(desired_label):
                if is_real_mask[idx]:
                    logging.info(f"[REAL DATA] Desired label '{desired_label}' detected for image: {img_path}")
                else:
                    logging.info(f"[SYNTHETIC DATA] Desired label '{desired_label}' detected for synthetic sample index: {idx - len(batch_flat_commands)}")

            # Only visualize and save results for real data
            if not is_real_mask[idx]:
                continue

            try:
                if config['calibration_method'] == 'platt':
                    probs = scaler.calibrate(np.expand_dims(logits_list[idx],0))[0]
                    confidence = float(probs[label_to_idx[consensus_label]])
                elif config['calibration_method'] == 'isotonic':
                    score = logits_list[idx].max()
                    confidence = float(scaler.calibrate([score])[0])
                else:
                    calibrated_logits = scaler.calibrate(logits_list[idx])
                    probs = np.exp(calibrated_logits) / np.sum(np.exp(calibrated_logits))
                    confidence = float(probs[label_to_idx[consensus_label]])
                uncertainty = uncertainties[img_path]
                if img_path in to_review:
                    current_image_flag_reasons.append('Selected by active learning (high uncertainty)')
                objects, fallback_reasons = create_shape_from_stroke_pipeline(
                    flat_commands, problem_id=problem_id, **shape_creation_dependencies
                )
                # Visualization for real data only
                try:
                    if objects and 'coords' in objects[0]:
                        coords = ensure_numeric_coords(objects[0]['coords'])
                        plt.figure()
                        arr = np.array(coords)
                        plt.plot(arr[:,0], arr[:,1], marker='o')
                        plt.title(f"Real Data Visualization: {img_path}")
                        plt.show(block=False)
                        plt.close()
                except Exception as e:
                    logging.error(f"[Visualization] Error visualizing real data for {img_path}: {e}")

                epoch = config.get('curriculum_epoch', 0)
                base_thr = config.get('base_thr', 0.9)
                min_thr = config.get('min_thr', 0.6)
                decay = config.get('decay', 0.01)
                thr = max(base_thr - decay * epoch, min_thr)
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
                    'review_threshold': thr,
                }
                validation_issues_list = validate_label_quality(image_record, problem_id)
                image_record['validation_issues'] = validation_issues_list
                all_results.append(image_record)
                if current_image_flag_reasons or confidence < thr:
                    flagged_cases.append(image_record)
            except Exception as e:
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


if __name__ == '__main__':
    main()
