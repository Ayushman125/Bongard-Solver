import logging
from logging.handlers import RotatingFileHandler

# --- LOGGING SETUP ---
LOG_FILENAME = 'logo_to_shape_debug.log'
file_handler = RotatingFileHandler(LOG_FILENAME, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in terminal
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

import importlib.util
import sys
sys.stdout.reconfigure(encoding='utf-8')
def fully_stringify(obj):
    if isinstance(obj, dict):
        return {fully_stringify(k): fully_stringify(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fully_stringify(x) for x in obj]
    elif hasattr(obj, 'raw_command'):
        return str(obj.raw_command)
    elif type(obj).__name__ in ['LineAction', 'ArcAction']:
        return str(getattr(obj, 'raw_command', str(obj)))
    else:
        return str(obj)

# Robust flatten and stringify for any nested actions list
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
def ensure_all_strings(lst):
    """Recursively convert all items in a (possibly nested) list to strings."""
    if isinstance(lst, list):
        return [ensure_all_strings(x) for x in lst]
    if hasattr(lst, 'raw_command'):
        return str(lst.raw_command)
    return str(lst)

def safe_join(lst, sep=','):
    """Join a list into a string, robustly converting all items to strings first using fully_stringify."""
    def fully_stringify(obj):
        if isinstance(obj, dict):
            return {fully_stringify(k): fully_stringify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fully_stringify(x) for x in obj]
        elif hasattr(obj, 'raw_command'):
            return str(obj.raw_command)
        elif type(obj).__name__ in ['LineAction', 'ArcAction']:
            return str(getattr(obj, 'raw_command', str(obj)))
        else:
            return str(obj)
    if isinstance(lst, list):
        safe_items = [fully_stringify(x) for x in lst]
        logger.debug(f"[SAFE_JOIN DEBUG] safe_items: {safe_items}")
        safe_items = [str(x) for x in safe_items]
        return sep.join(safe_items)
    return str(lst)


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
import numpy as np

# Ensure src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.data_pipeline.data_loader import load_action_programs
from src.bongard_augmentor.hybrid import HybridAugmentor
from src.Derive_labels.utils import robust_flatten_and_stringify
from bongard.bongard import BongardImage
from src.physics_inference import PhysicsInference
from src.Derive_labels.image_level import process_single_image
from src.Derive_labels.image_level import _calculate_image_features
from src.Derive_labels.physics_features import _calculate_physics_features
from src.Derive_labels.quality_monitor import quality_monitor
from src.Derive_labels.processing_monitor import processing_monitor
# After each major feature extraction step:
quality_monitor.log_quality('main_pipeline', {'step': 'feature_extraction'})
processing_monitor.log_event('feature_extraction_complete')

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
from src.Derive_labels.stroke_types import extract_action_type_prefixes
from src.Derive_labels.shape_utils import normalize_vertices, calculate_geometry, extract_position_and_rotation, ensure_vertex_list, _calculate_perimeter, _calculate_curvature_score, _calculate_edge_length_variance, json_safe, _calculate_irregularity, calculate_complexity
from src.Derive_labels.stroke_types import _extract_modifier_from_stroke, _calculate_stroke_specific_features, _calculate_stroke_type_differentiated_features
from src.Derive_labels.features import _actions_to_geometries, _extract_ngram_features, _extract_graph_features
from src.Derive_labels.features import _detect_alternation
from src.Derive_labels.shape_utils import _calculate_homogeneity, _calculate_angular_variance,safe_divide, _calculate_compactness, _calculate_eccentricity, _check_horizontal_symmetry, _check_vertical_symmetry, _check_rotational_symmetry
from src.Derive_labels.file_io import FileIO
from src.Derive_labels.features import extract_multiscale_features
from src.Derive_labels.context_features import BongardFeatureExtractor
from src.Derive_labels.spatial_topological_features import compute_spatial_topological_features
from src.Derive_labels.contextual_features import (
    positive_negative_contrast_score,
    support_set_mutual_information,
    label_consistency_ratio,
    concept_drift_score,
    support_set_shape_cooccurrence,
    category_consistency_score,
    class_prototype_distance,
    feature_importance_ranking,
    cross_set_symmetry_difference
)
from src.Derive_labels.compositional_features import (
    _calculate_composition_features,
    hierarchical_clustering_heights,
    composition_tree_depth,
    composition_tree_branching_factor,
    subgraph_isomorphism_frequencies,
    recursive_shape_patterns,
    multi_level_symmetry_chains,
    layered_edge_complexity,
    overlapping_substructure_ratios,
    composition_regularity_score,
    nested_convex_hull_levels
)
from src.Derive_labels.multi_modal_ensemble import aggregate_multi_modal_features




class ComprehensiveBongardProcessor:
    """
    Enhanced comprehensive processor for Bongard-LOGO data that handles:
    - Multi-stroke image composition with stroke-type specific calculations
    - Differentiated geometry analysis for line vs arc strokes
    - Shape-modifier aware feature extraction
    - Comprehensive flagging logic for suspicious entries
    - Physics and geometry computation with validation
    """
    def __init__(self):
        # ...existing code...
        self.context_extractor = BongardFeatureExtractor()

    def _calculate_vertices_from_action(self, action, stroke_index, bongard_image=None):
        """Always use analytic vertices from the action parser for all strokes."""
        try:
            if hasattr(action, 'vertices_from_command'):
                verts = action.vertices_from_command()
                if verts:
                    return verts
            # Fallback to previous extraction if analytic not available
            from src.Derive_labels.stroke_types import _extract_stroke_vertices, _compute_bounding_box
            verts = _extract_stroke_vertices(action, stroke_index, None, bongard_image=bongard_image)
            if verts:
                bbox = _compute_bounding_box(verts)
                logger.info(f"[_calculate_vertices_from_action] Bounding box: {bbox}")
            return verts
        except Exception as e:
            logger.debug(f"Failed to calculate vertices from action: {e}")
        return []


    def _calculate_pattern_regularity_from_modifiers(self, modifier_sequence: list) -> float:
        """
        Pattern regularity using PhysicsInference.pattern_regularity. Returns NaN if sequence too short.
        """
        # Use corrected modifiers (horizontal/vertical/arc) for n-gram and regularity
        return PhysicsInference.pattern_regularity(modifier_sequence)


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

    def _flag_case(self, category, problem_id, message, tags=None):
        """Log and store flagged cases for inspection."""
        logger.warning(f"[FLAG CASE] category={category}, problem_id={problem_id}, message={message}, tags={tags}")
        case = {
            'category': category,
            'problem_id': problem_id,
            'message': message,
            'tags': tags if tags else []
        }
        self.flagged_cases.append(case)

def main():
    def fully_stringify(obj):
        if isinstance(obj, dict):
            return {fully_stringify(k): fully_stringify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fully_stringify(x) for x in obj]
        elif hasattr(obj, 'raw_command'):
            return str(obj.raw_command)
        elif type(obj).__name__ in ['LineAction', 'ArcAction']:
            return str(getattr(obj, 'raw_command', str(obj)))
        else:
            return str(obj)
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
    action_type_prefixes = extract_action_type_prefixes(problems_data)
    logger.info(f"[DEBUG] Unique action type prefixes in dataset: {sorted(action_type_prefixes)}")
    # --- END DEBUG ---

    # Process all images

    all_results = []
    total_images = 0
    successful_images = 0
    problem_summaries = []

    # Defensive validation and logging for problem_data structure
    logger.info("[DEFENSIVE CHECK] Validating problem_data structure for all problems...")
    malformed_problems = []
    for pid, pdata in problems_data.items():
        if not (isinstance(pdata, list) and len(pdata) == 2):
            logger.warning(f"[DEFENSIVE CHECK] Problem {pid} has malformed data: type={type(pdata)}, value={pdata}")
            malformed_problems.append(pid)
    if malformed_problems:
        logger.error(f"[DEFENSIVE CHECK] Found malformed problems: {malformed_problems}")
    else:
        logger.info("[DEFENSIVE CHECK] All problems have valid [positive_examples, negative_examples] structure.")

    for problem_id, problem_data in problems_data.items():
        try:
            # Defensive: check problem_data structure
            if not (isinstance(problem_data, list) and len(problem_data) == 2):
                logger.warning(f"Problem {problem_id} has unexpected data format, skipping.")
                continue

            positive_examples, negative_examples = problem_data
            category = 'bd' if problem_id.startswith('bd_') else (
                'ff' if problem_id.startswith('ff_') else (
                'hd' if problem_id.startswith('hd_') else 'unknown'))

            pos_results, neg_results = [], []
            num_images_in_problem = 0
            problem_unique_shape_functions = set()
            problem_shape_function_counts = {}
            problem_modifiers = set()

            # Process positive examples
            degenerate_count = 0
            for i, action_commands in enumerate(positive_examples):
                total_images += 1
                num_images_in_problem += 1
                image_id = f"{problem_id}_pos_{i}"
                image_path = f"images/{problem_id}/category_1/{i}.png"
                result = process_single_image(
                    action_commands, image_id, True, problem_id, category, image_path,
                    processing_stats=processor.processing_stats,
                    flag_case=processor._flag_case,
                    calculate_vertices_from_action=processor._calculate_vertices_from_action,
                    calculate_composition_features=_calculate_composition_features,
                    calculate_physics_features=_calculate_physics_features
                )
                if result is not None and result.get('degenerate_case', False):
                    degenerate_count += 1
                    logger.warning(f"[PIPELINE] Degenerate case detected for image: {image_path}")
                logger.info(f"[PIPELINE] INPUT: {image_path}")
                logger.info(f"[PIPELINE] OUTPUT: {result}")
                if result:
                    result['is_positive'] = True
                    pos_results.append(result)
                    all_results.append(result)
                    successful_images += 1
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
                result = process_single_image(
                    action_commands, image_id, False, problem_id, category, image_path,
                    processing_stats=processor.processing_stats,
                    flag_case=processor._flag_case,
                    calculate_vertices_from_action=processor._calculate_vertices_from_action,
                    calculate_composition_features=_calculate_composition_features,
                    calculate_physics_features=_calculate_physics_features
                )
                if result is not None and result.get('degenerate_case', False):
                    degenerate_count += 1
                    logger.warning(f"[PIPELINE] Degenerate case detected for image: {image_path}")
                logger.info(f"[PIPELINE] INPUT: {image_path}")
                logger.info(f"[PIPELINE] OUTPUT: {result}")
                if result:
                    result['is_positive'] = False
                    neg_results.append(result)
                    all_results.append(result)
                    successful_images += 1
                    summary = result.get('image_canonical_summary', {})
                    for fn in summary.get('unique_shape_functions', []):
                        problem_unique_shape_functions.add(fn)
                    for fn, count in summary.get('shape_function_counts', {}).items():
                        problem_shape_function_counts[fn] = problem_shape_function_counts.get(fn, 0) + count
                    for mod in summary.get('modifiers', []):
                        problem_modifiers.add(mod)

            # --- Verification: Ensure 7-7 split ---
            assert len(pos_results) == 7 and len(neg_results) == 7, \
                f"Expected 7 positives/7 negatives, got {len(pos_results)}/{len(neg_results)} for problem {problem_id}"

            # --- Aggregate image-level feature vectors for each support set ---
            def extract_feature_vector(result):
                features = result.get('image_level_features', {})
                if isinstance(features, dict):
                    return features
                elif isinstance(features, list):
                    # Convert list to dict with keys feature_0, feature_1, ...
                    return {f'feature_{i}': v for i, v in enumerate(features)}
                else:
                    logger.warning(f"[DEBUG] Unexpected image_level_features type: {type(features)} value: {features}")
                    return {}

            # Debug the data structure
            for i, result in enumerate(pos_results[:2]):  # Check first 2 results
                features = result.get('image_level_features', {})
                logger.info(f"[DEBUG] pos_results[{i}] image_level_features type: {type(features)}")
                logger.info(f"[DEBUG] pos_results[{i}] image_level_features value: {features}")
                if hasattr(features, 'keys'):
                    logger.info(f"[DEBUG] pos_results[{i}] has keys: {list(features.keys())}")

            positive_vectors = [extract_feature_vector(r) for r in pos_results]
            negative_vectors = [extract_feature_vector(r) for r in neg_results]

            # Compute all contextual metrics at the problem level
            from src.Derive_labels.context_features import BongardFeatureExtractor
            bfe = BongardFeatureExtractor()
            problem_support_context = bfe.extract_support_set_context(positive_vectors, negative_vectors)
            # Attach only the problem-level context to each image result
            for result in pos_results + neg_results:
                result['support_set_context_image'] = problem_support_context
                result['discriminative_features_image'] = problem_support_context.get('discriminative', {})

            # Save problem-level canonical summary
            # --- Aggregate multiscale and spatial topological features at problem level ---
            import numpy as np
            def aggregate_feature_dicts(dicts):
                import logging
                logger = logging.getLogger("aggregate_feature_dicts")
                if not dicts:
                    logger.warning("aggregate_feature_dicts: Input is empty.")
                    return {}
                valid_dicts = []
                for idx, d in enumerate(dicts):
                    if not isinstance(d, dict):
                        logger.warning(f"aggregate_feature_dicts: Non-dict item at index {idx}: type={type(d)}, value={repr(d)[:200]}")
                    else:
                        valid_dicts.append(d)
                if not valid_dicts:
                    logger.warning(f"aggregate_feature_dicts: No valid dicts to aggregate; input was: {dicts!r}")
                    return {}
                keys = set().union(*(d.keys() for d in valid_dicts))
                agg = {}
                for k in keys:
                    vals = [d[k] for d in valid_dicts if k in d and isinstance(d[k], (int, float, np.integer, np.floating))]
                    if vals:
                        agg[k] = {
                            'mean': float(np.mean(vals)),
                            'var': float(np.var(vals)),
                            'min': float(np.min(vals)),
                            'max': float(np.max(vals))
                        }
                return agg

            all_results_for_problem = pos_results + neg_results
            multiscale_list = [r.get('multiscale_features', {}) for r in all_results_for_problem]
            spatial_topo_list = [r.get('advanced_spatial_topological', {}) for r in all_results_for_problem]
            problem_multiscale = aggregate_feature_dicts(multiscale_list)
            problem_spatial_topological = aggregate_feature_dicts(spatial_topo_list)

            # --- Compositional hierarchical features at problem level ---
            compositional_features = {}
            try:
                    shape_vertices_list = [r.get('geometry', {}).get('vertices', []) for r in all_results_for_problem if isinstance(r, dict) and 'geometry' in r]
                    compositional_features['hierarchical_clustering_heights'] = hierarchical_clustering_heights(shape_vertices_list)

                    def to_dummy_tree(vertices_list):
                        # If already a dict, return as is
                        if isinstance(vertices_list, dict):
                            return vertices_list, next(iter(vertices_list)) if vertices_list else None
                        # If list, create a dummy tree: {0: [1, 2, ...]}
                        if isinstance(vertices_list, list):
                            tree = {0: list(range(1, len(vertices_list)+1))}
                            for i, v in enumerate(vertices_list, 1):
                                tree[i] = []
                            return tree, 0
                        # Otherwise, return empty tree
                        return {}, None

                    tree, root = to_dummy_tree(shape_vertices_list)
                    compositional_features['composition_tree_depth'] = composition_tree_depth(tree, root) if root is not None else 0
                    compositional_features['composition_tree_branching_factor'] = composition_tree_branching_factor(tree, root) if root is not None else 0.0
                    compositional_features['subgraph_isomorphism_frequencies'] = subgraph_isomorphism_frequencies(shape_vertices_list)
                    compositional_features['recursive_shape_patterns'] = recursive_shape_patterns(shape_vertices_list)
                    compositional_features['multi_level_symmetry_chains'] = multi_level_symmetry_chains(shape_vertices_list)
                    compositional_features['layered_edge_complexity'] = layered_edge_complexity(shape_vertices_list)
                    compositional_features['overlapping_substructure_ratios'] = overlapping_substructure_ratios(shape_vertices_list)
                    compositional_features['composition_regularity_score'] = composition_regularity_score(shape_vertices_list)
                    compositional_features['nested_convex_hull_levels'] = nested_convex_hull_levels(shape_vertices_list)
            except Exception as e:
                logger.warning(f"[PROBLEM LEVEL] Failed to compute compositional hierarchical features: {e}")
                compositional_features = {}

            problem_summary = {
                'problem_id': problem_id,
                'unique_shape_functions': sorted(list(problem_unique_shape_functions)),
                'shape_function_counts': problem_shape_function_counts,
                'modifiers': sorted(list(problem_modifiers)),
                'num_images': num_images_in_problem,
                'compositional_hierarchical_features': compositional_features,
                'problem_multiscale_features': problem_multiscale,
                'problem_spatial_topological_features': problem_spatial_topological
            }
            logger.info(f"[PROBLEM SUMMARY] Problem: {problem_id}\n{json.dumps(problem_summary, indent=2, ensure_ascii=False)}")
            problem_summaries.append(problem_summary)

            # --- Verification: Ensure 7-7 split ---
            assert len(pos_results) == 7 and len(neg_results) == 7, \
                f"Expected 7 positives/7 negatives, got {len(pos_results)}/{len(neg_results)} for problem {problem_id}"

            # --- Aggregate image-level feature vectors for each support set ---
            positive_vectors = [extract_feature_vector(r) for r in pos_results]
            negative_vectors = [extract_feature_vector(r) for r in neg_results]
            # Compute all contextual metrics at the problem level
            from src.Derive_labels.context_features import BongardFeatureExtractor
            bfe = BongardFeatureExtractor()
            problem_support_context = bfe.extract_support_set_context(positive_vectors, negative_vectors)
            # Attach only the problem-level context to each image result
            for result in pos_results + neg_results:
                result['support_set_context_image'] = problem_support_context
                result['discriminative_features_image'] = problem_support_context.get('discriminative', {})

            # Save problem-level canonical summary
            problem_summary = {
                'problem_id': problem_id,
                'unique_shape_functions': sorted(list(problem_unique_shape_functions)),
                'shape_function_counts': problem_shape_function_counts,
                'modifiers': sorted(list(problem_modifiers)),
                'num_images': num_images_in_problem
            }
            logger.info(f"[PROBLEM SUMMARY] Problem: {problem_id}\n{json.dumps(problem_summary, indent=2, ensure_ascii=False)}")
            problem_summaries.append(problem_summary)

        except Exception as e:
            logger.error(f"Error processing problem {problem_id}: {e}")
            processor._flag_case('unknown', problem_id, f'Problem processing failed: {e}', ['problem_processing_error'])


    # Save results
    try:
        from src.Derive_labels.shape_utils import json_safe
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ensure all_results is json_safe
        safe_results = json_safe(all_results)
        # Defensive output patch: ensure all action lists are flattened and stringified before output

        # Before saving all_results

        def robust_action_list_to_str(lst):
            # Converts a list of actions (possibly nested) to a list of strings using raw_command
            if isinstance(lst, list):
                # Defensive: ensure all items are strings
                return [robust_action_list_to_str(x) for x in lst] if lst and isinstance(lst[0], list) else [getattr(x, 'raw_command', str(x)) if type(x).__name__ in ['LineAction', 'ArcAction'] else str(x) for x in lst]
            elif type(lst).__name__ in ['LineAction', 'ArcAction']:
                return str(getattr(lst, 'raw_command', str(lst)))
            else:
                return str(lst)



        # Helper: recursively sanitize None values in dicts/lists
        def sanitize_none(obj, path="root"):
            if isinstance(obj, dict):
                return {k: sanitize_none(v, f"{path}.{k}") for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_none(x, f"{path}[{i}]") for i, x in enumerate(obj)]
            elif obj is None:
                logger.warning(f"[SERIALIZE PATCH] None value found at {path}, replacing with safe default.")
                return 0
            else:
                return obj

        # Defensive patch: ensure all results, flagged cases, stats, and summaries are robustly stringified and sanitized before serialization
        safe_results = ensure_all_strings(all_results)
        safe_results = sanitize_none(safe_results)
        logger.info(f"[SERIALIZE DEBUG][main] Final processed_results before writing: {safe_results}")
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(safe_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[SERIALIZE DEBUG][main] Exception during output serialization: {e}")
            raise

        logger.info(f"Successfully processed {successful_images}/{total_images} images")
        logger.info(f"Saved {len(all_results)} records to {args.output}")
        logger.info(f"[PIPELINE] Degenerate cases detected: {degenerate_count}")

        # Save flagged cases
        if processor.flagged_cases:
            flagged_path = os.path.join(output_dir, 'flagged_cases.json')
            safe_flagged = ensure_all_strings(processor.flagged_cases)
            safe_flagged = sanitize_none(safe_flagged)
            logger.info(f"[SERIALIZE DEBUG][main][flagged_cases] Final processed_flagged before writing: {safe_flagged}")
            try:
                with open(flagged_path, 'w', encoding='utf-8') as f:
                    json.dump(safe_flagged, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"[SERIALIZE DEBUG][main][flagged_cases] Exception during flagged case serialization: {e}")
                raise

        # Save processing statistics
        processor.processing_stats['processing_summary'] = {
            'success_rate': processor.processing_stats['successful'] / max(processor.processing_stats['total_processed'], 1),
            'flag_rate': processor.processing_stats['flagged'] / max(processor.processing_stats['total_processed'], 1),
            'total_features_calculated': len(all_results) * 4 if all_results else 0  # 4 feature sets per record
        }

        stats_path = os.path.join(output_dir, 'processing_statistics.json')
        safe_stats = ensure_all_strings(processor.processing_stats)
        safe_stats = sanitize_none(safe_stats)
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(safe_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processing statistics to {stats_path}")

        # Save problem-level canonical summaries
        problem_summary_path = os.path.join(output_dir, 'problem_summaries.json')
        safe_summaries = ensure_all_strings(problem_summaries)
        safe_summaries = sanitize_none(safe_summaries)
        with open(problem_summary_path, 'w', encoding='utf-8') as f:
            json.dump(safe_summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(problem_summaries)} problem-level canonical summaries to {problem_summary_path}")

        return 0

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return 1


if __name__ == '__main__':
    exit(main())