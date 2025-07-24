def parse_logo_commands_to_tuples(commands):
    """
    Convert a list of LOGO commands (strings or tuples) to a list of (cmd, param) tuples.
    Accepts:
      - ['line_normal_0.354-0.500', ...]  → [('line_normal', '0.354-0.500'), ...]
      - [('fd', 30), ...]                 → unchanged
    """
    parsed = []
    for item in commands:
        if isinstance(item, tuple) and len(item) == 2:
            parsed.append(item)
        elif isinstance(item, str):
            # Try to split at the last underscore (for commands like 'line_normal_0.354-0.500')
            if '_' in item:
                cmd, param = item.rsplit('_', 1)
                parsed.append((cmd, param))
            else:
                parsed.append((item, None))
        else:
            # Malformed item, skip or log
            import logging
            logging.warning(f"parse_logo_commands_to_tuples: Skipping malformed item: {item!r}")
    return parsed
import random
import sys
import os
import argparse
import ijson
import json
import re
import logging
import multiprocessing
from collections import defaultdict

# Ensure that the project root is in the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project-specific modules
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference
from src.data_pipeline.verification import has_min_vertices, l2_shape_distance
from src.concepts.registry import get_concept_fn_for_problem
from src.hard_negative.evo_search import EvoPerturber
from src.data_pipeline.logo_mutator import mutate, RULE_SET

# --- Scorer wrapper for EvoPerturber ---
class Scorer:
    def predict_concept_confidence(self, prog):
        # Optionally, implement a confidence score for near-miss mining
        # Here, just return 0.5 as a placeholder, or use a real scoring function if available
        # You can replace this with your actual logic
        try:
            features = self.extract_features(prog)
            # If your concept_fn returns a probability/confidence, use it
            if hasattr(self.concept_fn, 'predict_concept_confidence'):
                return self.concept_fn.predict_concept_confidence(features)
            # Otherwise, use a simple proxy (e.g., area normalized)
            if 'area' in features:
                return min(1.0, max(0.0, features['area'] / 10000.0))
            return 0.5
        except Exception:
            return 0.5

    def __init__(self, concept_fn, orig_features):
        self.concept_fn = concept_fn
        self.orig_features = orig_features

    def is_flip(self, mutated_prog):
        # You may need to extract features from mutated_prog here
        # For now, assume mutated_prog is a dict with 'features' or similar
        # If not, you must add feature extraction logic
        features = self.extract_features(mutated_prog)
        return flips_label(self.orig_features, features, self.concept_fn)

    def geom_distance(self, orig_prog, mutated_prog):
        # Placeholder: use a simple edit distance or length diff
        # Replace with your actual geometry/command distance if available
        return abs(len(orig_prog) - len(mutated_prog))

    def extract_features(self, prog):
        # Robust feature extraction for LOGO program (list of (cmd, param) tuples or strings)
        try:
            # Convert (cmd, param) tuples to action strings if needed
            if isinstance(prog, list) and all(isinstance(x, tuple) and len(x) == 2 for x in prog):
                action_cmds = [f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in prog]
            elif isinstance(prog, list) and all(isinstance(x, str) for x in prog):
                action_cmds = prog
            else:
                action_cmds = [str(x) for x in prog]

            parser = BongardLogoParser()
            vertices = parser.parse_action_program(action_cmds)
            if not isinstance(vertices, list) or len(vertices) < 4:
                logging.warning(f"extract_features: Skipping feature extraction, only {len(vertices) if isinstance(vertices, list) else 'N/A'} vertices (need >=4) for valid polygon.")
                # Return safe defaults or original features
                return self.orig_features if self.orig_features else {}

            features = {}
            poly = PhysicsInference.polygon_from_vertices(vertices)
            features['centroid'] = PhysicsInference.centroid(poly)
            features['area'] = PhysicsInference.area(poly)
            features['is_convex'] = PhysicsInference.is_convex(poly)
            features['symmetry_score'] = PhysicsInference.symmetry_score(vertices)
            features['moment_of_inertia'] = PhysicsInference.moment_of_inertia(vertices)
            # Add other features if they are implemented in PhysicsInference
            if hasattr(PhysicsInference, 'num_straight'):
                features['num_straight'] = PhysicsInference.num_straight(vertices)
            if hasattr(PhysicsInference, 'has_quadrangle'):
                features['has_quadrangle'] = PhysicsInference.has_quadrangle(vertices)
            if hasattr(PhysicsInference, 'has_obtuse'):
                features['has_obtuse'] = PhysicsInference.has_obtuse(vertices)
            return features
        except Exception as e:
            logging.warning(f"Feature extraction failed in Scorer.extract_features: {e!r}")
            return self.orig_features if self.orig_features else {}

def is_valid_geometry(program):
    # Checks for minimum polygon validity: at least 4 coordinates (vertices)
    try:
        # Convert (cmd, param) tuples to action strings if needed
        if isinstance(program, list) and all(isinstance(x, tuple) and len(x) == 2 for x in program):
            action_cmds = [f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in program]
        elif isinstance(program, list) and all(isinstance(x, str) for x in program):
            action_cmds = program
        else:
            action_cmds = [str(x) for x in program]

        parser = BongardLogoParser()
        vertices = parser.parse_action_program(action_cmds)
        if not isinstance(vertices, list):
            return False
        if len(vertices) < 4:
            logging.warning(f"Rejected hard negative: only {len(vertices)} vertices (need >=4) for valid polygon.")
            return False
        return True
    except Exception as e:
        logging.warning(f"is_valid_geometry: Exception during geometry check: {e!r}")
        return False

def is_diverse(new, existing):
    # Placeholder: compute vector or label difference
    return True

def is_near_flip(conf, threshold=0.05):
    return abs(conf - 0.5) < threshold

# Structural perturbation
def structural_perturb(cmds):
    cmds = cmds.copy()
    ops = ['delete', 'duplicate', 'swap', 'toggle', 'insert_arc', 'reverse', 'block_swap']
    op = random.choice(ops)
    if op == 'delete' and len(cmds) > 3:
        cmds.pop(random.randrange(len(cmds)))
    elif op == 'duplicate':
        cmds.insert(random.randrange(len(cmds)), random.choice(cmds))
    elif op == 'swap' and len(cmds) > 1:
        i, j = random.sample(range(len(cmds)), 2)
        cmds[i], cmds[j] = cmds[j], cmds[i]
    elif op == 'toggle':
        k = random.randrange(len(cmds))
        cmds[k] = cmds[k].replace('line_', 'arc_') if cmds[k].startswith('line_') else cmds[k].replace('arc_', 'line_')
    elif op == 'insert_arc':
        idx = random.randrange(len(cmds)+1)
        arc_cmd = f"arc_normal_{random.random():.3f}-{random.random():.3f}"
        cmds.insert(idx, arc_cmd)
    elif op == 'reverse' and len(cmds) > 2:
        start = random.randrange(len(cmds)-1)
        end = random.randrange(start+1, len(cmds))
        cmds[start:end] = list(reversed(cmds[start:end]))
    elif op == 'block_swap' and len(cmds) > 4:
        block_indices = random.sample(range(len(cmds)), 3)
        block_indices.sort()
        if len(block_indices) == 3:
            cmds[block_indices[0]:block_indices[2]+1] = cmds[block_indices[0]:block_indices[2]+1][::-1]
    return cmds

def flatten_action_program(action_program):
    # Flatten nested lists to a single list of strings
    flat = []
    for item in action_program:
        if isinstance(item, list):
            flat.extend(flatten_action_program(item))
        else:
            flat.append(item)
    return flat

def flips_label(original_features, perturbed_features, concept_test):
    original_label = concept_test(original_features)
    perturbed_label = concept_test(perturbed_features)
    return original_label != perturbed_label

def concept_test(features):
    # Placeholder: implement your concept test logic here
    # For example, return True if area > threshold
    return features.get('area', 0) > 0.5 # Use .get with a default for robustness

# Move process_sample to top-level for multiprocessing
from src.hard_negative.multi_tier import process_sample_with_guaranteed_success

def process_sample(args_tuple):
    pid, sample, concept_fn, args = args_tuple
    # Use the multi-tier driver to guarantee at least one hard negative per sample
    try:
        hard_negative, used_tier = process_sample_with_guaranteed_success(sample, concept_fn, args)
        logging.info(f"Sample {pid}: Generated hard negative using {used_tier}")
        return hard_negative, None  # No near-miss needed since we have guarantee
    except Exception as e:
        logging.error(f"Error in process_sample_with_guaranteed_success for {pid}: {e!r}")
        return None, None
def main():
    parser = argparse.ArgumentParser(description="Generate hard negatives using evolutionary and grammar-based mutation.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing Bongard-LOGO problems')
    parser.add_argument('--output', required=True, help='Output file for hard negatives')
    parser.add_argument('--jitter-angle', type=float, default=15, help='Base angle jitter for centroid perturbation')
    parser.add_argument('--jitter-length', type=float, default=0.15, help='Base length jitter for area perturbation')
    parser.add_argument('--max-per-problem', type=int, default=14, help='(DEPRECATED) Use --max-per-sample instead')
    parser.add_argument('--max-per-sample', type=int, default=3, help='Max hard negatives per positive sample (default=3)')
    parser.add_argument('--trials-per-sample', type=int, default=500, help='Trials per positive sample')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--near-miss', action='store_true', help='Store near-miss samples')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Starting hard negative generation...")
    logging.info(f"Input dir: {args.input_dir}")
    logging.info(f"Output file: {args.output}")
    logging.info(f"Jitter angle: {args.jitter_angle}, Jitter length: {args.jitter_length}")
    logging.info(f"Trials per sample: {args.trials_per_sample}, Max per problem: {args.max_per_problem}")
    logging.info(f"Parallel workers: {args.parallel}, Near-miss: {args.near_miss}")

    derived_labels_path = os.path.join('data', 'derived_labels.json')
    if not os.path.exists(derived_labels_path):
        logging.error(f"Missing derived_labels.json at {derived_labels_path}")
        return

    all_entries = []
    try:
        with open(derived_labels_path, 'r') as f:
            all_entries = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {derived_labels_path}: {e}")
        return

    problems = defaultdict(list)
    for entry in all_entries:
        problems[entry['problem_id']].append(entry)

    hard_negatives_output = []
    near_misses_output = []
    total_problems = 0
    total_samples = 0

    sample_args = []
    for pid, entries in problems.items():
        total_problems += 1
        positives = [e for e in entries if e.get('label') == 'category_1']
        logging.info(f"Processing problem {total_problems}: {pid} ({len(positives)} positives)")


        try:
            concept_fn = get_concept_fn_for_problem(pid)
        except Exception as e:
            logging.error(f"No concept function registered for problem {pid}: {e}")
            raise

        for sample_idx, sample in enumerate(positives):
            total_samples += 1
            sample_args.append((pid, sample, concept_fn, args))

    logging.info(f"Prepared {len(sample_args)} sample arguments for processing.")

    results = []
    if args.parallel > 1:
        logging.info(f"Using multiprocessing with {args.parallel} workers.")
        try:
            with multiprocessing.Pool(args.parallel) as pool:
                results = pool.map(process_sample, sample_args)
        except Exception as e:
            logging.error(f"Exception during multiprocessing: {e}")
            import traceback
            logging.error(traceback.format_exc())
    else:
        for idx, arg in enumerate(sample_args):
            logging.info(f"Processing sample {idx+1}/{len(sample_args)}: {arg[0]}")
            try:
                res, near_res = process_sample(arg)
                results.append((res, near_res))
            except Exception as e:
                logging.error(f"Exception in process_sample for {arg[0]}: {e}")
                import traceback
                logging.error(traceback.format_exc())

    # Enforce diversity and non-degeneracy per positive sample
    unique_hard_negatives = []
    for hn_res, nm_res in results:
        if hn_res:
            vertices = hn_res.get('geometry', [])
            if not has_min_vertices(vertices, min_v=4):
                continue
            is_duplicate = False
            for e in unique_hard_negatives:
                if l2_shape_distance(vertices, e.get('geometry', [])) < 1e-3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_hard_negatives.append(hn_res)
        if nm_res:
            near_misses_output.append(nm_res)
    hard_negatives_output = unique_hard_negatives

    logging.info(f"Finished processing all samples. Writing output...")
    try:
        # Write hard negatives as a JSON list
        with open(args.output, 'w') as f:
            json.dump(hard_negatives_output, f, indent=2)
        logging.info(f"Hard negatives output written to {args.output} ({len(hard_negatives_output)} entries).")

        if args.near_miss:
            near_miss_output_path = args.output.replace('.txt', '_near_miss.json')
            if '.json' in near_miss_output_path:
                near_miss_output_path = args.output.replace('.json', '_near_miss.json')
            with open(near_miss_output_path, 'w') as f:
                json.dump(near_misses_output, f, indent=2)
            logging.info(f"Near-miss output written to {near_miss_output_path} ({len(near_misses_output)} entries).")

    except IOError as e:
        logging.error(f"Error writing output files: {e}")

    logging.info(f"Hard negative generation complete. Processed {total_problems} problems and {total_samples} samples.")

if __name__ == '__main__':
    main()