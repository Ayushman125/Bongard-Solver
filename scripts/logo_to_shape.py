
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"[DEBUG] Adding PROJECT_ROOT to sys.path: {PROJECT_ROOT}")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import json
# Add src to sys.path for imports
try:
    from src.data_pipeline.data_loader import load_action_programs
    from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser, OneStrokeShape, ensure_all_strings
    from src.physics_inference import symbolic_concept_features, extract_problem_level_features
except Exception as e:
    print(f"[IMPORT ERROR] Could not import from src.data_pipeline or src.physics_inference: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def split_support_holdout(examples):
    """
    Split 7 examples into 6 support and 1 hold-out.
    Returns: support_examples, holdout_example
    """
    return examples[:6], examples[6]

def assess_problem_level_concept(problem_id, positive_examples, negative_examples, logger):
    """
    Assess the concept at the problem level using symbolic features and context.
    Returns: dict with induced concept, hold-out validation, and logs.
    """
    # Defensive: ensure exactly 7 positive and 7 negative
    if len(positive_examples) != 7 or len(negative_examples) != 7:
        logger.warning(f"Problem {problem_id} does not have 7+7 examples, skipping.")
        return None
    # Split support/hold-out
    support_pos, holdout_pos = positive_examples[:6], positive_examples[6]
    support_neg, holdout_neg = negative_examples[:6], negative_examples[6]
    # Extract symbolic features for support sets
    support_pos_features = [symbolic_concept_features(ex) for ex in support_pos]
    support_neg_features = [symbolic_concept_features(ex) for ex in support_neg]
    # Induce concept: features present in positives, absent in negatives
    induced = extract_problem_level_features(support_pos, support_neg)
    logger.info(f"[{problem_id}] Induced concept: {induced}")
    # Hold-out validation
    holdout_results = []
    for ex, label in zip([holdout_pos, holdout_neg], ['positive', 'negative']):
        feats = symbolic_concept_features(ex)
        match = any(v for k, v in feats.items() if k in induced and induced[k])
        logger.info(f"[Hold-out] {problem_id} [{label}]: features={feats}, induced={induced}, match={match}")
        holdout_results.append({
            'problem_id': problem_id,
            'label': label,
            'features': feats,
            'induced_concept': induced,
            'match': match
        })
    return {
        'problem_id': problem_id,
        'induced_concept': induced,
        'holdout_results': holdout_results
    }
    parser = argparse.ArgumentParser(description="Extract derived labels for Bongard-LOGO problems.")
    parser.add_argument('--input-dir', required=True, help='Input directory containing ShapeBongard_V2')
    parser.add_argument('--output', required=True, help='Output JSON file for derived labels')
    parser.add_argument('--problems-list', required=True, help='File listing problem IDs to process')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("logo_to_shape")

    logger.info(f"Loading problem IDs from {args.problems_list}")
    with open(args.problems_list, 'r') as f:
        problem_ids = [line.strip() for line in f if line.strip()]

    logger.info(f"Loading action programs from {args.input_dir}")
    action_programs = load_action_programs(args.input_dir)

    derived_records = []
    for problem_id in problem_ids:
        logger.info(f"Parsing problem: {problem_id}")
        action_prog = action_programs.get(problem_id)
        if not action_prog or not isinstance(action_prog, list) or len(action_prog) != 2:
            logger.warning(f"No valid action program for problem_id: {problem_id}")
            continue
        positive_examples, negative_examples = action_prog
        # Problem-level concept assessment (contextual, set-level)
        result = assess_problem_level_concept(problem_id, positive_examples, negative_examples, logger)
        if result:
            derived_records.append(result)

    logger.info(f"Writing derived labels to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(derived_records, f, indent=2)
    logger.info("Finished writing derived labels.")

if __name__ == "__main__":
    main()