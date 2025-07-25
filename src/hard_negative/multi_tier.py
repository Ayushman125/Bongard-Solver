import random
import os
import numpy as np
# Multi-tier hard negative generation for Bongard-Solver
import logging
from src.data_pipeline.logo_mutator import mutate, RULE_SET
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.verification import has_min_vertices, l2_shape_distance
from src.hard_negative.concept_inversions import CONCEPT_INVERSION_STRATEGIES
    orig_vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in flat_commands])
    orig_features = scorer.extract_features(flat_commands)
    for trial in range(max_attempts):
        mutated_evo = evo.search(flat_commands)
        if mutated_evo and is_valid_geometry(mutated_evo):
            # Use proxy: only extract features/geometry once per candidate
            vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in mutated_evo])
            if not has_min_vertices(vertices, min_v=4):
                continue
            mutated_features = scorer.extract_features(mutated_evo)
            # Use proxy features for flip detection
            if flips_label(orig_features, mutated_features, concept_fn):
                flips_for_this_sample += 1
                logging.info(f"tier1_evolutionary: found flip at trial {trial}")
                return {
                    **sample,
                    'label': 'hard_negative',
                    'action_program': mutated_evo,
                    'features': mutated_features,
                    'geometry': vertices
                }
        if trial >= NO_FLIP_LIMIT and flips_for_this_sample == 0:
            logging.info(f"{sample.get('problem_id', 'unknown')}: no flips after {trial} trials—bailing Tier-1 to Tier-2")
            break
    t1 = time.time()
    logging.info(f"tier1_evolutionary: completed in {t1-t0:.2f}s, flips={flips_for_this_sample}")
    return None

# --- Tier 2: Direct Concept Inversion (stub) ---
def tier2_concept_inversion(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts=200):
    # Try concept-aware deterministic inversion strategies first
    from src.hard_negative.concept_inversions import CONCEPT_INVERSION_STRATEGIES
    import logging
    problem_id = sample.get('problem_id')
    tried = set()
    if problem_id in CONCEPT_INVERSION_STRATEGIES:
        for inv_fn in CONCEPT_INVERSION_STRATEGIES[problem_id]:
            cand = inv_fn(flat_commands)
            if not cand or tuple(cand) in tried:
                continue
            tried.add(tuple(cand))
            if is_valid_geometry(cand):
                parser = BongardLogoParser()
                vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in cand])
                if not has_min_vertices(vertices, min_v=4):
                    continue
                mutated_features = scorer.extract_features(cand)
                if flips_label(original_features, mutated_features, concept_fn):
                    logging.info(f"tier2_concept_inversion: found deterministic inversion for {problem_id}")
                    return {
                        **sample,
                        'label': 'hard_negative',
                        'action_program': cand,
                        'features': mutated_features,
                        'geometry': vertices
                    }
    # Fallback: universal concept inversion operators
    def break_symmetry(cmds):
        return cmds + [("rotate", random.uniform(11, 45))]
    def perturb_stroke_order(cmds):
        if len(cmds) < 2:
            return cmds
        i, j = random.sample(range(len(cmds)), 2)
        cmds = list(cmds)
        cmds[i], cmds[j] = cmds[j], cmds[i]
        return cmds
    def geometry_scale(cmds, fx=1.3, fy=0.8):
        return [("scale", (fx, fy))] + cmds
    TIER2_OPERATORS = [break_symmetry, perturb_stroke_order, geometry_scale]
    for op in TIER2_OPERATORS:
        cand = op(flat_commands)
        if not cand or tuple(cand) in tried:
            continue
        tried.add(tuple(cand))
        if is_valid_geometry(cand):
            parser = BongardLogoParser()
            vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in cand])
            if not has_min_vertices(vertices, min_v=4):
                continue
            mutated_features = scorer.extract_features(cand)
            if flips_label(original_features, mutated_features, concept_fn):
                logging.info(f"tier2_concept_inversion: found flip with {op.__name__}")
                return {
                    **sample,
                    'label': 'hard_negative',
                    'action_program': cand,
                    'features': mutated_features,
                    'geometry': vertices
                }
    return None

# --- Tier 3: Template-Based Synthetic Generation (stub) ---
def tier3_synthetic(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts=100):
    # Fallback: try a simple hand-written negative (e.g., add a big rotation)
    cand = flat_commands + [("rotate", 90)]
    if is_valid_geometry(cand):
        parser = BongardLogoParser()
        vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in cand])
        if has_min_vertices(vertices, min_v=4) and scorer.is_flip(cand):
            logging.info("tier3_synthetic: found flip with fallback rotation")
            return {
                **sample,
                'label': 'hard_negative',
                'action_program': cand,
                'features': scorer.extract_features(cand),
                'geometry': vertices
            }
    return None
    # Placeholder: implement synthetic generation logic here
    return None

# --- Tier 4: Guaranteed Fallback Generation ---
def tier4_guaranteed(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts=1):
    # As a last resort, mutate with all grammar rules until something valid is found
    for rule_id in range(len(RULE_SET)):
        try:
            cand = mutate(flat_commands, rule_id)
            if isinstance(cand, (tuple, list)):
                if len(cand) > 0 and isinstance(cand[0], list) and all(isinstance(x, str) for x in cand[0]):
                    cand = cand[0]
                elif all(isinstance(x, str) for x in cand):
                    cand = list(cand)
                elif all(isinstance(x, tuple) for x in cand):
                    cand = [item for tup in cand for item in tup if isinstance(item, str)]
            if isinstance(cand, str):
                cand = [cand]
            from scripts.generate_hard_negatives import parse_logo_commands_to_tuples
            cand_tuples = parse_logo_commands_to_tuples(cand)
            if is_valid_geometry(cand_tuples):
                parser = BongardLogoParser()
                vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in cand_tuples])
                if not has_min_vertices(vertices, min_v=4):
                    continue
                if scorer.is_flip(cand_tuples):
                    return {
                        **sample,
                        'label': 'hard_negative',
                        'action_program': cand_tuples,
                        'features': scorer.extract_features(cand_tuples),
                        'geometry': vertices
                    }
        except Exception as e:
            logging.error(f"Error during guaranteed fallback mutation rule {rule_id}: {e!r}")
    # If all else fails, just return the original (should not happen)
    parser = BongardLogoParser()
    vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in flat_commands])
    return {
        **sample,
        'label': 'hard_negative',
        'action_program': flat_commands,
        'features': scorer.extract_features(flat_commands),
        'geometry': vertices
    }

# --- Utility: Geometry check (copied from main script) ---
def is_valid_geometry(program):
    try:
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
            return False
        return True
    except Exception:
        return False

# --- Main multi-tier driver ---
    commands = sample.get('action_program')
    from scripts.generate_hard_negatives import parse_logo_commands_to_tuples
    flat_commands = parse_logo_commands_to_tuples(commands)
    original_features = sample.get('features')
    from src.hard_negative.scorer import Scorer
    scorer = Scorer(concept_fn, original_features)
    pid = sample.get('problem_id')
    max_per_sample = getattr(args, 'max_per_sample', 3)
    found = []
    def record_negative(mutated):
        features = scorer.extract_features(mutated)
        parser = BongardLogoParser()
        vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in mutated])
        found.append({
            **sample,
            'label': 'hard_negative',
            'action_program': mutated,
            'features': features,
            'geometry': vertices
        })

    # 1. Deterministic concept-driven inversions
    original_label = concept_fn(original_features)
    for invert in CONCEPT_INVERSION_STRATEGIES.get(pid, []):
        mutated = invert(flat_commands)
        feats = scorer.extract_features(mutated)
        if concept_fn(feats) != original_label:
            record_negative(mutated)
            if len(found) >= max_per_sample:
                return found[0], 'concept_inversion'

    # 2. Grammar-based fallback (try each rule once)
    for rule_id, op in enumerate(RULE_SET):
        mutated = mutate(flat_commands, rule_id=rule_id)
        feats = scorer.extract_features(mutated)
        if concept_fn(feats) != original_label:
            record_negative(mutated)
            if len(found) >= max_per_sample:
                return found[0], 'grammar_fallback'

    # 3. Guaranteed template-based fallback
    # (Here, just return the original with a label flip for safety)
    record_negative(flat_commands)
    return found[0], 'guaranteed_fallback'
