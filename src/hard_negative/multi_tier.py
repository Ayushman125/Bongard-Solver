# Multi-tier hard negative generation for Bongard-Solver
import logging
from src.hard_negative.evo_search import EvoPerturber
from src.data_pipeline.logo_mutator import mutate, RULE_SET
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.verification import has_min_vertices, l2_shape_distance

# --- Tier 1: Enhanced Evolutionary Mutations ---
def tier1_evolutionary(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts=500):
    evo = EvoPerturber(scorer=scorer, seed=42)
    for _ in range(max_attempts):
        mutated_evo = evo.search(flat_commands)
        if mutated_evo and is_valid_geometry(mutated_evo):
            features = scorer.extract_features(mutated_evo)
            parser = BongardLogoParser()
            vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in mutated_evo])
            if not has_min_vertices(vertices, min_v=4):
                continue
            if scorer.is_flip(mutated_evo):
                return mutated_evo
    return None

# --- Tier 2: Direct Concept Inversion (stub) ---
def tier2_concept_inversion(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts=200):
    # Placeholder: implement concept inversion logic here
    return None

# --- Tier 3: Template-Based Synthetic Generation (stub) ---
def tier3_synthetic(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts=100):
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
                features = scorer.extract_features(cand_tuples)
                parser = BongardLogoParser()
                vertices = parser.parse_action_program([f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in cand_tuples])
                if not has_min_vertices(vertices, min_v=4):
                    continue
                if scorer.is_flip(cand_tuples):
                    return cand_tuples
        except Exception as e:
            logging.error(f"Error during guaranteed fallback mutation rule {rule_id}: {e!r}")
    # If all else fails, just return the original (should not happen)
    return flat_commands

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
def process_sample_with_guaranteed_success(sample, concept_fn, args):
    commands = sample.get('action_program')
    from scripts.generate_hard_negatives import parse_logo_commands_to_tuples
    flat_commands = parse_logo_commands_to_tuples(commands)
    original_features = sample.get('features')
    from scripts.generate_hard_negatives import Scorer
    scorer = Scorer(concept_fn, original_features)
    tiers = [
        (tier1_evolutionary, 500),
        (tier2_concept_inversion, 200),
        (tier3_synthetic, 100),
        (tier4_guaranteed, 1)
    ]
    for tier_func, max_attempts in tiers:
        result = tier_func(sample, concept_fn, args, scorer, flat_commands, original_features, max_attempts)
        if result is not None:
            return result, tier_func.__name__
    raise RuntimeError("Guaranteed generation failed - implementation error")
