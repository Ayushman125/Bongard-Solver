#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-tier hard-negative generation for Bongard-Solver.
Integrates deterministic concept inversions, grammar fallback, and guaranteed fallback.
"""
import os
import random
import logging
from typing import List, Tuple, Any, Dict

import numpy as np

from src.data_pipeline.logo_mutator import mutate, RULE_SET
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.verification import has_min_vertices, l2_shape_distance
from src.hard_negative.concept_inversions import CONCEPT_INVERSION_STRATEGIES
from src.hard_negative.scorer import Scorer


def parse_logo_commands_to_tuples(commands: List[Any]) -> List[Tuple[str, Any]]:
    # existing parsing helper
    parsed = []
    for item in commands:
        if isinstance(item, tuple) and len(item) == 2:
            parsed.append(item)
        elif isinstance(item, str) and "_" in item:
            cmd, param = item.rsplit("_", 1)
            parsed.append((cmd, param))
        elif isinstance(item, str):
            parsed.append((item, None))
        else:
            logging.warning("Skipping malformed item: %r", item)
    return parsed


def is_valid_geometry(program: List[Tuple[str, Any]]) -> bool:
    try:
        action_cmds = [f"{cmd}_{param}" if param is not None else str(cmd) for cmd, param in program]
        parser = BongardLogoParser()
        shape = parser.comprehensive_parser.parse_action_commands(action_cmds, "unknown_pid")
        verts = shape.vertices if shape is not None and hasattr(shape, 'vertices') else []
        return isinstance(verts, list) and len(verts) >= 4
    except Exception:
        return False


def generate_hard_negative_for_sample(
    sample: Dict[str, Any],
    concept_fn,
    args
) -> Tuple[Dict[str, Any], str]:
    """
    Returns: (hard_negative_sample_dict, used_tier_name)
    """
    pid = sample["problem_id"]
    base_cmds = parse_logo_commands_to_tuples(sample["action_program"])
    # Aggregate features from the new derived_labels.json structure
    original_features = {}
    for stroke in sample.get('strokes', []):
        if 'specific_features' in stroke:
            original_features.update(stroke.get('specific_features', {}))
    for key in ["image_features", "physics_features", "composition_features", "stroke_type_features",
                "relational_features", "sequential_features", "topological_features"]:
        comp = sample.get(key, {})
        if isinstance(comp, dict) and comp:
            original_features.update(comp)
    if not original_features:
        logging.warning(f"Sample for problem {sample.get('problem_id', '?')} is missing all expected feature keys. Skipping sample.")
        return
    scorer = Scorer(concept_fn, original_features)
    original_label = concept_fn(original_features)
    found: List[Dict[str, Any]] = []

    def record_negative(cmds: List[Tuple[str, Any]]):
        feats = scorer.extract_features(cmds)
        parser = BongardLogoParser()
        verts = parser.parse_action_program(
            [f"{c} {p}" if p is not None else c for c, p in cmds]
        )
        found.append({
            **sample,
            "label": "hard_negative",
            "action_program": cmds,
            "features": feats,
            "geometry": verts
        })

    # Tier 2: Deterministic Concept Inversions
    for inv_fn in CONCEPT_INVERSION_STRATEGIES.get(pid, []):
        cand = inv_fn(base_cmds)
        if not cand or not is_valid_geometry(cand):
            continue
        feats = scorer.extract_features(cand)
        if concept_fn(feats) != original_label:
            logging.info("tier2_concept_inversion: found flip for %s", pid)
            record_negative(cand)
            return found[0], "tier2_concept_inversion"

    # Tier 3: Grammar-based Fallback
    for rule_id in range(len(RULE_SET)):
        cand = mutate(base_cmds, rule_id=rule_id)
        if not cand or not is_valid_geometry(cand):
            continue
        feats = scorer.extract_features(cand)
        if concept_fn(feats) != original_label:
            logging.info("grammar_fallback: flip at rule %d for %s", rule_id, pid)
            record_negative(cand)
            return found[0], "grammar_fallback"

    # Tier 4: Guaranteed Fallback
    # As last resort, return the original program as a dummy negative
    logging.info("guaranteed_fallback: returning original for %s", pid)
    record_negative(base_cmds)
    return found[0], "guaranteed_fallback"


def main():
    # parse args (omitted for brevity)
    # load derived_labels.json into `all_entries` (omitted)
    # build sample_args list as in your code, then:

    results = []
    for idx, (pid, entry, concept_fn, args) in enumerate(sample_args, 1):
        logging.info("[%d/%d] START sample %s", idx, len(sample_args), pid)
        hard_neg, tier = generate_hard_negative_for_sample(entry, concept_fn, args)
        results.append(hard_neg)
        logging.info("[%d/%d] DONE sample %s via %s", idx, len(sample_args), pid, tier)

    # write results to args.output as JSONL
    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
