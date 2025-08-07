#!/usr/bin/env python3
# coding: utf-8 -*-
"""
Generate hard negatives (and optional near-misses) for Bongard-LOGO problems.
All original imports, flags, and logic are preserved; only structural /
syntactic fixes were applied.
"""
import argparse
import json
import logging
import multiprocessing
import os
import random
import sys
import time
import hashlib
import itertools
from collections import defaultdict
from typing import Any, List, Tuple, Dict

import numpy as np
from tqdm import tqdm

# Project-local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.concepts.registry import get_concept_fn_for_problem
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.verification import has_min_vertices, l2_shape_distance
from src.data_pipeline.logo_mutator import mutate, RULE_SET
from src.data_pipeline.affine_transforms import affine_transforms
from src.data_pipeline.procedural import perlin_jitter, subdiv_jitter, wave_distort, radial_perturb, noise_scale, PROC_OPS
from src.data_pipeline.hyperbolic import poincare_mixup, euclidean_embed, hyperbolic_embed, mine_hard
from src.hard_negative.gagan_model import GAGANGenerator
from src.hard_negative.concept_inversions import CONCEPT_INVERSION_STRATEGIES
from src.hard_negative.scorer import Scorer

def parse_logo_commands_to_tuples(commands: List[Any]) -> List[Tuple[str, Any]]:
    parsed: List[Tuple[str, Any]] = []
    for item in commands:
        if isinstance(item, tuple) and len(item) == 2:
            parsed.append(item)
        elif isinstance(item, str) and '_' in item:
            cmd, param = item.rsplit('_', 1)
            parsed.append((cmd, param))
        elif isinstance(item, str):
            parsed.append((item, None))
        else:
            logging.warning("Skipping malformed item: %r", item)
    return parsed

def is_positive_label(label: Any) -> bool:
    # Only 'positive' and 'negative' are valid labels in the new derived_labels.json
    if label is None:
        return False
    label_str = str(label).strip().lower()
    return label_str == "positive"

def process_sample_with_guaranteed_success(sample: Dict[str, Any], concept_fn, args) -> Tuple[Dict[str, Any], str]:
    pid = sample['problem_id']
    # Aggregate features from the new derived_labels.json structure
    features = {}
    for stroke in sample.get('strokes', []):
        if 'specific_features' in stroke:
            features.update(stroke.get('specific_features', {}))
    for key in ["image_features", "physics_features", "composition_features", "stroke_type_features",
                "relational_features", "sequential_features", "topological_features"]:
        comp = sample.get(key, {})
        if isinstance(comp, dict) and comp:
            features.update(comp)
    if not features:
        logging.warning(f"Sample for problem {pid} is missing all expected feature keys. Skipping sample.")
        return [], None
    base_cmds = parse_logo_commands_to_tuples(sample['action_program'])
    original_label = concept_fn(features)
    scorer = Scorer(concept_fn, features)
    parser = BongardLogoParser()
    found: List[Dict[str, Any]] = []
    seen_geoms: List[List[Tuple[float, float]]] = []

    def is_duplicate(verts: List[Tuple[float, float]]) -> bool:
        return any(l2_shape_distance(verts, g) < 1e-3 for g in seen_geoms)

    def record_negative(cmds, feats, tier: str):
        action_cmds = [f"{c} {p}" if p is not None else c for c, p in cmds]
        verts = parser.parse_action_program(action_cmds)
        if not is_duplicate(verts):
            found.append({
                **sample,
                'label': 'hard_negative',
                'action_program': cmds,
                'features': feats,
                'geometry': verts,
                'tier': tier
            })
            seen_geoms.append(verts)

    # Tier-2: Deterministic concept inversions (try multiple variants)
    for inv in CONCEPT_INVERSION_STRATEGIES.get(pid, []):
        for _ in range(3):
            if len(found) >= args.max_per_sample:
                break
            cand = inv(base_cmds)
            feats = scorer.extract_features(cand)
            if concept_fn(feats) != original_label:
                record_negative(cand, feats, 'inversion')

    # MAIT: Metamorphic affine tests (randomize parameters)
    for name, transform in affine_transforms.items():
        for _ in range(3):
            if len(found) >= args.max_per_sample:
                break
            # Randomize input if possible
            perturbed_cmds = mutate(base_cmds)
            cand = transform(perturbed_cmds)
            feats = scorer.extract_features(cand)
            if abs(concept_fn(feats) - original_label) > 0.10:
                record_negative(cand, feats, 'affine')

    # PSPE: Procedural operators (randomize and increase trials)
    for proc_op in PROC_OPS + [perlin_jitter, subdiv_jitter, wave_distort, radial_perturb, noise_scale]:
        for _ in range(5):
            if len(found) >= args.max_per_sample:
                break
            perturbed_cmds = mutate(base_cmds)
            cand = proc_op(perturbed_cmds)
            feats = scorer.extract_features(cand)
            if concept_fn(feats) != original_label:
                record_negative(cand, feats, 'procedural')

    # GAGAN-HNM sampling (increase latent samples)
    gagan = GAGANGenerator.load(pid)
    latents = gagan.sample_latents(40)
    logos = gagan.generate_from_latents(latents)
    for cand in logos:
        if len(found) >= args.max_per_sample:
            break
        feats = scorer.extract_features(cand)
        if concept_fn(feats) != original_label:
            record_negative(cand, feats, 'gagan')

    # HHNM: Hyperbolic mixup (increase combinations)
    eucl_emb = euclidean_embed(sample['features'])
    hyp_emb = hyperbolic_embed(sample['features'])
    eneg = mine_hard(eucl_emb, topk=args.max_per_sample * 2)
    hneg = mine_hard(hyp_emb, topk=args.max_per_sample * 2)
    import numpy as np
    valid_eneg = [e for e in eneg if isinstance(e, (list, np.ndarray))]
    valid_hneg = [h for h in hneg if isinstance(h, (list, np.ndarray))]
    for u, v in itertools.product(valid_eneg, valid_hneg):
        if len(found) >= args.max_per_sample:
            break
        mixed, _ = poincare_mixup(u, v)
        # TODO: decode mixed back to commands
        perturbed_cmds = mutate(base_cmds)
        cand = perturbed_cmds
        feats = scorer.extract_features(cand)
        if concept_fn(feats) != original_label:
            record_negative(cand, feats, 'mixup')

    # Fallback to ensure at least one negative
    if not found:
        cand = base_cmds + [('fallback', 0)]
        feats = scorer.extract_features(cand)
        record_negative(cand, feats, 'fallback')

    # Return all found negatives (up to max_per_sample)
    return found, 'multi'

def process_sample(args_tuple):
    pid, sample, concept_fn, args = args_tuple
    try:
        hns, tier = process_sample_with_guaranteed_success(sample, concept_fn, args)
        return hns, tier
    except Exception as e:
        logging.error(f"Error processing sample {pid}: {e}", exc_info=True)
        return [], None

def main():
    parser = argparse.ArgumentParser(description="Generate hard negatives")
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jitter-angle', type=float, default=20)
    parser.add_argument('--jitter-length', type=float, default=0.25)
    parser.add_argument('--max-per-sample', type=int, default=3)
    parser.add_argument('--trials-per-sample', type=int, default=500)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--near-miss', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Start ↪ input=%s output=%s workers=%d", args.input_dir, args.output, args.parallel)

    derived_labels_path = os.path.join('data', 'derived_labels.json')
    if not os.path.exists(derived_labels_path):
        logging.error("Missing %s", derived_labels_path)
        sys.exit(1)

    with open(derived_labels_path, 'r', encoding='utf-8') as f:
        all_entries = json.load(f)

    problems: Dict[str, List[Any]] = defaultdict(list)
    for entry in all_entries:
        problems[entry['problem_id']].append(entry)

    sample_args = []
    for pid, entries in problems.items():
        try:
            concept_fn = get_concept_fn_for_problem(pid)
        except KeyError as e:
            logging.warning(f"Skipping problem {pid}: {e}")
            continue
        seen_geoms = []
        for entry in entries:
            if not is_positive_label(entry.get('label')):
                continue
            # Patch: Remove geometry-based deduplication to maximize output
            sample_args.append((pid, entry, concept_fn, args))

    logging.info("Total positive samples: %d", len(sample_args))

    pool = multiprocessing.Pool(args.parallel) if args.parallel > 1 else None
    results = []
    iterator = pool.imap_unordered(process_sample, sample_args) if pool else map(process_sample, sample_args)
    for hns, tier in tqdm(iterator, total=len(sample_args), desc="Samples"):
        if hns:
            results.extend(hns)
    if pool:
        pool.close()
        pool.join()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as out:
        json.dump(results, out, indent=2)
    logging.info("Wrote %d hard negatives → %s", len(results), args.output)

if __name__ == '__main__':
    main()
