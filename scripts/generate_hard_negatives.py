#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Project-local modules (path fix kept exactly as in your fragment)
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.concepts.registry import get_concept_fn_for_problem
from src.data_pipeline.logo_parser import BongardLogoParser
from src.data_pipeline.physics_infer import PhysicsInference
from src.data_pipeline.verification import has_min_vertices, l2_shape_distance
from src.hard_negative.multi_tier import process_sample_with_guaranteed_success
from src.hard_negative.evo_search import EvoPerturber
from src.data_pipeline.logo_mutator import RULE_SET
from src.hard_negative.scorer import Scorer

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def parse_logo_commands_to_tuples(commands: List[Any]) -> List[Tuple[str, str]]:
    """
    Normalise LOGO command list → list[(cmd, param)].

    Examples
    --------
    ['line_normal_0.354-0.500']   -> [('line_normal', '0.354-0.500')]
    [('fd', 30)]                  -> [('fd', 30)]
    """
    parsed: List[Tuple[str, str]] = []
    for item in commands:
        if isinstance(item, tuple) and len(item) == 2:
            parsed.append(item)
        elif isinstance(item, str):
            if '_' in item:
                cmd, param = item.rsplit('_', 1)
                parsed.append((cmd, param))
            else:
                parsed.append((item, None))
        else:
            logging.warning(
                "parse_logo_commands_to_tuples: Skipping malformed item: %r", item
            )
    return parsed


def is_positive_label(label: Any) -> bool:
    """
    Robust match for positive labels: category_1, positive, 1, true, yes (any case).
    """
    if label is None:
        return False
    label_str = str(label).strip().lower()
    return label_str in {"category_1", "positive", "1", "true", "yes"}


def is_valid_geometry(program: List[Any]) -> bool:
    """
    True if program yields ≥4 vertices. Keeps untouched logic; only defensive
    coding and logging tweaks.
    """
    try:
        if all(isinstance(x, tuple) and len(x) == 2 for x in program):
            action_cmds = [
                f"{cmd} {param}" if param is not None else str(cmd) for cmd, param in program
            ]
        elif all(isinstance(x, str) for x in program):
            action_cmds = program
        else:
            action_cmds = [str(x) for x in program]

        parser = BongardLogoParser()
        vertices = parser.parse_action_program(action_cmds)
        if not isinstance(vertices, list) or len(vertices) < 4:
            logging.warning("Rejected: only %s vertices (need ≥4).", len(vertices))
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logging.warning("is_valid_geometry: %r", exc)
        return False


# --------------------------------------------------------------------------- #
# Placeholder helpers retained as-is
# --------------------------------------------------------------------------- #
def is_diverse(new, existing) -> bool:  # noqa: ANN001
    # Batch-level diversity: ensure new is not too close to any in existing (L2 shape distance > threshold)
    if not existing:
        return True
    verts_new = new.get('geometry', [])
    for e in existing:
        verts_e = e.get('geometry', [])
        if l2_shape_distance(verts_new, verts_e) < 0.05:  # Stricter than dedup
            return False
    return True


def is_near_flip(conf: float, threshold: float = 0.05) -> bool:
    return abs(conf - 0.5) < threshold


# --------------------------------------------------------------------------- #
# process_sample wrapper that calls the multi-tier guarantor
# --------------------------------------------------------------------------- #
def process_sample(args_tuple):
    pid, sample, concept_fn, args = args_tuple
    try:
        hard_negative, used_tier = process_sample_with_guaranteed_success(
            sample, concept_fn, args
        )
        if isinstance(hard_negative, list):
            logging.error("Tier returned list instead of dict: %r", hard_negative)
            return None, None
        return hard_negative, None
    except Exception as exc:  # noqa: BLE001
        logging.error(f"process_sample failure for {pid}: {exc}", exc_info=True)
        return None, None


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #
def main():

    parser = argparse.ArgumentParser(
        description="Generate hard negatives using evolutionary and grammar-based mutation."
    )
    parser.add_argument('--input-dir', required=True, help='Input Bongard-LOGO root')
    parser.add_argument('--output', required=True, help='JSON file for hard negatives')
    parser.add_argument('--jitter-angle', type=float, default=15)
    parser.add_argument('--jitter-length', type=float, default=0.15)
    parser.add_argument('--max-per-problem', type=int, default=14)  # deprecated
    parser.add_argument('--max-per-sample', type=int, default=3)
    parser.add_argument('--trials-per-sample', type=int, default=500)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--near-miss', action='store_true')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(
        "Start ↪ input=%s  output=%s  workers=%d",
        args.input_dir,
        args.output,
        args.parallel,
    )

    derived_labels_path = os.path.join('data', 'derived_labels.json')
    if not os.path.exists(derived_labels_path):
        logging.error("Missing %s", derived_labels_path)
        sys.exit(1)

    try:
        with open(derived_labels_path, 'r', encoding='utf-8') as handle:
            all_entries = json.load(handle)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        logging.error("JSON decode error: %r", exc)
        sys.exit(1)

    problems: dict[str, list] = defaultdict(list)
    for entry in all_entries:
        problems[entry['problem_id']].append(entry)

    # Build argument tuples with deduplication by (problem_id, image_path)
    sample_args = []
    seen = set()
    for pid, entries in problems.items():
        concept_fn = get_concept_fn_for_problem(pid)
        for entry in entries:
            if not is_positive_label(entry.get('label')):
                continue
            # Normalize image_path for deduplication
            path = os.path.normpath(os.path.abspath(entry['image_path']))
            prog_str = json.dumps(entry['action_program'], sort_keys=True)
            import hashlib
            fingerprint = hashlib.sha1(prog_str.encode()).hexdigest()
            key = (pid, path, fingerprint)
            if key in seen:
                logging.debug(f"Dedup skip: {key}")
                continue
            seen.add(key)
            sample_args.append((pid, entry, concept_fn, args))

    logging.info("Total positive samples: %d", len(sample_args))

    # Parallel / serial execution
    results = []
    if args.parallel > 1:
        with multiprocessing.Pool(args.parallel) as pool:
            for res in tqdm(
                pool.imap_unordered(process_sample, sample_args),
                total=len(sample_args),
                desc="Samples",
            ):
                results.append(res)
    else:
        for idx, arg in enumerate(tqdm(sample_args, desc="Samples"), start=1):
            pid = arg[0]
            import time
            t0 = time.time()
            logging.info(f"[{idx}/{len(sample_args)}] START sample {pid}")
            hn, nm = process_sample(arg)
            t1 = time.time()
            results.append((hn, nm))
            # Log geometry/feature info if available
            if hn:
                if not isinstance(hn, dict):
                    logging.error(f"Expected dict for hard negative, got {type(hn)}: {hn}")
                else:
                    verts = hn.get('geometry', [])
                    n_verts = len(verts)
                    logging.info(f"[{idx}/{len(sample_args)}] Sample {pid} post-processing: vertices={n_verts}, time={t1-t0:.2f}s")
            logging.info(f"[{idx}/{len(sample_args)}] DONE  sample {pid}")


    # Post-process outputs with batch-level diversity and hardness scoring
    hard_negatives, near_misses = [], []
    for hn, nm in results:
        if hn:
            if not isinstance(hn, dict):
                logging.error(f"Expected dict for hard negative, got {type(hn)}: {hn}")
                continue
            verts = hn.get('geometry', [])
            if has_min_vertices(verts, min_v=4):
                # de-duplication (L2 < 1e-3) and batch-level diversity (L2 > 0.05)
                if not any(l2_shape_distance(verts, e.get('geometry', [])) < 1e-3 for e in hard_negatives):
                    if is_diverse(hn, hard_negatives):
                        # Compute hardness: min L2 distance to any positive sample in the same problem
                        pid = hn.get('problem_id') or hn.get('pid')
                        # Find all positive samples for this problem
                        pos_verts = [entry.get('geometry', []) for entry in problems.get(pid, []) if is_positive_label(entry.get('label'))]
                        if pos_verts:
                            hardness = min(l2_shape_distance(verts, v) for v in pos_verts)
                        else:
                            hardness = 0.0
                        hn['hardness'] = float(hardness)
                        hard_negatives.append(hn)
        if nm:
            near_misses.append(nm)

    # Sort hard negatives by hardness descending (optional, for batch mixing)
    hard_negatives.sort(key=lambda x: -x.get('hardness', 0.0))

    # (Optional) Log batch-level diversity statistics
    if len(hard_negatives) > 1:
        l2s = [l2_shape_distance(hard_negatives[i]['geometry'], hard_negatives[j]['geometry'])
               for i in range(len(hard_negatives)) for j in range(i+1, len(hard_negatives))]
        logging.info(f"Batch-level diversity: mean L2={np.mean(l2s):.4f}, min L2={np.min(l2s):.4f}, max L2={np.max(l2s):.4f}")

    # Write files
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(hard_negatives, handle, indent=2)
    logging.info("Wrote %d hard negatives → %s", len(hard_negatives), args.output)

    if args.near_miss:
        nm_path = args.output.replace('.json', '_near_miss.json')
        with open(nm_path, 'w', encoding='utf-8') as handle:
            json.dump(near_misses, handle, indent=2)
        logging.info("Wrote %d near-misses → %s", len(near_misses), nm_path)


# --------------------------------------------------------------------------- #
# Script guard
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    main()
