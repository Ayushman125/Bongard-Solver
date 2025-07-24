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
        return hard_negative, None
    except Exception as exc:  # noqa: BLE001
        logging.error(f"process_sample failure for {pid}: {exc}", exc_info=True)
        return None, None


# --------------------------------------------------------------------------- #
# CLI entry
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Generate hard negatives using evolutionary and grammar-based "
        "mutation."
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
    args = parser.parse_args()

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

    # Build argument tuples
    sample_args = []
    for pid, entries in problems.items():
        positives = [e for e in entries if is_positive_label(e.get('label'))]
        if not positives:
            continue
        concept_fn = get_concept_fn_for_problem(pid)
        for sample in positives:
            sample_args.append((pid, sample, concept_fn, args))

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
            logging.info(f"[{idx}/{len(sample_args)}] START sample {pid}")
            hn, nm = process_sample(arg)
            results.append((hn, nm))
            logging.info(f"[{idx}/{len(sample_args)}] DONE  sample {pid}")

    # Post-process outputs
    hard_negatives, near_misses = [], []
    for hn, nm in results:
        if hn:
            verts = hn.get('geometry', [])
            if has_min_vertices(verts, min_v=4):
                # de-duplication
                if not any(
                    l2_shape_distance(verts, e.get('geometry', [])) < 1e-3
                    for e in hard_negatives
                ):
                    hard_negatives.append(hn)
        if nm:
            near_misses.append(nm)

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
