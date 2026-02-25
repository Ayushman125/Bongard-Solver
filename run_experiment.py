import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system import DualSystemBayesianSolver
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dual-system Bayesian Bongard solver")
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid_image",
        choices=["symbolic", "hybrid_image"],
        help="symbolic: program-based S1+S2, hybrid_image: neural image S1 + Bayesian image S2",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw/ShapeBongard_V2",
        help="Path to ShapeBongard_V2 root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test_ff", "test_bd", "test_hd_comb", "test_hd_novel"],
        help="Split names in ShapeBongard_V2_split.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional max tasks per split")
    parser.add_argument("--query-index", type=int, default=6, help="Index for query image in each class (0-6)")
    parser.add_argument("--threshold", type=float, default=0.35, help="System1 confidence threshold")
    parser.add_argument("--learning-rate", type=float, default=2.0, help="Arbitration weight update rate")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--max-rule-len", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--nn-epochs", type=int, default=60)
    parser.add_argument("--nn-lr", type=float, default=1e-3)
    parser.add_argument("--nn-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="logs/metrics",
        help="Directory where run metrics JSON/CSV files are saved",
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Disable automatic metrics file output",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-episode calculations")
    parser.add_argument(
        "--log-limit",
        type=int,
        default=2,
        help="Max episodes per split to print in verbose mode",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for progress and debug output",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/progress.log",
        help="File to save detailed logs (DEBUG level)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Pre-train CNN backbone on train split before evaluation",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=30,
        help="Epochs for self-supervised backbone pretraining",
    )
    parser.add_argument(
        "--pretrain-batch-size",
        type=int,
        default=256,
        help="Batch size for self-supervised backbone pretraining",
    )
    parser.add_argument(
        "--pretrain-workers",
        type=int,
        default=4,
        help="DataLoader workers for self-supervised pretraining",
    )
    parser.add_argument(
        "--use-programs",
        action="store_true",
        default=True,
        help="Use LOGO program features in System 2 (symbolic features from stroke sequences)",
    )
    parser.add_argument(
        "--no-use-programs",
        dest="use_programs",
        action="store_false",
        help="Disable program features, use only pixel-level image features",
    )
    parser.add_argument(
        "--arbitration-strategy",
        type=str,
        default="always_blend",
        choices=["always_blend", "conflict_based"],
        help="Arbitration strategy: always_blend (default) or conflict_based (Improvement #3)",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=float,
        default=0.3,
        help="Disagreement threshold for conflict-based gating (|p1-p2| > threshold → use S2)",
    )
    parser.add_argument(
        "--max-system2-weight",
        type=float,
        default=0.995,
        help="Cap System 2 weight to avoid total saturation (range: 0.5-0.999)",
    )
    parser.add_argument(
        "--auto-cap",
        action="store_true",
        help="Auto-tune max_system2_weight on a validation split before test evaluation",
    )
    parser.add_argument(
        "--auto-cap-per-split",
        action="store_true",
        help="Auto-tune max_system2_weight per test split using filtered validation episodes",
    )
    parser.add_argument(
        "--cap-grid",
        type=str,
        default="0.95,0.97,0.98,0.99",
        help="Comma-separated cap candidates for auto-cap tuning",
    )
    parser.add_argument(
        "--cap-split",
        type=str,
        default="val",
        help="Split to use for auto-cap tuning (default: val)",
    )
    parser.add_argument(
        "--cap-limit",
        type=int,
        default=200,
        help="Max episodes to evaluate per cap during auto-cap tuning",
    )
    return parser.parse_args()


def _setup_logging(log_level: str, log_file: str) -> None:
    """Configure logging to console and file."""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    
    level = getattr(logging, log_level.upper())
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    
    # File handler (DEBUG and above)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(funcName)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured: level={log_level}, file={os.path.abspath(log_file)}")


def main() -> None:
    args = parse_args()
    
    # Setup logging
    _setup_logging(args.log_level, args.log_file)
    
    logger.info("")
    logger.info("="*60)
    logger.info("BONGARD DUAL-SYSTEM SOLVER - EXPERIMENT RUN")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Episodes per split: {args.limit if args.limit else 'ALL'}")
    logger.info(f"Device: CUDA" if torch.cuda.is_available() else f"Device: CPU")
    logger.info("")

    loader = RawShapeBongardLoader(root_path=args.data_root)
    if args.mode == "symbolic":
        solver = DualSystemBayesianSolver(
            system1_confidence_threshold=args.threshold,
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.epsilon,
            length_penalty=args.length_penalty,
            max_rule_len=args.max_rule_len,
        )
    else:
        # Pre-train if requested
        pretrained_model = None
        if args.pretrain:
            logger.info("")
            logger.info("="*60)
            logger.info("PRE-TRAINING CNN BACKBONE")
            logger.info("="*60)
            logger.info(
                f"Objective: ssl_rotation | epochs={args.pretrain_epochs}, "
                f"batch_size={args.pretrain_batch_size}, workers={args.pretrain_workers}"
            )
            
            # Load train split images
            logger.info("Loading train split images...")
            train_episodes = loader.iter_split(
                split_name="train",
                query_index=args.query_index,
                shuffle=False,
                seed=args.seed,
                max_episodes=None,
            )
            
            pos_paths, neg_paths = [], []
            for ep_idx, episode in enumerate(train_episodes):
                if (ep_idx + 1) % 1000 == 0:
                    logger.info(f"  Loading train episodes: {ep_idx+1}")
                for item in episode.support_pos:
                    pos_paths.append(item.image_path)
                for item in episode.support_neg:
                    neg_paths.append(item.image_path)
            
            logger.info(f"Loaded {len(pos_paths)} pos + {len(neg_paths)} neg images from train split")
            
            # Pre-train
            from bayes_dual_system.system1_nn import NeuralSystem1, TrainConfig
            config = TrainConfig(
                image_size=args.image_size,
                epochs=args.nn_epochs,
                lr=args.nn_lr,
                weight_decay=args.nn_weight_decay,
                seed=args.seed,
                pretrain_epochs=args.pretrain_epochs,
                pretrain_batch_size=args.pretrain_batch_size,
                pretrain_workers=args.pretrain_workers,
            )
            pretrained_model = NeuralSystem1.pretrain_backbone(config, pos_paths, neg_paths)
            logger.info("="*60)
            logger.info("")
        
        if args.auto_cap and args.auto_cap_per_split:
            logger.warning("Both --auto-cap and --auto-cap-per-split set; using per-split tuning")

        if args.auto_cap and not args.auto_cap_per_split:
            cap_grid = _parse_float_list(args.cap_grid)
            logger.info(
                f"Auto-cap tuning on split '{args.cap_split}' with {len(cap_grid)} candidates"
            )
            best_cap, best_acc = None, -1.0
            for cap in cap_grid:
                summary = _evaluate_cap(
                    loader=loader,
                    args=args,
                    pretrained_model=pretrained_model,
                    cap=cap,
                )
                acc = summary.get("concept_acc", 0.0)
                logger.info(f"  cap={cap:.3f} -> acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_cap = cap
            if best_cap is not None:
                args.max_system2_weight = best_cap
                logger.info(
                    f"Selected max_system2_weight={best_cap:.3f} (val acc={best_acc:.4f})"
                )

        def build_solver(cap: float) -> HybridDualSystemSolver:
            return HybridDualSystemSolver(
                image_size=args.image_size,
                nn_epochs=args.nn_epochs,
                nn_lr=args.nn_lr,
                nn_weight_decay=args.nn_weight_decay,
                system1_confidence_threshold=args.threshold,
                reliability_lr=args.learning_rate,
                seed=args.seed,
                pretrained_model=pretrained_model,
                use_pretrained=args.pretrain,
                use_programs=args.use_programs,
                arbitration_strategy=args.arbitration_strategy,  # Improvement #3
                conflict_threshold=args.conflict_threshold,      # Improvement #3
                max_system2_weight=cap,
            )

        solver = build_solver(args.max_system2_weight)

    split_metrics = defaultdict(list)
    split_summaries: Dict[str, Dict[str, float]] = {}

    for split_idx, split in enumerate(args.splits):
        if args.auto_cap_per_split:
            cap_grid = _parse_float_list(args.cap_grid)
            category_filter = _split_category_filter(split)
            logger.info(
                f"Auto-cap (per-split) on '{args.cap_split}' for '{split}' with {len(cap_grid)} candidates"
            )
            best_cap, best_acc = None, -1.0
            for cap in cap_grid:
                summary = _evaluate_cap(
                    loader=loader,
                    args=args,
                    pretrained_model=pretrained_model,
                    cap=cap,
                    category_filter=category_filter,
                )
                acc = summary.get("concept_acc", 0.0)
                logger.info(f"  cap={cap:.3f} -> acc={acc:.4f}")
                if acc > best_acc:
                    best_acc = acc
                    best_cap = cap
            if best_cap is not None:
                logger.info(
                    f"Selected max_system2_weight={best_cap:.3f} for split '{split}' (val acc={best_acc:.4f})"
                )
                solver = build_solver(best_cap)
        logger.info(f"[{split_idx+1}/{len(args.splits)}] Processing split: {split}")
        episodes = loader.iter_split(
            split_name=split,
            query_index=args.query_index,
            shuffle=False,
            seed=args.seed,
            max_episodes=args.limit,
        )
        
        episodes_list = list(episodes)
        logger.info(f"  Total episodes to evaluate: {len(episodes_list)}")

        for ep_idx, episode in enumerate(episodes_list):
            if (ep_idx + 1) % max(1, len(episodes_list) // 10) == 0:
                progress_pct = (ep_idx + 1) / len(episodes_list) * 100
                logger.info(f"  Progress: {ep_idx+1}/{len(episodes_list)} ({progress_pct:.1f}%)")
            
            logger.debug(f"  Evaluating task {ep_idx+1}: {episode.task_id}")
            metrics = solver.evaluate_episode(episode)
            split_metrics[split].append(metrics)
            
            if args.verbose and ep_idx < args.log_limit:
                _print_episode_trace(split=split, task_id=episode.task_id, metrics=metrics, mode=args.mode)

    logger.info("")
    logger.info("="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    
    for split in args.splits:
        records = split_metrics[split]
        if not records:
            logger.info(f"{split}: no episodes")
            continue

        summary = _summarize_split(records)
        split_summaries[split] = summary

        logger.info(
            f"\n{split}:\n"
            f"  Episodes: {int(summary['episodes'])}\n"
            f"  Concept Accuracy: {summary['concept_acc']:.4f}\n"
            f"  Concept Margin (avg): {summary['concept_margin']:.4f}\n"
            f"  Tie Rate: {summary['concept_tie_rate']:.4f}\n"
            f"  System 2 Usage Rate: {summary['s2_usage']:.4f}"
        )

    if not args.no_save_metrics:
        json_path, csv_path = _write_run_metrics(
            metrics_dir=args.metrics_dir,
            args=args,
            split_summaries=split_summaries,
            split_metrics=split_metrics,
        )
        logger.info(f"\nSaved metrics JSON: {os.path.abspath(json_path)}")
        logger.info(f"Saved metrics CSV : {os.path.abspath(csv_path)}")
    
    logger.info(f"\nDetailed logs saved to: {os.path.abspath(args.log_file)}")
    logger.info("="*60)
    logger.info("RUN COMPLETE")
    logger.info("="*60)
    logger.info("")


def _parse_float_list(values: str) -> list:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _split_category_filter(split_name: str) -> Optional[str]:
    if split_name == "test_ff":
        return "ff"
    if split_name == "test_bd":
        return "bd"
    if split_name in {"test_hd_comb", "test_hd_novel"}:
        return "hd"
    return None


def _evaluate_cap(
    loader: RawShapeBongardLoader,
    args: argparse.Namespace,
    pretrained_model,
    cap: float,
    category_filter: Optional[str] = None,
) -> Dict[str, float]:
    solver = HybridDualSystemSolver(
        image_size=args.image_size,
        nn_epochs=args.nn_epochs,
        nn_lr=args.nn_lr,
        nn_weight_decay=args.nn_weight_decay,
        system1_confidence_threshold=args.threshold,
        reliability_lr=args.learning_rate,
        seed=args.seed,
        pretrained_model=pretrained_model,
        use_pretrained=args.pretrain,
        use_programs=args.use_programs,
        arbitration_strategy=args.arbitration_strategy,
        conflict_threshold=args.conflict_threshold,
        max_system2_weight=cap,
    )

    episodes = loader.iter_split(
        split_name=args.cap_split,
        query_index=args.query_index,
        shuffle=False,
        seed=args.seed,
        max_episodes=None,
    )

    records = []
    for episode in episodes:
        if category_filter and episode.category != category_filter:
            continue
        records.append(solver.evaluate_episode(episode))
        if len(records) >= args.cap_limit:
            break

    return _summarize_split(records)


def _print_episode_trace(split: str, task_id: str, metrics: Dict[str, object], mode: str) -> None:
    print(f"\n--- TRACE split={split} task={task_id} ---")

    concept_correct = int(metrics["concept_correct"]) == 1

    print(
        "POS query: "
        f"p1={float(metrics['query_pos_p1']):.4f}, "
        f"p2={float(metrics['query_pos_p2']):.4f}, "
        f"combined={float(metrics['query_pos_combined']):.4f}"
    )
    print(
        "NEG query: "
        f"p1={float(metrics['query_neg_p1']):.4f}, "
        f"p2={float(metrics['query_neg_p2']):.4f}, "
        f"combined={float(metrics['query_neg_combined']):.4f}"
    )

    print(
        "Concept decision: "
        f"margin=pos-neg={float(metrics['concept_margin']):.4f}, "
        f"correct={concept_correct}"
    )

    pos_trace = metrics["query_pos_trace"]
    neg_trace = metrics["query_neg_trace"]
    if mode == "symbolic":
        print(
            "Top system2 hypotheses (POS query): "
            f"pos={pos_trace['system2_top_positive_hypothesis']}@{float(pos_trace['system2_top_positive_weight']):.4f}, "
            f"neg={pos_trace['system2_top_negative_hypothesis']}@{float(pos_trace['system2_top_negative_weight']):.4f}"
        )
        print(
            "Top system2 hypotheses (NEG query): "
            f"pos={neg_trace['system2_top_positive_hypothesis']}@{float(neg_trace['system2_top_positive_weight']):.4f}, "
            f"neg={neg_trace['system2_top_negative_hypothesis']}@{float(neg_trace['system2_top_negative_weight']):.4f}"
        )
    else:
        pos_trace = metrics["query_pos_trace"]
        neg_trace = metrics["query_neg_trace"]
        feature_type = pos_trace.get("feature_type", "unknown")
        print(
            f"System2 Bayesian evidence (feature_type={feature_type}) (POS query): "
            f"logp_pos={float(pos_trace['system2_logp_pos']):.4f}, "
            f"logp_neg={float(pos_trace['system2_logp_neg']):.4f}, "
            f"margin={float(pos_trace['system2_logp_margin']):.4f}"
        )
        print(
            f"System2 Bayesian evidence (feature_type={feature_type}) (NEG query): "
            f"logp_pos={float(neg_trace['system2_logp_pos']):.4f}, "
            f"logp_neg={float(neg_trace['system2_logp_neg']):.4f}, "
            f"margin={float(neg_trace['system2_logp_margin']):.4f}"
        )

    if "weight_update" in metrics:
        update = metrics["weight_update"]
        print(
            "Reliability update (episode): "
            f"s1_correct={bool(update['s1_correct'])}, s2_correct={bool(update['s2_correct'])}, "
            f"disagreement={float(update['disagreement']):.4f}, "
            f"w1 {float(update['old_w1']):.4f}->{float(update['new_w1']):.4f}, "
            f"w2 {float(update['old_w2']):.4f}->{float(update['new_w2']):.4f}"
        )

    print(
        "Final arbitration weights: "
        f"w1={float(metrics['system1_weight_final']):.4f}, "
        f"w2={float(metrics['system2_weight_final']):.4f}"
    )


def _summarize_split(records) -> Dict[str, float]:
    n = len(records)
    concept_acc = sum(r["concept_correct"] for r in records) / n
    concept_margin = sum(r["concept_margin"] for r in records) / n
    concept_tie_rate = sum(r["concept_tie"] for r in records) / n
    s2_rate = sum((r["used_system2_pos"] + r["used_system2_neg"]) / 2.0 for r in records) / n
    return {
        "episodes": float(n),
        "concept_acc": concept_acc,
        "concept_margin": concept_margin,
        "concept_tie_rate": concept_tie_rate,
        "s2_usage": s2_rate,
    }


def _write_run_metrics(metrics_dir: str, args: argparse.Namespace, split_summaries, split_metrics):
    os.makedirs(metrics_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    json_path = os.path.join(metrics_dir, f"{base_name}.json")
    csv_path = os.path.join(metrics_dir, f"{base_name}.csv")

    payload = {
        "timestamp": timestamp,
        "config": vars(args),
        "summaries": split_summaries,
        "episodes": split_metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "episodes", "concept_acc", "concept_margin", "concept_tie_rate", "s2_usage"],
        )
        writer.writeheader()
        for split, summary in split_summaries.items():
            writer.writerow(
                {
                    "split": split,
                    "episodes": int(summary["episodes"]),
                    "concept_acc": f"{summary['concept_acc']:.6f}",
                    "concept_margin": f"{summary['concept_margin']:.6f}",
                    "concept_tie_rate": f"{summary['concept_tie_rate']:.6f}",
                    "s2_usage": f"{summary['s2_usage']:.6f}",
                }
            )

    return json_path, csv_path


if __name__ == "__main__":
    import torch  # Import torch here for device check in main
    main()
