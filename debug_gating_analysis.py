import argparse
import json
from statistics import mean

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver


def run(split: str, limit: int, threshold: float, use_programs: bool):
    loader = RawShapeBongardLoader(root_path="data/raw/ShapeBongard_V2")
    solver = HybridDualSystemSolver(
        use_programs=use_programs,
        arbitration_strategy="conflict_based",
        conflict_threshold=threshold,
    )

    episodes = list(loader.iter_split(split_name=split, query_index=6, shuffle=False, seed=0, max_episodes=limit))

    trigger_count = 0
    total_items = 0
    concept_correct = 0

    disagreements = []
    when_trigger_correct = []
    when_no_trigger_correct = []

    for ep in episodes:
        solver.fit_episode(ep)

        pos = solver.predict_item(ep.query_pos)
        neg = solver.predict_item(ep.query_neg)

        d_pos = abs(pos.p1 - pos.p2)
        d_neg = abs(neg.p1 - neg.p2)
        disagreements.extend([d_pos, d_neg])

        t_pos = d_pos > threshold
        t_neg = d_neg > threshold
        trigger_count += int(t_pos) + int(t_neg)
        total_items += 2

        concept_margin = pos.p_combined - neg.p_combined
        c_ok = 1 if concept_margin > 0 else 0
        concept_correct += c_ok

        # classify episode by whether any trigger happened
        if t_pos or t_neg:
            when_trigger_correct.append(c_ok)
        else:
            when_no_trigger_correct.append(c_ok)

        solver._update_reliability_weights(pos, neg)

    out = {
        "split": split,
        "episodes": len(episodes),
        "threshold": threshold,
        "use_programs": use_programs,
        "concept_accuracy": concept_correct / max(1, len(episodes)),
        "trigger_rate_per_item": trigger_count / max(1, total_items),
        "mean_disagreement": mean(disagreements) if disagreements else 0.0,
        "medianish_disagreement": sorted(disagreements)[len(disagreements)//2] if disagreements else 0.0,
        "acc_when_triggered": mean(when_trigger_correct) if when_trigger_correct else None,
        "acc_when_not_triggered": mean(when_no_trigger_correct) if when_no_trigger_correct else None,
        "count_triggered_episodes": len(when_trigger_correct),
        "count_not_triggered_episodes": len(when_no_trigger_correct),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test_ff")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--use-programs", action="store_true", default=True)
    args = parser.parse_args()

    res = run(args.split, args.limit, args.threshold, args.use_programs)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
