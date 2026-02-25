"""
Deep analysis of our dual-system implementation to understand WHY it's lagging.

This script analyzes:
1. System 1: Confidence vs accuracy relationship (calibration)
2. System 2: Rule quality and discriminative power
3. Arbitration: Weight update dynamics
4. Episode-level behavior patterns
"""

import sys
import numpy as np
import json
from collections import defaultdict
sys.path.insert(0, ".")

from bayes_dual_system.data import RawShapeBongardLoader
from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver

def analyze_system_behavior(num_episodes=50, use_programs=True):
    """Comprehensive analysis of dual-system behavior."""
    
    loader = RawShapeBongardLoader("data/raw/ShapeBongard_V2")
    solver = HybridDualSystemSolver(
        image_size=64,
        use_programs=use_programs,
        seed=42,
    )
    
    # Storage for analysis
    data = {
        "system1": {
            "confidences": [],
            "accuracies": [],
            "predictions": [],
        },
        "system2": {
            "rule_counts": [],
            "rule_weights": [],
            "max_weights": [],
            "mean_weights": [],
            "predictions": [],
            "accuracies": [],
        },
        "arbitration": {
            "w1_trajectory": [],
            "w2_trajectory": [],
            "s1_corrects": [],
            "s2_corrects": [],
            "disagreements": [],
            "usage_s2": [],
        },
        "episodes": {
            "task_ids": [],
            "concept_correct": [],
            "margins": [],
        }
    }
    
    # Get episodes
    split = "test_ff"
    task_ids = loader.splits[split][:num_episodes]
    
    print("="*80)
    print(f"DEEP ANALYSIS: {num_episodes} episodes from {split}")
    print(f"Program features: {use_programs}")
    print("="*80)
    print()
    
    for idx, task_id in enumerate(task_ids):
        episode = loader.build_episode(task_id, query_index=6)
        
        # Evaluate
        metrics = solver.evaluate_episode(episode)
        
        # Extract System 1 data
        s1_conf_pos = metrics['query_pos_trace'].get('system1_confidence', 0)
        s1_conf_neg = metrics['query_neg_trace'].get('system1_confidence', 0)
        avg_s1_conf = (s1_conf_pos + s1_conf_neg) / 2
        
        s1_pred_pos = metrics['query_pos_p1']
        s1_pred_neg = metrics['query_neg_p1']
        s1_margin = s1_pred_pos - s1_pred_neg
        s1_correct = 1 if s1_margin > 0 else 0
        
        data['system1']['confidences'].append(avg_s1_conf)
        data['system1']['accuracies'].append(s1_correct)
        data['system1']['predictions'].append(s1_margin)
        
        # Extract System 2 data
        s2_rule_count = metrics['query_pos_trace']['system2_rule_count']
        
        # Get rule weights from solver
        if solver.system2._rules:
            rule_weights = [r.weight for r in solver.system2._rules]
            data['system2']['rule_weights'].extend(rule_weights)
            data['system2']['max_weights'].append(max(rule_weights))
            data['system2']['mean_weights'].append(np.mean(rule_weights))
        else:
            data['system2']['max_weights'].append(0)
            data['system2']['mean_weights'].append(0)
        
        data['system2']['rule_counts'].append(s2_rule_count)
        
        s2_pred_pos = metrics['query_pos_p2']
        s2_pred_neg = metrics['query_neg_p2']
        s2_margin = s2_pred_pos - s2_pred_neg
        s2_correct = 1 if s2_margin > 0 else 0
        
        data['system2']['predictions'].append(s2_margin)
        data['system2']['accuracies'].append(s2_correct)
        
        # Extract arbitration data
        weight_update = metrics['weight_update']
        data['arbitration']['w1_trajectory'].append(metrics['system1_weight_final'])
        data['arbitration']['w2_trajectory'].append(metrics['system2_weight_final'])
        data['arbitration']['s1_corrects'].append(1 if weight_update['s1_correct'] else 0)
        data['arbitration']['s2_corrects'].append(1 if weight_update['s2_correct'] else 0)
        data['arbitration']['disagreements'].append(weight_update['disagreement'])
        data['arbitration']['usage_s2'].append(metrics['used_system2_pos'])
        
        # Episode-level
        data['episodes']['task_ids'].append(task_id)
        data['episodes']['concept_correct'].append(metrics['concept_correct'])
        data['episodes']['margins'].append(metrics['concept_margin'])
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{num_episodes} episodes...")
    
    return data


def print_analysis(data):
    """Print comprehensive analysis."""
    
    print("\n" + "="*80)
    print("SYSTEM 1 ANALYSIS (CNN Perception)")
    print("="*80)
    
    s1_acc = np.mean(data['system1']['accuracies'])
    s1_conf = np.mean(data['system1']['confidences'])
    
    print(f"Overall Accuracy: {s1_acc:.1%}")
    print(f"Average Confidence: {s1_conf:.4f}")
    print(f"Confidence/Accuracy Ratio: {s1_conf/max(s1_acc, 0.01):.2f}x")
    
    # Calibration analysis
    high_conf = [acc for conf, acc in zip(data['system1']['confidences'], data['system1']['accuracies']) if conf > 0.8]
    if high_conf:
        print(f"Accuracy when confidence > 0.8: {np.mean(high_conf):.1%} (n={len(high_conf)})")
    
    low_conf = [acc for conf, acc in zip(data['system1']['confidences'], data['system1']['accuracies']) if conf < 0.5]
    if low_conf:
        print(f"Accuracy when confidence < 0.5: {np.mean(low_conf):.1%} (n={len(low_conf)})")
    
    print(f"\n❌ PROBLEM: System 1 is {'OVERCONFIDENT' if s1_conf/max(s1_acc, 0.01) > 1.5 else 'miscalibrated'}")
    print(f"   → Trained on only 12 examples per episode → overfitting")
    print(f"   → High confidence ({s1_conf:.1%}) but low accuracy ({s1_acc:.1%})")
    
    print("\n" + "="*80)
    print("SYSTEM 2 ANALYSIS (Bayesian Reasoning)")
    print("="*80)
    
    s2_acc = np.mean(data['system2']['accuracies'])
    avg_rules = np.mean(data['system2']['rule_counts'])
    avg_max_weight = np.mean(data['system2']['max_weights'])
    avg_mean_weight = np.mean(data['system2']['mean_weights'])
    
    print(f"Overall Accuracy: {s2_acc:.1%}")
    print(f"Average Rules Found: {avg_rules:.1f}")
    print(f"Average Max Rule Weight: {avg_max_weight:.4f}")
    print(f"Average Mean Rule Weight: {avg_mean_weight:.4f}")
    
    # Weight distribution
    if data['system2']['rule_weights']:
        all_weights = np.array(data['system2']['rule_weights'])
        print(f"\nRule Weight Distribution:")
        print(f"  Min: {all_weights.min():.4f}")
        print(f"  25%: {np.percentile(all_weights, 25):.4f}")
        print(f"  50%: {np.percentile(all_weights, 50):.4f}")
        print(f"  75%: {np.percentile(all_weights, 75):.4f}")
        print(f"  Max: {all_weights.max():.4f}")
    
    print(f"\n❌ PROBLEM: Rule weights are very small (~{avg_max_weight:.4f})")
    print(f"   → Threshold rules on statistical features have low discriminative power")
    print(f"   → Cannot express sequential patterns in LOGO programs")
    
    print("\n" + "="*80)
    print("ARBITRATION ANALYSIS (Weight Dynamics)")
    print("="*80)
    
    final_w1 = np.mean(data['arbitration']['w1_trajectory'])
    final_w2 = np.mean(data['arbitration']['w2_trajectory'])
    s1_correct_rate = np.mean(data['arbitration']['s1_corrects'])
    s2_correct_rate = np.mean(data['arbitration']['s2_corrects'])
    avg_disagreement = np.mean(data['arbitration']['disagreements'])
    s2_usage = np.mean(data['arbitration']['usage_s2'])
    
    print(f"Final Average Weights: w1={final_w1:.4f}, w2={final_w2:.4f}")
    print(f"System 1 Correctness Rate: {s1_correct_rate:.1%}")
    print(f"System 2 Correctness Rate: {s2_correct_rate:.1%}")
    print(f"Average Disagreement: {avg_disagreement:.4f}")
    print(f"System 2 Usage Rate: {s2_usage:.1%}")
    
    # Analyze weight trajectory
    w1_traj = np.array(data['arbitration']['w1_trajectory'])
    w2_traj = np.array(data['arbitration']['w2_trajectory'])
    
    print(f"\nWeight Trajectory:")
    print(f"  w1: {w1_traj[0]:.4f} → {w1_traj[-1]:.4f} (Δ={w1_traj[-1]-w1_traj[0]:+.4f})")
    print(f"  w2: {w2_traj[0]:.4f} → {w2_traj[-1]:.4f} (Δ={w2_traj[-1]-w2_traj[0]:+.4f})")
    
    if s2_correct_rate > s1_correct_rate and final_w1 > final_w2:
        print(f"\n❌ PROBLEM: S2 more accurate but S1 gets higher weight!")
        print(f"   → S2 is {s2_correct_rate:.1%} correct vs S1 {s1_correct_rate:.1%}")
        print(f"   → But w1={final_w1:.2f} > w2={final_w2:.2f}")
        print(f"   → Weight update mechanism may be broken")
    
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    overall_acc = np.mean(data['episodes']['concept_correct'])
    avg_margin = np.mean(data['episodes']['margins'])
    
    print(f"Concept Accuracy: {overall_acc:.1%}")
    print(f"Average Margin: {avg_margin:.4f}")
    
    # Correlation analysis
    s1_vs_s2_corr = np.corrcoef(data['system1']['predictions'], data['system2']['predictions'])[0, 1]
    print(f"\nSystem 1 vs System 2 prediction correlation: {s1_vs_s2_corr:.3f}")
    
    if abs(s1_vs_s2_corr) > 0.7:
        print(f"⚠️  WARNING: Systems are too correlated (redundant)")
    elif abs(s1_vs_s2_corr) < 0.3:
        print(f"✓ Systems are complementary (good diversity)")
    
    return {
        's1_accuracy': s1_acc,
        's1_confidence': s1_conf,
        's2_accuracy': s2_acc,
        'avg_rule_weight': avg_max_weight,
        'final_w1': final_w1,
        'final_w2': final_w2,
        'overall_accuracy': overall_acc,
        's1_vs_s2_correlation': s1_vs_s2_corr,
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEEP ANALYSIS: Understanding Our Dual-System Architecture")
    print("="*80)
    
    # Analysis with program features
    print("\n>>> Running analysis with PROGRAM features...")
    data_prog = analyze_system_behavior(num_episodes=50, use_programs=True)
    results_prog = print_analysis(data_prog)
    
    print("\n\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Analysis without program features
    print("\n>>> Running analysis with IMAGE features...")
    data_img = analyze_system_behavior(num_episodes=50, use_programs=False)
    results_img = print_analysis(data_img)
    
    # Compare
    print("\n" + "="*80)
    print("PROGRAM vs IMAGE FEATURES")
    print("="*80)
    
    print(f"\nSystem 1 (CNN) - identical in both:")
    print(f"  Accuracy: {results_prog['s1_accuracy']:.1%}")
    
    print(f"\nSystem 2 (Bayesian):")
    print(f"  Accuracy with PROGRAM features: {results_prog['s2_accuracy']:.1%}")
    print(f"  Accuracy with IMAGE features:   {results_img['s2_accuracy']:.1%}")
    print(f"  Δ Accuracy: {(results_prog['s2_accuracy']-results_img['s2_accuracy'])*100:+.1f}%")
    
    print(f"\nRule Quality:")
    print(f"  Avg max weight with PROGRAM: {results_prog['avg_rule_weight']:.4f}")
    print(f"  Avg max weight with IMAGE:   {results_img['avg_rule_weight']:.4f}")
    
    print(f"\nOverall Performance:")
    print(f"  With PROGRAM features: {results_prog['overall_accuracy']:.1%}")
    print(f"  With IMAGE features:   {results_img['overall_accuracy']:.1%}")
    print(f"  Δ Accuracy: {(results_prog['overall_accuracy']-results_img['overall_accuracy'])*100:+.1f}%")
    
    # Save detailed results
    with open("logs/deep_analysis_results.json", "w") as f:
        json.dump({
            "program_features": results_prog,
            "image_features": results_img,
            "raw_data_program": {k: {k2: [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v2] 
                                 for k2, v2 in v.items()} for k, v in data_prog.items()},
            "raw_data_image": {k: {k2: [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v2] 
                               for k2, v2 in v.items()} for k, v in data_img.items()},
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: logs/deep_analysis_results.json")
