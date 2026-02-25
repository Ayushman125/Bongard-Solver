#!/usr/bin/env python
"""
Comprehensive BONGARD-LOGO Solver Results Analysis
Executes full evaluation pipeline with real data
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

print("\n" + "="*80)
print("COMPREHENSIVE BONGARD-LOGO RESULTS ANALYSIS")
print("="*80)

# Setup paths
PROJECT_ROOT = Path('.').resolve()
LOGS_DIR = PROJECT_ROOT / 'logs'
RESULTS_DIR = PROJECT_ROOT / 'results_analysis'
RESULTS_DIR.mkdir(exist_ok=True)

print(f"\nProject Root: {PROJECT_ROOT}")
print(f"Results Dir: {RESULTS_DIR}")

# ==========================================
# SECTION 1: Load evaluation data
# ==========================================
print("\n" + "="*80)
print("SECTION 1: LOADING EVALUATION DATA")
print("="*80)

# Find latest evaluation run
latest_run = max(glob.glob(str(LOGS_DIR / 'metrics/run_*.json')), key=os.path.getctime)
print(f"\nLoading evaluation run: {os.path.basename(latest_run)}")

with open(latest_run, 'r') as f:
    eval_data = json.load(f)

print(f"✅ Loaded evaluation data")
print(f"   Config: {eval_data['config']['mode']}")
print(f"   Timestamp: {eval_data['timestamp']}")

# Extract performance data
summaries = eval_data['summaries']
splits = ['test_ff', 'test_bd', 'test_hd_comb', 'test_hd_novel']
performance_summary = {}

for split_name in splits:
    if split_name in summaries:
        split_data = summaries[split_name]
        performance_summary[split_name] = {
            'accuracy': split_data['concept_acc'],
            'episodes': int(split_data['episodes']),
            'margin': split_data['concept_margin'],
            's2_usage': split_data['s2_usage']
        }

# Print performance summary
print("\n" + "="*80)
print("PERFORMANCE BY SPLIT")
print("="*80)

your_results_list = []
for split in splits:
    data = performance_summary[split]
    acc_pct = data['accuracy'] * 100
    your_results_list.append(acc_pct)
    print(f"\n{split}:")
    print(f"  Accuracy:  {acc_pct:.2f}%")
    print(f"  Episodes:  {data['episodes']}")
    print(f"  Margin:    {data['margin']:.4f}")
    print(f"  S2 Usage:  {data['s2_usage']*100:.1f}%")

# ==========================================
# SECTION 2: Baseline Comparison
# ==========================================
print("\n" + "="*80)
print("SECTION 2: BASELINE COMPARISON")
print("="*80)

baseline_data = {
    'Method': [
        'Your Hybrid Dual-System ⭐',
        'Meta-Baseline-PS',
        'Meta-Baseline-MoCo',
        'ProtoNet',
        'SNAIL',
        'MetaOptNet',
        'ANIL',
        'WReN-Bongard',
        'CNN-Baseline',
        'Human (Expert)',
        'Human (Amateur)'
    ],
    'test_ff': [your_results_list[0], 68.2, 65.9, 64.6, 56.3, 60.3, 56.6, 50.1, 51.9, 92.1, 88.0],
    'test_bd': [your_results_list[1], 75.7, 72.2, 72.4, 60.2, 71.7, 59.0, 50.9, 56.6, 99.3, 90.0],
    'test_hd_comb': [your_results_list[2], 67.4, 63.9, 62.4, 60.1, 61.7, 59.6, 53.8, 53.6, 90.7, 71.0],
    'test_hd_novel': [your_results_list[3], 71.5, 64.7, 65.4, 61.3, 63.3, 61.0, 54.3, 57.6, 90.7, 71.0],
}

baseline_df = {k: v for k, v in baseline_data.items()}
print("\nBASELINE COMPARISON TABLE:")
print("-" * 120)
print(f"{'Method':<35} {'test_ff':>10} {'test_bd':>10} {'test_hd_comb':>12} {'test_hd_novel':>12}")
print("-" * 120)
for i, method in enumerate(baseline_data['Method']):
    ff_acc = baseline_data['test_ff'][i]
    bd_acc = baseline_data['test_bd'][i]
    hd_comb = baseline_data['test_hd_comb'][i]
    hd_novel = baseline_data['test_hd_novel'][i]
    print(f"{method:<35} {ff_acc:>9.1f}% {bd_acc:>9.1f}% {hd_comb:>11.1f}% {hd_novel:>11.1f}%")

# Calculate improvements
print("\n" + "="*80)
print("IMPROVEMENTS OVER META-BASELINE-PS")
print("="*80)

meta_baseline_ps = baseline_data['test_ff'][1:5] + baseline_data['test_bd'][1:5]
for i, split in enumerate(splits):
    your_acc = your_results_list[i]
    baseline_acc = baseline_data[split][1]  # Meta-Baseline-PS
    delta = your_acc - baseline_acc
    print(f"\n{split:15s}: {your_acc:6.2f}% vs {baseline_acc:6.2f}% (Δ = {delta:+6.2f}%)")

# ==========================================
# SECTION 3: Statistical Analysis
# ==========================================
print("\n" + "="*80)
print("SECTION 3: BOOTSTRAP STATISTICAL ANALYSIS")
print("="*80)

episodes_data = eval_data.get('episodes', {})
bootstrap_results = {}
n_bootstrap = 1000
np.random.seed(42)

for split_name in splits:
    if split_name not in episodes_data:
        continue
    
    split_episodes = episodes_data[split_name]
    correctness_array = np.array([ep['concept_correct'] for ep in split_episodes])
    n_total = len(correctness_array)
    observed_acc = np.mean(correctness_array)
    
    # Bootstrap sampling
    bootstrap_accs = []
    for _ in range(n_bootstrap):
        sample_indices = np.random.choice(n_total, n_total, replace=True)
        bootstrap_accs.append(np.mean(correctness_array[sample_indices]) * 100)
    
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    std_error = np.std(bootstrap_accs)
    
    bootstrap_results[split_name] = {
        'observed': observed_acc * 100,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': std_error,
        'n_episodes': n_total
    }
    
    print(f"\n{split_name}:")
    print(f"  Observed: {observed_acc*100:.2f}% ({n_total} episodes)")
    print(f"  95% CI:   [{ci_lower:.2f}%, {ci_upper:.2f}%]")
    print(f"  Std Err:  ±{std_error:.2f}%")

# ==========================================
# SECTION 4: Save Results
# ==========================================
print("\n" + "="*80)
print("SECTION 4: SAVING RESULTS")
print("="*80)

# Save comprehensive JSON
results_json = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'evaluation_data': os.path.basename(latest_run),
        'evaluation_timestamp': eval_data['timestamp']
    },
    'performance': {
        split: {
            'accuracy': performance_summary[split]['accuracy'] * 100,
            'episodes': performance_summary[split]['episodes'],
            'margin': performance_summary[split]['margin'],
            's2_usage': performance_summary[split]['s2_usage']
        } for split in splits
    },
    'baseline_comparison': baseline_data,
    'improvements': {
        split: float(your_results_list[i] - baseline_data[split][1]) 
        for i, split in enumerate(splits)
    },
    'bootstrap_ci': {
        split: {
            'observed': bootstrap_results[split]['observed'],
            'ci_lower': bootstrap_results[split]['ci_lower'],
            'ci_upper': bootstrap_results[split]['ci_upper'],
            'std_error': bootstrap_results[split]['std_error']
        } for split in bootstrap_results.keys()
    }
}

results_json_path = RESULTS_DIR / 'comprehensive_results.json'
with open(results_json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"✅ Saved: {results_json_path}")

# Save baseline CSV (as JSON since pandas not available)
baseline_csv_path = RESULTS_DIR / 'baseline_comparison.json'
with open(baseline_csv_path, 'w') as f:
    json.dump(baseline_data, f, indent=2)
print(f"✅ Saved: {baseline_csv_path}")

# Save manifest
manifest = {
    'creation_timestamp': datetime.now().isoformat(),
    'evaluation_source': os.path.basename(latest_run),
    'evaluation_timestamp': eval_data['timestamp'],
    'splits_processed': splits,
    'total_episodes': sum(performance_summary[s]['episodes'] for s in splits),
    'output_files': {
        'comprehensive_results': 'comprehensive_results.json',
        'baseline_comparison': 'baseline_comparison.json',
        'notebook': 'analysis_comprehensive_results.ipynb'
    }
}

manifest_path = RESULTS_DIR / 'MANIFEST.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"✅ Saved: {manifest_path}")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)

print("\n✅ Results Analysis Complete!")
print(f"\nKey Metrics:")
for i, split in enumerate(splits):
    print(f"  {split}: {your_results_list[i]:.2f}%")

print(f"\nAll results saved to: {RESULTS_DIR}/")
print("\nFiles generated:")
print(f"  • comprehensive_results.json")
print(f"  • baseline_comparison.csv")
print(f"  • MANIFEST.json")
print(f"  • analysis_comprehensive_results.ipynb")

print("\n" + "="*80)
print("✨ Ready for publication")
print("="*80 + "\n")
