"""
Test script for validating the three major improvements:
1. Meta-learning for S1 (pre-trained backbone)
2. Sequence matching in S2 (LCS + edit distance features)
3. Conflict-based gating (dynamic arbitration)

Usage:
    # Test with all improvements (after pre-training completes)
    python test_improvements.py --pretrain --arbitration-strategy conflict_based
    
    # Test sequence matching only
    python test_improvements.py --use-programs
    
    # Test conflict-based gating only
    python test_improvements.py --arbitration-strategy conflict_based
    
    # Baseline (like original results)
    python test_improvements.py --arbitration-strategy always_blend
"""

import subprocess
import sys
import time
from pathlib import Path

def run_evaluation(config_name: str, args: list[str], limit: int = 50):
    """Run evaluation with specific configuration."""
    print(f"\n{'='*70}")
    print(f"TESTING: {config_name}")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable,
        "run_experiment.py",
        "--mode", "hybrid_image",
        "--splits", "test_ff",  # Start with just one split for quick testing
        "--limit", str(limit),
        "--log-level", "INFO",
    ] + args
    
    print(f"Command: {' '.join(cmd)}")
    print(f"\nStarting evaluation...")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    
    # Extract accuracy from output
    for line in result.stdout.split('\n'):
        if 'test_ff:' in line or 'Concept Accuracy:' in line:
            print(line)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        print(result.stderr)
    
    return result


def main():
    """Run ablation study to measure improvement contributions."""
    
    # Check if pretrained model exists
    pretrain_checkpoint = Path("checkpoints/pretrain_backbone_ssl_rotation.pt")
    has_pretrained = pretrain_checkpoint.exists()
    
    if has_pretrained:
        print(f"✅ Found pre-trained model: {pretrain_checkpoint}")
    else:
        print(f"⚠️  No pre-trained model found. Will test without meta-learning (Improvement #1).")
        print(f"   Run with --pretrain to enable pre-training first.")
    
    # Configuration matrix
    configs = [
        {
            "name": "BASELINE (Fix 1 only)",
            "args": ["--arbitration-strategy", "always_blend", "--no-use-programs"],
            "expected": "~48% (without programs) or ~75% (with programs from Fix 1)",
        },
        {
            "name": "IMPROVEMENT #2: Sequence Matching",
            "args": ["--arbitration-strategy", "always_blend", "--use-programs"],
            "expected": "~77-79% (+2-4% from sequence features)",
        },
        {
            "name": "IMPROVEMENT #3: Conflict-Based Gating",
            "args": ["--arbitration-strategy", "conflict_based", "--use-programs", "--conflict-threshold", "0.3"],
            "expected": "~78-80% (+3-5% from better arbitration)",
        },
    ]
    
    # Add meta-learning config if pretrained model exists
    if has_pretrained:
        configs.append({
            "name": "IMPROVEMENT #1: Meta-Learning for S1",
            "args": ["--pretrain", "--arbitration-strategy", "always_blend", "--use-programs"],
            "expected": "~78-80% (+3-5% from better S1)",
        })
        configs.append({
            "name": "ALL IMPROVEMENTS COMBINED",
            "args": ["--pretrain", "--arbitration-strategy", "conflict_based", "--use-programs", "--conflict-threshold", "0.3"],
            "expected": "~82-85% (cumulative gains)",
        })
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: Testing {len(configs)} configurations")
    print(f"{'='*70}")
    print(f"Test split: test_ff (Free-Form shapes)")
    print(f"Episodes per config: 50 (quick validation)")
    print(f"")
    
    results = {}
    for config in configs:
        print(f"\n📊 Expected: {config['expected']}")
        result = run_evaluation(config["name"], config["args"], limit=50)
        results[config["name"]] = result
        time.sleep(2)  # Brief pause between runs
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for name in results:
        print(f"  {name}")
    
    print(f"\n✅ Ablation study complete!")
    print(f"\nNext steps:")
    print(f"  1. Review results above to verify improvements")
    print(f"  2. Run full evaluation: python run_experiment.py --pretrain --arbitration-strategy conflict_based --splits test_ff test_bd test_hd_comb test_hd_novel")
    print(f"  3. Compare with baseline (FINAL_RESULTS.md)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test improvements with ablation study")
    parser.add_argument("--limit", type=int, default=50, help="Episodes per test")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all splits)")
    args = parser.parse_args()
    
    if args.full:
        print("Running full evaluation on all test splits...")
        result = run_evaluation(
            "FULL EVALUATION (all improvements)",
            ["--pretrain", "--arbitration-strategy", "conflict_based", "--use-programs",
             "--splits", "test_ff", "test_bd", "test_hd_comb", "test_hd_novel"],
            limit=None
        )
    else:
        main()
