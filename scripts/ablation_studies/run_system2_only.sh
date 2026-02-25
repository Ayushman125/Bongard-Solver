#!/bin/bash
# Ablation Study: System 2 Only (Bayesian Rule Induction)
# Tests performance without Neural Network System 1

echo "========================================="
echo "ABLATION STUDY: System 2 Only"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Mode: system2_only"
echo "  - Pretrain: N/A (no neural component)"
echo "  - System 1: DISABLED"
echo "  - Expected: Good on basic shapes, worse on complex visual patterns"
echo ""

python run_experiment.py \
    --mode system2_only \
    --splits test_ff test_bd test_hd_comb test_hd_novel \
    --log-level INFO \
    --output-dir logs/ablation/system2_only

echo ""
echo "========================================="
echo "System 2 Only - Complete!"
echo "Results saved to: logs/ablation/system2_only/"
echo "========================================="
