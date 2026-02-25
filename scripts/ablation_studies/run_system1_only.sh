#!/bin/bash
# Ablation Study: System 1 Only (Neural Network)
# Tests performance without Bayesian System 2

echo "========================================="
echo "ABLATION STUDY: System 1 Only"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Mode: system1_only"
echo "  - Pretrain: YES (SSL rotation)"
echo "  - System 2: DISABLED"
echo "  - Expected: Lower accuracy on abstract splits"
echo ""

python run_experiment.py \
    --mode system1_only \
    --pretrain \
    --splits test_ff test_bd test_hd_comb test_hd_novel \
    --log-level INFO \
    --output-dir logs/ablation/system1_only

echo ""
echo "========================================="
echo "System 1 Only - Complete!"
echo "Results saved to: logs/ablation/system1_only/"
echo "========================================="
