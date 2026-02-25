#!/bin/bash
# Ablation Study: No Pretraining
# Tests importance of SSL pretraining for System 1

echo "========================================="
echo "ABLATION STUDY: No Pretraining"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Mode: hybrid_image"
echo "  - Pretrain: NO (train from scratch)"
echo "  - Expected: Significantly lower accuracy, especially on free-form shapes"
echo ""

python run_experiment.py \
    --mode hybrid_image \
    --no-pretrain \
    --auto-cap \
    --cap-grid 0.95,0.97,0.98,0.99 \
    --cap-split val \
    --cap-limit 200 \
    --splits test_ff test_bd test_hd_comb test_hd_novel \
    --use-programs \
    --arbitration-strategy always_blend \
    --log-level INFO \
    --output-dir logs/ablation/no_pretrain

echo ""
echo "========================================="
echo "No Pretraining - Complete!"
echo "Results saved to: logs/ablation/no_pretrain/"
echo "========================================="
