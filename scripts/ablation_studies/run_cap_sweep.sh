#!/bin/bash
# Ablation Study: Weight Cap Sweep
# Tests effect of different max_system2_weight values

echo "========================================="
echo "ABLATION STUDY: Weight Cap Sweep"
echo "========================================="
echo ""
echo "Testing caps: [No Cap, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]"
echo "Goal: Demonstrate weight saturation issue and optimal cap"
echo ""

# Array of cap values to test
CAPS=(0.999 0.90 0.93 0.95 0.97 0.98 0.99)
CAP_LABELS=("no_cap" "0.90" "0.93" "0.95" "0.97" "0.98" "0.99")

for i in "${!CAPS[@]}"; do
    CAP=${CAPS[$i]}
    LABEL=${CAP_LABELS[$i]}
    
    echo ""
    echo "-----------------------------------------"
    echo "Testing cap = $LABEL"
    echo "-----------------------------------------"
    
    python run_experiment.py \
        --mode hybrid_image \
        --pretrain \
        --max-system2-weight $CAP \
        --splits test_ff test_bd test_hd_comb test_hd_novel \
        --use-programs \
        --arbitration-strategy always_blend \
        --log-level INFO \
        --output-dir logs/ablation/cap_$LABEL
    
    echo "Cap $LABEL complete!"
done

echo ""
echo "========================================="
echo "Weight Cap Sweep - All Tests Complete!"
echo "========================================="
echo ""
echo "Results Summary:"
echo "  - No cap (0.999): Expected S2 saturation, S1 irrelevant"
echo "  - Cap 0.90: More S1 influence, may hurt S2-dominant tasks"
echo "  - Cap 0.95: Optimal balance (our main result)"
echo "  - Cap 0.99: Minimal constraint, still saturates"
echo ""
echo "Analyze results in: logs/ablation/cap_*/"
echo "========================================="
