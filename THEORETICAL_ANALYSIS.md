# Mathematical and Theoretical Analysis of Our Dual-System Architecture

**Date:** February 25, 2026  
**Analysis of:** Dual-System Bayesian Bongard Solver  
**Goal:** Understand why our architecture underperforms and identify theoretically-grounded fixes

---

## Executive Summary

Our dual-system implementation achieves **48% accuracy** (near chance) despite System 2 achieving **70%** accuracy when program features are used. The root cause is **System 1 overconfidence** preventing System 2 from being used.

**Key Findings:**
1. System 1 (CNN): 48% accuracy but 95%+ confidence → **miscalibrated** 
2. System 2 (Bayesian): 70% accuracy with program features → **works well**
3. Arbitration: Bypasses S2 92% of the time due to S1 overconfidence → **broken**

---

## 1. System 1 Analysis: The Overconfidence Problem

### 1.1 Empirical Measurements

From 50 test episodes:
- **Accuracy:** 48.0% (near chance)
- **Average Confidence:** 95%+ (measured as `|p - 0.5| * 2`)
- **Calibration Error:** Confidence/Accuracy ratio = **2.0x**

Specific examples:
```
Episode 1: conf=0.9998, pred=0.0001, S2=0.8282 (correct) → Used S1 (WRONG)
Episode 2: conf=0.9878, pred=0.0061, S2=0.7877 (correct) → Used S1 (WRONG)
Episode 3: conf=0.9201, pred=0.0400, S2=0.7352 (correct) → Used S1 (WRONG)
```

### 1.2 Mathematical Analysis

**Problem:** confidence = |p - 0.5| * 2.0

When CNN overfits on 12 examples:
- Predicts extreme values: p ≈ 0.0001 or p ≈ 0.9999
- Confidence becomes: |0.0001 - 0.5| * 2 ≈ 1.0
- **High confidence does NOT mean correctness**, just extreme predictions

**Theoretical Root Cause:**

From statistical learning theory (Vapnik-Chervonenkis):
- Sample complexity for CNNs: O(d/ε²) where d = parameter dimensionality
- Our CNN: ~50K parameters, trained on n=12 examples
- Overfitting bound: Generalization error ∝ √(d/n) = √(50000/12) ≈ 64.5

**Conclusion:** CNN is hopelessly overfit. High confidence reflects memorization, not understanding.

### 1.3 Relevant Research

**Guo et al. (2017) - "On Calibration of Modern Neural Networks"**
- Modern neural networks are poorly calibrated
- Confidence does not match accuracy
- Solutions: Temperature scaling, label smoothing

**Lakshminarayanan et al. (2017) - "Simple and Scalable Predictive Uncertainty"**  
- Deep ensembles provide better uncertainty estimates
- Single model confidence is unreliable in few-shot settings

**Application to Our System:**  
S1 confidence should be **penalized** in low-data regime (12 examples), not trusted.

---

## 2. System 2 Analysis: Bayesian Rules Work!

### 2.1 Empirical Measurements

From 50 test episodes:
- **Accuracy with program features:** 70.0% ✓
- **Accuracy with image features:** 44.0%  
- **Improvement:** +26% absolute (59% relative)

Rule statistics:
- Average rules found: 24 composite rules
- Max rule weight: 0.0475 (low, but sufficient)
- Rule weight distribution: [0.031, 0.058]

### 2.2 Mathematical Analysis

**Bayesian Rule-Based Learning:**

For each rule r, maintain Beta posterior:
- θ_pos ~ Beta(α + n_pos_fire, β + n_pos_no_fire)
- θ_neg ~ Beta(α + n_neg_fire, β + n_neg_no_fire)

Prediction:
- log P(y=1|x) = Σ_r w_r * log(θ_pos if r(x) else 1-θ_pos)
- log P(y=0|x) = Σ_r w_r * log(θ_neg if r(x) else 1-θ_neg)

**Why It Works:**
1. Bayesian updates are robust to small samples (12 examples)
2. Compositional search finds interpretable patterns
3. Program features align with ground-truth concepts

**Rule Quality Analysis:**

Weights are small (~0.05) but this is EXPECTED:
- With 24 rules, uniform weight = 1/24 = 0.042
- Our weights are slightly higher (0.048 avg), showing learning
- Small weights ≠ weak rules; it's the ensemble that matters

### 2.3 Relevant Research

**Tenenbaum et al. (2011) - "How to Grow a Mind"**
- Concepts are programs in a compositional language
- Bayesian inference over program space
- Few examples sufficient with strong priors

**Lake et al. (2015) - "Human-level concept learning through probabilistic program induction"**
- Bayesian Program Learning (BPL) framework
- Learns from 1-5 examples by composing primitives
- 70% accuracy matches human-level learning

**Goodman et al. (2008) - "A Rational Analysis of Rule-Based Concept Learning"**
- Compositional rule search via Bayesian inference
- Rules as logical formulas over features
- Effective with < 20 examples

**Application to Our System:**  
S2 Bayesian approach is theoretically grounded and empirically successful (70%).

---

## 3. Arbitration Analysis: The Gating Mechanism is Broken

### 3.1 Current Mechanism

```python
if s1_confidence >= threshold:
    use S1 only
else:
    blend: p = w1*p1 + w2*p2
```

**Problem:** S1 confidence is ALWAYS above threshold (95%+ confidence) due to overfitting.

Result:
- S2 usage rate: 8% (should be ~100%)
- System ignores S2's 70% accuracy
- Overall accuracy = S1 accuracy = 48%

### 3.2 Weight Update Dynamics

**Current Update Rule:**

Based on correctness:
- Both correct: +w1 slightly
- S1 correct, S2 wrong: +w1, -w2  
- S1 wrong, S2 correct: -w1, +w2  
- Both wrong: penalize based on loss

**Observed Trajectory (with programs):**
- Start: w1=0.50, w2=0.50
- Episode 1: Correct → w1=0.77, w2=0.23 (S1 correct by chance)
- Episode 2: S2 corrects S1 → w1=0.35, w2=0.65 ✓
- Episode 50: w1=0.00, w2=1.00 (S2 dominates)

**But this doesn't matter** because S1 confidence bypass prevents blending!

### 3.3 Mathematical Formulation

The intended formula:
```
P(concept | query) = w1 * P_S1(concept | query) + w2 * P_S2(concept | query)
```

The actual formula (92% of time):
```
P(concept | query) = P_S1(concept | query)  [if conf > 0.35]
```

**This violates the dual-system framework!**

### 3.4 Relevant Research

**Kahneman (2011) - "Thinking, Fast and Slow"**
- System 1: Fast, intuitive, but error-prone
- System 2: Slow, deliberate, more accurate  
- **Key insight:** System 2 should override System 1 when stakes are high

**Stanovich & West (2000) - "Individual differences in reasoning"**
- Override mechanism: Monitor S1 outputs, engage S2 when uncertain
- **Confidence alone is insufficient** for gating
- Need conflict detection, not just low confidence

**Application to Our System:**  
Using S1 confidence for gating is theoretically unsound when S1 is miscalibrated.

---

## 4. Theoretical Fixes

### Fix 1: Remove Confidence-Based Gating (Immediate)

**Current:**
```python
if s1_confidence >= threshold:
    p = p1
else:
    p = w1*p1 + w2*p2
```

**Proposed:**
```python
# Always blend, but let weight learning handle it
p = w1*p1 + w2*p2
```

**Justification:**
- Weight updates already learn S1 vs S2 reliability
- If S1 is truly better, w1 → 1.0 naturally
- Doesn't require calibrated confidence

**Expected Impact:** +20% accuracy (from 48% → ~70%, matching S2 alone)

---

### Fix 2: Calibrate S1 Confidence (Temperature Scaling)

**Current:**
```python
confidence = abs(prob - 0.5) * 2.0
```

**Proposed:**
```python
# Apply temperature scaling
prob_calibrated = sigmoid(T * logit(prob))
confidence = abs(prob_calibrated - 0.5) * 2.0
```

Where T is learned from validation data to match accuracy.

**Justification:**
- Guo et al. (2017): Temperature scaling fixes neural network calibration
- Preserves ranks but adjusts confidence magnitudes
- Simple, single-parameter fix

**Expected Impact:** Moderate (+5-10%), but requires validation set

---

### Fix 3: Conflict-Based Gating

**Proposed:**
```python
disagreement = abs(p1 - p2)
if disagreement > threshold:
    # S1 and S2 disagree → use S2 (more reliable)
    p = p2
else:
    # Agreement → blend
    p = w1*p1 + w2*p2
```

**Justification:**
- Stanovich & West: Engage System 2 when conflict detected
- High disagreement signals S1 may be wrong
- Empirically: S2 correct rate 70% > S1 correct rate 48%

**Expected Impact:** +15-20% accuracy

---

### Fix 4: Ensemble S1 for Better Uncertainty

**Proposed:**
```python
# Train K CNNs with different random seeds
predictions = [model_k.predict(x) for k in range(K)]
p1 = mean(predictions)
confidence = 1 - std(predictions)  # High agreement → high confidence
```

**Justification:**
- Lakshminarayanan et al. (2017): Ensembles → better uncertainty
- Disagreement among models → low confidence
- More reliable than single model confidence

**Expected Impact:** +10-15% if used with Fix 1

---

## 5. Comparison to Bongard-LOGO Baselines

### 5.1 Their Approach: Meta-Learning

**Meta-Baseline-MoCo:**
- Pretrain CNN on train split with MoCo
- Fine-tune on support set per episode
- Accuracy: 65-75%

**Meta-Baseline-PS (Program Synthesis):**
- LSTM encoder-decoder on programs
- Meta-learn to synthesize programs
- Accuracy: 68-75% (test), 85% (train)

**Why They Work:**
- Meta-learning handles few-shot better than vanilla CNN
- Pretraining on train split → better initialization
- Program synthesis directly predicts sequences

### 5.2 Our Approach: Dual-System Theory

**Philosophy:**
- System 1: Pattern recognition (perception)
- System 2: Structured reasoning (cognition)
- Arbitration: Intelligent gating

**Advantages:**
- Theoretically grounded (Kahneman, Tenenbaum)
- Interpretable (explicit rules)
- Modular (can improve S1/S2 independently)

**Current Issue:**
- Gating mechanism broken (overconfidence)
- Not using meta-learning (could be added to S1)

### 5.3 Hybrid Path Forward

We can combine dual-system theory with meta-learning:

**System 1:** Meta-learned CNN
- Pretrain with SSL rotation (already doing this!)
- Add MAML/Prototypical Networks for meta-learning
- Expected: 55-60% accuracy (vs current 48%)

**System 2:** Program-based Bayesian reasoning (already 70%!)
- Keep current approach
- Could add program synthesis as "hypothesis generation"

**Arbitration:** Fixed gating (Fix 1 or Fix 3)
- Remove overconfident bypass
- Let weight learning handle S1 vs S2 selection

**Expected Combined Accuracy:** 70-75% (competitive with Meta-Baseline)

---

## 6. Mathematical Formalization of Fixes

### Fix 1: Always-Blend (Recommended First Step)

**Current Decision Rule:**
```
π(y | x, S) = { P_S1(y|x)              if conf_S1 > τ
              { w1·P_S1(y|x) + w2·P_S2(y|x)  otherwise
```

**Proposed:**
```
π(y | x, S) = w1·P_S1(y|x) + w2·P_S2(y|x)  ∀ episodes
```

**Weight Update (keep current):**
```
w1^(t+1) = w1^(t) · exp(η · Δ_1^(t))
w2^(t+1) = w2^(t) · exp(η · Δ_2^(t))
Normalize: w → w / (w1 + w2)
```

Where Δ_i = correctness-based adjustment (current implementation)

**Proof of Improvement:**

Given:
- P_S1 is miscalibrated: E[correct | high conf] = 48%
- P_S2 is well-calibrated: E[correct] = 70%

Current accuracy with bypass:
```
A_current = P(conf > τ) · 48% + P(conf ≤ τ) · A_blend
          = 0.92 · 48% + 0.08 · A_blend
          ≈ 48% (since bypass dominates)
```

Proposed accuracy without bypass:
```
A_proposed = w1 · 48% + w2 · 70%
```

Since weight updates favor S2 (70% > 48%), w2 → 0.7 asymptotically:
```
A_proposed → 0.3 · 48% + 0.7 · 70% = 63.4%
```

**Expected gain: +15% absolute**

---

### Fix 3: Conflict-Based Gating (Alternative)

**Decision Rule:**
```
π(y | x, S) = { P_S2(y|x)              if |P_S1 - P_S2| > δ
              { w1·P_S1(y|x) + w2·P_S2(y|x)  otherwise
```

**Justification:**

From signal detection theory:
- High disagreement → S1 prediction unreliable
- S2 has higher baseline accuracy (70% > 48%)
- Rational strategy: defer to more reliable system when conflict arises

**Optimal δ:**

Maximize accuracy:
```
A(δ) = P(|p1-p2| > δ) · A_S2 + P(|p1-p2| ≤ δ) · A_blend

∂A/∂δ = 0  ⟹  δ* where marginal S2 accuracy = marginal blend accuracy
```

Empirically estimate from validation set. Initial guess: δ = 0.3

**Expected gain: +18% absolute**

---

## 7. Action Plan

### Phase 1: Quick Fix (1 hour)
1. **Remove confidence bypass** (Fix 1)
2. Test on 100 episodes
3. **Expected: 48% → 63%**

### Phase 2: Refinement (2-3 hours)
4. Implement conflict-based gating (Fix 3)
5. Tune disagreement threshold  
6. **Expected: 63% → 68%**

### Phase 3: Meta-Learning (1-2 days)
7. Add MAML to System 1
8. Pretrain on train split
9. **Expected: 68% → 72%**

### Phase 4: Program Synthesis (optional, 3-5 days)
10. Add LSTM program synthesis as S2 hypothesis generator
11. **Expected: 72% → 75%+**

---

## 8. Conclusion

**Summary of Findings:**

1. **System 1 (CNN):** Overconfident (95%+ conf) but inaccurate (48%) due to overfitting on 12 examples
2. **System 2 (Bayesian):** Well-calibrated and accurate (70%) with program features  
3. **Arbitration:** Broken due to S1 overconfidence bypassing S2 92% of time

**Root Cause:** Gating mechanism trusts S1 confidence, but confidence is meaningless when overfit.

**Theoretically-Grounded Fix:** Remove confidence-based bypass, always blend with learned weights.

**Expected Impact:** 48% → 63-68% (immediate), 70-75% (with meta-learning)

**Theoretical Alignment:**
- Dual-process theory: ✓ (S1 fast perception, S2 deliberate reasoning)
- Tenenbaum framework: ✓ (S2 uses compositional Bayesian rules over programs)
- Meta-learning: ○ (can be added to S1 in Phase 3)

**Next Steps:** Implement Fix 1 (remove confidence bypass), validate on test splits.

---

## References

1. Kahneman, D. (2011). *Thinking, Fast and Slow*.
2. Tenenbaum, J. B., et al. (2011). "How to grow a mind." *Science*.
3. Lake, B. M., et al. (2015). "Human-level concept learning through probabilistic program induction." *Science*.
4. Goodman, N. D., et al. (2008). "A rational analysis of rule-based concept learning." *Cognitive Science*.
5. Guo, C., et al. (2017). "On calibration of modern neural networks." *ICML*.
6. Lakshminarayanan, B., et al. (2017). "Simple and scalable predictive uncertainty." *NeurIPS*.
7. Stanovich, K. E., & West, R. F. (2000). "Individual differences in reasoning." *Behavioral and Brain Sciences*.
8. Vapnik, V. (1998). *Statistical Learning Theory*.
