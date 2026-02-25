# DIAGNOSIS: Why We're Getting ~50% Accuracy (Random Performance)

## 1. THEORETICAL FOUNDATION: What Tenenbaum/Lake Actually Propose

### Core References:
- **Lake et al. (2015)** "Human-level concept learning through probabilistic program induction"
- **Tenenbaum (1999)** "Rules and similarity in concept learning" 
- **Tenenbaum (1998)** "Bayesian modeling of human concept learning"

### Key Insight: Dual-System Should Work Like This

```
System 1 (Fast/Intuitive):
├─ Quick similarity judgment based on visual features
└─ Direct feature comparison: "Is this similar to positive examples?"

System 2 (Slow/Deliberate):  ← THE CRITICAL PART
├─ Structure search: Infer logical rules/programs
├─ E.g., "all shapes with >6 vertices AND symmetric" 
├─ Uses STRUCTURED HYPOTHESIS SPACE (not just Bayes on features)
└─ Corrects System 1 when confidence is misaligned

Arbitration:
└─ Use S1 when confident AND when S2 agrees
└─ Use S2 when S1 uncertain OR when they disagree
```

---

## 2. WHAT WE'RE ACTUALLY DOING (vs. What Theory Says)

### Current System 1 (Neural CNN)
```python
# Problem: Training CNN on 12 examples per episode
✗ CNN needs 1000+ examples to generalize; 12 is severe overfitting
✗ Per-episode retraining means no transfer learning
✗ Result: 99% confidence on decisions that are wrong
```

**Why this fails:**
- A CNN trained on only 6 pos + 6 neg images WILL memorize them perfectly
- Then applied to query images it's never seen = random guessing
- But it gives 99% confidence → blocks System 2 from helping
- **The network doesn't learn the CONCEPT, just the specific examples**

### Current System 2 (Bayesian on Features)
```python
# Problem: Hand-crafted features can't capture Bongard concepts
Features = [ink_density, centroid_x, centroid_y, spread_x, 
            spread_y, h_flip_sym, v_flip_sym, edge_strength, 
            quadrant_inks_1-4]

✗ These 12 features are TOO SIMPLE for Bongard tasks
✗ Bongard requires understanding:
   - Temporal relationships (alignment, relative positions)
   - Topological properties (connectedness, holes, bridges)
   - Structural hierarchies (nested shapes, layering)
   - Symmetry axis orientation and type
   
✗ Example: "All shapes have a horizontal line as spine"
  → Requires detecting LINE + ORIENTATION + ALIGNMENT
  → Hand-crafted features CAN'T capture this

✗ Bayesian classifier on poor features ≈ random guessing
  → ~50% accuracy is what we see
```

---

## 3. ROOT CAUSE: System 1 Confidence is WRONG

```
Our weight update logic:
  if System1_confidence >= 0.35:
      use S1 only
      
Problem: System1 confidence is measuring FIT TO TRAINING DATA
         not GENERALIZATION TO NEW DATA!

Real data flow:
  1. CNN trains on 6 pos + 6 neg → fits perfectly (loss→0)
  2. CNN computes confidence = difference between logits
  3. During training, diff is high (one class strongly preferred)
  4. But on NEW query images: random logits → confidence is meaningless!
  
Solution: Confidence SHOULD measure DISAGREEMENT between S1 and S2
         Not self-contained S1 certainty
```

---

## 4. MISSING: Structured Hypothesis Space (System 2 Should Infer Programs)

### What Lake et al. Do:
```
System 2 searches over:
  - Generative programs for shapes
  - Rules: AND, OR, NOT combinations of features
  - Concepts like "all X's have property P"
  
Example for Bongard:
  Hypothesis space: 
    {shape_type AND symmetry_type AND position_relation}
    {vertex_count AND area_ratio AND ...}
  
  S2 computes: P(concept | pos_examples, neg_examples)
  Returns: Most likely program + confidence
```

### What We Do:
```
System 2 just does:
  μ, σ for each class on 12 features
  P(concept | features) via Bayes
  
This is equivalent to:
  "Assume feature distribution is Gaussian"
  No structure search, no program induction
```

---

## 5. WEIGHT UPDATE ISN'T WORKING

Current logic:
```python
# After episode:
loss1 = BCE(S1_pred_pos, 1.0) + BCE(S1_pred_neg, 0.0)
loss2 = BCE(S2_pred_pos, 1.0) + BCE(S2_pred_neg, 0.0)

w1_new = w1 * exp(-lr * loss1)  # 2.0 is learning rate
w2_new = w2 * exp(-lr * loss2)
```

**Problem:**
- S1 has 99% confidence → BCE loss ≈ 0
- S2 has ~50% confidence → BCE loss ≈ 0.693
- Weight update: w1 *= exp(-2.0 × 0.01) ≈ 0.98 (barely moves)
- Weight update: w2 *= exp(-2.0 × 0.69) ≈ 0.25 (penalized)
- **Result: S2 gets progressively down-weighted despite being MORE HONEST about uncertainty**

The weight should flip toward S2 because S2 is uncertain (realistic) while S1 is falsely confident.

---

## 6. WHY 50% IS EXACTLY WHAT WE SHOULD GET

With this setup:
```
Per episode:
  1. S1 (CNN on 12 examples) learns perfect memorization
  2. S1 has 99% confidence on everything
     → Query prediction is biased but random w.r.t. unseen data
  3. S2 (Bayesian on weak features) gives ~50% guess
  4. System 1 blocks System 2 (conf >= 0.35)
  5. Result: Use S1's random prediction
  
Aggregated: ~50% accuracy (random)
```

---

## 7. HOW TO FIX (In Priority Order)

### Fix #1: RIGHT NOW - Restructure System 2 to Learn from Failures
**Severity: CRITICAL**

Current: w2 down-weighted when loss2 is high (uncertain)
Should be: w2 UP-weighted when S1 was WRONG but S2 was RIGHT

```python
# New logic:
s1_correct = (s1_pos > s1_neg)  # Did S1 get it right?
s2_correct = (s2_pos > s2_neg)  # Did S2 get it right?

# Reward when:
#   - S1 is wrong BUT S2 is right (should use S2)
#   - S2 is confident AND right (System 2 working well)

# Example:
if not s1_correct and s2_correct:
    w2 *= exp(+learning_rate)  # REWARD S2
    w1 *= exp(-learning_rate)
```

### Fix #2: Use Pre-trained Feature Extractor for System 1
**Severity: HIGH**

Instead of training CNN from scratch on 12 examples:
```python
# Pre-train on train split first:
# - Load all train episodes (~9300 episodes)
# - Train CNN on aggregate positive/negative examples (~56K images)
# - Then freeze backbone, only fine-tune last layer per-episode

# This gives S1:
# ✓ Learned visual features that generalize
# ✓ Meaningful confidence (based on learned representations)
# ✓ Faster per-episode training (only 1-2 layers to fine-tune)
```

### Fix #3: Implement Structured System 2 (Proper Program Induction)
**Severity: HIGH** (long-term improvement)

Replace Bayesian feature classifier with rule search:
```python
# System 2 searches over rules like:
rules = [
    "vertex_count > 5",
    "is_symmetric",
    "centroid_in_left_half",
    "has_hole",
    "touches_border"
]

# Generates hypotheses:
hypotheses = all_conjunctions(rules)  # AND, OR, NOT versions

# Scores each rule by: P(pos | rule) - P(neg | rule)

# Returns rule + confidence
```

### Fix #4: Proper Confidence Fusion
**Severity: MEDIUM**

```python
# Instead of threshold-based:
#   if S1_conf >= 0.35: use S1 only

# Use disagreement-based:
confidence_in_s1 = measure_agreement(S1, S2)

# Confidence is HIGH when:
#   - S1 confident AND matches S2
# Confidence is LOW when:
#   - S1 bullish but S2 disagrees
#   - Both uncertain
```

---

## 8. EXPECTED IMPROVEMENT PATH

```
Current: 50-52% (random)
  ↓ (Fix #1: Proper reward)
Target:  55-65% (System 2 starting to help)
  ↓ (Fix #2: Pre-training)
Target:  70-80% (S1 better generalization)
  ↓ (Fix #3: Structured rules)
Target:  85-92% (Proper program induction)
```

---

## 9. SUMMARY: What's Wrong

| Issue | Impact | Fix |
|-------|--------|-----|
| S1 trains on 12 examples → overfits | CNN gives false confidence (99% on random guesses) | Pre-train on train split (~56K images) |
| No transfer learning | Each episode is independent, no learning across tasks | Use frozen CNN backbone |
| S2 uses weak features | 12 hand-crafted features can't capture Bongard concepts | Implement structured program search |
| Weight reward is backward | S2 penalized for honest uncertainty | Reward S2 when it corrects S1 |
| Threshold arbitration | Ignores disagreement between S1 and S2 | Use confidence-based blending |
| No rule search | Just vanilla Bayes on features | Add logical rule induction |

---

## 10. Implementation Path

**Week 1 (Immediate):**
1. Fix weight update to reward S2 when S1 is wrong
2. Implement pre-training on train split
3. Test with 100 episodes → expect 60%+

**Week 2:**
4. Add simple rule search to System 2
5. Implement disagreement-based confidence fusion

**Week 3:**
6. Enhance rule search with negation, complex patterns
7. Data augmentation for training

Expected final: 80-90% on clean test sets.

