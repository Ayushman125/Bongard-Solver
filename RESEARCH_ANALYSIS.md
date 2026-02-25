# Critical Research Analysis: Bongard-LOGO Dual-System Implementation

**Date**: February 25, 2026  
**Issue**: Compositional rules didn't improve performance (50.83% baseline vs 50.83%-51.88% compositional)

---

## Executive Summary

**ROOT CAUSE IDENTIFIED**: We are **NOT using the LOGO action programs** available in our dataset. Instead, we extract 12 pixel-level features (centroid, symmetry, edge strength) that are fundamentally misaligned with the Bongard-LOGO task structure.

**Impact**: Our hypothesis space cannot express the actual concepts in the benchmark, making compositional reasoning ineffective.

---

## 1. Bongard-LOGO Paper Core Findings

### 1.1 What Concepts Actually Are

From Section 2.3:
- **Free-form shapes (3,600 problems)**: Concept IS the sequence of action strokes
  - Example: `["line_square_1.000-0.500", "arc_triangle_0.500_0.625-0.750", ...]`
  - Each positive image shares the EXACT stroke sequence
  - Negative images have subtle stroke perturbations (zigzag → normal, different angles)

- **Basic shapes (4,000 problems)**: Concept is shape category combinations
  - Example: "triangle AND trapezoid"  
  - Stroke types (zigzag vs normal) are NUISANCES here (analogy-making perception)

- **Abstract shapes (4,400 problems)**: Concept is geometric attributes
  - Examples: convex, symmetric, have_four_straight_lines
  - 25 total attributes, combinatorially mixed

### 1.2 Core Properties of Human Cognition

From Section 2.2:

1. **Context-dependent perception**:
   - Same shape pattern interpreted differently based on context
   - Example: 4 straight lines vs 6 straight lines depends on whether intersections split lines
   - **Our current features are context-free** ❌

2. **Analogy-making perception**:
   - Trade off representations: zigzag → straight line in basic shapes
   - Circles/zigzags form conceptual trapezoid  
   - **Our features can't trade off** (centroid is always centroid) ❌

3. **Few samples but infinite vocabulary**:
   - Free-form shapes have arbitrary stroke combinations → infinite space
   - **Our 12 features have fixed vocabulary** ❌

### 1.3 SOTA Model Results

From Table 1 (paper):
- **Best meta-learner**: Meta-Baseline-MoCo (81.2% train, 65.9% FF, 72.2% BA, 63.9% CM, 64.7% NV)
- **With program synthesis**: Meta-Baseline-PS (85.2% train, 68.2% FF, 75.7% BA, 67.4% CM, 71.5% NV)
- **Human experts**: 92.1% FF, 99.3% BA, 90.7% CM/NV

**Key insight from Appendix C**:
> "The ground-truth action programs provide useful supervision in guiding symbolic reasoning in the action space."

**Meta-Baseline-PS** (Figure 6):
- CNN encoder → LSTM → Action decoder (MDN + MoG)
- Decodes: action_type (MLP+softmax), moving_type (MLP+softmax), moving_length/angle (MDN+MoG)
- Pre-train on program synthesis, then meta-train on Bongard problems

---

## 2. Tenenbaum's Concept-as-Program Framework

From reference [3] (Lake et al., 2015):
- **Concepts are generative programs** that can produce examples
- **Bayesian Program Learning**: P(program | examples) via probabilistic inference
- **Compositional primitives**: Build complex concepts from base actions
- **Few-shot learning**: Infer program from 1-5 examples, generalize to new instances

### Applied to Bongard-LOGO:
- Concept = LOGO action program
- Inference = Deduce program from 6 positive images
- Generalization = Test if new image matches inferred program

**What we should do**:
1. Infer action programs from positive set
2. Score test images by P(image | inferred_program)
3. Use Bayesian posterior to handle uncertainty

**What we're doing**:
1. Extract 12 pixel features ❌
2. Score with threshold rules on features ❌  
3. Can't express stroke sequences ❌

---

## 3. Dual-System Theory (Kahneman)

Referenced implicitly in paper discussion:

### System 1 (Fast, Intuitive):
- Pattern recognition, perceptual features
- Neural networks, CNNs
- **Good for**: Quick judgments, learned patterns
- **Our implementation**: SmallCNN with SSL rotation pretraining ✓

### System 2 (Slow, Deliberate):
- Rule-based reasoning, symbolic logic
- Compositional search, hypothesis testing
- **Good for**: Novel concepts, abstract reasoning
- **Our implementation**: Bayesian rule search... but on WRONG features ❌

**Critical issue**:
- System 1 should provide **rich representations** (e.g., 128-dim CNN embeddings)
- System 2 should do **symbolic reasoning** (e.g., program induction, rule search)
- **Our System 1 provides 12 hand-crafted features** → too impoverished!
- **Our System 2 searches threshold rules** → can't express stroke sequences!

---

## 4. Our Current Implementation

### 4.1 What We Have (Good)

✓ **Action program data available**:
```python
# bayes_dual_system/data.py line 23
prog_file = os.path.join(root_path, category, f"{category}_action_programs.json")

# bayes_dual_system/types.py
@dataclass
class ExampleItem:
    program: Any  # Contains the LOGO action program!
```

✓ **Dual-system architecture**:
- System 1: SmallCNN with SSL rotation pretraining
- System 2: Bayesian rule search
- Confidence-based arbitration

✓ **Compositional rules**:
- Primitive: `f_j >= θ`, `(f_i - f_j) >= θ`
- Compositional: AND/OR over top primitives

### 4.2 What We're Missing (Critical)

❌ **Using programs for reasoning**:
```python
# bayes_dual_system/system2_bayes_image.py line 34-35
x_pos = np.asarray([extract_image_features(i.image_path, self.image_size) for i in support_pos])
# i.program is IGNORED!
```

❌ **Feature alignment with LOGO structure**:
```python
# bayes_dual_system/image_utils.py extract_image_features()
features = [
    ink.mean(),         # Pixel-level statistic
    centroid_x,         # Pixel-level statistic
    centroid_y,         # Pixel-level statistic
    spread_x, spread_y, # Pixel-level statistics
    h_sym, v_sym,       # Pixel correlation (not geometric symmetry!)
    edge_strength,      # Pixel gradient
    q1, q2, q3, q4,     # Quadrant ink distribution
]
# NO stroke detection, NO shape category, NO geometric properties!
```

❌ **Hypothesis space mismatch**:
- **Actual concept** (free-form): `["line_square_1.000-0.500", "arc_triangle_0.500_0.625-0.750", ...]`
- **Our rule**: `centroid_x >= 0.523`  
- **Why it fails**: Can't express "has this 4-stroke sequence"

- **Actual concept** (basic): "triangle AND trapezoid"
- **Our rule**: `(spread_x - spread_y) >= 0.15`
- **Why it fails**: Can't identify shape categories

- **Actual concept** (abstract): "convex"
- **Our rule**: `h_sym >= 0.05 AND edge_strength < 0.3`
- **Why it fails**: Pixel symmetry ≠ geometric symmetry

❌ **No meta-learning**:
- Paper shows meta-learning is crucial (Table 1)
- We evaluate each problem independently
- Can't learn to learn across 9,300 training problems

---

## 5. Why Compositional Rules Failed

### Baseline vs Compositional Results:
- **test_ff**: 50.83% → 50.83% (no change)
- **test_bd**: 52.92% → 52.92% (no change)  
- **test_hd_comb**: 49.75% → 49.75% (no change)
- **test_hd_novel**: 52.19% → 51.88% (-0.31%)

### Root Cause Analysis:

1. **Feature space is too weak**:
   - 12 pixel statistics can't capture:
     - Stroke sequences (free-form concepts)
     - Shape categories (basic concepts)
     - Geometric properties (abstract concepts)
   - AND/OR over weak primitives = still weak!

2. **Hypothesis space doesn't align with concepts**:
   - Threshold rules: `f_j >= θ`
   - Actual concepts: stroke sequences, shape combinations, geometric attributes
   - **No overlap!**

3. **Search insufficiency**:
   - max_rules=24 (12 primitives + 12 AND/OR compositions)
   - Concept space: infinite (free-form), 627² (basic), 2²⁵ (abstract)
   - **Vastly undersampling!**

4. **Bayesian scoring assumes independence**:
   - Beta-Bernoulli: P(fire | pos) independent across rules
   - Actual concepts: compositional dependencies (stroke sequences, shape parts)
   - **Wrong probabilistic model!**

---

## 6. Comparison to SOTA Approaches

### What Top Models Do:

#### Meta-Baseline-MoCo (best baseline):
1. **Representation learning**: MoCo pre-training on all images → 128-dim embeddings
2. **Meta-learning**: Learn to learn across 9,300 training problems (episodic training)
3. **Metric learning**: Nearest-centroid classification in learned embedding space
4. **Result**: 81.2% train, 65.9% FF, 72.2% BA, 63.9% CM, 64.7% NV

#### Meta-Baseline-PS (program synthesis):
1. **CNN encoder**: Extract visual features
2. **LSTM decoder**: Predict action programs (action_type, moving_type, moving_length, moving_angle)
3. **Pre-training**: Supervised on ground-truth programs (symbolic supervision!)
4. **Meta-learning**: Fine-tune on Bongard problems
5. **Result**: 85.2% train, 68.2% FF, 75.7% BA, 67.4% CM, 71.5% NV

### What We're Doing:
1. **Representation**: 12 hand-crafted pixel features (vs 128-dim learned embeddings)
2. **Learning**: No meta-learning, no training (vs episodic learning across 9,300 problems)
3. **Reasoning**: Threshold rules on pixel stats (vs program synthesis or metric learning)
4. **Result**: 50.83% (chance-level on most splits)

**Gap**: We're solving a different (easier) problem than Bongard-LOGO!

---

## 7. Correct Implementation Paths

### Option A: Program Synthesis (Tenenbaum-aligned)

**Architecture**:
1. CNN encoder (System 1) → visual embeddings
2. LSTM + Action Decoder (System 2) → action programs
3. Bayesian posterior scoring: P(program | positive_images)
4. Test: Does test_image match highest-scoring program?

**Pros**:
- Directly models concepts as programs ✓
- Aligns with paper's Meta-Baseline-PS ✓
- Can use ground-truth programs for pre-training ✓

**Cons**:
- Requires training (9,300 problems)
- Complex architecture (LSTM + MDN)
- May overfit to training program distribution

**Implementation**:
```python
class ProgramSynthesisSystem2:
    def __init__(self):
        self.encoder = SmallCNN(output_dim=128)
        self.decoder = LSTMActionDecoder(
            action_vocab=["line", "arc"],
            type_vocab=["normal", "zigzag", "triangle", "circle", "square"],
        )
    
    def fit(self, support_pos, support_neg):
        # Encode positive images
        embeddings = [self.encoder(img) for img in support_pos]
        
        # Decode to programs
        programs = [self.decoder(emb) for emb in embeddings]
        
        # Find consensus program (most common subsequences)
        self.consensus_program = self._consensus_search(programs)
    
    def predict(self, test_image):
        # Does test image match consensus program?
        test_emb = self.encoder(test_image)
        test_prog = self.decoder(test_emb)
        return self._program_similarity(test_prog, self.consensus_program)
```

---

### Option B: Symbolic Feature Extraction (Hybrid)

**Architecture**:
1. Stroke detection: Identify lines, arcs, zigzags from pixels
2. Shape recognition: Match detected strokes to 627 shape templates
3. Geometric analysis: Compute true convexity, symmetry, angles
4. Bayesian rule search: Over SYMBOLIC features (not pixel stats)

**Pros**:
- No training required (keeps our zero-shot approach)
- Can still use action programs as weak supervision
- Interpretable features

**Cons**:
- Hard to detect strokes from pixels (computer vision challenge)
- Need to implement 627 shape templates
- May not generalize to novel shapes

**Implementation**:
```python
def extract_symbolic_features(image_path, programs=None):
    img = load_image(image_path)
    
    # Stroke detection
    strokes = detect_strokes(img)  # Returns [(type, length, angle), ...]
    line_count = sum(1 for s in strokes if s.type == "line")
    arc_count = sum(1 for s in strokes if s.type == "arc")
    zigzag_count = sum(1 for s in strokes if s.moving_type == "zigzag")
    
    # Shape recognition
    shape_category = match_shape_template(strokes)  # e.g., "triangle"
    
    # Geometric analysis
    convex = is_convex(img)  # Proper convex hull check
    symmetric = has_reflection_symmetry(img)  # Geometric symmetry
    angle_count = count_angles(strokes)
    
    return [line_count, arc_count, zigzag_count, ..., convex, symmetric, ...]
```

---

### Option C: Meta-Learning with Better Representations

**Architecture**:
1. Pre-train CNN (System 1): SSL rotation or MoCo on all Bongard images
2. Meta-train (System 2): ProtoNet or MAML on 9,300 training problems
3. Episodic evaluation: Learn concept from support set, classify query

**Pros**:
- Matches SOTA approach (Meta-Baseline) ✓
- Proven to work (65.9% test accuracy) ✓
- Can combine with program synthesis later

**Cons**:
- Requires full dataset and GPU training
- Black-box learned embeddings (less interpretable)
- Doesn't directly use available programs

**Implementation**:
```python
# Pre-training (SSL)
backbone = SmallCNN()
pretrain_ssl_rotation(backbone, all_bongard_images, epochs=100)

# Meta-training (ProtoNet)
model = ProtoNet(backbone)
for episode in meta_train_loader:  # 9,300 problems
    support_pos, support_neg, query = episode
    
    # Compute prototypes
    proto_pos = support_pos.mean(dim=0)
    proto_neg = support_neg.mean(dim=0)
    
    # Classify query by nearest prototype
    pred = argmin(dist(query, proto_pos), dist(query, proto_neg))
    loss = cross_entropy(pred, label)
    loss.backward()
```

---

### Option D: Hybrid Neuro-Symbolic (Best Long-term)

**Architecture**:
1. **System 1 (Neural)**: CNN → 128-dim embeddings + program decoder
2. **System 2 (Symbolic)**: Rule search over decoded programs OR meta-learned embeddings
3. **Meta-learning**: Train across 9,300 problems
4. **Arbitration**: Use both neural similarity and symbolic rules

**Pros**:
- Combines strengths of neural (representation) and symbolic (reasoning) ✓
- Can use programs when detected, fallback to embeddings ✓
- Most aligned with paper's vision (Appendix C, Section 5)

**Cons**:
- Complex to implement and train
- Needs careful integration of neural/symbolic components

---

## 8. Immediate Actionable Steps

### Step 1: Verify Program Data Availability
```bash
python -c "
from bayes_dual_system.data import RawShapeBongardLoader
loader = RawShapeBongardLoader('data/raw/ShapeBongard_V2')
episode = loader.build_episode('ff_nact4_0103')
print('Positive programs:', episode.support_pos[0].program)
print('Negative programs:', episode.support_neg[0].program)
"
```

### Step 2: Implement Symbolic Feature Extraction (Quick Fix)
```python
def extract_symbolic_features_v2(image_path, program=None):
    # Use program if available (ground-truth supervision!)
    if program:
        features = program_to_features(program)
    else:
        # Fallback to vision-based detection
        features = detect_strokes_from_image(image_path)
    
    return features

def program_to_features(program):
    # Count stroke types from ground-truth program
    line_count = sum(1 for cmd in program if cmd.startswith("line"))
    arc_count = sum(1 for cmd in program if cmd.startswith("arc"))
    zigzag_count = sum(1 for cmd in program if "zigzag" in cmd)
    # ... extract all symbolic features
    return [line_count, arc_count, zigzag_count, ...]
```

### Step 3: Test on Free-Form Problems
- Free-form concepts ARE the stroke sequences
- If we can match program features, should see immediate improvement
- Target: > 65% (vs current 50.83%)

### Step 4: Implement Meta-Learning (Medium-term)
- Switch from per-problem evaluation to episodic training
- Use 9,300 training problems to learn representations
- Target: Match Meta-Baseline (~65-72%)

### Step 5: Add Program Synthesis (Long-term)
- Train LSTM decoder on ground-truth programs
- Combine with Bayesian hypothesis search
- Target: Match Meta-Baseline-PS (~68-75%)

---

## 9. Theoretical Grounding

### Why Our Approach is Principled (in theory):
1. **Dual-system**: Matches cognitive science (Kahneman) ✓
2. **Bayesian reasoning**: Matches Tenenbaum's framework ✓  
3. **Compositional rules**: Matches concept-as-program ✓

### Why It Failed (in practice):
1. **Wrong feature space**: Pixel stats ≠ LOGO programs ❌
2. **No learning**: Zero-shot ≠ meta-learning required by benchmark ❌
3. **Hypothesis space mismatch**: Threshold rules ≠ stroke sequences ❌

### To Fix:
1. **Use programs as features** OR **detect strokes from images**
2. **Add meta-learning** OR **improve zero-shot with better features**
3. **Align hypothesis space** with actual concept structure

---

## 10. Success Criteria

### Minimum Viable Fix (Use Programs):
- Extract features from `ExampleItem.program` field
- Re-run compositional rules on symbolic features
- **Target**: > 60% test accuracy (10% absolute improvement)

### Good Solution (Symbolic Features):
- Implement stroke detection + shape matching
- Use geometric analysis for abstract attributes
- **Target**: > 65% test accuracy (match ProtoNet baseline)

### Excellent Solution (Meta-Learning):
- Pre-train SSL + meta-train ProtoNet/MAML
- Episodic learning on 9,300 problems
- **Target**: > 70% test accuracy (match Meta-Baseline)

### Ideal Solution (Program Synthesis):
- Neural program decoder + symbolic reasoning
- Hybrid neuro-symbolic architecture
- **Target**: > 75% test accuracy (approach Meta-Baseline-PS)

### Human Parity (Long-term Goal):
- Context-dependent perception
- Analogy-making perception  
- Compositional generalization
- **Target**: > 90% test accuracy (match human experts)

---

## 11. Conclusion

**We have correctly identified the dual-system approach in principle**, but:
- ❌ **System 1 representation is too weak** (12 features vs 128-dim embeddings or programs)
- ❌ **System 2 reasoning is misaligned** (threshold rules vs stroke sequences)
- ❌ **No meta-learning** (required for Bongard-LOGO generalization)

**The compositional rules failed because**:
- Composing weak primitives yields weak compositions
- Hypothesis space has zero overlap with actual concepts
- Feature extraction ignores available LOGO program data

**Path forward**:
1. **Immediate**: Use `ExampleItem.program` field for symbolic features
2. **Short-term**: Implement stroke detection or shape matching
3. **Medium-term**: Add meta-learning (ProtoNet/MAML)
4. **Long-term**: Full hybrid neuro-symbolic with program synthesis

**Alignment with research**:
- ✓ Dual-system theory: Correct conceptually, needs better implementation
- ✓ Tenenbaum framework: Need to actually use programs, not ignore them!
- ✓ Bongard-LOGO paper: Meta-Baseline-PS is the north star (85% train, 68-75% test)

**Next command**: Implement symbolic feature extraction using available program data.
