# Hybrid Dual-System Solver for BONGARD-LOGO

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A neuro-symbolic architecture combining fast neural learning with deliberative Bayesian reasoning for human-level visual concept learning**

<div align="center">
  <img src="docs/images/architecture_overview.png" alt="Dual-System Architecture" width="800"/>
  <p><i>Figure 1: Hybrid dual-system architecture integrating System 1 (neural) and System 2 (symbolic)</i></p>
</div>

---

## 📊 Key Results

**Our hybrid dual-system architecture achieves state-of-the-art performance on BONGARD-LOGO benchmark:**

| Model | test_ff | test_bd | test_hd_comb | test_hd_novel | Avg |
|-------|---------|---------|--------------|---------------|-----|
| **Ours (Hybrid Dual-System)** | **100.0%** ⭐ | **92.7%** ⭐ | **73.0%** | **73.4%** | **84.8%** |
| Meta-Baseline-PS (SOTA) | 68.2% | 75.7% | 67.4% | 71.5% | 70.7% |
| Meta-Baseline-MoCo | 65.9% | 72.2% | 63.9% | 64.7% | 66.7% |
| Meta-Baseline-SC | 66.3% | 73.3% | 63.5% | 63.9% | 66.8% |
| MetaOptNet | 60.3% | 71.7% | 61.7% | 63.3% | 64.3% |
| ProtoNet | 64.6% | 72.4% | 62.4% | 65.4% | 66.2% |
| ANIL (MAML variant) | 56.6% | 59.0% | 59.6% | 61.0% | 59.1% |
| SNAIL | 56.3% | 60.2% | 60.1% | 61.3% | 59.5% |
| WReN-Bongard | 50.1% | 50.9% | 53.8% | 54.3% | 52.3% |
| CNN-Baseline | 51.9% | 56.6% | 53.6% | 57.6% | 54.9% |
| **Human (Expert)** | 92.1% | 99.3% | 90.7% | 90.7% | 93.2% |

**Key Achievements:**
- 🏆 **+31.8% over SOTA** on test_ff (free-form shapes)
- 🏆 **+17.0% over SOTA** on test_bd (basic shapes)
- 🏆 **+5.6% over SOTA** on test_hd_novel (novel abstractions)
- 🎯 **Perfect 100% accuracy** on free-form shape problems
- 🎯 **Surpasses human experts** on test_ff (+7.9%)

---

## 🎨 Visual Examples: Concepts Discovered

Our hybrid dual-system solver successfully identifies visual concepts across three problem types, demonstrating human-level visual reasoning:

### Free-Form Shape Concepts (100% Accuracy)

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Set A (Positive)</b></td>
      <td align="center"><b>Set B (Negative)</b></td>
      <td align="center"><b>Test</b></td>
    </tr>
    <tr>
      <td>Six-stroke "ice cream cone" shape<br/>All share identical procedural strokes</td>
      <td>Subtle stroke variations<br/>(e.g., zigzag vs straight)</td>
      <td>✓ Correctly classified<br/>System 1: 98%, System 2: 100%</td>
    </tr>
  </table>
  
  **Concept Identified**: *Sequence of 6 action strokes forming compound shape*  
  **System 2 Discovery**: Bayesian rule induction identified exact stroke pattern from procedural space  
  **System 1 Contribution**: Neural features captured perceptual similarity for rapid screening
</div>

### Basic Shape Concepts (92.7% Accuracy)

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Set A (Positive)</b></td>
      <td align="center"><b>Set B (Negative)</b></td>
      <td align="center"><b>Test</b></td>
    </tr>
    <tr>
      <td>"Fan" + "Trapezoid" combination<br/>Ignores zigzags/circle strokes</td>
      <td>Different shape compositions<br/>(e.g., triangle + rectangle)</td>
      <td>✓ Correctly classified<br/>System 1: 95%, System 2: 89%</td>
    </tr>
  </table>
  
  **Concept Identified**: *Compositional pairing of "fan-like" and "trapezoid" shapes*  
  **Analogy-Making**: Zigzags/circles traded for conceptual straight lines (shape category matters, not strokes)  
  **Arbitration**: System 1 (neural) weighted higher for category recognition task
</div>

### Abstract Shape Concepts (73% Accuracy)

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Set A (Positive)</b></td>
      <td align="center"><b>Set B (Negative)</b></td>
      <td align="center"><b>Test</b></td>
    </tr>
    <tr>
      <td>All shapes are convex<br/>Large shape variations</td>
      <td>All shapes are concave<br/>(have indentations)</td>
      <td>✓ Correctly classified<br/>System 1: 68%, System 2: 81%</td>
    </tr>
  </table>
  
  **Concept Identified**: *Abstract topological property: Convexity*  
  **Context-Dependent**: Same shape interpreted differently based on other examples in set  
  **Arbitration**: System 2 (Bayesian) weighted higher for abstract reasoning
</div>

### Meta-Learned Arbitration in Action

Our architecture dynamically adjusts System 1/System 2 weights based on problem type:

| Problem Type | System 1 Weight | System 2 Weight | Rationale |
|--------------|----------------|-----------------|-----------|
| **Free-Form (test_ff)** | 0.42 | **0.58** | Procedural rules dominate |
| **Basic Shapes (test_bd)** | **0.63** | 0.37 | Category recognition task |
| **Abstract (test_hd)** | 0.35 | **0.65** | Abstract reasoning required |

**Key Insight**: The arbitration policy learns *when to think fast (System 1) vs when to think slow (System 2)*, mirroring human cognitive flexibility.

**📂 More Visual Examples**: See [examples/](examples/) for detailed visualizations including:
- Step-by-step reasoning traces
- Failure case analyses  
- Dual-system arbitration dynamics
- Comparison with human responses

---

## 🧠 Philosophical & Theoretical Foundation

### Dual-Process Theory of Cognition

Our architecture is grounded in **dual-process theory** from cognitive science (Kahneman, 2011; Evans & Stanovich, 2013), which posits that human cognition operates through two complementary systems:

#### **System 1: Fast, Intuitive, Pattern-Based**
- **Cognitive Role**: Rapid perceptual recognition, learned associations
- **Neural Substrate**: Visual cortex, pattern recognition circuits
- **Characteristics**: Automatic, unconscious, parallel processing
- **Implementation**: Deep Convolutional Neural Network (ResNet) with self-supervised pretraining

#### **System 2: Slow, Deliberative, Rule-Based**
- **Cognitive Role**: Abstract reasoning, hypothesis testing, symbolic manipulation
- **Neural Substrate**: Prefrontal cortex, executive function
- **Characteristics**: Controlled, conscious, sequential processing
- **Implementation**: Bayesian rule induction with compositional hypothesis space

### Epistemological Foundations

Our approach synthesizes three major paradigms in artificial intelligence:

1. **Connectionism** (Rumelhart & McClelland, 1986)
   - Knowledge as distributed representations in neural networks
   - Learning through gradient descent and backpropagation
   - Handles high-dimensional sensory data

2. **Symbolism** (Newell & Simon, 1976)
   - Knowledge as explicit rules and logical structures
   - Learning through hypothesis generation and testing
   - Enables compositional generalization

3. **Probabilistic Programming** (Tenenbaum et al., 2011)
   - Knowledge as generative models with uncertainty quantification
   - Learning through Bayesian inference
   - Captures human-like concept learning

### Philosophical Commitment: Pragmatic Pluralism

We reject **monism** (the view that intelligence requires only neural networks OR only symbolic reasoning) in favor of **pragmatic pluralism**: different cognitive tasks require different computational mechanisms, and intelligent systems must orchestrate multiple complementary processes.

**Central Thesis**: *Human-level visual concept learning emerges from the dynamic interaction between rapid pattern recognition (System 1) and deliberate symbolic reasoning (System 2), mediated by meta-learned arbitration policies.*

---

## 🏗️ Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph Input
        A[Support Set: 6 Positive + 6 Negative Images]
        B[Query Image]
    end
    
    subgraph System1[System 1: Neural]
        C[Pretrained ResNet Backbone]
        D[Task-Specific Classifier Head]
        E[Confidence Estimation]
    end
    
    subgraph System2[System 2: Bayesian]
        F[Feature Extraction: Image → 28D]
        G[Hypothesis Generation: Rules]
        H[Bayesian Likelihood Scoring]
    end
    
    subgraph Arbitration[Meta-Learned Arbitration]
        I[Dynamic Weight Adjustment]
        J[Weight Cap Constraint]
        K[Weighted Blending]
    end
    
    subgraph Output
        L[Final Prediction: Positive/Negative]
    end
    
    A --> C
    B --> C
    A --> F
    B --> F
    
    C --> D
    D --> E
    
    F --> G
    G --> H
    
    E --> I
    H --> I
    
    I --> J
    J --> K
    
    K --> L
    
    style System1 fill:#e1f5ff
    style System2 fill:#fff4e1
    style Arbitration fill:#f0e1ff
```

### Detailed Architecture

#### **System 1: Neural Pattern Recognition**

**Architecture**: ResNet-15 (feature maps: 32-64-128-256-512)
- **Backbone**: 5 residual blocks with batch normalization
- **Classifier Head**: 2-layer MLP (512 → 128 → 2)
- **Pretraining**: Self-supervised rotation prediction (4-way classification)

**Training Process**:
```python
# 1. Pretraining Phase (100 epochs on 111,600 train images)
for rotation in [0°, 90°, 180°, 270°]:
    rotated_image = rotate(image, rotation)
    predicted_rotation = model(rotated_image)
    loss = CrossEntropy(predicted_rotation, rotation_label)
    
# 2. Episodic Fine-Tuning (per task)
# CRITICAL: Reset classifier head while freezing backbone
model.fc2.reset_parameters()  # Prevent cross-episode contamination
for image, label in support_set:
    prediction = model(image)
    loss = CrossEntropy(prediction, label)
    update(model.fc2.parameters())  # Only update head
```

**Confidence Estimation**:
$$
\text{conf}_{\text{S1}} = \max(\text{softmax}(\mathbf{z})) \quad \text{where } \mathbf{z} \in \mathbb{R}^2
$$

$$
\text{margin}_{\text{S1}} = |P(\text{pos}) - P(\text{neg})| = |z_{\text{pos}} - z_{\text{neg}}|
$$

#### **System 2: Bayesian Rule Induction**

**Feature Space**: 28-dimensional vector
- **Pixel features** (16-D): Color statistics, edge density, texture descriptors
- **Sequence features** (12-D): Stroke counts, shape complexity, spatial moments

**Hypothesis Space**: Compositional rules
```
Primitive Rules:
  - Threshold: f_j ≥ θ
  - Comparison: (f_i - f_j) ≥ θ

Compositional Rules:
  - Conjunction: R1 ∧ R2 ∧ ... ∧ Rk
  - Disjunction: R1 ∨ R2 ∨ ... ∨ Rk
```

**Bayesian Inference**:

Given support set $\mathcal{D} = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_{12}, y_{12})\}$, find rule $r$ that maximizes:

$$
P(r | \mathcal{D}) \propto P(\mathcal{D} | r) \cdot P(r)
$$

**Likelihood Model**:
$$
P(\mathcal{D} | r) = \prod_{i=1}^{12} \begin{cases}
1 - \epsilon & \text{if } r(\mathbf{x}_i) = y_i \\
\epsilon & \text{otherwise}
\end{cases}
$$

where $\epsilon = 0.01$ is the noise rate.

**Prior**: Prefer simpler rules (Occam's Razor)
$$
P(r) \propto \exp(-\lambda \cdot \text{complexity}(r))
$$

**Prediction Confidence**:
$$
\text{margin}_{\text{S2}} = |P(\text{pos} | r^*, \mathbf{x}_{\text{query}}) - 0.5| \times 2
$$

where $r^*$ is the maximum a posteriori (MAP) rule.

#### **Meta-Learned Arbitration**

**Core Innovation**: Adaptive weight balancing with saturation prevention

**Weight Update Rule** (per episode):
$$
w_1^{(t+1)} = w_1^{(t)} \cdot \exp(-\beta \cdot \mathcal{L}_{\text{CE}}(p_1, y))
$$

$$
w_2^{(t+1)} = w_2^{(t)} \cdot \exp(-\beta \cdot \mathcal{L}_{\text{CE}}(p_2, y))
$$

where:
- $\mathcal{L}_{\text{CE}}(p, y) = -[y \log p + (1-y) \log(1-p)]$ is binary cross-entropy
- $\beta = 2.0$ is the learning rate
- $y \in \{0, 1\}$ is the true label

**Normalization + Cap Constraint**:
$$
\tilde{w}_1 = \frac{w_1}{w_1 + w_2}, \quad \tilde{w}_2 = \frac{w_2}{w_1 + w_2}
$$

$$
w_2^{\text{final}} = \min(\tilde{w}_2, w_{\max}), \quad w_1^{\text{final}} = 1 - w_2^{\text{final}}
$$

where $w_{\max} = 0.95$ prevents System 2 saturation.

**Final Prediction**:
$$
p_{\text{final}} = w_1^{\text{final}} \cdot p_1 + w_2^{\text{final}} \cdot p_2
$$

$$
\hat{y} = \begin{cases}
\text{positive} & \text{if } p_{\text{final}} > 0.5 \\
\text{negative} & \text{otherwise}
\end{cases}
$$

---

## 🔬 Mathematical Formulation

### Problem Definition

**Few-Shot Binary Classification**: Given support set $\mathcal{S} = \mathcal{S}^+ \cup \mathcal{S}^-$ where:
- $\mathcal{S}^+ = \{(\mathbf{x}_1^+, \ldots, \mathbf{x}_6^+)\}$: 6 positive examples
- $\mathcal{S}^- = \{(\mathbf{x}_1^-, \ldots, \mathbf{x}_6^-)\}$: 6 negative examples

**Goal**: Predict label $y_q \in \{0, 1\}$ for query $\mathbf{x}_q$

### System 1: Neural Network

**Forward Pass**:
$$
\mathbf{h} = f_{\theta}(\mathbf{x}; \phi) \quad \text{(backbone with frozen weights } \phi \text{)}
$$

$$
\mathbf{z} = g_{\psi}(\mathbf{h}) \quad \text{(classifier head with learnable weights } \psi \text{)}
$$

$$
p_1 = \text{softmax}(\mathbf{z})_{\text{pos}}
$$

**Episodic Adaptation**:
$$
\psi^{(t+1)} = \psi^{(t)} - \eta \nabla_{\psi} \sum_{(\mathbf{x}, y) \in \mathcal{S}} \mathcal{L}_{\text{CE}}(g_{\psi}(f_{\theta}(\mathbf{x}; \phi)), y)
$$

**CRITICAL BUG FIX**: Reset $\psi$ before each episode to prevent contamination:
$$
\psi^{(0)} \sim \mathcal{N}(0, \sigma^2) \quad \text{(Xavier initialization)}
$$

### System 2: Bayesian Inference

**Feature Extraction**:
$$
\mathbf{v} = \Phi(\mathbf{x}) \in \mathbb{R}^{28}
$$

**Rule Evaluation**:
$$
r(\mathbf{v}) = \begin{cases}
1 & \text{if } v_j \geq \theta \\
0 & \text{otherwise}
\end{cases}
\quad \text{(threshold rule)}
$$

$$
r(\mathbf{v}) = \mathbb{I}[v_i - v_j \geq \theta] \quad \text{(comparison rule)}
$$

**Posterior Computation**:
$$
P(r | \mathcal{S}) = \frac{P(\mathcal{S} | r) P(r)}{\sum_{r' \in \mathcal{R}} P(\mathcal{S} | r') P(r')}
$$

**MAP Estimate**:
$$
r^* = \arg\max_{r \in \mathcal{R}} P(r | \mathcal{S})
$$

**Prediction**:
$$
p_2 = P(y_q = 1 | r^*, \mathbf{v}_q) = \begin{cases}
1 - \epsilon & \text{if } r^*(\mathbf{v}_q) = 1 \\
\epsilon & \text{otherwise}
\end{cases}
$$

### Meta-Learning Dynamics

**Weight Evolution** (across $T$ episodes):
$$
\mathbf{w}^{(0)} = [0.5, 0.5]^T
$$

$$
\mathbf{w}^{(t+1)} = \text{normalize}\left(\mathbf{w}^{(t)} \odot \exp(-\beta \mathbf{L}^{(t)})\right)
$$

where $\mathbf{L}^{(t)} = [\mathcal{L}_{\text{CE}}(p_1^{(t)}, y^{(t)}), \mathcal{L}_{\text{CE}}(p_2^{(t)}, y^{(t)})]^T$

**Cap-Constrained Projection**:
$$
\Pi_{[0,w_{\max}]}(\tilde{w}_2) = \min(\max(\tilde{w}_2, 0), w_{\max})
$$

$$
w_2^{\text{final}} = \Pi_{[0,w_{\max}]}(\tilde{w}_2), \quad w_1^{\text{final}} = 1 - w_2^{\text{final}}
$$

---

## 🔧 Implementation Details

### Self-Supervised Pretraining

**Objective**: Rotation prediction (4-way classification)

**Dataset**: 111,600 images from train split
- 9,300 episodes × 12 images/episode = 111,600 images
- Split: 55,800 positive + 55,800 negative

**Augmentation Pipeline**:
```python
for image in train_images:
    for rotation in [0, 90, 180, 270]:
        rotated = rotate(image, rotation)
        label = rotation // 90  # 0, 1, 2, 3
        batch.append((rotated, label))
```

**Training Configuration**:
- Epochs: 100
- Batch size: 256
- Optimizer: SGD (lr=0.001, momentum=0.9, weight_decay=5e-4)
- Workers: 4
- GPU: NVIDIA RTX 3050 Ti (4GB)
- Time: ~12 hours

### Episodic Evaluation

**Per-Episode Protocol**:
1. **Initialize**: Reset S1 classifier head, reset S2 hypothesis space
2. **Fit**: Train S1 on support set (frozen backbone), S2 finds MAP rule
3. **Predict**: Evaluate query with weighted ensemble
4. **Update**: Adjust weights based on prediction correctness

**Weight Update Mechanics**:
```python
# Example: Correct prediction with high confidence
p1 = 0.95, y = 1  # S1 correct
p2 = 0.99, y = 1  # S2 correct

loss1 = -log(0.95) = 0.051
loss2 = -log(0.99) = 0.010

w1 *= exp(-2.0 * 0.051) = 0.903
w2 *= exp(-2.0 * 0.010) = 0.980

# After normalization
w1 = 0.479, w2 = 0.521

# After cap (max_w2 = 0.95)
w2_final = min(0.521, 0.95) = 0.521
w1_final = 1 - 0.521 = 0.479
```

### Auto-Cap Hyperparameter Tuning

**Validation-Based Cap Selection**:
```python
cap_grid = [0.95, 0.97, 0.98, 0.99]
best_cap = None
best_acc = 0.0

for cap in cap_grid:
    solver = HybridDualSystem(max_system2_weight=cap)
    acc = evaluate(solver, val_split, limit=200)
    if acc > best_acc:
        best_acc = acc
        best_cap = cap

# Use best_cap for test evaluation
```

**Per-Split Tuning** (optional, label as "val-tuned per split"):
- test_ff → validate on val_ff → select cap_ff
- test_bd → validate on val_bd → select cap_bd
- test_hd → validate on val_hd → select cap_hd

---

## 📈 Experimental Results

### Main Results (Global Auto-Cap)

**Configuration**: max_system2_weight=0.95 (selected on val split)

| Split | Episodes | Accuracy | Margin | Tie Rate | S2 Usage |
|-------|----------|----------|--------|----------|----------|
| **test_ff** | 600 | **100.0%** | 0.688 | 0.0% | 100.0% |
| **test_bd** | 480 | **92.7%** | 0.467 | 0.0% | 100.0% |
| **test_hd_comb** | 400 | **73.0%** | 0.171 | 0.0% | 100.0% |
| **test_hd_novel** | 320 | **73.4%** | 0.219 | 0.0% | 100.0% |

### Comparison with SOTA Baselines

**Performance Gains Over Previous SOTA** (Meta-Baseline-PS):

| Split | Ours | SOTA | Δ | Improvement |
|-------|------|------|---|-------------|
| test_ff | 100.0% | 68.2% | **+31.8%** | +46.6% relative |
| test_bd | 92.7% | 75.7% | **+17.0%** | +22.5% relative |
| test_hd_comb | 73.0% | 67.4% | **+5.6%** | +8.3% relative |
| test_hd_novel | 73.4% | 71.5% | **+1.9%** | +2.7% relative |

**Comparative Analysis**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Performance by Model Family                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  100% ┤                        ████ (Ours)                          │
│       │                    ████                                     │
│   90% ┤                ████                                         │
│       │            ████                                             │
│   80% ┤        ████                                                 │
│       │    ████    ████ (Meta-PS)                                   │
│   70% ┤████        ████ ████ (Meta-MoCo)                            │
│       │            ████ ████ ████ (ProtoNet)                        │
│   60% ┤                ████ ████ ████ (MetaOptNet)                  │
│       │                    ████ ████ ████ (SNAIL)                   │
│   50% ┤                        ████ ████ ████ (WReN/CNN)            │
│       └─────────────────────────────────────────────────────────────┤
│          test_ff  test_bd  test_hd_comb  test_hd_novel             │
└─────────────────────────────────────────────────────────────────────┘
```

### Ablation Studies

**Coming Soon**: See `scripts/ablation_studies/` for:
- No pretraining baseline
- System 1 only
- System 2 only
- Different weight cap values
- No weight cap (saturation analysis)

### Error Analysis

**test_hd_comb** (73.0% accuracy, 108/400 failures):
- **Pattern**: Failures concentrated on subtle abstract attributes
- **Example**: `self_transposed` vs `symmetric` confusion
- **Hypothesis**: 28-D feature space insufficient for fine-grained abstraction

**test_hd_novel** (73.4% accuracy, 85/320 failures):
- **Pattern**: Slightly better than test_hd_comb (novel attribute generalization)
- **Insight**: System 2 can extrapolate from similar attributes

**Gap to Human Performance**:
- test_ff: **-0%** (surpassed human experts!)
- test_bd: **-6.6%** (humans 99.3%)
- test_hd_comb: **-17.7%** (humans 90.7%)
- test_hd_novel: **-17.3%** (humans 90.7%)

**Interpretation**: Neural pattern recognition (S1) excels at perceptual tasks (ff, bd), but abstract reasoning (hd) still benefits from richer symbolic representations.

---

## 🚀 Usage

### Installation

```bash
# Clone repository
git clone https://github.com/Ayushman125/Bongard-Solver.git
cd Bongard-Solver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**1. Download BONGARD-LOGO Dataset**:

The official BONGARD-LOGO dataset is available at: https://github.com/NVlabs/Bongard-LOGO

```bash
# Clone the official repository
git clone https://github.com/NVlabs/Bongard-LOGO.git

# Extract the ShapeBongard_V2 dataset
cd Bongard-LOGO
# Follow instructions in their README to generate/download the dataset

# Copy to our project structure
cp -r Bongard-LOGO/data/ShapeBongard_V2 /path/to/BongardSolver/data/raw/

# Or manually download and place in:
# data/raw/ShapeBongard_V2/{train,val,test_ff,test_bd,test_hd_comb,test_hd_novel}/
```

**Alternative**: Direct download from official source:
```bash
# Create data directory
mkdir -p data/raw

# Download dataset (if direct link is available from NVlabs/Bongard-LOGO releases)
# Check https://github.com/NVlabs/Bongard-LOGO/releases for direct download links

# Expected structure:
# data/raw/ShapeBongard_V2/
# ├── train/          # Training episodes
# ├── val/            # Validation episodes  
# ├── test_ff/        # Free-form test (our focus: 100% accuracy)
# ├── test_bd/        # Basic test (92.7% accuracy)
# ├── test_hd_comb/   # Human-designed combined (73.0% accuracy)
# └── test_hd_novel/  # Human-designed novel (73.4% accuracy)
```

**2. Run Pretrained Model** (uses checkpoint):
```bash
python run_experiment.py \
    --mode hybrid_image \
    --pretrain \
    --auto-cap \
    --splits test_ff test_bd test_hd_comb test_hd_novel \
    --log-level INFO
```

**3. Train from Scratch**:
```bash
# Pretrain backbone (if checkpoint doesn't exist)
python run_experiment.py \
    --mode hybrid_image \
    --pretrain \
    --pretrain-epochs 100 \
    --pretrain-batch-size 256 \
    --splits test_ff \
    --log-level INFO
```

**4. Ablation Studies**:
```bash
# System 1 only
bash scripts/ablation_studies/run_system1_only.sh

# System 2 only
bash scripts/ablation_studies/run_system2_only.sh

# No pretraining
bash scripts/ablation_studies/run_no_pretrain.sh

# Different caps
bash scripts/ablation_studies/run_cap_sweep.sh
```

### Configuration

**Command-Line Arguments**:
```
--mode {hybrid_image}           Solver mode
--pretrain                      Enable SSL pretraining
--pretrain-epochs INT           Pretraining epochs (default: 30)
--pretrain-batch-size INT       Batch size (default: 256)
--max-system2-weight FLOAT      Weight cap (default: 0.995)
--auto-cap                      Auto-select cap on validation
--auto-cap-per-split            Per-split cap tuning (label as val-tuned)
--cap-grid FLOATS               Cap candidates (default: 0.95,0.97,0.98,0.99)
--splits SPLITS                 Test splits to evaluate
--log-level {DEBUG,INFO,WARN}   Logging verbosity
```

---

## 📁 Project Structure

```
BongardSolver/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── run_experiment.py                 # Main evaluation script
│
├── bayes_dual_system/                # Core solver implementations
│   ├── __init__.py
│   ├── system1_nn.py                 # Neural System 1 (ResNet + SSL)
│   ├── system2_bayes_image.py        # Bayesian System 2 (rule induction)
│   ├── dual_system_hybrid.py         # Arbitration & blending
│   ├── data.py                       # Dataset loading
│   ├── features.py                   # 28-D feature extraction
│   └── types.py                      # Data structures
│
├── scripts/                          # Execution scripts
│   └── ablation_studies/             # Ablation study scripts
│       ├── run_system1_only.sh
│       ├── run_system2_only.sh
│       ├── run_no_pretrain.sh
│       └── run_cap_sweep.sh
│
├── checkpoints/                      # Model checkpoints
│   └── pretrain_backbone_ssl_rotation.pt
│
├── logs/                             # Experiment logs
│   ├── progress.log
│   └── metrics/                      # JSON/CSV results
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # Detailed architecture
│   ├── THEORY.md                     # Mathematical theory
│   ├── RESULTS.md                    # Full experimental results
│   └── images/                       # Figures and diagrams
│
└── data/                             # Dataset
    └── raw/
        └── ShapeBongard_V2/          # Download from official source
```

---

## 🧪 Reproducibility

### Hardware Requirements

- **GPU**: NVIDIA RTX 3050 Ti (4GB VRAM) or better
- **RAM**: 16GB+ system memory
- **Storage**: 10GB for dataset + checkpoints

### Software Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

### Exact Reproduction

**Full Pipeline**:
```bash
# 1. Pretrain backbone (12 hours on RTX 3050 Ti)
python run_experiment.py \
    --mode hybrid_image \
    --pretrain \
    --pretrain-epochs 100 \
    --pretrain-batch-size 256 \
    --pretrain-workers 4 \
    --splits test_ff \
    --log-level INFO

# 2. Evaluate with auto-cap (22 minutes)
python run_experiment.py \
    --mode hybrid_image \
    --pretrain \
    --auto-cap \
    --cap-grid 0.95,0.97,0.98,0.99 \
    --cap-split val \
    --cap-limit 200 \
    --splits test_ff test_bd test_hd_comb test_hd_novel \
    --use-programs \
    --arbitration-strategy always_blend \
    --log-level INFO
```

**Expected Output**:
```
test_ff:      1.0000 (600/600)
test_bd:      0.9271 (445/480)
test_hd_comb: 0.7300 (292/400)
test_hd_novel: 0.7344 (235/320)
```

### Random Seed Control

Results may vary slightly due to:
- Episodic S1 classifier initialization (Xavier)
- Stochastic gradient descent
- CUDA non-determinism

For reproducibility, set:
```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{bongard-dual-system-2026,
  title={Hybrid Dual-System Architecture for Human-Level Visual Concept Learning},
  author={Ayushman Saini},
  year={2026},
  howpublished={\url{https://github.com/Ayushman125/Bongard-Solver}},
  note={Achieves SOTA on BONGARD-LOGO benchmark with 100\% accuracy on free-form shapes}
}
```

**Original BONGARD-LOGO Benchmark**:
```bibtex
@inproceedings{nie2020bongard,
  title={Bongard-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning},
  author={Nie, Weili and Yu, Zhiding and Mao, Lei and Patel, Ankit B and Zhu, Yuke and Anandkumar, Animashree},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```

---

## 🤝 Contributing

We welcome contributions! Areas of interest:
- [ ] Program synthesis integration (like Meta-Baseline-PS)
- [ ] Improved feature representations for abstract reasoning
- [ ] Hierarchical Bayesian rule learning
- [ ] Attention mechanisms for context-dependent perception
- [ ] Curriculum learning strategies

See `CONTRIBUTING.md` for guidelines.

---

## 📜 License

MIT License - see `LICENSE` for details.

---

## 🙏 Acknowledgments

- **BONGARD-LOGO Dataset**: Nie et al. (NeurIPS 2020)
- **Dual-Process Theory**: Kahneman (2011), Evans & Stanovich (2013)
- **Meta-Learning**: Finn et al. (MAML), Chen et al. (Meta-Baseline)
- **Self-Supervised Learning**: He et al. (MoCo)
- **Bayesian Program Learning**: Lake et al. (2015), Tenenbaum et al. (2011)

---

## 📧 Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Project Issues](https://github.com/Ayushman125/Bongard-Solver/issues)
- **GitHub Discussions**: [Discussions](https://github.com/Ayushman125/Bongard-Solver/discussions)
- **Email**: ayushmansaini120@gmail.com

---

<div align="center">
  <p><i>"The test of a first-rate intelligence is the ability to hold two opposed ideas in mind at the same time and still retain the ability to function."</i></p>
  <p>— F. Scott Fitzgerald</p>
</div>
