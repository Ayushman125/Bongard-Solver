# Visual Examples: Hybrid Dual-System Solver in Action

This folder contains visual demonstrations of our hybrid dual-system solver tackling BONGARD-LOGO problems, showcasing the concepts it discovers and the reasoning process.

## 🎯 Performance Overview

Our solver achieves **state-of-the-art** performance across all test splits:

| Split | Accuracy | Description | Concepts Tested |
|-------|----------|-------------|-----------------|
| **test_ff** | **100.0%** | Free-form shapes | Infinite vocabulary, procedural strokes |
| **test_bd** | **92.7%** | Basic shapes | Shape category composition |
| **test_hd_comb** | **73.0%** | Abstract concepts (combinatorial) | Novel attribute combinations |
| **test_hd_novel** | **73.4%** | Abstract concepts (novel) | Held-out abstract attributes |

**Improvement over previous SOTA**: +31.8% on free-form concepts

---

## 📁 Example Categories

### 1. Perfect Performance: Free-Form Shapes (100% Accuracy)

Examples where our dual-system architecture achieves perfect accuracy by combining:
- **System 1 (Neural)**: ResNet-15 with SSL rotation pretraining captures perceptual patterns
- **System 2 (Bayesian)**: Rule induction discovers procedural stroke sequences

**View Examples**: [01-free-form-perfect/](01-free-form-perfect/)

### 2. Strong Performance: Basic Shapes (92.7% Accuracy)

Examples demonstrating analogy-making perception:
- Zigzags traded for straight lines
- Circles traded for conceptual trapezoids
- Stroke types ignored, shape categories recognized

**View Examples**: [02-basic-shapes/](02-basic-shapes/)

### 3. Abstract Reasoning: Human-Designed Concepts (73% Accuracy)

Examples requiring abstract concept discovery:
- Convexity, symmetry, topology
- Context-dependent interpretation
- Compositional generalization

**View Examples**: [03-abstract-concepts/](03-abstract-concepts/)

### 4. Dual-System Arbitration in Action

Visual demonstrations of meta-learned arbitration:
- When System 1 (neural) dominates (perceptual tasks)
- When System 2 (rule-based) dominates (abstract tasks)
- Weight distribution across problem types

**View Examples**: [04-arbitration-analysis/](04-arbitration-analysis/)

### 5. Failure Case Analysis

Transparent analysis of remaining challenges:
- Problems where both systems fail
- Edge cases in abstract reasoning
- Generalization limits

**View Examples**: [05-failure-cases/](05-failure-cases/)

---

## 🎨 Visualization Format

Each example follows the BONGARD-LOGO format:

```
Set A (Positive Examples - 6 images)    Set B (Negative Examples - 6 images)
[img1] [img2] [img3]                     [img1] [img2] [img3]
[img4] [img5] [img6]                     [img4] [img5] [img6]

Test Images: [positive ✓] [negative ✓]

Predicted Concept: [concept description]
System 1 Confidence: X.XX
System 2 Confidence: X.XX
Final Decision: ✓ CORRECT / ✗ INCORRECT
```

---

## 🔬 How to Generate Examples

To generate visual examples from your own evaluation:

```bash
# Run evaluation with visualization flag
python run_experiment.py \
    --mode hybrid_image \
    --pretrain \
    --auto-cap \
    --splits test_ff test_bd test_hd_comb test_hd_novel \
    --visualize \
    --save-examples examples/ \
    --num-examples 20

# Generate example gallery
python scripts/generate_example_gallery.py \
    --results-dir logs/ \
    --output-dir examples/
```

---

## 📊 Interactive Comparison with Baselines

Compare our solver against Meta-Baseline, ProtoNet, and WReN:

**View Comparison**: [06-baseline-comparison/](06-baseline-comparison/)

---

## 🧠 Cognitive Properties Demonstrated

Our examples showcase the three core properties from the BONGARD-LOGO benchmark:

### 1. Context-Dependent Perception
Same visual pattern, different interpretation based on context
- Example: [Four vs Six Straight Lines](03-abstract-concepts/context-dependent/)

### 2. Analogy-Making Perception  
Trading off meaningful concepts for other concepts
- Example: [Zigzags as Trapezoids](02-basic-shapes/analogy-making/)

### 3. Few-Shot with Infinite Vocabulary
Learning from 6+6 examples, generalizing to unseen compositions
- Example: [Novel Stroke Sequences](01-free-form-perfect/infinite-vocabulary/)

---

## 📖 Citation

If you use these examples in your work, please cite:

```bibtex
@misc{bongard-dual-system-2026,
  title={Hybrid Dual-System Architecture for Human-Level Visual Concept Learning},
  author={Ayushman Saini},
  year={2026},
  howpublished={\url{https://github.com/Ayushman125/Bongard-Solver}},
  note={Achieves SOTA on BONGARD-LOGO benchmark with 100\% accuracy on free-form shapes}
}
```

---

## 🤝 Contributing Examples

Want to contribute interesting examples? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Particularly interested in:
- Edge cases where the solver exhibits surprising behavior
- Examples demonstrating cognitive properties
- Comparative analyses with human reasoning
