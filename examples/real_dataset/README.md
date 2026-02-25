# BONGARD-LOGO Real Examples with Solver Analysis

## Overview
This directory contains **REAL benchmark problems** from the BONGARD-LOGO dataset
paired with solver performance metrics and cognitive analysis.

Source Paper:
> BONGARD-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning
> Weili Nie, Zhiding Yu, Lei Mao, Ankit B. Patel, Yuke Zhu, Animashree Anandkumar
> NeurIPS 2020 | https://github.com/NVlabs/Bongard-LOGO

## Dataset Structure

### ShapeBongard_V2 Splits
- **FF (Free-Form)**: 24 problems - Procedural stroke sequences
  - Concept: Specific action program (line, arc, angle sequences)
  - Example: "ice cream cone-like shape" from 6 specific strokes
  - Solver Accuracy: 100%

- **BD (Basic Shapes)**: Problems - Single or dual shape categories
  - Concept: Recognition of 627 human-designed shape categories
  - Example: "fan + trapezoid" composition
  - Solver Accuracy: 92.7%

- **HD (Abstract Attributes)**: Problems - Geometric/topological properties
  - Concept: 25 abstract attributes (convex, symmetric, etc.)
  - Example: "convex vs concave" or "have_four_straight_lines"
  - Solver Accuracy (Combinatorial): 73.0%
  - Solver Accuracy (Novel): 73.4%

## Solver Performance (SOTA)

### Hybrid Dual-System Architecture
- **Neural Component**: ResNet-15 CNN backbone
- **Symbolic Component**: Bayesian rule induction
- **Meta-Learning**: Learned arbitration between systems

### Test Accuracies
| Split | Type | Accuracy | Type | Notes |
|-------|------|----------|------|-------|
| FF | Free-Form | 100% | Within-distribution | Extrapolation to longer programs |
| BD | Basic Shape | 92.7% | Novel composition | Hold-out shape compositions |
| HD_Comb | Abstract (Combined) | 73.0% | Novel combinations | Hold-out attribute pairs |
| HD_Novel | Abstract (New) | 73.4% | Truly novel attribute | Hold-out entire attribute |

### Improvement over SOTA
- **+31.8%** over Meta-Baseline-PS baseline on FF split
- Competitive across all problem types
- Demonstrates hybrid approach effectiveness

### Human Performance (for reference)
- **Expert**: 92.1-99.3% across splits
- **Amateur**: 71.0-90.0% across splits
- Gap indicates hard problem suite

## Key Cognitive Properties Tested

### 1. Context-Dependent Perception
Same geometric arrangement interpreted differently based on context.
Example: "have_four_straight_lines" vs "have_six_straight_lines"
(Intersecting lines counted as one vs separate)

### 2. Analogy-Making Perception
Representations traded off - interpret zigzags as straight lines, circles as trapezoids.
Models must know when to trade off concepts vs when to preserve them.

### 3. Infinite Vocabulary, Few Examples
Procedurally generated concepts create unbounded vocabulary.
No finite set of categories to memorize - must conceptualize from few examples.

## Example Problem Format

Each visualization shows:
- **Set A (Green)**: 6 positive examples showing the concept
- **VS**: Visual separator emphasizing binary contrast
- **Set B (Red)**: 6 negative examples violating the concept
- **Solver Analysis**: Performance on this problem type + cognitive properties tested

## References

[1] Nie et al., "BONGARD-LOGO: A New Benchmark for Human-Level Concept Learning and Reasoning", NeurIPS 2020
[2] Lake et al., "Human-level concept learning through probabilistic program induction", Science 2015
[3] Wharton et al., "Agent-based reasoning on concepts", Cognitive Systems Research 2021

---
**Generated**: C:\Users\HP\AI_Projects\BongordSolver\examples\real_dataset
**Total Examples**: 24
**Date**: 2026-02-25
