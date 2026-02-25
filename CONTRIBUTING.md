# Contributing to BongardSolver

Thank you for your interest in contributing to the Hybrid Dual-System Solver! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Submitting Changes](#submitting-changes)
7. [Reporting Bugs](#reporting-bugs)
8. [Feature Requests](#feature-requests)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and constructive in all interactions
- Focus on what is best for the community
- Show empathy towards other contributors
- Accept constructive criticism gracefully

---

## How to Contribute

### Areas for Contribution

We welcome contributions in the following areas:

1. **New System Integrations**:
   - System 3 (program synthesis)
   - System 4 (graph neural networks)
   - Alternative arbitration strategies

2. **Performance Improvements**:
   - GPU optimization
   - Faster feature extraction
   - Efficient rule generation

3. **Documentation**:
   - Tutorial notebooks
   - Additional examples
   - API documentation

4. **Bug Fixes**:
   - See [GitHub Issues](https://github.com/YOUR_USERNAME/BongardSolver/issues)

5. **Benchmarking**:
   - Other few-shot learning datasets
   - Ablation studies
   - Computational efficiency analysis

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork repository on GitHub
# Then clone your fork
git clone https://github.com/Ayushman125/Bongard-Solver.git
cd Bongard-Solver
```

### 2. Create Virtual Environment

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt  # Development tools
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks before each commit:
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking
- `isort`: Import sorting

---

## Coding Standards

### Python Style Guide

Follow **PEP 8** with these specifics:

**1. Formatting**:
```python
# Use black formatter (line length = 100)
black bayes_dual_system/ --line-length 100
```

**2. Imports**:
```python
# Standard library
import os
from typing import List, Tuple

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local
from bayes_dual_system.types import EpisodeData
from bayes_dual_system.features import extract_image_features
```

**3. Type Hints**:
```python
def solve_episode(
    self,
    episode_data: EpisodeData,
    max_system2_weight: float = 0.95
) -> Tuple[int, float]:
    """
    Solve a single episode using hybrid dual-system approach.
    
    Args:
        episode_data: Contains support set, query image, and labels
        max_system2_weight: Cap on System 2 weight (default: 0.95)
    
    Returns:
        Tuple of (prediction, confidence)
    """
    ...
```

**4. Docstrings** (Google Style):
```python
def compute_posteriors(
    rules: List[Rule],
    support_set: List[Tuple[np.ndarray, int]],
    noise_rate: float = 0.01
) -> Dict[Rule, float]:
    """Compute Bayesian posterior probabilities for rule hypotheses.
    
    Uses MAP estimation with Occam's Razor penalty:
    P(r|S) ∝ P(S|r) · exp(-λ·complexity(r))
    
    Args:
        rules: List of candidate rule hypotheses
        support_set: List of (feature_vector, label) pairs
        noise_rate: Probability of mislabeled example (default: 0.01)
    
    Returns:
        Dictionary mapping each rule to its posterior probability
    
    Example:
        >>> rules = [ThresholdRule('edge', 0.5), ComparisonRule('mean', 'std', 0.1)]
        >>> support = [(np.array([0.6, 0.2]), 1), (np.array([0.3, 0.4]), 0)]
        >>> posteriors = compute_posteriors(rules, support)
        >>> best_rule = max(posteriors, key=posteriors.get)
    """
    ...
```

### Git Commit Messages

**Format**:
```
<type>(<scope>): <short summary>

<detailed description>

<footer: references, metrics>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvement
- `refactor`: Code restructuring
- `test`: Add tests
- `chore`: Maintenance

**Example**:
```
feat(system2): Add compositional rule generation

- Implement conjunction (AND) and disjunction (OR) rules
- Support up to 3-way compositions
- Improves test_hd accuracy from 67.0% to 73.0%

Closes #42
```

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_system1.py

# Run with coverage
pytest --cov=bayes_dual_system --cov-report=html
```

### Writing Tests

**1. Unit Tests** (`tests/unit/`):
```python
import pytest
from bayes_dual_system.system1_nn import NeuralSystem1

def test_head_reset():
    """Test that classifier head is properly reset between episodes"""
    system1 = NeuralSystem1(use_pretrain=False)
    
    # Get initial weights
    initial_weights = system1.model.fc2.weight.data.clone()
    
    # Fit on dummy data
    support = torch.randn(12, 1, 128, 128)
    labels = torch.randint(0, 2, (12,))
    system1.fit(support, labels)
    
    # Get weights after fitting
    fitted_weights = system1.model.fc2.weight.data.clone()
    
    # Fit again (should reset)
    system1.fit(support, labels)
    reset_weights = system1.model.fc2.weight.data.clone()
    
    # Check that weights changed after fitting
    assert not torch.equal(initial_weights, fitted_weights)
    
    # Check that weights were reset before second fit
    assert not torch.equal(fitted_weights, reset_weights)
```

**2. Integration Tests** (`tests/integration/`):
```python
def test_full_pipeline():
    """Test complete episode solving pipeline"""
    from bayes_dual_system.dual_system_hybrid import HybridDualSystemSolver
    from bayes_dual_system.data import load_test_split
    
    solver = HybridDualSystemSolver(max_system2_weight=0.95)
    episodes = load_test_split('ff', num_episodes=10)
    
    correct = 0
    for episode in episodes:
        prediction = solver.solve_episode(episode)
        if prediction == episode.query_label:
            correct += 1
    
    accuracy = correct / len(episodes)
    assert accuracy > 0.5  # Should beat random
```

---

## Submitting Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code
- Add tests
- Update documentation

### 3. Run Pre-commit Checks

```bash
# Formatting
black bayes_dual_system/

# Linting
flake8 bayes_dual_system/

# Type checking
mypy bayes_dual_system/

# Run tests
pytest tests/
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat(scope): Add feature X

Detailed description of changes..."
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Open Pull Request

- Go to GitHub
- Click "New Pull Request"
- Select your branch
- Fill out PR template:

```markdown
## Description
Brief summary of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] Commits are descriptive
```

---

## Reporting Bugs

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. Use parameters '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**:
- OS: [e.g. Windows 11]
- Python version: [e.g. 3.9.7]
- PyTorch version: [e.g. 2.0.1]
- CUDA version: [e.g. 11.7]

**Traceback**
```python
Paste full error traceback here
```

**Additional context**
Any other relevant information.
```

---

## Feature Requests

### Feature Request Template

```markdown
**Feature description**
Clear description of the proposed feature.

**Motivation**
Why would this feature be useful?

**Proposed implementation**
How could this be implemented?

**Alternatives considered**
What other approaches did you consider?

**Additional context**
Any other relevant information.
```

---

## Development Workflow Example

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create branch
git checkout -b fix/head-reset-bug

# 3. Make changes
# ... edit files ...

# 4. Run tests
pytest tests/test_system1.py

# 5. Format code
black bayes_dual_system/system1_nn.py
flake8 bayes_dual_system/system1_nn.py

# 6. Commit
git add bayes_dual_system/system1_nn.py tests/test_system1.py
git commit -m "fix(system1): Properly reset classifier head

- Add reset_parameters() call before each episode
- Prevents cross-episode contamination
- Improves test_ff from 87% to 100%

Fixes #123"

# 7. Push
git push origin fix/head-reset-bug

# 8. Open PR on GitHub
```

---

## Questions?

- **GitHub Discussions**: Ask questions, share ideas
- **Email**: bongard.solver@example.com
- **Discord**: [Link to Discord server]

Thank you for contributing! 🚀
