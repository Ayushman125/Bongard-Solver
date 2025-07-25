# Bongard-Solver: Phase 0 - System-1 Abstraction Layer

This README provides a comprehensive overview of Phase 0 of the Bongard-Solver project, which introduces the System-1 Abstraction Layer (S1-AL). This layer serves as the foundational perceptual engine, extracting low-level visual features and generating rapid heuristic guesses to guide the more complex reasoning modules.

---

## 1. Overview of Phase 0

Phase 0 establishes a continuously running, heuristic "System-1" inspired by dual-process theories of cognition. Its primary role is to provide a fast, intuitive, and low-level analysis of Bongard problems, transforming raw pixel data into a structured format that higher-level "System-2" processes can utilize.

### Key Objectives:
- **Feature Extraction**: To extract a rich set of coarse, low-level visual attributes (e.g., stroke count, curvature, symmetry) from each image in a Bongard problem.
- **Relational Computation**: To compute pairwise relational cues between objects, such as size comparisons, spatial adjacency, and containment.
- **Heuristic Guessing**: To emit a set of fast, plausible hypotheses about the underlying rule of the puzzle, complete with confidence scores.
- **Self-Supervision**: To create a feedback loop where the S1-AL can learn and adapt over time by comparing its initial guesses to the final, confirmed rule.

---

## 2. Architecture

The S1-AL is designed as a modular component that integrates seamlessly into the main processing pipeline.

```text
+-------------------+      +----------------------+      +--------------------+
|  Raw Bongard      |      |   S1-AL Module       |      | Downstream Layers  |
|  Images           +----->|  (system1_al.py)     +----->| (Grounder, CoT, …) |
+-------------------+      +----------------------+      +--------------------+
         |                          |
         |                          +--> Fuzzy Heuristic Engine
         |                                 (utils/fuzzy_tree.py)
         |
         +-- Replay Buffer Update <-- Final Rule Outcome
             (data/system1_replay.pkl)
```

### Core Components:
- **`src/system1_al.py`**: The main module containing the `System1AbstractionLayer` class. It orchestrates feature extraction, relation computation, and heuristic generation.
- **`src/utils/fuzzy_tree.py`**: A utility for the Fuzzy Decision Tree model, which is responsible for generating fast heuristic guesses. It includes methods for prediction, loading, saving, and a placeholder for self-supervised updates.
- **`data/system1_replay.pkl`**: A replay buffer that stores "surprising" events—instances where the S1-AL's heuristics significantly diverged from the final puzzle solution. This buffer is used for periodic retraining.
- **`tests/test_system1.py`**: A suite of unit tests to ensure the correctness, robustness, and performance of the S1-AL.

---

## 3. How to Run and Test

### 3.1. Prerequisites

Ensure you have Python 3.11+ and the required libraries installed. The primary dependencies for this phase are:
- `scikit-image`
- `networkx`
- `numpy`
- `scipy`
- `joblib`

You can install them using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3.2. Initial Setup

Before running the system for the first time, you need to generate the initial fuzzy tree model. This can be done by running the `fuzzy_tree.py` script directly:

```bash
python src/utils/fuzzy_tree.py
```

This will create a `data` directory (if it doesn't exist) and save the initial `fuzzy_tree.pkl` model inside it.

### 3.3. Running the Unit Tests

To verify that the S1-AL is functioning correctly, run the unit tests from the root directory of the project:

```bash
pytest tests/test_system1.py
```
or
```bash
python -m unittest tests/test_system1.py
```

The tests cover:
- **Attribute Correctness**: Verifies that features like area and circularity are computed correctly for synthetic shapes.
- **Relational Cues**: Checks that relationships (e.g., size comparison) are accurately identified.
- **Heuristic Output**: Ensures the fuzzy tree produces valid, structured hypotheses.
- **Performance**: Asserts that the processing time for a full 12-image puzzle remains within an acceptable threshold (e.g., < 200ms).
- **JSON Serialization**: Confirms that the output bundle is correctly structured and can be serialized to JSON without errors.
- **Self-Supervision Loop**: Tests the logic for adding samples to the replay buffer.

### 3.4. Example Usage

The `system1_al.py` module can be run as a standalone script to demonstrate its functionality. It will process two synthetic images (a square and a circle), print the resulting JSON feature bundle, and simulate a self-supervision step.

```bash
python src/system1_al.py
```

#### Expected Output:
The script will print a JSON object to the console, structured as follows:

```json
{
  "problem_id": "test_problem_01",
  "timestamp": "...",
  "duration_ms": ...,
  "images": [
    {
      "image_id": "img_0",
      "attrs": {
        "stroke_count": ...,
        "area": 3600,
        ...
      },
      "relations": { ... }
    },
    ...
  ],
  "heuristics": [
    {
      "rule": "strong_vertical_symmetry",
      "confidence": 0.85
    },
    ...
  ]
}
```

---

## 4. Integration with the Main Pipeline

The S1-AL is designed to be easily integrated into the main `professional_pipeline.py` or `pipeline_workers.py`.

**Step 1: Initialization**
At worker startup, create an instance of the `System1AbstractionLayer`.

```python
from src.system1_al import System1AbstractionLayer

# At worker startup:
s1_al = System1AbstractionLayer(
    fuzzy_model_path="data/fuzzy_tree.pkl",
    replay_path="data/system1_replay.pkl"
)
```

**Step 2: Processing**
For each puzzle, process the binary images to get the S1 feature bundle.

```python
# Assuming `bin_images` is a list of 12 binary numpy arrays
s1_output = s1_al.process(bin_images, problem_id=current_puzzle_id)

# Store the output in a shared context for downstream modules
context["s1_output"] = s1_output
```

**Step 3: Self-Supervision**
After the final rule for the puzzle has been determined, trigger the self-supervision mechanism.

```python
# Assuming `final_rule` is the confirmed solution and `true_labels` is a list
# of 0s and 1s indicating if an image belongs to the left or right set.
s1_al.self_supervise(s1_output, true_labels)

# Optionally, trigger a periodic update of the fuzzy model
if should_update_model(): # Based on time or number of puzzles solved
    s1_al.periodic_update()
```

---

## 5. Future Work

Phase 0 lays the critical groundwork for more advanced reasoning. Future phases will build directly on the S1-AL's outputs:
- **Phase 1 (Symbolic Grounding)**: Will consume the `attrs` and `relations` to ground abstract symbols (e.g., `shape`, `size`, `inside`) in concrete visual data.
- **Phase 2 (Causal Reasoning)**: Will use the `heuristics` as initial hypotheses to be tested and refined by a causal engine.
- **Continuous Improvement**: The self-supervision loop will ensure that the S1-AL becomes more accurate and efficient over time as it processes more puzzles.
