# Bongard-Solver: Data Preparation Pipeline

This document provides a comprehensive guide to the data preparation pipeline for the Bongard-Solver project. The pipeline is designed to be robust, auditable, and fully automated, transforming raw Bongard-LOGO data into feature-rich datasets ready for model training and evaluation.

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Directory Structure](#3-directory-structure)
4. [Step-by-Step Workflow](#4-step-by-step-workflow)
    - [Step 1: Initial Setup](#step-1-initial-setup)
    - [Step 2: Attribute & Feature Extraction](#step-2-attribute--feature-extraction)
    - [Step 3: Concept Induction](#step-3-concept-induction)
    - [Step 4: Hard Negative Mining](#step-4-hard-negative-mining)
    - [Step 5: Verification & Auditing](#step-5-verification--auditing)
5. [Key Components & Modules](#5-key-components--modules)
6. [Output Files Explained](#6-output-files-explained)
    - [`derived_labels.json`](#derived_labelsjson)
    - [`concepts_auto.yaml`](#concepts_autoyaml)
    - [`hard_negatives.txt`](#hard_negativestxt)
    - [`flagged_cases.txt`](#flagged_casestxt)
7. [Example Usage](#7-example-usage)

---

## 1. Overview

The data preparation pipeline is the foundation of the Bongard-Solver project. Its primary purpose is to process the raw Bongard-LOGO dataset and generate three critical assets:

1. **Derived Labels:** A structured JSON file containing rich physics, geometry, and semantic attributes for every puzzle sample.
2. **Concept Registry:** An automatically induced and cached set of rules that define the core concept for each puzzle.
3. **Hard Negatives:** A diverse set of challenging adversarial examples generated through multiple advanced strategies to improve model robustness.

The entire process is designed for determinism, scalability, and full auditability, ensuring high-quality data for all downstream modules.

## 2. Prerequisites

### System Requirements

- Python 3.9+
- A standard development environment (Linux, macOS, or Windows).

### Python Dependencies

Install all required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Key libraries include:

- `numpy`
- `torch`
- `shapely`
- `pymunk`
- `tqdm`
- `pyyaml`
- `opencv-python`

## 3. Directory Structure

The data pipeline operates on and creates files within the `data/` directory.

```
BongordSolver/
└── data/
    ├── raw/                         # INPUT: Place raw Bongard-LOGO JSON files here.
    |
    ├── derived_labels.json          # OUTPUT: Generated physics/geometry/label attributes.
    ├── concepts_auto.yaml           # OUTPUT: Auto-induced concept predicates for each puzzle.
    ├── hard_negatives.txt           # OUTPUT: Generated hard negative/adversarial samples.
    └── flagged_cases.txt            # OUTPUT: Samples flagged for manual review.
```

---

## 4. Step-by-Step Workflow

### Step 1: Initial Setup

Place the unmodified Bongard-LOGO dataset files into the `data/raw/` directory. The pipeline scripts will read from this location.

### Step 2: Attribute & Feature Extraction

This step computes a rich set of features for every sample in the dataset.

**Command:**

```bash
python scripts/logo_to_shape.py --input-dir data/raw/ --output data/derived_labels.json
```

**Process:**

- The `logo_to_shape.py` script iterates through all problems in `data/raw/`.
- For each sample, it uses modules from `src/data_pipeline/` (e.g., `physics_infer.py`) to calculate attributes like:
    - **Geometric:** Centroid, area, perimeter, convexity, symmetry.
    - **Physics-based:** Moment of inertia, stability.
    - **Meta-labels:** Problem type, source, etc.
- The final, structured data is saved to `data/derived_labels.json`.

### Step 3: Concept Induction

This step automatically discovers and caches the core logical rule (the "concept") for each puzzle.

**Process:**

- This step is **run automatically** as part of the hard negative mining pipeline (Step 4).
- The `Concept Registry` (`src/concepts/registry.py`) is invoked.
- It loads the features from `derived_labels.json`.
- For each problem, it finds a simple predicate (e.g., `is_convex`, `num_straight_lines > 2`) that perfectly separates the positive (`category_1`) and negative (`category_0`) examples.
- The discovered predicate is saved to `data/concepts_auto.yaml`. This ensures that every problem has a defined, auditable concept.

### Step 4: Hard Negative Mining

This step generates challenging adversarial examples to train a more robust solver.

**Command:**

```bash
python scripts/generate_hard_negatives.py --input-dir data/raw/ --output data/hard_negatives.txt --parallel 8
```

**Process:**

- The `generate_hard_negatives.py` script orchestrates a multi-strategy mining process.
- It uses the `Concept Registry` from Step 3 to understand the rule for each puzzle.
- It applies five advanced strategies to mutate positive samples into label-flipping "hard negatives":
    1.  **Deterministic Concept Inversions**
    2.  **Metamorphic Affine Inversion Testing (MAIT)**
    3.  **Procedural Shape Perturbation Ensemble (PSPE)**
    4.  **Geometry-Aware GAN Mining (GAGAN-HNM)**
    5.  **Hyperbolic Hard-Negative Mixup (HHNM)**
- All generated negatives are validated for geometric integrity and deduplicated.
- The final set of hard negatives is saved to `data/hard_negatives.txt`.

### Step 5: Verification & Auditing

This optional step allows for manual spot-checking of the generated data to ensure quality.

**Command:**

```bash
python scripts/verify_annotations.py --input data/derived_labels.json --output data/flagged_cases.txt
```

**Process:**

- The `verify_annotations.py` script loads the generated labels.
- It runs a series of checks to flag potential issues, such as:
    - Geometrically degenerate polygons (e.g., self-intersecting).
    - Mismatches between labels and visual properties.
- A list of problematic sample IDs is saved to `data/flagged_cases.txt` for manual review.

---

## 5. Key Components & Modules

- **`scripts/logo_to_shape.py`**: The main CLI for running the attribute extraction pipeline.
- **`scripts/generate_hard_negatives.py`**: The main CLI for orchestrating the multi-strategy hard negative mining.
- **`scripts/verify_annotations.py`**: The CLI for auditing and flagging potentially erroneous data.
- **`src/data_pipeline/`**: A package containing the core logic for feature extraction.
    - `physics_infer.py`: Extracts all shape and physics-based features.
    - `attributes.py`: Assigns categorical and meta-labels.
- **`src/concepts/`**: A package for the concept registry and auto-induction logic.
- **`src/hard_negative/`**: A package containing the implementation of the advanced hard negative mining strategies.

---

## 6. Output Files Explained

### `derived_labels.json`

A JSON file containing a list of objects, where each object represents a Bongard puzzle.

**Data Structure:**

```json
[
  {
    "problem_id": "logo_001",
    "category_1": [  // List of positive samples
      {
        "sample_id": "p01_1",
        "attributes": {
          "centroid": [32.5, 31.8],
          "area": 512.0,
          "is_convex": true,
          "symmetry": 0.98,
          "num_vertices": 4
        }
      },
      ...
    ],
    "category_0": [  // List of negative samples
      {
        "sample_id": "n01_1",
        "attributes": {
          "centroid": [28.1, 30.4],
          "area": 480.5,
          "is_convex": false,
          "symmetry": 0.45,
          "num_vertices": 7
        }
      },
      ...
    ]
  },
  ...
]
```

### `concepts_auto.yaml`

A YAML file that maps each `problem_id` to its automatically discovered logical predicate.

**Data Structure:**

```yaml
logo_001: "lambda x: x['is_convex']"
logo_002: "lambda x: x['num_vertices'] == 3"
logo_003: "lambda x: x['symmetry'] > 0.9"
...
```

### `hard_negatives.txt`

A text file (likely JSONL format) where each line is a JSON object representing a single hard negative sample.

**Data Structure (per line):**

```json
{
  "original_problem_id": "logo_001",
  "original_sample_id": "p01_1",
  "hard_negative_id": "hn_001",
  "generation_strategy": "affine_inversion",
  "mutator_chain": ["rotate(45)", "scale(1.2)"],
  "difficulty_score": 0.85,
  "attributes": {
    "centroid": [35.2, 33.1],
    "area": 614.4,
    "is_convex": false,
    ...
  }
}
```

### `flagged_cases.txt`

A simple text file listing the `sample_id` of any data point that failed a verification check and requires manual review.

**Data Structure:**

```
p01_5_degen
n07_2_intersect
...
```

---

## 7. Example Usage

Here is a typical sequence of commands to run the entire data preparation pipeline from scratch:

```bash
# 1. Extract features and attributes for all problems
python scripts/logo_to_shape.py --input-dir data/raw/ --output data/derived_labels.json

# 2. Generate hard negatives (this will also create concepts_auto.yaml)
# Use --parallel to leverage multiple CPU cores
python scripts/generate_hard_negatives.py --input-dir data/raw/ --output data/hard_negatives.txt --parallel 8

# 3. (Optional) Verify the generated data and flag cases for review
python scripts/verify_annotations.py --input data/derived_labels.json --output data/flagged_cases.txt
```
