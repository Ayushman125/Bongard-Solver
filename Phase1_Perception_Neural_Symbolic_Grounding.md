# Phase 1: Enhanced Perception & Neural-Symbolic Grounding

## Objective & Scope
Deliver end-to-end solving on the 50-puzzle Bongard-LOGO subset by integrating visual perception, symbol grounding, dynamic DSL evolution, and adversarial mining. This phase extends the KPI dashboard, integration tests, and schema validations to cover all new modules and data flows, maintaining a relaxed performance budget for local hardware (Ryzen 7, 16 GB, RTX 3050 Ti).

---

## Key Enhancements
- **Hybrid Commonsense Fusion:**
  - Load and query ConceptNet-lite in `src/physics_inference.py` and `src/cross_domain_reasoner.py`.
  - Human-loop bootstrapping for new social/intention concepts.
- **Dynamic DSL & Meta-Grammar Stub:**
  - Grammar coverage estimation and diff_ratio operator proposals in `src/grammar_extender.py`.
  - Near-deterministic sampling for template induction.
- **OOD-Aware Embedding Stubs:**
  - System-1 embeddings ready for adversarial domain-classifier integration.
  - DriftMonitor alerts on perception embedding shifts.
- **Adversarial & Hard-Negative Mining:**
  - Physics-informed augmentations in `data/hard_negative_miner.py`.
  - Diversity metrics and negative sample injection verified by CI.
- **Profiling & Scheduling Continuity:**
  - TaskProfiler and AdaptiveScheduler extended to `src/image_augmentor.py` and `src/physics_inference.py`.
  - CUDAStreamManager prefetches data for augmentation pipelines.
- **CI-Driven Contracts & Dependency Checks:**
  - All Phase 1 modules added to `integration/data_validator.py`.
  - Automated dependency-graph reports for new couplings.

---

## Directory Structure
```
bongard-solver/
├── data/
│   ├── raw/
│   ├── derived_labels.json
│   ├── augmented.pkl
│   ├── scene_graphs.pkl
│   ├── scene_graphs_physics.pkl
│   ├── hard_negatives.txt
│   ├── conceptnet_lite.json
│   ├── flagged_cases.txt
├── scripts/
│   ├── logo_to_shape.py
│   ├── image_augmentor.py
│   ├── build_scene_graphs.py
│   ├── physics_inference.py
│   ├── generate_hard_negatives.py
│   ├── export_conceptnet.py
│   ├── run_phase1.ps1
├── src/
│   ├── data_pipeline/
│   ├── physics_infer.py
│   ├── grammar_extender.py
│   ├── commonsense_kb.py
│   ├── cross_domain_reasoner.py
│   ├── quantifier_module.py
│   ├── image_augmentor.py
│   ├── drift_monitor.py
├── integration/
│   ├── task_profiler.py
│   ├── adaptive_scheduler.py
│   ├── cuda_stream_manager.py
│   ├── data_validator.py
│   ├── debug_dashboard.py
├── tests/
│   ├── test_augmentor.py
│   ├── test_physics_inference.py
│   ├── test_commonsense_kb.py
│   ├── test_cross_domain_reasoner.py
│   ├── test_anytime_inference.py
│   ├── test_grammar_extender.py
│   ├── test_hard_negative_miner.py
│   ├── test_data_validator.py
│   ├── test_debug_dashboard.py
├── README.md
```

---

## Step-By-Step Pipeline
### 1. Activate Virtual Environment
```powershell
cd Bongard-Solver
.\venv\Scripts\activate
```
### 2. Install Dependencies
```powershell
pip install shapely pymunk cupy-cuda12x cugraph-cuda12x conceptnet-lite==0.3.2 tqdm
```
### 3. Build ConceptNet-lite Snapshot
```powershell
python scripts/export_conceptnet.py --out data/conceptnet_lite.json --languages en,es,de,fr
```
### 4. Generate Geometry & Physics Attributes
```powershell
python scripts/logo_to_shape.py --input-dir data/raw/ --output data/derived_labels.json --parallel 8
```
### 5. GPU-Batched Image Augmentation
```powershell
python scripts/image_augmentor.py --input data/derived_labels.json --out data/augmented.pkl --parallel 8 --rotate 10 --scale 1.2 --shear 12
```
### 6. Build Scene Graphs
```powershell
python scripts/build_scene_graphs.py --aug data/augmented.pkl --out data/scene_graphs.pkl
```
### 7. Inject Physics + Commonsense Proxies
```powershell
python scripts/physics_inference.py --scene-graphs data/scene_graphs.pkl --kb data/conceptnet_lite.json --out data/scene_graphs_physics.pkl --gpu
```
### 8. Mine Hard Negatives
```powershell
python scripts/generate_hard_negatives.py --input-dir data/raw/ --output data/hard_negatives.txt --parallel 8 --near-miss
```
### 9. Validate Artifacts
```powershell
pytest tests/test_data_validator.py::test_phase1_schema -q
```
### 10. Run End-to-End MVI-1 Slice
```powershell
pytest tests/test_anytime_inference.py::test_mvi1_slice -q
```

---

## Automation Script
For full reproducibility, use:
```powershell
.\venv\Scripts\activate
python scripts/export_conceptnet.py --out data/conceptnet_lite.json
python scripts/logo_to_shape.py --input-dir data/raw/ --output data/derived_labels.json --parallel 8
python scripts/image_augmentor.py --input data/derived_labels.json --out data/augmented.pkl --parallel 8
python scripts/build_scene_graphs.py --aug data/augmented.pkl --out data/scene_graphs.pkl
python scripts/physics_inference.py --scene-graphs data/scene_graphs.pkl --kb data/conceptnet_lite.json --out data/scene_graphs_physics.pkl --gpu
python scripts/generate_hard_negatives.py --input-dir data/raw/ --output data/hard_negatives.txt --parallel 8 --near-miss
pytest tests/test_anytime_inference.py::test_mvi1_slice -q
```

---

## Artifact Details
| File                      | Description                                              | Usage                        |
|---------------------------|---------------------------------------------------------|------------------------------|
| derived_labels.json       | Per-image geometry & physics attributes                 | Model training/reasoning     |
| augmented.pkl             | GPU-batch augmented images & masks                      | Scene graph construction     |
| scene_graphs.pkl          | Fused scene graphs with physics proxies                 | Physics/commonsense fusion   |
| scene_graphs_physics.pkl  | Scene graphs with physics and commonsense predicates    | Downstream reasoning         |
| hard_negatives.txt        | Adversarial negatives for curriculum mining             | Robust training/evaluation   |
| conceptnet_lite.json      | Commonsense KB for fusion modules                       | KB queries                   |
| flagged_cases.txt         | Flagged samples for manual review                       | Data quality assurance       |

---

## Testing & CI
- All modules are covered by dedicated tests in `tests/`.
- CI enforces schema compliance, coverage, and performance targets.
- Use `pytest` to validate all artifacts and pipeline steps.

---

## KPIs for Completion
| KPI                        | Target         | Where to View                |
|----------------------------|---------------|------------------------------|
| Puzzle coverage            | 100% (50/50)  | Dashboard “Coverage”         |
| Augment throughput         | ≥150 img/s    | TaskProfiler                 |
| Drift alert count          | 0             | Dashboard “S1 Drift”         |
| Avg grounding latency      | ≤200 ms       | Smoke test log               |
| Schema violations          | 0             | DataValidator                |

---

## Troubleshooting
| Symptom                                 | Root Cause                        | Fix                                   |
|------------------------------------------|-----------------------------------|---------------------------------------|
| `CommonsenseKB: file not found`          | Missing `conceptnet_lite.json`    | Re-run ConceptNet export              |
| `CUDA_ERROR_OUT_OF_MEMORY`               | Batch too large                   | Set `--batch 32` on augmentor         |
| `Schema mismatch: field 'convexity'`     | Old cache of `derived_labels.json`| Delete file, re-run attribute script  |
| `Augmentor KeyError: shear`              | CuPy build missing SciPy ops      | `pip install cupyx-scikit-image`      |

---

## Best Practices
- **Determinism:** All random routines are seeded and logged.
- **Batch Processing:** Use vectorized routines for speed and consistency.
- **Configurable Thresholds:** Expose geometry/physics thresholds via CLI.
- **Comprehensive Logging:** All flagged or failed computations are logged.
- **Audit Trail:** All label values and sample selection steps are logged.

---

## Integration Notes
- All outputs are designed for direct ingestion by downstream modules (`PhysicsInference`, `HardNegativeMiner`, “alpha” model testing).
- CI and schema validation are mandatory before merging to `main`.

---

## References
- Bongard-Solver: https://github.com/Ayushman125/Bongard-Solver
- ConceptNet-lite: https://github.com/commonsense/conceptnet5
- NVLabs Bongard-LOGO dataset and tools

---

## Conclusion
Phase 1 establishes a robust, extensible foundation for enhanced perception and neural-symbolic grounding. By following this pipeline, you ensure reproducibility, auditability, and readiness for future causal reasoning and codelet swarms.

---

**For deeper module details or help drafting the Phase 1 smoke test harness, please reach out!**
