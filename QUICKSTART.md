# Quick Start Guide

This guide will help you get started with the Hybrid Dual-System Solver in 10 minutes.

---

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+ (with CUDA for GPU support)
- 8GB RAM minimum
- GPU with 4GB VRAM (recommended, but not required)

---

## Installation

### Step 1: Clone Repository

```powershell
git clone https://github.com/Ayushman125/Bongard-Solver.git
cd Bongard-Solver
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements/requirements.txt
```

**Expected packages:**
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `numpy>=1.21.0`
- `opencv-python>=4.5.0`
- `Pillow>=9.0.0`
- `scikit-image>=0.19.0`
- `tqdm>=4.62.0`

---

## Download Data

### Option 1: Automatic Download (Recommended)

```powershell
python scripts/download_data.py
```

This will download and extract BONGARD-LOGO dataset to `data/raw/`.

### Option 2: Manual Download

1. Visit [BONGARD-LOGO GitHub](https://github.com/NVlabs/Bongard-LOGO)
2. Download `ShapeBongard_V2.zip`
3. Extract to `data/raw/ShapeBongard_V2/`

**Expected structure:**
```
data/raw/ShapeBongard_V2/
├── train/
│   ├── ff/
│   ├── bd/
│   └── hd/
├── val/
└── test/
    ├── test_ff/
    ├── test_bd/
    ├── test_hd_comb/
    └── test_hd_novel/
```

---

## Self-Supervised Pretraining

### Option 1: Use Pretrained Weights (Fast)

Download our pretrained ResNet backbone:

```powershell
# Download from release
Invoke-WebRequest -Uri "https://github.com/Ayushman125/Bongard-Solver/releases/download/v1.0.0/pretrained_resnet.pth" -OutFile "pretrained/resnet_rotation_100ep.pth"
```

### Option 2: Train from Scratch (Slow, ~4 hours on RTX 3050 Ti)

```powershell
python scripts/pretrain_ssl.py --epochs 100 --batch-size 256 --lr 0.001
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--save-path`: Where to save weights (default: `pretrained/resnet_rotation_100ep.pth`)

**Expected output:**
```
Epoch [1/100] Loss: 1.386 Acc: 25.3%
Epoch [10/100] Loss: 0.524 Acc: 82.1%
Epoch [50/100] Loss: 0.087 Acc: 97.6%
Epoch [100/100] Loss: 0.042 Acc: 98.9%
Saved pretrained model to pretrained/resnet_rotation_100ep.pth
```

---

## Running Experiments

### Quick Test (5 minutes)

Test on a small subset to verify installation:

```powershell
python run_experiment.py --split ff --num-episodes 10 --cap 0.95
```

**Expected output:**
```
Loading data from data/raw/ShapeBongard_V2/test_ff...
Loaded 10 episodes
Running hybrid dual-system solver...
Episode 1/10: Correct ✓ (S1: 0.92, S2: 0.98, Final: 0.95)
Episode 2/10: Correct ✓ (S1: 0.88, S2: 0.87, Final: 0.88)
...
Final Accuracy: 100.0% (10/10)
```

### Full Evaluation (30 minutes per split)

**Test on Free-Form Split:**
```powershell
python run_experiment.py --split ff --cap 0.95
```

**Test on Basic Split:**
```powershell
python run_experiment.py --split bd --cap 0.95
```

**Test on Human-Designed Splits:**
```powershell
python run_experiment.py --split hd_comb --cap 0.95
python run_experiment.py --split hd_novel --cap 0.95
```

---

## Auto-Cap Tuning (Recommended)

To find the optimal weight cap on validation set:

```powershell
python scripts/auto_cap_tuning.py --mode global
```

**Output:**
```
Testing cap=0.95: Accuracy = 91.5%
Testing cap=0.97: Accuracy = 90.2%
Testing cap=0.98: Accuracy = 88.7%
Testing cap=0.99: Accuracy = 86.3%

Best cap: 0.95 (Accuracy: 91.5%)
```

**Use the best cap for testing:**
```powershell
python run_experiment.py --split ff --cap 0.95
```

---

## Understanding Results

### Interpreting Output

```
Episode 42/200: Correct ✓ (S1: 0.85, S2: 0.92, Final: 0.89)
```

- **S1**: System 1 (neural) prediction probability
- **S2**: System 2 (Bayesian) prediction probability
- **Final**: Weighted blend $w_1 \cdot p_1 + w_2 \cdot p_2$
- **Correct ✓**: Prediction matches ground truth

### Checking Weight Dynamics

View weight evolution across episodes:

```powershell
python scripts/visualize_weights.py --results logs/results_ff.json
```

**Output:**
![Weight Trajectory](docs/images/weight_trajectory.png)

---

## Command-Line Arguments

### `run_experiment.py`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--split` | str | `ff` | Test split: `ff`, `bd`, `hd_comb`, `hd_novel` |
| `--cap` | float | `0.95` | Maximum System 2 weight (cap constraint) |
| `--use-pretrain` | flag | True | Use pretrained backbone |
| `--pretrain-path` | str | `pretrained/resnet_rotation_100ep.pth` | Path to pretrained weights |
| `--num-episodes` | int | All | Limit number of episodes (for testing) |
| `--save-results` | str | `logs/results.json` | Save detailed results |
| `--verbose` | flag | False | Print detailed per-episode logs |

**Example:**
```powershell
python run_experiment.py `
    --split bd `
    --cap 0.97 `
    --use-pretrain `
    --pretrain-path pretrained/resnet_rotation_100ep.pth `
    --save-results logs/results_bd_cap097.json `
    --verbose
```

---

## Ablation Studies

Run predefined ablation scripts:

### 1. System 1 Only (Neural Baseline)

```powershell
.\scripts\ablation_studies\run_system1_only.bat
```

**Expected accuracy:** ~87% on test_ff

### 2. System 2 Only (Bayesian Baseline)

```powershell
bash scripts/ablation_studies/run_system2_only.sh
```

**Expected accuracy:** ~68% on test_ff

### 3. No Pretraining

```powershell
bash scripts/ablation_studies/run_no_pretrain.sh
```

**Expected accuracy:** ~62% on test_ff

### 4. Cap Sweep

Test multiple cap values:

```powershell
bash scripts/ablation_studies/run_cap_sweep.sh
```

**Output:**
```
Cap=0.50: Accuracy=85.2%
Cap=0.70: Accuracy=91.3%
Cap=0.90: Accuracy=97.5%
Cap=0.95: Accuracy=100.0% ← Best
Cap=0.97: Accuracy=98.0%
Cap=0.98: Accuracy=95.5%
Cap=0.99: Accuracy=90.2%
```

---

## Visualizations

### Generate Performance Plots

```powershell
python scripts/visualize_results.py --results logs/results_ff.json
```

**Outputs:**
- `plots/accuracy_per_episode.png`: Episode-by-episode accuracy
- `plots/weight_trajectory.png`: S1/S2 weight evolution
- `plots/confidence_distribution.png`: Confidence scores histogram
- `plots/conflict_analysis.png`: Cases where S1 ≠ S2

### Generate Architecture Diagram

```powershell
python scripts/generate_diagrams.py
```

**Outputs:**
- `docs/images/architecture_overview.png`
- `docs/images/system1_detail.png`
- `docs/images/system2_detail.png`
- `docs/images/arbitration_flow.png`

---

## Troubleshooting

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**Solution:**
Reduce batch size during pretraining:
```powershell
python scripts/pretrain_ssl.py --batch-size 128
```

### Missing Data Files

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/ShapeBongard_V2/test_ff'
```

**Solution:**
Re-download dataset:
```powershell
python scripts/download_data.py
```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'bayes_dual_system'
```

**Solution:**
Install in development mode:
```powershell
pip install -e .
```

### Slow Inference

**Problem:** Episode solving takes >5 seconds

**Solutions:**
1. **Use GPU**: Ensure CUDA is available
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   ```

2. **Reduce rule hypothesis space**: Edit `bayes_dual_system/system2_bayes_image.py`:
   ```python
   NUM_THRESHOLDS = 10  # Reduce from 20
   MAX_COMPOSITION_SIZE = 2  # Reduce from 3
   ```

3. **Cache features**: Precompute features for all images
   ```powershell
   python scripts/precompute_features.py --split ff
   ```

---

## Next Steps

### 1. Reproduce Paper Results

Run full evaluation pipeline:

```powershell
# Auto-cap tuning
python scripts/auto_cap_tuning.py --mode global

# Test all splits with best cap
python run_experiment.py --split ff --cap 0.95
python run_experiment.py --split bd --cap 0.95
python run_experiment.py --split hd_comb --cap 0.95
python run_experiment.py --split hd_novel --cap 0.95

# Generate comparison table
python scripts/generate_results_table.py
```

### 2. Experiment with Hyperparameters

Modify `config.yaml`:

```yaml
system1:
  learning_rate: 0.01
  num_adaptation_steps: 10
  dropout: 0.5

system2:
  noise_rate: 0.01
  complexity_penalty: 0.1
  num_thresholds: 20

arbitration:
  beta: 2.0
  max_system2_weight: 0.95
```

### 3. Extend Architecture

Add a third system (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#extensibility)).

### 4. Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Documentation**: [README.md](README.md), [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/Ayushman125/Bongard-Solver/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ayushman125/Bongard-Solver/discussions)
- **Email**: bongard.solver@example.com

---

## Summary Checklist

- [x] Clone repository
- [x] Install dependencies
- [x] Download data
- [x] Run quick test (10 episodes)
- [x] Perform SSL pretraining (or download weights)
- [x] Auto-cap tuning
- [x] Full evaluation on all splits
- [x] Generate visualizations
- [x] Review results in [docs/RESULTS.md](docs/RESULTS.md)

**Congratulations! You've successfully set up the Hybrid Dual-System Solver.** 🎉
