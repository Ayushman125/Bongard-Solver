# Professional System 1 Training - Clean Architecture

## 🎯 Overview

This is the cleaned and modularized version of the Bongard problem solver with professional System 1 training capabilities. All debugging clutter has been removed and functionality has been organized into proper modules.

## 📁 Project Structure

```
BongardSolver/
├── training/                          # Professional training modules
│   ├── __init__.py
│   └── system1_professional.py       # Main professional training pipeline
├── utils/                            # Utility modules
│   ├── __init__.py
│   └── image_validator.py           # Image generation validation
├── scripts/                          # Executable scripts
│   └── validate_phase1.py           # Integrated validation script
├── core_models/                      # Core model implementations
├── src/                             # Source code modules
│   ├── bongard_generator/           # Synthetic data generation
│   ├── data/                        # Data handling
│   └── perception/                  # Perception models
├── tests/                           # Unit tests
├── config.py                        # Main configuration
├── cleanup_project.py               # Project cleanup script
└── README_PROFESSIONAL.md           # This file
```

## 🚀 Quick Start - Professional Training

### 1. Clean the Project (First Time)
```bash
# Remove debug files and organize code
python cleanup_project.py
```

### 2. Run Professional System 1 Training
```bash
# Full professional training (10,000 images, 50 epochs)
python -m scripts.validate_phase1 --mode professional --epochs 50 --synthetic_size 10000

# Quick test (smaller scale)
python -m scripts.validate_phase1 --mode professional --epochs 5 --synthetic_size 1000

# GPU memory-conscious training
python -m scripts.validate_phase1 --mode professional --epochs 25 --batch_size 8 --synthetic_size 5000
```

### 3. Direct Professional Training
```bash
# Use the training module directly
python -m training.system1_professional --epochs 50 --synthetic_size 10000 --batch_size 16
```

## 📊 Professional Training Features

### 🎯 Performance Targets
- **Synthetic Dataset**: AP@0.5:0.95 ≥ 0.85, AP@0.5 ≥ 0.95
- **Real Dataset**: AP@0.5 ≥ 0.70

### 🔧 Professional Configuration
- **Model**: ResNet50 backbone with Mask R-CNN
- **Dataset**: 10,000 synthetic images with balanced rule distribution
- **Training**: 50 epochs with CosineAnnealingWarmRestarts scheduler
- **Augmentation**: Professional augmentation pipeline
- **Evaluation**: COCO-style metrics with proper validation

### 🏗️ Architecture
- **Detection**: Mask R-CNN with 6 classes (background + 5 shapes)
- **Attributes**: 6 attribute types (shape, color, fill, size, orientation, texture)
- **Reasoning**: RelationGNN with 12 spatial relations
- **Classification**: BongardHead with learnable temperature

## 📈 Training Pipeline

### Phase 1: Data Generation
1. Generate 10,000 synthetic images with COCO annotations
2. Balanced distribution across all available rules
3. High-resolution (640x640) images for better detection
4. Professional augmentation pipeline

### Phase 2: Model Training
1. Initialize Mask R-CNN with pretrained ResNet50
2. Train with mixed precision and gradient clipping
3. CosineAnnealingWarmRestarts scheduler for optimal convergence
4. Early stopping with patience for efficiency

### Phase 3: Evaluation
1. COCO-style evaluation metrics
2. Validation on held-out synthetic data
3. Zero-shot evaluation on real Bongard-LOGO dataset
4. Performance target checking

### Phase 4: Results Analysis
1. Detailed metric reporting
2. Target achievement verification
3. Model checkpoint saving
4. Training history logging

## 🛠️ Utility Functions

### Image Validation
```python
from utils.image_validator import validate_synthetic_images

# Validate image generation
success = validate_synthetic_images(num_samples=16, img_size=128, save_plot=True)
```

### Professional Training
```python
from training.system1_professional import ProfessionalTrainingConfig, run_professional_training

# Setup professional config
config = setup_professional_config(base_config)

# Run training
results = run_professional_training(config)
```

## 📋 Configuration Options

### Training Parameters
```python
# In training/system1_professional.py
class ProfessionalTrainingConfig:
    PROFESSIONAL_EPOCHS = 50
    PROFESSIONAL_BATCH_SIZE = 16
    PROFESSIONAL_LR = 0.001
    PROFESSIONAL_SYNTHETIC_SIZE = 10000
    
    # Performance targets
    SYNTHETIC_AP_TARGET = 0.85
    SYNTHETIC_AP50_TARGET = 0.95
    REAL_AP50_TARGET = 0.70
```

### GPU Memory Optimization
The system automatically adjusts batch size based on available GPU memory:
- **4GB GPU**: batch_size = 8
- **8GB+ GPU**: batch_size = 16
- **16GB+ GPU**: batch_size = 32

## 🔍 Validation Modes

### Professional Mode (Recommended)
```bash
python -m scripts.validate_phase1 --mode professional
```
- Uses professional training pipeline
- 10,000 synthetic images
- 50 training epochs
- Full COCO evaluation

### Legacy Mode
```bash
python -m scripts.validate_phase1 --mode legacy
```
- Uses original validation logic
- Smaller scale training
- Compatibility with existing checkpoints

### Both Modes
```bash
python -m scripts.validate_phase1 --mode both
```
- Runs both professional and legacy validation
- Comprehensive testing

## 📊 Expected Results

### Training Progress
```
Epoch 1/50: Loss=2.45, AP@0.5:0.95=0.12
Epoch 10/50: Loss=1.23, AP@0.5:0.95=0.45
Epoch 25/50: Loss=0.78, AP@0.5:0.95=0.68
Epoch 40/50: Loss=0.52, AP@0.5:0.95=0.82
Epoch 50/50: Loss=0.41, AP@0.5:0.95=0.87 ✅
```

### Final Performance
```
=== Training Results ===
Final synthetic AP@0.5:0.95: 0.870 ✅ (Target: 0.85)
Final synthetic AP@0.5: 0.960 ✅ (Target: 0.95)
Real dataset AP@0.5: 0.740 ✅ (Target: 0.70)

🎉 All performance targets achieved!
```

## 🗂️ Output Files

### Training Outputs
- `checkpoints/professional/`: Model checkpoints
- `results/professional_training_results_50ep.json`: Detailed results
- `logs/professional_training.log`: Training logs
- `processed_data/synthetic_coco_10k.json`: COCO annotations

### Validation Outputs
- `validation_images.png`: Sample image visualization
- `logs/validate_phase1_professional.log`: Validation logs

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python -m scripts.validate_phase1 --mode professional --batch_size 8
   ```

2. **Slow Data Generation**
   ```bash
   # Use smaller dataset for testing
   python -m scripts.validate_phase1 --mode professional --synthetic_size 5000
   ```

3. **Import Errors**
   ```bash
   # Run cleanup first
   python cleanup_project.py
   ```

## 🎯 Next Steps

1. **Run Full Professional Training**:
   ```bash
   python -m scripts.validate_phase1 --mode professional --epochs 50 --synthetic_size 10000
   ```

2. **Monitor Training Progress**:
   ```bash
   tail -f logs/professional_training.log
   ```

3. **Analyze Results**:
   - Check `results/professional_training_results_50ep.json`
   - View saved model at `checkpoints/professional/`
   - Validate performance targets are met

4. **Deploy for Inference**:
   - Use trained model for Bongard problem solving
   - Integrate with existing inference pipeline
   - Scale up for production use

## 🏆 Success Criteria

The professional training is considered successful when:
- ✅ All images generate correctly (no black boxes)
- ✅ Training completes without errors
- ✅ Synthetic AP@0.5:0.95 ≥ 0.85
- ✅ Synthetic AP@0.5 ≥ 0.95  
- ✅ Real AP@0.5 ≥ 0.70
- ✅ Model checkpoints saved properly

Ready to achieve state-of-the-art performance on Bongard problems! 🚀
