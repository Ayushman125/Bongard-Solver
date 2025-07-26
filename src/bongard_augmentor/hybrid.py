"""
Legacy hybrid.py is now refactored into modules:
- sam_ensemble.py: Multi-model SAM logic
- mask_refiners.py: Mask refinement techniques
- deep_segmentation.py: BongardSegmentationTrainer
- quality_controller.py: Quality metrics and QA
- augmentor_main.py: Main orchestration

from .augmentor_main import BongardAugmentor

# For backward compatibility, expose main entry points
augmentor = BongardAugmentor()

def generate_all_advanced_masks(image):
    return augmentor.generate_all_masks(image)

def generate_ensemble_masks(image, use_ensemble=True):
    return augmentor.generate_ensemble_masks(image, use_ensemble=use_ensemble)
"""