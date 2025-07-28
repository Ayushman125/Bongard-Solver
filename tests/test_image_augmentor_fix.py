import numpy as np
from src.bongard_augmentor.refiners import MaskRefiner

def test_mask_quality_on_synthetic():
    # Create a synthetic image and mask
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.rectangle(img, (16, 16), (48, 48), (255,255,255), -1)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (44, 44), 255, -1)
    refiner = MaskRefiner()
    validated_mask, quality, metrics = refiner.validate_mask_quality_with_confidence(mask, img, prediction_scores=[1.0])
    assert quality >= 0.5, f"Quality score too low: {quality}"
    fill_ratio = np.count_nonzero(validated_mask) / validated_mask.size
    assert 0.01 < fill_ratio < 0.5, f"Fill ratio out of range: {fill_ratio}"
