import torch
from src.image_augmentor import ImageAugmentor

def test_augment_batch_returns_dict():
    augmentor = ImageAugmentor()
    images = torch.zeros((4,3,64,64))
    out = augmentor.augment_batch(images)
    assert isinstance(out, dict)
    assert 'original' in out
