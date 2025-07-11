# albumentations_augmix.py
# This is a community-maintained implementation of RandomAugMix for Albumentations.
# It is provided as a standalone module to resolve import issues.
# Source: Based on common community implementations of AugMix for Albumentations.

import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class RandomAugMix(ImageOnlyTransform):
    """
    Applies AugMix data augmentation technique.
    AugMix is a data augmentation method that mixes multiple augmented versions
    of an image. It helps to improve model robustness and uncertainty estimation.

    Args:
        width (tuple): Range for the number of augmentation chains to mix. Default: (3, 10).
        alpha (float): Parameter for Beta distribution, controlling the mixing weights. Default: 1.0.
        p (float): Probability of applying the transform. Default: 0.5.
    """
    def __init__(self, width=(3, 10), alpha=1.0, p=0.5, always_apply=False):
        super().__init__(p, always_apply) # Pass p and always_apply to the parent class
        self.width = width
        self.alpha = alpha

    def apply(self, img, **params):
        return self._augmix_apply(img, self.width, self.alpha)

    @staticmethod
    def _augmix_apply(img, width, alpha):
        # Ensure image is float32 for calculations
        img_float = img.astype(np.float32)

        # Sample number of augmentation chains
        num_chains = np.random.randint(width[0], width[1] + 1)
        
        # Sample mixing weights from Dirichlet distribution
        ws = np.float32(np.random.dirichlet([alpha] * num_chains))
        
        # Sample Beta distribution for overall blending factor
        m = np.float32(np.random.beta(alpha, alpha))

        mix = np.zeros_like(img_float, dtype=np.float32)
        for i in range(num_chains):
            img_aug = img_float.copy()
            # Apply 1 to 3 random operations per chain
            for _ in range(np.random.randint(1, 4)): 
                op = np.random.choice([
                    A.RandomBrightnessContrast(p=1.0),
                    A.GaussNoise(p=1.0),
                    A.HueSaturationValue(p=1.0),
                    A.CLAHE(p=1.0),
                    A.ToGray(p=1.0),
                    A.Posterize(p=1.0, num_bits=np.random.randint(4, 8)),
                    A.Solarize(p=1.0, threshold=np.random.uniform(0.5, 1.0)),
                    A.ColorJitter(p=1.0),
                    A.Equalize(p=1.0),
                    A.Blur(p=1.0, blur_limit=(3,7)),
                    A.Sharpen(p=1.0),
                ])
                img_aug = op(image=img_aug)['image'].astype(np.float32) # Ensure output is float32

            mix += ws[i] * img_aug

        # Final blend of original image and mixed augmented images
        mixed_img = (m * img_float + (1 - m) * mix).astype(img.dtype)
        return mixed_img

    def get_transform_init_args_names(self):
        return ("width", "alpha")

