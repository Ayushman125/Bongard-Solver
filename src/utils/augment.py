# Folder: bongard_solver/src/utils/
# File: augment.py
import logging
import numpy as np
import random # For conditional augmentations if needed
import cv2 # Required by Albumentations for some transforms

# Import Albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    logging.warning("Albumentations not found. Image augmentation will be skipped.")
    HAS_ALBUMENTATIONS = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define the augmentation pipeline
# This needs to be configured based on your `config.py`
# For now, using a general set of augmentations as described.
# It expects a numpy array (H, W, C) and returns a PyTorch tensor (C, H, W).
if HAS_ALBUMENTATIONS:
    # Assuming ImageNet mean/std for normalization if not specified in config
    # You might want to load these from a global config if available.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    augmenter = A.Compose([
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
        ], p=0.5), # 50% chance for either GaussNoise or ISONoise
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=0.5), # blur_limit must be odd
            A.MedianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.5), # 50% chance for one of the blur types
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.5), # Fill with white
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=(255,255,255), p=0.5), # Simulate occlusion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), # ImageNet normalization
        ToTensorV2() # Converts numpy array (H, W, C) to PyTorch tensor (C, H, W)
    ])
else:
    # Dummy augmenter if Albumentations is not available
    class DummyAugmenter:
        def __call__(self, image: np.ndarray, **kwargs) -> torch.Tensor:
            logger.warning("Dummy augmenter used. No augmentations applied.")
            # Convert numpy HWC to torch CHW and normalize manually if needed
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Apply basic normalization if no Albumentations
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            return img_tensor
    augmenter = DummyAugmenter()

def augment_image(img_np: np.ndarray) -> torch.Tensor:
    """
    Applies the defined Albumentations augmentation pipeline to a single image.
    Args:
        img_np (np.ndarray): The input image as a NumPy array (H, W, C).
    Returns:
        torch.Tensor: The augmented and transformed image as a PyTorch tensor (C, H, W).
    """
    if HAS_ALBUMENTATIONS:
        return augmenter(image=img_np)['image']
    else:
        # If Albumentations is not installed, the dummy augmenter is used.
        # It will perform basic conversion and normalization.
        return augmenter(image=img_np)

if __name__ == '__main__':
    # Example Usage
    logger.info("Running augment.py example.")
    # Create a dummy image (e.g., a white square with a black circle)
    img_size = 256
    dummy_img_pil = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img_pil)
    draw.ellipse((50, 50, 200, 200), fill=(0, 0, 0)) # Black circle
    dummy_img_np = np.array(dummy_img_pil)
    logger.info(f"Original image shape: {dummy_img_np.shape}, dtype: {dummy_img_np.dtype}")
    # Apply augmentation
    augmented_tensor = augment_image(dummy_img_np)
    logger.info(f"Augmented tensor shape: {augmented_tensor.shape}, dtype: {augmented_tensor.dtype}")
    logger.info(f"Augmented tensor min value: {augmented_tensor.min():.4f}, max value: {augmented_tensor.max():.4f}")
    
    # To visualize, you'd need to denormalize and convert back to numpy
    # (This logic would typically be in xai.py or a visualization utility)
    # For a quick check:
    # denormalize = T.Compose([
    #     T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
    #     T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    #     T.ToPILImage()
    # ])
    # try:
    #     denormalized_img = denormalize(augmented_tensor.cpu())
    #     denormalized_img.save("augmented_example.png")
    #     logger.info("Saved augmented_example.png (might not be perfectly visible due to normalization).")
    # except Exception as e:
    #     logger.error(f"Error saving augmented image: {e}")
