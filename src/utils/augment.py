# Folder: bongard_solver/src/utils/
# File: augment.py
import logging
import numpy as np
import random  # For conditional augmentations if needed
import cv2  # Required by Albumentations for some transforms
import torch
from PIL import Image  # For converting PIL to numpy and back if needed for legacy

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
    # From config.py: IMAGENET_MEAN, IMAGENET_STD
    try:
        from config import IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS, CONFIG
        # Ensure mean/std are lists/tuples for Albumentations
        if not isinstance(IMAGENET_MEAN, (list, tuple)):
            IMAGENET_MEAN = [IMAGENET_MEAN] * NUM_CHANNELS
        if not isinstance(IMAGENET_STD, (list, tuple)):
            IMAGENET_STD = [IMAGENET_STD] * NUM_CHANNELS
        IMAGE_SIZE = CONFIG['data']['image_size']
    except ImportError:
        logger.warning("Could not import IMAGENET_MEAN/STD/NUM_CHANNELS/CONFIG from config. Using default ImageNet values.")
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        IMAGE_SIZE = [128, 128]  # Default size if config not available

    augmenter = A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1], always_apply=True),  # Ensure consistent size
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),  # Adjusted var_limit as per prompt
            A.ISONoise(color_shift=(0.01, 0.05), p=0.5),
        ], p=0.5),  # 50% chance for either GaussNoise or ISONoise
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),  # blur_limit must be odd
            A.MedianBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),  # Added GaussianBlur for more options
        ], p=0.3),  # 30% chance for one of the blur types
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Added for color variation
        A.Perspective(distortion_scale=0.05, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),  # Fill with white for perspective
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.5),  # Fill with white
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, fill_value=(255,255,255), p=0.3),  # Simulate occlusion, fill with white
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet normalization
        ToTensorV2()  # Converts numpy array (H, W, C) to PyTorch tensor (C, H, W)
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

def augment_image(img_pil: Image.Image) -> torch.Tensor:
    """
    Applies the defined Albumentations augmentation pipeline to a single image.
    Args:
        img_pil (PIL.Image.Image): The input image as a PIL Image.
    Returns:
        torch.Tensor: The augmented and transformed image as a PyTorch tensor (C, H, W).
    """
    # Albumentations expects numpy array (H, W, C)
    img_np = np.array(img_pil)
    if HAS_ALBUMENTATIONS:
        return augmenter(image=img_np)['image']
    else:
        # Simple fallback: random horizontal flip, keep as numpy array in [0,1]
        arr = img_np.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]  # Ensure (H, W, 1) for grayscale
        if np.random.rand() < 0.5:
            arr = np.fliplr(arr)
        # Return as float32 numpy array, caller can convert to tensor if needed
        return arr

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running augment.py example.")
    # Create a dummy image (e.g., a white square with a black circle)
    img_size = 256
    dummy_img_pil = Image.new('RGB', (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img_pil)
    draw.ellipse((50, 50, 200, 200), fill=(0, 0, 0))  # Black circle
    
    logger.info(f"Original image PIL mode: {dummy_img_pil.mode}, size: {dummy_img_pil.size}")
    # Apply augmentation
    augmented_tensor = augment_image(dummy_img_pil)
    logger.info(f"Augmented tensor shape: {augmented_tensor.shape}, dtype: {augmented_tensor.dtype}")
    logger.info(f"Augmented tensor min value: {augmented_tensor.min():.4f}, max value: {augmented_tensor.max():.4f}")
    
    # To visualize, you'd need to denormalize and convert back to numpy
    # (This logic would typically be in xai.py or a visualization utility)
    # For a quick check:
    if HAS_ALBUMENTATIONS:
        # Create a reverse normalization transform
        reverse_normalize = A.Compose([
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1/s for s in IMAGENET_STD]),
            A.Normalize(mean=[-m for m in IMAGENET_MEAN], std=[1.0, 1.0, 1.0]),
            ToTensorV2(transpose_mask=False)  # Keep HWC for numpy conversion
        ])
        
        # Convert tensor back to numpy HWC, then denormalize
        # Ensure it's on CPU and convert to numpy
        augmented_np_normalized = augmented_tensor.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
        
        # Apply reverse normalization
        denormalized_img_np = reverse_normalize(image=augmented_np_normalized)['image']
        
        # Convert to uint8 and save
        denormalized_img_np = (denormalized_img_np * 255).astype(np.uint8)
        denormalized_img_pil = Image.fromarray(denormalized_img_np)
        denormalized_img_pil.save("augmented_example.png")
        logger.info("Saved augmented_example.png (denormalized for viewing).")
    else:
        logger.warning("Albumentations not available, cannot denormalize for visualization.")
