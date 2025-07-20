"""
GAN-based stylization module for the Bongard-LOGO generator.
Applies a pre-trained CycleGAN model to images and ensures the output
remains strictly black and white to adhere to the problem constraints.
"""
import logging
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import GeneratorConfig
if TORCH_AVAILABLE:
    from .models import CycleGANGenerator  # Assuming model definition exists

logger = logging.getLogger(__name__)

class Styler:
    """GAN-based stylization that ensures output remains black and white."""
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.model = None
        self.transform = None
        self.reverse_transform = None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. GAN stylization disabled.")
            return
            
        if self.config.use_gan_stylization and self.config.gan_model_path:
            logger.info(f"GAN Styler: Loading model from {self.config.gan_model_path}")
            self._load_model(self.config.gan_model_path)
        else:
            logger.info("GAN Styler: Not enabled.")

    def _load_model(self, model_path: str):
        """Loads the CycleGAN generator model."""
        if not TORCH_AVAILABLE:
            logger.error("Cannot load GAN model: PyTorch not available")
            return
            
        try:
            self.model = CycleGANGenerator().eval()  # Add .cuda() if using GPU
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])  # Assuming single channel (L)
            ])
            self.reverse_transform = T.ToPILImage()
            logger.info("GAN Styler model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GAN model from {model_path}: {e}")
            self.model = None

    def apply(self, image: Image.Image) -> Image.Image:
        """Applies style transfer to the image and ensures output remains binary."""
        if not self.model or not TORCH_AVAILABLE:
            return image
        
        try:
            # Ensure image is in a compatible format (e.g., 'L' for grayscale)
            img_tensor = self.transform(image.convert('L')).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                stylized_tensor = self.model(img_tensor)
            
            # Post-process back to a PIL image
            stylized_tensor = (stylized_tensor.squeeze(0) + 1) / 2.0  # Denormalize
            stylized_img = self.reverse_transform(stylized_tensor.cpu())
            
            # CRITICAL: Re-binarize after GAN processing to maintain black/white constraint
            return stylized_img.convert("L").point(lambda p: 255 if p > 128 else 0, mode='L')
        except Exception as e:
            logger.error(f"GAN stylization failed: {e}")
            return image
