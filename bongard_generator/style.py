"""
Encapsulates the GAN-based stylization logic.
"""
import torch
from PIL import Image, ImageDraw
import numpy as np

class Styler:
    def __init__(self, gan_ckpt_path):
        self.gan_model = None
        if gan_ckpt_path and torch.cuda.is_available():
            try:
                # Placeholder for loading a real GAN model (e.g., CycleGAN)
                # self.gan_model = torch.load(gan_ckpt_path).to('cuda')
                # self.gan_model.eval()
                print(f"GAN styler initialized (placeholder).")
            except Exception as e:
                print(f"Could not load GAN model from {gan_ckpt_path}: {e}")
                self.gan_model = None

    def stylize(self, image: Image.Image) -> Image.Image:
        """
        Applies GAN-based stylization to an image.
        Ensures the final output is black and white.
        """
        if self.gan_model:
            # This is a placeholder for actual GAN inference
            # img_tensor = transforms.ToTensor()(image).unsqueeze(0).to('cuda')
            # with torch.no_grad():
            #     stylized_tensor = self.gan_model(img_tensor)
            # stylized_image = transforms.ToPILImage()(stylized_tensor.squeeze(0).cpu())
            # For now, just return the original image
            stylized_image = image
        else:
            stylized_image = image

        # IMPORTANT: Convert to black and white ('1' mode)
        return stylized_image.convert('1')

def apply_background(image: Image.Image, cfg) -> Image.Image:
    """
    Applies a background texture to the image.
    """
    if cfg.bg_texture == 'noise':
        noise = np.random.randint(0, 255, (*image.size, 3), dtype=np.uint8)
        bg = Image.fromarray(noise, 'RGB')
    elif cfg.bg_texture == 'checker':
        bg = Image.new('RGB', image.size)
        draw = ImageDraw.Draw(bg)
        tile_size = 20
        for y in range(0, image.height, tile_size):
            for x in range(0, image.width, tile_size):
                if (x // tile_size) % 2 == (y // tile_size) % 2:
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill='gray')
                else:
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill='white')
    else:
        return image

    # Composite the original image over the background
    # The original image should have a transparent background for this to work best
    image = image.convert("RGBA")
    bg = bg.convert("RGBA")
    
    # Create a new image to composite onto
    composite_img = Image.alpha_composite(bg, image)
    
    return composite_img.convert('RGB')
