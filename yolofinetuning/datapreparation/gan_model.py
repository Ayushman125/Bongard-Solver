import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import random
import logging
import os

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DCGAN Architectures ---

# Generator Code based on original DCGAN paper and common implementations
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim: int, img_channels: int, img_size: int, ngf: int = 64):
        super().__init__()
        self.img_size = img_size
        
        # Determine the initial spatial dimension based on img_size
        # For a 640x640 output, we need 5 upsampling layers (4->8->16->32->64->128->256->512->1024)
        # This architecture will try to scale up to the target size.
        # A common DCGAN for 64x64 output uses 4 ConvTranspose2d layers, starting from 4x4.
        # To reach 640x640, we'd ideally need more layers or different strides/kernel sizes.
        # For simplicity and to fit general image_size, we'll use a base of 4 ConvTranspose2d
        # and then resize the output if it doesn't match img_size.
        
        # Initial spatial dimension for first ConvTranspose2d
        self.init_spatial_dim = 4 

        # Calculate the number of upsampling layers needed to get close to img_size
        # Starting from 4x4, each ConvTranspose2d with stride 2 doubles the size.
        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256 -> 512
        num_upsampling_layers = int(np.log2(img_size / self.init_spatial_dim))
        
        # Adjust ngf based on the number of layers to maintain reasonable feature map sizes
        ngf_multiplier = 2**(num_upsampling_layers - 1) if num_upsampling_layers > 0 else 1
        current_ngf = ngf * ngf_multiplier

        layers = []
        # First layer: Project latent_dim to initial feature map size
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(latent_dim, current_ngf, self.init_spatial_dim, 1, 0, bias=False),
                nn.BatchNorm2d(current_ngf),
                nn.ReLU(True)
            )
        )
        
        # Middle layers: Upsample
        for i in range(num_upsampling_layers - 1): # -1 because the first layer already upsamples once
            next_ngf = current_ngf // 2
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_ngf, next_ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_ngf),
                    nn.ReLU(True)
                )
            )
            current_ngf = next_ngf

        # Output layer: to img_channels
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(current_ngf, img_channels, 4, 2, 1, bias=False),
                nn.Tanh() # Output pixel values between -1 and 1
            )
        )
        self.main = nn.Sequential(*layers)
        
        logging.info(f"DCGAN Generator initialized for {img_size}x{img_size} images with {num_upsampling_layers} upsampling layers.")

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DCGAN Generator.
        Args:
            noise (torch.Tensor): Random noise vector (batch_size, latent_dim, 1, 1).
        Returns:
            torch.Tensor: Generated image tensor (batch_size, img_channels, img_size, img_size).
        """
        # Ensure noise is reshaped to (batch_size, latent_dim, 1, 1) if not already
        if noise.ndim == 2:
            noise = noise.unsqueeze(2).unsqueeze(3)
        img = self.main(noise)
        return img

# Discriminator Code based on original DCGAN paper and common implementations
class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels: int, img_size: int, ndf: int = 64):
        super().__init__()
        self.img_size = img_size

        # Calculate number of downsampling layers needed
        # Starting from img_size, each Conv2d with stride 2 halves the size.
        # We want to end up with a 4x4 feature map before the final linear layer.
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        num_downsampling_layers = int(np.log2(img_size / 4)) # Assuming final spatial dim is 4

        current_ndf = ndf
        layers = []
        # Input layer: from img_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(img_channels, current_ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Middle layers: Downsample
        for i in range(num_downsampling_layers - 1):
            next_ndf = current_ndf * 2
            layers.append(
                nn.Sequential(
                    nn.Conv2d(current_ndf, next_ndf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_ndf),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            current_ndf = next_ndf

        # Output layer: to 1 (real/fake score)
        layers.append(
            nn.Sequential(
                nn.Conv2d(current_ndf, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        )
        self.main = nn.Sequential(*layers)
        logging.info(f"DCGAN Discriminator initialized for {img_size}x{img_size} images with {num_downsampling_layers} downsampling layers.")

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DCGAN Discriminator.
        Args:
            img (torch.Tensor): Input image tensor (batch_size, img_channels, img_size, img_size).
        Returns:
            torch.Tensor: Validity score (batch_size, 1, 1, 1).
        """
        validity = self.main(img)
        return validity.view(-1, 1) # Flatten to (batch_size, 1)


class GANModel:
    """
    Wrapper for a GAN model (e.g., DCGAN) for synthetic image generation.
    This class handles loading the generator and generating images.
    It's designed to be integrated into a data pipeline.
    """
    def __init__(self, config: dict):
        self.config = config
        self.gan_type = self.config.get('gan_type', 'dcgan').lower()
        self.latent_dim = self.config.get('latent_dim', 100)
        self.num_channels = self.config.get('num_channels', 3)
        self.num_classes = self.config.get('num_classes', 10) # Used for mock labels, not generator input for unconditional GAN
        self.image_size = self.config.get('image_size', [640, 640])
        self.device = torch.device(self.config.get('device', 'cpu'))

        if self.image_size[0] != self.image_size[1]:
            logging.warning(f"GAN expects square images, but got {self.image_size}. Using {self.image_size[0]} as size.")
            self.img_dim = self.image_size[0]
        else:
            self.img_dim = self.image_size[0]

        if self.gan_type == 'dcgan':
            self.generator = DCGAN_Generator(self.latent_dim, self.num_channels, self.img_dim).to(self.device)
            self.discriminator = DCGAN_Discriminator(self.num_channels, self.img_dim).to(self.device) # For completeness
            logging.info(f"DCGAN Generator initialized on {self.device} for {self.img_dim}x{self.img_dim} images.")
        else:
            raise ValueError(f"Unsupported GAN type: {self.gan_type}. Only 'dcgan' is implemented.")
        
        # Load pre-trained weights if specified
        pretrained_weights_path = self.config.get('pretrained_weights_path')
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            try:
                self.generator.load_state_dict(torch.load(pretrained_weights_path, map_location=self.device))
                logging.info(f"Loaded pre-trained GAN generator weights from: {pretrained_weights_path}")
            except Exception as e:
                logging.error(f"Failed to load GAN generator weights from {pretrained_weights_path}: {e}")
                logging.warning("Proceeding with randomly initialized GAN generator.")
        else:
            logging.warning("No pre-trained GAN generator weights specified or found. Using randomly initialized weights.")
            logging.warning("For high-quality synthetic data, you will need to train this DCGAN on a relevant dataset or find pre-trained weights online.")

        self.generator.eval() # Set generator to evaluation mode

    def generate_images_batch(self, batch_size: int, image_size: tuple = (640, 640)) -> list[Image.Image]:
        """
        Generates a batch of synthetic images using the GAN.
        Args:
            batch_size (int): Number of images to generate in this batch.
            image_size (tuple): (width, height) of the desired output images.
        Returns:
            list: List of PIL.Image objects.
        """
        with torch.no_grad():
            # Generate random noise batch
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            
            # Generate images batch
            gen_img_tensors = self.generator(noise)

        generated_pil_images = []
        for i in range(batch_size):
            # Convert tensor to PIL Image (scale from -1 to 1 to 0 to 255)
            # Permute from (C, H, W) to (H, W, C) for PIL
            gen_img_np = (gen_img_tensors[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
            gen_img_pil = Image.fromarray(gen_img_np.astype(np.uint8))
            
            # Resize to target image_size if the GAN's internal output size is different
            if gen_img_pil.size != image_size:
                gen_img_pil = gen_img_pil.resize(image_size, Image.LANCZOS)
            generated_pil_images.append(gen_img_pil)

        logging.debug(f"Generated a batch of {batch_size} synthetic images of size {image_size}.")
        return generated_pil_images

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    import shutil

    # Dummy class names for testing mock labels
    dummy_class_names = ["car", "person", "tree", "building"]
    num_dummy_classes = len(dummy_class_names)

    # Create a dummy output directory
    output_dir = "dcgan_test_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # GAN Model Configuration for DCGAN
    gan_config = {
        'gan_type': 'dcgan',
        'latent_dim': 100,
        'num_channels': 3,
        'num_classes': num_dummy_classes, # Used for mock labels in data_preparation_utils
        'image_size': [128, 128], # Smaller size for quick testing in this example
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # 'pretrained_weights_path': 'path/to/your/trained_dcgan_generator.pth' # Add path if you have one
    }

    print(f"Testing DCGANModel on device: {gan_config['device']}")
    gan_instance = GANModel(gan_config)

    batch_size = 4
    print(f"\nGenerating {batch_size} synthetic images in a batch...")
    generated_images = gan_instance.generate_images_batch(
        batch_size=batch_size,
        image_size=(256, 256) # Can generate at a different size than GAN's internal
    )
    
    for i, img in enumerate(generated_images):
        img_path = os.path.join(output_dir, f"synth_dcgan_{i:02d}.png")
        img.save(img_path)
        print(f"Saved {img_path}")

    print(f"\nSynthetic images saved to {output_dir}")
    print("IMPORTANT: This module only generates images. Pseudo-labeling will be handled by data_preparation_utils.")

    # Clean up dummy directory
    # shutil.rmtree(output_dir)
    # print(f"Cleaned up {output_dir}")
