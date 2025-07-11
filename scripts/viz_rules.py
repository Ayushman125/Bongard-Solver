# Folder: bongard_solver/
# File: scripts/viz_rules.py

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import argparse
import logging
import os

# Assume dsl.py is in the parent directory or accessible via PYTHONPATH
try:
    from dsl import parse_rule_to_mask
except ImportError:
    logging.error("Could not import parse_rule_to_mask from dsl.py. Please ensure dsl.py is accessible.")
    # Define a dummy function to prevent crashes if dsl.py is missing
    def parse_rule_to_mask(rule_description: str, image_size: Tuple[int, int]) -> Image.Image:
        logging.warning("Using dummy parse_rule_to_mask: Returning a blank mask.")
        return Image.new('L', image_size, 0) # Return a black (empty) mask

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def overlay_rule(image_path: str, rule_description: str, output_path: Optional[str] = None):
    """
    Overlays a symbolic rule's mask onto an image and displays or saves the result.

    Args:
        image_path (str): Path to the input image.
        rule_description (str): The symbolic rule description (e.g., "circle_smaller").
        output_path (str, optional): Path to save the output image. If None, displays the image.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return

    try:
        # Open the image and convert to RGBA for alpha blending
        img = Image.open(image_path).convert('RGBA')
        img_size = img.size # (width, height)

        # Generate the mask from the rule description
        # parse_rule_to_mask should return a PIL Image (mode 'L' for grayscale mask)
        mask_image = parse_rule_to_mask(rule_description, img_size)

        # Ensure mask is RGBA for blending
        # Create a transparent base for the mask
        mask_rgba = Image.new('RGBA', img_size, (0, 0, 0, 0))
        # Convert the grayscale mask to a color (e.g., red) with alpha
        # The 'jet' colormap is for matplotlib, for PIL blending, we need RGBA.
        # Let's use a semi-transparent blue for the mask.
        mask_color = (0, 0, 255, 128) # Blue with 50% alpha
        draw = ImageDraw.Draw(mask_rgba)
        # Iterate over the mask_image pixels and draw on mask_rgba
        # This is a simplified approach; a more direct way might be needed if parse_rule_to_mask
        # returns a numpy array or directly a color mask.
        # For a simple binary mask, we can just use the mask as the alpha channel.
        
        # Convert grayscale mask to a full RGBA image where mask values control alpha
        mask_array = np.array(mask_image)
        colored_mask = np.zeros((*img_size[::-1], 4), dtype=np.uint8) # H, W, RGBA
        
        # Apply a color to the masked areas (e.g., blue)
        colored_mask[mask_array > 0, 0] = 0   # Red
        colored_mask[mask_array > 0, 1] = 0   # Green
        colored_mask[mask_array > 0, 2] = 255 # Blue
        
        # Set alpha based on mask presence
        colored_mask[mask_array > 0, 3] = 128 # 50% opacity
        
        # Create PIL Image from numpy array
        overlay_img = Image.fromarray(colored_mask, 'RGBA')

        # Blend the original image and the overlay mask
        combined_img = Image.alpha_composite(img, overlay_img)

        # Display or save the image
        plt.figure(figsize=(8, 8))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title(f"Rule: '{rule_description}'")

        if output_path:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            combined_img.save(output_path)
            logger.info(f"Overlayed image saved to: {output_path}")
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error processing image or rule: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Overlay symbolic rules as masks on images.")
    parser.add_argument('img', type=str, help="Path to the input image.")
    parser.add_argument('rule', type=str, help="Symbolic rule description (e.g., 'circle_smaller').")
    parser.add_argument('--out', type=str, default=None,
                        help="Optional: Path to save the output image. If not provided, displays the image.")
    
    args = parser.parse_args()
    
    overlay_rule(args.img, args.rule, args.out)
