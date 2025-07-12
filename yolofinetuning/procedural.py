import numpy as np
import cv2
import random
from perlin_noise import PerlinNoise
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def gen_cellular_automata(size, rule=30, iterations=50, initial_density=0.5):
    """
    Generates an image using a simple 1D cellular automaton (e.g., Rule 30).
    The 1D pattern is expanded to 2D.
    Args:
        size (tuple): (height, width) of the output image.
        rule (int): Wolfram's Rule number (0-255).
        iterations (int): Number of generations for the 1D automaton.
        initial_density (float): Initial density of live cells for the first row.
    Returns:
        np.ndarray: Generated image (HWC, uint8, BGR).
    """
    height, width = size
    
    # Generate a 1D cellular automaton
    # Initial state: random binary array
    current_row = np.random.rand(width) < initial_density
    history = [current_row]

    # Convert rule to binary representation
    rule_binary = np.array([int(x) for x in np.binary_repr(rule, 8)], dtype=bool)

    for _ in range(iterations - 1):
        next_row = np.zeros_like(current_row)
        # Pad the row to handle edges
        padded_row = np.pad(current_row, (1, 1), mode='wrap')
        for i in range(width):
            # Get the 3-bit neighborhood (left, center, right)
            neighborhood = padded_row[i:i+3]
            # Convert neighborhood to an integer index (e.g., [1,0,1] -> 5)
            index = (neighborhood[0] << 2) | (neighborhood[1] << 1) | neighborhood[2]
            # Apply the rule
            next_row[i] = rule_binary[7 - index] # Rules are indexed from 0-7, 7-index maps to Wolfram's convention
        current_row = next_row
        history.append(current_row)
    
    # Expand 1D history to 2D image
    # If height > iterations, repeat the pattern or stretch
    if height > iterations:
        ca_pattern = np.vstack(history)
        # Repeat the pattern vertically to fill the image height
        num_repeats = (height + ca_pattern.shape[0] - 1) // ca_pattern.shape[0]
        full_pattern = np.tile(ca_pattern, (num_repeats, 1))[:height, :]
    else:
        full_pattern = np.vstack(history)[:height, :] # Take only necessary rows

    # Convert boolean array to uint8 image (0 for black, 255 for white)
    img_array = (full_pattern * 255).astype(np.uint8)
    
    # Randomly choose colors
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Create a 3-channel image with the chosen colors
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    output_img[img_array == 0] = color1
    output_img[img_array == 255] = color2

    logger.debug(f"Generated cellular automata image with size {size}, rule {rule}")
    return output_img

def gen_texture(size, octaves_range=(2, 6), seed=None):
    """
    Generates a Perlin noise-based texture.
    Args:
        size (tuple): (height, width) of the output image.
        octaves_range (tuple): Min and max number of octaves for Perlin noise.
        seed (int, optional): Seed for reproducibility.
    Returns:
        np.ndarray: Generated texture image (HWC, uint8, BGR).
    """
    height, width = size
    octaves = random.randint(octaves_range[0], octaves_range[1])
    
    # Use a random seed if not provided
    current_seed = random.randint(0, 100000) if seed is None else seed
    noise = PerlinNoise(octaves=octaves, seed=current_seed)

    pic = np.zeros(size)
    for i in range(height):
        for j in range(width):
            pic[i, j] = noise([i/height, j/width])
    
    # Normalize to 0-1 and then to 0-255
    pic = (pic - pic.min()) / (pic.max() - pic.min())
    img_array = (pic * 255).astype(np.uint8)

    # Apply a random color map or tint
    if random.random() < 0.5: # Grayscale
        output_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else: # Color tint
        tint_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        output_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        output_img = cv2.addWeighted(output_img, 0.7, np.full_like(output_img, tint_color), 0.3, 0)
    
    logger.debug(f"Generated texture image with size {size}, octaves {octaves}, seed {current_seed}")
    return output_img

if __name__ == '__main__':
    # Example usage and saving
    output_dir = Path("./procedural_test_output")
    output_dir.mkdir(exist_ok=True)

    # Generate Cellular Automata
    for i in range(5):
        img_ca = gen_cellular_automata((224, 224), rule=random.choice([30, 90, 110]))
        cv2.imwrite(str(output_dir / f"ca_example_{i}.png"), img_ca)
        logger.info(f"Saved ca_example_{i}.png")

    # Generate Texture
    for i in range(5):
        img_texture = gen_texture((224, 224))
        cv2.imwrite(str(output_dir / f"texture_example_{i}.png"), img_texture)
        logger.info(f"Saved texture_example_{i}.png")
