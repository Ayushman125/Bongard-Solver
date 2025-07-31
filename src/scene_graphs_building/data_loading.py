# --- Main Functions for Data Loading and Processing ---
import os
import pickle
import json
def load_data(input_path):
    """Simple pickle/json loader for demonstration; replace as needed."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported input file type: {input_path}")


def remap_path(path):
    """Remaps image paths to match expected dataset structure."""
    return path.replace('category_1', '1').replace('category_0', '0')

def robust_image_open(path, *args, **kwargs):
    """
    Open an image file after remapping the path. Use this everywhere in the pipeline for image loading.
    Example: img = robust_image_open(image_path)
    """
    from PIL import Image
    remapped = remap_path(path)
    if not os.path.exists(remapped):
        raise FileNotFoundError(f"Image file not found after remapping: {remapped}")
    return Image.open(remapped, *args, **kwargs)
