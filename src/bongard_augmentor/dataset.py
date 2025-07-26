import torch
from torch.utils import data
from PIL import Image
import numpy as np
import json
import os

class ImagePathDataset(data.Dataset):
    """Load images from a JSON file containing paths and geometries."""
    def __init__(self, image_paths, derived_labels_path=None, transform=None):
        # Remap image paths to match actual folder names
        def remap_path(path):
            return path.replace('category_1', '1').replace('category_0', '0')

        self.image_paths = [remap_path(p) for p in image_paths]
        self.transform = transform
        self.derived_labels = None
        if derived_labels_path:
            with open(derived_labels_path, 'r') as f:
                self.derived_labels = json.load(f)

        # Create a mapping from image path to geometry
        self.path_to_geometry = {}
        if self.derived_labels:
            for entry in self.derived_labels:
                mapped_path = remap_path(entry['image_path'])
                self.path_to_geometry[mapped_path] = entry['geometry']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Use PIL to open the image, convert to grayscale
            image = Image.open(img_path).convert('L')
            
            # Convert to numpy array
            image_np = np.array(image, dtype=np.float32) / 255.0
            
            # Add channel dimension to make it [1, H, W]
            tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            if self.transform:
                tensor = self.transform(tensor)
            
            # Get geometry for the current image path
            geometry = self.path_to_geometry.get(img_path, [])
            
            return tensor, img_path, geometry
            
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}")
            return None, None, None
        except Exception as e:
            print(f"Warning: Error loading {img_path}: {e}")
            return None, None, None
