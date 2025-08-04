import torch
from torch.utils import data
from PIL import Image, ImageDraw
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data_pipeline.logo_parser import UnifiedActionParser

class ActionProgramDataset(data.Dataset):
    """Load and synthesize images from action programs in derived_labels.json instead of real images."""
    
    def __init__(self, derived_labels_path, transform=None, image_size=(160, 160)):
        self.transform = transform
        self.image_size = image_size
        self.action_parser = UnifiedActionParser()
        
        # Load derived labels data
        with open(derived_labels_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} records from {derived_labels_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        
        try:
            # Extract action program from the record
            action_program = record.get('action_program', [])
            if not action_program:
                # Fallback: reconstruct from strokes
                action_program = [stroke['raw_command'] for stroke in record.get('strokes', [])]
            
            if not action_program:
                print(f"Warning: No action program found for {record.get('image_id', 'unknown')}")
                return None, None, None
            
            # Synthesize image from action program
            image_np = self._synthesize_image_from_actions(action_program)
            
            # Convert to tensor
            tensor = torch.from_numpy(image_np).unsqueeze(0).float()
            
            if self.transform:
                tensor = self.transform(tensor)
            
            # Get vertices from the record
            vertices = record.get('vertices', [])
            if not vertices:
                print(f"Warning: No vertices found for {record.get('image_id', 'unknown')}")
                vertices = []
            
            return tensor, record.get('image_id', f'idx_{idx}'), vertices
            
        except Exception as e:
            print(f"Error processing record {idx}: {e}")
            return None, None, None
    
    def _synthesize_image_from_actions(self, action_commands):
        """Synthesize a binary image from action program commands using turtle graphics."""
        try:
            # Parse action commands to get image program
            image_program = self.action_parser._parse_single_image(
                action_commands, 
                "synthetic", 
                True, 
                "test"
            )
            
            if not image_program or not image_program.vertices:
                # Return empty image
                return np.zeros(self.image_size, dtype=np.float32)
            
            # Create PIL image and draw vertices
            img = Image.new('L', self.image_size, 0)  # Black background
            draw = ImageDraw.Draw(img)
            
            # Scale and center vertices to fit image
            vertices = image_program.vertices
            if len(vertices) < 2:
                return np.zeros(self.image_size, dtype=np.float32)
            
            # Get bounding box
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # Scale to fit 80% of image with padding
            width = max_x - min_x
            height = max_y - min_y
            if width <= 0 or height <= 0:
                return np.zeros(self.image_size, dtype=np.float32)
            
            scale_x = (self.image_size[0] * 0.8) / width
            scale_y = (self.image_size[1] * 0.8) / height
            scale = min(scale_x, scale_y)
            
            # Center offset
            offset_x = (self.image_size[0] - width * scale) / 2 - min_x * scale
            offset_y = (self.image_size[1] - height * scale) / 2 - min_y * scale
            
            # Scale and translate vertices
            scaled_vertices = [
                (int(x * scale + offset_x), int(y * scale + offset_y))
                for x, y in vertices
            ]
            
            # Draw lines connecting consecutive vertices
            for i in range(len(scaled_vertices) - 1):
                draw.line([scaled_vertices[i], scaled_vertices[i + 1]], fill=255, width=2)
            
            # For closed shapes, connect last to first
            if len(scaled_vertices) > 2:
                draw.line([scaled_vertices[-1], scaled_vertices[0]], fill=255, width=1)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error synthesizing image: {e}")
            return np.zeros(self.image_size, dtype=np.float32)

# Keep old class for backward compatibility, but mark as deprecated
class ImagePathDataset(ActionProgramDataset):
    """Deprecated: Use ActionProgramDataset instead."""
    def __init__(self, image_paths=None, derived_labels_path=None, transform=None):
        print("Warning: ImagePathDataset is deprecated. Use ActionProgramDataset instead.")
        if derived_labels_path:
            super().__init__(derived_labels_path, transform)
        else:
            raise ValueError("derived_labels_path is required")
