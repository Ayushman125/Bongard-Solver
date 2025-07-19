"""
Visualization script for inspecting generated Bongard samples.
Creates a mosaic view of multiple samples to verify diversity.
"""


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import random

def safe_randint(a: int, b: int) -> int:
    """Safe random integer generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    return random.randint(lo, hi)

def safe_randrange(a: int, b: int) -> int:
    """Safe random range generator that handles inverted ranges."""
    lo, hi = min(a, b), max(a, b)
    # randrange excludes hi, so we +1
    return random.randrange(lo, hi+1)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


# Use hybrid generator for dataset creation
try:
    from src.bongard_generator.hybrid_sampler import HybridSampler
    from src.bongard_generator.config_loader import get_sampler_config
    GENERATOR_AVAILABLE = True
except ImportError:
    print("Warning: Could not import hybrid generator. Mosaic will use fallback test data.")
    GENERATOR_AVAILABLE = False

def show_mosaic(ds, N=16, C=4, show_cnn_score=False):
    """
    Display a mosaic of N samples from the dataset in a C-column grid.
    
    Args:
        ds: Dataset object with samples
        N: Total number of samples to show
        C: Number of columns in the grid
    """
    rows = N // C
    fig, axs = plt.subplots(rows, C, figsize=(C*3, rows*3))
    
    # Handle single row case
    if rows == 1:
        axs = axs.reshape(1, -1)
    
    for i, ax in enumerate(axs.flatten()):
        if i < N and i < len(ds):
            sample = ds[i]
            img = sample.get('image', None)
            if img is not None:
                if hasattr(img, 'squeeze'):
                    img = img.squeeze()
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                # Title: rule, polarity, (optional CNN score)
                rule_info = sample.get('rule', 'Unknown')
                polarity = sample.get('polarity', '?')
                title = f'{rule_info[:15]}... ({polarity})'
                if show_cnn_score and 'cnn_score' in sample:
                    title += f' | CNN: {sample["cnn_score"]:.2f}'
                ax.set_title(title, fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No image', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Empty', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_simple_dataset(num_samples=16):
    """
    Create a simple dataset for testing visualization.
    """
    samples = []
    rules = ['shape', 'count', 'fill', 'relation']
    
    for i in range(num_samples):
        # Create a simple black and white image
        img = np.ones((128, 128), dtype=np.uint8) * 255  # White background
        
        # Add some black shapes
        rule = random.choice(rules)
        
        if rule == 'shape':
            # Draw a circle or square
            center = (64, 64)
            if random.choice([True, False]):
                # Circle
                y, x = np.ogrid[:128, :128]
                mask = (x - center[0])**2 + (y - center[1])**2 <= 20**2
                img[mask] = 0
                shape_type = 'circle'
            else:
                # Square
                img[44:84, 44:84] = 0
                shape_type = 'square'
        else:
            # Default: random shapes
            for _ in range(safe_randint(1, 3)):
                x, y = safe_randint(20, 108), safe_randint(20, 108)
                size = safe_randint(10, 30)
                img[y:y+size, x:x+size] = 0
            shape_type = 'random'
        
        sample = {
            'image': img,
            'rule': f'{rule}({shape_type})',
            'label': random.choice([0, 1])
        }
        samples.append(sample)
    
    return samples


def main():
    """Main function to run the inspection tool."""
    print("Bongard Sample Inspector")
    print("========================")
    try:
        if GENERATOR_AVAILABLE:
            print("Generating Bongard scenes using hybrid generator...")
            config = get_sampler_config(total=16)
            if 'hybrid_split' not in config['data']:
                config['data']['hybrid_split'] = {'cp': 0.7, 'ga': 0.3}
            sampler = HybridSampler(config)
            imgs, labels = sampler.build_synth_holdout(n=16)
            
            # Convert to dataset format for mosaic display
            ds = []
            for img, label in zip(imgs, labels):
                ds.append({
                    'image': np.array(img) if hasattr(img, 'size') else img,
                    'label': label,
                    'rule': 'hybrid_generated',
                    'polarity': 'pos' if label == 1 else 'neg'
                })
            print(f"Generated {len(ds)} scenes using hybrid generator.")
        else:
            print("Hybrid generator not available, creating simple dataset...")
            ds = create_simple_dataset(16)
        
        print("Displaying mosaic...")
        show_mosaic(ds, N=min(16, len(ds)), C=4)
    except Exception as e:
        print(f"Error in main: {e}")
        print("\nFalling back to simple visualization...")
        ds = create_simple_dataset(16)
        show_mosaic(ds)

if __name__ == "__main__":
    main()
