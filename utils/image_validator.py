#!/usr/bin/env python3
"""
Image Generation Validator
Validates that synthetic Bongard images are generated correctly with actual shapes.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def validate_synthetic_images(num_samples: int = 8, img_size: int = 128, save_plot: bool = True) -> bool:
    """
    Validate that synthetic images contain actual shapes (not blank).
    
    Args:
        num_samples: Number of sample images to generate and check
        img_size: Size of generated images
        save_plot: Whether to save visualization plot
        
    Returns:
        bool: True if all images contain shapes, False otherwise
    """
    try:
        from bongard_generator.dataset import SyntheticBongardDataset
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print(f"🔍 Validating {num_samples} synthetic images...")
    
    # Create a small dataset with different rules
    rules = [
        ('All objects are circles.', 2),
        ('All objects are squares.', 2), 
        ('All objects have solid fill.', 2),
        ('There are exactly 2 objects in the image.', 2)
    ]
    
    try:
        dataset = SyntheticBongardDataset(rules, img_size=img_size)
        print(f"📊 Generated {len(dataset)} examples")
        
        # Check images for validity
        all_valid = True
        generated_examples = []
        
        for i in range(min(num_samples, len(dataset))):
            example = dataset[i]
            generated_examples.append(example)
            
            img = np.array(example['image'])
            unique_vals = np.unique(img)
            
            print(f"  📷 Image {i}: Rule='{example['rule']}', Shape={img.shape}, "
                  f"Unique values={len(unique_vals)}, Mean={img.mean():.1f}")
            
            if len(unique_vals) <= 1:
                print(f"❌ Image {i} is blank (uniform color)")
                all_valid = False
        
        # Create visualization if requested
        if save_plot and generated_examples:
            create_validation_plot(generated_examples)
        
        if all_valid:
            print(f"✅ SUCCESS: All {len(generated_examples)} images contain actual shapes!")
            return True
        else:
            print(f"❌ FAILED: Some images are blank")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False

def create_validation_plot(examples: list, output_path: str = "validation_images.png"):
    """Create and save visualization plot of generated images."""
    num_examples = len(examples)
    if num_examples == 0:
        return
        
    # Calculate grid dimensions
    cols = min(4, num_examples)
    rows = (num_examples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Generated Bongard Problem Images - Validation', fontsize=16)
    
    for i, example in enumerate(examples):
        if i >= len(axes):
            break
            
        img = np.array(example['image'])
        rule = example['rule']
        label = example['label']
        
        axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(f'{rule}\nLabel: {label}', fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(examples), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Saved validation plot to '{output_path}'")

if __name__ == "__main__":
    success = validate_synthetic_images()
    if success:
        print("\n🎉 Image generation validation PASSED!")
        print("The synthetic image generation is working correctly!")
    else:
        print("\n⚠️ Image generation validation FAILED!")
        sys.exit(1)
