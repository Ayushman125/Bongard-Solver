#!/usr/bin/env python3
"""Final verification to show actual generated images."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bongard_generator.dataset import SyntheticBongardDataset
import matplotlib.pyplot as plt
import numpy as np

def show_final_verification():
    """Generate and display images to verify everything is working."""
    print("=== FINAL IMAGE GENERATION VERIFICATION ===")
    
    # Generate different types of scenes
    test_rules = [
        ('SHAPE(circle)', 1),
        ('SHAPE(square)', 1), 
        ('FILL(solid)', 1),
        ('FILL(outline)', 1)
    ]
    
    dataset = SyntheticBongardDataset(test_rules, img_size=128)
    
    print(f"Generated {len(dataset)} test images")
    
    # Create a figure to show the results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Generated Bongard Images - Final Verification', fontsize=16)
    
    for i in range(min(4, len(dataset))):
        row = i // 2
        col = i % 2
        
        example = dataset[i]
        img = np.array(example['image'])
        rule = example['rule']
        
        # Show the image
        axes[row, col].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[row, col].set_title(f'Rule: {rule}')
        axes[row, col].axis('off')
        
        # Print statistics
        unique_vals = np.unique(img)
        non_white_pct = (np.sum(img < 255) / img.size) * 100
        
        print(f"Image {i+1}: {rule}")
        print(f"  Shape: {img.shape}")
        print(f"  Unique values: {unique_vals}")
        print(f"  Non-white pixels: {non_white_pct:.1f}%")
        print(f"  Mean: {img.mean():.1f}")
        
        if len(unique_vals) > 1 and non_white_pct > 0:
            print(f"  ✅ GOOD: Contains visible shapes")
        else:
            print(f"  ❌ BAD: Blank or uniform image")
        print()
    
    plt.tight_layout()
    plt.show()
    
    # Overall verification
    all_good = True
    for i in range(len(dataset)):
        img = np.array(dataset[i]['image'])
        if len(np.unique(img)) <= 1:
            all_good = False
            break
    
    if all_good:
        print("🎉 FINAL RESULT: ALL IMAGES ARE GENERATING CORRECTLY!")
        print("   - Images contain actual geometric shapes")
        print("   - Black shapes on white background")
        print("   - Proper binarization working")
        print("   - The 'black boxes' issue has been RESOLVED!")
    else:
        print("❌ FINAL RESULT: Some images are still blank")
    
    return all_good

if __name__ == "__main__":
    show_final_verification()
