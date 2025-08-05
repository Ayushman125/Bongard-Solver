#!/usr/bin/env python3
"""
Test parameter variation in semantic commands to verify we're using real parameter values.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.bongard_augmentor.hybrid import ActionMaskGenerator

def test_parameter_variation():
    """Test that different parameter values produce visually different results."""
    
    # Test data from the user's examples
    test_commands = [
        # Line normal with different parameters
        ['line_normal_1.000-0.833'],
        ['line_normal_1.000-0.500'], 
        ['line_normal_0.700-0.667'],
        
        # Line circle with different parameters
        ['line_circle_1.000-0.833'],
        ['line_circle_0.518-0.792'],
        ['line_circle_0.700-0.500'],
        
        # Arc normal with different parameters
        ['arc_normal_0.500_0.542-0.750'],
        ['arc_normal_0.500_0.875-0.574'],
        ['arc_normal_0.500_0.694-0.750'],
        
        # Arc circle with different parameters
        ['arc_circle_0.500_0.639-0.750'],
        ['arc_circle_0.500_0.875-0.500'],
        ['arc_circle_0.500_0.583-0.750'],
    ]
    
    # Create mask generator
    mask_gen = ActionMaskGenerator(image_size=(512, 512))
    
    # Generate masks for each command
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, commands in enumerate(test_commands):
        mask = mask_gen.generate_mask_from_actions(commands)
        
        # Calculate some statistics to verify variation
        pixel_count = np.sum(mask > 0)
        
        axes[i].imshow(mask, cmap='gray')
        axes[i].set_title(f'{commands[0][:20]}...\nPixels: {pixel_count}', fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_parameter_variation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Also test parameter parsing directly
    print("\n=== PARAMETER PARSING TEST ===")
    for commands in test_commands:
        cmd = commands[0]
        result = mask_gen._parse_semantic_command(cmd)
        if result:
            stroke_type, shape_type, params = result
            print(f"{cmd} -> {stroke_type}_{shape_type}: {params}")
        else:
            print(f"{cmd} -> FAILED TO PARSE")

if __name__ == "__main__":
    test_parameter_variation()
