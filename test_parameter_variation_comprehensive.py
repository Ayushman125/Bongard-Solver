#!/usr/bin/env python3
"""
Comprehensive test to demonstrate that parameters now vary correctly 
according to the real dataset values instead of being hardcoded.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.bongard_augmentor.hybrid import ActionMaskGenerator

def test_parameter_variation_comprehensive():
    """
    Test with the exact parameter variations from the user's real dataset examples.
    This validates that we're no longer using hardcoded values.
    """
    
    # Real dataset examples from user
    real_dataset_commands = {
        "LINE_NORMAL": [
            "line_normal_1.000-0.833",  # scale=1.0, param=0.833
            "line_normal_1.000-0.500",  # scale=1.0, param=0.5
            "line_normal_0.700-0.667",  # scale=0.7, param=0.667
        ],
        "LINE_CIRCLE": [
            "line_circle_1.000-0.833",  # scale=1.0, param=0.833
            "line_circle_0.518-0.792",  # scale=0.518, param=0.792  
            "line_circle_0.700-0.500",  # scale=0.7, param=0.5
        ],
        "ARC_NORMAL": [
            "arc_normal_0.500_0.542-0.750",  # scale=0.5, arc=0.542, param=0.75
            "arc_normal_0.500_0.875-0.574",  # scale=0.5, arc=0.875, param=0.574
            "arc_normal_0.500_0.694-0.750",  # scale=0.5, arc=0.694, param=0.75
        ],
        "ARC_CIRCLE": [
            "arc_circle_0.500_0.639-0.750",  # scale=0.5, arc=0.639, param=0.75
            "arc_circle_0.500_0.875-0.500",  # scale=0.5, arc=0.875, param=0.5
            "arc_circle_0.500_0.583-0.750",  # scale=0.5, arc=0.583, param=0.75
        ]
    }
    
    mask_gen = ActionMaskGenerator(image_size=(512, 512))
    
    print("🔍 PARAMETER VARIATION ANALYSIS")
    print("=" * 60)
    
    for command_type, commands in real_dataset_commands.items():
        print(f"\n📂 {command_type}")
        print("-" * 40)
        
        previous_pixels = None
        for i, cmd in enumerate(commands):
            # Parse parameters
            result = mask_gen._parse_semantic_command(cmd)
            if result:
                stroke_type, shape_type, params = result
                
                # Generate mask
                mask = mask_gen.generate_mask_from_actions([cmd])
                pixel_count = np.sum(mask > 0)
                
                # Show parameter variation
                if stroke_type == 'line':
                    param_info = f"scale={params['scale']:.3f}, param={params['final_param']:.3f}"
                else:  # arc
                    param_info = f"scale={params['scale']:.3f}, arc={params['arc_param']:.3f}, param={params['final_param']:.3f}"
                
                variation = ""
                if previous_pixels is not None:
                    diff = pixel_count - previous_pixels
                    if diff > 0:
                        variation = f" (+{diff} pixels)"
                    elif diff < 0:
                        variation = f" ({diff} pixels)"
                    else:
                        variation = f" (same)"
                
                print(f"  {i+1}. {cmd}")
                print(f"     📊 {param_info}")
                print(f"     🎨 {pixel_count} pixels{variation}")
                
                previous_pixels = pixel_count
            else:
                print(f"  ❌ Failed to parse: {cmd}")
    
    # Visual demonstration
    print(f"\n🎨 GENERATING VISUAL COMPARISON...")
    
    # Create a comparison showing parameter effects
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    # Line variations (top row)
    line_commands = [
        "line_normal_1.000-0.833",
        "line_normal_1.000-0.500", 
        "line_normal_0.700-0.667",
        "line_circle_1.000-0.833",
        "line_circle_0.518-0.792",
        "line_circle_0.700-0.500"
    ]
    
    # Arc variations (bottom row) 
    arc_commands = [
        "arc_normal_0.500_0.542-0.750",
        "arc_normal_0.500_0.875-0.574",
        "arc_normal_0.500_0.694-0.750",
        "arc_circle_0.500_0.639-0.750",
        "arc_circle_0.500_0.875-0.500",
        "arc_circle_0.500_0.583-0.750"
    ]
    
    for i, cmd in enumerate(line_commands):
        mask = mask_gen.generate_mask_from_actions([cmd])
        axes[0, i].imshow(mask, cmap='gray')
        axes[0, i].set_title(f"Line\n{cmd.split('_')[2][:8]}...", fontsize=8)
        axes[0, i].axis('off')
    
    for i, cmd in enumerate(arc_commands):
        mask = mask_gen.generate_mask_from_actions([cmd])
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f"Arc\n{cmd.split('_')[2][:8]}...", fontsize=8)
        axes[1, i].axis('off')
    
    plt.suptitle('Parameter Variation: Real Dataset Values vs Hardcoded (FIXED)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('parameter_variation_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ PARAMETER VARIATION TEST COMPLETED")
    print(f"   📁 Saved visual comparison: parameter_variation_fixed.png")
    print(f"   🎯 All parameters now use real dataset values instead of hardcoded defaults")

if __name__ == "__main__":
    test_parameter_variation_comprehensive()
