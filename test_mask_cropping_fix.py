#!/usr/bin/env python3
"""
Test the mask cropping fix to ensure canvas size is preserved.
"""

import numpy as np
import cv2
from src.bongard_augmentor.hybrid import HybridAugmentationPipeline

def test_mask_cropping_fix():
    """Test that masks maintain their original 512x512 canvas size."""
    
    # Create a basic config
    config = {
        'data': {
            'action_programs_dir': 'data/raw/ShapeBongard_V2',
            'output_path': 'test_output.pkl',
            'n_select': 1
        },
        'enable_post_processing': True  # Enable post-processing to test the fix
    }
    
    # Initialize pipeline
    pipeline = HybridAugmentationPipeline(config)
    
    # Test commands from the user's examples
    test_commands = [
        ['line_normal_1.000-0.833'],
        ['arc_circle_0.500_0.639-0.750'],
        ['line_triangle_1.000-0.500']
    ]
    
    print("🔍 TESTING MASK CROPPING FIX")
    print("=" * 50)
    
    for i, commands in enumerate(test_commands):
        print(f"\n📝 Test {i+1}: {commands[0]}")
        
        # Generate mask
        mask = pipeline.mask_generator.generate_mask_from_actions(commands)
        print(f"   🎨 Original mask shape: {mask.shape}")
        print(f"   📊 Original pixel count: {np.sum(mask > 0)}")
        
        # Apply post-processing (where cropping was happening)
        processed_mask = pipeline._post_process_mask(mask)
        print(f"   🔧 Processed mask shape: {processed_mask.shape}")
        print(f"   📊 Processed pixel count: {np.sum(processed_mask > 0)}")
        
        # Check if shape is preserved
        if mask.shape == processed_mask.shape:
            print(f"   ✅ Canvas size preserved: {mask.shape}")
        else:
            print(f"   ❌ Canvas size changed: {mask.shape} → {processed_mask.shape}")
        
        # Save test images to verify visually
        cv2.imwrite(f'test_mask_original_{i}.png', mask)
        cv2.imwrite(f'test_mask_processed_{i}.png', processed_mask)
        print(f"   💾 Saved: test_mask_original_{i}.png, test_mask_processed_{i}.png")
    
    print(f"\n✅ MASK CROPPING FIX TEST COMPLETED")
    print(f"   Check the saved images to verify masks are not cropped")

if __name__ == "__main__":
    test_mask_cropping_fix()
