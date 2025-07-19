#!/usr/bin/env python
"""Simple test to verify that the validation script can generate proper images."""
import os
import sys

# Add project root to path 
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = SCRIPT_DIR
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SRC_ROOT)

try:
    # Import exactly as the validation script does
    from bongard_generator.sampler import BongardSampler
    from bongard_generator.config_loader import get_sampler_config
    from bongard_generator.rule_loader import get_all_rules
    from core_models.training_args import config
    
    print("✓ All imports successful")
    
    # Create sampler with the correct config
    sampler_config = get_sampler_config(img_size=config['phase1']['img_size'], generator_mode='genetic')
    print(f"✓ Config created: img_size={sampler_config.img_size}")
    
    # Create sampler
    sampler = BongardSampler(sampler_config)
    print("✓ BongardSampler created")
    
    # Get rules
    rules = get_all_rules()
    
    # Try to generate a single image using the sampler's internal methods
    print("Attempting to generate a test image...")
    
    # Use the sampler's fallback scene generation
    try:
        scene = sampler._render_fallback_scene()
        if scene:
            print("✓ Fallback scene generated successfully")
            if hasattr(scene, 'get_image') and callable(getattr(scene, 'get_image')):
                img = scene.get_image()
                if img:
                    print(f"✓ Image generated: {type(img)}, size: {img.size}")
                    img.save("test_validation_fallback.png")
                    print("✓ Saved test_validation_fallback.png")
                    
                    # Check if image has content (not all black)
                    import numpy as np
                    img_array = np.array(img)
                    unique_vals = np.unique(img_array)
                    print(f"Image unique values: {unique_vals}")
                    
                    if len(unique_vals) > 1:
                        print("✅ SUCCESS: Image has varied content (not black boxes)")
                    else:
                        print("❌ FAIL: Image appears to be all same color (black boxes)")
                else:
                    print("❌ No image returned from scene")
            else:
                print(f"Scene type: {type(scene)}, has get_image: {hasattr(scene, 'get_image')}")
        else:
            print("❌ No scene generated")
    except Exception as e:
        print(f"❌ Fallback scene generation failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
