#!/usr/bin/env python
"""Test to reproduce the black box issue specifically."""
import os
import sys

# Add project root to path 
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = SCRIPT_DIR  # We're at root level
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)     # for core_models/
sys.path.insert(0, SRC_ROOT)      # for src/data, src/perception, src/utils

import numpy as np
from PIL import Image

def test_empty_dataset_issue():
    """Test what happens when BongardDataset has no examples."""
    
    print("🔍 Testing Empty Dataset Issue")
    print("=" * 50)
    
    # Test the exact scenario from the logs - a dataset with 0 examples
    from bongard_generator.dataset import BongardDataset
    
    # Create dataset for 'count_eq_4' which showed "→ Examples built: 0"
    print("Creating dataset for 'count_eq_4' rule...")
    dataset = BongardDataset(
        canvas_size=128,
        min_obj_size=20,
        max_obj_size=60,
        target_quota=2,  # Small quota for testing
        rule_list=['count_eq_4']  # This rule failed in the logs
    )
    
    print(f"Dataset has {len(dataset.examples)} examples")
    
    if len(dataset.examples) == 0:
        print("❌ Dataset is empty! This is the source of black boxes.")
        print("Trying to access index 0 from empty dataset...")
        try:
            example = dataset[0]
            print(f"Somehow got example: {example}")
        except IndexError as e:
            print(f"✓ Got expected IndexError: {e}")
        except Exception as e:
            print(f"❌ Got unexpected error: {e}")
            return False
    else:
        print(f"✓ Dataset has {len(dataset.examples)} examples")
        # Test accessing an example
        try:
            example = dataset[0]
            img = example['image']
            print(f"✓ Image type: {type(img)}")
            print(f"✓ Image size: {img.size if hasattr(img, 'size') else 'No size'}")
            
            # Save the image to check quality
            img.save("debug_dataset_example.png")
            print("✓ Saved debug_dataset_example.png")
            
            # Check if image is actually black
            img_array = np.array(img)
            print(f"Image stats: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.1f}")
            if img_array.max() == img_array.min():
                print("❌ Image is uniform color (likely black box)!")
                return False
            else:
                print("✓ Image has variation (not a black box)")
                
        except Exception as e:
            print(f"❌ Failed to access example: {e}")
            return False
    
    return True

def test_hybrid_sampler_robustness():
    """Test if the hybrid sampler handles empty datasets gracefully."""
    
    print("\n🔍 Testing Hybrid Sampler Robustness")
    print("=" * 50)
    
    try:
        # Simulate the hybrid sampler workflow
        from bongard_generator.rule_loader import get_all_rules
        from bongard_generator.dataset import BongardDataset
        from PIL import Image
        import numpy as np
        
        rules = get_all_rules()
        problematic_rules = ['count_eq_4']  # Known to generate 0 examples
        
        images = []
        labels = []
        
        for rule_name in problematic_rules:
            print(f"Testing rule: {rule_name}")
            
            # Find the actual rule object
            rule = None
            for r in rules:
                if hasattr(r, 'name') and r.name == rule_name:
                    rule = r
                    break
            
            if rule is None:
                print(f"❌ Could not find rule {rule_name}")
                continue
                
            # Create dataset for this rule (similar to hybrid sampler)
            dataset = BongardDataset(
                canvas_size=128,
                target_quota=2,
                rule_list=[rule_name]
            )
            
            print(f"  Dataset examples: {len(dataset.examples)}")
            
            if len(dataset.examples) == 0:
                print("  ❌ Empty dataset - this will cause black boxes!")
                # This is where the hybrid sampler might fail
                # Let's see what happens if we try to add a "fake" image
                black_img = Image.new('L', (128, 128), 0)  # Pure black image
                images.append(black_img)
                labels.append(0)
                print("  Added black placeholder image")
            else:
                # Get the first example
                example = dataset[0]
                images.append(example['image'])
                labels.append(example['label'])
                print("  ✓ Added valid image")
        
        print(f"\nFinal results: {len(images)} images, {len(labels)} labels")
        
        # Test the images
        for i, img in enumerate(images):
            if isinstance(img, Image.Image):
                img_array = np.array(img)
                if img_array.max() == img_array.min() == 0:
                    print(f"  Image {i}: BLACK BOX (all pixels = {img_array[0,0]})")
                else:
                    print(f"  Image {i}: Valid (min={img_array.min()}, max={img_array.max()})")
            else:
                print(f"  Image {i}: Invalid type {type(img)}")
        
        return len(images) > 0
        
    except Exception as e:
        print(f"❌ Hybrid sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Black Box Generation Issues")
    print("=" * 60)
    
    success1 = test_empty_dataset_issue()
    success2 = test_hybrid_sampler_robustness()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ Tests completed successfully")
    else:
        print("❌ Some tests failed - black box issue identified")
    print("=" * 60)
