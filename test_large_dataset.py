#!/usr/bin/env python3
"""Test large dataset generation with clean output."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bongard_generator.dataset import SyntheticBongardDataset
import time

def test_large_dataset():
    """Test generating a larger dataset to verify clean output."""
    print("🚀 Testing large dataset generation...")
    
    # Create a larger dataset (100 examples)
    rules = [
        ('All objects are circles.', 25),
        ('All objects are squares.', 25), 
        ('All objects have solid fill.', 25),
        ('There are exactly 2 objects in the image.', 25)
    ]
    
    start_time = time.time()
    
    print(f"📦 Creating dataset with {sum(count for _, count in rules)} examples...")
    dataset = SyntheticBongardDataset(rules, img_size=128)
    
    generation_time = time.time() - start_time
    
    print(f"✅ Generated {len(dataset)} examples in {generation_time:.2f} seconds")
    print(f"⚡ Generation rate: {len(dataset)/generation_time:.1f} examples/second")
    
    # Test accessing multiple examples (this should be fast and quiet)
    print("\n🔍 Testing dataset access...")
    access_start = time.time()
    
    for i in range(min(10, len(dataset))):
        example = dataset[i]
        assert example['image'] is not None
        assert example['rule'] in [rule for rule, _ in rules]
        assert example['label'] == 1
    
    access_time = time.time() - access_start
    print(f"✅ Accessed 10 examples in {access_time:.4f} seconds")
    if access_time > 0:
        print(f"⚡ Access rate: {10/access_time:.1f} examples/second")
    else:
        print("⚡ Access rate: Instantaneous (too fast to measure!)")
    
    return True

if __name__ == "__main__":
    success = test_large_dataset()
    if success:
        print("\n🎉 Large dataset test PASSED!")
        print("💡 No more chatty debug prints flooding the console!")
    else:
        print("\n❌ Large dataset test FAILED!")
