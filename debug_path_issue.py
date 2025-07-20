#!/usr/bin/env python3
"""
Debug script to isolate the path division error.
"""
import sys
import os
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).parent / 'src'))

# Try basic imports first
print("Testing basic imports...")

try:
    from src.bongard_generator.config import GeneratorConfig
    print("✅ Config import successful")
    
    config = GeneratorConfig(
        img_size=128,
        prototype_path="shapebongordV2",
        use_gan_stylization=False,
        use_meta_controller=True,
        bg_texture="none",
        noise_level=0.0
    )
    print("✅ Config creation successful")
    
except Exception as e:
    print(f"❌ Config error: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.bongard_generator.dataset import BongardDataset
    print("✅ Dataset import successful")
    
    dataset = BongardDataset(config, total_problems=1)
    print("✅ Dataset creation successful")
    
    problem = dataset[0]
    print("✅ Problem generation successful")
    
except Exception as e:
    print(f"❌ Dataset error: {e}")
    import traceback
    traceback.print_exc()
