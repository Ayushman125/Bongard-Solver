#!/usr/bin/env python3
"""
Minimal test to find the exact string/int division error.
"""
import sys
import os
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).parent / 'src'))

print("Testing basic imports...")

try:
    print("1. Testing config import...")
    from src.bongard_generator.config import GeneratorConfig
    print("✅ Config import successful")
    
    print("2. Creating config...")
    config = GeneratorConfig(
        img_size=128,
        prototype_path="shapebongordV2",
        use_gan_stylization=False,
        use_meta_controller=True,
        bg_texture="none",
        noise_level=0.0
    )
    print("✅ Config creation successful")
    print(f"   img_size type: {type(config.img_size)}, value: {config.img_size}")
    
    print("3. Testing YAML config load...")
    from config import load_config
    yaml_config = load_config()
    print("✅ YAML config loaded")
    
    print("4. Testing basic scene creation...")
    from src.bongard_generator.dataset import create_composite_scene
    objects = []  # Empty objects list for basic test
    scene_img = create_composite_scene(objects, config)
    print("✅ Scene creation successful")
    print(f"   Scene image size: {scene_img.size}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
