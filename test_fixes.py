#!/usr/bin/env python3
"""Simple test to verify canvas_size conversion fixes"""
import sys
import os
from types import SimpleNamespace

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

print("Testing canvas_size conversion fixes...")

# Test 1: create_composite_scene with string canvas_size
from src.bongard_generator.dataset import create_composite_scene

cfg = SimpleNamespace(canvas_size="128", bg_texture=None)
objects = [{'x': 64, 'y': 64, 'size': 20, 'shape': 'circle', 'color': 'black', 'fill': 'solid'}]

try:
    img = create_composite_scene(objects, cfg)
    print("✅ create_composite_scene works with string canvas_size")
except Exception as e:
    print(f"❌ create_composite_scene failed: {e}")

# Test 2: apply_noise with string canvas_size  
from src.bongard_generator.styler import apply_noise
from PIL import Image

img = Image.new('RGB', (128, 128), 'white')
cfg_str = SimpleNamespace(canvas_size="128")

try:
    result = apply_noise(img, cfg_str)
    print("✅ apply_noise works with string canvas_size")
except Exception as e:
    print(f"❌ apply_noise failed: {e}")

# Test 3: apply_checker with string canvas_size
from src.bongard_generator.styler import apply_checker

try:
    result = apply_checker(img, cfg_str)
    print("✅ apply_checker works with string canvas_size")
except Exception as e:
    print(f"❌ apply_checker failed: {e}")

# Test 4: build_scene_graph with string canvas_size
from src.bongard_generator.scene_graph import build_scene_graph

objects = [{'x': 64, 'y': 64, 'size': 20, 'shape': 'circle', 'color': 'black', 'fill': 'solid'}]
cfg_graph = SimpleNamespace(canvas_size="128", gnn_radius=0.3)

try:
    graph = build_scene_graph(objects, cfg_graph)
    print("✅ build_scene_graph works with string canvas_size")
except Exception as e:
    print(f"❌ build_scene_graph failed: {e}")

print("All tests completed!")

# Test 5: create_composite_scene with stroke_min fix
from src.bongard_generator.dataset import create_composite_scene
from PIL import Image

print("\nTesting create_composite_scene with stroke_min fix...")
test_imgs = [Image.new('RGB', (224, 224), 'white'), Image.new('RGB', (224, 224), 'black')]
cfg_composite = SimpleNamespace(canvas_size="224", stroke_min=1, jitter_px=2)

try:
    result = create_composite_scene(test_imgs, cfg_composite)
    print("✅ create_composite_scene works with stroke_min fix")
    print(f"Result size: {result.size}")
except Exception as e:
    print(f"❌ create_composite_scene failed: {e}")
    import traceback
    traceback.print_exc()
