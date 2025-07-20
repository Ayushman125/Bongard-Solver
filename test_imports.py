#!/usr/bin/env python3
"""Test script to check if all imports work correctly."""

def test_imports():
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ torch import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.builder import BongardGenerator
        print("✅ BongardGenerator imported successfully")
    except ImportError as e:
        print(f"❌ BongardGenerator import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.genetic_generator import GeneticSceneGenerator
        print("✅ GeneticSceneGenerator imported successfully")
    except ImportError as e:
        print(f"❌ GeneticSceneGenerator import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.cp_sampler import CPSATSampler
        print("✅ CPSATSampler imported successfully")
    except ImportError as e:
        print(f"❌ CPSATSampler import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.prototype_action import PrototypeAction
        print("✅ PrototypeAction imported successfully")
    except ImportError as e:
        print(f"❌ PrototypeAction import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.dataset import create_composite_scene
        print("✅ create_composite_scene imported successfully")
    except ImportError as e:
        print(f"❌ create_composite_scene import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.coverage import CoverageTracker
        print("✅ CoverageTracker imported successfully")
    except ImportError as e:
        print(f"❌ CoverageTracker import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.scene_graph import build_scene_graph
        print("✅ build_scene_graph imported successfully")
    except ImportError as e:
        print(f"❌ build_scene_graph import failed: {e}")
        return False
    
    try:
        from src.bongard_generator.gnn_model import SceneGNN
        print("✅ SceneGNN imported successfully")
    except ImportError as e:
        print(f"❌ SceneGNN import failed: {e}")
        return False
    
    print("🎉 All imports successful!")
    return True

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
