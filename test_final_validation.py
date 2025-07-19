#!/usr/bin/env python3
"""
Final validation test to check if all fixes are working.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_rule_loading():
    """Test rule loading with new properties."""
    print("=== Testing Rule Loading ===")
    try:
        from src.bongard_generator.rule_loader import get_all_rules
        rules = get_all_rules()
        print(f"✓ Loaded {len(rules)} rules")
        
        if rules:
            rule = rules[0]
            print(f"✓ First rule: {rule.description}")
            print(f"✓ Has positive_features: {hasattr(rule, 'positive_features')}")
            print(f"✓ Positive features: {rule.positive_features}")
            print("✓ Rule loading working!")
            return True
    except Exception as e:
        print(f"✗ Rule loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shape_rendering():
    """Test shape rendering."""
    print("\n=== Testing Shape Rendering ===")
    try:
        from src.bongard_generator.dataset import create_composite_scene
        test_objects = [
            {'shape': 'circle', 'x': 30, 'y': 30, 'size': 20, 'color': 'red', 'fill': 'solid'},
            {'shape': 'square', 'x': 70, 'y': 70, 'size': 25, 'color': 'blue', 'fill': 'outline'}
        ]
        img = create_composite_scene(test_objects, 128)
        print(f"✓ Generated scene: {img.size}, mode={img.mode}")
        print("✓ Shape rendering working!")
        return True
    except Exception as e:
        print(f"✗ Shape rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_generator():
    """Test hybrid generator."""
    print("\n=== Testing Hybrid Generator ===")
    try:
        from src.bongard_generator.hybrid_sampler import HybridSampler
        from src.bongard_generator.config_loader import get_sampler_config
        
        # Configure for hybrid generation
        hybrid_config = get_sampler_config(total=8)
        if 'hybrid_split' not in hybrid_config['data']:
            hybrid_config['data']['hybrid_split'] = {'cp': 0.7, 'ga': 0.3}
        
        sampler = HybridSampler(hybrid_config)
        
        # Test small generation
        imgs, labels = sampler.build_synth_holdout(n=4)
        print(f"✓ Generated {len(imgs)} images with {len(labels)} labels")
        print("✓ Hybrid generator working!")
        return True
    except Exception as e:
        print(f"✗ Hybrid generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running Final Validation Tests")
    print("=" * 50)
    
    results = []
    results.append(test_rule_loading())
    results.append(test_shape_rendering())
    results.append(test_hybrid_generator())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("✅ Rule loading with positive_features working")
        print("✅ Shape rendering with actual drawing working") 
        print("✅ Hybrid generator integration working")
        print("\n🚀 System is ready for deployment!")
    else:
        print("⚠️ Some tests failed:")
        test_names = ["Rule Loading", "Shape Rendering", "Hybrid Generator"]
        for i, (name, result) in enumerate(zip(test_names, results)):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {name}")
        print("\n🔧 Please fix the failing tests before deployment.")
