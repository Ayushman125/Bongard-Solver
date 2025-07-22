#!/usr/bin/env python3
"""
Simple test for key generator functionality
"""

import sys
import os

# Add project paths
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_basic_import():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        from src.bongard_generator.dataset import BongardDataset
        from src.bongard_generator.config import GeneratorConfig
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_config_creation():
    """Test config creation"""
    try:
        print("Testing config creation...")
        from src.bongard_generator.config import GeneratorConfig
        config = GeneratorConfig()
        print(f"✓ Config created: canvas_size={config.canvas_size}")
        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation"""
    try:
        print("Testing dataset creation...")
        from src.bongard_generator.dataset import BongardDataset
        from src.bongard_generator.config import GeneratorConfig
        
        config = GeneratorConfig()
        dataset = BongardDataset(config, total_problems=1)
        print(f"✓ Dataset created with {len(dataset.rules)} rules")
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

def test_single_image_generation():
    """Test generating a single image"""
    try:
        print("Testing single image generation...")
        from src.bongard_generator.dataset import BongardDataset
        from src.bongard_generator.config import GeneratorConfig
        
        config = GeneratorConfig(canvas_size=128)  # Smaller for faster test
        dataset = BongardDataset(config, total_problems=1)
        
        # Test single example generation
        rule = dataset.rules[0] if dataset.rules else None
        if rule:
            img = dataset._generate_example(rule, is_positive=True)
            if img and hasattr(img, 'size'):
                print(f"✓ Generated image: {img.size} {img.mode}")
                return True
            else:
                print("✗ Generated image is invalid")
                return False
        else:
            print("✗ No rules available")
            return False
    except Exception as e:
        print(f"✗ Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_tests():
    """Run simple tests"""
    print("=" * 50)
    print("SIMPLE GENERATOR TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Config Creation", test_config_creation),
        ("Dataset Creation", test_dataset_creation),
        ("Single Image Generation", test_single_image_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
        
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    return passed == len(results)

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
