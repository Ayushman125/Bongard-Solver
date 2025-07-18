#!/usr/bin/env python3
"""
Simple test script for the modular Bongard generator.
This script tests the basic functionality before integration.
"""

import os
import sys
import logging

# Add the src directory to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_ROOT)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test basic imports."""
    try:
        from bongard_generator import BongardSampler, SamplerConfig, validate_installation
        from bongard_generator.config_loader import get_sampler_config
        logger.info("✓ Basic imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from bongard_generator.config_loader import get_sampler_config
        config = get_sampler_config()
        logger.info(f"✓ Config loaded: img_size={config.img_size}")
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False

def test_validation():
    """Test validation suite."""
    try:
        from bongard_generator import validate_installation
        result = validate_installation()
        if result:
            logger.info("✓ Validation passed")
        else:
            logger.warning("⚠ Some validations failed")
        return True
    except Exception as e:
        logger.error(f"✗ Validation test failed: {e}")
        return False

def test_sampler():
    """Test basic sampler functionality."""
    try:
        from bongard_generator import BongardSampler, generate_single_problem
        
        # Test simple problem generation
        problem = generate_single_problem(rule="SHAPE(CIRCLE)", num_pos_scenes=2, num_neg_scenes=2)
        
        if problem:
            logger.info(f"✓ Generated problem: {problem['rule']['description']}")
            logger.info(f"  Positive scenes: {len(problem['positive_scenes'])}")
            logger.info(f"  Negative scenes: {len(problem['negative_scenes'])}")
            return True
        else:
            logger.error("✗ Problem generation returned None")
            return False
            
    except Exception as e:
        logger.error(f"✗ Sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    logger.info("Starting modular generator test...")
    
    tests = [
        ("Import Test", test_imports),
        ("Config Test", test_config),
        ("Validation Test", test_validation),
        ("Sampler Test", test_sampler),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Generator is ready for integration.")
        return True
    else:
        logger.warning("⚠ Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
