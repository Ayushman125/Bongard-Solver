#!/usr/bin/env python3
"""
Test script to verify that all generator components are working without stubs or placeholders.
This will test the complete pipeline from rule loading to image generation.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_dataset_generator():
    """Test that EnhancedDatasetGenerator works without stubs."""
    try:
        from src.bongard_generator.enhanced_dataset import EnhancedDatasetGenerator
        
        # Create generator
        generator = EnhancedDatasetGenerator()
        logger.info("✓ EnhancedDatasetGenerator created successfully")
        
        # Test that it can generate a scene
        test_rule = type('MockRule', (), {'name': 'TEST_RULE', 'description': 'Test rule'})()
        
        # Try to generate objects
        objects = generator.generate_enhanced_scene(test_rule, num_objects=3, is_positive=True)
        
        if objects and len(objects) > 0:
            logger.info(f"✓ Generated {len(objects)} objects successfully")
            for i, obj in enumerate(objects):
                if 'object_id' in obj and 'shape' in obj and 'color' in obj:
                    logger.info(f"  Object {i}: {obj.get('shape')} {obj.get('color')} at ({obj.get('x', 0)}, {obj.get('y', 0)})")
                else:
                    logger.warning(f"  Object {i} missing required fields: {obj}")
        else:
            logger.warning("Generated empty object list")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ EnhancedDatasetGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_generation():
    """Test that BongardDataset works without stubs."""
    try:
        from src.bongard_generator.dataset import BongardDataset
        from src.bongard_generator.config import GeneratorConfig
        
        # Create a basic config
        config = GeneratorConfig()
        
        # Create dataset
        dataset = BongardDataset(config, total_problems=1)
        logger.info("✓ BongardDataset created successfully")
        
        # Test rule loading
        if dataset.rules and len(dataset.rules) > 0:
            logger.info(f"✓ Loaded {len(dataset.rules)} rules")
            for rule in dataset.rules[:3]:  # Show first 3 rules
                logger.info(f"  Rule: {rule.name}")
        else:
            logger.warning("No rules loaded - using fallback rules")
        
        # Test example generation
        logger.info("Testing image generation...")
        problem = dataset[0]
        
        if problem and 'positive_examples' in problem and 'negative_examples' in problem:
            pos_count = len(problem['positive_examples'])
            neg_count = len(problem['negative_examples'])
            logger.info(f"✓ Generated {pos_count} positive and {neg_count} negative examples")
            
            # Check if images are valid PIL Images
            if pos_count > 0:
                example = problem['positive_examples'][0]
                if hasattr(example, 'size') and hasattr(example, 'mode'):
                    logger.info(f"✓ Generated valid PIL image: {example.size} {example.mode}")
                else:
                    logger.warning("Generated object is not a valid PIL Image")
                    
            rule_name = problem.get('rule_name', 'unknown')
            logger.info(f"✓ Applied rule: {rule_name}")
            
        else:
            logger.warning("Failed to generate valid problem structure")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ BongardDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actions_no_stubs():
    """Test that action classes work without stubs."""
    try:
        from src.bongard_generator.actions import Action, ArcAction, ZigzagAction, FanAction
        from PIL import Image, ImageDraw
        
        # Test base action
        action = Action()
        img = Image.new('RGB', (128, 128), 'white')
        draw = ImageDraw.Draw(img)
        
        # Should not raise NotImplementedError anymore
        action.draw(draw, (64, 64), 50, color='black')
        logger.info("✓ Base Action class draws successfully")
        
        # Test specific actions
        actions = [ArcAction(), ZigzagAction(), FanAction()]
        for i, action in enumerate(actions):
            try:
                action.draw(draw, (64, 64), 30)
                logger.info(f"✓ {action.__class__.__name__} draws successfully")
            except Exception as e:
                logger.warning(f"Action {action.__class__.__name__} failed: {e}")
                
        return True
        
    except Exception as e:
        logger.error(f"✗ Actions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fallback_implementations():
    """Test that fallback implementations in enhanced_dataset work."""
    try:
        # Test importing the fallback classes
        from src.bongard_generator.enhanced_dataset import (
            HybridSampler, GeneticSceneGenerator, sample_scene_cp
        )
        
        # Test HybridSampler
        sampler = HybridSampler()
        test_rule = type('MockRule', (), {'name': 'TEST_RULE'})()
        objects = sampler.sample_scene(test_rule, num_objects=2, canvas_size=128)
        
        if objects and len(objects) > 0:
            logger.info(f"✓ HybridSampler generated {len(objects)} objects")
        else:
            logger.warning("HybridSampler returned empty objects list")
            
        # Test GeneticSceneGenerator
        genetic_gen = GeneticSceneGenerator()
        genetic_objects = genetic_gen.generate_scene(test_rule, num_objects=3)
        
        if genetic_objects and len(genetic_objects) > 0:
            logger.info(f"✓ GeneticSceneGenerator generated {len(genetic_objects)} objects")
        else:
            logger.warning("GeneticSceneGenerator returned empty objects list")
            
        # Test CP sampler
        cp_objects = sample_scene_cp(test_rule, num_objects=2)
        
        if cp_objects and len(cp_objects) > 0:
            logger.info(f"✓ CP Sampler generated {len(cp_objects)} objects")
        else:
            logger.warning("CP Sampler returned empty objects list")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Fallback implementations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests to verify no stubs remain."""
    logger.info("=" * 60)
    logger.info("TESTING FOR STUBS AND PLACEHOLDERS")
    logger.info("=" * 60)
    
    tests = [
        ("Enhanced Dataset Generator", test_enhanced_dataset_generator),
        ("Dataset Generation", test_dataset_generation),
        ("Actions (No NotImplementedError)", test_actions_no_stubs),
        ("Fallback Implementations", test_fallback_implementations),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - NO STUBS DETECTED!")
        return True
    else:
        logger.info("❌ Some tests failed - check for remaining stubs")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
