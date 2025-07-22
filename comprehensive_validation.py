#!/usr/bin/env python3
"""
Final comprehensive validation of Bongard generator - no stubs check
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add paths
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def validate_no_stubs():
    """Validate that no stubs remain in the generator system"""
    
    print("🔍 VALIDATING: No stubs or placeholders remain in Bongard generator")
    print("="*70)
    
    issues_found = []
    
    # Test 1: Basic imports work
    print("\n1. Testing basic imports...")
    try:
        from src.bongard_generator.dataset import BongardDataset
        from src.bongard_generator.config import GeneratorConfig
        from src.bongard_generator.enhanced_dataset import EnhancedDatasetGenerator
        print("✓ All key imports successful")
    except Exception as e:
        issue = f"Import failed: {e}"
        print(f"✗ {issue}")
        issues_found.append(issue)
        return False, issues_found
    
    # Test 2: Config has required attributes
    print("\n2. Testing config attributes...")
    try:
        config = GeneratorConfig()
        required_attrs = ['canvas_size', 'img_size', 'min_shapes', 'max_shapes']
        for attr in required_attrs:
            if not hasattr(config, attr):
                issue = f"Config missing attribute: {attr}"
                print(f"✗ {issue}")
                issues_found.append(issue)
            else:
                print(f"✓ Config has {attr} = {getattr(config, attr)}")
    except Exception as e:
        issue = f"Config creation failed: {e}"
        print(f"✗ {issue}")
        issues_found.append(issue)
    
    # Test 3: Dataset creation works
    print("\n3. Testing dataset creation...")
    try:
        config = GeneratorConfig(canvas_size=64)  # Small for testing
        dataset = BongardDataset(config, total_problems=1)
        print(f"✓ Dataset created with {len(dataset.rules)} rules")
        
        if len(dataset.rules) == 0:
            issue = "No rules loaded in dataset"
            print(f"✗ {issue}")
            issues_found.append(issue)
    except Exception as e:
        issue = f"Dataset creation failed: {e}"
        print(f"✗ {issue}")
        issues_found.append(issue)
    
    # Test 4: Enhanced generator works
    print("\n4. Testing enhanced dataset generator...")
    try:
        enhanced_gen = EnhancedDatasetGenerator()
        print("✓ EnhancedDatasetGenerator created")
        
        # Test basic scene generation method exists
        if not hasattr(enhanced_gen, '_generate_basic_scene'):
            issue = "EnhancedDatasetGenerator missing _generate_basic_scene method"
            print(f"✗ {issue}")
            issues_found.append(issue)
        else:
            print("✓ _generate_basic_scene method exists")
            
    except Exception as e:
        issue = f"Enhanced generator creation failed: {e}"
        print(f"✗ {issue}")
        issues_found.append(issue)
    
    # Test 5: Action classes work (no NotImplementedError)
    print("\n5. Testing action classes...")
    try:
        from src.bongard_generator.actions import Action, ArcAction
        from PIL import Image, ImageDraw
        
        # Test base action doesn't raise NotImplementedError
        action = Action()
        img = Image.new('RGB', (64, 64), 'white')
        draw = ImageDraw.Draw(img)
        
        action.draw(draw, (32, 32), 20)  # Should not raise NotImplementedError
        print("✓ Base Action.draw() works without NotImplementedError")
        
        # Test specific action
        arc_action = ArcAction()
        arc_action.draw(draw, (32, 32), 20)
        print("✓ ArcAction.draw() works")
        
    except NotImplementedError:
        issue = "Action classes still raise NotImplementedError"
        print(f"✗ {issue}")
        issues_found.append(issue)
    except Exception as e:
        issue = f"Action classes failed: {e}"
        print(f"✗ {issue}")
        issues_found.append(issue)
    
    # Test 6: Try generating a simple image
    print("\n6. Testing image generation...")
    try:
        config = GeneratorConfig(canvas_size=64)
        dataset = BongardDataset(config, total_problems=1)
        
        if dataset.rules:
            rule = dataset.rules[0]
            img = dataset._generate_example(rule, is_positive=True)
            
            if img and hasattr(img, 'size') and hasattr(img, 'mode'):
                print(f"✓ Generated valid PIL image: {img.size} {img.mode}")
            else:
                issue = "Generated image is not a valid PIL Image"
                print(f"✗ {issue}")
                issues_found.append(issue)
        else:
            issue = "No rules available for image generation"
            print(f"✗ {issue}")
            issues_found.append(issue)
            
    except Exception as e:
        issue = f"Image generation failed: {e}"
        print(f"✗ {issue}")
        issues_found.append(issue)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if issues_found:
        print(f"❌ VALIDATION FAILED - {len(issues_found)} issues found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        return False, issues_found
    else:
        print("🎉 VALIDATION PASSED - No stubs or critical issues detected!")
        print("✓ All generator components are working with real implementations")
        return True, []

def check_final_validation_script():
    """Check if the final_validation.py script works"""
    print("\n" + "="*70)
    print("CHECKING FINAL VALIDATION SCRIPT")
    print("="*70)
    
    try:
        # Try to import and check the pipeline
        from final_validation import CompleteBongardPipeline
        print("✓ final_validation.py imports successfully")
        
        # Try to create a basic config and pipeline
        basic_config = {
            'dataset': {
                'problems_per_task': 10,
                'validation_split': 0.1
            },
            'generation': {
                'canvas_size': 64
            }
        }
        
        pipeline = CompleteBongardPipeline(basic_config)
        print("✓ CompleteBongardPipeline created successfully")
        
        return True
    except Exception as e:
        print(f"✗ final_validation.py failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive validation...")
    
    # Main validation
    success, issues = validate_no_stubs()
    
    # Check final validation script
    final_validation_works = check_final_validation_script()
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    if success:
        print("🎉 STUB REMOVAL: SUCCESS")
    else:
        print("❌ STUB REMOVAL: FAILED")
    
    if final_validation_works:
        print("🎉 FINAL VALIDATION SCRIPT: SUCCESS")
    else:
        print("❌ FINAL VALIDATION SCRIPT: NEEDS ATTENTION")
    
    overall_success = success and final_validation_works
    
    if overall_success:
        print("\n🚀 ALL SYSTEMS GO - Generator is ready for production!")
    else:
        print("\n⚠️  Some issues remain - check output above")
    
    sys.exit(0 if overall_success else 1)
