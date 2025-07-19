#!/usr/bin/env python
"""Debug script to replicate validation script image generation."""
import os
import sys

# Add project root to path 
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = SCRIPT_DIR  # We're at root level
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)     # for core_models/
sys.path.insert(0, SRC_ROOT)      # for src/data, src/perception, src/utils

print(f"REPO_ROOT: {REPO_ROOT}")
print(f"SRC_ROOT: {SRC_ROOT}")
print(f"Current working dir: {os.getcwd()}")

try:
    # Debug: Check if we can import the modules that __init__.py tries to import
    print("Testing individual imports...")
    try:
        from src.bongard_generator.config_loader import SamplerConfig, get_config, get_sampler_config as real_get_sampler_config
        print("✓ config_loader imports work")
    except Exception as e:
        print(f"❌ config_loader import failed: {e}")
    
    try:
        from src.bongard_generator.rule_loader import BongardRule, get_all_rules, get_rule_by_description
        print("✓ rule_loader imports work")
    except Exception as e:
        print(f"❌ rule_loader import failed: {e}")
    
    try:
        from src.bongard_generator.sampler import BongardSampler
        print("✓ sampler imports work")
    except Exception as e:
        print(f"❌ sampler import failed: {e}")
    
    # Now try the way validation script imports
    print("Testing validation script style imports...")
    from bongard_generator.config_loader import get_sampler_config
    from bongard_generator.rule_loader import get_all_rules
    from bongard_generator.sampler import BongardSampler
    from core_models.training_args import config
    
    print("✓ Successfully imported validation script components")
    
    # Debug the get_sampler_config function
    print(f"Phase1 img_size from config: {config['phase1']['img_size']}")
    
    # Test both versions
    print("Testing real_get_sampler_config (direct import):")
    print(f"Function location: {real_get_sampler_config}")
    print(f"Function file: {real_get_sampler_config.__module__}")
    real_config = real_get_sampler_config(img_size=config['phase1']['img_size'], generator_mode='genetic')
    print(f"Real config type: {type(real_config)}")
    
    if hasattr(real_config, 'img_size'):
        print(f"Real config img_size: {real_config.img_size}")
    else:
        print(f"Real config contents: {real_config}")
    
    print("Testing get_sampler_config (through package):")
    print(f"Function location: {get_sampler_config}")
    print(f"Function file: {get_sampler_config.__module__}")
    package_config = get_sampler_config(img_size=config['phase1']['img_size'], generator_mode='genetic')
    print(f"Package config type: {type(package_config)}")
    
    # Use the working version - both should work now
    sampler_config = real_config
    
    # Create sampler exactly like validation script
    sampler = BongardSampler(sampler_config)
    print("✓ Successfully created BongardSampler")
    
    # Get rules and test generation
    rules = get_all_rules()
    print(f"Loaded {len(rules)} rules")
    
    if rules:
        rule = rules[0]  # Use first rule
        print(f"Using rule: {rule.description}")
        
        # Generate problem like validation script does
        problem = sampler.sample_problem(rule_description=rule.description, num_pos_scenes=1, num_neg_scenes=1)
        
        if problem:
            print("✓ Problem generated successfully")
            print(f"Positive scenes: {len(problem.get('positive_scenes', []))}")
            print(f"Negative scenes: {len(problem.get('negative_scenes', []))}")
            
            # Check first positive scene
            if problem.get('positive_scenes'):
                scene = problem['positive_scenes'][0]
                img = scene.get('image')
                if img is not None:
                    print(f"First positive image type: {type(img)}")
                    if hasattr(img, 'size'):
                        print(f"Image size: {img.size}")
                    
                    # Save for inspection
                    img.save("debug_validation_positive.png")
                    print("✓ Saved debug_validation_positive.png")
                else:
                    print("❌ No image in positive scene")
            
            # Check first negative scene
            if problem.get('negative_scenes'):
                scene = problem['negative_scenes'][0]
                img = scene.get('image')
                if img is not None:
                    print(f"First negative image type: {type(img)}")
                    if hasattr(img, 'size'):
                        print(f"Image size: {img.size}")
                    
                    # Save for inspection
                    img.save("debug_validation_negative.png")
                    print("✓ Saved debug_validation_negative.png")
                else:
                    print("❌ No image in negative scene")
                
        else:
            print("❌ Problem generation failed")
    else:
        print("❌ No rules loaded")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
