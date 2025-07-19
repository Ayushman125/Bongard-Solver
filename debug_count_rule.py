#!/usr/bin/env python3
"""
Test the specific count_eq_4 rule more thoroughly to understand the failure.
"""
import os
import sys

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from bongard_generator.rule_loader import get_all_rules
from bongard_generator.dataset import BongardDataset

def test_count_eq_4_specifically():
    """Test the count_eq_4 rule in detail."""
    print("🔍 Debugging count_eq_4 Rule Specifically")
    print("=" * 60)
    
    # Find the count_eq_4 rule
    rules = get_all_rules()
    count_eq_4_rule = None
    for rule in rules:
        if "exactly 4 objects" in rule.description:
            count_eq_4_rule = rule
            break
    
    if not count_eq_4_rule:
        print("❌ Could not find count_eq_4 rule!")
        return False
    
    print(f"✓ Found rule: {count_eq_4_rule.description}")
    print(f"  Rule object: {count_eq_4_rule}")
    print(f"  Rule type: {type(count_eq_4_rule)}")
    
    # Test with different target quotas
    for target_quota in [1, 2, 5, 10]:
        print(f"\n🧪 Testing with target_quota={target_quota}")
        try:
            dataset = BongardDataset(
                canvas_size=128, 
                target_quota=target_quota, 
                rule_list=[count_eq_4_rule.description],
                max_obj_size=30,  # Smaller objects to fit 4 in scene
                min_obj_size=15
            )
            
            print(f"  Examples generated: {len(dataset.examples)}")
            if len(dataset.examples) > 0:
                example = dataset.examples[0]
                scene_graph = example.get('scene_graph', {})
                objects = scene_graph.get('objects', [])
                print(f"  First example has {len(objects)} objects")
                if len(objects) == 4:
                    print("  ✅ SUCCESS: Correctly generated 4 objects!")
                else:
                    print(f"  ⚠ WARNING: Expected 4 objects, got {len(objects)}")
                    
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

def test_different_count_rules():
    """Test all count-based rules to see which ones work."""
    print("\n🔢 Testing All Count-Based Rules")
    print("=" * 50)
    
    rules = get_all_rules()
    count_rules = []
    
    for rule in rules:
        if "exactly" in rule.description and "objects" in rule.description:
            count_rules.append(rule)
    
    print(f"Found {len(count_rules)} count-based rules:")
    
    for rule in count_rules:
        print(f"\n🧪 Testing: {rule.description}")
        try:
            dataset = BongardDataset(
                canvas_size=128, 
                target_quota=3, 
                rule_list=[rule.description],
                max_obj_size=25,  # Smaller to fit more objects
                min_obj_size=15
            )
            
            if len(dataset.examples) > 0:
                example = dataset.examples[0]
                scene_graph = example.get('scene_graph', {})
                objects = scene_graph.get('objects', [])
                print(f"  ✅ Generated {len(dataset.examples)} examples, first has {len(objects)} objects")
            else:
                print("  ❌ No examples generated")
                
        except Exception as e:
            print(f"  💥 ERROR: {e}")

if __name__ == "__main__":
    test_count_eq_4_specifically()
    test_different_count_rules()
