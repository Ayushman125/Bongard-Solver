#!/usr/bin/env python3
"""
Test all available rules to identify which ones can successfully generate scenes.
"""
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from bongard_generator.rule_loader import get_all_rules
from bongard_generator.dataset import BongardDataset

def test_all_rules():
    """Test all available rules to see which ones work."""
    print("🧪 Testing All Available Rules")
    print("=" * 50)
    
    rules = get_all_rules()
    print(f"Found {len(rules)} total rules")
    
    working_rules = []
    failing_rules = []
    
    for rule in tqdm(rules, desc="Testing rules"):
        try:
            # Create small dataset to test rule
            dataset = BongardDataset(canvas_size=128, target_quota=2, rule_list=[rule.description])
            
            if len(dataset.examples) > 0:
                working_rules.append((rule.description, len(dataset.examples)))
                print(f"✓ {rule.description[:50]:<50} -> {len(dataset.examples)} examples")
            else:
                failing_rules.append(rule.description)
                print(f"❌ {rule.description[:50]:<50} -> 0 examples")
                
        except Exception as e:
            failing_rules.append(rule.description)
            print(f"💥 {rule.description[:50]:<50} -> ERROR: {e}")
    
    print(f"\n📊 Results Summary:")
    print(f"✅ Working rules: {len(working_rules)}")
    print(f"❌ Failing rules: {len(failing_rules)}")
    print(f"Success rate: {len(working_rules)/(len(working_rules)+len(failing_rules))*100:.1f}%")
    
    if working_rules:
        print(f"\n🎯 Top Working Rules:")
        # Sort by number of examples generated
        working_rules.sort(key=lambda x: x[1], reverse=True)
        for rule, count in working_rules[:10]:
            print(f"  ✓ {rule[:60]:<60} ({count} examples)")
    
    if failing_rules:
        print(f"\n❌ Failing Rules (to avoid in validation):")
        for rule in failing_rules[:10]:
            print(f"  ❌ {rule}")
    
    return working_rules, failing_rules

def create_robust_rule_list():
    """Create a curated list of rules that reliably work."""
    working_rules, failing_rules = test_all_rules()
    
    # Create a whitelist of working rules
    robust_rules = [rule for rule, count in working_rules if count >= 2]
    
    print(f"\n✅ Created robust rule list with {len(robust_rules)} reliable rules")
    
    # Save to file for future use
    with open("robust_rules.txt", "w") as f:
        for rule in robust_rules:
            f.write(f"{rule}\n")
    
    print("💾 Saved to robust_rules.txt")
    return robust_rules

if __name__ == "__main__":
    robust_rules = create_robust_rule_list()
    
    print(f"\n🚀 Ready for validation with {len(robust_rules)} working rules!")
    print("This should eliminate the black box issue.")
