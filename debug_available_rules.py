#!/usr/bin/env python3
"""Debug what rules are actually available for the verification script."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bongard_generator.rule_loader import get_all_rules, get_rule_lookup

def show_available_rules():
    """Show what rules are actually available."""
    print("🔍 Available Rules:")
    print("=" * 50)
    
    rules = get_all_rules()
    lookup = get_rule_lookup()
    
    print(f"Total rules loaded: {len(rules)}")
    print(f"Rule lookup keys: {len(lookup)}")
    
    print("\n📋 All available rule descriptions:")
    for i, rule in enumerate(rules):
        print(f"{i+1:2d}. {rule.description} -> features: {rule.positive_features}")
    
    print(f"\n🔑 Rule lookup keys:")
    for key in sorted(lookup.keys()):
        print(f"  '{key}'")

if __name__ == "__main__":
    show_available_rules()
