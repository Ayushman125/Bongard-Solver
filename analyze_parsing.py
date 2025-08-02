import json
import os
import numpy as np
from pathlib import Path

def analyze_bongard_data_parsing():
    """Analyze how well the current implementation parses Bongard-LOGO data"""
    
    print("=== BONGARD DATA PARSING ANALYSIS ===\n")
    
    # 1. Analyze action program structure
    print("1. ACTION PROGRAM ANALYSIS")
    print("-" * 40)
    
    action_file = "data/raw/ShapeBongard_V2/bd/bd_action_programs.json"
    if os.path.exists(action_file):
        with open(action_file, 'r') as f:
            action_data = json.load(f)
        
        print(f"Total problems in action file: {len(action_data)}")
        
        # Sample a problem to understand structure
        sample_key = list(action_data.keys())[0]
        sample_data = action_data[sample_key]
        
        print(f"Sample problem: {sample_key}")
        print(f"Structure: {type(sample_data)}")
        
        if isinstance(sample_data, list) and len(sample_data) >= 2:
            pos_set = sample_data[0]  # Positive examples
            neg_set = sample_data[1]  # Negative examples
            
            print(f"Positive examples: {len(pos_set)}")
            print(f"Negative examples: {len(neg_set)}")
            
            # Analyze first positive example
            if pos_set:
                first_pos = pos_set[0]
                print(f"First positive example structure: {len(first_pos)} strokes")
                for i, stroke in enumerate(first_pos[:3]):  # Show first 3 strokes
                    print(f"  Stroke {i}: {stroke}")
    
    print("\n2. DERIVED LABELS ANALYSIS")
    print("-" * 40)
    
    # Analyze derived labels
    with open('data/derived_labels.json', 'r') as f:
        derived_data = json.load(f)
    
    # Count by category
    categories = {}
    labels = {}
    for item in derived_data:
        cat = item.get('category', 'unknown')
        lbl = item.get('label', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
        labels[lbl] = labels.get(lbl, 0) + 1
    
    print(f"Categories: {categories}")
    print(f"Labels: {labels}")
    
    # Sample geometry data
    sample_item = derived_data[0]
    geometry = sample_item.get('geometry', [])
    features = sample_item.get('features', {})
    
    print(f"Sample geometry points: {len(geometry)}")
    print(f"Sample features: {list(features.keys())}")
    print(f"Sample feature values: {[(k, v) for k, v in list(features.items())[:5]]}")
    
    print("\n3. SCENE GRAPH QUALITY COMPARISON")
    print("-" * 40)
    
    # Compare what we extract vs what we should extract
    print("Current extraction:")
    print("- Geometry: Raw vertex coordinates")
    print("- Features: Low-level geometric properties")
    print("- Relationships: Over-generated spatial predicates")
    
    print("\nWhat Bongard problems actually test:")
    print("- Shape recognition (triangle, square, circle)")
    print("- Count-based rules (3 vs 4 sides)")
    print("- Symmetry properties (symmetric vs asymmetric)")
    print("- Topological properties (closed vs open)")
    print("- Compositional rules (made of triangles vs squares)")
    
    print("\n4. PARSING QUALITY ASSESSMENT")
    print("-" * 40)
    
    # Check if we're extracting the right level of abstraction
    total_issues = 0
    
    # Issue 1: Only extracting primitives, not shapes
    if geometry and len(geometry) > 10:
        print("❌ Issue: Extracting too many low-level points")
        print(f"   {len(geometry)} geometry points suggests primitive-level parsing")
        print("   Should extract: ~1-5 high-level shapes per image")
        total_issues += 1
    
    # Issue 2: Missing semantic shape detection
    if 'num_straight' in features and 'num_arcs' in features:
        print("❌ Issue: Features focus on primitive counts")
        print("   Should focus on: Shape types, symmetry, topology")
        total_issues += 1
    
    # Issue 3: No action sequence analysis
    action_program = sample_item.get('action_program', [])
    if not action_program:
        print("❌ Issue: Missing action program analysis")
        print("   Action sequences contain the semantic intent")
        total_issues += 1
    
    print(f"\nTotal parsing issues identified: {total_issues}")
    
    print("\n5. RECOMMENDATIONS")
    print("-" * 40)
    
    print("Priority 1: Implement hierarchical shape detection")
    print("- Parse action sequences to detect intended shapes")
    print("- Map line/arc primitives to triangle/square/circle")
    print("- Extract shape-level properties (convex, symmetric, etc.)")
    
    print("\nPriority 2: Focus on Bongard-relevant features")
    print("- Count-based: number of sides, holes, components")
    print("- Symmetry: reflection, rotation axes")
    print("- Topology: connectivity, closure")
    print("- Composition: part-whole relationships")
    
    print("\nPriority 3: Validate semantic consistency")
    print("- Ensure extracted shapes match visual appearance")
    print("- Check relationships make geometric sense")
    print("- Verify action sequence alignment with features")

if __name__ == "__main__":
    analyze_bongard_data_parsing()
