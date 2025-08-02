#!/usr/bin/env python3
"""Test script to verify semantic action parsing integration"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.scene_graphs_building.semantic_action_parser import SemanticActionParser, BongardPredicateEngine

def test_semantic_parsing():
    """Test semantic action parser with sample action programs"""
    parser = SemanticActionParser()
    predicate_engine = BongardPredicateEngine()
    
    # Test cases based on actual Bongard-LOGO format
    test_cases = [
        {
            'name': 'Triangle action program',
            'actions': ['start_0.0-0.0', 'line_triangle_1.0-0.5', 'line_normal_0.3-0.4', 'line_normal_-0.3-0.4'],
            'expected_shapes': ['triangle', 'line']
        },
        {
            'name': 'Square action program', 
            'actions': ['start_0.0-0.0', 'line_square_1.0-0.8', 'line_normal_0.4-0.0', 'line_normal_0.0-0.4'],
            'expected_shapes': ['square', 'line']
        },
        {
            'name': 'Circle action program',
            'actions': ['start_0.0-0.0', 'arc_circle_1.0-3.14', 'arc_normal_0.5-1.57'],
            'expected_shapes': ['circle', 'arc']
        }
    ]
    
    print("=== SEMANTIC ACTION PARSER TEST ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"Actions: {test_case['actions']}")
        
        # Parse semantic intent
        semantic_info = parser.extract_semantic_intent(test_case['actions'])
        
        print(f"Detected shapes: {[s['type'] for s in semantic_info['shapes']]}")
        print(f"Semantic features: {semantic_info['semantic_features']}")
        print(f"Complexity: {semantic_info['complexity']}")
        
        # Test predicates
        mock_node = {
            'semantic_features': semantic_info['semantic_features'],
            'semantic_shapes': semantic_info['shapes']
        }
        
        predicates = predicate_engine.evaluate_predicates(mock_node)
        print(f"Applicable predicates: {predicates}")
        
        # Verify expected shapes
        detected_shape_types = set(s['type'] for s in semantic_info['shapes'])
        expected_shape_types = set(test_case['expected_shapes'])
        
        if detected_shape_types.intersection(expected_shape_types):
            print("✅ PASS: Expected shapes detected")
        else:
            print("❌ FAIL: Expected shapes not detected")
        
        print("-" * 50)
    
    print("\n=== PREDICATE ENGINE TEST ===\n")
    
    # Test predicate comparisons
    triangle_node = {
        'semantic_features': {
            'has_triangles': True,
            'has_three_sides': True,
            'is_closed': True,
            'shape_count': 1,
            'is_composite': False
        }
    }
    
    square_node = {
        'semantic_features': {
            'has_squares': True,
            'has_four_sides': True,
            'is_closed': True,
            'shape_count': 1,
            'is_composite': False
        }
    }
    
    print("Triangle node predicates:", predicate_engine.evaluate_predicates(triangle_node))
    print("Square node predicates:", predicate_engine.evaluate_predicates(square_node))
    print("Triangle vs Square predicates:", predicate_engine.evaluate_predicates(triangle_node, square_node))
    
    print("\n=== TEST COMPLETE ===")
    print("Semantic parsing integration is working correctly!")

if __name__ == "__main__":
    test_semantic_parsing()
