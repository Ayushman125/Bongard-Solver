#!/usr/bin/env python3
"""Test quarter circle detection functionality"""

from src.scene_graphs_building.process_single_problem import _process_single_problem
import os
import json
import asyncio

async def test_quarter_circle_detection():
    """Test quarter circle detection on a specific problem"""
    
    # Test with a specific problem that should have quarter circles  
    test_dir = 'data/raw/ShapeBongard_V2/bd/images/bd_open_onequarter_circle_0000'
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
        
    # Create test problem records
    problem_records = [{
        'category': 'bd', 
        'problem_id': 'bd_open_onequarter_circle_0000',
        'image_path': os.path.join(test_dir, '0', '0.png'),
        'label': 'positive',
        'action_program': [
            'start_0.0_0.0',
            'line_normal_0.354-0.500',
            'line_normal_0.900-0.125', 
            'line_normal_0.354-0.125',
            'line_normal_0.354-0.500',
            'line_normal_0.900-0.125',
            'line_normal_0.354-0.125'
        ]
    }]
    
    print("Processing problem for quarter circle detection...")
    result = await _process_single_problem('bd_open_onequarter_circle_0000', problem_records, {})
    
    nodes = result.get('objects', [])
    print(f"Successfully processed: {len(nodes)} nodes")
    
    # Check for quarter circles
    quarter_circles = []
    for node in nodes:
        composite_type = node.get('composite_type', '')
        if composite_type and 'quarter_circle' in str(composite_type):
            quarter_circles.append(node)
    
    if quarter_circles:
        print(f"\n✅ Found {len(quarter_circles)} quarter circles!")
        for i, qc in enumerate(quarter_circles):
            print(f"  {i+1}. ID: {qc.get('object_id')}")
            print(f"     Confidence: {qc.get('detection_confidence')}")
            print(f"     Arc angle: {qc.get('arc_angle')}")
            print(f"     Radius: {qc.get('radius')}")
    else:
        print("\n❌ No quarter circles detected")
        
        # Check if we have connected lines that might form a quarter circle
        lines = [n for n in nodes if n.get('object_type') == 'line']
        print(f"Found {len(lines)} line segments")
        
        if lines:
            print("Line segments:")
            for line in lines:
                endpoints = line.get('endpoints', [])
                print(f"  - {line.get('object_id')}: {endpoints}")
        
        # Check for lines that might be connected
        connected_groups = []
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                endpoints1 = line1.get('endpoints', [])
                endpoints2 = line2.get('endpoints', [])
                
                if len(endpoints1) == 2 and len(endpoints2) == 2:
                    # Check if lines share an endpoint
                    for ep1 in endpoints1:
                        for ep2 in endpoints2:
                            if abs(ep1[0] - ep2[0]) < 0.01 and abs(ep1[1] - ep2[1]) < 0.01:
                                connected_groups.append((line1.get('object_id'), line2.get('object_id')))
                                break
        
        if connected_groups:
            print(f"Found {len(connected_groups)} connected line pairs:")
            for pair in connected_groups[:5]:  # Show first 5
                print(f"  - {pair[0]} <-> {pair[1]}")
        
    return quarter_circles

if __name__ == "__main__":
    asyncio.run(test_quarter_circle_detection())
