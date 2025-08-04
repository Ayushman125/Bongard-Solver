#!/usr/bin/env python3
"""Test arc parsing functionality"""

from src.data_pipeline.logo_parser import UnifiedActionParser

# Test arc parsing
parser = UnifiedActionParser()

# Test an arc command
test_command = 'arc_normal_0.500_0.542-0.750'
result = parser._parse_stroke_command(test_command)

print(f"Arc command: {test_command}")
print(f"Parsed result: {result}")
print(f"Type: {result['type']}")
print(f"Parameters: {result['params']}")

# Test that it generates vertices
if result['type'] == 'arc':
    vertices = parser._generate_arc_vertices(**result['params'])
    print(f"Generated {len(vertices)} vertices")
    print(f"Sample vertices: {vertices[:3]}")
    print("✅ Arc parsing is working correctly!")
else:
    print("❌ Arc parsing failed")
