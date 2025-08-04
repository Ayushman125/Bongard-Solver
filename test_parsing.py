#!/usr/bin/env python3
"""
Test script to verify the corrected turtle graphics parsing logic
"""

import sys
import os
import logging
import json
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_pipeline.logo_parser import UnifiedActionParser

def test_parsing_improvements():
    """Test the corrected parsing logic with a few examples"""
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize parser
    parser = UnifiedActionParser()
    
    # Load action programs
    action_file = "data/bd_action_programs.json"
    if not os.path.exists(action_file):
        print(f"Action file not found: {action_file}")
        return
    
    # Parse the file
    parsed_data = parser.parse_action_file(action_file)
    
    print(f"Successfully parsed {len(parsed_data)} problems")
    
    # Test with problem 101
    if '101' in parsed_data:
        problem_101 = parsed_data['101']
        print(f"Problem 101 has {len(problem_101)} examples")
        
        # Test first positive example
        if problem_101:
            first_example = problem_101[0]
            print(f"First example - Problem ID: {first_example.problem_id}")
            print(f"First example - Image ID: {first_example.image_id}")
            print(f"First example - Is Positive: {first_example.is_positive}")
            print(f"First example - Strokes: {len(first_example.strokes)} strokes")
            print(f"First example - Vertices: {len(first_example.vertices)} vertices")
            
            # Show first few strokes to verify parsing
            print("\nFirst few strokes:")
            for i, stroke in enumerate(first_example.strokes[:3]):
                print(f"  Stroke {i}: {stroke}")
            
            # Show first few vertices to verify turtle graphics
            print(f"\nFirst few vertices:")
            for i, vertex in enumerate(first_example.vertices[:5]):
                print(f"  Vertex {i}: ({vertex[0]:.3f}, {vertex[1]:.3f})")
            
            # Create a simple visualization
            if first_example.vertices:
                create_simple_visualization(first_example.vertices, f"test_problem_101_example_0.png")
                print(f"\nVisualization saved as test_problem_101_example_0.png")
    
    else:
        print("Problem 101 not found in parsed data")
        print(f"Available problems: {list(parsed_data.keys())[:10]}...")

def create_simple_visualization(vertices, filename):
    """Create a simple visualization of the vertices"""
    if len(vertices) < 2:
        return
    
    # Create image
    img_size = 200
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Convert vertices to image coordinates
    points = []
    for vertex in vertices:
        if isinstance(vertex, (list, tuple)) and len(vertex) >= 2:
            # Convert from parser coordinates to image coordinates
            x = int((vertex[0] / 64.0 + 1) * img_size / 2)  # Assuming scale_factor was 64
            y = int((vertex[1] / 64.0 + 1) * img_size / 2)
            
            # Clamp to bounds
            x = max(0, min(img_size - 1, x))
            y = max(0, min(img_size - 1, y))
            points.append([x, y])
    
    # Draw lines connecting the vertices
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i+1]), (255, 255, 255), 2)
        
        # Close the shape if it looks like a polygon
        if len(points) > 3:
            cv2.line(img, tuple(points[-1]), tuple(points[0]), (255, 255, 255), 2)
    
    # Save the image
    cv2.imwrite(filename, img)

if __name__ == "__main__":
    test_parsing_improvements()
