#!/usr/bin/env python3
"""
Diagnostic script to analyze parsing issues in generated images
"""

import sys
import os
import logging
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data_pipeline.logo_parser import UnifiedActionParser

def diagnose_parsing_issues():
    """Diagnose what's wrong with the parsed images"""
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Find an action program file
    action_files = [
        "data/bd_action_programs.json",
        "data/raw/bd_action_programs.json", 
        "data/action_programs.json"
    ]
    
    action_file = None
    for f in action_files:
        if os.path.exists(f):
            action_file = f
            break
    
    if not action_file:
        print("No action program file found. Looking for any JSON files...")
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith('.json') and 'action' in file.lower():
                    action_file = os.path.join(root, file)
                    print(f"Found: {action_file}")
                    break
            if action_file:
                break
    
    if not action_file:
        print("No action program files found!")
        return
    
    print(f"Using action file: {action_file}")
    
    # Initialize parser
    parser = UnifiedActionParser()
    
    # Parse the file
    try:
        parsed_data = parser.parse_action_file(action_file)
        print(f"Successfully parsed {len(parsed_data)} problems")
    except Exception as e:
        print(f"Error parsing file: {e}")
        return
    
    # Get the first problem for detailed analysis
    if not parsed_data:
        print("No parsed data available")
        return
    
    problem_id = list(parsed_data.keys())[0]
    examples = parsed_data[problem_id]
    
    print(f"\nAnalyzing problem {problem_id} with {len(examples)} examples")
    
    # Analyze first few examples
    for i, example in enumerate(examples[:3]):
        print(f"\n=== Example {i} ({example.image_id}) ===")
        print(f"Is positive: {example.is_positive}")
        print(f"Number of strokes: {len(example.strokes)}")
        print(f"Number of vertices: {len(example.vertices)}")
        
        # Show the stroke commands
        print("Stroke commands:")
        for j, stroke in enumerate(example.strokes):
            print(f"  {j}: {stroke.raw_command}")
            print(f"      Type: {stroke.stroke_type.value}, Shape: {stroke.shape_modifier.value}")
            print(f"      Parameters: {stroke.parameters}")
        
        # Show vertices
        print("Generated vertices:")
        for j, vertex in enumerate(example.vertices[:10]):  # Show first 10
            print(f"  {j}: ({vertex[0]:.3f}, {vertex[1]:.3f})")
        if len(example.vertices) > 10:
            print(f"  ... and {len(example.vertices) - 10} more vertices")
        
        # Visualize this example
        visualize_example(example, f"diagnosis_example_{i}.png")
        
        print(f"Visualization saved as diagnosis_example_{i}.png")

def visualize_example(example, filename):
    """Create a detailed visualization of an example"""
    
    if not example.vertices or len(example.vertices) < 2:
        print(f"Not enough vertices to visualize {example.image_id}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Raw vertices as points
    ax1 = axes[0]
    xs = [v[0] for v in example.vertices]
    ys = [v[1] for v in example.vertices]
    ax1.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=20)
    ax1.plot(xs, ys, 'r-', alpha=0.5, linewidth=1)
    ax1.set_title('Raw Vertices (Turtle Path)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Add vertex numbers for first few points
    for i, (x, y) in enumerate(example.vertices[:10]):
        ax1.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Rendered as image (like the pipeline does)
    ax2 = axes[1]
    rendered_img = render_vertices_to_image(example.vertices, (64, 64))
    ax2.imshow(rendered_img, cmap='gray')
    ax2.set_title('Rendered Image (64x64)')
    ax2.axis('off')
    
    # Plot 3: Stroke-by-stroke breakdown
    ax3 = axes[2]
    
    # Parse each stroke and show its contribution
    parser = UnifiedActionParser()
    parser._reset_turtle()
    
    all_x, all_y = [], []
    colors = plt.cm.tab10(np.linspace(0, 1, len(example.strokes)))
    
    for i, stroke in enumerate(example.strokes):
        # Execute this stroke
        start_pos = (parser.turtle_x, parser.turtle_y)
        stroke_vertices = parser._execute_stroke(stroke)
        
        # Plot this stroke's path
        stroke_x = [start_pos[0]] + [v[0] for v in stroke_vertices]
        stroke_y = [start_pos[1]] + [v[1] for v in stroke_vertices]
        
        ax3.plot(stroke_x, stroke_y, color=colors[i], linewidth=2, 
                label=f'Stroke {i}: {stroke.stroke_type.value}_{stroke.shape_modifier.value}')
        
        all_x.extend(stroke_x)
        all_y.extend(stroke_y)
    
    ax3.set_title('Stroke-by-Stroke Execution')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def render_vertices_to_image(vertices, image_size=(64, 64)):
    """Render vertices to an image like the pipeline does"""
    
    if len(vertices) < 2:
        return np.zeros(image_size, dtype=np.uint8)
    
    # Create image
    img = np.zeros(image_size, dtype=np.uint8)
    
    # Find bounds of vertices
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    
    if not xs or not ys:
        return img
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Scale vertices to image coordinates
    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1
    
    points = []
    for vertex in vertices:
        # Normalize to [0, 1] range
        x_norm = (vertex[0] - min_x) / (max_x - min_x)
        y_norm = (vertex[1] - min_y) / (max_y - min_y)
        
        # Scale to image size with some padding
        padding = 0.1
        x = int((x_norm * (1 - 2*padding) + padding) * image_size[1])
        y = int((y_norm * (1 - 2*padding) + padding) * image_size[0])
        
        # Clamp to bounds
        x = max(0, min(image_size[1] - 1, x))
        y = max(0, min(image_size[0] - 1, y))
        points.append([x, y])
    
    # Draw lines connecting the points
    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i+1]), 255, 1)
    
    return img

if __name__ == "__main__":
    diagnose_parsing_issues()
