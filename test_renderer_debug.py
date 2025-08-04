#!/usr/bin/env python3
"""
Debug script to diagnose rendering issues with parsed images
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add the specific paths to find the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'bongard_augmentor'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'data_pipeline'))

from hybrid import HybridAugmentationPipeline
from logo_parser import UnifiedActionParser

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_specific_rendering():
    """Test rendering for a specific problematic image"""
    
    # Test case from logs
    commands = ['line_normal_0.860-0.500', 'line_normal_0.300-0.151', 'line_normal_0.860-0.151', 'line_normal_0.700-0.849']
    image_id = "test_debug"
    
    print(f"\n=== TESTING RENDERING FOR {image_id} ===")
    print(f"Commands: {commands}")
    
    # Initialize components
    pipeline = HybridAugmentationPipeline()
    parser = UnifiedActionParser()
    
    # Parse commands to get vertices and object boundaries
    print("\n--- PARSING PHASE ---")
    parsed_program = parser._parse_single_image(commands, image_id, True, "test_problem")
    
    if not parsed_program or not parsed_program.vertices:
        print("ERROR: Failed to parse commands or no vertices generated")
        return
    
    print(f"Total vertices: {len(parsed_program.vertices)}")
    print(f"Object boundaries: {len(parsed_program.object_boundaries) if hasattr(parsed_program, 'object_boundaries') else 'None'}")
    
    # Debug object boundaries
    if hasattr(parsed_program, 'object_boundaries'):
        for i, obj in enumerate(parsed_program.object_boundaries):
            print(f"  Object {i+1}: {len(obj['vertices'])} vertices, modifiers: {obj.get('shape_modifiers', [])}")
    
    # Test rendering
    print("\n--- RENDERING PHASE ---")
    rendered_image = pipeline._render_parsed_image_from_vertices(
        parsed_program.vertices, image_id, parsed_program
    )
    
    if rendered_image is None:
        print("ERROR: Failed to render image")
        return
    
    print(f"Rendered image shape: {rendered_image.shape}")
    print(f"Pixel value range: {rendered_image.min()} - {rendered_image.max()}")
    print(f"Non-zero pixels: {np.count_nonzero(rendered_image)}")
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show rendered image
    axes[0].imshow(rendered_image, cmap='gray')
    axes[0].set_title(f"Rendered: {image_id}")
    axes[0].axis('off')
    
    # Show vertex plot
    if parsed_program.vertices:
        vertices_array = np.array(parsed_program.vertices)
        axes[1].scatter(vertices_array[:, 0], vertices_array[:, 1], alpha=0.7, s=20)
        axes[1].plot(vertices_array[:, 0], vertices_array[:, 1], alpha=0.5)
        axes[1].set_title(f"Vertices ({len(parsed_program.vertices)} points)")
        axes[1].grid(True)
        axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'debug_rendering_{image_id}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save individual images for inspection
    cv2.imwrite(f'debug_rendered_{image_id}.png', rendered_image)
    
    return rendered_image

def test_manual_rendering():
    """Test manual rendering with simple coordinates"""
    
    print("\n=== TESTING MANUAL RENDERING ===")
    
    # Simple test vertices
    test_vertices = [
        (0.0, 0.0), (10.0, 10.0), (20.0, 0.0), (30.0, 15.0)
    ]
    
    # Create mock object boundaries
    mock_boundaries = [
        {
            'vertices': test_vertices[:2],
            'strokes': ['line_normal_0.5-0.0'],
            'shape_modifiers': ['normal'],
            'stroke_types': ['line']
        },
        {
            'vertices': test_vertices[2:],
            'strokes': ['line_normal_0.5-0.25'],
            'shape_modifiers': ['normal'],
            'stroke_types': ['line']
        }
    ]
    
    # Initialize pipeline
    pipeline = HybridAugmentationPipeline()
    
    # Create high-resolution canvas
    scale_factor = 8
    base_size = (64, 64)
    high_res_size = (base_size[0] * scale_factor, base_size[1] * scale_factor)
    canvas = np.zeros((*high_res_size, 3), dtype=np.uint8)
    
    center_x = high_res_size[1] // 2
    center_y = high_res_size[0] // 2
    
    print(f"Canvas size: {canvas.shape}")
    print(f"Center: ({center_x}, {center_y})")
    print(f"Scale factor: {scale_factor}")
    
    # Test rendering
    pipeline._render_multiple_objects(canvas, mock_boundaries, scale_factor, center_x, center_y)
    
    # Convert to grayscale and downsample
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    final_image = cv2.resize(canvas_gray, base_size, interpolation=cv2.INTER_AREA)
    
    print(f"Final image shape: {final_image.shape}")
    print(f"Pixel range: {final_image.min()} - {final_image.max()}")
    print(f"Non-zero pixels: {np.count_nonzero(final_image)}")
    
    # Display result
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(canvas_gray, cmap='gray')
    plt.title('High-res canvas')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(final_image, cmap='gray')
    plt.title('Final downsampled')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_manual_rendering.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return final_image

if __name__ == "__main__":
    print("=== RENDERER DEBUG SCRIPT ===")
    
    # Test 1: Specific problematic case
    rendered1 = test_specific_rendering()
    
    # Test 2: Manual rendering test
    rendered2 = test_manual_rendering()
    
    print("\n=== SUMMARY ===")
    if rendered1 is not None:
        print(f"Specific case: {rendered1.shape}, {np.count_nonzero(rendered1)} non-zero pixels")
    if rendered2 is not None:
        print(f"Manual test: {rendered2.shape}, {np.count_nonzero(rendered2)} non-zero pixels")
