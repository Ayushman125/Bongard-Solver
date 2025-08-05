"""
Test the headless NVLabs-compatible renderer to ensure proper coordinate mapping.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.bongard_augmentor.headless_nvlabs_renderer import HeadlessNVLabsRenderer, HeadlessActionMaskGenerator
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_headless_nvlabs_renderer():
    """Test the headless NVLabs-compatible renderer."""
    
    print("=== Testing Headless NVLabs-Compatible Renderer ===")
    
    # Test 1: Simple geometric shape
    print("\\n1. Testing simple rectangle in NVLabs coordinates...")
    
    rectangle_vertices = [
        (-100, -100),  # Bottom left
        (100, -100),   # Bottom right  
        (100, 100),    # Top right
        (-100, 100),   # Top left
        (-100, -100)   # Close the shape
    ]
    
    renderer = HeadlessNVLabsRenderer()
    mask = renderer.render_vertices_to_image(rectangle_vertices)
    
    coverage = np.mean(mask > 0) * 100
    print(f"Rectangle: shape={mask.shape}, coverage={coverage:.2f}%")
    
    # Save test image
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Headless NVLabs Rectangle (Coverage: {coverage:.1f}%)')
    plt.axis('off')
    plt.savefig('test_headless_rectangle.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 2: Circle-like shape
    print("\\n2. Testing circle-like shape...")
    
    # Create circle vertices
    circle_vertices = []
    num_points = 16
    radius = 120
    for i in range(num_points + 1):  # +1 to close the circle
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        circle_vertices.append((x, y))
    
    mask = renderer.render_vertices_to_image(circle_vertices)
    coverage = np.mean(mask > 0) * 100
    print(f"Circle: shape={mask.shape}, coverage={coverage:.2f}%")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Headless NVLabs Circle (Coverage: {coverage:.1f}%)')
    plt.axis('off')
    plt.savefig('test_headless_circle.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 3: Real Bongard-LOGO action commands
    print("\\n3. Testing real action commands...")
    
    sample_commands = [
        "line_normal_0.640_0.556-0.389",
        "arc_normal_0.500_0.306-0.694", 
        "line_normal_0.640_0.556-0.611",
        "arc_normal_0.500_0.194-0.306"
    ]
    
    try:
        parser = ComprehensiveNVLabsParser()
        vertices = parser.parse_to_vertices(sample_commands)
        
        print(f"Parsed {len(sample_commands)} commands -> {len(vertices)} vertices")
        if vertices:
            print(f"Vertex range: x=({min(v[0] for v in vertices):.1f}, {max(v[0] for v in vertices):.1f}), "
                  f"y=({min(v[1] for v in vertices):.1f}, {max(v[1] for v in vertices):.1f})")
        
        mask_generator = HeadlessActionMaskGenerator()
        mask = mask_generator.generate_mask_from_vertices(vertices)
        
        coverage = np.mean(mask > 0) * 100
        print(f"Action commands: shape={mask.shape}, coverage={coverage:.2f}%")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Headless Action Commands (Coverage: {coverage:.1f}%)')
        plt.axis('off')
        plt.savefig('test_headless_actions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error testing action commands: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Coordinate system verification
    print("\\n4. Testing coordinate system boundaries...")
    
    # Full canvas border
    border_vertices = [
        (-360, -360),  # Bottom left corner
        (360, -360),   # Bottom right corner
        (360, 360),    # Top right corner
        (-360, 360),   # Top left corner
        (-360, -360)   # Close
    ]
    
    mask = renderer.render_vertices_to_image(border_vertices)
    coverage = np.mean(mask > 0) * 100
    print(f"Full border: shape={mask.shape}, coverage={coverage:.2f}%")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Headless Full Canvas Border (Coverage: {coverage:.1f}%)')
    plt.axis('off')
    plt.savefig('test_headless_border.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 5: Small shape (center)
    print("\\n5. Testing small centered shape...")
    
    small_vertices = [
        (-20, -20),
        (20, -20),
        (20, 20),
        (-20, 20),
        (-20, -20)
    ]
    
    mask = renderer.render_vertices_to_image(small_vertices)
    coverage = np.mean(mask > 0) * 100
    print(f"Small center: shape={mask.shape}, coverage={coverage:.2f}%")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f'Headless Small Center Shape (Coverage: {coverage:.1f}%)')
    plt.axis('off')
    plt.savefig('test_headless_small.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\\n=== Test Complete ===")
    print("Generated test images:")
    print("- test_headless_rectangle.png")
    print("- test_headless_circle.png")
    print("- test_headless_actions.png")
    print("- test_headless_border.png")
    print("- test_headless_small.png")
    print("\\nAll tests use the headless NVLabs-compatible coordinate system.")

if __name__ == "__main__":
    test_headless_nvlabs_renderer()
