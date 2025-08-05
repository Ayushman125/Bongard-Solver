"""
Test the NVLabs-compatible renderer to ensure it matches the official coordinate system and image generation.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.bongard_augmentor.nvlabs_compatible_renderer import NVLabsCompatibleRenderer, ActionMaskGeneratorNVLabsCompatible
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_nvlabs_renderer():
    """Test the NVLabs-compatible renderer with sample coordinates."""
    
    # Test 1: Simple geometric shape in NVLabs coordinate system
    print("=== Test 1: Simple Rectangle in NVLabs Coordinates ===")
    
    # Rectangle in NVLabs coordinate system (-360 to 360)
    rectangle_vertices = [
        (-100, -100),  # Bottom left
        (100, -100),   # Bottom right  
        (100, 100),    # Top right
        (-100, 100),   # Top left
        (-100, -100)   # Close the shape
    ]
    
    renderer = NVLabsCompatibleRenderer()
    mask = renderer.render_vertices_to_image(rectangle_vertices)
    
    coverage = np.mean(mask > 0) * 100
    print(f"Rectangle mask generated: {mask.shape}, coverage: {coverage:.2f}%")
    
    # Save test image
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'NVLabs Rectangle Test (Coverage: {coverage:.1f}%)')
    plt.axis('off')
    plt.savefig('test_nvlabs_rectangle.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 2: Real Bongard-LOGO action commands
    print("\\n=== Test 2: Real Bongard-LOGO Action Commands ===")
    
    # Example action commands from real Bongard-LOGO dataset
    sample_commands = [
        "line_normal_0.640_0.556-0.389",
        "arc_normal_0.500_0.306-0.694", 
        "line_normal_0.640_0.556-0.611",
        "arc_normal_0.500_0.194-0.306"
    ]
    
    try:
        # Parse commands to get vertices
        parser = ComprehensiveNVLabsParser()
        vertices = parser.parse_to_vertices(sample_commands)
        
        print(f"Parsed {len(sample_commands)} commands -> {len(vertices)} vertices")
        print(f"Vertices: {vertices[:3]}..." if len(vertices) > 3 else f"Vertices: {vertices}")
        
        # Generate mask using NVLabs renderer
        mask_generator = ActionMaskGeneratorNVLabsCompatible()
        mask = mask_generator.generate_mask_from_vertices(vertices)
        
        coverage = np.mean(mask > 0) * 100
        print(f"Action commands mask generated: {mask.shape}, coverage: {coverage:.2f}%")
        
        # Save test image
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='gray')
        plt.title(f'NVLabs Action Commands Test (Coverage: {coverage:.1f}%)')
        plt.axis('off')
        plt.savefig('test_nvlabs_action_commands.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error testing action commands: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Compare coordinate mapping
    print("\\n=== Test 3: Coordinate System Verification ===")
    
    # Test extreme coordinates
    extreme_vertices = [
        (-360, -360),  # Bottom left corner
        (360, -360),   # Bottom right corner
        (360, 360),    # Top right corner
        (-360, 360),   # Top left corner
        (-360, -360)   # Close
    ]
    
    mask = renderer.render_vertices_to_image(extreme_vertices)
    coverage = np.mean(mask > 0) * 100
    print(f"Full canvas border: {mask.shape}, coverage: {coverage:.2f}%")
    
    # Save test image
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'NVLabs Full Canvas Border (Coverage: {coverage:.1f}%)')
    plt.axis('off')
    plt.savefig('test_nvlabs_full_canvas.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\\n=== Tests Complete ===")
    print("Generated test images:")
    print("- test_nvlabs_rectangle.png")
    print("- test_nvlabs_action_commands.png") 
    print("- test_nvlabs_full_canvas.png")

if __name__ == "__main__":
    test_nvlabs_renderer()
