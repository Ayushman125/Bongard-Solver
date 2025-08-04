#!/usr/bin/env python3
"""
Compare parsed images before and after the turtle graphics corrections
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def compare_existing_images():
    """Look at existing parsed images to assess quality"""
    
    # Find all parsed image files
    data_dir = "data"
    parsed_files = [f for f in os.listdir(data_dir) if f.endswith("_parsed.png")]
    
    print(f"Found {len(parsed_files)} parsed image files")
    
    if len(parsed_files) >= 4:
        # Compare a few different examples
        sample_files = parsed_files[:4]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Parsed Images vs Masks Quality Comparison", fontsize=16)
        
        for i, parsed_file in enumerate(sample_files):
            # Load parsed image
            parsed_path = os.path.join(data_dir, parsed_file)
            parsed_img = cv2.imread(parsed_path)
            
            # Load corresponding mask
            mask_file = parsed_file.replace("_parsed.png", "_mask.png")
            mask_path = os.path.join(data_dir, mask_file)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if parsed_img is not None and mask_img is not None:
                # Convert parsed image to grayscale for comparison
                parsed_gray = cv2.cvtColor(parsed_img, cv2.COLOR_BGR2GRAY)
                
                # Display parsed image
                axes[0, i].imshow(parsed_gray, cmap='gray')
                axes[0, i].set_title(f"Parsed: {parsed_file[:20]}...")
                axes[0, i].axis('off')
                
                # Display mask
                axes[1, i].imshow(mask_img, cmap='gray')
                axes[1, i].set_title(f"Mask: {mask_file[:20]}...")
                axes[1, i].axis('off')
                
                # Print statistics
                print(f"\\nFile: {parsed_file}")
                print(f"  Parsed image non-zero pixels: {np.count_nonzero(parsed_gray)}")
                print(f"  Mask non-zero pixels: {np.count_nonzero(mask_img)}")
                print(f"  Parsed image max value: {np.max(parsed_gray)}")
                print(f"  Mask max value: {np.max(mask_img)}")
        
        plt.tight_layout()
        plt.savefig("quality_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\\nComparison saved as quality_comparison.png")
        
    else:
        print("Not enough parsed images found for comparison")

def analyze_turtle_graphics_accuracy():
    """Create a simple test to verify turtle graphics accuracy"""
    
    from data_pipeline.logo_parser import UnifiedActionParser
    
    # Create a simple test command set
    test_commands = [
        ["line_triangle_1.000-0.000"],  # Move right
        ["line_triangle_1.000-1.571"],  # Move down  
        ["line_triangle_1.000-3.142"],  # Move left
        ["line_triangle_1.000-4.712"]   # Move up (should complete square)
    ]
    
    parser = UnifiedActionParser()
    
    print("\\nTesting turtle graphics accuracy:")
    for i, commands in enumerate(test_commands):
        print(f"\\nTest {i+1}: {commands}")
        
        # Reset parser state
        parser.turtle_x = 0.0
        parser.turtle_y = 0.0
        parser.turtle_heading = 0.0
        vertices = []
        
        # Process commands
        for cmd in commands:
            stroke_cmd = parser._parse_stroke_command(cmd)
            if stroke_cmd and stroke_cmd.command_type == "line":
                result = parser._execute_line_stroke(stroke_cmd.parameters, vertices)
                if result:
                    vertices.extend(result)
        
        print(f"  Generated vertices: {len(vertices)}")
        for j, vertex in enumerate(vertices[:3]):  # Show first 3
            print(f"    Vertex {j}: ({vertex[0]:.3f}, {vertex[1]:.3f})")

if __name__ == "__main__":
    print("=== Existing Parsed Images Quality Assessment ===")
    compare_existing_images()
    
    print("\\n=== Turtle Graphics Accuracy Test ===")
    analyze_turtle_graphics_accuracy()
