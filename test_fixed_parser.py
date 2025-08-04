#!/usr/bin/env python3
"""
Test script to validate the corrected Bongard-LOGO parser and hybrid augmentation.
Tests the key fixes: coordinate system, arc parsing, and rendering quality.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'bongard_augmentor'))

from logo_parser import UnifiedActionParser
from hybrid import HybridAugmentationPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_corrected_parsing():
    """Test the corrected parameter parsing and coordinate system"""
    
    print("=== Testing Corrected Bongard-LOGO Parser ===\n")
    
    parser = UnifiedActionParser()
    
    # Test cases covering the key issues
    test_cases = [
        # Basic line commands
        {
            'name': 'Simple Line',
            'commands': ['line_normal_0.500-0.500'],
            'expected': 'Horizontal line from center'
        },
        # Arc commands with corrected parsing
        {
            'name': 'Simple Arc',
            'commands': ['arc_normal_0.500_0.250-0.000'],
            'expected': 'Quarter circle arc'
        },
        # Complex shape modifiers
        {
            'name': 'Zigzag Line',
            'commands': ['line_zigzag_0.500-0.250'],
            'expected': 'Zigzag pattern line'
        },
        # Multi-stroke image
        {
            'name': 'Multi-stroke Shape',
            'commands': [
                'line_normal_0.300-0.000',
                'arc_normal_0.200_0.500-0.000',
                'line_normal_0.300-0.500'
            ],
            'expected': 'Connected line-arc-line shape'
        },
        # Complex modifiers
        {
            'name': 'Triangle Arc',
            'commands': ['arc_triangle_0.400_0.750-0.250'],
            'expected': 'Arc with triangular protrusions'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        print(f"Commands: {test_case['commands']}")
        print(f"Expected: {test_case['expected']}")
        
        try:
            # Parse the commands
            parsed_program = parser._parse_single_image(
                test_case['commands'], 
                f"test_{i+1}", 
                True, 
                "test_problem"
            )
            
            if parsed_program and parsed_program.vertices:
                vertex_count = len(parsed_program.vertices)
                
                # Analyze vertex distribution
                vertices = parsed_program.vertices
                x_coords = [v[0] for v in vertices]
                y_coords = [v[1] for v in vertices]
                
                x_range = (min(x_coords), max(x_coords)) if x_coords else (0, 0)
                y_range = (min(y_coords), max(y_coords)) if y_coords else (0, 0)
                
                print(f"✓ SUCCESS: Generated {vertex_count} vertices")
                print(f"  X range: [{x_range[0]:.2f}, {x_range[1]:.2f}]")
                print(f"  Y range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
                
                # Check if coordinates are reasonable (within expected bounds)
                reasonable_bounds = all(-50 <= x <= 50 for x in x_coords) and all(-50 <= y <= 50 for y in y_coords)
                print(f"  Coordinates within reasonable bounds: {'✓' if reasonable_bounds else '✗'}")
                
                results.append({
                    'test': test_case['name'],
                    'success': True,
                    'vertices': vertex_count,
                    'x_range': x_range,
                    'y_range': y_range,
                    'reasonable_bounds': reasonable_bounds
                })
            else:
                print("✗ FAILED: No vertices generated")
                results.append({
                    'test': test_case['name'],
                    'success': False,
                    'error': 'No vertices generated'
                })
        
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': str(e)
            })
        
        print()
    
    return results

def test_corrected_rendering():
    """Test the corrected rendering pipeline with anti-aliasing"""
    
    print("=== Testing Corrected Rendering Pipeline ===\n")
    
    pipeline = HybridAugmentationPipeline()
    
    # Test cases for rendering
    test_commands = [
        ['line_normal_0.500-0.000'],  # Horizontal line
        ['arc_normal_0.500_0.500-0.000'],  # Semicircle
        ['line_normal_0.300-0.000', 'line_normal_0.300-0.500'],  # L-shape
        ['arc_circle_0.300_0.750-0.000'],  # Beaded arc
    ]
    
    rendered_images = []
    
    for i, commands in enumerate(test_commands):
        print(f"Rendering test {i+1}: {commands}")
        
        try:
            # Generate mask using the hybrid pipeline
            mask = pipeline.process_action_commands(commands, f"render_test_{i+1}")
            
            if mask is not None:
                non_zero_pixels = np.count_nonzero(mask)
                fill_ratio = non_zero_pixels / mask.size
                
                print(f"✓ SUCCESS: Generated {mask.shape} mask")
                print(f"  Non-zero pixels: {non_zero_pixels} ({fill_ratio:.1%})")
                
                # Check for reasonable fill ratio
                reasonable_fill = 0.01 <= fill_ratio <= 0.5
                print(f"  Reasonable fill ratio: {'✓' if reasonable_fill else '✗'}")
                
                rendered_images.append({
                    'commands': commands,
                    'mask': mask,
                    'fill_ratio': fill_ratio,
                    'success': True
                })
            else:
                print("✗ FAILED: No mask generated")
                rendered_images.append({
                    'commands': commands,
                    'mask': None,
                    'success': False
                })
        
        except Exception as e:
            print(f"✗ FAILED: {e}")
            rendered_images.append({
                'commands': commands,
                'mask': None,
                'success': False,
                'error': str(e)
            })
        
        print()
    
    return rendered_images

def visualize_results(parsing_results, rendering_results):
    """Create visualizations to compare before/after quality"""
    
    print("=== Creating Visualizations ===\n")
    
    # Create a comprehensive comparison plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Corrected Bongard-LOGO Parser Results', fontsize=16)
    
    # Plot rendering results
    for i, result in enumerate(rendering_results[:4]):
        row = i // 2
        col = i % 2 * 2 + 1
        
        if result['success'] and result['mask'] is not None:
            axes[row, col].imshow(result['mask'], cmap='gray')
            axes[row, col].set_title(f"Rendered: {' + '.join(result['commands'])}")
        else:
            axes[row, col].text(0.5, 0.5, 'FAILED', ha='center', va='center', 
                               transform=axes[row, col].transAxes, fontsize=14, color='red')
            axes[row, col].set_title(f"Failed: {' + '.join(result['commands'])}")
        
        axes[row, col].axis('off')
    
    # Plot parsing statistics
    for i, result in enumerate(parsing_results[:4]):
        row = i // 2
        col = i % 2 * 2
        
        if result['success']:
            # Create a simple visualization of coordinate bounds
            x_range = result['x_range']
            y_range = result['y_range']
            
            # Plot coordinate bounds as rectangles
            axes[row, col].add_patch(plt.Rectangle(
                (x_range[0], y_range[0]), 
                x_range[1] - x_range[0], 
                y_range[1] - y_range[0],
                fill=False, edgecolor='blue', linewidth=2
            ))
            axes[row, col].set_xlim(-30, 30)
            axes[row, col].set_ylim(-30, 30)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_title(f"Bounds: {result['test']}")
            axes[row, col].set_xlabel('X coordinate')
            axes[row, col].set_ylabel('Y coordinate')
        else:
            axes[row, col].text(0.5, 0.5, 'PARSE\nFAILED', ha='center', va='center',
                               transform=axes[row, col].transAxes, fontsize=14, color='red')
            axes[row, col].set_title(f"Failed: {result['test']}")
        
        axes[row, col].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('corrected_parser_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("=== Summary Results ===")
    
    successful_parses = sum(1 for r in parsing_results if r['success'])
    successful_renders = sum(1 for r in rendering_results if r['success'])
    
    print(f"Parsing Success Rate: {successful_parses}/{len(parsing_results)} ({100*successful_parses/len(parsing_results):.1f}%)")
    print(f"Rendering Success Rate: {successful_renders}/{len(rendering_results)} ({100*successful_renders/len(rendering_results):.1f}%)")
    
    if successful_renders > 0:
        avg_fill_ratio = np.mean([r['fill_ratio'] for r in rendering_results if r['success']])
        print(f"Average Fill Ratio: {avg_fill_ratio:.1%}")
    
    print()

def main():
    """Main test function"""
    
    print("Testing Corrected Bongard-LOGO Parser Implementation")
    print("=" * 60)
    print()
    
    # Test parsing corrections
    parsing_results = test_corrected_parsing()
    
    # Test rendering corrections
    rendering_results = test_corrected_rendering()
    
    # Create visualizations
    visualize_results(parsing_results, rendering_results)
    
    print("Test completed! Check 'corrected_parser_results.png' for visual results.")

if __name__ == "__main__":
    main()
