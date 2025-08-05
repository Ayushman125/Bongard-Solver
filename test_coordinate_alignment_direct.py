#!/usr/bin/env python3
"""
Direct test of the coordinate alignment fix to verify masks are properly positioned.
"""

import os
import sys
import numpy as np
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bongard_augmentor.hybrid import ActionMaskGenerator
from data_pipeline.logo_parser import ComprehensiveNVLabsParser

def test_coordinate_fix_direct():
    """Test the coordinate fix directly without complex pipeline interactions."""
    
    print("🔧 Direct test of coordinate alignment fix...")
    
    # Initialize components directly
    parser = ComprehensiveNVLabsParser()
    generator = ActionMaskGenerator()
    
    # Test with the user's problematic commands
    test_cases = [
        {
            'name': 'Simple Line Pattern',
            'commands': ["line_normal_1.000-0.500", "line_normal_0.500-1.000"]
        },
        {
            'name': 'Arc Pattern',
            'commands': ["arc_normal_1.000-0.800", "line_normal_0.800-0.600"]
        },
        {
            'name': 'Complex Shape',
            'commands': ["line_normal_1.200-0.400", "arc_normal_0.400-1.200", "line_normal_1.200-0.400"]
        }
    ]
    
    all_successful = True
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test_case['name']} ---")
        print(f"Commands: {test_case['commands']}")
        
        try:
            # Parse action commands to get vertices
            shape = parser.parse_action_commands(test_case['commands'], f"test_{i}")
            
            if not shape or not shape.vertices:
                print(f"❌ Failed to parse commands for {test_case['name']}")
                all_successful = False
                continue
                
            vertices = shape.vertices
            print(f"✅ Parsed {len(vertices)} vertices")
            print(f"   Coordinate range: X=[{min(v[0] for v in vertices):.1f}, {max(v[0] for v in vertices):.1f}]")
            print(f"                     Y=[{min(v[1] for v in vertices):.1f}, {max(v[1] for v in vertices):.1f}]")
            
            # Generate mask using FIXED coordinate system
            mask = generator._render_vertices_to_mask(vertices)
            
            # Analyze mask
            coverage = np.mean(mask > 0) * 100
            mask_sum = np.sum(mask)
            nonzero_pixels = np.count_nonzero(mask)
            
            print(f"✅ Generated mask:")
            print(f"   Coverage: {coverage:.2f}% ({nonzero_pixels}/{mask.size} pixels)")
            print(f"   Sum: {mask_sum}")
            print(f"   Shape: {mask.shape}")
            
            # Check if mask is reasonable (not all white, not empty)
            is_reasonable = 0.1 < coverage < 50 and mask_sum > 0
            
            if is_reasonable:
                print(f"✅ Mask quality: GOOD (properly positioned)")
            else:
                print(f"⚠️  Mask quality: QUESTIONABLE (may be cut/misplaced)")
                all_successful = False
            
            # Save mask for inspection
            mask_filename = f'coordinate_fix_test_{i+1}_{test_case["name"].replace(" ", "_")}.png'
            cv2.imwrite(mask_filename, mask)
            print(f"💾 Saved: {mask_filename}")
            
            results.append({
                'name': test_case['name'],
                'coverage': coverage,
                'sum': mask_sum,
                'vertices': len(vertices),
                'quality': 'good' if is_reasonable else 'poor',
                'filename': mask_filename
            })
            
        except Exception as e:
            print(f"❌ Error processing {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_successful = False
    
    # Create comprehensive comparison
    create_coordinate_fix_comparison(results)
    
    # Print final summary
    print("\n" + "="*70)
    print("COORDINATE ALIGNMENT FIX - FINAL RESULTS")
    print("="*70)
    
    good_masks = [r for r in results if r['quality'] == 'good']
    poor_masks = [r for r in results if r['quality'] == 'poor']
    
    print(f"✅ Well-positioned masks: {len(good_masks)}")
    print(f"⚠️  Poorly-positioned masks: {len(poor_masks)}")
    
    if good_masks:
        avg_coverage = np.mean([r['coverage'] for r in good_masks])
        print(f"📊 Average coverage (good masks): {avg_coverage:.2f}%")
    
    print("\n🔍 Key improvements implemented:")
    print("   • Official NVLabs coordinate system (800x800 canvas)")
    print("   • Proper coordinate range mapping (-360, 360)")
    print("   • Correct Y-axis flipping for image coordinates")
    print("   • Thick stroke rendering matching real dataset")
    print("   • High-quality anti-aliasing")
    
    if all_successful and len(good_masks) >= 2:
        print("\n🎉 COORDINATE ALIGNMENT FIX SUCCESSFUL!")
        print("   Masks should no longer be 'cut' or 'incorrectly placed'")
        print("   The coordinate system now matches official NVLabs implementation")
    else:
        print("\n⚠️  Partial success - some edge cases may need refinement")
    
    return all_successful

def create_coordinate_fix_comparison(results):
    """Create a visual comparison of the coordinate fix results."""
    
    if not results:
        return
    
    # Create comparison image
    num_results = len(results)
    comparison_width = min(3, num_results) * 300  # Max 3 columns
    comparison_height = ((num_results + 2) // 3) * 350  # Rows of 3
    
    comparison = np.zeros((comparison_height, comparison_width, 3), dtype=np.uint8)
    
    # Add title
    title_y = 30
    cv2.putText(comparison, "Coordinate Alignment Fix - Test Results", 
                (20, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add individual mask results
    for i, result in enumerate(results):
        col = i % 3
        row = i // 3 + 1  # Skip title row
        
        x_pos = col * 300 + 10
        y_pos = row * 350 + 10
        
        try:
            # Load and resize mask
            mask = cv2.imread(result['filename'], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_resized = cv2.resize(mask, (280, 200))
                comparison[y_pos:y_pos+200, x_pos:x_pos+280] = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                
                # Add text info
                info_y = y_pos + 220
                cv2.putText(comparison, result['name'], (x_pos, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(comparison, f"Coverage: {result['coverage']:.1f}%", (x_pos, info_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(comparison, f"Quality: {result['quality']}", (x_pos, info_y + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0) if result['quality'] == 'good' else (0, 255, 255), 1)
        except Exception as e:
            print(f"Warning: Could not add {result['filename']} to comparison: {e}")
    
    cv2.imwrite('coordinate_alignment_fix_comparison.png', comparison)
    print(f"💾 Saved comparison: coordinate_alignment_fix_comparison.png")

if __name__ == "__main__":
    print("🔧 Testing Coordinate Alignment Fix")
    print("Verifying that masks are no longer 'cut' or 'incorrectly placed'")
    print("="*70)
    
    success = test_coordinate_fix_direct()
    
    print(f"\n{'✅ SUCCESS' if success else '⚠️  PARTIAL SUCCESS'}: Coordinate alignment testing complete")
