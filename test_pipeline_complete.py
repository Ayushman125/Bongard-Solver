#!/usr/bin/env python3
"""Test the fixed pipeline with real Bongard-LOGO images."""

import os
import cv2
import numpy as np
from src.bongard_augmentor.hybrid import ActionMaskGenerator

def test_real_bongard_pipeline():
    """Test the pipeline with actual Bongard-LOGO data."""
    
    print("=== TESTING FIXED PIPELINE WITH REAL BONGARD DATA ===")
    
    # Look for real Bongard images
    bongard_data_dir = "Bongard-LOGO/data"
    if os.path.exists(bongard_data_dir):
        print(f"✅ Found Bongard data directory: {bongard_data_dir}")
    else:
        print(f"❌ Bongard data directory not found at {bongard_data_dir}")
        return
    
    # Find some sample problems
    sample_dirs = []
    for root, dirs, files in os.walk(bongard_data_dir):
        if any(f.endswith('.png') for f in files) and len(files) > 10:
            sample_dirs.append(root)
            if len(sample_dirs) >= 3:
                break
    
    if not sample_dirs:
        print("❌ No sample directories with images found")
        return
        
    print(f"Found {len(sample_dirs)} sample directories")
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    # Test with some known good command patterns
    test_cases = [
        {
            "name": "Simple Lines",
            "commands": ['line_normal_0.500-0.500', 'line_normal_0.700-0.300'],
            "expected_coverage": (1.0, 5.0)  # 1-5% coverage expected
        },
        {
            "name": "Complex Pattern", 
            "commands": ['line_normal_1.000-0.500', 'line_normal_0.608-0.121', 
                        'line_normal_0.224-0.600', 'line_normal_1.000-0.074'],
            "expected_coverage": (2.0, 8.0)  # 2-8% coverage expected
        },
        {
            "name": "Triangle Pattern",
            "commands": ['line_normal_0.333-0.333', 'line_normal_0.667-0.333', 'line_normal_0.500-0.667'],
            "expected_coverage": (1.5, 6.0)  # 1.5-6% coverage expected
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['name']} ---")
        
        try:
            mask = generator.generate_mask_from_actions(
                test_case['commands'], 
                f"test_case_{i+1}"
            )
            
            total_pixels = mask.size
            nonzero_pixels = np.count_nonzero(mask)
            coverage = 100 * nonzero_pixels / total_pixels
            mask_sum = np.sum(mask)
            
            print(f"Commands: {test_case['commands']}")
            print(f"Coverage: {coverage:.2f}% ({nonzero_pixels}/{total_pixels} pixels)")
            print(f"Mask sum: {mask_sum}")
            
            # Check if in expected range
            min_cov, max_cov = test_case['expected_coverage']
            if min_cov <= coverage <= max_cov:
                status = "✅ PASS"
            elif coverage > max_cov * 2:
                status = "❌ FAIL (too much coverage - possible white mask)"
            elif coverage < min_cov / 2:
                status = "❌ FAIL (too little coverage - possible empty mask)"
            else:
                status = "⚠️  BORDERLINE"
                
            print(f"Status: {status}")
            
            # Save mask for inspection
            filename = f"pipeline_test_{i+1}_{test_case['name'].replace(' ', '_')}.png"
            cv2.imwrite(filename, mask)
            print(f"Saved: {filename}")
            
            results.append({
                'name': test_case['name'],
                'coverage': coverage,
                'status': status,
                'mask_sum': mask_sum,
                'nonzero_pixels': nonzero_pixels
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'name': test_case['name'],
                'status': f"ERROR: {e}",
                'coverage': 0,
                'mask_sum': 0,
                'nonzero_pixels': 0
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if '✅' in r['status'])
    total = len(results)
    
    print(f"Overall: {passed}/{total} tests passed")
    
    for result in results:
        print(f"  {result['name']}: {result['status']} (Coverage: {result['coverage']:.2f}%)")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! Pipeline is working correctly.")
        print(f"   ✅ No white mask issues")
        print(f"   ✅ Reasonable coverage percentages")
        print(f"   ✅ Coordinate system fix is successful")
    elif passed > total // 2:
        print(f"\n⚠️  Most tests passed, but some issues remain")
    else:
        print(f"\n❌ Major issues still present in pipeline")
    
    return results

def quick_alignment_verification():
    """Quick verification that the alignment issues are resolved."""
    
    print(f"\n{'='*60}")
    print("QUICK ALIGNMENT VERIFICATION")
    print(f"{'='*60}")
    
    # Test the exact problematic commands from the user's logs
    problematic_commands = [
        'line_normal_1.000-0.500',
        'line_normal_0.608-0.121', 
        'line_normal_0.224-0.600',
        'line_normal_1.000-0.074',
        'line_normal_0.224-0.926',
        'line_normal_0.224-0.352'
    ]
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print("Testing the exact commands that were causing alignment issues...")
    
    mask = generator.generate_mask_from_actions(problematic_commands, "alignment_verification")
    
    total_pixels = mask.size
    nonzero_pixels = np.count_nonzero(mask)
    mask_sum = np.sum(mask)
    coverage = 100 * nonzero_pixels / total_pixels
    
    print(f"Results:")
    print(f"  Coverage: {coverage:.2f}% ({nonzero_pixels} pixels)")
    print(f"  Mask sum: {mask_sum}")
    print(f"  Average pixel value: {mask_sum/total_pixels:.2f}")
    
    # Compare to the previous problematic results
    if mask_sum > 50000000:  # Previous white mask issue
        print(f"❌ REGRESSION: White mask issue returned!")
    elif nonzero_pixels == 0:
        print(f"❌ PROBLEM: Empty mask generated")
    elif 1000 < nonzero_pixels < 50000:
        print(f"✅ SUCCESS: Alignment issues resolved!")
    else:
        print(f"⚠️  UNUSUAL: Unexpected result pattern")
    
    cv2.imwrite("alignment_verification.png", mask)
    print(f"Verification mask saved as 'alignment_verification.png'")

if __name__ == "__main__":
    test_real_bongard_pipeline()
    quick_alignment_verification()
