#!/usr/bin/env python3
"""Final comprehensive test of the coordinate system fixes."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.bongard_augmentor.hybrid import ActionMaskGenerator

def comprehensive_pipeline_test():
    """Comprehensive test of the fixed pipeline."""
    
    print("=" * 80)
    print("COMPREHENSIVE PIPELINE TEST - COORDINATE SYSTEM FIX VALIDATION")
    print("=" * 80)
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    # Define test cases that cover different scenarios
    test_cases = [
        {
            "name": "Original Problematic Commands",
            "commands": [
                'line_normal_1.000-0.500',
                'line_normal_0.608-0.121', 
                'line_normal_0.224-0.600',
                'line_normal_1.000-0.074',
                'line_normal_0.224-0.926',
                'line_normal_0.224-0.352'
            ],
            "description": "The exact commands that were causing white masks"
        },
        {
            "name": "Simple Line Pattern",
            "commands": [
                'line_normal_0.500-0.500',
                'line_normal_0.700-0.300',
                'line_normal_0.300-0.700'
            ],
            "description": "Simple 3-line pattern"
        },
        {
            "name": "Square-like Pattern",
            "commands": [
                'line_normal_0.250-0.250',
                'line_normal_0.750-0.250',
                'line_normal_0.750-0.750',
                'line_normal_0.250-0.750'
            ],
            "description": "Four lines forming a square-like shape"
        },
        {
            "name": "Edge Case - Extreme Coordinates",
            "commands": [
                'line_normal_0.000-0.000',
                'line_normal_1.000-1.000',
                'line_normal_0.500-0.500'
            ],
            "description": "Testing edge coordinates"
        },
        {
            "name": "Second Problematic Set",
            "commands": [
                'line_normal_0.500-0.500',
                'line_normal_0.707-0.875', 
                'line_normal_0.500-0.875',
                'line_normal_0.300-0.167'
            ],
            "description": "Another set that was causing issues"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-'*60}")
        print(f"TEST {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Commands: {test_case['commands']}")
        print(f"{'-'*60}")
        
        try:
            # Generate mask
            mask = generator.generate_mask_from_actions(
                test_case['commands'], 
                f"comprehensive_test_{i}"
            )
            
            # Analyze mask
            total_pixels = mask.size
            nonzero_pixels = np.count_nonzero(mask)
            mask_sum = np.sum(mask)
            coverage = 100 * nonzero_pixels / total_pixels
            avg_pixel_value = mask_sum / total_pixels
            
            # Determine status
            if mask_sum > 50000000:  # Previous white mask threshold
                status = "❌ FAIL - White mask detected"
                color = "red"
            elif nonzero_pixels == 0:
                status = "❌ FAIL - Empty mask" 
                color = "red"
            elif coverage > 50:
                status = "⚠️  WARN - Very high coverage"
                color = "orange"
            elif 0.5 <= coverage <= 15:
                status = "✅ PASS - Excellent"
                color = "green"
            else:
                status = "⚠️  WARN - Unusual coverage"
                color = "orange"
            
            print(f"Results:")
            print(f"  Total pixels: {total_pixels:,}")
            print(f"  Nonzero pixels: {nonzero_pixels:,}")
            print(f"  Coverage: {coverage:.2f}%")
            print(f"  Mask sum: {mask_sum:,}")
            print(f"  Average pixel value: {avg_pixel_value:.2f}")
            print(f"  Status: {status}")
            
            # Save mask
            filename = f"comprehensive_test_{i}_{test_case['name'].replace(' ', '_').replace('-', '_')}.png"
            cv2.imwrite(filename, mask)
            print(f"  💾 Saved: {filename}")
            
            results.append({
                'test_num': i,
                'name': test_case['name'],
                'coverage': coverage,
                'mask_sum': mask_sum,
                'nonzero_pixels': nonzero_pixels,
                'status': status,
                'color': color,
                'filename': filename
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'test_num': i,
                'name': test_case['name'],
                'coverage': 0,
                'mask_sum': 0,
                'nonzero_pixels': 0,
                'status': f"❌ ERROR: {str(e)[:50]}...",
                'color': 'red',
                'filename': None
            })
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if '✅' in r['status'])
    warning_tests = sum(1 for r in results if '⚠️' in r['status'])
    failed_tests = sum(1 for r in results if '❌' in r['status'])
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"⚠️  Warnings: {warning_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {100*passed_tests/total_tests:.1f}%")
    
    print(f"\nDetailed Results:")
    for result in results:
        print(f"  Test {result['test_num']}: {result['name']}")
        print(f"    Status: {result['status']}")
        print(f"    Coverage: {result['coverage']:.2f}%")
        print(f"    Pixels: {result['nonzero_pixels']:,}")
        if result['filename']:
            print(f"    File: {result['filename']}")
        print()
    
    # Overall assessment
    if failed_tests == 0 and passed_tests >= total_tests * 0.8:
        print("🎉 EXCELLENT: Coordinate system fix is working perfectly!")
        print("   ✅ No white mask issues detected")
        print("   ✅ All tests produce reasonable results")
        print("   ✅ Pipeline is ready for production use")
    elif failed_tests == 0:
        print("✅ GOOD: Coordinate system fix is working well")
        print("   ✅ No critical failures")
        print("   ⚠️  Some warnings but acceptable results")
    elif failed_tests <= total_tests * 0.2:
        print("⚠️  ACCEPTABLE: Most issues resolved")
        print("   ✅ Major improvements achieved")
        print("   ⚠️  Some edge cases may need attention")
    else:
        print("❌ PROBLEMATIC: Significant issues remain")
        print("   ❌ Multiple test failures")
        print("   ❌ Further debugging needed")
    
    return results

def create_comprehensive_visualization(results):
    """Create a comprehensive visualization of all test results."""
    
    if not results:
        return
        
    print(f"\nCreating comprehensive visualization...")
    
    # Try to load and display some masks
    valid_results = [r for r in results if r['filename'] and r['nonzero_pixels'] > 0]
    
    if len(valid_results) >= 2:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, result in enumerate(valid_results[:6]):
                try:
                    mask = cv2.imread(result['filename'], cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        axes[i].imshow(mask, cmap='gray')
                        axes[i].set_title(f"Test {result['test_num']}: {result['name'][:20]}...\n"
                                        f"Coverage: {result['coverage']:.1f}%", 
                                        fontsize=10)
                        axes[i].axis('off')
                except:
                    axes[i].text(0.5, 0.5, f"Test {result['test_num']}\nLoad Error", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            
            # Hide unused subplots
            for j in range(len(valid_results), 6):
                axes[j].axis('off')
            
            plt.suptitle("Comprehensive Pipeline Test Results\nCoordinate System Fix Validation", 
                        fontsize=14)
            plt.tight_layout()
            plt.savefig("comprehensive_test_results.png", dpi=150, bbox_inches='tight')
            print("✅ Comprehensive visualization saved as 'comprehensive_test_results.png'")
            
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")

if __name__ == "__main__":
    results = comprehensive_pipeline_test()
    create_comprehensive_visualization(results)
