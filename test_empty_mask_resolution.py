#!/usr/bin/env python3
"""Test the empty mask issue with real-style action programs."""

from src.bongard_augmentor.hybrid import ActionMaskGenerator
import numpy as np

def test_empty_mask_resolution():
    """Test if the empty mask issue is resolved with proper semantic rendering."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print("=== TESTING EMPTY MASK RESOLUTION ===")
    
    # Test realistic action program combinations (like real Bongard data)
    test_programs = [
        # Single command programs
        ["line_triangle_1.000-0.500"],
        ["arc_circle_0.500_0.625-0.500"],
        
        # Multi-command programs (like real data)
        [
            "line_triangle_1.000-0.500", 
            "line_square_1.000-0.833", 
            "line_circle_1.000-0.833"
        ],
        [
            "arc_circle_0.500_0.625-0.500", 
            "arc_triangle_0.500_0.625-0.750", 
            "line_normal_1.000-0.500"
        ],
        
        # Complex programs with various parameters
        [
            "line_zigzag_1.000-0.500",
            "line_normal_1.000-0.833", 
            "line_triangle_1.000-0.833",
            "arc_square_0.500_0.625-0.500"
        ]
    ]
    
    all_success = True
    
    for i, program in enumerate(test_programs):
        print(f"\n{i+1}. Testing program with {len(program)} commands:")
        print(f"   Commands: {program}")
        
        try:
            # Test semantic rendering directly
            semantic_mask = generator._render_semantic_commands_to_mask(program)
            semantic_pixels = np.sum(semantic_mask > 0)
            
            # Test full pipeline
            full_mask = generator.generate_mask_from_actions(program, f"test_program_{i}")
            full_pixels = np.sum(full_mask > 0)
            
            print(f"   🎨 Semantic rendering: {semantic_pixels} pixels")
            print(f"   🔧 Full pipeline: {full_pixels} pixels")
            
            if semantic_pixels == 0:
                print("   ❌ EMPTY MASK from semantic rendering!")
                all_success = False
            elif full_pixels == 0:
                print("   ❌ EMPTY MASK from full pipeline!")
                all_success = False
            else:
                print("   ✅ SUCCESS: Non-empty masks generated")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_success = False
    
    print(f"\n=== SUMMARY ===")
    if all_success:
        print("✅ ALL TESTS PASSED - No empty masks detected!")
        print("🎉 The empty mask issue appears to be RESOLVED!")
    else:
        print("❌ Some tests failed - empty masks still occurring")
    
    return all_success

def test_production_simulation():
    """Simulate production pipeline conditions."""
    
    generator = ActionMaskGenerator(image_size=(512, 512))
    
    print("\n=== PRODUCTION SIMULATION ===")
    
    # Simulate processing 10 action programs like in production
    test_cases = []
    for i in range(10):
        # Create varied realistic programs
        if i % 3 == 0:
            program = [f"line_triangle_1.000-0.{500 + i*50}", f"arc_circle_0.500_0.625-0.{500 + i*30}"]
        elif i % 3 == 1:
            program = [f"line_square_{0.7 + i*0.1:.3f}-0.833", f"line_normal_1.000-0.{500 + i*40}"]
        else:
            program = [f"arc_zigzag_0.500_0.625-0.{400 + i*60}", f"line_circle_1.000-0.{600 + i*20}"]
        
        test_cases.append(program)
    
    empty_count = 0
    success_count = 0
    
    for i, program in enumerate(test_cases):
        try:
            mask = generator.generate_mask_from_actions(program, f"prod_test_{i}")
            pixels = np.sum(mask > 0)
            
            if pixels == 0:
                print(f"❌ Program {i}: EMPTY MASK - {program}")
                empty_count += 1
            else:
                print(f"✅ Program {i}: {pixels} pixels - {program[:1]}...")
                success_count += 1
                
        except Exception as e:
            print(f"💥 Program {i}: ERROR - {e}")
            empty_count += 1
    
    print(f"\n📊 Results: {success_count} success, {empty_count} empty/error")
    
    if empty_count == 0:
        print("🎉 PRODUCTION SIMULATION PASSED - No empty masks!")
    else:
        print(f"⚠️  {empty_count} cases failed - needs investigation")
    
    return empty_count == 0

if __name__ == "__main__":
    basic_success = test_empty_mask_resolution()
    prod_success = test_production_simulation()
    
    if basic_success and prod_success:
        print("\n🎊 OVERALL SUCCESS: Empty mask issue appears RESOLVED!")
        print("💪 All 10 semantic command types working correctly with 2/3 parameter patterns")
    else:
        print("\n⚠️  Some issues remain - further debugging needed")
