#!/usr/bin/env python3
"""
Test the visualization system - this will show you the full canvas with multiple images.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

def test_visualization():
    """Test the full visualization with multiple images in a grid."""
    print("🎨 Testing Full Canvas Visualization")
    print("=" * 50)
    
    try:
        from src.bongard_generator.dataset import SyntheticBongardDataset
        from src.bongard_generator.visualize import show_mosaic, plot_rule_distribution
        import matplotlib.pyplot as plt
        
        # Create rules for testing different shapes and features  
        rules = [
            ('SHAPE(circle)', 4),
            ('SHAPE(square)', 4), 
            ('FILL(solid)', 3),
            ('FILL(outline)', 3),
            ('COUNT(2)', 3),
            ('RELATION(overlap)', 3)
        ]
        
        print("📊 Creating synthetic dataset with multiple rules...")
        ds = SyntheticBongardDataset(rules=rules, img_size=128, grayscale=True)
        
        print(f"✅ Dataset created with {len(ds)} examples")
        
        # Show the full canvas mosaic - this is what you were looking for!
        print("🖼️ Displaying FULL CANVAS MOSAIC with multiple images...")
        n_images = min(16, len(ds))  # Don't try to show more images than we have
        show_mosaic(ds, n=n_images, cols=4)  # This shows images in a 4x4 grid
        
        # Also show rule distribution
        print("📈 Displaying rule distribution...")
        plot_rule_distribution(ds)
        
        print("🎉 Full canvas visualization completed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_image_generation():
    """Test individual image generation to confirm shapes are being drawn."""
    print("\n🔍 Testing Individual Image Generation")
    print("=" * 50)
    
    try:
        from src.bongard_generator.dataset import create_composite_scene
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Create test objects with different shapes
        test_objects = [
            {'shape': 'circle', 'x': 30, 'y': 30, 'size': 20, 'color': 'red', 'fill': 'solid'},
            {'shape': 'square', 'x': 70, 'y': 30, 'size': 20, 'color': 'blue', 'fill': 'solid'},
            {'shape': 'triangle', 'x': 50, 'y': 70, 'size': 20, 'color': 'green', 'fill': 'solid'},
            {'shape': 'pentagon', 'x': 90, 'y': 70, 'size': 15, 'color': 'purple', 'fill': 'outline'}
        ]
        
        print("🎨 Generating test image with multiple shapes...")
        img = create_composite_scene(test_objects, 128)
        
        # Display the single test image
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.title("Test Scene: Circle, Square, Triangle, Pentagon")
        plt.axis('off')
        plt.show()
        
        # Save for verification
        img.save('test_multi_shapes.png')
        print("✅ Test image saved as 'test_multi_shapes.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Full Canvas Visualization Test")
    print("=" * 60)
    
    # Test individual image generation first
    individual_success = test_individual_image_generation()
    
    # Test full canvas visualization
    visualization_success = test_visualization()
    
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    if individual_success:
        print("✅ Individual image generation working")
    else:
        print("❌ Individual image generation failed")
        
    if visualization_success:
        print("✅ Full canvas mosaic visualization working")
        print("🎉 You should see multiple matplotlib windows with:")
        print("   • A 4x4 grid of generated Bongard scenes")
        print("   • A bar chart showing rule distribution")
        print("   • Individual test image with multiple shapes")
    else:
        print("❌ Full canvas visualization failed")
    
    if individual_success and visualization_success:
        print("\n🚀 ALL VISUALIZATION TESTS PASSED!")
        print("💡 The full canvas with multiple images should now be visible!")
    else:
        print("\n🔧 Some visualization tests failed. Check the errors above.")
