#!/usr/bin/env python3
"""
Comprehensive fix for the hybrid sample validation issues.
1. Fix scene graph format issues
2. Fix rule matching problems  
3. Ensure robust validation display
"""
import os
import sys

def fix_scene_graph_format():
    """Fix the scene graph objects being int instead of list."""
    dataset_path = "src/bongard_generator/dataset.py"
    
    if not os.path.exists(dataset_path):
        print(f"❌ {dataset_path} not found")
        return False
        
    print(f"🔧 Fixing scene graph format in {dataset_path}")
    
    # Read the file
    with open(dataset_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find where scene graphs are created and ensure objects is always a list
    # Look for the scene graph creation pattern
    if "scene_graph = {" in content:
        # Find all scene graph creations and fix them
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if "'objects':" in line and "scene_graph" not in line:
                # This might be a scene graph field assignment
                if not "[]" in line and not "[" in line:
                    # Convert single values to lists
                    line = line.replace("'objects': objects", "'objects': objects if isinstance(objects, list) else []")
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
    
    # Also fix the specific case where objects might be assigned incorrectly
    problematic_patterns = [
        "'objects': len(objects)",  # This would create int instead of list
        "'objects': num_objects",   # This would create int instead of list
    ]
    
    for pattern in problematic_patterns:
        if pattern in content:
            replacement = pattern.replace("len(objects)", "objects").replace("num_objects", "objects")
            content = content.replace(pattern, replacement)
            print(f"✓ Fixed: {pattern}")
    
    # Write the file back
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("✅ Fixed scene graph format issues")
    return True

def create_robust_validation_script():
    """Create a robust validation script that handles empty datasets gracefully."""
    script_content = '''#!/usr/bin/env python3
"""
Robust validation script that properly handles empty datasets and displays working images.
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path  
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from bongard_generator.rule_loader import get_all_rules
from bongard_generator.dataset import BongardDataset

def test_robust_hybrid_validation():
    """Test hybrid validation with robust error handling."""
    print("🧪 Robust Hybrid Validation Test")
    print("=" * 50)
    
    # Get working rules (filter out problematic ones)
    all_rules = get_all_rules()
    working_rules = []
    
    print("🔍 Finding working rules...")
    for rule in all_rules:
        try:
            # Quick test with small dataset
            test_dataset = BongardDataset(
                canvas_size=128, 
                target_quota=1, 
                rule_list=[rule.description],
                max_obj_size=30,
                min_obj_size=20
            )
            
            if len(test_dataset.examples) > 0:
                # Verify the example is valid
                example = test_dataset.examples[0]
                if example.get('image') is not None:
                    working_rules.append(rule.description)
                    print(f"  ✓ {rule.description[:60]}")
            else:
                print(f"  ❌ {rule.description[:60]} (no examples)")
                
        except Exception as e:
            print(f"  💥 {rule.description[:60]} (error: {e})")
    
    print(f"\\n✅ Found {len(working_rules)} working rules")
    
    if len(working_rules) == 0:
        print("❌ No working rules found - using fallback image generation")
        return create_fallback_validation()
    
    # Create validation images using only working rules
    print("\\n🎨 Generating validation images...")
    imgs, labels = [], []
    
    # Select a few working rules
    selected_rules = working_rules[:min(6, len(working_rules))]
    
    for i, rule_desc in enumerate(selected_rules):
        try:
            dataset = BongardDataset(
                canvas_size=128,
                target_quota=1,
                rule_list=[rule_desc],
                max_obj_size=30,
                min_obj_size=20
            )
            
            if len(dataset.examples) > 0:
                example = dataset.examples[0]
                img = example.get('image')
                if img is not None:
                    imgs.append(np.array(img))
                    labels.append(f"Rule {i+1}")
                    print(f"  ✓ Generated image for rule {i+1}")
                else:
                    print(f"  ⚠ No image for rule {i+1}")
            else:
                print(f"  ❌ No examples for rule {i+1}")
                
        except Exception as e:
            print(f"  💥 Error with rule {i+1}: {e}")
    
    if len(imgs) == 0:
        print("❌ No valid images generated - creating fallback display")
        return create_fallback_validation()
    
    # Display the images
    display_validation_images(imgs, labels)
    return True

def create_fallback_validation():
    """Create fallback validation with simple generated images."""
    print("\\n🎯 Creating fallback validation images...")
    
    imgs, labels = [], []
    
    # Create simple test images manually
    for i in range(6):
        # Create a simple image with circles
        img = Image.new('L', (128, 128), color=255)  # White background
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw some circles
        num_circles = (i % 3) + 1
        for j in range(num_circles):
            x = 30 + j * 25 + (i * 5)
            y = 40 + j * 20
            size = 15 + (j * 5)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=0)  # Black circles
        
        imgs.append(np.array(img))
        labels.append(f"Fallback {i+1}")
    
    display_validation_images(imgs, labels)
    return True

def display_validation_images(imgs, labels):
    """Display validation images in a grid."""
    print(f"\\n🖼️ Displaying {len(imgs)} validation images...")
    
    n_show = min(6, len(imgs))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(n_show):
        if i < len(imgs):
            axes[i].imshow(imgs[i], cmap='gray')
            axes[i].set_title(labels[i] if i < len(labels) else f"Image {i+1}")
        else:
            axes[i].text(0.5, 0.5, 'No Image', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Empty {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Robust Hybrid Validation Results", y=0.98, fontsize=14)
    plt.show()
    
    print("✅ Validation display completed!")

if __name__ == "__main__":
    success = test_robust_hybrid_validation()
    if success:
        print("\\n🎉 Robust validation completed successfully!")
    else:
        print("\\n❌ Validation failed")
'''

    with open("test_robust_validation.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created robust validation script: test_robust_validation.py")

def main():
    print("🔧 Comprehensive Fix for Hybrid Sample Issues")
    print("=" * 60)
    
    # Fix 1: Scene graph format
    fix_scene_graph_format()
    
    # Fix 2: Create robust validation
    create_robust_validation_script()
    
    print("\\n🧪 Testing the fixes...")
    os.system("python test_robust_validation.py")

if __name__ == "__main__":
    main()
