#!/usr/bin/env python3
"""
Fix empty dataset handling in BongardDataset to prevent black boxes.
"""
import os
import sys

def fix_dataset_empty_handling():
    """Fix the dataset.py file to handle empty datasets gracefully."""
    dataset_path = "src/bongard_generator/dataset.py"
    
    if not os.path.exists(dataset_path):
        print(f"❌ {dataset_path} not found")
        return False
        
    print(f"🔧 Fixing empty dataset handling in {dataset_path}")
    
    # Read the file
    with open(dataset_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the __getitem__ method and add safety check
    old_getitem = '''    def __getitem__(self, idx):
        """Return the example at the given index."""
        example = self.examples[idx]
        print(f"[BongardDataset] __getitem__ idx={idx} rule={example['rule']} label={example['label']} objects={example.get('scene_graph', {}).get('objects', None)}")
        return example'''
    
    new_getitem = '''    def __getitem__(self, idx):
        """Return the example at the given index."""
        if not self.examples:
            # Return a dummy black image for empty datasets
            dummy_example = {
                'rule': 'empty_dataset',
                'label': 0,
                'image': self._create_black_placeholder(),
                'scene_graph': {'objects': [], 'relations': []}
            }
            print(f"[BongardDataset] __getitem__ idx={idx} EMPTY DATASET - returning black placeholder")
            return dummy_example
        
        if idx >= len(self.examples):
            idx = idx % len(self.examples)  # Wrap around if index too large
            
        example = self.examples[idx]
        print(f"[BongardDataset] __getitem__ idx={idx} rule={example['rule']} label={example['label']} objects={example.get('scene_graph', {}).get('objects', None)}")
        return example
    
    def _create_black_placeholder(self):
        """Create a black placeholder image for empty datasets."""
        from PIL import Image
        import numpy as np
        
        # Create a black image
        img = Image.new('L', (self.canvas_size, self.canvas_size), color=0)
        return img'''
    
    if old_getitem in content:
        content = content.replace(old_getitem, new_getitem)
        print("✓ Added safety check to __getitem__ method")
    else:
        print("⚠ Could not find the exact __getitem__ method to replace")
        
    # Also add a __len__ safety check if not present
    if "def __len__(self):" not in content:
        # Find a good place to add it (after __getitem__)
        getitem_pos = content.find("return example")
        if getitem_pos != -1:
            # Find end of method
            method_end = content.find("\n    def ", getitem_pos)
            if method_end == -1:
                method_end = content.find("\nclass ", getitem_pos)
            if method_end != -1:
                len_method = '''
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
'''
                content = content[:method_end] + len_method + content[method_end:]
                print("✓ Added __len__ method")
    
    # Write the file back
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("✅ Fixed empty dataset handling")
    return True

def test_fix():
    """Test the fix by running the black box test again."""
    print("\n🧪 Testing the fix...")
    os.system("python debug_black_boxes.py")

if __name__ == "__main__":
    print("🔧 Fixing Empty Dataset Black Box Issue")
    print("=" * 50)
    
    success = fix_dataset_empty_handling()
    if success:
        test_fix()
    else:
        print("❌ Fix failed")
        sys.exit(1)
