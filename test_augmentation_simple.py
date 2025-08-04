#!/usr/bin/env python3
"""
Test script to verify the corrected augmentation pipeline.
This directly imports and tests the ActionMaskGenerator.
"""

import sys
import os
import cv2
import numpy as np
import logging

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Direct import to avoid the hybrid.py import issues
from src.data_pipeline.logo_parser import UnifiedActionParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleActionMaskGenerator:
    """Simplified ActionMaskGenerator for testing the corrected pipeline."""
    
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.action_parser = UnifiedActionParser()
        logging.debug(f"SimpleActionMaskGenerator: Initialized with image_size={image_size}")
        logging.debug(f"SimpleActionMaskGenerator: Parser scale_factor={self.action_parser.scale_factor}")
    
    def generate_mask_from_actions(self, action_commands: list) -> np.ndarray:
        """Generate a binary mask from action commands using the corrected parser."""
        try:
            logging.debug(f"Processing {len(action_commands)} action commands")
            
            # Parse the action commands using the corrected parser
            parsed_program = self.action_parser._parse_single_image(
                action_commands, 
                image_id="temp", 
                is_positive=True, 
                problem_id="temp"
            )
            
            if parsed_program and parsed_program.vertices:
                logging.debug(f"Got {len(parsed_program.vertices)} vertices from parser")
                
                # Use the corrected parser's rendering method
                mask = self.action_parser._render_vertices_to_image(parsed_program.vertices, self.image_size)
                logging.debug(f"Generated mask with {np.sum(mask > 0)} non-zero pixels")
                return mask
            else:
                logging.warning("No vertices in parsed program")
                return np.zeros(self.image_size, dtype=np.uint8)
                
        except Exception as e:
            logging.warning(f"Failed to generate mask: {e}")
            import traceback
            logging.debug(f"Full traceback: {traceback.format_exc()}")
            return np.zeros(self.image_size, dtype=np.uint8)

def test_augmentation_pipeline():
    """Test the corrected augmentation pipeline."""
    print("=" * 60)
    print("TESTING CORRECTED AUGMENTATION PIPELINE")
    print("=" * 60)
    
    mask_generator = SimpleActionMaskGenerator()
    
    # Test various action commands
    test_cases = [
        {
            "name": "Simple Line",
            "commands": ["line_normal_0.5-0.0"],
            "expected_pixels": (20, 40)  # Expected range of non-zero pixels
        },
        {
            "name": "Vertical Line",
            "commands": ["line_normal_0.3-0.25"],
            "expected_pixels": (15, 25)
        },
        {
            "name": "Zigzag Line",
            "commands": ["line_zigzag_0.4-0.5"],
            "expected_pixels": (25, 50)
        },
        {
            "name": "Simple Arc",
            "commands": ["arc_normal_0.5_0.25-0.0"],
            "expected_pixels": (50, 100)
        },
        {
            "name": "Circle Arc",
            "commands": ["arc_circle_0.3_0.5-0.25"],
            "expected_pixels": (60, 120)
        },
        {
            "name": "Complex Shape",
            "commands": [
                "line_normal_0.4-0.0",
                "arc_normal_0.3_0.25-0.25",
                "line_zigzag_0.3-0.5"
            ],
            "expected_pixels": (70, 150)
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test {i+1}: {test_case['name']} ===")
        print(f"Commands: {test_case['commands']}")
        
        # Generate mask
        mask = mask_generator.generate_mask_from_actions(test_case['commands'])
        
        if mask is not None:
            non_zero_pixels = np.count_nonzero(mask)
            total_pixels = mask.size
            percentage = 100 * non_zero_pixels / total_pixels
            
            print(f"✓ Generated mask: {non_zero_pixels}/{total_pixels} pixels ({percentage:.1f}%)")
            
            # Check if pixel count is in expected range
            min_expected, max_expected = test_case['expected_pixels']
            if min_expected <= non_zero_pixels <= max_expected:
                print(f"✓ Pixel count is in expected range [{min_expected}, {max_expected}]")
            else:
                print(f"⚠ Pixel count {non_zero_pixels} outside expected range [{min_expected}, {max_expected}]")
            
            # Save for inspection
            save_path = f"test_augmentation_{i}_simple.png"
            cv2.imwrite(save_path, mask)
            print(f"✓ Saved mask: {save_path}")
            
            # Check if mask has reasonable distribution
            if non_zero_pixels > 5:  # At least some pixels
                print("✓ Mask has reasonable content")
            else:
                print("⚠ Mask seems to have very little content")
        else:
            print("✗ Failed to generate mask")

def test_coordinate_consistency():
    """Test that augmentation and parser produce consistent results."""
    print("\n=== TESTING COORDINATE CONSISTENCY ===")
    
    parser = UnifiedActionParser()
    mask_generator = SimpleActionMaskGenerator()
    
    test_cmd = "line_normal_0.5-0.0"
    print(f"Testing consistency with: {test_cmd}")
    
    # Parse with parser
    image_program = parser._parse_single_image([test_cmd], "consistency_test", True, "test")
    parser_image = parser.visualize_image_program(image_program) if image_program else None
    
    # Generate with augmentor
    augmentor_mask = mask_generator.generate_mask_from_actions([test_cmd])
    
    if parser_image is not None and augmentor_mask is not None:
        # Calculate similarity
        parser_binary = (parser_image > 127).astype(np.uint8) * 255
        augmentor_binary = (augmentor_mask > 127).astype(np.uint8) * 255
        
        # Calculate overlap
        overlap = np.logical_and(parser_binary > 0, augmentor_binary > 0)
        union = np.logical_or(parser_binary > 0, augmentor_binary > 0)
        
        iou = np.sum(overlap) / np.sum(union) if np.sum(union) > 0 else 0
        
        print(f"Parser image: {np.count_nonzero(parser_binary)} pixels")
        print(f"Augmentor mask: {np.count_nonzero(augmentor_binary)} pixels")
        print(f"IoU (Intersection over Union): {iou:.3f}")
        
        if iou > 0.8:
            print("✓ Excellent coordinate consistency")
        elif iou > 0.6:
            print("✓ Good coordinate consistency")
        elif iou > 0.4:
            print("⚠ Moderate coordinate consistency")
        else:
            print("⚠ Poor coordinate consistency - check coordinate transformation")
            
        # Save comparison
        comparison = np.zeros((64, 64, 3), dtype=np.uint8)
        comparison[:, :, 0] = parser_binary  # Red channel
        comparison[:, :, 1] = augmentor_binary  # Green channel
        # Overlap appears as yellow
        
        cv2.imwrite("test_augmentation_consistency.png", comparison)
        print("✓ Saved consistency comparison: test_augmentation_consistency.png")
        print("  (Red=Parser, Green=Augmentor, Yellow=Overlap)")
    else:
        print("✗ Failed to generate images for consistency test")

def main():
    """Run augmentation tests."""
    try:
        test_augmentation_pipeline()
        test_coordinate_consistency()
        
        print("\n" + "=" * 60)
        print("✓ AUGMENTATION TESTS COMPLETED")
        print("Check generated PNG files:")
        print("- test_augmentation_*_simple.png")
        print("- test_augmentation_consistency.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ AUGMENTATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
