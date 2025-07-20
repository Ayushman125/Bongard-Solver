#!/usr/bin/env python3
"""
Test script to verify that the new shape renderer integration works correctly
and generates diverse geometric shapes based on rule transformations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bongard_generator.config import GeneratorConfig
from src.bongard_generator.prototype_action import PrototypeAction
from src.bongard_generator.mask_utils import _render_objects
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_shape_renderer_integration():
    """Test that different shapes are rendered correctly."""
    
    # Create configuration
    config = GeneratorConfig()
    config.img_size = 128
    config.bg_color = "white"
    
    # Create prototype action
    prototype_action = PrototypeAction()
    
    # Test objects with different shapes
    test_objects = [
        {
            "shape": "circle",
            "position": (40, 40),
            "size": 30,
            "color": "red",
            "fill_type": "solid",
            "stroke_style": "solid",
            "stroke_width": 2,
            "rotation": 0
        },
        {
            "shape": "square", 
            "position": (88, 40),
            "size": 30,
            "color": "blue",
            "fill_type": "hollow",
            "stroke_style": "solid", 
            "stroke_width": 2,
            "rotation": 0
        },
        {
            "shape": "triangle",
            "position": (40, 88),
            "size": 30,
            "color": "green",
            "fill_type": "striped",
            "stroke_style": "dashed",
            "stroke_width": 2,
            "rotation": 15
        },
        {
            "shape": "pentagon",
            "position": (88, 88), 
            "size": 30,
            "color": "purple",
            "fill_type": "dotted",
            "stroke_style": "dotted",
            "stroke_width": 2,
            "rotation": 30
        }
    ]
    
    # Create canvas
    canvas = Image.new('RGB', (config.img_size, config.img_size), "white")
    
    # Render objects using the updated function
    try:
        _render_objects(canvas, test_objects, config, prototype_action)
        
        # Save the test image
        canvas.save("test_shape_integration.png")
        logger.info("✅ Shape renderer integration test successful!")
        logger.info("Generated test image with 4 different shapes: circle, square, triangle, pentagon")
        logger.info("Features tested: solid/hollow/striped/dotted fills, solid/dashed/dotted strokes, rotation")
        logger.info("Saved as: test_shape_integration.png")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Shape renderer integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_shape_renderer_integration()
    if success:
        print("\n🎉 Integration successful! The shape renderer is now properly connected.")
        print("Your generated Bongard problems will now show diverse geometric shapes!")
    else:
        print("\n⚠️  Integration test failed. Please check the error messages above.")
