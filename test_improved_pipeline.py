#!/usr/bin/env python3
"""
Test the updated parsing with the hybrid pipeline
"""

import sys
import os
import logging
import json
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.bongard_augmentor.hybrid import HybridAugmentationPipeline

def test_improved_pipeline():
    """Test the pipeline with improved parsing"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create pipeline config
    config = {
        'data': {
            'action_programs_dir': 'data/raw/ShapeBongard_V2/bd/',
            'output_path': 'data/test_improved_output',
            'n_select': 3  # Just test 3 problems
        },
        'processing': {
            'batch_size': 1
        },
        'image_size': (64, 64)
    }
    
    # Initialize pipeline
    pipeline = HybridAugmentationPipeline(config=config)
    
    print("Testing improved parsing pipeline...")
    
    # Run pipeline
    try:
        pipeline.run_pipeline()
        print("Pipeline completed successfully!")
        
        # Check the output
        output_dir = Path(config['data']['output_path'])
        if output_dir.exists():
            files = list(output_dir.glob('*'))
            print(f"Generated {len(files)} files in {output_dir}")
            
            # Count different file types
            masks = len(list(output_dir.glob('*_mask.png')))
            parsed = len(list(output_dir.glob('*_parsed.png')))
            
            print(f"Generated {masks} mask files and {parsed} parsed image files")
            
            if parsed > 0:
                print("✅ Parsed images are being generated successfully!")
                
                # Look at a sample parsed image
                sample_parsed = list(output_dir.glob('*_parsed.png'))[0]
                img = cv2.imread(str(sample_parsed), cv2.IMREAD_GRAYSCALE)
                non_zero = np.count_nonzero(img)
                print(f"Sample parsed image '{sample_parsed.name}' has {non_zero} non-zero pixels")
                
                if non_zero > 50:  # Reasonable threshold
                    print("✅ Parsed images have good content!")
                else:
                    print("⚠️ Parsed images might be too sparse")
                    
            else:
                print("❌ No parsed images generated")
        else:
            print("❌ Output directory not created")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_pipeline()
