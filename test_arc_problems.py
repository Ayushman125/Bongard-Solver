#!/usr/bin/env python3
"""
Test the improved parsing pipeline with arc-containing problems
"""

import os
import sys
sys.path.append('src')

from bongard_augmentor.hybrid import HybridAugmentationPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_arc_problems():
    """Test pipeline with problems that contain arc commands"""
    
    print("Testing improved parsing pipeline with arc-containing problems...")
    
    # Configuration for testing with arc problems
    config = {
        'sam_ensemble': ['vit_h'],
        'sam_checkpoint_dir': 'sam_checkpoints',
        'action_programs_dir': 'data/raw/ShapeBongard_V2/bd/',  # Directory not file
        'output_base_dir': 'data',
        'image_size': 64
    }
    
    # Initialize pipeline
    pipeline = HybridAugmentationPipeline(config)
    
    # Process only arc-containing problems for testing
    arc_problems = [
        'bd_sector30_0000',
        'bd_inverse_sector30_0000', 
        'bd_sector90_0000',
        'bd_closed_semi_circle_0000',
        'bd_open_semi_circle_0000'
    ]
    
    print(f"Processing {len(arc_problems)} arc-containing problems...")
    
    # Generate augmented data for these specific problems
    output_file = os.path.join(config['output_base_dir'], 'test_arc_problems_output')
    pipeline.process_action_programs(
        limit_problems=arc_problems,
        output_file=output_file
    )
    
    print("Arc problem processing completed!")
    
    # Check output files
    data_dir = config['output_base_dir']
    parsed_files = [f for f in os.listdir(data_dir) if f.endswith('_parsed.png')]
    mask_files = [f for f in os.listdir(data_dir) if f.endswith('_mask.png')]
    
    print(f"\nGenerated {len(parsed_files)} parsed image files")
    print(f"Generated {len(mask_files)} mask files")
    
    # Show some examples
    arc_parsed_files = [f for f in parsed_files if any(prob in f for prob in arc_problems)]
    print(f"\nArc problem parsed images: {len(arc_parsed_files)}")
    for f in arc_parsed_files[:5]:
        print(f"  {f}")

if __name__ == "__main__":
    test_arc_problems()
