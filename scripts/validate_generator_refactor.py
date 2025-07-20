import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import logging
from src.bongard_generator.config import GeneratorConfig
from src.bongard_generator.dataset import generate_and_save_dataset

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to configure and run the refactored Bongard problem generator.
    """
    logging.info("Starting validation of the refactored Bongard generator...")

    # 1. Define the output directory
    save_dir = "refactor_validation_output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.info(f"Output will be saved to: {os.path.abspath(save_dir)}")

    # 2. Create a GeneratorConfig instance
    # This config uses the new features: real prototypes and a placeholder for textures.
    config = GeneratorConfig(
        img_size=(256, 256),
        min_objects=2,
        max_objects=4,
        prototype_path="ShapeBongard_V2/",  # Point to the directory with real shape prototypes
        use_advanced_textures=False, # Keep false unless you have a texture library
        texture_library_path="textures/", # Placeholder path
        use_meta_controller=True,
        coverage_goals={
            "default_quota": 5,
            "priority_quota": 10,
            "priority_cells": [('shape', 'triangle')]
        }
    )
    logging.info(f"Generator configured with prototype path: {config.prototype_path}")

    # 3. Define the number of problems to generate for this test
    num_problems_to_generate = 5

    # 4. Run the generation process
    logging.info(f"Generating {num_problems_to_generate} problems...")
    try:
        generate_and_save_dataset(
            config=config,
            num_problems=num_problems_to_generate,
            save_dir=save_dir
        )
        logging.info("Validation script finished successfully.")
        logging.info(f"Please check the '{save_dir}' directory for the generated Bongard problems.")
    except Exception as e:
        logging.error(f"An error occurred during dataset generation: {e}", exc_info=True)

if __name__ == "__main__":
    main()
