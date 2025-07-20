"""
Final Validation Script for the Professional Bongard-LOGO Generator

This script performs an end-to-end test of the entire generation pipeline,
including:
- Unified configuration management (GeneratorConfig)
- Modular components (MetaController, Styler, PrototypeAction)
- Advanced features (textures, prototype shapes, GAN stylization hooks)
- Final data generation and saving

It generates a small, representative dataset to verify that all parts of the
system are working together correctly and producing the expected black-and-white
Bongard problems.
"""

import os
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# It's good practice to put imports after logging configuration
# to ensure they don't interfere with it.
try:
    from src.bongard_generator.config import GeneratorConfig
    from src.bongard_generator.dataset import generate_and_save_dataset
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    logging.error(
        "Please ensure you are running this script from the 'BongordSolver' root directory."
    )
    exit(1)


def create_validation_config() -> GeneratorConfig:
    """
    Creates a comprehensive configuration to test all major features of the
    generator pipeline.
    """
    logging.info("Creating validation configuration...")
    return GeneratorConfig(
        img_size=256,
        min_shapes=2,
        max_shapes=4,
        
        # --- Test Advanced Features ---
        # Enable textures (randomly chosen per scene later)
        bg_texture="checker",
        noise_level=0.15,
        noise_opacity=0.4,
        checker_size=25,
        
        # Enable prototype shapes.
        # Point to a plausible directory. If it doesn't exist, the system
        # will gracefully fall back to procedural shapes.
        prototype_path="data/shapebongordV2",

        # Enable GAN stylization hook (will be skipped if model not found)
        use_gan_stylization=True,
        gan_model_path="checkpoints/gan_generator_placeholder.pth",

        # Enable the adaptive rule selection
        use_meta_controller=True,
        rule_paths=["src/bongard_generator/rules"],
        
        # --- Standard Settings ---
        enable_jitter=True,
        jitter_strength=0.03,
        enable_rotation=True,
        fill_type='solid' # Keep shapes solid for clarity
    )


def run_validation():
    """
    Executes the end-to-end validation test.
    """
    logging.info("Starting final validation of the Bongard-LOGO generator.")
    
    # 1. Configuration
    config = create_validation_config()
    
    # 2. Setup Parameters
    num_problems_to_generate = 5  # Small number for a quick test
    output_directory = "validation_output"
    
    logging.info(f"Generator will create {num_problems_to_generate} problems.")
    logging.info(f"Output will be saved to: {output_directory}")

    # Clean up previous validation runs
    if os.path.exists(output_directory):
        logging.warning(f"Removing existing validation directory: {output_directory}")
        shutil.rmtree(output_directory)
    
    # 3. Run the Dataset Generation
    try:
        logging.info("Starting dataset generation...")
        generate_and_save_dataset(
            config=config,
            num_problems=num_problems_to_generate,
            save_dir=output_directory
        )
        logging.info("Dataset generation completed successfully.")
        
    except Exception as e:
        logging.error(f"An error occurred during dataset generation: {e}", exc_info=True)
        logging.error("Validation failed.")
        return

    # 4. Final Verification
    logging.info("-" * 50)
    logging.info("Final Validation Summary:")
    
    if os.path.exists(output_directory):
        generated_folders = [f for f in os.listdir(output_directory) if os.path.isdir(os.path.join(output_directory, f))]
        if len(generated_folders) == num_problems_to_generate:
            logging.info(f"✅ Success: Correct number of problem folders generated ({len(generated_folders)}).")
            
            # Check content of the first problem folder
            first_problem_dir = os.path.join(output_directory, generated_folders[0])
            files = os.listdir(first_problem_dir)
            if "positive_0.png" in files and "negative_0.png" in files and "rule.txt" in files:
                logging.info(f"✅ Success: First problem folder '{generated_folders[0]}' contains expected files.")
            else:
                logging.error(f"❌ Failure: First problem folder is missing expected files.")
        else:
            logging.error(f"❌ Failure: Incorrect number of problem folders generated. Expected {num_problems_to_generate}, found {len(generated_folders)}.")
    else:
        logging.error(f"❌ Failure: Output directory '{output_directory}' was not created.")

    logging.info("-" * 50)
    logging.info("Validation script finished. Please review the images in the 'validation_output' directory.")


if __name__ == "__main__":
    run_validation()
