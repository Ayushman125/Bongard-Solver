"""
Final Validation Script for the Unified Bongard-LOGO Generator

This script performs an end-to-end test of the unified generation pipeline,
driven by the master BongardGenerator class.
"""
import os
import logging
import shutil
from bongard_generator.builder import BongardGenerator
from bongard_rules import get_all_rules, BongardRule
from config import load_config, CONFIG
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def save_scenes(rule_name: str, scenes: list, base_dir: str):
    """Saves generated scenes to a directory structure."""
    rule_dir = os.path.join(base_dir, rule_name)
    os.makedirs(rule_dir, exist_ok=True)
    
    for i, (img, rule, tag) in enumerate(scenes):
        # Ensure image is in a savable format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Sanitize tag for filename
        safe_tag = tag.replace(" ", "_")
        
        # Construct filename
        filename = f"{i:03d}_{safe_tag}.png"
        filepath = os.path.join(rule_dir, filename)
        
        # Save the image
        img.save(filepath)

    logging.info(f"Saved {len(scenes)} scenes for rule '{rule_name}' in '{rule_dir}'")


def run_unified_validation():
    """
    Executes the end-to-end validation test for the unified generator.
    """
    logging.info("Starting unified validation of the Bongard-LOGO generator.")
    
    # 1. Load configuration from config.py and config.yaml
    cfg = CONFIG
    
    # 2. Instantiate the master generator
    try:
        gen = BongardGenerator(cfg)
    except Exception as e:
        logging.error(f"Failed to initialize BongardGenerator: {e}", exc_info=True)
        return

    # 3. Setup Parameters
    all_rules = get_all_rules()
    output_directory = "unified_validation_output"
    
    logging.info(f"Found {len(all_rules)} rules to process.")
    logging.info(f"Output will be saved to: {output_directory}")

    # Clean up previous validation runs
    if os.path.exists(output_directory):
        logging.warning(f"Removing existing validation directory: {output_directory}")
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    # 4. Run Generation for each rule
    for rule in all_rules:
        if not isinstance(rule, BongardRule):
            logging.warning(f"Skipping invalid rule object: {rule}")
            continue
            
        logging.info(f"--- Generating for rule: {rule.name} ---")
        try:
            scenes = gen.generate_for_rule(rule, n_scenes=50)
            if scenes:
                save_scenes(rule.name, scenes, output_directory)
            else:
                logging.warning(f"No scenes were generated for rule: {rule.name}")
        except Exception as e:
            logging.error(f"An error occurred during generation for rule {rule.name}: {e}", exc_info=True)

    # 5. Final Verification and Coverage Report
    logging.info("-" * 50)
    logging.info("Final Validation Summary:")
    
    if os.path.exists(output_directory):
        generated_folders = [f for f in os.listdir(output_directory) if os.path.isdir(os.path.join(output_directory, f))]
        if generated_folders:
            logging.info(f"✅ Success: {len(generated_folders)} rule folders generated.")
        else:
            logging.error("❌ Failure: No rule folders were generated.")
    else:
        logging.error(f"❌ Failure: Output directory '{output_directory}' was not created.")

    # Print coverage report
    gen.get_coverage_report()

    logging.info("-" * 50)
    logging.info("Unified validation script finished. Please review the images in the 'unified_validation_output' directory.")


if __name__ == "__main__":
    run_unified_validation()
