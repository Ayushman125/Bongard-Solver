import argparse
import logging
from .hybrid import HybridAugmentor
from .utils import setup_logging, get_base_config

def main():
    """Main entry point for the augmentation script."""
    parser = argparse.ArgumentParser(description="Bongard Problem Image Augmentation")
    parser.add_argument('--input', type=str, required=True, help='Path to derived_labels.json')
    parser.add_argument('--out', type=str, required=True, help='Path to output augmented.pkl')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--type', type=str, default='geometric', choices=['geometric', 'photometric', 'both'], help='Type of augmentation')
    parser.add_argument('--sam-model', type=str, default='vit_h', help='SAM model type (e.g., vit_h, vit_l, vit_b)')
    parser.add_argument('--qa-fail-threshold', type=float, default=0.15, help='QA failure rate threshold to trigger fallback')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (e.g., DEBUG, INFO, WARNING)')
    parser.add_argument('--log-file', type=str, default='logs/augmentor.log', help='Path to the log file.')
    # Deprecated/handled by config:
    # --parallel, --rotate, --scale, --enable-hybrid, --test-corruption-fixes, --force-emergency-qa, --fallback-empty
    args = parser.parse_args()

    # 1. Setup Logging
    setup_logging(args.log_level, args.log_file)
    log = logging.getLogger(__name__)

    # 2. Create Configuration
    log.info("Loading base configuration.")
    config = get_base_config()

    # Update config with CLI arguments
    log.info("Updating configuration with CLI arguments.")
    config['data']['input_path'] = args.input
    config['data']['output_path'] = args.out
    config['processing']['batch_size'] = args.batch_size
    config['augmentation']['type'] = args.type
    config['sam']['model_type'] = args.sam_model
    config['qa']['failure_rate_threshold'] = args.qa_fail_threshold

    log.info(f"Final configuration loaded for pipeline.")
    log.debug(f"Config: {config}")

    # 3. Instantiate and Run the Pipeline
    log.info("Initializing Bongard Hybrid Augmentation Pipeline.")
    try:
        augmentor = HybridAugmentor(config)
        augmentor.run_pipeline()
        log.info(f"Augmentation complete. Results saved to {config['data']['output_path']}")
    except Exception as e:
        log.critical(f"An unrecoverable error occurred in the pipeline: {e}", exc_info=True)
        # In a real scenario, you might want to perform cleanup here

if __name__ == "__main__":
    main()
