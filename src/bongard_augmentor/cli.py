import argparse
import logging
from .hybrid import HybridAugmentor
from .utils import setup_logging, get_base_config
from src.data_pipeline.logo_parser import UnifiedActionParser
from src.data_pipeline.logo_parser import ComprehensiveNVLabsParser as NVLabsActionParser

def main():
    """Main entry point for the action-based augmentation script."""
    parser = argparse.ArgumentParser(description="Bongard Problem Action-Based Augmentation")
    
    # Input options - either derived_labels.json or action programs directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Path to derived_labels.json (from logo_to_shape.py)')
    input_group.add_argument('--action-programs-dir', type=str, help='Path to action programs directory')
    
    # Output and processing options
    parser.add_argument('--out', type=str, required=True, help='Path to output augmented.pkl')
    parser.add_argument('--problems-list', type=str, help='Optional file containing list of problems to process')
    parser.add_argument('--n-select', type=int, default=50, help='Number of problems to select from action programs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--image-size', type=int, nargs=2, default=[64, 64], help='Image size for mask generation (height width)')
    parser.add_argument('--enable-post-processing', action='store_true', help='Enable post-processing of generated masks')
    parser.add_argument('--inspection-dir', type=str, help='Directory to save inspection images and action files')
    parser.add_argument("--use-nvlabs", action="store_true", help="Use NVLabs canonical parser")
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (e.g., DEBUG, INFO, WARNING)')
    parser.add_argument('--log-file', type=str, default='logs/augmentor.log', help='Path to the log file.')
    
    args = parser.parse_args()

    # 1. Setup Logging
    setup_logging(args.log_level, args.log_file)
    log = logging.getLogger(__name__)

    # 2. Create Configuration
    log.info("Loading base configuration.")
    config = get_base_config()

    # Update config with CLI arguments
    log.info("Updating configuration with action-based processing arguments.")
    config['data']['output_path'] = args.out
    config['processing']['batch_size'] = args.batch_size
    config['image_size'] = tuple(args.image_size)
    config['enable_post_processing'] = args.enable_post_processing
    
    # Set input source
    if args.input:
        config['data']['input_path'] = args.input
        log.info(f"Using derived_labels.json input: {args.input}")
    elif args.action_programs_dir:
        config['data']['action_programs_dir'] = args.action_programs_dir
        if args.problems_list:
            config['data']['problems_list'] = args.problems_list
        config['data']['n_select'] = args.n_select
        log.info(f"Using action programs directory: {args.action_programs_dir}")
    
    # Optional inspection directory
    if args.inspection_dir:
        config['inspection_dir'] = args.inspection_dir

    log.info(f"Final configuration loaded for action-based pipeline.")
    log.debug(f"Config: {config}")

    # Select parser
    if args.use_nvlabs:
        log.info("Using NVLabs canonical parser")
        action_parser = NVLabsActionParser()
    else:
        log.info("Using custom UnifiedActionParser")
        action_parser = UnifiedActionParser()

    # 3. Instantiate and Run the Pipeline
    log.info("Initializing Action-Based Augmentation Pipeline.")
    try:
        augmentor = HybridAugmentor(config)
        augmentor.run_pipeline()
        log.info(f"Action-based augmentation complete. Results saved to {config['data']['output_path']}")
    except Exception as e:
        log.critical(f"An unrecoverable error occurred in the pipeline: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()