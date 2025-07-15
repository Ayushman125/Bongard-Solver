import typer
import yaml
from pathlib import Path
import logging
import os

# Import necessary functions and classes from your project modules
# Adjust these imports based on your actual file structure
try:
    from .embedding_model import EmbedDetector # If EmbedDetector is in its own file
except ImportError:
    # Fallback if EmbedDetector is in auto_labeling.py
    try:
        from .auto_labeling import EmbedDetector
    except ImportError:
        EmbedDetector = None
        logging.warning("EmbedDetector not found. OWL-ViT functionality will be disabled.")

from .data_preparation_utils import parse_args, process_dataset
from .metrics import Evaluator # Assuming Evaluator is in metrics.py
from .logger import setup_logging # Assuming setup_logging is in logger.py

# Assuming dali_pipeline.py exists and has prepare_dataset
try:
    from .dali_pipeline import prepare_dataset
except ImportError:
    prepare_dataset = None
    logging.warning("dali_pipeline module not found. DALI-based dataset preparation will be unavailable.")


app = typer.Typer()

def load_config(path: Path):
    """Loads a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

@app.command()
def main(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml (optional, overrides default)"),
    model_name: str = typer.Option(None, "--model-name", help="Override model.name from config"),
    threshold: float = typer.Option(None, "--threshold", help="Override detection threshold from config"),
    use_owlvit: bool = typer.Option(False, "--use-owlvit", help="Use OWL-ViT detection pipeline for initial labeling"),
    device: str = typer.Option("cpu", "--device", help="Device to run OWL-ViT on (e.g., 'cuda', 'cpu')"),
    input_dir: Path = typer.Option(None, "--input-dir", "-i", help="Input directory with raw images (overrides config)"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Output directory for YOLO dataset (overrides config)"),
    annotations: Path = typer.Option(None, "--annotations", "-a", help="Path to annotation file (JSON, CSV, or XML) (overrides config)"),
    split_ratio: float = typer.Option(None, "--split-ratio", help="Train/val split ratio (overrides config)"),
    classes_file: Path = typer.Option(None, "--classes-file", help="Path to class names file (overrides config)"),
    seed: int = typer.Option(None, "--seed", help="Random seed (overrides config)"),
    num_workers: int = typer.Option(4, "--num-workers", help="Number of workers for multiprocessing (e.g., augmentation)"),
):
    """
    Main entry point for the dataset preparation pipeline.
    Loads configuration, initializes models, and runs the data processing.
    """
    # Configure basic logging at the start
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    cfg = {}
    if config_path:
        try:
            cfg = load_config(config_path)
            logging.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logging.error(f"Config file not found at {config_path}. Proceeding with CLI defaults/empty config.")
        except Exception as e:
            logging.error(f"Error loading config from {config_path}: {e}. Proceeding with CLI defaults/empty config.")

    # Override config values with CLI arguments if provided
    if model_name:
        cfg.setdefault('model', {})['name'] = model_name
    if threshold:
        cfg.setdefault('model', {})['detection_threshold'] = threshold
    if device:
        cfg.setdefault('model', {})['device'] = device # Ensure device is set for models
    
    # Update paths in config based on CLI args if provided
    if input_dir:
        cfg.setdefault('paths', {})['raw_images'] = str(input_dir)
    if output_dir:
        cfg.setdefault('paths', {})['output_dir'] = str(output_dir)
    if annotations:
        cfg.setdefault('paths', {})['annotations'] = str(annotations)
    if classes_file:
        cfg.setdefault('paths', {})['classes_file'] = str(classes_file)
    if split_ratio is not None:
        cfg.setdefault('split', {})['train_size'] = split_ratio
    if seed is not None:
        cfg['seed'] = seed
    
    # Set up logging based on config (if available)
    setup_logging(cfg.get('logging', {}))

    # Create a dummy argparse.Namespace object to pass to process_dataset
    # This bridges the Typer CLI arguments to the existing process_dataset function
    # which expects an argparse.Namespace object.
    cli_args_for_process_dataset = argparse.Namespace(
        input_dir=input_dir if input_dir else cfg.get('paths', {}).get('raw_images'),
        output_dir=output_dir if output_dir else cfg.get('paths', {}).get('output_dir'),
        annotations=annotations if annotations else cfg.get('paths', {}).get('annotations'),
        split_ratio=split_ratio if split_ratio is not None else cfg.get('split', {}).get('train_size', 0.8),
        classes_file=classes_file if classes_file else cfg.get('paths', {}).get('classes_file'),
        seed=seed if seed is not None else cfg.get('seed', 42),
        config=config_path, # Pass the config path for process_dataset to load it internally
        use_owlvit=use_owlvit,
        device=device,
        num_workers=num_workers # Pass num_workers
    )

    logging.info("Starting dataset preparation pipeline...")
    logging.info(f"Effective Configuration: {cfg}") # Log the effective config

    # Run the main dataset processing logic
    process_dataset(cli_args_for_process_dataset)

    logging.info("Dataset preparation pipeline finished.")

if __name__ == "__main__":
    app()
