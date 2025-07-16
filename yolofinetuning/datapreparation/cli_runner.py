import typer
import yaml
from pathlib import Path
import logging
import os
import argparse # Needed to create Namespace object for process_dataset

# Import necessary functions and classes from your project modules
# Adjust these imports based on your actual file structure
try:
    from .data_preparation_utils import process_dataset
    # Import specific helper functions that might be called directly by CLI subcommands
    from .data_preparation_utils import (
        collect_stats, # Assuming this is now in data_preparation_utils
        parse_classes, # Needed for visualize_aug to get class names
    )
    from .augmentations import visualize_augmentations # For 'visualize_aug' command
except ImportError as e:
    logging.critical(f"Core modules not found: {e}. Please ensure PYTHONPATH is set correctly and all dependencies are installed.")
    process_dataset = None
    collect_stats = None
    parse_classes = None
    visualize_augmentations = None

# Optional: MLflow for logging
try:
    import mlflow
except ImportError:
    mlflow = None
    logging.warning("MLflow not installed. MLflow logging will be skipped.")


app = typer.Typer()

def load_config(path: Path):
    """Loads a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

@app.command(name="run_pipeline")
def run_pipeline(
    config_path: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml (optional, overrides default)"),
    # Model parameters
    owlvit_model: str = typer.Option(None, "--owlvit-model", help="Override OWL-ViT model name (e.g., 'google/owlvit-base-patch32')"),
    min_score: float = typer.Option(None, "--min-score", help="Override detection threshold for models"),
    device: str = typer.Option(None, "--device", help="Override device for models (e.g., 'cuda', 'cpu')"),
    prompts: list[str] = typer.Option(None, "--prompts", help="Override detection prompts for OWL-ViT (e.g., 'a photo of a person', 'a car')"),
    class_map_path: Path = typer.Option(None, "--class-map", help="Path to JSON file for OWL-ViT label to dataset class ID mapping"),

    # Path parameters
    input_dir: Path = typer.Option(None, "--input-dir", "-i", help="Override input directory with raw images"),
    output_dir: Path = typer.Option(None, "--output-dir", "-o", help="Override output directory for YOLO dataset"),
    annotations: Path = typer.Option(None, "--annotations", "-a", help="Override path to annotation file (JSON, CSV, or XML)"),
    classes_file: Path = typer.Option(None, "--classes-file", help="Override path to class names file"),
    raw_labels_output_dir: Path = typer.Option(None, "--raw-labels-output-dir", help="Override output directory for initial auto-generated labels"),

    # Split parameters
    split_ratio: float = typer.Option(None, "--split-ratio", help="Override train/val split ratio"),
    
    # General parameters
    seed: int = typer.Option(None, "--seed", help="Override random seed"),
    num_workers: int = typer.Option(None, "--num-workers", help="Override number of workers for multiprocessing"),

    # Feature flags (boolean overrides)
    use_owlvit: bool = typer.Option(None, "--use-owlvit/--no-use-owlvit", help="Enable/Disable OWL-ViT detection pipeline for initial labeling"),
    auto_labeling_enabled: bool = typer.Option(None, "--auto-labeling/--no-auto-labeling", help="Enable/Disable initial auto-labeling"),
    label_correction_enabled: bool = typer.Option(None, "--label-correction/--no-label-correction", help="Enable/Disable label correction"),
    class_balancing_enabled: bool = typer.Option(None, "--class-balancing/--no-class-balancing", help="Enable/Disable class balancing (oversampling)"),
    smote_enabled: bool = typer.Option(None, "--smote/--no-smote", help="Enable/Disable SMOTE oversampling"),
    curriculum_learning_enabled: bool = typer.Option(None, "--curriculum-learning/--no-curriculum-learning", help="Enable/Disable curriculum learning"),
    hard_negative_mining_enabled: bool = typer.Option(None, "--hard-negative-mining/--no-hard-negative-mining", help="Enable/Disable hard negative mining"),
    synthetic_data_enabled: bool = typer.Option(None, "--synthetic-data/--no-synthetic-data", help="Enable/Disable synthetic data generation"),
    test_time_augmentation_enabled: bool = typer.Option(None, "--tta/--no-tta", help="Enable/Disable Test-Time Augmentation"),
    error_analysis_enabled: bool = typer.Option(None, "--error-analysis/--no-error-analysis", help="Enable/Disable error analysis"),
    visualization_enabled: bool = typer.Option(None, "--visualization/--no-visualization", help="Enable/Disable visualization generation"),
    mlflow_enabled: bool = typer.Option(None, "--mlflow/--no-mlflow", help="Enable/Disable MLflow tracking"),
    dali_enabled: bool = typer.Option(None, "--dali/--no-dali", help="Enable/Disable DALI pipeline integration"),

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
    # This ensures CLI args take precedence over config file
    
    # Model Config
    cfg.setdefault('model', {})
    if owlvit_model is not None: cfg['model']['name'] = owlvit_model
    if min_score is not None: cfg['model']['detection_threshold'] = min_score
    if device is not None: cfg['model']['device'] = device
    if prompts is not None: cfg['model']['detection_prompts'] = prompts
    if class_map_path is not None: cfg['model']['class_map_path'] = str(class_map_path)

    # Paths Config
    cfg.setdefault('paths', {})
    if input_dir is not None: cfg['paths']['raw_images'] = str(input_dir)
    if output_dir is not None: cfg['paths']['output_dir'] = str(output_dir)
    if annotations is not None: cfg['paths']['annotations'] = str(annotations)
    if classes_file is not None: cfg['paths']['classes_file'] = str(classes_file)
    if raw_labels_output_dir is not None: cfg['paths']['raw_labels_output'] = str(raw_labels_output_dir)

    # Split Config
    cfg.setdefault('split', {})
    if split_ratio is not None: cfg['split']['train_size'] = split_ratio
    
    # General Config
    if seed is not None: cfg['seed'] = seed
    if num_workers is not None: cfg['num_workers'] = num_workers

    # Feature Flags (nested under their respective sections)
    cfg.setdefault('auto_labeling', {})
    if auto_labeling_enabled is not None: cfg['auto_labeling']['enabled'] = auto_labeling_enabled
    if use_owlvit is not None: cfg['auto_labeling']['enabled'] = use_owlvit # `use_owlvit` is now an alias for auto_labeling enabled

    cfg.setdefault('label_correction', {})
    if label_correction_enabled is not None: cfg['label_correction']['enable'] = label_correction_enabled

    cfg.setdefault('class_balancing', {})
    if class_balancing_enabled is not None: cfg['class_balancing']['enable'] = class_balancing_enabled
    cfg.setdefault('smote', {})
    if smote_enabled is not None: cfg['smote']['enable'] = smote_enabled

    cfg.setdefault('curriculum_learning', {})
    if curriculum_learning_enabled is not None: cfg['curriculum_learning']['enable'] = curriculum_learning_enabled

    cfg.setdefault('hard_negative_mining', {})
    if hard_negative_mining_enabled is not None: cfg['hard_negative_mining']['enable'] = hard_negative_mining_enabled

    cfg.setdefault('synthetic_data', {})
    if synthetic_data_enabled is not None: cfg['synthetic_data']['enabled'] = synthetic_data_enabled

    cfg.setdefault('test_time_augmentation', {})
    if test_time_augmentation_enabled is not None: cfg['test_time_augmentation']['enable'] = test_time_augmentation_enabled

    cfg.setdefault('error_analysis', {})
    if error_analysis_enabled is not None: cfg['error_analysis']['enable'] = error_analysis_enabled

    cfg.setdefault('visualization', {})
    if visualization_enabled is not None: cfg['visualization']['enable'] = visualization_enabled

    cfg.setdefault('mlflow', {})
    if mlflow_enabled is not None: cfg['mlflow']['enable'] = mlflow_enabled

    cfg.setdefault('dali', {})
    if dali_enabled is not None: cfg['dali']['enabled'] = dali_enabled


    # Set up logging based on the (potentially updated) config
    setup_logging(cfg.get('logging', {}))

    logging.info("Starting dataset preparation pipeline...")
    logging.info(f"Effective Configuration: {yaml.dump(cfg, indent=2)}") # Log the effective config

    # Create a dummy argparse.Namespace object to pass to process_dataset
    # This bridges the Typer CLI arguments to the existing process_dataset function
    # which expects an argparse.Namespace object.
    # Populate with values from the final `cfg` dictionary
    cli_args_for_process_dataset = argparse.Namespace(
        input_dir=cfg['paths'].get('raw_images'),
        output_dir=cfg['paths'].get('output_dir'),
        annotations=cfg['paths'].get('annotations'),
        split_ratio=cfg['split'].get('train_size'),
        classes_file=cfg['paths'].get('classes_file'),
        seed=cfg.get('seed'),
        config=config_path, # Pass the original config path if available
        use_owlvit=cfg['auto_labeling'].get('enabled', False), # Use the unified flag
        device=cfg['model'].get('device'),
        num_workers=cfg.get('num_workers'),
        # Pass other relevant CLI args that process_dataset might expect
        owlvit_model=cfg['model'].get('name'),
        min_score=cfg['model'].get('detection_threshold'),
        prompts=cfg['model'].get('detection_prompts'),
        class_map_path=cfg['model'].get('class_map_path'),
        raw_labels_output_dir=cfg['paths'].get('raw_labels_output')
    )

    # Run the main dataset processing logic
    if process_dataset:
        process_dataset(cli_args_for_process_dataset)
    else:
        logging.critical("`process_dataset` function is not available. Cannot run pipeline.")


    logging.info("Dataset preparation pipeline finished.")


@app.command(name="visualize_aug")
def visualize_aug_cli(
    img: Path = typer.Argument(..., help="Path to image"),
    lbl: Path = typer.Argument(..., help="Path to label txt"),
    out: Path = typer.Option("aug_preview", "--out", "-o", help="Output directory for preview images"),
    n: int = typer.Option(5, "--num-samples", "-n", help="Number of augmented samples to visualize"),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml (optional, for augmentation settings)"),
):
    """
    Visualize augmentations on a sample image and save the results.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    cfg_for_aug = None
    if config and config.exists():
        try:
            cfg_for_aug = load_config(config)
            logging.info(f"Loaded augmentation config from {config}")
        except Exception as e:
            logging.error(f"Error loading config for visualization: {e}. Proceeding without config.")

    if visualize_augmentations:
        visualize_augmentations(str(img), str(lbl), str(out), n, cfg=cfg_for_aug)
        logging.info(f"Augmentation visualizations saved to {out}")
    else:
        logging.critical("`visualize_augmentations` function not available. Cannot run visualization.")


@app.command(name="collect_stats")
def collect_stats_cli(
    data_dir: Path = typer.Argument(..., help="Root directory of the processed dataset (e.g., output_dir)"),
    output_prefix: str = typer.Option("metadata", "--output-prefix", help="Prefix for output metadata files (JSON/CSV)"),
):
    """
    Collect and log dataset statistics from a processed dataset directory.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if collect_stats:
        collect_stats(str(data_dir), os.path.join(str(data_dir), output_prefix))
        logging.info(f"Dataset statistics collected and saved with prefix {output_prefix} in {data_dir}")
    else:
        logging.critical("`collect_stats` function not available. Cannot collect statistics.")


if __name__ == "__main__":
    app()
