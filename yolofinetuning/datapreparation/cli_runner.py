
# Typer CLI for research-grade data prep
import typer
import yaml
from pathlib import Path
from embedding_model import EmbedDetector
from dali_pipeline import prepare_dataset
from metrics import Evaluator
from logger import setup_logging
from data_preparation_utils import parse_args, process_dataset

app = typer.Typer()

def load_config(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)

@app.command()
def main(
    config_path: Path = typer.Option(None, help="Path to config.yaml (optional, overrides default)"),
    model_name: str = typer.Option(None, "--model-name", help="Override model.name"),
    threshold: float = typer.Option(None, "--threshold", help="Override detection threshold"),
    use_owlvit: bool = typer.Option(False, help="Use OWL-ViT detection pipeline (default: False)"),
    # Add more CLI overrides as needed
):
    # If config_path is provided, load YAML config, else fallback to argparse
    if config_path:
        cfg = load_config(config_path)
        if model_name:    cfg['model']['name'] = model_name
        if threshold:     cfg['model']['detection_threshold'] = threshold
        setup_logging(cfg['logging'])
        if use_owlvit:
            detector = EmbedDetector(cfg['model'])
            evaluator = Evaluator(cfg['metrics'])
            prepare_dataset(cfg['data'], detector, evaluator)
        else:
            # Fallback to classic YOLO prep
            args = parse_args()
            process_dataset(args)
    else:
        # No config: fallback to argparse for classic YOLO prep
        args = parse_args()
        process_dataset(args)

if __name__ == "__main__":
    app()
