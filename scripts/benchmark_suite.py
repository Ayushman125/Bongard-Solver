# Folder: scripts/
# File: benchmark_suite.py
import json
import argparse
import torch
import os
import sys
import logging
from typing import Dict, Any

# Add the parent directory to the Python path to import modules from bongard_solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bongard_solver')))

# Assuming metrics.py and config.py are in the bongard_solver directory
from metrics import compute_bongard_accuracy, compute_relation_map # Ensure these functions are defined
from config import load_config, DEVICE # Assuming load_config and DEVICE are available

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run automated benchmark suite for Bongard Solver models.")
    parser.add_argument("--models", nargs='+', required=True,
                        help="List of paths to model checkpoint files (e.g., best.pt, other.pt).")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    parser.add_argument("--output_file", type=str, default="benchmarks.json",
                        help="Path to save the benchmark results JSON file.")
    args = parser.parse_args()

    # Load the base configuration
    cfg = load_config(args.config)

    results = {}
    for model_path in args.models:
        logger.info(f"--- Benchmarking model: {model_path} ---")
        model_name = os.path.basename(model_path)
        
        try:
            # Load the model. This might need to be adapted based on how your models are saved.
            # Assuming it's a state_dict of PerceptionModule or LitBongard.
            # You might need to instantiate the model class first.
            from models import PerceptionModule # Import your model class
            model = PerceptionModule(cfg).to(DEVICE)
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if 'state_dict' in checkpoint: # For Lightning checkpoints
                model_state_dict = {k.replace('perception_module.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('perception_module.')}
                model.load_state_dict(model_state_dict)
            else: # For raw state_dict
                model.load_state_dict(checkpoint)
            model.eval() # Set model to evaluation mode

            # Compute metrics
            # These functions (compute_bongard_accuracy, compute_relation_map)
            # need to be implemented in metrics.py and accept the model and config.
            bongard_acc = compute_bongard_accuracy(model, cfg)
            relation_map = compute_relation_map(model, cfg)
            
            results[model_name] = {
                "bongard_accuracy": bongard_acc,
                "relation_map": relation_map
            }
            logger.info(f"Results for {model_name}: Bongard Accuracy={bongard_acc:.4f}, Relation mAP={relation_map:.4f}")

        except FileNotFoundError:
            logger.error(f"Model checkpoint not found at {model_path}. Skipping.")
            results[model_name] = {"error": "Model file not found"}
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Benchmark results written to {args.output_file}")

if __name__ == "__main__":
    main()
