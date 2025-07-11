# Folder: scripts/
# File: ci_report_metrics.py
import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def summarize_ci_metrics(log_dir: str, out_file: str):
    """
    Loads various metric logs from a specified directory, extracts key metrics,
    and summarizes them into a single JSON file.

    Args:
        log_dir (str): The directory containing the log files (hpo.json, metrics.json, diversity.json).
        out_file (str): The path to the output JSON file for the summarized report.
    """
    logger.info(f"Summarizing CI metrics from {log_dir} to {out_file}")
    
    summary_report: Dict[str, Any] = {}

    # 1. Load HPO best parameters
    hpo_path = os.path.join(log_dir, "best_hpo_config_optuna.json") # Assuming Optuna HPO output
    if os.path.exists(hpo_path):
        with open(hpo_path, 'r') as f:
            hpo_data = json.load(f)
            summary_report['hpo_best_params'] = hpo_data
            logger.info(f"Loaded HPO best parameters from {hpo_path}")
    else:
        logger.warning(f"HPO results not found at {hpo_path}. Skipping HPO summary.")
        summary_report['hpo_best_params'] = "N/A"

    # 2. Load benchmark_report.json (which now contains bongard_acc, relation_map, mAP, diversity)
    benchmark_report_path = os.path.join(log_dir, "benchmark_report.json")
    if os.path.exists(benchmark_report_path):
        with open(benchmark_report_path, 'r') as f:
            benchmark_data = json.load(f)
            
            # Extract bongard_acc
            summary_report['bongard_accuracy'] = benchmark_data.get('bongard_accuracy', 'N/A')
            
            # Extract relation_map
            summary_report['relation_map'] = benchmark_data.get('relation_map', 'N/A')
            
            # Extract mAP (Object Detection mAP)
            summary_report['mAP'] = benchmark_data.get('mAP', 'N/A')
            
            # Extract diversity metrics
            diversity_data = benchmark_data.get('diversity', {})
            summary_report['diversity'] = {
                'entropy': diversity_data.get('entropy', 'N/A'),
                'disagreement': diversity_data.get('disagreement', 'N/A')
            }
            logger.info(f"Loaded and extracted metrics from {benchmark_report_path}")
    else:
        logger.warning(f"Benchmark report not found at {benchmark_report_path}. Skipping benchmark metrics.")
        summary_report['bongard_accuracy'] = "N/A"
        summary_report['relation_map'] = "N/A"
        summary_report['mAP'] = "N/A"
        summary_report['diversity'] = {'entropy': "N/A", 'disagreement': "N/A"}

    # 3. Placeholder for prune/quantize metrics (assuming separate logs or part of benchmark_report)
    # If prune_size and quant_drop are logged elsewhere, load them here.
    # For now, let's assume they might be in a 'pruning_quantization_report.json'
    prune_quant_report_path = os.path.join(log_dir, "pruning_quantization_report.json")
    if os.path.exists(prune_quant_report_path):
        with open(prune_quant_report_path, 'r') as f:
            prune_quant_data = json.load(f)
            summary_report['prune_size_reduction_ratio'] = prune_quant_data.get('prune_size_reduction_ratio', 'N/A')
            summary_report['quantization_accuracy_drop'] = prune_quant_data.get('quantization_accuracy_drop', 'N/A')
            logger.info(f"Loaded pruning and quantization metrics from {prune_quant_report_path}")
    else:
        logger.warning(f"Pruning/Quantization report not found at {prune_quant_report_path}. Skipping these metrics.")
        summary_report['prune_size_reduction_ratio'] = "N/A"
        summary_report['quantization_accuracy_drop'] = "N/A"


    # Dump the summarized report to the output file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(summary_report, f, indent=4)
    logger.info(f"CI metrics summary saved to: {out_file}")

def main():
    """
    Main entry point for the CI metrics summarization script.
    Parses command-line arguments for log directory and output file.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Summarize CI metrics from log directory.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Path to the directory containing HPO, benchmark, and other metric logs.")
    parser.add_argument("--out_file", type=str, default="ci_summary_report.json",
                        help="Path to the output JSON file for the summarized report.")
    args = parser.parse_args()
    
    summarize_ci_metrics(args.log_dir, args.out_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

