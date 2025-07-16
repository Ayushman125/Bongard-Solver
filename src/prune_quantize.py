# Folder: bongard_solver/
# File: prune_quantize.py
import torch
import torch.nn as nn
import logging
import os
import sys
from typing import Dict, Any, Optional
import numpy as np
import argparse

# Add the 'src' directory to the Python path to import modules from it
# This assumes prune_quantize.py is in the project root (bongard_solver/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Now imports from src/ can be done directly
from config import CONFIG, load_config, DEVICE, HAS_TORCH_QUANTIZATION
from models import PerceptionModule, LitBongard
from data import get_dataloader, build_dali_image_processor, BongardSyntheticDataset, RealBongardDataset, BongardGenerator
from training import compute_layer_sensitivity, apply_structured_pruning, quantize_model_qat, quantize_model_ptq
from bongard_rules import ALL_BONGARD_RULES

# Assume HAS_DALI is defined in config or globally if DALI is used
try:
    import nvidia.dali.plugin.pytorch as dali_pytorch
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    logging.warning("NVIDIA DALI not found. DALI will be disabled.")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def sensitivity_prune(model, cfg, val_loader, groups: int = 1):
    """
    Applies sensitivity-based structured pruning to the model.
    Goal: Use Module Robustness Sensitivity (MRS) to decide layer-wise sparsity.
    Args:
        model (nn.Module): The model to prune.
        cfg (Dict[str, Any]): Configuration dictionary.
        val_loader (DataLoader): Validation data loader for sensitivity evaluation.
        groups (int): The number of groups for structured pruning.
    Returns:
        nn.Module: The pruned model.
    """
    logger.info("Starting sensitivity-based structured pruning.")
    # 1) Compute layer sensitivities (MRS): higher ↦ prune less
    sens = compute_layer_sensitivity(model, val_loader, dali_image_processor=None, current_rank=0)
    
    # Ensure sensitivities are a numpy array for min/max operations
    sens_values = np.array(list(sens.values()))
    if len(sens_values) == 0:
        logger.warning("No sensitivities computed. Skipping sensitivity-based pruning.")
        return model

    # 2) Normalize to per-layer pruning ratios
    max_s, min_s = sens_values.max(), sens_values.min()
    if (max_s - min_s) == 0:
        logger.warning("All sensitivities are the same. Cannot normalize for adaptive pruning. Applying uniform pruning if enabled.")
        ratios = np.ones_like(sens_values) * cfg['pruning'].get('amount', 0.2)
    else:
        ratios = 1 - (sens_values - min_s) / (max_s - min_s)
    
    # 3) Apply structured pruning per layer
    prunable_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d)):
            prunable_modules.append((name, module))
    
    sens_map = sens
    
    if len(prunable_modules) != len(sens_values):
        logger.error("Mismatch between number of prunable modules and sensitivity ratios. Cannot apply adaptive pruning.")
        return model

    for i, (name, module) in enumerate(prunable_modules):
        r = ratios[i]
        
        if r > cfg['pruning'].get('min_sparsity', 0.0):
            logger.info(f"Applying structured pruning to {name} with amount={float(r):.4f} and groups={groups}")
            apply_structured_pruning(module, amount=float(r), n=groups)
    logger.info("Sensitivity-based structured pruning completed.")
    return model

def run_pruning_and_quantization(cfg: Dict[str, Any], groups: int = 1):
    """
    Orchestrates the pruning and quantization pipeline.
    """
    logger.info("--- Starting Pruning and Quantization Pipeline ---")
    # 1. Load Model
    model = PerceptionModule(cfg).to(DEVICE)
    checkpoint_path = cfg['training']['pruning']['checkpoint']
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if 'state_dict' in checkpoint:
                model_state_dict = {k.replace('perception_module.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('perception_module.')}
                model.load_state_dict(model_state_dict)
                logger.info(f"Loaded PerceptionModule state_dict from Lightning checkpoint: {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded raw model state_dict from: {checkpoint_path}")
            model.eval()
        except Exception as e:
            logger.error(f"Error loading model checkpoint from {checkpoint_path}: {e}. Starting with uninitialized model.")
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Starting with uninitialized model.")

    # 2. Prepare DataLoaders and DALI processor
    val_dataset = None
    if cfg['data']['use_synthetic_data']:
        dummy_generator = BongardGenerator(cfg['data'], ALL_BONGARD_RULES)
        val_dataset = BongardSyntheticDataset(cfg, dummy_generator, num_samples=cfg['data']['synthetic_data_config'].get('num_val_problems', 100))
    else:
        logger.warning("Real data not supported for standalone `prune_quantize.py` without proper data loading setup. Using dummy synthetic data if available.")
        dummy_generator = BongardGenerator(cfg['data'], ALL_BONGARD_RULES)
        val_dataset = BongardSyntheticDataset(cfg, dummy_generator, num_samples=cfg['data']['synthetic_data_config'].get('num_val_problems', 100))
    
    if val_dataset is None:
        logger.error("Could not create a validation dataset for pruning/quantization. Exiting.")
        return

    train_loader = get_dataloader(cfg, dataset=val_dataset, is_train=True, rank=0, world_size=1)
    val_loader = get_dataloader(cfg, dataset=val_dataset, is_train=False, rank=0, world_size=1)

    dali_image_processor = None
    if cfg['training']['use_dali'] and HAS_DALI:
        try:
            dali_image_processor = build_dali_image_processor(
                batch_size=cfg['training']['batch_size'],
                num_threads=cfg['data']['dataloader_workers'],
                device_id=0,
                image_size=cfg['data']['image_size'],
                is_training=False,
                curriculum_config=cfg['training']['curriculum_config'],
                augmentation_config=cfg['training']['augmentation_config']
            )
        except ImportError:
            logger.warning("DALI not available. Proceeding without DALI image processor.")
        except Exception as e:
            logger.error(f"Error building DALI image processor: {e}. Proceeding without DALI.")

    # 3. Pruning
    if cfg['training']['pruning']['enabled']:
        logger.info("Initiating model pruning.")
        if cfg['training']['pruning']['use_sensitivity']:
            model = sensitivity_prune(model, cfg, val_loader, groups=groups)
        else:
            sensitivity = None
            model = apply_structured_pruning(
                model,
                cfg,
                train_loader,
                val_loader,
                dali_image_processor,
                current_rank=0,
                is_ddp_initialized=False,
                sensitivity_scores=sensitivity,
                groups=groups
            )
        logger.info("Model pruning completed.")
    else:
        logger.info("Model pruning is disabled in config.")

    # 4. Quantization
    if HAS_TORCH_QUANTIZATION:
        if cfg['training']['quantization']['qat']:
            logger.info("Initiating Quantization Aware Training (QAT).")
            model = quantize_model_qat(model, cfg)
            logger.info("Model prepared for QAT. For full QAT benefits, fine-tune this model.")
            model.eval()
            model = torch.quantization.convert(model, inplace=False)
            logger.info("QAT prepared model converted to quantized model.")
        elif cfg['training']['quantization']['ptq']:
            logger.info("Initiating Post-Training Quantization (PTQ).")
            model.eval()
            model = quantize_model_ptq(model, val_loader, cfg)
            logger.info("Model quantized using PTQ.")
        else:
            logger.info("Model quantization is disabled in config.")
    else:
        logger.warning("PyTorch quantization is not available. Skipping QAT/PTQ steps.")

    # 5. Save the optimized model
    optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "optimized_bongard_model.pth")
    os.makedirs(os.path.dirname(optimized_model_path), exist_ok=True)
    torch.save(model.state_dict(), optimized_model_path)
    logger.info(f"Optimized model saved to: {optimized_model_path}")
    logger.info("--- Pruning and Quantization Pipeline finished. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pruning and Quantization for Bongard Solver.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    parser.add_argument('--groups', type=int, default=1,
                        help="Number of groups for structured pruning (e.g., for channel pruning).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    # Update config with output_dir for saving study and best config
    # This ensures that the checkpoints are saved in the correct place,
    # especially if the user overrides save_model_checkpoints via CLI.
    # If not explicitly provided, it will use the default from config.yaml.
    if 'save_model_checkpoints' not in cfg['debug']:
        cfg['debug']['save_model_checkpoints'] = './checkpoints' # Default if not set in config.yaml

    # Create dummy directories if they don't exist for testing
    os.makedirs(cfg['debug']['save_model_checkpoints'], exist_ok=True)
    
    run_pruning_and_quantization(cfg, groups=args.groups)
