# Folder: bongard_solver/
# File: prune_quantize.py
import torch
import torch.nn as nn
import logging
import os
import sys
from typing import Dict, Any, Optional
import numpy as np  # Added for sensitivity_prune
import argparse # Added for command-line arguments

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, load_config, DEVICE, HAS_TORCH_QUANTIZATION
from models import PerceptionModule, LitBongard    # Import your model classes
from data import get_dataloader, build_dali_image_processor    # For data loaders and DALI processor
from training import compute_layer_sensitivity, apply_structured_pruning, quantize_model_qat, quantize_model_ptq    # Import functions from training.py
from data import BongardSyntheticDataset, RealBongardDataset, BongardGenerator  # Import necessary classes for dataset creation
from bongard_rules import ALL_BONGARD_RULES  # Assuming this is available for synthetic data
# Assume HAS_DALI is defined in config or globally if DALI is used
try:
    import nvidia.dali.plugin.pytorch as dali_pytorch
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    logging.warning("NVIDIA DALI not found. DALI will be disabled.")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def sensitivity_prune(model, cfg, val_loader, groups: int = 1): # Added groups argument
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
    # 1) Compute layer sensitivities (MRS): higher  ↦  prune less
    # The `evaluate_sensitivity` function needs to be robust to accept a raw nn.Module
    # and handle the DALI processor if provided.
    sens = compute_layer_sensitivity(model, val_loader, dali_image_processor=None, current_rank=0)  # Assuming no DALI processor needed for sensitivity eval directly here, or pass it if evaluate_sensitivity expects it.
    
    # Ensure sensitivities are a numpy array for min/max operations
    sens_values = np.array(list(sens.values()))  # Assuming evaluate_sensitivity returns a dict of sensitivities
    if len(sens_values) == 0:
        logger.warning("No sensitivities computed. Skipping sensitivity-based pruning.")
        return model
    # 2) Normalize to per-layer pruning ratios
    max_s, min_s = sens_values.max(), sens_values.min()
    if (max_s - min_s) == 0:
        logger.warning("All sensitivities are the same. Cannot normalize for adaptive pruning. Applying uniform pruning if enabled.")
        # Fallback to a default uniform ratio or skip adaptive pruning
        ratios = np.ones_like(sens_values) * cfg['pruning'].get('amount', 0.2)  # Use a default amount
    else:
        ratios = 1 - (sens_values - min_s) / (max_s - min_s)    # invert: high sensitivity→low prune
    
    # 3) Apply structured pruning per layer
    # Need to iterate through named_modules and match with calculated sensitivities
    # This requires `evaluate_sensitivity` to return sensitivities in a consistent order or with module names.
    # For simplicity, let's assume `evaluate_sensitivity` returns a list of sensitivities corresponding to `model.named_modules()`
    # or that we can map them.
    
    # Re-iterate over named_modules to apply pruning
    prunable_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module, (nn.Linear, nn.Conv2d)):
            prunable_modules.append((name, module))
    
    # Map sensitivities to module names
    # This assumes `sens` is a dict where keys are module names
    # If `sens` is just a list, you'd need to ensure order matches `prunable_modules`
    sens_map = sens # Assuming sens is already a dictionary with module names as keys
    
    if len(prunable_modules) != len(sens_values):
        logger.error("Mismatch between number of prunable modules and sensitivity ratios. Cannot apply adaptive pruning.")
        return model  # Or raise an error

    for i, (name, module) in enumerate(prunable_modules):
        r = ratios[i] # Use the ratio from the `ratios` array
        
        if r > cfg['pruning'].get('min_sparsity', 0.0):  # Use a default min_sparsity if not in config
            logger.info(f"Applying structured pruning to {name} with amount={float(r):.4f} and groups={groups}")
            apply_structured_pruning(module, amount=float(r),
                                     n=groups)  # Use the passed 'groups' argument
    logger.info("Sensitivity-based structured pruning completed.")
    return model
def run_pruning_and_quantization(cfg: Dict[str, Any], groups: int = 1): # Added groups argument
    """
    Orchestrates the pruning and quantization pipeline.
    """
    logger.info("--- Starting Pruning and Quantization Pipeline ---")
    # 1. Load Model
    model = PerceptionModule(cfg).to(DEVICE)    # Or LitBongard if you want to load a full PL model
    checkpoint_path = cfg['training']['pruning']['checkpoint']
    if os.path.exists(checkpoint_path):
        try:
            # If it's a Lightning checkpoint, load state_dict from 'state_dict' key
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            if 'state_dict' in checkpoint:
                # Remove 'perception_module.' prefix if loading into bare PerceptionModule
                model_state_dict = {k.replace('perception_module.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('perception_module.')}
                model.load_state_dict(model_state_dict)
                logger.info(f"Loaded PerceptionModule state_dict from Lightning checkpoint: {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded raw model state_dict from: {checkpoint_path}")
            model.eval()    # Set to eval mode for pruning/quantization
        except Exception as e:
            logger.error(f"Error loading model checkpoint from {checkpoint_path}: {e}. Starting with uninitialized model.")
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Starting with uninitialized model.")
    # 2. Prepare DataLoaders and DALI processor
    # For pruning/quantization, we often use the validation set for calibration/evaluation.
    # Ensure that `get_dataloader` is called with a dataset instance, not just `is_train`.
    # Assuming a dummy dataset can be created for this script if not part of a full training run.
    # For now, assuming `get_dataloader` can handle `cfg` directly for simplicity.
    # To get a proper dataset for `get_dataloader`, we need to provide one.
    # Let's assume for this script, we can create a minimal dataset if `cfg['data']['use_synthetic_data']` is True.
    # Dummy dataset creation for `prune_quantize.py` if not part of a full training pipeline
    # In a real scenario, you'd load your actual validation dataset here.
    val_dataset = None
    if cfg['data']['use_synthetic_data']:
        # Create a dummy generator for the synthetic dataset
        dummy_generator = BongardGenerator(cfg['data'], ALL_BONGARD_RULES)
        val_dataset = BongardSyntheticDataset(cfg, dummy_generator, num_samples=cfg['data']['synthetic_data_config'].get('num_val_problems', 100))
    else:
        # If using real data, you'd need to load it here
        # For this script, we'll assume synthetic data or that real data is already set up.
        # If real data is used, ensure `load_bongard_data` is properly called.
        logger.warning("Real data not supported for standalone `prune_quantize.py` without proper data loading setup. Using dummy synthetic data if available.")
        # Fallback to dummy synthetic if real data setup is complex for this script
        dummy_generator = BongardGenerator(cfg['data'], ALL_BONGARD_RULES)
        val_dataset = BongardSyntheticDataset(cfg, dummy_generator, num_samples=cfg['data']['synthetic_data_config'].get('num_val_problems', 100))
    if val_dataset is None:
        logger.error("Could not create a validation dataset for pruning/quantization. Exiting.")
        return
    # Now pass the dataset instance to get_dataloader
    train_loader = get_dataloader(cfg, dataset=val_dataset, is_train=True, rank=0, world_size=1)    # Using val_dataset as train for calibration if needed
    val_loader   = get_dataloader(cfg, dataset=val_dataset, is_train=False, rank=0, world_size=1)
    # Build DALI image processor (needed for `_validate_model` and `fine_tune_model`)
    dali_image_processor = None
    if cfg['training']['use_dali'] and HAS_DALI:    # Check if DALI is globally available
        try:
            dali_image_processor = build_dali_image_processor(
                batch_size=cfg['training']['batch_size'],
                num_threads=cfg['data']['dataloader_workers'],
                device_id=0,    # Use rank 0 for this script
                image_size=cfg['data']['image_size'],
                is_training=False,    # For calibration/evaluation
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
            # Use sensitivity_prune function for adaptive pruning
            model = sensitivity_prune(model, cfg, val_loader, groups=groups) # Pass groups
        else:
            # Fallback to generic structured pruning if sensitivity is not used
            sensitivity = None  # No sensitivity scores needed for non-adaptive pruning
            model = apply_structured_pruning(
                model,
                cfg,
                train_loader,    # Pass train_loader for fine-tuning during iterative pruning
                val_loader,
                dali_image_processor,    # Pass DALI processor
                current_rank=0,
                is_ddp_initialized=False,    # Not in DDP context for this script
                sensitivity_scores=sensitivity,
                groups=groups # Pass groups
            )
        logger.info("Model pruning completed.")
    else:
        logger.info("Model pruning is disabled in config.")
    # 4. Quantization
    if HAS_TORCH_QUANTIZATION:    # Check if PyTorch quantization is available
        if cfg['training']['quantization']['qat']:
            logger.info("Initiating Quantization Aware Training (QAT).")
            # `quantize_model_qat` prepares the model. For full QAT, it needs fine-tuning.
            # In this standalone script, we will just prepare and then convert for demonstration.
            # A real QAT workflow involves a training loop after `prepare_qat`.
            # Prepare for QAT
            model = quantize_model_qat(model, cfg)
            logger.info("Model prepared for QAT. For full QAT benefits, fine-tune this model.")
            # For this script, we'll convert it immediately after preparation.
            # This is more akin to a static PTQ with QAT-like qconfigs if no training occurs.
            model.eval()    # Ensure eval mode before conversion
            model = torch.quantization.convert(model, inplace=False)
            logger.info("QAT prepared model converted to quantized model.")
        elif cfg['training']['quantization']['ptq']:
            logger.info("Initiating Post-Training Quantization (PTQ).")
            # `quantize_model_ptq` needs a data loader for calibration.
            # We pass the `val_loader` for this purpose.
            # Ensure model is in eval mode before PTQ
            model.eval()
            # Pass the model, val_loader, and config to the PTQ function
            model = quantize_model_ptq(model, val_loader, cfg)
            logger.info("Model quantized using PTQ.")
        else:
            logger.info("Model quantization is disabled in config.")
    else:
        logger.warning("PyTorch quantization is not available. Skipping QAT/PTQ steps.")
    # 5. Save the optimized model
    optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "optimized_bongard_model.pth")
    os.makedirs(os.path.dirname(optimized_model_path), exist_ok=True)    # Ensure directory exists
    torch.save(model.state_dict(), optimized_model_path)
    logger.info(f"Optimized model saved to: {optimized_model_path}")
    logger.info("--- Pruning and Quantization Pipeline finished. ---")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pruning and Quantization for Bongard Solver.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    # 10.1 Add --groups Argument
    parser.add_argument('--groups', type=int, default=1,
                        help="Number of groups for structured pruning (e.g., for channel pruning).")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)    # Ensure config.yaml is updated with pruning/quantization settings
    
    # Example usage:
    # python prune_quantize.py --config config.yaml --groups 16
    # Ensure your config.yaml has:
    # training:
    #   pruning:
    #     enabled: True
    #     method: "ln_structured" # or "l1_unstructured", "random_unstructured"
    #     amount: 0.2
    #     iterations: 2
    #     fine_tune_epochs_per_iter: 3
    #     use_sensitivity: True
    #     pruning_target_layers_ratio: 0.5
    #     checkpoint: "./checkpoints/best_perception_module.pth" # Path to a trained model
    #     min_sparsity: 0.0 # Added for sensitivity_prune
    #     # n_groups: 1 # This will now be controlled by the --groups argument
    #     metric: "l1_norm" # Example metric for sensitivity evaluation
    #   quantization:
    #     qat: False
    #     ptq: True # Set to True for PTQ
    #     ptq_type: "static" # or "dynamic"
    #   data:
    #     image_size: 224
    #     dataloader_workers: 4
    #     use_dali: True # Or False if not using DALI
    #     synthetic_data_config: # Needed if using synthetic data in dataloader
    #       num_val_problems: 100 # Number of samples for calibration
    #       min_objects_per_image: 1
    #       max_objects_per_image: 5
    #       min_support_images_per_problem: 2
    #       max_support_images_per_problem: 5
    # debug:
    #   save_model_checkpoints: "./checkpoints"
    # model: # Ensure these are set if using topological features
    #   use_persistent_homology: True
    #   ph_pixel_thresh: 0.5
    #   ph_feature_dim: 64
    # Create dummy directories if they don't exist for testing
    os.makedirs(cfg['debug']['save_model_checkpoints'], exist_ok=True)
    run_pruning_and_quantization(cfg, groups=args.groups) # Pass groups to the function
