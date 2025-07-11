# Folder: bongard_solver/
# File: prune_quantize.py
import torch
import torch.nn as nn
import logging
import os
import sys
from typing import Dict, Any, Optional

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import CONFIG, load_config, DEVICE, HAS_TORCH_QUANTIZATION
from models import PerceptionModule, LitBongard # Import your model classes
from data import get_dataloader, build_dali_image_processor # For data loaders and DALI processor
from training import compute_layer_sensitivity, apply_structured_pruning, quantize_model_qat, quantize_model_ptq # Import functions from training.py

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_pruning_and_quantization(cfg: Dict[str, Any]):
    """
    Orchestrates the pruning and quantization pipeline.
    """
    logger.info("--- Starting Pruning and Quantization Pipeline ---")

    # 1. Load Model
    model = PerceptionModule(cfg).to(DEVICE) # Or LitBongard if you want to load a full PL model
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
            model.eval() # Set to eval mode for pruning/quantization
        except Exception as e:
            logger.error(f"Error loading model checkpoint from {checkpoint_path}: {e}. Starting with uninitialized model.")
    else:
        logger.warning(f"Model checkpoint not found at {checkpoint_path}. Starting with uninitialized model.")

    # 2. Prepare DataLoaders and DALI processor
    # For pruning/quantization, we often use the validation set for calibration/evaluation.
    train_loader = get_dataloader(cfg, is_train=True, rank=0, world_size=1)
    val_loader   = get_dataloader(cfg, is_train=False, rank=0, world_size=1)
    
    # Build DALI image processor (needed for `_validate_model` and `fine_tune_model`)
    dali_image_processor = None
    if cfg['training']['use_dali'] and build_dali_image_processor: # Check if function exists
        try:
            dali_image_processor = build_dali_image_processor(
                batch_size=cfg['training']['batch_size'],
                num_threads=cfg['data']['dataloader_workers'],
                device_id=0, # Use rank 0 for this script
                image_size=cfg['data']['image_size'],
                is_training=False, # For calibration/evaluation
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
        sensitivity = None
        if cfg['training']['pruning']['use_sensitivity']:
            sensitivity = compute_layer_sensitivity(model, val_loader, dali_image_processor, current_rank=0)
        
        model = apply_structured_pruning(
            model, 
            cfg, 
            train_loader, 
            val_loader, 
            dali_image_processor, # Pass DALI processor
            current_rank=0, 
            is_ddp_initialized=False, # Not in DDP context for this script
            sensitivity_scores=sensitivity
        )
        logger.info("Model pruning completed.")
    else:
        logger.info("Model pruning is disabled in config.")

    # 4. Quantization
    if cfg['training']['quantization']['qat']:
        logger.info("Initiating Quantization Aware Training (QAT).")
        model = quantize_model_qat(model, cfg)
        # After QAT preparation, the model needs to be fine-tuned.
        # This fine-tuning loop is typically part of your main training process
        # or a dedicated short training phase.
        logger.info("Model prepared for QAT. Remember to fine-tune it in your training loop.")
        # Example: fine_tune_model(model, train_loader, val_loader, dali_image_processor, epochs=5, learning_rate=1e-5)
    elif cfg['training']['quantization']['ptq']:
        logger.info("Initiating Post-Training Quantization (PTQ).")
        if dali_image_processor:
            model = quantize_model_ptq(model, val_loader, dali_image_processor, cfg)
            logger.info("Model quantized using PTQ.")
        else:
            logger.error("DALI Image Processor is required for PTQ calibration. Skipping PTQ.")
    else:
        logger.info("Model quantization is disabled in config.")

    # 5. Save the optimized model
    optimized_model_path = os.path.join(cfg['debug']['save_model_checkpoints'], "optimized_bongard_model.pth")
    torch.save(model.state_dict(), optimized_model_path)
    logger.info(f"Optimized model saved to: {optimized_model_path}")

    logger.info("--- Pruning and Quantization Pipeline finished. ---")

if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config.yaml") # Ensure config.yaml is updated with pruning/quantization settings

    # Example usage:
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
    #     checkpoint: "./checkpoints/best_bongard_model.pth" # Path to a trained model
    #   quantization:
    #     qat: False
    #     ptq: True # Set to True for PTQ
    #     calibration_batches: 10
    # data:
    #   image_size: 224
    #   dataloader_workers: 4
    #   use_dali: True # Or False if not using DALI
    #   synthetic_data_config: # Needed if using synthetic data in dataloader
    #     min_objects_per_image: 1
    #     max_objects_per_image: 5
    #     min_support_images_per_problem: 2
    #     max_support_images_per_problem: 5
    # debug:
    #   save_model_checkpoints: "./checkpoints"

    run_pruning_and_quantization(cfg)

