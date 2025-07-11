# Folder: bongard_solver/
# File: training_moco.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
from typing import Dict, Any

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming SimCLREncoder is defined in models.py
from models import SimCLREncoder, PerceptionModule # Assuming PerceptionModule might be used for the backbone
from data import get_loader # Assuming get_loader is in data.py (or dataloader.py as per user's snippet)
from config import load_config, DEVICE, CONFIG # Import CONFIG for dummy values if needed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    """
    Main entry point for MoCo-V2 pretraining.
    Leverages SimCLREncoder with MoCo loss.
    """
    parser = argparse.ArgumentParser(description="Run MoCo-V2 Pretraining.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger.info("--- Starting MoCo-V2 Pretraining ---")

    # Initialize the encoder for MoCo
    # SimCLREncoder needs to be designed to handle `use_moco` flag.
    # It should have a backbone (e.g., PerceptionModule) and projection/prediction heads.
    # For MoCo, it typically has a query encoder and a momentum encoder.
    # Let's assume SimCLREncoder handles this internally based on `use_moco`.
    
    # Dummy optimizer and config values if not fully defined in config.yaml
    # Ensure cfg has 'moco' section, e.g.:
    # moco:
    #   epochs: 10
    #   learning_rate: 0.03
    #   momentum: 0.99
    #   temperature: 0.07
    #   queue_size: 65536
    #   mlp: True # Whether to use MLP projection head
    #   cos_lr: True # Cosine LR schedule
    #   optimizer: "SGD"
    #   weight_decay: 1e-4

    # Ensure model config has feat_dim and proj_dim
    # model:
    #   feat_dim: 2048 # Output feature dimension of backbone
    #   proj_dim: 128 # Dimension of projection head output
    
    # Instantiate the encoder. Assuming SimCLREncoder takes cfg and handles MoCo setup.
    encoder = SimCLREncoder(cfg=cfg, use_moco=True).to(DEVICE)

    # Define optimizer for the query encoder (momentum encoder is updated via EMA)
    # The user snippet implies `optimizer` is available. Let's define it here.
    # Assuming the optimizer should only optimize the query encoder's parameters.
    # SimCLREncoder should expose `query_encoder_params` or similar.
    # If SimCLREncoder is a LightningModule, it handles optimizers.
    # For a simple script, we'll optimize all encoder parameters.
    
    # Use SGD as is common for MoCo
    optimizer = optim.SGD(encoder.parameters(), 
                          lr=cfg['moco'].get('learning_rate', 0.03),
                          momentum=cfg['moco'].get('momentum', 0.9),
                          weight_decay=cfg['moco'].get('weight_decay', 1e-4))

    # Get data loader. `get_loader` is assumed to be in `data.py` (or `dataloader.py`).
    # It needs a `cfg` and `train` flag.
    train_loader = get_loader(cfg, train=True)

    for epoch in range(cfg['moco'].get('epochs', 10)):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # Assuming batch contains images for MoCo.
            # For MoCo, we typically need two augmented views of the same image.
            # The `get_loader` should provide this, or we apply augmentations here.
            # User's snippet: `q, k = x, x.flip(dims=[-1])` implies `x` is the input image.
            # `x.flip(dims=[-1])` is a simple augmentation (horizontal flip).
            # In a real MoCo setup, you'd use two strong augmentation pipelines.
            
            # Assuming `batch` directly contains the input images `x`
            # If `get_loader` returns a dict (like custom_collate_fn), adapt here.
            # For simplicity, let's assume `batch` is a tuple (images, labels, ...)
            # and we take the first element as the image.
            
            # If using DALI, `batch` will be a list of dictionaries with outputs.
            # For MoCo, you'd typically have two augmented views as outputs.
            # e.g., `batch[0]['view1']`, `batch[0]['view2']`
            
            # For this example, let's assume `get_loader` provides `x` directly
            # or `batch['query_img1']` for the first view.
            # We'll simulate `x` and `x_aug2` (second augmented view)
            
            # If the dataloader provides `query_img1` and `query_img2` as two views:
            if isinstance(batch, dict) and 'query_img1' in batch and 'query_img2' in batch:
                q = batch['query_img1'].to(DEVICE) # Query view
                k = batch['query_img2'].to(DEVICE) # Key view
            else:
                # Fallback if batch format is unexpected, or for dummy data.
                # Create dummy images if `train_loader` doesn't yield images directly.
                img_size = cfg['data'].get('image_size', 224)
                q = torch.randn(cfg['training']['batch_size'], 3, img_size, img_size).to(DEVICE)
                k = torch.randn(cfg['training']['batch_size'], 3, img_size, img_size).to(DEVICE)
                logger.warning("DataLoader did not provide 'query_img1' and 'query_img2'. Using dummy inputs for MoCo.")

            # Calculate MoCo loss
            # The `encoder.moco(q, k)` method should compute the loss.
            loss = encoder.moco(q, k)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {i+1}: loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

    logger.info("--- MoCo-V2 Pretraining finished. ---")

    # Optionally save the pretrained encoder
    pretrained_encoder_path = os.path.join(cfg['debug']['save_model_checkpoints'], "moco_encoder.pth")
    os.makedirs(os.path.dirname(pretrained_encoder_path), exist_ok=True)
    torch.save(encoder.state_dict(), pretrained_encoder_path)
    logger.info(f"Pretrained MoCo encoder saved to: {pretrained_encoder_path}")

if __name__ == "__main__":
    import argparse
    # Ensure config.yaml has necessary MoCo and data settings
    # Example config structure (add to your actual config.yaml):
    # moco:
    #   epochs: 50
    #   learning_rate: 0.03
    #   momentum: 0.99
    #   temperature: 0.07
    #   queue_size: 65536
    #   mlp: True
    #   cos_lr: True
    #   optimizer: "SGD"
    #   weight_decay: 1e-4
    # data:
    #   image_size: 224
    #   dataloader_workers: 4
    #   use_synthetic_data: True # or False
    #   synthetic_data_config:
    #     num_train_problems: 1000 # Number of samples for pretraining
    #     max_support_images_per_problem: 5 # Dummy, not used for MoCo directly
    #   use_dali: False # Set to True if you want to use DALI for MoCo pretraining
    # model:
    #   feat_dim: 2048
    #   proj_dim: 128
    # debug:
    #   save_model_checkpoints: "./checkpoints"

    main()
