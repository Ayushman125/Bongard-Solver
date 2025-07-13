# Folder: bongard_solver/
# File: training_moco.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import argparse # Import argparse
from typing import Dict, Any
import numpy as np # For dummy data generation
import torchvision.transforms as T # For dummy data transforms

# Add the parent directory to the Python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming SimCLREncoder is defined in models.py
from models import SimCLREncoder, PerceptionModule  # Assuming PerceptionModule might be used for the backbone
from data import get_loader  # Assuming get_loader is in data.py (or dataloader.py as per user's snippet)
from config import load_config, DEVICE, CONFIG  # Import CONFIG for dummy values if needed

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
    # SimCLREncoder's __init__ should take `feat_dim`, `proj_dim`, `head_layers`, and `use_moco`.
    # It will use the `cfg` internally to get `moco_k`, `moco_m`, `temperature`.
    encoder = SimCLREncoder(
        feat_dim=cfg['model'].get('feat_dim', 2048), # Default if not in config
        proj_dim=cfg['model'].get('proj_dim', 128), # Default if not in config
        head_layers=cfg['model']['simclr_config'].get('head_layers', 2), # Default if not in config
        use_moco=True
    ).to(DEVICE)

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

    # Create a dummy dataset for get_loader if it expects one
    class MoCoDummyDataset(torch.utils.data.Dataset):
        def __init__(self, cfg):
            self.cfg = cfg
            self.image_size = cfg['data']['image_size']
            self.num_samples = cfg['data']['synthetic_data_config'].get('num_train_problems', 100) # Use num_train_problems as samples

            # Dummy transforms for two views
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Return raw numpy arrays simulating augmented views for DALI/data.py processing
            img_np = np.random.randint(0, 256, size=(self.image_size, self.image_size, 3), dtype=np.uint8)
            # Simulate a simple augmentation for the second view (e.g., horizontal flip)
            img_np_flipped = np.flip(img_np, axis=1).copy() # .copy() to ensure contiguous array

            # get_loader's custom_collate_fn expects a specific tuple structure.
            # (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
            # query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
            # raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
            # tree_indices, is_weights)
            
            # For MoCo, we primarily care about raw_query_images_view1_np and raw_query_images_view2_np.
            # The rest can be dummy values.
            return (img_np, img_np_flipped, 0, b'{}', b'{}', 0.0, np.eye(3).tolist(), np.eye(3).tolist(), idx,
                    np.zeros((1, cfg['data']['image_size'], cfg['data']['image_size'], 3), dtype=np.uint8), # Dummy support image
                    0, b'{}', 0, 0, 1.0) # Dummy support labels, tree_indices, is_weights

    moco_dataset = MoCoDummyDataset(cfg)
    train_loader = get_loader(cfg, train=True, dataset=moco_dataset) # Pass the dummy dataset

    epochs = cfg['moco'].get('epochs', 10) # Default epochs for MoCo
    logger.info(f"Starting MoCo pretraining for {epochs} epochs.")

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # Unpack batch data as expected by custom_collate_fn
            (raw_query_images_view1_np, raw_query_images_view2_np, _, # query_labels are not used in SimCLR loss
             _, _, _, _, _, _, # other unused fields
             _, _, _, _, _) = batch # support images, labels, etc. not used

            # DALI Image Processor
            # Access the DALI processor from the DataLoader's dataset if it was set up there.
            dali_processor = getattr(train_loader.dataset, 'dali_image_processor', None)
            
            if dali_processor is None or not cfg['training'].get('use_dali', False):
                # Fallback for testing without DALI
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((cfg['data']['image_size'], cfg['data']['image_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Assuming ImageNet normalization
                ])
                q = torch.stack([transform(img_np) for img_np in raw_query_images_view1_np]).to(DEVICE)
                k = torch.stack([transform(img_np) for img_np in raw_query_images_view2_np]).to(DEVICE)
            else:
                # DALI returns processed tensors
                # The `dali_processor.run` method expects lists of numpy arrays.
                # It returns a tuple of tensors (query_img1, query_img2, support_imgs_flat).
                # We only need the first two for MoCo.
                q, k, _ = dali_processor.run(
                    raw_query_images_view1_np,
                    raw_query_images_view2_np,
                    # Provide dummy support images if the DALI pipeline expects 3 inputs,
                    # even if not used by MoCo.
                    [np.zeros((1, cfg['data']['image_size'], cfg['data']['image_size'], 3), dtype=np.uint8)] * len(raw_query_images_view1_np)
                )
            
            # Calculate MoCo loss
            # The `encoder.moco(q, k)` method should compute the logits and labels for InfoNCE.
            logits, labels = encoder.moco(q, k)
            
            # Calculate NT-Xent loss using standard CrossEntropyLoss
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, labels)

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
    # Setup basic logging for the script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy config.yaml for testing if it doesn't exist
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.warning(f"Dummy config.yaml not found at {config_path}. Creating a sample.")
        sample_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model': {
                'backbone': 'mobilenet_v3_small',
                'pretrained': True,
                'feat_dim': 576, # Example feature dim for mobilenet_v3_small
                'proj_dim': 128, # Example projection dim
                'simclr_config': {
                    'projection_dim': 128,
                    'head_layers': 2,
                    'use_moco': True,
                    'moco_k': 65536,
                    'moco_m': 0.999,
                    'temperature': 0.07,
                    'pretrain_epochs': 10 # For LitSimCLR
                }
            },
            'data': {
                'image_size': 224,
                'dataloader_workers': 4,
                'use_synthetic_data': True, # For dummy data
                'synthetic_data_config': {
                    'num_train_problems': 100, # Number of samples for pretraining
                    'max_support_images_per_problem': 0 # Not relevant for MoCo pretraining
                }
            },
            'moco': { # Specific MoCo training parameters
                'lr': 0.03,
                'epochs': 5,
                'momentum': 0.9,
                'weight_decay': 1e-4
            },
            'training': { # General training config, might be used by get_loader
                'batch_size': 64,
                'use_dali': False # Set to True if DALI is installed and configured
            },
            'debug': {
                'save_model_checkpoints': './checkpoints' # Directory to save checkpoints
            }
        }
        # Ensure the checkpoint directory exists for saving the model
        os.makedirs(sample_config['debug']['save_model_checkpoints'], exist_ok=True)

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(sample_config, f, indent=4)
        logger.info(f"Created sample config at: {config_path}")

    main()
