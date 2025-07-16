# Folder: bongard_solver/core_models/
# File: training_moco.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import argparse
from typing import Dict, Any, Tuple
import numpy as np
import torchvision.transforms as T

# Import PyTorch Lightning
import pytorch_lightning as pl

# Import models (from current directory)
from .models import LitSimCLR, PerceptionModule # LitSimCLR internally uses SimCLREncoder/MoCo

# Import data module and loader (from parent directory's src folder)
from ..data import get_loader, BongardDataModule # Import BongardDataModule for trainer

# Import config (from parent directory)
from ..config import load_config, DEVICE, CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dummy Dataset for MoCo pretraining if not using a full data pipeline
class MoCoDummyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.image_size = cfg['data']['image_size']
        self.num_samples = cfg['data']['synthetic_data_config'].get('num_train_problems', 100) # Use num_train_problems for samples

        # Define basic transforms for MoCo views
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(self.image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, bytes, bytes, float, List[List[float]], List[List[float]], int, np.ndarray, int, bytes, int, int, float, List[List[List[float]]], List[List[np.ndarray]], List[List[List[float]]], List[List[np.ndarray]], List[List[List[float]]], List[List[np.ndarray]]]:
        """
        Returns a tuple matching the expected output of custom_collate_fn in data.py.
        For MoCo, we primarily care about raw_query_images_view1_np and raw_query_images_view2_np.
        The rest are dummy values.
        """
        # Generate a dummy image (e.g., random noise)
        img_np = np.random.randint(0, 256, size=(self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Apply two different augmentations for MoCo views
        img_view1 = self.transform(img_np).cpu().numpy() # Convert back to numpy for DALI compatibility
        img_view2 = self.transform(img_np).cpu().numpy()

        # Dummy bounding boxes and masks (empty lists for no detections)
        dummy_bboxes = []
        dummy_masks = []
        
        # The full tuple expected by custom_collate_fn:
        # (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
        #  query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
        #  raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
        #  tree_indices, is_weights,
        #  query_bboxes_view1, query_masks_view1,
        #  query_bboxes_view2, query_masks_view2,
        #  support_bboxes_flat, support_masks_flat)
        return (img_view1, img_view2, 0, b'{}', b'{}', 0.0, np.eye(3).tolist(), np.eye(3).tolist(), idx,
                np.zeros((1, self.image_size, self.image_size, 3), dtype=np.uint8),  # Dummy support image
                0, b'{}', 0, 0, 1.0,  # Dummy support labels, tree_indices, is_weights
                dummy_bboxes, dummy_masks,  # query_bboxes_view1, query_masks_view1
                dummy_bboxes, dummy_masks,  # query_bboxes_view2, query_masks_view2
                dummy_bboxes, dummy_masks)  # support_bboxes_flat, support_masks_flat

def main():
    """
    Main entry point for MoCo-V2 pretraining using PyTorch Lightning.
    """
    parser = argparse.ArgumentParser(description="Run MoCo-V2 Pretraining.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger.info("--- Starting MoCo-V2 Pretraining ---")

    # Initialize the LitSimCLR module, which internally handles SimCLREncoder and MoCo
    model = LitSimCLR(cfg)

    # Initialize data module with the dummy dataset
    # If you have a real dataset for pretraining, use that instead.
    moco_dataset = MoCoDummyDataset(cfg)
    # The BongardDataModule will wrap this dataset and create loaders
    data_module = BongardDataModule(cfg, train_dataset=moco_dataset, val_dataset=None)
    data_module.setup(stage='fit') # Call setup to prepare data loaders

    # Setup PyTorch Lightning Trainer
    # Use cfg['debug']['save_model_checkpoints'] for checkpointing
    checkpoint_dir = cfg['debug']['save_model_checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define callbacks
    callbacks = []
    # Add ModelCheckpoint to save the best model
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='moco_pretrained_encoder',
        monitor='simclr_train_loss',  # Monitor the training loss for pretraining
        mode='min',
        save_top_k=1,
        verbose=True
    ))

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,  # Use a single device for this script
        max_epochs=cfg['model']['simclr_config'].get('pretrain_epochs', 10),
        logger=pl.loggers.TensorBoardLogger(cfg['debug']['logs_dir'], name="moco_pretraining"),  # Use cfg['debug']['logs_dir']
        callbacks=callbacks,
        precision=16 if cfg['training'].get('use_amp', False) else 32,  # Use mixed precision if enabled
        enable_progress_bar=True,
        log_every_n_steps=cfg['training'].get('log_every_n_steps', 50)
    )

    # Train the model
    trainer.fit(model, data_module)
    logger.info("--- MoCo-V2 Pretraining finished. ---")
    
    # Optionally load and save the best model explicitly if not handled by ModelCheckpoint
    # The ModelCheckpoint callback should handle saving the best model.
    # You can access the path to the best model from trainer.checkpoint_callback.best_model_path
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_model_path = trainer.checkpoint_callback.best_model_path
        logger.info(f"Best MoCo pretrained encoder saved by checkpoint callback to: {best_model_path}")
        # Save just the encoder (PerceptionModule.attribute_model) for downstream tasks
        # Assuming LitSimCLR.feature_extractor is the AttributeClassifier
        encoder_state_dict = model.feature_extractor.state_dict()
        final_encoder_path = os.path.join(checkpoint_dir, "moco_final_encoder.pth")
        torch.save(encoder_state_dict, final_encoder_path)
        logger.info(f"Final MoCo encoder (AttributeClassifier) saved to: {final_encoder_path}")
    else:
        logger.warning("No best model path found from checkpoint callback. Ensuring final encoder is saved.")
        # Fallback: save the final state if no best model was explicitly saved by callback
        # Assuming LitSimCLR.feature_extractor is the AttributeClassifier
        final_encoder_path = os.path.join(checkpoint_dir, "moco_final_encoder.pth")
        torch.save(model.feature_extractor.state_dict(), final_encoder_path)
        logger.info(f"Final MoCo encoder (AttributeClassifier) saved to: {final_encoder_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.warning(f"Dummy config.yaml not found at {config_path}. Creating a sample.")
        sample_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model': {
                'backbone': 'mobilenet_v3_small',
                'pretrained': True,
                'feat_dim': 576, # Example feature dim
                'proj_dim': 128, # Example projection dim for SimCLR
                'simclr_config': {
                    'projection_dim': 128,
                    'head_layers': 2,
                    'mlp_hidden_size': 2048, # Added for projection head
                    'use_moco': True,
                    'moco_k': 65536,
                    'moco_m': 0.999,
                    'temperature': 0.07,
                    'pretrain_epochs': 5  # Reduced for quick test
                },
                'bongard_head_config': {  # Needed for LitBongard in case it's used as a base
                    'num_classes': 2,
                    'hidden_dim': 256
                },
                'relation_gnn_config': {  # Needed for PerceptionModule
                    'hidden_dim': 256,
                    'num_relations': 11
                },
                'attribute_classifier_config': { # Added for AttributeClassifier
                    'shape': 5, 'color': 7, 'size': 3, 'fill': 2, 'orientation': 4, 'texture': 2,
                    'mlp_dim': 256, 'head_dropout_prob': 0.3
                }
            },
            'data': {
                'image_size': 224,
                'dataloader_workers': 4,
                'use_synthetic_data': True,
                'synthetic_data_config': {
                    'num_train_problems': 100,
                    'num_val_problems': 20, # Added for completeness
                    'max_support_images_per_problem': 0
                }
            },
            'training': {
                'batch_size': 64,
                'use_dali': False,
                'use_amp': False,  # Added for mixed precision
                'log_every_n_steps': 50,  # For PyTorch Lightning logger
                'optimizer': 'AdamW', # Default optimizer
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'scheduler': 'CosineAnnealingLR',
                'scheduler_config': {
                    'CosineAnnealingLR': {'eta_min': 1e-6}
                },
                'early_stop_patience': 5,
                'early_stop_delta': 0.001
            },
            'debug': {
                'save_model_checkpoints': './checkpoints',
                'logs_dir': './logs'  # For TensorBoard logger
            }
        }
        os.makedirs(sample_config['debug']['save_model_checkpoints'], exist_ok=True)
        os.makedirs(sample_config['debug']['logs_dir'], exist_ok=True)
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(sample_config, f, indent=4)
        logger.info(f"Created sample config at: {config_path}")
    main()

