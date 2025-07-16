# Folder: bongard_solver/core_models/
# File: training_moco.py
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import sys
import argparse
from typing import Dict, Any, Tuple, List
import numpy as np
import torchvision.transforms as T
from PIL import Image # For loading real images

# Import PyTorch Lightning
import pytorch_lightning as pl

# Import models (from current directory)
from .models import LitSimCLR, BongardPerceptionModel, PerceptionModule # LitSimCLR internally uses SimCLREncoder/MoCo
from .optimizers import get_optimizer # For train_moco_on_real
from .losses import MoCoLoss # For train_moco_on_real

# Import data module and loader (from parent directory's src folder)
# Assuming data.py and its components are in `bongard_solver/data/`
# and config.py is in `bongard_solver/`
try:
    from ..data import get_loader, BongardDataModule, custom_collate_fn, RealBongardDataset # Import RealBongardDataset
    from ..data.generator import LogoGenerator # For synthetic data in MoCoDummyDataset
    HAS_DATA_MODULE = True
except ImportError:
    HAS_DATA_MODULE = False
    logger = logging.getLogger(__name__)
    logger.warning("Could not import BongardDataModule or get_loader from ..data. MoCo pretraining will use dummy dataset.")
    # Dummy custom_collate_fn if not available
    def custom_collate_fn(batch):
        # This dummy collate function will just stack the first two elements (images)
        # and return dummy values for the rest to match the expected tuple structure.
        raw_query_images_view1_np = np.stack([item[0] for item in batch])
        raw_query_images_view2_np = np.stack([item[1] for item in batch])
        
        batch_size = len(batch)
        dummy_label = torch.zeros(batch_size, dtype=torch.long)
        dummy_json = [b'{}'] * batch_size
        dummy_float = torch.zeros(batch_size, dtype=torch.float32)
        dummy_affine = [np.eye(3).tolist()] * batch_size
        dummy_indices = torch.arange(batch_size, dtype=torch.long)
        dummy_support_img = np.zeros((batch_size, 1, 224, 224, 3), dtype=np.uint8) # Assuming 1 support image, 224x224
        dummy_support_label = torch.zeros(batch_size, dtype=torch.long)
        dummy_support_sg = [b'{}'] * batch_size
        dummy_num_support = torch.zeros(batch_size, dtype=torch.long)
        dummy_tree_idx = torch.zeros(batch_size, dtype=torch.long)
        dummy_is_weights = torch.ones(batch_size, dtype=torch.float32)
        dummy_bboxes = [[]] * batch_size
        dummy_masks = [[]] * batch_size
        dummy_confs = [0.0] * batch_size

        return (
            raw_query_images_view1_np, raw_query_images_view2_np, dummy_label,
            dummy_json, dummy_json, dummy_float, dummy_affine, dummy_affine, dummy_indices,
            dummy_support_img, dummy_support_label, dummy_support_sg, dummy_num_support,
            dummy_tree_idx, dummy_is_weights,
            dummy_bboxes, dummy_masks,
            dummy_bboxes, dummy_masks,
            dummy_bboxes, dummy_masks,
            dummy_confs, dummy_confs, dummy_confs # Added dummy confidences
        )
    # Dummy BongardDataModule if not available
    class BongardDataModule(pl.LightningDataModule):
        def __init__(self, cfg: Dict[str, Any], train_dataset=None, val_dataset=None):
            super().__init__()
            self.cfg = cfg
            self._train_dataset = train_dataset
            self._val_dataset = val_dataset
            self.image_size = cfg['data']['image_size']
            self.batch_size = cfg['training']['batch_size']
            self.num_workers = cfg['data']['dataloader_workers']
            self.use_dali = cfg['training'].get('use_dali', False)
            self.dali_image_processor = None # Dummy processor

        def setup(self, stage: Optional[str] = None):
            if stage == 'fit' or stage is None:
                if self._train_dataset is None:
                    self._train_dataset = MoCoDummyDataset(self.cfg)
                if self._val_dataset is None:
                    # For MoCo, validation might not be strictly necessary or can use a small dummy set
                    self._val_dataset = MoCoDummyDataset(self.cfg) 
            
            if self.use_dali:
                # Dummy DALI pipeline for MoCo pretraining if DALI is requested but not fully set up
                logger.warning("DALI is requested but not fully implemented for dummy dataset. Using basic DALI-like behavior.")
                class DummyDALIProcessor:
                    def __init__(self, image_size, mean, std):
                        self.image_size = image_size
                        self.transform = T.Compose([
                            T.ToPILImage(),
                            T.Resize((image_size, image_size)),
                            T.ToTensor(),
                            T.Normalize(mean=mean, std=std)
                        ])
                        self.mean = mean
                        self.std = std

                    def run(self, img1_np_batch, img2_np_batch, support_img_np_batch):
                        # Simulate DALI output: tensors on device
                        img1_tensors = torch.stack([self.transform(img) for img in img1_np_batch])
                        img2_tensors = torch.stack([self.transform(img) for img in img2_np_batch])
                        # For support images, assume they are already processed or handle them similarly
                        support_tensors = torch.stack([self.transform(img.squeeze(0)) for img in support_img_np_batch]) # Squeeze dummy dim
                        return img1_tensors.to(DEVICE), img2_tensors.to(DEVICE), support_tensors.to(DEVICE)
                
                self.dali_image_processor = DummyDALIProcessor(
                    self.image_size, 
                    [0.485, 0.456, 0.406], # ImageNet stats for 3-channel images
                    [0.229, 0.224, 0.225]
                )
                logger.info("Dummy DALI image processor initialized.")
            
        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                self._train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                collate_fn=custom_collate_fn if not self.use_dali else None # DALI handles collation internally
            )

        def val_dataloader(self):
            if self._val_dataset:
                return torch.utils.data.DataLoader(
                    self._val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    shuffle=False,
                    collate_fn=custom_collate_fn if not self.use_dali else None
                )
            return None

# Import config (from parent directory)
try:
    from ..config import load_config, DEVICE, CONFIG, IMAGENET_MEAN, IMAGENET_STD, DATA_ROOT_PATH
    from ..src.utils.augment import augment_image # Import augment_image for real data
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import config or augment_image. Using dummy CONFIG and DEVICE.")
    # Dummy CONFIG and DEVICE for standalone execution
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIG = {
        'device': str(DEVICE),
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
                'pretrain_epochs': 5 # Reduced for quick test
            },
            'bongard_head_config': { # Needed for LitBongard in case it's used as a base
                'num_classes': 2,
                'hidden_dim': 256
            },
            'relation_gnn_config': { # Needed for PerceptionModule
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
            'dataloader_workers': 0, # Set to 0 for dummy dataset to avoid multiprocessing issues
            'use_synthetic_data': True,
            'synthetic_data_config': {
                'num_train_problems': 100,
                'num_val_problems': 20, # Added for completeness
                'max_support_images_per_problem': 0
            },
            'real_data_path': './data/real_bongard' # Dummy path for real data
        },
        'training': {
            'batch_size': 64,
            'use_dali': False,
            'use_amp': False, # Added for mixed precision
            'log_every_n_steps': 50, # For PyTorch Lightning logger
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
            'logs_dir': './logs' # For TensorBoard logger
        }
    }
    def load_config(path: str) -> Dict[str, Any]:
        """Dummy load_config function."""
        return CONFIG
    def augment_image(img_pil): # Dummy augment_image
        return T.ToTensor()(img_pil) # Just convert to tensor
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DATA_ROOT_PATH = "./data"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dummy Dataset for MoCo pretraining if not using a full data pipeline
class MoCoDummyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.image_size = cfg['data']['image_size']
        self.num_samples = cfg['data']['synthetic_data_config'].get('num_train_problems', 100) # Use num_train_problems for samples
        
        # Define basic transforms for MoCo views
        # These transforms should match what's expected for contrastive learning
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(self.image_size[0], scale=(0.2, 1.0)), # Use height for square crop
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # ImageNet stats
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, bytes, bytes, float, List[List[float]], List[List[float]], int, np.ndarray, int, bytes, int, int, float, List[List[List[float]]], List[List[np.ndarray]], List[List[List[float]]], List[List[np.ndarray]], List[List[List[float]]], List[List[np.ndarray]], float, float, List[float]]:
        """
        Returns a tuple matching the expected output of custom_collate_fn in data.py.
        For MoCo, we primarily care about raw_query_images_view1_np and raw_query_images_view2_np.
        The rest are dummy values.
        """
        # Generate a dummy image (e.g., random noise)
        # Ensure it's 3-channel for ImageNet normalization
        # Use the height for square image generation
        img_np = np.random.randint(0, 256, size=(self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        # Apply two different augmentations for MoCo views
        # augment_image expects PIL Image, so convert from numpy
        img_view1 = augment_image(Image.fromarray(img_np)).cpu().numpy() # Convert back to numpy for DALI compatibility
        img_view2 = augment_image(Image.fromarray(img_np)).cpu().numpy()

        # Dummy bounding boxes and masks (empty lists for no detections)
        dummy_bboxes = []
        dummy_masks = []
        dummy_confs = [1.0] # Dummy confidence for synthetic images

        # The full tuple expected by custom_collate_fn:
        # (raw_query_images_view1_np, raw_query_images_view2_np, query_labels,
        #  query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
        #  raw_support_images_flat_np, support_labels_flat, support_sgs_flat_bytes, num_support_per_problem,
        #  tree_indices, is_weights,
        #  query_bboxes_view1, query_masks_view1,
        #  query_bboxes_view2, query_masks_view2,
        #  support_bboxes_flat, support_masks_flat,
        #  query_conf1, query_conf2, support_confs_flat)
        return (img_view1, img_view2, 0, b'{}', b'{}', 0.0, np.eye(3).tolist(), np.eye(3).tolist(), idx,
                np.zeros((self.cfg['data']['synthetic_data_config'].get('max_support_images_per_problem', 0), self.image_size[0], self.image_size[1], 3), dtype=np.uint8), # Dummy support image (shape for max support images)
                [0] * self.cfg['data']['synthetic_data_config'].get('max_support_images_per_problem', 0), b'{}', 0, 0, 1.0, # Dummy support labels, tree_indices, is_weights
                dummy_bboxes, dummy_masks, # query_bboxes_view1, query_masks_view1
                dummy_bboxes, dummy_masks, # query_bboxes_view2, dummy_masks_view2
                dummy_bboxes, dummy_masks, # support_bboxes_flat, dummy_masks_flat
                dummy_confs[0], dummy_confs[0], dummy_confs * self.cfg['data']['synthetic_data_config'].get('max_support_images_per_problem', 0) # Added dummy confidences
                )

def train_moco_on_real(config: Dict[str, Any]):
    """
    Performs MoCo-style self-supervised pretraining on unlabeled real Bongard images.
    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """
    logger.info("Starting MoCo-V2 self-supervised pretraining on real Bongard images.")

    # Initialize query and key encoders (BongardPerceptionModel as the backbone)
    # The BongardPerceptionModel is the feature extractor part of PerceptionModule
    # We create it directly for MoCo pretraining.
    model_q = BongardPerceptionModel(config).to(DEVICE)
    model_k = BongardPerceptionModel(config).to(DEVICE)
    
    # Initialize key encoder with query encoder's weights
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False # Key encoder is not updated by backprop

    optimizer = get_optimizer(model_q, config)
    moco_loss_fn = MoCoLoss(temperature=config['model']['simclr_config']['temperature'],
                            queue_size=config['model']['simclr_config']['moco_k']).to(DEVICE)

    # Prepare data for real Bongard images
    real_data_root = config['data']['real_data_path']
    if not os.path.exists(real_data_root):
        logger.error(f"Real data path '{real_data_root}' does not exist. Cannot perform MoCo pretraining on real data.")
        return

    # Get list of all image paths in the real data directory
    real_image_paths = []
    for root, _, files in os.walk(real_data_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                real_image_paths.append(os.path.join(root, file))
    
    if not real_image_paths:
        logger.warning(f"No real images found in '{real_data_root}'. Skipping MoCo pretraining on real data.")
        return

    logger.info(f"Found {len(real_image_paths)} real images for MoCo pretraining.")
    
    # Create a simple dataset for real images
    class RealImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths: List[str], image_size: List[int]):
            self.image_paths = image_paths
            self.image_size = image_size
            # No normalization here, augment_image handles it.
            self.resize_transform = T.Resize(tuple(self.image_size))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img_pil = Image.open(img_path).convert('RGB') # Ensure RGB for augment_image
            img_pil = self.resize_transform(img_pil) # Resize before augmentation

            # Apply two different augmentations for MoCo views
            x1 = augment_image(img_pil) # augment_image returns a tensor (C, H, W)
            x2 = augment_image(img_pil)
            return x1, x2

    real_dataset = RealImageDataset(real_image_paths, config['data']['image_size'])
    real_loader = torch.utils.data.DataLoader(
        real_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['dataloader_workers'],
        pin_memory=True
    )

    moco_epochs = config['model']['simclr_config'].get('pretrain_epochs', 100)
    moco_m = config['model']['simclr_config'].get('moco_m', 0.999)

    for epoch in range(moco_epochs):
        total_loss = 0.0
        for batch_idx, (x1, x2) in enumerate(real_loader):
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)

            # Compute query features
            # model_q (BongardPerceptionModel) expects a batch of images and returns features
            # It does not need bboxes/masks for simple feature extraction
            q_features = model_q(x1)

            # Compute key features
            with torch.no_grad(): # No gradient for key encoder
                # Update key encoder with momentum
                for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
                    param_k.data = param_k.data * moco_m + param_q.data * (1. - moco_m)
                k_features = model_k(x2)
            
            # Calculate MoCo loss
            loss = moco_loss_fn(q_features, k_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % config['training']['log_every_n_steps'] == 0:
                logger.info(f"MoCo Real Epoch {epoch+1}/{moco_epochs}, Batch {batch_idx+1}/{len(real_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(real_loader)
        logger.info(f"MoCo Real Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # Save the final query encoder (which is the main model being trained)
    moco_real_ckpt_path = config['debug']['save_model_checkpoints']
    os.makedirs(moco_real_ckpt_path, exist_ok=True)
    final_moco_encoder_path = os.path.join(moco_real_ckpt_path, config['model']['simclr_config'].get('moco_real_ckpt_name', 'moco_real_encoder.pth'))
    torch.save(model_q.state_dict(), final_moco_encoder_path)
    logger.info(f"MoCo pretrained encoder on real data saved to: {final_moco_encoder_path}")

def main():
    """
    Main entry point for MoCo-V2 Pretraining using PyTorch Lightning.
    This function primarily handles synthetic data pretraining.
    For real data pretraining, `train_moco_on_real` should be called directly.
    """
    parser = argparse.ArgumentParser(description="Run MoCo-V2 Pretraining.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the main configuration YAML file.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    logger.info("--- Starting MoCo-V2 Pretraining (Synthetic Data) ---")

    # Initialize the LitSimCLR module, which internally handles SimCLREncoder and MoCo
    model = LitSimCLR(cfg)

    # Initialize data module with the dummy dataset
    moco_dataset = MoCoDummyDataset(cfg)
    
    # The BongardDataModule will wrap this dataset and create loaders
    data_module = BongardDataModule(cfg, train_dataset=moco_dataset, val_dataset=None)
    data_module.setup(stage='fit') # Call setup to prepare data loaders

    # Setup PyTorch Lightning Trainer
    checkpoint_dir = cfg['debug']['save_model_checkpoints']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define callbacks
    callbacks = []
    # Add ModelCheckpoint to save the best model
    callbacks.append(pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='moco_pretrained_encoder_synthetic',
        monitor='simclr_train_loss', # Monitor the training loss for pretraining
        mode='min',
        save_top_k=1,
        verbose=True
    ))

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, # Use a single device for this script
        max_epochs=cfg['model']['simclr_config'].get('pretrain_epochs', 10),
        logger=pl.loggers.TensorBoardLogger(cfg['debug']['logs_dir'], name="moco_pretraining_synthetic"), # Use cfg['debug']['logs_dir']
        callbacks=callbacks,
        precision=16 if cfg['training'].get('use_amp', False) else 32, # Use mixed precision if enabled
        enable_progress_bar=True,
        log_every_n_steps=cfg['training'].get('log_every_n_steps', 50)
    )

    # Train the model
    trainer.fit(model, data_module)
    logger.info("--- MoCo-V2 Pretraining on Synthetic Data finished. ---")
    
    # Save the final encoder
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        best_model_path = trainer.checkpoint_callback.best_model_path
        logger.info(f"Best MoCo pretrained encoder (synthetic) saved by checkpoint callback to: {best_model_path}")
        encoder_state_dict = model.feature_extractor.state_dict()
        final_encoder_path = os.path.join(checkpoint_dir, "moco_final_encoder_synthetic.pth")
        torch.save(encoder_state_dict, final_encoder_path)
        logger.info(f"Final MoCo encoder (AttributeClassifier) on synthetic data saved to: {final_encoder_path}")
    else:
        logger.warning("No best model path found from checkpoint callback. Ensuring final encoder is saved.")
        final_encoder_path = os.path.join(checkpoint_dir, "moco_final_encoder_synthetic.pth")
        torch.save(model.feature_extractor.state_dict(), final_encoder_path)
        logger.info(f"Final MoCo encoder (AttributeClassifier) on synthetic data saved to: {final_encoder_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create a dummy config.yaml if it doesn't exist for standalone testing
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
                    'enabled': True, # Enable synthetic pretraining by default for main()
                    'projection_dim': 128,
                    'head_layers': 2,
                    'mlp_hidden_size': 2048,
                    'use_moco': True,
                    'moco_k': 65536,
                    'moco_m': 0.999,
                    'temperature': 0.07,
                    'pretrain_epochs': 5, # Reduced for quick test
                    'moco_real_ckpt_name': 'moco_real_encoder.pth' # Name for real data checkpoint
                },
                'bongard_head_config': {
                    'num_classes': 2,
                    'hidden_dim': 256
                },
                'relation_gnn_config': {
                    'hidden_dim': 256,
                    'num_relations': 11
                },
                'attribute_classifier_config': {
                    'shape': 5, 'color': 7, 'size': 3, 'fill': 2, 'orientation': 4, 'texture': 2,
                    'mlp_dim': 256, 'head_dropout_prob': 0.3
                }
            },
            'data': {
                'image_size': [224, 224],
                'dataloader_workers': 0, # Set to 0 for dummy dataset to avoid multiprocessing issues
                'use_synthetic_data': True,
                'synthetic_data_config': {
                    'num_train_problems': 100,
                    'num_val_problems': 20,
                    'max_support_images_per_problem': 0,
                    'background_texture_path': './data/textures', # Ensure this path exists for testing
                    'num_positive_examples': 6,
                    'num_negative_examples': 6,
                    'occluder_prob': 0.3,
                    'blur_prob': 0.2,
                    'min_occluder_size': 5,
                    'max_occluder_size': 20,
                    'jitter_width_range': [1, 3],
                    'jitter_dash_options': [None, (4,2), (2,2,2)]
                },
                'real_data_path': './data/real_bongard' # Dummy path for real data
            },
            'training': {
                'batch_size': 64,
                'use_dali': False,
                'use_amp': False,
                'log_every_n_steps': 50,
                'optimizer': 'AdamW',
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
                'logs_dir': './logs'
            }
        }
        os.makedirs(sample_config['debug']['save_model_checkpoints'], exist_ok=True)
        os.makedirs(sample_config['debug']['logs_dir'], exist_ok=True)
        # Create dummy data/textures and data/real_bongard for testing
        os.makedirs('./data/textures', exist_ok=True)
        os.makedirs('./data/real_bongard', exist_ok=True)
        # Create a dummy texture image
        dummy_texture = Image.new('RGB', (128, 128), color = (100, 100, 100))
        dummy_texture.save('./data/textures/dummy_texture.png')
        # Create a dummy real image
        dummy_real_img = Image.new('RGB', (224, 224), color = (50, 50, 50))
        dummy_real_img.save('./data/real_bongard/dummy_real_image.png')

        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(sample_config, f, indent=4)
        logger.info(f"Created sample config at: {config_path}")
    
    # Run the main function for synthetic data pretraining
    main()

    # Example of how to call train_moco_on_real separately
    logger.info("\n--- Running MoCo Pretraining on Real Data ---")
    cfg = load_config(config_path) # Reload config to ensure it's up-to-date
    train_moco_on_real(cfg)

