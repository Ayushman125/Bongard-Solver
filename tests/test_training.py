# Folder: tests/
# File: tests/test_training.py
import pytest
import torch
import torch.nn as nn
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dummy configuration for testing purposes
class DummyConfig:
    def __init__(self):
        self.model = {
            'backbone': 'dummy_backbone',
            'bongard_head_config': {'num_classes': 2, 'dropout_prob': 0.0},
            'feature_dim': 10
        }
        self.training = {
            'epochs': 1, # Only 1 epoch for smoke test
            'batch_size': 2,
            'learning_rate': 0.001,
            'optimizer': 'Adam',
            'scheduler': 'None',
            'label_smoothing_epsilon': 0.0,
            'use_amp': False,
            'gradient_accumulation_steps': 1,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 0.001,
            'early_stopping_monitor_metric': 'val_loss',
            'use_swa': False,
            'use_qat': False,
            'use_sam_optimizer': False,
            'curriculum_learning': False,
            'seed': 42,
        }
        self.data = {
            'image_size': 32, # Very small image size for fast test
            'synthetic_data_config': {
                'max_support_images_per_problem': 0,
                'num_train_problems': 10, # Small number of problems
                'num_val_problems': 5
            },
            'dataloader_workers': 0,
            'use_synthetic_data': True,
            'use_dali': False, # Use PyTorch DataLoader for simplicity
        }
        self.debug = {
            'log_dir': './test_logs', # Will be created by tmp_path
            'save_model_checkpoints': './test_checkpoints' # Will be created by tmp_path
        }
    
    def copy(self):
        # Simple deep copy for dictionary-like structure
        import copy
        return copy.deepcopy(self.__dict__)

    def get(self, key, default=None):
        keys = key.split('.')
        val = self.__dict__
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            elif hasattr(val, k):
                val = getattr(val, k)
            else:
                return default
        return val

# Mock PerceptionModule
class MockPerceptionModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1) # Minimal layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(cfg.data['image_size'] * cfg.data['image_size'], cfg.model['bongard_head_config']['num_classes'])
    
    def forward(self, x, gts_json_strings_batch):
        # Simulate a forward pass
        x = self.conv(x)
        x = self.flatten(x)
        logits = self.linear(x)
        # Return dummy detected_objects and aggregated_outputs
        return logits, [], {}

# Mock training function (smoke test)
def mock_train(cfg: Dict[str, Any]):
    """
    A mock training function that simulates a single epoch of training
    and saves a dummy checkpoint.
    """
    logger.info(f"Mock training started for {cfg['training']['epochs']} epoch(s).")
    
    # Create dummy model
    model = MockPerceptionModule(cfg).to('cpu') # Run on CPU for tests
    
    # Create dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # Create dummy data loader
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, img_size):
            self.num_samples = num_samples
            self.img_size = img_size
        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            return {
                'query_img1': torch.randn(3, self.img_size, self.img_size),
                'query_img2': torch.randn(3, self.img_size, self.img_size),
                'query_labels': torch.tensor(0, dtype=torch.long),
                'query_gts_json_view1': b'{}',
                'query_gts_json_view2': b'{}',
                'difficulties': torch.tensor(0.5),
                'affine1': [[1,0,0],[0,1,0],[0,0,1]],
                'affine2': [[1,0,0],[0,1,0],[0,0,1]],
                'original_indices': torch.tensor(idx),
                'padded_support_imgs': torch.randn(0, 3, self.img_size, self.img_size),
                'padded_support_labels': torch.tensor([]),
                'padded_support_sgs_bytes': [],
                'num_support_per_problem': torch.tensor(0),
                'tree_indices': torch.tensor(idx),
                'is_weights': torch.tensor(1.0)
            }

    train_dataset = MockDataset(cfg['data']['synthetic_data_config']['num_train_problems'], cfg['data']['image_size'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=0, # No workers for testing simplicity
        collate_fn=lambda x: {k: torch.stack([d[k] for d in x]) if isinstance(x[0][k], torch.Tensor) else [d[k] for d in x] for k in x[0].keys()}
    )
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg['training']['epochs']):
        model.train()
        for batch_idx, data in enumerate(train_loader):
            images = data['query_img1']
            labels = data['query_labels']
            gts_json_strings = data['query_gts_json_view1']

            optimizer.zero_grad()
            logits, _, _ = model(images, gts_json_strings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            logger.debug(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    
    # Simulate saving a checkpoint
    checkpoint_path = os.path.join(cfg['debug']['save_model_checkpoints'], 'checkpoint.pt')
    os.makedirs(cfg['debug']['save_model_checkpoints'], exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Mock training finished. Dummy checkpoint saved to {checkpoint_path}")

# Test for a single epoch smoke test
def test_training_loop(tmp_path):
    """
    Runs a single epoch of training as a smoke test to ensure the training loop
    can execute without immediate crashes and creates a checkpoint file.
    """
    logger.info("Running test_training_loop...")
    
    # Create a local copy of config and update paths to use tmp_path
    cfg_local = DummyConfig().copy()
    cfg_local['training']['epochs'] = 1
    cfg_local['debug']['log_dir'] = str(tmp_path / "logs")
    cfg_local['debug']['save_model_checkpoints'] = str(tmp_path / "checkpoints")

    # Ensure the checkpoint directory exists for the mock_train function
    os.makedirs(cfg_local['debug']['save_model_checkpoints'], exist_ok=True)

    # Call the mock training function
    mock_train(cfg_local)
    
    # Assert that the checkpoint file was created
    checkpoint_file = tmp_path / "checkpoints" / "checkpoint.pt"
    assert checkpoint_file.exists(), f"Checkpoint file not found at {checkpoint_file}"
    logger.info(f"Checkpoint file found at {checkpoint_file}.")
    logger.info("test_training_loop passed.")

