# Folder: bongard_solver/
# File: scripts/baseline_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf # For Hydra config
import hydra

# Assume data.py is in the parent directory or accessible via PYTHONPATH
try:
    from data import get_loader, BongardDataModule # get_loader is the main entry point
    from config import CONFIG as FALLBACK_CONFIG # Fallback if not using Hydra
    from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD # For image normalization
except ImportError as e:
    logging.error(f"Could not import necessary modules: {e}. Please ensure data.py and config.py are accessible.")
    # Define dummy values to prevent crashes
    class DummyDataLoader:
        def __init__(self): self.dataset = []; self.batch_size = 1
        def __len__(self): return 0
        def __iter__(self): yield from []
    def get_loader(cfg, train): return DummyDataLoader()
    FALLBACK_CONFIG = {'data': {'image_size': 128, 'synthetic_data_config': {'num_train_problems': 10, 'num_val_problems': 5}, 'dataloader_workers': 0, 'use_synthetic_data': True}, 'training': {'batch_size': 1, 'epochs': 1}, 'debug': {'save_model_checkpoints': './checkpoints'}}
    DEVICE = torch.device('cpu')
    IMAGENET_MEAN = [0.0, 0.0, 0.0]
    IMAGENET_STD = [1.0, 1.0, 1.0]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    Designed as a baseline to quickly verify data loading and basic training.
    """
    def __init__(self, n_classes: int = 2, image_size: int = 224):
        """
        Initializes the SimpleCNN.

        Args:
            n_classes (int): Number of output classes (e.g., 2 for Bongard problems: positive/negative).
            image_size (int): The input image size (e.g., 224x224).
        """
        super().__init__()
        # Feature extractor: two convolutional blocks
        # Input: (Batch, 3, image_size, image_size)
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Output: (Batch, 32, image_size, image_size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # Output: (Batch, 32, image_size/2, image_size/2)
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (Batch, 64, image_size/2, image_size/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                   # Output: (Batch, 64, image_size/4, image_size/4)
        )

        # Calculate the flattened size after feature extraction
        # image_size / 4 for both height and width
        self._flattened_size = 64 * (image_size // 4) * (image_size // 4)

        # Classifier head: Flatten, Linear, ReLU, Linear
        self.classifier = nn.Sequential(
            nn.Flatten(),                                       # Output: (Batch, flattened_size)
            nn.Linear(self._flattened_size, 128),               # Output: (Batch, 128)
            nn.ReLU(),
            nn.Linear(128, n_classes)                           # Output: (Batch, n_classes)
        )
        logger.info(f"SimpleCNN initialized with input size {image_size}x{image_size} and {n_classes} classes.")
        logger.info(f"Flattened feature size: {self._flattened_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleCNN.

        Args:
            x (torch.Tensor): Input image tensor (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits for each class.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

# Main function with Hydra integration
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the CNN baseline training and evaluation.
    Integrates with Hydra for configuration.
    """
    logger.info("--- Starting CNN Baseline Training ---")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set up device
    device = DEVICE # Using global DEVICE from config.py
    logger.info(f"Using device: {device}")

    # Initialize model
    # Ensure n_classes matches your Bongard problem setup (typically 2: positive/negative)
    # Ensure image_size matches your data pipeline's output
    model = SimpleCNN(n_classes=cfg.model.bongard_head_config.num_classes,
                      image_size=cfg.data.image_size).to(device)
    logger.info("SimpleCNN model initialized.")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    logger.info(f"Optimizer: Adam with learning rate {cfg.training.learning_rate}")

    # Loss function (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()
    logger.info("Loss function: CrossEntropyLoss")

    # Data loaders
    # Use BongardDataModule for consistent data loading
    data_module = BongardDataModule(cfg)
    data_module.setup(stage='fit') # Prepare train and val datasets
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    if len(train_loader) == 0:
        logger.error("Training loader is empty. Cannot proceed with training.")
        return
    if len(val_loader) == 0:
        logger.warning("Validation loader is empty. Skipping validation steps.")

    num_epochs = cfg.training.epochs
    best_val_accuracy = 0.0
    model_save_path = os.path.join(cfg.debug.save_model_checkpoints, "baseline_cnn_best_model.pt")
    os.makedirs(cfg.debug.save_model_checkpoints, exist_ok=True)

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_correct_predictions = 0
        train_total_samples = 0
        
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, batch in enumerate(train_loop):
            # Extract query images and labels from the batch
            # Assuming 'query_img1' is the primary image and 'query_labels' are the labels
            images = batch['query_img1'].to(device)
            labels = batch['query_labels'].to(device).long() # Ensure labels are long for CrossEntropyLoss

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            train_correct_predictions += (predictions == labels).sum().item()
            train_total_samples += labels.size(0)

            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct_predictions / train_total_samples
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}")

        # Validation phase
        if len(val_loader) > 0:
            model.eval()
            total_val_loss = 0
            val_correct_predictions = 0
            val_total_samples = 0
            val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            with torch.no_grad():
                for batch_idx_val, batch_val in enumerate(val_loop):
                    images_val = batch_val['query_img1'].to(device)
                    labels_val = batch_val['query_labels'].to(device).long()

                    outputs_val = model(images_val)
                    loss_val = criterion(outputs_val, labels_val)

                    total_val_loss += loss_val.item()
                    
                    predictions_val = torch.argmax(outputs_val, dim=1)
                    val_correct_predictions += (predictions_val == labels_val).sum().item()
                    val_total_samples += labels_val.size(0)
                    val_loop.set_postfix(loss=loss_val.item())

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = val_correct_predictions / val_total_samples
            logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"New best model saved to {model_save_path} with accuracy: {best_val_accuracy:.4f}")
        else:
            logger.info("No validation data available. Skipping validation and model saving.")

    logger.info("--- CNN Baseline Training Finished ---")
    logger.info(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

if __name__ == '__main__':
    # Ensure multiprocessing start method is set for DALI/PyTorch DataLoader
    # when running from a script, especially on Windows.
    # This is handled in main.py, but good to have here for standalone execution.
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
