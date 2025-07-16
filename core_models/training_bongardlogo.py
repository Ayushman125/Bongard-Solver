# Folder: bongard_solver/core_models/
# File: training_bongardlogo.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import logging
import os

# Import necessary components from your project
from .models import BongardPerceptionModel # Assuming this is your backbone for binary classification
from .optimizers import get_optimizer
from .metrics import classification_accuracy # Assuming this function exists
from data.bongardlogo_dataset import BongardLogoDataset # The new dataset
# Import the Config from training_args.py
from .training_args import Config as TrainingConfig # Renamed to avoid conflict with local 'config' var

logger = logging.getLogger(__name__)

def train_on_bongardlogo(cfg: TrainingConfig):
    """
    Performs supervised fine-tuning on the Bongard-LOGO dataset.

    Args:
        cfg (TrainingConfig): The configuration object containing training parameters.
    """
    logger.info("--- Starting Supervised Fine-Tuning on Bongard-LOGO Dataset ---")

    # 1) Prepare dataset and DataLoader
    try:
        train_ds = BongardLogoDataset(cfg.bongardlogo_root,
                                      split="train",
                                      img_size=cfg.img_size)
        # For validation, you might want a separate split if available
        val_ds = BongardLogoDataset(cfg.bongardlogo_root,
                                    split="val", # Assuming a 'val' split exists
                                    img_size=cfg.img_size)
    except FileNotFoundError as e:
        logger.error(f"Failed to load BongardLogoDataset: {e}. Skipping fine-tuning.")
        return

    train_loader = DataLoader(train_ds,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.dataloader_workers, # Use cfg.data.dataloader_workers
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.dataloader_workers,
                            pin_memory=True)

    # 2) Model & criterion for binary classification
    # The BongardPerceptionModel should output a single logit for binary classification
    # It needs to be adapted to handle 1-channel input if its backbone expects 3.
    # For now, assuming BongardPerceptionModel can handle 1-channel input implicitly
    # or its backbone is modified to accept it.
    model = BongardPerceptionModel(num_classes=1, cfg=cfg.model).to(cfg.device) # Pass model config
    
    # Optimizer from centralized function
    optimizer = get_optimizer(model.parameters(), cfg.training) # Pass model.parameters() and training config
    
    # BCEWithLogitsLoss is good for binary classification with raw logits
    criterion = nn.BCEWithLogitsLoss()

    # Define checkpoint path
    os.makedirs(os.path.dirname(cfg.bongardlogo_ckpt), exist_ok=True)
    best_val_accuracy = 0.0
    
    # 3) Training loop
    logger.info(f"Training for {cfg.logo_epochs} epochs.")
    for epoch in range(cfg.logo_epochs):
        model.train()
        total_train_loss, total_train_acc = 0.0, 0.0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{cfg.logo_epochs} [Train Bongard-LOGO]")
        for imgs, labels in train_loop:
            imgs = imgs.to(cfg.device)
            labels = labels.float().to(cfg.device) # Ensure labels are float for BCEWithLogitsLoss

            # Model forward pass: BongardPerceptionModel should output features, then a single logit
            # Assuming BongardPerceptionModel has a classification head for num_classes=1
            # If BongardPerceptionModel is just a feature extractor, you'd need a separate head.
            # Let's assume it has a flexible head that can output 1 class.
            # The output of BongardPerceptionModel is typically features, so we need to pass it through a linear layer.
            # For this context, let's assume `model` is the full classification model.
            # If `BongardPerceptionModel` is intended to be the backbone, then `model` here should be `nn.Sequential(BongardPerceptionModel(...), nn.Linear(feat_dim, 1))`
            
            # Assuming BongardPerceptionModel's forward method returns features, and we need a final linear layer.
            # Or, if it's a full classifier for this task, it should return logits directly.
            # Given the snippet, it implies `model(imgs)` returns logits directly.
            # We'll adjust its init in models.py if needed.
            
            # For now, assume BongardPerceptionModel is adapted to output 1 logit for binary classification.
            logits = model(imgs).squeeze(1) # [B]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_train_acc += classification_accuracy(labels.cpu().numpy(), preds.cpu().numpy())
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item(), acc=total_train_acc / (train_loop.n + 1)) # Update postfix

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        logger.info(f"[BongardLogo][Epoch {epoch+1}] Train Loss={avg_train_loss:.4f} Train Acc={avg_train_acc:.4f}")

        # 4) Validation loop
        model.eval()
        total_val_loss, total_val_acc = 0.0, 0.0
        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{cfg.logo_epochs} [Val Bongard-LOGO]")
        with torch.no_grad():
            for imgs, labels in val_loop:
                imgs = imgs.to(cfg.device)
                labels = labels.float().to(cfg.device)

                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

                preds = (torch.sigmoid(logits) > 0.5).long()
                total_val_acc += classification_accuracy(labels.cpu().numpy(), preds.cpu().numpy())
                total_val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item(), acc=total_val_acc / (val_loop.n + 1))

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        logger.info(f"[BongardLogo][Epoch {epoch+1}] Val Loss={avg_val_loss:.4f} Val Acc={avg_val_acc:.4f}")

        # Save best model based on validation accuracy
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            torch.save(model.state_dict(), cfg.bongardlogo_ckpt)
            logger.info(f"Saved best model checkpoint to {cfg.bongardlogo_ckpt} with accuracy {best_val_accuracy:.4f}")

    logger.info("--- Supervised Fine-Tuning on Bongard-LOGO Completed ---")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Dummy Config for standalone testing
    class DummyModelConfig:
        def __init__(self):
            self.backbone = "mobilenet_v3_small"
            self.pretrained = True
            self.feat_dim = 576 # Example for mobilenet_v3_small
            self.bongard_head_config = {'num_classes': 1, 'hidden_dim': 256} # For binary classification
    
    class DummyTrainingConfig:
        def __init__(self):
            self.optimizer = "AdamW"
            self.learning_rate = 1e-3
            self.weight_decay = 1e-5
            self.epochs = 5 # Reduced for quick test
    
    class DummyDataConfig:
        def __init__(self):
            self.dataloader_workers = 0 # Set to 0 for dummy dataset to avoid multiprocessing issues
            self.image_size = 128
            self.bongardlogo_root = "./dummy_bongardlogo_data" # Point to the dummy data
    
    class DummyConfig:
        def __init__(self):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.img_size = 128
            self.bongardlogo_root = "./dummy_bongardlogo_data"
            self.batch_size = 32
            self.logo_epochs = 5
            self.bongardlogo_ckpt = 'checkpoints/bongardlogo_cnn.pth'
            self.model = DummyModelConfig()
            self.training = DummyTrainingConfig()
            self.data = DummyDataConfig()
            
            # Ensure checkpoint directory exists for dummy run
            os.makedirs(os.path.dirname(self.bongardlogo_ckpt), exist_ok=True)

    dummy_cfg = DummyConfig()

    # Create dummy dataset structure for testing
    dummy_root = dummy_cfg.bongardlogo_root
    os.makedirs(os.path.join(dummy_root, "train", "problem_0001", "images", "pos"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "train", "problem_0001", "images", "neg"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "val", "problem_0002", "images", "pos"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "val", "problem_0002", "images", "neg"), exist_ok=True)

    # Create dummy image files (grayscale)
    Image.new('L', (dummy_cfg.img_size, dummy_cfg.img_size), color=0).save(os.path.join(dummy_root, "train", "problem_0001", "images", "pos", "pos_01.png"))
    Image.new('L', (dummy_cfg.img_size, dummy_cfg.img_size), color=255).save(os.path.join(dummy_root, "train", "problem_0001", "images", "neg", "neg_01.png"))
    Image.new('L', (dummy_cfg.img_size, dummy_cfg.img_size), color=100).save(os.path.join(dummy_root, "val", "problem_0002", "images", "pos", "pos_01.png"))

    train_on_bongardlogo(dummy_cfg)

    # Clean up dummy data
    import shutil
    if os.path.exists(dummy_root):
        shutil.rmtree(dummy_root)
        print(f"\nCleaned up dummy data at {dummy_root}")
    if os.path.exists(os.path.dirname(dummy_cfg.bongardlogo_ckpt)):
        shutil.rmtree(os.path.dirname(dummy_cfg.bongardlogo_ckpt))
        print(f"Cleaned up dummy checkpoint directory at {os.path.dirname(dummy_cfg.bongardlogo_ckpt)}")
