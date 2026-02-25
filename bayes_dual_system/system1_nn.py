import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .image_utils import load_image_array
from .types import ExampleItem

logger = logging.getLogger(__name__)


class SmallCNN(nn.Module):
    def __init__(self, image_size: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        feat_size = image_size // 4
        self.fc1 = nn.Linear(32 * feat_size * feat_size, 64)
        self.fc2 = nn.Linear(64, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.encode(x))


@dataclass
class TrainConfig:
    image_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: Optional[str] = None  # 'cuda', 'cpu', or None (auto-detect)
    pretrain_epochs: int = 30
    pretrain_batch_size: int = 256
    pretrain_workers: int = 4


# Module-level dataset class for pickling support (required for multiprocessing on Windows)
class PretrainDataset(torch.utils.data.Dataset):
    """Lazy-loading image-only dataset for self-supervised pretraining."""
    
    def __init__(self, image_paths: list, image_size: int):
        self.image_paths = image_paths
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        try:
            arr = load_image_array(path, image_size=self.image_size)
            arr = np.asarray(arr, dtype=np.float32)[None, :, :]
            return torch.from_numpy(arr)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)


class NeuralSystem1:
    def __init__(self, config: TrainConfig, pretrained_model: Optional[SmallCNN] = None) -> None:
        self.config = config
        self.using_pretrained = pretrained_model is not None
        
        # Auto-detect device if not specified
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        logger.info(f"NeuralSystem1 initialized on device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Capability: {torch.cuda.get_device_capability(0)}")
        
        if pretrained_model is not None:
            self.model = pretrained_model.to(self.device)
            logger.info(f"[NN-S1] Using pre-trained backbone")
        else:
            self.model = SmallCNN(image_size=config.image_size).to(self.device)
            logger.info(f"[NN-S1] Training from scratch (no pre-training)")

    def fit(self, support_pos: Iterable[ExampleItem], support_neg: Iterable[ExampleItem], freeze_backbone: bool = False) -> None:
        """Train or fine-tune the model.
        
        Args:
            support_pos: Positive support images
            support_neg: Negative support images
            freeze_backbone: If True and using pretrained, only train last layer
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # If not using pretrained or freeze_backbone is False, reinitialize full model
        if not self.using_pretrained or not freeze_backbone:
            self.model = SmallCNN(image_size=self.config.image_size).to(self.device)
        else:
            # Critical in episodic evaluation: reset only the classifier head each episode.
            # Keep pretrained backbone fixed, but avoid cross-episode contamination of fc2.
            self.model.fc2.reset_parameters()
        
        self.model.train()

        # Freeze backbone if requested
        if freeze_backbone and self.using_pretrained:
            logger.debug(f"[NN-S1] Freezing backbone, training only last layer")
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.conv2.parameters():
                param.requires_grad = False
            for param in self.model.fc1.parameters():
                param.requires_grad = False
            for param in self.model.fc2.parameters():
                param.requires_grad = True
            # Only fc2 (last layer) has requires_grad = True (default)
        else:
            logger.debug(f"[NN-S1] Training all parameters")

        pos_list = list(support_pos)
        neg_list = list(support_neg)
        logger.debug(f"[NN-S1] Building dataset from {len(pos_list)} pos and {len(neg_list)} neg examples")
        x_arr, y_arr = self._build_dataset(pos_list, neg_list)
        x = torch.from_numpy(x_arr).to(self.device)
        y = torch.from_numpy(y_arr).to(self.device)
        logger.debug(f"[NN-S1] Dataset on {self.device}: x={x.shape}, y={y.shape}")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        logger.debug(f"[NN-S1] Training {self.config.epochs} epochs on {self.device}")
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                logger.debug(f"[NN-S1] Epoch {epoch+1}/{self.config.epochs}, loss={loss.item():.6f}")

        logger.debug(f"[NN-S1] Training complete")
        self.model.eval()

    def predict_proba(self, item: ExampleItem) -> Tuple[float, float]:
        arr = load_image_array(item.image_path, image_size=self.config.image_size)
        x = torch.from_numpy(arr[None, None, :, :]).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            prob = probs[1].item()
            confidence = max(probs).item()

        confidence = abs(prob - 0.5) * 2.0
        return float(prob), float(confidence)

    @classmethod
    def pretrain_backbone(
        cls,
        config: TrainConfig,
        image_paths_pos: list,
        image_paths_neg: list,
    ) -> SmallCNN:
        """Self-supervised pretraining via rotation prediction.
        
        Args:
            config: Training configuration
            image_paths_pos: List of paths to positive images
            image_paths_neg: List of paths to negative images
            
        Returns:
            Pre-trained SmallCNN model
        """
        logger.info(
            f"[NN-S1] SSL pretraining (rotation) on {len(image_paths_pos)} pos + {len(image_paths_neg)} neg images"
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SmallCNN(image_size=config.image_size).to(device)
        rotation_head = nn.Linear(64, 4).to(device)
        model.train()
        rotation_head.train()
        
        image_paths = list(image_paths_pos) + list(image_paths_neg)
        
        # Shuffle for better training
        np.random.shuffle(image_paths)
        
        dataset = PretrainDataset(image_paths, config.image_size)
        batch_size = config.pretrain_batch_size
        worker_count = max(0, config.pretrain_workers)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(worker_count > 0),
        )
        
        logger.info(
            f"[PRETRAIN] Objective=ssl_rotation, dataset={len(dataset)} images, "
            f"batches={len(dataloader)}, batch_size={batch_size}, workers={worker_count}"
        )
        
        backbone_params = (
            list(model.conv1.parameters())
            + list(model.conv2.parameters())
            + list(model.fc1.parameters())
        )
        optimizer = torch.optim.Adam(
            backbone_params + list(rotation_head.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        pretrain_epochs = max(1, config.pretrain_epochs)
        logger.info(f"[PRETRAIN] Training {pretrain_epochs} epochs on {device}")
        
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "pretrain_backbone_ssl_rotation.pt")
        
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            logger.info(f"[PRETRAIN] Found checkpoint, resuming from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if checkpoint.get("objective") == "ssl_rotation":
                model.load_state_dict(checkpoint["model_state"])
                rotation_head.load_state_dict(checkpoint["rotation_head_state"])
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_epoch = checkpoint["epoch"]
                logger.info(f"[PRETRAIN] Resumed from epoch {start_epoch}")
            else:
                logger.warning("[PRETRAIN] Checkpoint objective mismatch, ignoring old checkpoint")
        
        for epoch in tqdm(range(start_epoch, pretrain_epochs), desc="Pre-training epochs", unit="epoch"):
            total_loss = 0.0
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{pretrain_epochs}", unit="batch", leave=False)
            
            for batch_x in batch_pbar:
                batch_x = batch_x.to(device, non_blocking=True)
                bs = batch_x.size(0)

                x_rot = torch.cat(
                    [
                        batch_x,
                        torch.rot90(batch_x, 1, dims=(2, 3)),
                        torch.rot90(batch_x, 2, dims=(2, 3)),
                        torch.rot90(batch_x, 3, dims=(2, 3)),
                    ],
                    dim=0,
                )
                y_rot = torch.cat(
                    [
                        torch.full((bs,), 0, dtype=torch.long, device=device),
                        torch.full((bs,), 1, dtype=torch.long, device=device),
                        torch.full((bs,), 2, dtype=torch.long, device=device),
                        torch.full((bs,), 3, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                
                optimizer.zero_grad()
                features = model.encode(x_rot)
                rot_logits = rotation_head(features)
                loss = F.cross_entropy(rot_logits, y_rot)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * x_rot.size(0)
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / (len(dataset) * 4.0)
            
            torch.save(
                {
                    "objective": "ssl_rotation",
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "rotation_head_state": rotation_head.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            
            if (epoch + 1) % max(1, pretrain_epochs // 10) == 0:
                logger.info(f"[PRETRAIN] Epoch {epoch+1}/{pretrain_epochs}, loss={avg_loss:.6f}, checkpoint saved")
        
        logger.info(f"[PRETRAIN] Complete! Backbone ready for fine-tuning")
        model.eval()
        return model

    def _build_dataset(
        self,
        support_pos: Iterable[ExampleItem],
        support_neg: Iterable[ExampleItem],
    ):
        xs = []
        ys = []
        for item in support_pos:
            xs.append(load_image_array(item.image_path, image_size=self.config.image_size))
            ys.append(1)
        for item in support_neg:
            xs.append(load_image_array(item.image_path, image_size=self.config.image_size))
            ys.append(0)

        x_arr = np.asarray(xs, dtype=np.float32)[:, None, :, :]
        y_arr = np.asarray(ys, dtype=np.int64)
        return x_arr, y_arr
