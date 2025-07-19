"""
Instance Segmentation Training for Bongard Logo Detection
Using Mask R-CNN with custom attribute heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
from torchvision.ops import MultiScaleRoIAlign
import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..data.bongard_logo_detection import BongardDetectionDataset, collate_fn

class BongardMaskRCNN(nn.Module):
    """
    Mask R-CNN with custom attribute prediction heads for Bongard Logo detection.
    Predicts shape, fill, and color attributes for each detected object.
    """
    
    def __init__(self, 
                 num_shape_classes: int = 5,  # circle, square, triangle, pentagon, star
                 num_fill_classes: int = 4,   # solid, outline, striped, gradient  
                 num_color_classes: int = 7,  # red, blue, green, yellow, purple, orange, black
                 pretrained: bool = True):
        super(BongardMaskRCNN, self).__init__()
        
        # Load pretrained Mask R-CNN backbone
        self.backbone = maskrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Replace the box classifier head for shape detection
        in_features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_shape_classes)
        
        # Replace mask predictor if needed
        in_features_mask = self.backbone.roi_heads.mask_predictor.conv5_mask.in_channels
        self.backbone.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_shape_classes)
        
        # Add custom attribute heads
        self.fill_head = nn.Linear(in_features, num_fill_classes)
        self.color_head = nn.Linear(in_features, num_color_classes)
        
        # Store class counts
        self.num_shape_classes = num_shape_classes
        self.num_fill_classes = num_fill_classes
        self.num_color_classes = num_color_classes
        
    def forward(self, images, targets=None):
        """
        Forward pass through the network.
        
        Args:
            images: List of images as tensors
            targets: List of target dictionaries (during training)
            
        Returns:
            During training: Loss dictionary
            During inference: List of prediction dictionaries
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        # Get backbone features and predictions
        backbone_output = self.backbone(images, targets)
        
        if self.training:
            # During training, add attribute losses
            losses = backbone_output
            
            # Extract ROI features for attribute prediction
            # This is a simplified version - in practice you'd need to properly
            # extract features from the ROI heads
            if 'roi_features' in backbone_output:
                roi_features = backbone_output['roi_features']
                
                # Predict attributes
                fill_logits = self.fill_head(roi_features)
                color_logits = self.color_head(roi_features)
                
                # Compute attribute losses (if targets contain attribute labels)
                if targets and 'fill_labels' in targets[0]:
                    fill_targets = torch.cat([t['fill_labels'] for t in targets])
                    fill_loss = F.cross_entropy(fill_logits, fill_targets)
                    losses['loss_fill'] = fill_loss
                
                if targets and 'color_labels' in targets[0]:
                    color_targets = torch.cat([t['color_labels'] for t in targets])
                    color_loss = F.cross_entropy(color_logits, color_targets)
                    losses['loss_color'] = color_loss
            
            return losses
        else:
            # During inference, add attribute predictions
            predictions = backbone_output
            # Add attribute predictions to each detection
            # This would require extracting features for each detection
            # and running through attribute heads
            return predictions

def get_bongard_maskrcnn(num_shape_classes: int = 5, 
                        num_fill_classes: int = 4,
                        num_color_classes: int = 7,
                        pretrained: bool = True) -> torch.nn.Module:
    """
    Create a Bongard Mask R-CNN model.
    
    Args:
        num_shape_classes: Number of shape classes (including background)
        num_fill_classes: Number of fill pattern classes
        num_color_classes: Number of color classes
        pretrained: Whether to use pretrained backbone
        
    Returns:
        Configured Mask R-CNN model
    """
    # For now, use the standard Mask R-CNN and modify it
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the box classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_shape_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_shape_classes)
    
    return model

def get_transforms(training: bool = True):
    """Get image transforms for training or validation"""
    transforms = []
    transforms.append(T.ToTensor())
    
    if training:
        transforms.extend([
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    return T.Compose(transforms)

def train_perception(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Train the perception module on synthetic and real data.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Trained model
    """
    print("🚀 Starting Bongard Logo Perception Training")
    print("=" * 60)
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n📊 Loading datasets...")
    
    # Training transforms
    train_transforms = get_transforms(training=True)
    val_transforms = get_transforms(training=False)
    
    datasets = []
    
    # Load synthetic dataset if specified
    if 'synthetic_images_dir' in config and 'synthetic_annotations' in config:
        print(f"Loading synthetic data from {config['synthetic_images_dir']}")
        
        # Load annotations
        if isinstance(config['synthetic_annotations'], str):
            # Load from JSON file
            from ..data.bongard_logo_detection import load_annotations_from_json
            synth_annotations = load_annotations_from_json(config['synthetic_annotations'])
        else:
            synth_annotations = config['synthetic_annotations']
        
        synth_ds = BongardDetectionDataset(
            config['synthetic_images_dir'],
            synth_annotations,
            train_transforms
        )
        datasets.append(synth_ds)
        print(f"✅ Loaded {len(synth_ds)} synthetic examples")
    
    # Load real dataset if specified
    if 'real_images_dir' in config and 'real_annotations' in config:
        print(f"Loading real data from {config['real_images_dir']}")
        
        # Load annotations
        if isinstance(config['real_annotations'], str):
            from ..data.bongard_logo_detection import load_annotations_from_json
            real_annotations = load_annotations_from_json(config['real_annotations'])
        else:
            real_annotations = config['real_annotations']
            
        real_ds = BongardDetectionDataset(
            config['real_images_dir'],
            real_annotations,
            train_transforms
        )
        datasets.append(real_ds)
        print(f"✅ Loaded {len(real_ds)} real examples")
    
    if not datasets:
        raise ValueError("No datasets specified in config!")
    
    # Combine datasets
    if len(datasets) == 1:
        train_dataset = datasets[0]
    else:
        train_dataset = ConcatDataset(datasets)
    
    print(f"📦 Total training examples: {len(train_dataset)}")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn
    )
    
    # Create model
    print("\n🏗️ Building model...")
    model = get_bongard_maskrcnn(
        num_shape_classes=config.get('num_shape_classes', 5),
        num_fill_classes=config.get('num_fill_classes', 4),
        num_color_classes=config.get('num_color_classes', 7),
        pretrained=config.get('pretrained', True)
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.005),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get('lr_step_size', 3),
        gamma=config.get('lr_gamma', 0.1)
    )
    
    # Training loop
    print("\n🎯 Starting training...")
    model.train()
    
    num_epochs = config.get('epochs', 10)
    save_every = config.get('save_every', 5)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"🏋️ Training Epoch {epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Forward pass
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                # Track losses
                epoch_losses.append(losses.item())
                
                # Update progress bar
                if len(epoch_losses) > 0:
                    avg_loss = np.mean(epoch_losses[-10:])  # Moving average of last 10 batches
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                    })
                    
            except Exception as e:
                print(f"❌ Error in batch {batch_idx}: {e}")
                continue
        
        # End of epoch
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        print(f"📊 Epoch {epoch + 1} completed - Average Loss: {avg_epoch_loss:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            checkpoint_path = config.get('checkpoint_path', 'checkpoints/maskrcnn_bongard.pth')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved checkpoint to {checkpoint_path}")
    
    print("\n🎉 Training completed!")
    return model

def load_trained_model(checkpoint_path: str, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Model configuration
        
    Returns:
        Loaded model
    """
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Create model
    model = get_bongard_maskrcnn(
        num_shape_classes=config.get('num_shape_classes', 5),
        num_fill_classes=config.get('num_fill_classes', 4),
        num_color_classes=config.get('num_color_classes', 7),
        pretrained=False  # Don't load pretrained when loading from checkpoint
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"✅ Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model

# Test function
def test_training():
    """Test training setup"""
    print("Testing training setup...")
    
    # Create dummy config
    config = {
        'device': 'cpu',  # Use CPU for testing
        'batch_size': 2,
        'epochs': 1,
        'lr': 0.001,
        'num_shape_classes': 5,
        'num_fill_classes': 4,
        'num_color_classes': 7,
        'pretrained': False,  # Faster for testing
    }
    
    # Create dummy dataset
    import tempfile
    from PIL import Image
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy images and annotations
        images_dir = os.path.join(tmp_dir, "images")
        os.makedirs(images_dir)
        
        dummy_annotations = {}
        for i in range(4):  # Just 4 images for testing
            img_id = f"test_{i:03d}"
            
            # Create dummy image
            img = Image.new('RGB', (128, 128), color='white')
            img.save(os.path.join(images_dir, f"{img_id}.png"))
            
            # Create dummy annotation
            dummy_annotations[img_id] = [{
                'bbox': [10 + i*10, 10 + i*10, 30, 30],
                'shape_label': i % 5,
                'fill_label': i % 4,
                'color_label': i % 7,
                'mask': np.ones((128, 128), dtype=np.uint8)
            }]
        
        # Update config with paths
        config['synthetic_images_dir'] = images_dir
        config['synthetic_annotations'] = dummy_annotations
        
        # Test model creation
        model = get_bongard_maskrcnn()
        print(f"✅ Model created successfully")
        
        # Test dataset creation
        dataset = BongardDetectionDataset(images_dir, dummy_annotations)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        print("Training setup test completed!")

if __name__ == "__main__":
    test_training()
