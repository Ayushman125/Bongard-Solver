#!/usr/bin/env python3
"""
Professional System 1 Training Pipeline
Integrated training pipeline for Bongard problem detection with professional parameters.
"""

import logging
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
from tqdm import tqdm

# Import core modules
from config import CONFIG
from core_models.training import train_detection_system1, evaluate_detection_model
from core_models.models import PerceptionModule


class ProfessionalTrainingConfig:
    """Professional training configuration for System 1."""
    
    # Professional training parameters
    PROFESSIONAL_EPOCHS = 50
    PROFESSIONAL_BATCH_SIZE = 16  # Adjust based on GPU memory
    PROFESSIONAL_LR = 0.001
    PROFESSIONAL_SYNTHETIC_SIZE = 10000
    
    # Performance targets
    SYNTHETIC_AP_TARGET = 0.85  # AP@0.5:0.95
    SYNTHETIC_AP50_TARGET = 0.95  # AP@0.5
    REAL_AP50_TARGET = 0.70  # AP@0.5 on real dataset
    
    # Advanced training features
    USE_AUGMENTATION = True
    USE_MIXED_PRECISION = True
    USE_WARMUP = True
    WARMUP_EPOCHS = 5
    USE_SCHEDULER = True
    EARLY_STOPPING_PATIENCE = 10


def setup_professional_config(base_config: Dict) -> Dict:
    """Setup professional training configuration."""
    config = base_config.copy()
    
    # Detection model configuration
    model_config = {
        'detector_config': {
            'num_classes': 6,  # background + 5 object types
            'confidence_threshold': 0.5,  # Lower for better recall
            'nms_threshold': 0.4,
            'min_size': 32,
            'max_size': 512
        },
        'backbone': 'resnet50',  # More powerful backbone
        'pretrained': True,
        'feature_dim': 2048,  # ResNet50 feature dimension
        'use_scene_gnn': True,  # Enable for better reasoning
        
        # Attribute classifier configuration
        'attribute_classifier_config': {
            'shape': 6,  # circle, triangle, square, star, pentagon, hexagon
            'color': 8,  # red, blue, green, yellow, orange, purple, black, white
            'fill': 4,  # solid, outline, pattern, gradient
            'size': 5,  # tiny, small, medium, large, huge
            'orientation': 4,  # up, down, left, right
            'texture': 4,  # smooth, rough, dotted, striped
            'mlp_dim': 512,
            'head_dropout_prob': 0.2
        },
        
        # Relation GNN configuration
        'relation_gnn_config': {
            'hidden_dim': 512,
            'num_layers': 4,
            'dropout': 0.1,
            'dropout_prob': 0.1,
            'num_relations': 12,  # More spatial relations
            'use_edge_features': True,
            'node_features_dim': 2048
        },
        
        # Bongard head configuration
        'bongard_head_config': {
            'num_classes': 2,
            'hidden_dim': 512,
            'num_layers': 3,
            'dropout_prob': 0.1
        }
    }
    
    # Professional training configuration
    training_config = {
        'epochs': ProfessionalTrainingConfig.PROFESSIONAL_EPOCHS,
        'batch_size': ProfessionalTrainingConfig.PROFESSIONAL_BATCH_SIZE,
        'detection_lr': ProfessionalTrainingConfig.PROFESSIONAL_LR,
        'weight_decay': 0.0001,
        'use_mixed_precision': ProfessionalTrainingConfig.USE_MIXED_PRECISION,
        'warmup_epochs': ProfessionalTrainingConfig.WARMUP_EPOCHS if ProfessionalTrainingConfig.USE_WARMUP else 0,
        'scheduler': 'CosineAnnealingWarmRestarts' if ProfessionalTrainingConfig.USE_SCHEDULER else None,
        'scheduler_config': {
            'CosineAnnealingWarmRestarts': {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6
            }
        },
        'early_stopping_patience': ProfessionalTrainingConfig.EARLY_STOPPING_PATIENCE,
        'gradient_clip_norm': 1.0,
        'label_smoothing': 0.1
    }
    
    # Professional data configuration
    data_config = {
        'synthetic_size': ProfessionalTrainingConfig.PROFESSIONAL_SYNTHETIC_SIZE,
        'validation_split': 0.15,  # 15% for validation
        'image_size': [640, 640],  # Higher resolution
        'augmentation': ProfessionalTrainingConfig.USE_AUGMENTATION,
        'augmentation_config': {
            'horizontal_flip': 0.5,
            'vertical_flip': 0.3,
            'rotation_degrees': 15,
            'scale_range': (0.8, 1.2),
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1,
            'noise_std': 0.02
        }
    }
    
    # Update configuration
    config['model'] = {**config.get('model', {}), **model_config}
    config['training'] = {**config.get('training', {}), **training_config}
    config['data'] = {**config.get('data', {}), **data_config}
    config['targets'] = {
        'synthetic_ap': ProfessionalTrainingConfig.SYNTHETIC_AP_TARGET,
        'synthetic_ap50': ProfessionalTrainingConfig.SYNTHETIC_AP50_TARGET,
        'real_ap50': ProfessionalTrainingConfig.REAL_AP50_TARGET
    }
    
    return config


class DictConfig:
    """Configuration object with attribute access."""
    
    def __init__(self, data: Dict):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
    def items(self):
        return self._data.items()
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


def create_professional_data_loaders(config: Dict) -> Tuple:
    """Create professional data loaders with enhanced augmentation."""
    from torch.utils.data import DataLoader
    from src.data.bongard_logo_detection import BongardDetectionDataset
    
    logger = logging.getLogger(__name__)
    
    # Load existing synthetic data or generate new
    annotations_file = "processed_data/synthetic_coco_10k.json"
    images_dir = "processed_data/synthetic_images_10k"
    
    if not Path(annotations_file).exists():
        logger.info(f"Generating {config['data']['synthetic_size']} synthetic images...")
        # Generate new dataset
        generate_professional_synthetic_data(config)
    
    logger.info(f"Loading synthetic dataset from {annotations_file}")
    
    # Create datasets
    full_dataset = BongardDetectionDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transform=create_augmentation_transform(config) if config['data']['augmentation'] else None
    )
    
    # Split dataset
    val_size = int(len(full_dataset) * config['data']['validation_split'])
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(42)  # For reproducible splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=BongardDetectionDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=BongardDetectionDataset.collate_fn
    )
    
    logger.info(f"Created professional data loaders: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_loader, val_loader


def create_augmentation_transform(config: Dict):
    """Create professional augmentation pipeline."""
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as F
    
    class ProfessionalAugmentation:
        def __init__(self, aug_config):
            self.aug_config = aug_config
            
        def __call__(self, image, target):
            # Apply augmentations while preserving bounding boxes
            # This is a placeholder - would need full implementation
            return image, target
    
    return ProfessionalAugmentation(config['data']['augmentation_config'])


def generate_professional_synthetic_data(config: Dict):
    """Generate professional synthetic dataset."""
    from src.bongard_generator.dataset import SyntheticBongardDataset, get_all_rules
    import json
    from torchvision.utils import save_image
    
    logger = logging.getLogger(__name__)
    num_images = config['data']['synthetic_size']
    
    logger.info(f"Generating {num_images} professional synthetic images...")
    
    # Create output directories
    images_dir = Path("processed_data/synthetic_images_10k")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all available rules for balanced distribution
    rules = get_all_rules()
    images_per_rule = max(1, num_images // len(rules))
    rule_distribution = [(rule.description, images_per_rule) for rule in rules]
    
    # Adjust for exact count
    total_so_far = sum(count for _, count in rule_distribution)
    if total_so_far < num_images:
        rule_distribution[-1] = (rule_distribution[-1][0], 
                                rule_distribution[-1][1] + (num_images - total_so_far))
    
    # Create synthetic dataset
    img_size = config['data']['image_size'][0]
    synthetic_dataset = SyntheticBongardDataset(
        rules=rule_distribution,
        img_size=img_size,
        grayscale=False,
        flush_cache=True
    )
    
    # Generate COCO annotations
    coco_annotations = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 1, 'name': 'circle'},
            {'id': 2, 'name': 'triangle'}, 
            {'id': 3, 'name': 'square'},
            {'id': 4, 'name': 'star'},
            {'id': 5, 'name': 'pentagon'}
        ]
    }
    
    annotation_id = 1
    
    for idx, example in enumerate(tqdm(synthetic_dataset, desc="Generating professional dataset")):
        if idx >= num_images:
            break
        
        # Save image
        image = example['image']
        image_path = images_dir / f"image_{idx:06d}.png"
        
        if isinstance(image, torch.Tensor):
            save_image(image, image_path)
        else:
            image.save(image_path)
        
        # Create COCO annotation
        height, width = img_size, img_size
        coco_annotations['images'].append({
            'id': idx,
            'file_name': f"image_{idx:06d}.png",
            'height': height,
            'width': width
        })
        
        # Extract objects from scene graph
        scene_graph = example.get('scene_graph', {})
        objects = scene_graph.get('objects', [])
        
        for obj in objects:
            x = obj.get('x', width//2)
            y = obj.get('y', height//2)
            size = obj.get('size', 40)
            shape = obj.get('shape', 'circle')
            
            # Map shape to class ID
            shape_mapping = {
                'circle': 1, 'triangle': 2, 'square': 3,
                'star': 4, 'pentagon': 5
            }
            class_id = shape_mapping.get(shape.lower(), 1)
            
            # Create bounding box
            bbox = [max(0, x - size//2), max(0, y - size//2), 
                   min(size, width - (x - size//2)), min(size, height - (y - size//2))]
            area = bbox[2] * bbox[3]
            
            if area > 0:
                coco_annotations['annotations'].append({
                    'id': annotation_id,
                    'image_id': idx,
                    'category_id': class_id,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0
                })
                annotation_id += 1
    
    # Save annotations
    annotations_path = "processed_data/synthetic_coco_10k.json"
    with open(annotations_path, 'w') as f:
        json.dump(coco_annotations, f, indent=2)
    
    logger.info(f"Generated {len(coco_annotations['images'])} images with "
                f"{len(coco_annotations['annotations'])} annotations")
    logger.info(f"Saved to {annotations_path}")


def run_professional_training(config_dict: Dict) -> Dict[str, Any]:
    """Run professional System 1 training pipeline."""
    logger = logging.getLogger(__name__)
    
    # Convert to DictConfig for model compatibility
    config = DictConfig(config_dict)
    
    logger.info("=== Professional System 1 Training Pipeline ===")
    logger.info(f"Training for {config.training.epochs} epochs with batch size {config.training.batch_size}")
    logger.info(f"Target synthetic AP@0.5:0.95: {config_dict['targets']['synthetic_ap']}")
    logger.info(f"Target real AP@0.5: {config_dict['targets']['real_ap50']}")
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints/professional")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Create professional data loaders
        logger.info("Step 1: Creating professional data loaders...")
        train_loader, val_loader = create_professional_data_loaders(config_dict)
        
        # Step 2: Professional training
        logger.info("Step 2: Starting professional training...")
        best_checkpoint, training_metrics = train_detection_system1(
            cfg=config,
            detection_dataloader=train_loader,
            val_detection_dataloader=val_loader,
            num_epochs=config.training.epochs
        )
        
        # Step 3: Final evaluation
        logger.info("Step 3: Final evaluation...")
        if best_checkpoint and Path(best_checkpoint).exists():
            final_metrics = evaluate_detection_model(
                best_checkpoint, val_loader, 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            final_metrics = training_metrics
        
        # Step 4: Performance analysis
        synthetic_ap = final_metrics.get('AP@0.5:0.95', 0.0)
        synthetic_ap50 = final_metrics.get('AP@0.5', 0.0)
        
        # Check targets
        targets_met = {
            'synthetic_ap': synthetic_ap >= config_dict['targets']['synthetic_ap'],
            'synthetic_ap50': synthetic_ap50 >= config_dict['targets']['synthetic_ap50']
        }
        
        # Results summary
        results = {
            'training_metrics': training_metrics,
            'final_metrics': final_metrics,
            'best_checkpoint': best_checkpoint,
            'targets_met': targets_met,
            'config': config_dict
        }
        
        # Log results
        logger.info("=== Training Results ===")
        logger.info(f"Final synthetic AP@0.5:0.95: {synthetic_ap:.3f} "
                   f"({'✅' if targets_met['synthetic_ap'] else '❌'} Target: {config_dict['targets']['synthetic_ap']})")
        logger.info(f"Final synthetic AP@0.5: {synthetic_ap50:.3f} "
                   f"({'✅' if targets_met['synthetic_ap50'] else '❌'} Target: {config_dict['targets']['synthetic_ap50']})")
        
        # Save results
        results_file = f"results/professional_training_results_{config.training.epochs}ep.json"
        Path("results").mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return results
        
    except Exception as e:
        logger.error(f"Professional training failed: {e}")
        raise


def main():
    """Main professional training entry point."""
    parser = argparse.ArgumentParser(description='Professional System 1 Training')
    parser.add_argument('--epochs', type=int, default=ProfessionalTrainingConfig.PROFESSIONAL_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=ProfessionalTrainingConfig.PROFESSIONAL_BATCH_SIZE,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=ProfessionalTrainingConfig.PROFESSIONAL_LR,
                       help='Learning rate')
    parser.add_argument('--synthetic_size', type=int, default=ProfessionalTrainingConfig.PROFESSIONAL_SYNTHETIC_SIZE,
                       help='Synthetic dataset size')
    parser.add_argument('--gpu_memory_gb', type=int, default=4, 
                       help='Available GPU memory in GB (for auto-adjustment)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/professional_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Auto-adjust parameters based on GPU memory
    if args.gpu_memory_gb <= 4:
        args.batch_size = min(args.batch_size, 8)
        logger.info(f"Adjusted batch size to {args.batch_size} for {args.gpu_memory_gb}GB GPU")
    
    # Setup professional configuration
    config_dict = setup_professional_config(CONFIG)
    
    # Apply command line overrides
    if args.epochs:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size:
        config_dict['training']['batch_size'] = args.batch_size
    if args.lr:
        config_dict['training']['detection_lr'] = args.lr
    if args.synthetic_size:
        config_dict['data']['synthetic_size'] = args.synthetic_size
    
    # Run professional training
    try:
        results = run_professional_training(config_dict)
        
        # Final summary
        if all(results['targets_met'].values()):
            logger.info("🎉 All performance targets achieved!")
        else:
            logger.warning("⚠️ Some performance targets not met. Consider:")
            logger.warning("  - Increasing training epochs")
            logger.warning("  - Adjusting learning rate")
            logger.warning("  - Adding more synthetic data")
            logger.warning("  - Tuning model architecture")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
