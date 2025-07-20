#!/usr/bin/env python3
"""
Professional System 1 Training Script
Integrates with validate_phase1 structure and scales up for production training.
"""

import os, sys, time, logging
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, classification_report

# Setup path imports like validate_phase1
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
SRC_ROOT   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, REPO_ROOT)     # for core_models/
sys.path.insert(0, SRC_ROOT)      # for src/data, src/perception, src/utils

# Core imports
from core_models.training_args import config
from core_models.models import PerceptionModule
from core_models.training import train_detection_system1, evaluate_detection_model
from bongard_generator.sampler import BongardSampler
from bongard_generator.config_loader import get_sampler_config
from bongard_generator.rule_loader import get_all_rules
from bongard_generator.validation import ValidationSuite
from src.data.bongard_logo_detection import BongardDetectionDataset
from torch.utils.data import DataLoader


class ProfessionalSystem1Trainer:
    """Professional trainer for System 1 perception with comprehensive validation."""
    
    def __init__(self, config_dict):
        self.config = config_dict
        self.logger = self._setup_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # Professional training parameters optimized for 4GB GPU
        professional_epochs = 30 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 6e9 else 25
        professional_batch = 16 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 6e9 else 12
        professional_resolution = 224 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 6e9 else 160
        
        self.config.update({
            'training': {
                **self.config.get('training', {}),
                'epochs': professional_epochs,
                'batch_size': professional_batch,
                'detection_lr': 0.0008,  # Optimized learning rate
                'weight_decay': 0.0001,
                'early_stop_patience': 12,
                'use_mixed_precision': True,
                'warmup_epochs': 4,
                'scheduler': 'CosineAnnealingLR',
                'gradient_accumulation_steps': 2 if professional_batch < 16 else 1
            },
            'data': {
                **self.config.get('data', {}),
                'synthetic_size': 10000,  # Professional dataset size
                'validation_split': 0.2,
                'image_size': [professional_resolution, professional_resolution],
                'augmentation': True,
                'augmentation_strength': 0.7
            },
            'model': {
                **self.config.get('model', {}),
                'backbone': 'resnet34',  # Balanced backbone for memory
                'pretrained': True,
                'use_scene_gnn': True,  # Enable full architecture
                'detector_config': {
                    'num_classes': 6,
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.3
                },
                'attribute_classifier_config': {
                    'shape': 5,  # More shape categories
                    'color': 8,  # More colors
                    'fill': 4,
                    'size': 4,  # More size categories
                    'orientation': 4,  # More orientations
                    'texture': 5,  # More textures
                    'position': 9,  # Add position attribute
                    'mlp_dim': 384,  # Optimized MLP size
                    'head_dropout_prob': 0.3
                },
                'relation_gnn_config': {
                    'hidden_dim': 384,
                    'num_layers': 3,
                    'dropout_prob': 0.15,
                    'num_relations': 12,  # More spatial relations
                    'use_edge_features': True,
                    'node_features_dim': 384
                },
                'bongard_head_config': {
                    'num_classes': 2,
                    'hidden_dim': 384
                }
            }
        })
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs/system1_professional")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("System1Professional")
        
    def validate_generator_infrastructure(self):
        """Validate the Bongard generator infrastructure like validate_phase1."""
        self.logger.info("==== Validating Generator Infrastructure ====")
        
        try:
            # Run validation suite
            validator = ValidationSuite()
            validation_results = validator.run_all_validations()
            validator.print_validation_report()
            
            if not all(validation_results.values()):
                self.logger.warning("⚠ Some validations failed but continuing")
                return False
            else:
                self.logger.info("✓ All generator validations passed")
                
            # Test sampler configuration
            self.logger.info("Testing professional sampler configuration...")
            config_obj = get_sampler_config(
                img_size=self.config['data']['image_size'][0],
                max_objs=8,  # More objects for complexity
                generator_mode='genetic'
            )
            self.logger.info(f"✓ Config loaded: img_size={config_obj.img_size}, max_objs={config_obj.max_objs}")
            
            # Test rule coverage
            rules = get_all_rules()
            self.logger.info(f"✓ Loaded {len(rules)} canonical rules")
            
            # Test problem generation with different complexities
            sampler = BongardSampler(config_obj)
            test_rules = [
                "SHAPE(circle)",
                "COUNT(3)", 
                "RELATION(overlap)",
                "SIZE(large)",
                "COLOR(red)",
                "FILL(solid)",
                "TEXTURE(smooth)"
            ]
            
            successful_generations = 0
            for rule_desc in test_rules:
                try:
                    problem = sampler.sample_problem(
                        rule_description=rule_desc,
                        num_pos_scenes=4,
                        num_neg_scenes=4
                    )
                    if problem:
                        successful_generations += 1
                        self.logger.info(f"✓ Generated problem for rule: {rule_desc}")
                except Exception as e:
                    self.logger.warning(f"⚠ Failed to generate problem for rule {rule_desc}: {e}")
            
            success_rate = successful_generations / len(test_rules)
            self.logger.info(f"✓ Rule generation success rate: {success_rate:.2%}")
            
            return success_rate >= 0.8  # Require 80% success rate
            
        except Exception as e:
            self.logger.error(f"✗ Generator validation failed: {e}")
            return False
    
    def create_dict_config(self, config_dict):
        """Convert dictionary to DictConfig object compatible with existing model code."""
        class DictConfig:
            def __init__(self, data):
                for key, value in data.items():
                    if isinstance(value, dict):
                        setattr(self, key, DictConfig(value))
                    else:
                        setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
                
            def items(self):
                return [(k, v) for k, v in self.__dict__.items()]
                
            def __contains__(self, key):
                return hasattr(self, key)
                
            def __getitem__(self, key):
                return getattr(self, key)
                
            def __setitem__(self, key, value):
                setattr(self, key, value)
                
            def __repr__(self):
                return f"DictConfig({self.__dict__})"
        
        return DictConfig(config_dict)
    
    def generate_professional_dataset(self):
        """Generate large-scale synthetic dataset for professional training."""
        self.logger.info(f"==== Generating Professional Dataset ({self.config['data']['synthetic_size']} samples) ====")
        
        # Create comprehensive rule distribution
        rules = get_all_rules()
        
        # Professional rule distribution with balanced complexity
        rule_categories = {
            'shape': ['SHAPE(circle)', 'SHAPE(triangle)', 'SHAPE(square)', 'SHAPE(pentagon)', 'SHAPE(star)'],
            'count': ['COUNT(1)', 'COUNT(2)', 'COUNT(3)', 'COUNT(4)', 'COUNT(5)'],
            'color': ['COLOR(red)', 'COLOR(blue)', 'COLOR(green)', 'COLOR(yellow)', 'COLOR(black)', 'COLOR(white)', 'COLOR(gray)', 'COLOR(orange)'],
            'size': ['SIZE(small)', 'SIZE(medium)', 'SIZE(large)', 'SIZE(varied)'],
            'fill': ['FILL(solid)', 'FILL(outline)', 'FILL(striped)', 'FILL(dotted)'],
            'texture': ['TEXTURE(smooth)', 'TEXTURE(rough)', 'TEXTURE(metallic)', 'TEXTURE(wood)', 'TEXTURE(fabric)'],
            'position': ['POSITION(center)', 'POSITION(corner)', 'POSITION(edge)', 'POSITION(random)', 'POSITION(grid)', 'POSITION(circle)', 'POSITION(line)', 'POSITION(cluster)', 'POSITION(scattered)'],
            'relation': ['RELATION(overlap)', 'RELATION(near)', 'RELATION(far)', 'RELATION(inside)', 'RELATION(outside)', 'RELATION(adjacent)', 'RELATION(aligned)', 'RELATION(symmetric)']
        }
        
        # Calculate samples per category for balanced distribution
        total_samples = self.config['data']['synthetic_size']
        samples_per_category = total_samples // len(rule_categories)
        
        rule_distribution = []
        for category, rules_list in rule_categories.items():
            samples_per_rule = max(1, samples_per_category // len(rules_list))
            for rule_desc in rules_list:
                rule_distribution.append((rule_desc, samples_per_rule))
        
        # Ensure we reach exact target count
        current_total = sum(count for _, count in rule_distribution)
        if current_total < total_samples:
            rule_distribution[-1] = (rule_distribution[-1][0], rule_distribution[-1][1] + (total_samples - current_total))
        
        self.logger.info(f"✓ Created balanced rule distribution: {len(rule_distribution)} rules, {sum(count for _, count in rule_distribution)} total samples")
        
        # Generate dataset using existing infrastructure
        try:
            from src.bongard_generator.dataset import SyntheticBongardDataset
            
            dataset = SyntheticBongardDataset(
                rules=rule_distribution,
                img_size=self.config['data']['image_size'][0],
                grayscale=False,  # Color images for better detection
                flush_cache=True
            )
            
            self.logger.info(f"✓ Generated professional dataset with {len(dataset)} samples")
            
            # Validate dataset quality
            sample_count = min(100, len(dataset))
            valid_samples = 0
            
            for i in tqdm(range(sample_count), desc="Validating dataset quality"):
                try:
                    sample = dataset[i]
                    if 'image' in sample and 'scene_graph' in sample and sample['scene_graph'].get('objects'):
                        valid_samples += 1
                except:
                    continue
            
            quality_ratio = valid_samples / sample_count
            self.logger.info(f"✓ Dataset quality validation: {quality_ratio:.2%} valid samples")
            
            if quality_ratio < 0.9:
                self.logger.warning("⚠ Dataset quality below threshold, but continuing")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"✗ Dataset generation failed: {e}")
            raise
    
    def create_data_loaders(self, dataset):
        """Create professional data loaders with proper splits."""
        self.logger.info("==== Creating Professional Data Loaders ====")
        
        # Calculate split sizes
        total_size = len(dataset)
        val_size = int(total_size * self.config['data']['validation_split'])
        train_size = total_size - val_size
        
        # Create train/val split
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create dataset subsets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        # Professional data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=8,  # More workers for speed
            pin_memory=True,
            persistent_workers=True,
            collate_fn=BongardDetectionDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=BongardDetectionDataset.collate_fn
        )
        
        self.logger.info(f"✓ Created loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
        self.logger.info(f"✓ Batch size: {self.config['training']['batch_size']}, {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader
    
    def train_professional_model(self, train_loader, val_loader):
        """Train System 1 model with professional parameters."""
        self.logger.info("==== Professional System 1 Training ====")
        
        # Convert config to DictConfig for model compatibility
        config_obj = self.create_dict_config(self.config)
        
        # Create checkpoint directory
        checkpoint_dir = Path("checkpoints/system1_professional")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Train the model
            best_checkpoint, training_metrics = train_detection_system1(
                cfg=config_obj,
                detection_dataloader=train_loader,
                val_detection_dataloader=val_loader,
                num_epochs=self.config['training']['epochs']
            )
            
            self.logger.info(f"✓ Training completed. Best checkpoint: {best_checkpoint}")
            self.logger.info(f"✓ Training metrics: {training_metrics}")
            
            self.results['training'] = {
                'best_checkpoint': str(best_checkpoint),
                'metrics': training_metrics,
                'epochs': self.config['training']['epochs'],
                'batch_size': self.config['training']['batch_size']
            }
            
            return best_checkpoint, training_metrics
            
        except Exception as e:
            self.logger.error(f"✗ Training failed: {e}")
            raise
    
    def comprehensive_evaluation(self, checkpoint_path, val_loader):
        """Comprehensive evaluation of trained model."""
        self.logger.info("==== Comprehensive Model Evaluation ====")
        
        try:
            # Evaluate on validation set
            val_metrics = evaluate_detection_model(
                checkpoint_path, val_loader, device=self.device
            )
            
            self.logger.info(f"✓ Validation metrics: {val_metrics}")
            
            # Performance analysis
            ap50 = val_metrics.get('AP@0.5', 0.0)
            ap75 = val_metrics.get('AP@0.75', 0.0)
            ap_all = val_metrics.get('AP@0.5:0.95', 0.0)
            
            # Check professional targets
            targets = {
                'AP@0.5:0.95': 0.75,  # Professional target
                'AP@0.5': 0.90,       # High precision target
                'AP@0.75': 0.60       # Strict localization target
            }
            
            targets_met = {}
            for metric, target in targets.items():
                actual = val_metrics.get(metric, 0.0)
                met = actual >= target
                targets_met[metric] = met
                status = "✅" if met else "⚠️"
                self.logger.info(f"{status} {metric}: {actual:.3f} (target: {target})")
            
            # Overall assessment
            overall_success = sum(targets_met.values()) >= 2  # At least 2/3 targets
            if overall_success:
                self.logger.info("🎉 Professional performance targets achieved!")
            else:
                self.logger.warning("⚠️ Some professional targets not met - consider additional training")
            
            self.results['evaluation'] = {
                'validation_metrics': val_metrics,
                'targets_met': targets_met,
                'overall_success': overall_success
            }
            
            return val_metrics
            
        except Exception as e:
            self.logger.error(f"✗ Evaluation failed: {e}")
            return {}
    
    def generate_training_report(self):
        """Generate comprehensive training report."""
        self.logger.info("==== Generating Training Report ====")
        
        report = {
            'configuration': self.config,
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Save report
        report_dir = Path("results/system1_professional")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"training_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"✓ Training report saved to {report_file}")
        
        # Print summary
        self.logger.info("==== Training Summary ====")
        if 'training' in self.results:
            training = self.results['training']
            self.logger.info(f"✓ Epochs: {training['epochs']}")
            self.logger.info(f"✓ Batch size: {training['batch_size']}")
            self.logger.info(f"✓ Best checkpoint: {training['best_checkpoint']}")
        
        if 'evaluation' in self.results:
            evaluation = self.results['evaluation']
            metrics = evaluation['validation_metrics']
            self.logger.info(f"✓ AP@0.5:0.95: {metrics.get('AP@0.5:0.95', 0):.3f}")
            self.logger.info(f"✓ AP@0.5: {metrics.get('AP@0.5', 0):.3f}")
            self.logger.info(f"✓ AP@0.75: {metrics.get('AP@0.75', 0):.3f}")
            
            if evaluation['overall_success']:
                self.logger.info("🎉 Professional training completed successfully!")
            else:
                self.logger.warning("⚠️ Professional targets not fully achieved")
        
        return report_file
    
    def run_professional_training(self):
        """Run complete professional training pipeline."""
        self.logger.info("🚀 Starting Professional System 1 Training Pipeline")
        
        try:
            # Phase 1: Validate infrastructure
            if not self.validate_generator_infrastructure():
                raise RuntimeError("Generator infrastructure validation failed")
            
            # Phase 2: Generate professional dataset
            dataset = self.generate_professional_dataset()
            
            # Phase 3: Create data loaders
            train_loader, val_loader = self.create_data_loaders(dataset)
            
            # Phase 4: Train model
            checkpoint, training_metrics = self.train_professional_model(train_loader, val_loader)
            
            # Phase 5: Comprehensive evaluation
            val_metrics = self.comprehensive_evaluation(checkpoint, val_loader)
            
            # Phase 6: Generate report
            report_file = self.generate_training_report()
            
            self.logger.info("✅ Professional System 1 training pipeline completed successfully!")
            return report_file
            
        except Exception as e:
            self.logger.error(f"❌ Professional training pipeline failed: {e}")
            raise


def main():
    """Main entry point for professional training."""
    parser = argparse.ArgumentParser(description='Professional System 1 Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--synthetic_size', type=int, default=10000, help='Synthetic dataset size')
    parser.add_argument('--image_size', type=int, default=224, help='Image resolution')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Model backbone')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load base configuration (from config.py)
    base_config = config.copy()
    
    # Override with command line arguments
    if args.epochs:
        base_config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        base_config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.synthetic_size:
        base_config.setdefault('data', {})['synthetic_size'] = args.synthetic_size
    if args.image_size:
        base_config.setdefault('data', {})['image_size'] = [args.image_size, args.image_size]
    if args.backbone:
        base_config.setdefault('model', {})['backbone'] = args.backbone
    if args.lr:
        base_config.setdefault('training', {})['detection_lr'] = args.lr
    
    # Create trainer and run
    trainer = ProfessionalSystem1Trainer(base_config)
    report_file = trainer.run_professional_training()
    
    print(f"\n🎉 Professional training completed! Report saved to: {report_file}")


if __name__ == "__main__":
    main()
