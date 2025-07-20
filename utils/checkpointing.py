#!/usr/bin/env python3
"""
Checkpointing system to avoid redundant operations.
Tracks completion of data generation, training, and evaluation stages.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages checkpoints to avoid redundant operations."""
    
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint files for different stages
        self.data_checkpoint = self.checkpoint_dir / "data_generation.json"
        self.training_checkpoint = self.checkpoint_dir / "training.json"
        self.evaluation_checkpoint = self.checkpoint_dir / "evaluation.json"
        
        # Load existing checkpoints
        self.data_status = self._load_checkpoint(self.data_checkpoint)
        self.training_status = self._load_checkpoint(self.training_checkpoint)
        self.evaluation_status = self._load_checkpoint(self.evaluation_checkpoint)
    
    def _load_checkpoint(self, checkpoint_file):
        """Load checkpoint status from file."""
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        return {}
    
    def _save_checkpoint(self, checkpoint_file, status):
        """Save checkpoint status to file."""
        try:
            # Robust tuple-to-list conversion that handles nested structures
            def deep_convert_tuples(obj):
                """Recursively convert all tuples to lists for JSON serialization."""
                if isinstance(obj, tuple):
                    return [deep_convert_tuples(item) for item in obj]
                elif isinstance(obj, list):
                    return [deep_convert_tuples(item) for item in obj]
                elif isinstance(obj, dict):
                    return {str(k): deep_convert_tuples(v) for k, v in obj.items()}
                else:
                    return obj
            
            # Apply deep conversion to the entire status object
            converted_status = deep_convert_tuples(status)
            
            with open(checkpoint_file, 'w') as f:
                json.dump(converted_status, f, indent=2, default=str)
            logger.info(f"✅ Checkpoint saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint {checkpoint_file}: {e}")
            # Print the problematic data for debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Status data keys: {list(status.keys())}")
            for key, value in status.items():
                logger.error(f"  {key}: {type(value)}")
    
    def _get_config_hash(self, config):
        """Generate hash of configuration for comparison."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    # Data Generation Checkpointing
    def is_data_generated(self, dataset_size, image_size, config_hash=None):
        """Check if synthetic data is already generated with same parameters."""
        if not self.data_status:
            return False
        
        # Check if parameters match
        return (self.data_status.get('dataset_size') == dataset_size and
                self.data_status.get('image_size') == image_size and
                self.data_status.get('config_hash') == config_hash and
                self.data_status.get('completed', False))
    
    def mark_data_generated(self, dataset_size, image_size, output_dir, num_annotations, config_hash=None):
        """Mark synthetic data generation as completed."""
        # Convert tuples to lists for JSON serialization
        if isinstance(image_size, tuple):
            image_size = list(image_size)
        
        self.data_status = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'dataset_size': dataset_size,
            'image_size': image_size,
            'output_dir': str(output_dir),
            'num_annotations': num_annotations,
            'config_hash': config_hash
        }
        self._save_checkpoint(self.data_checkpoint, self.data_status)
        logger.info(f"✅ Data generation checkpoint saved: {dataset_size} images")
    
    def get_data_info(self):
        """Get information about generated data."""
        if self.data_status.get('completed'):
            return {
                'dataset_size': self.data_status.get('dataset_size'),
                'output_dir': self.data_status.get('output_dir'),
                'timestamp': self.data_status.get('timestamp'),
                'num_annotations': self.data_status.get('num_annotations', 0)
            }
        return None
    
    # Training Checkpointing
    def is_training_completed(self, config):
        """Check if training is already completed with same configuration."""
        if not self.training_status:
            return False
        
        config_hash = self._get_config_hash(config)
        return (self.training_status.get('config_hash') == config_hash and
                self.training_status.get('completed', False))
    
    def mark_training_completed(self, config, best_checkpoint, metrics, model_path=None):
        """Mark training as completed."""
        config_hash = self._get_config_hash(config)
        self.training_status = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'config_hash': config_hash,
            'best_checkpoint': str(best_checkpoint) if best_checkpoint else None,
            'model_path': str(model_path) if model_path else None,
            'metrics': metrics,
            'epochs_trained': config.get('training', {}).get('epochs', 0)
        }
        self._save_checkpoint(self.training_checkpoint, self.training_status)
        logger.info(f"✅ Training checkpoint saved: {best_checkpoint}")
    
    def get_training_info(self):
        """Get information about completed training."""
        if self.training_status.get('completed'):
            return {
                'best_checkpoint': self.training_status.get('best_checkpoint'),
                'model_path': self.training_status.get('model_path'),
                'metrics': self.training_status.get('metrics'),
                'timestamp': self.training_status.get('timestamp'),
                'epochs_trained': self.training_status.get('epochs_trained', 0)
            }
        return None
    
    # Evaluation Checkpointing
    def is_evaluation_completed(self, model_checkpoint, eval_config_hash=None):
        """Check if evaluation is already completed for this model."""
        if not self.evaluation_status:
            return False
        
        return (self.evaluation_status.get('model_checkpoint') == str(model_checkpoint) and
                self.evaluation_status.get('eval_config_hash') == eval_config_hash and
                self.evaluation_status.get('completed', False))
    
    def mark_evaluation_completed(self, model_checkpoint, synthetic_metrics, real_metrics, eval_config_hash=None):
        """Mark evaluation as completed."""
        self.evaluation_status = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': str(model_checkpoint),
            'eval_config_hash': eval_config_hash,
            'synthetic_metrics': synthetic_metrics,
            'real_metrics': real_metrics
        }
        self._save_checkpoint(self.evaluation_checkpoint, self.evaluation_status)
        logger.info(f"✅ Evaluation checkpoint saved for model: {model_checkpoint}")
    
    def get_evaluation_info(self):
        """Get information about completed evaluation."""
        if self.evaluation_status.get('completed'):
            return {
                'model_checkpoint': self.evaluation_status.get('model_checkpoint'),
                'synthetic_metrics': self.evaluation_status.get('synthetic_metrics'),
                'real_metrics': self.evaluation_status.get('real_metrics'),
                'timestamp': self.evaluation_status.get('timestamp')
            }
        return None
    
    # Utility methods
    def reset_checkpoints(self, stage=None):
        """Reset checkpoints for specified stage or all stages."""
        if stage is None or stage == 'data':
            self.data_status = {}
            self._save_checkpoint(self.data_checkpoint, self.data_status)
        
        if stage is None or stage == 'training':
            self.training_status = {}
            self._save_checkpoint(self.training_checkpoint, self.training_status)
        
        if stage is None or stage == 'evaluation':
            self.evaluation_status = {}
            self._save_checkpoint(self.evaluation_checkpoint, self.evaluation_status)
        
        logger.info(f"🔄 Reset checkpoints for: {stage or 'all stages'}")
    
    def get_status_summary(self):
        """Get summary of all checkpoint statuses."""
        return {
            'data_generation': {
                'completed': self.data_status.get('completed', False),
                'timestamp': self.data_status.get('timestamp'),
                'dataset_size': self.data_status.get('dataset_size')
            },
            'training': {
                'completed': self.training_status.get('completed', False),
                'timestamp': self.training_status.get('timestamp'),
                'epochs_trained': self.training_status.get('epochs_trained')
            },
            'evaluation': {
                'completed': self.evaluation_status.get('completed', False),
                'timestamp': self.evaluation_status.get('timestamp'),
                'model_checkpoint': self.evaluation_status.get('model_checkpoint')
            }
        }

    def force_checkpoint_validation(self):
        """Validate that checkpoint files and referenced files exist."""
        # Validate data checkpoint
        if self.data_status.get('completed'):
            output_dir = Path(self.data_status.get('output_dir', ''))
            if not output_dir.exists():
                logger.warning(f"Data output directory missing: {output_dir}. Resetting data checkpoint.")
                self.reset_checkpoints('data')
        
        # Validate training checkpoint
        if self.training_status.get('completed'):
            checkpoint_path = self.training_status.get('best_checkpoint')
            if checkpoint_path and not Path(checkpoint_path).exists():
                logger.warning(f"Training checkpoint missing: {checkpoint_path}. Resetting training checkpoint.")
                self.reset_checkpoints('training')
        
        # Validate evaluation checkpoint
        if self.evaluation_status.get('completed'):
            model_path = self.evaluation_status.get('model_checkpoint')
            if model_path and not Path(model_path).exists():
                logger.warning(f"Model checkpoint missing: {model_path}. Resetting evaluation checkpoint.")
                self.reset_checkpoints('evaluation')
