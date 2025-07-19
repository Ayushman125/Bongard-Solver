# Folder: bongard_solver/
# File: test_system1_implementation.py
"""
Test script to validate System 1 Mask R-CNN implementation.
This script tests the core components before running the full training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_maskrcnn_detector():
    """Test MaskRCNNDetector class."""
    logger.info("Testing MaskRCNNDetector...")
    
    try:
        from core_models.models import MaskRCNNDetector
        
        # Initialize detector
        detector = MaskRCNNDetector(num_classes=6, confidence_threshold=0.7)
        logger.info(f"✅ MaskRCNNDetector initialized with {detector.num_classes} classes")
        
        # Test forward pass
        batch_size = 2
        test_images = torch.randn(batch_size, 3, 512, 512)
        
        # Inference mode
        detector.eval()
        with torch.no_grad():
            results = detector(test_images)
        
        logger.info(f"✅ Forward pass successful, got {len(results['detections'])} detection results")
        
        # Test feature extraction
        bboxes, masks = detector.extract_features_for_batch(test_images)
        logger.info(f"✅ Feature extraction successful: {len(bboxes)} bbox lists, {len(masks)} mask lists")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MaskRCNNDetector test failed: {e}")
        return False

def test_perception_module():
    """Test PerceptionModule with integrated detector."""
    logger.info("Testing PerceptionModule...")
    
    try:
        from core_models.models import PerceptionModule
        from config import CONFIG
        
        # Create comprehensive config for testing
        test_config = {
            'model': {
                'detector_config': {'num_classes': 6, 'confidence_threshold': 0.7},
                'backbone': 'mobilenet_v2',
                'pretrained': True,
                'feature_dim': 256,
                'use_scene_gnn': False,
                'relation_gnn_config': {
                    'hidden_dim': 256, 
                    'num_layers': 2, 
                    'num_relations': 4,
                    'dropout_prob': 0.1,
                    'norm': 'batch',
                    'activation': 'relu',
                    'global_pool': 'mean'
                },
                'bongard_head_config': {
                    'num_classes': 2,
                    'attn_dim': 256,
                    'hidden_dim': 256,
                    'dropout_prob': 0.1
                },
                'attribute_classifier_config': {'mlp_dim': 256, 'head_dropout_prob': 0.3}
            },
            'data': {
                'image_size': [224, 224]  # Required for PerceptionModule
            }
        }
        
        # Convert dict to object with dot notation
        class DictConfig:
            def __init__(self, d):
                self._data = d
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, DictConfig(v))
                    else:
                        setattr(self, k, v)
                        
            def get(self, key, default=None):
                """Support .get() method for compatibility"""
                return getattr(self, key, default)
                
            def items(self):
                """Support .items() method for compatibility"""
                return self._data.items()
                
            def __contains__(self, key):
                """Support 'in' operator"""
                return key in self._data
                
            def __getitem__(self, key):
                """Support subscript notation"""
                return getattr(self, key)
                
            def __setitem__(self, key, value):
                """Support subscript assignment"""
                setattr(self, key, value)
        
        config = DictConfig(test_config)
        
        # Initialize PerceptionModule
        model = PerceptionModule(config)
        logger.info("✅ PerceptionModule initialized successfully")
        
        # Test forward pass with automatic detection
        batch_size = 1
        test_images = torch.randn(batch_size, 3, 512, 512)
        
        model.eval()
        with torch.no_grad():
            # Test without providing external detections (should use integrated detector)
            results = model(test_images)
        
        logger.info("✅ PerceptionModule forward pass with auto-detection successful")
        logger.info(f"Results keys: {list(results.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PerceptionModule test failed: {e}")
        return False

def test_detection_dataset():
    """Test BongardDetectionDataset class."""
    logger.info("Testing BongardDetectionDataset...")
    
    try:
        from src.data.bongard_logo_detection import BongardDetectionDataset
        import tempfile
        import json
        
        # Create dummy test data
        dummy_annotations = {
            'images': [
                {'id': 0, 'file_name': 'test_image.jpg', 'height': 224, 'width': 224}
            ],
            'annotations': [
                {
                    'id': 1, 'image_id': 0, 'category_id': 1,
                    'bbox': [50, 50, 100, 100], 'area': 10000, 'iscrowd': 0,
                    'segmentation': [[50, 50, 150, 50, 150, 150, 50, 150]]
                }
            ],
            'categories': [
                {'id': 1, 'name': 'triangle'},
                {'id': 2, 'name': 'quadrilateral'}
            ]
        }
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dummy_annotations, f)
            ann_file = f.name
        
        # Create a dummy image
        from PIL import Image
        dummy_img = Image.new('RGB', (224, 224), color='red')
        with tempfile.TemporaryDirectory() as temp_dir:
            img_path = Path(temp_dir) / 'test_image.jpg'
            dummy_img.save(img_path)
            
            # Test dataset creation
            dataset = BongardDetectionDataset(
                annotations_file=ann_file,
                images_dir=temp_dir,
                transform_config={'image_size': [224, 224], 'augmentation': False}
            )
            
            logger.info(f"✅ Dataset created with {len(dataset)} samples")
            
            # Test getting an item
            image, target = dataset[0]
            logger.info(f"✅ Dataset item access successful: image {image.shape}, target keys: {target.keys()}")
        
        # Clean up
        Path(ann_file).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BongardDetectionDataset test failed: {e}")
        return False

def test_synthetic_data_generation():
    """Test synthetic data generation."""
    logger.info("Testing synthetic data generation...")
    
    try:
        from src.bongard_generator.dataset import SyntheticBongardDataset
        
        # Create small synthetic dataset with correct rule names
        rules = [
            ('SHAPE(TRIANGLE)', 2),  # Rule description and count
            ('FILL(SOLID)', 1)
        ]
        
        dataset = SyntheticBongardDataset(
            rules=rules,
            img_size=224,
            grayscale=False
        )
        
        logger.info(f"✅ Synthetic dataset created with {len(dataset)} samples")
        
        # Test accessing samples
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            image = example['image']
            label = example['label'] 
            logger.info(f"Sample {i}: image type {type(image)}, label {label}, rule {example['rule']}")
        
        logger.info("✅ Synthetic data generation test successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Synthetic data generation test failed: {e}")
        return False

def run_all_tests():
    """Run all component tests."""
    logger.info("🚀 Starting System 1 Implementation Tests...")
    
    tests = [
        ("MaskRCNNDetector", test_maskrcnn_detector),
        ("PerceptionModule", test_perception_module),
        ("BongardDetectionDataset", test_detection_dataset),
        ("Synthetic Data Generation", test_synthetic_data_generation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info('='*50)
        
        try:
            success = test_func()
            if success:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.info(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    logger.info(f"✅ PASSED: {passed}/{len(tests)}")
    logger.info(f"❌ FAILED: {failed}/{len(tests)}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! System 1 implementation is ready.")
        return True
    else:
        logger.error(f"⚠️ {failed} tests failed. Please fix issues before running full training.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
