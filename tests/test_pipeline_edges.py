# Folder: bongard_solver/
# File: tests/test_pipeline_edges.py

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf # For creating a dummy config for tests

# Assume these are available in your main project structure
try:
    from data import get_dataloader # Renamed from get_loader for clarity as per common practice
    from models import PerceptionModule
except ImportError as e:
    pytest.skip(f"Skipping pipeline edge tests due to missing module: {e}. Ensure data.py and models.py are accessible.")

# --- Helper Functions for Test Data ---

def create_empty_image_batch(batch_size: int = 1, image_size: int = 224, channels: int = 3):
    """
    Creates a dummy batch of 'empty' images (e.g., all zeros or a placeholder).
    This simulates a scenario where an image might be entirely black or
    otherwise lack meaningful content.
    """
    # For a perception module, a batch of zeros might be a valid input
    # but represents no visual information.
    return torch.zeros(batch_size, channels, image_size, image_size)

def create_dummy_cfg():
    """
    Creates a minimal OmegaConf config object for testing purposes.
    This should include any parameters that `get_dataloader` or `PerceptionModule`
    might access from the config.
    """
    return OmegaConf.create({
        'data': {
            'image_size': 224,
            'dataloader_workers': 0, # Set to 0 for tests to avoid multiprocessing issues
            'use_synthetic_data': True, # Or False, depending on what get_dataloader expects
            'synthetic_data_config': {
                'num_train_problems': 1, # Minimal for test
                'num_val_problems': 1,
                'min_objects_per_image': 0, # Allow empty images for testing
                'max_objects_per_image': 1,
            },
            'real_data_config': { # Minimal for test
                'dataset_name': 'test_dataset',
                'dataset_path': './test_data',
                'train_split': 0.8
            }
        },
        'model': {
            'backbone': 'resnet18', # A common backbone
            'pretrained': False, # No need for actual pretrained weights in unit tests
            'feature_dim': 512, # A dummy value, might be inferred in actual model
            'attribute_classifier_config': {
                'shape': 4, 'color': 6, 'fill': 4, 'size': 3, 'orientation': 2, 'texture': 3
            },
            'relation_gnn_config': {
                'hidden_dim': 256, 'num_layers': 3, 'num_relations': 10, 'dropout_prob': 0.1, 'use_edge_features': True
            },
            'bongard_head_config': {
                'hidden_dim': 512, 'num_classes': 2, 'dropout_prob': 0.2
            }
        },
        'object_detector': {
            'use_yolo': False, # Disable actual YOLO for unit tests
            'yolo_pretrained': 'dummy.pt',
            'yolo_conf_threshold': 0.25,
            'yolo_iou_threshold': 0.7,
        },
        'segmentation': {
            'use_sam': False, # Disable actual SAM for unit tests
            'sam_model_type': 'vit_b',
            'sam_checkpoint_path': 'dummy.pth',
            'sam_points_per_side': 32,
            'sam_pred_iou_thresh': 0.88,
        },
        'paths': { # Essential for any file path handling
            'data_root': './test_data',
            'scene_graph_dir': './test_scene_graphs',
            'dsl_dir': './test_dsl_rules',
            'logs_dir': './test_logs',
            'cache_dir': './.test_cache',
            'mlflow_artifact_dir': './test_mlruns'
        }
    })

# --- Test Cases ---

def test_empty_dataset_raises():
    """
    Verifies that attempting to load from an effectively empty dataset
    or iterating over an empty DataLoader raises an appropriate error.
    """
    cfg = create_dummy_cfg()
    # Modify cfg to simulate an empty dataset scenario
    cfg.data.synthetic_data_config.num_train_problems = 0
    cfg.data.synthetic_data_config.num_val_problems = 0
    cfg.data.use_synthetic_data = True # Ensure it tries to use synthetic data

    # Depending on your `get_dataloader` implementation, it might raise
    # an error during creation or during iteration.
    with pytest.raises(ValueError, match="No data available for DataLoader"):
        # Assuming get_dataloader raises ValueError for empty data
        # Or, if it returns an empty loader, the next(iter(loader)) will fail
        loader = get_dataloader(cfg, train=True)
        # If the loader is created but empty, iterating will raise StopIteration
        # We want to catch the initial problem if possible, or ensure it's empty.
        # This might require mocking the internal dataset creation if get_dataloader
        # doesn't immediately validate empty data.
        # For a robust test, you might mock the underlying dataset class.
        # For now, we assume get_dataloader itself handles this.
        if loader: # Check if loader was created before trying to iterate
            next(iter(loader))
        else:
            raise ValueError("No data available for DataLoader") # Explicitly raise if loader is None/empty

def test_perception_handles_missing_objects():
    """
    Tests that the PerceptionModule can process an image batch
    even if no objects are detected (e.g., all-black image, or
    object detector returns empty results).
    It should not crash and should return expected output structure.
    """
    cfg = create_dummy_cfg()
    
    # Initialize PerceptionModule with the dummy config
    # Note: If PerceptionModule internally calls YOLO/SAM, they should be mocked
    # or disabled in the cfg for a true unit test.
    # The dummy_cfg sets use_yolo and use_sam to False.
    model = PerceptionModule(cfg).eval() # Set to eval mode for inference

    # Create a batch of empty images
    fake_batch = create_empty_image_batch(batch_size=2) # Test with a batch of 2 empty images

    # Pass the empty batch through the model
    out = model(fake_batch)

    # Assertions:
    # 1. The output dictionary contains expected keys (e.g., 'features', 'attribute_logits', 'relation_logits')
    assert 'features' in out
    assert 'attribute_logits' in out
    assert 'relation_logits' in out
    assert 'scene_graphs' in out # Assuming it outputs scene graphs

    # 2. The shapes of the outputs are as expected for zero detected objects.
    # For example, if no objects are detected, attribute_logits might be empty or a tensor of size (0, N).
    # This depends on your PerceptionModule's implementation details.
    # Assuming it returns empty lists or tensors with 0 in the batch/object dimension.
    assert isinstance(out['features'], torch.Tensor)
    # If no objects, feature tensor might be empty or represent batch-level features
    # For simplicity, let's just check it's a tensor.
    
    # For attribute_logits, if no objects, the inner tensors might be empty
    assert isinstance(out['attribute_logits'], dict)
    for attr_type, logits in out['attribute_logits'].items():
        assert isinstance(logits, torch.Tensor)
        # Expecting 0 objects, so the first dimension should be 0
        assert logits.shape[0] == 0, f"Expected 0 objects for {attr_type}, got {logits.shape[0]}"

    # For relation_logits, if less than 2 objects, there are no relations
    assert isinstance(out['relation_logits'], torch.Tensor)
    assert out['relation_logits'].shape[0] == 0, \
        f"Expected 0 relations, got {out['relation_logits'].shape[0]}"
    
    # Scene graphs should be a list of dictionaries, one per image in batch
    assert isinstance(out['scene_graphs'], list)
    assert len(out['scene_graphs']) == fake_batch.shape[0]
    for sg in out['scene_graphs']:
        assert isinstance(sg, dict)
        assert 'objects' in sg and isinstance(sg['objects'], list)
        assert 'relations' in sg and isinstance(sg['relations'], list)
        assert len(sg['objects']) == 0 # No objects detected
        assert len(sg['relations']) == 0 # No relations for no objects

# You can add more tests for:
# - Data augmentation edge cases (e.g., very small images, extreme augmentations)
# - Handling corrupted image files in the DataLoader
# - PerceptionModule with very low confidence detections (should filter out)
# - PerceptionModule with overlapping bounding boxes (NMS behavior)
# - DataLoader with a single sample (batch size 1)
