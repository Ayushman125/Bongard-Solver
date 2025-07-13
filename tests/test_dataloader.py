# Folder: tests/
# File: tests/test_dataloader.py
import pytest
import torch
import numpy as np
import os
import json
import logging

# Configure logging for tests
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Dummy configuration for testing purposes
# This should mirror the structure of your actual config.py
class DummyConfig:
    def __init__(self):
        self.data = {
            'image_size': 64, # Small image size for faster tests
            'synthetic_data_config': {
                'max_support_images_per_problem': 2, # Small number for tests
                'num_train_problems': 10,
                'num_val_problems': 5
            },
            'dataloader_workers': 0, # 0 workers for simpler debugging in tests
            'use_synthetic_data': True, # Start with synthetic for simplicity
            'use_dali': False, # Will be overridden in tests to check both
            'real_data_config': {
                'dataset_path': './data_test', # Dummy path
                'dataset_name': 'dummy_real',
                'train_split': 0.8
            },
            'data_root_path': './data_test'
        }
        self.training = {
            'batch_size': 2, # Small batch size for tests
            'curriculum_learning': False,
            'curriculum_config': {'difficulty_sampling': False, 'beta_anneal_epochs': 1},
            'augmentation_config': {} # No augmentations for simple shape validation
        }
        self.dali = {
            'num_threads': 1,
            'device_id': 0,
            'queue_size': 1,
            'monitor_interval': 0.1,
            'erase_fill': 0.0,
            'erase_prob': 0.0,
            'mixup_prob': 0.0,
            'mixup_alpha': 0.0
        }
    
    def get(self, key, default=None):
        # Allow dictionary-like access for nested configs
        keys = key.split('.')
        val = self
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k, default)
            elif hasattr(val, k):
                val = getattr(val, k)
            else:
                return default
        return val

# Mock the necessary imports from data.py
# In a real test setup, you would import directly if paths are configured.
# For standalone test file, we mock or provide minimal implementations.
try:
    from data import build_pt_loader, build_dali_loader, BongardSyntheticDataset, RealBongardDataset, BongardGenerator, load_bongard_data, custom_collate_fn, HAS_DALI
    from bongard_rules import ALL_BONGARD_RULES # Assuming this exists
except ImportError:
    logger.warning("Could not import data loading components directly. Providing mock implementations for testing.")
    # Mock classes/functions if imports fail (e.g., when running tests in isolation)
    HAS_DALI = False # Assume DALI is not available for mock
    
    class MockBongardGenerator:
        def __init__(self, data_config, all_bongard_rules): pass
        def generate_problem(self):
            img_size = 64
            # Mimic the output of the real BongardGenerator
            return (np.zeros((img_size, img_size, 3), dtype=np.uint8), # query_img1_np
                    np.zeros((img_size, img_size, 3), dtype=np.uint8), # query_img2_np
                    0, # label
                    b'{}', b'{}', # gt_json_view1, gt_json_view2
                    0.5, # difficulty
                    np.eye(3).tolist(), np.eye(3).tolist(), # affine1, affine2
                    0, # original_index
                    [np.zeros((img_size, img_size, 3), dtype=np.uint8)] * 2, # padded_support_imgs_np
                    [-1, -1], # padded_support_labels
                    [b'{}', b'{}'], # padded_support_sgs_bytes
                    torch.tensor(0, dtype=torch.long), # num_support_per_problem
                    torch.tensor(0, dtype=torch.long), # tree_indices
                    torch.tensor(1.0, dtype=torch.float)) # is_weights

    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, cfg, generator=None, num_samples=10, data_list=None, transform=None):
            self.num_samples = num_samples if data_list is None else len(data_list)
            self.generator = generator if generator else MockBongardGenerator(cfg.data, [])
            self.transform = transform
            self.cfg = cfg
            self.data_list = data_list
            self.replay_buffer = None # For HardExampleSampler testing
            if cfg.training['curriculum_learning'] and cfg.training['curriculum_config']['difficulty_sampling']:
                class MockReplayBuffer:
                    def __init__(self, capacity): self.buffer = []; self.priorities = np.zeros(capacity); self.position = 0
                    def add(self, sample, original_index, initial_priority=1.0):
                        if len(self.buffer) < self.capacity: self.buffer.append(sample)
                        else: self.buffer[self.position] = sample
                        self.priorities[self.position] = initial_priority
                        self.position = (self.position + 1) % self.capacity
                    def sample(self, batch_size, beta=0.4): return [0]*batch_size, np.arange(batch_size), np.ones(batch_size)
                    def update_priorities(self, original_indices, errors): pass
                    def __len__(self): return len(self.buffer)
                self.replay_buffer = MockReplayBuffer(num_samples * 2)
                for i in range(num_samples): self.replay_buffer.add(i, i)

        def __len__(self): return self.num_samples
        def __getitem__(self, idx):
            if isinstance(idx, tuple): # From HardExampleSampler
                original_idx, is_weight = idx
            else:
                original_idx = idx
                is_weight = 1.0
            
            if self.data_list: # Real dataset mock
                problem = self.data_list[original_idx]
                # Mimic RealBongardDataset output
                img_size = self.cfg.data['image_size']
                query_img1_np = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                query_img2_np = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                
                if self.transform:
                    query_img1_tensor = self.transform(Image.fromarray(query_img1_np))
                    query_img2_tensor = self.transform(Image.fromarray(query_img2_np))
                    padded_support_imgs_tensor = torch.stack([self.transform(Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))) for _ in range(self.cfg.data['synthetic_data_config']['max_support_images_per_problem'])])
                else:
                    query_img1_tensor = query_img1_np
                    query_img2_tensor = query_img2_np
                    padded_support_imgs_tensor = np.stack([np.zeros((img_size, img_size, 3), dtype=np.uint8)] * self.cfg.data['synthetic_data_config']['max_support_images_per_problem'])

                return (query_img1_tensor, query_img2_tensor, 0, # label
                        b'{}', b'{}', # gt_json_view1, gt_json_view2
                        0.5, # difficulty
                        np.eye(3).tolist(), np.eye(3).tolist(), # affine1, affine2
                        original_idx, # original_index
                        padded_support_imgs_tensor,
                        [-1]*self.cfg.data['synthetic_data_config']['max_support_images_per_problem'], # padded_support_labels
                        [b'{}']*self.cfg.data['synthetic_data_config']['max_support_images_per_problem'], # padded_support_sgs_bytes
                        torch.tensor(0, dtype=torch.long), # num_support_per_problem
                        torch.tensor(original_idx, dtype=torch.long), # tree_indices
                        torch.tensor(is_weight, dtype=torch.float)) # is_weights

            else: # Synthetic dataset mock
                item = self.generator.generate_problem()
                item_list = list(item)
                if self.transform:
                    item_list[0] = self.transform(Image.fromarray(item_list[0]))
                    item_list[1] = self.transform(Image.fromarray(item_list[1]))
                    transformed_support_imgs = []
                    for img_np in item_list[9]:
                        transformed_support_imgs.append(self.transform(Image.fromarray(img_np)))
                    item_list[9] = torch.stack(transformed_support_imgs)
                item_list[8] = original_idx # Ensure original_index is correct
                item_list[13] = torch.tensor(original_idx, dtype=torch.long)
                item_list[14] = torch.tensor(is_weight, dtype=torch.float)
                return tuple(item_list)

    build_pt_loader = lambda cfg, dataset, is_train, rank, world_size: torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training['batch_size'], shuffle=is_train, num_workers=0, collate_fn=custom_collate_fn
    )
    build_dali_loader = lambda cfg, dataset, is_train, rank, world_size: None # DALI mock
    BongardSyntheticDataset = MockDataset
    RealBongardDataset = MockDataset
    BongardGenerator = MockBongardGenerator
    ALL_BONGARD_RULES = []
    
    def load_bongard_data(dataset_path, dataset_name, train_split):
        # Create dummy image files for real data mock
        dummy_image_dir = os.path.join(dataset_path, "dummy_images")
        os.makedirs(dummy_image_dir, exist_ok=True)
        for i in range(10): # 10 dummy problems
            dummy_img = Image.new('RGB', (64, 64), color = 'blue')
            dummy_img.save(os.path.join(dummy_image_dir, f"img1_{i}.png"))
        
        all_data = [{'id': i, 'label': i % 2, 'image_path_view1': os.path.join(dummy_image_dir, f"img1_{i}.png")} for i in range(10)]
        split_idx = int(len(all_data) * train_split)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        train_file_paths = [d['image_path_view1'] for d in train_data]
        val_file_paths = [d['image_path_view1'] for d in val_data]
        train_labels_np = np.array([d['label'] for d in train_data])
        val_labels_np = np.array([d['label'] for d in val_data])
        return (train_data, val_data, train_file_paths, val_file_paths, train_labels_np, val_labels_np)

    # Re-define custom_collate_fn for the mock setup if it's not imported
    def custom_collate_fn(batch):
        is_torch_tensor = isinstance(batch[0][0], torch.Tensor)
        def stack_and_convert(items, dtype=None):
            if is_torch_tensor: return torch.stack(items)
            else: return torch.tensor(np.stack(items), dtype=dtype if dtype else torch.float32).permute(0, 3, 1, 2)
        
        query_img1_tensors = stack_and_convert([item[0] for item in batch])
        query_img2_tensors = stack_and_convert([item[1] for item in batch])
        query_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        gt_json_view1_list = [item[3] for item in batch]
        gt_json_view2_list = [item[4] for item in batch]
        difficulties = torch.tensor([item[5] for item in batch], dtype=torch.float)
        affine1_list = [item[6] for item in batch]
        affine2_list = [item[7] for item in batch]
        original_indices = torch.tensor([item[8] for item in batch], dtype=torch.long)
        
        # Handle padded_support_imgs_tensors based on whether they are already tensors or numpy arrays
        if is_torch_tensor:
            padded_support_imgs_tensors = torch.stack([item[9] for item in batch])
        else:
            stacked_np = np.stack([item[9] for item in batch])
            padded_support_imgs_tensors = torch.tensor(stacked_np, dtype=torch.float32).permute(0, 1, 4, 2, 3)

        padded_support_labels_list = torch.tensor([item[10] for item in batch], dtype=torch.long)
        padded_support_sgs_bytes_list = [item[11] for item in batch]
        num_support_per_problem = torch.tensor([item[12] for item in batch], dtype=torch.long)
        tree_indices = torch.tensor([item[13] for item in batch], dtype=torch.long)
        is_weights = torch.tensor([item[14] for item in batch], dtype=torch.float)

        return {
            'query_img1': query_img1_tensors,
            'query_img2': query_img2_tensors,
            'query_labels': query_labels,
            'query_gts_json_view1': gt_json_view1_list,
            'query_gts_json_view2': gt_json_view2_list,
            'difficulties': difficulties,
            'affine1': affine1_list,
            'affine2': affine2_list,
            'original_indices': original_indices,
            'padded_support_imgs': padded_support_imgs_tensors,
            'padded_support_labels': padded_support_labels_list,
            'padded_support_sgs_bytes': padded_support_sgs_bytes_list,
            'num_support_per_problem': num_support_per_problem,
            'tree_indices': tree_indices,
            'is_weights': is_weights
        }


# Global dummy config for tests
cfg = DummyConfig()

# Test for DataLoader shapes consistency
def test_loader_shapes():
    """
    Validates that PyTorch and DALI data loaders produce batches with consistent shapes.
    This test will run for both synthetic and real data paths.
    """
    if not HAS_DALI:
        pytest.skip("NVIDIA DALI not found, skipping DALI loader tests.")

    # Test with synthetic data
    cfg.data['use_synthetic_data'] = True
    cfg.data['use_dali'] = False # PyTorch loader first
    pt_dataset_synth = BongardSyntheticDataset(cfg, BongardGenerator(cfg.data, ALL_BONGARD_RULES), num_samples=cfg.data['synthetic_data_config']['num_train_problems'], transform=T.Compose([T.ToTensor()]))
    pt_loader_synth = build_pt_loader(cfg, pt_dataset_synth, is_train=True, rank=0, world_size=1)
    
    cfg.data['use_dali'] = True # DALI loader
    dali_dataset_synth = BongardSyntheticDataset(cfg, BongardGenerator(cfg.data, ALL_BONGARD_RULES), num_samples=cfg.data['synthetic_data_config']['num_train_problems'], transform=None) # DALI expects raw numpy
    dali_loader_synth = build_dali_loader(cfg, dali_dataset_synth, is_train=True, rank=0, world_size=1)

    logger.info("Testing synthetic data loader shapes...")
    pt_batch_synth = next(iter(pt_loader_synth))
    dali_batch_synth = next(dali_loader_synth) # DALI returns a dict of tensors

    # Compare shapes of 'query_img1' and 'query_labels'
    assert pt_batch_synth['query_img1'].shape == dali_batch_synth['query_img1'].shape, \
        f"Synthetic: query_img1 shape mismatch: PT {pt_batch_synth['query_img1'].shape} vs DALI {dali_batch_synth['query_img1'].shape}"
    assert pt_batch_synth['query_labels'].shape == dali_batch_synth['query_labels'].shape, \
        f"Synthetic: query_labels shape mismatch: PT {pt_batch_synth['query_labels'].shape} vs DALI {dali_batch_synth['query_labels'].shape}"
    assert pt_batch_synth['padded_support_imgs'].shape == dali_batch_synth['padded_support_imgs'].shape, \
        f"Synthetic: padded_support_imgs shape mismatch: PT {pt_batch_synth['padded_support_imgs'].shape} vs DALI {dali_batch_synth['padded_support_imgs'].shape}"
    logger.info("Synthetic data loader shapes match.")

    # Test with real data
    # Create dummy data files for real data testing
    dummy_data_root = cfg.data['data_root_path']
    os.makedirs(dummy_data_root, exist_ok=True)
    (train_data_list, val_data_list, 
     train_file_paths, val_file_paths, 
     train_labels_np, val_labels_np) = load_bongard_data(
        cfg.data['real_data_config']['dataset_path'],
        cfg.data['real_data_config']['dataset_name'],
        cfg.data['real_data_config']['train_split']
    )
    # Update global config with file lists for DALI FileReader
    cfg.data['real_data_config']['train_file_list'] = train_file_paths
    cfg.data['real_data_config']['val_file_list'] = val_file_paths
    cfg.data['real_data_config']['train_labels_np'] = train_labels_np
    cfg.data['real_data_config']['val_labels_np'] = val_labels_np

    cfg.data['use_synthetic_data'] = False
    cfg.data['use_dali'] = False # PyTorch loader first
    pt_dataset_real = RealBongardDataset(train_data_list, transform=T.Compose([T.ToTensor()]))
    pt_loader_real = build_pt_loader(cfg, pt_dataset_real, is_train=True, rank=0, world_size=1)
    
    cfg.data['use_dali'] = True # DALI loader
    dali_dataset_real = RealBongardDataset(train_data_list, transform=None) # DALI expects raw numpy
    dali_loader_real = build_dali_loader(cfg, dali_dataset_real, is_train=True, rank=0, world_size=1)

    logger.info("Testing real data loader shapes...")
    pt_batch_real = next(iter(pt_loader_real))
    dali_batch_real = next(dali_loader_real)

    # Compare shapes of 'query_img1' and 'query_labels'
    assert pt_batch_real['query_img1'].shape == dali_batch_real['query_img1'].shape, \
        f"Real: query_img1 shape mismatch: PT {pt_batch_real['query_img1'].shape} vs DALI {dali_batch_real['query_img1'].shape}"
    assert pt_batch_real['query_labels'].shape == dali_batch_real['query_labels'].shape, \
        f"Real: query_labels shape mismatch: PT {pt_batch_real['query_labels'].shape} vs DALI {dali_batch_real['query_labels'].shape}"
    # Note: padded_support_imgs might not be directly comparable if real data doesn't use it or it's handled differently
    # Assert for padded_support_imgs only if it's expected to be populated and consistent
    if 'padded_support_imgs' in pt_batch_real and 'padded_support_imgs' in dali_batch_real:
        assert pt_batch_real['padded_support_imgs'].shape == dali_batch_real['padded_support_imgs'].shape, \
            f"Real: padded_support_imgs shape mismatch: PT {pt_batch_real['padded_support_imgs'].shape} vs DALI {dali_batch_real['padded_support_imgs'].shape}"
    logger.info("Real data loader shapes match.")

