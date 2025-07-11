# Folder: bongard_solver/
# File: data.py (formerly dataloader.py)
import os
import glob
import json
import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2  # For synthetic data generation and mask processing
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler
import threading  # For DALI queue monitoring
import time  # For DALI queue monitoring

# Imports for rich augmentations
from torchvision import transforms as T
try:
    from torchvision.transforms import AutoAugment, RandAugment, RandomErasing
    HAS_RICH_AUGMENTATIONS = True
    logger = logging.getLogger(__name__)
    logger.info("AutoAugment, RandAugment, RandomErasing found and enabled.")
except ImportError:
    HAS_RICH_AUGMENTATIONS = False
    logger = logging.getLogger(__name__)
    logger.warning("AutoAugment, RandAugment, RandomErasing not found. Rich augmentations will be disabled.")

# DALI imports
try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.pytorch import performance # Added for DALI timing
    HAS_DALI = True
    logger = logging.getLogger(__name__)
    logger.info("NVIDIA DALI found and enabled.")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("NVIDIA DALI not found. Falling back to PyTorch DataLoader for data loading.")
    HAS_DALI = False

# Import configuration (assuming it exists and has relevant maps)
try:
    from config import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE
    from bongard_rules import ALL_BONGARD_RULES # Assuming this is defined for BongardGenerator
    # Import pl (pytorch_lightning) if it's used for LightningDataModule
    import pytorch_lightning as pl
except ImportError:
    logger.error("Could not import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE or ALL_BONGARD_RULES from config.py/bongard_rules.py. Data loading might fail.")
    # Define dummy values if imports fail to prevent crashes
    CONFIG = {'data': {'image_size': 128, 'synthetic_data_config': {'max_support_images_per_problem': 5, 'num_train_problems': 100, 'num_val_problems': 20, 'num_test_problems': 10}, 'dataloader_workers': 0, 'use_synthetic_data': True, 'use_dali': False, 'real_data_config': {'dataset_path': '', 'dataset_name': '', 'train_split': 0.8}, 'data_root_path': './data'}, 'training': {'batch_size': 16, 'curriculum_learning': False, 'curriculum_config': {'difficulty_sampling': False, 'difficulty_update_frequency_batches': 100, 'beta_anneal_epochs': 10}, 'augmentation_config': {'use_autoaugment': False, 'use_randaugment': False, 'randaugment_num_ops': 2, 'randaugment_magnitude': 9, 'use_random_erasing': False, 'random_erasing_prob': 0.5, 'random_erasing_scale_min': 0.02, 'random_erasing_scale_max': 0.33, 'random_erasing_ratio_min': 0.3, 'random_erasing_ratio_max': 3.3, 'use_color_jitter': False, 'color_jitter_brightness': 0.8, 'color_jitter_contrast': 0.8, 'color_jitter_saturation': 0.8, 'color_jitter_hue': 0.2, 'use_gaussian_blur': False, 'gaussian_blur_sigma': 1.0, 'use_random_flip': False, 'use_random_resized_crop': False, 'random_resized_crop_area_min': 0.08, 'random_resized_crop_ratio_min': 0.75, 'random_resized_crop_ratio_max': 1.33, 'use_augmix': False, 'augmix_alpha': 0.5}}, 'dali': {'num_threads': 4, 'device_id': 0, 'queue_size': 3, 'monitor_interval': 1.0, 'erase_fill': 0.0, 'erase_prob': 0.5, 'mixup_prob': 0.5, 'mixup_alpha': 0.2}} # Added DALI specific config for dummy, and augmix
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ALL_BONGARD_RULES = [] # Dummy rule list
    import torch.nn as pl_nn # Dummy for pl.LightningDataModule
    pl = type('pl', (object,), {'LightningDataModule': object}) # Dummy pl

logger = logging.getLogger(__name__)

# --- Replay Buffer for HardExampleSampler (Simple PER-like) ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.original_index_to_buffer_pos = {} # Maps original_dataset_index to buffer_position

    def add(self, sample: Any, original_index: int, initial_priority: float = 1.0):
        # If the original_index is already in the buffer, update its priority
        if original_index in self.original_index_to_buffer_pos:
            pos = self.original_index_to_buffer_pos[original_index]
            self.buffer[pos] = sample # Update sample
            self.priorities[pos] = (initial_priority if initial_priority > 0 else self.priorities.max() if self.buffer else 1.0) ** self.alpha
        else:
            # Add new sample
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
                pos = len(self.buffer) - 1
            else:
                pos = self.position
                self.buffer[self.position] = sample
            
            self.priorities[pos] = (initial_priority if initial_priority > 0 else self.priorities.max() if self.buffer else 1.0) ** self.alpha
            self.original_index_to_buffer_pos[original_index] = pos
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        current_buffer_size = len(self.buffer)
        total_priority = self.priorities[:current_buffer_size].sum()
        
        if total_priority == 0: # Avoid division by zero if all priorities are 0
            probabilities = np.ones(current_buffer_size) / current_buffer_size
        else:
            probabilities = self.priorities[:current_buffer_size] / total_priority
        # Sample indices based on probabilities
        # Replace=True allows sampling the same item multiple times in a batch
        indices = np.random.choice(current_buffer_size, batch_size, p=probabilities, replace=True)
        
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        # IS_weight = (1 / N * 1 / P(i)) ** beta
        # P(i) = priorities[i] / total_priority
        is_weights = (1 / current_buffer_size * 1 / probabilities[indices]) ** beta
        is_weights = is_weights / is_weights.max() # Normalize for stability
        return samples, indices, is_weights

    def update_priorities(self, original_indices: List[int], errors: List[float]):
        """
        Updates the priorities in the replay buffer.
        `original_indices` are the original dataset indices.
        `errors` are the per-sample losses (used as new priorities).
        """
        for original_idx, error in zip(original_indices, errors):
            if original_idx in self.original_index_to_buffer_pos:
                buffer_pos = self.original_index_to_buffer_pos[original_idx]
                self.priorities[buffer_pos] = (abs(error) + 1e-6) ** self.alpha # Add small epsilon to avoid zero priority
            else:
                logger.warning(f"Original index {original_idx} not found in replay buffer map. Cannot update priority.")

    def __len__(self):
        return len(self.buffer)
    
    def get_original_index_map(self):
        return {v: k for k, v in self.original_index_to_buffer_pos.items()} # Reverse map for debugging

# --- Samplers ---
class HardExampleSampler(Sampler):
    """
    Samples data based on a replay buffer with prioritized experience replay.
    """
    def __init__(self, dataset: Dataset, replay_buffer: PrioritizedReplayBuffer, batch_size: int, beta_anneal_epochs: int, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True):
        self.dataset = dataset
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.beta_anneal_epochs = beta_anneal_epochs
        self.epoch = 0
        self.shuffle = shuffle # This refers to shuffling the order of batches, not within a batch
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Number of samples this rank will process per epoch
        self.num_samples_per_rank = len(self.dataset) // self.num_replicas
        # Ensure total samples are divisible by batch_size for consistent batches
        self.num_batches_per_rank = self.num_samples_per_rank // self.batch_size
        self.total_samples_this_epoch = self.num_batches_per_rank * self.batch_size
        
        # Populate replay buffer initially with uniform priorities
        if len(self.replay_buffer) == 0:
            logger.info("Initializing replay buffer for HardExampleSampler with uniform priorities.")
            for i in range(len(self.dataset)):
                # Store original_index as sample, initial priority 1.0
                self.replay_buffer.add(sample=i, original_index=i, initial_priority=1.0)
        else:
            logger.info(f"Replay buffer already initialized with {len(self.replay_buffer)} samples.")

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        # Anneal beta from initial 0.4 to 1.0 over beta_anneal_epochs
        beta = 0.4 + (1.0 - 0.4) * min(1.0, self.epoch / self.beta_anneal_epochs)
        
        # Sample `total_samples_this_epoch` indices and IS weights for this epoch
        # The `sample` method of replay_buffer returns (samples, indices_in_buffer, is_weights).
        # Here, `samples` are the original dataset indices.
        original_dataset_indices_and_weights = []
        for _ in range(self.num_batches_per_rank):
            # Sample `self.batch_size` items from the replay buffer
            # The `samples` returned by `replay_buffer.sample` are the original dataset indices
            # that were added during `replay_buffer.add(sample=original_index, ...)`.
            sampled_original_indices, _, is_weights_batch = self.replay_buffer.sample(self.batch_size, beta=beta)
            
            for i in range(self.batch_size):
                original_dataset_indices_and_weights.append((sampled_original_indices[i], is_weights_batch[i]))
        
        if self.shuffle: # Shuffle the order of batches for this epoch
            random.shuffle(original_dataset_indices_and_weights)
        
        # Distribute samples across ranks (if DDP)
        # This sampler yields (original_index, is_weight) pairs.
        # The DataLoader's worker will receive these and call dataset.__getitem__(original_index).
        # The `is_weight` needs to be passed through `__getitem__` to `custom_collate_fn`.
        
        # Divide the total samples for this rank into chunks for iteration
        for i in range(self.total_samples_this_epoch):
            yield original_dataset_indices_and_weights[i]

    def __len__(self) -> int:
        return self.total_samples_this_epoch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def update(self, original_indices: List[int], losses: List[float]):
        """
        Updates the priorities in the replay buffer.
        `original_indices` are the original dataset indices.
        `losses` are the per-sample losses.
        """
        self_indices = list(range(len(self.dataset)))
        # Filter indices to only those relevant to this rank if DDP is active
        # This is a simplification; in true DDP, each rank only updates for its own samples.
        # For this setup, the `training.py` will pass `original_indices` from the batch.
        
        self.replay_buffer.update_priorities(original_indices, losses)
        logger.debug(f"HardExampleSampler: Updated priorities for {len(original_indices)} samples.")

class CurriculumSampler(DistributedSampler):
    """
    A simple curriculum sampler that extends DistributedSampler.
    Can be expanded to implement specific curriculum strategies.
    """
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        logger.info(f"CurriculumSampler initialized for rank {self.rank}/{self.num_replicas}.")

    def __iter__(self):
        # The default DistributedSampler __iter__ provides shuffled indices for the current rank.
        # Add curriculum logic here if needed (e.g., sorting by difficulty, then sampling)
        # For now, it behaves like a standard DistributedSampler.
        return super().__iter__()

class PrototypicalSampler(DistributedSampler):
    """
    A sampler for prototypical networks, ensuring episodes are formed correctly.
    Extends DistributedSampler for DDP compatibility.
    """
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        logger.info(f"PrototypicalSampler initialized for rank {self.rank}/{self.num_replicas}.")

    def __iter__(self):
        # This would typically involve ensuring that each batch contains N-way K-shot episodes.
        # For simplicity, it currently behaves like a DistributedSampler.
        # A full implementation would group samples by class/problem type to form episodes.
        return super().__iter__()

# --- Dataset Classes ---
class BongardGenerator:
    """
    Generates synthetic Bongard problems. (Dummy for now)
    """
    def __init__(self, data_config: Dict[str, Any], all_bongard_rules: List[Any]):
        self.data_config = data_config
        self.all_bongard_rules = all_bongard_rules
        logger.info("BongardGenerator initialized (dummy).")

    def generate_problem(self) -> Dict[str, Any]:
        # This is a dummy implementation. In reality, it would generate images,
        # ground truth scene graphs, and labels.
        img_size = self.data_config['image_size']
        
        # Generate dummy images (numpy arrays)
        query_img1_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        query_img2_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        
        # Dummy labels (0 for negative, 1 for positive)
        label = random.choice([0, 1])
        
        # Dummy ground truth JSON strings (empty for now)
        gt_json_view1 = json.dumps({'objects': [], 'relations': []}).encode('utf-8')
        gt_json_view2 = json.dumps({'objects': [], 'relations': []}).encode('utf-8')
        
        # Dummy difficulties, affine matrices, original_indices
        difficulty = random.random()
        affine1 = np.eye(3).tolist()
        affine2 = np.eye(3).tolist()
        original_index = random.randint(0, 100000) # Dummy index
        
        # Dummy support images and labels for few-shot
        max_support = self.data_config['synthetic_data_config']['max_support_images_per_problem']
        num_support = random.randint(0, max_support)
        padded_support_imgs_np = [np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8) for _ in range(num_support)]
        padded_support_labels = [random.choice([0,1]) for _ in range(num_support)]
        
        # Pad support images/labels to max_support_images_per_problem
        while len(padded_support_imgs_np) < max_support:
            padded_support_imgs_np.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
            padded_support_labels.append(-1) # Use -1 for padding labels
        
        padded_support_sgs_bytes = [json.dumps({'objects': [], 'relations': []}).encode('utf-8')] * max_support
        
        num_support_per_problem = torch.tensor(num_support, dtype=torch.long)
        tree_indices = torch.tensor(original_index, dtype=torch.long) # Dummy
        is_weights = torch.tensor(1.0, dtype=torch.float) # Dummy
        
        # For MoCo, we need two augmented views of the same image.
        # Let's return the same image twice for simplicity, and augmentations will be applied by DALI/PyTorch transforms.
        # Or, if this generator is used for MoCo, it should generate two views.
        # For now, stick to the original Bongard problem structure.
        # The `training_moco.py` will handle the `q, k = x, x.flip(dims=[-1])` or similar.
        # So, we return `query_img1_np` as the primary image `x`.
        # The `get_item` will handle passing `is_weight` and `original_index`.
        return (query_img1_np, query_img2_np, label,
                gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index,
                padded_support_imgs_np, padded_support_labels, padded_support_sgs_bytes,
                num_support_per_problem, tree_indices, is_weights)

class BongardSyntheticDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], generator: BongardGenerator, num_samples: int, transform: Optional[T.Compose] = None):
        self.cfg = cfg
        self.generator = generator
        self.num_samples = num_samples
        self.transform = transform # Store transform
        self.cache = [] # To store generated problems if caching is desired
        
        # Initialize replay buffer if curriculum learning with difficulty sampling is enabled
        self.replay_buffer = None
        if self.cfg['training']['curriculum_learning'] and self.cfg['training']['curriculum_config']['difficulty_sampling']:
            buffer_capacity = num_samples * 2 # Example capacity
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
            logger.info(f"BongardSyntheticDataset: Initialized PrioritizedReplayBuffer with capacity {buffer_capacity}.")
        logger.info(f"BongardSyntheticDataset initialized with {num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: Union[int, Tuple[int, float]]) -> Tuple[Any, ...]:
        # If idx is a tuple (from HardExampleSampler), extract original index and IS weight
        if isinstance(idx, tuple):
            original_idx, is_weight = idx
        else:
            original_idx = idx
            is_weight = 1.0 # Default IS weight if not using HardExampleSampler
        
        # Generate a new problem on the fly
        problem_data = self.generator.generate_problem()
        
        # problem_data is: (query_img1_np, query_img2_np, label, gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index, padded_support_imgs_np, padded_support_labels, padded_support_sgs_bytes, num_support_per_problem, tree_indices, is_weights)
        
        # Convert problem_data to a list to modify
        problem_list = list(problem_data)
        
        # Apply transform to images (if transform is provided, i.e., PyTorch path)
        # If transform is None (i.e., DALI path), return raw numpy arrays.
        if self.transform is not None:
            # Transform query images
            problem_list[0] = self.transform(Image.fromarray(problem_list[0])) # query_img1_np
            problem_list[1] = self.transform(Image.fromarray(problem_list[1])) # query_img2_np
            
            # Transform support images
            transformed_support_imgs = []
            for img_np in problem_list[9]: # padded_support_imgs_np
                transformed_support_imgs.append(self.transform(Image.fromarray(img_np)))
            problem_list[9] = torch.stack(transformed_support_imgs) # Stack into a single tensor (N_support, C, H, W)
        # Else (self.transform is None), return raw numpy arrays for DALI to process.
        
        # Update the is_weights with the sampler's IS weight
        problem_list[-1] = torch.tensor(is_weight, dtype=torch.float) 
        problem_list[8] = original_idx # Ensure original_index is correct (used by collate_fn for tree_indices)
        problem_list[13] = torch.tensor(original_idx, dtype=torch.long) # tree_indices maps to original_index in replay buffer
        
        return tuple(problem_list)

class RealBongardDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]], transform: Optional[T.Compose] = None):
        self.data_list = data_list
        self.transform = transform
        logger.info(f"RealBongardDataset initialized with {len(data_list)} samples.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: Union[int, Tuple[int, float]]) -> Tuple[Any, ...]:
        # If idx is a tuple (from HardExampleSampler), extract original index and IS weight
        if isinstance(idx, tuple):
            original_idx, is_weight = idx
        else:
            original_idx = idx
            is_weight = 1.0 # Default IS weight if not using HardExampleSampler
        
        problem = self.data_list[original_idx]
        
        # Load images (assuming paths are in problem dict)
        # This is a placeholder, actual image loading logic would go here.
        # For now, return dummy numpy arrays.
        img_size = CONFIG['data']['image_size']
        query_img1_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        query_img2_np = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        label = problem.get('label', random.choice([0,1]))
        gt_json_view1 = json.dumps(problem.get('scene_graph_view1', {'objects': [], 'relations': []})).encode('utf-8')
        gt_json_view2 = json.dumps(problem.get('scene_graph_view2', {'objects': [], 'relations': []})).encode('utf-8')
        
        difficulty = problem.get('difficulty', 0.5)
        affine1 = problem.get('affine_matrix_view1', np.eye(3).tolist())
        affine2 = problem.get('affine_matrix_view2', np.eye(3).tolist())
        
        # Support images and labels (dummy for now)
        max_support = CONFIG['data']['synthetic_data_config']['max_support_images_per_problem']
        padded_support_imgs_np = [np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8) for _ in range(max_support)]
        padded_support_labels = [random.choice([0,1]) for _ in range(max_support)]
        padded_support_sgs_bytes = [json.dumps({'objects': [], 'relations': []}).encode('utf-8')] * max_support
        num_support_per_problem = torch.tensor(max_support, dtype=torch.long) # Dummy count
        tree_indices = torch.tensor(original_idx, dtype=torch.long) # Use original_idx as tree_index
        
        # Apply transform to images if PyTorch path
        if self.transform is not None:
            query_img1_tensor = self.transform(Image.fromarray(query_img1_np))
            query_img2_tensor = self.transform(Image.fromarray(query_img2_np))
            
            transformed_support_imgs = []
            for img_np in padded_support_imgs_np:
                transformed_support_imgs.append(self.transform(Image.fromarray(img_np)))
            padded_support_imgs_tensor = torch.stack(transformed_support_imgs)
        else: # DALI path, return raw numpy arrays
            query_img1_tensor = query_img1_np
            query_img2_tensor = query_img2_np
            padded_support_imgs_tensor = np.stack(padded_support_imgs_np) # Stack numpy arrays
        
        return (query_img1_tensor, query_img2_tensor, label,
                gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_idx,
                padded_support_imgs_tensor, padded_support_labels, padded_support_sgs_bytes,
                num_support_per_problem, tree_indices, torch.tensor(is_weight, dtype=torch.float))

def load_bongard_data(dataset_path: str, dataset_name: str, train_split: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Dummy function to load real Bongard data.
    In a real scenario, this would parse JSON files or similar.
    It also returns file paths and labels for DALI's FileReader.
    """
    logger.info(f"Loading dummy real Bongard data from {dataset_path}/{dataset_name}.")
    all_data = []
    all_file_paths = []
    all_labels = []
    
    # Create a dummy directory for images if it doesn't exist
    dummy_image_dir = os.path.join(dataset_path, "dummy_images")
    os.makedirs(dummy_image_dir, exist_ok=True)

    for i in range(100): # 100 dummy problems
        label = random.choice([0, 1])
        
        # Create dummy image files
        dummy_img_path1 = os.path.join(dummy_image_dir, f"img1_{i}.png")
        dummy_img_path2 = os.path.join(dummy_image_dir, f"img2_{i}.png")
        
        # Create dummy image content (e.g., a blank image)
        dummy_img = Image.new('RGB', (CONFIG['data']['image_size'], CONFIG['data']['image_size']), color = 'red')
        dummy_img.save(dummy_img_path1)
        dummy_img.save(dummy_img_path2)

        all_data.append({
            'id': i,
            'label': label,
            'image_path_view1': dummy_img_path1,
            'image_path_view2': dummy_img_path2,
            'scene_graph_view1': {'objects': [{'id': 0, 'shape': 'circle'}], 'relations': []},
            'scene_graph_view2': {'objects': [{'id': 0, 'shape': 'square'}], 'relations': []},
            'difficulty': random.random()
        })
        all_file_paths.append(dummy_img_path1) # DALI FileReader expects a list of image paths
        all_labels.append(label)
    
    # Shuffle data consistently
    combined = list(zip(all_data, all_file_paths, all_labels))
    random.shuffle(combined)
    all_data, all_file_paths, all_labels = zip(*combined)
    all_data = list(all_data)
    all_file_paths = list(all_file_paths)
    all_labels = np.array(all_labels)

    split_idx = int(len(all_data) * train_split)
    
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    train_file_paths = all_file_paths[:split_idx]
    val_file_paths = all_file_paths[split_idx:]
    train_labels_np = all_labels[:split_idx]
    val_labels_np = all_labels[split_idx:]

    logger.info(f"Loaded {len(train_data)} train and {len(val_data)} validation dummy real Bongard problems.")
    return (train_data, val_data, train_file_paths, val_file_paths, train_labels_np, val_labels_np)

# --- Custom Collate Function ---
def custom_collate_fn(batch: List[Tuple[Any, ...]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length lists and tensors
    from the dataset. Assumes images are already transformed to tensors by __getitem__
    for PyTorch path, or are raw numpy arrays for DALI path.
    """
    # Batch structure from dataset:
    # (query_img1_tensor, query_img2_tensor, label,
    #  gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index,
    #  padded_support_imgs_tensor, padded_support_labels, padded_support_sgs_bytes,
    #  num_support_per_problem, tree_indices, is_weights)
    
    # Separate components and stack tensors
    # For PyTorch path, these are already tensors. For DALI path, they are numpy arrays.
    # We will convert numpy arrays to tensors here, as DALI's DALIGenericIterator
    # handles the conversion itself. This collate function is primarily for PyTorch DataLoader.
    
    # Check if the first item in the batch for images is a torch.Tensor or numpy.ndarray
    is_torch_tensor = isinstance(batch[0][0], torch.Tensor)

    def stack_and_convert(items, dtype=None):
        if is_torch_tensor:
            return torch.stack(items)
        else: # numpy array, convert to tensor
            stacked_np = np.stack(items)
            return torch.tensor(stacked_np, dtype=dtype if dtype else torch.float32).permute(0, 3, 1, 2) # HWC to NCHW for images

    query_img1_tensors = stack_and_convert([item[0] for item in batch])
    query_img2_tensors = stack_and_convert([item[1] for item in batch])
    
    query_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    gt_json_view1_list = [item[3] for item in batch]
    gt_json_view2_list = [item[4] for item in batch]
    difficulties = torch.tensor([item[5] for item in batch], dtype=torch.float)
    affine1_list = [item[6] for item in batch]
    affine2_list = [item[7] for item in batch]
    original_indices = torch.tensor([item[8] for item in batch], dtype=torch.long) # Original indices from dataset
    
    # padded_support_imgs_tensor is already (N_support, C, H, W) per item for PyTorch,
    # or (N_support, H, W, C) numpy for DALI.
    # For PyTorch, stack directly. For DALI, stack numpy and convert.
    if is_torch_tensor:
        padded_support_imgs_tensors = torch.stack([item[9] for item in batch]) # Result: (B, N_support, C, H, W)
    else:
        # Stack numpy arrays and convert to tensor (B, N_support, H, W, C) -> (B, N_support, C, H, W)
        stacked_np = np.stack([item[9] for item in batch])
        padded_support_imgs_tensors = torch.tensor(stacked_np, dtype=torch.float32).permute(0, 1, 4, 2, 3)

    padded_support_labels_list = torch.tensor([item[10] for item in batch], dtype=torch.long)
    padded_support_sgs_bytes_list = [item[11] for item in batch]
    num_support_per_problem = torch.tensor([item[12] for item in batch], dtype=torch.long)
    tree_indices = torch.tensor([item[13] for item in batch], dtype=torch.long) # Indices in replay buffer tree
    is_weights = torch.tensor([item[14] for item in batch], dtype=torch.float) # Importance sampling weights

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

# --- DALI Pipeline and Loader ---
@fn.pipeline_def
def dali_pipeline_synthetic(image_size: int, is_training: bool, curriculum_config: Dict[str, Any], augmentation_config: Dict[str, Any], cfg: Dict[str, Any]):
    # Use external_source to get data from Python side for synthetic data.
    query_img1_raw = fn.external_source(name="query_img1", device="cpu")
    query_img2_raw = fn.external_source(name="query_img2", device="cpu")
    padded_support_imgs_raw = fn.external_source(name="padded_support_imgs", device="cpu")
    query_labels = fn.external_source(name="query_labels", device="cpu")
    query_gts_json_view1 = fn.external_source(name="query_gts_json_view1", device="cpu")
    query_gts_json_view2 = fn.external_source(name="query_gts_json_view2", device="cpu")
    difficulties = fn.external_source(name="difficulties", device="cpu")
    affine1 = fn.external_source(name="affine1", device="cpu")
    affine2 = fn.external_source(name="affine2", device="cpu")
    original_indices = fn.external_source(name="original_indices", device="cpu")
    padded_support_labels = fn.external_source(name="padded_support_labels", device="cpu")
    padded_support_sgs_bytes = fn.external_source(name="padded_support_sgs_bytes", device="cpu")
    num_support_per_problem = fn.external_source(name="num_support_per_problem", device="cpu")
    tree_indices = fn.external_source(name="tree_indices", device="cpu")
    is_weights = fn.external_source(name="is_weights", device="cpu")

    # Image processing function within DALI pipeline (for synthetic data)
    def process_image_dali_synthetic_base(img_raw):
        # Resize and normalize (common steps)
        output_images = fn.resize(img_raw, resize_x=image_size, resize_y=image_size)
        output_images = fn.cast(output_images, dtype=types.FLOAT)
        output_images = fn.crop_mirror_normalize(
            output_images,
            dtype=types.FLOAT,
            output_layout=types.NCHW, # Convert to NCHW for PyTorch
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
        return output_images

    # Define augmentation branches for AugMix
    def apply_augmix_branch1(img):
        # Apply color jitter
        if augmentation_config.get('use_color_jitter', False):
            img = fn.color_twist(img,
                                 brightness=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_brightness'], 1.0 + augmentation_config['color_jitter_brightness']]),
                                 contrast=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_contrast'], 1.0 + augmentation_config['color_jitter_contrast']]),
                                 saturation=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_saturation'], 1.0 + augmentation_config['color_jitter_saturation']]),
                                 hue=fn.random.uniform(range=[-augmentation_config['color_jitter_hue'], augmentation_config['color_jitter_hue']]))
        # Apply random flip
        if augmentation_config.get('use_random_flip', False):
            img = fn.coin_flip(img, probability=0.5)
        return img

    def apply_augmix_branch2(img):
        # Apply gaussian blur
        if augmentation_config.get('use_gaussian_blur', False):
            img = fn.gaussian_blur(img, sigma=augmentation_config['gaussian_blur_sigma'])
        # Apply RandomErasing
        if augmentation_config.get('use_random_erasing', False):
            img = fn.erase(img,
                           anchor=fn.random.uniform(range=[0.0, 1.0], shape=[2]),
                           shape=fn.random.uniform(range=[augmentation_config['random_erasing_scale_min'], augmentation_config['random_erasing_scale_max']], shape=[2]),
                           axis_names="HW",
                           fill_value=cfg['dali']['erase_fill'],
                           normalized=True,
                           output_dtype=types.UINT8,
                           axis=[0,1])
        return img

    # Process query images
    processed_query_img1 = process_image_dali_synthetic_base(query_img1_raw)
    processed_query_img2 = process_image_dali_synthetic_base(query_img2_raw)
    
    # Process support images (assuming it's a batch of images)
    processed_padded_support_imgs = process_image_dali_synthetic_base(padded_support_imgs_raw)

    if is_training and augmentation_config.get('use_augmix', False):
        logger.debug("Applying AugMix to synthetic data.")
        aug1_view1 = apply_augmix_branch1(query_img1_raw) # Apply to raw image before base processing
        aug2_view1 = apply_augmix_branch2(query_img1_raw) # Apply to raw image before base processing
        
        # Blend the two augmented branches, then apply base processing
        blended_query_img1_raw = fn.blend(aug1_view1, aug2_view1, alpha=augmentation_config.get('augmix_alpha', 0.5))
        processed_query_img1 = process_image_dali_synthetic_base(blended_query_img1_raw)
        # AugMix is typically applied to a single input, not necessarily both views for Bongard.
        # For query_img2, we can apply standard augmentations or none.
        # For simplicity, let's keep query_img2 processing as is, or apply a different mix.
        # If `use_mixup_cutmix` is also true, AugMix will take precedence for query_img1.
    elif is_training and augmentation_config.get('use_mixup_cutmix', False):
        logger.debug("Applying MixUp/CutMix to synthetic data.")
        # MixUp/CutMix logic as previously implemented
        mixed_query_img1 = fn.mix(processed_query_img1, processed_query_img2,
                                  prob=cfg['dali']['mixup_prob'], alpha=cfg['dali']['mixup_alpha'])
        processed_query_img1 = mixed_query_img1

    return (processed_query_img1, processed_query_img2, query_labels,
            query_gts_json_view1, query_gts_json_view2, difficulties, affine1, affine2, original_indices,
            processed_padded_support_imgs, padded_support_labels, padded_support_sgs_bytes,
            num_support_per_problem, tree_indices, is_weights)

@fn.pipeline_def
def dali_pipeline_real(cfg: Dict[str, Any], is_training: bool):
    # DALI pipeline for real data using FileReader
    image_size = cfg['data']['image_size']
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    augmentation_config = cfg['training']['augmentation_config']
    
    # FileReader for image paths and labels
    images, labels = fn.readers.file(
        file_root=cfg['data']['data_root_path'],
        file_list=cfg['data']['real_data_config']['train_file_list'] if is_training else cfg['data']['real_data_config']['val_file_list'],
        labels=cfg['data']['real_data_config']['train_labels_np'] if is_training else cfg['data']['real_data_config']['val_labels_np'],
        random_shuffle=is_training,
        name="Reader"
    )
    
    # Decode images (mixed device for CPU decoding, then GPU for further ops)
    decoded_images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
    
    # Base image processing (resize, cast, normalize)
    def process_image_dali_real_base(img):
        output_images = fn.resize(img, resize_x=image_size, resize_y=image_size, device="gpu")
        output_images = fn.crop_mirror_normalize(
            output_images,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=mean,
            std=std,
            device="gpu"
        )
        return output_images

    # Define augmentation branches for AugMix (similar to synthetic)
    def apply_augmix_branch1_real(img):
        if augmentation_config.get('use_color_jitter', False):
            img = fn.color_twist(img,
                                 brightness=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_brightness'], 1.0 + augmentation_config['color_jitter_brightness']]),
                                 contrast=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_contrast'], 1.0 + augmentation_config['color_jitter_contrast']]),
                                 saturation=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_saturation'], 1.0 + augmentation_config['color_jitter_saturation']]),
                                 hue=fn.random.uniform(range=[-augmentation_config['color_jitter_hue'], augmentation_config['color_jitter_hue']]),
                                 device="gpu")
        if augmentation_config.get('use_random_flip', False):
            img = fn.flip(img, horizontal=fn.random.coin_flip(probability=0.5), device="gpu")
        return img

    def apply_augmix_branch2_real(img):
        if augmentation_config.get('use_gaussian_blur', False):
            img = fn.gaussian_blur(img, sigma=augmentation_config['gaussian_blur_sigma'], device="gpu")
        if augmentation_config.get('use_random_erasing', False):
            img = fn.erase(img,
                           anchor=fn.random.uniform(range=[0.0, 1.0], shape=[2]),
                           shape=fn.random.uniform(range=[augmentation_config['random_erasing_scale_min'], augmentation_config['random_erasing_scale_max']], shape=[2]),
                           axis_names="HW",
                           fill_value=cfg['dali']['erase_fill'],
                           normalized=True,
                           output_dtype=types.UINT8,
                           axis=[0,1])
        return img

    if is_training and augmentation_config.get('use_augmix', False):
        logger.debug("Applying AugMix to real data.")
        aug1_decoded = apply_augmix_branch1_real(decoded_images)
        aug2_decoded = apply_augmix_branch2_real(decoded_images)
        
        blended_images = fn.blend(aug1_decoded, aug2_decoded, alpha=augmentation_config.get('augmix_alpha', 0.5), device="gpu")
        final_processed_images = process_image_dali_real_base(blended_images)
    else:
        # Apply standard sequential augmentations if not using AugMix
        current_images = decoded_images
        if is_training and augmentation_config.get('use_color_jitter', False):
            current_images = fn.color_twist(current_images,
                                             brightness=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_brightness'], 1.0 + augmentation_config['color_jitter_brightness']]),
                                             contrast=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_contrast'], 1.0 + augmentation_config['color_jitter_contrast']]),
                                             saturation=fn.random.uniform(range=[1.0 - augmentation_config['color_jitter_saturation'], 1.0 + augmentation_config['color_jitter_saturation']]),
                                             hue=fn.random.uniform(range=[-augmentation_config['color_jitter_hue'], augmentation_config['color_jitter_hue']]),
                                             device="gpu")
        if is_training and augmentation_config.get('use_gaussian_blur', False):
            current_images = fn.gaussian_blur(current_images, sigma=augmentation_config['gaussian_blur_sigma'], device="gpu")
        if is_training and augmentation_config.get('use_random_flip', False):
            current_images = fn.flip(current_images, horizontal=fn.random.coin_flip(probability=0.5), device="gpu")
        if is_training and augmentation_config.get('use_random_erasing', False):
            current_images = fn.erase(current_images,
                                       anchor=fn.random.uniform(range=[0.0, 1.0], shape=[2]),
                                       shape=fn.random.uniform(range=[augmentation_config['random_erasing_scale_min'], augmentation_config['random_erasing_scale_max']], shape=[2]),
                                       axis_names="HW",
                                       fill_value=cfg['dali']['erase_fill'],
                                       normalized=True,
                                       output_dtype=types.UINT8,
                                       axis=[0,1])
        final_processed_images = process_image_dali_real_base(current_images)

    # Dummy outputs for other elements expected by collate_fn if not directly from DALI
    batch_size_dummy = cfg['training']['batch_size']
    num_support_dummy = cfg['data']['synthetic_data_config']['max_support_images_per_problem']
    dummy_query_img2 = fn.constant(0.0, shape=[image_size, image_size, 3], dtype=types.FLOAT, device="gpu")
    dummy_query_gts_json_view1 = fn.constant(np.array([b'{}']*batch_size_dummy, dtype=object), device="cpu")
    dummy_query_gts_json_view2 = fn.constant(np.array([b'{}']*batch_size_dummy, dtype=object), device="cpu")
    dummy_difficulties = fn.constant(0.5, shape=[1], dtype=types.FLOAT, device="cpu")
    dummy_affine1 = fn.constant(np.eye(3).tolist(), shape=[3,3], dtype=types.FLOAT, device="cpu")
    dummy_affine2 = fn.constant(np.eye(3).tolist(), shape=[3,3], dtype=types.FLOAT, device="cpu")
    dummy_original_indices = fn.constant(0, shape=[1], dtype=types.INT32, device="cpu")
    dummy_padded_support_imgs = fn.constant(0.0, shape=[num_support_dummy, image_size, image_size, 3], dtype=types.FLOAT, device="gpu")
    dummy_padded_support_labels = fn.constant(-1, shape=[num_support_dummy], dtype=types.INT32, device="cpu")
    dummy_padded_support_sgs_bytes = fn.constant(np.array([b'{}']*num_support_dummy, dtype=object), device="cpu")
    dummy_num_support_per_problem = fn.constant(0, shape=[1], dtype=types.INT32, device="cpu")
    dummy_tree_indices = fn.constant(0, shape=[1], dtype=types.INT32, device="cpu")
    dummy_is_weights = fn.constant(1.0, shape=[1], dtype=types.FLOAT, device="cpu")

    # Set outputs matching the custom_collate_fn structure
    return (final_processed_images, dummy_query_img2, labels, # labels from FileReader
            dummy_query_gts_json_view1, dummy_query_gts_json_view2, dummy_difficulties,
            dummy_affine1, dummy_affine2, dummy_original_indices,
            dummy_padded_support_imgs, dummy_padded_support_labels, dummy_padded_support_sgs_bytes,
            dummy_num_support_per_problem, dummy_tree_indices, dummy_is_weights)

def build_dali_loader(cfg: Dict[str, Any], dataset: Dataset, is_train: bool, rank: int, world_size: int) -> DALIGenericIterator:
    """
    Builds a DALI DataLoader using DALIGenericIterator.
    Updated with DALI fusion and monitoring.
    """
    if not HAS_DALI:
        logger.error("DALI not available. Cannot build DALI loader.")
        raise ImportError("NVIDIA DALI is not installed or configured.")
    
    batch_size = cfg['training']['batch_size']
    num_threads = cfg['data']['dataloader_workers']
    device_id = rank # Use rank for device_id in DDP
    prefetch_queue_depth = cfg['dali'].get('queue_size', 3) # Default queue size

    if cfg['data']['use_synthetic_data']:
        logger.info(f"Building DALI pipeline for synthetic data on device {device_id}.")
        dali_pipe_instance = dali_pipeline_synthetic(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            prefetch_queue_depth=prefetch_queue_depth,
            image_size=cfg['data']['image_size'],
            is_training=is_train,
            curriculum_config=cfg['training']['curriculum_config'],
            augmentation_config=cfg['training']['augmentation_config'],
            cfg=cfg # Pass full config for DALI specific params
        )
        # For synthetic data, the DALIGenericIterator needs to be fed by the dataset's __getitem__
        class ExternalSourceIteratorWrapper:
            def __init__(self, dataset_obj: Dataset):
                self.dataset = dataset_obj
                self.indices = list(range(len(self.dataset)))
                random.shuffle(self.indices) # Shuffle for training
                self.current_idx = 0

            def __iter__(self):
                self.current_idx = 0
                random.shuffle(self.indices) # Reshuffle each epoch
                return self

            def __next__(self):
                if self.current_idx >= len(self.indices):
                    raise StopIteration
                batch_data = []
                for _ in range(batch_size):
                    if self.current_idx >= len(self.indices):
                        break
                    # Get data from the PyTorch dataset (which returns raw numpy arrays for DALI path)
                    item = self.dataset[self.indices[self.current_idx]]
                    batch_data.append(item)
                    self.current_idx += 1
                
                if not batch_data:
                    raise StopIteration
                
                # Transpose and stack the data to match DALI external_source expectations
                # Assuming `item` is the tuple from `BongardSyntheticDataset.__getitem__`
                # (query_img1_np, query_img2_np, label, gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index, padded_support_imgs_np, padded_support_labels, padded_support_sgs_bytes, num_support_per_problem, tree_indices, is_weights)
                
                # Stack each component across the batch
                query_img1_batch = np.stack([d[0] for d in batch_data])
                query_img2_batch = np.stack([d[1] for d in batch_data])
                padded_support_imgs_batch = np.stack([d[9] for d in batch_data]) # (B, N_support, H, W, C)
                query_labels_batch = np.stack([d[2] for d in batch_data])
                query_gts_json_view1_batch = np.array([d[3] for d in batch_data], dtype=object)
                query_gts_json_view2_batch = np.array([d[4] for d in batch_data], dtype=object)
                difficulties_batch = np.stack([d[5] for d in batch_data])
                affine1_batch = np.stack([d[6] for d in batch_data])
                affine2_batch = np.stack([d[7] for d in batch_data])
                original_indices_batch = np.stack([d[8] for d in batch_data])
                padded_support_labels_batch = np.stack([d[10] for d in batch_data])
                padded_support_sgs_bytes_batch = np.array([d[11] for d in batch_data], dtype=object)
                num_support_per_problem_batch = np.stack([d[12].item() for d in batch_data]) # Convert tensor to scalar
                tree_indices_batch = np.stack([d[13].item() for d in batch_data])
                is_weights_batch = np.stack([d[14].item() for d in batch_data])

                # Return as a tuple in the order of `external_source` ops in `dali_pipeline_synthetic`
                return (query_img1_batch, query_img2_batch, padded_support_imgs_batch,
                        query_labels_batch, query_gts_json_view1_batch, query_gts_json_view2_batch,
                        difficulties_batch, affine1_batch, affine2_batch, original_indices_batch,
                        padded_support_labels_batch, padded_support_sgs_bytes_batch,
                        num_support_per_problem_batch, tree_indices_batch, is_weights_batch)

        # Build the DALI pipeline
        dali_pipe_instance.build()
        
        # Create the DALI iterator with the external source wrapper
        loader = DALIGenericIterator(
            dali_pipe_instance,
            output_map=[
                "query_img1", "query_img2", "padded_support_imgs",
                "query_labels", "query_gts_json_view1", "query_gts_json_view2",
                "difficulties", "affine1", "affine2", "original_indices",
                "padded_support_labels", "padded_support_sgs_bytes",
                "num_support_per_problem", "tree_indices", "is_weights"
            ],
            size=len(dataset), # Total number of samples in the dataset
            auto_reset=True,
            fill_last_batch=False,
            dynamic_shape=True,
            # Provide the Python iterable as the source for external_source ops
            # The `source` argument to DALIGenericIterator should be an instance of `ExternalSourceIteratorWrapper`.
            source=ExternalSourceIteratorWrapper(dataset)
        )
    else: # Use FileReader for real data
        logger.info(f"Building DALI pipeline for real data on device {device_id}.")
        # The `dali_pipeline_real` needs to be initialized with the actual file lists and labels.
        # These are passed to `fn.readers.file` inside the pipeline.
        # The `cfg` should contain these lists.
        
        # Ensure the config has the real data lists (populated in `BongardDataModule.setup`)
        if 'train_file_list' not in cfg['data']['real_data_config'] or \
           'val_file_list' not in cfg['data']['real_data_config']:
            logger.error("Real data file lists not found in config. Please ensure `load_bongard_data` populates them.")
            raise ValueError("Missing real data file lists for DALI FileReader.")

        dali_pipe_instance = dali_pipeline_real(
            cfg=cfg,
            is_training=is_train,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            prefetch_queue_depth=prefetch_queue_depth
        )
        dali_pipe_instance.build()
        
        loader = DALIGenericIterator(
            dali_pipe_instance,
            output_map=[
                "query_img1", "query_img2", "query_labels",
                "query_gts_json_view1", "query_gts_json_view2", "difficulties",
                "affine1", "query_img2_affine", "original_indices", # Renamed affine2 to query_img2_affine for clarity
                "padded_support_imgs", "padded_support_labels", "padded_support_sgs_bytes",
                "num_support_per_problem", "tree_indices", "is_weights"
            ],
            size=len(dataset), # Total number of samples in the dataset
            auto_reset=True,
            fill_last_batch=False,
            dynamic_shape=True,
            reader_name="Reader" # For FileReader, specify reader_name
        )
    
    # Start DALI queue monitoring thread
    def monitor_queue():
        while True:
            try:
                # Accessing `pipeline.queue_depth` might not be directly available on `DALIGenericIterator`
                # You typically access it on the `Pipeline` object itself.
                # The `dali_pipe_instance` is the Pipeline object.
                queue_depth = dali_pipe_instance.queue_depth()
                logger.debug(f"DALI queue size: {queue_depth}")
            except Exception as e:
                logger.warning(f"Failed to get DALI queue depth: {e}")
            time.sleep(cfg['dali'].get('monitor_interval', 1.0))
    
    threading.Thread(target=monitor_queue, daemon=True).start()

    # Start DALI performance tracing
    # The output will be saved to cfg['paths']['logs_dir']
    performance.start_tracing(dali_pipe_instance, interval_s=30, output_dir=cfg['paths']['logs_dir'])
    logger.info(f"DALI performance tracing started, outputting to {cfg['paths']['logs_dir']}.")

    logger.info(f"DALI loader built for mode {'training' if is_train else 'validation'} with size {len(dataset)}.")
    return loader

# Update BongardDataModule to use get_loader
class BongardDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.train_file_paths = None
        self.val_file_paths = None
        self.train_labels_np = None
        self.val_labels_np = None
        self.dali_train_loader = None
        self.dali_val_loader = None
        self.setup_called = False

    def prepare_data(self):
        if self.cfg['data']['use_synthetic_data']:
            os.makedirs(os.path.join(self.cfg['data']['data_root_path'], 'synthetic_temp'), exist_ok=True)
        else:
            # Ensure dummy image directory exists for real data loading
            dummy_image_dir = os.path.join(self.cfg['data']['data_root_path'], "dummy_images")
            os.makedirs(dummy_image_dir, exist_ok=True)
            # No need to check real data path existence here, `load_bongard_data` handles dummy files.
        pass

    def setup(self, stage: Optional[str] = None):
        if self.setup_called and stage != 'test':
            return
        
        logger.info(f"Setting up data for rank {self.trainer.global_rank} / {self.trainer.world_size}")
        
        bongard_generator = None
        if self.cfg['data']['use_synthetic_data']:
            try:
                from bongard_rules import ALL_BONGARD_RULES
                bongard_generator = BongardGenerator(self.cfg['data'], ALL_BONGARD_RULES)
            except ImportError:
                logger.error("Could not import BongardGenerator or ALL_BONGARD_RULES. Synthetic data generation will fail.")
                self.cfg['data']['use_synthetic_data'] = False
        
        if stage == 'fit' or stage is None:
            if self.cfg['data']['use_synthetic_data']:
                self.train_dataset = BongardSyntheticDataset(
                    self.cfg, bongard_generator, num_samples=self.cfg['data']['synthetic_data_config']['num_train_problems']
                )
                self.val_dataset = BongardSyntheticDataset(
                    self.cfg, bongard_generator, num_samples=self.cfg['data']['synthetic_data_config']['num_val_problems']
                )
            else:
                (train_data_list, val_data_list, 
                 self.train_file_paths, self.val_file_paths, 
                 self.train_labels_np, self.val_labels_np) = load_bongard_data(
                    self.cfg['data']['real_data_config']['dataset_path'],
                    self.cfg['data']['real_data_config']['dataset_name'],
                    self.cfg['data']['real_data_config']['train_split']
                )
                self.train_dataset = RealBongardDataset(train_data_list)
                self.val_dataset = RealBongardDataset(val_data_list)
                
                # Store file lists and labels in config for DALI FileReader
                self.cfg['data']['real_data_config']['train_file_list'] = self.train_file_paths
                self.cfg['data']['real_data_config']['val_file_list'] = self.val_file_paths
                self.cfg['data']['real_data_config']['train_labels_np'] = self.train_labels_np
                self.cfg['data']['real_data_config']['val_labels_np'] = self.val_labels_np

            # Initialize DALI loaders if use_dali is true
            if self.cfg['data']['use_dali'] and HAS_DALI:
                self.dali_train_loader = build_dali_loader(
                    self.cfg, self.train_dataset, is_train=True,
                    rank=self.trainer.local_rank, world_size=self.trainer.world_size
                )
                self.dali_val_loader = build_dali_loader(
                    self.cfg, self.val_dataset, is_train=False,
                    rank=self.trainer.local_rank, world_size=self.trainer.world_size
                )
        self.setup_called = True

    def train_dataloader(self) -> Union[DataLoader, DALIGenericIterator]:
        if self.cfg['data']['use_dali'] and HAS_DALI:
            logger.info(f"Using DALI DALIGenericIterator for training on rank {self.trainer.global_rank}.")
            return self.dali_train_loader
        else:
            logger.info(f"Using PyTorch DataLoader for training on rank {self.trainer.global_rank}.")
            return build_pt_loader(
                self.cfg,
                dataset=self.train_dataset,
                is_train=True,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )

    def val_dataloader(self) -> Union[DataLoader, DALIGenericIterator]:
        if self.cfg['data']['use_dali'] and HAS_DALI:
            logger.info(f"Using DALI DALIGenericIterator for validation on rank {self.trainer.global_rank}.")
            return self.dali_val_loader
        else:
            logger.info(f"Using PyTorch DataLoader for validation on rank {self.trainer.global_rank}.")
            return build_pt_loader(
                self.cfg,
                dataset=self.val_dataset,
                is_train=False,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )

    def teardown(self, stage: Optional[str] = None):
        # DALI loaders do not have a `release()` method directly on the iterator,
        # but the underlying pipeline might need cleanup.
        # DALIGenericIterator handles pipeline lifecycle.
        logger.info(f"DataModule teardown for rank {self.trainer.global_rank}.")

def build_dali_image_processor(batch_size, num_threads, device_id, image_size, is_training, curriculum_config, augmentation_config):
    """
    Dummy function for build_dali_image_processor.
    In a real scenario, this would return a DALI pipeline or similar.
    """
    logger.warning("`build_dali_image_processor` is a dummy function. It should return a DALI pipeline.")
    return None # Return None as a placeholder

def build_pt_loader(cfg: Dict[str, Any], dataset: Dataset, is_train: bool, rank: int, world_size: int) -> DataLoader:
    """
    Builds a PyTorch DataLoader.
    """
    sampler = None
    shuffle = is_train
    if world_size > 1: # For Distributed Data Parallel
        if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling']:
            # For HardExampleSampler, it manages its own sampling based on replay buffer
            # It already handles distributed sampling internally if num_replicas/rank are set.
            # We need to pass the replay buffer to the dataset if it's used by the sampler.
            # For simplicity, assume the dataset is already prepared for HardExampleSampler.
            # This part needs careful integration with how HardExampleSampler gets its data.
            # For now, let's just use DistributedSampler as a fallback for DDP with curriculum.
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
            shuffle = False # Sampler handles shuffling
            logger.warning("HardExampleSampler not fully integrated with build_pt_loader for DDP. Using DistributedSampler.")
        else:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train)
            shuffle = False # Sampler handles shuffling

    if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['difficulty_sampling'] and sampler is None:
        # If not DDP, and curriculum is enabled, use HardExampleSampler
        # This requires the dataset to have a replay_buffer initialized.
        # The dataset's __getitem__ should handle the (idx, is_weight) tuple from the sampler.
        # This is for single-GPU/CPU training with curriculum.
        replay_buffer = dataset.replay_buffer # Assuming dataset has this attribute
        if replay_buffer is None:
            logger.error("Replay buffer not initialized in dataset for HardExampleSampler.")
            raise ValueError("Replay buffer required for HardExampleSampler.")
        sampler = HardExampleSampler(
            dataset=dataset,
            replay_buffer=replay_buffer,
            batch_size=cfg['training']['batch_size'],
            beta_anneal_epochs=cfg['training']['curriculum_config']['beta_anneal_epochs'],
            shuffle=is_train # Shuffle batches from replay buffer
        )
        shuffle = False # Sampler handles shuffling

    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=shuffle,
        num_workers=cfg['data']['dataloader_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=custom_collate_fn,
        sampler=sampler,
        drop_last=is_train # Drop last batch in training if incomplete
    )
    return loader

def get_loader(cfg, train=True, rank=0, world_size=1):
    """
    Main entry point for getting the DataLoader (PyTorch or DALI).
    """
    # To make this function runnable standalone, let's add dummy dataset creation.
    bongard_generator = None
    if cfg['data']['use_synthetic_data']:
        try:
            from bongard_rules import ALL_BONGARD_RULES
            bongard_generator = BongardGenerator(cfg['data'], ALL_BONGARD_RULES)
        except ImportError:
            logger.error("Could not import BongardGenerator or ALL_BONGARD_RULES. Synthetic data generation will fail.")
            cfg['data']['use_synthetic_data'] = False
    
    dataset = None
    if cfg['data']['use_synthetic_data']:
        dataset = BongardSyntheticDataset(
            cfg, bongard_generator, num_samples=cfg['data']['synthetic_data_config']['num_train_problems'] if train else cfg['data']['synthetic_data_config']['num_val_problems']
        )
    else:
        (train_data_list, val_data_list, 
         train_file_paths, val_file_paths, 
         train_labels_np, val_labels_np) = load_bongard_data(
            cfg['data']['real_data_config']['dataset_path'],
            cfg['data']['real_data_config']['dataset_name'],
            cfg['data']['real_data_config']['train_split']
        )
        cfg['data']['real_data_config']['train_file_list'] = train_file_paths
        cfg['data']['real_data_config']['val_file_list'] = val_file_paths
        cfg['data']['real_data_config']['train_labels_np'] = train_labels_np
        cfg['data']['real_data_config']['val_labels_np'] = val_labels_np
        dataset = RealBongardDataset(train_data_list if train else val_data_list)

    if cfg['data']['use_dali'] and HAS_DALI:
        return build_dali_loader(cfg, dataset, train, rank, world_size)
    else:
        return build_pt_loader(cfg, dataset, train, rank, world_size)
