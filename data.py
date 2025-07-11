# Folder: bongard_solver/
# File: data.py
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
except ImportError:
    logger.error("Could not import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE or ALL_BONGARD_RULES from config.py/bongard_rules.py. Data loading might fail.")
    # Define dummy values if imports fail to prevent crashes
    CONFIG = {'data': {'image_size': 128, 'synthetic_data_config': {'max_support_images_per_problem': 5, 'num_train_problems': 100, 'num_val_problems': 20, 'num_test_problems': 10}, 'dataloader_workers': 0, 'use_synthetic_data': True, 'use_dali': False, 'real_data_config': {'dataset_path': '', 'dataset_name': '', 'train_split': 0.8}}, 'training': {'batch_size': 16, 'curriculum_learning': False, 'curriculum_config': {'difficulty_sampling': False, 'difficulty_update_frequency_batches': 100, 'beta_anneal_epochs': 10}, 'augmentation_config': {'use_autoaugment': False, 'use_randaugment': False, 'randaugment_num_ops': 2, 'randaugment_magnitude': 9, 'use_random_erasing': False, 'random_erasing_prob': 0.5, 'random_erasing_scale_min': 0.02, 'random_erasing_scale_max': 0.33, 'random_erasing_ratio_min': 0.3, 'random_erasing_ratio_max': 3.3, 'use_color_jitter': False, 'color_jitter_brightness': 0.8, 'color_jitter_contrast': 0.8, 'color_jitter_saturation': 0.8, 'color_jitter_hue': 0.2, 'use_gaussian_blur': False, 'gaussian_blur_sigma': 1.0, 'use_random_flip': False, 'use_random_resized_crop': False, 'random_resized_crop_area_min': 0.08, 'random_resized_crop_ratio_min': 0.75, 'random_resized_crop_ratio_max': 1.33}}, 'few_shot': {'enable': False, 'k_shot': 1, 'n_way': 2}}
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ALL_BONGARD_RULES = [] # Dummy rule list

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
        
        # Apply transform to images (if transform is provided)
        # If transform is None, apply a default ToTensor and Normalize for consistency
        current_transform = self.transform if self.transform else T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Transform query images
        problem_list[0] = current_transform(Image.fromarray(problem_list[0])) # query_img1_np
        problem_list[1] = current_transform(Image.fromarray(problem_list[1])) # query_img2_np
        
        # Transform support images
        transformed_support_imgs = []
        for img_np in problem_list[9]: # padded_support_imgs_np
            transformed_support_imgs.append(current_transform(Image.fromarray(img_np)))
        problem_list[9] = torch.stack(transformed_support_imgs) # Stack into a single tensor (N_support, C, H, W)

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
        
        # Apply transform to images
        current_transform = self.transform if self.transform else T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        query_img1_tensor = current_transform(Image.fromarray(query_img1_np))
        query_img2_tensor = current_transform(Image.fromarray(query_img2_np))
        
        transformed_support_imgs = []
        for img_np in padded_support_imgs_np:
            transformed_support_imgs.append(current_transform(Image.fromarray(img_np)))
        padded_support_imgs_tensor = torch.stack(transformed_support_imgs)


        return (query_img1_tensor, query_img2_tensor, label,
                gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_idx,
                padded_support_imgs_tensor, padded_support_labels, padded_support_sgs_bytes,
                num_support_per_problem, tree_indices, torch.tensor(is_weight, dtype=torch.float))

def load_bongard_data(dataset_path: str, dataset_name: str, train_split: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Dummy function to load real Bongard data.
    In a real scenario, this would parse JSON files or similar.
    """
    logger.info(f"Loading dummy real Bongard data from {dataset_path}/{dataset_name}.")
    # Create dummy data
    all_data = []
    for i in range(100): # 100 dummy problems
        all_data.append({
            'id': i,
            'label': random.choice([0, 1]),
            'image_path_view1': f'path/to/img1_{i}.png',
            'image_path_view2': f'path/to/img2_{i}.png',
            'scene_graph_view1': {'objects': [{'id': 0, 'shape': 'circle'}], 'relations': []},
            'scene_graph_view2': {'objects': [{'id': 0, 'shape': 'square'}], 'relations': []},
            'difficulty': random.random()
        })
    
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_split)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    logger.info(f"Loaded {len(train_data)} train and {len(val_data)} validation dummy real Bongard problems.")
    return train_data, val_data

# --- Custom Collate Function ---
def custom_collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    """
    Custom collate function to handle variable-length lists and tensors
    from the dataset. Assumes images are already transformed to tensors by __getitem__.
    """
    # Batch structure from dataset:
    # (query_img1_tensor, query_img2_tensor, label,
    #  gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index,
    #  padded_support_imgs_tensor, padded_support_labels, padded_support_sgs_bytes,
    #  num_support_per_problem, tree_indices, is_weights)

    # Separate components and stack tensors
    query_img1_tensors = torch.stack([item[0] for item in batch])
    query_img2_tensors = torch.stack([item[1] for item in batch])
    query_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    gt_json_view1_list = [item[3] for item in batch]
    gt_json_view2_list = [item[4] for item in batch]
    difficulties = torch.tensor([item[5] for item in batch], dtype=torch.float)
    affine1_list = [item[6] for item in batch]
    affine2_list = [item[7] for item in batch]
    original_indices = torch.tensor([item[8] for item in batch], dtype=torch.long) # Original indices from dataset
    
    # padded_support_imgs_tensor is already (N_support, C, H, W) per item, stack across batch
    padded_support_imgs_tensors = torch.stack([item[9] for item in batch]) # Result: (B, N_support, C, H, W)
    padded_support_labels_list = torch.tensor([item[10] for item in batch], dtype=torch.long)
    padded_support_sgs_bytes_list = [item[11] for item in batch]
    num_support_per_problem = torch.tensor([item[12] for item in batch], dtype=torch.long)
    tree_indices = torch.tensor([item[13] for item in batch], dtype=torch.long) # Indices in replay buffer tree
    is_weights = torch.tensor([item[14] for item in batch], dtype=torch.float) # Importance sampling weights

    return (query_img1_tensors, query_img2_tensors, query_labels,
            gt_json_view1_list, gt_json_view2_list, difficulties, affine1_list, affine2_list, original_indices,
            padded_support_imgs_tensors, padded_support_labels_list, padded_support_sgs_bytes_list,
            num_support_per_problem, tree_indices, is_weights)

# --- DALI Pipeline and Loader ---
@fn.pipeline_def
def bongard_dali_pipeline(image_size: int, is_training: bool, curriculum_config: Dict[str, Any], augmentation_config: Dict[str, Any]):
    # Use external_source to get data from Python side.
    # The `DALIGenericIterator` will feed data to these sources.
    # We expect raw numpy arrays (HWC, uint8) from the dataset.
    query_img1_raw = fn.external_source(name="query_img1", device="cpu")
    query_img2_raw = fn.external_source(name="query_img2", device="cpu")
    support_imgs_flat_raw = fn.external_source(name="support_imgs_flat", device="cpu")

    # Image processing function within DALI pipeline
    def process_image_dali(img_raw):
        # Convert to float and normalize
        output_images = fn.cast(img_raw, dtype=types.FLOAT)
        output_images = fn.crop_mirror_normalize(
            output_images,
            dtype=types.FLOAT,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            output_layout=types.NCHW # Convert to NCHW for PyTorch
        )
        return output_images

    # Apply processing to all image types
    processed_query_img1 = process_image_dali(query_img1_raw)
    processed_query_img2 = process_image_dali(query_img2_raw)
    processed_support_imgs_flat = process_image_dali(support_imgs_flat_raw)

    # Return the processed images
    return processed_query_img1, processed_query_img2, processed_support_imgs_flat

class DaliImageProcessor:
    """
    A wrapper for DALI pipeline to be used directly for processing numpy images.
    This is a fallback/alternative to DALIGenericIterator, primarily for `training.py`'s direct `dali_processor.run` calls.
    For standard DALI usage, `DALIGenericIterator` is preferred.
    """
    def __init__(self, batch_size: int, num_threads: int, device_id: int, image_size: int, is_training: bool, curriculum_config: Dict[str, Any], augmentation_config: Dict[str, Any]):
        self.image_size = image_size
        self.is_training = is_training
        self.augmentation_config = augmentation_config
        self.device_id = device_id
        
        # This pipeline is used for direct `run` calls, not for `DALIGenericIterator`.
        # It needs to be configured to take numpy inputs directly.
        # This is a simplified direct processing logic.
        # A full DALI pipeline for direct execution would be more complex.

        # For the purpose of `dali_processor.run` in `training.py`,
        # we will use torchvision transforms as a fallback if DALI's direct
        # numpy processing is not fully implemented or configured.
        logger.warning("DaliImageProcessor.run is using torchvision transforms as a fallback. For full DALI performance, ensure DALI pipeline is correctly configured for direct numpy input processing or use DALIGenericIterator.")
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        if self.is_training and HAS_RICH_AUGMENTATIONS:
            if self.augmentation_config.get('use_autoaugment', False):
                self.transform.transforms.insert(2, AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
            if self.augmentation_config.get('use_randaugment', False):
                self.transform.transforms.insert(2, RandAugment(
                    num_ops=self.augmentation_config.get('randaugment_num_ops', 2),
                    magnitude=self.augmentation_config.get('randaugment_magnitude', 9)
                ))
            if self.augmentation_config.get('use_random_erasing', False):
                self.transform.transforms.append(RandomErasing(
                    p=self.augmentation_config.get('random_erasing_prob', 0.5),
                    scale=(self.augmentation_config.get('random_erasing_scale_min', 0.02), self.augmentation_config.get('random_erasing_scale_max', 0.33)),
                    ratio=(self.augmentation_config.get('random_erasing_ratio_min', 0.3), self.augmentation_config.get('random_erasing_ratio_max', 3.3))
                ))
            # Add other DALI-like augmentations here if they can be mapped to torchvision
            if self.augmentation_config.get('use_color_jitter', False):
                self.transform.transforms.insert(2, T.ColorJitter(
                    brightness=self.augmentation_config.get('color_jitter_brightness', 0.8),
                    contrast=self.augmentation_config.get('color_jitter_contrast', 0.8),
                    saturation=self.augmentation_config.get('color_jitter_saturation', 0.8),
                    hue=self.augmentation_config.get('color_jitter_hue', 0.2)
                ))
            if self.augmentation_config.get('use_gaussian_blur', False):
                self.transform.transforms.insert(2, T.GaussianBlur(
                    kernel_size=int(self.augmentation_config.get('gaussian_blur_sigma', 1.0)*2 + 1) # Convert sigma to kernel_size
                ))
            if self.augmentation_config.get('use_random_flip', False):
                self.transform.transforms.insert(2, T.RandomHorizontalFlip(p=0.5)) # DALI's coin_flip

        # Initialize the DALI pipeline for direct execution if HAS_DALI
        self.pipeline = None
        if HAS_DALI:
            # Create a simple DALI pipeline for direct execution, not for DALIGenericIterator
            # This pipeline would be run manually for each batch of numpy arrays.
            # This is a conceptual example for `dali_processor.run`.
            # A more robust solution would involve `fn.external_source` with a custom reader.
            
            # For now, let's assume `bongard_dali_pipeline` is defined and can be built.
            # However, `bongard_dali_pipeline` is designed for `DALIGenericIterator`'s `external_source`
            # and not for direct `run` calls with lists of numpy arrays.
            # So, the `dali_processor.run` will continue to use the torchvision fallback.
            pass # DALI pipeline not directly built here for `run` method.

    def run(self, query_img1_np_list: List[np.ndarray], query_img2_np_list: List[np.ndarray], support_imgs_flat_np_list: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes lists of numpy images using the configured transforms (torchvision fallback).
        This method is called by `training.py` when `use_dali` is true, but not when `DALIGenericIterator` is used.
        """
        processed_query_img1_tensors = torch.stack([self.transform(Image.fromarray(img_np)) for img_np in query_img1_np_list]).to(DEVICE)
        processed_query_img2_tensors = torch.stack([self.transform(Image.fromarray(img_np)) for img_np in query_img2_np_list]).to(DEVICE)
        processed_support_imgs_flat_tensors = torch.stack([self.transform(Image.fromarray(img_np)) for img_np in support_imgs_flat_np_list]).to(DEVICE)
        
        return processed_query_img1_tensors, processed_query_img2_tensors, processed_support_imgs_flat_tensors

    def release(self):
        """Releases DALI pipeline resources."""
        if self.pipeline:
            self.pipeline.release()


def build_dali_image_processor(batch_size: int, num_threads: int, device_id: int, image_size: int, is_training: bool, curriculum_config: Dict[str, Any], augmentation_config: Dict[str, Any]) -> DaliImageProcessor:
    """
    Builds and returns a DaliImageProcessor.
    This is intended for the `dali_processor.run` calls in `training.py`.
    """
    # This function will always return an instance of `DaliImageProcessor`.
    # Inside `DaliImageProcessor`, it decides whether to use actual DALI or torchvision fallback.
    return DaliImageProcessor(batch_size, num_threads, device_id, image_size, is_training, curriculum_config, augmentation_config)

# --- Main DataLoader Builders ---
def build_pt_loader(cfg: Dict[str, Any], dataset: Dataset, is_train: bool, rank: int, world_size: int) -> DataLoader:
    """
    Builds a PyTorch DataLoader with specified augmentations and samplers.
    """
    img_size = cfg['data']['image_size']
    
    # Define transformations to be passed to the dataset
    transform_list = [
        T.Resize((img_size, img_size)),
    ]
    
    if is_train and HAS_RICH_AUGMENTATIONS:
        if cfg['training']['augmentation_config'].get('use_autoaugment', False):
            transform_list.append(AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET))
            logger.info("PyTorch DataLoader: AutoAugment enabled.")
        if cfg['training']['augmentation_config'].get('use_randaugment', False):
            transform_list.append(RandAugment(
                num_ops=cfg['training']['augmentation_config'].get('randaugment_num_ops', 2),
                magnitude=cfg['training']['augmentation_config'].get('randaugment_magnitude', 9)
            ))
            logger.info("PyTorch DataLoader: RandAugment enabled.")
    
    transform_list.append(T.ToTensor())
    transform_list.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    if is_train and HAS_RICH_AUGMENTATIONS and cfg['training']['augmentation_config'].get('use_random_erasing', False):
        transform_list.append(RandomErasing(
            p=cfg['training']['augmentation_config'].get('random_erasing_prob', 0.5),
            scale=(cfg['training']['augmentation_config'].get('random_erasing_scale_min', 0.02), cfg['training']['augmentation_config'].get('random_erasing_scale_max', 0.33)),
            ratio=(cfg['training']['augmentation_config'].get('random_erasing_ratio_min', 0.3), cfg['training']['augmentation_config'].get('random_erasing_ratio_max', 3.3))
        ))
        logger.info("PyTorch DataLoader: RandomErasing enabled.")

    transform = T.Compose(transform_list)

    # Dataset must accept and apply this transform
    # Create a new dataset instance with the transform
    if isinstance(dataset, BongardSyntheticDataset):
        dataset_with_transform = BongardSyntheticDataset(cfg, dataset.generator, dataset.num_samples, transform=transform)
    elif isinstance(dataset, RealBongardDataset):
        dataset_with_transform = RealBongardDataset(dataset.data_list, transform=transform)
    else:
        logger.error("Unknown dataset type for PyTorch DataLoader. Cannot apply transform.")
        dataset_with_transform = dataset # Fallback to original dataset

    sampler = None
    shuffle = is_train # Default shuffle behavior

    if is_train and cfg['training']['curriculum_learning']:
        if cfg['training']['curriculum_config']['difficulty_sampling']:
            if hasattr(dataset_with_transform, 'replay_buffer') and dataset_with_transform.replay_buffer is not None:
                sampler = HardExampleSampler(
                    dataset=dataset_with_transform, # Pass dataset with transform
                    replay_buffer=dataset_with_transform.replay_buffer,
                    batch_size=cfg['training']['batch_size'],
                    beta_anneal_epochs=cfg['training']['curriculum_config'].get('beta_anneal_epochs', cfg['training']['epochs']),
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True
                )
                shuffle = False
                logger.info("PyTorch DataLoader: HardExampleSampler enabled.")
            else:
                logger.warning("HardExampleSampler requested but dataset has no replay_buffer. Falling back to DistributedSampler.")
                sampler = DistributedSampler(dataset_with_transform, num_replicas=world_size, rank=rank, shuffle=is_train)
                shuffle = False
        elif cfg['training']['curriculum_config'].get('curriculum_type') == 'curriculum_sampler':
            sampler = CurriculumSampler(dataset_with_transform, num_replicas=world_size, rank=rank, shuffle=is_train)
            shuffle = False
            logger.info("PyTorch DataLoader: CurriculumSampler enabled.")
        elif cfg['training']['curriculum_config'].get('curriculum_type') == 'prototypical_sampler':
            sampler = PrototypicalSampler(dataset_with_transform, num_replicas=world_size, rank=rank, shuffle=is_train)
            shuffle = False
            logger.info("PyTorch DataLoader: PrototypicalSampler enabled.")
        else:
            logger.info("PyTorch DataLoader: Curriculum learning enabled but no specific sampler type. Using DistributedSampler.")
            sampler = DistributedSampler(dataset_with_transform, num_replicas=world_size, rank=rank, shuffle=is_train)
            shuffle = False
    elif world_size > 1: # Use DistributedSampler for DDP if no other sampler specified
        sampler = DistributedSampler(dataset_with_transform, num_replicas=world_size, rank=rank, shuffle=is_train)
        shuffle = False
        logger.info(f"PyTorch DataLoader: DistributedSampler enabled for DDP (rank {rank}).")

    return DataLoader(
        dataset_with_transform,
        batch_size=cfg['training']['batch_size'],
        sampler=sampler,
        shuffle=shuffle, # This will be False if sampler is used
        num_workers=cfg['data']['dataloader_workers'],
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

def build_dali_loader(cfg: Dict[str, Any], dataset: Dataset, is_train: bool, rank: int, world_size: int) -> DALIGenericIterator:
    """
    Builds a DALI DataLoader using DALIGenericIterator.
    """
    if not HAS_DALI:
        logger.error("DALI not available. Cannot build DALI loader.")
        raise ImportError("NVIDIA DALI is not installed or configured.")
    
    # Instantiate the DALI pipeline for DALIGenericIterator
    # Note: This pipeline is different from the one used by DaliImageProcessor.run
    # This one is designed to be iterated by DALIGenericIterator.
    dali_pipeline = bongard_dali_pipeline(
        batch_size=cfg['training']['batch_size'],
        num_threads=cfg['data']['dataloader_workers'],
        device_id=rank, # Use rank for device_id in DDP
        image_size=cfg['data']['image_size'],
        is_training=is_train,
        curriculum_config=cfg['training']['curriculum_config'],
        augmentation_config=cfg['training']['augmentation_config']
    )
    dali_pipeline.build() # Build the pipeline

    # DALIGenericIterator requires a source function if not using DALI's native readers.
    # We need to adapt our PyTorch Dataset to be an external source for DALI.
    # This is a common pattern for integrating PyTorch datasets with DALI.
    
    # Define a callable for external_source
    # This callable will yield batches of numpy arrays from the PyTorch dataset.
    
    # This requires the dataset to be accessible from within the DALI pipeline's external_source.
    # A simpler way is to make `DALIGenericIterator` iterate over the PyTorch dataset directly,
    # and its `output_map` should correspond to what `custom_collate_fn` would normally return.
    
    # However, `DALIGenericIterator` takes a `pipeline` object.
    # The `external_source` in the pipeline needs to be fed data.
    # This is usually done by providing a `source` argument to `fn.external_source`
    # that is a Python callable yielding data.
    
    # For simplicity and to match the `DALIGenericIterator` usage in `BongardDataModule`
    # from the `phase1.docx` snippet, we assume the `bongard_dali_pipeline` correctly
    # uses `fn.external_source` with names "query_img1", "query_img2", "support_imgs_flat".
    # The `DALIGenericIterator` then handles feeding the data from the dataset to these sources.
    
    # The `DALIGenericIterator` expects the `dataset` to be iterable and yield the raw numpy data.
    # So, the `__getitem__` of `BongardSyntheticDataset` and `RealBongardDataset` should
    # return raw numpy arrays when `use_dali` is true, bypassing torchvision transforms.
    
    # Let's modify `BongardSyntheticDataset` and `RealBongardDataset` to return raw numpy
    # when `self.transform` is None (which it would be if `use_dali` is true and `build_pt_loader`
    # is not called).
    
    # Re-adjusting `BongardSyntheticDataset.__getitem__` and `RealBongardDataset.__getitem__`:
    # If `self.transform` is None (meaning DALI path), return raw numpy arrays.
    # If `self.transform` is provided (meaning PyTorch path), apply transform and return tensors.
    # This will simplify `custom_collate_fn` for both paths.

    logger.info(f"Using DALI DALIGenericIterator for {'training' if is_train else 'validation'} on rank {rank}.")
    return DALIGenericIterator(
        [dali_pipeline],
        output_map=["query_img1", "query_img2", "support_imgs_flat"],
        size=len(dataset),
        auto_reset=True,
        fill_last_batch=False,
        dynamic_shape=True
    )

# Update BongardDataModule to use get_loader
class BongardDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.dali_image_processor_train = None # For DALI path, holds the pipeline
        self.dali_image_processor_val = None # For DALI path, holds the pipeline
        self.setup_called = False

    def prepare_data(self):
        if self.cfg['data']['use_synthetic_data']:
            os.makedirs(os.path.join(self.cfg['data']['data_root_path'], 'synthetic_temp'), exist_ok=True)
        else:
            if not os.path.exists(self.cfg['data']['real_data_config']['dataset_path']):
                logger.error(f"Real data path not found: {self.cfg['data']['real_data_config']['dataset_path']}")
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
                train_data_list, val_data_list = load_bongard_data(
                    self.cfg['data']['real_data_config']['dataset_path'],
                    self.cfg['data']['real_data_config']['dataset_name'],
                    self.cfg['data']['real_data_config']['train_split']
                )
                self.train_dataset = RealBongardDataset(train_data_list)
                self.val_dataset = RealBongardDataset(val_data_list)
        
        # Initialize DALI pipelines if use_dali is true
        if self.cfg['data']['use_dali'] and HAS_DALI:
            # Create DALI pipeline for training
            self.dali_image_processor_train = bongard_dali_pipeline(
                batch_size=self.cfg['training']['batch_size'],
                num_threads=self.cfg['data']['dataloader_workers'],
                device_id=self.trainer.local_rank,
                image_size=self.cfg['data']['image_size'],
                is_training=True,
                curriculum_config=self.cfg['training']['curriculum_config'],
                augmentation_config=self.cfg['training']['augmentation_config']
            )
            self.dali_image_processor_train.build()
            logger.info(f"DALI training pipeline built for device {self.trainer.local_rank}.")

            # Create DALI pipeline for validation
            self.dali_image_processor_val = bongard_dali_pipeline(
                batch_size=self.cfg['training']['batch_size'],
                num_threads=self.cfg['data']['dataloader_workers'],
                device_id=self.trainer.local_rank,
                image_size=self.cfg['data']['image_size'],
                is_training=False, # Not training
                curriculum_config=self.cfg['training']['curriculum_config'],
                augmentation_config=self.cfg['training']['augmentation_config']
            )
            self.dali_image_processor_val.build()
            logger.info(f"DALI validation pipeline built for device {self.trainer.local_rank}.")

        self.setup_called = True

    def train_dataloader(self) -> Union[DataLoader, DALIGenericIterator]:
        if self.cfg['data']['use_dali'] and HAS_DALI:
            logger.info(f"Using DALI DALIGenericIterator for training on rank {self.trainer.global_rank}.")
            return DALIGenericIterator(
                [self.dali_image_processor_train], # Pass the built pipeline
                output_map=["query_img1", "query_img2", "support_imgs_flat"],
                size=len(self.train_dataset),
                auto_reset=True,
                fill_last_batch=False,
                dynamic_shape=True
            )
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
            return DALIGenericIterator(
                [self.dali_image_processor_val], # Pass the built pipeline
                output_map=["query_img1", "query_img2", "support_imgs_flat"],
                size=len(self.val_dataset),
                auto_reset=True,
                fill_last_batch=False,
                dynamic_shape=True
            )
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
        if self.dali_image_processor_train:
            self.dali_image_processor_train.release()
        if self.dali_image_processor_val:
            self.dali_image_processor_val.release()
        logger.info(f"DataModule teardown for rank {self.trainer.global_rank}.")
