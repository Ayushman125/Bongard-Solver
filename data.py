# Folder: bongard_solver/
# File: data.py
import os
import glob
import json
import random
import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Iterator # Added Iterator for Sampler
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2  # For synthetic data generation and mask processing
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler  # For DDP with PyTorch DataLoader
from torch.utils.data import Sampler # For PrototypicalSampler

# DALI imports
try:
    from nvidia.dali.pipeline import pipeline_def
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

# Import from config
try:
    from config import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE
    from bongard_rules import ALL_BONGARD_RULES  # For BongardGenerator
    from replay_buffer import KnowledgeReplayBuffer
    # Import _calculate_iou from utils.py
    from utils import _calculate_iou
except ImportError:
    logger.warning("Could not import CONFIG, IMAGENET_MEAN, IMAGENET_STD, DEVICE, ALL_BONGARD_RULES, KnowledgeReplayBuffer, or _calculate_iou. Using dummy values/classes.")
    # Dummy CONFIG for standalone testing if config.py is not fully set up
    CONFIG = {
        'data': {
            'image_size': 224,
            'synthetic_data_config': {
                'min_objects_per_image': 1, 'max_objects_per_image': 3,
                'object_size_range': (20, 80), 'padding': 10, 'font_path': None,
                'scene_layout_types': ['random'],
                'min_support_images_per_problem': 2, 'max_support_images_per_problem': 5,
            }
        },
        'training': {
            'use_dali': False,
            'batch_size': 16, # Dummy value for few-shot testing
            'dataloader_workers': 0, # Dummy value
            'dali': {
                'sample_per_gpu': 1, 'sample_mode': 'random', 'decode_device': 'mixed', 'aug_device': 'gpu',
            }
        },
        'few_shot': { # Added for dummy config
            'enable': False,
            'n_way': 2, 'k_shot': 1, 'q_query': 1, 'episodes': 10
        }
    }
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = torch.device("cpu")
    ALL_BONGARD_RULES = {}
    class KnowledgeReplayBuffer:
        def __init__(self, *args, **kwargs): pass
        def add(self, *args): pass
        def sample(self, *args): return [], [], []
        def update_beta(self, *args): pass
        def set_current_batch_info(self, *args): pass
    # Dummy _calculate_iou if utils.py cannot be imported
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(box1_area + box2_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0


logger = logging.getLogger(__name__)

# --- Dataset Classes ---
class BongardSyntheticDataset(Dataset):
    """
    A PyTorch Dataset for generating synthetic Bongard problems on the fly.
    Each item now includes query images and a set of support images.
    Returns raw NumPy arrays for images.
    """
    def __init__(self, config: Dict[str, Any], bongard_generator: Any, num_samples: int):
        self.config = config
        self.bongard_generator = bongard_generator
        self.num_samples = num_samples
        self.curriculum_sampler = None  # Will be set by training.py if curriculum is used
        self.current_image_size = CONFIG['data']['image_size']  # Default, can be updated by sampler
        
        # Store labels for PrototypicalSampler if few-shot is enabled
        # For synthetic data, we can generate dummy labels or pre-generate them.
        # For few-shot, the sampler needs to know the class labels of each sample in the dataset.
        # Here, `labels` refers to the Bongard problem label (0 or 1).
        # Assuming problem labels are 0 or 1 for positive/negative.
        self.labels = [random.randint(0, 1) for _ in range(num_samples)] # Dummy labels for sampler
        logger.info(f"BongardSyntheticDataset initialized with {num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, bytes, bytes, float, np.ndarray, np.ndarray, int, List[np.ndarray], List[int], List[bytes], int, Optional[int], Optional[float]]:
        """
        Generates a single Bongard problem (query image pair, label, GT scene graphs, difficulty)
        and a set of support images with their labels and scene graphs.
        Returns:
            Tuple: (query_image_view1_np, query_image_view2_np, query_label,
                    query_gt_json_bytes_view1, query_gt_json_bytes_view2,
                    difficulty, affine_matrix_view1, affine_matrix_view2, original_index,
                    support_images_np_list, support_labels_list, support_sgs_bytes_list,
                    num_support_per_problem,
                    [tree_index, is_weight] - optional, for PER)
        """
        # Get current image size from curriculum sampler if available, else use default
        image_size_for_gen = self.current_image_size if self.curriculum_sampler else CONFIG['data']['image_size']
        original_index = idx  # Default original index
        tree_index = None
        is_weight = None
        # Generate a Bongard problem with support set
        (query_img1, query_img2, query_label, query_sg1, query_sg2,
         support_imgs, support_labels, support_sgs) = \
            self.bongard_generator.generate_bongard_problem(image_size=image_size_for_gen)
        # Convert PIL Images to NumPy arrays (H, W, C)
        query_img1_np = np.array(query_img1)
        query_img2_np = np.array(query_img2)
        # Convert scene graphs to JSON bytes
        query_sg1_bytes = json.dumps(query_sg1).encode('utf-8')
        query_sg2_bytes = json.dumps(query_sg2).encode('utf-8')
        support_imgs_np_list = [np.array(img) for img in support_imgs]
        support_sgs_bytes_list = [json.dumps(sg).encode('utf-8') for sg in support_sgs]
        num_actual_support_images = len(support_imgs_np_list)
        # Dummy difficulties and affine matrices for now
        difficulty = 0.5 
        affine_matrix_view1 = np.eye(3, dtype=np.float32)
        affine_matrix_view2 = np.eye(3, dtype=np.float32)
        return (query_img1_np, query_img2_np, query_label, query_sg1_bytes, query_sg2_bytes,
                difficulty, affine_matrix_view1, affine_matrix_view2, original_index,
                support_imgs_np_list, support_labels, support_sgs_bytes_list,
                num_actual_support_images, tree_index, is_weight)

class RealBongardDataset(Dataset):
    """
    A PyTorch Dataset for loading real Bongard problems from disk.
    Returns raw NumPy arrays for images.
    """
    def __init__(self, data_list: List[Dict[str, Any]], transform=None):
        self.data_list = data_list
        self.transform = transform
        self.curriculum_sampler = None  # Will be set by training.py if curriculum is used
        self.current_image_size = CONFIG['data']['image_size']  # Default, can be updated by sampler
        
        # Extract labels for PrototypicalSampler
        self.labels = [item['label'] for item in data_list] # Assuming 'label' key exists
        logger.info(f"RealBongardDataset initialized with {len(data_list)} samples.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int, bytes, bytes, float, np.ndarray, np.ndarray, int, List[np.ndarray], List[int], List[bytes], int, Optional[int], Optional[float]]:
        item = self.data_list[idx]
        
        # Load query images
        query_img1 = Image.open(item['image_path_view1']).convert("RGB")
        query_img2 = Image.open(item['image_path_view2']).convert("RGB")
        query_img1_np = np.array(query_img1)
        query_img2_np = np.array(query_img2)
        query_label = item['label']
        
        query_sg1 = item.get('scene_graph_view1', {})
        query_sg2 = item.get('scene_graph_view2', {})
        
        query_sg1_bytes = json.dumps(query_sg1).encode('utf-8')
        query_sg2_bytes = json.dumps(query_sg2).encode('utf-8')
        difficulty = item.get('difficulty', 0.5)
        affine_matrix_view1 = np.eye(3, dtype=np.float32)  # Dummy
        affine_matrix_view2 = np.eye(3, dtype=np.float32)  # Dummy
        tree_index = None
        is_weight = None
        # --- Support Set for Real Data ---
        support_imgs_np_list = []
        support_labels = []
        support_sgs_bytes_list = []
        num_actual_support_images = 0
        # If your real dataset includes explicit support sets, load them here.
        if 'support_images' in item and 'support_labels' in item and 'support_scene_graphs' in item:
            for s_img_path, s_label, s_sg_data in zip(item['support_images'], item['support_labels'], item['support_scene_graphs']):
                try:
                    s_img = Image.open(s_img_path).convert("RGB")
                    support_imgs_np_list.append(np.array(s_img))
                    support_labels.append(s_label)
                    support_sgs_bytes_list.append(json.dumps(s_sg_data).encode('utf-8'))
                except FileNotFoundError:
                    logger.warning(f"Support image not found: {s_img_path}. Skipping.")
            num_actual_support_images = len(support_imgs_np_list)
        
        return (query_img1_np, query_img2_np, query_label, query_sg1_bytes, query_sg2_bytes,
                difficulty, affine_matrix_view1, affine_matrix_view2, idx,
                support_imgs_np_list, support_labels, support_sgs_bytes_list,
                num_actual_support_images, tree_index, is_weight)

def load_bongard_data(data_root_path: str, dataset_name: str, train_split_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads real Bongard problem data from a specified directory structure.
    Assumes data is organized as:
    data_root_path/dataset_name/
        images/
            problem_id/
                view1.png
                view2.png
        labels.json (or similar, containing labels and potentially scene graphs)
    
    The 'labels.json' is expected to have entries like:
    {
        "problem_id_001": {
            "label": 1,
            "scene_graph_view1": {...},
            "scene_graph_view2": {...},
            "difficulty": 0.7,
            "support_images": ["path/to/support_pos1.png", "path/to/support_neg1.png"], # Optional
            "support_labels": [1, 0], # Optional
            "support_scene_graphs": [{...}, {...}] # Optional
        },
        ...
    }
    """
    dataset_path = os.path.join(data_root_path, dataset_name)
    image_dir = os.path.join(dataset_path, 'images')
    labels_file = os.path.join(dataset_path, 'labels.json')  # Example labels file
    if not os.path.exists(image_dir) or not os.path.exists(labels_file):
        logger.error(f"Real data directory or labels file not found: {dataset_path}, {labels_file}")
        return [], []
    all_problems = []
    try:
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading labels.json: {e}")
        return [], []
    for problem_id, problem_info in labels_data.items():
        image_path_view1 = os.path.join(image_dir, problem_id, 'view1.png')
        image_path_view2 = os.path.join(image_dir, problem_id, 'view2.png')
        if os.path.exists(image_path_view1) and os.path.exists(image_path_view2):
            problem_entry = {
                'problem_id': problem_id,
                'image_path_view1': image_path_view1,
                'image_path_view2': image_path_view2,
                'label': problem_info['label'],  # Assuming 'label' key exists
                'scene_graph_view1': problem_info.get('scene_graph_view1', {}),  # Optional
                'scene_graph_view2': problem_info.get('scene_graph_view2', {}),  # Optional
                'difficulty': problem_info.get('difficulty', 0.5)  # Optional
            }
            # Add support set info if available in labels.json (paths, labels, SGs)
            if 'support_images' in problem_info and 'support_labels' in problem_info and 'support_scene_graphs' in problem_info:
                # Convert relative paths to absolute paths
                problem_entry['support_images'] = [os.path.join(dataset_path, img_path) for img_path in problem_info['support_images']]
                problem_entry['support_labels'] = problem_info['support_labels']
                problem_entry['support_scene_graphs'] = problem_info['support_scene_graphs']
            
            all_problems.append(problem_entry)
        else:
            logger.warning(f"Images not found for problem_id {problem_id}. Skipping.")
    random.shuffle(all_problems)
    split_idx = int(len(all_problems) * train_split_ratio)
    train_data = all_problems[:split_idx]
    val_data = all_problems[split_idx:]
    logger.info(f"Loaded {len(train_data)} training problems and {len(val_data)} validation problems from {dataset_path}.")
    return train_data, val_data

# --- Curriculum Sampler (remains largely the same, interacts with PyTorch DataLoader) ---
class CurriculumSampler(torch.utils.data.Sampler):
    """
    A sampler that implements curriculum learning and prioritized experience replay.
    It can adjust image size and sample based on difficulty.
    This sampler works with a standard PyTorch DataLoader.
    """
    def __init__(self, dataset: Dataset, training_config: Dict[str, Any], num_replicas: Optional[int] = None, rank: Optional[int] = None):
        self.dataset = dataset
        self.training_config = training_config
        self.curriculum_config = training_config['curriculum_config']
        self.per_config = training_config['per_config']
        self.num_samples = len(self.dataset)
        
        # For distributed training
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.num_samples_per_replica = int(np.ceil(self.num_samples * 1.0 / self.num_replicas))
        self.total_size = self.num_samples_per_replica * self.num_replicas
        self.current_epoch = 0 
        
        self.current_image_size = self.curriculum_config['start_image_size']
        self.use_difficulty_sampling = self.curriculum_config['difficulty_sampling']
        if self.use_difficulty_sampling:
            self.replay_buffer = KnowledgeReplayBuffer(
                capacity=self.per_config['capacity'],
                alpha=self.per_config['alpha']
            )
            # Initialize buffer with dummy priorities (e.g., 1.0 for all)
            # Add original indices to the buffer
            for i in range(self.num_samples):
                self.replay_buffer.add(data_index=i, priority=1.0)
            logger.info(f"CurriculumSampler: Prioritized Experience Replay enabled with capacity {self.per_config['capacity']}.")
        else:
            self.replay_buffer = None
            logger.info("CurriculumSampler: Difficulty sampling (PER) disabled.")
        logger.info(f"CurriculumSampler initialized. Initial image size: {self.current_image_size}")

    def __iter__(self):
        # Update image size based on curriculum
        if self.curriculum_config['enabled']:
            if self.current_epoch < self.curriculum_config['size_increase_epochs']:
                progress = self.current_epoch / self.curriculum_config['size_increase_epochs']
                self.current_image_size = int(self.curriculum_config['start_image_size'] + 
                                             (self.curriculum_config['end_image_size'] - self.curriculum_config['start_image_size']) * progress)
            else:
                self.current_image_size = self.curriculum_config['end_image_size']
            
            # Update the dataset's current_image_size for generation
            if hasattr(self.dataset, 'current_image_size'):
                self.dataset.current_image_size = self.current_image_size
            logger.debug(f"CurriculumSampler: Epoch {self.current_epoch}, setting image size to {self.current_image_size}")

        if self.use_difficulty_sampling and self.replay_buffer:
            # Sample indices from replay buffer
            original_data_indices, tree_indices_batch, is_weights_batch = self.replay_buffer.sample(self.num_samples_per_replica)  # Sample for this replica
            
            # Store PER info in the replay buffer for retrieval by the training loop
            self.replay_buffer.set_current_batch_info(original_data_indices, tree_indices_batch, is_weights_batch)
            
            yield from original_data_indices
        else:
            # Standard random sampling if PER is not used
            indices = list(range(self.num_samples))
            random.shuffle(indices)
            
            # Distributed sampling logic
            start_idx = self.rank * self.num_samples_per_replica
            end_idx = min((self.rank + 1) * self.num_samples_per_replica, self.num_samples)
            indices_this_replica = indices[start_idx:end_idx]
            
            yield from indices_this_replica

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int):
        """
        Sets the current epoch for curriculum learning.
        """
        self.current_epoch = epoch
        if self.use_difficulty_sampling and self.replay_buffer:
            self.replay_buffer.update_beta(epoch, CONFIG['training']['epochs'])  # Anneal beta over total epochs

# --- Prototypical Sampler for Few-Shot Learning ---
class PrototypicalSampler(Sampler):
    """
    A sampler for few-shot learning that creates episodes with N-way K-shot Q-query structure.
    It samples `n_way` classes, then `k_shot` support examples and `q_query` query examples
    for each chosen class.
    
    Args:
        labels (list or array): List or array of class labels per sample in the dataset.
                                For Bongard problems, this would be the problem label (0 or 1).
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support examples per class.
        q_query (int): Number of query examples per class.
        episodes (int): Total number of episodes to generate per epoch.
    """
    def __init__(self, labels: List[int], n_way: int, k_shot: int, q_query: int, episodes: int):
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes

        # Map labels to indices for efficient sampling
        self.label2indices = {}
        for idx, lbl in enumerate(self.labels):
            self.label2indices.setdefault(lbl, []).append(idx)
        
        # Filter out classes that don't have enough samples for an episode
        self.available_classes = [
            lbl for lbl, indices in self.label2indices.items()
            if len(indices) >= (self.k_shot + self.q_query)
        ]
        if len(self.available_classes) < self.n_way:
            logger.warning(f"Not enough classes ({len(self.available_classes)}) with sufficient samples "
                           f"for {self.n_way}-way {self.k_shot}-shot episodes. "
                           "Few-shot sampler might not generate full episodes or will fail.")
        logger.info(f"PrototypicalSampler initialized for {n_way}-way {k_shot}-shot with {episodes} episodes.")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Generates indices for one episode.
        An episode consists of (n_way * k_shot) support samples + (n_way * q_query) query samples.
        """
        for _ in range(self.episodes):
            # Sample N classes (ways)
            if len(self.available_classes) < self.n_way:
                logger.error("Cannot sample enough unique classes for N-way. Skipping episode.")
                continue # Skip this episode if not enough classes
            
            ways = random.sample(self.available_classes, self.n_way)
            
            support_indices = []
            query_indices = []
            
            for w in ways:
                # Sample (k_shot + q_query) indices for the current class
                class_indices = self.label2indices[w]
                if len(class_indices) < (self.k_shot + self.q_query):
                    logger.error(f"Class {w} does not have enough samples for {self.k_shot}-shot {self.q_query}-query. Skipping episode.")
                    # This should ideally be caught during __init__ by filtering available_classes
                    continue # Skip this class, or the entire episode if critical
                
                sampled_indices = random.sample(class_indices, self.k_shot + self.q_query)
                support_indices.extend(sampled_indices[:self.k_shot])
                query_indices.extend(sampled_indices[self.k_shot:])
            
            # Yield indices for the current episode: support first, then query
            # The DataLoader's collate_fn will then process these indices.
            if len(support_indices) == self.n_way * self.k_shot and \
               len(query_indices) == self.n_way * self.q_query:
                yield support_indices + query_indices
            else:
                logger.warning(f"Generated incomplete episode. Support: {len(support_indices)}, Query: {len(query_indices)}. Skipping.")

    def __len__(self) -> int:
        return self.episodes

# --- DALI Augmentation Pipeline ---
class BongardAugmentationPipeline(Pipeline):
    """
    NVIDIA DALI Pipeline for Bongard problem image augmentation and preprocessing.
    It takes raw NumPy image arrays as input via ExternalSource and outputs
    augmented, normalized, and resized GPU tensors.
    """
    def __init__(self,
                 batch_size: int,
                 num_threads: int,
                 device_id: int,
                 image_size: int,  # This is the target image size for resize
                 is_training: bool,
                 curriculum_config: Dict[str, Any]):
        
        super().__init__(batch_size, num_threads, device_id, seed=CONFIG['training']['seed'] + device_id)
        
        self.dali_op_device = "gpu" if device_id >= 0 else "cpu"
        self.image_size = image_size  # This is the target image size for resize
        self.is_training = is_training
        self.curriculum_config = curriculum_config  # Used to update self.image_size in iter_setup
        self.mean = [m * 255 for m in IMAGENET_MEAN]
        self.std = [s * 255 for s in IMAGENET_STD]
        # DALI Augmentation parameters from config
        dali_aug_config = CONFIG['data']['dali_augmentations']
        self.jpeg_p = dali_aug_config['jpeg_p']
        self.jpeg_q_range = dali_aug_config['jpeg_q']
        self.gaussian_blur_p = dali_aug_config['gaussian_blur_p']
        self.gaussian_blur_sigma_range = dali_aug_config['gaussian_blur_sigma_range']
        self.color_twist_p = dali_aug_config['color_twist_p']
        self.color_twist_brightness_range = dali_aug_config['color_twist_brightness_range']
        self.color_twist_contrast_range = dali_aug_config['color_twist_contrast_range']
        self.color_twist_saturation_range = dali_aug_config['color_twist_saturation_range']
        self.color_twist_hue_range = dali_aug_config['color_twist_hue_range']
        # Define external sources for raw image inputs
        self.query_img1_input = fn.external_source(name="query_img1", device="cpu", dtype=types.UINT8, layout="HWC")
        self.query_img2_input = fn.external_source(name="query_img2", device="cpu", dtype=types.UINT8, layout="HWC")
        self.support_imgs_flat_input = fn.external_source(name="support_imgs_flat", device="cpu", dtype=types.UINT8, layout="HWC")
        logger.info(f"DALI Augmentation Pipeline initialized. Device: {self.dali_op_device}, Target Image Size: {self.image_size}")

    def define_graph(self):
        """
        Defines the DALI graph of operations for data augmentation and preprocessing.
        It takes raw image data from external sources.
        """
        # Get raw image inputs from external sources
        query_img1_raw = self.query_img1_input()
        query_img2_raw = self.query_img2_input()
        support_imgs_flat_raw = self.support_imgs_flat_input()
        # Process Query Images
        processed_query_img1 = self._apply_transforms(query_img1_raw)
        processed_query_img2 = self._apply_transforms(query_img2_raw)
        # Process Support Images (flattened list)
        processed_support_imgs_flat = self._apply_transforms(support_imgs_flat_raw)  # Apply same transforms
        # Return augmented GPU tensors
        return processed_query_img1, processed_query_img2, processed_support_imgs_flat

    def _apply_transforms(self, imgs: types.Tensor) -> types.Tensor:
        """
        Splits out common transforms to apply to a batch of images.
        """
        # Cast to float for augmentations
        imgs = fn.cast(imgs, dtype=types.FLOAT, device=self.dali_op_device)
        # 1. Resize to target dimensions
        imgs = fn.resize(imgs,
                         resize_x=self.image_size,
                         resize_y=self.image_size,
                         interp_type=types.INTERP_LINEAR,
                         device=self.dali_op_device)
        
        # 2. Optional JPEG compression distortion
        if self.is_training and self.jpeg_p > 0:
            coin_jpeg = fn.random.coin_flip(probability=self.jpeg_p, device=self.dali_op_device)
            quality = fn.random.uniform(range=[self.jpeg_q_range[0], self.jpeg_q_range[1]], dtype=types.INT32, device=self.dali_op_device)
            compressed = fn.jpeg_compression_distortion(
                imgs,
                quality=quality,
                device=self.dali_op_device
            )
            imgs = fn.cast(coin_jpeg, dtype=types.FLOAT) * compressed + \
                   (1 - fn.cast(coin_jpeg, dtype=types.FLOAT)) * imgs
        
        # 3. Optional Gaussian Blur
        if self.is_training and self.gaussian_blur_p > 0:
            coin_blur = fn.random.coin_flip(probability=self.gaussian_blur_p, device=self.dali_op_device)
            sigma = fn.random.uniform(range=[self.gaussian_blur_sigma_range[0], self.gaussian_blur_sigma_range[1]], device=self.dali_op_device)
            blurred = fn.gaussian_blur(
                imgs,
                sigma=sigma,
                device=self.dali_op_device
            )
            imgs = fn.cast(coin_blur, dtype=types.FLOAT) * blurred + \
                   (1 - fn.cast(coin_blur, dtype=types.FLOAT)) * imgs
        
        # 4. Optional Color Twist
        if self.is_training and self.color_twist_p > 0:
            coin_color = fn.random.coin_flip(probability=self.color_twist_p, device=self.dali_op_device)
            brightness = fn.random.uniform(range=[self.color_twist_brightness_range[0], self.color_twist_brightness_range[1]], device=self.dali_op_device)
            contrast = fn.random.uniform(range=[self.color_twist_contrast_range[0], self.color_twist_contrast_range[1]], device=self.dali_op_device)
            saturation = fn.random.uniform(range=[self.color_twist_saturation_range[0], self.color_twist_saturation_range[1]], device=self.dali_op_device)
            hue = fn.random.uniform(range=[self.color_twist_hue_range[0], self.color_twist_hue_range[1]], device=self.dali_op_device)
            color_twisted = fn.color_twist(
                imgs,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                device=self.dali_op_device
            )
            imgs = fn.cast(coin_color, dtype=types.FLOAT) * color_twisted + \
                   (1 - fn.cast(coin_color, dtype=types.FLOAT)) * imgs
        
        # 5. Normalize and convert layout to NCHW float [0,1]
        output = fn.crop_mirror_normalize(
            imgs,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=self.mean,
            stddev=self.std,
            mirror=fn.random.coin_flip(probability=0.5) if self.is_training else False,  # Apply random horizontal flip
            device=self.dali_op_device
        )
        return output

class DALIImageProcessor:
    """
    A wrapper to run the DALI augmentation pipeline.
    It takes raw NumPy image batches and returns augmented GPU tensors.
    NOTE: This class is for handling external source data. For file-based DALI,
    DALIGenericIterator is used directly with the pipeline.
    This class might be deprecated or refactored if full DALI file reader is used.
    """
    def __init__(self,
                 batch_size: int,
                 num_threads: int,
                 device_id: int,
                 image_size: int,  # This will be the fixed target size for DALI's resize op
                 is_training: bool,
                 curriculum_config: Dict[str, Any]):
        
        if not HAS_DALI:
            raise ImportError("NVIDIA DALI is not installed. Cannot use DALIImageProcessor.")
        self.pipeline = BongardAugmentationPipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            image_size=image_size,
            is_training=is_training,
            curriculum_config=curriculum_config
        )
        self.pipeline.build()
        logger.info(f"DALIImageProcessor built for device {device_id}.")

    def run(self,
            query_img1_batch_np: List[np.ndarray],
            query_img2_batch_np: List[np.ndarray],
            support_imgs_flat_batch_np: List[np.ndarray]
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Feeds raw NumPy image batches to the DALI pipeline and runs it.
        Returns augmented GPU tensors.
        """
        # Feed inputs to the external sources
        self.pipeline.feed_input("query_img1", query_img1_batch_np)
        self.pipeline.feed_input("query_img2", query_img2_batch_np)
        self.pipeline.feed_input("support_imgs_flat", support_imgs_flat_batch_np)
        # Run the pipeline and get outputs
        output = self.pipeline.run()
        
        # DALI outputs are usually tuples of DALI Tensors. Convert to PyTorch Tensors.
        processed_query_img1 = output[0].as_tensor()
        processed_query_img2 = output[1].as_tensor()
        processed_support_imgs_flat = output[2].as_tensor()
        return processed_query_img1, processed_query_img2, processed_support_imgs_flat

    def release(self):
        """Releases DALI pipeline resources."""
        self.pipeline = None
        logger.info("DALIImageProcessor pipeline released.")

def build_dali_image_processor(
    batch_size: int,
    num_threads: int,
    device_id: int,
    image_size: int,  # This will be the fixed target size for DALI's resize op
    is_training: bool,
    curriculum_config: Dict[str, Any]
) -> DALIImageProcessor:
    """
    Builds and returns a DALIImageProcessor.
    """
    if not HAS_DALI:
        logger.error("NVIDIA DALI is not installed. Cannot build DALIImageProcessor.")
        raise ImportError("NVIDIA DALI is not installed.")
    dali_processor = DALIImageProcessor(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        image_size=image_size,
        is_training=is_training,
        curriculum_config=curriculum_config
    )
    logger.info(f"DALI Image Processor built for device {device_id}.")
    return dali_processor

# --- Synthetic Data Generation Logic ---
class BongardGenerator:
    """
    Generates synthetic Bongard problems based on predefined rules,
    including a query pair and a support set.
    """
    def __init__(self, data_config: Dict[str, Any], all_bongard_rules: Dict[str, Any]):
        self.data_config = data_config
        self.synthetic_config = data_config['synthetic_data_config']
        self.all_bongard_rules = all_bongard_rules
        self.object_types = ['circle', 'square', 'triangle', 'star']  # Supported shapes
        self.colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        self.fills = ['solid', 'hollow', 'striped', 'dotted']
        self.sizes = ['small', 'medium', 'large']
        self.orientations = ['upright', 'inverted']  # For triangles
        self.textures = ['none', 'striped', 'dotted']  # For more complex textures
        self.color_map = {
            'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
            'yellow': (255, 255, 0), 'black': (0, 0, 0), 'white': (255, 255, 255)
        }
        self.size_map = {'small': 0.15, 'medium': 0.25, 'large': 0.35}  # Relative to image_size
        self.font = None
        if self.synthetic_config['font_path'] and os.path.exists(self.synthetic_config['font_path']):
            try:
                self.font = ImageFont.truetype(self.synthetic_config['font_path'], 20)
            except Exception as e:
                logger.warning(f"Could not load font from {self.synthetic_config['font_path']}: {e}")
        
        logger.info("BongardGenerator initialized.")

    def _draw_object(self, draw: ImageDraw.ImageDraw, shape: str, color: str, fill: str,
                     size_factor: float, center_x: int, center_y: int,
                     image_size: int, orientation: str = 'upright', texture: str = 'none') -> Tuple[List[float], Dict[str, Any]]:
        """
        Draws a single object on the image and returns its bounding box and attributes.
        Returns: (bbox_xyxy, attributes_dict)
        """
        fill_color = self.color_map[color]
        outline_color = (0, 0, 0)  # Always black outline for now
        obj_width = int(image_size * size_factor)
        obj_height = int(image_size * size_factor)
        min_obj_size = self.synthetic_config['object_size_range'][0]
        obj_width = max(obj_width, min_obj_size)
        obj_height = max(obj_height, min_obj_size)
        x1 = center_x - obj_width // 2
        y1 = center_y - obj_height // 2
        x2 = center_x + obj_width // 2
        y2 = center_y + obj_height // 2
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_size, x2)
        y2 = min(image_size, y2)
        bbox_xyxy = [x1, y1, x2, y2]
        # Draw shape
        if shape == 'circle':
            draw.ellipse(bbox_xyxy, fill=fill_color, outline=outline_color, width=2)
        elif shape == 'square':
            draw.rectangle(bbox_xyxy, fill=fill_color, outline=outline_color, width=2)
        elif shape == 'triangle':
            if orientation == 'upright':
                vertices = [(center_x, y1), (x1, y2), (x2, y2)]
            else:  # 'inverted'
                vertices = [(center_x, y2), (x1, y1), (x2, y1)]
            draw.polygon(vertices, fill=fill_color, outline=outline_color, width=2)
        elif shape == 'star':
            # Simple 5-point star placeholder
            draw.line([(x1, center_y), (x2, center_y)], fill=fill_color, width=2)
            draw.line([(center_x, y1), (center_x, y2)], fill=fill_color, width=2)
            draw.ellipse(bbox_xyxy, outline=outline_color, width=2)
        else:
            logger.warning(f"Unsupported shape: {shape}. Drawing a square instead.")
            draw.rectangle(bbox_xyxy, fill=fill_color, outline=outline_color, width=2)
        # Apply fill/texture
        if fill == 'striped' and texture == 'striped':
            for y_line in range(y1, y2, 5):
                draw.line([(x1, y_line), (x2, y_line)], fill=(0,0,0), width=1)
        elif fill == 'dotted' and texture == 'dotted':
            for y_dot in range(y1, y2, 10):
                for x_dot in range(x1, x2, 10):
                    draw.ellipse((x_dot-1, y_dot-1, x_dot+1, y_dot+1), fill=(0,0,0))
        attributes = {
            'shape': shape,
            'color': color,
            'fill': fill,
            'size': self._get_size_label(size_factor, image_size),
            'orientation': orientation,
            'texture': texture
        }
        return bbox_xyxy, attributes

    def _get_size_label(self, size_factor: float, image_size: int) -> str:
        """Determines size label based on object area relative to image."""
        area_ratio = (size_factor * image_size)**2 / (image_size**2)
        if area_ratio < 0.02: return 'small'
        if area_ratio < 0.08: return 'medium'
        return 'large'

    def _generate_scene_layout(self, num_objects: int, image_size: int, layout_type: str) -> List[Tuple[int, int]]:
        """
        Generates object center positions based on a structured layout.
        """
        positions = []
        padding = self.synthetic_config['padding']
        
        if layout_type == 'grid':
            grid_size = int(np.ceil(np.sqrt(num_objects)))
            cell_width = (image_size - 2 * padding) // grid_size
            cell_height = (image_size - 2 * padding) // grid_size
            for i in range(num_objects):
                row = i // grid_size
                col = i % grid_size
                center_x = padding + col * cell_width + cell_width // 2
                center_y = padding + row * cell_height + cell_height // 2
                positions.append((center_x, center_y))
        
        elif layout_type == 'polar':
            center_x, center_y = image_size // 2, image_size // 2
            radius = min(image_size / 2 - padding, image_size / 4)
            
            angles = np.linspace(0, 2 * np.pi, num_objects, endpoint=False)
            for angle in angles:
                x = center_x + int(radius * np.cos(angle))
                y = center_y + int(radius * np.sin(angle))
                positions.append((x, y))
        
        else:  # 'random' layout
            for _ in range(num_objects):
                center_x = random.randint(padding, image_size - padding)
                center_y = random.randint(padding, image_size - padding)
                positions.append((center_x, center_y))
        
        return positions

    def _generate_single_image_with_rule(self, image_size: int, num_objects: int,
                                         target_rule: Any, satisfy_rule: bool) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generates a single image and its scene graph, attempting to satisfy or violate a given rule.
        This function now tries to procedurally enforce the rule.
        """
        img = Image.new('RGB', (image_size, image_size), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        scene_graph = {'objects': [], 'relations': []}
        objects_data = []  # List of (bbox, attributes, id)
        
        layout_type = random.choice(self.synthetic_config['scene_layout_types'])
        positions = self._generate_scene_layout(num_objects, image_size, layout_type)
        # First, generate base attributes for all objects randomly
        base_objects_attrs = []
        for i in range(num_objects):
            attrs = {
                'shape': random.choice(self.object_types),
                'color': random.choice(self.colors),
                'fill': random.choice(self.fills),
                'size': random.choice(self.sizes),
                'orientation': random.choice(self.orientations),
                'texture': random.choice(self.textures)
            }
            if attrs['shape'] != 'triangle': attrs['orientation'] = 'upright'
            if attrs['fill'] not in ['striped', 'dotted']: attrs['texture'] = 'none'
            base_objects_attrs.append(attrs)
        # Apply rule constraints for generation
        final_objects_attrs = self._apply_rule_constraints_for_generation(
            base_objects_attrs, target_rule, satisfy_rule, num_objects
        )
        
        # Draw objects based on final attributes and positions
        for i in range(num_objects):
            attrs = final_objects_attrs[i]
            center_x, center_y = positions[i]
            size_factor = self.size_map[attrs['size']]
            bbox_xyxy, actual_attributes = self._draw_object(
                draw, attrs['shape'], attrs['color'], attrs['fill'], size_factor,
                center_x, center_y, image_size, attrs['orientation'], attrs['texture']
            )
            
            obj_id = i
            scene_graph['objects'].append({'id': obj_id, 'bbox': bbox_xyxy, 'attributes': actual_attributes})
            objects_data.append({'id': obj_id, 'bbox': bbox_xyxy, 'attributes': actual_attributes})
        # Generate relations based on spatial proximity (simplified)
        for i in range(len(objects_data)):
            for j in range(i + 1, len(objects_data)):
                obj1 = objects_data[i]
                obj2 = objects_data[j]
                cx1 = (obj1['bbox'][0] + obj1['bbox'][2]) / 2
                cy1 = (obj1['bbox'][1] + obj1['bbox'][3]) / 2
                cx2 = (obj2['bbox'][0] + obj2['bbox'][2]) / 2
                cy2 = (obj2['bbox'][1] + obj2['bbox'][3]) / 2
                if cx1 < cx2 - 10:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'left_of'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'right_of'})
                elif cx1 > cx2 + 10:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'right_of'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'left_of'})
                if cy1 < cy2 - 10:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'above'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'below'})
                elif cy1 > cy2 + 10:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'below'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'above'})
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < image_size * 0.2:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'is_close_to'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'is_close_to'})
                
                if abs(cy1 - cy2) < 5:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'aligned_horizontally'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'aligned_horizontally'})
                
                if abs(cx1 - cx2) < 5:
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'aligned_vertically'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'aligned_vertically'})
                
                # Simple intersection check (bounding box overlap)
                bbox1 = obj1['bbox']
                bbox2 = obj2['bbox']
                if not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
                    scene_graph['relations'].append({'subject_id': obj1['id'], 'object_id': obj2['id'], 'type': 'intersects'})
                    scene_graph['relations'].append({'subject_id': obj2['id'], 'object_id': obj1['id'], 'type': 'intersects'})
        return img, scene_graph

    def _apply_rule_constraints_for_generation(self,
                                               base_objects_attrs: List[Dict[str, Any]],
                                               target_rule: Any,
                                               satisfy_rule: bool,
                                               num_objects: int) -> List[Dict[str, Any]]:
        """
        Procedurally modifies object attributes to satisfy or violate a given rule.
        This is a simplified procedural generation based on rule name.
        For complex rules, this would involve more sophisticated logic.
        """
        modified_attrs = [attrs.copy() for attrs in base_objects_attrs]
        # Example rule enforcement (simplified)
        if target_rule and target_rule.name == "all_circles":
            for attrs in modified_attrs:
                attrs['shape'] = 'circle'
            if not satisfy_rule and num_objects > 0:  # Violate: make one non-circle
                modified_attrs[0]['shape'] = random.choice([s for s in self.object_types if s != 'circle'])
        
        elif target_rule and target_rule.name == "all_same_color_red":
            for attrs in modified_attrs:
                attrs['color'] = 'red'
            if not satisfy_rule and num_objects > 0:  # Violate: make one non-red
                modified_attrs[0]['color'] = random.choice([c for c in self.colors if c != 'red'])
        elif target_rule and target_rule.name == "exists_large_square":
            if satisfy_rule:  # Ensure at least one large square
                if num_objects > 0:
                    modified_attrs[0]['shape'] = 'square'
                    modified_attrs[0]['size'] = 'large'
            else:  # Violate: Ensure no large squares
                for attrs in modified_attrs:
                    if attrs['shape'] == 'square' and attrs['size'] == 'large':
                        attrs['size'] = random.choice([s for s in self.sizes if s != 'large'])
                        if attrs['size'] == 'large':  # If still large, change shape
                            attrs['shape'] = random.choice([s for s in self.object_types if s != 'square'])
        
        elif target_rule and target_rule.name == "small_above_large":
            # This rule is relational and harder to enforce procedurally without position control.
            # For now, we'll assume the random placement might satisfy/violate, or generate simple attributes.
            # A more advanced generator would adjust positions.
            if satisfy_rule and num_objects >= 2:
                modified_attrs[0]['size'] = 'small'
                modified_attrs[1]['size'] = 'large'
            elif not satisfy_rule and num_objects >= 2:
                # Try to make sure no small is above large by changing sizes
                modified_attrs[0]['size'] = 'large'
                modified_attrs[1]['size'] = 'small'
        # Add more rule-specific generation logic here based on ALL_BONGARD_RULES
        # This is the most complex part of synthetic data generation for arbitrary rules.
        # For now, it's a simplified mapping.
        return modified_attrs

    def generate_bongard_problem(self, image_size: int) -> Tuple[Image.Image, Image.Image, int, Dict[str, Any], Dict[str, Any], List[Image.Image], List[int], List[Dict[str, Any]]]:
        """
        Generates a single Bongard problem, including a query pair and a support set.
        The support set will consist of `num_pos_support` positive examples and
        `num_neg_support` negative examples for the chosen rule.
        """
        # Dummy rule if ALL_BONGARD_RULES is empty (e.g., for testing)
        if not self.all_bongard_rules:
            class DummyRule:
                def __init__(self, name): self.name = name
            rule_name = "all_circles"
            rule = DummyRule(rule_name)
            logger.warning("ALL_BONGARD_RULES is empty. Using a dummy rule 'all_circles' for generation.")
        else:
            rule_name = random.choice(list(self.all_bongard_rules.keys()))
            rule = self.all_bongard_rules[rule_name]

        min_objects = self.synthetic_config['min_objects_per_image']
        max_objects = self.synthetic_config['max_objects_per_image']
        num_objects_query = random.randint(min_objects, max_objects)
        # Generate Query Pair
        query_img_pos, query_sg_pos = self._generate_single_image_with_rule(image_size, num_objects_query, rule, satisfy_rule=True)
        query_img_neg, query_sg_neg = self._generate_single_image_with_rule(image_size, num_objects_query, rule, satisfy_rule=False)
        # Randomly assign query_img1 and query_img2
        if random.random() < 0.5:
            query_image_view1 = query_img_pos
            query_scene_graph_view1 = query_sg_pos
            query_image_view2 = query_img_neg
            query_scene_graph_view2 = query_sg_neg
            query_label = 1  # View1 is positive
        else:
            query_image_view1 = query_img_neg
            query_scene_graph_view1 = query_sg_neg
            query_image_view2 = query_img_pos
            query_scene_graph_view2 = query_sg_pos
            query_label = 0  # View1 is negative
        # Generate Support Set
        support_images = []
        support_labels = []
        support_scene_graphs = []
        num_pos_support = self.synthetic_config['min_support_images_per_problem']  # At least this many positive
        num_neg_support = self.synthetic_config['min_support_images_per_problem']  # At least this many negative
        
        # Ensure total support images does not exceed max_support_images_per_problem
        total_support_needed = num_pos_support + num_neg_support
        if total_support_needed > self.synthetic_config['max_support_images_per_problem']:
            num_pos_support = self.synthetic_config['max_support_images_per_problem'] // 2
            num_neg_support = self.synthetic_config['max_support_images_per_problem'] - num_pos_support
        for _ in range(num_pos_support):
            num_objects_support = random.randint(min_objects, max_objects)
            img, sg = self._generate_single_image_with_rule(image_size, num_objects_support, rule, satisfy_rule=True)
            support_images.append(img)
            support_labels.append(1)
            support_scene_graphs.append(sg)
        for _ in range(num_neg_support):
            num_objects_support = random.randint(min_objects, max_objects)
            img, sg = self._generate_single_image_with_rule(image_size, num_objects_support, rule, satisfy_rule=False)
            support_images.append(img)
            support_labels.append(0)
            support_scene_graphs.append(sg)
        
        # Shuffle support set
        combined_support = list(zip(support_images, support_labels, support_scene_graphs))
        random.shuffle(combined_support)
        support_images, support_labels, support_scene_graphs = zip(*combined_support)
        support_images = list(support_images)
        support_labels = list(support_labels)
        support_scene_graphs = list(support_scene_graphs)
        return (query_image_view1, query_image_view2, query_label, query_scene_graph_view1, query_scene_graph_view2,
                support_images, support_labels, support_scene_graphs)

# --- DALI Pipeline Definition ---
@pipeline_def
def dali_bongard_pipeline(file_root, batch_size, sample_per_gpu, random_shuffle, decode_device, aug_device, image_size, mean_norm, std_norm):
    """
    DALI pipeline for loading and augmenting Bongard problem images.
    It expects a file_root containing subdirectories for each problem,
    with 'view1.png', 'view2.png', and potentially support images.
    Labels are assumed to be derived or loaded separately if not directly from file names.
    This pipeline is tailored for image data, not for scene graph or other metadata.
    """
    # 1) File reader - This needs to be adapted to read pairs of images (view1, view2)
    # and potentially support images. DALI's file reader is typically for single files.
    # For Bongard problems, a custom ExternalSource might be more suitable if images are
    # not in a flat directory structure with simple labels.
    # For demonstration, let's assume a simplified scenario where file_root points to
    # a directory of individual images, and labels are generated or mapped.
    # In a real Bongard setup with DALI, you'd feed paths/metadata via ExternalSource.

    # For now, let's assume the file_root contains images that represent
    # either view1 or view2, and we'll process them in pairs outside DALI,
    # or rely on an ExternalSource for more complex data structures.
    # The user's example implies a file_root for images and labels.
    # This part needs careful consideration for the actual Bongard dataset structure.

    # As per user's request, we'll use fn.readers.file as a placeholder for image loading.
    # This will read image files and (dummy) labels.
    # For actual Bongard problems, you'd need a more sophisticated data source.
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=random_shuffle,
        pad_last_batch=True, # Ensures all batches have the same size
        name="Reader",
        # sample_per_gpu is handled by DALIGenericIterator's epoch_size parameter
        # and the overall batch_size in the pipeline.
        # It's not a direct parameter to fn.readers.file for this use case.
        # The DALIGenericIterator will manage splitting the batch across GPUs.
    )

    # 2) Decode on mixed device (CPU/GPU)
    # This decodes JPEG images. If your images are PNG, use fn.decoders.image instead.
    images = fn.decoders.image(
        jpegs, device=decode_device, output_type=types.RGB
    )

    # 3) GPU-based augmentations
    # Resize to 256x256 first, then crop to 224x224 for common ImageNet preprocessing
    images = fn.resize(images, device=aug_device, resize_x=256, resize_y=256)
    images = fn.crop_mirror_normalize(
        images,
        device=aug_device,
        crop=(image_size, image_size), # Use image_size from config (e.g., 224)
        mean=[m * 255 for m in mean_norm], # DALI expects 0-255 range for mean/std
        std=[s * 255 for s in std_norm],
        mirror=fn.random.coin_flip(probability=0.5) if random_shuffle else False,  # Apply random horizontal flip
    )

    return images, labels # Returning images and (dummy) labels

def get_dataloader(cfg: Dict[str, Any], is_train: bool, rank: int = 0, world_size: int = 1):
    """
    Provides a DataLoader (either PyTorch's or DALI's DALIGenericIterator)
    based on the configuration.
    """
    if cfg['training']['use_dali'] and HAS_DALI:
        logger.info(f"Using DALI DataLoader for {'training' if is_train else 'validation'}.")
        # Determine file_root based on synthetic vs real data
        if cfg['data']['use_synthetic_data']:
            # DALI's file reader needs actual files. For synthetic data, we'd need
            # to pre-generate images to disk or use an ExternalSource.
            # For this demonstration, we'll use a dummy file_root and warn.
            # In a real scenario, synthetic data would be generated and saved,
            # or the DALI pipeline would be fed via an ExternalSource.
            file_root = os.path.join(cfg['data']['data_root_path'], 'synthetic_dali_temp')
            os.makedirs(file_root, exist_ok=True)
            logger.warning(f"DALI with synthetic data requires pre-generated images. Using dummy file_root: {file_root}. "
                           "You need to ensure images are present here for DALI to work correctly.")
            # Create some dummy image files for DALI to read
            for i in range(10): # Create 10 dummy images
                img = Image.new('RGB', (cfg['data']['image_size'], cfg['data']['image_size']), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                img.save(os.path.join(file_root, f'dummy_img_{i}.png'))
        else:
            file_root = os.path.join(cfg['data']['data_root_path'], cfg['data']['real_data_config']['dataset_name'], 'images')
            if not os.path.exists(file_root):
                logger.error(f"Real data root for DALI not found: {file_root}. DALI will fail.")
                raise FileNotFoundError(f"Real data root for DALI not found: {file_root}")

        pipe = dali_bongard_pipeline(
            batch_size=cfg['training']['batch_size'], # This is the per-GPU batch size
            file_root=file_root,
            sample_per_gpu=cfg['training']['dali']['sample_per_gpu'],
            random_shuffle=(cfg['training']['dali']['sample_mode'] == "random") and is_train, # Only shuffle for training
            decode_device=cfg['training']['dali']['decode_device'],
            aug_device=cfg['training']['dali']['aug_device'],
            num_threads=cfg['data']['dataloader_workers'],
            device_id=rank, # Assign pipeline to specific GPU
            image_size=cfg['data']['image_size'],
            mean_norm=IMAGENET_MEAN,
            std_norm=IMAGENET_STD
        )
        pipe.build()

        # DALIGenericIterator acts as the PyTorch DataLoader interface for DALI pipelines
        # reader_name is important if you have multiple readers in your DALI pipeline.
        # epoch_size is crucial for defining the length of an epoch and for distributed training.
        # It should be the total number of samples / world_size for each GPU.
        # For a file reader, DALI automatically determines this based on the files.
        # For synthetic data, you might need to manage this more carefully.
        
        # For the file reader, DALI automatically handles the number of samples.
        # The effective batch size for PyTorch will be pipe.batch_size * world_size.
        # Each DALIGenericIterator instance on each GPU will yield batches of size pipe.batch_size.
        return DALIGenericIterator(
            [pipe],
            ['images', 'labels'], # Output names from define_graph
            reader_name='Reader',
            auto_reset=True, # Reset pipeline at the end of each epoch
            fill_last_batch=True, # Ensure last batch is filled (padded if necessary)
            # For distributed training, each DALIGenericIterator should have its own pipeline
            # and the epoch_size should be set to the number of samples it should process.
            # DALI's file reader handles sharding automatically if device_id is set.
            # epoch_size=len(dataset) // world_size # This would be needed for external source
        )
    else:
        logger.info(f"Using PyTorch DataLoader for {'training' if is_train else 'validation'}.")
        # Your existing PyTorch DataLoader logic
        if cfg['data']['use_synthetic_data']:
            # Create a dummy BongardGenerator if not already defined
            if 'bongard_generator' not in globals():
                global bongard_generator
                bongard_generator = BongardGenerator(cfg['data'], ALL_BONGARD_RULES)
            dataset = BongardSyntheticDataset(
                cfg,
                bongard_generator,
                num_samples=cfg['training']['batch_size'] * 1000 if is_train else cfg['training']['batch_size'] * 100
            )
        else:
            train_data_list, val_data_list = load_bongard_data(
                cfg['data']['data_root_path'],
                cfg['data']['real_data_config']['dataset_name'],
                cfg['data']['real_data_config']['train_split_ratio']
            )
            dataset = RealBongardDataset(train_data_list if is_train else val_data_list)
        
        # Check for few-shot enablement
        if cfg['few_shot']['enable'] and is_train:
            logger.info("Few-shot learning enabled. Using PrototypicalSampler.")
            sampler = PrototypicalSampler(
                labels=dataset.labels, # Dataset must expose a .labels attribute
                n_way=cfg['few_shot']['n_way'],
                k_shot=cfg['few_shot']['k_shot'],
                q_query=cfg['few_shot']['q_query'],
                episodes=cfg['few_shot']['episodes']
            )
            # When using batch_sampler, batch_size, shuffle, and drop_last should not be set
            return DataLoader(
                dataset,
                batch_sampler=sampler, # Use batch_sampler for episodic sampling
                num_workers=cfg['data']['dataloader_workers'],
                pin_memory=True,
                collate_fn=custom_collate_fn # Still use custom collate function
            )
        else:
            # Initialize CurriculumSampler if enabled (only if few-shot is not enabled or not training)
            curriculum_sampler = None
            if cfg['training']['curriculum_learning'] and cfg['training']['curriculum_config']['enabled']:
                curriculum_sampler = CurriculumSampler(
                    dataset,
                    cfg['training'],
                    num_replicas=world_size,
                    rank=rank
                )
                dataset.curriculum_sampler = curriculum_sampler # Link sampler to dataset
            
            # Use DistributedSampler for DDP if not using curriculum sampler
            pytorch_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train) \
                            if world_size > 1 and curriculum_sampler is None else curriculum_sampler

            return DataLoader(
                dataset,
                batch_size=cfg['training']['batch_size'],
                sampler=pytorch_sampler, # Use custom sampler or DistributedSampler
                shuffle=is_train if pytorch_sampler is None else False, # Shuffle only if no sampler is used
                num_workers=cfg['data']['dataloader_workers'],
                pin_memory=True,
                collate_fn=custom_collate_fn, # Use existing custom collate function
                drop_last=True # Ensure consistent batch sizes
            )

# --- Synthetic Data Generation Logic (rest of the file remains the same) ---
# ... (BongardGenerator and related functions) ...
