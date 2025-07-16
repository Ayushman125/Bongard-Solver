# coremodels/replay_buffer.py

import numpy as np
import random
import logging
from collections import deque
from typing import List, Tuple, Optional, Any, Union, Dict
import torch

logger = logging.getLogger(__name__)

class SumTree:
    """
    A SumTree data structure for efficient sampling and priority updates
    in Prioritized Experience Replay.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree stores priorities, leaves are at indices capacity-1 to 2*capacity-2
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        # Data stores the actual data (in this case, original dataset indices)
        self.data = np.zeros(capacity, dtype=object)  # Stores actual data indices
        self.data_pointer = 0  # Pointer to the next available data leaf
        self.current_size = 0  # Tracks the number of elements currently in the buffer

    def _propagate(self, idx: int, change: float):
        """Propagates priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieves a leaf node index based on a sampled value 's'."""
        left_child = 2 * idx + 1
        right_child = left_child + 1
        # If we are at a leaf node (or beyond the tree's extent for children)
        if left_child >= len(self.tree):
            return idx  # Reached a leaf node
        # Traverse down based on 's'
        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            return self._retrieve(right_child, s - self.tree[left_child])

    def total_priority(self) -> float:
        """Returns the sum of all priorities (root of the tree)."""
        return self.tree[0]

    def add(self, priority: float, data_index: int):
        """Adds a new data item with its priority."""
        # Calculate the tree index for the new leaf node
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Store the actual data (original dataset index)
        self.data[self.data_pointer] = data_index
        
        # Update the tree with the new priority
        self.update(tree_idx, priority)
        
        # Move the data pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Wrap around if capacity is reached
        
        self.current_size = min(self.current_size + 1, self.capacity)  # Update current size

    def update(self, tree_idx: int, priority: float):
        """Updates the priority of an existing data item."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Gets a (tree_idx, priority, data_index) tuple for a sampled value 's'.
        """
        tree_idx = self._retrieve(0, s)
        # Convert tree index to data index
        data_idx = tree_idx - self.capacity + 1
        
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

class KnowledgeReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer for Bongard problems.
    Stores problem indices and their associated priorities (e.g., based on loss).
    """
    def __init__(self, capacity: int, alpha_start: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
            alpha_start (float): Initial alpha value for prioritization (0 = uniform, 1 = full).
            beta_start (float): Initial beta value for importance sampling (0 = no IS, 1 = full IS).
            beta_frames (int): Number of frames over which beta will be annealed to 1.0.
        """
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha_start  # Controls how much prioritization is used (0 = uniform, 1 = full)
        self.beta = beta_start    # Controls importance sampling (IS) weight compensation
        self.beta_increment_per_sampling = (1.0 - beta_start) / beta_frames if beta_frames > 0 else 0.0  # How much beta increases per sample operation
        self.epsilon = 1e-6       # Small epsilon to prevent zero priority
        self.max_priority = 1.0   # Initial max priority (used for normalizing IS weights)
        self.frame = 0            # Total frames/steps processed, for annealing beta
        
        # Store batch-specific info for retrieval by collate_fn or training loop
        self._current_batch_original_indices = []
        self._current_batch_tree_indices = []
        self._current_batch_is_weights = []
        logger.info(f"KnowledgeReplayBuffer initialized with capacity {capacity}, alpha_start={alpha_start}, beta_start={beta_start}.")

    def add(self, data_index: int, priority: float):
        """
        Adds a new data index with its priority.
        The priority is transformed using alpha.
        Args:
            data_index (int): The original index of the data sample in the dataset.
            priority (float): The initial priority of this sample (e.g., its initial loss).
        """
        # Transform priority using alpha
        transformed_priority = (abs(priority) + self.epsilon) ** self.alpha
        self.sum_tree.add(transformed_priority, data_index)
        self.max_priority = max(self.max_priority, transformed_priority)  # Update max priority seen

    def sample(self, batch_size: int) -> Tuple[List[int], List[int], List[float]]:
        """
        Samples a batch of data indices based on their priorities.
        Returns:
            Tuple[List[int], List[int], List[float]]:
                - original_data_indices (List[int]): Indices of the sampled data in the original dataset.
                - tree_indices (List[int]): Indices of the sampled data in the SumTree.
                - is_weights (List[float]): Importance Sampling weights for each sampled data.
        """
        original_data_indices = []
        tree_indices = []
        is_weights = []
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        total_priority = self.sum_tree.total_priority()
        if total_priority == 0 or self.sum_tree.current_size < batch_size:
            logger.warning(f"Attempted to sample {batch_size} from replay buffer with {self.sum_tree.current_size} items. Returning empty lists.")
            return [], [], []
        
        segment = total_priority / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)  # Sample a value uniformly from each segment
            
            (tree_idx, priority, data_idx) = self.sum_tree.get(s)
            
            sampling_probability = priority / total_priority
            
            # Calculate Importance Sampling weight
            if sampling_probability == 0:
                is_weight = 0.0  # Should not happen if epsilon is used
            else:
                is_weight = (self.sum_tree.current_size * sampling_probability) ** (-self.beta)
            
            # Normalize IS weights by the maximum weight in the current batch or buffer
            # Normalizing by max_priority in the buffer is common.
            if self.max_priority > 0:  # Avoid division by zero
                is_weight = is_weight / self.max_priority
            else:
                is_weight = 0.0
            
            original_data_indices.append(data_idx)
            tree_indices.append(tree_idx)
            is_weights.append(is_weight)
        
        # Store for later update_priorities call
        self._current_batch_original_indices = original_data_indices
        self._current_batch_tree_indices = tree_indices
        self._current_batch_is_weights = is_weights
        
        self.frame += batch_size  # Increment frame count
        return original_data_indices, tree_indices, is_weights

    def update_priorities(self, tree_indices: List[int], losses: List[float], cfg: Dict[str, Any]):
        """
        Updates priorities of sampled experiences based on their new losses.
        The alpha value for prioritization is annealed based on the current epoch/frame.
        Args:
            tree_indices (List[int]): The SumTree indices of the experiences to update.
            losses (List[float]): New loss values for these experiences.
            cfg (Dict[str, Any]): Configuration dictionary containing replay buffer annealing settings.
        """
        if len(tree_indices) != len(losses):
            logger.error(f"Mismatch in lengths: tree_indices ({len(tree_indices)}) vs losses ({len(losses)}) for priority update.")
            return

        # Anneal alpha (prioritization exponent)
        alpha_start = cfg['replay'].get('alpha_start', 0.6)
        alpha_end = cfg['replay'].get('alpha_end', 0.0)
        anneal_frames = cfg['replay'].get('anneal_frames', 100000)  # Number of frames over which alpha anneals
        
        if anneal_frames <= 0:
            anneal_frames = 1  # Prevent division by zero
            logger.warning("replay.anneal_frames is zero or negative, setting to 1 to prevent division by zero.")
            
        # Linear annealing of alpha
        self.alpha = alpha_start + self.frame * (alpha_end - alpha_start) / anneal_frames
        self.alpha = max(alpha_end, min(alpha_start, self.alpha))  # Clamp alpha within bounds
        
        new_max_priority = 0.0
        for tree_idx, loss in zip(tree_indices, losses):
            # Transform loss into new priority using the current alpha
            new_priority = (abs(loss) + self.epsilon) ** self.alpha
            self.sum_tree.update(tree_idx, new_priority)
            new_max_priority = max(new_max_priority, new_priority)
        
        # Update the overall max_priority of the buffer
        self.max_priority = new_max_priority
        
        logger.debug(f"Replay buffer priorities updated. Frame: {self.frame}, Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}, Max Priority: {self.max_priority:.4f}")

    def set_current_batch_info(self, original_indices: List[int], tree_indices: List[int], is_weights: List[float]):
        """
        Stores the PER-related information for the current batch.
        This is a workaround for PyTorch DataLoader's inability to pass extra info
        from sampler to collate_fn directly.
        """
        self._current_batch_original_indices = original_indices
        self._current_batch_tree_indices = tree_indices
        self._current_batch_is_weights = is_weights

    def get_current_batch_info(self, original_index: int) -> Tuple[Optional[int], Optional[float]]:
        """
        Retrieves the tree_index and is_weight for a given original_index
        from the last sampled batch.
        This method is called by the Dataset's __getitem__ when PER is active.
        """
        try:
            # Find the position of the original_index within the last sampled batch
            pos_in_batch = self._current_batch_original_indices.index(original_index)
            return self._current_batch_tree_indices[pos_in_batch], self._current_batch_is_weights[pos_in_batch]
        except ValueError:
            logger.warning(f"Original index {original_index} not found in current sampled batch info. Returning None for PER info.")
            return None, None

# --- New simple ReplayBuffer for perception module training ---
class ReplayBuffer:
    """
    A simple FIFO (First-In, First-Out) Replay Buffer for storing experiences.
    Used for general perception module training (e.g., Phase 1 initial training).
    """
    def __init__(self, capacity: int = 10000):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences the buffer can hold.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, label: Union[int, torch.Tensor]):
        """
        Adds a new experience (state, label) to the buffer.

        Args:
            state (torch.Tensor): The input state (e.g., image tensor [C, H, W]).
            label (Union[int, torch.Tensor]): The corresponding label (e.g., int or one-hot tensor).
        """
        self.buffer.append((state, label))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of experiences uniformly from the buffer.

        Args:
            batch_size (int): The number of samples to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - states (torch.Tensor): A batch of state tensors.
                - labels (torch.Tensor): A batch of label tensors.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Replay buffer has only {len(self.buffer)} samples, but {batch_size} requested.")
            
        samples = random.sample(self.buffer, batch_size)
        states, labels = zip(*samples)
        
        # Ensure labels are converted to a tensor if they are integers
        # If labels are already tensors (e.g., one-hot), torch.stack will handle it.
        if isinstance(labels[0], int):
            return torch.stack(states), torch.tensor(labels)
        else: # Assume labels are already tensors (e.g., one-hot)
            return torch.stack(states), torch.stack(labels)

    def __len__(self) -> int:
        """
        Returns the current number of experiences in the buffer.
        """
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Checks if the buffer has enough samples to form a batch.
        """
        return len(self.buffer) >= batch_size

