# Folder: bongard_solver/
# File: replay_buffer.py
import numpy as np
import random
import logging
from typing import List, Tuple, Optional, Any # Added Any for config type
import torch # Added for torch.from_numpy

logger = logging.getLogger(__name__)

class SumTree:
    """
    A SumTree data structure for efficient sampling and priority updates
    in Prioritized Experience Replay.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Stores actual data indices
        self.data_pointer = 0  # Pointer to the next available data leaf

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

        if left_child >= len(self.tree):
            return idx  # Reached a leaf node

        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            return self._retrieve(right_child, s - self.tree[left_child])

    def total_priority(self) -> float:
        """Returns the sum of all priorities (root of the tree)."""
        return self.tree[0]

    def add(self, priority: float, data_index: int):
        """Adds a new data item with its priority."""
        tree_idx = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data_index  # Store the original dataset index
        self.update(tree_idx, priority)  # Update the tree with the new priority
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # Wrap around if capacity is reached

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
        data_idx = tree_idx - self.capacity + 1  # Convert tree index to data index
        
        return tree_idx, self.tree[tree_idx], self.data[data_idx]

class KnowledgeReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer for Bongard problems.
    Stores problem indices and their associated priorities (e.g., based on loss).
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4):
        self.sum_tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # Controls how much prioritization is used (0 = uniform, 1 = full)
        self.beta = beta_start  # Controls importance sampling (IS) weight compensation
        self.beta_increment_per_sampling = 0.001  # How much beta increases per sample operation
        self.epsilon = 1e-6  # Small epsilon to prevent zero priority
        self.max_priority = 1.0  # Initial max priority
        self.epoch = 0 # Track epochs for annealing
        
        # Store batch-specific info for retrieval by collate_fn or training loop
        self._current_batch_original_indices = []
        self._current_batch_tree_indices = []
        self._current_batch_is_weights = []
        logger.info(f"KnowledgeReplayBuffer initialized with capacity {capacity}, alpha={alpha}, beta_start={beta_start}.")

    def add(self, data_index: int, priority: float):
        """Adds a new data index with its priority."""
        # Add a small epsilon to priority to avoid zero priority
        priority = (priority + self.epsilon) ** self.alpha
        self.sum_tree.add(priority, data_index)
        self.max_priority = max(self.max_priority, priority)  # Update max priority

    def sample(self, batch_size: int) -> Tuple[List[int], List[int], List[float]]:
        """
        Samples a batch of data indices based on their priorities.
        Returns: (original_data_indices, tree_indices, is_weights)
        """
        original_data_indices = []
        tree_indices = []
        is_weights = []
        segment = self.sum_tree.total_priority() / batch_size
        
        # Anneal beta (moved to update method for epoch-based annealing)
        # self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)  # Sample a value within the segment
            (tree_idx, priority, data_idx) = self.sum_tree.get(s)
            
            # Calculate importance sampling (IS) weight
            # P(i) = priority_i / total_priority
            sampling_probability = priority / self.sum_tree.total_priority()
            is_weight = (self.capacity * sampling_probability) ** (-self.beta)  # (N * P(i))^-beta
            is_weight = is_weight / self.max_priority  # Normalize by max_priority for stability

            original_data_indices.append(data_idx)
            tree_indices.append(tree_idx)
            is_weights.append(is_weight)
            
            # Update priority for the sampled experience (e.g., to a very low value initially)
            # This is typically done *after* the loss calculation, so we don't update here.
            # The `update_priorities` method will handle this.
        
        # Store for retrieval by collate_fn or training loop
        self._current_batch_original_indices = original_data_indices
        self._current_batch_tree_indices = tree_indices
        self._current_batch_is_weights = is_weights
        return original_data_indices, tree_indices, is_weights

    # 6.1 Annealing via Config
    def update(self, losses: List[float], cfg: Dict[str, Any]): # Pass cfg here
        """
        Updates priorities of sampled experiences based on their new losses.
        Also handles annealing of alpha and beta based on epoch.
        Args:
            losses (List[float]): New losses for the sampled experiences.
            cfg (Dict[str, Any]): Configuration dictionary containing replay buffer annealing settings.
        """
        if len(self.sum_tree.data_pointer) != len(losses): # Check against actual number of elements in tree
            logger.error(f"Mismatch in lengths: tree_indices ({len(self.sum_tree.data_pointer)}) vs losses ({len(losses)})")
            return

        # Anneal alpha and beta based on current epoch
        anneal_epochs = cfg['replay'].get('anneal_epochs', 1) # Default to 1 to avoid division by zero
        
        # Ensure ep does not exceed anneal_epochs
        ep = min(self.epoch, anneal_epochs) 

        # Linear annealing for alpha (priority exponent)
        alpha_start = cfg['replay'].get('alpha_start', 0.6)
        alpha_end = cfg['replay'].get('alpha_end', 0.0) # Anneal towards less prioritization
        self.alpha = alpha_start + ep * (alpha_end - alpha_start) / anneal_epochs

        # Linear annealing for beta (importance sampling exponent)
        beta_start = cfg['replay'].get('beta_start', 0.4)
        beta_end = cfg['replay'].get('beta_end', 1.0) # Anneal towards full importance sampling
        self.beta = beta_start + ep * (beta_end - beta_start) / anneal_epochs

        # Update priorities in the SumTree
        for tree_idx, loss in zip(self._current_batch_tree_indices, losses): # Use _current_batch_tree_indices
            new_priority = (abs(loss) + self.epsilon) ** self.alpha  # Use absolute loss for priority
            self.sum_tree.update(tree_idx, new_priority)
            self.max_priority = max(self.max_priority, new_priority)  # Update max priority

        # Increment epoch counter for annealing
        self.epoch += 1
        logger.debug(f"Replay buffer updated. Epoch: {self.epoch}, Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}")

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
            # Find the position of original_index in the last sampled batch
            pos_in_batch = self._current_batch_original_indices.index(original_index)
            return self._current_batch_tree_indices[pos_in_batch], self._current_batch_is_weights[pos_in_batch]
        except ValueError:
            # This should ideally not happen if the sampler is working correctly
            logger.warning(f"Original index {original_index} not found in current sampled batch info. Returning None for PER info.")
            return None, None

