import copy
import random
import numpy as np
import logging

# MCTS specific helper functions (from image_features)
from derive_label.geometric_detectors import detect_vertices
from derive_label.confidence_scoring import consensus_vote, DETECTOR_RELIABILITY
from derive_label.image_features import extract_clean_mask_and_skeleton

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state  # state: {'shapes': [...], 'unlabeled': [idxs], 'labels': {idx:label}}
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_terminal(self):
        return len(self.state['unlabeled']) == 0

    def expand(self, label_options_fn):
        # Expand by assigning a label to the next unlabeled shape
        if self.is_terminal():
            return []
        idx = self.state['unlabeled'][0]
        options = label_options_fn(self.state, idx)
        for label in options:
            new_state = copy.deepcopy(self.state)
def label_options_fn(state, idx):
    """
    Uses geometric/statistical techniques to propose label options for a shape, using consensus_vote and robust mask features.
    """
    shape = state['shapes'][idx]
    # Collect candidate labels from all detectors
    candidate_labels = shape.get('possible_labels', {})
    detector_confidences = shape.get('detector_confidences', {})
    # Optionally, add mask/skeleton QA as context
    mask, skeleton, qa = extract_clean_mask_and_skeleton(np.array(shape.get('vertices', [])))
    context = {'fill_ratio': qa.get('fill_ratio', 0.0)}
    final_label, label_ranking, label_sources = consensus_vote(
        candidate_labels, detector_confidences, DETECTOR_RELIABILITY, context)
    options = [final_label] if final_label != 'AMBIGUOUS' else [l for l, _ in label_ranking[:2]]
    return options

    new_state['labels'][idx] = label
    new_state['unlabeled'] = [i for i in new_state['unlabeled'] if i != idx]
    child = MCTSNode(new_state, parent=self, action=(idx, label))
    self.children.append(child)
    return self.children

    def best_child(self, c_param=1.4):
        choices_weights = [child.value / (child.visits + 1e-6) + \
                           c_param * ((2 * (self.visits + 1e-6))**0.5 / (child.visits + 1e-6))
                           for child in self.children]
        return self.children[choices_weights.index(max(choices_weights))]

def mcts_search(initial_state, label_options_fn, reward_fn, n_sim=100):
    root = MCTSNode(initial_state)
    for _ in range(n_sim):
        node = root
        # Selection
        while node.children:
            node = node.best_child()
        # Expansion
def reward_fn(state):
    """
    Calculates a reward for a given labeled state, incorporating mask/skeleton QA.
    """
    score = 0
    labels = list(state['labels'].values())
    if not labels:
        return 0
    for idx, l in enumerate(labels):
        shape = state['shapes'][idx]
        mask, skeleton, qa = extract_clean_mask_and_skeleton(np.array(shape.get('vertices', [])))
        fill_ratio = qa.get('fill_ratio', 0.0)
        score += fill_ratio
    # Uniformity bonus
    if len(set(labels)) == 1:
        score += 1
    return score / len(labels)
    if not node.is_terminal():
        node.expand(label_options_fn)
        if node.children:
            node = random.choice(node.children)
        # Simulation
        sim_state = copy.deepcopy(node.state)
        while sim_state['unlabeled']:
            idx = sim_state['unlabeled'][0]
            options = label_options_fn(sim_state, idx)
            if not options:
                break
            sim_state['labels'][idx] = random.choice(options)
            sim_state['unlabeled'] = [i for i in sim_state['unlabeled'] if i != idx]
        # Reward
        reward = reward_fn(sim_state)
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    # Return best labeling
    best = max(root.children, key=lambda c: c.value / c.visits if c.visits > 0 else -1, default=None)
    if best:
        return best.state['labels']
    return {}

def label_options_fn(state, idx):
    """
    Uses geometric/statistical techniques to propose label options for a shape.
    This function should ideally use properties already computed and stored in `state['shapes'][idx]`.
    """
    shape = state['shapes'][idx]
    options = []
    
    # Access properties directly from the shape object
    imgproc_features = shape.get('properties', {}).get('img_proc_features', {})
    
    if imgproc_features.get('n_contours', 0) == 1 and imgproc_features.get('n_holes', 0) == 0:
        options.append('polygon_imgproc')
    
    if shape.get('properties', {}).get('n_holes_shapely', 0) > 0:
        options.append('polygon_with_holes')
        
    if shape.get('properties', {}).get('symmetry', 'none') != 'none':
        options.append(f"symmetric_{shape['properties']['symmetry']}")
        
    if shape.get('degenerate', False):
        options.append('degenerate')
    
    if not options:
        options.append('unknown')
    return options

def reward_fn(state):
    """
    Calculates a reward for a given labeled state.
    """
    score = 0
    labels = list(state['labels'].values())
    if not labels:
        return 0
    for l in labels:
        if l.startswith('polygon'):
            score += 1
        if l.startswith('symmetric'):
            score += 1
        if l == 'degenerate':
            score -= 1
    # Uniformity bonus
    if len(set(labels)) == 1:
        score += 2
    return score / len(labels)

def label_scene_with_mcts(shapes, n_sim=100):
    """
    Applies MCTS to label shapes in a scene.
    """
    initial_state = {'shapes': shapes, 'unlabeled': list(range(len(shapes))), 'labels': {}}
    best_labels = mcts_search(initial_state, label_options_fn, reward_fn, n_sim=n_sim)
    # Attach best labels to shapes
    for idx, label in best_labels.items():
        shapes[idx]['mcts_label'] = label
    return shapes
