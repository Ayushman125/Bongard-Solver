# Folder: bongard_solver/
# File: imports.py
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw, ImageFont # Import ImageFilter, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss, roc_auc_score, accuracy_score
import logging
import yaml
import argparse
import math
import random
import zipfile
import requests
from tqdm.auto import tqdm
import collections
import functools
import time
import json
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Any, Optional, Union # Added Union
import io # Needed for BytesIO

# PyTorch specific imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights
from torchvision.models import mobilenet_v3_small, efficientnet_b0
import torchvision.transforms as T
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim.swa_utils as swa_utils
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision.io
import torch.profiler

# DALI imports
try:
    import nvidia.dali
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    from nvidia.dali import pipeline_def # NEW: For decorated DALI pipelines
    HAS_DALI = True
except ImportError:
    HAS_DALI = False
    logging.warning("NVIDIA DALI not found. DALI functionalities will be disabled.")

# PyTorch Geometric imports
try:
    # Ensure these are installed: pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
    from torch_geometric.nn import GCNConv, Set2Set
    from torch_geometric.data import Data, Batch # Import Data and Batch for graph handling
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    logging.warning("PyTorch Geometric not found. GNN functionalities will be disabled.")
    # Dummy classes/functions to prevent errors if PyG is not installed
    class GCNConv(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); logging.warning("Dummy GCNConv used.")
        def forward(self, x, edge_index): return x
    class Set2Set(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.output_size = kwargs.get('hidden_size', 256) * 2; logging.warning("Dummy Set2Set used.")
        def forward(self, x, batch): return torch.zeros(batch.max().item() + 1 if batch.numel() > 0 else 1, self.output_size, device=x.device)
    class Data:
        def __init__(self, x, edge_index, edge_attr, batch):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.batch = batch
    class Batch:
        pass # Dummy Batch class

# Set up logging before any logger calls
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log DALI version
if HAS_DALI:
    try:
        logger.info(f"NVIDIA DALI version: {nvidia.dali.__version__}")
    except Exception as e:
        logger.warning(f"Could not determine DALI version: {e}")

# Conditional imports for optional libraries
HAS_ELASTIC_TRANSFORM = False
try:
    from torchvision.transforms import ElasticTransform
    HAS_ELASTIC_TRANSFORM = True
except ImportError:
    pass

# SAM (Sharpness-Aware Minimization) for optimizer
HAS_SAM = False
try:
    # Attempt to import your local SAM implementation
    from sam import SAM  # Assuming sam.py is in the same directory or on PYTHONPATH
    HAS_SAM = True
    logger.info("Successfully imported SAM from local sam.py.")
except ImportError:
    logger.warning("Could not import local sam.py. SAM optimizer will not be available.")
    # Dummy SAM class to prevent NameError if SAM is not available
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
            logger.warning("Using dummy SAM optimizer. Please install sam-pytorch or ensure sam.py is accessible for full SAM functionality.")
            self.base_optimizer = base_optimizer
            super().__init__(params, base_optimizer.defaults)
        
        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for p in self.param_groups:
                for d in p['params']:
                    if d.grad is None: continue
                    self.state[d]['old_p'] = d.data.clone() # Store original parameter
                    eps = p['rho'] / (grad_norm + 1e-12) # Use rho from param_group
                    d.add_(d.grad, alpha=eps) # Update for first step
            if zero_grad: self.zero_grad()
        
        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for p in self.param_groups:
                for d in p['params']:
                    if d.grad is None: continue
                    d.data = self.state[d]['old_p'] # Restore original parameter
            self.base_optimizer.step() # Apply base optimizer step
            if zero_grad: self.zero_grad()
        def _grad_norm(self):
            # Calculate the gradient norm for SAM
            norm = torch.norm(torch.stack([
                (p.grad if p.grad is not None else torch.zeros_like(p)).norm(p=2)
                for group in self.param_groups for p in group['params']
            ]), p=2)
            return norm

HAS_WANDB = False
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    pass

HAS_KORNIA = False
try:
    import kornia.augmentation as K
    import kornia.geometry.transform as KGT
    import kornia.color as KC
    import kornia.filters as KF
    from kornia.models import SegmentAnything # For SAM mask-based fallback
    HAS_KORNIA = True
except ImportError:
    pass

# Dummy for timm and sophia-pytorch if not installed
HAS_TIMM = False
try:
    import timm
    HAS_TIMM = True
except ImportError:
    pass

HAS_SOPHIA = False
try:
    from sophia import SophiaG
    HAS_SOPHIA = True
    logger.info("Successfully imported SophiaG.")
except ImportError:
    logger.warning("Could not import SophiaG. Sophia optimizer will not be available.")
    # Dummy SophiaG class if not installed
    class SophiaG(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
            logger.warning("Using dummy SophiaG optimizer. Please install sophia-pytorch for full functionality.")
            defaults = dict(lr=lr, weight_decay=weight_decay)
            super().__init__(params, defaults)
        def step(self, closure=None):
            if closure is not None:
                closure()
            logger.debug("Dummy SophiaG step called.")

# LoRA Adapters
HAS_LORA = False
try:
    from loralib import LoRA
    HAS_LORA = True
except ImportError:
    logger.warning("Could not import loralib. LoRA adapters will not be available.")
    # Dummy LoRA class
    class LoRA(nn.Module):
        def __init__(self, model, r=4, lora_alpha=32):
            super().__init__()
            self.model = model
            logger.warning("Using dummy LoRA. Please install loralib for full functionality.")
        def forward(self, x):
            return self.model(x)

# timm optimizers (Lion, MADGRAD)
HAS_TIMM_OPTIM = False
try:
    # timm.optim is not a direct module, optimizers are usually accessed via timm.create_optimizer_v2
    # For direct import, we might need torch_optimizer
    from torch_optimizer import Lion, MADGRAD # Assuming torch_optimizer is installed
    HAS_TIMM_OPTIM = True
except ImportError:
    logger.warning("Could not import Lion or MADGRAD from torch_optimizer. These optimizers will not be available.")

# Quantization imports
HAS_TORCH_QUANTIZATION = False
try:
    import torch.quantization as tq # CRITICAL FIX: Added alias 'tq'
    HAS_TORCH_QUANTIZATION = True
except ImportError:
    logger.warning("torch.quantization not available. QAT will be disabled.")

# Ultralytics (YOLO) imports
HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO as RealYOLO # Import RealYOLO from ultralytics
    HAS_ULTRALYTICS = True
except ImportError:
    logger.warning("Ultralytics YOLO not found. YOLO functionalities will be disabled.")
    # Dummy YOLO class to prevent NameError
    class RealYOLO:
        def __init__(self, model_path):
            logger.warning("Using dummy YOLO. Please install ultralytics for full YOLO functionality.")
            self.model_path = model_path
        def __call__(self, img, verbose=False, augment=False):
            # Return dummy results
            class DummyBoxes:
                def __init__(self):
                    self.conf = torch.tensor([])
                    self.cls = torch.tensor([])
                    self.xyxy = torch.tensor([])
            class DummyResult:
                def __init__(self):
                    self.boxes = DummyBoxes()
            return [DummyResult()]

# Segment Anything Model (SAM) for segmentation
HAS_SAM_SEG = False
try:
    # Assuming segment_anything is installed
    # from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    # For now, relying on Kornia's SegmentAnything if available.
    # If a direct SAM import is needed, add it here.
    pass # Handled by HAS_KORNIA for now
except ImportError:
    logger.warning("Segment Anything Model (SAM) not found. SAM segmentation functionalities will be disabled.")


# --- Utility Functions (moved from config.py and main.py for better organization) ---

def set_seed(seed: int):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"All random seeds set to {seed}.")

def _calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Compute the area of the intersection rectangle
    inter_width = xB - xA
    inter_height = yB - yA
    
    if inter_width <= 0 or inter_height <= 0:
        return 0.0 # No intersection
    inter_area = inter_width * inter_height
    # Compute the area of both the prediction and ground-truth rectangles
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou

# --- Dummy Classes/Functions for modules not in imports.py ---
# These are placeholders for classes/functions that are expected to be imported
# from other project files (e.g., models.py, data.py, etc.) but are not defined
# within this imports.py file itself. They allow other modules to import them
# without immediate NameErrors if those files haven't been loaded yet or are missing.

class BongardGenerator:
    """Dummy BongardGenerator for type hinting and preventing NameErrors."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy BongardGenerator used. Please ensure the real BongardGenerator is imported.")
    def generate_problem(self):
        # Return dummy data matching the expected structure
        img_size = (96, 96) # Example size
        query_img1_np = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        query_img2_np = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        label = 0
        gt_json_view1 = b'{}'
        gt_json_view2 = b'{}'
        difficulty = 0.5
        affine1 = np.eye(3).tolist()
        affine2 = np.eye(3).tolist()
        original_index = 0
        padded_support_imgs_np = [np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)] * 5
        padded_support_labels = [-1] * 5
        padded_support_sgs_bytes = [b'{}'] * 5
        num_support_per_problem_tensor = torch.tensor(0, dtype=torch.long)
        tree_indices_tensor = torch.tensor(0, dtype=torch.long)
        is_weights_tensor = torch.tensor(1.0, dtype=torch.float)
        dummy_query_bboxes_view1 = [[]]
        dummy_query_masks_view1 = [[]]
        dummy_query_bboxes_view2 = [[]]
        dummy_query_masks_view2 = [[]]
        dummy_support_bboxes_flat = [[] for _ in range(5)]
        dummy_support_masks_flat = [[] for _ in range(5)]

        return (query_img1_np, query_img2_np, label,
                gt_json_view1, gt_json_view2, difficulty, affine1, affine2, original_index,
                padded_support_imgs_np, padded_support_labels, padded_support_sgs_bytes,
                num_support_per_problem_tensor, tree_indices_tensor, is_weights_tensor,
                dummy_query_bboxes_view1, dummy_query_masks_view1,
                dummy_query_bboxes_view2, dummy_query_masks_view2,
                dummy_support_bboxes_flat, dummy_support_masks_flat)

class BongardSyntheticDataset(Dataset):
    """Dummy BongardSyntheticDataset."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy BongardSyntheticDataset used.")
        self.num_samples = 10
        self.data = [(b'', b'{}')] * self.num_samples # Dummy data
    def __len__(self): return self.num_samples
    def __getitem__(self, idx): return self.data[idx % self.num_samples]

class BongardExternalSource:
    """Dummy BongardExternalSource."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy BongardExternalSource used.")
        self.dataset = BongardSyntheticDataset()
        self.batch_size = 1
        self.indices = list(range(len(self.dataset)))
        self.i = 0
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i >= len(self.indices): raise StopIteration
        batch_imgs = []
        batch_gts = []
        for _ in range(self.batch_size):
            if self.i >= len(self.indices): break
            img_bytes, gt_json_string = self.dataset[self.indices[self.i]]
            batch_imgs.append(np.frombuffer(img_bytes, dtype=np.uint8))
            batch_gts.append(np.array(gt_json_string, dtype=np.object_))
            self.i += 1
        return batch_imgs, batch_gts

@pipeline_def
def dali_pipe(*args, **kwargs):
    """Dummy DALI pipeline definition."""
    logger.warning("Dummy DALI pipeline used. Please ensure the real dali_pipe is imported.")
    # Define dummy outputs matching the expected structure of the real pipeline
    batch_size = kwargs.get('batch_size', 1)
    height = kwargs.get('height', 96)
    width = kwargs.get('width', 96)
    num_channels = kwargs.get('num_channels', 3)

    dummy_img = fn.constant(0.0, shape=[height, width, num_channels], dtype=types.FLOAT, device="gpu")
    dummy_img = fn.crop_mirror_normalize(dummy_img, output_layout=types.NCHW)

    dummy_label = fn.constant(0, shape=[1], dtype=types.INT64, device="gpu")
    dummy_json = fn.constant(np.array(["{}"], dtype=np.object_), shape=(1,), device="cpu")
    dummy_affine = fn.constant(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), shape=(2, 3), dtype=types.FLOAT, device="cpu")

    # Stack for batch size
    dummy_img_batch = fn.stack(*([dummy_img] * batch_size))
    dummy_label_batch = fn.stack(*([dummy_label] * batch_size))
    dummy_json_batch = fn.stack(*([dummy_json] * batch_size))
    dummy_affine_batch = fn.stack(*([dummy_affine] * batch_size))

    return (dummy_img_batch, dummy_img_batch, dummy_label_batch, dummy_json_batch, dummy_affine_batch, dummy_affine_batch)


class CurriculumSampler(torch.utils.data.Sampler):
    """Dummy CurriculumSampler."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy CurriculumSampler used.")
        self.num_samples = 100
        self.indices = list(range(self.num_samples))
        self.current_epoch = 0
    def set_epoch(self, epoch): self.current_epoch = epoch
    def get_epoch_data(self): return [f"dummy_path_{i}.png" for i in self.indices], [0] * self.num_samples
    def __iter__(self): return iter(self.indices)
    def __len__(self): return self.num_samples

class RealObjectDetector:
    """Dummy RealObjectDetector."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy RealObjectDetector used.")
        self._sam_model = None # Placeholder for SAM model
    def detect_objects(self, image_pil: Image.Image, overall_confidence: float = 0.3):
        # Return dummy detections
        return [{"id": 0, "bbox": [10, 10, 50, 50], "type": "dummy_shape", "confidence": 0.9}]

class AttributeClassifier(nn.Module):
    """Dummy AttributeClassifier."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Dummy AttributeClassifier used.")
        self.dummy_feature_dim = 1280 # Example feature dim
        self.fill_head = nn.Linear(self.dummy_feature_dim, 1)
        self.color_head = nn.Linear(self.dummy_feature_dim, 1)
        self.size_head = nn.Linear(self.dummy_feature_dim, 1)
        self.orientation_head = nn.Linear(self.dummy_feature_dim, 1)
        self.shape_head = nn.Linear(self.dummy_feature_dim, 1)
        self.texture_head = nn.Linear(self.dummy_feature_dim, 1)
    def forward(self, x):
        features = torch.zeros(x.shape[0], self.dummy_feature_dim, device=x.device)
        return self.fill_head(features), self.color_head(features), self.size_head(features), \
               self.orientation_head(features), self.shape_head(features), self.texture_head(features), features

class RelationGNN(nn.Module):
    """Dummy RelationGNN."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Dummy RelationGNN used.")
        self.set2set = Set2Set(kwargs.get('node_feature_dim', 256), processing_steps=2, num_layers=1) # Dummy Set2Set
        self.num_relation_types = kwargs.get('num_relation_types', 21) # Example value
        self.edge_mlp = nn.Linear(512 + kwargs.get('edge_feature_dim', 6), self.num_relation_types) # Example dimension
    def forward(self, x, edge_index, batch):
        # Dummy forward pass
        graph_embedding = self.set2set(x, batch)
        return graph_embedding, x # Return dummy node features
    def classify_edges(self, node_features, edge_index, edge_attributes, node_types_for_edges):
        # Dummy edge classification
        num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
        return torch.zeros(num_edges, self.num_relation_types, device=node_features.device)

class PerceptionModule(nn.Module):
    """Dummy PerceptionModule."""
    def __init__(self, config):
        super().__init__()
        logger.warning("Dummy PerceptionModule used.")
        self.config = config
        self.real_object_detector = RealObjectDetector()
        self.attribute_classifier = AttributeClassifier()
        self.relation_gnn = RelationGNN(node_feature_dim=1280, edge_feature_dim=6, num_relation_types=21)
        self.bongard_classifier = nn.Linear(self.relation_gnn.set2set.output_size, config['model']['num_classes'])
        self.device = torch.device("cpu") # Dummy device
        self.use_knowledge_distillation = False
        self.teacher_model = None

    def forward(self, image_input: Any, is_synthetic: bool = False):
        # Dummy forward pass, return empty symbolic outputs and aggregated outputs
        symbolic_outputs_batch = []
        aggregated_outputs = {
            'attribute_logits': {'fill': torch.empty(0), 'color': torch.empty(0), 'size': torch.empty(0), 'orientation': torch.empty(0), 'shape': torch.empty(0), 'texture': torch.empty(0)},
            'attribute_gt': {'fill': torch.empty(0, dtype=torch.long), 'color': torch.empty(0, dtype=torch.long), 'size': torch.empty(0, dtype=torch.long), 'orientation': torch.empty(0, dtype=torch.long), 'shape': torch.empty(0, dtype=torch.long), 'texture': torch.empty(0, dtype=torch.long)},
            'relation_logits': torch.empty(0),
            'relation_gt': torch.empty(0, dtype=torch.long),
            'bongard_labels': torch.empty(0, dtype=torch.long),
            'image_features_student': torch.empty(0, self.relation_gnn.set2set.output_size),
            'image_features_teacher': torch.empty(0, self.relation_gnn.set2set.output_size)
        }
        return symbolic_outputs_batch, aggregated_outputs
    
    def mc_dropout_predict(self, *args, **kwargs):
        logger.warning("Dummy mc_dropout_predict used.")
        return [] # Return empty list for dummy

    def export_onnx(self, *args, **kwargs):
        logger.warning("Dummy export_onnx used.")

class DistillationLoss(nn.Module):
    """Dummy DistillationLoss."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Dummy DistillationLoss used.")
    def forward(self, *args, **kwargs): return torch.tensor(0.0)

class MixupCutmix:
    """Dummy MixupCutmix."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy MixupCutmix used.")
    def __call__(self, input, target): return input, target

class LabelSmoothingCrossEntropy(nn.Module):
    """Dummy LabelSmoothingCrossEntropy."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Dummy LabelSmoothingCrossEntropy used.")
    def forward(self, *args, **kwargs): return torch.tensor(0.0)

class FeatureConsistencyLoss(nn.Module):
    """Dummy FeatureConsistencyLoss."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Dummy FeatureConsistencyLoss used.")
    def forward(self, *args, **kwargs): return torch.tensor(0.0)

class KnowledgeReplayBuffer:
    """Dummy KnowledgeReplayBuffer."""
    def __init__(self, *args, **kwargs):
        logger.warning("Dummy KnowledgeReplayBuffer used.")
        self.buffer = collections.deque(maxlen=10)
    def add(self, *args, **kwargs): pass
    def sample(self, *args, **kwargs): return [], [], []
    def update_priorities(self, *args, **kwargs): pass
    def __len__(self): return 0

def enhance_logo_contrast(*args, **kwargs):
    logger.warning("Dummy enhance_logo_contrast used.")
    return Image.new('RGB', (1,1)) # Return a dummy image

def fallback_shape_detection_bw(*args, **kwargs):
    logger.warning("Dummy fallback_shape_detection_bw used.")
    return [] # Return empty list

def apply_symbolic_priors(*args, **kwargs):
    logger.warning("Dummy apply_symbolic_priors used.")
    return args[0] # Return original logits

def _train_step(*args, **kwargs):
    logger.warning("Dummy _train_step used.")
    return 0.0, {}, {}, 0.0, 0.0, 0.0 # Return dummy metrics

def _validate_model(*args, **kwargs):
    logger.warning("Dummy _validate_model used.")
    return 0.0, 0.0, 0.0, 0.0 # Return dummy metrics

def _plot_training_history(*args, **kwargs):
    logger.warning("Dummy _plot_training_history used.")

def _plot_reliability_diagram(*args, **kwargs):
    logger.warning("Dummy _plot_reliability_diagram used.")

def calibrate_model(*args, **kwargs):
    logger.warning("Dummy calibrate_model used.")

def apply_structured_pruning(*args, **kwargs):
    logger.warning("Dummy apply_structured_pruning used.")

def quantize_model_ptq(*args, **kwargs):
    logger.warning("Dummy quantize_model_ptq used.")

def quantize_model_qat(*args, **kwargs):
    logger.warning("Dummy quantize_model_qat used.")

def load_bongard_data(*args, **kwargs):
    logger.warning("Dummy load_bongard_data used.")
    # Return dummy data matching the expected format of load_bongard_data
    return [], [], [], [], np.array([]), np.array([])

