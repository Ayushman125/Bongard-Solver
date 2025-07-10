import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import copy
from tqdm.auto import tqdm
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime # For TensorBoard log directory
import math # For OneCycleLR total_steps
import random # For rand_bbox
import json # Ensure json is imported for parsing DALI ground_truth_json
import jsonschema # For GT JSON validation
from sklearn.model_selection import KFold # For Meta-Learner Cross-Validation

# Set up logging for this module
logger = logging.getLogger(__name__)

# Define a basic JSON schema for ground truth validation
# You might need to expand this based on your actual GT JSON structure
BONGARD_GT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "bongard_label": {"type": "integer", "minimum": 0, "maximum": 1},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["id", "bbox", "attributes", "confidence"]
            }
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "src": {"type": "integer"},
                    "dst": {"type": "integer"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["src", "dst", "type", "confidence"]
            }
        }
    },
    "required": ["bongard_label", "objects", "relations"]
}


# IMPORTANT: These imports assume that phase1_code_part1.py and phase1_code_part2b.py
# (specifically the model definitions and core globals) have already been executed
# in your Jupyter Notebook session, making them globally available.
try:
    from phase1_code_part1 import (
        CONFIG, DEVICE, load_bongard_data, set_seed, IMAGENET_MEAN, IMAGENET_STD, NUM_CHANNELS,
        DATA_ROOT_PATH, BongardSyntheticDataset, BongardExternalSource,
        dali_pipe, DALIGenericIterator, LastBatchPolicy,
        load_config, # Assuming load_config is in phase1_code_part1
        CurriculumSampler, # Assuming CurriculumSampler is in phase1_code_part1
        ATTRIBUTE_FILL_MAP, # Assuming attribute maps are in phase1_code_part1
        RELATION_MAPS # Assuming relation maps are in phase1_code_part1
    )
    # Import core model definitions and necessary helper from phase1_code_part2b.py
    from phase1_code_part2b import (
        PerceptionModule, RealObjectDetector, # Core models
        HAS_SAM, HAS_SOPHIA, HAS_TIMM_OPTIM, HAS_TORCH_QUANTIZATION, HAS_WANDB, # Feature flags
        tq, swa_utils, autocast, GradScaler, SummaryWriter, # Training utilities
        # Assuming these are defined in phase1_code_part2b.py if not in current file
        LabelSmoothingCrossEntropy, FeatureConsistencyLoss, DistillationLoss, MixupCutmix
    )
    from sklearn.model_selection import train_test_split # Assuming this is imported in phase1_code_part1 or phase1_code_part2b
    from sklearn.metrics import accuracy_score, brier_score_loss # Assuming this is imported in phase1_code_part1 or phase1_code_part2b
    from PIL import Image # For loading images for symbolic output/MC dropout
    import matplotlib.pyplot as plt # For plotting history
    import torchvision.transforms as T # For converting PIL Image to Tensor for symbolic output saving
    
    # Try importing netcal for calibration
    try:
        from netcal.scaling import Dirichlet
        HAS_NETCAL = True
    except ImportError:
        logger.warning("netcal library not found. Advanced calibration (Dirichlet) will be skipped.")
        HAS_NETCAL = False

    # Try importing swag for SWAG
    try:
        from swag.posteriors import SWAG
        HAS_SWAG = True
    except ImportError:
        logger.warning("SWAG library not found. SWAG integration will be skipped.")
        HAS_SWAG = False

except ImportError as e:
    logger.error(f"Failed to import core components. Ensure phase1_code_part1.py and phase1_code_part2b.py are run first. Error: {e}")
    # Dummy definitions for standalone testing if imports fail
    # Added new ensemble-specific config parameters
    CONFIG = {'model': {'num_classes': 2, 'batch_size': 4, 'epochs': 10, 'initial_learning_rate': 0.001, 'max_learning_rate': 0.01, 'attribute_backbone': 'mobilenet_v2', 'gnn_depth': 2, 'object_detector_model_path': 'dummy.pt', 'random_seed': 42, 'mc_dropout_samples': 0}, 'data': {'image_size': [96, 96], 'initial_image_size': [64, 64], 'dataloader_num_workers': 2, 'use_synthetic_data': False, 'synthetic_samples': 100, 'train_test_split_ratio': 0.2, 'curriculum_annealing_epochs': 5, 'dali_device_memory_padding': 64*1024*1024, 'dali_host_memory_padding': 64*1024*1024, 'dali_prebuild_streams': 0}, 'training': {'use_amp': True, 'use_wandb': False, 'early_stopping_patience': 5, 'early_stopping_monitor_metric': 'val_loss', 'early_stopping_min_delta': 0.0001, 'checkpoint_dir': './checkpoints', 'model_checkpoint_name': 'best_model.pt', 'distillation_temperature': 2.0, 'distillation_alpha': 0.5, 'use_knowledge_distillation': False, 'weight_decay': 1e-4, 'scheduler': 'OneCycleLR', 'onecycle_pct_start': 0.3, 'gradient_accumulation_steps': 1, 'mixup_alpha': 0.0, 'cutmix_alpha': 0.0, 'mixup_cutmix_ratio': 0.5, 'label_smoothing_epsilon': 0.0, 'feature_consistency_alpha': 0.0, 'feature_consistency_loss_type': 'mse', 'symbolic_consistency_alpha': 0.0, 'use_sam_optimizer': False, 'use_swa': False, 'swa_start_epoch_ratio': 0.75, 'swa_lr': 0.05, 'use_qat': False, 'qat_start_epoch': 5, 'enable_profiler': False, 'profiler_schedule_wait': 1, 'profiler_schedule_warmup': 2, 'profiler_schedule_active': 10, 'profiler_schedule_repeat': 3, 'save_symbolic_outputs_interval_epochs': 5, 'validation_frequency_batches': 0, 'attribute_loss_weight': 1.0, 'relation_loss_weight': 1.0, 'max_grad_norm': 1.0, 'knowledge_replay_enabled': False, 'replay_buffer_size': 1000, 'calibrate_model': False, 'use_ptq': False, 'enable_structured_pruning': False, 'pruning_amount': 0.3, 'use_swag': False}, 'debug': {'log_level': 'INFO', 'visualize_training_history': True, 'plot_reliability_diagram': True}, 'ensemble': {'num_members': 3, 'train_members': True, 'inference_mode': 'simple_average', 'use_stacking': False, 'use_distillation': False, 'distilled_student_config_override': None}}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_ROOT_PATH = "./data" # Dummy path
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    NUM_CHANNELS = 3
    logger.warning("Using dummy CONFIG and DEVICE due to import failure. This script may not run correctly standalone.")

    # Dummy classes/functions if core imports fail (with updated signatures/methods)
    class PerceptionModule(nn.Module):
        def __init__(self, config, object_detector): super().__init__()
        def forward(self, images_tensor, ground_truth_json_strings=None): # Added detected_objects return
            # Dummy aggregated_outputs for _run_single_training_session_ensemble
            dummy_attr_logits = {attr: torch.randn(images_tensor.shape[0], 2) for attr in ['fill', 'color', 'size', 'orientation', 'shape', 'texture']}
            dummy_attr_gt = {attr: torch.randint(0, 2, (images_tensor.shape[0],)) for attr in ['fill', 'color', 'size', 'orientation', 'shape', 'texture']}
            dummy_rel_logits = torch.randn(images_tensor.shape[0], 2)
            dummy_rel_gt = torch.randint(0, 2, (images_tensor.shape[0],))
            dummy_image_features_student = torch.randn(images_tensor.shape[0], 128) # Dummy feature size
            dummy_image_features_teacher = torch.randn(images_tensor.shape[0], 128)
            dummy_bongard_labels = torch.randint(0, 2, (images_tensor.shape[0],))
            
            # Dummy detected_objects for scene graph extraction
            dummy_detected_objects = []
            for _ in range(images_tensor.shape[0]):
                num_obj = random.randint(1, 3)
                objs = []
                for i_obj in range(num_obj):
                    x1, y1 = random.randint(0, 50), random.randint(0, 50)
                    x2, y2 = random.randint(x1+10, 90), random.randint(y1+10, 90)
                    objs.append({"bbox": [x1, y1, x2, y2], "score": random.random(), "class_id": random.randint(0, 5)})
                dummy_detected_objects.append(objs)

            aggregated_outputs = {
                'attribute_logits': dummy_attr_logits,
                'attribute_gt': dummy_attr_gt,
                'relation_logits': dummy_rel_logits,
                'relation_gt': dummy_rel_gt,
                'image_features_student': dummy_image_features_student,
                'image_features_teacher': dummy_image_features_teacher,
                'bongard_labels': dummy_bongard_labels
            }
            return torch.randn(images_tensor.shape[0], 2), dummy_detected_objects, aggregated_outputs # Dummy bongard_logits, detected_objects, aggregated_outputs
        def load_weights(self, path): pass
        def mc_dropout_predict(self, images, num_samples, is_synthetic):
            return [{"image_path": "dummy.png", "bongard_prediction_mean": [0.5, 0.5], "bongard_prediction_variance": [0.01, 0.01]}]
        def export_onnx(self, output_path): logger.info(f"Dummy ONNX export to {output_path}")
        @property
        def device(self): return DEVICE
        def extract_scene_graph(self, detected_objects, attribute_logits, relation_logits):
            # Dummy implementation for now, actual implementation needs to be in phase1_code_part2b.py
            objects = []
            # detected_objects here is a list of objects for a single image, not a batch
            for i, obj in enumerate(detected_objects): 
                attrs = {
                    name: f"attr_val_{i}" # Dummy attribute value
                    for i, name in enumerate(attribute_logits.keys())
                }
                objects.append({
                    "id": i,
                    "bbox": obj["bbox"],
                    "attributes": attrs,
                    "confidence": float(obj["score"])
                })
            relations = []
            if relation_logits: # Dummy relations
                relations.append({"src": 0, "dst": 1, "type": "touching", "confidence": 0.9})
            
            # Add dummy uncertainty for symbolic output
            uncertainty = {"epistemic": 0.0, "aleatoric": 0.0}
            
            return {"objects": objects, "relations": relations, "uncertainty": uncertainty}
    class RealObjectDetector:
        def __init__(self, model_path, sam_model=None): pass
        def detect_objects(self, image_batch):
            # Dummy detected objects
            detected_objects_batch = []
            for _ in range(image_batch.shape[0]):
                num_obj = random.randint(1, 3)
                objs = []
                for i_obj in range(num_obj):
                    x1, y1 = random.randint(0, 50), random.randint(0, 50)
                    x2, y2 = random.randint(x1+10, 90), random.randint(y1+10, 90)
                    objs.append({"bbox": [x1, y1, x2, y2], "score": random.random(), "class_id": random.randint(0, 0)}) # class_id 0 for simplicity
                detected_objects_batch.append(objs)
            return detected_objects_batch
        def predict_with_sam(self, image_batch):
            # Dummy SAM masks
            return torch.randn(image_batch.shape[0], 1, image_batch.shape[2], image_batch.shape[3])
    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, **kwargs): self.base_optimizer = base_optimizer; super().__init__(params, base_optimizer.defaults)
        def first_step(self, zero_grad=False): pass
        def second_step(self, zero_grad=False): pass
    class SophiaG(torch.optim.Optimizer):
        def __init__(self, params, **kwargs): super().__init__(params, {})
        def step(self, closure=None): pass
    class Lion(torch.optim.Optimizer):
        def __init__(self, params, **kwargs): super().__init__(params, {})
        def step(self, closure=None): pass
    class MADGRAD(torch.optim.Optimizer):
        def __init__(self, params, **kwargs): super().__init__(params, {})
        def step(self, closure=None): pass
    class DALIGenericIterator:
        def __init__(self, pipeline, output_map, size, auto_reset, last_batch_policy):
            self.size = size; self.i = 0; self.output_map = output_map
            self.iterator = type('obj', (object,), {'get_current_image_size': lambda: (CONFIG['data']['image_size'][0], CONFIG['data']['image_size'][1]), 'set_epoch': lambda x: None, 'get_epoch_data': lambda: ([],[])})()
            self.pipeline = pipeline # Store pipeline to access args
        def __iter__(self): return self
        def __next__(self):
            if self.i * CONFIG['model']['batch_size'] >= self.size: raise StopIteration
            self.i += 1
            batch_size = min(CONFIG['model']['batch_size'], self.size - (self.i-1)*CONFIG['model']['batch_size'])
            return {
                'view1': torch.randn(batch_size, 3, 96, 96),
                'view2': torch.randn(batch_size, 3, 96, 96),
                'labels': torch.randint(0, 2, (batch_size, 1)),
                'gts_json': np.array([b'{"bongard_label":0, "objects":[], "relations":[]}']*batch_size, dtype=object),
                'affine1': torch.randn(batch_size, 2, 3),
                'affine2': torch.randn(batch_size, 2, 3)
            }
        def reset(self): self.i = 0
        def __len__(self): return math.ceil(self.size / CONFIG['model']['batch_size'])
        def release_gpu_memory(self): pass
    class LastBatchPolicy: PARTIAL = None; DROP = None
    class SummaryWriter:
        def __init__(self, log_dir): pass
        def add_scalar(self, tag, scalar_value, global_step=None): pass
        def close(self): pass
    class GradScaler:
        def __init__(self, enabled): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def update(self): pass
    class autocast:
        def __init__(self, enabled): pass
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class swa_utils:
        class AveragedModel(nn.Module):
            def __init__(self, model): super().__init__()
            def update_parameters(self, model): pass
        def update_bn(loader, model, device): pass
        def SWALR(optimizer, swa_lr): return type('SWALR', (object,), {'step': lambda: None})()
        def swap_parameters_with_avg(model, swa_model): pass
    class tq:
        def get_default_qat_qconfig(self, backend): return None
        def prepare_qat(self, model, inplace): pass
        def enable_observer(self, module): pass
        def enable_fake_quant(self, module): pass
        def convert(self, model, inplace): return model
        def get_default_qconfig(backend): return None
        def prepare(model, inplace): pass
    HAS_SAM = False; HAS_SOPHIA = False; HAS_TIMM_OPTIM = False; HAS_TORCH_QUANTIZATION = False; HAS_WANDB = False
    HAS_NETCAL = False; HAS_SWAG = False
    
    # Dummy rand_bbox and distillation_loss if they are not imported
    def rand_bbox(size, lam):
        W = size[2]; H = size[3]
        cut_rat = np.sqrt(1. - lam); cut_w = np.int(W * cut_rat); cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W); cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
    def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)
        dist_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
        hard_loss = F.cross_entropy(student_logits, labels)
        return alpha * dist_loss + (1.0 - alpha) * hard_loss
    
    # Dummy classes for losses and transforms if not imported
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.0): super().__init__(); self.smoothing = smoothing
        def forward(self, input, target): return F.cross_entropy(input, target, label_smoothing=self.smoothing)
    class FeatureConsistencyLoss(nn.Module):
        def __init__(self, loss_type='mse'): super().__init__(); self.loss_type = loss_type
        def forward(self, f1, f2):
            if self.loss_type == 'mse': return F.mse_loss(f1, f2)
            elif self.loss_type == 'cosine': return (1 - F.cosine_similarity(f1, f2)).mean()
            elif self.loss_type == 'kl_div': return F.kl_div(F.log_softmax(f1, dim=-1), F.softmax(f2, dim=-1), reduction='batchmean')
            else: raise ValueError("Unknown loss type")
    class DistillationLoss(nn.Module):
        def __init__(self, alpha, temperature, base_loss_fn): super().__init__(); self.alpha = alpha; self.temperature = temperature; self.base_loss_fn = base_loss_fn
        def forward(self, student_logits, teacher_logits, labels):
            return distillation_loss(student_logits, teacher_logits, labels, self.temperature, self.alpha)
    class MixupCutmix(nn.Module):
        def __init__(self, mixup_alpha, cutmix_alpha, prob, switch_prob, num_classes): super().__init__()
        def forward(self, img, target): return img, (target, None, 1.0, 'none') # Return mixinfo tuple
    class KnowledgeReplayBuffer:
        def __init__(self, capacity): pass
        def add(self, experience): pass
        def sample(self, batch_size): return []
    class CurriculumSampler:
        def __init__(self, *args, **kwargs): pass
        def set_epoch(self, epoch): pass
        def get_epoch_data(self): return ([], [])
        def get_current_image_size(self): return (CONFIG['data']['image_size'][0], CONFIG['data']['image_size'][1])

    # Dummy data loading functions
    def load_bongard_data(path): return [], [], []
    def load_config(path): return CONFIG
    def train_test_split(*args, **kwargs): return [], [], [], [], [], []
    
    # Dummy attribute/relation maps
    ATTRIBUTE_FILL_MAP = {'fill': {0: 'solid', 1: 'hollow'}} # Dummy map
    RELATION_MAPS = {0: 'touching', 1: 'overlapping'} # Dummy map
    logger.warning("Dummy classes/functions are active. Ensure all core dependencies are loaded in your main notebook.")


# --- Ensemble Utilities ---

class MetaLearner(nn.Module):
    """
    A simple neural network (MLP) to act as the meta-learner in a stacking ensemble.
    It takes the concatenated logits/probabilities from base models as input
    and learns to combine them to make a final prediction.

    Args:
        input_dim (int): The total input dimension, which should be
                         (num_base_models * num_classes_per_base_model).
        num_classes (int): The number of final output classes for the Bongard Problem.
        hidden_dim (int): The dimension of the hidden layer in the MLP.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Added dropout for regularization
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        logger.info(f"MetaLearner initialized with input_dim={input_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MetaLearner.
        Args:
            x (torch.Tensor): Concatenated predictions (logits or probabilities)
                              from base models. Shape: [batch_size, input_dim].
        Returns:
            torch.Tensor: Final logits for the Bongard Problem classification.
                          Shape: [batch_size, num_classes].
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def calculate_model_weights(
    member_metrics: List[Dict[str, Any]],
    metric_name: str = 'val_accuracy',
    minimize_metric: bool = False
) -> List[float]:
    """
    Calculates weights for ensemble members based on their performance metrics.

    Args:
        member_metrics (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                contains metrics for an ensemble member
                                                (e.g., from _run_single_training_session).
        metric_name (str): The name of the metric to use for weighting (e.g., 'val_accuracy', 'val_loss').
        minimize_metric (bool): If True, a lower value of the metric indicates better performance
                                (e.g., for 'val_loss'). If False, a higher value is better.

    Returns:
        List[float]: A list of weights for each ensemble member, normalized to sum to 1.
    """
    if not member_metrics:
        logger.warning("No member metrics provided. Returning equal weights.")
        return []

    metric_values = []
    for metrics_dict in member_metrics:
        if metric_name not in metrics_dict:
            logger.warning(f"Metric '{metric_name}' not found in one of the member's metrics. Using default weight 1.0.")
            metric_values.append(1.0) # Default to 1.0 if metric is missing
        else:
            metric_values.append(metrics_dict[metric_name])

    metric_values_np = np.array(metric_values)

    if minimize_metric:
        # For metrics like loss, smaller is better, so use inverse
        # Add a small epsilon to avoid division by zero for very small losses
        weights = 1.0 / (metric_values_np + 1e-6)
        logger.info(f"Calculated weights based on inverse of '{metric_name}' (minimize).")
    else:
        # For metrics like accuracy, larger is better
        weights = metric_values_np
        logger.info(f"Calculated weights based on '{metric_name}' (maximize).")

    # Normalize weights to sum to 1
    total_weight = np.sum(weights)
    if total_weight == 0:
        logger.warning("Sum of weights is zero. Returning equal weights.")
        return [1.0 / len(member_metrics)] * len(member_metrics)
    
    normalized_weights = weights / total_weight
    logger.info(f"Normalized ensemble weights: {normalized_weights.tolist()}")
    return normalized_weights.tolist()

def compute_diversity_metrics(probs_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Computes diversity metrics (e.g., pairwise KL divergence) for ensemble members.
    Args:
        probs_list (List[np.ndarray]): A list of probability arrays from different models.
                                       Each array shape: [num_samples, num_classes].
    Returns:
        Dict[str, float]: A dictionary of diversity metrics.
    """
    if len(probs_list) < 2:
        logger.warning("Need at least two models to compute diversity metrics. Skipping.")
        return {"avg_kl_divergence": 0.0}

    import itertools
    divergences = []
    
    # Ensure probabilities are not zero for log
    epsilon = 1e-10
    
    for p_arr, q_arr in itertools.combinations(probs_list, 2):
        p_t = torch.tensor(p_arr + epsilon, device=DEVICE, dtype=torch.float32)
        q_t = torch.tensor(q_arr + epsilon, device=DEVICE, dtype=torch.float32)
        
        # Calculate KL divergence: D_KL(P || Q) = sum(P * log(P / Q))
        kl = (p_t * (p_t.log() - q_t.log())).sum(dim=1).mean().item()
        divergences.append(kl)
    
    avg_kl = sum(divergences) / len(divergences) if divergences else 0.0
    logger.info(f"Computed ensemble diversity: Average KL Divergence = {avg_kl:.4f}")
    return {"avg_kl_divergence": avg_kl}


# --- DALI Loader Refactoring ---

def build_dali_loader(file_list: List[str], labels_list: np.ndarray, config: Dict[str, Any], mode: str = 'train', external_source_iterator: Optional[Any] = None) -> Tuple[Any, DALIGenericIterator]:
    """
    Builds and returns a DALI pipeline and iterator.
    Args:
        file_list (List[str]): List of image file paths.
        labels_list (np.ndarray): NumPy array of labels.
        config (Dict): Configuration dictionary.
        mode (str): 'train', 'val', 'inference', or 'calibration'. Affects image size and is_train flag.
        external_source_iterator (Optional[Any]): Iterator for synthetic data.
    Returns:
        Tuple[Any, DALIGenericIterator]: The DALI pipeline and iterator.
    """
    is_train = (mode == 'train')
    
    if config['data']['use_synthetic_data'] and external_source_iterator:
        file_root = "" # Not used by external_source
        file_list_for_dali = [] # Not used by external_source
        labels_list_for_dali = [] # Not used by external_source
        size = len(external_source_iterator.dataset) # Use dataset size for synthetic
    else:
        file_root = DATA_ROOT_PATH
        file_list_for_dali = file_list
        labels_list_for_dali = labels_list
        size = len(file_list)

    height = config['data']['initial_image_size'][0] if is_train and mode == 'train' else config['data']['image_size'][0]
    width = config['data']['initial_image_size'][1] if is_train and mode == 'train' else config['data']['image_size'][1]

    # DALI Pipeline Flags
    device_memory_padding = config['data'].get('dali_device_memory_padding', 64 * 1024 * 1024)
    host_memory_padding = config['data'].get('dali_host_memory_padding', 64 * 1024 * 1024)
    prebuild_streams = config['data'].get('dali_prebuild_streams', 0) # 0 means disabled, 2 is common for prebuild

    pipeline = dali_pipe(
        file_root=file_root,
        file_list=file_list_for_dali,
        labels_list=labels_list_for_dali,
        batch_size=config['model']['batch_size'],
        num_threads=config['data']['dataloader_num_workers'],
        device_id=0 if str(DEVICE) == 'cuda' else -1,
        height=height,
        width=width,
        is_train=is_train,
        num_channels=NUM_CHANNELS,
        feature_consistency_alpha=config['training']['feature_consistency_alpha'] if is_train else 0.0,
        imagenet_mean=IMAGENET_MEAN,
        imagenet_std=IMAGENET_STD,
        use_synthetic_data=config['data']['use_synthetic_data'],
        external_source_iterator=external_source_iterator,
        device_memory_padding=device_memory_padding,
        host_memory_padding=host_memory_padding,
        prebuild_streams=prebuild_streams # Add prebuild_streams
    )
    pipeline.build() # Build the pipeline once
    
    loader = DALIGenericIterator(
        pipeline,
        ['view1', 'view2', 'labels', 'gts_json', 'affine1', 'affine2'],
        size=size,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL
    )
    logger.info(f"DALI loader built for mode '{mode}' with size {size}.")
    return pipeline, loader

# --- Augmentation Utilities ---

class AugmentMix(nn.Module):
    """
    Applies MixUp or CutMix augmentation to a batch of images and labels.
    """
    def __init__(self, mixup_alpha: float, cutmix_alpha: float, mixup_cutmix_ratio: float):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.ratio = mixup_cutmix_ratio # Probability of applying MixUp vs CutMix

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], float, str]]:
        """
        Applies MixUp or CutMix.
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W].
            labels (torch.Tensor): Batch of labels [B].
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], float, str]]:
                - Augmented images.
                - Tuple: (labels_a, labels_b, lambda, mode_str)
                  labels_a: Original labels
                  labels_b: Labels of mixed image (None if no mix)
                  lambda: Mixing coefficient
                  mode_str: 'mixup', 'cutmix', or 'none'
        """
        if self.mixup_alpha > 0 and random.random() < self.ratio:
            # Apply MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam) # Ensure lambda >= 0.5
            
            index = torch.randperm(images.size(0)).to(images.device)
            mixed_images = lam * images + (1 - lam) * images[index, :]
            labels_a, labels_b = labels, labels[index]
            
            return mixed_images, (labels_a, labels_b, lam, 'mixup')
        
        elif self.cutmix_alpha > 0 and random.random() >= self.ratio:
            # Apply CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = max(lam, 1 - lam)
            
            index = torch.randperm(images.size(0)).to(images.device)
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            labels_a, labels_b = labels, labels[index]
            
            return mixed_images, (labels_a, labels_b, lam, 'cutmix')
        
        else:
            # No MixUp/CutMix
            return images, (labels, None, 1.0, 'none')

# Helper function for CutMix (from torchvision examples)
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    return bbx1, bby1, bbx2, bby2

# Helper function for distillation loss
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Calculates the knowledge distillation loss.
    Combines KL Divergence for soft targets with CrossEntropy for hard labels.
    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model (ensemble).
        labels (torch.Tensor): True hard labels.
        temperature (float): Temperature for softening probabilities.
        alpha (float): Weight for the distillation loss component.
    Returns:
        torch.Tensor: Combined distillation loss.
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    # Soft predictions from student
    soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL Divergence for distillation
    dist_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Hard loss (cross-entropy with true labels)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined loss
    return alpha * dist_loss + (1.0 - alpha) * hard_loss

def symbolic_consistency_loss(attr_logits1: Dict[str, torch.Tensor], attr_logits2: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Calculates symbolic consistency loss between attribute predictions from two views.
    Uses Jaccard distance on one-hot argmax predictions.
    Args:
        attr_logits1 (Dict[str, torch.Tensor]): Dictionary of attribute logits from view 1.
        attr_logits2 (Dict[str, torch.Tensor]): Dictionary of attribute logits from view 2.
    Returns:
        torch.Tensor: Average symbolic consistency loss.
    """
    loss = 0.0
    num_attributes = 0
    for name in attr_logits1:
        if name in attr_logits2:
            num_attributes += 1
            # Get one-hot argmax predictions
            pred1 = F.one_hot(attr_logits1[name].argmax(dim=-1), num_classes=attr_logits1[name].shape[-1]).float()
            pred2 = F.one_hot(attr_logits2[name].argmax(dim=-1), num_classes=attr_logits2[name].shape[-1]).float()
            
            # Calculate Jaccard distance: 1 - (Intersection / Union)
            intersection = (pred1 * pred2).sum(dim=-1) # Sum over classes (0 or 1)
            union = (pred1 + pred2).sum(dim=-1) - intersection
            
            # Avoid division by zero if union is 0 (both are 0 vectors)
            jaccard_similarity = torch.where(union > 0, intersection / union, torch.tensor(1.0, device=pred1.device))
            
            loss += (1 - jaccard_similarity).mean() # Average over batch
            
    return loss / num_attributes if num_attributes > 0 else torch.tensor(0.0, device=DEVICE)


# --- Core Training and Validation ---

def _run_single_training_session_ensemble(
    current_config: Dict[str, Any],
    member_id: int,
    random_seed: int,
    output_dir: str = './checkpoints',
    epochs_override: Optional[int] = None,
    train_image_paths: List[str] = None,
    train_labels: np.ndarray = None,
    train_difficulty_scores: Optional[np.ndarray] = None, # Added for difficulty-weighted loss
    val_image_paths: List[str] = None,
    val_labels: np.ndarray = None,
    val_difficulty_scores: Optional[np.ndarray] = None,
    teacher_model: Optional[Any] = None # For knowledge distillation
) -> Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Runs a single training session for a PerceptionModule model.
    This function is designed to be called by the ensemble orchestrator.
    Args:
        current_config (Dict): The configuration for this specific training session.
        member_id (int): Unique ID for the ensemble member (for naming checkpoints/logs).
        random_seed (int): Random seed for reproducibility of this member.
        output_dir (str): Directory to save model checkpoints.
        epochs_override (Optional[int]): If provided, overrides 'model.epochs' in config.
        train_image_paths (List[str]): Paths to training images.
        train_labels (np.ndarray): Labels for training images.
        train_difficulty_scores (Optional[np.ndarray]): Difficulty scores for training images.
        val_image_paths (List[str]): Paths to validation images.
        val_labels (np.ndarray): Labels for validation images.
        val_difficulty_scores (np.ndarray): Difficulty scores for validation images.
        teacher_model (Optional[Any]): A pre-trained teacher model (e.g., an ensemble) for knowledge distillation.
    Returns:
        Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
            - Path to the best saved model checkpoint.
            - All validation predictions (logits) from the best model.
            - All validation true labels.
            - Dictionary of best validation metrics.
    """
    set_seed(random_seed)
    logger.info(f"--- Starting Training Session for Member {member_id} (Seed: {random_seed}) ---")
    
    # Override epochs if specified
    epochs = epochs_override if epochs_override is not None else current_config['model']['epochs']
    logger.info(f"Training for {epochs} epochs.")

    # Initialize TensorBoard writer
    log_dir = os.path.join('runs', datetime.now().strftime(f'member_{member_id}_%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs for member {member_id} at: {log_dir}")

    # Initialize Weights & Biases (WandB)
    if current_config['training']['use_wandb'] and HAS_WANDB:
        import wandb # Import here to avoid circular dependency if not globally available
        wandb.init(project="Bongard_Perception_Ensemble",
                   group="ensemble_training",
                   name=f"member_{member_id}_seed_{random_seed}",
                   config=current_config,
                   reinit=True) # reinit=True for multiple runs in the same script
        logger.info(f"WandB initialized for member {member_id}.")

    # Data Loaders (using DALI)
    batch_size = current_config['model']['batch_size']
    num_workers = current_config['data']['dataloader_num_workers']
    
    # Initialize synthetic dataset and external source if enabled
    train_synthetic_dataset = None
    val_synthetic_dataset = None
    train_external_source_iterator = None
    val_external_source_iterator = None

    if current_config['data']['use_synthetic_data']:
        logger.info("Using synthetic data for training and validation.")
        train_synthetic_dataset = BongardSyntheticDataset(current_config['data']['synthetic_samples'],
                                                          image_size=current_config['data']['image_size'],
                                                          num_classes=current_config['model']['num_classes'])
        val_synthetic_dataset = BongardSyntheticDataset(current_config['data']['synthetic_samples'] // 5, # Smaller val set
                                                        image_size=current_config['data']['image_size'],
                                                        num_classes=current_config['model']['num_classes'])
        
        train_external_source_iterator = BongardExternalSource(train_synthetic_dataset, batch_size, shuffle=True)
        val_external_source_iterator = BongardExternalSource(val_synthetic_dataset, batch_size, shuffle=False)
        
        # For synthetic data, file_root, file_list, labels_list are not directly used by fn.readers.file
        # They are passed as dummy values or None, as DALI will use external_source.
        train_file_root = ""
        train_file_list = []
        train_labels_list = []
        val_file_root = ""
        val_file_list = []
        val_labels_list = []
    else:
        # For real data, use the provided paths and labels
        train_file_root = DATA_ROOT_PATH
        train_file_list = train_image_paths
        train_labels_list = train_labels
        val_file_root = DATA_ROOT_PATH
        val_file_list = val_image_paths
        val_labels_list = val_labels

    # Refactored DALI Loader Creation
    train_pipeline, train_loader = build_dali_loader(
        file_list=train_file_list,
        labels_list=train_labels_list,
        config=current_config,
        mode='train',
        external_source_iterator=train_external_source_iterator
    )
    logger.info("DALI training loader initialized.")

    val_pipeline, val_loader = build_dali_loader(
        file_list=val_file_list,
        labels_list=val_labels_list,
        config=current_config,
        mode='val',
        external_source_iterator=val_external_source_iterator
    )
    logger.info("DALI validation loader initialized.")

    # Model, Optimizer, Scheduler
    object_detector = RealObjectDetector(model_path=current_config['model']['object_detector_model_path'])
    model = PerceptionModule(current_config, object_detector).to(DEVICE)
    logger.info("PerceptionModule model initialized.")

    # Initialize SWA model if enabled
    if current_config['training']['use_swa']:
        swa_model = swa_utils.AveragedModel(model)
        logger.info("SWA AveragedModel initialized.")

    # Quantization Aware Training (QAT) setup
    if current_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION:
        logger.info("Setting up Quantization Aware Training (QAT).")
        model.eval() # Model must be in eval mode for fusion
        if hasattr(model, 'fuse_model'): # Assuming fuse_model is a method of PerceptionModule
            model.fuse_model()
            logger.info("Model modules fused for QAT.")
        else:
            logger.warning("Model does not have a fuse_model method. Fusion skipped for QAT.")
        
        # Prepare model for QAT
        model.qconfig = tq.get_default_qat_qconfig('fbgemm') # Use fbgemm for server-side inference
        tq.prepare_qat(model, inplace=True)
        logger.info("Model prepared for QAT.")
        model.train() # Set back to train mode

    # Loss function (for Bongard classification)
    criterion_bongard = nn.CrossEntropyLoss(label_smoothing=current_config['training']['label_smoothing_epsilon'])
    logger.info(f"Using CrossEntropyLoss with label_smoothing_epsilon={current_config['training']['label_smoothing_epsilon']}.")

    # Loss functions for attribute and relation classification (if used in joint symbolic loss)
    criterion_attr = nn.CrossEntropyLoss()
    criterion_rel = nn.CrossEntropyLoss()

    # Optimizer
    optimizer_name = current_config['model']['optimizer']
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=current_config['model']['initial_learning_rate'],
                                weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=current_config['model']['initial_learning_rate'],
                              momentum=0.9, weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'SophiaG' and HAS_SOPHIA:
        # Ensure SophiaG is imported if HAS_SOPHIA is True
        from sophia import SophiaG
        optimizer = SophiaG(model.parameters(), lr=current_config['model']['initial_learning_rate'],
                            weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'Lion' and HAS_TIMM_OPTIM:
        # Ensure Lion is imported if HAS_TIMM_OPTIM is True
        from torch_optimizer import Lion
        optimizer = Lion(model.parameters(), lr=current_config['model']['initial_learning_rate'],
                         weight_decay=current_config['training']['weight_decay'])
    elif optimizer_name == 'MADGRAD' and HAS_TIMM_OPTIM:
        # Ensure MADGRAD is imported if HAS_TIMM_OPTIM is True
        from torch_optimizer import MADGRAD
        optimizer = MADGRAD(model.parameters(), lr=current_config['model']['initial_learning_rate'],
                            weight_decay=current_config['training']['weight_decay'])
    else:
        logger.warning(f"Optimizer '{optimizer_name}' not supported or not available. Falling back to AdamW.")
        optimizer = optim.AdamW(model.parameters(), lr=current_config['model']['initial_learning_rate'],
                                weight_decay=current_config['training']['weight_decay'])

    if current_config['training']['use_sam_optimizer'] and HAS_SAM:
        # Ensure SAM is imported if HAS_SAM is True
        from sam import SAM
        optimizer = SAM(model.parameters(), optimizer, rho=0.05, adaptive=True)
        logger.info("Wrapped optimizer with SAM (Sharpness-Aware Minimization).")
    logger.info(f"Optimizer initialized: {optimizer_name}.")

    # Learning Rate Scheduler
    scheduler = None
    if current_config['model']['scheduler'] == 'OneCycleLR':
        total_samples = len(train_image_paths) if not current_config['data']['use_synthetic_data'] else current_config['data']['synthetic_samples']
        total_steps = math.ceil(total_samples / batch_size) * epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=current_config['model']['max_learning_rate'],
            total_steps=total_steps,
            pct_start=current_config['training']['onecycle_pct_start'],
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        logger.info(f"Using OneCycleLR scheduler with max_lr={current_config['model']['max_learning_rate']} and total_steps={total_steps}.")
    elif current_config['model']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=current_config['training']['lr_scheduler_factor'],
            patience=current_config['training']['lr_scheduler_patience'],
            verbose=True
        )
        logger.info(f"Using ReduceLROnPlateau scheduler.")

    scaler = GradScaler(enabled=current_config['training']['use_amp'])
    logger.info(f"AMP scaler initialized (enabled={current_config['training']['use_amp']}).")

    # Early Stopping
    best_val_loss = float('inf')
    best_val_accuracy = -float('inf')
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f'member_{member_id}_{current_config["training"]["model_checkpoint_name"]}')

    # Store validation predictions for stacking
    all_val_predictions_logits = []
    all_val_labels = []

    # Initialize AugmentMix
    augmenter = AugmentMix(
        mixup_alpha=current_config['training']['mixup_alpha'],
        cutmix_alpha=current_config['training']['cutmix_alpha'],
        mixup_cutmix_ratio=current_config['training']['mixup_cutmix_ratio']
    )
    logger.info(f"AugmentMix initialized with mixup_alpha={augmenter.mixup_alpha}, cutmix_alpha={augmenter.cutmix_alpha}.")

    # Training Loop
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples_epoch = 0
        
        # Update image size for progressive resizing (only for real data with CurriculumSampler)
        if not current_config['data']['use_synthetic_data'] and isinstance(train_loader.iterator, CurriculumSampler):
            current_image_size = train_loader.iterator.get_current_image_size()
            if current_image_size != train_pipeline.args.height: # Check if size changed
                logger.info(f"Epoch {epoch}: Adjusting DALI pipeline image size to {current_image_size}.")
                train_pipeline.args.height = current_image_size[0]
                train_pipeline.args.width = current_image_size[1]
                val_pipeline.args.height = current_config['data']['image_size'][0] # Validation always uses final size
                val_pipeline.args.width = current_config['data']['image_size'][1]
                train_pipeline.build()
                val_pipeline.build()
                train_loader.reset()
                val_loader.reset()

            # Set epoch for curriculum sampler
            train_loader.iterator.set_epoch(epoch)
            epoch_train_paths, epoch_train_labels = train_loader.iterator.get_epoch_data()
            train_pipeline.args.file_list = epoch_train_paths
            train_pipeline.args.labels_list = epoch_train_labels
            train_pipeline.build() # Rebuild pipeline with new file list
            train_loader.reset() # Reset iterator
        else: # For synthetic data or if not using CurriculumSampler
            train_loader.reset() # Always reset loader for new epoch

        # QAT specific: enable observer for last few epochs
        if current_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION and epoch >= current_config['training']['qat_start_epoch']:
            logger.info(f"Epoch {epoch}: Enabling QAT observers.")
            model.apply(tq.enable_observer)
            model.apply(tq.enable_fake_quant)
        
        # Profiler setup
        profiler = None
        if current_config['training']['enable_profiler'] and epoch == 0:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=current_config['training']['profiler_schedule_wait'],
                    warmup=current_config['training']['profiler_schedule_warmup'],
                    active=current_config['training']['profiler_schedule_active'],
                    repeat=current_config['training']['profiler_schedule_repeat']
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                with_stack=True
            )
            profiler.start()
            logger.info("PyTorch profiler started.")

        for batch_idx, data in enumerate(train_loader):
            images_view1 = data['view1'].permute(0, 3, 1, 2) # NCHW
            images_view2 = data['view2'].permute(0, 3, 1, 2) # NCHW
            labels_bongard = data['labels'].squeeze(-1) # Remove last dim from DALI
            gts_json_strings_batch = [s.decode('utf-8') for s in data['gts_json'].cpu().numpy()] # Decode JSON strings

            # Ground-Truth JSON Validation
            for i, gt_json_str in enumerate(gts_json_strings_batch):
                try:
                    gt_dict = json.loads(gt_json_str)
                    jsonschema.validate(instance=gt_dict, schema=BONGARD_GT_JSON_SCHEMA)
                    # For synthetic data, extract actual labels from GT JSON
                    if current_config['data']['use_synthetic_data']:
                        labels_bongard[i] = torch.tensor(gt_dict['bongard_label'], dtype=torch.long, device=DEVICE)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode GT JSON: {gt_json_str}. Using dummy label 0 for this sample.")
                    labels_bongard[i] = torch.tensor(0, dtype=torch.long, device=DEVICE) # Assign dummy label
                except jsonschema.ValidationError as ve:
                    logger.error(f"GT JSON validation error for sample {i}: {ve.message}. Using dummy label 0.")
                    labels_bongard[i] = torch.tensor(0, dtype=torch.long, device=DEVICE) # Assign dummy label

            # Apply MixUp/CutMix
            images_view1_aug, mixinfo = augmenter(images_view1, labels_bongard)
            labels_a, labels_b, lam, mode_aug = mixinfo

            # Difficulty-Weighted Loss
            current_batch_difficulty_weights = None
            if train_difficulty_scores is not None and len(train_difficulty_scores) > 0:
                # Assuming train_loader provides indices or labels map to original dataset
                # For simplicity here, we'll assume a direct mapping or use dummy weights if not available
                # In a real DALI setup with CurriculumSampler, you'd get actual indices.
                # For now, if train_difficulty_scores is provided, use it directly for the batch
                # This is a simplification; proper indexing would be needed.
                current_batch_difficulty_weights = torch.tensor(train_difficulty_scores[batch_idx * batch_size : (batch_idx + 1) * batch_size], dtype=torch.float32, device=DEVICE)
                if current_batch_difficulty_weights.shape[0] != labels_bongard.shape[0]:
                    logger.warning("Difficulty weights batch size mismatch. Using uniform weights.")
                    current_batch_difficulty_weights = torch.ones_like(labels_bongard, dtype=torch.float32)
            else:
                current_batch_difficulty_weights = torch.ones_like(labels_bongard, dtype=torch.float32) # Uniform weights if no difficulty scores

            # Forward pass
            with autocast(enabled=current_config['training']['use_amp']):
                bongard_logits, detected_objects_view1, aggregated_outputs_view1 = model(images_view1_aug, gts_json_strings_batch)
                
                # Bongard Classification Loss (potentially with MixUp/CutMix)
                if mode_aug == 'mixup' or mode_aug == 'cutmix':
                    loss_bongard = lam * criterion_bongard(bongard_logits, labels_a) + (1 - lam) * criterion_bongard(bongard_logits, labels_b)
                else:
                    loss_bongard = criterion_bongard(bongard_logits, labels_bongard)
                
                # Apply difficulty weights to Bongard loss
                # Note: If reduction='none' is used in criterion_bongard, then element-wise multiplication is direct.
                # If criterion_bongard returns a scalar (mean/sum), then apply weights before averaging.
                # Assuming criterion_bongard returns a scalar loss per batch.
                loss_bongard = (loss_bongard * current_batch_difficulty_weights).mean() # Apply mean after weighting
                
                total_loss_batch = loss_bongard

                # Attribute and Relation Losses (Joint Symbolic Loss)
                if 'attribute_logits' in aggregated_outputs_view1 and 'attribute_gt' in aggregated_outputs_view1:
                    for attr_name, attr_logits in aggregated_outputs_view1['attribute_logits'].items():
                        if attr_name in aggregated_outputs_view1['attribute_gt']:
                            attr_gt = aggregated_outputs_view1['attribute_gt'][attr_name]
                            total_loss_batch += current_config['training']['attribute_loss_weight'] * criterion_attr(attr_logits, attr_gt)
                
                if 'relation_logits' in aggregated_outputs_view1 and 'relation_gt' in aggregated_outputs_view1:
                    rel_logits = aggregated_outputs_view1['relation_logits']
                    rel_gt = aggregated_outputs_view1['relation_gt']
                    total_loss_batch += current_config['training']['relation_loss_weight'] * criterion_rel(rel_logits, rel_gt)

                # Feature Consistency Loss
                if current_config['training']['feature_consistency_alpha'] > 0:
                    _, _, aggregated_outputs_view2 = model(images_view2, gts_json_strings_batch)
                    
                    # Compute feature consistency loss on image features
                    consistency_loss_features = 0.0
                    if 'image_features_student' in aggregated_outputs_view1 and 'image_features_student' in aggregated_outputs_view2:
                        feature_consistency_criterion = FeatureConsistencyLoss(loss_type=current_config['training']['feature_consistency_loss_type'])
                        consistency_loss_features = feature_consistency_criterion(
                            aggregated_outputs_view1['image_features_student'],
                            aggregated_outputs_view2['image_features_student']
                        )
                    total_loss_batch += current_config['training']['feature_consistency_alpha'] * consistency_loss_features
                    logger.debug(f"Added feature consistency loss: {consistency_loss_features.item():.4f}")

                # Symbolic Consistency Loss
                if current_config['training']['symbolic_consistency_alpha'] > 0:
                    symbolic_cons_loss = symbolic_consistency_loss(
                        aggregated_outputs_view1['attribute_logits'],
                        aggregated_outputs_view2['attribute_logits']
                    )
                    total_loss_batch += current_config['training']['symbolic_consistency_alpha'] * symbolic_cons_loss
                    logger.debug(f"Added symbolic consistency loss: {symbolic_cons_loss.item():.4f}")

            # Knowledge Distillation (if teacher model provided)
            if teacher_model is not None and current_config['training']['use_knowledge_distillation']:
                with autocast(enabled=current_config['training']['use_amp']):
                    # Get soft targets from teacher
                    teacher_logits_list = []
                    # If teacher_model is a list of models (ensemble teacher)
                    if isinstance(teacher_model, list):
                        for t_model in teacher_model:
                            t_model.eval() # Ensure teacher is in eval mode
                            t_logits, _, _ = t_model(images_view1, gts_json_strings_batch)
                            teacher_logits_list.append(t_logits)
                        # Average teacher logits
                        teacher_logits = torch.mean(torch.stack(teacher_logits_list, dim=0), dim=0)
                    else: # Single teacher model
                        teacher_model.eval()
                        teacher_logits, _, _ = teacher_model(images_view1, gts_json_strings_batch)
                    
                    # Calculate distillation loss
                    dist_loss = distillation_loss(
                        student_logits=bongard_logits,
                        teacher_logits=teacher_logits,
                        labels=labels_bongard, # Hard labels for the hard loss component
                        temperature=current_config['training']['distillation_temperature'],
                        alpha=current_config['training']['distillation_alpha']
                    )
                    total_loss_batch = dist_loss # Distillation loss replaces main loss
                    logger.debug(f"Applied knowledge distillation loss: {dist_loss.item():.4f}")
            
            # Scale loss for gradient accumulation
            total_loss_batch = total_loss_batch / current_config['training']['gradient_accumulation_steps']

            scaler.scale(total_loss_batch).backward()

            if (batch_idx + 1) % current_config['training']['gradient_accumulation_steps'] == 0:
                if current_config['training']['use_sam_optimizer']:
                    scaler.unscale_(optimizer) # Unscale gradients for SAM's first step
                    # Clip gradients before SAM's first step if needed
                    if current_config['training'].get('max_grad_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), current_config['training']['max_grad_norm'])
                    optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass for SAM
                    with autocast(enabled=current_config['training']['use_amp']):
                        # Re-calculate loss for the second step of SAM
                        bongard_logits_second_step, detected_objects_second_step, aggregated_outputs_second_step = model(images_view1_aug, gts_json_strings_batch)
                        
                        loss_bongard_second_step = criterion_bongard(bongard_logits_second_step, labels_bongard)
                        if current_batch_difficulty_weights is not None:
                            loss_bongard_second_step = (loss_bongard_second_step * current_batch_difficulty_weights).mean()
                        
                        total_loss_second_step = loss_bongard_second_step

                        # Re-apply attribute and relation losses
                        if 'attribute_logits' in aggregated_outputs_second_step and 'attribute_gt' in aggregated_outputs_second_step:
                            for attr_name, attr_logits in aggregated_outputs_second_step['attribute_logits'].items():
                                if attr_name in aggregated_outputs_second_step['attribute_gt']:
                                    attr_gt = aggregated_outputs_second_step['attribute_gt'][attr_name]
                                    total_loss_second_step += current_config['training']['attribute_loss_weight'] * criterion_attr(attr_logits, attr_gt)
                        
                        if 'relation_logits' in aggregated_outputs_second_step and 'relation_gt' in aggregated_outputs_second_step:
                            rel_logits = aggregated_outputs_second_step['relation_logits']
                            rel_gt = aggregated_outputs_second_step['relation_gt']
                            total_loss_second_step += current_config['training']['relation_loss_weight'] * criterion_rel(rel_logits, rel_gt)

                        # Re-apply feature consistency loss if enabled
                        if current_config['training']['feature_consistency_alpha'] > 0:
                            _, _, aggregated_outputs_view2_second_step = model(images_view2, gts_json_strings_batch)
                            consistency_loss_features_second_step = 0.0
                            if 'image_features_student' in aggregated_outputs_second_step and 'image_features_student' in aggregated_outputs_view2_second_step:
                                feature_consistency_criterion_second_step = FeatureConsistencyLoss(loss_type=current_config['training']['feature_consistency_loss_type'])
                                consistency_loss_features_second_step = feature_consistency_criterion_second_step(
                                    aggregated_outputs_second_step['image_features_student'],
                                    aggregated_outputs_view2_second_step['image_features_student']
                                )
                            total_loss_second_step += current_config['training']['feature_consistency_alpha'] * consistency_loss_features_second_step

                        # Re-apply symbolic consistency loss
                        if current_config['training']['symbolic_consistency_alpha'] > 0:
                            symbolic_cons_loss_second_step = symbolic_consistency_loss(
                                aggregated_outputs_second_step['attribute_logits'],
                                aggregated_outputs_view2_second_step['attribute_logits']
                            )
                            total_loss_second_step += current_config['training']['symbolic_consistency_alpha'] * symbolic_cons_loss_second_step

                        # Re-apply distillation loss if enabled
                        if teacher_model is not None and current_config['training']['use_knowledge_distillation']:
                            teacher_logits_list_second_step = []
                            if isinstance(teacher_model, list):
                                for t_model in teacher_model:
                                    t_model.eval()
                                    t_logits, _, _ = t_model(images_view1, gts_json_strings_batch)
                                    teacher_logits_list_second_step.append(t_logits)
                                teacher_logits_second_step = torch.mean(torch.stack(teacher_logits_list_second_step, dim=0), dim=0)
                            else:
                                teacher_model.eval()
                                teacher_logits_second_step, _, _ = teacher_model(images_view1, gts_json_strings_batch)

                            dist_loss_second_step = distillation_loss(
                                student_logits=bongard_logits_second_step,
                                teacher_logits=teacher_logits_second_step,
                                labels=labels_bongard,
                                temperature=current_config['training']['distillation_temperature'],
                                alpha=current_config['training']['distillation_alpha']
                            )
                            total_loss_second_step = dist_loss_second_step

                    total_loss_second_step = total_loss_second_step / current_config['training']['gradient_accumulation_steps']
                    scaler.scale(total_loss_second_step).backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    # Apply gradient clipping if not using SAM
                    if current_config['training'].get('max_grad_norm', 0) > 0:
                        scaler.unscale_(optimizer) # Unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), current_config['training']['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad() # Clear gradients after step
                
                if scheduler and current_config['model']['scheduler'] == 'OneCycleLR':
                    scheduler.step() # Step OneCycleLR after optimizer.step()

            total_loss += total_loss_batch.item() * current_config['training']['gradient_accumulation_steps'] # Re-scale for logging
            
            # Predictions for accuracy
            _, predicted = torch.max(bongard_logits.data, 1)
            correct_predictions += (predicted == labels_bongard).sum().item()
            total_samples_epoch += labels_bongard.size(0)

            if profiler:
                profiler.step()

            if current_config['training']['validation_frequency_batches'] > 0 and \
               (batch_idx + 1) % current_config['training']['validation_frequency_batches'] == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: Performing intermediate validation.")
                val_loss_intermediate, val_accuracy_intermediate, _, _ = _validate_model_ensemble(model, val_loader, criterion_bongard, current_config)
                logger.info(f"Intermediate Val Loss: {val_loss_intermediate:.4f}, Val Accuracy: {val_accuracy_intermediate:.4f}")
                model.train() # Set back to train mode after validation

        if profiler:
            profiler.stop()
            logger.info("PyTorch profiler stopped.")

        avg_train_loss = total_loss / len(train_loader) # DALI loader len is num_samples / batch_size
        train_accuracy = correct_predictions / total_samples_epoch if total_samples_epoch > 0 else 0.0

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        if current_config['training']['use_wandb'] and HAS_WANDB:
            import wandb
            wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_accuracy, "epoch": epoch})

        # Validation
        val_loss, val_accuracy, val_predictions_logits, val_true_labels = _validate_model_ensemble(model, val_loader, criterion_bongard, current_config)
        logger.info(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        if current_config['training']['use_wandb'] and HAS_WANDB:
            import wandb
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch})
        
        # Collect validation predictions and labels for stacking
        all_val_predictions_logits.extend(val_predictions_logits)
        all_val_labels.extend(val_true_labels)

        # SWA update
        if current_config['training']['use_swa'] and epoch >= int(epochs * current_config['training']['swa_start_epoch_ratio']):
            swa_model.update_parameters(model)
            logger.debug(f"SWA model updated at epoch {epoch}.")
            # Update SWA LR if using SWALR
            if current_config['training']['swa_lr'] is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_config['training']['swa_lr']
                logger.debug(f"SWA LR set to {current_config['training']['swa_lr']}.")

        # Early Stopping check
        monitor_metric = val_loss
        if current_config['training']['early_stopping_monitor_metric'] == 'val_accuracy':
            monitor_metric = -val_accuracy # For accuracy, we want to maximize, so minimize negative accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving model.")
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                logger.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{current_config['training']['early_stopping_patience']}")
        else: # Default to val_loss
            if val_loss < best_val_loss - current_config['training']['early_stopping_min_delta']:
                best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model.")
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{current_config['training']['early_stopping_patience']}")

        if scheduler and current_config['model']['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss) # Step ReduceLROnPlateau with validation loss

        if patience_counter >= current_config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

        # Save symbolic outputs periodically
        if (epoch + 1) % current_config['training']['save_symbolic_outputs_interval_epochs'] == 0:
            logger.info(f"Saving example structured symbolic outputs at epoch {epoch+1}.")
            # Take a small sample from validation set for visualization
            sample_val_paths = val_image_paths[:min(5, len(val_image_paths))] # Save up to 5 samples
            
            if sample_val_paths:
                sample_pil_images = [Image.open(p).convert('RGB') for p in sample_val_paths]
                
                model.eval()
                with torch.no_grad():
                    # Pass dummy GT JSON strings for inference
                    dummy_gts_json = [json.dumps({"bongard_label": 0, "objects": [], "relations": []})] * len(sample_pil_images)
                    # The PerceptionModule.forward now returns bongard_logits, detected_objects, and aggregated_outputs
                    bongard_logits_sample, detected_objects_sample_batch, aggregated_outputs_sample = model(
                        torch.stack([T.ToTensor()(img).to(DEVICE) for img in sample_pil_images]),
                        dummy_gts_json
                    )
                model.train()
                
                json_serializable_outputs = []
                for i in range(len(sample_pil_images)):
                    # Call the extract_scene_graph method
                    # detected_objects_sample_batch[i] is for a single image
                    # attribute_logits and relation_logits need to be indexed for a single image and unsqueezed
                    single_image_attr_logits = {k: v[i].unsqueeze(0) for k, v in aggregated_outputs_sample['attribute_logits'].items()}
                    single_image_rel_logits = {k: v[i].unsqueeze(0) for k, v in aggregated_outputs_sample['relation_logits'].items()}

                    scene_graph = model.extract_scene_graph(
                        detected_objects_sample_batch[i], # Pass single image's detected objects
                        single_image_attr_logits,
                        single_image_rel_logits
                    )
                    
                    scene_graph["image_path"] = sample_val_paths[i]
                    scene_graph["predicted_bongard_label"] = int(torch.argmax(bongard_logits_sample[i]).item())
                    scene_graph["bongard_prediction_probs"] = F.softmax(bongard_logits_sample[i], dim=-1).cpu().numpy().tolist()
                    # Symbolic uncertainty is now part of extract_scene_graph's output (even if dummy)
                    
                    json_serializable_outputs.append(scene_graph)

                output_json_path = os.path.join(current_config['training']['checkpoint_dir'], f"symbolic_outputs_member_{member_id}_epoch_{epoch+1}.json")
                with open(output_json_path, 'w') as f:
                    json.dump(json_serializable_outputs, f, indent=4)
                logger.info(f"Sample structured symbolic outputs exported to: {output_json_path}")
            else:
                logger.warning("Not enough samples to save symbolic outputs.")

    # Finalize SWA model if enabled
    if current_config['training']['use_swa']:
        swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
        torch.save(swa_model.state_dict(), best_model_path.replace('.pt', '_swa.pt'))
        logger.info(f"SWA model finalized and saved to {best_model_path.replace('.pt', '_swa.pt')}")
        # If SWA is used, the best_model_path should point to the SWA model for inference
        best_model_path = best_model_path.replace('.pt', '_swa.pt')

    # QAT specific: convert to quantized model
    if current_config['training']['use_qat'] and HAS_TORCH_QUANTIZATION:
        logger.info("Finalizing QAT model conversion.")
        model.eval()
        # Move model to CPU before conversion if it's on GPU, as convert often works best on CPU
        model_to_convert = model.cpu()
        model_int8 = tq.convert(model_to_convert, inplace=False)
        quantized_model_path = best_model_path.replace('.pt', '_quantized.pt')
        torch.save(model_int8.state_dict(), quantized_model_path)
        logger.info(f"Quantized model saved to {quantized_model_path}")
        best_model_path = quantized_model_path # Use quantized model for inference

    writer.close()
    if current_config['training']['use_wandb'] and HAS_WANDB:
        import wandb
        wandb.finish()
    
    # Convert collected logits and labels to numpy arrays
    final_val_predictions_logits = np.array(all_val_predictions_logits)
    final_val_labels = np.array(all_val_labels)

    best_metrics = {
        'val_loss': best_val_loss,
        'val_accuracy': best_val_accuracy
    }

    logger.info(f"--- Training Session for Member {member_id} Finished. Best model saved to {best_model_path} ---")
    return best_model_path, final_val_predictions_logits, final_val_labels, best_metrics

def _validate_model_ensemble(model: PerceptionModule, data_loader: DALIGenericIterator, criterion: nn.Module, config: Dict[str, Any]):
    """
    Validates the model on the given data loader.
    Args:
        model (PerceptionModule): The model to validate.
        data_loader (DALIGenericIterator): DALI data loader for validation data.
        criterion (nn.Module): Loss function.
        config (Dict): Configuration dictionary.
    Returns:
        Tuple[float, float, List[np.ndarray], List[int]]:
            - Average validation loss.
            - Average validation accuracy.
            - List of numpy arrays of predictions (logits) for each validation sample.
            - List of true labels for each validation sample.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions_logits = []
    all_true_labels = []

    # Reset DALI iterator for validation
    data_loader.reset()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images_view1 = data['view1'].permute(0, 3, 1, 2) # NCHW
            labels_bongard = data['labels'].squeeze(-1) # Remove last dim from DALI
            gts_json_strings_batch = [s.decode('utf-8') for s in data['gts_json'].cpu().numpy()] # Decode JSON strings

            # Ground-Truth JSON Validation
            for i, gt_json_str in enumerate(gts_json_strings_batch):
                try:
                    gt_dict = json.loads(gt_json_str)
                    jsonschema.validate(instance=gt_dict, schema=BONGARD_GT_JSON_SCHEMA)
                    # For synthetic data, extract actual labels from GT JSON
                    if config['data']['use_synthetic_data']:
                        labels_bongard[i] = torch.tensor(gt_dict['bongard_label'], dtype=torch.long, device=DEVICE)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode GT JSON: {gt_json_str}. Using dummy label 0 for this sample.")
                    labels_bongard[i] = torch.tensor(0, dtype=torch.long, device=DEVICE) # Assign dummy label
                except jsonschema.ValidationError as ve:
                    logger.error(f"GT JSON validation error for sample {i}: {ve.message}. Using dummy label 0.")
                    labels_bongard[i] = torch.tensor(0, dtype=torch.long, device=DEVICE) # Assign dummy label


            with autocast(enabled=config['training']['use_amp']):
                bongard_logits, _, _ = model(images_view1, gts_json_strings_batch) # Pass GT for synthetic
                loss = criterion(bongard_logits, labels_bongard)

            total_loss += loss.item()
            
            _, predicted = torch.max(bongard_logits.data, 1)
            correct_predictions += (predicted == labels_bongard).sum().item()
            total_samples += labels_bongard.size(0)

            all_predictions_logits.extend(bongard_logits.cpu().numpy())
            all_true_labels.extend(labels_bongard.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, accuracy, all_predictions_logits, all_true_labels

def ensemble_predict_orchestrator_base(
    models: List[PerceptionModule],
    image_paths: List[str],
    config: Dict[str, Any],
    use_mc_dropout: bool = False,
    mc_dropout_samples: int = 0,
    model_weights: Optional[List[float]] = None # Optional list of weights for weighted averaging
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Performs ensemble prediction by averaging probabilities from multiple PerceptionModule models.
    This is the base prediction function, now internal to the combined module.
    Args:
        models (List[PerceptionModule]): List of trained PerceptionModule instances.
        image_paths (List[str]): List of image paths for inference.
        config (Dict): Configuration dictionary.
        use_mc_dropout (bool): If True, enable MC Dropout during inference.
        mc_dropout_samples (int): Number of forward passes for MC Dropout.
        model_weights (Optional[List[float]]): Weights for each model for weighted averaging.
                                                If None, simple averaging is used.
    Returns:
        Tuple[np.ndarray, List[Dict]]:
            - Averaged Bongard problem probabilities (after softmax) [num_images, num_classes].
            - List of symbolic outputs (from the first model's inference, for example).
    """
    if not models:
        logger.error("No models provided for ensemble prediction.")
        return np.empty((len(image_paths), config['model']['num_classes'])), []

    all_bongard_probs = []
    first_model_symbolic_outputs = [] # Collect symbolic outputs from the first model for example

    # Create DALI loader for inference using refactored function
    inference_pipeline, inference_loader = build_dali_loader(
        file_list=image_paths,
        labels_list=[0] * len(image_paths), # Dummy labels for inference
        config=config,
        mode='inference'
    )
    logger.info("DALI inference loader initialized.")

    for model_idx, model in enumerate(models):
        model.eval()
        if use_mc_dropout:
            # Enable dropout layers for MC Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train() # Set dropout to training mode to enable dropout during inference
            logger.info(f"MC Dropout enabled for model {model_idx}.")
        else:
            # Ensure dropout layers are in eval mode if not using MC Dropout
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.eval()
        
        member_probs = []
        member_symbolic_outputs_mc_samples = [] # Store symbolic outputs for each MC sample

        with torch.no_grad():
            # Efficient MC Dropout Sampling
            for mc_sample_idx in tqdm(range(mc_dropout_samples if use_mc_dropout else 1), desc=f"Model {model_idx} MC Inference"):
                inference_loader.reset() # Reset for each MC Dropout sample or single pass
                
                current_sample_probs = []
                current_batch_symbolic_outputs = []

                for batch_idx, data in enumerate(inference_loader):
                    images_view1 = data['view1'].permute(0, 3, 1, 2) # NCHW
                    
                    # Pass None for ground_truth_json_strings as it's inference on real images
                    # PerceptionModule.forward now returns bongard_logits, detected_objects, aggregated_outputs
                    bongard_logits, detected_objects_batch, aggregated_outputs = model(images_view1, None)
                    
                    probs = F.softmax(bongard_logits, dim=-1).cpu().numpy()
                    current_sample_probs.append(probs)

                    # Collect structured symbolic outputs from the first model, first MC dropout sample
                    if model_idx == 0: # Only collect for the first model
                        for b_idx in range(images_view1.shape[0]):
                            # Ensure detected_objects_batch and aggregated_outputs are correctly indexed for a single image
                            single_detected_objects = detected_objects_batch[b_idx]
                            single_attribute_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['attribute_logits'].items()}
                            single_relation_logits = {k: v[b_idx].unsqueeze(0) for k, v in aggregated_outputs['relation_logits'].items()}

                            scene_graph = model.extract_scene_graph(
                                single_detected_objects,
                                single_attribute_logits,
                                single_relation_logits
                            )
                            scene_graph["bongard_prediction"] = int(torch.argmax(bongard_logits[b_idx]).item())
                            
                            # Store raw logits for uncertainty calculation later
                            scene_graph["raw_bongard_logits"] = bongard_logits[b_idx].cpu().numpy().tolist()
                            scene_graph["bongard_probs"] = probs[b_idx].tolist()
                            
                            current_batch_symbolic_outputs.append(scene_graph)
                
                member_probs.append(np.concatenate(current_sample_probs, axis=0))
                if model_idx == 0: # Only store symbolic outputs for the first model across MC samples
                    member_symbolic_outputs_mc_samples.append(current_batch_symbolic_outputs)
        
        # Average MC Dropout samples for this model
        if use_mc_dropout:
            avg_member_probs = np.mean(np.stack(member_probs, axis=0), axis=0)
            logger.info(f"Model {model_idx} averaged {mc_dropout_samples} MC Dropout samples.")
            
            # If MC Dropout, calculate epistemic and aleatoric uncertainty for symbolic outputs
            if model_idx == 0 and member_symbolic_outputs_mc_samples:
                # Reorganize symbolic outputs to be per-image, across MC samples
                num_images_in_batch = len(member_symbolic_outputs_mc_samples[0])
                for img_idx in range(num_images_in_batch):
                    all_mc_logits_for_image = []
                    for mc_sample_data in member_symbolic_outputs_mc_samples:
                        all_mc_logits_for_image.append(mc_sample_data[img_idx]["raw_bongard_logits"])
                    
                    all_mc_logits_for_image_np = np.array(all_mc_logits_for_image) # [mc_samples, num_classes]
                    
                    # Epistemic uncertainty: variance of predictions across MC samples
                    epistemic_probs = F.softmax(torch.tensor(all_mc_logits_for_image_np), dim=-1).numpy()
                    epistemic_uncertainty = np.mean(np.var(epistemic_probs, axis=0)) # Mean variance across classes
                    
                    # Aleatoric uncertainty: average of (p * (1-p)) over MC samples
                    # This is often approximated by the mean of the variance of the individual predictions
                    # For classification, a common proxy is mean of (p_i * (1-p_i))
                    aleatoric_uncertainty = np.mean(epistemic_probs * (1 - epistemic_probs))
                    
                    # Update the first MC sample's symbolic output for this image with uncertainty
                    member_symbolic_outputs_mc_samples[0][img_idx]["uncertainty"] = {
                        "epistemic": float(epistemic_uncertainty),
                        "aleatoric": float(aleatoric_uncertainty)
                    }
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0] # Use the first sample's structure, updated with uncertainty
        else:
            avg_member_probs = member_probs[0] # Only one sample
            if model_idx == 0:
                # If no MC dropout, symbolic output's uncertainty will be 0.0
                for img_data in member_symbolic_outputs_mc_samples[0]:
                    img_data["uncertainty"] = {"epistemic": 0.0, "aleatoric": 0.0}
                first_model_symbolic_outputs = member_symbolic_outputs_mc_samples[0] # Just the single sample's output
        
        all_bongard_probs.append(avg_member_probs)

    stacked_probs = np.stack(all_bongard_probs, axis=0) # Shape: [num_models, num_images, num_classes]

    if model_weights is not None and len(model_weights) == len(models):
        # Normalize weights to sum to 1
        normalized_weights = np.array(model_weights) / np.sum(model_weights)
        # Reshape weights to [num_models, 1, 1] for broadcasting
        weighted_probs = stacked_probs * normalized_weights[:, np.newaxis, np.newaxis]
        ensemble_averaged_probs = np.sum(weighted_probs, axis=0) # Sum along the model dimension
        logger.info("Ensemble prediction using weighted averaging.")
    else:
        ensemble_averaged_probs = np.mean(stacked_probs, axis=0)
        logger.info("Ensemble prediction using simple averaging.")

    logger.info(f"Ensemble prediction complete. Averaged probabilities for {len(image_paths)} images.")
    return ensemble_averaged_probs, first_model_symbolic_outputs

